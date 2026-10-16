"""torch Dataset / collate wrappers around hvlm.data_core."""
import hashlib
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from common_io import desegment_cjk, iter_records  # noqa: E402
from hvlm.data_core import (TokenMixer, collate_sequences, encode_conversation,  # noqa: E402
                            normalize_messages)

SOURCE_KINDS = ("jsonl", "json", "parquet", "hf")


def is_source_spec(source):
    """True for `hf:...` / `parquet:...` / `jsonl:<glob>` rather than a path to a local jsonl file."""
    kind, _, rest = source.partition(":")
    return bool(rest) and kind in SOURCE_KINDS


def spool(sources, cache_dir):
    """Stream remote / glob sources once into a canonical jsonl file.

    ChatDataset needs byte offsets for random access (that is what makes --resume exact), and holding a
    multi-million-record dataset in memory is not an option, so the records are normalised once into
    `cache_dir` and reused on later runs."""
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.sha1("|".join(sources).encode()).hexdigest()[:12]
    path = os.path.join(cache_dir, f"records-{key}.jsonl")
    if os.path.exists(path):
        return path
    tmp, n, bad = path + ".part", 0, 0
    with open(tmp, "w", encoding="utf-8") as f:
        for spec in sources:
            for rec in iter_records(spec):
                try:
                    rec = dict(rec)
                    msgs = normalize_messages(rec)
                    if not any(m["role"] == "self" for m in msgs):
                        raise ValueError("no 'self' turn: nothing to train on")
                except Exception:  # unknown role / unrecognised shape: not a usable training sample
                    bad += 1
                    continue
                out = {"messages": msgs}
                if rec.get("image"):
                    out["image"] = rec["image"]
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                n += 1
    os.replace(tmp, path)
    print(f"cached {n} records ({bad} unusable) from {len(sources)} source(s) -> {path}")
    return path


class PretrainDataset(Dataset):
    """Packed text windows from a weighted mixture of tokenised sources (see TokenMixer)."""

    def __init__(self, specs, seq_len, planned_samples, seed=0):
        self.mixer = TokenMixer(specs, seq_len, planned_samples, seed)

    def __len__(self):
        return self.mixer.planned

    def __getitem__(self, i):
        return torch.from_numpy(self.mixer.get(i))


class ValWindows(Dataset):
    """Sequential windows of a val.bin file."""

    def __init__(self, path, seq_len):
        self.arr = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len

    def __len__(self):
        return len(self.arr) // self.seq_len

    def __getitem__(self, i):
        return torch.from_numpy(np.asarray(self.arr[i * self.seq_len:(i + 1) * self.seq_len], dtype=np.int64))


def load_image(path, size):
    """Pad to a square with mid-grey (keeps the whole image), resize, normalise to [-1, 1], CHW float32."""
    from PIL import Image

    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), (127, 127, 127))
    canvas.paste(im, ((s - w) // 2, (s - h) // 2))
    canvas = canvas.resize((size, size), Image.Resampling.BICUBIC)
    arr = (np.asarray(canvas, dtype=np.float32) / 255.0 - 0.5) / 0.5
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())


class ChatDataset(Dataset):
    """Chat records: {"messages": [{"role","content"}], "image": "rel/path.jpg" (optional)}.

    Local jsonl files are read in place; remote / glob sources (`hf:`, `parquet:`, `jsonl:<glob>`, see
    common_io.py) are spooled into `cache_dir` first, one file per source so a source keeps its own row
    count (that is what `--mm_weights` mixes).  Images are loaded lazily; unusable records (unknown role,
    missing image, truncated image block, ...) are skipped by falling through to the next record.

    `set_mix(weights, planned_samples)` folds a mixture into the dataset: the sample index then walks a
    deterministic weighted schedule instead of the raw concatenation, so an SFT slice's share of the
    epochs is a parameter rather than whatever its line count happens to be."""

    def __init__(self, sources, image_root, tokenizer, max_len, vision_cfg, token_ids, cache_dir="data/cache",
                 desegment=False):
        self.sources = list(sources)
        self.files = [spool([s], cache_dir) if is_source_spec(s) else s for s in self.sources]
        self.offsets = []  # (file index, byte offset)
        self.counts = []
        for fi, p in enumerate(self.files):
            start = len(self.offsets)
            with open(p, "rb") as f:
                off = 0
                for line in f:
                    if line.strip():
                        self.offsets.append((fi, off))
                    off += len(line)
            self.counts.append(len(self.offsets) - start)
        self.counts = np.asarray(self.counts, dtype=np.int64)
        self.schedule = None  # int32 positions into self.offsets once set_mix() has been called
        self.mix_weights = None
        self.root = image_root
        self.tok = tokenizer
        self.max_len = max_len
        self.vc = vision_cfg
        self.sp = token_ids  # dict: im_start, im_end, vision_start, vision_end, image_pad
        self.desegment = desegment  # LCCC-style word-segmented turns (see common_io.desegment_cjk)
        self.skipped = 0

    def set_mix(self, weights, planned_samples, seed=0):
        from hvlm.data_core import weighted_schedule

        if len(weights) != len(self.sources):
            raise ValueError(f"{len(weights)} weights for {len(self.sources)} sources")
        self.mix_weights = [float(w) for w in weights]
        self.schedule = weighted_schedule(self.counts, self.mix_weights, int(planned_samples), seed)

    def source_counts(self):
        rows = [int(c) for c in self.counts]
        weights = self.mix_weights or [1.0] * len(rows)
        return list(zip(self.sources, weights, rows))

    def __len__(self):
        return len(self.schedule) if self.schedule is not None else len(self.offsets)

    def _encode(self, s):
        return self.tok(desegment_cjk(s) if self.desegment else s, add_special_tokens=False)["input_ids"]

    def _load(self, idx):
        if self.schedule is not None:
            idx = int(self.schedule[idx])
        fi, off = self.offsets[idx]
        with open(self.files[fi], "rb") as f:
            f.seek(off)
            rec = json.loads(f.readline())
        msgs = normalize_messages(rec)
        has_image = bool(rec.get("image"))
        ids, labels, used = encode_conversation(self._encode, self.sp, msgs, self.vc.num_image_tokens, has_image, self.max_len)
        img = load_image(os.path.join(self.root, rec["image"]), self.vc.image_size) if used else None
        return ids, labels, img

    def __getitem__(self, idx):
        for k in range(16):
            try:
                return self._load((idx + k) % len(self))
            except Exception:  # bad record / unreadable or huge image / odd content type: try the next record
                self.skipped += 1
                continue
        raise RuntimeError("16 consecutive unusable records - check the data files / image_root")


class ChatCollator:
    def __init__(self, pad_id, image_pad_id, grid):
        self.pad_id, self.image_pad_id, self.grid = pad_id, image_pad_id, grid

    def __call__(self, batch):
        ids, lab, pos = collate_sequences([(b[0], b[1]) for b in batch], self.pad_id, self.image_pad_id, self.grid)
        out = dict(input_ids=torch.from_numpy(ids), labels=torch.from_numpy(lab), position_ids=torch.from_numpy(pos))
        imgs = [b[2] for b in batch if b[2] is not None]
        if imgs:
            out["pixel_values"] = torch.stack(imgs)
        return out


def special_ids(tokenizer):
    t = tokenizer.convert_tokens_to_ids
    return dict(im_start=t("<|im_start|>"), im_end=t("<|im_end|>"), vision_start=t("<|vision_start|>"),
                vision_end=t("<|vision_end|>"), image_pad=t("<|image_pad|>"))
