"""torch Dataset / collate wrappers around hvlm.data_core."""
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hvlm.data_core import (TokenMixer, collate_sequences, encode_conversation,  # noqa: E402
                            normalize_messages)


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
    """jsonl records: {"image": "rel/path.jpg" (optional), "conversations": [...]} or {"messages": [...]}.
    Images are loaded lazily; unusable records (missing image, truncated image block, ...) are skipped by
    falling through to the next record."""

    def __init__(self, jsonl_paths, image_root, tokenizer, max_len, vision_cfg, token_ids):
        self.files = list(jsonl_paths)
        self.offsets = []  # (file index, byte offset)
        for fi, p in enumerate(self.files):
            with open(p, "rb") as f:
                off = 0
                for line in f:
                    if line.strip():
                        self.offsets.append((fi, off))
                    off += len(line)
        self.root = image_root
        self.tok = tokenizer
        self.max_len = max_len
        self.vc = vision_cfg
        self.sp = token_ids  # dict: im_start, im_end, vision_start, vision_end, image_pad
        self.skipped = 0

    def __len__(self):
        return len(self.offsets)

    def _encode(self, s):
        return self.tok(s, add_special_tokens=False)["input_ids"]

    def _load(self, idx):
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
