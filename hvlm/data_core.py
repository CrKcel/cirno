"""Data logic that needs only numpy (unit-testable without torch); torch wrappers live in scripts/data.py."""
import glob
import os

import numpy as np

from .positions import build_mrope_positions


# ----------------------------------------------------------------------------------------------
# resumable sample order
# ----------------------------------------------------------------------------------------------
def sample_order(n, seed, start=0):
    """Infinite stream of dataset indices: a fresh permutation per epoch, resumable at global sample `start`."""
    epoch, offset = divmod(start, n)
    while True:
        perm = np.random.default_rng([seed, epoch]).permutation(n)
        for i in perm[offset:]:
            yield int(i)
        epoch, offset = epoch + 1, 0


# ----------------------------------------------------------------------------------------------
# pre-training: weighted mixture of tokenised sources, deterministic in the global sample index
# ----------------------------------------------------------------------------------------------
class TokenMixer:
    """specs: list of (directory, weight).  Each directory holds train_*.bin (uint16) from prepare_data.py.
    get(i) returns the i-th window of `seq_len` tokens; the same i always returns the same window, so training
    resumes exactly by restarting at the saved sample counter.  Each source is traversed in a shuffled order
    without repetition until exhausted, then wraps (a new epoch for that source)."""

    def __init__(self, specs, seq_len, planned_samples, seed=0):
        self.seq_len = seq_len
        self.planned = planned_samples
        self.shards, self.starts, weights = [], [], []
        for d, w in specs:
            paths = sorted(glob.glob(os.path.join(d, "train_*.bin")))
            if not paths:
                raise FileNotFoundError(f"no train_*.bin in {d}")
            arrs = [np.memmap(p, dtype=np.uint16, mode="r") for p in paths]
            counts = np.array([len(a) // seq_len for a in arrs], dtype=np.int64)
            self.shards.append(arrs)
            self.starts.append(np.concatenate([[0], np.cumsum(counts)]))
            weights.append(w)
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / weights.sum()
        rng = np.random.default_rng(seed)
        self.src_of = rng.choice(len(specs), size=planned_samples, p=weights).astype(np.int32)
        self.local = np.zeros(planned_samples, dtype=np.int64)
        self.perms = []
        for s in range(len(specs)):
            mask = self.src_of == s
            self.local[mask] = np.arange(int(mask.sum()))
            self.perms.append(np.random.default_rng([seed, 1000 + s]).permutation(int(self.starts[s][-1])))

    def source_chunks(self):
        return [int(s[-1]) for s in self.starts]

    def get(self, i):
        i = i % self.planned
        s = int(self.src_of[i])
        c = int(self.perms[s][self.local[i] % len(self.perms[s])])
        k = int(np.searchsorted(self.starts[s], c, side="right")) - 1
        w = c - int(self.starts[s][k])
        return np.asarray(self.shards[s][k][w * self.seq_len:(w + 1) * self.seq_len], dtype=np.int64)


# ----------------------------------------------------------------------------------------------
# weighted sampling over chat sources
# ----------------------------------------------------------------------------------------------
def weighted_schedule(counts, weights, total, seed=0):
    """Deterministic sample-index schedule realising a target mixture exactly.

    `counts[i]` is how many records group i holds and `weights[i]` its target share.  The share is
    realised by *repeating* a source that is too small for it (records are cycled through fresh
    permutations), which is what makes a mixture target meaningful for a small but important slice such
    as the SFT refusal set: capping at one pass instead would let a 500-row source never exceed 3.7% of
    a 13.5k-row corpus however it is weighted.  Returns int32 indices into the concatenation of the
    groups, deterministic in (counts, weights, total, seed), so `sample_order` resumes it exactly.
    """
    counts = np.asarray(counts, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if len(weights) != len(counts):
        raise ValueError(f"{len(weights)} weights for {len(counts)} groups")
    total = int(total)
    live = counts > 0
    if total <= 0 or not live.any():
        return np.zeros(0, dtype=np.int32)
    w = np.where(live, np.maximum(weights, 0.0), 0.0)
    if w.sum() <= 0:  # no usable weights: fall back to the sources' own proportions
        w = np.where(live, counts.astype(np.float64), 0.0)
    w = w / w.sum()
    want = total * w
    take = np.floor(want).astype(np.int64)
    for i in np.argsort(-(want - take), kind="stable"):  # largest-remainder rounding
        if total - int(take.sum()) <= 0:
            break
        if live[i]:
            take[i] += 1
    base = np.concatenate([[0], np.cumsum(counts)])[:-1]
    chunks = []
    for i, k in enumerate(take):
        if k <= 0:
            continue
        perms = [np.random.default_rng([seed, e, i]).permutation(int(counts[i]))
                 for e in range(int(np.ceil(k / counts[i])))]
        chunks.append(np.concatenate(perms)[:k] + base[i])
    sched = np.concatenate(chunks)
    np.random.default_rng([seed, 997]).shuffle(sched)
    return sched.astype(np.int32)


# ----------------------------------------------------------------------------------------------
# chat / multimodal formatting
# ----------------------------------------------------------------------------------------------
# The imitated speaker is role "self": the model is a human chat partner, not an assistant.  "gpt" (LLaVA)
# and "self" are the accepted input spellings; anything else - notably "assistant" - is rejected and the
# record is dropped as a bad sample.
ROLE_MAP = {"human": "user", "user": "user", "gpt": "self", "self": "self", "system": "system"}

# Column names the same content ships under in the wild (see doc/training.md 4.1).  Mapping them here is
# what lets a Hugging Face dataset be pointed at directly instead of being converted first.
_TURNS = ("messages", "conversations")                      # already turn-structured
_UTTERANCES = ("dialog", "dialogue", "utterances")          # one conversation as a flat list of lines
_PERSONA = ("system_prompt", "persona", "personality", "system")
_PROMPTS = ("instruction", "question", "prompt", "INSTRUCTION", "问")
_REPLIES = ("output", "answer", "response", "utterance", "RESPONSE", "答")
# Side info to glue onto the prompt.  "context" is deliberately NOT here: EmpatheticDialogues (and others)
# use that column for an emotion *label*, so gluing it on would inject "sentimental" into the user turn.
_INPUTS = ("input",)


def _as_text(v):
    if isinstance(v, (list, tuple)):
        return "\n".join(str(x) for x in v)
    return v if isinstance(v, str) else str(v)


def _first(rec, keys):
    for k in keys:
        if rec.get(k):
            return _as_text(rec[k])
    return None


def _from_utterances(utts, speakers):
    """A flat utterance list is one conversation: whoever opens is the prompt, the other side is self."""
    if isinstance(speakers, (list, tuple)) and len(speakers) == len(utts):
        first = speakers[0]
        return [{"role": "user" if spk == first else "self", "content": _as_text(u)}
                for spk, u in zip(speakers, utts)]
    return [{"role": "user" if i % 2 == 0 else "self", "content": _as_text(u)} for i, u in enumerate(utts)]


def normalize_messages(rec):
    """Map one record onto [{"role","content"}] with role in {system, user, self}.

    Native shapes: {"conversations":[{"from","value"}]} (LLaVA) / {"messages":[{"role","content"}]} (OpenAI).
    The shapes Hugging Face dialogue sets actually ship are accepted too, so no conversion step is needed:
      * a flat utterance list under "dialog"/"dialogue"/"utterances" (LCCC, SODA, DailyDialog), optionally
        with a parallel "speakers" list; turns alternate and the opener is the prompt;
      * a persona ("system"/"system_prompt"/"persona") plus an "instruction"/"input" -> "output" pair, or
        the "question"/"answer" spelling;
      * PersonaChat dumps: "personality" + "history" + "candidates" (the last candidate is the true reply).
    """
    turns = next((rec[k] for k in _TURNS if rec.get(k)), None)
    if turns:
        out = []
        first = str(turns[0].get("role") or turns[0].get("from") or "").lower()
        persona = _first(rec, _PERSONA) if first != "system" else None
        if persona:
            out.append({"role": "system", "content": persona})
        for t in turns:
            name = str(t.get("role") or t.get("from")).lower()
            if name not in ROLE_MAP:
                raise ValueError(f"unknown chat role '{name}': the imitated side must be spelled self/gpt/human")
            content = t.get("content") if "content" in t else t.get("value")
            out.append({"role": ROLE_MAP[name], "content": content or ""})
        return out

    utts = next((rec[k] for k in _UTTERANCES if isinstance(rec.get(k), (list, tuple)) and rec[k]), None)
    if utts:
        return _from_utterances(list(utts), rec.get("speakers"))

    if rec.get("history") and rec.get("candidates"):  # PersonaChat: persona + history + true reply candidates
        persona = _first(rec, _PERSONA)
        out = [{"role": "system", "content": persona}] if persona else []
        out += [{"role": "user", "content": _as_text(rec["history"][-1])},
                {"role": "self", "content": _as_text(rec["candidates"][-1])}]
        return out

    prompt, reply = _first(rec, _PROMPTS), _first(rec, _REPLIES)
    if reply:
        persona = _first(rec, _PERSONA)
        out = [{"role": "system", "content": persona}] if persona else []
        user = "\n".join(x for x in (prompt, _first(rec, _INPUTS)) if x)
        if user:
            out.append({"role": "user", "content": user})
        out.append({"role": "self", "content": reply})
        return out

    raise ValueError("record has neither 'messages' nor 'conversations' nor a known prompt/reply pair")


def encode_conversation(encode, sp, messages, n_img_tokens, has_image, max_len):
    """ChatML-style encoding with loss only on the imitated side's replies (role "self", + its <|im_end|>).

    encode : callable str -> list[int]   (plain text, no special tokens added)
    sp     : dict with ids for im_start, im_end, vision_start, vision_end, image_pad
    The image block <|vision_start|> <|image_pad|> x n <|vision_end|> replaces the first '<image>' tag
    (or is put at the start of the first user turn when the record has an image but no tag).
    Returns (ids, labels, image_used).  Raises ValueError for unusable samples (image cut by truncation, or
    no trainable token left)."""
    ids, labels = [], []
    image_used = False
    nl = encode("\n")
    for m in messages:
        role, content = m["role"], m["content"]
        head = [sp["im_start"]] + encode(role + "\n")
        ids += head
        labels += [-100] * len(head)
        if role == "self":
            body = encode(content) + [sp["im_end"]]
            ids += body
            labels += body
            ids += nl
            labels += [-100] * len(nl)
            continue
        parts = content.split("<image>")
        if has_image and not image_used and len(parts) == 1 and role == "user":
            parts = ["", content]  # no tag: image goes first
        body = []
        for pi, part in enumerate(parts):
            if pi > 0 and has_image and not image_used:
                body += [sp["vision_start"]] + [sp["image_pad"]] * n_img_tokens + [sp["vision_end"]]
                image_used = True
            if part:
                body += encode(part)
        body += [sp["im_end"]] + nl
        ids += body
        labels += [-100] * len(body)
    ids, labels = ids[:max_len], labels[:max_len]
    n_pad = sum(1 for t in ids if t == sp["image_pad"])
    if image_used and n_pad != n_img_tokens:
        raise ValueError("image block truncated")
    if not image_used and n_pad:
        raise ValueError("stray image tokens")
    if all(l == -100 for l in labels):
        raise ValueError("no trainable tokens")
    return ids, labels, image_used


def collate_sequences(items, pad_id, image_pad_id, grid):
    """items: list of (ids, labels).  Right-pads and builds M-RoPE position ids [3, B, T]."""
    B = len(items)
    T = max(len(i) for i, _ in items)
    ids = np.full((B, T), pad_id, dtype=np.int64)
    lab = np.full((B, T), -100, dtype=np.int64)
    for b, (i, l) in enumerate(items):
        ids[b, :len(i)] = i
        lab[b, :len(l)] = l
    pos = np.stack([build_mrope_positions(ids[b], image_pad_id, grid) for b in range(B)], axis=1)
    return ids, lab, pos
