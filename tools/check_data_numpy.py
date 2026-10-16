#!/usr/bin/env python
"""Torch-free checks of hvlm/data_core.py: mixture sampling, resumability, chat/label masking, collation."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from common_io import WHOLE, _split_file_spec, desegment_cjk, field_text, iter_texts, split_limit  # noqa: E402
from tokenizers import Regex, pre_tokenizers  # noqa: E402
from train_tokenizer import SPLIT_REGEX  # noqa: E402
from hvlm.data_core import (TokenMixer, collate_sequences, encode_conversation,  # noqa: E402
                            normalize_messages, sample_order, weighted_schedule)

ok = True


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")


# ---- resumable order -------------------------------------------------------------------------
full = []
g = sample_order(10, seed=3)
for _ in range(25):
    full.append(next(g))
resumed = []
g2 = sample_order(10, seed=3, start=13)
for _ in range(12):
    resumed.append(next(g2))
check("sample_order resumes mid-epoch exactly", resumed == full[13:25])
check("sample_order is a permutation per epoch", sorted(full[:10]) == list(range(10)) and sorted(full[10:20]) == list(range(10)))

# ---- token mixer -----------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as td:
    S = 8
    for name, base, n_tok in [("a", 1000, 8 * 50), ("b", 2000, 8 * 30 + 3)]:
        d = os.path.join(td, name)
        os.makedirs(d)
        arr = (base + np.arange(n_tok)).astype(np.uint16)
        arr[:200].tofile(os.path.join(d, "train_0000.bin")) if n_tok > 200 else arr.tofile(os.path.join(d, "train_0000.bin"))
        if n_tok > 200:
            arr[200:].tofile(os.path.join(d, "train_0001.bin"))
    mix = TokenMixer([(os.path.join(td, "a"), 0.7), (os.path.join(td, "b"), 0.3)], seq_len=S, planned_samples=2000, seed=1)
    srcs = np.array([int(mix.get(i)[0] // 1000) - 1 for i in range(2000)])  # 1 -> source a, 2 -> source b
    frac_a = float((srcs == 0).mean())
    check("mixture ratio ~ 0.7/0.3", abs(frac_a - 0.7) < 0.04, f"(a={frac_a:.3f})")
    w = mix.get(5)
    check("windows are contiguous token runs of seq_len", len(w) == S and (np.diff(w) == 1).all())
    check("get() is deterministic", (mix.get(123) == mix.get(123)).all())
    # no repeats inside one source until it is exhausted: first 50 'a' samples are 50 distinct windows
    a_idx = [i for i in range(2000) if srcs[i] == 0][:50]
    starts = {int(mix.get(i)[0]) for i in a_idx}
    check("no repetition within an epoch of a source", len(starts) == 50)
    check("shard boundary handled (source b spans 2 shards)", mix.source_chunks() == [50, 30])

# ---- weighted chat schedule --------------------------------------------------------------------
# An SFT mixture target is exact: a source smaller than its target share is repeated, which is what
# lets a small refusal slice hold 5% of the corpus.  The plan's own document says the share is a knob.
sched = weighted_schedule([100, 20], [0.5, 0.5], 240, seed=7)
check("the schedule has exactly the requested length", len(sched) == 240)
check("every scheduled index is a real record", sched.min() >= 0 and sched.max() < 120)
check("weights above a source's size are realised by repeating it",
      int((sched < 100).sum()) == 120 and int((sched >= 100).sum()) == 120,
      f"({int((sched < 100).sum())}/{int((sched >= 100).sum())})")
check("every record of the repeated source is used at least once",
      len(set(sched[sched >= 100].tolist())) == 20)
sched = weighted_schedule([100, 100], [0.75, 0.25], 200, seed=7)
used = np.bincount(sched[sched < 100], minlength=100)
check("a mixture that fits a single pass repeats only what the share needs",
      int((sched < 100).sum()) == 150 and used.max() == 2 and int((used == 2).sum()) == 50,
      f"({int((sched < 100).sum())}, max {used.max()}, {int((used == 2).sum())} doubled)")
check("the schedule is deterministic", (weighted_schedule([100, 100], [0.75, 0.25], 200, 7) == sched).all())
check("an empty source takes no share", int((weighted_schedule([10, 0], [0.5, 0.5], 20,
                                                               seed=1) >= 10).sum()) == 0)
check("zero records is an empty schedule", len(weighted_schedule([0, 0], [1.0, 1.0], 10)) == 0)

# ---- chat encoding ---------------------------------------------------------------------------
enc = lambda s: [20 + (ord(c) % 100) for c in s]
sp = dict(im_start=1, im_end=2, vision_start=3, vision_end=4, image_pad=5)
n_img = 49
msgs = normalize_messages({"conversations": [
    {"from": "human", "value": "<image>\nWhat is this?"},
    {"from": "gpt", "value": "A cat."},
    {"from": "human", "value": "Color?"},
    {"from": "gpt", "value": "Grey."},
]})
ids, lab, used = encode_conversation(enc, sp, msgs, n_img, True, 512)
trained = [t for t, l in zip(ids, lab) if l != -100]
expected = enc("A cat.") + [2] + enc("Grey.") + [2]
check("image used and 49 image tokens in order", used and ids.count(5) == 49 and ids[ids.index(3) + 1:ids.index(3) + 50] == [5] * 49 and ids[ids.index(3) + 50] == 4)
check("loss only on self (imitated) text + <|im_end|>", trained == expected)
check("labels never point at image/vision tokens", all(l not in (3, 4, 5) for l in lab))
check("sequence starts with <|im_start|>user", ids[:1] == [1] and ids[1:1 + len(enc("user\n"))] == enc("user\n"))

ids2, lab2, used2 = encode_conversation(enc, sp, normalize_messages({"messages": [
    {"role": "user", "content": "hi"}, {"role": "self", "content": "yo"}]}), n_img, False, 512)
check("text-only record has no image tokens", not used2 and 5 not in ids2)
check("'gpt' input maps to role self",
      normalize_messages({"messages": [{"role": "gpt", "content": "x"}]})[0]["role"] == "self")

ids3, _, used3 = encode_conversation(enc, sp, normalize_messages({"messages": [
    {"role": "user", "content": "describe"}, {"role": "self", "content": "ok"}]}), n_img, True, 512)
check("image without <image> tag is prepended to first user turn", used3 and ids3[len(enc("user\n")) + 1] == 3)

ids4, _, used4 = encode_conversation(enc, sp, normalize_messages({"messages": [
    {"role": "user", "content": "<image> x"}, {"role": "self", "content": "ok"}]}), n_img, False, 512)
check("<image> tag without an image is dropped", not used4 and 5 not in ids4)

try:
    encode_conversation(enc, sp, msgs, n_img, True, 30)
    check("truncating inside the image block is rejected", False)
except ValueError:
    check("truncating inside the image block is rejected", True)
try:
    encode_conversation(enc, sp, msgs, n_img, True, 70)
    check("all-masked sample (answer cut off) is rejected", False)
except ValueError:
    check("all-masked sample (answer cut off) is rejected", True)

# ---- collate ---------------------------------------------------------------------------------
a = encode_conversation(enc, sp, msgs, n_img, True, 512)[:2]
b = encode_conversation(enc, sp, normalize_messages({"messages": [{"role": "user", "content": "hi"}, {"role": "self", "content": "yo"}]}), n_img, False, 512)[:2]
ids_b, lab_b, pos = collate_sequences([a, b], pad_id=6, image_pad_id=5, grid=7)
T = max(len(a[0]), len(b[0]))
check("collate shapes", ids_b.shape == (2, T) and lab_b.shape == (2, T) and pos.shape == (3, 2, T))
check("right padding with pad id / -100", (ids_b[1, len(b[0]):] == 6).all() and (lab_b[1, len(b[0]):] == -100).all())
first_img = int(np.where(ids_b[0] == 5)[0][0])
check("image tokens carry (t,h,w) grid positions", pos[0, 0, first_img] == pos[0, 0, first_img + 48] and pos[1, 0, first_img + 7] == pos[1, 0, first_img] + 1 and pos[2, 0, first_img + 1] == pos[2, 0, first_img] + 1)
check("text-only row has plain arange positions", (pos[:, 1, :len(b[0])] == np.arange(len(b[0]))[None]).all())

# ---- source specs: field grammar (scripts/common_io.py) ---------------------------------------
check("split_limit strips a trailing @N", split_limit("hf:org/name:text:train@300000") == ("hf:org/name:text:train", 300000))
check("split_limit leaves other specs alone", split_limit("jsonl:data/a.jsonl:text") == ("jsonl:data/a.jsonl:text", None))

lccc = {"dialog": ["在吗", "在的咋啦"]}
check("whole-record field flattens list columns", field_text(lccc, "-") == "在吗\n在的咋啦")
check("named list column joins with newlines", field_text(lccc, "dialog") == "在吗\n在的咋啦")

forum = {"主题": "今天吃什么", "回复": [{"楼ID": "1", "回复": "火锅"}, {"楼ID": "2", "回复": "随便"}]}
check("dotted path walks into a list of dicts", field_text(forum, "回复.回复") == "火锅\n随便")
check("'+' concatenates fields in order", field_text(forum, "主题+回复.回复") == "今天吃什么\n火锅\n随便")
check("missing field yields no document", field_text(forum, "没有这个字段") is None)

# ---- source specs: record shapes (hvlm/data_core.normalize_messages) --------------------------
check("LCCC 'dialog' maps onto alternating user/self turns",
      [t["role"] for t in normalize_messages({"dialog": ["在吗", "在的", "忙啥"]})] == ["user", "self", "user"])
soda = normalize_messages({"dialogue": ["a", "b", "a2", "b2"], "speakers": ["A", "B", "A", "B"]})
check("SODA 'speakers' decides who is self", [t["role"] for t in soda] == ["user", "self", "user", "self"])
inst = normalize_messages({"system": "你是店员", "instruction": "打个招呼", "input": "客人进来了", "output": "来啦"})
check("instruction/input/output + system maps to system/user/self",
      [(t["role"], t["content"]) for t in inst] == [("system", "你是店员"), ("user", "打个招呼\n客人进来了"), ("self", "来啦")])
rp = normalize_messages({"system_prompt": "你是林小雨", "conversations": [{"from": "human", "value": "在吗"},
                                                                        {"from": "gpt", "value": "在的"}]})
check("system_prompt is prepended without duplicating a system turn", [t["role"] for t in rp] == ["system", "user", "self"])
pc = normalize_messages({"personality": ["我喜欢滑雪"], "history": ["你好", "今天干嘛"], "candidates": ["x", "去滑雪"]})
check("PersonaChat: personality -> system, last candidate -> self",
      [(t["role"], t["content"]) for t in pc] == [("system", "我喜欢滑雪"), ("user", "今天干嘛"), ("self", "去滑雪")])
emp = normalize_messages({"conv_id": "hit:0_conv:1", "context": "sentimental", "prompt": "I miss my dog",
                          "utterance": "that sounds rough", "speaker_idx": 1})
check("EmpatheticDialogues: one utterance per row maps to prompt -> reply",
      [(t["role"], t["content"]) for t in emp] == [("user", "I miss my dog"), ("self", "that sounds rough")])
check("the emotion label in 'context' is not glued onto the prompt", "sentimental" not in emp[0]["content"])
try:
    normalize_messages({"messages": [{"role": "assistant", "content": "hi"}]})
    check("'assistant' role is rejected (would turn the model into an assistant)", False)
except ValueError:
    check("'assistant' role is rejected (would turn the model into an assistant)", True)
try:
    normalize_messages({"foo": 1})
    check("unrecognised record shape is rejected", False)
except ValueError:
    check("unrecognised record shape is rejected", True)

# ---- source specs ----------------------------------------------------------------------------
# A remote glob keeps its scheme, so the `hf` in front of `://` is not mistaken for the glob and a
# trailing `:field` is still split off.  Getting this wrong silently reads the wrong files.
check("remote parquet spec keeps its scheme and splits the field",
      _split_file_spec("hf://datasets/o/n@refs/convert/parquet/base/train/*.parquet:dialog")
      == ("hf://datasets/o/n@refs/convert/parquet/base/train/*.parquet", "dialog"))
check("remote parquet spec without a field is all glob",
      _split_file_spec("hf://datasets/o/n@rev/x/*.parquet") == ("hf://datasets/o/n@rev/x/*.parquet", WHOLE))
check("local glob spec still splits the field",
      _split_file_spec("data/x/*.parquet:text") == ("data/x/*.parquet", "text"))
check("'@' inside a remote path is not read as a document limit",
      split_limit("parquet:hf://datasets/o/n@refs/convert/parquet/x/*.parquet:dialog")[1] is None)

# A glob that matches nothing used to tokenise to an empty bin with exit status 0, so a typo in a spec
# (e.g. `.../convert/parquet/<config>/train/*` for a dataset that only has `partial-train`) looked like
# "the corpus is small" instead of an error.
try:
    list(iter_texts("jsonl:/nonexistent-dir-xyz/*.jsonl:text"))
    check("a glob matching no file fails loudly", False)
except FileNotFoundError as e:
    check("a glob matching no file fails loudly", "matched no files" in str(e))

# ---- de-segmentation -------------------------------------------------------------------------
# LCCC ships its dialog pre-segmented.  Training on it uncorrected teaches the model to put spaces
# between characters, so the join has to happen, and it must not eat English word spacing.
check("segmented Chinese is rejoined",
      desegment_cjk("你 去 那儿 竟然 不喊 我 生气 了 ， 快点 给 我 道歉") == "你去那儿竟然不喊我生气了，快点给我道歉")
check("CJK/Latin boundary loses its space, Latin/Latin keeps it",
      desegment_cjk("我用 SEED.24 小时 签到") == "我用SEED.24小时签到")
check("English spaces survive", desegment_cjk("no cjk here  at all") == "no cjk here  at all")
check("newlines still separate turns", desegment_cjk("第一 行\n第二 行") == "第一行\n第二行")

# ---- tokenizer pre-split ---------------------------------------------------------------------
# Digit runs must stay whole: Qwen2 splits them one at a time to help arithmetic, which this model does
# not do, and chat is full of numbers (`25岁`, `100块`).  The regex uses \p{...}, so Python's own `re`
# cannot even compile it - the same `tokenizers` pre-tokenizer the trainer uses has to answer.
_split = pre_tokenizers.Split(Regex(SPLIT_REGEX), behavior="isolated", invert=False)
pieces = [p[0] for p in _split.pre_tokenize_str("2024年12月")]
check("digit runs are not split one by one", "2024" in pieces and "12" in pieces, f"({pieces})")
check("letters are still split off a trailing digit",
      [p[0] for p in _split.pre_tokenize_str("gpt4")] == ["gpt", "4"])

print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
