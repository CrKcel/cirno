#!/usr/bin/env python
"""Train a 32k byte-level BPE tokenizer (Chinese + English) with the special tokens the model expects.

Special-token ids are FIXED (they are referenced by configs/hvlm_130m.json):
    0 <|endoftext|>   1 <|im_start|>   2 <|im_end|>   3 <|vision_start|>   4 <|vision_end|>
    5 <|image_pad|>   6 <|pad|>        7..15 reserved

Example (about 1-2 GB of mixed text is plenty; weights keep the languages balanced):
    python scripts/train_tokenizer.py --out tokenizer \
        --source "hf:HuggingFaceFW/fineweb-edu:text:sample-10BT:train@300000" \
        --source "jsonl:data/zh/*.jsonl:text@300000" \
        --source "jsonl:data/code/*.jsonl:content@100000"
(`@N` limits a source to its first N documents.)
"""
import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common_io import iter_texts  # noqa: E402

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>",
            "<|image_pad|>", "<|pad|>"] + [f"<|reserved_{i}|>" for i in range(9)]

# Qwen2-style pre-tokenisation: digits are split one by one (helps arithmetic), contractions, words, punctuation.
SPLIT_REGEX = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")

CHAT_TEMPLATE = (
    "{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>' + '\\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True, help="source spec, optionally suffixed with @max_docs")
    ap.add_argument("--out", default="tokenizer")
    ap.add_argument("--vocab_size", type=int, default=32768)
    args = ap.parse_args()

    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(SPLIT_REGEX), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    def corpus():
        iters = []
        for s in args.source:
            spec, _, lim = s.partition("@") if "@" in s.rsplit(":", 1)[-1] else (s, "", "")
            iters.append(iter_texts(spec, int(lim) if lim else None))
        # round-robin so every language contributes evenly while the trainer streams
        for batch in zip(*[iter(it) for it in iters]):
            yield from batch
        # (zip stops at the shortest source: that is intentional, it keeps the mix balanced)

    tok.train_from_iterator(corpus(), trainer=trainer)
    assert tok.token_to_id("<|image_pad|>") == 5 and tok.token_to_id("<|vision_start|>") == 3, "special ids moved"

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
        clean_up_tokenization_spaces=False,
    )
    fast.chat_template = CHAT_TEMPLATE
    os.makedirs(args.out, exist_ok=True)
    fast.save_pretrained(args.out)
    print(f"saved tokenizer with {len(fast)} tokens to {args.out}")
    for t in ["你好，世界！Hello world 12345", "def f(x):\n    return x**2"]:
        ids = fast(t).input_ids
        print(len(t), "chars ->", len(ids), "tokens;", fast.decode(ids) == t)


if __name__ == "__main__":
    main()
