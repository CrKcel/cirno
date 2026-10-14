#!/usr/bin/env python
"""Train a 32k byte-level BPE tokenizer (Chinese + English) with the special tokens the model expects.

Special-token ids are FIXED (they are referenced by configs/hvlm_130m.json):
    0 <|endoftext|>   1 <|im_start|>   2 <|im_end|>   3 <|vision_start|>   4 <|vision_end|>
    5 <|image_pad|>   6 <|pad|>        7..15 reserved

Example (about 0.5-2 GB of mixed text is plenty):
    python scripts/train_tokenizer.py --out tokenizer \
        --source "hf:HuggingFaceFW/fineweb-edu:text:sample-10BT:train@300000" \
        --source "jsonl:data/zh/*.jsonl:text@300000" \
        --source "jsonl:data/code/*.jsonl:content@100000"

`@N` limits a source to its first N documents, and it is the **per-source budget** - the mix is whatever
those N values say, so pick them from the mean document length rather than from a hunch.  Balancing by
document count does NOT balance the text: a chat turn is ~50 chars while an article is ~6300, so equal N
values would give the target distribution ~1% of the corpus.
"""
import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common_io import desegment_cjk, iter_texts, split_limit  # noqa: E402

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>",
            "<|image_pad|>", "<|pad|>"] + [f"<|reserved_{i}|>" for i in range(9)]

# Qwen2-style pre-tokenisation, with one change: digit runs stay together (`\p{N}+`, not `\p{N}`).
# Qwen2 splits digits one at a time because it targets arithmetic; this project explicitly does not do
# maths, and numbers are everywhere in chat (几点 / 多少钱 / 25岁), where "2024年" costing 5 tokens instead
# of 2 is pure waste.  Unseen numbers still decompose into digits under BPE, so nothing is lost.
SPLIT_REGEX = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}+| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")

CHAT_TEMPLATE = (
    "{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>' + '\\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>self\\n' }}{% endif %}"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True, help="source spec, optionally suffixed with @max_docs")
    ap.add_argument("--out", default="tokenizer")
    ap.add_argument("--vocab_size", type=int, default=32768)
    ap.add_argument("--desegment", action="store_true",
                    help="remove spaces between CJK tokens (word-segmented corpora such as LCCC's dialog)")
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
            spec, max_docs = split_limit(s)
            texts = iter_texts(spec, max_docs)
            iters.append((desegment_cjk(t) for t in texts) if args.desegment else texts)
        # Round-robin, but let each source run to its own `@N`.  `zip` would stop at the *shortest*
        # source, and since a chat turn is ~50 chars while an article is ~6300, balancing by document
        # count leaves the target distribution at ~1% of the tokenizer's text.  With `zip_longest` the
        # `@N` values are the per-source budget, so the mix is whatever those say (BPE itself only counts
        # pairs, so the interleaving order does not change what is learned).
        for batch in itertools.zip_longest(*iters):
            yield from (t for t in batch if t is not None)

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
