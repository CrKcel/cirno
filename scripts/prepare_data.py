#!/usr/bin/env python
"""Tokenise text sources into flat uint16 token streams (documents separated by <|endoftext|>).

    python scripts/prepare_data.py --tokenizer tokenizer --out data/bin/fineweb_edu \
        --source "hf:HuggingFaceFW/fineweb-edu:text:sample-10BT:train" --max_tokens 1200000000 --val_tokens 4000000

`--source` takes any source spec from common_io.py (jsonl / parquet / txt / hf, with the `-`, `a.b` and
`a+b` field forms) and may be suffixed with `@N` to use only the first N documents, e.g.
    --source "hf:silver/lccc:-:base:train@2000000"        # LCCC conversations as whole documents
    --source "hf:liwu/MNBVC:主题+回复.回复:forum:train"     # forum threads: title + replies

Output directory: train_0000.bin, train_0001.bin, ... (about --shard_tokens each), val.bin and meta.json.
Run it once per data source (English / Chinese / code / math ...) and mix them at training time with
`train.py --data dir_a:0.5,dir_b:0.3,...` - mixing weights can then be changed without re-tokenising.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common_io import iter_texts, split_limit  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--source", required=True, help="source spec (see common_io.py), optionally suffixed with @max_docs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_tokens", type=int, default=None)
    ap.add_argument("--val_tokens", type=int, default=0, help="the first N tokens go to val.bin")
    ap.add_argument("--shard_tokens", type=int, default=100_000_000)
    ap.add_argument("--batch_docs", type=int, default=512)
    ap.add_argument("--max_doc_chars", type=int, default=200_000)
    args = ap.parse_args()
    spec, max_docs = split_limit(args.source)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    assert len(tok) <= 65536, "uint16 storage needs vocab_size <= 65536"
    eos = tok.convert_tokens_to_ids("<|endoftext|>")
    os.makedirs(args.out, exist_ok=True)

    state = dict(total=0, shard=0, in_shard=0, val=0)
    shard_f = None
    val_f = open(os.path.join(args.out, "val.bin"), "wb") if args.val_tokens > 0 else None

    def write(arr):
        nonlocal shard_f
        arr = arr.astype(np.uint16)
        if val_f is not None and state["val"] < args.val_tokens:
            take = min(len(arr), args.val_tokens - state["val"])
            arr[:take].tofile(val_f)
            state["val"] += take
            arr = arr[take:]
            if len(arr) == 0:
                return
        while len(arr):
            if shard_f is None or state["in_shard"] >= args.shard_tokens:
                if shard_f is not None:
                    shard_f.close()
                    state["shard"] += 1
                shard_f = open(os.path.join(args.out, f"train_{state['shard']:04d}.bin"), "wb")
                state["in_shard"] = 0
            take = min(len(arr), args.shard_tokens - state["in_shard"])
            arr[:take].tofile(shard_f)
            state["in_shard"] += take
            state["total"] += take
            arr = arr[take:]

    t0, buf, n_docs = time.time(), [], 0

    def flush():
        nonlocal buf
        if not buf:
            return
        enc = tok(buf, add_special_tokens=False)["input_ids"]
        out = []
        for ids in enc:
            out.extend(ids)
            out.append(eos)
        write(np.asarray(out, dtype=np.int64))
        buf = []

    for text in iter_texts(spec, max_docs):
        buf.append(text[: args.max_doc_chars])
        n_docs += 1
        if len(buf) >= args.batch_docs:
            flush()
            seen = state["total"] + state["val"]
            if (n_docs // args.batch_docs) % 20 == 0:
                print(f"{n_docs} docs, {seen / 1e6:.1f}M tokens, {seen / (time.time() - t0):.0f} tok/s", flush=True)
            if args.max_tokens and seen >= args.max_tokens:
                break
    flush()
    if shard_f is not None:
        shard_f.close()
    if val_f is not None:
        val_f.close()
    meta = dict(train_tokens=state["total"], val_tokens=state["val"], shards=state["shard"] + 1,
                vocab_size=len(tok), dtype="uint16", source=args.source)
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=1)
    print(meta)


if __name__ == "__main__":
    main()
