#!/usr/bin/env python
"""Reduce an .ass/.ssa subtitle corpus to plain dialogue text (one jsonl record per episode).

`chenrm/anime-subtitles` stores whole subtitle scripts, and the format is most of that: script
headers, style tables, karaoke/position overrides and timing fields.  Measured on the local snapshot,
11.43B raw characters reduce to 2.93B of dialogue text (25.6%), so training on the raw field would
spend three quarters of the subtitle bucket's token budget on markup.

ASS quirk: a `Dialogue:` line is
    Dialogue: layer,start,end,style,name,marginL,marginR,marginV,effect,text
- exactly nine comma-separated fields, and `text` may itself contain commas, hence `split(",", 9)`.
Override blocks (`{\\pos(..)}`, `{\\k50}`) are stripped and `\\N` / `\\n` / `\\h` become spaces.

The collection mixes Chinese subtitles with Japanese raw/karaoke scripts, so `--zh-only` drops the
Japanese episodes and, inside the Chinese ones, the kana-only lines (the OP/ED songs, 3.6% of lines).
`--t2s` converts traditional to simplified: roughly half of the fansub groups release traditional, and
keeping both scripts in the chat bucket teaches the model to answer in traditional.  Exact-duplicate
episodes (re-releases) are dropped by default.

    python scripts/clean_ass.py --zh-only --t2s \
        --source "parquet:/mnt/data/LLM/hf/chenrm--anime-subtitles/data/train-*.parquet:data" \
        --out /mnt/data/LLM/hf/chenrm--anime-subtitles/dialogue.jsonl
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common_io import iter_texts  # noqa: E402

_OVERRIDE = re.compile(r"\{[^}]*\}")            # {\pos(..)} / {\k50} / {\an8} style blocks
_KANA = re.compile(r"[\u3040-\u30ff]")          # hiragana + katakana: the "this is Japanese" signal
_HAN = re.compile(r"[\u4e00-\u9fff]")


def clean_ass(text, kana_ratio=None):
    """Dialogue lines of one .ass script, joined by newlines; empty when there is no dialogue.

    With `kana_ratio`, lines above that kana share are dropped as well: an otherwise Chinese episode
    carries its Japanese OP/ED song as dialogue lines (measured 3.6% of the kept lines).
    """
    lines = []
    for line in text.splitlines():
        if not line.startswith("Dialogue:"):
            continue
        fields = line.split(",", 9)
        if len(fields) < 10:
            continue
        body = _OVERRIDE.sub("", fields[9])
        for esc, rep in (("\\N", " "), ("\\n", " "), ("\\h", " ")):
            body = body.replace(esc, rep)
        body = body.strip()
        if not body:
            continue
        kana = len(_KANA.findall(body))
        if kana_ratio is not None and kana and kana / (kana + len(_HAN.findall(body))) > kana_ratio:
            continue
        lines.append(body)
    return "\n".join(lines)


def is_chinese(text, kana_ratio):
    """False for Japanese scripts and for text with barely any Han characters (English/romaji)."""
    kana, han = len(_KANA.findall(text)), len(_HAN.findall(text))
    if kana + han == 0:
        return False
    return kana / (kana + han) <= kana_ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True, help="source spec (see common_io.py)")
    ap.add_argument("--out", required=True, help="output jsonl path")
    ap.add_argument("--zh-only", action="store_true", help="drop episodes that are not Chinese (see --kana_ratio)")
    ap.add_argument("--t2s", action="store_true",
                    help="convert traditional Chinese to simplified (needs opencc); about half of the "
                         "fansub groups release traditional, which would otherwise teach mixed script")
    ap.add_argument("--kana_ratio", type=float, default=0.2,
                    help="max kana share of the kana+Han characters for an episode to count as Chinese")
    ap.add_argument("--min_chars", type=int, default=50, help="drop episodes shorter than this after cleaning")
    ap.add_argument("--keep_duplicates", action="store_true", help="keep exact-duplicate episodes")
    ap.add_argument("--max_docs", type=int, default=None, help="stop after this many source records")
    args = ap.parse_args()

    t0 = time.time()
    cc = None
    if args.t2s:
        from opencc import OpenCC

        cc = OpenCC("t2s")
    seen = set()
    read = kept = 0
    chars = in_chars = 0
    drop = dict(empty=0, short=0, lang=0, dup=0)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for spec in args.source:
            for raw in iter_texts(spec, args.max_docs):
                read += 1
                in_chars += len(raw)
                text = clean_ass(raw, args.kana_ratio if args.zh_only else None)
                if cc is not None:
                    text = cc.convert(text) if text else text
                digest = None if args.keep_duplicates else hashlib.blake2b(text.encode(), digest_size=8).digest()
                if not text:
                    drop["empty"] += 1
                elif len(text) < args.min_chars:
                    drop["short"] += 1
                elif args.zh_only and not is_chinese(text, args.kana_ratio):
                    drop["lang"] += 1
                elif digest is not None and digest in seen:
                    drop["dup"] += 1
                else:
                    if digest is not None:
                        seen.add(digest)
                    f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    kept += 1
                    chars += len(text)
                if read % 5000 == 0:
                    print(f"{read} records, kept {kept}, {chars / 1e9:.3f}B chars, "
                          f"{in_chars / (time.time() - t0) / 1e6:.0f}M raw chars/s", flush=True)
    print(f"{args.out}: {kept}/{read} episodes, {chars / 1e9:.3f}B chars "
          f"(raw {in_chars / 1e9:.3f}B, {100 * chars / max(in_chars, 1):.1f}%), dropped {drop}, "
          f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
