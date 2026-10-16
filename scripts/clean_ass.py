#!/usr/bin/env python
"""Reduce an .ass/.ssa subtitle corpus to plain dialogue text (one jsonl record per ~4000-character scene).

`chenrm/anime-subtitles` stores whole subtitle scripts, and the format is most of that: script
headers, style tables, karaoke/position overrides, timing fields, and - in typeset episodes - ASS
vector drawing commands.  Measured on the local snapshot, 11.43B raw characters reduce to 0.282B of
dialogue text (2.5%): training on the raw field would spend almost the whole subtitle budget on markup.

ASS quirks this handles:
- a `Dialogue:` line is
      Dialogue: layer,start,end,style,name,marginL,marginR,marginV,effect,text
  exactly nine comma-separated fields, and `text` may itself contain commas, hence `split(",", 9)`;
- `{\\pos(..)}` / `{\\k50}` override blocks are stripped, `\\N` / `\\n` / `\\h` become line breaks;
- typesetting draws shapes through `{\\p1}`: the override is stripped and the path itself - bare
  coordinates interleaved with the `m`/`l`/`b`/`c`/`s`/`p` operators - is left behind as dialogue text
  (over 80% of the raw characters on this corpus), so path lines are dropped;
- the same line is often emitted once per karaoke/scroll frame, so a repeated event is emitted once;
- vertical lyric lines leave runs of one/two-character lines, dropped as a run (`--short_run`).

The collection mixes Chinese subtitles with Japanese raw/karaoke scripts, so `--zh-only` drops the
Japanese episodes and, inside the Chinese ones, the kana-only lines (the OP/ED songs).  `--t2s`
converts traditional to simplified: roughly half of the fansub groups release traditional, and keeping
both scripts in the chat bucket teaches the model to answer in traditional.  Exact-duplicate episodes
(re-releases) are dropped by default.

An episode is one document otherwise, and `prepare_data --max_doc_chars 200000` keeps only a
document's head; the cleaned text is cut into ~4000-character scene chunks at line boundaries, which
is also the granularity the other chat/novel buckets use.

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
# A drawing path is made only of coordinates, separators and the path commands; requiring three numbers
# keeps a short line such as `1 2 3` from being read as text while never matching prose (which holds no
# coordinates and no path command).
_PATH_CHARS = re.compile(r"^[\smnlbscp0-9.,+\-]+$", re.I)
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
# Decorative lines: a subtitle separator (`==== ====OP JP==== ====`) or a run of one punctuation mark
# (`————`, `-----`).  Both are checked by hand rather than by regex: the natural regexes for them
# (`(?:\s*\1){2,}` on drawing-length lines) backtrack exponentially.
_DECOR_CHARS = frozenset("=-–—─_~*#·※.。")
CHUNK_CHARS = 4000


def is_drawing(line):
    """True for an ASS vector path (`m 0 0 l 24 -1 b 34 -2 ...`) left behind by `{\\p1}`."""
    if not _PATH_CHARS.match(line):
        return False
    numbers = _NUMBER.findall(line)
    return len(numbers) >= 3 and sum(len(n) for n in numbers) / len(line) >= 0.4


def is_furniture(line):
    """True for a fansub separator or a line that is one punctuation mark repeated at least three times."""
    stripped = line.strip()
    if len(stripped) >= 4 and stripped[0] == "=" and stripped[-1] == "=":
        return True
    packed = "".join(line.split())
    return len(packed) >= 3 and len(set(packed)) == 1 and packed[0] in _DECOR_CHARS


def clean_lines(text, kana_ratio=None):
    """Dialogue lines of one .ass script in order: markup, drawings, furniture and repeats removed.

    With `kana_ratio`, lines above that kana share are dropped as well: an otherwise Chinese episode
    carries its Japanese OP/ED song as dialogue lines.  A repeat is detected on the whole `Dialogue:`
    event, because karaoke and scrolling credits re-emit the same text under a new timestamp.
    """
    lines, prev = [], None
    for raw in text.splitlines():
        if not raw.startswith("Dialogue:"):
            continue
        fields = raw.split(",", 9)      # `text` may hold commas of its own, hence the maxsplit
        if len(fields) < 10:
            continue
        body = _OVERRIDE.sub("", fields[9])
        for esc in ("\\N", "\\n", "\\h"):
            body = body.replace(esc, "\n")
        event = []
        for piece in body.splitlines():
            line = piece.strip()
            if not line or is_drawing(line) or is_furniture(line):
                continue
            if kana_ratio is not None:
                kana = len(_KANA.findall(line))
                if kana and kana / (kana + len(_HAN.findall(line))) > kana_ratio:
                    continue
            event.append(line)
        event_text = "\n".join(event)
        if not event or event_text == prev:
            continue
        lines.extend(event)
        prev = event_text
    return lines


def drop_short_runs(lines, min_run, max_len=2):
    """Drop runs of `min_run` consecutive lines of at most `max_len` characters.

    Vertical lyrics and karaoke leave one character per line (`你\\n的\\n世\\n界…`); an exchange of short
    dialogue lines rarely reaches `min_run` in a row, so the run length is what separates the two."""
    if min_run <= 0:
        return lines
    out, i = [], 0
    while i < len(lines):
        j = i
        while j < len(lines) and len(lines[j]) <= max_len:
            j += 1
        if j - i >= min_run:
            i = j
            continue
        end = max(j, i + 1)     # a run shorter than min_run, or a line too long to be in one
        out.extend(lines[i:end])
        i = end
    return out


def chunk_lines(lines, target=CHUNK_CHARS):
    """Group lines into documents of about `target` characters, breaking between lines."""
    buf, size = [], 0
    for line in lines:
        if size and size + len(line) + 1 > target:
            yield "\n".join(buf)
            buf, size = [], 0
        buf.append(line)
        size += len(line) + 1
    if buf:
        yield "\n".join(buf)


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
    ap.add_argument("--short_run", type=int, default=6,
                    help="drop a run of this many consecutive 1-2 character lines as karaoke/lyrics "
                         "(0 disables the filter)")
    ap.add_argument("--chunk_chars", type=int, default=CHUNK_CHARS,
                    help="characters per output document, cut at line boundaries (0 writes one per episode)")
    ap.add_argument("--keep_duplicates", action="store_true", help="keep exact-duplicate episodes")
    ap.add_argument("--max_docs", type=int, default=None, help="stop after this many source records")
    args = ap.parse_args()

    t0 = time.time()
    cc = None
    if args.t2s:
        from opencc import OpenCC

        cc = OpenCC("t2s")
    seen = set()
    read = kept = docs = 0
    chars = in_chars = 0
    drop = dict(empty=0, short=0, lang=0, dup=0)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        for spec in args.source:
            for raw in iter_texts(spec, args.max_docs):
                read += 1
                in_chars += len(raw)
                lines = drop_short_runs(clean_lines(raw, args.kana_ratio if args.zh_only else None),
                                        args.short_run)
                text = "\n".join(lines)
                if cc is not None and text:
                    text = cc.convert(text)
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
                    episodes = [text] if not args.chunk_chars else chunk_lines(text.splitlines(),
                                                                              args.chunk_chars)
                    for chunk in episodes:
                        f.write(json.dumps({"text": chunk}, ensure_ascii=False) + "\n")
                        docs += 1
                        chars += len(chunk)
                    kept += 1
                if read % 5000 == 0:
                    print(f"{read} records, kept {kept}, {docs} documents, {chars / 1e6:.1f}M chars, "
                          f"{in_chars / (time.time() - t0) / 1e6:.0f}M raw chars/s", flush=True)
    os.replace(tmp, args.out)
    print(f"{args.out}: {kept}/{read} episodes -> {docs} documents, {chars / 1e9:.3f}B chars "
          f"(raw {in_chars / 1e9:.3f}B, {100 * chars / max(in_chars, 1):.1f}%), dropped {drop}, "
          f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
