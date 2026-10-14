"""Record iteration shared by train_tokenizer.py, prepare_data.py and the align / sft data loader.

A *source spec* is one of
    jsonl:<glob>[:<field>]               e.g.  jsonl:data/zh/*.jsonl:text
    parquet:<glob>[:<field>]             e.g.  parquet:data/fineweb/*.parquet:text
                                               parquet:hf://datasets/org/name@refs/convert/parquet/base/train/*.parquet:dialog
    txt:<glob>                           one document per file
    hf:<dataset>[:<field>[:<config>[:<split>]]]   streamed from the Hugging Face Hub,
                                         e.g.  hf:HuggingFaceFW/fineweb-edu:text:sample-10BT:train

`parquet:` accepts remote `hf://` globs, which is how datasets that only ship a loading script (LCCC,
DailyDialog, EmpatheticDialogues) stay readable once `datasets` refuses scripts, and how a non-config
subfolder (Fineweb-Edu-Chinese's score bands) is addressed.

<field> says what becomes the document text (default: the whole record):
    text            one column of the record
    -               the whole record, every text it holds
    a.b             a dotted path into nested records (MNBVC forum rows: 回复.回复)
    a+b             several fields concatenated in order (MNBVC forum rows: 主题+回复.回复)
List values are joined with newlines, so a column of utterances (LCCC "dialog", SODA "dialogue",
DailyDialog "dialog") reads as one document per conversation with no conversion step in between.

`iter_records()` yields the raw dicts instead of text - that is what align / sft data wants.
A trailing `@N` limits a source to its first N documents; `split_limit()` strips the suffix off.
"""
import glob
import json
import re
from collections.abc import Mapping

WHOLE = "-"


def split_limit(spec):
    """`hf:org/name:text:train@300000` -> ("hf:org/name:text:train", 300000); no suffix -> (spec, None)."""
    head, sep, tail = spec.rpartition("@")
    return (head, int(tail)) if sep and tail.isdigit() else (spec, None)


def _select(rec, parts):
    """Walk a dotted path through nested dicts/lists, mapping the same path over list elements."""
    if rec is None or not parts:
        return rec
    if isinstance(rec, Mapping):
        return _select(rec.get(parts[0]), parts[1:])
    if isinstance(rec, (list, tuple)):
        vals = [v for v in (_select(x, parts) for x in rec) if v not in (None, "", [], {})]
        return vals or None
    return None  # a leaf before the path is exhausted: no such field


def _to_text(value):
    """Flatten whatever a column holds (str / list / dict, arbitrarily nested) into one document."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        value = list(value.values())
    if isinstance(value, (list, tuple)):
        return "\n".join(t for t in (_to_text(v) for v in value) if t) or None
    return str(value)


def field_text(rec, field):
    """Document text selected by `field` (see the module docstring), or None when the record lacks it."""
    if field == WHOLE:
        return _to_text(rec)
    chunks = (_to_text(_select(rec, f.split("."))) for f in field.split("+"))
    return "\n".join(t for t in chunks if t) or None


# CJK blocks whose neighbours a word-segmented corpus has separated with spaces (see desegment_cjk):
# ideographs, CJK punctuation (、。「」), kana, compatibility ideographs and fullwidth forms (，！？).
_CJK = ((0x2E80, 0x2FFF), (0x3000, 0x303F), (0x3040, 0x30FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF),
        (0xF900, 0xFAFF), (0xFF00, 0xFFEF), (0x20000, 0x3FFFF))
# A run of spaces (or ideographic spaces) with a CJK character on either side.  Regex, not a character
# loop: this runs over every document of a multi-GB corpus.
_CJK_CLASS = "".join(f"\\U{lo:08x}-\\U{hi:08x}" for lo, hi in _CJK)
_CJK_SPACE = re.compile(r"(?<=[%s])[ \u3000]+|[ \u3000]+(?=[%s])" % (_CJK_CLASS, _CJK_CLASS))


def desegment_cjk(text):
    """Undo the spaces a word-segmented corpus puts between Chinese tokens.

    LCCC ships `dialog` pre-segmented (`你 去 那儿 竟然 不喊 我 生气 了 ，`), and training on that
    teaches the model to answer with spaces between characters.  A space run is dropped only when a
    CJK character sits on one side of it, so `SEED.24 小时` -> `SEED.24小时` and `我用 SEED` -> `我用SEED`
    while spaces inside English/digit runs survive and newlines keep separating turns."""
    return _CJK_SPACE.sub("", text) if (" " in text or "\u3000" in text) else text


def _split_file_spec(rest):
    """`<glob>[:<field>]` -> (glob, field); the glob may contain no colon at all.

    A remote spec keeps its scheme: in `hf://datasets/org/name@rev/x/*.parquet:text` the `hf` in front
    of `://` is not the glob, so the colon split has to happen after the scheme."""
    if "://" in rest:
        scheme, _, tail = rest.partition("://")
        pattern, sep, field = tail.rpartition(":")
        return (f"{scheme}://{pattern}", field) if sep and pattern else (rest, WHOLE)
    pattern, sep, field = rest.rpartition(":")
    return (pattern, field) if sep and pattern else (rest, WHOLE)


def _no_match(pattern, what):
    """A glob that matches nothing is a typo, not an empty corpus: say so instead of yielding nothing.

    This has bitten twice already - `@refs/convert/parquet/<config>/train/*` for a dataset that only has
    `partial-train`, and a misspelled local directory both tokenise to an empty bin with exit status 0."""
    raise FileNotFoundError(f"{what} glob matched no files: {pattern!r}")


def _iter_source(spec):
    """Yield (record, field) for every document of `spec`."""
    kind, _, rest = spec.partition(":")
    if kind in ("jsonl", "json"):
        pattern, field = _split_file_spec(rest)
        paths = sorted(glob.glob(pattern))
        if not paths:
            _no_match(pattern, "jsonl")
        for path in paths:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        if rec:
                            yield rec, field
    elif kind == "parquet":
        import pyarrow.parquet as pq

        pattern, field = _split_file_spec(rest)
        # read only the columns the field expression needs (a nested path needs only its top level)
        cols = None if field == WHOLE else sorted({f.split(".")[0] for f in field.split("+")})
        # `hf://` globs reach the Hub's auto-converted parquet branch (`@refs/convert/parquet/...`), which
        # is what keeps script-only datasets (LCCC, DailyDialog, EmpatheticDialogues) loadable under
        # datasets >= 4, and non-config subfolders such as Fineweb-Edu-Chinese's score bands.
        fs = None
        if "://" in pattern:
            import fsspec

            scheme, _, remote = pattern.partition("://")
            fs = fsspec.filesystem(scheme)
            paths = sorted(fs.glob(remote))
        else:
            paths = sorted(glob.glob(pattern))
        if not paths:
            _no_match(pattern, "parquet")
        for path in paths:
            handle = fs.open(path, "rb") if fs is not None else path
            try:
                for batch in pq.ParquetFile(handle).iter_batches(columns=cols, batch_size=1024):
                    for rec in batch.to_pylist():
                        if rec:
                            yield rec, field
            finally:
                if fs is not None:
                    handle.close()
    elif kind == "hf":
        from datasets import load_dataset

        parts = rest.split(":")
        name = parts[0]
        field = parts[1] if len(parts) > 1 and parts[1] else WHOLE
        config = parts[2] if len(parts) > 2 and parts[2] else None
        split = parts[3] if len(parts) > 3 else "train"
        for rec in load_dataset(name, config, split=split, streaming=True):
            yield dict(rec), field
    elif kind == "txt":
        paths = sorted(glob.glob(rest))
        if not paths:
            _no_match(rest, "txt")
        for path in paths:
            with open(path, encoding="utf-8") as f:
                text = f.read()
            if text:
                yield {"text": text}, "text"
    else:
        raise ValueError(f"unknown source kind '{kind}' in spec '{spec}'")


def iter_texts(spec, max_docs=None):
    """One document string per record of `spec`."""
    n = 0
    for rec, field in _iter_source(spec):
        text = field_text(rec, field)
        if text:
            yield text
            n += 1
            if max_docs is not None and n >= max_docs:
                return


def iter_records(spec, max_docs=None):
    """Raw dict records of `spec` (align / sft input; normalize_messages() maps them onto chat roles)."""
    n = 0
    for rec, _field in _iter_source(spec):
        yield rec
        n += 1
        if max_docs is not None and n >= max_docs:
            return
