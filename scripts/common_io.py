"""Record iteration shared by train_tokenizer.py, prepare_data.py and the align / sft data loader.

A *source spec* is one of
    jsonl:<glob>[:<field>]               e.g.  jsonl:data/zh/*.jsonl:text
    parquet:<glob>[:<field>]             e.g.  parquet:data/fineweb/*.parquet:text
    txt:<glob>                           one document per file
    hf:<dataset>[:<field>[:<config>[:<split>]]]   streamed from the Hugging Face Hub,
                                         e.g.  hf:HuggingFaceFW/fineweb-edu:text:sample-10BT:train

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


def _split_file_spec(rest):
    """`<glob>[:<field>]` -> (glob, field); the glob may contain no colon at all."""
    pattern, sep, field = rest.rpartition(":")
    return (pattern, field) if sep and pattern else (rest, WHOLE)


def _iter_source(spec):
    """Yield (record, field) for every document of `spec`."""
    kind, _, rest = spec.partition(":")
    if kind in ("jsonl", "json"):
        pattern, field = _split_file_spec(rest)
        for path in sorted(glob.glob(pattern)):
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
        for path in sorted(glob.glob(pattern)):
            for batch in pq.ParquetFile(path).iter_batches(columns=cols, batch_size=1024):
                for rec in batch.to_pylist():
                    if rec:
                        yield rec, field
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
        for path in sorted(glob.glob(rest)):
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
