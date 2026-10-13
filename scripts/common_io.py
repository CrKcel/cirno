"""Text-source iteration shared by train_tokenizer.py and prepare_data.py.

A *source spec* is one of
    jsonl:<glob>:<text_field>            e.g.  jsonl:data/zh/*.jsonl:text
    parquet:<glob>:<text_field>          e.g.  parquet:data/fineweb/*.parquet:text
    txt:<glob>                           one document per file
    hf:<dataset>:<field>[:<config>[:<split>]]   streamed from the Hugging Face Hub,
                                         e.g.  hf:HuggingFaceFW/fineweb-edu:text:sample-10BT:train
"""
import glob
import json


def iter_texts(spec, max_docs=None):
    kind, _, rest = spec.partition(":")
    n = 0

    def done():
        return max_docs is not None and n >= max_docs

    if kind in ("jsonl", "json"):
        pattern, _, field = rest.rpartition(":")
        for path in sorted(glob.glob(pattern)):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    t = json.loads(line).get(field)
                    if t:
                        yield t
                        n += 1
                        if done():
                            return
    elif kind == "parquet":
        import pyarrow.parquet as pq

        pattern, _, field = rest.rpartition(":")
        for path in sorted(glob.glob(pattern)):
            for batch in pq.ParquetFile(path).iter_batches(columns=[field], batch_size=1024):
                for t in batch.column(0).to_pylist():
                    if t:
                        yield t
                        n += 1
                        if done():
                            return
    elif kind == "txt":
        for path in sorted(glob.glob(rest)):
            with open(path, encoding="utf-8") as f:
                t = f.read()
            if t:
                yield t
                n += 1
                if done():
                    return
    elif kind == "hf":
        from datasets import load_dataset

        parts = rest.split(":")
        name, field = parts[0], parts[1]
        config = parts[2] if len(parts) > 2 and parts[2] else None
        split = parts[3] if len(parts) > 3 else "train"
        ds = load_dataset(name, config, split=split, streaming=True)
        for ex in ds:
            t = ex.get(field)
            if t:
                yield t
                n += 1
                if done():
                    return
    else:
        raise ValueError(f"unknown source kind '{kind}' in spec '{spec}'")
