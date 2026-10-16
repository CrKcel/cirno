#!/usr/bin/env python
"""Compare eval reports across checkpoints/stages: one row per metric, one column per run.

    python tools/eval_table.py runs/eval/*/report.json
    python tools/eval_table.py runs/eval/sft/report.json --baseline runs/eval/pretrain/report.json
    python tools/eval_table.py runs/eval/*/report.json --filter tone,accuracy,ratio

Reads report.json only, so it needs no torch and can run anywhere the reports landed.  Runs evaluated with
a different tokenizer are not comparable (the reply-length numbers are in tokens), so the table refuses to
mix them unless --force is given.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load(paths):
    reports = []
    for path in paths:
        if not os.path.exists(path):
            raise SystemExit(f"no report at {path}")
        with open(path, encoding="utf-8") as f:
            reports.append(json.load(f))
    return reports


def metric_rows(reports):
    rows = []
    for report in reports:
        for set_name, block in (report.get("sets") or {}).items():
            for key, value in (block.get("metrics") or {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    rows.append((f"{set_name}.{key}", report, value))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+", help="report.json files (runs/eval/<tag>/report.json)")
    ap.add_argument("--baseline", default=None, help="report.json to show deltas against")
    ap.add_argument("--filter", default=None, help="comma-separated substrings a metric path must contain")
    ap.add_argument("--force", action="store_true", help="compare reports that used different tokenizers")
    a = ap.parse_args()

    reports = load(a.reports)
    tags = [r.get("tag") or os.path.basename(os.path.dirname(str(r.get("model", "")))) for r in reports]
    tokenizers = {r.get("tokenizer") for r in reports}
    if len(tokenizers) > 1 and not a.force:
        raise SystemExit(f"reports used different tokenizers {sorted(tokenizers)}: reply lengths are not "
                         f"comparable (--force to print anyway)")

    baseline = load([a.baseline])[0] if a.baseline else None
    names = sorted({row[0] for row in metric_rows(reports + ([baseline] if baseline else []))})
    if a.filter:
        keep = tuple(x.strip() for x in a.filter.split(",") if x.strip())
        names = [n for n in names if any(k in n for k in keep)]

    def cell(report, name):
        set_name, _, key = name.partition(".")
        block = ((report or {}).get("sets") or {}).get(set_name) or {}
        value = (block.get("metrics") or {}).get(key)
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    head = ["metric"] + tags + (["Δ " + (baseline.get("tag") or "baseline")] if baseline else [])
    rows = []
    for name in names:
        cells = []
        for report in reports:
            value = cell(report, name)
            cells.append("-" if value is None else f"{value:.4g}")
        if baseline:
            base = cell(baseline, name)
            values = [cell(r, name) for r in reports]
            delta = next((v - base for v in values if v is not None and base is not None), None)
            cells.append("-" if delta is None else f"{delta:+.4g}")
        rows.append([name] + cells)

    widths = [max(len(row[i]) for row in [head] + rows) for i in range(len(head))]
    line = "  ".join(h.ljust(w) for h, w in zip(head, widths))
    print(line)
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)))
    gates = [(r.get("tag"), (r.get("gates") or {}).get("passed")) for r in reports]
    print()
    print("gate: " + ", ".join(f"{tag}={'PASS' if ok else 'FAIL' if ok is False else 'n/a'}"
                               for tag, ok in gates))
    return 0


if __name__ == "__main__":
    sys.exit(main())
