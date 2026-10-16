#!/usr/bin/env python
"""Live terminal dashboard for a training run (no dependencies beyond the standard library).

    python tools/watch_training.py --dir runs/pretrain          # refresh in place until Ctrl-C
    python tools/watch_training.py --dir runs/sft --once        # one snapshot, for logs / CI
    python tools/watch_training.py --dir runs/pretrain --interval 5 --events 20

Reads the structured streams `scripts/train.py` writes (`metrics.jsonl`, `health.jsonl`) and falls back
to the older `log.jsonl` for runs started before monitoring existed.  It only ever reads: training is not
affected by whether a dashboard is attached, and a partially flushed line is ignored until it completes.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hvlm.monitor import read_records, render_dashboard, tail_records  # noqa: E402

CLEAR = "\033[2J\033[H"


def resolve(run_dir):
    if os.path.isdir(run_dir):
        metrics = os.path.join(run_dir, "metrics.jsonl")
        if not os.path.exists(metrics):
            metrics = os.path.join(run_dir, "log.jsonl")
        return metrics, os.path.join(run_dir, "health.jsonl")
    return run_dir, os.path.join(os.path.dirname(run_dir), "health.jsonl")


def snapshot(metrics_path, health_path, *, events, width, color):
    records = read_records(metrics_path)
    evs = tail_records(health_path, events) if os.path.exists(health_path) else []
    return render_dashboard(records, evs, width=width, run=os.path.dirname(metrics_path) or metrics_path,
                            color=color)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/pretrain", help="run directory (out_dir), or a metrics.jsonl path")
    ap.add_argument("--interval", type=float, default=2.0, help="refresh period in seconds")
    ap.add_argument("--once", action="store_true", help="print one snapshot and exit")
    ap.add_argument("--events", type=int, default=20, help="how many recent health events to read")
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--no-color", action="store_true", help="plain text (also the default when not a tty)")
    a = ap.parse_args()

    metrics_path, health_path = resolve(a.dir)
    if not os.path.exists(metrics_path):
        raise SystemExit(f"no metrics at {metrics_path}: is the run writing to {a.dir}?")
    color = (not a.no_color) and sys.stdout.isatty()
    if a.once:
        print(snapshot(metrics_path, health_path, events=a.events, width=a.width, color=color))
        return 0
    try:
        while True:
            print(CLEAR + snapshot(metrics_path, health_path, events=a.events, width=a.width, color=color),
                  flush=True)
            time.sleep(max(a.interval, 0.2))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
