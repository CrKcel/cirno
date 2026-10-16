#!/usr/bin/env python
"""Torch-free checks of hvlm/monitor.py (health policy, metric streams, dashboard) and the recipe allocator."""
import json
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hvlm.monitor import (EMA, HealthPolicy, MetricsLog, dashboard_summary,  # noqa: E402
                          normalize_eval, read_records, render_dashboard, tail_records)
from plan_mix import plan_bins, resolve_plan  # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")


# ---- EMA -------------------------------------------------------------------------------------
e = EMA(0.5)
for x in (1.0, 3.0, 5.0):
    e.update(x)
check("EMA seeds on the first sample and then blends", abs(e.value - 3.5) < 1e-9, f"({e.value})")

# ---- loss-spike detection and learning-rate backoff -------------------------------------------
p = HealthPolicy(intervene=True, min_steps=5, spike_sigma=3.0, spike_ratio=1.5,
                 lr_backoff_factor=0.5, backoff_steps=10, backoff_recover=0.1)
steady = [dict(step=i, t=0.0, loss=5.0, gnorm=1.0, tok_s=1000.0) for i in range(1, 20)]
events = [ev for rec in steady for ev in p.observe(rec)]
check("a steady loss raises no alarm", not events, f"({len(events)} events)")
events = p.observe(dict(step=21, t=0.0, loss=40.0, gnorm=1.0, tok_s=1000.0))
check("a 8x loss spike is reported", any(ev["code"] == "loss_spike" for ev in events))
check("the spike backs the learning rate off", abs(p.lr_scale(21) - 0.5) < 1e-9, f"({p.lr_scale(21)})")
check("a second spike halves it again", abs(p.observe(dict(step=22, t=0.0, loss=80.0, gnorm=1.0,
                                                           tok_s=1000.0))[0].get("lr_scale", 0) - 0.25) < 1e-9)
check("the backoff is held, not released immediately", p.lr_scale(30) == 0.25, f"({p.lr_scale(30)})")
recovered = [p.lr_scale(s) for s in range(33, 60)]
check("the backoff is released gradually afterwards",
      recovered[0] > 0.25 and recovered[-1] == 1.0 and all(b >= a for a, b in zip(recovered, recovered[1:])),
      f"({recovered[:6]} ... {recovered[-1]})")

p2 = HealthPolicy(intervene=False, min_steps=5, spike_sigma=3.0, spike_ratio=1.5)
for rec in steady:
    p2.observe(rec)
ev = p2.observe(dict(step=21, t=0.0, loss=40.0, gnorm=1.0, tok_s=1000.0))
check("--no_intervention still reports the spike", bool(ev) and p2.lr_scale(21) == 1.0)

# ---- gradient / non-finite / memory / throughput ----------------------------------------------
p3 = HealthPolicy(min_steps=3, clip=1.0, gnorm_spike_factor=20.0, throughput_ratio=0.5,
                  throughput_patience=2, mem_frac=0.9, budget_gb=8.0)
for i in range(1, 6):
    p3.observe(dict(step=i, t=0.0, loss=5.0, gnorm=1.0, tok_s=1000.0, mem_gb=4.0))
codes = {ev["code"] for ev in p3.observe(dict(step=6, t=0.0, loss=5.0, gnorm=50.0, tok_s=1000.0, mem_gb=7.8))}
check("a grad-norm explosion is reported", "grad_spike" in codes, f"({codes})")
check("memory above the budget share is reported", "memory_pressure" in codes)
codes = set()
for i in range(7, 11):
    codes |= {ev["code"] for ev in p3.observe(dict(step=i, t=0.0, loss=5.0, gnorm=1.0, tok_s=100.0, mem_gb=4.0))}
check("a sustained throughput drop is reported after patience", "throughput_drop" in codes, f"({codes})")
check("a non-finite gradient is critical",
      any(ev["level"] == "critical" for ev in HealthPolicy(min_steps=1).observe(dict(step=1, t=0.0, loss=5.0, nonfinite=True))))

# ---- validation plateau / early stop -----------------------------------------------------------
p4 = HealthPolicy(val_patience=3, intervene=True)
seen = []
seen += p4.observe_eval(100, {"val": {"loss": 4.0, "ppl": 55.0}})
seen += p4.observe_eval(200, {"val": {"loss": 4.2, "ppl": 66.0}})
seen += p4.observe_eval(300, {"val": {"loss": 4.3, "ppl": 73.0}})
fine = p4.observe_eval(400, {"val": {"loss": 3.9, "ppl": 49.0}})
check("a new best val loss clears the counter", not fine and p4.best_val() == 3.9)
for step in (500, 600):
    p4.observe_eval(step, {"val": {"loss": 4.1, "ppl": 60.0}})
last = p4.observe_eval(700, {"val": {"loss": 4.1, "ppl": 60.0}})
check("three evaluations without improvement stop the run",
      any(ev["action"] == "stop" for ev in last), f"({[ev['code'] for ev in last]})")
check("the first two are a warning and then silence",
      sum(1 for ev in seen if ev["action"] == "stop") == 0)

# ---- metric stream ----------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as td:
    path = os.path.join(td, "metrics.jsonl")
    with MetricsLog(path) as log:
        log.write(dict(step=1, loss=9.0, lr=1e-3))
        log.write(dict(step=2, loss=8.0, lr=1e-3))
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"step": 3, "loss"')  # a writer caught mid-flush must not break the dashboard
    check("read_records ignores a partial trailing line", len(read_records(path)) == 2)
    check("tail_records slices the head off", [r["step"] for r in tail_records(path, 1)] == [2])
    recs = read_records(path) + [dict(step=3, loss=7.0, lr=1e-3, lr_scale=0.5, gnorm=1.0, tok_s=1000.0,
                                      mem_gb=4.0, stage="pretrain", phase="B", total_steps=10,
                                      tokens_total=100, budget_tokens=1000) for _ in range(1)]
    text = dashboard_summary(recs)
    check("the summary reports step progress and the latest loss",
          "3/10" in text and "7.0000" in text, f"({text.splitlines()[1] if len(text.splitlines()) > 1 else text})")
    live = render_dashboard(recs, [dict(level="warn", step=3, code="loss_spike", message="too big")], color=False)
    check("the live view renders alerts", "loss_spike" in live and "step 3/10" in live)
    check("an empty run renders without raising", "waiting" in render_dashboard([]))
check("eval records of either schema normalise", normalize_eval(dict(eval={"a": {"loss": 1.0, "ppl": 2.0},
                                                                        "b": 3.0}))["b"]["loss"] == 3.0)

# ---- recipe allocator --------------------------------------------------------------------------
plan = dict(
    budget_tokens=1000, schedule=dict(peak_lr=1e-3),
    bins={"big": dict(dir="/nonexistent/big", expected_tokens=800, allocate=800, val_tokens=0),
          "small": dict(dir="/nonexistent/small", expected_tokens=200, allocate=200, val_tokens=0),
          "replay": dict(dir="/nonexistent/replay", expected_tokens=100, allocate=150, val_tokens=0)},
    pretrain=[dict(name="A", tokens=300, shares={"big": 0.5, "small": 0.5}),
              dict(name="B"),
              dict(name="C", tokens=300, fixed={"big": 0.5}, rest=["small"], absorber="big")],
    align={}, sft={})
bins = plan_bins(plan)
resolved = resolve_plan(plan, bins, warn=lambda *a, **k: None)
alloc = {n: b["allocated"] for n, b in resolved["bins"].items()}
check("every phase is allocated its declared budget",
      all(abs(resolved["phases"][p]["tokens"] - want) < 1 for p, want in (("A", 300), ("C", 300))),
      f"({[resolved['phases'][p]['tokens'] for p in 'AC']})")
check("the remainder phase takes the rest of the budget",
      abs(sum(alloc.values()) - 1000) < 1, f"({alloc})")
check("no bin is asked for more than its allocation", all(alloc[n] <= bins[n]["allocate"] + 1 for n in bins),
      f"({alloc})")
check("a bin read partly is not tokenised past the cut",
      resolved["bins"]["big"]["prepare_max_tokens"] == 800)
check("the remainder phase never goes negative", all(v >= 0 for v in alloc.values()))
check("phase data strings are normalised weights",
      abs(sum(float(x.split(":")[1]) for x in resolved["phases"]["A"]["data"]) - 1.0) < 1e-3,
      f"({resolved['phases']['A']['data']})")
check("cumulative offsets chain the phases",
      resolved["phases"]["B"]["cum_start"] == 300 and abs(resolved["phases"]["C"]["cum_start"] - 700) < 1,
      f"({[(p, resolved['phases'][p]['cum_start']) for p in 'ABC']})")

print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
