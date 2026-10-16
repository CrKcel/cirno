"""Training health monitoring: metric aggregation, anomaly policy and the on-disk metric streams.

Torch-free on purpose.  ``scripts/train.py`` feeds plain floats in, ``tools/watch_training.py`` reads the
same files back out, and ``tools/check_monitor.py`` exercises both without a GPU.  The two file schemas are
the contract between the trainer and the dashboard:

``metrics.jsonl`` - one object per log interval, plus one per evaluation::

    {"t": 1729.4, "step": 10, "stage": "pretrain", "phase": "B", "tokens": 5242880,
     "tokens_total": 5242880, "total_steps": 8220, "budget_tokens": 5390000000,
     "loss": 6.31, "lm": 6.11, "mtp": 6.5, "lr": 0.001, "lr_scale": 1.0, "gnorm": 1.2,
     "grad_scale": null, "tok_s": 12000.0, "step_s": 21.8, "mem_gb": 5.1, "elapsed_s": 220.0}
    {"t": ..., "step": 500, "eval": {"chat": {"loss": 4.1, "ppl": 60.3}}, "best_val": 4.1}

``health.jsonl`` - one object per anomaly or intervention::

    {"t": ..., "step": 40, "level": "warn", "code": "loss_spike", "message": "...",
     "action": "lr_backoff", "value": 9.1, "threshold": 7.4, "lr_scale": 0.5}

The policy distinguishes *observation* from *intervention*: every anomaly is written to health.jsonl, but
only ``lr_backoff`` and ``stop`` change what the trainer does, and both can be switched off
(``--no_intervention``) so an instrumented run stays reproducible when it has to be.
"""
import json
import math
import os
import time

ACTIONS = ("none", "lr_backoff", "stop")
# Sparkline ramp; index = value between the minimum and maximum of the plotted window.
_SPARK = "▁▂▃▄▅▆▇█"


def json_safe(value):
    """Non-finite floats become null: `json.dumps` would otherwise emit NaN, which is not valid JSON.

    A diverging run really does record a NaN grad norm or loss - that record is the evidence - so the
    streams stay parseable by anything stricter than Python's own `json` instead of dropping the row."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


class EMA:
    """Exponential moving average; the first sample seeds it, so early values are not dragged towards 0."""

    def __init__(self, alpha=0.05):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self.n = 0
        self.value = 0.0

    def update(self, x):
        self.n += 1
        self.value = float(x) if self.n == 1 else (1 - self.alpha) * self.value + self.alpha * float(x)
        return self.value

    def reset(self):
        self.n, self.value = 0, 0.0


class HealthPolicy:
    """Turns a stream of per-step metrics into anomaly events and, optionally, interventions.

    Detection is deliberately conservative and scale-free: a loss spike has to clear both a multiple of
    the recent standard deviation *and* a multiple of the recent mean, so the ordinary noise of a
    well-behaved run never trips it.  ``min_steps`` keeps the first records (where the EMA is still
    meaningless) from being judged at all.

    Interventions are limited to two, both reversible:
      * a learning-rate backoff, held for ``backoff_steps`` and then released ``backoff_recover`` per step;
      * an early stop after ``val_patience`` evaluations without a new best.
    """

    def __init__(self, *, peak_lr=1e-3, clip=1.0, budget_gb=None,
                 spike_sigma=4.0, spike_ratio=1.5, min_steps=25,
                 gnorm_spike_factor=20.0,
                 throughput_ratio=0.5, throughput_patience=3,
                 mem_frac=0.95,
                 val_patience=0, val_tolerance=0.0, val_min_steps=0,
                 intervene=True, lr_backoff_factor=0.5, backoff_steps=300,
                 min_backoff=0.05, backoff_recover=0.01, alpha=0.05):
        self.peak_lr, self.clip, self.budget_gb = peak_lr, clip, budget_gb
        self.spike_sigma, self.spike_ratio, self.min_steps = spike_sigma, spike_ratio, min_steps
        self.gnorm_spike_factor = gnorm_spike_factor
        self.throughput_ratio, self.throughput_patience = throughput_ratio, throughput_patience
        self.mem_frac = mem_frac
        self.val_patience, self.val_tolerance, self.val_min_steps = val_patience, val_tolerance, val_min_steps
        self.intervene = intervene
        self.lr_backoff_factor, self.backoff_steps = lr_backoff_factor, backoff_steps
        self.min_backoff, self.backoff_recover = min_backoff, backoff_recover

        self._loss = EMA(alpha)
        self._loss_sq = EMA(alpha)
        self._gnorm = EMA(alpha)
        self._tok_s = EMA(alpha)
        self._slow = 0
        self._backoff = 1.0
        self._backoff_until = -1
        self._best_val = None
        self._bad_evals = 0
        self.events = 0
        self.backoffs = 0

    # -- learning-rate intervention -------------------------------------------------------------
    def lr_scale(self, step):
        """Multiplier applied on top of the scheduled lr for `step`; 1.0 when nothing was backed off."""
        if step > self._backoff_until and self._backoff < 1.0:
            self._backoff = min(1.0, self._backoff + self.backoff_recover)
        return self._backoff

    def _apply_backoff(self, step, now):
        self._backoff = max(self.min_backoff, self._backoff * self.lr_backoff_factor)
        self._backoff_until = step + self.backoff_steps
        self.backoffs += 1
        return self._backoff

    @property
    def backoff(self):
        return self._backoff

    def best_val(self):
        return self._best_val

    # -- per-step observation -------------------------------------------------------------------
    def observe(self, rec):
        """Feed one logged interval.  Returns the list of events raised by this record."""
        events = []
        step = int(rec.get("step", 0))
        t = rec.get("t") or time.time()

        if rec.get("nonfinite"):
            events.append(self._event(t, step, "critical", "nonfinite_gradient",
                                      "non-finite gradient norm: the step was skipped", "lr_backoff",
                                      value=rec.get("gnorm"), threshold=self.clip))

        loss = rec.get("loss")
        if loss is not None and math.isfinite(float(loss)):
            loss = float(loss)
            mean, var = self._loss.value, max(self._loss_sq.value - self._loss.value ** 2, 0.0)
            if self._loss.n >= self.min_steps:
                std = math.sqrt(var)
                limit = mean + self.spike_sigma * max(std, 1e-3)
                if loss > limit and loss > self.spike_ratio * mean:
                    events.append(self._event(t, step, "warn", "loss_spike",
                                              f"loss {loss:.4g} above {limit:.4g} (ema {mean:.4g} sd {std:.3g})",
                                              "lr_backoff", value=loss, threshold=limit))
            self._loss.update(loss)
            self._loss_sq.update(loss * loss)

        gnorm = rec.get("gnorm")
        if gnorm is not None and math.isfinite(float(gnorm)):
            gnorm = float(gnorm)
            limit = self.gnorm_spike_factor * self.clip
            if self._gnorm.n >= self.min_steps and gnorm > limit:
                events.append(self._event(t, step, "warn", "grad_spike",
                                          f"grad norm {gnorm:.4g} above {limit:.4g}", "lr_backoff",
                                          value=gnorm, threshold=limit))
            self._gnorm.update(gnorm)

        tok_s = rec.get("tok_s")
        if tok_s is not None and float(tok_s) > 0:
            tok_s = float(tok_s)
            if self._tok_s.n >= self.min_steps:
                if tok_s < self.throughput_ratio * self._tok_s.value:
                    self._slow += 1
                    if self._slow == self.throughput_patience:
                        events.append(self._event(t, step, "warn", "throughput_drop",
                                                  f"throughput {tok_s:.0f} tok/s below "
                                                  f"{self.throughput_ratio:.0%} of the {self._tok_s.value:.0f} tok/s "
                                                  f"running mean for {self._slow} intervals", "none",
                                                  value=tok_s, threshold=self.throughput_ratio * self._tok_s.value))
                else:
                    self._slow = 0
            self._tok_s.update(tok_s)

        mem = rec.get("mem_gb")
        if self.budget_gb and mem is not None and float(mem) > self.mem_frac * self.budget_gb:
            events.append(self._event(t, step, "warn", "memory_pressure",
                                      f"peak {float(mem):.2f} GB above {self.mem_frac:.0%} of the "
                                      f"{self.budget_gb:.2f} GB budget", "none",
                                      value=float(mem), threshold=self.mem_frac * self.budget_gb))

        return self._maybe_intervene(events, step, t)

    def _maybe_intervene(self, events, step, t):
        if not self.intervene:
            return events
        for e in events:
            if e["action"] == "lr_backoff":
                scale = self._apply_backoff(step, t)
                e["lr_scale"] = scale
                e["message"] += f"; lr x{scale:.3g} for {self.backoff_steps} steps"
                break  # one backoff per record, however many anomalies came with it
        return events

    # -- evaluation observation -----------------------------------------------------------------
    def observe_eval(self, step, results, *, tokens=None):
        """Feed one evaluation.  `results` maps loader name to {"loss", "ppl"} (or a bare loss float)."""
        events = []
        if not results:
            return events
        name = sorted(results)[0]
        r = results[name]
        loss = float(r["loss"] if isinstance(r, dict) else r)
        t = time.time()
        if self._best_val is None or loss < self._best_val - self.val_tolerance:
            self._best_val = loss
            self._bad_evals = 0
            return events
        self._bad_evals += 1
        if self.val_patience > 0 and step >= self.val_min_steps:
            if self._bad_evals >= self.val_patience:
                events.append(self._event(t, step, "critical", "val_plateau",
                                          f"{name} loss {loss:.4f} has not improved on best "
                                          f"{self._best_val:.4f} for {self._bad_evals} evaluations", "stop",
                                          value=loss, threshold=self._best_val))
            elif self._bad_evals == self.val_patience - 1:
                events.append(self._event(t, step, "warn", "val_plateau",
                                          f"{name} loss {loss:.4f} worse than best {self._best_val:.4f} for "
                                          f"{self._bad_evals} evaluations", "none",
                                          value=loss, threshold=self._best_val))
        return events

    @staticmethod
    def _event(t, step, level, code, message, action, *, value=None, threshold=None):
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r}")
        return json_safe(dict(t=round(float(t), 3), step=int(step), level=level, code=code, message=message,
                              action=action, value=value, threshold=threshold))


class MetricsLog:
    """Append-only JSONL writer; flushes every record so a dashboard can tail it while training runs."""

    def __init__(self, path):
        self.path = path
        self.f = open(path, "a", encoding="utf-8")

    def write(self, rec):
        self.f.write(json.dumps(json_safe(rec), ensure_ascii=False) + "\n")
        self.f.flush()

    def close(self):
        if not self.f.closed:
            self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ------------------------------------------------------------------------------------------------
# reading side (dashboard / tests)
# ------------------------------------------------------------------------------------------------
def read_records(path):
    """Read a JSONL metric file, ignoring a trailing partial line (a writer may be mid-flush)."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def tail_records(path, n):
    if n <= 0:
        return []
    return read_records(path)[-n:]


def normalize_eval(rec):
    """Return {loader: {"loss", "ppl"}} for either metric schema (`log.jsonl` stores bare losses)."""
    raw = rec.get("eval")
    if not isinstance(raw, dict):
        return {}
    out = {}
    for name, v in raw.items():
        loss = v.get("loss") if isinstance(v, dict) else v
        try:
            loss = float(loss)
        except (TypeError, ValueError):
            loss = float("nan")
        ppl = v.get("ppl") if isinstance(v, dict) else None
        out[name] = {"loss": loss, "ppl": float(ppl) if ppl is not None else math.exp(min(loss, 700.0))}
    return out


def _spark(values, width):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not values:
        return ""
    values = values[-width:]
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return _SPARK[0] * len(values)
    return "".join(_SPARK[min(len(_SPARK) - 1, int((v - lo) / (hi - lo) * (len(_SPARK) - 1) + 0.5))]
                   for v in values)


def _bar(frac, width):
    frac = min(max(frac, 0.0), 1.0)
    filled = int(round(frac * width))
    return "█" * filled + "·" * (width - filled)


def _fmt_s(seconds):
    if seconds is None or not math.isfinite(seconds):
        return "--"
    seconds = int(seconds)
    if seconds >= 86400:
        return f"{seconds // 86400}d{seconds % 86400 // 3600:02d}h"
    if seconds >= 3600:
        return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds}s"


def _fmt_tokens(n):
    if n is None:
        return "--"
    n = float(n)
    for unit, div in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return f"{n:.0f}"


def dashboard_summary(records, events=(), *, width=96):
    """Plain-text snapshot of a run (no ANSI), used by `tools/watch_training.py --once` and by tests."""
    steps = [r for r in records if "step" in r and "eval" not in r and "loss" in r]
    evals = [r for r in records if r.get("eval")]
    lines = []
    if not records:
        return "no metrics yet"
    last = (steps or records)[-1]
    stage = last.get("stage", "?")
    phase = last.get("phase")
    lines.append(f"run      {stage}" + (f"/{phase}" if phase else ""))
    total = last.get("total_steps")
    if total:
        pct = last["step"] / total
        lines.append(f"step     {last['step']}/{total}  [{_bar(pct, 40)}] {pct:5.1%}")
    budget = last.get("budget_tokens")
    if budget and last.get("tokens_total"):
        lines.append(f"tokens   {_fmt_tokens(last['tokens_total'])}/{_fmt_tokens(budget)} "
                     f"({last['tokens_total'] / budget:5.1%})")
    if steps:
        losses = [r["loss"] for r in steps if r.get("loss") is not None]
        if losses:
            ema = EMA(alpha=0.05)
            for x in losses:
                ema.update(x)
            lines.append(f"loss     {losses[-1]:.4f}   ema {ema.value:.4f}   min {min(losses):.4f}")
            lines.append(f"         {_spark(losses, width - 9)}")
        for key, fmt in (("lr", ".3g"), ("gnorm", ".3g"), ("tok_s", ".0f"), ("mem_gb", ".2f")):
            vals = [r[key] for r in steps if r.get(key) is not None]
            if vals:
                lines.append(f"{key:<8} {format(vals[-1], fmt):>10}  {_spark(vals, width - 20)}")
    if evals:
        ev = normalize_eval(evals[-1])
        best = evals[-1].get("best_val")
        txt = "  ".join(f"{k} loss {v['loss']:.4f} ppl {v['ppl']:.1f}" for k, v in ev.items())
        lines.append(f"eval     {txt}" + (f"   best {best:.4f}" if best is not None else ""))
    if events:
        lines.append("alerts   " + f"{len(events)} total, latest:")
        for e in events[-3:]:
            lines.append(f"         [{e['level']:<8}] step {e['step']} {e['code']}: {e['message']}")
    return "\n".join(lines)


def render_dashboard(records, events=(), *, width=96, run=None, color=True, now=None):
    """Full live view for the terminal: progress, metric sparklines, ETA and the alert panel."""
    now = time.time() if now is None else now
    reset, dim, bold = ("\033[0m", "\033[2m", "\033[1m") if color else ("", "", "")
    level_color = {"info": "\033[36m", "warn": "\033[33m", "critical": "\033[31m"} if color else {}
    steps = [r for r in records if "loss" in r and "eval" not in r]
    evals = [r for r in records if r.get("eval")]
    out = [f"{bold}cirno training monitor{reset}" + (f"  {dim}{run}{reset}" if run else "")]

    if not records:
        out.append(f"{dim}waiting for the first metrics record...{reset}")
        return "\n".join(out)

    last = (steps or records)[-1]
    head = f"{last.get('stage', '?')}" + (f"/{last.get('phase')}" if last.get("phase") else "")
    total = last.get("total_steps")
    step = last.get("step", 0)
    if total:
        pct = step / total
        out.append(f"{head:<16} step {step}/{total}  [{_bar(pct, 40)}] {pct:5.1%}")
    else:
        out.append(f"{head:<16} step {step}")

    budget = last.get("budget_tokens")
    tks = last.get("tokens_total")
    eta = None
    if budget and tks:
        out.append(f"{'':<16} tokens {_fmt_tokens(tks)}/{_fmt_tokens(budget)} ({tks / budget:5.1%})")
    elapsed = last.get("elapsed_s")
    if elapsed and budget and tks and tks > 0:
        rate = tks / elapsed
        if rate > 0:
            eta = (budget - tks) / rate
    if steps and last.get("tok_s"):
        out.append(f"{'':<16} {last['tok_s']:.0f} tok/s   elapsed {_fmt_s(elapsed)}   eta {_fmt_s(eta)}")

    out.append("")
    losses = [r["loss"] for r in steps if r.get("loss") is not None]
    if losses:
        ema = EMA(alpha=0.05)
        for x in losses:
            ema.update(x)
        out.append(f"  loss    {losses[-1]:>9.4f}   ema {ema.value:>8.4f}   min {min(losses):>8.4f}"
                   f"   {_spark(losses, 40)}")
    for key, label, fmt in (("lm", "lm", ".4f"), ("mtp", "mtp", ".4f"), ("lr", "lr", ".3g"),
                            ("gnorm", "gnorm", ".3g"), ("tok_s", "tok/s", ".0f"), ("mem_gb", "mem", ".2f")):
        vals = [r[key] for r in steps if r.get(key) is not None]
        if vals:
            out.append(f"  {label:<6}  {format(vals[-1], fmt):>9}   {_spark(vals, 40)}")

    if evals:
        out.append("")
        ev = normalize_eval(evals[-1])
        best = evals[-1].get("best_val")
        for name, v in ev.items():
            out.append(f"  eval {name:<10} loss {v['loss']:.4f}   ppl {v['ppl']:.1f}"
                       + (f"   best {best:.4f}" if best is not None else ""))
        val_losses = [normalize_eval(r).get("val", {}).get("loss") for r in evals]
        val_losses = [v for v in val_losses if v is not None]
        if val_losses:
            out.append(f"       {'':<10} {_spark(val_losses, 40)}")

    out.append("")
    if events:
        out.append(f"{bold}alerts{reset} ({len(events)}, latest {min(5, len(events))})")
        for e in events[-5:]:
            c = level_color.get(e.get("level", "info"), "")
            out.append(f"  {c}[{e.get('level', '?'):<8}]{reset} step {e.get('step')} {e.get('code')}: {e.get('message')}")
    else:
        out.append(f"{dim}no alerts{reset}")
    return "\n".join(out)
