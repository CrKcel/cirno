#!/usr/bin/env python
"""Derive the three pre-training phase mixes from the bins' own meta.json (no torch needed).

    python tools/plan_mix.py \
        --bin chat=data/bin/chat --bin en_chat=data/bin/en_chat --bin fiction=data/bin/fiction \
        --bin long=data/bin/long --bin wiki=data/bin/wiki --bin roleplay=data/bin/roleplay \
        --bin roleplay_char=data/bin/roleplay_char --bin roleplay_catgirl=data/bin/roleplay_catgirl

The weights scripts/train.py takes are not a free parameter: `prepare_data.py` already fixed how many
tokens each bin has (`train_tokens` in its meta.json), and a weight above that count is a bin the
TokenMixer repeats.  This script reads the bins, sets every bin's whole-pool budget, and splits it over
the three phases so that the `--data` weights come out of the measured token counts instead of a guess:

  * phase A takes only the long-form bins (--a-bins at --a-shares), the phase that builds the base;
  * phase C takes --c-shares plus --c-rest, whose remaining share is split by measured pool size, so the
    anneal stays chat / role-play heavy while the small role-play bins are read about once;
  * phase B gets whatever is left, which is where the dialogue-heavy part of the curriculum lives.

Bins named by --slack absorb the difference between the sum of the pools and the phase budgets; without
one, every bin is scaled down together.  A negative allocation for a bin means its pool is smaller than
its phase A + C shares, and the script says so instead of silently wrapping.

`--val` prints the matching `prepare_data.py --max_tokens`, which must sit at or above the bin's pool:
a cap that binds is a bin that stops mid-source, which is what the mix is trying to avoid.  The printed
number is the minimum that never binds; add headroom when a small source is read last inside its bin (the
human-written English sets after soda), so a tokenizer-ratio surprise cannot cut that source off.
"""
import argparse
import json
import os
import sys

DEFAULT_PHASES = "0.9e9,2.3e9,0.8e9"
DEFAULT_A = "fiction=0.45,long=0.35,wiki=0.20"
DEFAULT_C = "chat=0.34,en_chat=0.036,fiction=0.28,long=0.04,wiki=0.284"
DEFAULT_C_REST = "roleplay,roleplay_char,roleplay_catgirl"


def num(s):
    return int(float(s))


def parse_pairs(s):
    out = {}
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, share = item.partition("=")
        out[name] = float(share) if share else None
    return out


def load_bins(specs, bins):
    for spec in specs:
        name, _, path = spec.partition("=")
        meta_path = os.path.join(path, "meta.json")
        if not os.path.exists(meta_path):
            raise SystemExit(f"{name}: no {meta_path} - run prepare_data.py first")
        meta = json.load(open(meta_path, encoding="utf-8"))
        if name in bins:
            raise SystemExit(f"duplicate bin name {name!r}")
        bins[name] = dict(path=path, tokens=int(meta["train_tokens"]), val=int(meta.get("val_tokens", 0)))


def distribute(total, weights, caps):
    """Give `total` out along `weights`, never above `caps`; leftovers go back proportionally."""
    out = {k: 0.0 for k in weights}
    pool = dict(caps)
    left = float(total)
    for _ in range(3):
        w = {k: v for k, v in weights.items() if pool[k] > 0}
        if not w or left <= 0:
            break
        s = sum(w.values()) or 1.0
        for k, v in w.items():
            out[k] += min(left * v / s, pool[k])
        left = float(total) - sum(out.values())
        for k in out:
            pool[k] = caps[k] - out[k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--phases", default=DEFAULT_PHASES, help="A,B,C token budgets (default %(default)s)")
    ap.add_argument("--a-bins", default=DEFAULT_A, help="phase A bins and shares (default %(default)s)")
    ap.add_argument("--c-shares", default=DEFAULT_C, help="phase C bins and shares (default %(default)s)")
    ap.add_argument("--c-rest", default=DEFAULT_C_REST,
                    help="phase C bins sharing the leftover, split by measured pool (default %(default)s)")
    ap.add_argument("--slack", default="", help="bins that absorb the pool/budget difference, in order")
    ap.add_argument("--val", action="append", default=[], metavar="NAME=N",
                    help="val_tokens per bin, for the printed --max_tokens recommendation")
    ap.add_argument("--decimals", type=int, default=4)
    args = ap.parse_args()

    bins = {}
    load_bins(args.bin, bins)
    phases = [num(x) for x in args.phases.split(",")]
    if len(phases) != 3:
        raise SystemExit("--phases wants three comma-separated budgets")
    a_spec = parse_pairs(args.a_bins)
    c_spec = parse_pairs(args.c_shares)
    c_rest = [b.strip() for b in args.c_rest.split(",") if b.strip()]
    slack = [b.strip() for b in args.slack.split(",") if b.strip()]
    vals = {}
    for item in args.val:
        name, _, v = item.partition("=")
        vals[name] = num(v)

    for name in list(a_spec) + list(c_spec) + c_rest + slack:
        if name not in bins:
            raise SystemExit(f"unknown bin {name!r}: --bin knows {sorted(bins)}")

    pools = {n: float(b["tokens"]) for n, b in bins.items()}
    alloc = dict(pools)
    surplus = sum(alloc.values()) - sum(phases)
    if surplus > 0:
        if slack:
            for name in slack:
                take = min(alloc[name], surplus)
                alloc[name] -= take
                surplus -= take
                if surplus <= 0:
                    break
        if surplus > 0:  # no slack bin took it all: scale everything down
            scale = sum(phases) / sum(alloc.values())
            alloc = {n: v * scale for n, v in alloc.items()}
            print(f"# no --slack absorbed the shortfall: every bin scaled by {scale:.4f}", file=sys.stderr)
    elif surplus < 0:
        scale = sum(phases) / sum(alloc.values())
        alloc = {n: v * scale for n, v in alloc.items()}
        print(f"# budget exceeds the pools by {-surplus/1e6:.1f}M tokens: every bin repeats {scale:.3f}x",
              file=sys.stderr)

    t_a = distribute(phases[0], {n: a_spec[n] for n in a_spec if a_spec[n] is not None}, alloc)
    c_fixed = {n: v for n, v in c_spec.items() if v is not None}
    t_c = {n: v * phases[2] for n, v in c_fixed.items()}
    rest_budget = phases[2] - sum(t_c.values())
    if c_rest and rest_budget > 0:
        caps = {n: max(alloc[n] - t_c.get(n, 0.0), 0.0) for n in c_rest}
        for n, v in distribute(rest_budget, {n: pools[n] for n in c_rest}, caps).items():
            t_c[n] = t_c.get(n, 0.0) + v
    t_b = {}
    for n in bins:
        left = alloc[n] - t_a.get(n, 0.0) - t_c.get(n, 0.0)
        if left < -1.0:
            print(f"# {n}: phase A + C ask for {-left/1e6:.1f}M more than its pool; lower a/c shares",
                  file=sys.stderr)
        t_b[n] = max(left, 0.0)

    def weights(t):
        s = sum(t.values()) or 1.0
        return [f"{bins[n]['path']}:{t[n] / s:.{args.decimals}f}"
                for n in bins if t.get(n, 0.0) > 0]

    print("# phase A (long-form base, seq 2048)")
    print("--data " + ",".join(weights(t_a)))
    print("# phase B (dialogue introduced, seq 1024)")
    print("--data " + ",".join(weights(t_b)))
    print("# phase C (anneal on chat + role-play, seq 1024)")
    print("--data " + ",".join(weights(t_c)))

    print()
    head = f"{'bin':<18}{'pool':>10}{'A':>10}{'B':>10}{'C':>10}{'total':>10}{'passes':>9}"
    print(head)
    print("-" * len(head))
    for n in bins:
        a, b, c = t_a.get(n, 0.0), t_b.get(n, 0.0), t_c.get(n, 0.0)
        print(f"{n:<18}{pools[n]/1e6:>9.1f}M{a/1e6:>9.1f}M{b/1e6:>9.1f}M{c/1e6:>9.1f}M"
              f"{(a+b+c)/1e6:>9.1f}M{alloc[n]/pools[n]:>8.2f}x")
    print(f"{'TOTAL':<18}{sum(pools.values())/1e6:>9.1f}M{sum(t_a.values())/1e6:>9.1f}M"
          f"{sum(t_b.values())/1e6:>9.1f}M{sum(t_c.values())/1e6:>9.1f}M"
          f"{sum(alloc.values())/1e6:>9.1f}M{sum(alloc.values())/sum(pools.values()):>8.2f}x")

    if vals:
        print("\n# caps that never bind (pool + val), for prepare_data.py --max_tokens")
        for n in bins:
            print(f"  {n:<18} --max_tokens {int(alloc[n] + vals.get(n, bins[n]['val']))}")


if __name__ == "__main__":
    main()
