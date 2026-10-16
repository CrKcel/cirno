#!/usr/bin/env python
"""Derive the pre-training phase mixes from a recipe file and the bins' own meta.json (no torch needed).

    # recipe mode: reads the plan, resolves the mixture against the measured pools, writes a runnable plan
    python tools/plan_mix.py --plan configs/pretrain_plan.json --write runs/pretrain/plan.resolved.json

    # legacy mode: raw bin list, for experiments outside the recipe
    python tools/plan_mix.py --bin chat=data/bin/chat --bin wiki=data/bin/wiki \
        --a-bins fiction=0.45,long=0.35,wiki=0.20 --c-shares chat=0.34,fiction=0.28,wiki=0.284

The weights ``scripts/train.py`` takes are not a free parameter: ``prepare_data.py`` already fixed how many
tokens each bin holds (``train_tokens`` in its meta.json), and a weight above that count is a bin the
TokenMixer repeats.  This script reads the bins, sets every bin's whole-pool budget (times its declared
``max_passes``), and splits it over the three phases so the ``--data`` weights come out of the measured
token counts instead of a guess.

Recipe mode additionally writes ``plan.resolved.json``, which is what ``scripts/train.py --plan --phase``
consumes: it carries the phase hyper-parameters, the cumulative token offsets (so one global WSD schedule
spans all three runs) and the per-phase ``--data`` strings.  A bin whose directory has no meta.json yet
falls back to its ``expected_tokens`` in the recipe and says so on stderr, so the plan can be reviewed
before the data is tokenised and re-derived once it is.

``--print-commands`` emits the exact command lines, including the align / SFT stages.
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


# ------------------------------------------------------------------------------------------------
# recipe mode
# ------------------------------------------------------------------------------------------------
def plan_bins(plan, fallback_dir="data/bin"):
    """Bin name -> {path, tokens, val, allocate, passes, prepare_max_tokens, measured}.

    `tokens` is the pool (measured `train_tokens`, or the recipe's `expected_tokens` before tokenising);
    `allocate` is how much of it the recipe is allowed to read across all phases.  allocate > pool is a
    deliberate replay, allocate < pool is a corpus read only up to the useful part (baidu-baike is ordered
    common entity -> rare entity, so the cut lands on the long tail)."""
    bins = {}
    for name, spec in plan["bins"].items():
        path = spec.get("dir", os.path.join(fallback_dir, name))
        meta_path = os.path.join(path, "meta.json")
        measured = os.path.exists(meta_path)
        pool = int(spec["expected_tokens"])
        val = int(spec.get("val_tokens", 0))
        if measured:
            meta = json.load(open(meta_path, encoding="utf-8"))
            pool = int(meta["train_tokens"])
            val = int(meta.get("val_tokens", val))
        allocate = float(spec["allocate"])
        bins[name] = dict(path=path, tokens=pool, val=val, allocate=allocate, measured=measured,
                          expected=int(spec["expected_tokens"]),
                          passes=allocate / pool if pool else 0.0,
                          # a corpus we only read part of is not tokenised past that part
                          prepare_max_tokens=int(round(min(allocate, pool) + val)))
    return bins


def allocate_phase(p, want, remaining, pools, warn):
    """Split one phase's budget over the bins, never past what is left of each bin's allocation."""
    got = {}
    fixed = dict(p.get("fixed", {}))
    if fixed:
        got.update(distribute(want * sum(fixed.values()), fixed, remaining))
        left = want - sum(got.values())
        rest = list(p.get("rest", []))
        if rest and left > 1.0:
            caps = {n: remaining[n] for n in rest}
            for n, v in distribute(left, {n: max(pools[n], 1.0) for n in rest}, caps).items():
                got[n] = got.get(n, 0.0) + v
        left = want - sum(got.values())
        absorber = p.get("absorber")
        if left > 1.0 and absorber:  # tiny bins cannot use their share: the target-distribution bin takes it
            take = min(left, max(remaining[absorber] - got.get(absorber, 0.0), 0.0))
            got[absorber] = got.get(absorber, 0.0) + take
    else:
        shares = dict(p.get("shares", {}))
        if not shares:
            raise SystemExit(f"phase {p['name']} has neither shares nor a remainder role")
        got.update(distribute(want, shares, remaining))
        left = want - sum(got.values())
        if left > 1.0:  # caps stopped it short: spread the leftover over the same shares
            for k, v in distribute(left, shares, {k: remaining[k] - got.get(k, 0.0) for k in shares}).items():
                got[k] = got.get(k, 0.0) + v
    left = want - sum(got.values())
    if left > 1.0:
        warn(f"# phase {p['name']}: {left/1e6:.1f}M of its budget unallocated (bin caps exhausted)")
    return got


def resolve_plan(plan, bins, warn=print):
    phases = plan["pretrain"]
    budget = float(plan["budget_tokens"])
    fixed_phases = [p for p in phases if "tokens" in p]
    tail_phases = [p for p in phases if "tokens" not in p]
    if len(tail_phases) != 1:
        raise SystemExit("exactly one pretrain phase must omit 'tokens': it is the remainder of the budget")
    fixed_sum = sum(float(p["tokens"]) for p in fixed_phases)
    if fixed_sum >= budget:
        warn(f"# phases A + C already ask for {fixed_sum/1e9:.3f}B of the {budget/1e9:.3f}B budget")

    remaining = {n: float(b["allocate"]) for n, b in bins.items()}
    pools = {n: float(b["tokens"]) for n, b in bins.items()}
    alloc, got_by_phase = {}, {}
    for p in fixed_phases:
        got = allocate_phase(p, float(p["tokens"]), remaining, pools, warn)
        for n, v in got.items():
            remaining[n] -= v
            alloc[n] = alloc.get(n, 0.0) + v
        got_by_phase[p["name"]] = got

    tail = tail_phases[0]
    want = budget - fixed_sum
    available = sum(remaining.values())
    if available > want + 1.0:  # the bins hold more than the budget: every bin keeps its exact share
        scale = want / available
        got = {n: v * scale for n, v in remaining.items()}
        warn(f"# {tail['name']}: budget leaves {want/1e9:.3f}B, the bins still hold {available/1e9:.3f}B - "
             f"scaling that phase by {scale:.4f}")
    else:
        got = dict(remaining)
        if available < want - 1.0:
            warn(f"# {tail['name']}: wants {want/1e9:.3f}B but only {available/1e9:.3f}B is left in the bins")
    got_by_phase[tail["name"]] = got
    for n, v in got.items():
        alloc[n] = alloc.get(n, 0.0) + v

    sched = plan["schedule"]
    out = dict(version=plan.get("version", 1), budget_tokens=budget, schedule=dict(sched), phases={},
               bins={}, align=dict(plan.get("align", {})), sft=dict(plan.get("sft", {})))
    cum = 0.0
    for p in phases:
        got = got_by_phase[p["name"]]
        total = sum(got.values()) or 1.0
        data = [f"{bins[n]['path']}:{got[n] / total:.4f}" for n in bins if got.get(n, 0.0) > 0]
        per_bin = {n: dict(tokens=got[n], share=got[n] / total, passes=got[n] / bins[n]["tokens"])
                   for n in bins if got.get(n, 0.0) > 0}
        length = float(p["tokens"]) if "tokens" in p else sum(got.values())
        rec = dict(name=p["name"], stage="pretrain", tokens=length, cum_start=cum, cum_end=cum + length,
                   data=data, per_bin=per_bin)
        for k in ("seq_len", "micro_batch", "grad_accum", "lr", "warmup", "mtp_weight",
                  "mtp_weight_final", "train", "compile"):
            if k in p:
                rec[k] = p[k]
        out["phases"][p["name"]] = rec
        cum += length
    for name, b in bins.items():
        out["bins"][name] = dict(path=b["path"], pool=b["tokens"], val=b["val"],
                                 allocate=b["allocate"], allocated=alloc.get(name, 0.0),
                                 passes=alloc.get(name, 0.0) / b["tokens"] if b["tokens"] else 0.0,
                                 prepare_max_tokens=b["prepare_max_tokens"],
                                 measured=b["measured"], expected=b["expected"])
    for name in ("align", "sft"):
        out["phases"].setdefault(name, out[name])
    return out


def print_plan(resolved, bins, order=("A", "B", "C")):
    phases = resolved["phases"]
    order = [p for p in order if p in phases]
    print("# mixture per phase (weights are the phase's own share of its budget)")
    head = (f"{'bin':<18}{'pool':>10}{'allocate':>10}"
            + "".join(f"{p:>11}" for p in order) + f"{'total':>11}{'passes':>9}")
    print(head)
    print("-" * len(head))
    for n in bins:
        row = f"{n:<18}{bins[n]['tokens']/1e6:>9.1f}M{bins[n]['allocate']/1e6:>9.1f}M"
        tot = 0.0
        for p in order:
            v = phases.get(p, {}).get("per_bin", {}).get(n, {}).get("tokens", 0.0)
            tot += v
            row += f"{v/1e6:>10.1f}M"
        row += f"{tot/1e6:>10.1f}M{tot/bins[n]['tokens']:>8.2f}x"
        print(row)
    print()
    for p in order:
        rec = phases.get(p)
        if rec:
            print(f"# phase {p}: {rec['tokens']/1e6:.0f}M tokens, cum {rec['cum_start']/1e9:.2f}B -> "
                  f"{rec['cum_end']/1e9:.2f}B, seq {rec.get('seq_len')}, "
                  f"{rec.get('micro_batch')} x {rec.get('grad_accum')}")
            print("--data " + ",".join(rec["data"]))
    print()
    print("# caps that never bind, for prepare_data.py --max_tokens (a bin read partly is not tokenised past it)")
    for n, b in bins.items():
        print(f"  {n:<18} --max_tokens {b['prepare_max_tokens']}")


def print_commands(resolved, out_dir, tokenizer, python="python"):
    phases = resolved["phases"]
    print(f"# text pre-training: {resolved['budget_tokens']/1e9:.2f}B tokens in three phases")
    for p in ("A", "B", "C"):
        rec = phases[p]
        args = [f"{python} scripts/train.py", "--stage pretrain", f"--plan {out_dir}/plan.resolved.json",
                f"--phase {p}", f"--tokenizer {tokenizer}", f"--out_dir {out_dir}"]
        if rec.get("compile"):
            args.append("--compile")
        args.append("--resume")
        print(" ".join(args))
    align = phases.get("align")
    if align:
        print(f"\n# optional vision alignment (LLM frozen; skip for a text-only deployment)")
        print(f"{python} scripts/train.py --stage align --plan {out_dir}/plan.resolved.json --phase align \\\n"
              f"    --tokenizer {tokenizer} --init_from {out_dir}/final --out_dir runs/align \\\n"
              f"    --mm_data data/mm/chat_image.jsonl --image_root data/mm/images --resume")
    sft = phases.get("sft")
    if sft:
        print(f"\n# dialogue / role-play SFT (--mm_weights follows the recipe's {sft.get('mix_weights')} slice)")
        print(f"{python} scripts/train.py --stage sft --plan {out_dir}/plan.resolved.json --phase sft \\\n"
              f"    --tokenizer {tokenizer} --init_from runs/align/final --out_dir runs/sft \\\n"
              f"    --mm_data data/sft/human_dialog.jsonl,data/sft/persona.jsonl,data/sft/refuse.jsonl \\\n"
              f"    --mm_weights {','.join(str(x) for x in sft.get('mix_weights', []))} --desegment --resume")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=None, help="recipe file (configs/pretrain_plan.json)")
    ap.add_argument("--write", default=None, help="where to write the resolved plan")
    ap.add_argument("--print-commands", action="store_true", help="emit the matching train.py invocations")
    ap.add_argument("--out_dir", default="runs/pretrain")
    ap.add_argument("--tokenizer", default="tokenizer")
    ap.add_argument("--python", default="python")
    # legacy bin-mode arguments
    ap.add_argument("--bin", action="append", default=[], metavar="NAME=DIR")
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

    if args.plan:
        return plan_mode(args)
    return legacy_mode(args)


def plan_mode(args):
    plan = json.load(open(args.plan, encoding="utf-8"))
    bins = plan_bins(plan)
    missing = [n for n, b in bins.items() if not b["measured"]]
    for n in missing:
        print(f"# {n}: no meta.json yet, using the recipe's expected_tokens "
              f"({bins[n]['expected']/1e6:.1f}M) - re-run after prepare_data.py", file=sys.stderr)
    resolved = resolve_plan(plan, bins)
    print_plan(resolved, bins, order=[p["name"] for p in plan["pretrain"]])
    if args.write:
        os.makedirs(os.path.dirname(args.write) or ".", exist_ok=True)
        with open(args.write, "w", encoding="utf-8") as f:
            json.dump(resolved, f, ensure_ascii=False, indent=2)
        print(f"\n# wrote {args.write}")
    if args.print_commands:
        print()
        print_commands(resolved, args.out_dir, args.tokenizer, args.python)
    return 0


def legacy_mode(args):
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
