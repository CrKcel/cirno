#!/usr/bin/env python
"""Generation-based evaluation stage: run an eval set against a checkpoint and score the replies.

    python scripts/eval.py --model runs/sft/final --out runs/eval/sft          # every set in configs/eval.json
    python scripts/eval.py --model runs/sft/final --out runs/eval/sft --sets knowledge --limit 20
    python scripts/eval.py --model runs/sft/final --out runs/eval/sft --blind 30   # + human blind-test pairs
    python scripts/eval.py --model runs/sft/final --out runs/eval/sft --bench      # + edge cost numbers
    python scripts/eval.py --model runs/align/final --out runs/eval/align \
        --set synthetic=data/smoke/data/eval.jsonl --image_root data/smoke/data  # vision probes
    python scripts/eval.py --tokenizer tokenizer --desegment --length-prior data/sft/human_dialog.jsonl \
        --out runs/eval/imform_prior.json                                      # human length prior

Everything the samples *mean* is decided in hvlm/eval_core.py; this script owns the model, the sampling
and the artifacts (report.json / report.md / generations.jsonl / blind.json + blind.key.json / cost.json).
It never touches training: evaluation is a separate pass over a checkpoint directory, so the recipe's step
accounting stays what the plan resolved.
"""
import argparse
import contextlib
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from common_io import desegment_cjk, iter_records  # noqa: E402
from data import is_source_spec, load_image, special_ids  # noqa: E402
from hvlm import eval_core as ec  # noqa: E402
from hvlm.data_core import encode_prompt, normalize_messages  # noqa: E402
from hvlm.gated_delta import resolve_impl  # noqa: E402
from hvlm.modeling_hvlm import HVLMForConditionalGeneration  # noqa: E402
from hvlm.monitor import json_safe  # noqa: E402
from train import resolve_precision  # noqa: E402

DEFAULT_SAMPLING = dict(samples=1, temperature=0.0, top_k=40, max_new_tokens=48)
BENCH_PROMPT = [{"role": "user", "content": "在吗？今天过得怎么样？"}]
SKIP_SPECIALS = ("<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>",
                 "<|pad|>", "<|endoftext|>")


def build_parser():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--config", default="configs/eval.json", help="sets, sampling defaults and gates")
    ap.add_argument("--model", default=None, help="checkpoint directory (save_pretrained)")
    ap.add_argument("--tokenizer", default=None, help="defaults to --model; 'last/' carries no tokenizer")
    ap.add_argument("--out", required=True, help="report directory (runs/eval/<tag>), or the prior file")
    ap.add_argument("--tag", default=None, help="name this run carries in comparisons (default: basename of --out)")
    ap.add_argument("--sets", default=None, help="comma-separated subset of the config's sets")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=PATH",
                    help="add or replace a set (repeatable); the kind comes from the file")
    ap.add_argument("--limit", type=int, default=None, help="cap the records read per set")
    ap.add_argument("--samples", type=int, default=None, help="generations per item (overrides the config)")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=None)
    ap.add_argument("--max_prompt_tokens", type=int, default=1024, help="longer prompts are skipped and counted")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--desegment", action="store_true", help="word-segmented context (LCCC-style)")
    ap.add_argument("--image_root", default=".")
    ap.add_argument("--blind", type=int, default=0, help="also write this many human-vs-model blind pairs")
    ap.add_argument("--blind_human", default="data/sft/human_dialog.jsonl",
                    help="jsonl whose records supply the human arm of the blind test")
    ap.add_argument("--bench", action="store_true", help="measure prefill/decode cost and peak memory")
    ap.add_argument("--gate", action="store_true", help="exit non-zero when a configured gate fails")
    ap.add_argument("--length-prior", default=None, metavar="JSONL",
                    help="write the human reply-length prior of this chat jsonl to --out and exit")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gdn_impl", choices=["auto", "torch", "fla"], default="auto")
    ap.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    return ap


# ------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------
def as_spec(path):
    """A bare local path is a jsonl source spec (common_io spells the kind explicitly)."""
    return path if is_source_spec(path) else f"jsonl:{path}"


def sample_seed(seed, set_name, item_id, index):
    """Per-(set, item, sample) seed: the same command reproduces the same generation."""
    return random.Random(f"{seed}:{set_name}:{item_id}:{index}").getrandbits(31)


def git_rev():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # not a checkout / git missing: the report simply carries no revision
        return None


def load_config(path):
    if not os.path.exists(path):
        raise SystemExit(f"--config {path} does not exist")
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg, cfg.get("sets", {}), cfg.get("sampling", {}), cfg.get("gates", {}) or {}


def write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, ensure_ascii=False, indent=2)
        f.write("\n")


def human_replies(path, limit=None):
    """(context, reply) pairs from a chat jsonl: the last self turn is the human reply being compared to.

    The context is the whole conversation before that turn, so the model continues a real dialogue rather
    than answering an isolated line.  Records carrying an image are skipped: the human arm is text-only."""
    out = []
    for rec in iter_records(as_spec(path)):
        if rec.get("image"):
            continue
        try:
            msgs = normalize_messages(rec)
        except Exception:
            continue
        last = next((i for i in range(len(msgs) - 1, -1, -1) if msgs[i]["role"] == "self"), None)
        if last is None or not any(m["role"] != "system" for m in msgs[:last]):
            continue
        out.append((msgs[:last], msgs[last]["content"]))
        if limit is not None and len(out) >= limit:
            break
    return out


# ------------------------------------------------------------------------------------------------
# length prior
# ------------------------------------------------------------------------------------------------
def length_prior(args):
    """Human reply-length distribution, in the project tokenizer's tokens."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    counts = []
    for _context, reply in human_replies(args.length_prior):
        text = desegment_cjk(reply) if args.desegment else reply
        counts.append(len(tok(text, add_special_tokens=False)["input_ids"]))
    if not counts:
        raise SystemExit(f"no usable self turns in {args.length_prior}")
    stats = ec.length_stats(counts)
    write_json(args.out, dict(source=args.length_prior, tokenizer=args.tokenizer, desegment=bool(args.desegment),
                              n=len(counts), tokens={k: stats[k] for k in ("mean", "median", "p90", "p99", "max")},
                              long_share=stats["long_share"], reply_tokens_im=ec.REPLY_TOKENS_IM))
    print(f"prior from {len(counts)} human replies -> {args.out}: median {stats['median']:.0f} tokens, "
          f"p90 {stats['p90']:.0f}, >{ec.REPLY_TOKENS_IM} tokens {stats['long_share'] * 100:.1f}%")


# ------------------------------------------------------------------------------------------------
# generation
# ------------------------------------------------------------------------------------------------
class Generator:
    """One place where ids/texts cross the model boundary (generation runs through the model's
    InferenceCache: prefill once, single-token decode steps after that)."""

    def __init__(self, model, tok, sp, device, dtype_ctx):
        self.model, self.tok, self.sp, self.device, self.dtype_ctx = model, tok, sp, device, dtype_ctx
        self.skip = {tok.convert_tokens_to_ids(t) for t in SKIP_SPECIALS}
        self.eos = tok.convert_tokens_to_ids("<|im_end|>")

    def __call__(self, ids, pixel_values=None, max_new_tokens=48, temperature=0.0, top_k=40, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        with self.dtype_ctx():
            out = self.model.generate_simple(torch.tensor([ids], device=self.device), pixel_values=pixel_values,
                                             max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
                                             eos_token_id=self.eos)
        new = out[0, len(ids):].tolist()
        text = self.tok.decode(new, skip_special_tokens=True).strip()
        return text, sum(1 for t in new if t not in self.skip)


def image_tensor(rec, args, vc, device):
    if not rec.get("image"):
        return None
    path = os.path.join(args.image_root, rec["image"])
    if not os.path.exists(path):
        raise SystemExit(f"{rec['id']}: image not found at {path} (--image_root {args.image_root})")
    return load_image(path, vc.image_size)[None].to(device)


def work_items(records, kind):
    """(record, probe index) units: persona probes are scored separately, everything else per record."""
    if kind == "persona":
        return [(rec, i) for rec in records for i in range(len(rec["probes"]))]
    return [(rec, None) for rec in records]


def sampling_for(name, kind, cfg_sampling, args):
    spec = dict(DEFAULT_SAMPLING)
    spec.update(cfg_sampling.get(kind, {}))
    for key in ("samples", "temperature", "top_k", "max_new_tokens"):
        if getattr(args, key) is not None:
            spec[key] = getattr(args, key)
    return spec


def evaluate_set(name, path, gen, args, vc, cfg_sampling, prior):
    records = ec.load_set(path, args.limit)
    kind = ec.set_kind(records)
    sampling = sampling_for(name, kind, cfg_sampling, args)
    items, samples_by_id, details, skipped, n_gen = [], {}, [], 0, 0
    t0 = time.time()
    for rec, probe_index in work_items(records, kind):
        probe = rec["probes"][probe_index] if probe_index is not None else None
        item_id = f"{rec['id']}.{probe['field']}" if probe is not None else rec["id"]
        messages = ec.prompt_messages(rec, probe)
        enc = lambda s: gen.tok(desegment_cjk(s) if args.desegment else s,  # noqa: E731
                                add_special_tokens=False)["input_ids"]
        ids = encode_prompt(enc, gen.sp, messages, vc.num_image_tokens, bool(rec.get("image")))
        if len(ids) > args.max_prompt_tokens:
            skipped += 1
            details.append(dict(id=item_id, skipped="prompt too long", prompt_tokens=len(ids)))
            continue
        pixel_values = image_tensor(rec, args, vc, gen.device)
        samples = []
        for k in range(sampling["samples"]):
            text, n_tokens = gen(ids, pixel_values, max_new_tokens=sampling["max_new_tokens"],
                                 temperature=sampling["temperature"], top_k=sampling["top_k"],
                                 seed=sample_seed(args.seed, name, item_id, k))
            samples.append(dict(text=text, n_tokens=n_tokens))
        n_gen += len(samples)
        samples_by_id[item_id] = samples
        if kind == "knowledge":
            item = ec.score_knowledge(rec, samples)
        elif kind == "persona":
            item = ec.score_persona(rec, probe_index, samples)
        elif kind == "roleplay":
            item = ec.score_roleplay(rec, samples)
        else:
            item = dict(id=item_id)
        items.append(item)
        details.append(dict(id=item_id, messages=messages, image=rec.get("image"),
                            samples=[s["text"] for s in samples],
                            item={k: v for k, v in item.items() if k != "values"}))
    metrics = ec.aggregate(kind, items, samples_by_id, prior=prior if kind == "imform" else None)
    block = dict(n_items=len(items), n_generations=n_gen, skipped=skipped,
                 seconds=round(time.time() - t0, 1), metrics=metrics)
    return kind, sampling, block, details


# ------------------------------------------------------------------------------------------------
# cost and blind test
# ------------------------------------------------------------------------------------------------
def bench(gen, args, vc, model, load_s):
    device = gen.device
    cuda = device.type == "cuda"
    enc = lambda s: gen.tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    ids = encode_prompt(enc, gen.sp, BENCH_PROMPT, vc.num_image_tokens, False)
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    t1 = time.time()
    gen(ids, None, max_new_tokens=1, temperature=0.0, seed=args.seed)
    if cuda:
        torch.cuda.synchronize(device)
    prefill = time.time() - t1
    t2 = time.time()
    gen(ids, None, max_new_tokens=32, temperature=0.0, seed=args.seed)
    if cuda:
        torch.cuda.synchronize(device)
    decode = time.time() - t2
    return dict(load_s=round(load_s, 1),
                params_m=round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
                prompt_tokens=len(ids), prefill_ms=round(prefill * 1000, 1),
                decode_tok_s=round(31 / decode, 1) if decode > 0 else None,
                peak_mem_gb=round(torch.cuda.max_memory_allocated(device) / 1024 ** 3, 2) if cuda else None,
                note="decode goes through generate_simple's KV/state cache (prefix processed once, one token "
                     "per step after that); it is a real measurement but still of this machine - measure on "
                     "the target device for on-device cost")


def write_blind(args, gen, vc, out_dir):
    """Blind pairs where both arms answer the same human context: the model is sampled on it, the human
    reply is the one a real person gave in `--blind_human`.  Which arm holds the model reply is written
    only to the key, so the labeling pass sees two unlabelled replies."""
    humans = human_replies(args.blind_human)
    if len(humans) < args.blind:
        raise SystemExit(f"--blind {args.blind} needs {args.blind} human replies from {args.blind_human}, "
                         f"found {len(humans)}")
    picked = random.Random(args.seed).sample(humans, args.blind)
    items, model_replies = [], []
    for i, (messages, _reply) in enumerate(picked):
        enc = lambda s: gen.tok(desegment_cjk(s) if args.desegment else s,  # noqa: E731
                                add_special_tokens=False)["input_ids"]
        ids = encode_prompt(enc, gen.sp, messages, vc.num_image_tokens, False)
        text, _n = gen(ids, None, max_new_tokens=64, temperature=0.7, top_k=40,
                       seed=sample_seed(args.seed, "blind", f"b{i:03d}", 0))
        items.append((f"b{i:03d}", messages))
        model_replies.append(text)
    pairs, key = ec.make_blind_pairs(items, model_replies, [h[1] for h in picked], seed=args.seed)
    write_json(os.path.join(out_dir, "blind.json"), pairs)
    write_json(os.path.join(out_dir, "blind.key.json"), key)
    print(f"blind: {len(pairs)} pairs -> {out_dir}/blind.json (key: blind.key.json; "
          f"label with tools/blind_test.py, then --score folds the win rate into report.json)")


# ------------------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------------------
def main():
    args = build_parser().parse_args()
    cfg, cfg_sets, cfg_sampling, gates = load_config(args.config)
    if args.length_prior:
        if not args.tokenizer:
            raise SystemExit("--length-prior needs --tokenizer")
        length_prior(args)
        return 0
    if not args.model:
        raise SystemExit("--model is required (a save_pretrained directory)")

    sets = dict(cfg_sets)
    for spec in args.set:
        name, _, path = spec.partition("=")
        if not name or not path:
            raise SystemExit(f"--set expects NAME=PATH, got {spec!r}")
        sets[name] = path
    if args.sets:
        want = [s.strip() for s in args.sets.split(",") if s.strip()]
        missing = [s for s in want if s not in sets]
        if missing:
            raise SystemExit(f"unknown set(s) {missing}; config has {sorted(sets)} (add with --set NAME=PATH)")
        sets = {s: sets[s] for s in want}
    if not sets:
        raise SystemExit("no sets to evaluate")

    device = torch.device(args.device)
    cuda = device.type == "cuda"
    ac_dtype, precision_label = resolve_precision(args.precision, device)
    dtype_ctx = contextlib.nullcontext if ac_dtype is None else (
        lambda: torch.autocast(device_type=device.type, dtype=ac_dtype))
    gdn_impl = resolve_impl(args.gdn_impl, device, strict=True)

    from transformers import AutoTokenizer

    tok_path = args.tokenizer or args.model
    tok = AutoTokenizer.from_pretrained(tok_path)
    t0 = time.time()
    model = HVLMForConditionalGeneration.from_pretrained(args.model).float().to(device).eval()
    load_s = time.time() - t0
    model.config.linear_attn_impl = gdn_impl
    for m in model.modules():
        if hasattr(m, "impl"):
            m.impl = gdn_impl
    sp = special_ids(tok)
    gen = Generator(model, tok, sp, device, dtype_ctx)
    vc = model.config.vision_config
    print(f"eval {args.model} on {device} ({precision_label}, {gdn_impl})")

    prior = None
    if cfg.get("prior"):
        if os.path.exists(cfg["prior"]):
            with open(cfg["prior"], encoding="utf-8") as f:
                prior = json.load(f)
        else:
            print(f"note: prior {cfg['prior']} is missing; imform length ratios are left empty "
                  f"(build it with --length-prior)")

    report = dict(tag=args.tag or os.path.basename(os.path.normpath(args.out)),
                  model=args.model, tokenizer=tok_path, git=git_rev(),
                  created=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                  device=str(device), precision=precision_label, gdn_impl=gdn_impl,
                  seed=args.seed, sets={}, gates={}, cost=None, blind=None)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "generations.jsonl"), "w", encoding="utf-8") as gf:
        for name, path in sets.items():
            kind, sampling, block, details = evaluate_set(name, path, gen, args, vc, cfg_sampling, prior)
            block.update(kind=kind, file=path, sampling=sampling)
            report["sets"][name] = block
            for d in details:
                gf.write(json.dumps(json_safe(dict(set=name, kind=kind, **d)), ensure_ascii=False) + "\n")
            headline = {k: v for k, v in block["metrics"].items()
                        if k in ("accuracy", "unknown_rate", "tone_rate", "self_reference_rate",
                                 "overall_pass_rate", "median_ratio", "consistency")}
            print(f"  {name}: {block['n_items']} items, {block['n_generations']} generations, "
                  f"{block['skipped']} skipped, {block['seconds']}s {headline}")

    if args.bench:
        report["cost"] = bench(gen, args, vc, model, load_s)
        write_json(os.path.join(args.out, "cost.json"), report["cost"])
    if args.blind:
        write_blind(args, gen, vc, args.out)
        report["blind"] = dict(n_pairs=args.blind,
                               pairs=os.path.join(args.out, "blind.json"),
                               key=os.path.join(args.out, "blind.key.json"),
                               score_with=f"python tools/blind_test.py --score {args.out}")

    report["gates"] = ec.evaluate_gates({n: b["metrics"] for n, b in report["sets"].items()}, gates)
    write_json(os.path.join(args.out, "report.json"), report)
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write(ec.render_markdown(report))
    print(f"report: {args.out}/report.json, report.md, generations.jsonl")
    if report["gates"]["checks"]:
        failed = [c["metric"] for c in report["gates"]["checks"] if not c["ok"]]
        print(f"gate: {'PASS' if report['gates']['passed'] else 'FAIL'}"
              + (f" ({', '.join(failed)})" if failed else ""))
    return 1 if (args.gate and not report["gates"]["passed"]) else 0


if __name__ == "__main__":
    sys.exit(main())
