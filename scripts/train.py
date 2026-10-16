#!/usr/bin/env python
import argparse
import contextlib
import json
import math
import os
import shutil
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from data import ChatCollator, ChatDataset, PretrainDataset, ValWindows, special_ids  # noqa: E402
from hvlm.configuration_hvlm import HVLMConfig  # noqa: E402
from hvlm.data_core import sample_order  # noqa: E402
from hvlm.gated_delta import fla_available, fla_error, fla_version, resolve_impl  # noqa: E402
from hvlm.modeling_hvlm import HVLMForConditionalGeneration  # noqa: E402
from hvlm.monitor import HealthPolicy, MetricsLog, json_safe  # noqa: E402

STAGE_DEFAULTS = {
    "pretrain": dict(lr=1e-3, warmup=0.02, seq_len=1024, micro_batch=2, grad_accum=256, train="llm",
                     mtp_weight=0.3, vit_lr_mult=1.0),
    "align":    dict(lr=5e-4, warmup=0.05, seq_len=512, micro_batch=4, grad_accum=64, train="vit",
                     mtp_weight=0.0, vit_lr_mult=1.0),
    "sft":      dict(lr=2e-4, warmup=0.03, seq_len=1024, micro_batch=2, grad_accum=128, train="llm,vit",
                     mtp_weight=0.1, vit_lr_mult=0.5),
}

# Recipe records (configs/pretrain_plan.json, resolved by tools/plan_mix.py) use the same names as the
# command line wherever the meaning is identical; the schedule block is global and the phase record wins.
PLAN_ARG_MAP = {"seq_len": "seq_len", "micro_batch": "micro_batch", "grad_accum": "grad_accum",
                "lr": "lr", "warmup": "warmup", "min_lr_ratio": "min_lr_ratio",
                "weight_decay": "weight_decay", "mtp_weight": "mtp_weight",
                "mtp_weight_final": "mtp_weight_final", "train": "train", "epochs": "epochs"}
PLAN_SCHEDULE_MAP = {"peak_lr": "lr", "warmup_frac": "warmup", "decay_frac": "decay_frac",
                     "min_lr_ratio": "min_lr_ratio", "weight_decay": "weight_decay", "clip": "clip"}


def build_parser():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--stage", choices=list(STAGE_DEFAULTS), default=None,
                    help="required unless --plan names a phase that carries a stage")
    ap.add_argument("--config", default="configs/hvlm_130m.json", help="used when --init_from is not given")
    ap.add_argument("--init_from", default=None, help="save_pretrained directory to start from (stages 2/3)")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--plan", dest="plan_path", default=None,
                    help="resolved recipe (tools/plan_mix.py --write); supplies the phase's mixture, "
                         "shape and learning-rate schedule")
    ap.add_argument("--phase", dest="phase_name", default=None, help="which phase of --plan to run")
    # data
    ap.add_argument("--data", default=None, help="pretrain: dir:weight,dir:weight  (dirs from prepare_data.py)")
    ap.add_argument("--mm_data", default=None, help="align/sft: comma-separated jsonl files and/or source "
                                                   "specs (hf:..., parquet:..., jsonl:<glob>; see doc/training.md 5)")
    ap.add_argument("--mm_weights", default=None,
                    help="align/sft: comma-separated sampling weights, one per --mm_data item; defaults to "
                         "uniform.  Without it a source's share is decided by its line count")
    ap.add_argument("--image_root", default=".")
    ap.add_argument("--val_mm_data", default=None, help="optional jsonl/spec for validation in align/sft")
    ap.add_argument("--mm_cache", default="data/cache",
                    help="where remote --mm_data sources are spooled as jsonl (reused on later runs)")
    ap.add_argument("--desegment", action="store_true",
                    help="align/sft: remove spaces between CJK tokens, for word-segmented corpora "
                         "such as LCCC's dialog (see common_io.desegment_cjk)")
    # budget
    ap.add_argument("--seq_len", type=int, default=None)
    ap.add_argument("--micro_batch", type=int, default=None,
                    help="stage defaults: pretrain 2, align 4, sft 2; measured safe max on this 8 GB card "
                         "is 2 @seq1024 (pretrain, plus --compile for 4 or for seq2048 x 2), 4 @seq512 "
                         "(align) and 2 @seq1024 (sft) - see doc/architecture.md 3.3, tools/measure_batch.py")
    ap.add_argument("--grad_accum", type=int, default=None,
                    help="stage defaults: pretrain 256, align 64, sft 128 (mb*accum*seq = 524k/131k/262k tokens per step)")
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--tokens", type=float, default=None, help="pretrain: total training tokens (alternative to --max_steps)")
    ap.add_argument("--epochs", type=float, default=None, help="align/sft: number of passes over the data")
    # optimisation
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--vit_lr_mult", type=float, default=None)
    ap.add_argument("--min_lr_ratio", type=float, default=None)
    ap.add_argument("--warmup", type=float, default=None, help="fraction of steps")
    ap.add_argument("--schedule", choices=["wsd", "cosine"], default="wsd")
    ap.add_argument("--decay_frac", type=float, default=0.2, help="wsd: final fraction of steps spent decaying")
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--betas", default="0.9,0.95")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--train", default=None, help="modules to train: llm, vit or llm,vit")
    ap.add_argument("--mtp_weight", type=float, default=None)
    ap.add_argument("--mtp_weight_final", type=float, default=None, help="pretrain: switch to this weight for the last --decay_frac")
    ap.add_argument("--gdn_impl", choices=["auto", "torch", "fla"], default="auto",
                    help="Gated DeltaNet kernel: auto = flash-linear-attention when it is installed, else pure PyTorch")
    ap.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
                    help="autocast dtype; auto = bf16 if the GPU supports it, else fp16 (fp16 adds a GradScaler)")
    # system
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_mem_gb", type=float, default=8.0,
                    help="hard cap on this process's CUDA memory, also bounded by what is free at start (0 = off)")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the training forward (Inductor). Measured +48%% on an 8 GB laptop card. "
                         "The first step pays a one-off JIT cost (seconds to a minute; cached in "
                         "TORCHINDUCTOR_CACHE_DIR, default /tmp/torchinductor_*). Falls back to eager when "
                         "Inductor is unavailable")
    ap.add_argument("--tf32", action="store_true",
                    help="allow TF32 for fp32 matmuls (off by default: the pure-PyTorch delta-rule core is an "
                         "fp32 island; harmless with --gdn_impl fla, where that core lives in the Triton kernel)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_batches", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    # monitoring (hvlm/monitor.py; see doc/monitoring.md)
    ap.add_argument("--no_monitor", action="store_true", help="write only the legacy log.jsonl")
    ap.add_argument("--no_intervention", action="store_true",
                    help="report anomalies in health.jsonl but never change the learning rate or stop early")
    ap.add_argument("--spike_sigma", type=float, default=4.0, help="loss-spike threshold in recent std devs")
    ap.add_argument("--gnorm_spike_factor", type=float, default=20.0, help="grad-norm spike, as a multiple of --clip")
    ap.add_argument("--throughput_drop", type=float, default=0.5, help="alert below this fraction of the running tok/s")
    ap.add_argument("--lr_backoff", type=float, default=0.5, help="lr multiplier per loss/grad spike (1.0 = off)")
    ap.add_argument("--backoff_steps", type=int, default=300, help="how long a backoff is held before release")
    ap.add_argument("--early_stop_patience", type=int, default=0,
                    help="stop after this many evaluations without a new best val loss (0 = off)")
    ap.add_argument("--early_stop_min_frac", type=float, default=0.5,
                    help="ignore --early_stop_patience before this fraction of the run")
    return ap


def load_plan(path, phase):
    """Read a resolved recipe and return (plan, phase record)."""
    if not os.path.exists(path):
        raise SystemExit(f"--plan {path} does not exist; write it with "
                         f"`python tools/plan_mix.py --plan <recipe> --write {path}`")
    plan = json.load(open(path, encoding="utf-8"))
    phases = plan.get("phases", {})
    if not phases:  # the editable recipe, not the resolved one: it still needs the bins' token counts
        raise SystemExit(f"{path} has no 'phases' - resolve the recipe first: "
                         f"`python tools/plan_mix.py --plan {path} --write <out_dir>/plan.resolved.json`")
    if phase is None:
        candidates = [n for n, p in phases.items() if p.get("stage") == "pretrain"]
        if len(candidates) != 1:
            raise SystemExit(f"--plan needs --phase (one of {sorted(phases)})")
        phase = candidates[0]
    if phase not in phases:
        raise SystemExit(f"--phase {phase!r} is not in {path}: {sorted(phases)}")
    return plan, phases[phase]


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = build_parser()
    a = ap.parse_args(argv)
    a.plan, a.phase = None, None
    if a.plan_path:
        plan, rec = load_plan(a.plan_path, a.phase_name)
        defaults = {}
        for key, dest in PLAN_ARG_MAP.items():
            if key in rec:
                defaults[dest] = rec[key]
        for key, dest in PLAN_SCHEDULE_MAP.items():
            if key in plan.get("schedule", {}):
                defaults.setdefault(dest, plan["schedule"][key])
        if "betas" in plan.get("schedule", {}):
            defaults.setdefault("betas", ",".join(str(b) for b in plan["schedule"]["betas"]))
        if rec.get("compile"):
            defaults["compile"] = True
        if rec.get("stage"):
            defaults["stage"] = rec["stage"]
        ap.set_defaults(**defaults)  # explicit command-line flags still win over the recipe
        a = ap.parse_args(argv)
        a.plan, a.phase = plan, rec
        if rec.get("stage") == "pretrain":
            if "--data" not in argv:
                a.data = ",".join(rec["data"])
            a.tokens = a.max_steps or rec["cum_end"]
    if a.stage is None:
        ap.error("--stage is required unless --plan names a phase that carries one")
    d = STAGE_DEFAULTS[a.stage]
    for k, v in d.items():
        if getattr(a, k, None) is None:
            setattr(a, k, v)
    if a.min_lr_ratio is None:  # not a per-stage default, but the recipes can set it
        a.min_lr_ratio = 0.1
    if a.mm_weights:
        a.mm_weights = [float(x) for x in a.mm_weights.split(",")]
    return a


def resolve_precision(name, device):
    """Pick the autocast dtype for this device.  Returns (dtype or None, human-readable label)."""
    if name == "fp32":
        return None, "fp32 (autocast off, fp32 master weights)"
    if name == "fp16":
        if device.type != "cuda":
            raise SystemExit("--precision fp16 needs CUDA (use bf16 or fp32 on CPU)")
        return torch.float16, "fp16 autocast + GradScaler (fp32 master weights)"
    if name == "bf16":
        return torch.bfloat16, "bf16 autocast (fp32 master weights)"
    # auto: bf16 wherever it is native (Ampere+), fp16 with a GradScaler otherwise
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        return torch.float16, "fp16 autocast + GradScaler (fp32 master weights; no native bf16 here)"
    return torch.bfloat16, f"bf16 autocast (fp32 master weights{'; CPU' if device.type != 'cuda' else ''})"


def lr_at(step, total, warmup_steps, base, min_ratio, schedule, decay_frac):
    if step < warmup_steps:
        return base * (step + 1) / max(1, warmup_steps)
    if schedule == "cosine":
        prog = (step - warmup_steps) / max(1, total - warmup_steps)
        return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))
    decay_start = int(total * (1 - decay_frac))
    if step < decay_start:
        return base
    return base * (1 - (1 - min_ratio) * (step - decay_start) / max(1, total - decay_start))


def lr_at_tokens(tokens_done, budget, warm_frac, decay_frac, base, min_ratio):
    """Same WSD shape as `lr_at`, but indexed by tokens over the *whole* recipe.

    A phase run only knows its own end step, so the plan's three runs each evaluate this with the global
    token count (`step * tokens_per_step`): warmup, the flat middle and the final decay then form one
    continuous curve without the `--decay_frac 0` / cumulative `--tokens` bookkeeping the recipe used to
    need."""
    warm = warm_frac * budget
    if tokens_done < warm:
        return base * max(tokens_done, 0.0) / max(warm, 1.0) if warm else base
    decay_start = (1.0 - decay_frac) * budget
    if tokens_done < decay_start:
        return base
    return base * (1 - (1 - min_ratio) * (tokens_done - decay_start) / max(budget - decay_start, 1.0))


def build_param_groups(model, a):
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_vit = n.startswith("visual.")
        no_decay = p.ndim < 2 or "embed_tokens" in n or "A_log" in n or "dt_bias" in n
        key = (is_vit, no_decay)
        groups.setdefault(key, []).append(p)
    out = []
    for (is_vit, no_decay), params in groups.items():
        out.append(dict(params=params, weight_decay=0.0 if no_decay else a.weight_decay,
                        lr_mult=a.vit_lr_mult if is_vit else 1.0))
    return out


def set_trainable(model, which):
    which = set(which.split(","))
    for n, p in model.named_parameters():
        is_vit = n.startswith("visual.")
        p.requires_grad = ("vit" in which) if is_vit else ("llm" in which)


def count(model, trainable=False):
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable))


def setup_compile(a, model):
    """Wrap the training forward in torch.compile, or fall back to eager when Inductor is unavailable.

    Compilation is lazy: the first forward traces and compiles, so its cost lands in the first logged
    step (the artifact is cached under TORCHINDUCTOR_CACHE_DIR across runs)."""
    if not a.compile:
        return model, "off"
    try:
        fwd_model = torch.compile(model, dynamic=False)
    except Exception as exc:  # pragma: no cover - depends on the local toolchain
        return model, f"unavailable ({type(exc).__name__}: {exc})"
    return fwd_model, "on (the first logged step includes the one-off JIT)"


@torch.no_grad()
def evaluate(model, loaders, device, dtype_ctx, n_batches):
    """Mean LM loss (and ppl) per validation loader; feeds both the log and the health policy."""
    model.eval()
    res = {}
    for name, dl in loaders.items():
        tot, n = 0.0, 0
        for i, batch in enumerate(dl):
            if i >= n_batches:
                break
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with dtype_ctx():
                out = model(**batch, compute_mtp=False)
            tot += float(out.lm_loss.detach())
            n += 1
        if n:
            loss = tot / n
            res[name] = dict(loss=loss, ppl=math.exp(min(loss, 700.0)))
    model.train()
    return res


def report_events(events_log, events):
    """Write policy events to health.jsonl and echo them next to the training line."""
    for e in events:
        events_log.write(e)
        print(f"[{e['level'].upper()}] step {e['step']} {e['code']}: {e['message']}", flush=True)


def save_checkpoint(model, opt, out_dir, tag, step, samples, tokenizer_path=None, scaler=None):
    tmp, dst = os.path.join(out_dir, tag + ".tmp"), os.path.join(out_dir, tag)
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp)
    if opt is not None:
        torch.save(dict(opt=opt.state_dict(), step=step, samples=samples,
                        scaler=scaler.state_dict() if scaler is not None else None),
                   os.path.join(tmp, "train_state.pt"))
    if tokenizer_path:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(tokenizer_path).save_pretrained(tmp)
    shutil.rmtree(dst, ignore_errors=True)
    os.rename(tmp, dst)


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    torch.manual_seed(a.seed)
    device = torch.device(a.device)
    cuda = device.type == "cuda"
    budget_gb = None  # CUDA memory this process may use; feeds the allocator cap and the health policy
    if cuda:
        if a.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        dev_idx = device.index if device.index is not None else torch.cuda.current_device()
        total_gb = torch.cuda.get_device_properties(dev_idx).total_memory / 1024 ** 3
        free_gb = torch.cuda.mem_get_info(dev_idx)[0] / 1024 ** 3
        budget_gb = min(a.max_mem_gb, free_gb) if a.max_mem_gb else free_gb
        if a.max_mem_gb and a.max_mem_gb > free_gb:
            print(f"note: --max_mem_gb {a.max_mem_gb:.1f} exceeds the {free_gb:.2f} GB currently free "
                  f"(desktop/other processes hold the rest) - capping to {budget_gb:.2f} GB instead")
        if budget_gb < total_gb:
            torch.cuda.set_per_process_memory_fraction(max(budget_gb / total_gb, 1e-6), dev_idx)
            print(f"CUDA allocator capped at {budget_gb:.2f} GB (device has {total_gb:.2f} GB, {free_gb:.2f} GB free)")
    ac_dtype, precision_label = resolve_precision(a.precision, device)
    if ac_dtype is None:
        dtype_ctx = contextlib.nullcontext
    else:
        dtype_ctx = lambda: torch.autocast(device_type=device.type, dtype=ac_dtype)  # noqa: E731
    scaler = torch.amp.GradScaler("cuda") if (ac_dtype is torch.float16 and cuda) else None
    gdn_impl = resolve_impl(a.gdn_impl, device, strict=True)  # fails fast on an unusable --gdn_impl fla
    print(f"precision: {precision_label}")
    print(f"gdn_impl:  {gdn_impl}" + ("" if gdn_impl == a.gdn_impl else f" (--gdn_impl {a.gdn_impl})") + "  ["
          + (f"flash-linear-attention {fla_version()}]" if fla_available()
             else f"fla unavailable: {str(fla_error())[:160]}]"))

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)

    # ---- monitoring -----------------------------------------------------------------------------
    # The recipe's schedule is what the policy judges against, so it is resolved before the model is
    # built: without a plan there is no global budget and only the per-run thresholds apply.
    budget_tokens = float((a.plan or {}).get("budget_tokens") or 0.0) or None
    mon = events_log = policy = None
    if not a.no_monitor:
        mon = MetricsLog(os.path.join(a.out_dir, "metrics.jsonl"))
        events_log = MetricsLog(os.path.join(a.out_dir, "health.jsonl"))
        policy = HealthPolicy(peak_lr=a.lr, clip=a.clip, budget_gb=budget_gb,
                              spike_sigma=a.spike_sigma, gnorm_spike_factor=a.gnorm_spike_factor,
                              throughput_ratio=a.throughput_drop,
                              val_patience=a.early_stop_patience,
                              intervene=not a.no_intervention,
                              lr_backoff_factor=a.lr_backoff, backoff_steps=a.backoff_steps)
        print(f"monitoring: {a.out_dir}/metrics.jsonl + health.jsonl"
              + ("" if not a.no_intervention else " (interventions off)")
              + (f", lr backoff x{a.lr_backoff}" if not a.no_intervention and a.lr_backoff < 1.0 else ""))

    # ---- model ----------------------------------------------------------------------------------
    resume_dir = os.path.join(a.out_dir, "last")
    if a.resume and os.path.isdir(resume_dir):
        model = HVLMForConditionalGeneration.from_pretrained(resume_dir).float()
        print(f"resuming weights from {resume_dir}")
    elif a.init_from:
        model = HVLMForConditionalGeneration.from_pretrained(a.init_from).float()
    else:
        raw = json.load(open(a.config))
        raw.pop("model_type", None)
        cfg = HVLMConfig(**raw)
        model = HVLMForConditionalGeneration(cfg)
    model.config.mtp_loss_weight = a.mtp_weight
    # The saved config keeps the *request* ("auto" stays portable: a checkpoint must still load and sample
    # on a machine without flash-linear-attention); the live modules get what this device actually resolved.
    model.config.linear_attn_impl = a.gdn_impl
    for m in model.modules():
        if hasattr(m, "impl"):
            m.impl = gdn_impl
    assert len(tok) <= model.config.vocab_size, "tokenizer larger than model vocab"
    model.to(device)
    set_trainable(model, a.train)
    model.train()
    fwd_model, compile_msg = setup_compile(a, model)
    print(f"torch.compile: {compile_msg}")
    vc = model.config.vision_config
    print(f"params: total {count(model) / 1e6:.1f}M, trainable {count(model, True) / 1e6:.1f}M "
          f"(stage={a.stage}, train={a.train}, mtp_weight={a.mtp_weight})")

    # ---- data -----------------------------------------------------------------------------------
    S = a.seq_len
    sampler_len, val_loaders = None, {}
    if a.stage == "pretrain":
        assert a.data, "--data dir:weight,... is required for pretrain"
        specs = [(p.split(":")[0], float(p.split(":")[1]) if ":" in p else 1.0) for p in a.data.split(",")]
        toks_per_step = a.micro_batch * a.grad_accum * S
        assert a.max_steps or a.tokens, "pretrain needs --tokens or --max_steps"
        total_steps = a.max_steps or int(a.tokens / toks_per_step)
        planned = total_steps * a.micro_batch * a.grad_accum + a.micro_batch * a.grad_accum
        ds = PretrainDataset(specs, S, planned, a.seed)
        collate = None
        for d, _ in specs:
            vp = os.path.join(d, "val.bin")
            if os.path.exists(vp):
                val_loaders[os.path.basename(os.path.normpath(d))] = DataLoader(
                    ValWindows(vp, S), batch_size=a.micro_batch, shuffle=False, num_workers=0,
                    collate_fn=lambda b: dict(input_ids=torch.stack(b), labels=torch.stack(b)))
    else:
        assert a.mm_data, "--mm_data file.jsonl|spec[,...] is required for align/sft"
        sp = special_ids(tok)
        sources = a.mm_data.split(",")
        ds = ChatDataset(sources, a.image_root, tok, S, vc, sp, a.mm_cache, a.desegment)
        rows = len(ds)  # rows, not samples: --mm_weights lays out its own `epochs x rows` schedule
        if a.mm_weights:
            if len(a.mm_weights) != len(sources):
                raise SystemExit(f"--mm_weights has {len(a.mm_weights)} values for {len(sources)} --mm_data items")
            ds.set_mix(a.mm_weights, int((a.epochs or 1.0) * rows))
        collate = ChatCollator(tok.convert_tokens_to_ids("<|pad|>"), sp["image_pad"], vc.merged_grid_size)
        per_step = a.micro_batch * a.grad_accum
        total_steps = a.max_steps or int((a.epochs or 1.0) * rows / per_step)
        if a.val_mm_data:
            vds = ChatDataset(a.val_mm_data.split(","), a.image_root, tok, S, vc, sp, a.mm_cache, a.desegment)
            val_loaders["val"] = DataLoader(vds, batch_size=a.micro_batch, shuffle=False, num_workers=0, collate_fn=collate)
    total_steps = max(total_steps, 1)
    n_samples = len(ds)
    print(f"{total_steps} optimizer steps, {a.micro_batch} x {a.grad_accum} sequences per step, {S} tokens each")
    if a.stage != "pretrain" and a.mm_weights:
        print("mix: " + ", ".join(f"{os.path.basename(p)} weight={w:g} rows={n}"
                                  for p, w, n in ds.source_counts()))

    # ---- optimiser ------------------------------------------------------------------------------
    betas = tuple(float(x) for x in a.betas.split(","))
    opt = torch.optim.AdamW(build_param_groups(model, a), lr=a.lr, betas=betas, eps=1e-8, fused=cuda)
    start_step, samples = 0, 0
    state_path = os.path.join(resume_dir, "train_state.pt")
    if a.resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location="cpu")
        opt.load_state_dict(st["opt"])
        if scaler is not None and st.get("scaler"):
            scaler.load_state_dict(st["scaler"])
        start_step, samples = st["step"], st["samples"]
        print(f"resumed at step {start_step}, sample {samples}")

    order = sample_order(n_samples, a.seed, samples) if a.stage != "pretrain" else iter(range(samples, 10 ** 12))
    dl = DataLoader(ds, batch_size=a.micro_batch, sampler=order, num_workers=a.workers, collate_fn=collate,
                    pin_memory=cuda, drop_last=True, persistent_workers=a.workers > 0,
                    prefetch_factor=4 if a.workers > 0 else None)
    it = iter(dl)

    warmup_steps = max(1, int(a.warmup * total_steps))
    log_f = open(os.path.join(a.out_dir, "log.jsonl"), "a")
    if policy is not None:
        policy.val_min_steps = int(a.early_stop_min_frac * total_steps)
        # A backoff that outlasts a twentieth of the run is not a correction, it is a new schedule.
        policy.backoff_steps = min(a.backoff_steps, max(10, total_steps // 20))
    stage_tokens = a.micro_batch * a.grad_accum * S if a.stage == "pretrain" else 0.0
    t_start = time.time()
    t0, tok_count, bad_steps, nonfinite = time.time(), 0, 0, False
    last_log_step = start_step
    stopped_early = False
    step = start_step
    while step < total_steps:
        done_tokens = (step + 1) * stage_tokens
        if a.plan and a.stage == "pretrain":
            lr = lr_at_tokens(done_tokens, budget_tokens, a.warmup, a.decay_frac, a.lr, a.min_lr_ratio)
        else:
            lr = lr_at(step, total_steps, warmup_steps, a.lr, a.min_lr_ratio, a.schedule, a.decay_frac)
        lr_scale = policy.lr_scale(step) if policy is not None else 1.0
        lr *= lr_scale
        for g in opt.param_groups:
            g["lr"] = lr * g["lr_mult"]
        if a.stage == "pretrain" and a.mtp_weight_final is not None:
            switch = (done_tokens >= (1 - a.decay_frac) * budget_tokens
                      if a.plan else step >= int(total_steps * (1 - a.decay_frac)))
            if switch:
                model.config.mtp_loss_weight = a.mtp_weight_final

        loss_sum = lm_sum = mtp_sum = 0.0
        n_mtp = 0
        for _ in range(a.grad_accum):
            batch = next(it)
            if a.stage == "pretrain":
                batch = dict(input_ids=batch, labels=batch)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            tok_count += batch["input_ids"].numel()
            with dtype_ctx():
                out = fwd_model(**batch)
            if out.loss.requires_grad:  # e.g. align stage with a micro-batch that has no image
                scaled = out.loss / a.grad_accum
                (scaler.scale(scaled) if scaler is not None else scaled).backward()
            loss_sum += float(out.loss.detach())
            lm_sum += float(out.lm_loss.detach())
            if out.mtp_loss is not None:
                mtp_sum += float(out.mtp_loss.detach())
                n_mtp += 1
        samples += a.micro_batch * a.grad_accum
        params = [p for p in model.parameters() if p.requires_grad]
        if scaler is not None:
            scaler.unscale_(opt)  # grads are still scaled: unscale before clipping / finiteness check
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, a.clip))
        if not math.isfinite(gnorm):
            bad_steps += 1
            nonfinite = True
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                # fp16 overflow: halve the loss scale and let the next step retry.  This is routine for the
                # first few steps (the initial scale of 65536 usually overflows once); only a run of them
                # means the training itself is broken.
                scaler.update()
                print(f"step {step}: fp16 overflow, step skipped, GradScaler scale -> {scaler.get_scale():.0f}")
            else:
                print(f"step {step}: non-finite gradient norm, step skipped ({bad_steps} in a row)")
            if bad_steps >= 10:
                raise RuntimeError("10 consecutive non-finite steps - lower the learning rate or check the data")
            step += 1
            if policy is not None:
                report_events(events_log, policy.observe(
                    dict(t=time.time(), step=step, loss=loss_sum / a.grad_accum, gnorm=gnorm, nonfinite=True)))
            loss_sum = lm_sum = mtp_sum = 0.0
            n_mtp = 0
            continue
        if scaler is not None:
            scaler.step(opt)  # also skips the update if an inf/nan slipped through the clip
            scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        bad_steps = 0  # the counter is about *consecutive* failures
        step += 1

        if step % a.log_every == 0 or step == total_steps:
            if cuda:
                torch.cuda.synchronize()
            dt = time.time() - t0
            rec = dict(t=time.time(), step=step, stage=a.stage, phase=(a.phase or {}).get("name"),
                       tokens=tok_count, tokens_total=done_tokens if stage_tokens else None,
                       total_steps=total_steps, budget_tokens=budget_tokens, elapsed_s=time.time() - t_start,
                       loss=loss_sum / a.grad_accum, lm=lm_sum / a.grad_accum,
                       mtp=(mtp_sum / n_mtp) if n_mtp else None, lr=lr, lr_scale=lr_scale, gnorm=gnorm,
                       grad_scale=(scaler.get_scale() if scaler is not None else None),
                       step_s=dt / max(1, step - last_log_step),
                       tok_s=tok_count / dt, nonfinite=nonfinite,
                       mem_gb=(torch.cuda.max_memory_allocated() / 1024 ** 3) if cuda else None)
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in rec.items() if k not in ("t", "tokens_total", "elapsed_s", "step_s")),
                  flush=True)
            log_f.write(json.dumps(json_safe({k: v for k, v in rec.items() if k not in ("nonfinite",)})) + "\n")
            log_f.flush()
            if mon is not None:
                mon.write(rec)
                report_events(events_log, policy.observe(rec))
            t0, tok_count, nonfinite, last_log_step = time.time(), 0, False, step
        if val_loaders and (step % a.eval_every == 0 or step == total_steps):
            res = evaluate(fwd_model, val_loaders, device, dtype_ctx, a.eval_batches)
            print("eval", {k: f"loss {v['loss']:.4f} ppl {v['ppl']:.2f}" for k, v in res.items()}, flush=True)
            events = policy.observe_eval(step, res) if policy is not None else []
            eval_rec = dict(t=time.time(), step=step, stage=a.stage, phase=(a.phase or {}).get("name"),
                            eval=res, best_val=policy.best_val() if policy is not None else None,
                            tokens_total=done_tokens if stage_tokens else None, total_steps=total_steps,
                            budget_tokens=budget_tokens, elapsed_s=time.time() - t_start)
            log_f.write(json.dumps(dict(step=step, eval={k: v["loss"] for k, v in res.items()})) + "\n")
            log_f.flush()
            if mon is not None:
                mon.write(eval_rec)
                report_events(events_log, events)
                if any(e["action"] == "stop" for e in events):
                    stopped_early = True
        if step % a.save_every == 0 and step < total_steps:
            save_checkpoint(model, opt, a.out_dir, "last", step, samples, scaler=scaler)
        if stopped_early:
            break

    save_checkpoint(model, opt, a.out_dir, "last", step, samples, scaler=scaler)
    save_checkpoint(model, None, a.out_dir, "final", step, samples, tokenizer_path=a.tokenizer)
    if mon is not None:
        mon.close()
        events_log.close()
    log_f.close()
    if stopped_early:
        print(f"stopped early at step {step}: {os.path.join(a.out_dir, 'last')} is the stopping point")
    print(f"done. final model: {os.path.join(a.out_dir, 'final')}")


if __name__ == "__main__":
    main()
