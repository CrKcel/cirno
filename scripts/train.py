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

STAGE_DEFAULTS = {
    "pretrain": dict(lr=1e-3, warmup=0.02, seq_len=1024, micro_batch=2, grad_accum=256, train="llm",
                     mtp_weight=0.3, vit_lr_mult=1.0),
    "align":    dict(lr=5e-4, warmup=0.05, seq_len=512, micro_batch=8, grad_accum=32, train="vit",
                     mtp_weight=0.0, vit_lr_mult=1.0),
    "sft":      dict(lr=2e-4, warmup=0.03, seq_len=1024, micro_batch=8, grad_accum=32, train="llm,vit",
                     mtp_weight=0.1, vit_lr_mult=0.5),
}


def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--stage", choices=list(STAGE_DEFAULTS), required=True)
    ap.add_argument("--config", default="configs/hvlm_130m.json", help="used when --init_from is not given")
    ap.add_argument("--init_from", default=None, help="save_pretrained directory to start from (stages 2/3)")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out_dir", required=True)
    # data
    ap.add_argument("--data", default=None, help="pretrain: dir:weight,dir:weight  (dirs from prepare_data.py)")
    ap.add_argument("--mm_data", default=None, help="align/sft: comma-separated jsonl files")
    ap.add_argument("--image_root", default=".")
    ap.add_argument("--val_mm_data", default=None, help="optional jsonl for validation in align/sft")
    # budget
    ap.add_argument("--seq_len", type=int, default=None)
    ap.add_argument("--micro_batch", type=int, default=None,
                    help="stage defaults: pretrain 2, align 8, sft 8; measured safe max on 8 GB with a "
                         "desktop running is 24 @seq1024 and 12 @seq2048 (see doc/architecture.md 3.2, "
                         "tools/measure_batch.py)")
    ap.add_argument("--grad_accum", type=int, default=None,
                    help="stage defaults: pretrain 256, align 32, sft 32 (mb*accum*seq = 524k/131k/262k tokens per step)")
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--tokens", type=float, default=None, help="pretrain: total training tokens (alternative to --max_steps)")
    ap.add_argument("--epochs", type=float, default=None, help="align/sft: number of passes over the data")
    # optimisation
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--vit_lr_mult", type=float, default=None)
    ap.add_argument("--min_lr_ratio", type=float, default=0.1)
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
    ap.add_argument("--grad_ckpt", choices=["auto", "on", "off"], default="auto",
                    help="per-layer activation checkpointing: auto = off when a micro-batch of the "
                         "configured shape fits without it (probed, costs one fwd/bwd), on otherwise")
    ap.add_argument("--ce_ckpt", choices=["auto", "on", "off"], default="auto",
                    help="recompute the vocab-head matmul in the CE backward: auto = off when the kept "
                         "logits fit (worth ~10%%), on otherwise")
    ap.add_argument("--no_grad_ckpt", action="store_true",
                    help="deprecated alias for --grad_ckpt off")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the training forward (Inductor). Measured +48%% on an 8 GB laptop card "
                         "(it also cuts activation memory by ~35%%, which is what lets --grad_ckpt auto turn "
                         "checkpointing off at a larger micro_batch). The first step pays a one-off JIT cost "
                         "(seconds to a minute; cached in TORCHINDUCTOR_CACHE_DIR, default /tmp/torchinductor_*). "
                         "Falls back to eager if compilation fails")
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
    a = ap.parse_args()
    if a.no_grad_ckpt:  # deprecated alias, kept so old command lines keep working
        a.grad_ckpt = "off"
    d = STAGE_DEFAULTS[a.stage]
    for k, v in d.items():
        if getattr(a, k, None) is None:
            setattr(a, k, v)
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


def probe_batch(model, device, micro_batch, seq_len, stage):
    """A synthetic batch of exactly the shape the trainer will use, for the memory probe below."""
    cfg = model.config
    ids = torch.randint(8, cfg.vocab_size, (micro_batch, seq_len), device=device)
    batch = dict(input_ids=ids, labels=ids)
    if stage != "pretrain":  # make the vision tower take part, so its activations count too
        vc = cfg.vision_config
        n = vc.num_image_tokens
        if seq_len < 4 + n + 2:
            return None
        ids[:, 2] = cfg.vision_start_token_id
        ids[:, 3:3 + n] = cfg.image_token_id
        ids[:, 3 + n] = cfg.vision_end_token_id
        labels = ids.clone()
        labels[:, :4 + n] = -100
        batch = dict(input_ids=ids, labels=labels,
                     pixel_values=torch.zeros(micro_batch, 3, vc.image_size, vc.image_size, device=device))
    return batch


def probe_step(fwd_model, batch, dtype_ctx, raw_model):
    """One fwd+bwd of `batch` that leaves no gradients behind.  Raises like a normal training step."""
    raw_model.zero_grad(set_to_none=True)
    with dtype_ctx():
        out = fwd_model(**batch)
    if out.loss.requires_grad:
        out.loss.backward()
    raw_model.zero_grad(set_to_none=True)


CKPT_HEADROOM_GB = 0.5  # allocator fragmentation + cuBLAS workspaces + the batch on its way to the GPU


def probe_fits(model, fwd_model, device, dtype_ctx, a, budget_gb):
    """One fwd+bwd of the real training shape; True when its peak leaves CKPT_HEADROOM_GB to spare.

    The AdamW moments (2 fp32 per trainable parameter) do not exist until the first opt.step(), so a
    stand-in buffer of exactly that size is held during the probe - without it the probe under-reports
    by ~1.1 GB here and cheerfully approves a micro_batch that OOMs two steps later."""
    batch = probe_batch(model, device, a.micro_batch, a.seq_len, a.stage)
    if batch is None:
        return None, "cannot probe (sequence too short for an image batch)"
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    torch.cuda.reset_peak_memory_stats()
    dummy = torch.empty(2 * n, device=device, dtype=torch.float32)
    try:
        probe_step(fwd_model, batch, dtype_ctx, model)
        peak = torch.cuda.max_memory_reserved() / 1024 ** 3
        why = f"peak {peak:.2f} GB vs {budget_gb:.2f} GB budget"
        return peak + CKPT_HEADROOM_GB <= budget_gb, why
    except torch.cuda.OutOfMemoryError:
        return False, "probe OOM"
    finally:
        model.zero_grad(set_to_none=True)
        del dummy, batch
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _candidates(a):
    """(grad_ckpt, ce_ckpt) pairs to try, cheapest-first: every "off" is work not repeated in backward.

    Both knobs are pure recompute-for-memory trades, so the only reason to switch either on is that the
    configured micro_batch otherwise does not fit.  An explicit --grad_ckpt/--ce_ckpt pins that axis."""
    grads = [False, True] if a.grad_ckpt == "auto" else [a.grad_ckpt == "on"]
    ces = [False, True] if a.ce_ckpt == "auto" else [a.ce_ckpt == "on"]
    return sorted(((g, c) for g in grads for c in ces), key=lambda gc: (gc[0], gc[1]))


def setup_execution(a, model, device, dtype_ctx, budget_gb):
    """Pick activation/CE checkpointing and (optionally) torch.compile by probing the real shape.

    Both checkpointing switches are mathematically free but repeat work in the backward pass (whole
    layers, and the vocab-head matmul), which on a power-limited card is exactly the compute worth not
    doing.  torch.compile fuses the elementwise chains and drops activation memory by ~35%, so it can
    turn a micro_batch that needs checkpointing into one that does not - hence probing the compiled
    graph, not just the eager one.  Each candidate costs one probe (one Inductor trace when compiled)."""
    fwd_model, compile_msg, compiled = model, "off", False
    ckpt_on = ce_on = True
    label = "grad_ckpt on, ce_ckpt on (no CUDA budget to probe against)"
    for ckpt_on, ce_on in _candidates(a):
        model.gradient_checkpointing_enable() if ckpt_on else model.gradient_checkpointing_disable()
        model.ce_checkpoint = ce_on
        if a.compile and not compiled:
            try:
                fwd_model = torch.compile(model, dynamic=False)
            except Exception as exc:  # pragma: no cover - depends on the local toolchain
                fwd_model, compile_msg = model, f"unavailable ({type(exc).__name__}: {exc})"
            else:
                compiled = True
        if budget_gb is None:  # CPU: nothing to probe, keep the conservative flags
            break
        t0 = time.time()
        fits, why = probe_fits(model, fwd_model, device, dtype_ctx, a, budget_gb)
        if compiled:
            compile_msg = f"on (first step {time.time() - t0:.0f}s, cached afterwards)"
        label = (f"grad_ckpt {'on' if ckpt_on else 'off'}, ce_ckpt {'on' if ce_on else 'off'} "
                 f"(auto: {why})")
        if fits or (ckpt_on and ce_on):  # accepted, or the most conservative candidate anyway
            break
    if a.grad_ckpt != "auto":  # an explicit choice wins over whatever the probe picked
        ckpt_on = a.grad_ckpt == "on"
    if a.ce_ckpt != "auto":
        ce_on = a.ce_ckpt == "on"
    model.gradient_checkpointing_enable() if ckpt_on else model.gradient_checkpointing_disable()
    model.ce_checkpoint = ce_on
    return fwd_model, ckpt_on, ce_on, label, compile_msg


@torch.no_grad()
def evaluate(model, loaders, device, dtype_ctx, n_batches):
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
            tot += float(out.lm_loss)
            n += 1
        if n:
            res[name] = tot / n
    model.train()
    return res


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
    budget_gb = None  # CUDA memory this process may use; drives the checkpointing probe
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
    fwd_model, ckpt_on, ce_on, ckpt_label, compile_msg = setup_execution(a, model, device, dtype_ctx, budget_gb)
    model.config.ce_checkpoint = ce_on  # record it, like the resolved gdn_impl
    print(f"checkpointing: {ckpt_label}")
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
        assert a.mm_data, "--mm_data file.jsonl[,file2.jsonl] is required for align/sft"
        sp = special_ids(tok)
        ds = ChatDataset(a.mm_data.split(","), a.image_root, tok, S, vc, sp)
        collate = ChatCollator(tok.convert_tokens_to_ids("<|pad|>"), sp["image_pad"], vc.merged_grid_size)
        per_step = a.micro_batch * a.grad_accum
        total_steps = a.max_steps or int((a.epochs or 1.0) * len(ds) / per_step)
        if a.val_mm_data:
            vds = ChatDataset(a.val_mm_data.split(","), a.image_root, tok, S, vc, sp)
            val_loaders["val"] = DataLoader(vds, batch_size=a.micro_batch, shuffle=False, num_workers=0, collate_fn=collate)
    total_steps = max(total_steps, 1)
    n_samples = len(ds)
    print(f"{total_steps} optimizer steps, {a.micro_batch} x {a.grad_accum} sequences per step, {S} tokens each")

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
    t0, tok_count, bad_steps = time.time(), 0, 0
    step = start_step
    while step < total_steps:
        lr = lr_at(step, total_steps, warmup_steps, a.lr, a.min_lr_ratio, a.schedule, a.decay_frac)
        for g in opt.param_groups:
            g["lr"] = lr * g["lr_mult"]
        if a.stage == "pretrain" and a.mtp_weight_final is not None and step >= int(total_steps * (1 - a.decay_frac)):
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
            loss_sum += float(out.loss)
            lm_sum += float(out.lm_loss)
            if out.mtp_loss is not None:
                mtp_sum += float(out.mtp_loss)
                n_mtp += 1
        samples += a.micro_batch * a.grad_accum
        params = [p for p in model.parameters() if p.requires_grad]
        if scaler is not None:
            scaler.unscale_(opt)  # grads are still scaled: unscale before clipping / finiteness check
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, a.clip))
        if not math.isfinite(gnorm):
            bad_steps += 1
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
            rec = dict(step=step, loss=loss_sum / a.grad_accum, lm=lm_sum / a.grad_accum,
                       mtp=(mtp_sum / n_mtp) if n_mtp else None, lr=lr, gnorm=gnorm,
                       tok_s=tok_count / dt, mem_gb=(torch.cuda.max_memory_allocated() / 1024 ** 3) if cuda else None)
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            t0, tok_count = time.time(), 0
        if val_loaders and (step % a.eval_every == 0 or step == total_steps):
            res = evaluate(fwd_model, val_loaders, device, dtype_ctx, a.eval_batches)
            print("eval", {k: f"loss {v:.4f} ppl {math.exp(v):.2f}" for k, v in res.items()}, flush=True)
            log_f.write(json.dumps(dict(step=step, eval=res)) + "\n")
            log_f.flush()
        if step % a.save_every == 0 and step < total_steps:
            save_checkpoint(model, opt, a.out_dir, "last", step, samples, scaler=scaler)

    save_checkpoint(model, opt, a.out_dir, "last", step, samples, scaler=scaler)
    save_checkpoint(model, None, a.out_dir, "final", step, samples, tokenizer_path=a.tokenizer)
    print(f"done. final model: {os.path.join(a.out_dir, 'final')}")


if __name__ == "__main__":
    main()
