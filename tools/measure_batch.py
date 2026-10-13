#!/usr/bin/env python
"""Measure peak GPU memory and throughput for (seq_len, micro_batch) on this machine.

Mirrors scripts/train.py: fp32 weights, autocast (--precision), per-layer checkpointing, fused AdamW, and
the Gated DeltaNet kernel selected by --gdn_impl.  Run it once per (precision, kernel) pair you intend to
train with - the fla kernels change both the memory ceiling and the throughput.
"""
import argparse
import json
import os
import sys
import time
from contextlib import nullcontext

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from hvlm.configuration_hvlm import HVLMConfig  # noqa: E402
from hvlm.gated_delta import resolve_impl  # noqa: E402
from hvlm.modeling_hvlm import HVLMForConditionalGeneration  # noqa: E402
from train import build_param_groups, resolve_precision, set_trainable  # noqa: E402


class _A:
    def __init__(self, lr, weight_decay, vit_lr_mult):
        self.lr = lr
        self.weight_decay = weight_decay
        self.vit_lr_mult = vit_lr_mult


def make_text_batch(mb, seq, device):
    ids = torch.randint(8, 32768, (mb, seq), device=device)
    return dict(input_ids=ids, labels=ids)


def make_image_batch(mb, seq, cfg, device):
    ids = torch.randint(8, 32768, (mb, seq), device=device)
    n = cfg.vision_config.num_image_tokens
    ids[:, 2] = cfg.vision_start_token_id
    ids[:, 3 : 3 + n] = cfg.image_token_id
    ids[:, 3 + n] = cfg.vision_end_token_id
    labels = ids.clone()
    labels[:, : 4 + n] = -100
    px = torch.randn(mb, 3, cfg.vision_config.image_size, cfg.vision_config.image_size, device=device)
    return dict(input_ids=ids, labels=ids, pixel_values=px)


def build_model(cfg, train, device, impl="auto", grad_ckpt=True, ce_ckpt=True, compile_=False):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    m = HVLMForConditionalGeneration(HVLMConfig(**cfg)).to(device).train()
    resolved = resolve_impl(impl, device, strict=(impl == "fla"))
    m.config.linear_attn_impl = resolved
    for mod in m.modules():
        if hasattr(mod, "impl"):
            mod.impl = resolved
    set_trainable(m, train)
    m.ce_checkpoint = ce_ckpt
    if grad_ckpt:
        m.gradient_checkpointing_enable()
    fwd = torch.compile(m, dynamic=False) if compile_ else m
    a = _A(lr=1e-4, weight_decay=0.1, vit_lr_mult=1.0 if train == "vit" else 0.5)
    opt = torch.optim.AdamW(build_param_groups(m, a), lr=a.lr, betas=(0.9, 0.95), eps=1e-8, fused=True)
    return m, opt, fwd


def trial(model, opt, batch, fwd=None, ac_dtype=torch.bfloat16, steps=3):
    """Peak memory over `steps` fwd/bwd/step iterations; throughput measured over steps 1..n-1
    (step 0 warms up the allocator, cuBLAS handles and any Triton autotuning/jit)."""
    fwd = fwd if fwd is not None else model
    scaler = torch.amp.GradScaler("cuda") if ac_dtype is torch.float16 else None
    ctx = torch.autocast("cuda", dtype=ac_dtype) if ac_dtype is not None else nullcontext()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    tokens = 0
    for i in range(steps):
        with ctx:
            loss = fwd(**batch).loss
        (scaler.scale(loss) if scaler is not None else loss).backward()
        if scaler is not None:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        if scaler is not None:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        tokens += batch["input_ids"].numel()
        if i == 0:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            tokens = 0
    torch.cuda.synchronize()
    dt = max(time.perf_counter() - t0, 1e-6)
    return dict(
        allocated_gb=torch.cuda.max_memory_allocated() / 1024 ** 3,
        reserved_gb=torch.cuda.max_memory_reserved() / 1024 ** 3,
        tok_s=tokens / dt,
        oom=False,
    )


def parse_grid(s):
    return [int(x) for x in s.split(",") if x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs/hvlm_130m.json"))
    ap.add_argument("--stage", choices=["pretrain", "align", "sft"], default="pretrain")
    ap.add_argument("--seq", default=None, help="comma-separated seq_len values")
    ap.add_argument("--micro_batch", default=None, help="comma-separated micro_batch values")
    ap.add_argument("--steps", type=int, default=3, help="fwd/bwd/step iterations per trial (first is warmup)")
    ap.add_argument("--gdn_impl", choices=["auto", "torch", "fla"], default="auto",
                    help="Gated DeltaNet kernel, as in scripts/train.py")
    ap.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
                    help="autocast dtype, as in scripts/train.py")
    ap.add_argument("--no_grad_ckpt", action="store_true", help="measure without per-layer checkpointing")
    ap.add_argument("--no_ce_ckpt", action="store_true", help="measure without recomputing the CE head matmul")
    ap.add_argument("--compile", action="store_true", help="measure with torch.compile, as in scripts/train.py")
    ap.add_argument("--max_mem_gb", type=float, default=0.0,
                    help="hard cap on this process's CUDA memory, like train.py (0 = off)")
    ap.add_argument("--reserve_gb", type=float, default=0.8,
                    help="headroom that must stay free after the peak (desktop, fragmentation) for the summary")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    cfg.pop("model_type", None)
    device = torch.device("cuda")
    ac_dtype, precision_label = resolve_precision(args.precision, device)
    args.gdn_impl = resolve_impl(args.gdn_impl, device, strict=True)
    print(f"precision {precision_label};  gdn_impl {args.gdn_impl};  grad_ckpt {not args.no_grad_ckpt};  "
          f"ce_ckpt {not args.no_ce_ckpt};  torch.compile {args.compile}")
    total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    free, _ = torch.cuda.mem_get_info()
    free_gb = free / 1024 ** 3
    print(f"device {torch.cuda.get_device_name(0)}  torch_total {total:.2f} GB  free {free_gb:.2f} GB")
    if args.max_mem_gb:
        budget = min(args.max_mem_gb, free_gb)
        torch.cuda.set_per_process_memory_fraction(max(budget / total, 1e-6), 0)
        print(f"CUDA allocator capped at {budget:.2f} GB")

    stage = {
        "pretrain": dict(train="llm", image=False, seq="512,1024,2048,3072,4096",
                         mb="4,8,12,16,20,24,32,40,48"),
        "align": dict(train="vit", image=True, seq="512,1024",
                      mb="4,8,16,24,32,48,64"),
        "sft": dict(train="llm,vit", image=True, seq="512,1024,2048",
                    mb="4,8,12,16,24,32"),
    }[args.stage]
    seqs = parse_grid(args.seq or stage["seq"])
    mbs = parse_grid(args.micro_batch or stage["mb"])
    make_batch = make_image_batch if stage["image"] else make_text_batch

    rows = []
    model = opt = None
    for seq in seqs:
        for mb in mbs:
            try:
                if model is None:
                    model, opt, fwd = build_model(cfg, stage["train"], device, args.gdn_impl,
                                                  grad_ckpt=not args.no_grad_ckpt,
                                                  ce_ckpt=not args.no_ce_ckpt, compile_=args.compile)
                batch = make_batch(mb, seq, model.config, device) if stage["image"] else make_batch(mb, seq, device)
                rec = trial(model, opt, batch, fwd=fwd, ac_dtype=ac_dtype, steps=args.steps)
                rec.update(stage=args.stage, seq_len=seq, micro_batch=mb, tokens=seq * mb,
                           precision=args.precision, gdn_impl=args.gdn_impl,
                           grad_ckpt=not args.no_grad_ckpt, ce_ckpt=not args.no_ce_ckpt, compile=args.compile)
                print(
                    f"  seq {seq:4d} x mb {mb:2d} ({seq * mb:6d} tok): "
                    f"alloc {rec['allocated_gb']:.2f}  reserved {rec['reserved_gb']:.2f}  "
                    f"tok/s {rec['tok_s']:.0f}"
                )
                del batch
                rows.append(rec)
            except torch.cuda.OutOfMemoryError:
                print(f"  seq {seq:4d} x mb {mb:2d}: OOM")
                rows.append(dict(
                    stage=args.stage, seq_len=seq, micro_batch=mb, tokens=seq * mb,
                    allocated_gb=None, reserved_gb=None, tok_s=None, oom=True,
                ))
                del model, opt
                model = opt = None
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                break
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")

    print(f"\nlargest micro_batch whose peak reserved + {args.reserve_gb:g} GB stays within the "
          f"{free_gb:.2f} GB that were free at start:")
    for seq in sorted({r["seq_len"] for r in rows}):
        ok = [r for r in rows if r["seq_len"] == seq and not r["oom"]
              and r["reserved_gb"] + args.reserve_gb <= free_gb]
        oomed = [r["micro_batch"] for r in rows if r["seq_len"] == seq and r["oom"]]
        if ok:
            best = max(ok, key=lambda r: r["micro_batch"])
            line = (f"  seq {seq:4d}: micro_batch <= {best['micro_batch']:2d}   "
                    f"(reserved {best['reserved_gb']:.2f} GB, {best['tok_s']:.0f} tok/s)")
        else:
            line = f"  seq {seq:4d}: nothing fits with {args.reserve_gb:g} GB to spare"
        if oomed:
            line += f"   first OOM at micro_batch {min(oomed)}"
        print(line)


if __name__ == "__main__":
    main()
