#!/usr/bin/env python
"""Single-GPU trainer for the hybrid VLM.  Designed to stay under ~8 GB:

  * fp32 master weights + AdamW (fp32 states) + bf16 autocast
  * per-layer activation checkpointing (--no_grad_ckpt to disable)
  * chunked, checkpointed cross-entropy against the tied head (never materialises [tokens, vocab] logits)
  * optional hard cap on the CUDA allocator (--max_mem_gb 8, never above what is currently free) so an
    out-of-budget config fails fast instead of competing with the desktop or OOMing the device.

Stages (see README):
  pretrain : text only, packed windows from prepared token bins        (--stage pretrain --data dirA:0.6,dirB:0.4)
  align    : image-caption data, ViT + projector trained, LLM frozen   (--stage align    --mm_data caps.jsonl --image_root ...)
  sft      : instruction / VQA / text chat data, everything trained    (--stage sft      --mm_data sft.jsonl  --image_root ...)
"""
import argparse
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
from hvlm.modeling_hvlm import HVLMForConditionalGeneration  # noqa: E402

STAGE_DEFAULTS = {
    #            lr     warmup  seq   mb  accum  train_modules  mtp_weight  vit_lr_mult
    # micro_batch / grad_accum keep the stage's tokens per step unchanged (pretrain 16*32*1024 =
    # 524k, align 8*32*512 = 131k, sft 8*32*1024 = 262k); see README for the measured caps.
    "pretrain": dict(lr=1e-3, warmup=0.02, seq_len=1024, micro_batch=16, grad_accum=32, train="llm",
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
                    help="stage defaults: pretrain 16, align 8, sft 8; measured safe max on 8 GB with a "
                         "desktop running is 24 @seq1024 and 12 @seq2048 (see README, tools/measure_batch.py)")
    ap.add_argument("--grad_accum", type=int, default=None,
                    help="stage defaults: pretrain 32, align 32, sft 32 (mb*accum*seq = 524k/131k/262k tokens per step)")
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
    ap.add_argument("--gdn_impl", choices=["torch", "fla"], default="torch")
    # system
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_mem_gb", type=float, default=8.0,
                    help="hard cap on this process's CUDA memory, also bounded by what is free at start (0 = off)")
    ap.add_argument("--no_grad_ckpt", action="store_true")
    ap.add_argument("--tf32", action="store_true", help="allow TF32 for fp32 matmuls (off by default: the delta-rule core is an fp32 island)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_batches", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    d = STAGE_DEFAULTS[a.stage]
    for k, v in d.items():
        if getattr(a, k, None) is None:
            setattr(a, k, v)
    return a


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


def save_checkpoint(model, opt, out_dir, tag, step, samples, tokenizer_path=None):
    tmp, dst = os.path.join(out_dir, tag + ".tmp"), os.path.join(out_dir, tag)
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp)
    if opt is not None:
        torch.save(dict(opt=opt.state_dict(), step=step, samples=samples), os.path.join(tmp, "train_state.pt"))
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
    ac_dtype = torch.bfloat16
    dtype_ctx = lambda: torch.autocast(device_type=device.type, dtype=ac_dtype)

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
    model.config.linear_attn_impl = a.gdn_impl
    for m in model.modules():
        if hasattr(m, "impl"):
            m.impl = a.gdn_impl
    assert len(tok) <= model.config.vocab_size, "tokenizer larger than model vocab"
    model.to(device)
    set_trainable(model, a.train)
    if not a.no_grad_ckpt:
        model.gradient_checkpointing_enable()
    model.train()
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
                out = model(**batch)
            if out.loss.requires_grad:  # e.g. align stage with a micro-batch that has no image
                (out.loss / a.grad_accum).backward()
            loss_sum += float(out.loss)
            lm_sum += float(out.lm_loss)
            if out.mtp_loss is not None:
                mtp_sum += float(out.mtp_loss)
                n_mtp += 1
        samples += a.micro_batch * a.grad_accum
        params = [p for p in model.parameters() if p.requires_grad]
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, a.clip))
        if not math.isfinite(gnorm):
            bad_steps += 1
            opt.zero_grad(set_to_none=True)
            print(f"step {step}: non-finite gradient norm, step skipped ({bad_steps} so far)")
            if bad_steps >= 10:
                raise RuntimeError("10 non-finite steps - lower the learning rate or check the data")
            step += 1
            continue
        opt.step()
        opt.zero_grad(set_to_none=True)
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
            res = evaluate(model, val_loaders, device, dtype_ctx, a.eval_batches)
            print("eval", {k: f"loss {v:.4f} ppl {math.exp(v):.2f}" for k, v in res.items()}, flush=True)
            log_f.write(json.dumps(dict(step=step, eval=res)) + "\n")
            log_f.flush()
        if step % a.save_every == 0 and step < total_steps:
            save_checkpoint(model, opt, a.out_dir, "last", step, samples)

    save_checkpoint(model, opt, a.out_dir, "last", step, samples)
    save_checkpoint(model, None, a.out_dir, "final", step, samples, tokenizer_path=a.tokenizer)
    print(f"done. final model: {os.path.join(a.out_dir, 'final')}")


if __name__ == "__main__":
    main()
