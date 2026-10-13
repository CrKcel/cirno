#!/usr/bin/env python
"""Pure-Python parameter count and training-memory estimate (no torch needed).

Usage:
    python tools/estimate.py configs/hvlm_130m.json
    python tools/estimate.py configs/hvlm_130m.json --seq_len 2048 --micro_batch 8

The memory model is an *analytic estimate* for: bf16 autocast, fp32 master
weights, AdamW (fp32 m and v), per-layer activation checkpointing, and a
chunked cross-entropy.  Measure the real number with
`torch.cuda.max_memory_allocated()` (train.py logs it).
"""
import argparse
import json


def count_params(c):
    d = c["hidden_size"]
    V = c["vocab_size"]
    L = c["num_hidden_layers"]
    ffn = c["intermediate_size"]
    nh, nkv, hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    Hk, Hv = c["linear_num_key_heads"], c["linear_num_value_heads"]
    Dk, Dv = c["linear_key_head_dim"], c["linear_value_head_dim"]
    K = c["linear_conv_kernel_dim"]
    interval = c["full_attention_interval"]

    key_dim, val_dim = Hk * Dk, Hv * Dv
    conv_ch = 2 * key_dim + val_dim

    mlp = 3 * d * ffn
    attn = d * (nh * hd * 2) + 2 * d * (nkv * hd) + nh * hd * d + 2 * hd  # q(+gate),k,v,o,qk-norm
    gdn = (
        d * (2 * key_dim + 2 * val_dim)  # in_proj_qkvz
        + d * 2 * Hv  # in_proj_ba
        + conv_ch * K  # depthwise conv
        + 2 * Hv  # A_log, dt_bias
        + Dv  # gated norm weight
        + val_dim * d  # out_proj
    )
    norms = 2 * d

    n_full = sum(1 for i in range(L) if (i + 1) % interval == 0)
    n_lin = L - n_full

    embed = V * d
    backbone_layers = n_full * (attn + mlp + norms) + n_lin * (gdn + mlp + norms)
    final_norm = d

    mtp = 0
    for _ in range(c.get("mtp_num_layers", 0)):
        mtp += 2 * d + 2 * d * d + (attn + mlp + norms) + d  # 2 pre-norms, fc, block, final norm

    v = c["vision_config"]
    dv, fv, nl, p = v["hidden_size"], v["intermediate_size"], v["depth"], v["patch_size"]
    g = v["image_size"] // p
    m = v["spatial_merge_size"]
    patch = v["in_channels"] * p * p * dv + dv
    pos = g * g * dv
    vblock = 4 * dv + (dv * 3 * dv + 3 * dv) + (dv * dv + dv) + (dv * fv + fv) + (fv * dv + dv)
    merger_in = dv * m * m
    merger = 2 * dv + (merger_in * merger_in + merger_in) + (merger_in * d + d)
    vision = patch + pos + nl * vblock + merger

    out = dict(
        embed_tied_head=embed,
        backbone_layers=backbone_layers,
        final_norm=final_norm,
        n_full=n_full,
        n_linear=n_lin,
        per_layer_full=attn + mlp + norms,
        per_layer_linear=gdn + mlp + norms,
        mtp=mtp,
        vision=vision,
    )
    out["llm_inference"] = embed + backbone_layers + final_norm
    out["total_with_vision_no_mtp"] = out["llm_inference"] + vision
    out["total_training"] = out["total_with_vision_no_mtp"] + mtp
    out["non_embedding_llm"] = backbone_layers + final_norm
    return out


def estimate_memory(c, n_params, seq_len, micro_batch, ce_chunk=2048, overhead_gb=1.0):
    d = c["hidden_size"]
    V = c["vocab_size"]
    L = c["num_hidden_layers"]
    ffn = c["intermediate_size"]
    Hv, Dv = c["linear_num_value_heads"], c["linear_value_head_dim"]
    Hk, Dk = c["linear_num_key_heads"], c["linear_key_head_dim"]
    nh, hd = c["num_attention_heads"], c["head_dim"]
    tokens = seq_len * micro_batch
    GB = 1024 ** 3

    weights = 4 * n_params  # fp32 master
    grads = 4 * n_params
    adam = 8 * n_params
    states = weights + grads + adam

    # per-layer checkpoint: fp32 residual stream saved at each layer boundary (+MTP block input)
    ckpt = (L + 2) * tokens * d * 4

    # transient peak while recomputing / back-propagating ONE layer (rough, deliberately generous)
    chunk = 64
    gdn_fp32 = tokens * (2 * Hv * Dk + 2 * Hv * Dv) * 4 * 3  # q,k,v,o fp32 copies (+grad + temp)
    gdn_chunk = micro_batch * Hv * seq_len * chunk * 4 * 8  # decay mask / attn / T matrices
    gdn_proj = tokens * (2 * Hk * Dk + 2 * Hv * Dv) * 2 * 2
    mlp = tokens * ffn * 2 * 4 + tokens * d * 4 * 4
    attn = tokens * (nh * hd * 2 * 2 + nh * hd * 2 * 3)
    layer_peak = max(gdn_fp32 + gdn_chunk + gdn_proj, attn) + mlp

    ce = ce_chunk * V * (2 + 4 + 4)  # bf16 logits + fp32 logits + fp32 grad
    embed_act = tokens * d * 4 * 2

    transient = layer_peak + ce + embed_act
    total = states + ckpt + transient + overhead_gb * GB
    return dict(
        tokens_per_micro_batch=tokens,
        weights_GB=weights / GB,
        grads_GB=grads / GB,
        adam_GB=adam / GB,
        ckpt_GB=ckpt / GB,
        layer_peak_GB=layer_peak / GB,
        ce_GB=ce / GB,
        overhead_GB=overhead_gb,
        total_GB=total / GB,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--micro_batch", type=int, default=8)
    args = ap.parse_args()
    c = json.load(open(args.config))
    p = count_params(c)
    M = 1e6
    print(f"layers: {c['num_hidden_layers']}  ({p['n_linear']} GatedDeltaNet + {p['n_full']} GatedAttention, 3:1)")
    print(f"embedding (tied lm_head)   : {p['embed_tied_head'] / M:7.2f} M")
    print(f"decoder layers             : {p['backbone_layers'] / M:7.2f} M"
          f"   (linear layer {p['per_layer_linear'] / M:.2f}M, full layer {p['per_layer_full'] / M:.2f}M)")
    print(f"LLM at inference           : {p['llm_inference'] / M:7.2f} M   (non-embedding {p['non_embedding_llm'] / M:.2f} M)")
    print(f"vision tower + merger      : {p['vision'] / M:7.2f} M")
    print(f"MTP module (train only)    : {p['mtp'] / M:7.2f} M")
    print(f"TOTAL inference (LLM+ViT)  : {p['total_with_vision_no_mtp'] / M:7.2f} M")
    print(f"TOTAL training (+MTP)      : {p['total_training'] / M:7.2f} M")
    print()
    print("Estimated peak GPU memory (analytic):")
    for mb in sorted({2, 4, args.micro_batch, 16}):
        m = estimate_memory(c, p["total_training"], args.seq_len, mb)
        print(f"  seq {args.seq_len} x micro_batch {mb:2d} ({m['tokens_per_micro_batch']:6d} tok): "
              f"states {m['weights_GB'] + m['grads_GB'] + m['adam_GB']:.2f} + ckpt {m['ckpt_GB']:.2f} "
              f"+ layer {m['layer_peak_GB']:.2f} + CE {m['ce_GB']:.2f} + overhead {m['overhead_GB']:.1f} "
              f"= {m['total_GB']:.2f} GB")


if __name__ == "__main__":
    main()
