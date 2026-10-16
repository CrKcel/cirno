#!/usr/bin/env python
"""Pure-Python parameter count and training-memory estimate (no torch needed).

Usage:
    python tools/estimate.py configs/hvlm_130m.json
    python tools/estimate.py configs/hvlm_130m.json --seq_len 2048 --micro_batch 2

The memory model is an *analytic estimate* for: bf16 autocast, fp32 master
weights, AdamW (fp32 m and v), no activation checkpointing, and a chunked
cross-entropy.  It counts the eager graph, so a `--compile` run comes out
measurably below it.  Measure the real number with
`torch.cuda.max_memory_allocated()` (`tools/measure_batch.py`, or the `mem_gb`
that train.py logs).
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


def _activation_bytes(c, seq_len, micro_batch):
    """Bytes the backward pass keeps for one micro-batch of bf16 activations (nothing recomputed).

    Every linear keeps its input and output; the delta rule additionally keeps q/k/v/g/beta and its
    per-chunk matrices, the full-attention layers keep their q/k/v and the projection output (the
    memory-efficient SDPA backend used on CUDA does not materialise the score matrix), and the fp32
    residual stream is retained at every layer boundary.  The path from the vision output to the loss
    is part of the graph even when the backbone is frozen, so align pays the same activation memory as
    pretrain - only the optimizer `states` differ between stages.
    """
    d, ffn = c["hidden_size"], c["intermediate_size"]
    Hk, Dk = c["linear_num_key_heads"], c["linear_key_head_dim"]
    Hv, Dv = c["linear_num_value_heads"], c["linear_value_head_dim"]
    nh, hd, nkv = c["num_attention_heads"], c["head_dim"], c["num_key_value_heads"]
    tokens = seq_len * micro_batch
    key_dim, val_dim = Hk * Dk, Hv * Dv
    conv_ch = 2 * key_dim + val_dim

    residual = 4 * d                                       # fp32 residual stream at the layer boundary
    norms = 3 * 2 * d                                      # two norms' inputs, one norm's output
    mlp = 2 * ((d + ffn) * 2 + (ffn + d) + 2 * ffn)        # gate/up io, silu product, down io
    linear = 2 * ((d + 2 * key_dim + 2 * val_dim)          # in_proj_qkvz
                  + (d + 2 * Hv)                           # in_proj_ba
                  + (val_dim + d)                          # out_proj
                  + conv_ch                                # conv output
                  + (2 * key_dim + val_dim + 2 * Hv)       # q/k/v/g/beta handed to the delta rule
                  + Dv) + residual + norms                 # gated norm
    full = 2 * ((d + 2 * nh * hd)                          # q proj (+ gate)
                + (d + nkv * hd) * 2                       # k/v proj
                + (nh * hd + d)                            # o proj
                + 2 * nh * hd + 2 * nkv * hd) + residual + norms
    # the chunked delta rule keeps one fp32 [chunk, chunk] matrix per batch, value head and chunk
    chunk_matrices = micro_batch * Hv * seq_len * c.get("linear_chunk_size", 64) * 4 * 8
    n_full = sum(1 for i in range(c["num_hidden_layers"]) if (i + 1) % c["full_attention_interval"] == 0)
    n_lin = c["num_hidden_layers"] - n_full
    mtp = c.get("mtp_num_layers", 0) * (full + mlp + 4 * d)  # fusion fc + one attention block
    return tokens * (n_lin * (linear + mlp) + n_full * (full + mlp) + mtp) + n_lin * chunk_matrices


def estimate_memory(c, n_params, seq_len, micro_batch, ce_chunk=2048, overhead_gb=1.0):
    V = c["vocab_size"]
    tokens = seq_len * micro_batch
    GB = 1024 ** 3

    weights = 4 * n_params  # fp32 master
    grads = 4 * n_params
    adam = 8 * n_params
    states = weights + grads + adam

    acts = _activation_bytes(c, seq_len, micro_batch)
    # the head's bf16 logits are kept for its gradient; the fp32 cross-entropy internals of one chunk
    # exist only while that chunk is computed
    ce = tokens * V * 2 + ce_chunk * V * 8
    embed_act = tokens * c["hidden_size"] * 4 * 2

    total = states + acts + ce + embed_act + overhead_gb * GB
    return dict(
        tokens_per_micro_batch=tokens,
        states_GB=states / GB,
        activations_GB=acts / GB,
        ce_GB=(ce + embed_act) / GB,
        overhead_GB=overhead_gb,
        total_GB=total / GB,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--micro_batch", type=int, default=2)
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
    for mb in sorted({1, 2, 4, args.micro_batch}):
        m = estimate_memory(c, p["total_training"], args.seq_len, mb)
        print(f"  seq {args.seq_len} x micro_batch {mb:2d} ({m['tokens_per_micro_batch']:6d} tok): "
              f"states {m['states_GB']:.2f} + activations {m['activations_GB']:.2f} "
              f"+ CE {m['ce_GB']:.2f} + overhead {m['overhead_GB']:.1f} = {m['total_GB']:.2f} GB")


if __name__ == "__main__":
    main()
