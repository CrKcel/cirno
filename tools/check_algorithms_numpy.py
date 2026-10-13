#!/usr/bin/env python
"""Torch-free verification of the algorithmic parts (numpy only).

1. Chunked gated delta rule  ==  token-by-token recurrence (the definition).
   The chunked function below mirrors, line by line, hvlm/gated_delta.py::chunk_gated_delta_rule.
2. M-RoPE axis map and position-id builder.

Run:  python tools/check_algorithms_numpy.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hvlm.positions import build_mrope_positions, mrope_axis_map  # noqa: E402


# ----------------------------------------------------------------------------------------
# reference: the definition of the gated delta rule, one token at a time
#   S_t = a_t * S_{t-1} + b_t * k_t (v_t - a_t * S_{t-1}^T k_t)^T      (S: [Dk, Dv])
#   o_t = S_t^T q_t
# ----------------------------------------------------------------------------------------
def recurrent_gated_delta_rule(q, k, v, g, beta, S0=None):
    B, T, H, Dk = k.shape
    Dv = v.shape[-1]
    q = q.astype(np.float64)
    k = k.astype(np.float64)
    v = v.astype(np.float64)
    q = q / np.sqrt(np.sum(q * q, -1, keepdims=True) + 1e-6)
    k = k / np.sqrt(np.sum(k * k, -1, keepdims=True) + 1e-6)
    q = q * Dk ** -0.5
    S = np.zeros((B, H, Dk, Dv)) if S0 is None else S0.copy()
    out = np.zeros((B, T, H, Dv))
    for t in range(T):
        a = np.exp(g[:, t])[..., None, None]  # [B,H,1,1]
        S = S * a
        kt, vt, qt, bt = k[:, t], v[:, t], q[:, t], beta[:, t][..., None]
        pred = np.einsum("bhkd,bhk->bhd", S, kt)
        S = S + np.einsum("bhk,bhd->bhkd", kt, bt * (vt - pred))
        out[:, t] = np.einsum("bhkd,bhk->bhd", S, qt)
    return out, S


# ----------------------------------------------------------------------------------------
# chunked algorithm (mirror of the torch implementation)
# ----------------------------------------------------------------------------------------
def chunk_gated_delta_rule(q, k, v, g, beta, chunk=64, S0=None, dtype=np.float64):
    B, T, H, Dk = k.shape
    Dv = v.shape[-1]
    tr = lambda x: np.ascontiguousarray(np.swapaxes(x, 1, 2)).astype(dtype)  # [B,T,H,*] -> [B,H,T,*]
    q, k, v = tr(q), tr(k), tr(v)
    g, beta = tr(g[..., None])[..., 0], tr(beta[..., None])[..., 0]
    q = q / np.sqrt(np.sum(q * q, -1, keepdims=True) + dtype(1e-6))
    k = k / np.sqrt(np.sum(k * k, -1, keepdims=True) + dtype(1e-6))
    q = q * dtype(Dk ** -0.5)

    pad = (-T) % chunk
    if pad:
        q = np.pad(q, ((0, 0), (0, 0), (0, pad), (0, 0)))
        k = np.pad(k, ((0, 0), (0, 0), (0, pad), (0, 0)))
        v = np.pad(v, ((0, 0), (0, 0), (0, pad), (0, 0)))
        beta = np.pad(beta, ((0, 0), (0, 0), (0, pad)))
        g = np.pad(g, ((0, 0), (0, 0), (0, pad)))
    Tp = T + pad
    N = Tp // chunk
    C = chunk

    v_beta = v * beta[..., None]
    k_beta = k * beta[..., None]
    q, k, v, k_beta, v_beta = [x.reshape(B, H, N, C, -1) for x in (q, k, v, k_beta, v_beta)]
    g = np.cumsum(g.reshape(B, H, N, C), axis=-1)

    lower = np.tril(np.ones((C, C), dtype=bool))
    strict = np.tril(np.ones((C, C), dtype=bool), -1)
    diff = g[..., :, None] - g[..., None, :]
    decay = np.where(lower, np.exp(np.where(lower, diff, 0.0)), 0.0)  # exp(g_i - g_j), zero above diagonal

    Lm = np.where(strict, (k_beta @ np.swapaxes(k, -1, -2)) * decay, 0.0)
    eye = np.eye(C, dtype=dtype)
    Tm = np.linalg.solve(eye + Lm, np.broadcast_to(eye, Lm.shape))  # (I + L)^-1   (lower-triangular solve)
    u = Tm @ v_beta
    w = Tm @ (k_beta * np.exp(g)[..., None])

    S = np.zeros((B, H, Dk, Dv), dtype=dtype) if S0 is None else S0.astype(dtype).copy()
    outs = []
    for i in range(N):
        q_i, k_i, u_i, w_i, g_i = q[:, :, i], k[:, :, i], u[:, :, i], w[:, :, i], g[:, :, i]
        attn_i = (q_i @ np.swapaxes(k_i, -1, -2)) * decay[:, :, i]
        v_new = u_i - w_i @ S
        o_i = (q_i * np.exp(g_i)[..., None]) @ S + attn_i @ v_new
        outs.append(o_i)
        g_last = g_i[..., -1]
        S = S * np.exp(g_last)[..., None, None] + np.swapaxes(k_i * np.exp(g_last[..., None] - g_i)[..., None], -1, -2) @ v_new
    out = np.stack(outs, axis=2).reshape(B, H, Tp, Dv)[:, :, :T]
    return np.swapaxes(out, 1, 2), S


def make_inputs(rng, B, T, H, Dk, Dv, g_scale):
    q = rng.standard_normal((B, T, H, Dk))
    k = rng.standard_normal((B, T, H, Dk))
    v = rng.standard_normal((B, T, H, Dv))
    # same parameterisation as the model: g = -exp(A_log) * softplus(a + dt_bias) <= 0 ; beta = sigmoid(b)
    A = np.exp(rng.uniform(np.log(1e-3), np.log(g_scale), size=(H,)))
    a = rng.standard_normal((B, T, H))
    g = -A * np.log1p(np.exp(a + 1.0))
    beta = 1.0 / (1.0 + np.exp(-rng.standard_normal((B, T, H))))
    return q, k, v, g, beta


def check_gdn():
    rng = np.random.default_rng(0)
    ok = True
    cases = [
        ("T multiple of chunk", dict(B=2, T=128, H=3, Dk=16, Dv=24, g=16.0), 32),
        ("T not multiple (padding)", dict(B=2, T=77, H=3, Dk=16, Dv=24, g=16.0), 32),
        ("T < chunk", dict(B=1, T=5, H=2, Dk=8, Dv=8, g=16.0), 64),
        ("slow decay (long memory)", dict(B=2, T=200, H=2, Dk=32, Dv=32, g=0.05), 64),
        ("very fast decay", dict(B=2, T=200, H=2, Dk=32, Dv=32, g=500.0), 64),
        ("model-size heads, chunk 64", dict(B=1, T=256, H=8, Dk=64, Dv=64, g=16.0), 64),
    ]
    for name, c, chunk in cases:
        q, k, v, g, beta = make_inputs(rng, c["B"], c["T"], c["H"], c["Dk"], c["Dv"], c["g"])
        ref, S_ref = recurrent_gated_delta_rule(q, k, v, g, beta)
        got, S_got = chunk_gated_delta_rule(q, k, v, g, beta, chunk=chunk)
        got32, _ = chunk_gated_delta_rule(q, k, v, g, beta, chunk=chunk, dtype=np.float32)
        e64 = np.max(np.abs(ref - got))
        e32 = np.max(np.abs(ref - got32))
        es = np.max(np.abs(S_ref - S_got))
        flag = e64 < 1e-9 and es < 1e-9 and e32 < 1e-4 and np.isfinite(got32).all()
        ok &= bool(flag)
        print(f"[{'PASS' if flag else 'FAIL'}] {name:28s} max|chunk64-ref|={e64:.2e}  state={es:.2e}  chunk-fp32={e32:.2e}")

    # carrying a state across two calls must equal one long call (needed for cache-based decoding)
    q, k, v, g, beta = make_inputs(rng, 1, 150, 2, 16, 16, 16.0)
    full, _ = chunk_gated_delta_rule(q, k, v, g, beta, chunk=32)
    o1, S1 = chunk_gated_delta_rule(q[:, :90], k[:, :90], v[:, :90], g[:, :90], beta[:, :90], chunk=32)
    o2, _ = chunk_gated_delta_rule(q[:, 90:], k[:, 90:], v[:, 90:], g[:, 90:], beta[:, 90:], chunk=32, S0=S1)
    e = np.max(np.abs(full - np.concatenate([o1, o2], 1)))
    flag = e < 1e-9
    ok &= bool(flag)
    print(f"[{'PASS' if flag else 'FAIL'}] {'state carry across calls':28s} max err={e:.2e}")

    # causality: changing a future token must not change earlier outputs
    q, k, v, g, beta = make_inputs(rng, 1, 100, 2, 16, 16, 16.0)
    a, _ = chunk_gated_delta_rule(q, k, v, g, beta, chunk=32)
    v2 = v.copy()
    v2[:, 70:] += 5.0
    b, _ = chunk_gated_delta_rule(q, k, v2, g, beta, chunk=32)
    e = np.max(np.abs(a[:, :70] - b[:, :70]))
    flag = e == 0.0 or e < 1e-12
    ok &= bool(flag)
    print(f"[{'PASS' if flag else 'FAIL'}] {'causality':28s} max change in past={e:.2e}")
    return ok


def check_positions():
    ok = True
    sec = [8, 4, 4]
    ax = mrope_axis_map(sec)
    counts = [int((ax == a).sum()) for a in range(3)]
    flag = counts == sec and len(ax) == sum(sec) and list(ax[:6]) == [0, 1, 2, 0, 1, 2]
    ok &= flag
    print(f"[{'PASS' if flag else 'FAIL'}] mrope axis map {ax.tolist()} counts={counts}")

    IMG, START, END = 5, 3, 4
    g = 7
    ids = np.array([10, 11, 12, START] + [IMG] * (g * g) + [END, 20, 21])
    pos = build_mrope_positions(ids, IMG, g)
    exp_t = [0, 1, 2, 3] + [4] * 49 + [11, 12, 13]
    exp_h = [0, 1, 2, 3] + [4 + r for r in range(7) for _ in range(7)] + [11, 12, 13]
    exp_w = [0, 1, 2, 3] + [4 + c for _ in range(7) for c in range(7)] + [11, 12, 13]
    flag = pos[0].tolist() == exp_t and pos[1].tolist() == exp_h and pos[2].tolist() == exp_w
    ok &= flag
    print(f"[{'PASS' if flag else 'FAIL'}] mrope positions: image block + text resume (next text pos = {pos[0, -3]})")

    txt = np.arange(30) + 100
    p = build_mrope_positions(txt, IMG, g)
    flag = (p == np.arange(30)[None]).all()
    ok &= bool(flag)
    print(f"[{'PASS' if flag else 'FAIL'}] text-only: three axes equal arange (plain RoPE)")

    two = np.array([START] + [IMG] * 49 + [END, 30, START] + [IMG] * 49 + [END])
    p = build_mrope_positions(two, IMG, g)
    # layout (index: token -> t): 0 START->0 | 1..49 image1->1 (text resumes at 1+7=8) | 50 END->8 |
    # 51 text->9 | 52 START->10 | 53..101 image2->11 (rows/cols 11..17) | 102 END->18
    flag = (
        p[0, 0] == 0 and p[0, 1] == 1 and p[0, 49] == 1 and p[0, 50] == 8 and p[0, 51] == 9
        and p[0, 52] == 10 and p[0, 53] == 11 and p[0, 101] == 11 and p[0, 102] == 18
        and p[1, 101] == 17 and p[2, 101] == 17 and p[1, 53] == 11 and p[2, 59] == 17
    )
    ok &= bool(flag)
    print(f"[{'PASS' if flag else 'FAIL'}] two images in one sequence (second image t={p[0, 53]}, next text={p[0, 102]})")
    return ok


if __name__ == "__main__":
    a = check_gdn()
    b = check_positions()
    print("\nALL PASS" if (a and b) else "\nSOME CHECKS FAILED")
    sys.exit(0 if (a and b) else 1)
