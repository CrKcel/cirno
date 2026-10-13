"""Gated delta rule (the linear-attention core of Gated DeltaNet) in pure PyTorch.

Recurrence, per head, state S in R^{Dk x Dv}:
    S_t = a_t * S_{t-1} + b_t * k_t (v_t - a_t * S_{t-1}^T k_t)^T        a_t = exp(g_t) in (0, 1]
    o_t = S_t^T q_t
q and k are L2-normalised, q is scaled by Dk^-0.5.

`chunk_gated_delta_rule` is the training-time algorithm (parallel inside chunks of C tokens, a short
sequential loop over the T/C chunks).  It is a line-by-line twin of the numpy implementation that
tools/check_algorithms_numpy.py verifies against the recurrence.  `recurrent_gated_delta_rule` is the
definition itself (slow; used for tests and, with a carried state, for token-by-token decoding).

Inputs are [B, T, H, D] (g and beta: [B, T, H]).  All math is done in float32.
"""
import torch
import torch.nn.functional as F


def l2norm(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def recurrent_gated_delta_rule(q, k, v, g, beta, initial_state=None):
    B, T, H, Dk = k.shape
    Dv = v.shape[-1]
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    q = l2norm(q) * Dk ** -0.5
    k = l2norm(k)
    S = k.new_zeros(B, H, Dk, Dv) if initial_state is None else initial_state.float()
    outs = []
    for t in range(T):
        S = S * g[:, t].exp()[..., None, None]
        kt, vt, qt = k[:, t], v[:, t], q[:, t]
        bt = beta[:, t][..., None]
        pred = (S * kt[..., None]).sum(-2)  # S^T k   [B,H,Dv]
        S = S + kt[..., None] * (bt * (vt - pred))[..., None, :]
        outs.append((S * qt[..., None]).sum(-2))
    return torch.stack(outs, dim=1), S


def chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=64, initial_state=None):
    B, T, H, Dk = k.shape
    Dv = v.shape[-1]
    C = chunk_size
    q, k, v = (x.transpose(1, 2).float() for x in (q, k, v))  # [B,H,T,D]
    g, beta = (x.transpose(1, 2).float() for x in (g, beta))  # [B,H,T]
    q = l2norm(q) * Dk ** -0.5
    k = l2norm(k)

    pad = (-T) % C
    if pad:  # zero padding is inert: k = v = beta = 0 (no write), g = 0 (no decay)
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        g, beta = (F.pad(x, (0, pad)) for x in (g, beta))
    Tp = T + pad
    N = Tp // C

    k_beta = k * beta[..., None]
    v_beta = v * beta[..., None]
    q, k, v, k_beta, v_beta = (x.reshape(B, H, N, C, -1) for x in (q, k, v, k_beta, v_beta))
    g = g.reshape(B, H, N, C).cumsum(-1)  # cumulative log-decay inside each chunk

    dev = q.device
    lower = torch.tril(torch.ones(C, C, dtype=torch.bool, device=dev))  # i >= j
    strict = torch.tril(torch.ones(C, C, dtype=torch.bool, device=dev), diagonal=-1)  # i > j
    diff = g[..., :, None] - g[..., None, :]  # g_i - g_j
    decay = diff.masked_fill(~lower, 0.0).exp().masked_fill(~lower, 0.0)  # exp(g_i - g_j) for i >= j else 0

    L = ((k_beta @ k.transpose(-1, -2)) * decay).masked_fill(~strict, 0.0)  # strictly lower triangular
    eye = torch.eye(C, dtype=q.dtype, device=dev)
    Tm = torch.linalg.solve_triangular(eye + L, eye.expand_as(L).contiguous(), upper=False)  # (I + L)^-1
    u = Tm @ v_beta  # [B,H,N,C,Dv]
    w = Tm @ (k_beta * g.exp()[..., None])  # [B,H,N,C,Dk]

    S = q.new_zeros(B, H, Dk, Dv) if initial_state is None else initial_state.float()
    outs = []
    for i in range(N):
        q_i, k_i, u_i, w_i, g_i, d_i = q[:, :, i], k[:, :, i], u[:, :, i], w[:, :, i], g[:, :, i], decay[:, :, i]
        attn_i = (q_i @ k_i.transpose(-1, -2)) * d_i  # causal inside the chunk (d_i is zero above the diagonal)
        v_new = u_i - w_i @ S  # delta-rule "error" values
        outs.append((q_i * g_i.exp()[..., None]) @ S + attn_i @ v_new)
        g_last = g_i[..., -1]  # [B,H]
        k_dec = k_i * (g_last[..., None] - g_i).exp()[..., None]
        S = S * g_last.exp()[..., None, None] + k_dec.transpose(-1, -2) @ v_new
    out = torch.stack(outs, dim=2).reshape(B, H, Tp, Dv)[:, :, :T]
    return out.transpose(1, 2), S  # [B,T,H,Dv], [B,H,Dk,Dv]


def _fla_chunk(q, k, v, g, beta, initial_state=None):  # optional fast path, needs `pip install flash-linear-attention`
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_fn

    dtype = torch.bfloat16 if q.is_cuda and torch.is_autocast_enabled() else q.dtype
    o, S = fla_fn(
        q.to(dtype), k.to(dtype), v.to(dtype), g=g.float(), beta=beta.to(dtype),
        initial_state=initial_state, output_final_state=True, use_qk_l2norm_in_kernel=True,
    )
    return o.float(), S


def gated_delta_rule(q, k, v, g, beta, impl="torch", chunk_size=64, initial_state=None):
    """Dispatch.  Returns (out [B,T,H,Dv] float32, final_state [B,H,Dk,Dv])."""
    if impl == "fla":
        return _fla_chunk(q, k, v, g, beta, initial_state)
    # Autocast would silently turn every fp32 `@` below into a bf16 matmul (the (I+L)^-1 solve and the
    # recurrent state are precision sensitive), so the core always runs with autocast disabled.
    with torch.autocast(device_type=q.device.type, enabled=False):
        if impl == "recurrent":
            return recurrent_gated_delta_rule(q, k, v, g, beta, initial_state)
        return chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=chunk_size, initial_state=initial_state)
