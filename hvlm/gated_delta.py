"""Gated delta rule (the linear-attention core of Gated DeltaNet) in pure PyTorch.

Recurrence, per head, state S in R^{Dk x Dv}:
    S_t = a_t * S_{t-1} + b_t * k_t (v_t - a_t * S_{t-1}^T k_t)^T        a_t = exp(g_t) in (0, 1]
    o_t = S_t^T q_t
q and k are L2-normalised, q is scaled by Dk^-0.5.

`chunk_gated_delta_rule` is the training-time algorithm (parallel inside chunks of C tokens, a short
sequential loop over the T/C chunks).  It is a line-by-line twin of the numpy implementation that
tools/check_algorithms_numpy.py verifies against the recurrence.  `recurrent_gated_delta_rule` is the
definition itself (slow; used for tests and, with a carried state, for token-by-token decoding).

Inputs are [B, T, H, D] (g and beta: [B, T, H]).  The pure-PyTorch path does all math in float32.

`impl` selects the kernel:
  "torch"     pure PyTorch, fp32, runs anywhere (CPU/CUDA) - the reference implementation
  "fla"       flash-linear-attention Triton kernels, CUDA only, needs `pip install flash-linear-attention`
              (see the note on the two failure modes below)
  "auto"      fla when it is installed and the inputs are on CUDA, otherwise torch.  If the fla kernel
              turns out not to support this shape at runtime, auto falls back to torch for the rest of the
              process (with a warning) instead of crashing.
  "fla"       honoured when the package and CUDA are there.  Otherwise `resolve_impl(strict=True)` (what
              the trainer does with --gdn_impl) raises, while the lenient default warns once and uses
              torch - so a checkpoint whose config says "fla" still loads for CPU inference.  A kernel
              failure *during* an explicit "fla" run always raises.
  "recurrent" the definition itself, token by token

With fla the operands are cast to the active autocast dtype (bf16/fp16), which is what makes the kernel
fast; the observed difference to the fp32 reference is ~1e-3 absolute / ~1e-2 relative, i.e. ordinary
bf16 rounding.  See README section 2 for the measured throughput.
"""
import warnings

import torch
import torch.nn.functional as F

_FLA_FN = None
_FLA_ERROR = None
_FLA_PROBED = False
_FLA_BROKEN = False  # set when an "auto" run hit a kernel-level failure and fell back to torch
_FLA_WARNED = False  # set once we have warned that "fla" is unusable on this device


def _fla_probe():
    """Import the fla kernel once.  Returns (callable or None, error string or None)."""
    global _FLA_FN, _FLA_ERROR, _FLA_PROBED
    if not _FLA_PROBED:
        _FLA_PROBED = True
        try:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fn

            _FLA_FN = fn
        except Exception as exc:  # pragma: no cover - depends on the environment
            _FLA_ERROR = f"{type(exc).__name__}: {exc}"
    return _FLA_FN, _FLA_ERROR


def fla_available():
    """True when the flash-linear-attention kernel can be imported in this environment."""
    return _fla_probe()[0] is not None


def fla_error():
    """Why fla is unavailable (None when it is available)."""
    return _fla_probe()[1]


def fla_version():
    try:
        import fla

        return getattr(fla, "__version__", "unknown")
    except Exception:  # pragma: no cover
        return None


def resolve_impl(impl="auto", device=None, strict=False):
    """Map an implementation name to something runnable on `device`.

    "auto" never fails - it just picks the best option.  A request for "fla" that cannot be honoured is an
    error when `strict` (the trainer validates its own --gdn_impl at startup that way) and a one-time
    warning plus a torch fallback otherwise (so a checkpoint saved by an fla run still loads on a machine
    without flash-linear-attention, e.g. CPU inference)."""
    if impl in ("torch", "recurrent"):
        return impl
    if impl not in ("auto", "fla"):
        raise ValueError(f"unknown gated-delta-rule impl {impl!r} (expected auto/torch/fla/recurrent)")
    cuda = torch.cuda.is_available() if device is None else torch.device(device).type == "cuda"
    if impl == "auto":
        return "fla" if (cuda and fla_available() and not _FLA_BROKEN) else "torch"
    fn, err = _fla_probe()
    if fn is None:
        problem = f"flash-linear-attention is not importable (pip install flash-linear-attention; got {err})"
    elif not cuda:
        problem = "flash-linear-attention needs CUDA tensors"
    else:
        return "fla"
    if strict:
        raise RuntimeError(f"linear_attn_impl='fla' cannot run here: {problem}")
    global _FLA_WARNED
    if not _FLA_WARNED:
        _FLA_WARNED = True
        warnings.warn(f"linear_attn_impl='fla' cannot run here ({problem}); using the pure-PyTorch gated "
                      "delta rule instead", RuntimeWarning, stacklevel=2)
    return "torch"


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


def _fla_dtype(x):  # dtype the kernel should run in, given an active autocast region
    if torch.is_autocast_enabled(x.device.type):
        d = torch.get_autocast_dtype(x.device.type)
        # fp32/other autocast dtypes: feed the kernel fp32 rather than guessing
        return d if d in (torch.bfloat16, torch.float16) else torch.float32
    return x.dtype if x.dtype in (torch.bfloat16, torch.float16) else torch.float32


def _fla_chunk(q, k, v, g, beta, initial_state=None):  # needs `pip install flash-linear-attention`
    fn, err = _fla_probe()
    if fn is None:
        raise RuntimeError(f"flash-linear-attention is not importable ({err})")
    dtype = _fla_dtype(q)
    o, S = fn(
        q.to(dtype), k.to(dtype), v.to(dtype), g=g.float(), beta=beta.to(dtype),
        initial_state=None if initial_state is None else initial_state.to(dtype),
        output_final_state=True, use_qk_l2norm_in_kernel=True,
    )
    return o.float(), S.float()  # fp32 result, like the pure-PyTorch path


def gated_delta_rule(q, k, v, g, beta, impl="auto", chunk_size=64, initial_state=None):
    """Dispatch.  Returns (out [B,T,H,Dv] float32, final_state [B,H,Dk,Dv])."""
    global _FLA_BROKEN
    requested, impl = impl, resolve_impl(impl, q.device)
    if impl == "fla":
        try:
            return _fla_chunk(q, k, v, g, beta, initial_state)
        except Exception as exc:
            if requested == "fla":  # explicit request: never silently degrade
                raise
            if not _FLA_BROKEN:  # warn once, then stay on the fallback for the rest of the process
                _FLA_BROKEN = True
                warnings.warn(
                    f"flash-linear-attention failed on this shape ({type(exc).__name__}: {exc}); "
                    "falling back to the pure-PyTorch gated delta rule for the rest of this process "
                    "(pass --gdn_impl fla to turn this into an error, or --gdn_impl torch to silence it)",
                    RuntimeWarning, stacklevel=2,
                )
            impl = "torch"
    # Autocast would silently turn every fp32 `@` below into a bf16 matmul (the (I+L)^-1 solve and the
    # recurrent state are precision sensitive), so the core always runs with autocast disabled.
    with torch.autocast(device_type=q.device.type, enabled=False):
        if impl == "recurrent":
            return recurrent_gated_delta_rule(q, k, v, g, beta, initial_state)
        return chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=chunk_size, initial_state=initial_state)
