#!/usr/bin/env python
"""Torch-side tests.  Run on your machine (CPU is enough for all but the GPU memory test):

    python tests/test_model.py            # everything
    python tests/test_model.py -k mtp     # only tests whose name contains 'mtp'

The model code was written without being able to execute PyTorch, so run this FIRST and fix anything it reports.
"""
import json
import os
import sys
import tempfile
import traceback
import warnings

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from hvlm.configuration_hvlm import HVLMConfig  # noqa: E402
from hvlm.gated_delta import (chunk_gated_delta_rule, fla_available,  # noqa: E402
                              gated_delta_rule, recurrent_gated_delta_rule, resolve_impl)
from hvlm.modeling_hvlm import HVLMForConditionalGeneration, chunked_cross_entropy  # noqa: E402
from hvlm.positions import build_mrope_positions  # noqa: E402

FULL = json.load(open(os.path.join(ROOT, "configs", "hvlm_130m.json")))


def tiny_config(**over):
    c = dict(FULL)
    c.update(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
             num_key_value_heads=2, head_dim=16, mrope_section=[2, 1, 1], linear_num_key_heads=2,
             linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16, linear_chunk_size=8,
             ce_chunk_size=16, vision_config=dict(FULL["vision_config"], hidden_size=32, intermediate_size=64,
                                                  depth=2, num_heads=2))
    c.update(over)
    return HVLMConfig(**c)


def tiny_model(**over):
    torch.manual_seed(0)
    return HVLMForConditionalGeneration(tiny_config(**over)).eval()


def rand_batch(cfg, B=2, T=40, with_image=False):
    g = torch.Generator().manual_seed(1)
    if with_image:  # the image prefix needs 4 + n tokens; callers may read 3 more
        T = max(T, 4 + cfg.vision_config.num_image_tokens + 3)
    ids = torch.randint(8, cfg.vocab_size, (B, T), generator=g)
    batch = dict(input_ids=ids, labels=ids.clone())
    if with_image:
        vc = cfg.vision_config
        n = vc.num_image_tokens
        ids[:, 2] = cfg.vision_start_token_id
        ids[:, 3:3 + n] = cfg.image_token_id
        ids[:, 3 + n] = cfg.vision_end_token_id
        batch["labels"] = ids.clone()
        batch["labels"][:, :4 + n] = -100
        batch["pixel_values"] = torch.randn(B, 3, vc.image_size, vc.image_size, generator=g)
        batch["position_ids"] = torch.from_numpy(np.stack(
            [build_mrope_positions(ids[b].numpy(), cfg.image_token_id, vc.merged_grid_size) for b in range(B)], axis=1))
    return batch


# ---------------------------------------------------------------------------------------------
def test_gdn_chunk_matches_recurrent():
    torch.manual_seed(0)
    for T, C in [(64, 16), (50, 16), (7, 64)]:
        B, H, Dk, Dv = 2, 3, 16, 24
        q, k, v = torch.randn(B, T, H, Dk), torch.randn(B, T, H, Dk), torch.randn(B, T, H, Dv)
        g = -torch.rand(B, T, H) * 2
        beta = torch.rand(B, T, H)
        a, Sa = chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=C)
        b, Sb = recurrent_gated_delta_rule(q, k, v, g, beta)
        assert torch.allclose(a, b, atol=2e-4, rtol=1e-3), (T, C, (a - b).abs().max().item())
        assert torch.allclose(Sa, Sb, atol=2e-4, rtol=1e-3)


def test_gdn_gradients_match_recurrent():
    torch.manual_seed(0)
    B, T, H, Dk, Dv = 1, 33, 2, 8, 8
    base = [torch.randn(B, T, H, Dk), torch.randn(B, T, H, Dk), torch.randn(B, T, H, Dv),
            -torch.rand(B, T, H), torch.rand(B, T, H)]
    grads = []
    for fn in (lambda *x: chunk_gated_delta_rule(*x, chunk_size=8), recurrent_gated_delta_rule):
        xs = [t.clone().requires_grad_(True) for t in base]
        out, _ = fn(*xs)
        (out ** 2).sum().backward()
        grads.append([x.grad for x in xs])
    for i, (ga, gb) in enumerate(zip(*grads)):
        assert torch.allclose(ga, gb, atol=1e-3, rtol=1e-2), (i, (ga - gb).abs().max().item())


def test_gdn_impl_dispatch():
    """`auto` must never ask for fla on CPU / without the package, and must not need CUDA to be importable."""
    assert resolve_impl("auto", "cpu") == "torch"
    assert resolve_impl("torch", "cuda") == "torch"
    assert resolve_impl("recurrent", "cpu") == "recurrent"
    if not (torch.cuda.is_available() and fla_available()):
        try:
            resolve_impl("fla", "cuda", strict=True)
            raise AssertionError("strict resolve_impl('fla') must refuse when fla/CUDA is unusable")
        except RuntimeError:
            pass
        # ... while the lenient path (what a checkpoint's saved config goes through) degrades to torch.
        # The warning is emitted once per process, so clear that latch to keep this test deterministic.
        import hvlm.gated_delta as gd

        gd._FLA_WARNED = False
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert resolve_impl("fla", "cpu") == "torch"
        assert any(issubclass(w.category, RuntimeWarning) for w in caught), "expected a RuntimeWarning"
    try:
        resolve_impl("nope", "cpu")
        raise AssertionError("unknown impl must raise")
    except ValueError:
        pass
    # "auto" on CPU silently lands on the fp32 reference path
    torch.manual_seed(0)
    B, T, H, Dk, Dv = 2, 40, 3, 16, 16
    q, k, v = torch.randn(B, T, H, Dk), torch.randn(B, T, H, Dk), torch.randn(B, T, H, Dv)
    g, beta = -torch.rand(B, T, H) * 2, torch.rand(B, T, H)
    a, _ = gated_delta_rule(q, k, v, g, beta, impl="auto", chunk_size=8)
    b, _ = gated_delta_rule(q, k, v, g, beta, impl="torch", chunk_size=8)
    assert torch.equal(a, b)


def test_fla_matches_torch_impl():
    """The optional flash-linear-attention kernel must agree with the fp32 reference within bf16 rounding."""
    if not torch.cuda.is_available():
        print("    (skipped: no CUDA device)")
        return
    if not fla_available():
        print("    (skipped: fla not installed)")
        return
    torch.manual_seed(0)
    for T, H, Dk, Dv in [(512, 8, 64, 64), (100, 4, 64, 64)]:  # incl. a length that needs chunk padding
        q, k = torch.randn(2, T, H, Dk, device="cuda"), torch.randn(2, T, H, Dk, device="cuda")
        v = torch.randn(2, T, H, Dv, device="cuda")
        g, beta = -torch.rand(2, T, H, device="cuda") * 2, torch.rand(2, T, H, device="cuda")
        with torch.no_grad():
            ref, s_ref = gated_delta_rule(q, k, v, g, beta, impl="torch", chunk_size=64)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out, s_out = gated_delta_rule(q, k, v, g, beta, impl="fla")
        assert out.dtype == torch.float32 and s_out.dtype == torch.float32
        err = (ref - out).abs().max().item()
        serr = ((s_ref - s_out).abs().max() / s_ref.abs().max().clamp_min(1e-3)).item()
        assert err < 5e-3, (T, H, Dk, Dv, err)
        assert serr < 2e-2, (T, H, Dk, Dv, serr)  # the carried state accumulates bf16 rounding over T steps
        print(f"    fla vs torch (T={T}): out max|diff| {err:.2e}, state rel {serr:.2e}")
        # gradients must flow through the kernel and stay finite
        xs = [t.clone().requires_grad_(True) for t in (q[:1], k[:1], v[:1], g[:1], beta[:1])]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o, _ = gated_delta_rule(*xs, impl="fla")
        o.float().pow(2).sum().backward()
        assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in xs)


def test_param_count_matches_estimator():
    from estimate import count_params

    m = HVLMForConditionalGeneration(HVLMConfig(**FULL))
    total = sum(p.numel() for p in m.parameters())
    est = count_params(FULL)["total_training"]
    assert abs(total - est) / est < 0.002, f"model {total} vs estimate {est}"
    print(f"    params: {total / 1e6:.2f}M (estimate {est / 1e6:.2f}M); "
          f"without MTP {(total - sum(p.numel() for p in m.mtp.parameters())) / 1e6:.2f}M")


def test_layer_pattern_is_3_to_1():
    cfg = HVLMConfig(**FULL)
    assert cfg.layer_types.count("full_attention") == 5 and cfg.layer_types.count("linear_attention") == 15
    assert cfg.layer_types[3] == "full_attention" and cfg.layer_types[-1] == "full_attention"


def test_forward_loss_and_logits():
    m = tiny_model()
    b = rand_batch(m.config)
    out = m(**b)
    assert out.loss.ndim == 0 and torch.isfinite(out.loss)
    assert abs(out.lm_loss.item() - np.log(m.config.vocab_size)) < 1.0, "untrained loss should be ~ln(V)"
    assert out.mtp_loss is not None and abs(out.mtp_loss.item() - np.log(m.config.vocab_size)) < 1.0
    lg = m(input_ids=b["input_ids"]).logits
    assert lg.shape == (2, 40, m.config.vocab_size)
    assert m(input_ids=b["input_ids"], logits_to_keep=1).logits.shape == (2, 1, m.config.vocab_size)


def test_chunked_ce_equals_full_ce():
    torch.manual_seed(0)
    h = torch.randn(50, 16, requires_grad=True)
    W = torch.randn(40, 16, requires_grad=True)
    y = torch.randint(0, 40, (50,))
    y[::7] = -100
    s, n = chunked_cross_entropy(h, W, y, chunk_size=12)
    ref = torch.nn.functional.cross_entropy(h @ W.T, y, ignore_index=-100, reduction="sum")
    assert abs(s.item() - ref.item()) < 1e-3 and n == int((y != -100).sum())
    g1 = torch.autograd.grad(s, [h, W])
    g2 = torch.autograd.grad(ref, [h, W])
    assert all(torch.allclose(a, b, atol=1e-4) for a, b in zip(g1, g2))


def test_causality_text_and_image():
    m = tiny_model()
    for with_image in (False, True):
        b = rand_batch(m.config, with_image=with_image)
        cut = 30
        ids2 = b["input_ids"].clone()
        ids2[:, cut:] = (ids2[:, cut:] + 7) % m.config.vocab_size
        ids2[b["input_ids"] == m.config.image_token_id] = m.config.image_token_id
        kw = {k: v for k, v in b.items() if k not in ("input_ids", "labels")}
        with torch.no_grad():
            a = m(input_ids=b["input_ids"], **kw).logits
            c = m(input_ids=ids2, **kw).logits
        assert torch.allclose(a[:, :cut], c[:, :cut], atol=1e-4), f"future tokens leaked (image={with_image})"
        assert not torch.allclose(a[:, cut:], c[:, cut:], atol=1e-4)


def test_right_padding_is_inert():
    m = tiny_model()
    ids = rand_batch(m.config, B=1, T=24)["input_ids"]
    padded = torch.cat([ids, torch.full((1, 8), 6)], dim=1)
    with torch.no_grad():
        a = m(input_ids=ids).logits
        b = m(input_ids=padded).logits[:, :24]
    assert torch.allclose(a, b, atol=1e-4)


def test_mtp_label_alignment():
    """MTP head k=1 must be trained to predict token t+2 from (hidden[t], embed[t+1]).  A sequence where a
    single position carries the only non-ignored label makes the alignment observable."""
    m = tiny_model()
    cfg = m.config
    ids = rand_batch(cfg, B=1, T=20)["input_ids"]
    for target_pos in (5, 11):
        lab = torch.full_like(ids, -100)
        lab[0, target_pos] = ids[0, target_pos]
        out = m(input_ids=ids, labels=lab)
        # recompute the MTP loss by hand: position t = target_pos-2 predicts token at target_pos
        with torch.no_grad():
            emb = m.embed_tokens(ids)
            pos = torch.arange(20)[None, None].expand(3, 1, 20)
            cos, sin = m.rotary(pos)
            h = emb
            for layer in m.layers:
                h = layer(h, cos, sin)
            mod = m.mtp[0]
            prev = mod(h[:, :19], emb[:, 1:], cos[:, :19], sin[:, :19])
            logits = prev @ m.embed_tokens.weight.T
            manual = torch.nn.functional.cross_entropy(logits[0, target_pos - 2][None], ids[0, target_pos][None])
        assert abs(out.mtp_loss.item() - manual.item()) < 1e-3, (target_pos, out.mtp_loss.item(), manual.item())


def test_mtp_disabled_paths():
    m = tiny_model()
    b = rand_batch(m.config)
    o = m(**b, compute_mtp=False)
    assert o.mtp_loss is None and torch.allclose(o.loss, o.lm_loss)
    m2 = tiny_model(mtp_num_layers=0)
    assert m2(**b).mtp_loss is None
    m3 = tiny_model(mtp_loss_weight=0.0)
    assert m3(**b).mtp_loss is None


def test_backward_all_parameters_get_gradient():
    m = tiny_model().train()
    out = m(**rand_batch(m.config, with_image=True))
    out.loss.backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, f"no gradient for: {missing}"
    bad = [n for n, p in m.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f"non-finite gradients: {bad}"


def test_bf16_autocast_runs_and_is_close():
    m = tiny_model().train()
    b = rand_batch(m.config, with_image=True)
    ref = m(**b)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = m(**b)
        out.loss.backward()
    assert torch.isfinite(out.loss) and abs(out.loss.item() - ref.loss.item()) < 0.1


def test_align_stage_trains_only_vision():
    m = tiny_model().train()
    for n, p in m.named_parameters():
        p.requires_grad = n.startswith("visual.")
    m(**rand_batch(m.config, with_image=True)).loss.backward()
    assert all(p.grad is not None for n, p in m.named_parameters() if n.startswith("visual.")), "ViT got no gradient"
    assert all(p.grad is None for n, p in m.named_parameters() if not n.startswith("visual."))


def test_image_changes_output_and_slot_mismatch_raises():
    m = tiny_model()
    b = rand_batch(m.config, with_image=True)
    kw = dict(input_ids=b["input_ids"], position_ids=b["position_ids"])
    with torch.no_grad():
        a = m(**kw, pixel_values=b["pixel_values"]).logits
        c = m(**kw, pixel_values=b["pixel_values"] + 1.0).logits
    assert not torch.allclose(a, c)
    try:
        m(input_ids=b["input_ids"], pixel_values=b["pixel_values"][:1])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_save_load_roundtrip():
    m = tiny_model()
    b = rand_batch(m.config, with_image=True)
    with torch.no_grad():
        ref = m(input_ids=b["input_ids"], pixel_values=b["pixel_values"], position_ids=b["position_ids"]).logits
    with tempfile.TemporaryDirectory() as d:
        m.save_pretrained(d)
        m2 = HVLMForConditionalGeneration.from_pretrained(d).float().eval()
        assert m2.config.vision_config.depth == m.config.vision_config.depth
        assert m2.config.layer_types == m.config.layer_types
        with torch.no_grad():
            out = m2(input_ids=b["input_ids"], pixel_values=b["pixel_values"], position_ids=b["position_ids"]).logits
    assert torch.allclose(ref, out, atol=1e-5)


def test_tied_embedding_single_tensor():
    m = tiny_model()
    names = [n for n, _ in m.named_parameters()]
    assert not any("lm_head" in n for n in names)
    assert sum(p.numel() for n, p in m.named_parameters() if "embed_tokens" in n) == m.config.vocab_size * m.config.hidden_size


def test_init_scales_and_decay_init():
    m = HVLMForConditionalGeneration(HVLMConfig(**FULL))
    gdn = next(l.linear_attn for l in m.layers if l.layer_type == "linear_attention")
    A = gdn.A_log.exp()
    assert A.min() >= 1e-3 * 0.999 and A.max() <= 16.0 * 1.001, "A_log init range"
    assert (gdn.dt_bias == 1).all()
    o = next(l.mlp.down_proj for l in m.layers)
    assert o.weight.std().item() < 0.02, "residual output projections must use the smaller init"
    assert abs(m.embed_tokens.weight.std().item() - 0.02) < 0.002
    assert (m.layers[0].input_layernorm.weight == 0).all()


def test_generate_simple_runs():
    m = tiny_model()
    b = rand_batch(m.config, with_image=True, B=1)
    out = m.generate_simple(b["input_ids"][:, :4 + m.config.vision_config.num_image_tokens + 3],
                            pixel_values=b["pixel_values"], max_new_tokens=3, temperature=0)
    assert out.shape[1] == 4 + m.config.vision_config.num_image_tokens + 3 + 3


def test_tiny_overfit_learns():
    """A few dozen steps on one batch must drive the loss well below ln(V): end-to-end trainability."""
    m = tiny_model().train()
    b = rand_batch(m.config, B=2, T=32)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.0)
    first = None
    for i in range(60):
        loss = m(**b).loss
        first = first or loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
    assert loss.item() < 0.35 * first, (first, loss.item())
    print(f"    overfit: {first:.2f} -> {loss.item():.2f}")


def test_chat_dataset_reads_source_specs():
    """--mm_data takes source specs, not just local jsonl: a Hub-shaped record is spooled and encoded."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from data import ChatDataset  # noqa: E402

    class Tok:  # character ids are enough: this test is about the record plumbing, not the tokenizer
        def __call__(self, s, add_special_tokens=False):
            return {"input_ids": [10 + (ord(c) % 50) for c in s]}

        def convert_tokens_to_ids(self, t):
            return 99

    vc = tiny_config().vision_config
    sp = dict(im_start=1, im_end=2, vision_start=3, vision_end=4, image_pad=5)
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "lccc.jsonl")
        with open(src, "w", encoding="utf-8") as f:  # LCCC's on-Hub shape: one "dialog" list per line
            f.write(json.dumps({"dialog": ["在吗", "在的咋啦"]}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"dialog": ["吃了吗", "刚吃完"]}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"dialog": ["你好"]}, ensure_ascii=False) + "\n")  # one turn only: unusable
        ds = ChatDataset([f"jsonl:{src}:-"], ".", Tok(), 128, vc, sp, cache_dir=os.path.join(td, "cache"))
        assert len(ds) == 2, len(ds)  # the single-turn record is dropped, the two usable ones survive
        ids, labels, img = ds[0]
        assert img is None and len(ids) == len(labels) and any(l != -100 for l in labels)


def test_eval_prompt_matches_training_encoding():
    """The eval/sample prompt is the training-time prefix of the same conversation: same ChatML layout and
    the image block in the same slot.  A drift here would score generations from a distribution the model
    was never trained on, and every eval number would be silently meaningless."""
    from hvlm.data_core import encode_conversation, encode_prompt  # noqa: E402

    enc = lambda s: [10 + (ord(c) % 50) for c in s]  # noqa: E731
    sp = dict(im_start=1, im_end=2, vision_start=3, vision_end=4, image_pad=5)
    msgs = [{"role": "user", "content": "在吗"}, {"role": "self", "content": "在的咋啦"},
            {"role": "user", "content": "吃了吗"}]
    prompt = encode_prompt(enc, sp, msgs, 4, False)
    ids, labels, used = encode_conversation(enc, sp, msgs + [{"role": "self", "content": "刚吃完"}], 4, False, 4096)
    assert not used and ids[:len(prompt)] == prompt, "the generation prompt drifted from the training encoding"
    head = [sp["im_start"]] + enc("self\n")  # the opening turn the model continues from
    assert prompt[-len(head):] == head and labels[len(prompt) - len(head):len(prompt)] == [-100] * len(head), \
        "the turn the model generates from must carry no loss"

    img = [{"role": "user", "content": "<image>\n这是什么"}]
    prompt_i = encode_prompt(enc, sp, img, 4, True)
    ids_i, _labels, used_i = encode_conversation(enc, sp, img + [{"role": "self", "content": "喵"}], 4, True, 4096)
    assert used_i and ids_i[:len(prompt_i)] == prompt_i and prompt_i.count(sp["image_pad"]) == 4


def test_eval_generation_is_reproducible():
    """Same seed, same sample: report.json's numbers have to be reproducible from the recorded seed."""
    m = tiny_model().eval()
    ids = rand_batch(m.config, B=1, T=16)["input_ids"][0].tolist()

    def sample():
        torch.manual_seed(7)
        return m.generate_simple(torch.tensor([ids]), max_new_tokens=6, temperature=0.7, top_k=40)

    assert torch.equal(sample(), sample())

    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from eval import sample_seed  # noqa: E402

    assert sample_seed(1234, "imform", "im01", 0) == sample_seed(1234, "imform", "im01", 0)
    assert sample_seed(1234, "imform", "im01", 1) != sample_seed(1234, "imform", "im01", 0), \
        "the per-sample seed must vary, or K samples would be one sample repeated"


def test_gpu_memory_under_8gb():
    """No activation checkpointing: these are the shapes the trainer can afford on an 8 GB card."""
    if not torch.cuda.is_available():
        print("    (skipped: no CUDA device)")
        return

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    m = HVLMForConditionalGeneration(HVLMConfig(**FULL)).cuda().train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True)

    def one_step(mb, seq):  # one fwd/bwd/step; lazy cuBLAS handles and the Triton JIT warm up below
        ids = torch.randint(8, 32768, (mb, seq), device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = m(input_ids=ids, labels=ids).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    one_step(1, 1024)  # warm-up, not measured: this is what pays for lazy init / kernel compilation
    for mb, seq in ((2, 1024), (1, 2048)):
        torch.cuda.reset_peak_memory_stats()
        one_step(mb, seq)
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"    micro_batch {mb} x {seq}: peak allocated {peak:.2f} GB")
        assert peak < 7.0, f"peak {peak:.2f} GB leaves too little headroom for the CUDA context under 8 GB"


# ---------------------------------------------------------------------------------------------
def main():
    flt = sys.argv[sys.argv.index("-k") + 1] if "-k" in sys.argv else ""
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and flt in n]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception:
            failed += 1
            print(f"[FAIL] {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
