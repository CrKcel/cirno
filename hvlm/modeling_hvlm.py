"""Hybrid vision-language model (~130M): Gated DeltaNet / Gated Attention (3:1), dense SwiGLU MLP,
ViT tower with 2x2 merge projector, and multi-token-prediction (MTP) blocks.

Conventions
-----------
* `labels` are the *unshifted* token ids (-100 = ignore).  Shifting happens inside the model.
* Padding must be on the RIGHT.  Both token mixers are causal, so trailing pad tokens cannot influence
  earlier positions and no attention mask is needed (`attention_mask` is accepted and ignored).
* The residual stream stays in fp32 under bf16 autocast; matmuls run in bf16.
* The LM head is tied to the embedding (no separate `lm_head` module); logits are only materialised when
  `labels is None`.  With labels the loss is computed in checkpointed chunks (see `chunked_cross_entropy`).
"""
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_hvlm import HVLMConfig, HVLMVisionConfig
from .gated_delta import gated_delta_rule
from .positions import build_mrope_positions, mrope_axis_map


# --------------------------------------------------------------------------------------------------
# norms, MLP, rotary
# --------------------------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Zero-centred RMSNorm (Qwen3-Next / Qwen3.5): y = x_hat * (1 + w), w initialised to 0."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * (1.0 + self.weight.float())).to(dtype)


class GatedRMSNorm(nn.Module):
    """norm(x) * w * silu(gate)  - the output norm of Gated DeltaNet."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x, gate):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = self.weight.float() * x
        return (x * F.silu(gate.float())).to(dtype)


class MLP(nn.Module):
    """Dense SwiGLU MLP."""

    def __init__(self, config):
        super().__init__()
        d, f = config.hidden_size, config.intermediate_size
        self.gate_proj = nn.Linear(d, f, bias=False)
        self.up_proj = nn.Linear(d, f, bias=False)
        self.down_proj = nn.Linear(f, d, bias=False)
        self.down_proj._residual_out = True

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class RotaryEmbedding(nn.Module):
    """Partial-rotary interleaved M-RoPE.  position_ids: [3, B, T] (temporal, height, width).
    No buffers are registered (frequencies are recomputed per call) so meta-device init is harmless."""

    def __init__(self, config):
        super().__init__()
        self.rot_dim = config.rotary_dim
        self.theta = config.rope_theta
        self._axis_map = mrope_axis_map(config.mrope_section)

    @torch.no_grad()
    def forward(self, position_ids):
        dev = position_ids.device
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.rot_dim, 2, device=dev, dtype=torch.float32) / self.rot_dim))
        axis = torch.as_tensor(self._axis_map, device=dev)
        pos = position_ids[axis]  # [F, B, T] - each frequency reads its own axis
        freqs = pos.permute(1, 2, 0).float() * inv_freq  # [B, T, F]
        emb = torch.cat([freqs, freqs], dim=-1)  # [B, T, rot_dim]
        return emb.cos(), emb.sin()


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def apply_rope(x, cos, sin):
    """x: [B, T, H, D]; cos/sin: [B, T, rot].  Only the first `rot` channels are rotated."""
    rot = cos.shape[-1]
    cos, sin = cos[:, :, None, :], sin[:, :, None, :]
    x_rot, x_pass = x[..., :rot].float(), x[..., rot:]
    x_rot = (x_rot * cos + rotate_half(x_rot) * sin).to(x.dtype)
    return torch.cat([x_rot, x_pass], dim=-1)


# --------------------------------------------------------------------------------------------------
# token mixers
# --------------------------------------------------------------------------------------------------
class GatedAttention(nn.Module):
    """Full causal attention with GQA, QK-norm, partial M-RoPE and a sigmoid output gate (Qwen3-Next style)."""

    def __init__(self, config):
        super().__init__()
        d = config.hidden_size
        self.nh, self.nkv, self.hd = config.num_attention_heads, config.num_key_value_heads, config.head_dim
        self.q_proj = nn.Linear(d, self.nh * self.hd * 2, bias=False)  # query and gate
        self.k_proj = nn.Linear(d, self.nkv * self.hd, bias=False)
        self.v_proj = nn.Linear(d, self.nkv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.nh * self.hd, d, bias=False)
        self.o_proj._residual_out = True
        self.q_norm = RMSNorm(self.hd, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.hd, config.rms_norm_eps)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q, gate = self.q_proj(x).view(B, T, self.nh, 2 * self.hd).chunk(2, dim=-1)
        k = self.k_proj(x).view(B, T, self.nkv, self.hd)
        v = self.v_proj(x).view(B, T, self.nkv, self.hd)
        q = apply_rope(self.q_norm(q), cos, sin).transpose(1, 2)  # [B, nh, T, hd]
        k = apply_rope(self.k_norm(k), cos, sin).transpose(1, 2)  # [B, nkv, T, hd]
        v = v.transpose(1, 2)
        rep = self.nh // self.nkv
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # [B, nh, T, hd]
        o = o.transpose(1, 2).reshape(B, T, self.nh * self.hd)
        o = o * torch.sigmoid(gate.reshape(B, T, self.nh * self.hd))
        return self.o_proj(o)


class GatedDeltaNet(nn.Module):
    """Gated DeltaNet token mixer: short causal conv on q/k/v, gated delta rule core, gated RMSNorm output."""

    def __init__(self, config):
        super().__init__()
        d = config.hidden_size
        self.Hk, self.Hv = config.linear_num_key_heads, config.linear_num_value_heads
        self.Dk, self.Dv = config.linear_key_head_dim, config.linear_value_head_dim
        self.key_dim, self.val_dim = self.Hk * self.Dk, self.Hv * self.Dv
        self.conv_dim = 2 * self.key_dim + self.val_dim
        self.kernel = config.linear_conv_kernel_dim
        self.impl = config.linear_attn_impl
        self.chunk_size = config.linear_chunk_size

        self.in_proj_qkvz = nn.Linear(d, 2 * self.key_dim + 2 * self.val_dim, bias=False)
        self.in_proj_ba = nn.Linear(d, 2 * self.Hv, bias=False)
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, self.kernel, groups=self.conv_dim,
                                padding=self.kernel - 1, bias=False)
        self.A_log = nn.Parameter(torch.zeros(self.Hv))  # initialised in HVLM._init_weights
        self.dt_bias = nn.Parameter(torch.ones(self.Hv))
        self.norm = GatedRMSNorm(self.Dv, config.rms_norm_eps)
        self.out_proj = nn.Linear(self.val_dim, d, bias=False)
        self.out_proj._residual_out = True

    def forward(self, x, cos=None, sin=None):  # cos/sin unused: this layer needs no positional encoding
        B, T, _ = x.shape
        q, k, v, z = torch.split(self.in_proj_qkvz(x), [self.key_dim, self.key_dim, self.val_dim, self.val_dim], dim=-1)
        b, a = self.in_proj_ba(x).chunk(2, dim=-1)  # [B, T, Hv] each

        mixed = torch.cat([q, k, v], dim=-1).transpose(1, 2)  # [B, C, T]
        mixed = F.silu(self.conv1d(mixed)[..., :T]).transpose(1, 2)  # causal depthwise conv, [B, T, C]
        q, k, v = torch.split(mixed, [self.key_dim, self.key_dim, self.val_dim], dim=-1)
        q = q.reshape(B, T, self.Hk, self.Dk)
        k = k.reshape(B, T, self.Hk, self.Dk)
        v = v.reshape(B, T, self.Hv, self.Dv)
        if self.Hv != self.Hk:
            rep = self.Hv // self.Hk
            q = q.repeat_interleave(rep, dim=2)
            k = k.repeat_interleave(rep, dim=2)

        beta = torch.sigmoid(b.float())  # write strength in (0, 1)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())  # log-decay <= 0
        o, _ = gated_delta_rule(q, k, v, g, beta, impl=self.impl, chunk_size=self.chunk_size)  # fp32 [B,T,Hv,Dv]
        o = self.norm(o, z.reshape(B, T, self.Hv, self.Dv)).to(x.dtype)
        return self.out_proj(o.reshape(B, T, self.val_dim))


class DecoderLayer(nn.Module):
    def __init__(self, config, layer_type):
        super().__init__()
        self.layer_type = layer_type
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        if layer_type == "full_attention":
            self.self_attn = GatedAttention(config)
        elif layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(config)
        else:
            raise ValueError(f"unknown layer type {layer_type}")
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x, cos, sin):
        h = self.input_layernorm(x)
        mixer = self.self_attn if self.layer_type == "full_attention" else self.linear_attn
        x = x + mixer(h, cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


# --------------------------------------------------------------------------------------------------
# multi-token prediction
# --------------------------------------------------------------------------------------------------
class MTPModule(nn.Module):
    """DeepSeek-V3 / Qwen3.5 style MTP block: fuse hidden state at t with the embedding of token t+k,
    run one full-attention decoder layer, and predict token t+k+1 through the shared (tied) head."""

    def __init__(self, config):
        super().__init__()
        d = config.hidden_size
        self.pre_fc_norm_hidden = RMSNorm(d, config.rms_norm_eps)
        self.pre_fc_norm_embed = RMSNorm(d, config.rms_norm_eps)
        self.fc = nn.Linear(2 * d, d, bias=False)
        self.layer = DecoderLayer(config, "full_attention")
        self.norm = RMSNorm(d, config.rms_norm_eps)

    def forward(self, hidden, embeds, cos, sin):
        x = torch.cat([self.pre_fc_norm_hidden(hidden), self.pre_fc_norm_embed(embeds)], dim=-1)
        x = self.fc(x).to(hidden.dtype)
        return self.norm(self.layer(x, cos, sin))


# --------------------------------------------------------------------------------------------------
# vision tower
# --------------------------------------------------------------------------------------------------
class VisionAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.nh = c.num_heads
        self.hd = c.hidden_size // c.num_heads
        self.qkv = nn.Linear(c.hidden_size, 3 * c.hidden_size)
        self.proj = nn.Linear(c.hidden_size, c.hidden_size)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.nh, self.hd).permute(2, 0, 3, 1, 4).unbind(0)
        o = F.scaled_dot_product_attention(q, k, v)  # bidirectional
        return self.proj(o.transpose(1, 2).reshape(B, N, C))


class VisionBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm1 = nn.LayerNorm(c.hidden_size, eps=c.layer_norm_eps)
        self.attn = VisionAttention(c)
        self.norm2 = nn.LayerNorm(c.hidden_size, eps=c.layer_norm_eps)
        self.fc1 = nn.Linear(c.hidden_size, c.intermediate_size)
        self.fc2 = nn.Linear(c.intermediate_size, c.hidden_size)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x)), approximate="tanh"))


class VisionTower(nn.Module):
    """Fixed-resolution ViT -> LayerNorm -> 2x2 merge -> MLP into the LLM width.
    pixel_values [N, 3, S, S] -> [N, (grid/merge)^2, llm_hidden], tokens in row-major order."""

    def __init__(self, c: HVLMVisionConfig, llm_hidden):
        super().__init__()
        self.c = c
        g, dv, m = c.grid_size, c.hidden_size, c.spatial_merge_size
        self.patch_embed = nn.Conv2d(c.in_channels, dv, c.patch_size, stride=c.patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, g * g, dv))
        self.blocks = nn.ModuleList([VisionBlock(c) for _ in range(c.depth)])
        self.norm = nn.LayerNorm(dv, eps=c.layer_norm_eps)
        self.merge_fc1 = nn.Linear(dv * m * m, dv * m * m)
        self.merge_fc2 = nn.Linear(dv * m * m, llm_hidden)

    def forward(self, pixel_values):
        c = self.c
        g, m = c.grid_size, c.spatial_merge_size
        x = self.patch_embed(pixel_values.to(self.patch_embed.weight.dtype))  # [N, dv, g, g]
        x = x.flatten(2).transpose(1, 2) + self.pos_embed  # [N, g*g, dv], row-major
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        N, _, C = x.shape
        x = x.reshape(N, g // m, m, g // m, m, C).permute(0, 1, 3, 2, 4, 5).reshape(N, (g // m) ** 2, m * m * C)
        return self.merge_fc2(F.gelu(self.merge_fc1(x), approximate="tanh"))


# --------------------------------------------------------------------------------------------------
# loss
# --------------------------------------------------------------------------------------------------
def _ce_chunk_sum(h, weight, y):
    return F.cross_entropy(F.linear(h, weight), y, reduction="sum")


def chunked_cross_entropy(hidden, weight, labels, chunk_size=2048, use_checkpoint=True):
    """Cross-entropy against the tied head without ever holding the full [N, V] logits.
    hidden [N, d], weight [V, d], labels [N] (-100 ignored).  Returns (sum_of_losses, n_valid_tokens).
    With use_checkpoint each chunk is recomputed in backward (peak logits memory = chunk_size * V) at the
    cost of redoing the head matmul; without it every chunk's logits stay alive until their backward."""
    valid = labels != -100
    hidden, labels = hidden[valid], labels[valid]
    n = int(labels.numel())
    if n == 0:
        return (hidden.sum() + weight.sum()) * 0.0, 0
    total = None
    needs_grad = torch.is_grad_enabled() and (hidden.requires_grad or weight.requires_grad)
    for i in range(0, n, chunk_size):
        h, y = hidden[i:i + chunk_size], labels[i:i + chunk_size]
        if use_checkpoint and needs_grad:
            part = checkpoint(_ce_chunk_sum, h, weight, y, use_reentrant=False, preserve_rng_state=False)
        else:
            part = _ce_chunk_sum(h, weight, y)
        total = part if total is None else total + part
    return total, n


@dataclass
class HVLMOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    lm_loss: Optional[torch.Tensor] = None
    mtp_loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


# --------------------------------------------------------------------------------------------------
# full model
# --------------------------------------------------------------------------------------------------
class HVLMForConditionalGeneration(PreTrainedModel):
    config_class = HVLMConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = True
    _no_split_modules = ["DecoderLayer"]

    def __init__(self, config: HVLMConfig):
        super().__init__(config)
        d = config.hidden_size
        self.embed_tokens = nn.Embedding(config.vocab_size, d)
        self.rotary = RotaryEmbedding(config)
        self.layers = nn.ModuleList([DecoderLayer(config, t) for t in config.layer_types])
        self.norm = RMSNorm(d, config.rms_norm_eps)
        self.visual = VisionTower(config.vision_config, d)
        self.mtp = nn.ModuleList([MTPModule(config) for _ in range(config.mtp_num_layers)])
        self.use_checkpoint = False        # per-layer activation checkpointing (--grad_ckpt)
        self.ce_checkpoint = getattr(config, "ce_checkpoint", True)  # recompute the head matmul in the CE
        self.post_init()

    # --- HF plumbing -------------------------------------------------------------------------
    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):  # head is tied; there is no separate module
        return None

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None, **kwargs):
        self.use_checkpoint = True

    def gradient_checkpointing_disable(self):
        self.use_checkpoint = False

    @torch.no_grad()
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            s = std / math.sqrt(2 * self.config.num_hidden_layers) if getattr(module, "_residual_out", False) else std
            nn.init.normal_(module.weight, mean=0.0, std=s)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Conv1d):
            bound = module.kernel_size[0] ** -0.5
            nn.init.uniform_(module.weight, -bound, bound)
        elif isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.zeros_(module.weight)
        elif isinstance(module, GatedRMSNorm):
            nn.init.ones_(module.weight)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, GatedDeltaNet):
            # decay rate A is log-uniform in [1e-3, 16] -> heads cover timescales from ~1 to ~1000 tokens
            nn.init.uniform_(module.A_log, math.log(1e-3), math.log(16.0))
            nn.init.ones_(module.dt_bias)
        elif isinstance(module, VisionTower):
            nn.init.normal_(module.pos_embed, mean=0.0, std=std)

    # --- forward -----------------------------------------------------------------------------
    def _embed(self, input_ids, pixel_values):
        embeds = self.embed_tokens(input_ids)
        if pixel_values is None:
            return embeds
        img = self.visual(pixel_values)  # [N, n_tok, d]
        mask = input_ids == self.config.image_token_id
        n_slots = int(mask.sum())
        if n_slots != img.shape[0] * img.shape[1]:
            raise ValueError(
                f"{n_slots} <|image_pad|> slots in input_ids but the vision tower produced "
                f"{img.shape[0]} x {img.shape[1]} = {img.shape[0] * img.shape[1]} image tokens"
            )
        mask = mask.unsqueeze(-1).expand_as(embeds)
        return embeds.masked_scatter(mask, img.reshape(-1, img.shape[-1]).to(embeds.dtype))

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,  # ignored: right-padding + causal mixers need no mask
        compute_mtp: bool = True,
        logits_to_keep: int = 0,
    ) -> HVLMOutput:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("provide input_ids or inputs_embeds")
            inputs_embeds = self._embed(input_ids, pixel_values)
        elif pixel_values is not None:
            raise ValueError("pixel_values requires input_ids (to locate the image slots)")
        B, T, d = inputs_embeds.shape

        if position_ids is None:
            position_ids = torch.arange(T, device=inputs_embeds.device)[None, None, :].expand(3, B, T)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None].expand(3, -1, -1)
        cos, sin = self.rotary(position_ids)

        h = inputs_embeds
        ckpt = self.use_checkpoint and self.training and torch.is_grad_enabled()
        for layer in self.layers:
            h = checkpoint(layer, h, cos, sin, use_reentrant=False, preserve_rng_state=False) if ckpt else layer(h, cos, sin)
        hn = self.norm(h)
        W = self.embed_tokens.weight

        if labels is None:
            if logits_to_keep > 0:
                hn = hn[:, -logits_to_keep:]
            return HVLMOutput(logits=F.linear(hn, W))

        chunk, ce_ckpt = self.config.ce_chunk_size, self.ce_checkpoint
        s, n = chunked_cross_entropy(hn[:, :-1].reshape(-1, d), W, labels[:, 1:].reshape(-1), chunk, ce_ckpt)
        lm_loss = s / max(n, 1)
        loss, mtp_loss = lm_loss, None

        if compute_mtp and len(self.mtp) > 0 and self.config.mtp_loss_weight > 0:
            prev, losses = h, []
            for k, mod in enumerate(self.mtp, start=1):
                if T - k - 1 < 1:
                    break
                # position t (0 .. T-k-1) fuses prev[t] with the embedding of token t+k, predicts token t+k+1
                args = (prev[:, :T - k], inputs_embeds[:, k:], cos[:, :T - k], sin[:, :T - k])
                prev = checkpoint(mod, *args, use_reentrant=False, preserve_rng_state=False) if ckpt else mod(*args)
                ms, mn = chunked_cross_entropy(prev[:, :-1].reshape(-1, d), W,
                                               labels[:, k + 1:].reshape(-1), chunk, ce_ckpt)
                losses.append(ms / max(mn, 1))
            if losses:
                mtp_loss = torch.stack(losses).mean()
                loss = lm_loss + self.config.mtp_loss_weight * mtp_loss
        return HVLMOutput(loss=loss, lm_loss=lm_loss.detach(), mtp_loss=None if mtp_loss is None else mtp_loss.detach())

    # --- simple sampling (no KV / state cache: recomputes the prefix every step; for smoke tests) -----
    @torch.no_grad()
    def generate_simple(self, input_ids, pixel_values=None, max_new_tokens=64, temperature=0.8, top_k=40,
                        eos_token_id=None):
        assert input_ids.shape[0] == 1, "generate_simple supports batch size 1"
        vc = self.config.vision_config
        ids = input_ids
        for _ in range(max_new_tokens):
            pos = None
            if pixel_values is not None:
                p = build_mrope_positions(ids[0].cpu().numpy(), self.config.image_token_id, vc.merged_grid_size)
                pos = torch.from_numpy(p)[:, None, :].to(ids.device)
            logits = self(input_ids=ids, pixel_values=pixel_values, position_ids=pos, logits_to_keep=1).logits[:, -1]
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits.float() / temperature
                if top_k:
                    kth = torch.topk(logits, min(top_k, logits.shape[-1])).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            ids = torch.cat([ids, nxt], dim=1)
            if eos_token_id is not None and int(nxt) == eos_token_id:
                break
        return ids


HVLMForCausalLM = HVLMForConditionalGeneration
