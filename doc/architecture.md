# Cirno-130M 架构

Cirno-130M 是一个约 130M 参数的混合视觉语言模型：20 层 decoder 按 3 层线性注意力: 1 层全注意力 交替，外挂一个 6 层 ViT 和一个 MTP 块。

## 1. 模型设计

| 项 | 配置 | 说明 |
|---|---|---|
| 层数 / 隐层 | 20 层，d=640 | 3 层 Gated DeltaNet : 1 层 Gated Attention，共 15 + 5 |
| 线性注意力 | Gated DeltaNet；K 头 4、V 头 8，头维 64；depthwise conv 核 4 | q/k 做 L2 norm，衰减 `g=-exp(A_log)·softplus(a+dt_bias)`，写入强度 `β=sigmoid(b)`，输出过 gated RMSNorm |
| 全注意力 | 10 Q 头 / 2 KV 头（GQA），头维 64 | QK-norm、sigmoid 输出门、部分 RoPE（rotary 一半维度）、`rope_theta=1e6` |
| MLP | Dense SwiGLU，ffn=1792 | 每层一个，无 MoE |
| 归一化 | zero-centered RMSNorm（`y = x̂·(1+w)`） | |
| 位置编码 | 交错式 M-RoPE，`mrope_section=[8,4,4]` | 文本三轴相同，等价普通 RoPE；图像 token 带 (t,h,w) 网格位置 |
| 视觉塔 | ViT 6 层 / 384 维 / 6 头，224px、patch16，2×2 merge + 两层 MLP 投影 | 每图 49 个 LLM token，约 14.4M 参数 |
| MTP | 1 个 MTP 块（融合 hidden[t] 与 embed[t+1]，一层全注意力 decoder，共享词表头）预测 t+2 | |
| 词表 | 32768，中英代码字节级 BPE；LM 头与词嵌入共享 | |

## 2. 参数量

| 部分 | 参数 |
|---|---|
| 词嵌入（= LM 头） | 20.97 M |
| 20 层 decoder | 95.68 M（线性层 4.77M，全注意力层 4.83M） |
| **纯 LLM** | **116.65 M** |
| 视觉塔 + 投影 | 14.36 M |
| **LLM + 视觉** | **131.0 M** |
| MTP | 5.66 M |
| **训练时总计** | **136.7 M** |

## 3. 显存&吞吐

### 3.1 显存控制

1. **自动混合精度**（`--precision auto|bf16|fp16|fp32`）+ fp32 主权重：权重 + 梯度 + AdamW 两个状态
   = 16 B/参数 ≈ 2.0 GB（137M 参数）。`auto`（默认）在有原生 bf16 的卡（Ampere+）上用 bf16，否则用 fp16
   并自动挂 `GradScaler`（其状态随 checkpoint 一起存/取）；`fp32` 关闭 autocast，只用于排查数值问题。
2. **不做激活检查点**：反向直接用前向保存的激活，不重算。峰值因此由 `micro_batch × seq_len` 决定，
   每步 token 数交给 `--grad_accum` 凑；`--compile` 融掉大部分中间量，可再放松约 35%（见 §3.3）。
3. **分块交叉熵**（`chunked_cross_entropy`）：词表头 logits 分块算，不构造整块 `[tokens, 32768]` fp32；
   反向保留的是 bf16 logits，fp32 的交叉熵中间量只在一个块内存在（`ce_chunk_size`，默认 2048）。
4. **共享词表头**：MTP 与主头共用词嵌入，不额外占参数。
5. **`--max_mem_gb 8`**：分配器上限，超预算OOM。

### 3.2 精度与 kernel

| 参数 | 说明 |
|---|---|
| `--precision auto`（默认） | 有原生 bf16 用 bf16，否则 fp16 + GradScaler。本卡上 bf16 与 fp16 同速。 |
| `--gdn_impl auto`（默认） | 装了 `flash-linear-attention` 就用它的 Triton 内核（快 ≈1.7×、省 ≈0.5 GB）；没装或不可用则退回纯 PyTorch。显式 `fla` 在本机跑不了时启动即报错。 |
| `--compile` | `torch.compile(model, dynamic=False)`，再快 ≈1.4×（首步一次性 JIT，之后走 `TORCHINDUCTOR_CACHE_DIR` 缓存；编译失败自动回退 eager）。 |
| `--tf32` | 允许 fp32 matmul 用 TF32。纯 PyTorch 的 delta rule 是 fp32 精度岛，默认关闭。 |

数值上这些路径都与 fp32 参照对齐：fla 内核对 fp32 参照的前向最大差 ≈1e-3（普通 bf16 舍入），
`--compile` 只改变算子融合方式，loss 差 7.6e-4、梯度绝对差 ~4e-4。

### 3.3 实测

使用 RTX 4060 Laptop 8 GB，524k token/步（`tools/measure_batch.py` 的 peak reserved 与 tok/s，绝对值随机器负载变化）

| 用途 | 关键参数 | tok/s | peak reserved |
|---|---|---|---|
| 默认 | `--seq_len 1024 --micro_batch 2 --grad_accum 256` | 11174 | 4.87 GB |
| 最快 | 同上但 `--micro_batch 4 --grad_accum 128 --compile` | **14753** | 4.80 GB |
| 长上下文 | `--seq_len 2048 --micro_batch 2 --grad_accum 128 --compile` | 14051 | 4.80 GB |

三者每步都是 524k token（`micro_batch × grad_accum × seq_len`）。**micro_batch 取"刚好放得下"的最大值，
每步 token 数交给 `--grad_accum` 凑**：显存与吞吐几乎只由 micro_batch × seq_len 决定，用
`python tools/measure_batch.py --stage pretrain [--compile]` 可在本机重测，
它最后会打印各 seq_len 下的安全 micro_batch。