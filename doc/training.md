# Cirno-130M 训练

训练分三阶段：**文本预训练 → 视觉对齐 → 指令 / VQA 微调**。单一入口是
[`scripts/train.py`](../scripts/train.py)（单卡 trainer，`--stage` 选择阶段），整条链路见
[`scripts/run_pipeline.sh`](../scripts/run_pipeline.sh)。每个阶段的产物写到 `--out_dir`：
`last/`（含优化器状态，供 `--resume`）与 `final/`（权重 + tokenizer，供下一阶段 `--init_from`）。

## 1. 阶段默认值

以下取自 `scripts/train.py` 的 `STAGE_DEFAULTS`，都可以用同名命令行参数覆盖：

| 阶段 | `--train` | lr | warmup | seq_len | micro_batch | grad_accum | token/步 | mtp_weight | vit_lr_mult |
|---|---|---|---|---|---|---|---|---|---|
| `pretrain` | `llm` | 1e-3 | 2% | 1024 | 2 | 256 | 524k | 0.3 | 1.0 |
| `align` | `vit` | 5e-4 | 5% | 512 | 8 | 32 | 131k | 0.0 | 1.0 |
| `sft` | `llm,vit` | 2e-4 | 3% | 1024 | 8 | 32 | 262k | 0.1 | 0.5 |

`--train` 决定哪些模块可训练（`align` 阶段 LLM 冻结，所以该阶段没有可训练参数的 micro-batch 会被跳过反向）；
`vit_lr_mult` 是 ViT 相对于 LLM 的学习率倍数。

## 2. 阶段 1：文本预训练

- **数据与配比**：英文教育类网页 50%、中文网页 30%、代码 12%、数学 8%。
- **规模**：137M 参数，Chinchilla 约 2.7B token；小模型多喂更值。
- **批量**：脚本默认 `2 × 256 × 1024 = 524k token/步`。
- **优化器**：AdamW（β=0.9/0.95，wd=0.1，不对 norm/bias/嵌入/`A_log`/`dt_bias` 做衰减），梯度裁剪 1.0。
- **学习率**：峰值 1e-3，2% warmup，**WSD 调度**（稳定 → 最后 20% 线性降到 10%）。WSD 的好处是中途任意 checkpoint 都能接着延长训练。若 `gnorm` 经常尖峰，降到 6e-4。
- **MTP**：主体阶段权重 0.3，最后衰减阶段切到 0.1（`--mtp_weight_final 0.1`）。MTP 在这里的主要作用是提升表征质量/训练信号密度。
- **序列**：1024。需要长上下文用长上下文配置，每步 token 数不变；也可以绝大部分 token 用 1024、最后 10–20% 切到 2048 作为长上下文收尾。同一序列内的文档用 `<|endoftext|>` 连接。

## 3. 阶段 2：视觉对齐

- 冻结 LLM，只训练 ViT + 投影器（`--stage align`），数据为图像-描述对（如 LLaVA-Pretrain 558k 一类），seq 512，1 个 epoch，lr 5e-4。
- **数据须全是带图记录**：LLM 冻结时，没有图的 micro-batch 没有可训练参数（脚本会跳过其反向）。
- 这个 ViT 只有 14M 且从零训练，视觉识别能力有限（更像"能认颜色/形状/粗略场景"）。要更强：把 `vision_config` 加大（代价是显存），
  或先用对比学习/蒸馏预训练 ViT 再接入——本仓库未包含这一步。

## 4. 阶段 3：指令 / VQA 微调

- 全部参数可训练，ViT 学习率为 LLM 的 0.5 倍，lr 2e-4，1–2 个 epoch，MTP 权重 0.1。
- **务必混入纯文本对话数据**（建议 ≥ 30%），否则文本能力会退化。
- 数据格式（二选一，可混用）：
  `{"image": "相对路径", "conversations": [{"from":"human","value":"<image>\n问题"},{"from":"gpt","value":"回答"}]}` 或 `{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}`。
- 只对 assistant 回复（及其 `<|im_end|>`）计损失；图像块被截断或回复被截没的样本会被丢弃。

## 5. 数据准备

```bash
# 0. tokenizer：32k BPE，中英 + 代码
python scripts/train_tokenizer.py --out tokenizer \
    --source "hf:HuggingFaceFW/fineweb-edu:text:sample-10BT:train@300000" --source "jsonl:data/zh/*.jsonl:text@300000"

# 1. 每个来源单独 tokenize，训练时再用权重混合，改配比不必重新 tokenize
python scripts/prepare_data.py --tokenizer tokenizer --source "jsonl:data/zh/*.jsonl:text" \
    --out data/bin/zh --max_tokens 1800000000 --val_tokens 4000000
```

- `--source` 支持 `hf:`（流式 datasets）、`jsonl:`、`parquet:` 等来源，末尾 `@N` 限制文档数；
  文档之间用 `<|endoftext|>` 分隔，产物是扁平 uint16 token 流 + `meta.json`。
- `--val_tokens` 会把前 N 个 token 切给 `val.bin`；训练时定期在验证集上打印 `loss` / `ppl`。
- 训练时用 `--data dirA:0.5,dirB:0.3,...` 指定混合权重（见阶段 1 的配比）。
- 真实数据跑之前先跑 `bash scripts/smoke_test.sh` 合成数据冒烟（见 [`AGENTS.md`](../AGENTS.md)）。

## 6. 训练开关

| 参数 | 说明 |
|---|---|
| `--tokens` / `--max_steps` / `--epochs` | 训练长度：pretrain 用总 token 数或步数，align/sft 用 epoch 数 |
| `--max_mem_gb` | CUDA 分配器硬上限（默认 8），超预算快速失败；`0` 关闭上限 |
| `--schedule cosine` / `--decay_frac` / `--min_lr_ratio` | 调度器选择与衰减比例 |
| `--resume` | 从 `out_dir/last` 续训，含数据顺序与 GradScaler 状态 |
| `--eval_every` / `--eval_batches` / `--save_every` / `--log_every` | 验证频率与批次、保存与日志频率 |
| `--precision` / `--gdn_impl` / `--grad_ckpt` / `--ce_ckpt` / `--compile` / `--tf32` | 精度与 kernel，见[架构 §3.3](architecture.md#33-精度与-kernel) |
