# 开发进度 & TODO

## 1. 完成开发

- **模型**：20 层 Gated DeltaNet : Gated Attention = 3:1 混合主干、zero-centered RMSNorm、交错式 M-RoPE、
  GQA 全注意力、Dense SwiGLU、32768 词表共享词嵌入；6 层 ViT + 2×2 merge 投影；1 个 MTP 块（仅训练）。
- **算法**：chunk 内并行的 gated delta rule（纯 PyTorch fp32 参考实现）与 `fla` Triton 内核分派
  （`auto` / `torch` / `fla`，运行时形状不支持可安全回退）；分块交叉熵，不物化 `[tokens, 32768]` logits。
- **数据**：32k 中英代码 BPE tokenizer；`hf:` / `jsonl:` / `parquet:` 流式来源；按来源分别 tokenize、
  训练时按权重混合；文档 `<|endoftext|>` 分隔的扁平 uint16 分片；多模态 collator（图像 49 token、
  (t,h,w) 网格位置、只对 assistant 回复计损失、非法样本丢弃）。
- **训练**：单卡三阶段 trainer（`pretrain` / `align` / `sft`）、WSD 与 cosine 调度、`--resume` 续训
  （含数据顺序与 GradScaler 状态）、自动 mixed precision、按需梯度/CE 检查点（启动探针决定）、
  可选 `torch.compile`、CUDA 分配器硬上限 `--max_mem_gb`。
- **推理**：`scripts/sample.py`（chat / `--raw` 续写、可选图像、temperature / top_k）；`hvlm.register()`
  可把模型注册进 transformers 的 Auto 类。
- **工程**：numpy-only 检查脚本可在无 torch 环境跑；合成数据冒烟；`tools/measure_batch.py` 在本机重测
  显存/吞吐并给出安全 micro_batch。

## 2. TODO

- **KV / 状态缓存**：`generate_simple` 每步重算前缀，只适合冒烟测试；未接入 `GenerationMixin.generate()`。
  需补 GDN 的递推状态 + conv 状态缓存（`recurrent_gated_delta_rule` 已支持带状态递推）。
- **视觉输入**：固定 224×224 单图（等比填充成正方形）；不支持原生动态分辨率、多图交错与视频。
- **预训练细节**：按文档拼接打包，没有跨文档掩码；评测未集成，验证集只给出 ppl。
- **Auto 类**：模型类未自动注册；如需 `AutoModel`，调用 `hvlm.register()`。
- **ViT 预训练**：视觉塔从零训练（14M），未含对比学习/蒸馏预训练这一步，识别能力有限。