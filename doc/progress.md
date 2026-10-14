# 开发进度 & TODO

## 1. 完成开发

- **模型**：20 层 Gated DeltaNet : Gated Attention = 3:1 混合主干、zero-centered RMSNorm、交错式 M-RoPE、
  GQA 全注意力、Dense SwiGLU、32768 词表共享词嵌入；6 层 ViT + 2×2 merge 投影；1 个 MTP 块（仅训练）。
- **算法**：chunk 内并行的 gated delta rule（纯 PyTorch fp32 参考实现）与 `fla` Triton 内核分派
  （`auto` / `torch` / `fla`，运行时形状不支持可安全回退）；分块交叉熵，不物化 `[tokens, 32768]` logits。
- **数据**：32k 中英代码 BPE tokenizer；`hf:` / `jsonl:` / `parquet:` 流式来源；按来源分别 tokenize、
  训练时按权重混合；文档 `<|endoftext|>` 分隔的扁平 uint16 分片；多模态 collator（图像 49 token、
  (t,h,w) 网格位置、`system` 人设不计损失、只对 `self`（被模仿方）回复计损失、非法样本丢弃）。
- **训练**：单卡三阶段 trainer（`pretrain` / `align` / `sft`）、WSD 与 cosine 调度、`--resume` 续训
  （含数据顺序与 GradScaler 状态）、自动 mixed precision、按需梯度/CE 检查点（启动探针决定）、
  可选 `torch.compile`、CUDA 分配器上限 `--max_mem_gb`。
- **推理**：`scripts/sample.py`（chat / `--raw` 续写、可选图像、temperature / top_k）；`hvlm.register()`
  可把模型注册进 transformers 的 Auto 类。
- **工程**：numpy-only 检查脚本可在无 torch 环境跑；合成数据冒烟；`tools/measure_batch.py` 在本机重测
  显存/吞吐并给出安全 micro_batch。

## 2. TODO

- **KV / 状态缓存**：`generate_simple` 每步重算前缀，只适合冒烟测试；未接入 `GenerationMixin.generate()`。需补 GDN 的递推状态 + conv 状态缓存。
- **端侧量化**
- **身份与 persona**：`system` 预设已支持且不计损失，但没有身份注入模板、也没有跨会话一致性检查；无预设时身份从语料随机采样、可能前后不一致（定位内的预期行为，需要评测覆盖而不是"修复"）。
- **评测**：未集成对话评测，验证集只给出 ppl。缺人类盲测、越界拒答率/误拒率、回复长度分布、助理腔模式词命中等脚本。
- **拒答数据**：无内置越界 prompt 库与人类口吻拒答模板
- **视觉输入**：固定 224×224 单图（等比填充成正方形）；不支持原生动态分辨率、多图交错与视频。
- **视觉对齐**：ViT 从零训练（14.4M），未含对比学习/蒸馏预训练这一步，识别能力有限；纯文本部署可整个跳过 align 阶段。
- **预训练细节**：按文档拼接打包，没有跨文档掩码；会话边界依赖 `<|endoftext|>` 与打包策略。
- **Auto 类**：模型类未自动注册；如需 `AutoModel`，调用 `hvlm.register()`。