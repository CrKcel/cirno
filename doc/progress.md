# TODO

- **端侧量化**
- **身份与 persona**：`system` 预设已支持且不计损失，人设一致性与无人设时的身份分布由 `evals/persona.jsonl`
  覆盖（见 [evaluation.md](evaluation.md)）；仍缺身份注入模板。无预设时身份从语料随机采样、可能前后不一致，
  这是定位内的预期行为。
- **chat 桶是最紧的一环**：LCCC base+large 453M + 字幕 0.282B 字符 ≈200M + b-corpus `v-corpus-zh` ≈130M
  ≈ **0.78B** token；配方让它读 1.20 遍（多出的 156M 落在退火段，占该段 41.4%）。再往上只能扩源：
  播客/群聊转写、论坛口语帖。
- **语料清洗剩下的缺**：`古籍` 偏文言文，是否留在 long 桶要看训练后的语体评测；`empathetic` 有 22 条
  CSV 坏行来自源数据。
- **池子仍是字符估算**：`configs/pretrain_plan.json` 的 `expected_tokens` 已按本次清洗改写，误差仍有 ±20%；
  `prepare_data` 出 `train_tokens` 后重跑 `tools/plan_mix.py`，把三段权重、`allocate` 与各桶 `--max_tokens`
  换成实测值。
- **SFT 来源的报告**：`--mm_weights` 已经能按目标份额采样，但还没有各来源的行数/长度分布报告，要做形态评测得先有这份报表。`data/sft/refuse.jsonl` 仍未编写，`run_pipeline.sh` 的 74/21/5 配方直接引用它。
- **分片顺序偏差**：`prepare_data` 按 `parquet:` glob 的排序顺序读、遇 `--max_tokens` 即停，而分片顺序
  是有结构的，要支持按目录/按书采样，得给本地 `parquet:` glob 加「按步长/随机挑分片」的能力
  （如 `--shard_stride N` 或 spec 后缀 `@stride=N`）。另一个后果是 cap 卡住时整源的取舍只能靠排列顺序
  表达，所以 `run_pipeline.sh` 把量小的源排在前面。
- **LCCC base 与 large 有重叠**：实测在 large 的一个分片里检索 base 采样，直接命中 6.7%；按分片占比
  （2.8M/12M）折算，两者真实重叠约 **30%**。base 本来就是 large 的清洗子集，所以同时用两个来源会让
  ~7% 的 chat 桶 token 是重复对话；配方又让 chat 读 1.20 遍，重复量因此还要乘 1.2。要去重的话，
  直接去掉 base（损失约 7% 唯一 token）比在 tokenize 之后做近似去重更划算。
- **视觉输入**：固定 224×224 单图（等比填充成正方形）；不支持原生动态分辨率、多图交错与视频。
- **视觉对齐**：ViT 从零训练（14.4M），未含对比学习/蒸馏预训练这一步，识别能力有限；纯文本部署可整个跳过 align 阶段。开源的对话内图文集把图片存在数据集内部，`--mm_data` 还指不过去，需要先把图导出到 `--image_root` 下。
- **预训练细节**：按文档拼接打包，没有跨文档掩码；会话边界依赖 `<|endoftext|>` 与打包策略。
- **Auto 类**：模型类未自动注册；如需 `AutoModel`，调用 `hvlm.register()`。