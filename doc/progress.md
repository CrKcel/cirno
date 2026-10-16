# TODO

- **KV / 状态缓存**：`generate_simple` 每步重算前缀，只适合冒烟测试；未接入 `GenerationMixin.generate()`。需补 GDN 的递推状态 + conv 状态缓存。
- **端侧量化**
- **身份与 persona**：`system` 预设已支持且不计损失，人设一致性与无人设时的身份分布由 `evals/persona.jsonl`
  覆盖（见 [evaluation.md](evaluation.md)）；仍缺身份注入模板。无预设时身份从语料随机采样、可能前后不一致，
  这是定位内的预期行为。
- **字幕集还有两个坑**：快照里的 `chenrm--anime-subtitles/dialogue.jsonl` 是 `scripts/clean_ass.py
  --zh-only --t2s` 的产物（59,806 集 / 2.93B 字符：全量 11.43B 原始字符里只有 25.6% 是对白，`--zh-only`
  丢掉 16.3 亿字符的日语集与中文集里的假名 OP/ED 行，`--t2s` 把繁体统一成简体）。两个坑还没解决：
  ① `prepare_data` 的 `--max_doc_chars 200000` 之后只有 0.61B 字符能进桶，1,470 篇被截断（共 2.61B
  字符，最大 67 MB）；② 50.6M 行里 9.75M 行（19.3%）是 1–2 字行，即卡拉OK 逐字行（`佐\n佐\n佐…`，
  220 篇里单字行过半），既是字符数虚高的来源，也是噪声。要在 `clean_ass.py` 里丢掉单双字行并按场景切分。
  做完这两步字幕才能按"全量对白"来算，chat 桶也才有从 1.01B 再往上扩的空间。
- **chat 桶是最紧的一环**：LCCC base+large 453M + 字幕有效 428M + b-corpus `v-corpus-zh` ≈130M ≈ 1.01B
  token；配方已经让它读 1.20 遍（退火段 41.4%），全程对话 37.8%。再往上只能扩源：修好卡拉OK 问题后的
  字幕余量、播客/群聊转写。
- **语料清洗剩下的缺**：前四项（`ChineseNovels` 拆章去广告、互动小说 3k 去 `img`/作者注、b-corpus 引擎标记
  与水印、LooksJuicy 人设卡 wikitext）已落进 `prepare_local_data.py`，剩下的：
  - 英文小集：cornell 12.9% 带 `<u>` 与 `&quot;`/`__EX__`，kajiwoto 8.2% 带 `{{name}}`/`#mem:`，
    `chat_haruhi_en` 6.45% 带 `[[…]]` 与 `<span style=…>`；`Narrator：` 前缀因冒号后不接引号被
    `strip_speaker` 漏掉。
  - moegirl 边角：`&nbsp;` 205 条、`[[幼儿体型||…]]` 162 条、`�` 16 条，`_LINK`/`_MEDIA` 覆盖不到。
  - `lars1234/baidu-baike-dataset` 用 `-` 取整条，每篇尾部带 `url` 字段的 `http://baike.baidu.com/view/….htm`。
  - `古籍` 偏文言文，是否留在 long 桶要看训练后的语体评测；`empathetic` 有 22 条 CSV 坏行来自源数据。
  - 清洗产物已变，`configs/pretrain_plan.json` 里的池子仍是字符估算：`prepare_data` 出 `train_tokens` 后重跑
    `tools/plan_mix.py`，把三段权重与各桶 `--max_tokens` 换成实测值。
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