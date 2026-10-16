# 开发进度 & TODO

## 1. 完成开发

- **模型**：20 层 Gated DeltaNet : Gated Attention = 3:1 混合主干、zero-centered RMSNorm、交错式 M-RoPE、
  GQA 全注意力、Dense SwiGLU、32768 词表共享词嵌入；6 层 ViT + 2×2 merge 投影；1 个 MTP 块（仅训练）。
- **算法**：chunk 内并行的 gated delta rule（纯 PyTorch fp32 参考实现）与 `fla` Triton 内核分派
  （`auto` / `torch` / `fla`，运行时形状不支持可安全回退）；分块交叉熵，不物化 `[tokens, 32768]` logits。
- **数据**：32k 中英代码 BPE tokenizer（`\p{N}+` 让数字整段成词，不学 Qwen2 的逐位切分——那是为算术
  准备的，而本项目不做数学、闲聊里数字却很多；`@N` 是**每个源各自的预算**，round-robin 用 `zip_longest`
  而不是 `zip`，否则最短源会把整轮卡住、并让目标分布只占 tokenizer 语料的 ~1%）；`hf:` / `jsonl:` /
  `parquet:` / `txt:` 流式来源（`@N` 限制文档
  数；字段支持整条记录 `-`、嵌套路径 `a.b` 与拼接 `a+b`，列表型字段自动拼成一篇文档）；`parquet:` 支持
  远端 `hf://` glob，用来读 Hub 的自动转换 parquet 分支；
  `--desegment` 去掉分词型中文语料（LCCC）里汉字之间的空格，tokenizer / prepare_data / train(align,sft)
  三个入口都有；按来源分别 tokenize、训练时按权重混合；文档 `<|endoftext|>` 分隔的扁平 uint16 分片；
  多模态 collator（图像 49 token、(t,h,w) 网格位置、`system` 人设不计损失、只对 `self`（被模仿方）回复
  计损失、非法样本丢弃）。
  `--mm_data` 既收本地 jsonl，也收 HF/parquet/glob 来源（先流式落一份缓存 jsonl 再训练）；SFT 直接认
  `messages` / `conversations` / 整段对话列表 / `instruction→output` / PersonaChat 这几种开源数据集形状。
- **训练**：单卡三阶段 trainer（`pretrain` / `align` / `sft`）、WSD 与 cosine 调度、`--resume` 续训
  （含数据顺序与 GradScaler 状态）、自动 mixed precision、按需梯度/CE 检查点（启动探针决定）、
  可选 `torch.compile`、CUDA 分配器上限 `--max_mem_gb`。
  训练配方是文件：`configs/pretrain_plan.json` 声明三段预算、形状、每桶 `allocate` 与配比目标，
  `tools/plan_mix.py --plan ... --write` 用各桶实测 `train_tokens` 解析出 `plan.resolved.json`，
  `train.py --plan --phase A|B|C` 直接执行——权重与 `--tokens` 都是推导值，学习率是一条以 token 为轴的
  全局 WSD（2% warmup → 恒定 → 最后 20% 衰减，衰减起点 = 退火段起点），不再需要手写累计 token 与
  `--decay_frac 0`。5.39B token 的三段课程见 `doc/training.md` §2：chat 刻意读 1.20 遍、
  wiki 读 0.86 遍，把 200M token 从百科长尾挪到退火段的对话重放，全程对话占比 34.2% → 37.5%。
  SFT 的配比有了旋钮：`--mm_weights` 按目标份额采样（小于份额的源被重复到该份额），
  §4.4 的拒答 2–8% 不再靠人工数行数。
- **监控**：每个 run 写 `metrics.jsonl`（结构化的 loss / lm / mtp / lr / lr_scale / gnorm / tok_s / step_s /
  mem / 验证 ppl / 全局 token 进度）与 `health.jsonl`（异常与干预事件）；`tools/watch_training.py`
  零依赖实时看板（进度、loss EMA、sparkline、ETA、告警）；`hvlm/monitor.py` 的判定与受控干预
  （loss/梯度尖峰触发 lr 回退并逐步释放、可选按验证集早停、吞吐与显存告警），
  `--no_intervention` / `--no_monitor` 可分别关掉干预与整个指标流。见 `doc/monitoring.md`。
- **推理**：`scripts/sample.py`（chat / `--raw` 续写、可选图像、temperature / top_k）；`hvlm.register()`
  可把模型注册进 transformers 的 Auto 类。
- **工程**：numpy-only 检查脚本可在无 torch 环境跑；合成数据冒烟；`tools/measure_batch.py` 在本机重测
  显存/吞吐并给出安全 micro_batch。

## 2. TODO

- **KV / 状态缓存**：`generate_simple` 每步重算前缀，只适合冒烟测试；未接入 `GenerationMixin.generate()`。需补 GDN 的递推状态 + conv 状态缓存。
- **端侧量化**
- **身份与 persona**：`system` 预设已支持且不计损失，但没有身份注入模板、也没有跨会话一致性检查；无预设时身份从语料随机采样、可能前后不一致（定位内的预期行为，需要评测覆盖而不是"修复"）。
- **评测**：未集成对话评测，验证集只给出 ppl（监控会跟踪它的最好值与平台期，但不替代评测）。缺人类盲测、越界拒答率/误拒率、回复长度分布、助理腔模式词命中等脚本。**尤其是没有知识评测**——想判断"通识够不够"必须先有 50–200 题的常识集（天气/心情/身边事/简单事实 + 若干"该答不知道"），用生成而不是 ppl 打分；在那之前"加百科"是盲加（见 `doc/training.md` §2）。
- **拒答数据**：无内置越界 prompt 库与人类口吻拒答模板
- **字幕集清洗**：`scripts/clean_ass.py`（`--zh-only --t2s`）。`chenrm/anime-subtitles` 的 `data` 字段是整份
  `.ass`（见 `doc/datasets.md` §1.1），全量 11.43B 原始字符里只有 2.93B 是对白（25.6%）——不清洗等于把四分之
  三的字幕 token 预算花在 `[Script Info]` 头、样式表和覆盖标签上。集内还混着 16.3 亿字符的日语（假名占多
  数），`--zh-only` 丢掉这些整集、并顺带删掉中文集里的假名 OP/ED 行（3.6% 的行）；约半数字幕组出繁体，
  `--t2s` 走 opencc 统一成简体（繁体标记字占比 4.74% → 0.33%）。产物 59,806 集 / 2.93B 字符。
  **两个还没处理的问题**：① `prepare_data` 的 `--max_doc_chars 200000` 之后只有 0.61B 字符能进桶，
  1,470 篇被截断（最大 67 MB）；② 这些超长文档是卡拉OK 逐字行（`佐\n佐\n佐…`），既是字符数虚高的来源，
  也是噪声，需要在 `clean_ass.py` 里丢掉单字重复行。做完这两步字幕才能按"全量对白"来算，chat 桶也才有
  从 1.01B 再往上扩的空间。
- **chat 桶是最紧的一环**：LCCC base+large 453M + 字幕有效 428M + b-corpus `v-corpus-zh` ≈130M ≈ 1.01B
  token；配方已经让它读 1.20 遍（退火段 41.4%），全程对话 37.5%。再往上只能扩源：修好卡拉OK 问题后的
  字幕余量、播客/群聊转写——重采样第二遍的边际收益低于新语料。
- **流行文化语料没有清洗脚本**：`Limour/b-corpus` 全量已归入六个桶（1,819 个文件 / 552.6 MB，新增
  ≈370M token，映射见 `doc/datasets.md` §1），但每个子目录都要同一个前置步骤：解压 `.txt.gz`、按行
  或按章、按场景切分、去掉 `角色名：`/`system：` 前缀，仓库里还混着 130 多个 `.py`/`.bat` 提取脚本；
  `互动小说`（mrzjy 3k，≈149M token）自标 uncleaned，且 `content_tag`/`role` 只能在落盘时丢掉；
  `q-corpus` 是 R18，要逐篇筛；`Miaowuawa/ChineseNovels` 整本一篇，要拆章，否则每本只进前 200k 字符；
  `mrzjy/chinese_moegirl_wiki_corpus_raw` 是原始 wikitext，要剥模板/表格/注释并按篇截断。清洗落盘后按
  新的 `train_tokens` 用 `tools/plan_mix.py` 重算三段权重与各桶 `--max_tokens`。现状与取法见
  `doc/datasets.md` §1.1 / §1.5 / §1.7。
- **英文对话清洗**：英文桶里**真人写的小集**各自带格式噪声，进桶前都要过一遍（逐集的字段/噪声说明见
  `doc/datasets.md` §1.2 英文对话表）；只有 SODA 干净，不用清洗。
- **英文桶以合成语料为主**：SODA 是 InstructGPT 蒸馏出来的（`language_creators:machine-generated`），
  新增的 `noanabeshima/TinyStoriesV2`（≈550M token）是 GPT-4 生成的儿童故事且是叙事不是对话，两者合计
  约占该桶九成，都偏离 `doc/training.md` §0"用人类对话调风格"的原则；定性、抽查结论与「为什么不加英文
  百科」都写在 `doc/datasets.md` §1.2 英文对话部分。仍需评测覆盖：在 `doc/training.md` §6 的回复长度
  分布与助理腔模式词统计里单独看它们的影响，中文闲聊一旦回退就优先降它们的权重。
- **SFT 来源的报告**：`--mm_weights` 已经能按目标份额采样，但还没有各来源的行数/长度分布报告
  （回复长度分位数、助理腔模式词命中），要做 §6 的形态评测得先有这份报表。
- **小说桶的来源限制**：现代出版中文小说有版权、开源界不存在；小说桶能上量的只有网文，所以"多来点
  人类写的小说"只能靠在小说桶内替换提质（1.176B token 就是网文 parquet 分支的全量），不能靠新增来源。
- **分片顺序偏差**：`prepare_data` 按 `parquet:` glob 的排序顺序读、遇 `--max_tokens` 即停，而分片顺序
  是有结构的——各源首/末片实测见 `doc/training.md` §5，baidu 的排序让"取前几片"正好落在常见实体上
  （配方只读 0.86 遍，截断落在冷门实体上，是有意的），字幕则偏得很厉害（按目录分片，只取一部分就只有
  剧场版、没有 TV 系列）。当前配方靠"字幕跑全 60 片、网文下全 10 片、其余桶预算覆盖池子"绕开这个问题；
  要支持按目录/按书采样，得给 `parquet:` 的 `hf://` glob 加「按步长/随机挑分片」的能力
  （如 `--shard_stride N` 或 spec 后缀 `@stride=N`）。
- **LCCC base 与 large 有重叠**：实测在 large 的一个分片里检索 base 采样，直接命中 6.7%；按分片占比
  （2.8M/12M）折算，两者真实重叠约 **30%**。base 本来就是 large 的清洗子集，所以同时用两个来源会让
  ~7% 的 chat 桶 token 是重复对话；配方又让 chat 读 1.20 遍，重复量因此还要乘 1.2。要去重的话，
  直接去掉 base（损失约 7% 唯一 token）比在 tokenize 之后做近似去重更划算。
- **视觉输入**：固定 224×224 单图（等比填充成正方形）；不支持原生动态分辨率、多图交错与视频。
- **视觉对齐**：ViT 从零训练（14.4M），未含对比学习/蒸馏预训练这一步，识别能力有限；纯文本部署可整个跳过 align 阶段。开源的对话内图文集把图片存在数据集内部，`--mm_data` 还指不过去，需要先把图导出到 `--image_root` 下。
- **预训练细节**：按文档拼接打包，没有跨文档掩码；会话边界依赖 `<|endoftext|>` 与打包策略。
- **Auto 类**：模型类未自动注册；如需 `AutoModel`，调用 `hvlm.register()`。