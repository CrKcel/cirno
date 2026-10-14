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
- **推理**：`scripts/sample.py`（chat / `--raw` 续写、可选图像、temperature / top_k）；`hvlm.register()`
  可把模型注册进 transformers 的 Auto 类。
- **工程**：numpy-only 检查脚本可在无 torch 环境跑；合成数据冒烟；`tools/measure_batch.py` 在本机重测
  显存/吞吐并给出安全 micro_batch。

## 2. TODO

- **KV / 状态缓存**：`generate_simple` 每步重算前缀，只适合冒烟测试；未接入 `GenerationMixin.generate()`。需补 GDN 的递推状态 + conv 状态缓存。
- **端侧量化**
- **身份与 persona**：`system` 预设已支持且不计损失，但没有身份注入模板、也没有跨会话一致性检查；无预设时身份从语料随机采样、可能前后不一致（定位内的预期行为，需要评测覆盖而不是"修复"）。
- **评测**：未集成对话评测，验证集只给出 ppl。缺人类盲测、越界拒答率/误拒率、回复长度分布、助理腔模式词命中等脚本。**尤其是没有知识评测**——想判断"通识够不够"必须先有 50–200 题的常识集（天气/心情/身边事/简单事实 + 若干"该答不知道"），用生成而不是 ppl 打分；在那之前"加百科"是盲加（见 `doc/training.md` §2）。
- **拒答数据**：无内置越界 prompt 库与人类口吻拒答模板
- **字幕集清洗**：`scripts/clean_ass.py`（`--zh-only --t2s`）。`chenrm/anime-subtitles` 的 `data` 字段是整份
  `.ass`（见 `doc/datasets.md` §1），全量 11.43B 原始字符里只有 2.93B 是对白（25.6%）——不清洗等于把四分之
  三的字幕 token 预算花在 `[Script Info]` 头、样式表和覆盖标签上。集内还混着 16.3 亿字符的日语（假名占多
  数），`--zh-only` 丢掉这些整集、并顺带删掉中文集里的假名 OP/ED 行（3.6% 的行）；约半数字幕组出繁体，
  `--t2s` 走 opencc 统一成简体（繁体标记字占比 4.74% → 0.33%）。产物 59,806 集 / 2.93B 字符 ≈ 2.0B token，
  是 chat 桶的主力来源。
- **MNBVC 取不到**：它没有自动转换的 parquet 分支（`refs/convert/parquet` revision 不存在），所以
  `parquet:hf://` 也拿不到它——论坛语料暂时不可用。两条路：把 `datasets`
  钉到 3.x（回退加载脚本），或把 MNBVC 落成本地 parquet/jsonl 再喂。
- **英文对话清洗**：英文桶里**真人写的小集**各自带格式噪声，进桶前都要过一遍（逐集的字段/噪声说明见
  `doc/datasets.md` §1 英文对话表）；只有 SODA 干净，不用清洗。
- **SODA 是合成语料**：它占英文桶（5%）的绝大部分，却是 InstructGPT 蒸馏出来的（`language_creators:
  machine-generated`），偏离 `doc/training.md` §0"用人类对话调风格"的原则；定性、抽查结论与「为什么不加
  英文百科」都写在 `doc/datasets.md` §1 英文对话部分。仍需评测覆盖：在 `doc/training.md` §6 的回复长度
  分布与助理腔模式词统计里单独看它的影响，中文闲聊一旦回退就优先降它的权重。
- **SFT 没有配比控制点**：`ChatDataset` 把 `--mm_data` 的各项**按行数拼接**成一个 offsets 列表，没有权重
  参数——`human_dialog.jsonl : persona.jsonl : refuse.jsonl` 的比例谁的行数多谁说了算，`doc/training.md`
  §4.4 要求的"拒答 2–8%"目前只能靠人工数行数落实。所以说"对话比例放到 SFT 调"在当前代码里**没有可调的
  旋钮**。需要补：给 SFT 来源加权重（或按目标比例重采样），并打一份各来源的行数/长度分布报告。
- **百科与长文的语料质量问题**（都是实测出来的，进桶前要处理）：
  - `wikimedia/wikipedia` 中文版 **400 篇里 366 篇繁简混用**，对简体 IM 模型会诱导输出繁体字，
    需要一步繁→简统一（opencc）。
  - `wikimedia/wikisource` 中文版是真公版书，但 **68% 偏文言文**（《中華民國憲法》《南京條約》《正氣歌》
    这类），对 IM 模型是风格污染；要用必须先筛出白话子集。
  - 现代出版中文小说有版权、开源界不存在；小说桶能上量的只有网文，所以"多来点人类写的小说"只能靠
    在现有 20% 内替换提质，不能靠新增来源。
- **分片顺序偏差**：`prepare_data` 按 `parquet:` glob 的排序顺序读、遇 `--max_tokens` 即停，而分片顺序
  是有结构的——各源首/末片实测见 `doc/training.md` §5，wiki / baidu 的顺序截断反而有利，Fineweb-Edu 与
  字幕则偏得很厉害。要处理的话两个方向：给 `parquet:` 的 `hf://` glob 加「按步长/随机挑分片」的能力
  （如 `--shard_stride N` 或 spec 后缀 `@stride=N`），让 Fineweb-Edu `4_5` 取 1% 时能均匀覆盖 `CCI3` /
  `OpenCSG-CC` / `map-cc`，而不是清一色 `CCI3`；同样能用在字幕按目录分组、网文按书排序这两处。或者这几
  个桶先下全量再在上游采样——字幕反正要写清洗脚本，清洗时在全 60 片上采样即可。
- **LCCC base 与 large 有重叠**：实测在 large 的一个分片里检索 base 采样，直接命中 6.7%；按分片占比
  （2.8M/12M）折算，两者真实重叠约 **30%**。base 本来就是 large 的清洗子集，所以同时用两个来源会让
  ~7% 的 chat 桶 token 是重复对话（1.29 遍的桶里再叠一层）。要在意就去重，或只用 large。
- **视觉输入**：固定 224×224 单图（等比填充成正方形）；不支持原生动态分辨率、多图交错与视频。
- **视觉对齐**：ViT 从零训练（14.4M），未含对比学习/蒸馏预训练这一步，识别能力有限；纯文本部署可整个跳过 align 阶段。开源图文对话集（PhotoChat、聊天截图集，见 `doc/datasets.md` §3）把图片存在数据集内部，`--mm_data` 还指不过去，需要先把图导出到 `--image_root` 下。
- **预训练细节**：按文档拼接打包，没有跨文档掩码；会话边界依赖 `<|endoftext|>` 与打包策略。
- **Auto 类**：模型类未自动注册；如需 `AutoModel`，调用 `hvlm.register()`。