# 训练数据集

预训练按**桶**组织——每个桶一个 `prepare_data --out`，训练时按权重混合；对话 / 角色扮演 SFT 另列。
取数语法见 [common_io.py](../scripts/common_io.py) 的模块 docstring：
`hf:<数据集>:<字段>[:<config>[:<split>]]`，字段 `-` 表示整条记录、`a.b` 走嵌套路径、`a+b` 顺序拼接，
列表型字段自动用换行拼成一篇文档，末尾 `@N` 限制文档数。

## 0. 取数约定

> 1. `datasets` ≥ 4 已移除加载脚本支持，凡是只带 `*.py` 加载脚本的数据集（LCCC、DailyDialog、
>    EmpatheticDialogues）用 `hf:` 直连都会报 `Dataset scripts are no longer supported`。改走 Hub 自动转换的
>    parquet 分支：`parquet:hf://datasets/<repo>@refs/convert/parquet/<config>/<split>/*.parquet:<field>`。
>    路径里的 `@refs/...` 是 revision，**不是** `@N` 文档上限。凡是"只有加载脚本"的集，下面一律只给
>    parquet 写法。
> 2. `silver/lccc` 的 `dialog` 字段是**空格分词**的（`你 去 那儿 竟然` 这种），所有吃它的步骤都要带
>    `--desegment`。
> 3. **字符 → token**：中文按 0.70 token/字符、英文 0.25 估算（32k BPE，实测口径 1.18B token / 1.68B 字符
>    ≈ 0.70），误差 ±20%，真实值以 `prepare_data` 的 `train_tokens` 为准。压缩语料先抽样标定「字符/压缩
>    字节」再外推。
> 4. **`txt:` 是一个文件一篇文档**，且只吃 UTF-8 纯文本：`.txt.gz` 必须先解压；一个作品几千行会被算成
>    一篇，`--max_doc_chars`（默认 200000）只保留前 20 万字符，要按行/场景切分后再落盘。

## 1. 预训练（按桶）

桶由 [run_pipeline.sh](../scripts/run_pipeline.sh) 的 DATA 段定义。各桶训练配额不是手写的，而是从各桶
`meta.json` 的 `train_tokens` 反算：`该桶的三段课程权重 × 对应阶段 token 之和 ÷ 该桶 train_tokens`
（各段 token 见 [training.md §2](training.md)；`tools/plan_mix.py` 直接把三段 `--data` 串算出来）。当前预算是
各桶池子之和，每个桶恰好走一遍，所以这个比值应当是 1：明显低于 1 说明有语料没被训练（先看
`prepare_data` 的 `--max_tokens` 有没有卡在池子下面），高于 1 就是在背语料。

八个桶的现状：

- **`chat`** — 中文人类对话 / 口语
  - 阶段权重 A / B / C：— / 20.8% / 34%
  - 全程 token 1.010B；池子 1.010B；遍数 1.00
  - 来源：LCCC base+large、字幕、b-corpus `v-corpus-zh`
- **`en_chat`** — 英文对话
  - 阶段权重 A / B / C：— / 24.9% / 3.6%
  - 全程 token 810M；池子 810M；遍数 1.00
  - 来源：soda、真人小集、b-corpus `v-corpus-en`+`emoji`、`TinyStoriesV2`
- **`fiction`** — 小说 / 网文 / 剧本对白
  - 阶段权重 A / B / C：45% / 24.1% / 28%
  - 全程 token 1.593B；池子 1.593B；遍数 1.00
  - 来源：webnovel-chinese、b-corpus `互动小说`/`近现代小说`/`翻译名著`/`轻小说`、ChineseNovels
- **`long`** — 长文 / 非对话文本
  - 阶段权重 A / B / C：35% / 1.9% / 4%
  - 全程 token 527M；池子 527M；遍数 1.00
  - 来源：Zhihu-KOL、b-corpus `提高逼格`/`畅销书`/`散文集`/`古籍`
- **`wiki`** — 通识 / 百科 / 常识
  - 阶段权重 A / B / C：20% / 28.3% / 28.4%
  - 全程 token 1.425B；池子 1.425B；遍数 1.00
  - 来源：baidu-baike、萌娘百科
- **`roleplay`** — 人设卡 / 单轮扮演
  - 阶段权重 A / B / C：— / — / 1.6%
  - 全程 token 17.2M；池子 17.2M；遍数 1.00
  - 来源：shibing624、Johnson8187、b-corpus `q-corpus`、LooksJuicy
- **`roleplay_char`** — 小说 / 影视角色的问答
  - 阶段权重 A / B / C：— / — / 0.47%
  - 全程 token 5.1M；池子 5.1M；遍数 1.00
  - 来源：ChatHaruhi 54K、b-corpus `ChatHaruhi`
- **`roleplay_catgirl`** — 猫娘人设
  - 阶段权重 A / B / C：— / — / 0.01%
  - 全程 token 0.15M；池子 0.15M；遍数 1.00
  - 来源：kxdw2580 v2

合计：**预算 5.39B = 各桶池子之和**，实际消耗 5.39B、重复 0，没有哪个桶被 `--max_tokens` 截断；chat
（1.010B）仍是最紧的一环。roleplay 三个桶合计 22.5M token，任何一桶超过 2 遍就是在背语料而不是学风格。

**`Limour/b-corpus` 全量入桶**：这份合辑（1,819 个文件 / 552.6 MB）按内容拆到六个桶——`v-corpus-zh` → chat，
`v-corpus-en`、`emoji/kajiwoto` → en_chat，`互动小说`、`近现代小说`、`翻译名著`、`轻小说` → fiction，
`提高逼格`、`畅销书`、`散文集`、`古籍` → long，`q-corpus` → roleplay，`ChatHaruhi` → roleplay_char。
按压缩字节抽样外推合计新增 ≈370M token（chat +130M、en_chat +52M、fiction +163M、long +21M、
roleplay +2.2M、roleplay_char +1.6M）。

**英文桶另加 `noanabeshima/TinyStoriesV2`**（2.7M 篇 GPT-4 儿童故事，≈550M token，见 §1.2），两份合计把
池子推到 ≈5.05B。

**另有三份进桶**：`Miaowuawa/ChineseNovels` → fiction（截断后 ≈254M token）、
`mrzjy/chinese_moegirl_wiki_corpus_raw` → wiki（原始 wikitext，清洗后抽样外推 ≈82M token）、
`LooksJuicy/Chinese-Roleplay-SingleTurn` → roleplay（≈3.3M token）。池子合计 **≈5.39B**，预训练预算就取
这个总量（1.21B / 3.10B / 1.08B 三段），所以每个桶各读一遍、`passes` 全是 1.00x。ChineseNovels 与萌娘
百科是抽样外推（±20%）：落盘后按 `prepare_data` 的实测 `train_tokens` 重跑 `tools/plan_mix.py`，把预算与
各桶 `--max_tokens` 一起挪到新的池子之和。

每节列该桶的来源与已知噪声，【流行文化】标记的是补"网感"（ACG 梗、视觉小说台词、作品角色扮演）的那几份。

### 1.1 chat — 中文人类对话 / 口语

- **[silver/lccc](https://hf.co/datasets/silver/lccc)** — 人类对话；base 6.8M + large 12M 条 / 1.63 GB
  parquet，该桶 453M token
  - 取法：`parquet:hf://datasets/silver/lccc@refs/convert/parquet/base/train/*.parquet:dialog`（large 同理）
  - 微博闲聊二人会话，该桶最大来源，SFT 也用它。`dialog` 是列表，会自动拼成一篇文档；**分词文本，
    需 `--desegment`**。⚠️ base 是 large 的清洗子集，实测重叠约 **30%**，同时用两个来源会让 ~7% 的
    chat token 是重复对话；要在意就去重或只用 large
- **[chenrm/anime-subtitles](https://hf.co/datasets/chenrm/anime-subtitles)** — 人类对话；67k 部字幕 /
  1.3 GB parquet，清洗后有效 **428M token**
  - 取法：`jsonl:data/chat/subtitles.jsonl:text`（先 `scripts/clean_ass.py --zh-only --t2s` 落盘，见
    [progress.md](progress.md)）
  - 动画字幕，短句口语多。⚠️ `data` 字段是**整份 `.ass` 文件**：含 `[Script Info]` 头、样式定义与时间轴，
    必须先清洗。全量 11.43B 原始字符只剩 2.93B 对白，而 `prepare_data` 的 `--max_doc_chars 200000` 之后
    **真正能进桶的只有 0.61B 字符 ≈ 0.43B token**：1,470 篇超过 200k 字符（最大 67 MB），里面是卡拉OK
    逐字行（`佐\n佐\n佐…`），这批文档是待清理的噪声
- **[Limour/b-corpus](https://hf.co/datasets/Limour/b-corpus) `v-corpus-zh/**`** — 人类对话【流行文化】；
  820 个 `.txt.gz` / 187.6 MB 压缩，抽样 0.96 字符/压缩字节 → ≈180M 字符 ≈ **130M token 量级**
  - 取法：`hf download` → 解压 → 按行/按场景落 jsonl → 作为该桶的额外 `--source`
  - 视觉小说台词，按 `厂商/作品/线路` 组织（`v-corpus-zh/07thExpansion/寒蝉鸣泣之时/…`、
    `v-corpus-zh/5pb/命运石之门@MCSeekeri/本体/SG01..txt.gz`），行是 `角色名：「台词」` 或纯旁白，中文行
    中位 21 字，是二次元口语与角色语气最浓的一档，也是 chat 桶（最紧的一环）目前最大的新增量。
    ⚠️ cc-by-nc-sa-4.0 + `not-for-all-audiences`，来源含 R18 作品；仓库里混着 130 多个 `.py`/`.bat`
    提取脚本，glob 要写成 `*.txt.gz`；`txt:` 一个文件一篇文档，必须先拆行

同仓的 `v-corpus-en`（135 个 `.txt.gz` / 69.3 MB）是英文视觉小说对白，归 §1.2。该桶的池子 = LCCC
base+large 453M + 字幕有效 428M + `v-corpus-zh` ≈130M ≈ **1.01B**；再要扩只能从修好卡拉OK
问题后的字幕余量上想办法。

### 1.2 en_chat — 英文对话

- **[allenai/soda](https://hf.co/datasets/allenai/soda)** — ⚠️ **合成**（InstructGPT）；1.2M 会话 /
  ~190M token
  - 取法：`hf:allenai/soda:dialogue:default:train`
  - 该桶主力。语料卡自标 `language_creators:machine-generated`，客服腔/助理腔不明显但确实不是人类语料；
    取 `dialogue` 字段即可（`speakers` 与它对齐，SFT 里能自动判谁是被模仿方），`narrative` 是场景设定、
    预训练不用。它排在 `TinyStoriesV2` 之后、真人小集之前，所以 cap 必须放在整个池子之上（≈810M），
    否则排在它后面的小集会被整段丢掉
- **[noanabeshima/TinyStoriesV2](https://hf.co/datasets/noanabeshima/TinyStoriesV2)** — ⚠️ **合成**（GPT-4）；
  2.7M 篇 / parquet 1.1 GB（validation 27.6K），抽样 1 万篇均 814 字符（中位 726、163 词）→ ≈2.2B 字符
  ≈ **550M token**
  - 取法：`hf:noanabeshima/TinyStoriesV2:text:default:train`
  - GPT-4 生成的儿童短篇故事，ASCII 化、词汇刻意简单（`roneneldan/TinyStories` 的 V2-GPT4 子集），
    cdla-sharing-1.0，字段只有 `text`、开箱即用。⚠️ 是**叙事不是对话**，放进"英文对话"桶会明显稀释对话
    分布；它是该桶最大的一份（550M token，约占池子七成），也是**列在最前**的一份：cap 必须高过整个池子
    （≈810M），否则它后面的 SODA 与真人小集一个 token 都进不来
- **[li2017dailydialog/daily_dialog](https://hf.co/datasets/li2017dailydialog/daily_dialog)** — ✅ 人写；
  13k 会话
  - 取法：`parquet:hf://datasets/li2017dailydialog/daily_dialog@refs/convert/parquet/default/train/*.parquet:dialog`
  - 日常口语，短句多，英文闲聊里最干净的一档。⚠️ 实测标点被空格隔开（`going for a few beers after
    dinner ?`），进桶前要去掉标点前空格。只有加载脚本，`hf:` 直连已失效
- **[ParlAI/blended_skill_talk](https://hf.co/datasets/ParlAI/blended_skill_talk)** — ✅ 众包；4.8k 会话
  - 取法：`hf:ParlAI/blended_skill_talk:free_messages:default:train`
  - 同时含 persona / 共情 / 知识三种模式，和本项目"人设 + 陪伴"的定位最接近；`personas` 是列表，SFT 里
    能当 system。量很小
- **[facebook/empathetic_dialogues](https://hf.co/datasets/facebook/empathetic_dialogues)** — ✅ 人写；
  25k 会话
  - 取法：`parquet:hf://datasets/facebook/empathetic_dialogues@refs/convert/parquet/default/train/*.parquet:prompt+utterance`
  - cc-by-nc-4.0，情绪陪伴向。**一行一句**，取字段要写 `prompt+utterance`：用 `-` 会把 `conv_id`/
    `speaker_idx`/情绪标签一起拼进去。⚠️ 逗号被转义成 `_comma_`，要还原。只有加载脚本
- **[Fishball02/topical-chat](https://hf.co/datasets/Fishball02/topical-chat)** — ✅ MTurk；9.7k 会话
  - 取法：`hf:Fishball02/topical-chat:-:default:train`
  - Topical-Chat 的镜像（官方 `Conversational-Reasoning/Topical-Chat` 已加载失败、无 parquet 分支）。
    两人无预设角色、知识带聊，接近真实网聊。⚠️ 文本带 `<s>[agent_1]…</s><s>[agent_2]…</s>` 标记，要剥掉
- **[mylesmharrison/cornell-movie-dialog](https://hf.co/datasets/mylesmharrison/cornell-movie-dialog)** —
  ✅ 剧本；305k 行
  - 取法：`parquet:hf://datasets/mylesmharrison/cornell-movie-dialog@refs/convert/parquet/default/train/*.parquet:text`
  - 影视对白，口语味足、情绪起伏大。⚠️ **每行一句**不是整段会话，且带角色名前缀（`BIANCA   They do
    not!`）与 `\r\n`，要按角色切分并去掉前缀。官方 `cornell-movie-dialog/cornell_movie_dialog` 无 parquet
    分支，已不可用
- **[thu-coai/esconv](https://hf.co/datasets/thu-coai/esconv)** — ✅ 人写；1.3k 段对话
  - 取法：`hf:thu-coai/esconv:text:default:train`
  - 情感支持对话；量很小。⚠️ `text` 字段实测是**一整条 JSON 字符串**（含 `experience_type` 等键），要么
    改嵌套字段路径，要么先清洗
- **[Limour/b-corpus](https://hf.co/datasets/Limour/b-corpus) `v-corpus-en/**`** — ✅ 剧本【流行文化】；
  135 个 `.txt.gz` / 69.3 MB 压缩，抽样 2.86 字符/压缩字节 → ≈200M 字符 ≈ **50M token 量级**（Han 0.00，
  纯英文）
  - 取法：同 §1.1 的 `v-corpus-zh`：解压 → 拆行 → `jsonl:`
  - 英文视觉小说台词，格式与中文侧一致（`角色名：「台词」`），是该桶最大的一份人类写作文本（+≈50M，
    该桶池子与顺序见本节末）
- **[Limour/b-corpus](https://hf.co/datasets/Limour/b-corpus) `emoji/kajiwoto.txt.gz`** — ⚠️ 模型生成；
  6.70M 字符 / 147k 行（Han 0.00）≈ **1.7M token**
  - 取法：解压 → 按行落 jsonl → 作为该桶的额外 `--source`
  - 目录名叫 emoji，内容是 **`user：`/`gpt：` 的英文对话日志**（kajiwoto 是 AI 角色平台），`gpt：` 一侧
    是模型生成，角色扮演味重。⚠️ 整份 6.7M 字符会被 `txt:` 当成一篇并因 `--max_doc_chars` 整篇丢掉，
    必须按行拆；`user：`/`gpt：` 前缀要在清洗时去掉

除 SODA 外，英文桶里**真人写的小集各自带格式噪声**（标点前空格、`_comma_`、`<s>[agent_1]` 标记、角色名
前缀、整条 JSON 字符串），进桶前都要过一遍。

该桶的池子：soda 191M + 真人小集 17M = 208M → b-corpus `v-corpus-en`/`emoji` +52M →
`TinyStoriesV2` +550M ≈ **810M**，其中三分之二是合成语料（SODA 蒸馏自 InstructGPT、TinyStoriesV2 出自
GPT-4）。

`--source` 的顺序是 `TinyStoriesV2` → SODA → 真人小集。桶内顺序**不是**训练顺序（`TokenMixer` 对整桶的
窗口做 shuffle 采样），它只决定两件事：cap 卡住时先读谁——cap 低于 `TinyStoriesV2` 自己的 550M，排在它
后面的 SODA 与小集一个 token 都进不来，所以 `--max_tokens` 要放到 ≈810M 之上；以及 `val.bin` 取自谁——
`prepare_data` 把**第一个源**的头 `--val_tokens` 个 token 写进 `val.bin`，`train.py` 按桶用它算 ppl，
因此该桶的验证集是纯 `TinyStoriesV2` 故事，ppl 反映的是简单英文叙事的拟合程度。

### 1.3 fiction — 小说 / 网文 / 剧本对白

- **[wdndev/webnovel-chinese](https://hf.co/datasets/wdndev/webnovel-chinese)** — 网文；600k 章 / 3.2 GB
  （实测 1.68B 字符 ≈ 1.18B token）
  - 取法：`hf:wdndev/webnovel-chinese:text:default:train`
  - 该桶主力（原本是池子的全部）。字段干净（`title`/`chapter`/`text`），parquet 分支就是小说的全量池，
    一遍读完；apache-2.0
- **[Miaowuawa/ChineseNovels](https://hf.co/datasets/Miaowuawa/ChineseNovels)** — 网文；2,292 本 / 623 MB
  parquet，抽样 300 本中位 22.3 万字符（Han 0.80）→ 全量 ≈755M 字符 ≈ **529M token**；`--max_doc_chars`
  默认 200000 会按篇截断，**56% 的书被砍**，截断后 ≈363M 字符 ≈ **254M token**
  - 取法：`hf:Miaowuawa/ChineseNovels:-:default:train`
  - 同人/耽美占比高，单列结构（viewer 打不开，用 `-` 取整条记录）。⚠️ **一整本一篇文档**，进桶前应拆章，
    否则每本只进前 200k 字符；token 数是抽样外推，真实值以 `prepare_data` 的 `train_tokens` 为准
- **[mrzjy/Chinese_interactive_novels_3k](https://hf.co/datasets/mrzjy/Chinese_interactive_novels_3k)** —
  互动小说【流行文化】；3,534 部 / parquet 362.6 MB，抽样 30 部均 60.3k 字符（中位 31.3k，长尾）→
  ≈213M 字符 ≈ **149M token**
  - 取法：`parquet:hf://datasets/mrzjy/Chinese_interactive_novels_3k@refs/convert/parquet/default/train/*.parquet:book_chapter`
  - b-corpus `互动小说/**` 的原始集（那里打成 21 个 7z 分卷，直接读 Hub 这份更省事）。`book_chapter` →
    `chapter_title` + `content` 列表（`content_tag`/`role`/`content`），同人与互动小说为主。⚠️ 卡片自标
    **uncleaned**；字段表达式清不掉 `content_tag`（`narration`/`dialog`）与 `role`，它们会被 `_to_text`
    拼进正文，落盘时要只取 `content`；apache-2.0 是上游标注，内容是二次创作
- **[Limour/b-corpus](https://hf.co/datasets/Limour/b-corpus) `近现代小说/**`** — 出版小说【流行文化】；
  43 个 `.txt.gz` / 15.0 MB，抽样 0.78 字符/压缩字节 → ≈11.7M 字符 ≈ **8.2M token**
  - 取法：解压 → 按章落 jsonl → 作为该桶的额外 `--source`
  - 张承志《黑骏马》这类近现代小说，简体、叙事完整。⚠️ 上游以 cc-by-nc-sa-4.0 发布，但内容是出版作品
    本身，授权链不清；`txt:` 一个文件一篇文档，长篇要拆章
- **[Limour/b-corpus](https://hf.co/datasets/Limour/b-corpus) `翻译名著/**`** — 翻译小说【流行文化】；
  18 个 `.txt.gz` / 9.2 MB，抽样 0.82 字符/压缩字节 → ≈7.5M 字符 ≈ **5.3M token**
  - 取法：同上
  - 《小王子》《一千零一夜》这类中文译本，叙事文本；授权问题同 `近现代小说`
- **[Limour/b-corpus](https://hf.co/datasets/Limour/b-corpus) `轻小说/**`** — 轻小说【流行文化】；
  1 个 `.txt.gz` / 0.1 MB ≈ 0.1M 字符 ≈ **0.07M token**
  - 取法：同上
  - 《可塑性记忆 Heartfelt Thanks》，量小，随该桶一起进

该桶原本的 1.176B token 是 `wdndev/webnovel-chinese` parquet 分支的全量（网文，一遍读完）；b-corpus 再补
`互动小说`（≈149M，同人、未清洗）与 `近现代小说`/`翻译名著`/`轻小说`（≈14M，出版作品、授权链不清），
池子 ≈1.34B；`Miaowuawa/ChineseNovels` 再补 ≈254M token（整本一篇，不拆章就只进每本前 200k 字符），
池子 ≈1.59B。真公版书（b-corpus `古籍`）偏文言文，是风格污染，不进该桶。

### 1.4 long — 长文 / 非对话文本

- **[wangrui6/Zhihu-KOL](https://hf.co/datasets/wangrui6/Zhihu-KOL)** — 长文；1.0M 条 / 1.5 GB parquet
  （实测 0.72B 字符 ≈ 0.51B token）
  - 取法：`hf:wangrui6/Zhihu-KOL:RESPONSE:default:train`
  - 真人类写的**长回答**（实测 600–6700 字），不属于 IM 闲聊，放进该桶补篇章连贯性；原本是该桶唯一的源，
    一遍读完；无 license 声明
- **[Limour/b-corpus](https://hf.co/datasets/Limour/b-corpus) `提高逼格/history_24/**`** — 白话史书
  【流行文化】；38 个 `.txt.gz` / 26.9 MB，抽样 0.75 字符/压缩字节 → ≈20.1M 字符 ≈ **14.1M token**
  - 取法：解压 → 按章落 jsonl → 作为该桶的额外 `--source`
  - 白话版二十四史（《白话北齐书》等），现代汉语长篇叙事，是该桶除知乎之外最大的一份。⚠️ `txt:` 一个
    文件一篇文档，要拆章
- **[Limour/b-corpus](https://hf.co/datasets/Limour/b-corpus) `畅销书/**`** — 非虚构【流行文化】；
  4 个 `.txt.gz` / 1.1 MB，抽样 0.80 字符/压缩字节 → ≈0.9M 字符 ≈ **0.6M token**
  - 取法：同上
  - 《人类简史》《追风筝的人》这类叙事型非虚构。⚠️ 出版作品，授权链不清
- **[Limour/b-corpus](https://hf.co/datasets/Limour/b-corpus) `散文集/**`** — 散文【流行文化】；
  5 个 `.txt.gz` / 1.1 MB，抽样 0.75 字符/压缩字节 → ≈0.8M 字符 ≈ **0.6M token**
  - 取法：同上
  - 余秋雨《中国文脉》这类散文，篇章连贯、书面语，补该桶文体。⚠️ 授权同上
- **[Limour/b-corpus](https://hf.co/datasets/Limour/b-corpus) `古籍/**`** — 古籍（**慎用**）【流行文化】；
  16 个 `.txt.gz` / 11.4 MB，抽样 0.75 字符/压缩字节 → ≈8.5M 字符 ≈ **6.0M token**
  - 取法：同上
  - 《史记》《古文观止》《人间词话》原文，文言与古汉语语法会渗进生成，要用先筛白话子集

### 1.5 wiki — 通识 / 百科 / 常识

- **[lars1234/baidu-baike-dataset](https://hf.co/datasets/lars1234/baidu-baike-dataset)** — 通识；
  parquet 分支 1.7M 条 / 3.2 GB（原 json 5.63M 条 / 16.8 GB）
  - 取法：`parquet:hf://datasets/lars1234/baidu-baike-dataset@refs/convert/parquet/default/partial-train/*.parquet:-`
  - 该桶主力。**简体、现代白话**的百科条目，实测信息密度很高：**中位条目仅 130 字**（`title+summary`
    中位 78 字），比论述型长条目更贴近"一句话说清一个事实"的形态；parquet 分支实测 1.73M 条 /
    1.92B 字符 ≈ **1.34B token**，按池子全量读一遍（它按"常见词 → 冷门实体"排序，尾部是冷门实体）。
    字段 `title`/`summary`/`sections`/`tags`/`url`，`-` 取整条（自动拼 sections），要高事实密度用
    `title+summary`。⚠️ 语料卡未声明 license，爬取年份约 2019；**优先走 parquet 分支**（3.2 GB），`hf:`
    直连会拉原始的 16.8 GB 单文件 json
- **[mrzjy/chinese_moegirl_wiki_corpus_raw](https://hf.co/datasets/mrzjy/chinese_moegirl_wiki_corpus_raw)** —
  通识【流行文化】；131.5k 条 / 2 个 parquet 分片共 301 MB，合计 ≈291M 字符 ≈ **2 亿 token 量级
  （清洗前）**
  - 取法：`hf:mrzjy/chinese_moegirl_wiki_corpus_raw:text:default:train`（或
    `parquet:hf://datasets/mrzjy/chinese_moegirl_wiki_corpus_raw@refs/convert/parquet/default/train/*.parquet:text`）；
    原始 wikitext 不能直接进桶，剥完模板/表格/注释后落成 `jsonl:data/wiki/*.jsonl`
  - 萌娘百科 20230814 dump，ACG 梗与术语的密度是这批里最高的，给"网感"补事实性内容；语料卡标
    cc-by-4.0，但内容是站点 dump，对外发布前按萌娘百科自身的著作权信息再核对一次。⚠️ 是**原始
    wikitext**：较小的分片（9,765 条 / 26.0M 字符）实测 99.6% 的条目含 `[[`、91.5% 含 `{{`、9.0% 含
    `<!--`、88% 含 `'''`，另有 14.2% 带表格语法，Han 比例只有 0.34，中位 1736 字符、最长 233k 字符。进桶
    前必须剥模板/表格/注释、保留正文与 `'''强调'''`，并按 `--max_doc_chars` 截断——超过默认 200000 的长页
    只进前 200k 字符。清洗后 Han 比例会显著回升，token 数要重测，不能直接按 291M 字符配比

**"通识内容会不会太少"的答案不在数据侧**：116.65M 参数能存下的事实本来就有限，
超过容量后不同事实还会互相干扰；要动手先补知识评测，而不是继续加百科（[training.md §2](training.md)）。

### 1.6 roleplay — 人设卡 / 多轮扮演

- **[shibing624/roleplay-zh-sharegpt-gpt4-data](https://hf.co/datasets/shibing624/roleplay-zh-sharegpt-gpt4-data)**
  — 角色扮演；`evol-gpt4` 档 2.8k 条 / 2.7 MB ≈ 1.5M token
  - 取法：`hf:shibing624/roleplay-zh-sharegpt-gpt4-data:-:sharegpt_formatted_data-evol-gpt4:train`
  - 带 `system_prompt` 人设，AI 生成，助理腔要过滤
- **[Johnson8187/role-play-chinese](https://hf.co/datasets/Johnson8187/role-play-chinese)** — 角色扮演；
  22.4k 条 / 9.0 MB ≈ 10.3M token
  - 取法：`hf:Johnson8187/role-play-chinese:-:default:train`
  - `instruction`/`input`/`output`/`system`，人设齐全。AI 生成，同样要过滤
- **[LooksJuicy/Chinese-Roleplay-SingleTurn](https://hf.co/datasets/LooksJuicy/Chinese-Roleplay-SingleTurn)**
  — 角色扮演【流行文化】；7,593 条 / **4.66M 字符 ≈ 3.3M token**
  - 取法：`hf:LooksJuicy/Chinese-Roleplay-SingleTurn:-:default:train`
  - 作品角色的官方设定 + 单轮扮演回复，是这批里最贴"角色扮演"定位的一份；apache-2.0。实测 `instruction`
    （人设）中位 **514** 字、`input`（用户）中位 8 字、`output`（角色回复，带 `（动作描写）`）中位 100 字、
    p90 156 字，Han 比例 0.87。⚠️ `instruction` **不在** `normalize_messages` 的 `_PERSONA` 里（只认
    `system_prompt`/`persona`/`personality`/`system`），所以整段人设会拼进 **user 轮**、每轮重复；单条
    600–1000 字，`--seq_len 1024` 下占掉一半以上上下文。⚠️ 数据文件是 `Roleplay_Anime_xhj_alpaca.json`
    （单文件 JSON 数组，不是 JSONL），`hf:` 能直接流式读，但要落盘成 `jsonl:` 得先转格式
- **[Limour/b-corpus](https://hf.co/datasets/Limour/b-corpus) `q-corpus/**`** — 角色场景【流行文化】；
  61 个 `.txt.gz` / 3.8 MB，抽样 0.82 字符/压缩字节 → ≈3.1M 字符 ≈ **2.2M token**
  - 取法：解压 → 按场景落 jsonl → 作为该桶的额外 `--source`
  - 情境化的角色扮演文本（`Choice_选择`、`侠女` 等），第二人称场景 + 对白，语气贴近该桶。⚠️ 内容是 R18，
    要逐篇筛；`txt:` 一个文件一篇文档，要按场景拆

### 1.7 roleplay_char — 小说 / 影视角色的单轮问答

- **[silk-road/ChatHaruhi-54K-Role-Playing-Dialogue](https://hf.co/datasets/silk-road/ChatHaruhi-54K-Role-Playing-Dialogue)**
  — 角色扮演；54.7k 条 / 25.9 MB，对白占 51%（中文侧 ~2.0M token、英文侧 ~1.5M）
  - 取法：`parquet:hf://datasets/silk-road/ChatHaruhi-54K-Role-Playing-Dialogue@refs/convert/parquet/default/train/*.parquet:user_question+agent_response`
  - 小说/影视角色的单轮扮演问答，字段 `user_role`/`user_question`/`agent_role`/`agent_response`/
    `question_source`/`more_dialogues`。回复中位 57 字、p90 211 字，长度贴近 IM；中文占 69.3%，其余是
    Harry Potter / Sheldon 这类英文角色（按 `agent_role_name_en` 可分出去）。中文侧 64% 的问题是模型生成的
    （`question_source=synthesized`），回复是古白话/武侠腔并带 `「」`。字段名不在 `normalize_messages` 的
    别名表里，**只进预训练文本桶**，SFT 需先转成 `messages`
- **[Limour/b-corpus](https://hf.co/datasets/Limour/b-corpus) `ChatHaruhi/**`** — 角色扮演【流行文化】；
  38 个 `.txt.gz` / 2.46 MB，实测 4.00M 字符（Han 0.26）≈ **1.6M token**（中文侧 27 个文件 ~1.0M +
  英文侧 11 个 ~0.6M）
  - 取法：解压 → 按行落 jsonl → 作为该桶的额外 `--source`
  - 按角色切好的语料（`ChatHaruhi/RolePlaying/令狐冲.txt.gz`、`ChatHaruhi/Waifu/艾希.txt.gz`），首行是
    `system：I want you to act like …`，其后每行 `角色：「台词」` 或 `旁白：…`。⚠️ 与
    `silk-road/ChatHaruhi-54K` 同源；同一目录里英文侧（Harry Potter、Sheldon）与中文侧（金庸、原神）混放，
    要按 Han 比例分语言后再进桶；首行 system 与行尾残留的 `\` 要在清洗时去掉

### 1.8 roleplay_catgirl — 猫娘人设

- **[kxdw2580/catgirl-dataset](https://hf.co/datasets/kxdw2580/catgirl-dataset)** — 猫娘；全仓 6.07 MB，
  取 `v2/common-v2` 682 KB + `v2/create-v2` 377 KB + `v2/create-v2-next-nothink` 65 KB ≈ **0.15M token**
  - 取法：先 `hf download` 再落成本地 jsonl：`jsonl:data/roleplay_catgirl/*.jsonl:text`
  - 猫娘女仆人设，Alpaca 形状（`instruction`/`input`/`output`/`history`），问题取自
    `Chinese-DeepSeek-R1-Distill-data-110k-SFT` 与 Muice-Dataset 并筛到初中及以下的日常询问。⚠️ 三个
    前置：**gated**（数据文件 401，需先接受条款并带上有权限的 token；README 是公开的）、**文件按
    LLaMA-Factory 的 Alpaca 惯例存放，README 里的样例是 JSON 数组**（若是数组，`jsonl:`/`json:` 的逐行
    解析读不了，落盘时要转成 JSONL）、**思维链内联在 `output` 里**（`<think>…</think>`，字段表达式清不掉，
    只能在文本层去掉；实测不清会让它落在被计损失的目标上）。作者明确不用的：v1（自指回答会露出"规则/
    身份"）、`v2/common-v2-extra`（问答基本不相关，1.83 MB 看着最大）、`v2-qwen3` 整个系列（common 子集
    自 06/06 起被标注避免使用，且所有问题都触发长思维链）。`history` 会被 `normalize_messages` 忽略，
    多轮的 `create-next` 因此塌成单轮

## 2. 对话 / 角色扮演 SFT

- **[silver/lccc](https://hf.co/datasets/silver/lccc)** — 主力：人类口吻
  - 取法：`--mm_data "parquet:hf://datasets/silver/lccc@refs/convert/parquet/base/train/*.parquet:dialog"` +
    `--desegment`
  - 每条会话按 `user`/`self` 轮流交替，正好是"被模仿的一方"。想两个方向都学，按
    [training.md §4.1](training.md) 各出一条记录。无 parquet 以外取法（见 §0 约定 1）；文本是分词格式，
    必须 `--desegment`
- **[shibing624/roleplay-zh-sharegpt-gpt4-data](https://hf.co/datasets/shibing624/roleplay-zh-sharegpt-gpt4-data)**
  — 人设多轮
  - 取法：`--mm_data "hf:shibing624/roleplay-zh-sharegpt-gpt4-data:-:sharegpt_formatted_data-evol-gpt4:train"`
  - `system_prompt` 作为首条 `system` 注入且不计损失
- **[Johnson8187/role-play-chinese](https://hf.co/datasets/Johnson8187/role-play-chinese)** — 人设多轮
  - 取法：`--mm_data "hf:Johnson8187/role-play-chinese:-:default:train"`
  - 同上，`instruction`/`input`/`output`/`system` 形状
- **[LooksJuicy/Chinese-Roleplay-SingleTurn](https://hf.co/datasets/LooksJuicy/Chinese-Roleplay-SingleTurn)**
  — 人设单轮【流行文化】
  - 取法：落盘时把 `instruction` 改名 `system_prompt` 再写 jsonl 给 `--mm_data`
  - 作品角色的官方人设 + 扮演回复，直接可用（别名表实测通过：`output`→`self`、`input`→附带在 user 轮）。
    改名后人设进 `system` 且不计损失；不改名则整段人设拼进 user 轮并在每轮重复（见 §1.6）

SFT 的多个来源是**按行数拼接**的（`ChatDataset` 把 `--mm_data` 的各项顺序连成一个 offsets 列表），没有
权重参数：`human_dialog.jsonl, persona.jsonl, refuse.jsonl` 谁的行数多谁就主导。

## 3. 使用前自检

正式 tokenize 之前先用 `@N` 抽一小批看看文本长什么样、字段有没有取错：

```bash
python - <<'PY'
import sys; sys.path.insert(0, "scripts")
from common_io import desegment_cjk, iter_texts
spec = "parquet:hf://datasets/silver/lccc@refs/convert/parquet/base/train/*.parquet:dialog"
for t in iter_texts(spec, max_docs=3):
    print("raw  ", repr(t[:200]))
    print("clean", repr(desegment_cjk(t)[:200])); print("-" * 40)
PY
```

角色映射用仓库自己的 loader 验，不要用 datasets 的 viewer：

```bash
python - <<'PY'
import sys; sys.path.insert(0, "scripts")
from common_io import iter_records
from hvlm.data_core import normalize_messages
spec = "hf:Johnson8187/role-play-chinese:-:default:train"
print(normalize_messages(next(iter_records(spec, max_docs=1))))
PY
```

规模与分片不看文件头，直接问 datasets-server：

```bash
curl -s "https://datasets-server.huggingface.co/size?dataset=mrzjy%2FChinese_interactive_novels_3k"
```

清洗落盘后按正常流程进桶（`v-corpus-zh`、字幕清洗产物都是 `data/chat/` 下的 jsonl）。`--max_tokens` 要放在
该桶池子之上，`train_tokens` 变了以后三段权重再用 `tools/plan_mix.py` 重算：

```bash
python scripts/prepare_data.py --tokenizer tokenizer --desegment \
  --source "jsonl:data/chat/*.jsonl:text" \
  --out data/bin/chat --max_tokens 1100000000 --val_tokens 4000000
```

SFT 数据在落盘阶段会被 `normalize_messages` 过一遍，跑起来会打印
`cached N records (M unusable) -> data/cache/records-xxxx.jsonl`。`M` 很大说明字段或角色选错了，
去 `--mm_cache` 指向的那个文件里看留下的记录对不对。缓存文件可以随便删，下次会重新生成。
