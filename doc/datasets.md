# 训练数据集

取数据的语法见 [common_io.py](../scripts/common_io.py) 的模块 docstring：
`hf:<数据集>:<字段>[:<config>[:<split>]]`，字段 `-` 表示整条记录、`a.b` 走嵌套路径、`a+b` 顺序拼接，列表型字段自动用换行拼成一篇文档，末尾 `@N` 限制文档数。

## 1. 预训练

> **约束：**
> 1. `datasets` ≥ 4 已移除加载脚本支持，凡是只带 `*.py` 加载脚本的数据集（LCCC、MNBVC、DailyDialog、
>    EmpatheticDialogues）用 `hf:` 直连都会报 `Dataset scripts are no longer supported`。改走 Hub 自动转换的
>    parquet 分支：`parquet:hf://datasets/<repo>@refs/convert/parquet/<config>/<split>/*.parquet:<field>`。
>    路径里的 `@refs/...` 是 revision，**不是** `@N` 文档上限。凡是"只有加载脚本"的集，下表一律只给
>    parquet 写法。
> 2. `silver/lccc` 的 `dialog` 字段是**空格分词**的（`你 去 那儿 竟然` 这种），所有吃它的步骤都要带
>    `--desegment`。

| 数据集 | 类型 | 规模 | 取法 | 简介 |
|---|---|---|---|---|
| [silver/lccc](https://hf.co/datasets/silver/lccc) | 人类对话 | base 6.8M + large 12M 条 / 1.63 GB parquet | `parquet:hf://datasets/silver/lccc@refs/convert/parquet/base/train/*.parquet:dialog`（large 同理） | 微博闲聊二人会话。`dialog` 是列表，会自动拼成一篇文档；**分词文本，需 `--desegment`**；SFT 也用它 |
| [chenrm/anime-subtitles](https://hf.co/datasets/chenrm/anime-subtitles) | 人类对话 | 67k 部字幕 / 1.3 GB parquet | `jsonl:data/chat/subtitles.jsonl:text`（先 `scripts/clean_ass.py --zh-only --t2s` 落盘，见 [progress.md](progress.md)） | 动画字幕，短句口语多。⚠️ `data` 字段是**整份 `.ass` 文件**：含 `[Script Info]` 头、样式定义与时间轴，必须先清洗。全量 11.43B 原始字符只剩 2.93B 对白，而 `prepare_data` 的 `--max_doc_chars 200000` 之后**真正能进桶的只有 0.61B 字符 ≈ 0.43B token**：1,470 篇超过 200k 字符（最大 67 MB），里面是卡拉OK 逐字行（`佐\n佐\n佐…`），这批文档是待清理的噪声 |
| [Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset](https://hf.co/datasets/Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset) | 人类对话 | 4.2k 段对话 | `hf:Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset:text:default:train` | 日常对话 + 影视对白 + AI 生成混合，自带情绪标签；量小（161 KB parquet），当风格补充。`text` 字段**一行一句**，不是整段会话 |
| [liwu/MNBVC](https://hf.co/datasets/liwu/MNBVC) `forum`/`blog` | 人类对话 | `forum` 等 18 个子集 | 仅加载脚本，无 parquet 分支 | 论坛对白/博客，是扩 chat 桶最对口的补充。要么把 `datasets` 降到 3.x，要么自己落成本地 parquet/jsonl 再 `jsonl:`/`parquet:` |
| [wdndev/webnovel-chinese](https://hf.co/datasets/wdndev/webnovel-chinese) | 网文 | 600k 章 / 3.2 GB（实测 1.68B 字符 ≈ 1.18B token） | `hf:wdndev/webnovel-chinese:text:default:train` | 网文，字段干净（`title`/`chapter`/`text`），parquet 分支就是小说的全量池，一遍读完 |
| [Miaowuawa/ChineseNovels](https://hf.co/datasets/Miaowuawa/ChineseNovels) | 网文 | 2.3k 本 / 623 MB | `hf:Miaowuawa/ChineseNovels:-:default:train` | 同人/耽美占比高，单列结构（viewer 打不开，用 `-` 取整条记录） |
| [wangrui6/Zhihu-KOL](https://hf.co/datasets/wangrui6/Zhihu-KOL) | **长文** | 1.0M 条 / 1.5 GB parquet（实测 0.72B 字符 ≈ 0.51B token） | `hf:wangrui6/Zhihu-KOL:RESPONSE:default:train` | 真人类写的**长回答**（实测 600–6700 字），不属于 IM 闲聊，放进长文桶补篇章连贯性；长文桶就它一个源，一遍读完；无 license 声明 |
| [opencsg/Fineweb-Edu-Chinese-V2.1](https://hf.co/datasets/opencsg/Fineweb-Edu-Chinese-V2.1) | **长文** | 4_5 档 70 GB / ~46B token；3_4 档 800 GB | `parquet:hf://datasets/opencsg/Fineweb-Edu-Chinese-V2.1@main/4_5/*.parquet:text` | 教育打分过滤过的中文网页正文，apache-2.0，长文桶原本的**多样性来源**。⚠️ 不进当前配方：长文桶就是知乎的全部（0.51B token），再往下读只会拿到 `4_5` 档的头几片，而它按来源语料分组、头几片是清一色 `CCI3`（[training.md §5](training.md)）。`4_5` 是分数档目录、不是 config，所以必须走 `parquet:hf://`；字段 `text`/`score`/`source`。`hf:` 直连只能拿 `default`（三档混在一起） |
| [HuggingFaceFW/fineweb-2](https://hf.co/datasets/HuggingFaceFW/fineweb-2) | **长文** | `cmn_Hani` 档 | `hf:HuggingFaceFW/fineweb-2:text:cmn_Hani:train` | 通用中文网页正文，去重过、量大，作为长文桶的兜底；不如 Fineweb-Edu 干净 |
| [wikimedia/wikipedia](https://hf.co/datasets/wikimedia/wikipedia) | 通识 | zh 1.4M 篇 / 1.7 GB（实测 1.05B 字符 ≈ 0.74B token） | `hf:wikimedia/wikipedia:text:20231101.zh:train` | 通识候选，字段 `text`，实测单篇近万字，同时在补长文。⚠️ **繁简混用**，进桶前需统一成简体；事实密度不如百度百科，当前不进配方 |
| [liwu/MNBVC](https://hf.co/datasets/liwu/MNBVC) `wiki`/`news`/`qa` | 通识 | 见仓库 README | 同上（无 parquet 分支） | 和 forum 同一仓库不同 config，处理方式同上 |
| [lars1234/baidu-baike-dataset](https://hf.co/datasets/lars1234/baidu-baike-dataset) | 通识 | parquet 分支 1.7M 条 / 3.2 GB（原 json 5.63M 条 / 16.8 GB） | `parquet:hf://datasets/lars1234/baidu-baike-dataset@refs/convert/parquet/default/partial-train/*.parquet:-` | **简体、现代白话**的百科条目，建议作为通识桶主力。实测信息密度很高：**中位条目仅 130 字**（`title+summary` 中位 78 字），比中文维基（中位 3624 字）更贴近"一句话说清一个事实"的形态；parquet 分支实测 1.73M 条 / 1.92B 字符 ≈ **1.34B token**，配比取它的 90%（顺序截断正好落在常见实体上，砍掉的是冷门实体长尾）。字段 `title`/`summary`/`sections`/`tags`/`url`，`-` 取整条（自动拼 sections），要高事实密度用 `title+summary`。⚠️ 语料卡未声明 license，爬取年份约 2019；**优先走 parquet 分支**（3.2 GB），`hf:` 直连会拉原始的 16.8 GB 单文件 json |
| [wikimedia/wikisource](https://hf.co/datasets/wikimedia/wikisource) `20231201.zh` | 长文（**慎用**） | 265.7k 篇 / 3.3 GB，平均 4573 字 | `parquet:hf://datasets/wikimedia/wikisource@refs/convert/parquet/20231201.zh/train/*.parquet:text` | 公版书/文献（cc-by-sa-3.0）。⚠️ **实测 68% 偏文言文**，对 IM 聊天模型是风格污染——古汉语词汇/语法会渗进生成。默认不用 |
| [shibing624/roleplay-zh-sharegpt-gpt4-data](https://hf.co/datasets/shibing624/roleplay-zh-sharegpt-gpt4-data) | 角色扮演 | 6.6k 条（`evol-gpt4` 档仅 2.8k 条 / 2.7 MB） | `hf:shibing624/roleplay-zh-sharegpt-gpt4-data:-:sharegpt_formatted_data-evol-gpt4:train` | 带 `system_prompt` 人设。AI 生成，助理腔要过滤 |
| [Johnson8187/role-play-chinese](https://hf.co/datasets/Johnson8187/role-play-chinese) | 角色扮演 | 22.4k 条 / 9.0 MB | `hf:Johnson8187/role-play-chinese:-:default:train` | `instruction`/`input`/`output`/`system`，人设齐全。AI 生成，同样要过滤 |
| [silk-road/ChatHaruhi-54K-Role-Playing-Dialogue](https://hf.co/datasets/silk-road/ChatHaruhi-54K-Role-Playing-Dialogue) | 角色扮演 | 54.7k 条 / 25.9 MB，对白占 51%（中文侧 ~2.0M token、英文侧 ~1.5M） | `parquet:hf://datasets/silk-road/ChatHaruhi-54K-Role-Playing-Dialogue@refs/convert/parquet/default/train/*.parquet:user_question+agent_response` | 小说/影视角色的单轮扮演问答，字段 `user_role`/`user_question`/`agent_role`/`agent_response`/`question_source`/`more_dialogues`。回复中位 57 字、p90 211 字，长度贴近 IM；中文占 69.3%，其余是 Harry Potter / Sheldon 这类英文角色（按 `agent_role_name_en` 可分出去）。中文侧 64% 的问题是模型生成的（`question_source=synthesized`），回复是古白话/武侠腔并带 `「」`。字段名不在 `normalize_messages` 的别名表里，**只进预训练文本桶**，SFT 需先转成 `messages` |
| [silk-road/ChatHaruhi-Expand-118K](https://hf.co/datasets/silk-road/ChatHaruhi-Expand-118K) | 角色扮演 | 118.1k 条；parquet 分支只有 `context`+`target` | `parquet:hf://datasets/silk-road/ChatHaruhi-Expand-118K@refs/convert/parquet/default/train/*.parquet:context+target` | 上面那份的超集，但 `context` 是**英文**人设指令（`I want you to act like …`）+ 原文场景，约 3 KB/条，只有 `target` 是那一句对白。只要对白就用 54K 那份 |
| [kxdw2580/catgirl-dataset](https://hf.co/datasets/kxdw2580/catgirl-dataset) | 角色扮演（猫娘） | 全仓 6.07 MB；取 `v2/common-v2` 682 KB + `v2/create-v2` 377 KB + `v2/create-v2-next-nothink` 65 KB ≈ **0.15M token** | 先 `hf download` 再落成本地 jsonl：`jsonl:data/roleplay_catgirl/*.jsonl:text` | 猫娘女仆人设，Alpaca 形状（`instruction`/`input`/`output`/`history`），问题取自 `Chinese-DeepSeek-R1-Distill-data-110k-SFT` 与 Muice-Dataset 并筛到初中及以下的日常询问。⚠️ 三个前置：**gated**（数据文件 401，需先接受条款并带上有权限的 token；README 是公开的）、**文件按 LLaMA-Factory 的 Alpaca 惯例存放，README 里的样例是 JSON 数组**（若是数组，`jsonl:`/`json:` 的逐行解析读不了，落盘时要转成 JSONL）、**思维链内联在 `output` 里**（`<think>…</think>`，字段表达式清不掉，只能在文本层去掉；实测不清会让它落在被计损失的目标上）。作者明确不用的：v1（自指回答会露出"规则/身份"）、`v2/common-v2-extra`（问答基本不相关，1.83 MB 看着最大）、`v2-qwen3` 整个系列（common 子集自 06/06 起被标注避免使用，且所有问题都触发长思维链）。`history` 会被 `normalize_messages` 忽略，多轮的 `create-next` 因此塌成单轮 |
| [cyberlangke/nana-catgirl-wildchat-534K](https://hf.co/datasets/cyberlangke/nana-catgirl-wildchat-534K) | 角色扮演（猫娘·**待清洗**） | 534.7k 条 / 785 MB，75 语 | `parquet:hf://datasets/cyberlangke/nana-catgirl-wildchat-534K@refs/convert/parquet/default/train/*.parquet:query+reply` | 真实用户 query（WildChat 抽取）+ DeepSeek 猫娘回复。**思维链在独立的 `thinking` 列**，只取 `query+reply` 时 parquet 只读这两列，思维链自动不进语料。⚠️ 两个 blocker：`language` 列不可信（中文 query 被标成 Afrikaans，抽样 1000 行里 Han 主导的回复只占 **29.3%**），而 `prepare_data` 没有语言过滤，直接指过去会把七成非中文文本灌进 roleplay 桶；回复中位 237 字、37.5% 带 markdown/加粗。要用得先写清洗脚本筛 Han 比例并压长度 |
| liwu/MNBVC `code`/`math` | 代码/数学 | — | —  | 跳过。130M 上代码/数学买不到推理泛化，只会稀释对话分布并把风格往助理腔拉 |

桶的配额不是手写的，而是从各桶 `meta.json` 的 `train_tokens` 反算：
`该桶的三段课程权重 × 对应阶段 token 之和 ÷ 该桶 train_tokens`
（各段 token 见 [training.md §2](training.md)；`tools/plan_mix.py` 直接把三段 `--data` 串算出来）。
当前配比让每个桶恰好走一遍（wiki 取 90%），所以这个比值应当 ≈1：明显低于 1 说明有语料没被训练
（先看 `prepare_data` 的 `--max_tokens` 有没有卡在池子下面），高于 1 就是在背语料。roleplay 三个桶
合计 15.4M token，任何一桶超过 2 遍就是在背语料而不是学风格。

英文对话：

| 数据集 | 规模 | 人类 | 取法 | 简介 |
|---|---|---|---|---|
| [allenai/soda](https://hf.co/datasets/allenai/soda) | 1.2M 会话 / ~190M token | ⚠️ **合成**（InstructGPT） | `hf:allenai/soda:dialogue:default:train` | 英文桶的主力。语料卡自标 `language_creators:machine-generated`，客服腔/助理腔不明显但确实不是人类语料；取 `dialogue` 字段即可（`speakers` 与它对齐，SFT 里能自动判谁是被模仿方），`narrative` 是场景设定、预训练不用 |
| [li2017dailydialog/daily_dialog](https://hf.co/datasets/li2017dailydialog/daily_dialog) | 13k 会话 | ✅ 人写 | `parquet:hf://datasets/li2017dailydialog/daily_dialog@refs/convert/parquet/default/train/*.parquet:dialog` | 日常口语，短句多，英文闲聊里最干净的一档。⚠️ 实测标点被空格隔开（`going for a few beers after dinner ?`），进桶前要去掉标点前空格。只有加载脚本，`hf:` 直连已失效 |
| [ParlAI/blended_skill_talk](https://hf.co/datasets/ParlAI/blended_skill_talk) | 4.8k 会话 | ✅ 众包 | `hf:ParlAI/blended_skill_talk:free_messages:default:train` | 同时含 persona / 共情 / 知识三种模式，和本项目"人设 + 陪伴"的定位最接近；`personas` 是列表，SFT 里能当 system。量很小 |
| [facebook/empathetic_dialogues](https://hf.co/datasets/facebook/empathetic_dialogues) | 25k 会话 | ✅ 人写 | `parquet:hf://datasets/facebook/empathetic_dialogues@refs/convert/parquet/default/train/*.parquet:prompt+utterance` | cc-by-nc-4.0，情绪陪伴向。**一行一句**，取字段要写 `prompt+utterance`：用 `-` 会把 `conv_id`/`speaker_idx`/情绪标签一起拼进去。⚠️ 逗号被转义成 `_comma_`，要还原。只有加载脚本 |
| [Fishball02/topical-chat](https://hf.co/datasets/Fishball02/topical-chat) | 9.7k 会话 | ✅ MTurk | `hf:Fishball02/topical-chat:-:default:train` | Topical-Chat 的镜像（官方 `Conversational-Reasoning/Topical-Chat` 已加载失败、无 parquet 分支）。两人无预设角色、知识带聊，接近真实网聊。⚠️ 文本带 `<s>[agent_1]…</s><s>[agent_2]…</s>` 标记，要剥掉 |
| [mylesmharrison/cornell-movie-dialog](https://hf.co/datasets/mylesmharrison/cornell-movie-dialog) | 305k 行 | ✅ 剧本 | `parquet:hf://datasets/mylesmharrison/cornell-movie-dialog@refs/convert/parquet/default/train/*.parquet:text` | 影视对白，口语味足、情绪起伏大。⚠️ **每行一句**不是整段会话，且带角色名前缀（`BIANCA   They do not!`）与 `\r\n`，要按角色切分并去掉前缀。官方 `cornell-movie-dialog/cornell_movie_dialog` 无 parquet 分支，已不可用 |
| [thu-coai/esconv](https://hf.co/datasets/thu-coai/esconv) | 1.3k 段对话 | ✅ 人写 | `hf:thu-coai/esconv:text:default:train` | 情感支持对话；量很小。⚠️ `text` 字段实测是**一整条 JSON 字符串**（含 `experience_type` 等键），要么改嵌套字段路径，要么先清洗 |
| [bavard/personachat_truecased](https://hf.co/datasets/bavard/personachat_truecased) | 131k 条 | ✅ 人写 | `parquet:hf://datasets/bavard/personachat_truecased@refs/convert/parquet/full/train/*.parquet:-` | `personality`+`history`+`candidates`，只用于 SFT（喂预训练会因候选重复而大量重复）。⚠️ 只有加载脚本；`hf:` 直连已失效。注意别用 `AlekseyKorshuk/persona-chat` 之类的旧镜像，实测是小写且标点前带空格（`i like to remodel homes .`）的脏版本 |

## 2. 对话 / 角色扮演 SFT

| 数据集 | 用途 | 取法 | 说明 |
|---|---|---|---|
| [silver/lccc](https://hf.co/datasets/silver/lccc) | 主力：人类口吻 | `--mm_data "parquet:hf://datasets/silver/lccc@refs/convert/parquet/base/train/*.parquet:dialog"` + `--desegment` | 每条会话按 `user`/`self` 轮流交替，正好是"被模仿的一方"。想两个方向都学，按 §4.1 各出一条记录。无 parquet 以外取法（见 §1 硬约束 1）；文本是分词格式，必须 `--desegment` |
| [shibing624/roleplay-zh-sharegpt-gpt4-data](https://hf.co/datasets/shibing624/roleplay-zh-sharegpt-gpt4-data) | 人设多轮 | `--mm_data "hf:shibing624/roleplay-zh-sharegpt-gpt4-data:-:sharegpt_formatted_data-evol-gpt4:train"` | `system_prompt` 作为首条 `system` 注入且不计损失 |
| [Johnson8187/role-play-chinese](https://hf.co/datasets/Johnson8187/role-play-chinese) | 人设多轮 | `--mm_data "hf:Johnson8187/role-play-chinese:-:default:train"` | 同上，`instruction`/`input`/`output`/`system` 形状 |
| [facebook/empathetic_dialogues](https://hf.co/datasets/facebook/empathetic_dialogues) | 情绪陪伴 | `--mm_data "parquet:hf://datasets/facebook/empathetic_dialogues@refs/convert/parquet/default/train/*.parquet:-"` | 英文，人写。对应情绪陪伴这条用途；只学 `utterance` 那一侧。只有加载脚本，`hf:` 直连已失效 |
| [li2017dailydialog/daily_dialog](https://hf.co/datasets/li2017dailydialog/daily_dialog) | 日常口语 | `--mm_data "parquet:hf://datasets/li2017dailydialog/daily_dialog@refs/convert/parquet/default/train/*.parquet:dialog"` | 英文日常对话，按 `dialog` 列表交替映射。同样只有加载脚本 |
| [lorinma/Slim-LCCC-zh](https://hf.co/datasets/lorinma/Slim-LCCC-zh) | 闲聊风味种子 | `--mm_data "hf:lorinma/Slim-LCCC-zh:-:default:train"` | LCCC-large 里挑出的 1 万条 casual 闲聊，转成 ShareGPT 格式，专门用来对比"AI 腔"。⚠️ **gated**：需先 `hf auth login` 并接受条款，否则报 `DatasetNotFoundError` |
| [Gryphe/Sonnet3.5-Charcard-Roleplay](https://hf.co/datasets/Gryphe/Sonnet3.5-Charcard-Roleplay) | 角色卡扮演 | `--mm_data "hf:Gryphe/Sonnet3.5-Charcard-Roleplay:-:default:train"` | 英文、AI 生成，助理腔过滤要更狠 |
| [lemonilia/Roleplay-Forums_2023-04](https://hf.co/datasets/lemonilia/Roleplay-Forums_2023-04) | 真实论坛扮演 | `--mm_data "hf:lemonilia/Roleplay-Forums_2023-04:-:default:train"` | 真实人写，但作者注明 HTML 标签间的空格在转换时损坏，需要自己修文本 |

SFT 的多个来源是**按行数拼接**的（`ChatDataset` 把 `--mm_data` 的各项顺序连成一个 offsets 列表），没有
权重参数：`human_dialog.jsonl, persona.jsonl, refuse.jsonl` 谁的行数多谁就主导。

## 3. 视觉对齐

| 数据集 | 内容 | 简介 |
|---|---|---|
| [passing2961/photochat_plus](https://hf.co/datasets/passing2961/photochat_plus) | 真实人—人分享照片聊天 | 贴合"图在对话中的语气"；图片存在数据集里，见下方说明 |
| [friedrichor/PhotoChat_image](https://hf.co/datasets/friedrichor/PhotoChat_image) | PhotoChat 图片侧 | ⚠️ 同上 |
| [Mobiusi/Chat-Record-Image-Dataset](https://hf.co/datasets/Mobiusi/Chat-Record-Image-Dataset) | 手机聊天截图 + OCR 文本 | ⚠️ 对应"聊天截图"场景。|

**阻塞**：这些集的图片以 PIL/bytes 存在数据集内部，而 `--mm_data` 目前只认 `--image_root` 下的相对路径，所以它们还不能像预训练集那样一条命令指过去。这一步（落盘时把图一并导出到缓存目录、再设`--image_root`）还没做；需要时先在本地把图导成 `data/mm/images/...` 再写 jsonl。

## 4. 使用前自检

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

SFT 数据在落盘阶段会被 `normalize_messages` 过一遍，跑起来会打印
`cached N records (M unusable) -> data/cache/records-xxxx.jsonl`。`M` 很大说明字段或角色选错了，
去 `--mm_cache` 指向的那个文件里看留下的记录对不对。缓存文件可以随便删，下次会重新生成。
