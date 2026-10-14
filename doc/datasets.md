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
| [chenrm/anime-subtitles](https://hf.co/datasets/chenrm/anime-subtitles) | 人类对话 | 67k 部字幕 / 1.3 GB parquet | `hf:chenrm/anime-subtitles:data:default:train` | 动画字幕，短句口语多。⚠️ `data` 字段是**整份 `.ass` 文件**：含 `[Script Info]` 头、样式定义与时间轴，需要先写一步清洗 |
| [Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset](https://hf.co/datasets/Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset) | 人类对话 | 4.2k 段对话 | `hf:Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset:text:default:train` | 日常对话 + 影视对白 + AI 生成混合，自带情绪标签；量小（161 KB parquet），当风格补充。`text` 字段**一行一句**，不是整段会话 |
| [liwu/MNBVC](https://hf.co/datasets/liwu/MNBVC) `forum`/`blog` | 人类对话 | `forum` 等 18 个子集 | 仅加载脚本，无 parquet 分支 | 论坛对白/博客，是扩 chat 桶最对口的补充。要么把 `datasets` 降到 3.x，要么自己落成本地 parquet/jsonl 再 `jsonl:`/`parquet:` |
| [wdndev/webnovel-chinese](https://hf.co/datasets/wdndev/webnovel-chinese) | 网文 | 600k 章 / 3.2 GB（作者称约 9B token） | `hf:wdndev/webnovel-chinese:text:default:train` | 网文，字段干净（`title`/`chapter`/`text`） |
| [Miaowuawa/ChineseNovels](https://hf.co/datasets/Miaowuawa/ChineseNovels) | 网文 | 2.3k 本 / 623 MB | `hf:Miaowuawa/ChineseNovels:-:default:train` | 同人/耽美占比高，单列结构（viewer 打不开，用 `-` 取整条记录） |
| [wangrui6/Zhihu-KOL](https://hf.co/datasets/wangrui6/Zhihu-KOL) | **长文** | 1.0M 条 / 1.5 GB parquet（2.3 GB 文本） | `hf:wangrui6/Zhihu-KOL:RESPONSE:default:train` | 真人类写的**长回答**（实测 600–6700 字），不属于 IM 闲聊，放进长文桶补篇章连贯性；无 license 声明 |
| [opencsg/Fineweb-Edu-Chinese-V2.1](https://hf.co/datasets/opencsg/Fineweb-Edu-Chinese-V2.1) | **长文** | 4_5 档 70 GB / ~46B token；3_4 档 800 GB | `parquet:hf://datasets/opencsg/Fineweb-Edu-Chinese-V2.1@main/4_5/*.parquet:text` | 教育打分过滤过的中文网页正文，apache-2.0，**长文桶首选**。`4_5` 是分数档目录、不是 config，所以必须走 `parquet:hf://`；字段 `text`/`score`/`source`。`hf:` 直连只能拿 `default`（三档混在一起） |
| [HuggingFaceFW/fineweb-2](https://hf.co/datasets/HuggingFaceFW/fineweb-2) | **长文** | `cmn_Hani` 档 | `hf:HuggingFaceFW/fineweb-2:text:cmn_Hani:train` | 通用中文网页正文，去重过、量大，作为长文桶的兜底；不如 Fineweb-Edu 干净 |
| [wikimedia/wikipedia](https://hf.co/datasets/wikimedia/wikipedia) | 通识 | zh 1.4M 篇 / 1.7 GB | `hf:wikimedia/wikipedia:text:20231101.zh:train` | 通识主力之一，字段 `text`，实测单篇近万字，同时也在补长文。⚠️ **繁简混用**，进桶前需统一成简体 |
| [liwu/MNBVC](https://hf.co/datasets/liwu/MNBVC) `wiki`/`news`/`qa` | 通识 | 见仓库 README | 同上（无 parquet 分支） | 和 forum 同一仓库不同 config，处理方式同上 |
| [lars1234/baidu-baike-dataset](https://hf.co/datasets/lars1234/baidu-baike-dataset) | 通识 | parquet 分支 1.7M 条 / 3.2 GB（原 json 5.63M 条 / 16.8 GB） | `parquet:hf://datasets/lars1234/baidu-baike-dataset@refs/convert/parquet/default/partial-train/*.parquet:-` | **简体、现代白话**的百科条目，建议作为通识桶主力。实测信息密度很高：**中位条目仅 130 字**（`title+summary` 中位 78 字），比中文维基（中位 3624 字）更贴近"一句话说清一个事实"的形态；parquet 分支 ~1.3B token，够 435M 配额的三倍。字段 `title`/`summary`/`sections`/`tags`/`url`，`-` 取整条（自动拼 sections），要高事实密度用 `title+summary`。⚠️ 语料卡未声明 license，爬取年份约 2019；**优先走 parquet 分支**（3.2 GB），`hf:` 直连会拉原始的 16.8 GB 单文件 json |
| [wikimedia/wikisource](https://hf.co/datasets/wikimedia/wikisource) `20231201.zh` | 长文（**慎用**） | 265.7k 篇 / 3.3 GB，平均 4573 字 | `parquet:hf://datasets/wikimedia/wikisource@refs/convert/parquet/20231201.zh/train/*.parquet:text` | 公版书/文献（cc-by-sa-3.0）。⚠️ **实测 68% 偏文言文**，对 IM 聊天模型是风格污染——古汉语词汇/语法会渗进生成。默认不用 |
| [shibing624/roleplay-zh-sharegpt-gpt4-data](https://hf.co/datasets/shibing624/roleplay-zh-sharegpt-gpt4-data) | 角色扮演 | 6.6k 条（`evol-gpt4` 档仅 2.8k 条 / 2.7 MB） | `hf:shibing624/roleplay-zh-sharegpt-gpt4-data:-:sharegpt_formatted_data-evol-gpt4:train` | 带 `system_prompt` 人设。AI 生成，助理腔要过滤 |
| [Johnson8187/role-play-chinese](https://hf.co/datasets/Johnson8187/role-play-chinese) | 角色扮演 | 22.4k 条 / 9.0 MB | `hf:Johnson8187/role-play-chinese:-:default:train` | `instruction`/`input`/`output`/`system`，人设齐全。AI 生成，同样要过滤 |
| liwu/MNBVC `code`/`math` | 代码/数学 | — | —  | 跳过。130M 上代码/数学买不到推理泛化，只会稀释对话分布并把风格往助理腔拉 |

权重定完要反算一遍重复遍数：
`权重 × 3e9 ÷ 该桶 meta.json 的 train_tokens`。中文 chat 桶落在 1–1.5 遍、其余桶低于 1 遍是健康区间；
roleplay 这种只有 ~6M token 的桶一旦超过 2 遍，就是在背语料而不是学风格。

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
