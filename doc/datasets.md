# 训练数据集

取数据的语法见 [common_io.py](../scripts/common_io.py) 的模块 docstring：
`hf:<数据集>:<字段>[:<config>[:<split>]]`，字段 `-` 表示整条记录、`a.b` 走嵌套路径、`a+b` 顺序拼接，列表型字段自动用换行拼成一篇文档，末尾 `@N` 限制文档数。

## 1. 预训练

> **两条硬约束，写在前面：**
> 1. `datasets` ≥ 4 已移除加载脚本支持，凡是只带 `*.py` 加载脚本的数据集（LCCC、MNBVC、DailyDialog、
>    EmpatheticDialogues）用 `hf:` 直连都会报 `Dataset scripts are no longer supported`。改走 Hub 自动转换的
>    parquet 分支：`parquet:hf://datasets/<repo>@refs/convert/parquet/<config>/<split>/*.parquet:<field>`。
>    路径里的 `@refs/...` 是 revision，**不是** `@N` 文档上限。
> 2. `silver/lccc` 的 `dialog` 字段是**空格分词**的（`你 去 那儿 竟然`）。不加 `--desegment` 直接训，
>    模型会学着在汉字之间输出空格。所有吃 LCCC 的步骤都要带这个开关。

| 数据集 | 类型 | 规模 | 取法 | 简介 |
|---|---|---|---|---|
| [silver/lccc](https://hf.co/datasets/silver/lccc) | 人类对话 | base 6.8M + large 12M 条 / 1.63 GB parquet | `parquet:hf://datasets/silver/lccc@refs/convert/parquet/base/train/*.parquet:dialog`（large 同理） | 微博闲聊二人会话。`dialog` 是列表，会自动拼成一篇文档；**分词文本，需 `--desegment`**；SFT 也用它 |
| [chenrm/anime-subtitles](https://hf.co/datasets/chenrm/anime-subtitles) | 人类对话 | 67k 部字幕 / 1.3 GB parquet | `hf:chenrm/anime-subtitles:data:default:train` | 动画字幕，短句口语多。⚠️ `data` 字段是**整份 `.ass` 文件**：含 `[Script Info]` 头、样式定义与时间轴，喂之前必须剥掉，否则大量训练信号花在格式标记上 |
| [Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset](https://hf.co/datasets/Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset) | 人类对话 | 4.2k 段对话 | `hf:Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset:text:default:train` | 日常对话 + 影视对白 + AI 生成混合，自带情绪标签；量小（161 KB parquet），当风格补充。`text` 字段**一行一句**，不是整段会话 |
| [liwu/MNBVC](https://hf.co/datasets/liwu/MNBVC) `forum`/`blog` | 人类对话 | `forum` 等 18 个子集 | 仅加载脚本，`hf:` 直连已失效 | 论坛对白/博客，是扩 chat 桶最对口的补充。要么把 `datasets` 降到 3.x，要么自己落成本地 parquet/jsonl 再 `jsonl:`/`parquet:` |
| [wdndev/webnovel-chinese](https://hf.co/datasets/wdndev/webnovel-chinese) | 网文 | 600k 章 / 3.2 GB（作者称约 9B token） | `hf:wdndev/webnovel-chinese:text:default:train` | 网文，字段干净（`title`/`chapter`/`text`） |
| [Miaowuawa/ChineseNovels](https://hf.co/datasets/Miaowuawa/ChineseNovels) | 网文 | 2.3k 本 / 623 MB | `hf:Miaowuawa/ChineseNovels:-:default:train` | 同人/耽美占比高，单列结构（viewer 打不开，用 `-` 取整条记录） |
| [wangrui6/Zhihu-KOL](https://hf.co/datasets/wangrui6/Zhihu-KOL) | **长文** | 1.0M 条 / 1.5 GB parquet（2.3 GB 文本） | `hf:wangrui6/Zhihu-KOL:RESPONSE:default:train` | 真人类写的**长回答**（实测 600–6700 字），不属于 IM 闲聊，放进长文桶补篇章连贯性；无 license 声明 |
| [opencsg/Fineweb-Edu-Chinese-V2.1](https://hf.co/datasets/opencsg/Fineweb-Edu-Chinese-V2.1) | **长文** | 4_5 档 70 GB / ~46B token；3_4 档 800 GB | `parquet:hf://datasets/opencsg/Fineweb-Edu-Chinese-V2.1@main/4_5/*.parquet:text` | 教育打分过滤过的中文网页正文，apache-2.0，**长文桶首选**。`4_5` 是分数档目录、不是 config，所以必须走 `parquet:hf://`；字段 `text`/`score`/`source`。`hf:` 直连只能拿 `default`（三档混在一起） |
| [HuggingFaceFW/fineweb-2](https://hf.co/datasets/HuggingFaceFW/fineweb-2) | **长文** | `cmn_Hani` 档 | `hf:HuggingFaceFW/fineweb-2:text:cmn_Hani:train` | 通用中文网页正文，去重过、量大，作为长文桶的兜底；不如 Fineweb-Edu 干净 |
| [wikimedia/wikipedia](https://hf.co/datasets/wikimedia/wikipedia) | 通识 | zh 1.4M 篇 / 1.7 GB | `hf:wikimedia/wikipedia:text:20231101.zh:train` | 通识主力之一，字段 `text`，实测单篇近万字，同时也在补长文。⚠️ **实测 400 篇里 366 篇繁简混用**（同一段里 `數量` 和 `结构` 并存），对简体 IM 模型会诱导输出繁体字——进桶前应过一遍繁→简统一（opencc） |
| [liwu/MNBVC](https://hf.co/datasets/liwu/MNBVC) `wiki`/`news`/`qa` | 通识 | 见仓库 README | 仅加载脚本，`hf:` 直连已失效 | 和 forum 同一仓库不同 config，处理方式同上 |
| [lars1234/baidu-baike-dataset](https://hf.co/datasets/lars1234/baidu-baike-dataset) | 通识 | parquet 分支 1.7M 条 / 3.2 GB（原 json 5.63M 条 / 16.8 GB） | `parquet:hf://datasets/lars1234/baidu-baike-dataset@refs/convert/parquet/default/partial-train/*.parquet:-` | **简体、现代白话**的百科条目，建议作为通识桶主力。实测信息密度很高：**中位条目仅 130 字**（`title+summary` 中位 78 字），比中文维基（中位 3624 字）更贴近"一句话说清一个事实"的形态；parquet 分支 ~1.3B token，够 435M 配额的三倍。字段 `title`/`summary`/`sections`/`tags`/`url`，`-` 取整条（自动拼 sections），要高事实密度用 `title+summary`。⚠️ 语料卡未声明 license，爬取年份约 2019；**优先走 parquet 分支**（3.2 GB），`hf:` 直连会拉原始的 16.8 GB 单文件 json |
| [wikimedia/wikisource](https://hf.co/datasets/wikimedia/wikisource) `20231201.zh` | 长文（**慎用**） | 265.7k 篇 / 3.3 GB，平均 4573 字 | `parquet:hf://datasets/wikimedia/wikisource@refs/convert/parquet/20231201.zh/train/*.parquet:text` | 真正的公版书/文献（cc-by-sa-3.0）。⚠️ **实测 68% 偏文言文**（350/512；样本是《中華民國憲法》《南京條約》《正氣歌》），对 IM 聊天模型是明确的风格污染——古汉语词汇/语法会渗进生成。**默认不用**；真要用必须先筛白话子集 |
| [shibing624/roleplay-zh-sharegpt-gpt4-data](https://hf.co/datasets/shibing624/roleplay-zh-sharegpt-gpt4-data) | 角色扮演 | 6.6k 条（`evol-gpt4` 档仅 2.8k 条 / 2.7 MB） | `hf:shibing624/roleplay-zh-sharegpt-gpt4-data:-:sharegpt_formatted_data-evol-gpt4:train` | 带 `system_prompt` 人设。AI 生成，助理腔要过滤 |
| [Johnson8187/role-play-chinese](https://hf.co/datasets/Johnson8187/role-play-chinese) | 角色扮演 | 22.4k 条 / 9.0 MB | `hf:Johnson8187/role-play-chinese:-:default:train` | `instruction`/`input`/`output`/`system`，人设齐全。AI 生成，同样要过滤 |
| liwu/MNBVC `code`/`math` | 代码/数学 | — | —  | 跳过。130M 上代码/数学买不到推理泛化，只会稀释对话分布并把风格往助理腔拉；"听懂代码话题"靠 §2 的越界拒答覆盖即可 |

整个角色扮演类的公开语料加起来只有 **~6M token**（2.7 MB + 9.0 MB parquet），这是 1% 以上权重必然重复
五遍以上的量级——它不是预训练语料，是 SFT 语料。所以 §2 只给它 0.5%，作用是让 `system` 人设这个格式
有意义；人设能力请放在 §2 里学，想提高这个权重只能自己把小说/剧本改造成多轮扮演。

按 §2 配比拼起来（权重按 token 算，不是按文件大小；权重会被归一化，不必凑成 1）：

```bash
# 中文对话 45% + 英文对话 5% + 网文 20% + 长文 15% + 通识 14.5% + 角色扮演 0.5%
python scripts/train.py --stage pretrain --tokenizer tokenizer --out_dir runs/pretrain --tokens 3e9 --resume \
    --data data/bin/chat:0.45,data/bin/en_chat:0.05,data/bin/fiction:0.20,data/bin/long:0.15,data/bin/wiki:0.145,data/bin/roleplay:0.005
```

权重定完要反算一遍重复遍数：`权重 × 3e9 ÷ 该桶 meta.json 的 train_tokens`。中文 chat 桶落在 1–1.5 遍、
其余桶低于 1 遍是健康区间；roleplay 这种只有 ~6M token 的桶一旦超过 2 遍，就是在背语料而不是学风格。

英文对话：

> **先看量**：真人写的英文闲聊源加总只有 ~14M token（DailyDialog ~1.2M + EmpatheticDialogues ~0.9M +
> BlendedSkillTalk ~1.2M + Topical-Chat ~5M + Cornell ~3.7M + PersonaChat ~2M），和 roleplay 桶一个量级，
> 单靠它们撑不起一个桶。能上量的只有 **SODA**（实测 1.2M 会话、平均 632 字符对话、~190M token），所以
> §2 给了英文一个 5% 的桶，主要是 SODA 在扛，真人小集并进去补风格多样性。
>
> **SODA 的定性**：语料卡标注 `language_creators:machine-generated`，对话由 InstructGPT 在常识三元组上
> 蒸馏而来，属于合成数据，不是人类—人类语料。工程上人工抽查下来助理腔不算明显、格式干净（20k 行抽样里
> 含方括号的只有 0.25%、认不出标记的 1 行），所以**项目决定启用**——但它与 §0"用人类对话调风格"的原则
> 是偏离的，因此权重压在 5% 封顶，并且要当作合成数据盯：单独看它对回复长度分布和助理腔模式词的影响，
> 一旦中文闲聊质量回退，优先降它的权重。英文在这个项目里的定位是**防中英混输/拉丁字母退化**，不是提能力。
>
> **英文百科不在这里，也不该为了"补通识"加进来**：`wikimedia/wikipedia` 的 `20231101.en`（6.4M 篇 /
> 11.6GB）随时可取，但它教的是英文事实，130M 参数上不存在可靠的中英跨语言迁移；中文通识的可用量本身
> 已经是配额的 8 倍（见 §2 的"通识"一条）。要用它，理由只能是英文语言建模，并且得从英文桶那 5% 里出。

| 数据集 | 规模 | 人写? | 取法 | 简介 |
|---|---|---|---|---|
| [allenai/soda](https://hf.co/datasets/allenai/soda) | 1.2M 会话 / ~190M token | ⚠️ **合成**（InstructGPT） | `hf:allenai/soda:dialogue:default:train` | 英文桶的主力。语料卡自标 `language_creators:machine-generated`，客服腔/助理腔不明显但确实不是人类语料；取 `dialogue` 字段即可（`speakers` 与它对齐，SFT 里能自动判谁是被模仿方），`narrative` 是场景设定、预训练不用 |
| [li2017dailydialog/daily_dialog](https://hf.co/datasets/li2017dailydialog/daily_dialog) | 13k 会话 | ✅ 人写 | `parquet:hf://datasets/li2017dailydialog/daily_dialog@refs/convert/parquet/default/train/*.parquet:dialog` | 日常口语，短句多，英文闲聊里最干净的一档。⚠️ 实测标点被空格隔开（`going for a few beers after dinner ?`），进桶前要去掉标点前空格。只有加载脚本，`hf:` 直连已失效 |
| [ParlAI/blended_skill_talk](https://hf.co/datasets/ParlAI/blended_skill_talk) | 4.8k 会话 | ✅ 众包 | `hf:ParlAI/blended_skill_talk:free_messages:default:train` | 同时含 persona / 共情 / 知识三种模式，和本项目"人设 + 陪伴"的定位最接近；`personas` 是列表，SFT 里能当 system。量很小 |
| [facebook/empathetic_dialogues](https://hf.co/datasets/facebook/empathetic_dialogues) | 25k 会话 | ✅ 人写 | `parquet:hf://datasets/facebook/empathetic_dialogues@refs/convert/parquet/default/train/*.parquet:prompt+utterance` | cc-by-nc-4.0，情绪陪伴向。**一行一句**，取字段要写 `prompt+utterance`：用 `-` 会把 `conv_id`/`speaker_idx`/情绪标签一起拼进去。⚠️ 逗号被转义成 `_comma_`，要还原。只有加载脚本 |
| [Fishball02/topical-chat](https://hf.co/datasets/Fishball02/topical-chat) | 9.7k 会话 | ✅ MTurk | `hf:Fishball02/topical-chat:-:default:train` | Topical-Chat 的镜像（官方 `Conversational-Reasoning/Topical-Chat` 已加载失败、无 parquet 分支）。两人无预设角色、知识带聊，接近真实网聊。⚠️ 文本带 `<s>[agent_1]…</s><s>[agent_2]…</s>` 标记，要剥掉 |
| [mylesmharrison/cornell-movie-dialog](https://hf.co/datasets/mylesmharrison/cornell-movie-dialog) | 305k 行 | ✅ 剧本 | `parquet:hf://datasets/mylesmharrison/cornell-movie-dialog@refs/convert/parquet/default/train/*.parquet:text` | 影视对白，口语味足、情绪起伏大。⚠️ **每行一句**不是整段会话，且带角色名前缀（`BIANCA   They do not!`）与 `\r\n`，要按角色切分并去掉前缀。官方 `cornell-movie-dialog/cornell_movie_dialog` 无 parquet 分支，已不可用 |
| [thu-coai/esconv](https://hf.co/datasets/thu-coai/esconv) | 1.3k 段对话 | ✅ 人写 | `hf:thu-coai/esconv:text:default:train` | 情感支持对话；量很小。⚠️ `text` 字段实测是**一整条 JSON 字符串**（含 `experience_type` 等键），要么改嵌套字段路径，要么先清洗 |
| [bavard/personachat_truecased](https://hf.co/datasets/bavard/personachat_truecased) | 131k 条 | ✅ 人写 | `parquet:hf://datasets/bavard/personachat_truecased@refs/convert/parquet/full/train/*.parquet:-` | `personality`+`history`+`candidates`，只用于 SFT（喂预训练会因候选重复而大量重复）。⚠️ 只有加载脚本；`hf:` 直连已失效。注意别用 `AlekseyKorshuk/persona-chat` 之类的旧镜像，实测是小写且标点前带空格（`i like to remodel homes .`）的脏版本 |
| [allenai/prosocial-dialog](https://hf.co/datasets/allenai/prosocial-dialog) | 120k 条 | 人机协作 | — | ❌ 不适合：内容是"如何回应冒犯性言论"的安全改写，实测还带 `__needs` 之类标注残留，分布与闲聊差太远 |

## 2. 对话 / 角色扮演 SFT

取法和预训练一样，`--mm_data` 直接收 `hf:` 来源（会先落一份缓存 jsonl，见 training.md §5）。

| 数据集 | 用途 | 取法 | 说明 |
|---|---|---|---|
| [silver/lccc](https://hf.co/datasets/silver/lccc) | 主力：人类口吻 | `--mm_data "parquet:hf://datasets/silver/lccc@refs/convert/parquet/base/train/*.parquet:dialog"` + `--desegment` | 每条会话按 `user`/`self` 轮流交替，正好是"被模仿的一方"。想两个方向都学，按 §4.1 各出一条记录。只有加载脚本，`hf:` 直连已失效；文本还是分词格式，必须 `--desegment` |
| [shibing624/roleplay-zh-sharegpt-gpt4-data](https://hf.co/datasets/shibing624/roleplay-zh-sharegpt-gpt4-data) | 人设多轮 | `--mm_data "hf:shibing624/roleplay-zh-sharegpt-gpt4-data:-:sharegpt_formatted_data-evol-gpt4:train"` | `system_prompt` 作为首条 `system` 注入且不计损失 |
| [Johnson8187/role-play-chinese](https://hf.co/datasets/Johnson8187/role-play-chinese) | 人设多轮 | `--mm_data "hf:Johnson8187/role-play-chinese:-:default:train"` | 同上，`instruction`/`input`/`output`/`system` 形状 |
| [facebook/empathetic_dialogues](https://hf.co/datasets/facebook/empathetic_dialogues) | 情绪陪伴 | `--mm_data "parquet:hf://datasets/facebook/empathetic_dialogues@refs/convert/parquet/default/train/*.parquet:-"` | 英文，人写。对应情绪陪伴这条用途；只学 `utterance` 那一侧。只有加载脚本，`hf:` 直连已失效 |
| [li2017dailydialog/daily_dialog](https://hf.co/datasets/li2017dailydialog/daily_dialog) | 日常口语 | `--mm_data "parquet:hf://datasets/li2017dailydialog/daily_dialog@refs/convert/parquet/default/train/*.parquet:dialog"` | 英文日常对话，按 `dialog` 列表交替映射。同样只有加载脚本 |
| [lorinma/Slim-LCCC-zh](https://hf.co/datasets/lorinma/Slim-LCCC-zh) | 闲聊风味种子 | `--mm_data "hf:lorinma/Slim-LCCC-zh:-:default:train"` | LCCC-large 里挑出的 1 万条 casual 闲聊，转成 ShareGPT 格式，专门用来对比"AI 腔"。⚠️ **gated**：需先 `hf auth login` 并接受条款，否则报 `DatasetNotFoundError` |
| [Gryphe/Sonnet3.5-Charcard-Roleplay](https://hf.co/datasets/Gryphe/Sonnet3.5-Charcard-Roleplay) | 角色卡扮演 | `--mm_data "hf:Gryphe/Sonnet3.5-Charcard-Roleplay:-:default:train"` | 英文、AI 生成，助理腔过滤要更狠 |
| [lemonilia/Roleplay-Forums_2023-04](https://hf.co/datasets/lemonilia/Roleplay-Forums_2023-04) | 真实论坛扮演 | `--mm_data "hf:lemonilia/Roleplay-Forums_2023-04:-:default:train"` | 真实人写，但作者注明 HTML 标签间的空格在转换时损坏，需要自己修文本 |
| 越界拒答样本 | "我不会，你问 AI" | 自造，占 2–8% | 自造 prompt 库 + 口语化回复 |

SFT 的多个来源是**按行数拼接**的（`ChatDataset` 把 `--mm_data` 的各项顺序连成一个 offsets 列表），没有权重
参数：`human_dialog.jsonl, persona.jsonl, refuse.jsonl` 谁的行数多谁就主导。上线前先数一遍各来源的行数与长度
分布，把 §2 里"拒答 2–8%"这类比例按行数落实，别让它悄悄漂走。

## 3. 视觉对齐

| 数据集 | 内容 | 简介 |
|---|---|---|
| [passing2961/photochat_plus](https://hf.co/datasets/passing2961/photochat_plus) | 真实人—人分享照片聊天 | 贴合"图在对话中的语气"；图片存在数据集里，见下方说明 |
| [friedrichor/PhotoChat_image](https://hf.co/datasets/friedrichor/PhotoChat_image) | PhotoChat 图片侧 | ⚠️ 同上 |
| [Mobiusi/Chat-Record-Image-Dataset](https://hf.co/datasets/Mobiusi/Chat-Record-Image-Dataset) | 手机聊天截图 + OCR 文本 | ⚠️ 对应"聊天截图"场景。|

**阻塞**：这些集的图片以 PIL/bytes 存在数据集内部，而 `--mm_data` 目前只认 `--image_root` 下的相对路径，所以它们还不能像预训练集那样一条命令指过去。这一步（落盘时把图一并导出到缓存目录、再设
`--image_root`）还没做；需要时先在本地把图导成 `data/mm/images/...` 再写 jsonl。

## 5. 使用前自检

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
