# 训练数据集

取数据的语法见 [common_io.py](../scripts/common_io.py) 的模块 docstring：
`hf:<数据集>:<字段>[:<config>[:<split>]]`，字段 `-` 表示整条记录、`a.b` 走嵌套路径、`a+b` 顺序拼接，列表型字段自动用换行拼成一篇文档，末尾 `@N` 限制文档数。

## 1. 预训练

| 数据集 | 类型 | 规模 | 取法 | 简介 |
|---|---|---|---|---|
| [silver/lccc](https://hf.co/datasets/silver/lccc) | 人类对话 | base 6.8M / large 12M 条 | `hf:silver/lccc:-:base:train` | 微博闲聊二人会话。`dialog` 是列表，会自动拼成一篇文档；SFT 也用它 |
| [chenrm/anime-subtitles](https://hf.co/datasets/chenrm/anime-subtitles) | 人类对话 | 67k 部字幕，原始 12 GB | `hf:chenrm/anime-subtitles:data:default:train` | 动画字幕，短句口语多；下载 1.3 GB。|
| [Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset](https://hf.co/datasets/Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset) | 人类对话 | 4.2k 段对话 | `hf:Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset:text:default:train` | 日常对话 + 影视对白 + AI 生成混合，自带情绪标签；量小（161 KB parquet），当风格补充 |
| [wangrui6/Zhihu-KOL](https://hf.co/datasets/wangrui6/Zhihu-KOL) | 人类对话 | 1.0M 条 | `hf:wangrui6/Zhihu-KOL:RESPONSE:default:train` | 真人类长回答，但形态偏"知识问答"。可能需要按长度截断 |
| [liwu/MNBVC](https://hf.co/datasets/liwu/MNBVC) | 人类对话 | `forum` 等 18 个子集 | `hf:liwu/MNBVC:主题+回复.回复:forum:train` | 论坛对白/博客/新闻/问答都在这个仓库的不同 config 里。`回复` 是 `list[dict]`，用 `a.b` 嵌套路径取；带下载脚本、会先整包下载 |
| [wdndev/webnovel-chinese](https://hf.co/datasets/wdndev/webnovel-chinese) | 网文 | 600k 章 / 3.2 GB（作者称约 9B token） | `hf:wdndev/webnovel-chinese:text:default:train` | 网文，字段干净（`title`/`chapter`/`text`） |
| [Miaowuawa/ChineseNovels](https://hf.co/datasets/Miaowuawa/ChineseNovels) | 网文 | 2.3k 本 / 623 MB | `hf:Miaowuawa/ChineseNovels:-:default:train` | 同人/耽美占比高，单列结构（viewer 打不开，用 `-` 取整条记录） |
| [shibing624/roleplay-zh-sharegpt-gpt4-data](https://hf.co/datasets/shibing624/roleplay-zh-sharegpt-gpt4-data) | 角色扮演 | 6.6k 条 | `hf:shibing624/roleplay-zh-sharegpt-gpt4-data:-:sharegpt_formatted_data-evol-gpt4:train` | 带 `system_prompt` 人设。AI 生成，助理腔要过滤 |
| [Johnson8187/role-play-chinese](https://hf.co/datasets/Johnson8187/role-play-chinese) | 角色扮演 | 22.4k 条 | `hf:Johnson8187/role-play-chinese:-:default:train` | `instruction`/`input`/`output`/`system`，人设齐全。AI 生成，同样要过滤 |
| [wikimedia/wikipedia](https://hf.co/datasets/wikimedia/wikipedia) | 通识 | zh 1.4M 篇 / 1.7 GB | `hf:wikimedia/wikipedia:text:20231101.zh:train` | **通识首选**，字段 `text` |
| [liwu/MNBVC](https://hf.co/datasets/liwu/MNBVC) `wiki`/`news`/`qa` | 通识 | 见仓库 README | `hf:liwu/MNBVC:问+答:qa:train` | 和 forum 同一仓库不同 config，取法同上 |
| [lars1234/baidu-baike-dataset](https://hf.co/datasets/lars1234/baidu-baike-dataset) | 通识 | 百科条目 | `hf:lars1234/baidu-baike-dataset:-:default:train` | 字段名未核验，先用 `-` 取整条 |
| liwu/MNBVC `code`/`math` | 代码/数学 | — | —  |  跳过或只留 1–2% |

按 §2 配比拼起来（权重按 token 算，不是按文件大小）：

```bash
# 人类对话 55% + 网文 20% + 角色扮演 12% + 通识 13%
python scripts/train.py --stage pretrain --tokenizer tokenizer --out_dir runs/pretrain --tokens 3e9 --resume \
    --data data/bin/chat:0.55,data/bin/fiction:0.20,data/bin/roleplay:0.12,data/bin/wiki:0.13
```

英文对话：

| 数据集 | 规模 | 取法 | 简介 |
|---|---|---|---|
| [allenai/soda](https://hf.co/datasets/allenai/soda) | 1.2M 会话 | `hf:allenai/soda:dialogue:default:train` | 社交对话，话很自然；`speakers` 与 `dialogue` 对齐，SFT 里能自动判谁是被模仿方 |
| [li2017dailydialog/daily_dialog](https://hf.co/datasets/li2017dailydialog/daily_dialog) | 13k 会话 | `hf:li2017dailydialog/daily_dialog:dialog:default:train` | 日常对话，人写、干净、短句多。带下载脚本 |
| [facebook/empathetic_dialogues](https://hf.co/datasets/facebook/empathetic_dialogues) | 25k 会话 | `hf:facebook/empathetic_dialogues:-:default:train` | cc-by-nc-4.0 | 情绪陪伴向。**一行一句**：`prompt` 是上一句、`utterance` 是回复（各自映射成 user/self），`context` 是情绪标签、不会被拼进 prompt。|
| [thu-coai/esconv](https://hf.co/datasets/thu-coai/esconv) | 1.3k 段对话 | `hf:thu-coai/esconv:text:default:train` |  真实人写的情感支持对话，段落是整段会话的纯文本，适合当预训练文本；量很小，只能当风格补充。|
| [bavard/personachat_truecased](https://hf.co/datasets/bavard/personachat_truecased) | 131k 条 | `hf:bavard/personachat_truecased:-:full:train` |  `personality`+`history`+`candidates`，每条是"对话历史 + 候选回复"，直接喂预训练会大量重复，只用于 SFT |

## 2. 对话 / 角色扮演 SFT

取法和预训练一样，`--mm_data` 直接收 `hf:` 来源（会先落一份缓存 jsonl，见 training.md §5）。

| 数据集 | 用途 | 取法 | 说明 |
|---|---|---|---|
| [silver/lccc](https://hf.co/datasets/silver/lccc) | 主力：人类口吻 | `--mm_data "hf:silver/lccc:-:base:train"` | 每条会话按 `user`/`self` 轮流交替，正好是"被模仿的一方"。想两个方向都学，按 §4.1 各出一条记录 |
| [shibing624/roleplay-zh-sharegpt-gpt4-data](https://hf.co/datasets/shibing624/roleplay-zh-sharegpt-gpt4-data) | 人设多轮 | `--mm_data "hf:shibing624/roleplay-zh-sharegpt-gpt4-data:-:sharegpt_formatted_data-evol-gpt4:train"` | `system_prompt` 作为首条 `system` 注入且不计损失 |
| [Johnson8187/role-play-chinese](https://hf.co/datasets/Johnson8187/role-play-chinese) | 人设多轮 | `--mm_data "hf:Johnson8187/role-play-chinese:-:default:train"` | 同上，`instruction`/`input`/`output`/`system` 形状 |
| [facebook/empathetic_dialogues](https://hf.co/datasets/facebook/empathetic_dialogues) | 情绪陪伴 | `--mm_data "hf:facebook/empathetic_dialogues:-:default:train"` | 英文，人写。对应情绪陪伴这条用途；只学 `utterance` 那一侧 |
| [li2017dailydialog/daily_dialog](https://hf.co/datasets/li2017dailydialog/daily_dialog) | 日常口语 | `--mm_data "hf:li2017dailydialog/daily_dialog:-:default:train"` | 英文日常对话，按 `dialog` 列表交替映射 |
| [lorinma/Slim-LCCC-zh](https://hf.co/datasets/lorinma/Slim-LCCC-zh) | 闲聊风味种子 | `--mm_data "hf:lorinma/Slim-LCCC-zh:-:default:train"` | LCCC-large 里挑出的 1 万条 casual 闲聊，转成 ShareGPT 格式，专门用来对比"AI 腔" |
| [Gryphe/Sonnet3.5-Charcard-Roleplay](https://hf.co/datasets/Gryphe/Sonnet3.5-Charcard-Roleplay) | 角色卡扮演 | `--mm_data "hf:Gryphe/Sonnet3.5-Charcard-Roleplay:-:default:train"` | 英文、AI 生成，助理腔过滤要更狠 |
| [lemonilia/Roleplay-Forums_2023-04](https://hf.co/datasets/lemonilia/Roleplay-Forums_2023-04) | 真实论坛扮演 | `--mm_data "hf:lemonilia/Roleplay-Forums_2023-04:-:default:train"` | 真实人写，但作者注明 HTML 标签间的空格在转换时损坏，需要自己修文本 |
| 越界拒答样本 | "我不会，你问 AI" | 自造，占 2–8% | 自造 prompt 库 + 口语化回复 |

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
from common_io import iter_texts
for t in iter_texts("hf:silver/lccc:-:base:train", max_docs=3):
    print(repr(t[:200])); print("-" * 40)
PY
```

SFT 数据在落盘阶段会被 `normalize_messages` 过一遍，跑起来会打印
`cached N records (M unusable) -> data/cache/records-xxxx.jsonl`。`M` 很大说明字段或角色选错了，
去 `--mm_cache` 指向的那个文件里看留下的记录对不对。缓存文件可以随便删，下次会重新生成。
