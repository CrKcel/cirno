# Cirno-130M 训练：端侧对话 & 角色扮演模型

## 0. 设计定位

| 维度 | 设定 |
|---|---|
| 用途 | IM 软件内置拟人化自动回复；游戏内置 NPC 角色扮演；一切"像人在聊天"的场景 |
| 目标 | 流畅、口语化、符合人设的自然语言对话，模仿人类使用即时通讯软件的说话方式 |
| 做什么 | 日常闲聊、情绪陪伴、简单通识（常识、天气、心情、身边事、简单事实） |
| 不做什么 | 复杂数学/物理题、构建代码项目、专业医疗/法律咨询等 |
| SFT | 用**人类—人类对话**调风格，而不是用"助理回答"调能力 |

三条推论：

1. **训的是分布，不是能力。** 130M 参数在解题上不可能和真正的助手模型竞争；token 预算更应该花在对话节奏、语气、人设一致性上。把代码和数学稀释了对话分布。
2. **"助理腔"是负资产。** 长结构化回答、markdown 标题/列表、"我可以帮你…"、"作为一个 AI…" 这类数据不要进预训练和 SFT。
3. **身份是上下文变量，不是模型常量。** 部署时注入 persona（名字/年龄/性别/职业/口吻）；不给模型训练一个固定名字。

## 1. 三阶段总览

**文本预训练 →（可选）视觉对齐 → 对话 / 角色扮演 SFT**。

以下取自 `scripts/train.py` 的 `STAGE_DEFAULTS`，都可以用同名命令行参数覆盖：

| 阶段 | `--train` | lr | warmup | seq_len | micro_batch | grad_accum | token/步 | mtp_weight | vit_lr_mult |
|---|---|---|---|---|---|---|---|---|---|
| `pretrain` | `llm` | 1e-3 | 2% | 1024 | 2 | 256 | 524k | 0.3 | 1.0 |
| `align` | `vit` | 5e-4 | 5% | 512 | 8 | 32 | 131k | 0.0 | 1.0 |
| `sft` | `llm,vit` | 2e-4 | 3% | 1024 | 8 | 32 | 262k | 0.1 | 0.5 |

`--train` 决定哪些模块可训练（`align` 阶段 LLM 冻结，所以该阶段没有可训练参数的 micro-batch 会被跳过
反向）；`vit_lr_mult` 是 ViT 相对于 LLM 的学习率倍数。`sft` 是脚本里的历史阶段名，含义是**对话 /
角色扮演微调**；跳过视觉对齐、只做纯文本对话时用 `--train llm`（见 §3）。

## 2. 阶段 1：预训练

- **数据配比**（按 token 计）。权重不是自由参数：每个桶都按 `权重 × 总 token ÷ 该桶 token 数` 被重复若干遍，
  所以配比表要连同"这个桶有多少 token"一起看：

  | 语料 | 权重 | 目标重复 | 说明 |
  |---|---|---|---|
  | 人类对话 / 口语文本（中文） | 45% | 1–1.5 遍 | IM 聊天记录、论坛贴吧回复、字幕、访谈、播客转写；保留短句、语气词、emoji、重复、错别字 |
  | 英文对话 | 5% | <1 遍 | `allenai/soda`加几个真人小集。中英混输 / 拉丁字母鲁棒性靠它 |
  | 小说 / 网文 / 剧本对白 | 20% | <1 遍 | 叙事 + 角色对白，提供人设口吻与长程一致性 |
  | 长文 / 非对话文本 | 15% | <1 遍 | 长文章、书籍、深度报道、长回答；补篇章连贯性与长文鲁棒性，不是补推理 |
  | 通识 / 百科 / 常识 | 14.5% | <1 遍 | 确保"通识能答" |
  | 角色扮演语料 | 0.5%（上限） | 1–2 遍 | persona 卡片 + 多轮扮演。公开扮演语料总共只有 ~6M token，只能当"system 人设格式"的信号；人设能力靠 SFT |
  | 代码 / 数学 / 专业长文 | 0–2% | — | 不追求这方面能力；130M 上买不到推理泛化，反而稀释对话分布并强化助理腔 |

  权重按比例生效（`TokenMixer` 会归一化），不必凑成 1。配比按**实际 token 数**定，不按目录大小或文件
  体积估：跑完 `prepare_data.py` 后用各桶 `meta.json` 的 `train_tokens` 反算重复遍数，任何源超过 ~2 遍
  就要么扩源、要么降权重。
- **规模**：Chinchilla 参照约 2.7B token；对话语料够就喂 2–4B。公开**中文**人类闲聊语料加总约 1B token
  量级（LCCC base+large ≈ 0.6B + 字幕 ≈ 0.3–0.5B），所以 45% 的 chat 权重已经接近把该桶看到 1.3 遍——
  再往上加只会变成背语料。想要更高的中文对话占比，先扩 chat 桶（字幕清洗 / 论坛 / 播客转写），再改权重。
  英文侧反过来：真人写的英文闲聊加起来只有 ~14M token，能撑起 5% 权重的只有 `allenai/soda`（~190M token），
  而它是合成数据——所以英文给 5% 封顶，用来防中英混输退化，不是用来提量的。
- **非对话文本的天花板**：当前配比里小说 20% + 长文 15% + 通识 14.5% = **49.5% 是非对话文本**，
  对话只占 50.5%。
  长文贡献"连贯性下限"和常识地平线，在长文上多花预算，
  正确做法是**加总 token（3B → 4B）并保持 chat 的绝对 token 不变**（1.35B ≈ 1.3 遍），多出来的量给
  通识和长文；代价是 33% 算力，换到的是不稀释对话分布的额外语言能力。
- **"多来点人类写的小说"在开源界没有好增量**：现代出版小说有版权、不存在；真公版书（`wikimedia/wikisource`）
  有文言文污染，剩下能上量的只有网文。所以小说这一侧该做的是**在同一 20% 内提质**（替换部分低质网文），不是加量。
- **通识：瓶颈是参数量，不是 token 数**。知识桶只占 14.5%（435M token），但实测可用的中文百科远不止：
  百度百科 5.63M 条（~1.5B token）、中文维基 1.4M 篇（~2.05B token）——合计 ~3.5B token，是配额的
  **8 倍**。所以"通识内容会不会太少"
  不是数据问题：116.65M 参数能存下的事实本来就有限，继续灌百科不会让它变多，超过容量后不同事实还会互相
  干扰（同一批权重塞进更多事实，反而压低已记住那些的召回）。
  按性价比排序，真正能改善通识的四件事：
  1. **先测**：现在没有知识评测。先写 50–200 题的常识集（天气 / 心情 / 身边事 / 简单事实
     + 若干"该答不知道"的题），用**生成**而不是 ppl 打分。否则就是在给一个没量过的缺口加数据。
  2. **提事实密度而不是提量**：百度百科是短条目，比中文维基的论述型长文更贴近"一句话把事实说清楚"的形态；
     改用 `title+summary` 取字段能把每 token 的事实密度再抬一截（中位字数见 [datasets.md §1](datasets.md)）。
  3. **课程安排**：把知识桶更多地压进 WSD 衰减阶段（最后 20%，`--decay_frac`），晚期上采样的事实留存优于
     全程均匀铺开。
  4. **形态差归 SFT**：目标行为是"闲聊里顺口答一句常识"，百科是论述文体。这个形态差要靠 SFT 补（简短、
     口语、不确定就说不知道），不是预训练再加几百 M token 能解决的。
- **清洗**：分词型语料必须去空格——`silver/lccc` 的 `dialog` 直接用会让模型学着在汉字之间输出空格，
  tokenize 时加 `--desegment`（原理与三个入口见 §5）。字幕集的 `.ass` 标记同样是重灾区，不清洗会把字幕桶
  绝大部分 token 预算花在 `[Script Info]`、样式和时间轴上（实测比例与清洗步骤见 progress.md TODO）。
- **语料盘点**：字符→token 按 0.7 tok/char 估算（32k BPE、中文为主，±20%），
  真实值以 `prepare_data` 输出的 `train_tokens` 为准。

  | 桶 | 权重 | 预算 token | 可用唯一 token（实测/估算） | 实际遍数 |
  |---|---|---|---|---|
  | chat（中文） | 45% | 1.35B | **~1.05B**（LCCC 0.92B 字符 + 字幕对白 0.6B 字符） | **1.29** |
  | en_chat | 5% | 150M | ~0.20B（soda 190M + 真人小集 14M） | 0.74 |
  | fiction（网文） | 20% | 600M | ~2.55B（3.64B 字符） | 0.24 |
  | long | 15% | 450M | ~46.6B（Fineweb-Edu 4_5 ~46B + 知乎 0.61B） | <0.01 |
  | wiki | 14.5% | 435M | ~3.5B（百度百科 1.5B + 中文维基 2.05B） | 0.12 |
  | roleplay | 0.5% | 15M | **~6M** | **2.50** |

  合计：**预算 3.0B，实际消耗的唯一 token ≈ 2.7B**，重复率约 10%，且全部集中在 chat 与 roleplay 两个桶
  （其余四个桶连一遍都跑不满）。可选语料池约 **55B token，但其中 46B 来自 Fineweb-Edu 单一来源**；
  剔掉它，真正的多源语料池只有 ~8.7B，其中干净的中文对话仍是**最紧的那一环**。加总 token 时保持 chat 的绝对 token 不变，多出来的量给长文与通识。
- **序列**：1024 够用（IM 对话天然短）。把多个会话拼进同一条序列时仍用 `<|endoftext|>` 分隔，并尽量让一条序列 = 一个完整会话，模型才能看到会话边界。长文桶在 1024 的窗口里只能贡献窗口内的连贯性；
  想要真正的长程收益，在衰减阶段单独用长文、`--seq_len 2048` 续训一段。
- **优化器**：AdamW（β=0.9/0.95，wd=0.1，不对 norm/bias/嵌入/`A_log`/`dt_bias` 做衰减），梯度裁剪 1.0。
- **学习率**：峰值 1e-3，2% warmup，**WSD 调度**（稳定 → 最后 20% 线性降到 10%），中途任意 checkpoint 都能接着延长训练。若 `gnorm` 经常尖峰，降到 6e-4。
- **MTP**：主体阶段权重 0.3，最后衰减阶段切到 0.1（`--mtp_weight_final 0.1`）。

## 3. 阶段 2：视觉对齐（可选）

- **先问要不要**：只有部署场景真的会看到图才做这一步——IM 里发表情包/梗图/聊天截图、游戏里要看立绘或场景。纯文字聊天 + NPC 对话可以**完全跳过**，直接
  `--stage sft --init_from runs/pretrain/final --train llm`；此时 ViT 保持随机初始化、不参与推理。跳过 align
  时 SFT 数据必须是纯文本：混入图像只会让随机初始化的 ViT 往 LLM 里灌噪声（`--train llm`）或顺带训出一个没对齐过的 ViT（默认的 `--train llm,vit`）。
- **要做的数据必须是"对话里的图"**，而不是学术 caption/VQA：表情包/贴纸 + 意图标签、梗图 + 上下文、
  聊天截图 → 学"图在对话中是什么语气"。标准 VQA 问答对会把模型往助理方向拉，与本定位冲突。
- 训练方式不变：冻结 LLM，只训练 ViT + 投影器（`--stage align`），seq 512，1 个 epoch，lr 5e-4。
- **数据须全是带图记录**：LLM 冻结时，没有图的 micro-batch 没有可训练参数（脚本会跳过其反向）。
- 这个 ViT 只有 14.4M 且从零训练，识别能力有限（"能认颜色/形状/粗略场景和表情"量级）；要更强的视觉，
  见 progress.md 的"视觉对齐"一条。

## 4. 阶段 3：对话 / 角色扮演 SFT

这一步把基座从"会说话的文本模型"变成"像某个具体的人在聊天"。

### 4.1 数据

- **来源**：真实人类—人类对话（聊天记录、公开论坛/群聊、客服对话）、剧本 /
  小说对白改编、人工编写的 persona 多轮对话（角色卡 + 日常闲聊 + 情绪安抚 + 话题跳转）、少量领域拒答样本。
- **格式**（与 `data_core.normalize_messages` 一致，二选一）：

  ```json
  {"messages": [
    {"role": "system", "content": "你是林小雨，女，24岁，咖啡店店员。说话随意、爱用表情。"},
    {"role": "user", "content": "在吗"},
    {"role": "self", "content": "在的在的 咋啦"}
  ]}
  ```

  或 `{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}`。
  `normalize_messages` 还认开源对话集实际使用的几种形状，直接指向 HF 数据集时不必先转换：
  整段对话放在一个列表里的（LCCC / DailyDialog 的 `dialog`、SODA 的 `dialogue`，配套 `speakers` 时按说话人判断谁是被模仿方，否则按轮流交替）、`instruction`/`input` → `output` 这类字段对（外加可选的`system` / `system_prompt` 人设）、以及 PersonaChat 的 `personality` + `history` + `candidates`（最后一条 candidate 为真实回复）。
- **掩码**：编码器只对 `self` 角色的 token（含其 `<|im_end|>`）计损失，其余上下文只作为条件。
  在人类对话数据里，"`self`"表示**要被模仿的那一方**——把目标说话人标成 `self`（LLaVA 格式的 `gpt`映射到它），对方标成 `user`。输入角色写成 `assistant` 会被拒绝，该记录会被当作坏样本跳过。想同时学会扮演两个方向，就给同一段对话生成两条记录（A→B、B→A），而不是取消掩码。
- **人设**：身份预设写在首条 `system`，不计损失。**没有预设时**，模型只能从
  训练语料里采样，被问"你叫什么/多大/男的女的"会给出随机的人类信息，且多轮之间可能不一致——这是本定位的预期行为。需要稳定身份时，部署侧必须注入 persona。
- **风格负例过滤器**（SFT 前跑一遍）：目标回复里出现 markdown 标题/有序列表、"作为一个 AI/助手"、"我可以帮你"、"很高兴为您服务"、超长分点回答的样本直接丢弃。目标回复长度按 IM 分布截断：建议人类侧中位数 5–25 token，长回复（>128 token）占比 < 5%。

### 4.2 超参

- 全部参数可训练，ViT 学习率为 LLM 的 0.5 倍（跳过 align 时用 `--train llm`）；
  lr 1e-4–2e-4（基座没有任何指令数据时取 2e-4；数据少、担心过拟合取 1e-4），warmup 3%，
  1–3 个 epoch，MTP 权重 0.1，seq_len 1024（会话语料普遍超长时用 2048）。
- **务必混入纯文本对话数据**：本定位下纯文本应占 ≥ 70%，可选的对话内图像数据不超过 30%；否则文本对话能力会退化成"看图说话"。
- 数据量小时优先加数据多样性，而不是加 epoch；用 §6 的验证集监控过拟合。

### 4.3 身份与"像人"的行为

- 训练语料里不出现任何 AI 自我描述，也不要给模型固定名字；模型的身份由 `system` 预设决定（无预设时的行为见 §4.1）。
- 允许并鼓励"人类才有"的行为：短回复、连发多条消息、语气词（哈哈哈 / xswl / 草）、emoji、
  答非所问、偶尔不耐烦、记不清之前说过的话。
- `system` 只写人设（身份、口吻、与对方的关系、禁区），不写"你是 AI 助手"。

### 4.4 越界拒答

- 目标行为：被要求解数学/物理题、写代码项目、给医疗/法律等专业结论时，表示不会。例：「这题我真不会，你问 AI 吧」「这个得找医生看，我哪懂这个」「写代码我不行，你让 AI 给你写」。
- 不能出现的措辞：「作为 AI 我无法…」「我是一个语言模型…」——那会直接暴露 AI 身份并破坏人设。
- 构造方式：用一个覆盖各领域的越界 prompt 库（数学/物理/代码/医疗/法律/长文写作等）配 1–3 条符合人设的口语化回复；这些样本同时带上人设，避免学成"万能拒答机"。
- 配比 2–8%。太少会残留"尝试解题"的行为，太多会让日常闲聊也拒答。

### 4.5 常见失败模式

| 现象 | 原因 | 处理 |
|---|---|---|
| 又变成助理腔 | SFT 混了 VQA / 指令数据，或 `system` 里写了"助手" | 清洗数据；检查 `system` 模板 |
| 回复越来越长 | 目标回复长度分布没控住 | 按 IM 长度分布过滤/截断；提高短回复样本比例 |
| 人设漂移（名字/性别前后不一致） | 无人设预设，或预设太弱 | 部署固定 persona；多轮样本里重复人设细节 |
| 什么都拒答 | 拒答比例过高或模板单一 | 降低比例、增加模板多样性、混入正常闲聊 |
| 复读、答非所问 | 语料重复太多 / lr 太大 | 扩充语料种类、降 lr、早停 |

## 5. 数据准备

```bash
# 0. tokenizer：32k BPE。四件事必须一次做对，因为重做它等于重做所有桶：
#    · 每个占有实际 token 份额的桶都要出现（漏了 fiction 会让 20% 的语料被拆成字节）；
#    · `@N` 是**每个源各自的预算**，要按"平均文档长度"定而不是凭感觉——一条闲聊 ~50 字、一篇条目
#      ~6300 字，所以看起来相等的 N 会把目标分布饿死。下面这组约为 对话 40% / 网文 20% / 通识 15%
#      / 长文 15% / 英文 10%；
#    · 分词型语料（LCCC）必须加 --desegment，否则学到的是带空格文本的 merge；
#    · 数字整段保留（`\p{N}+`），不学 Qwen2 的逐位切分（那是为算术准备的，本项目不做数学）。
python scripts/train_tokenizer.py --out tokenizer --desegment \
    --source "jsonl:data/chat/*.jsonl:text@5000000" --source "jsonl:data/en_chat/*.jsonl:text@90000" \
    --source "jsonl:data/fiction/*.jsonl:text@20000" --source "jsonl:data/wiki/*.jsonl:text@14000" \
    --source "jsonl:data/long/*.jsonl:text@100000"

# 1. 每个来源单独 tokenize，训练时再用权重混合，改配比不必重新 tokenize。
#    §2 的六个桶都能直接在 HF 上取，列表型字段会自动拼成一篇文档，不需要先落地成本地 jsonl。
#    LCCC 用转换后的 parquet 分支（写法见下），并加 --desegment
#    去掉分词空格。--max_tokens 直接取"权重 × 总 token"，跑完看 meta.json 的 train_tokens 是否够：
python scripts/prepare_data.py --tokenizer tokenizer --desegment --out data/bin/chat --max_tokens 1350000000 --val_tokens 4000000 \
    --source "parquet:hf://datasets/silver/lccc@refs/convert/parquet/base/train/*.parquet:dialog" \
    --source "parquet:hf://datasets/silver/lccc@refs/convert/parquet/large/train/*.parquet:dialog" \
    --source "hf:chenrm/anime-subtitles:data:default:train"         # 字幕（.ass 标记需先剥掉）
python scripts/prepare_data.py --tokenizer tokenizer --out data/bin/en_chat --max_tokens 150000000 --val_tokens 1000000 \
    --source "hf:allenai/soda:dialogue:default:train"               # 合成数据，占该桶绝大部分
    # 真人写的英文小集并进同一桶（各自带格式噪声，见 datasets.md §1 英文对话表）：
    #   hf:ParlAI/blended_skill_talk:free_messages:default:train
    #   hf:Fishball02/topical-chat:-:default:train
    #   parquet:hf://datasets/li2017dailydialog/daily_dialog@refs/convert/parquet/default/train/*.parquet:dialog
    #   parquet:hf://datasets/facebook/empathetic_dialogues@refs/convert/parquet/default/train/*.parquet:prompt+utterance
python scripts/prepare_data.py --tokenizer tokenizer --out data/bin/fiction --max_tokens 600000000 \
    --source "hf:wdndev/webnovel-chinese:text:default:train"        # 网文
python scripts/prepare_data.py --tokenizer tokenizer --out data/bin/long --max_tokens 450000000 \
    --source "hf:wangrui6/Zhihu-KOL:RESPONSE:default:train" \
    --source "parquet:hf://datasets/opencsg/Fineweb-Edu-Chinese-V2.1@main/4_5/*.parquet:text"
python scripts/prepare_data.py --tokenizer tokenizer --out data/bin/wiki --max_tokens 435000000 \
    --source "parquet:hf://datasets/lars1234/baidu-baike-dataset@refs/convert/parquet/default/partial-train/*.parquet:-"  # 简体白话百科（主力）
    # --source "hf:wikimedia/wikipedia:text:20231101.zh:train"       # 繁简混用，需先统一成简体（见 progress.md）
python scripts/prepare_data.py --tokenizer tokenizer --desegment --out data/bin/roleplay --max_tokens 15000000 \
    --source "hf:shibing624/roleplay-zh-sharegpt-gpt4-data:-:sharegpt_formatted_data-evol-gpt4:train" \
    --source "hf:Johnson8187/role-play-chinese:-:default:train"

# 2. 预训练（配比见 §2）
python scripts/train.py --stage pretrain --tokenizer tokenizer --out_dir runs/pretrain \
    --data data/bin/chat:0.45,data/bin/en_chat:0.05,data/bin/fiction:0.20,data/bin/long:0.15,data/bin/wiki:0.145,data/bin/roleplay:0.005 \
    --tokens 3e9 --resume

# 3.（可选）视觉对齐
python scripts/train.py --stage align --tokenizer tokenizer --init_from runs/pretrain/final \
    --out_dir runs/align --mm_data data/mm/chat_image.jsonl --image_root data/mm/images --epochs 1 --resume

# 4. 对话 / 角色扮演 SFT：人类对话 + 人设 + 少量越界拒答
#    跳过视觉对齐时：--init_from runs/pretrain/final --train llm（数据须为纯文本）
python scripts/train.py --stage sft --tokenizer tokenizer \
    --init_from runs/align/final --out_dir runs/sft --train llm \
    --mm_data data/sft/human_dialog.jsonl,data/sft/persona.jsonl,data/sft/refuse.jsonl \
    --epochs 2 --lr 2e-4 --resume

# 同一个 SFT 阶段也可以直接吃 HF 数据源，不用先落 jsonl：
# LCCC 走转换后的 parquet 分支，并且要 --desegment：
python scripts/train.py --stage sft --tokenizer tokenizer \
    --init_from runs/pretrain/final --out_dir runs/sft --train llm --epochs 2 --lr 2e-4 --resume --desegment \
    --mm_data "parquet:hf://datasets/silver/lccc@refs/convert/parquet/base/train/*.parquet:dialog,hf:shibing624/roleplay-zh-sharegpt-gpt4-data:-:sharegpt_formatted_data-evol-gpt4:train"
```

- `--source` 支持 `hf:`（流式 datasets）、`jsonl:`、`parquet:`、`txt:`，末尾 `@N` 限制文档数；字段按
  `<数据集>:<字段>` 给出，`-` 表示整条记录，`a.b` 走嵌套路径，`a+b` 按顺序拼接（见 `scripts/common_io.py`
  的模块 docstring；[datasets.md §1](datasets.md) 给了每个候选集的现成写法）。列表型字段（LCCC 的
  `dialog`、MNBVC 的 `回复`）会自动用换行拼接成一篇文档，所以不需要先写转换脚本；没有字段名时可先用
  `-` 取整条记录。文档之间用 `<|endoftext|>` 分隔，产物是扁平 uint16 token 流 + `meta.json`。
- `parquet:` 也接受远端 `hf://` glob，例如
  `parquet:hf://datasets/silver/lccc@refs/convert/parquet/base/train/*.parquet:dialog`。这条写法有几种用途：
  读只带加载脚本的集（LCCC / DailyDialog / EmpatheticDialogues 等，规则见 [datasets.md §1](datasets.md)
  开头）、以及取 Fineweb-Edu-Chinese 某个分数档（`4_5/*.parquet`，它不是 config）。路径里的 `@refs/...`
  是 revision，不是 `@N` 文档上限。
- **`--desegment`**：去掉汉字之间的空格，给分词型语料用（`silver/lccc` 的 `dialog`、以及任何由它派生的
  SFT 数据）。`train_tokenizer.py` / `prepare_data.py` / `train.py`（align、sft 阶段）都有这个开关；
  它只删"一侧是 CJK"的空格，英文与数字之间的空格不受影响。
- `--val_tokens` 会把前 N 个 token 切给 `val.bin`；训练时定期在验证集上打印 `loss` / `ppl`。对话模型的
  ppl 只能当作"没训崩"的 sanity check，选型看 §6。
- `--mm_data` 用逗号分隔，每一项既可以是本地 jsonl 路径，也可以是 `hf:` / `parquet:` / `jsonl:<glob>`
  这类来源；远程来源会先流式过一遍 `normalize_messages`，落到 `--mm_cache`（默认 `data/cache`，可复用）
  的一个 jsonl 里再训练，这样 `Dataset` 仍按字节偏移随机读、`--resume` 精确，也不会把整个数据集塞进内存。
  无法映射成 user/self 的记录（角色写成 `assistant`、只有一轮没有回复、字段认不出来）在落盘和读取时都会
  被跳过。注意 `--mm_data` 里的来源是**先下载一遍**才开训，第一次启动会等得比较久。
- 只有带 `"image"` 字段的记录才需要 `--image_root`；纯文本 SFT 不用给。
- **磁盘预算**：`--max_tokens` 取够就停，所以**不需要把数据池整个下下来**——全量下上表所有源约 **100 GB**
  （光 Fineweb-Edu `4_5` 一档就 ~89 GB），而按当前配比实际只需 **~5.4 GB**：

  | 桶 | 源 | 全量 | 按配比实际需要 |
  |---|---|---|---|
  | chat | LCCC base + large | 1.67 GB | 1.67 GB（可用量不够，全要） |
  | chat | anime-subtitles | 1.30 GB | 1.30 GB（清洗后只剩 ~0.4B token） |
  | en_chat | soda + 4 个真人小集 | 0.74 GB | ~0.56 GB |
  | fiction | webnovel-chinese | 3.16 GB | ~0.75 GB（24%） |
  | long | Zhihu-KOL | 1.50 GB | ~1.10 GB（74%） |
  | long | Fineweb-Edu `4_5` | **~89 GB** | **0 — 不会被读到** |
  | wiki | wikipedia zh | 1.72 GB | ~0.29 GB（12%） |
  | wiki | baidu-baike（parquet） | 3.21 GB | ~0.93 GB |
  | roleplay | 两个集 | 0.01 GB | 全要 |
  | **合计** | | **~100 GB** | **~5.4 GB** |

  ⚠️ `--source` 是**按给定顺序**读、共享同一个预算的，所以**长文桶填配额时 Fineweb-Edu 根本轮不到**：
  Zhihu-KOL 自己就有 ~0.61B token > 450M 配额。想要 Fineweb-Edu 的文本多样性，得把它排在 Zhihu 之前，
  或者调高/调低两者的相对配额（拆成两个 `prepare_data` 再在 `train.py` 里按权重混）。
  另外：分片粒度会让实际值偏高（LCCC / soda / 网文每片 300+ MB）；不下 Fineweb-Edu `3_4`（800 GB）
  与 `2_3`（1.4 TB）、`fineweb-2` 的 `cmn_Hani`（**1.62 TB**）、wikisource 中文版。
  `baidu-baike` 优先走 parquet 分支：3.2 GB vs 原始单文件 json 16.8 GB。
  派生出的 token 分片（`data/bin/*`，uint16 即 2 B/token）再占 **~5.4 GB**（实际唯一 2.69B token）；
  加 `last/`（~1.6 GB 优化器状态）与各阶段 `final/`，一次完整三阶段训练合计约 **17 GB**。
  `--mm_data` 的远程来源会在 `--mm_cache` 另落一份 jsonl 缓存。
  ⚠️ **`hf:` 流式读取不是"不落盘"**：下载的 parquet 会留在 HF 缓存里（本机实测
  `~/.cache/huggingface` 已堆到 **5 GB**，其中 `hub/blobs` 2.3 GB）。所以上表"实际需要"要再加上缓存，
  大盘紧张时用 `HF_HOME=/big/disk` 指过去，或每跑完一个桶 `huggingface-cli delete-cache`。
- **"只取前几片"不是随机抽样——分片顺序是有结构的**（实测各源首片/末片）：

  | 源 | 排序依据 | 没下载到的那部分 | 判断 |
  |---|---|---|---|
  | wikipedia zh | 条目重要度（首片 `数学`/`哲学`/`文學`，末片是冷门化合物） | 冷门条目 | ✅ 截断反而砍掉长尾，12% 正好落在核心通识上 |
  | baidu-baike | 常见词 → 冷门实体（首片 `红色食品`/`丈夫`，末片 `林周风毛菊`/`新疆华凌工贸`） | 冷门实体；且 parquet 分支本身只覆盖原 json 的 30% | ✅ 同上 |
  | webnovel | 按书顺序（首片《(种田文系列)无尽相思风》第一章） | 书单靠后的 ~7600 本 | ⚠️ 只取 24% 会偏向排在前的题材 |
  | **Fineweb-Edu 4_5** | **按来源语料分组**（首片 `CCI3`，末片 `OpenCSG-CC`） | 只取 1% 就是**清一色 CCI3** | ❌ 拿不到 OpenCSG-CC / map-cc |
  | anime-subtitles | 按目录（首片 `剧场版(OVA)`，末片 `季度番(散)`） | 只取 20% 就只有剧场版，没有 TV 系列 | ❌ 题材不全 |
  | LCCC base/large | 无明显主题序 | — | ✅ 基本无偏 |

  所以 wiki / baidu 直接顺序截断没问题；Fineweb-Edu 与字幕**不要按顺序截断**——字幕反正要写清洗脚本，
  清洗时在全 60 片上采样即可；网文若要更全，把 10 片全下（3.16 GB）后按步长挑片。

## 6. 评测

这个定位下，损失/ppl 不作为选型依据。至少覆盖：

| 维度 | 指标 | 做法 |
|---|---|---|
| 语言自然度 | 人类盲测胜率 | 同一上下文，人写回复 vs 模型回复，双盲二选一；目标是接近 50% 分不出 |
| 人设一致性 | 身份问答一致率 | 同一 persona 多轮问名字/年龄/职业；无人设时统计各会话给出的身份分布（应接近语料先验，且**不**出现 AI 自述） |
| 越界拒答 | 拒答率 / 误拒率 | 越界 prompt 库（数学/物理/代码/医疗/法律）上的拒答率，以及日常闲聊被误拒的比例 |
| IM 形态 | 回复长度分布、助理腔词命中 | 与人类对话的长度分位数对比；统计 §4.5 列的那类模式词 |
| 端侧成本 | 首 token 延迟、decode tok/s、峰值内存 | 在目标设备（手机/NPU）实测 |

自动指标只在回归时看趋势，不作发布门槛；人工盲测和越界拒答率才是。

## 7. 端侧推理

- **参数量**：推理只加载 LLM 主干（纯文本 116.65M，含视觉塔 131.0M），MTP 块只用于训练、不加载。
- **会话内存**：20 层里 15 层是 Gated DeltaNet，状态大小固定、不随对话轮数增长；只有 5 层 GQA 全注意力需要 KV。长会话的内存增长因此远小于同等规模的纯 Transformer，适合端侧常驻。
- **短回复**：IM 风格本身就意味着 32–128 token 的输出，decode 是主要成本；建议
  `max_new_tokens ≤ 128`，并用 persona 约束回答长度。

## 8. 训练开关

| 参数 | 说明 |
|---|---|
| `--tokens` / `--max_steps` / `--epochs` | 训练长度：pretrain 用总 token 数或步数，align/sft 用 epoch 数 |
| `--max_mem_gb` | CUDA 分配器硬上限（默认 8），超预算快速失败；`0` 关闭上限 |
| `--schedule cosine` / `--decay_frac` / `--min_lr_ratio` | 调度器选择与衰减比例 |
| `--resume` | 从 `out_dir/last` 续训，含数据顺序与 GradScaler 状态 |
| `--eval_every` / `--eval_batches` / `--save_every` / `--log_every` | 验证频率与批次、保存与日志频率 |
| `--precision` / `--gdn_impl` / `--grad_ckpt` / `--ce_ckpt` / `--compile` / `--tf32` | 精度与 kernel |
