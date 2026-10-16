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

- **配方是文件，不是命令行**。`configs/pretrain_plan.json` 声明三段各自的预算、形状、配比目标，以及每个桶的
  `allocate`（跨三段最多从这个桶取多少 token）；`tools/plan_mix.py --plan configs/pretrain_plan.json
  --write runs/pretrain/plan.resolved.json --print-commands` 把各桶 `prepare_data` 实测的 `train_tokens` 代进去，
  算出每段的 `--data` 权重并写出可直接执行的配方；`scripts/train.py --plan ... --phase A|B|C` 读这份文件拿
  `--tokens`、形状、配比与调度。权重因此永远是推导值：没有桶会被要到超过它 `allocate` 的量，改配比只需改配方
  再重跑一次这个脚本，不用手抄权重。
- **预算 5.39B，三段 1.21B / 3.10B / 1.08B（22.4% / 57.5% / 20.0%）**，每段一份配比：

  | 语料 | A 长文<br>1.21B seq 2048 | B 对话<br>3.10B seq 1024 | C 退火<br>1.08B seq 1024 | 全程 token | 池子 | allocate | 实读 |
  |---|---|---|---|---|---|---|---|
  | 人类对话 / 口语文本（中文） | — | 24.7% | 41.4% | 1.212B | 1.010B | 1.212B | **1.20 遍** |
  | 英文对话 | — | 25.1% | 3.0% | 810M | 810M | 810M | 1.00 遍 |
  | 小说 / 网文 / 剧本对白 | 45% | 24.4% | 27.0% | 1.593B | 1.593B | 1.593B | 1.00 遍 |
  | 长文 / 非对话文本 | 35% | 2.3% | 3.0% | 527M | 527M | 527M | 1.00 遍 |
  | 通识 / 百科 / 常识 | 20% | 23.5% | 23.5% | 1.225B | ≈1.39B | 1.225B | **0.88 遍** |
  | 角色扮演（persona 卡片 / 小说角色 / 猫娘） | — | — | 2.1% | 22.5M | 22.5M | 22.5M | 1.00 遍 |

  池子的构成（估算与取法见 [datasets.md §1](datasets.md)）：`Limour/b-corpus` 的正文类目录已全部归入各桶
  （chat +130M、en_chat +52M、fiction +163M、long +21M、roleplay +2.2M、roleplay_char +1.6M），英文桶另加
  `noanabeshima/TinyStoriesV2`（+≈550M），fiction / wiki / roleplay 再收 `Miaowuawa/ChineseNovels`（截断后
  ≈254M）、`mrzjy/chinese_moegirl_wiki_corpus_raw`（清洗后 72.1M 字符 ≈ 50M）、
  `LooksJuicy/Chinese-Roleplay-SingleTurn`（≈3.3M）。这些数字是抽样外推（±20%）：按 `prepare_data` 输出的
  `train_tokens` 重跑一次 `tools/plan_mix.py`，`expected_tokens` / `allocate` 跟着实测值走。

  每段的理由：

  1. **A 长文（1.21B，seq 2048）**：最早的 token 在最高 lr 下训练、梯度更新最大，决定基底分布，所以只放书面语
     与叙事——小说 45% + 长文 35% + 通识 20%，同时用 2048 的窗口让长文真的贡献长程。百科在这一段只占 20%：
     继续加百科的收益受 116.65M 参数的容量限制（见下）。
  2. **B 对话（3.10B，seq 1024）**：引入 chat 24.7% + 英文 25.1%，IM 语域、语气、回合节奏在这一段学；小说 /
     长文 / 通识给 24.4% / 2.3% / 23.5% 继续提供多样性与常识地平线。
  3. **C 退火（1.08B，seq 1024）**：与 WSD 的最后 20%、MTP 权重切换重合，配比回到部署分布——chat 41.4% 保住
     风格，角色扮演 2.1% 只在这里出现，长文压到 3.0%，让最后的梯度更新几乎只改"怎么说话"。

  两处刻意的重采样：

  - **chat 读 1.20 遍**（1.212B / 1.010B）。对话是这个定位唯一的目标分布，而公开中文人类闲聊只有 1.01B；
    多出的 202M 全部落在退火段（C 段 chat 占 41.4%），是给最后一段梯度做分布温度，而不是全程均匀重放。
    会话级语料重复第二遍的边际收益低于"最后怎么说话"，这也是它能排在百科尾部前面的原因。
  - **wiki 读 0.88 遍**（1.225B / ≈1.39B）。`baidu-baike` 的顺序是常见词 → 冷门实体，截掉的 ≈165M 正好是
    长尾；116.65M 参数本来也存不下它们，超过容量后不同事实互相干扰，反而压低已记住那些的召回。
    `prepare_data` 的 `--max_tokens` 因此按 allocate 给（1228050000 + val），少 tokenize 165M；量小的萌娘百科
    排在这个 cap 之前读，不会被百度百科吃满。

  全程对话 **37.5%**（chat 22.5% + 英文 15.0%），非对话 62.5%（小说 29.6% + 长文 9.8% + 通识 22.7% + 角色扮演 0.4%）：
- **规模**：5.39B token ÷ 116.65M 参数 = **46.2 token/参数**，约为 Chinchilla 参照（2.7B）的 2 倍。
- **chat 桶是最紧的一环**：公开中文人类闲聊加总 **1.01B token**（LCCC base+large 453M，其中 base 是
  large 的清洗子集、两者重叠约 30%；字幕清洗后有效 428M；b-corpus `v-corpus-zh` ≈130M）。要再扩只能从被
  卡拉OK 噪声占掉的字幕余量（先给 `clean_ass.py` 加逐字行过滤）与论坛/群聊转写里找；配比能做的只是把它放在
  退火段多读一遍。
- **长文桶以知乎为主**（实测 1.006M 条 / 0.72B 字符 ≈ **0.51B token**，一遍读完）；b-corpus 的
  `提高逼格`（白话二十四史）、`畅销书`、`散文集`（合计 ≈15M token）与 `古籍`（文言，慎用，≈6M）也归
  该桶，池子 ≈527M。
- **小说这一侧**：真公版书（b-corpus `古籍`）偏文言文，是风格污染；能上量的是
  `wdndev/webnovel-chinese` parquet 分支的全量（600k 章 / 1.68B 字符 ≈ 1.176B token）。b-corpus 再补
  `互动小说`（同人、未清洗，≈149M token）与 `近现代小说`/`翻译名著`/`轻小说`（≈14M，出版作品、授权链
  不清），`Miaowuawa/ChineseNovels` 补 ≈254M（整本一篇，不拆章只进每本前 200k 字符），池子 ≈1.59B。
- **通识：瓶颈在参数量**。知识桶 allocate 1.225B token（22.7%）：百度百科 parquet 分支实测 1.73M 条 /
  1.92B 字符 ≈ **1.34B token**（按常见实体优先读，尾部不读），萌娘百科清洗后 72.1M 字符 ≈ 50M。
- **清洗**：分词型语料必须去空格——`silver/lccc` 的 `dialog` 直接用会让模型学着在汉字之间输出空格，
  tokenize 时加 `--desegment`（原理与三个入口见 §5）。字幕集的 `.ass` 标记同样是重灾区，不清洗会把字幕桶
  绝大部分 token 预算花在 `[Script Info]`、样式和时间轴上（清洗产物已在快照里，见 progress.md TODO）。
  其余文本层清洗——b-corpus 的 `.txt.gz`、萌娘百科原始 wikitext、JSON 数组角色扮演集、英文小集的
  标点/标记噪声——统一由 `scripts/prepare_local_data.py` 落成 `data/<bucket>/*.jsonl`。
- **语料盘点**：字符→token 按 0.7 tok/char（中文）/ 0.25（英文）估算（32k BPE、中文为主，±20%），
  真实值以 `prepare_data` 输出的 `train_tokens` 为准；下表是本地快照的实测字符数与由此得到的池子。

  | 桶 | 池子（实测，200k 字符/篇截断后） | allocate | 实读 |
  |---|---|---|---|
  | chat（中文） | **1.010B**（LCCC base+large 453M + 字幕有效 428M + b-corpus `v-corpus-zh` 130M） | 1.212B | **1.20 遍** |
  | en_chat | 810M（soda 191M + 真人小集 17M + b-corpus 52M + TinyStoriesV2 550M） | 810M | 1.00 遍 |
  | fiction（网文） | 1.593B（webnovel 1.176B + 互动小说 149M + 近现代/翻译/轻小说 14M + ChineseNovels 254M） | 1.593B | 1.00 遍 |
  | long | 527M（知乎 506M + b-corpus 21M） | 527M | 1.00 遍 |
  | wiki | ≈1.39B（百度百科 1.343B + 萌娘百科 ≈50M，72.1M 字符实测） | 1.225B | **0.88 遍** |
  | roleplay（persona 卡片） | 17.2M（shibing624 1.5M + Johnson8187 10.3M + `q-corpus` 2.2M + LooksJuicy 3.3M） | 17.2M | 1.00 遍 |
  | roleplay_char（小说角色） | 5.1M（ChatHaruhi 对白 3.5M + b-corpus 1.6M） | 5.1M | 1.00 遍 |
  | roleplay_catgirl（猫娘） | 0.15M（kxdw2580 v2，去掉 `<think>` 后） | 0.15M | 1.00 遍 |

  合计：**预算 5.39B = allocate 之和**；唯一被重复的是 chat（+202M），唯一没读完的是 wiki（−165M）。
- **序列**：阶段 A 用 **2048** 训长文，B / C 用 1024（IM 对话天然短）。把多个会话拼进同一条序列时仍用
  `<|endoftext|>` 分隔，并尽量让一条序列 = 一个完整会话，模型才能看到会话边界。2048 不需要位置缩放
  （`max_position_embeddings=4096`、`rope_theta=1e6`），代价也几乎为零：8 GB 卡上 seq 2048 × mb 2 与
  seq 1024 × mb 2 的吞吐、显存同档（见 `scripts/run_pipeline.sh` 步骤 2 的实测表）。
- **优化器**：AdamW（β=0.9/0.95，wd=0.1，不对 norm/bias/嵌入/`A_log`/`dt_bias` 做衰减），梯度裁剪 1.0。
- **学习率**：峰值 1e-3，2% warmup，**一条以 token 为轴的全局 WSD**（恒定 → 最后 20% 线性降到 10%）。三段是三次
  `--resume` run，但调度由配方统一计算（`lr_at_tokens`：`tokens_done / budget_tokens`），不需要"中间段
  `--decay_frac 0`、累计 `--tokens`"这套手写约定，衰减起点天然落在退火段起点（0.8 × 5.39B = 4.31B）。
  若 `gnorm` 经常尖峰，把配方里的 `schedule.peak_lr` 降到 6e-4；监控的受控回退也会在尖峰时临时压 lr
  （见 [monitoring.md](monitoring.md)）。
- **MTP**：主体阶段权重 0.3，最后衰减阶段切到 0.1（配方 `mtp_weight_final`），切换点与退火段起点重合。
- **监控**：每次 run 同时写 `metrics.jsonl` 与 `health.jsonl`，`python tools/watch_training.py --dir
  runs/pretrain` 实时看指标、进度、tok/s、ETA 与告警；异常判定与受控干预（lr 回退、可选早停）见
  [monitoring.md](monitoring.md)。

## 3. 阶段 2：视觉对齐（可选）

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
- **`--mm_weights` 是配比旋钮**：`ChatDataset` 把 `--mm_data` 的各项拼成一个 offsets 列表，不给权重时
  谁的行数多谁说了算（一个 500 行的拒答集在 100 万行闲聊旁边等于不存在）。给了权重就按目标比例采样，
  小于目标份额的源会被重复到那个份额——§4.4 的 2–8% 由此变成一个可执行的数字，而不是人工数行数。
  配方里的 74% / 21% / 5% 见 `configs/pretrain_plan.json` 的 `sft.mix_weights`。
- **务必混入纯文本对话数据**：本定位下纯文本应占 ≥ 70%，可选的对话内图像数据不超过 30%；否则文本对话能力会退化成"看图说话"。
- 数据量小时优先加数据多样性，而不是加 epoch；用 §6 的验证集监控过拟合，并给
  `--early_stop_patience`（例如 4）让监控在验证集不再改善时停下（见 [monitoring.md](monitoring.md)）。

### 4.5 常见失败模式

| 现象 | 原因 | 处理 |
|---|---|---|
| 又变成助理腔 | SFT 混了 VQA / 指令数据，或 `system` 里写了"助手" | 清洗数据；检查 `system` 模板 |
| 回复越来越长 | 目标回复长度分布没控住 | 按 IM 长度分布过滤/截断；提高短回复样本比例 |
| 人设漂移（名字/性别前后不一致） | 无人设预设，或预设太弱 | 部署固定 persona；多轮样本里重复人设细节 |
| 什么都拒答 | 拒答比例过高或模板单一 | 降低比例、增加模板多样性、混入正常闲聊 |
| 复读、答非所问 | 语料重复太多 / lr 太大 | 扩充语料种类、降 lr、早停 |

## 5. 数据准备

数据全部来自本地快照（`CIRNO_SNAPSHOT`，默认 `/mnt/data/LLM/hf`；每个仓库一个目录、保留 Hub 的文件
布局），所以整条流水线不需要联网。下面的命令与 [run_pipeline.sh](../scripts/run_pipeline.sh) 的前四步
等价，完整清单以脚本为准——桶内的源、顺序、cap 与理由都在它的 DATA 段。

```bash
SNAP="${CIRNO_SNAPSHOT:-/mnt/data/LLM/hf}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # 残留的 hf: 立即报错，不会下载
```

```bash
# 0a. 文本层清洗：b-corpus 的 .txt.gz、萌娘百科原始 wikitext、两个 JSON 数组角色扮演集、带标点/标记
#     噪声的英文小集 -> data/<bucket>/*.jsonl；SFT 的两份切片（human_dialog / persona）也在这一步落盘。其余源直接在快照上按字段表达式读（`parquet:` 的 `a.b`
#     能走进嵌套列表，例如互动小说的 `book_chapter.content.content`，正文之外的 content_tag/role 不进
#     文档）。已存在的产物会跳过，--force 重做。
python scripts/prepare_local_data.py --snapshot "$SNAP" --out data

# 0b. tokenizer：32k BPE。四件事必须一次做对，因为重做它等于重做所有桶：
#     · 每个占有实际 token 份额的桶都要出现（漏了 fiction 会让近三成的语料被拆成字节）；
#     · `@N` 是**每个源各自的预算**，要按"平均文档长度"定而不是凭感觉——一条闲聊 ~80 字、一篇条目
#       ~1000 字，看起来相等的 N 会把目标分布饿死。run_pipeline.sh 步骤 0 是完整清单：按 §2 的全程
#       token 总量配成 对话 19% / 网文 30% / 通识 26% / 长文 10% / 英文 15%，重新推导用
#       `python tools/check_local_data.py --lengths`（按源打印平均/中位文档长度）；
#     · 分词型语料（LCCC）必须加 --desegment，否则学到的是带空格文本的 merge；
#     · 数字整段保留（`\p{N}+`），不学 Qwen2 的逐位切分（那是为算术准备的，本项目不做数学）。
python scripts/train_tokenizer.py --out tokenizer --desegment \
    --source "parquet:$SNAP/silver--lccc/base/train/*.parquet:dialog@800000" \
    --source "jsonl:data/wiki/*.jsonl:text@38000" \
    --source "parquet:$SNAP/wangrui6--Zhihu-KOL/data/train-*.parquet:RESPONSE@160000"   # 其余见 run_pipeline.sh

# 1. 每个桶单独 tokenize，训练时再用权重混合，改配比不必重新 tokenize。八个桶在 run_pipeline.sh 步骤 1，
#    这里是 chat 桶的形状：
#      · cap 是**安全上限**而不是预算：放在该桶池子之上，`train_tokens` 就是各阶段共用的池子；卡在池子
#        下面就是有语料没被训练，而不是"配比修正"——权重只从这些池子派生（tools/plan_mix.py），不会向
#        任何一个桶要超过它自己的量。唯一的例外是 wiki：cap 有意低于池子，只读百度百科的常见实体部分；
#      · 顺序决定 cap 卡住时先读谁，以及 `val.bin` 取自谁：chat 的 val 来自 LCCC base，en_chat 的来自
#        TinyStoriesV2（它因此排第一），wiki 把量小的萌娘百科排在百度百科前面，否则 cap 会被百度百科
#        吃满、萌娘一个 token 都进不来。
python scripts/prepare_data.py --tokenizer tokenizer --desegment --out data/bin/chat --max_tokens 1210000000 --val_tokens 4000000 \
    --source "parquet:$SNAP/silver--lccc/base/train/*.parquet:dialog" \
    --source "parquet:$SNAP/silver--lccc/large/train/*.parquet:dialog" \
    --source "jsonl:$SNAP/chenrm--anime-subtitles/dialogue.jsonl:text" \
    --source "jsonl:data/chat/*.jsonl:text"

# 2. 预训练：三段课程。先把配方解析成可执行文件，再按段跑：
python tools/plan_mix.py --plan configs/pretrain_plan.json \
    --write runs/pretrain/plan.resolved.json --print-commands
#    这一步读各桶 meta.json 的 train_tokens，按 allocate 把每个桶摊到三段（A 只放长文三桶，C 按
#    chat / role-play 目标配比，B 拿剩下的），权重与 --tokens 都写进 plan.resolved.json。
#    三次 run 写同一个 --out_dir，后一段从 runs/pretrain/last 续；--tokens、形状、配比与 lr 全部来自配方，
#    学习率是一条以 token 为轴的全局 WSD（2% warmup → 恒定 → 最后 20% 衰减），不需要手写 --decay_frac 0：
python scripts/train.py --stage pretrain --plan runs/pretrain/plan.resolved.json --phase A \
    --tokenizer tokenizer --out_dir runs/pretrain --compile --resume
python scripts/train.py --stage pretrain --plan runs/pretrain/plan.resolved.json --phase B \
    --tokenizer tokenizer --out_dir runs/pretrain --compile --resume
python scripts/train.py --stage pretrain --plan runs/pretrain/plan.resolved.json --phase C \
    --tokenizer tokenizer --out_dir runs/pretrain --compile --resume
# 训练期间实时看指标与告警（细节见 doc/monitoring.md）：
#   python tools/watch_training.py --dir runs/pretrain

# 3.（可选）视觉对齐
python scripts/train.py --stage align --plan runs/pretrain/plan.resolved.json --phase align \
    --tokenizer tokenizer --init_from runs/pretrain/final \
    --out_dir runs/align --mm_data data/mm/chat_image.jsonl --image_root data/mm/images --resume

# 4. 对话 / 角色扮演 SFT：人类对话 + 人设 + 少量越界拒答
#    跳过视觉对齐时：--init_from runs/pretrain/final --train llm（数据须为纯文本）
#    --mm_weights 是配比旋钮：三项分别占 74% / 21% / 5%，不给就是"谁行数多谁说了算"。
#    前两项由 prepare_local_data.py 从快照落盘（human_dialog = LCCC base，persona = shibing624 +
#    Johnson8187 + LooksJuicy，后者的 instruction 已改名 system_prompt，否则整段人设会拼进 user 轮
#    并在每轮重复，见 datasets.md §1.6）；refuse.jsonl 没有现成来源，按 §4.4 自己写。
python scripts/train.py --stage sft --plan runs/pretrain/plan.resolved.json --phase sft \
    --tokenizer tokenizer --init_from runs/align/final --out_dir runs/sft \
    --mm_data data/sft/human_dialog.jsonl,data/sft/persona.jsonl,data/sft/refuse.jsonl \
    --mm_weights 0.74,0.21,0.05 --desegment --resume
```

- `--source` 支持 `parquet:`（本地 glob，字段可走 `a.b` 嵌套路径）、`jsonl:`、`txt:`，末尾 `@N` 限制文档数；
  字段按 `<glob>:<字段>` 给出，`-` 表示整条记录，`a.b` 走嵌套路径，`a+b` 按顺序拼接（见 `scripts/common_io.py`
  的模块 docstring；[datasets.md §1](datasets.md) 给了每个集的现成写法）。列表型字段（LCCC 的
  `dialog`、SODA 的 `dialogue`）会自动用换行拼接成一篇文档，所以不需要先写转换脚本；没有字段名时可先用
  `-` 取整条记录。文档之间用 `<|endoftext|>` 分隔，产物是扁平 uint16 token 流 + `meta.json`。
- **`--desegment`**：去掉汉字之间的空格，给分词型语料用（`silver/lccc` 的 `dialog`、以及任何由它派生的
  SFT 数据）。`train_tokenizer.py` / `prepare_data.py` / `train.py`（align、sft 阶段）都有这个开关；
  它只删"一侧是 CJK"的空格，英文与数字之间的空格不受影响。
- `--val_tokens` 会把前 N 个 token 切给 `val.bin`；训练时定期在验证集上打印 `loss` / `ppl`。对话模型的
  ppl 只能当作"没训崩"的 sanity check，选型看 §6。
- `--mm_data` 用逗号分隔，每一项既可以是本地 jsonl 路径（`data/sft/*.jsonl` 就是），也可以是
  `parquet:` / `jsonl:<glob>` 这类源；这类来源会先流式过一遍 `normalize_messages`，落到 `--mm_cache`
  （默认 `data/cache`，可复用）的一个 jsonl 里再训练，这样 `Dataset` 仍按字节偏移随机读、`--resume`
  精确，也不会把整个数据集塞进内存。无法映射成 user/self 的记录（角色写成 `assistant`、只有一轮没有回复、
  字段认不出来）在落盘和读取时都会被跳过，`prepare_local_data.py` 落盘时会打印同样的计数。
- 只有带 `"image"` 字段的记录才需要 `--image_root`；纯文本 SFT 不用给。
- **磁盘预算**：快照本体 **18.33 GiB**（只读，训练不复制它）；训练侧另需 `data/<bucket>/*.jsonl` 与
  `data/sft/*.jsonl` 清洗产物约 **2.7 GB**（SFT 的 human_dialog 1.6 GB + persona 65 MB + 各桶 1.1 GB）、
  token 分片 `data/bin/*`（uint16 即 2 B/token）约 **10.6 GB**（allocate 之和 ≈5.39B token），加 `last/`
  （~1.6 GB 优化器状态）与各阶段 `final/`，一次完整三阶段训练合计约 **15 GB**。
- **来源顺序是取舍**：`--source` 按给定顺序读、共享同一个 cap，cap 卡住就停在那个源中间。各桶实测的
  排序依据：

  | 源 | 排序依据 | cap 卡住时丢掉的部分 | 判断 |
  |---|---|---|---|
  | baidu-baike | 常见词 → 冷门实体（首片 `红色食品`/`丈夫`，末片 `林周风毛菊`/`新疆华凌工贸`） | 冷门实体；且 parquet 分支本身只覆盖原 json 的 30% | ✅ wiki 有意只读前 1.18B，丢掉冷门尾部 |
  | webnovel | 按书顺序（首片《(种田文系列)无尽相思风》第一章） | 书单靠后的部分 | ✅ 全量 10 片，cap 不卡 |
  | anime-subtitles | 按目录（首片 `剧场版(OVA)`，末片 `季度番(散)`） | 只取一部分就只有剧场版，没有 TV 系列 | ✅ 清洗产物跑全 60 片，全量保留 |
  | LCCC base/large | 无明显主题序 | — | ✅ 基本无偏（两者重叠 ~30%，在意唯一 token 时只用 large） |

  其余桶的 cap 都在池子之上，整份读入；只有 wiki 有意卡在池子下面，所以它把萌娘百科排在百度百科前面。

## 6. 评测

这个定位下，损失/ppl 不作为选型依据。至少覆盖：

| 维度 | 指标 | 做法 |
|---|---|---|
| 语言自然度 | 人类盲测胜率 | 同一上下文，人写回复 vs 模型回复，双盲二选一；目标是接近 50% 分不出 |
| 人设一致性 | 身份问答一致率 | 同一 persona 多轮问名字/年龄/职业；无人设时统计各会话给出的身份分布（应接近语料先验，且**不**出现 AI 自述） |
| 越界拒答 | 拒答率 / 误拒率 | 越界 prompt 库（数学/物理/代码/医疗/法律）上的拒答率，以及日常闲聊被误拒的比例 |
| 角色扮演 | 场景复述得分 | `MiniMaxAI/role-play-bench`（zh 1.6k / en 1.5k dialogues，各带场景 seed 与评分项）。扮演没有唯一正解，按 rubric 评，不用 ppl |
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
| `--plan` / `--phase` | 读 `tools/plan_mix.py` 解析出的配方：形状、配比、`--tokens` 与以 token 为轴的全局 WSD |
| `--mm_weights` | align/sft：每个 `--mm_data` 项的采样权重（不给则按行数） |
| `--no_monitor` / `--no_intervention` | 关掉指标流 / 保留告警但不自动回退与早停 |
| `--early_stop_patience` / `--spike_sigma` / `--lr_backoff` / `--backoff_steps` | 监控阈值与受控干预，见 [monitoring.md](monitoring.md) |
| `--precision` / `--gdn_impl` / `--grad_ckpt` / `--ce_ckpt` / `--compile` / `--tf32` | 精度与 kernel |
