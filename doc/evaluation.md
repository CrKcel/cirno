# Cirno-130M 评测

这个定位下损失与 ppl 只回答"有没有训崩"（[monitoring.md](monitoring.md)），选型与发布看生成结果。评测因此是
一个独立环节：对任意 checkpoint 目录跑一遍探针集，用生成而不是 ppl 打分，产出报告、门限判定、人工盲测素材
与端侧成本数字。

## 1. 位置与产物

```bash
python scripts/eval.py --model runs/sft/final --tokenizer tokenizer --out runs/eval/sft
python scripts/eval.py --model runs/sft/final --out runs/eval/sft --sets knowledge --limit 20   # 单集快跑
python scripts/eval.py --model runs/sft/final --out runs/eval/sft --blind 30 --bench            # 盲测 + 成本
python tools/eval_table.py runs/eval/*/report.json --baseline runs/eval/pretrain/report.json    # 跨 checkpoint 对比
```

评测不进 `scripts/train.py` 的步进循环：生成式评分是分钟级开销，且它面向的是产物（`final/`、`last/`、任意
checkpoint），而不是某一次 run。配置在 `configs/eval.json`（集合、每个 kind 的采样参数、门限），与训练配方
一样是文件而不是命令行。

一次 run 在 `runs/eval/<tag>/` 下写四类产物：

| 文件 | 内容 |
|---|---|
| `report.json` | 元信息（模型、tokenizer、git、精度、kernel、seed）、逐集指标、门限判定、成本与盲测块 |
| `report.md` | 同一份内容的人类可读表格 |
| `generations.jsonl` | 每个探针的上下文与全部采样，模式打分有误判时用来人工抽查 |
| `cost.json` | `--bench` 的加载耗时、prefill、无缓存 decode tok/s、峰值显存 |
| `blind.json` + `blind.key.json` | `--blind N` 的 A/B 盲测素材与答案键（哪一臂是模型） |

`report.json` 记下 seed 与逐样本 seed（`seed:集合:条目:序号`），同一条命令重跑得到同一份报告。

## 2. 评测集

探针是仓库内自撰内容（`evals/*.jsonl`），不来自训练快照，因而不会与语料混同；格式与校验由
`hvlm/eval_core.py` 定义，`tools/check_eval.py` 每次校验全部文件。

| 集合 | 规模 | 记录字段 | 打分 |
|---|---|---|---|
| `evals/knowledge.jsonl` | 150 题：`fact` 60 / `weather` 20 / `everyday` 20 / `unknown` 50 | `id, category, prompt, any_of[], not_any_of[], expect_unknown, context?, image?` | 命中率、`unknown` 组的正确拒答率与硬答率 |
| `evals/persona.jsonl` | 14 条记录 / 59 个探针：8 条人设（每条 4–5 探针）+ 6 条无人设 | `persona`（可为 null）、`probes[{field, question, any_of?}]` | 身份答对率、K 次采样一致性、AI 自述率、无人设时的身份分布 |
| `evals/imform.jsonl` | 60 条（闲聊/情绪/身边事/跳题/游戏内） | `id, tags, messages[]`（末条为 user，约 1/4 带 system 人设） | 回复长度分布、助理腔、复读、空回复 |
| `evals/roleplay.jsonl` | 20 个场景 / 62 条 rubric | `persona, history[], rubric[{name, any_of?/not_any_of?}]` | 逐项与整体 rubric 通过率 |

匹配口径是归一化之后的子串（NFKC、去大小写与标点、去空白），所以 `１２` 与 `12`、`24。` 与 `24` 等价；
中文数字不转换，数字类答案要在 `any_of` 里同时列出阿拉伯与中文写法（`24`、`二十四`）；
`not_any_of` 用来排除常见错误答案。`knowledge` 的 50 条 `unknown` 覆盖私人信息、实时信息与无法确定的问题，
预期行为是"不知道/说不准"，凭空给答案记 `over_answer`。rubric 用确定性模式匹配（口癖、称呼、出戏词），
不依赖 LLM judge；需要更强评判时属于外部环节。

`imform` 的长度分布对比需要人类侧先验：`runs/eval/imform_prior.json` 由 SFT 人类对话切片统计而来，
`run_pipeline.sh` 步骤 1b 生成。缺这份文件时长度比值留空，其余指标照常。

## 3. 指标与门限

每个集合的指标块都带一组通用风格指标：`tone_rate`（markdown 标题/有序列表/"很高兴为您服务"这类助理腔）、
`self_reference_rate`（自称 AI/助手）、`empty_rate`、`repeat_rate`（4-gram 复读）、`length`（token 分位与
`long_share`，即 >128 token 的占比）。集合自身的指标：

| 指标 | 含义 |
|---|---|
| `knowledge.accuracy` | `fact`/`weather`/`everyday` 的命中率（`by_category` 分组给出） |
| `knowledge.unknown_rate` / `over_answer_rate` | 该答"不知道"判对的比例 / 硬答的比例 |
| `persona.identity_accuracy` | 探针答案命中人设卡的比例 |
| `persona.consistency` | 同一探针 K 次采样互相一致的比例（`by_field` 分组） |
| `persona.identity_distribution` | 无人设记录自报身份的分布，用来确认它接近语料先验且不自称 AI |
| `imform.median_ratio` / `p90_ratio` | 回复长度中位/p90 与人类先验的比值，1.0 表示同分布 |
| `roleplay.rubric_pass_rate` / `overall_pass_rate` | 单项通过率 / 全项通过率 |

`configs/eval.json` 的 `gates` 只在词表可靠的维度设为发布门限：`imform.tone_rate ≤ 0.02`、
`imform.self_reference_rate ≤ 0.02`、`imform.empty_rate ≤ 0.01`、`persona.self_reference_rate ≤ 0.02`。
`--gate` 让不达标时退出码非 0（供 CI 与冒烟测试用）；该集合跑了但指标缺失按失败处理，集合本身没跑
（`--sets` 只选了部分集合）则跳过该项。其余指标（通识命中率、人设
一致率、长度比、rubric 通过率）只用于看趋势：它们受采样温度与模型规模影响，做发布门槛会掩盖真实回归。

门限阈值按当前定位给出，首个 SFT checkpoint 出来后按实测校准；`report.json` 里每次判定都记了值、约束与结论。

## 4. 人工盲测

自动化指标不能回答"像不像人说话"。`--blind N` 生成 N 组 A/B：同一个上下文下，一臂是模型采样，另一臂是
`--blind_human`（默认 `data/sft/human_dialog.jsonl`）里真实人类回复；哪一臂是模型只写在 `blind.key.json`。

```bash
python scripts/eval.py --model runs/sft/final --out runs/eval/sft --blind 30
python tools/blind_test.py --dir runs/eval/sft      # 逐条 a/b/tie/skip，随手 q 退出，已选项落盘
python tools/blind_test.py --score runs/eval/sft    # 胜率写回 report.json / report.md
```

打分给出模型胜率、人类胜率、平局率与 `parrot_rate`（两臂近似逐字相同的比例）。人类臂取自训练语料，所以
模型可能背下它：`parrot_rate` 是这条捷径的可见信号，读胜率时一起看。目标是模型胜率接近 0.5——人分不出
哪条是模型写的。

## 5. 端侧成本

`--bench` 报加载耗时、参数量、prompt token 数、prefill 毫秒、decode tok/s 与峰值显存。当前
`generate_simple` 每步重算前缀（KV/状态缓存见 [progress.md](progress.md)），所以这里的 decode 吞吐是无缓存
上界；真实端侧数字需要在目标设备上、且缓存落地后测，`cost.json` 的 `note` 也这么记录。版本之间横向对比时
它是同口径的相对量。

## 6. 校验

```bash
python tools/check_eval.py     # 无 torch：归一化/匹配/词表、逐 kind 打分与聚合、门限、报告渲染、schema 拒绝、盲测打分、已提交集合校验
python tests/test_model.py -k eval   # 提示词与训练编码同源、同 seed 采样可复现
bash scripts/smoke_test.sh     # 端到端：synthetic 四类探针 + --bench + --gate 退出码与报告一致
```

覆盖的可观察契约：答案命中与否定守卫、`unknown` 与硬答的区分、人设跨采样一致性、复读检测、聚合数值、
门限的通过/失败/缺指标、空报告不崩、盲测胜率与 parrot 计数，以及 `evals/*.jsonl` 整体通过 schema（集合
内容坏了在 CI 里就暴露，而不是等训练完才发现）。

## 7. 未覆盖

- **越界拒答/误拒率**：没有可用的越界 prompt 集，`refusal` 类指标与门限暂缺；接入时补一个
  `evals/refusal.jsonl`（`in_scope`/`out_of_scope` 两分）与对应打分分支即可。
- **LLM judge**：rubric 与自然度都用确定性规则；需要更强评判时是外部环节。
- **视觉**：评测支持 `image` 字段与 `--image_root`（`tools/make_synthetic.py` 的合成探针覆盖这条路径），
  真实图文集的导入仍受 [progress.md](progress.md) 里"图片存在数据集内部"的限制。
