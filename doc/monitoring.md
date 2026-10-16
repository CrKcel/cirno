# Cirno-130M 训练监控

监控要回答三个问题：**这一步还正常吗**（指标流）、**哪里开始不对**（告警）、**不对的时候训练自己做了什么**
（受控干预）。三者是一套东西：`scripts/train.py` 把每个记录点写成结构化文件，`hvlm/monitor.py` 是判定与干预
逻辑，`tools/watch_training.py` 只是把这些文件渲染出来。核心逻辑不 import torch，所以无 GPU 也能测
（`tools/check_monitor.py`），看板也不需要任何额外依赖。

## 1. 训练写出的文件

每个 `--out_dir` 下同时有：

| 文件 | 内容 | 谁写 | 谁读 |
|---|---|---|---|
| `log.jsonl` | 旧格式的一行文本记录（loss / lr / gnorm / tok_s / mem） | 训练 | 人工、旧脚本 |
| `metrics.jsonl` | 结构化指标流，每个 `--log_every` 一条、每次验证一条 | 训练 | 看板、告警、离线分析 |
| `health.jsonl` | 异常与干预事件，一条一个 JSON | 训练 | 看板、`--no_intervention` 时的审计 |

`metrics.jsonl` 的记录（节选，完整字段见 `hvlm/monitor.py` 顶部 docstring）：

```json
{"t": 1792153504.1, "step": 2, "stage": "pretrain", "phase": "B", "tokens": 8192,
 "tokens_total": 5242880, "total_steps": 8220, "budget_tokens": 5390000000,
 "loss": 10.23, "lm": 7.64, "mtp": 8.63, "lr": 0.001, "lr_scale": 1.0, "gnorm": 13.3,
 "grad_scale": null, "tok_s": 11281.0, "step_s": 0.36, "mem_gb": 5.08, "elapsed_s": 0.72}
{"t": ..., "step": 500, "eval": {"chat": {"loss": 4.1, "ppl": 60.3}}, "best_val": 4.1}
```

约定两条：**读取方要容忍写到一半的行**（`read_records` 直接丢弃无法解析的尾行，所以看板可以在训练还在写的时候
读）；**文件只追加**，`--resume` 不清空，因此一个 `out_dir` 下的历史就是这条 run 的完整曲线。

`health.jsonl` 的事件：

```json
{"t": ..., "step": 40, "level": "warn", "code": "loss_spike",
 "message": "loss 40 above 18.2 (ema 5.0 sd 3.3); lr x0.5 for 300 steps",
 "action": "lr_backoff", "value": 40.0, "threshold": 18.2, "lr_scale": 0.5}
```

`level` 是 `info` / `warn` / `critical`，`code` 见 §3，`action` 是 `none` / `lr_backoff` / `stop`。

## 2. 实时看板

```bash
python tools/watch_training.py --dir runs/pretrain            # 原地刷新，Ctrl-C 退出
python tools/watch_training.py --dir runs/pretrain --once     # 打一份快照（写日志、CI 用）
python tools/watch_training.py --dir runs/sft --interval 5 --events 20
```

显示内容：阶段与步进度条、已训 token / 总预算（配方里 `budget_tokens`）、tok/s 与预计剩余时间、loss 及
EMA/min、lm / mtp / lr / gnorm / mem 的即时值与 sparkline、验证 loss/ppl 与历史最好、最近若干条告警。
`--dir` 也可以直接给 `metrics.jsonl` 的路径。旧 run 没有 `metrics.jsonl` 时自动回退到 `log.jsonl`。

看板只读：挂不挂它都不影响训练。

## 3. 判定规则

`HealthPolicy` 在每次记录（以及每次验证）后更新滑动统计并判定。所有阈值都是"同时越过两个条件"才触发，避免
把正常噪声当成异常；前 25 条记录完全不判定（`HealthPolicy.min_steps`），因为 EMA 还没有意义。

| code | 触发条件 | level | 动作 |
|---|---|---|---|
| `loss_spike` | loss > EMA + 4×最近标准差 **且** loss > 1.5×EMA | warn | `lr_backoff` |
| `grad_spike` | 裁剪前的 grad norm > 20×`--clip` | warn | `lr_backoff` |
| `nonfinite_gradient` | 梯度非有限（该步已被跳过） | critical | `lr_backoff` |
| `throughput_drop` | tok/s 连续 3 个记录点低于滑动均值的 50% | warn | `none` |
| `memory_pressure` | 峰值显存 > 预算的 95% | warn | `none` |
| `val_plateau` | 连续 `--early_stop_patience` 次验证没有新的最好 loss | critical | `stop` |

对应命令行开关：`--spike_sigma`、`--gnorm_spike_factor`、`--throughput_drop`、`--early_stop_patience`、
`--early_stop_min_frac`（默认 0.5，前一半步数里不因为验证集判停）。

`throughput_drop` 与 `memory_pressure` 只报不动手：前者通常是别的进程抢卡或数据加载变慢，后者说明该调
micro_batch / `--max_mem_gb`，都不是训练程序能自己决定的。

## 4. 受控自动干预

只有两个动作会改变训练，且都可逆：

1. **学习率回退**（`lr_backoff`）：每次触发把当前 lr 乘 `--lr_backoff`（默认 0.5），**保持**
   `min(--backoff_steps, 总步数/20)` 步后按每步 0.01 逐步释放回调度曲线，下限 `0.05`。保持时长被封顶，
   是因为一次回退如果占据整个 run 的很大一块，它就不是修正而是另一条调度曲线了。回退期间
   `lr_scale` 会写进 `metrics.jsonl`，回退原因与倍率写进 `health.jsonl`。
2. **提前停止**（`stop`）：只在显式给了 `--early_stop_patience N` 时生效，且不早于
   `--early_stop_min_frac × 总步数`。停止时 `last/` 与 `final/` 都写成停止点，并把原因打进 `health.jsonl`。

`--no_intervention` 保留全部告警（仍写 `health.jsonl`）但不回退、不停止，用于需要完全可复现的对比实验；
`--no_monitor` 连指标流都不写，退回只有 `log.jsonl` 的行为。

预训练阶段默认 `--early_stop_patience 0`（关闭）：验证集 ppl 在课程切换点本来就会跳，按它停下会打断配方。
SFT 阶段建议 `--early_stop_patience 4 --eval_every 100`。

## 5. 与训练配方的关系

`--plan` 运行时，`metrics.jsonl` 里的 `budget_tokens`、`total_steps` 与 `lr` 都来自
`runs/<stage>/plan.resolved.json`（`configs/pretrain_plan.json` 经 `tools/plan_mix.py` 解析）。三段预训练
是三次独立 run，但学习率是一条**以 token 为轴的全局 WSD 曲线**（`lr_at_tokens`）：warmup 2%、中段恒定、
最后 20% 衰减，衰减起点正好落在退火段的开始（0.8 × 5.07B = 4.06B）。所以监控看到的是连续的进度，而不是
三次互不相干的 run。

## 6. 常见读数

| 现象 | 含义 | 处理 |
|---|---|---|
| `gnorm` 长期尖峰、`grad_spike` 反复出现 | 峰值 lr 对当前配比偏高 | 把配方 `schedule.peak_lr` 降到 6e-4，或保留回退让监控自动压 |
| `throughput_drop` 单个记录点 | 一般是别的进程占用 GPU / 数据加载抖动 | 看 `step_s` 与 `nvidia-smi`；持续出现再查 `--workers` |
| `mem_gb` 逼近预算并出现 `memory_pressure` | 形状压得太满 | 降 `--micro_batch` 或提 `--grad_accum`，重跑 `tools/measure_batch.py` |
| `val_plateau` 但 train loss 还在降 | 过拟合/数据重复 | 加数据多样性，不要加 epoch（`doc/training.md` §4.2） |
| `loss_spike` 后 lr 回退，loss 没有恢复 | 数据里有坏批或 lr 不稳 | 看 `health.jsonl` 的 step，抽查该步附近的数据分片 |

## 7. 校验

```bash
python tools/check_monitor.py     # 无 torch：EMA、各类判定、回退曲线、指标流解析、看板渲染、配方分配
bash scripts/smoke_test.sh        # 端到端：训练写出 metrics.jsonl/health.jsonl，看板 --once 能读出来
```

`tools/check_monitor.py` 覆盖的是可观察契约：稳定 loss 不误报、8 倍尖峰必报、`--no_intervention` 只报不动、
回退保持后逐步释放、验证集连续不改善才停、写到一半的 JSONL 行不影响读取、看板对空 run 也不会崩。
