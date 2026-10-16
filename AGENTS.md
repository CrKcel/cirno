# Repository Guidelines

Cirno-130M is a ~130M-parameter hybrid vision-language model (Gated DeltaNet + Gated Attention)
in Python/PyTorch, positioned as an **on-device chat / role-play model** rather than a general assistant.

## Project Structure & Module Organization

- `hvlm/` — model package. `modeling_hvlm.py` holds the model, ViT, MTP block, and chunked CE;
  `gated_delta.py` the delta rule and kernel dispatch; `configuration_hvlm.py` the configs;
  `monitor.py` the training-health policy and the metric/event streams; `eval_core.py` the evaluation
  schema, scoring, gates and report rendering.
  `positions.py`, `data_core.py`, `monitor.py` and `eval_core.py` are numpy-only and must stay importable
  without torch.
- `scripts/` — CLI entry points: `train_tokenizer.py`, `prepare_data.py`, `prepare_local_data.py`
  (turns the local dataset snapshot into the `data/<bucket>/*.jsonl` the recipe reads), `train.py`,
  `sample.py`, `eval.py` (generation-based evaluation of a checkpoint), plus the `smoke_test.sh` and
  `run_pipeline.sh` orchestration scripts.
- `tools/` — standalone estimators and checks: `estimate.py`, `measure_batch.py`,
  `check_algorithms_numpy.py`, `check_data_numpy.py`, `check_local_data.py`, `check_monitor.py`,
  `check_eval.py`, `make_synthetic.py`, `plan_mix.py` (resolves `configs/pretrain_plan.json` against the bins' `meta.json` into the runnable
  `plan.resolved.json`), `watch_training.py` (terminal dashboard over the metric streams),
  `eval_table.py` (cross-checkpoint metric table), `blind_test.py` (human A/B labelling and scoring).
- `evals/` — the authored evaluation probes (`knowledge.jsonl`, `persona.jsonl`, `imform.jsonl`,
  `roleplay.jsonl`); their schema and metrics are `hvlm/eval_core.py`, see `doc/evaluation.md`.
- `tests/` — `test_model.py`, the torch test suite.
- `configs/` — `hvlm_130m.json` model config, `pretrain_plan.json`, the training recipe (phase
  budgets, shapes, per-bin token allocations, mixture targets) and `eval.json` (eval sets, sampling
  defaults, gates). `data/` and `runs/` hold generated artifacts and are gitignored.
- Training data is the local Hugging Face snapshot (`CIRNO_SNAPSHOT`, default `/mnt/data/LLM/hf`, one
  directory per repo); `run_pipeline.sh` exports the `*_OFFLINE` switches, so a residual `hf:` spec fails
  instead of downloading. `doc/datasets.md` maps every dataset to its snapshot path.
- `doc/` — the long-form docs: `architecture.md` (model design, parameter count, memory and
  throughput, precision/kernel choices), `training.md` (design positioning, the three training
  stages, data prep, evaluation, edge inference, training flags), `evaluation.md` (probe sets, metrics
  and gates, the blind test, edge cost), `datasets.md` (which open datasets
  each stage points at, their source specs, licences and caveats), `monitoring.md` (metric streams,
  dashboard, health thresholds and the controlled interventions), `progress.md` (what is done, known
  limitations, TODO). Keep them in sync with the code.

`hvlm/__init__.py` loads model classes lazily; keep torch imports out of `positions.py`,
`data_core.py`, `monitor.py` and `eval_core.py`.

## Build, Test, and Development Commands

```bash
pip install -r requirements.txt
pip install flash-linear-attention          # optional CUDA Triton kernels for Gated DeltaNet
python tools/check_algorithms_numpy.py      # chunked delta rule vs recurrence, M-RoPE (no torch)
python tools/check_data_numpy.py            # data pipeline (no torch)
python tools/check_monitor.py               # health policy, metric streams, dashboard, recipe (no torch)
python tools/check_local_data.py            # offline data path: cleaners, snapshot inputs, recipe wiring
python tools/check_eval.py                  # eval schema, scoring, gates, the probes in evals/ (no torch)
python tools/estimate.py configs/hvlm_130m.json
python tools/plan_mix.py --plan configs/pretrain_plan.json   # resolve the recipe against the bins' meta.json
python tests/test_model.py                  # torch suite
python tests/test_model.py -k mtp           # run tests whose name contains "mtp"
python scripts/eval.py --model runs/sft/final --out runs/eval/sft --bench   # evaluate a checkpoint
python tools/eval_table.py runs/eval/*/report.json          # compare checkpoints / stages
bash scripts/smoke_test.sh                  # synthetic end-to-end run into data/smoke/
bash scripts/run_pipeline.sh                # full training recipe
```

## Usage

The whole pipeline is three stages — `pretrain` (text), `align` (vision, LLM frozen; optional, only
when the deployment sees images) and `sft` (dialogue / role-play fine-tuning on human-to-human
conversations) — driven by `scripts/train.py --stage <stage>`, with a generation-based evaluation pass
over each stage's artifact as a separate fourth step; see `doc/training.md` for the
positioning, per-stage hyperparameters and evaluation, `doc/evaluation.md` for the probes, metrics and
gates, and `doc/architecture.md` for the memory/throughput trade-offs.

```bash
# tokenizer -> per-source token bins -> pretrain (three phases) -> (optional) align -> sft -> eval -> sample
export CIRNO_SNAPSHOT=/mnt/data/LLM/hf      # local snapshot; every source is a path inside it
python scripts/prepare_local_data.py --snapshot "$CIRNO_SNAPSHOT" --out data   # cleaning -> data/<bucket>/*.jsonl
python scripts/train_tokenizer.py --out tokenizer --desegment --source "jsonl:data/chat/*.jsonl:text@8000"
python scripts/prepare_data.py --tokenizer tokenizer --desegment --source "parquet:$CIRNO_SNAPSHOT/silver--lccc/base/train/*.parquet:dialog" --source "jsonl:data/chat/*.jsonl:text" --out data/bin/chat --max_tokens 1210000000 --val_tokens 4000000
# the recipe declares each phase's budget, shape and mixture; plan_mix.py fills in the weights from the
# bins' meta.json and writes runs/pretrain/plan.resolved.json, which every phase then executes
python tools/plan_mix.py --plan configs/pretrain_plan.json --write runs/pretrain/plan.resolved.json --print-commands
python scripts/train.py --stage pretrain --plan runs/pretrain/plan.resolved.json --phase A \
    --tokenizer tokenizer --out_dir runs/pretrain --compile --resume
python scripts/train.py --stage pretrain --plan runs/pretrain/plan.resolved.json --phase B \
    --tokenizer tokenizer --out_dir runs/pretrain --compile --resume
python scripts/train.py --stage pretrain --plan runs/pretrain/plan.resolved.json --phase C \
    --tokenizer tokenizer --out_dir runs/pretrain --compile --resume
python tools/watch_training.py --dir runs/pretrain        # live metrics + alerts (--once for a snapshot)
python scripts/train.py --stage align --plan runs/pretrain/plan.resolved.json --phase align \
    --tokenizer tokenizer --init_from runs/pretrain/final --out_dir runs/align \
    --mm_data data/mm/chat_image.jsonl --image_root data/mm/images --resume
python scripts/train.py --stage sft --plan runs/pretrain/plan.resolved.json --phase sft \
    --tokenizer tokenizer --init_from runs/align/final --out_dir runs/sft \
    --mm_data data/sft/human_dialog.jsonl,data/sft/persona.jsonl,data/sft/refuse.jsonl \
    --mm_weights 0.74,0.21,0.05 --desegment --resume
python scripts/sample.py --model runs/sft/final --prompt "在吗" --max_new_tokens 32
python scripts/eval.py --model runs/sft/final --tokenizer tokenizer --out runs/eval/sft --blind 30
python tools/blind_test.py --dir runs/eval/sft              # label the pairs, then --score the win rate
```

`scripts/eval.py` is the fourth step and stays out of the training loop: it loads any checkpoint
directory, scores generation against `evals/*.jsonl` and writes `runs/eval/<tag>/report.json`,
`report.md`, `generations.jsonl` and (with `--bench` / `--blind`) `cost.json` / `blind.json`.
`tools/eval_table.py` compares those reports across stages; `configs/eval.json` holds the sets, sampling
defaults and the gates `--gate` enforces. See `doc/evaluation.md`.

To skip the vision stage (text-only chat), drop the `align` line and use
`--init_from runs/pretrain/final --train llm` for `sft` with pure-text data.

`bash scripts/run_pipeline.sh` runs the same recipe end to end against the snapshot (set
`CIRNO_SNAPSHOT` if it lives elsewhere). Each stage writes `out_dir/last` (weights + optimizer + GradScaler state, used by
`--resume`) and `out_dir/final` (weights + tokenizer, used by the next stage's `--init_from`).

Useful flags beyond the per-stage defaults: `--compile` for speed, `--seq_len / --micro_batch /
--grad_accum` for the shape, `--max_mem_gb 0` to drop the allocator cap, and `--resume` to continue
(with the data order and GradScaler state preserved). Precision and kernel switches (`--precision`,
`--gdn_impl`, `--tf32`) are documented in `doc/architecture.md` §3.
`--plan / --phase` read the resolved recipe (phase budget, mixture, and one WSD curve over the whole
token budget); `--mm_weights` sets the align/SFT source mixture. Monitoring switches
(`--no_monitor`, `--no_intervention`, `--early_stop_patience`, `--lr_backoff`) are documented in
`doc/monitoring.md`.

## Verification

1. `tools/check_algorithms_numpy.py`, `tools/check_data_numpy.py`, `tools/check_local_data.py`,
   `tools/check_monitor.py`, `tools/check_eval.py`, `tools/estimate.py` — no torch needed; all pass in the
   sandbox.
2. `tests/test_model.py` — 25 tests, ~19 s on the RTX 4060 Laptop (GPU paths included). Coverage:
   chunked delta rule ≡ recurrence (forward and gradients), parameter count matches the estimator,
   causality (including images), right padding is inert, MTP label alignment, chunked CE ≡ full CE
   (loss and gradients), bf16 autocast, align-stage trains only the ViT, save/load round-trip,
   tiny-model overfit. Kernel and precision tests: `test_gdn_impl_dispatch` (`auto` must fall back to
   torch on CPU or without fla), `test_fla_matches_torch_impl` (Triton kernels vs the fp32 reference,
   error bound). Evaluation tests:
   `test_eval_prompt_matches_training_encoding` (the generation prompt is the training-time prefix:
   eval and training cannot drift apart) and `test_eval_generation_is_reproducible`
   (same seed → same sample, so report.json's numbers can be reproduced). With a GPU it also measures
   peak memory.
3. `scripts/smoke_test.sh` — chains `make_synthetic.py` → tiny tokenizer → `prepare_data` →
   2–3 steps each of pretrain/align/sft → `sample.py` → `scripts/eval.py`, printing per-step timings
   (~45 s on the RTX 4060 Laptop here). It only verifies that the pipeline runs end to end, not that
   anything is learned: the final sample uses random weights and is expected to be gibberish. To see the
   vision path actually learn, run it with
   `KEEP=1` and train a few hundred more steps on the `data/smoke` data (the synthetic answers are
   deterministic: colors and shapes). The script deletes each stage's checkpoint as soon as the next
   one has loaded it (`last/` carries ~1.6 GB of optimizer state), so a full pass uses ~0.5 GB of
   disk and keeps only `sft/final`; `KEEP=1` keeps everything. It also asserts two contracts: the
   monitoring one (`metrics.jsonl` / `health.jsonl` are written and `tools/watch_training.py --once`
   reads them back at the right step) and the evaluation one (all four probe kinds scored, report /
   generations / cost written, `--gate`'s exit code agreeing with the report).
4. Before a real `run_pipeline.sh` run, confirm that the logged `mem_gb` and `nvidia-smi` both stay
   inside the budget, and record `tok_s`.

## Testing Guidelines

- Tests are plain `test_*` functions in `tests/test_model.py`, discovered by the file's own `main()`;
  pytest is not used. Run with `python tests/test_model.py` (optionally `-k <substring>`).
- Add tests for real regressions, invariants, and boundaries, such as chunk-vs-recurrent equivalence,
  causality, or checkpoint/gradient parity. Avoid tests that only restate literals or mappings.
- Keep logic that must run without torch covered by the `tools/check_*.py` scripts.
