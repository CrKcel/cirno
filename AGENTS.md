# Repository Guidelines

Cirno-130M is a ~130M-parameter hybrid vision-language model (Gated DeltaNet + Gated Attention)
in Python/PyTorch, positioned as an **on-device chat / role-play model** rather than a general assistant.

## Project Structure & Module Organization

- `hvlm/` — model package. `modeling_hvlm.py` holds the model, ViT, MTP block, and chunked CE;
  `gated_delta.py` the delta rule and kernel dispatch; `configuration_hvlm.py` the configs;
  `monitor.py` the training-health policy and the metric/event streams.
  `positions.py`, `data_core.py` and `monitor.py` are numpy-only and must stay importable without torch.
- `scripts/` — CLI entry points: `train_tokenizer.py`, `prepare_data.py`, `train.py`, `sample.py`,
  plus the `smoke_test.sh` and `run_pipeline.sh` orchestration scripts.
- `tools/` — standalone estimators and checks: `estimate.py`, `measure_batch.py`,
  `check_algorithms_numpy.py`, `check_data_numpy.py`, `check_monitor.py`, `make_synthetic.py`,
  `plan_mix.py` (resolves `configs/pretrain_plan.json` against the bins' `meta.json` into the runnable
  `plan.resolved.json`), `watch_training.py` (terminal dashboard over the metric streams).
- `tests/` — `test_model.py`, the torch test suite.
- `configs/` — `hvlm_130m.json` model config and `pretrain_plan.json`, the training recipe (phase
  budgets, shapes, per-bin token allocations, mixture targets). `data/` and `runs/` hold generated
  artifacts and are gitignored.
- `doc/` — the long-form docs: `architecture.md` (model design, parameter count, memory and
  throughput, precision/kernel choices), `training.md` (design positioning, the three training
  stages, data prep, evaluation, edge inference, training flags), `datasets.md` (which open datasets
  each stage points at, their source specs, licences and caveats), `monitoring.md` (metric streams,
  dashboard, health thresholds and the controlled interventions), `progress.md` (what is done, known
  limitations, TODO). Keep them in sync with the code.

`hvlm/__init__.py` loads model classes lazily; keep torch imports out of `positions.py` and
`data_core.py`.

## Build, Test, and Development Commands

```bash
pip install -r requirements.txt
pip install flash-linear-attention          # optional CUDA Triton kernels for Gated DeltaNet
python tools/check_algorithms_numpy.py      # chunked delta rule vs recurrence, M-RoPE (no torch)
python tools/check_data_numpy.py            # data pipeline (no torch)
python tools/check_monitor.py               # health policy, metric streams, dashboard, recipe (no torch)
python tools/estimate.py configs/hvlm_130m.json
python tools/plan_mix.py --plan configs/pretrain_plan.json   # resolve the recipe against the bins' meta.json
python tests/test_model.py                  # torch suite
python tests/test_model.py -k mtp           # run tests whose name contains "mtp"
bash scripts/smoke_test.sh                  # synthetic end-to-end run into data/smoke/
bash scripts/run_pipeline.sh                # full training recipe
```

## Usage

The whole pipeline is three stages — `pretrain` (text), `align` (vision, LLM frozen; optional, only
when the deployment sees images) and `sft` (dialogue / role-play fine-tuning on human-to-human
conversations) — driven by `scripts/train.py --stage <stage>`; see `doc/training.md` for the
positioning, per-stage hyperparameters and evaluation, and `doc/architecture.md` for the
memory/throughput trade-offs.

```bash
# tokenizer -> per-source token bins -> pretrain (three phases) -> (optional) align -> sft -> sample
python scripts/train_tokenizer.py --out tokenizer --desegment --source "jsonl:data/chat/*.jsonl:text@3200000"
python scripts/prepare_data.py --tokenizer tokenizer --desegment --source "jsonl:data/chat/*.jsonl:text" --out data/bin/chat --max_tokens 1014000000 --val_tokens 4000000
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
    --mm_weights 0.74,0.21,0.05 --resume
python scripts/sample.py --model runs/sft/final --prompt "在吗" --max_new_tokens 32
```

To skip the vision stage (text-only chat), drop the `align` line and use
`--init_from runs/pretrain/final --train llm` for `sft` with pure-text data.

`bash scripts/run_pipeline.sh` runs the same recipe end to end; edit its DATA section for real
corpora. Each stage writes `out_dir/last` (weights + optimizer + GradScaler state, used by
`--resume`) and `out_dir/final` (weights + tokenizer, used by the next stage's `--init_from`).

Useful flags beyond the per-stage defaults: `--compile` for speed, `--seq_len / --micro_batch /
--grad_accum` for the shape, `--max_mem_gb 0` to drop the allocator cap, and `--resume` to continue
(with the data order and GradScaler state preserved). Precision and kernel switches (`--precision`,
`--gdn_impl`, `--grad_ckpt`, `--ce_ckpt`, `--tf32`) are documented in `doc/architecture.md` §3.
`--plan / --phase` read the resolved recipe (phase budget, mixture, and one WSD curve over the whole
token budget); `--mm_weights` sets the align/SFT source mixture. Monitoring switches
(`--no_monitor`, `--no_intervention`, `--early_stop_patience`, `--lr_backoff`) are documented in
`doc/monitoring.md`.

## Verification

1. `tools/check_algorithms_numpy.py`, `tools/check_data_numpy.py`, `tools/check_monitor.py`,
   `tools/estimate.py` — no torch needed; all pass in the sandbox.
2. `tests/test_model.py` — 26 tests, ~19 s on the RTX 4060 Laptop (GPU paths included). Coverage:
   chunked delta rule ≡ recurrence (forward and gradients), parameter count matches the estimator,
   causality (including images), right padding is inert, MTP label alignment, chunked CE ≡ full CE
   (loss and gradients), gradient/CE checkpoint parity, bf16 autocast, align-stage trains only the
   ViT, save/load round-trip, tiny-model overfit. Kernel and precision tests:
   `test_gdn_impl_dispatch` (`auto` must fall back to torch on CPU or without fla),
   `test_fla_matches_torch_impl` (Triton kernels vs the fp32 reference, error bound),
   `test_ce_checkpoint_only_changes_recompute` (`--ce_ckpt` only moves the recompute: loss and
   gradients bit-identical), `test_probe_batch_is_a_valid_model_input` (the memory probe's synthetic
   batch must really be a valid model input). With a GPU it also measures peak memory.
3. `scripts/smoke_test.sh` — chains `make_synthetic.py` → tiny tokenizer → `prepare_data` →
   2–3 steps each of pretrain/align/sft → `sample.py`, printing per-step timings (~45 s on the
   RTX 4060 Laptop here). It only verifies that the pipeline runs end to end, not that anything is
   learned: the final sample uses random weights and is expected to be gibberish. To see the vision
   path actually learn, run it with
   `KEEP=1` and train a few hundred more steps on the `data/smoke` data (the synthetic answers are
   deterministic: colors and shapes). The script deletes each stage's checkpoint as soon as the next
   one has loaded it (`last/` carries ~1.6 GB of optimizer state), so a full pass uses ~0.5 GB of
   disk and keeps only `sft/final`; `KEEP=1` keeps everything. It also asserts the monitoring
   contract: `metrics.jsonl` / `health.jsonl` are written and `tools/watch_training.py --once` reads
   them back at the right step.
4. Before a real `run_pipeline.sh` run, confirm that the logged `mem_gb` and `nvidia-smi` both stay
   inside the budget, and record `tok_s`.

## Testing Guidelines

- Tests are plain `test_*` functions in `tests/test_model.py`, discovered by the file's own `main()`;
  pytest is not used. Run with `python tests/test_model.py` (optionally `-k <substring>`).
- Add tests for real regressions, invariants, and boundaries, such as chunk-vs-recurrent equivalence,
  causality, or checkpoint/gradient parity. Avoid tests that only restate literals or mappings.
- Keep logic that must run without torch covered by the `tools/check_*.py` scripts.
