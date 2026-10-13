set -euo pipefail
cd "$(dirname "$0")/.."

if [ -n "${PY:-}" ]; then :; elif [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python; fi
WORK=${1:-${SMOKE_DIR:-data/smoke}}
rm -rf "$WORK"
mkdir -p "$WORK"

T0=$SECONDS
step() {  # step <label> <command...>
  local label=$1; shift
  local t0=$SECONDS
  if ! "$@" >"$WORK/$label.log" 2>&1; then
    echo "FAILED: $label  (tail of $WORK/$label.log)"
    tail -20 "$WORK/$label.log"
    exit 1
  fi
  printf '  %3ds  %s\n' $((SECONDS - t0)) "$label"
}

echo "smoke test -> $WORK   (python: $PY)"
step 1-make-data "$PY" tools/make_synthetic.py --out "$WORK/data" --n_images 4 --n_text 800
step 2-tokenizer "$PY" scripts/train_tokenizer.py --out "$WORK/tok" --vocab_size 1024 \
    --source "jsonl:$WORK/data/text.jsonl:text@800"
step 3-prepare   "$PY" scripts/prepare_data.py --tokenizer "$WORK/tok" \
    --source "jsonl:$WORK/data/text.jsonl:text" --out "$WORK/bin" --max_tokens 120000 --val_tokens 1500
step 4-pretrain  "$PY" scripts/train.py --stage pretrain --tokenizer "$WORK/tok" --out_dir "$WORK/pretrain" \
    --data "$WORK/bin" --max_steps 3 --seq_len 512 --micro_batch 4 --grad_accum 2 --log_every 1 --workers 0
drop() { [ -n "${KEEP:-}" ] || rm -rf "$@"; }
drop "$WORK/pretrain/last"
step 5-align     "$PY" scripts/train.py --stage align --tokenizer "$WORK/tok" \
    --init_from "$WORK/pretrain/final" --out_dir "$WORK/align" --mm_data "$WORK/data/caps.jsonl" \
    --image_root "$WORK/data" --max_steps 2 --seq_len 512 --micro_batch 4 --grad_accum 2 --workers 0
drop "$WORK/align/last" "$WORK/pretrain/final"
step 6-sft       "$PY" scripts/train.py --stage sft --tokenizer "$WORK/tok" \
    --init_from "$WORK/align/final" --out_dir "$WORK/sft" --mm_data "$WORK/data/sft.jsonl" \
    --image_root "$WORK/data" --max_steps 2 --seq_len 512 --micro_batch 4 --grad_accum 2 --workers 0
drop "$WORK/sft/last" "$WORK/align/final"
step 7-sample    "$PY" scripts/sample.py --model "$WORK/sft/final" --prompt "What color is the shape?" \
    --image "$WORK/data/images/00000.png" --max_new_tokens 8

echo
echo "sample said: $(tail -1 "$WORK/7-sample.log")"
echo "OK: pretrain + align + sft + sample all ran in $((SECONDS - T0))s"
echo "    logs: $WORK/*.log   kept: $WORK/sft/final   ($(du -sh "$WORK" 2>/dev/null | cut -f1) on disk;"
echo "    KEEP=1 bash scripts/smoke_test.sh keeps every stage's checkpoints instead)"
