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
    --source "jsonl:$WORK/data/text.jsonl:text" --source "jsonl:$WORK/data/text.jsonl:text" \
    --out "$WORK/bin" --max_tokens 120000 --val_tokens 1500
# --source is repeatable; a regression to a plain (non-append) argument silently drops every source but
# the last, which would just look like "less data" everywhere else.  meta.json is where it shows up.
"$PY" - "$WORK/bin/meta.json" <<'EOF'
import json, sys
src = json.load(open(sys.argv[1]))["source"]
assert src.count(",") == 1, f"prepare_data --source is not appendable any more: {src!r}"
EOF
step 4-pretrain  "$PY" scripts/train.py --stage pretrain --tokenizer "$WORK/tok" --out_dir "$WORK/pretrain" \
    --data "$WORK/bin" --max_steps 3 --seq_len 512 --micro_batch 4 --grad_accum 2 --log_every 1 --workers 0
# The monitoring contract: every run writes metrics.jsonl + health.jsonl, and the dashboard reads them
# back.  A regression here (renamed stream, unreadable JSONL) is invisible in the loss line above.
"$PY" - "$WORK/pretrain" <<'EOF'
import json, os, subprocess, sys
d = sys.argv[1]
metrics = [json.loads(line) for line in open(os.path.join(d, "metrics.jsonl")) if line.strip()]
assert metrics, "metrics.jsonl is empty"
assert any("loss" in m and "step" in m for m in metrics), metrics
assert os.path.exists(os.path.join(d, "health.jsonl")), "health.jsonl missing"
out = subprocess.run([sys.executable, "tools/watch_training.py", "--dir", d, "--once", "--no-color"],
                     capture_output=True, text=True, check=True).stdout
assert "loss" in out and "step 3/3" in out, out
print("  monitoring: metrics.jsonl, health.jsonl and the dashboard all agree")
EOF
drop() { [ -n "${KEEP:-}" ] || rm -rf "$@"; }
drop "$WORK/pretrain/last"
# The intervention path has to reach health.jsonl, not just the policy's unit tests: an lr far outside the
# stable range makes gradients non-finite, which must be reported and answered with a backoff.
step 4b-alert    "$PY" scripts/train.py --stage pretrain --tokenizer "$WORK/tok" --out_dir "$WORK/alert" \
    --data "$WORK/bin" --max_steps 2 --seq_len 512 --micro_batch 4 --grad_accum 2 --log_every 1 \
    --workers 0 --lr 1e6 --backoff_steps 2
"$PY" - "$WORK/alert" <<'EOF'
import json, os, sys
events = [json.loads(line) for line in open(os.path.join(sys.argv[1], "health.jsonl")) if line.strip()]
assert any(e["code"] == "nonfinite_gradient" and e["action"] == "lr_backoff" for e in events), events
print("  monitoring: a diverging lr is detected, reported and backed off")
EOF
drop "$WORK/alert"
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

# 8. Evaluation is a separate pass over a checkpoint (doc/evaluation.md): it must score generation rather
#    than ppl, cover every kind of probe and agree with its own --gate verdict.  The synthetic probes carry
#    known answers, so a random-weight model still exercises the whole scoring path.
rc=0
"$PY" scripts/eval.py --model "$WORK/sft/final" --tokenizer "$WORK/tok" --out "$WORK/eval" \
    --set knowledge="$WORK/data/eval_knowledge.jsonl" --set persona="$WORK/data/eval_persona.jsonl" \
    --set imform="$WORK/data/eval_imform.jsonl" --set roleplay="$WORK/data/eval_roleplay.jsonl" \
    --image_root "$WORK/data" --max_new_tokens 8 --bench --gate >"$WORK/8-eval.log" 2>&1 || rc=$?
"$PY" - "$WORK/eval" "$rc" <<'EOF'
import json, os, sys
d, rc = sys.argv[1], int(sys.argv[2])
report = json.load(open(os.path.join(d, "report.json"), encoding="utf-8"))
assert set(report["sets"]) == {"knowledge", "persona", "imform", "roleplay"}, list(report["sets"])
for name, block in report["sets"].items():
    assert block["n_items"] > 0 and block["n_generations"] == block["n_items"] * block["sampling"]["samples"], block
    assert block["skipped"] == 0 and block["metrics"]["n_samples"] == block["n_generations"], block
rows = [json.loads(l) for l in open(os.path.join(d, "generations.jsonl"), encoding="utf-8") if l.strip()]
assert len(rows) == sum(b["n_items"] for b in report["sets"].values()), len(rows)
assert os.path.exists(os.path.join(d, "report.md")) and os.path.exists(os.path.join(d, "cost.json"))
assert report["gates"]["checks"], "the config's gates produced no checks"
assert rc == (0 if report["gates"]["passed"] else 1), (rc, report["gates"])
print("  eval: four kinds scored, report/generations/cost written, --gate agrees with the report")
EOF
printf '  %3ds  %s\n' $((SECONDS - T0)) "8-eval"
"$PY" tools/eval_table.py "$WORK/eval/report.json"

echo
echo "sample said: $(tail -1 "$WORK/7-sample.log")"
echo "OK: pretrain + align + sft + sample + eval all ran in $((SECONDS - T0))s"
echo "    logs: $WORK/*.log   kept: $WORK/sft/final   ($(du -sh "$WORK" 2>/dev/null | cut -f1) on disk;"
echo "    KEEP=1 bash scripts/smoke_test.sh keeps every stage's checkpoints instead)"
