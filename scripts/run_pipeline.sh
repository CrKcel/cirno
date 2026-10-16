#!/usr/bin/env bash
# End-to-end recipe for the on-device chat / role-play model (see doc/training.md).
# Every source is a path inside the local Hub snapshot, so the pipeline needs no network access.
set -euo pipefail
cd "$(dirname "$0")/.."

# ---------------- DATA --------------------------------------------------------------------------
# The datasets doc/datasets.md points at live in one local snapshot: one directory per repo keeping the
# Hub's own file layout (see /mnt/data/LLM/README.md).  Sources are therefore ordinary local `parquet:`
# / `jsonl:` paths, and the three *_OFFLINE switches turn an accidental `hf:` spec into an immediate
# error instead of a download.
SNAP="${CIRNO_SNAPSHOT:-/mnt/data/LLM/hf}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -d "$SNAP" ] || { echo "snapshot not found: $SNAP (set CIRNO_SNAPSHOT)" >&2; exit 2; }

# The sources whose cleaning happens at the text level cannot be written as a field expression and are
# converted into data/<bucket>/*.jsonl first: b-corpus `.txt.gz` works (one document per ~4k characters),
# the raw moegirl wikitext, the two JSON-array role-play sets and the English small sets that ship
# punctuation/markup noise.  It skips files that already exist; `--force` redoes them.
python scripts/prepare_local_data.py --snapshot "$SNAP" --out data

# One array per bucket.  `--source` is repeatable: the whole array shares one --max_tokens budget and is
# read in order.  The order is a priority/validation knob, not a training order (`TokenMixer` shuffles
# every window of a bin); it decides what a binding cap drops and where `val.bin` comes from (the head of
# the first source).  Caps sit ~20% above the estimated pool because the character -> token conversion is
# only ±20% accurate; wiki is the exception - the recipe allocates it less than its pool on purpose.
CHAT=(
    --source "parquet:$SNAP/silver--lccc/base/train/*.parquet:dialog"    # word-segmented: needs --desegment
    --source "parquet:$SNAP/silver--lccc/large/train/*.parquet:dialog"   # ~30% overlaps base (datasets.md 1.1)
    --source "jsonl:$SNAP/chenrm--anime-subtitles/dialogue.jsonl:text"   # scripts/clean_ass.py --zh-only --t2s output
    --source "jsonl:data/chat/*.jsonl:text"                              # b-corpus v-corpus-zh
)
EN=(
    # TinyStoriesV2 is listed first: it is the largest source, a low cap would cut everything after it,
    # and its head is what val.bin holds (train.py reports this bin's ppl on simple English story text).
    --source "parquet:$SNAP/noanabeshima--TinyStoriesV2/default/train/*.parquet:text"
    --source "parquet:$SNAP/allenai--soda/train.parquet:dialogue"        # synthetic (InstructGPT), the 2nd largest
    --source "parquet:$SNAP/ParlAI--blended_skill_talk/data/train-*.parquet:free_messages"
    # the human-written small sets ride along for style: b-corpus v-corpus-en and kajiwoto, plus the
    # cleaned DailyDialog / EmpatheticDialogues / Topical-Chat / Cornell / ESConv
    --source "jsonl:data/en_chat/*.jsonl:text"
)
FICTION=(
    --source "parquet:$SNAP/wdndev--webnovel-chinese/default/partial-train/*.parquet:text"
    --source "parquet:$SNAP/mrzjy--Chinese_interactive_novels_3k/default/train/*.parquet:book_chapter.content.content"
    --source "jsonl:$SNAP/Miaowuawa--ChineseNovels/haitang.jsonl:text"   # one book per record, cut at 200k chars
    --source "jsonl:data/fiction/*.jsonl:text"                           # b-corpus 近现代小说 / 翻译名著 / 轻小说
)
LONG=(
    --source "parquet:$SNAP/wangrui6--Zhihu-KOL/data/train-*.parquet:RESPONSE"
    --source "jsonl:data/long/*.jsonl:text"                              # b-corpus 提高逼格 / 畅销书 / 散文集 / 古籍
)
WIKI=(
    # moegirl first: the wiki cap is deliberately below the pool (~1.23B of ~1.39B), and baidu-baike is
    # ordered common entity -> rare entity, so the tokens this cap drops are its cold tail.  Listed
    # after baidu it would get nothing at all.
    --source "jsonl:data/wiki/*.jsonl:text"                              # moegirl, wikitext stripped
    --source "parquet:$SNAP/lars1234--baidu-baike-dataset/default/partial-train/*.parquet:-"
)
ROLEPLAY=(
    --source "jsonl:$SNAP/shibing624--roleplay-zh/sharegpt_formatted_data-evol-gpt4.jsonl:-"
    --source "jsonl:$SNAP/Johnson8187--role-play-chinese/alpaca_format_simplified.jsonl:-"
    --source "jsonl:data/roleplay/*.jsonl:text"                          # LooksJuicy + b-corpus q-corpus
)
CHAR=(
    --source "parquet:$SNAP/silk-road--ChatHaruhi-54K/default/train/*.parquet:user_question+agent_response"
    --source "jsonl:data/roleplay_char/*.jsonl:text"                     # b-corpus ChatHaruhi, split by language
)
CATGIRL=(
    --source "jsonl:data/roleplay_catgirl/*.jsonl:text"                  # kxdw2580 v2, <think> removed
)

# 0. tokenizer (32k BPE).  Four things to get right here, because redoing it means redoing every bucket:
#    * every bucket with a real token share must appear, or its vocabulary gets byte-split;
#    * `@N` is a per-source budget, so it is picked from the mean document length, not by feel - a chat
#      turn is ~35 chars and an article ~2800, so equal-looking N values would starve the target
#      distribution.  The numbers below follow the pre-training token totals of step 1 (role-play
#      excluded), i.e. roughly chat 19% / fiction 30% / wiki 26% / long 10% / en 15%; re-derive with
#      `python tools/check_local_data.py --lengths` after the sources change;
#    * word-segmented corpora (LCCC) need --desegment or the merges are learned on spaced-out text;
#    * numbers stay whole (`\p{N}+`), not digit-by-digit as in Qwen2.
python scripts/train_tokenizer.py --out tokenizer --desegment \
    --source "parquet:$SNAP/silver--lccc/base/train/*.parquet:dialog@800000" \
    --source "parquet:$SNAP/silver--lccc/large/train/*.parquet:dialog@700000" \
    --source "jsonl:$SNAP/chenrm--anime-subtitles/dialogue.jsonl:text@17000" \
    --source "jsonl:data/chat/*.jsonl:text@8000" \
    --source "parquet:$SNAP/noanabeshima--TinyStoriesV2/default/train/*.parquet:text@170000" \
    --source "parquet:$SNAP/allenai--soda/train.parquet:dialogue@70000" \
    --source "parquet:$SNAP/ParlAI--blended_skill_talk/data/train-*.parquet:free_messages@3000" \
    --source "jsonl:data/en_chat/bcorpus_en.jsonl:text@3000" \
    --source "jsonl:data/en_chat/daily_dialog.jsonl:text@1000" \
    --source "jsonl:data/en_chat/empathetic.jsonl:text@3000" \
    --source "jsonl:data/en_chat/topical_chat.jsonl:text@700" \
    --source "jsonl:data/en_chat/cornell.jsonl:text@300" \
    --source "jsonl:data/en_chat/esconv.jsonl:text@600" \
    --source "jsonl:data/en_chat/kajiwoto.jsonl:text@100" \
    --source "parquet:$SNAP/wdndev--webnovel-chinese/default/partial-train/*.parquet:text@120000" \
    --source "parquet:$SNAP/mrzjy--Chinese_interactive_novels_3k/default/train/*.parquet:book_chapter.content.content@1100" \
    --source "jsonl:$SNAP/Miaowuawa--ChineseNovels/haitang.jsonl:text@190" \
    --source "jsonl:data/fiction/*.jsonl:text@1000" \
    --source "jsonl:data/wiki/*.jsonl:text@38000" \
    --source "parquet:$SNAP/lars1234--baidu-baike-dataset/default/partial-train/*.parquet:-@320000" \
    --source "parquet:$SNAP/wangrui6--Zhihu-KOL/data/train-*.parquet:RESPONSE@160000" \
    --source "jsonl:data/long/*.jsonl:text@1400"

# 1. tokenise each bucket separately so the mixture can be re-weighted without re-tokenising.
#    `--max_tokens` is a safety limit, not a budget: it sits above the bucket's pool so the read reaches
#    the end of the list, and `train_tokens` in <out>/meta.json is the pool every phase samples from.
#    A cap that binds stops the bin mid-source, which is a corpus left untrained rather than a mixture
#    correction - the weights come from those pools (tools/plan_mix.py) and no bin is asked for more than
#    the recipe allocates it.  Re-derive every number afterwards with:
#      python tools/plan_mix.py --plan configs/pretrain_plan.json
python scripts/prepare_data.py --tokenizer tokenizer --desegment --out data/bin/chat --max_tokens 1210000000 --val_tokens 4000000 "${CHAT[@]}"
python scripts/prepare_data.py --tokenizer tokenizer             --out data/bin/en_chat --max_tokens 1000000000 --val_tokens 1000000 "${EN[@]}"
python scripts/prepare_data.py --tokenizer tokenizer             --out data/bin/fiction --max_tokens 1900000000 --val_tokens 2500000 "${FICTION[@]}"
python scripts/prepare_data.py --tokenizer tokenizer             --out data/bin/long --max_tokens 660000000 --val_tokens 2000000 "${LONG[@]}"
python scripts/prepare_data.py --tokenizer tokenizer             --out data/bin/wiki --max_tokens 1228050000 --val_tokens 2500000 "${WIKI[@]}"
python scripts/prepare_data.py --tokenizer tokenizer --desegment --out data/bin/roleplay --max_tokens 22400000 --val_tokens 200000 "${ROLEPLAY[@]}"
python scripts/prepare_data.py --tokenizer tokenizer --desegment --out data/bin/roleplay_char --max_tokens 6600000 --val_tokens 100000 "${CHAR[@]}"
python scripts/prepare_data.py --tokenizer tokenizer --desegment --out data/bin/roleplay_catgirl --max_tokens 210000 --val_tokens 20000 "${CATGIRL[@]}"

# 1b. the one evaluation input that needs the tokenizer: the human reply-length prior the imform set
#     compares against (real human replies from the SFT slice, in this tokenizer's tokens).  The probes
#     themselves are authored in evals/ and need nothing built.
python scripts/eval.py --tokenizer tokenizer --desegment --length-prior data/sft/human_dialog.jsonl \
    --out runs/eval/imform_prior.json

# 2. text pre-training: configs/pretrain_plan.json is the recipe, and plan_mix.py is what turns it into
#    runnable phases.  The mixture is a schedule, not a constant - phase A spends the first 1.21B on
#    long-form only at seq 2048, phase B brings dialogue in for 3.10B, phase C decays on chat + role-play
#    for the last 1.08B - and the weights come out of the bins' measured pools, so they are derived
#    rather than typed.  chat is the one bin read more than once (allocate 1.212B for a ~1.01B pool):
#    the replay is what lifts dialogue from 34% to 37.5% of the budget, and it is concentrated in the
#    anneal where the last gradients pick the style (doc/training.md 2).
#    --print-commands also emits the align / SFT lines below.
python tools/plan_mix.py --plan configs/pretrain_plan.json --write runs/pretrain/plan.resolved.json --print-commands

#    Each phase is a separate run into the SAME --out_dir, so B and C pick up runs/pretrain/last and keep
#    the optimizer, the sample counter and the data order.  --tokens, the phase boundary, the mixture and
#    the shape all come from the resolved plan, and the learning rate is one WSD curve over the whole
#    5.39B tokens (warmup 2%, flat middle, final 20% decay), so there is no --decay_frac 0 bookkeeping
#    and no hand-maintained cumulative token count to get wrong.
#    Measured on an 8 GB card at 524k tokens/step (doc/architecture.md 3.2, absolute numbers differ per
#    session):
#       mb16 x accum32   seq 1024         6938 tok/s   3.9 GB
#       mb2  x accum256  seq 1024        13042 tok/s   5.1 GB   <- phases B and C
#       mb4  x accum128  seq 1024 --compile  18143 tok/s   5.2 GB
#       mb2  x accum128  seq 2048         4785 tok/s   2.8 GB   <- phase A (the probe turns
#                                                                  checkpointing on; with it off this
#                                                                  shape OOMs).  Per token this is the same
#                                                                  cost as seq 1024 x mb 2 (4575 tok/s),
#                                                                  i.e. 2048 is nearly free.
#    Re-measure with:
#      python tools/measure_batch.py --stage pretrain [--gdn_impl fla|torch] [--precision bf16|fp16|fp32]
#                                    [--no_grad_ckpt] [--compile]
python scripts/train.py --stage pretrain --plan runs/pretrain/plan.resolved.json --phase A \
    --tokenizer tokenizer --out_dir runs/pretrain --compile --resume
python scripts/train.py --stage pretrain --plan runs/pretrain/plan.resolved.json --phase B \
    --tokenizer tokenizer --out_dir runs/pretrain --compile --resume
python scripts/train.py --stage pretrain --plan runs/pretrain/plan.resolved.json --phase C \
    --tokenizer tokenizer --out_dir runs/pretrain --compile --resume

#    Every run also writes metrics.jsonl + health.jsonl next to log.jsonl; watch it live, or take one
#    snapshot for a log, with:
#      python tools/watch_training.py --dir runs/pretrain
#      python tools/watch_training.py --dir runs/pretrain --once
#    Defaults are alert-and-correct: a loss or grad-norm spike halves the lr for up to a twentieth of the
#    run, sustained throughput loss and memory pressure are reported, and --early_stop_patience N turns
#    "N evaluations without a new best val loss" into a stop.  --no_intervention keeps the alerts only;
#    doc/monitoring.md has the thresholds.

# 2b. evaluate the pre-training artifact once the anneal finished: the probes are scored on generation
#     (doc/evaluation.md), so this is the first point where "does it still chat" is visible as a number.
#     The gates belong to the SFT stage, so this run is trend-only; --bench adds the edge cost numbers.
python scripts/eval.py --model runs/pretrain/final --tokenizer tokenizer --out runs/eval/pretrain --bench

# 3. (optional) vision alignment: only when the deployment really sees images (stickers / memes /
#    chat screenshots).  LLM frozen, ViT + projector learn from IN-CONVERSATION images, not academic
#    VQA captions - VQA data pulls the model back towards assistant behaviour.  Skip this stage for a
#    text-only chat model and point step 4 at runs/pretrain/final instead.
#    Chat images live inside the open image-chat sets, and the snapshot has no exported copy, so this
#    line expects an already-written jsonl plus the images under --image_root.
python scripts/train.py --stage align --plan runs/pretrain/plan.resolved.json --phase align \
    --tokenizer tokenizer --init_from runs/pretrain/final --out_dir runs/align \
    --mm_data data/mm/chat_image.jsonl --image_root data/mm/images --resume

#    The same probes on the aligned checkpoint.  The authored sets are text-only; image probes enter with
#    `--set <name>=<jsonl>` plus `--image_root <dir>` (tools/make_synthetic.py writes one such set for the
#    smoke test).
python scripts/eval.py --model runs/align/final --tokenizer tokenizer --out runs/eval/align

# 4. dialogue / role-play SFT: human-to-human conversations + personas + a small out-of-scope refusal
#    set ("I don't know, ask an AI").  Only the imitated side (the `self` role in the encoder)
#    is scored; `--train llm` matches the text-only path (use runs/pretrain/final if step 3 was skipped).
#    `prepare_local_data.py` wrote the first two slices from the snapshot (human dialogue = LCCC base,
#    persona = shibing624 + Johnson8187 + LooksJuicy, with `instruction` renamed `system_prompt` or the
#    card would repeat in every user turn, doc/datasets.md 1.6); the refusal slice (2-8%) has no
#    off-the-shelf source, so write data/sft/refuse.jsonl yourself per doc/training.md 4.4.
#    --mm_weights is what turns the recipe's 74 / 21 / 5 slice into an actual mixture: without it a
#    source's share of the batches is whatever its line count happens to be, which is how a 500-row
#    refusal set disappears next to a million-row chat dump.  LCCC needs --desegment here as well, or the
#    SFT targets are word-segmented.
python scripts/train.py --stage sft --plan runs/pretrain/plan.resolved.json --phase sft \
    --tokenizer tokenizer --init_from runs/align/final --out_dir runs/sft \
    --mm_data data/sft/human_dialog.jsonl,data/sft/persona.jsonl,data/sft/refuse.jsonl \
    --mm_weights 0.74,0.21,0.05 --desegment --resume

# 5. evaluate the deployed artifact.  The automated numbers are trend-only; the two things that decide a
#    release are the blind-test win rate and the assistant-tone / AI-self-reference gates (doc/evaluation.md).
#    --blind N writes N pairs where the model's reply and a real human reply answer the same context:
#      python tools/blind_test.py --dir runs/eval/sft        # label (a/b/tie/skip)
#      python tools/blind_test.py --score runs/eval/sft      # fold the win rate into report.json
python scripts/eval.py --model runs/sft/final --tokenizer tokenizer --out runs/eval/sft --blind 30
python tools/eval_table.py runs/eval/pretrain/report.json runs/eval/align/report.json \
    runs/eval/sft/report.json --baseline runs/eval/pretrain/report.json

python scripts/sample.py --model runs/sft/final --prompt "在吗" --max_new_tokens 32
