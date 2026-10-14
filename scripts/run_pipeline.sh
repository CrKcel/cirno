#!/usr/bin/env bash
# End-to-end recipe for the on-device chat / role-play model (see doc/training.md).
# Edit the DATA section for real corpora: the goal is human-like IM dialogue, not task solving.
set -euo pipefail
cd "$(dirname "$0")/.."

# ---------------- DATA (placeholders; doc/datasets.md lists the open datasets that fill each bucket) -----
# Each variable is a source spec, so a Hub dataset can be dropped in as-is - no local jsonl needed.
# `prepare_data.py --source` is repeatable and the sources share one --max_tokens budget, read in the
# order given, so a bucket is filled by listing several specs: give the scarce/priority source first.
#   CHAT     parquet:hf://datasets/silver/lccc@refs/convert/parquet/base/train/*.parquet:dialog     # (MIT)
#   CHAT     parquet:hf://datasets/silver/lccc@refs/convert/parquet/large/train/*.parquet:dialog
#   CHAT     jsonl:data/chat/subtitles.jsonl:text                     # spoken-style subtitles, but only
#            # the `scripts/clean_ass.py --zh-only --t2s` output: the raw `data` field is the whole .ass
#            # script (headers, style tables, karaoke timing), and 1,470 of the 59,806 episodes hit
#            # prepare_data.py's 200k-char cut (largest 67 MB) with karaoke runs like `佐\n佐\n佐...`
#   EN       hf:allenai/soda:dialogue:default:train                   # en, ~190M tok; INSTRUCTGPT-GENERATED
#   EN       hf:ParlAI/blended_skill_talk:free_messages:default:train # en, crowd-sourced: persona + empathy + knowledge
#   EN       hf:Fishball02/topical-chat:-:default:train               # en, MTurk; strip the <s>[agent_1] markup
#   EN       parquet:hf://datasets/li2017dailydialog/daily_dialog@refs/convert/parquet/default/train/*.parquet:dialog
#   EN       parquet:hf://datasets/facebook/empathetic_dialogues@refs/convert/parquet/default/train/*.parquet:prompt+utterance
#   FICTION  hf:wdndev/webnovel-chinese:text:default:train            # web novels (apache-2.0)
#   LONG     hf:wangrui6/Zhihu-KOL:RESPONSE:default:train             # human long-form answers
#   LONG     parquet:hf://datasets/opencsg/Fineweb-Edu-Chinese-V2.1@main/4_5/*.parquet:text  # (apache-2.0)
#            # not in the mix: Zhihu-KOL alone is 0.51B tokens, i.e. the whole long-form pool, and the
#            # first 4_5 shards are one source corpus (CCI3) - see the shard-order table in doc/training.md 5
#   ROLEPLAY hf:shibing624/roleplay-zh-sharegpt-gpt4-data:-:sharegpt_formatted_data-evol-gpt4:train
#   ROLEPLAY hf:Johnson8187/role-play-chinese:-:default:train
#   CHAR     parquet:hf://datasets/silk-road/ChatHaruhi-54K-Role-Playing-Dialogue@refs/convert/parquet/default/train/*.parquet:user_question+agent_response
#            # novel/film characters; only the dialogue pair, the file's other 49% is metadata
#   CATGIRL  hf download kxdw2580/catgirl-dataset --include "v2/*"  ->  clean to jsonl in data/roleplay_catgirl
#            # gated (accept the terms first), the v2 files are JSON arrays rather than JSONL, and the
#            # answer's chain of thought is inline (<think>...</think>) - field selection cannot remove
#            # it, the text has to be cleaned before tokenising.  v2 only: the author rules out v1
#            # (identity leaks), v2/common-v2-extra (irrelevant Q&A) and the whole v2-qwen3 series.
#   WIKI     hf:wikimedia/wikipedia:text:20231101.zh:train            # general knowledge (CC-BY-SA);
#            # measured 1.05B chars / ~0.74B tokens, lower fact density than baidu-baike and 繁简混用,
#            # so it stays out of the mix (opencc first if it is ever added)
#   WIKI     parquet:hf://datasets/lars1234/baidu-baike-dataset@refs/convert/parquet/default/partial-train/*.parquet:-
#                                                                     # 简体白话百科；parquet 3.2 GB，
#                                                                     # `hf:` 直连会拉 16.8 GB 的原始 json
# The EN bucket exists because allenai/soda is large (measured: 1.2M conversations, mean 632 chars of
# dialogue, ~191M tokens) - it is the only English set that can carry a weight.  It is synthetic
# (InstructGPT distilled on commonsense triples, `language_creators:machine-generated`); the human-written
# sets add up to ~17M tokens (doc/datasets.md §1) and ride along in the same bucket for style diversity,
# each needing cleaning (DailyDialog spaces before punctuation, EmpatheticDialogues `_comma_`,
# Topical-Chat `<s>[agent_1]`).  Order: soda first, the human-written sets after it.  The bin's cap
# (260M) sits above the whole pool with headroom (soda 191M + the sets ~17M), because soda is read first:
# a cap that only matches the estimate would let a tokenizer-ratio surprise cut the small sets off.
# Two things about the specs above, both of which bite silently:
#   * LCCC / DailyDialog / EmpatheticDialogues / PersonaChat only ship a loading script, and `datasets`
#     >= 4 refuses scripts.  The `parquet:hf://...@refs/convert/parquet/...:field` form reads the Hub's
#     auto-converted parquet branch instead - keep the `@refs/...` revision, it is not a document limit.
#     MNBVC is the exception: it has no parquet branch, so it is unreachable either way (see progress.md TODO).
#   * LCCC's `dialog` is word-segmented (`你 去 那儿 竟然`), so everything that takes it is tokenised
#     with --desegment, or the model learns to answer with spaces between characters.
# This project is non-commercial, so the NC-licensed sets are fine to use; `doc/datasets.md` §1 flags the
# sets with no licence at all (Zhihu-KOL) - those are the ones worth double-checking before publishing.
CHAT="jsonl:data/chat/*.jsonl:text"                  # human-to-human chat / forum / subtitles (zh)
EN="jsonl:data/en_chat/*.jsonl:text"                 # English conversation: soda first, then the human sets
FICTION="jsonl:data/fiction/*.jsonl:text"            # novels / scripts (dialogue-heavy)
LONG="jsonl:data/long/*.jsonl:text"                  # long-form non-dialogue: articles, books, long answers
ROLEPLAY="jsonl:data/roleplay/*.jsonl:text"          # persona cards + multi-turn role-play
CHAR="jsonl:data/roleplay_char/*.jsonl:text"         # novel/film character Q&A (ChatHaruhi)
CATGIRL="jsonl:data/roleplay_catgirl/*.jsonl:text"   # catgirl persona chat
WIKI="jsonl:data/wiki/*.jsonl:text"                  # general knowledge, keeps trivia answerable

# 0. tokenizer (32k BPE).  Four things to get right here, because redoing it means redoing every bucket:
#    * every bucket with a real token share must appear, or its vocabulary gets byte-split;
#    * `@N` is a per-source budget, so it is picked from the mean document length, not by feel - a chat
#      turn is ~35 chars and an article ~2800, so equal-looking N values would starve the target
#      distribution.  The numbers below follow the pre-training token totals of step 2 (role-play excluded),
#      i.e. roughly chat 22% / fiction 29% / wiki 30% / long 13% / en 5%;
#    * word-segmented corpora (LCCC) need --desegment or the merges are learned on spaced-out text;
#    * numbers stay whole (`\p{N}+`), not digit-by-digit as in Qwen2.
python scripts/train_tokenizer.py --out tokenizer --desegment \
    --source "$CHAT@3200000" --source "$EN@120000" --source "$FICTION@40000" \
    --source "$WIKI@39000" --source "$LONG@110000"

# 1. tokenise each source separately so the mixture can be re-weighted without re-tokenising.
#    The cap is a safety limit, not a budget: it sits at or above the bin's own token count, so the sources
#    are read to the end and `train_tokens` in <out>/meta.json is the pool every phase samples from.  A cap
#    that binds stops the bin mid-source, which is a corpus left untrained, not a mixture correction - the
#    weights come from those pools (tools/plan_mix.py) and no bin is asked for more than it holds.
python scripts/prepare_data.py --tokenizer tokenizer --desegment --source "$CHAT"     --out data/bin/chat     --max_tokens  884000000 --val_tokens 4000000
python scripts/prepare_data.py --tokenizer tokenizer             --source "$EN"       --out data/bin/en_chat  --max_tokens  260000000 --val_tokens 1000000
python scripts/prepare_data.py --tokenizer tokenizer             --source "$FICTION"  --out data/bin/fiction  --max_tokens 1179000000 --val_tokens 2500000
python scripts/prepare_data.py --tokenizer tokenizer             --source "$LONG"     --out data/bin/long     --max_tokens  508000000 --val_tokens 2000000
python scripts/prepare_data.py --tokenizer tokenizer --desegment --source "$ROLEPLAY" --out data/bin/roleplay         --max_tokens   12000000 --val_tokens 200000
python scripts/prepare_data.py --tokenizer tokenizer --desegment --source "$CHAR"     --out data/bin/roleplay_char    --max_tokens    3600000 --val_tokens 100000
python scripts/prepare_data.py --tokenizer tokenizer --desegment --source "$CATGIRL" --out data/bin/roleplay_catgirl --max_tokens     180000 --val_tokens  20000
python scripts/prepare_data.py --tokenizer tokenizer             --source "$WIKI"     --out data/bin/wiki     --max_tokens 1218000000 --val_tokens 2500000

# 2. text pre-training: 4B tokens in three phases, dialogue-dominant mixture (doc/training.md 2).  The
#    mixture is a schedule, not a constant: phase A spends the first 0.9B on long-form only at seq 2048,
#    phase B brings dialogue in for 2.3B, phase C decays on chat + role-play for the last 0.8B.  Weights are
#    derived from the bins' `train_tokens` (tools/plan_mix.py), so every bin is read about once - chat 880M,
#    en_chat 208M, fiction 1.18B, long 0.51B, role-play 15.4M - and the wiki bin, the only pool bigger than
#    what the budget needs, gives up its tail (0.90 pass).  The ordering is what keeps the long-form tokens
#    from diluting the chat style: chat starts in B and the last 0.8B is 34% chat + role-play.
#    Everything else uses the fast path: --precision auto (bf16 on Ampere+, else fp16 + GradScaler),
#    --gdn_impl auto (flash-linear-attention Triton kernels once `pip install flash-linear-attention` has
#    been run) and --grad_ckpt auto / --ce_ckpt auto, which probe the configured shape once and skip both
#    activation checkpointing and the CE's head-matmul recompute when they fit (~1.6x together).  micro_batch
#    is therefore kept small: what matters is that a micro-batch fits WITHOUT checkpointing, and
#    --grad_accum makes up 524k tokens per step in every phase.
#    Measured on an 8 GB card at 524k tokens/step (doc/architecture.md 3.2, absolute numbers differ per session):
#       mb16 x accum32   seq 1024         6938 tok/s   3.9 GB
#       mb2  x accum256  seq 1024        13042 tok/s   5.1 GB   <- phases B and C
#       mb4  x accum128  seq 1024 --compile  18143 tok/s   5.2 GB
#       mb2  x accum128  seq 2048         4785 tok/s   2.8 GB   <- phase A (probe turns checkpointing on;
#                                                                  with it off this shape OOMs).  Per token
#                                                                  this is the same cost as seq 1024 x mb 2
#                                                                  (4575 tok/s), i.e. 2048 is nearly free.
#    Re-measure with:
#      python tools/measure_batch.py --stage pretrain [--gdn_impl fla|torch] [--precision bf16|fp16|fp32]
#                                    [--no_grad_ckpt] [--compile]
#    Each phase is a separate run into the SAME --out_dir, so B and C pick up runs/pretrain/last and keep
#    the optimizer, the sample counter and the data order.  Two things to know before editing:
#      * --tokens is the CUMULATIVE target of the run, not the length of this phase: A passes 0.9e9, B
#        passes 3.2e9 (= 0.9 + 2.3), C passes 4.0e9.  All three use 524k tokens/step, so the step counter
#        lines up across runs (1716 / 6103 / 7629 steps) and C's decay starts exactly where C starts.
#      * --decay_frac 0 on A and B keeps the WSD schedule flat at the peak lr for an intermediate phase
#        (it also holds --mtp_weight_final off, which is what phase C is for).
#    Weights are derived, not chosen: tools/plan_mix.py reads the bins' `train_tokens` and splits each
#    whole pool over the three phases, so a weight can never exceed what the bin holds.  chat's pool is
#    880M, which is why its share drops from 47% of B to 26% - reading it once at 34% of both B and C
#    would make it 1.5 passes.  role-play's 15.4M tokens appear ONLY in the decay phase (1.9% of it), so
#    each of the three small bins is read about once; run plan_mix after prepare_data to refresh the
#    numbers.  TokenMixer normalises the weights, so only their ratios matter.
python scripts/train.py --stage pretrain --tokenizer tokenizer --out_dir runs/pretrain \
    --data data/bin/fiction:0.45,data/bin/long:0.35,data/bin/wiki:0.20 \
    --tokens 0.9e9 --seq_len 2048 --micro_batch 2 --grad_accum 128 --lr 1e-3 --decay_frac 0 \
    --mtp_weight 0.3 --compile --resume

python scripts/train.py --stage pretrain --tokenizer tokenizer --out_dir runs/pretrain \
    --data data/bin/chat:0.2645,data/bin/en_chat:0.0777,data/bin/fiction:0.2378,data/bin/long:0.0689,data/bin/wiki:0.3511 \
    --tokens 3.2e9 --seq_len 1024 --micro_batch 2 --grad_accum 256 --lr 1e-3 --decay_frac 0 \
    --mtp_weight 0.3 --compile --resume

python scripts/train.py --stage pretrain --tokenizer tokenizer --out_dir runs/pretrain \
    --data data/bin/chat:0.34,data/bin/en_chat:0.036,data/bin/fiction:0.28,data/bin/long:0.04,data/bin/wiki:0.284,data/bin/roleplay:0.0147,data/bin/roleplay_char:0.0044,data/bin/roleplay_catgirl:0.0002 \
    --tokens 4.0e9 --seq_len 1024 --micro_batch 2 --grad_accum 256 --lr 1e-3 --mtp_weight 0.3 \
    --mtp_weight_final 0.1 --compile --resume

# 3. (optional) vision alignment: only when the deployment really sees images (stickers / memes /
#    chat screenshots).  LLM frozen, ViT + projector learn from IN-CONVERSATION images, not academic
#    VQA captions - VQA data pulls the model back towards assistant behaviour.  Skip this stage for a
#    text-only chat model and point step 4 at runs/pretrain/final instead.
#    PhotoChat / chat-screenshot sets (doc/datasets.md §3) keep their images inside the dataset, so they
#    still have to be exported to --image_root by hand; this line expects an already-written jsonl.
python scripts/train.py --stage align --tokenizer tokenizer --init_from runs/pretrain/final --out_dir runs/align \
    --mm_data data/mm/chat_image.jsonl --image_root data/mm/images --epochs 1 --micro_batch 8 --grad_accum 32 --resume

# 4. dialogue / role-play SFT: human-to-human conversations + personas + a small out-of-scope refusal
#    set ("I don't know, ask an AI").  Only the imitated side (the `self` role in the encoder)
#    is scored; `--train llm` matches the text-only path (use runs/pretrain/final if step 3 was skipped).
#    --mm_data takes source specs too, and remote ones are spooled into --mm_cache (data/cache) once.
#    LCCC comes through its converted parquet branch and needs --desegment here as well, or the SFT
#    targets are word-segmented:
#      --mm_data "parquet:hf://datasets/silver/lccc@refs/convert/parquet/base/train/*.parquet:dialog,hf:shibing624/roleplay-zh-sharegpt-gpt4-data:-:sharegpt_formatted_data-evol-gpt4:train,hf:Johnson8187/role-play-chinese:-:default:train" --desegment
#    The refusal slice (2-8%) has no off-the-shelf source: write it yourself per doc/training.md 4.4.
python scripts/train.py --stage sft --tokenizer tokenizer --init_from runs/align/final --out_dir runs/sft \
    --train llm --mm_data data/sft/human_dialog.jsonl,data/sft/persona.jsonl,data/sft/refuse.jsonl --desegment --epochs 2 \
    --micro_batch 8 --grad_accum 16 --resume

python scripts/sample.py --model runs/sft/final --prompt "在吗" --max_new_tokens 32
