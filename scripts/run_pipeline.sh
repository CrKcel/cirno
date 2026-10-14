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
#   CHAT     hf:chenrm/anime-subtitles:data:default:train             # spoken-style subtitles
#   EN       hf:allenai/soda:dialogue:default:train                   # en, ~190M tok; INSTRUCTGPT-GENERATED
#   EN       hf:ParlAI/blended_skill_talk:free_messages:default:train # en, crowd-sourced: persona + empathy + knowledge
#   EN       hf:Fishball02/topical-chat:-:default:train               # en, MTurk; strip the <s>[agent_1] markup
#   EN       parquet:hf://datasets/li2017dailydialog/daily_dialog@refs/convert/parquet/default/train/*.parquet:dialog
#   EN       parquet:hf://datasets/facebook/empathetic_dialogues@refs/convert/parquet/default/train/*.parquet:prompt+utterance
#   FICTION  hf:wdndev/webnovel-chinese:text:default:train            # web novels (apache-2.0)
#   LONG     hf:wangrui6/Zhihu-KOL:RESPONSE:default:train             # human long-form answers
#   LONG     parquet:hf://datasets/opencsg/Fineweb-Edu-Chinese-V2.1@main/4_5/*.parquet:text  # (apache-2.0)
#   ROLEPLAY hf:shibing624/roleplay-zh-sharegpt-gpt4-data:-:sharegpt_formatted_data-evol-gpt4:train
#   ROLEPLAY hf:Johnson8187/role-play-chinese:-:default:train
#   WIKI     hf:wikimedia/wikipedia:text:20231101.zh:train            # general knowledge (CC-BY-SA);
#                                                                     # 实测繁简混用，建议先 opencc 统一
#   WIKI     parquet:hf://datasets/lars1234/baidu-baike-dataset@refs/convert/parquet/default/partial-train/*.parquet:-
#                                                                     # 简体白话百科；parquet 3.2 GB，
#                                                                     # `hf:` 直连会拉 16.8 GB 的原始 json
# The EN bucket exists because allenai/soda is large (measured: 1.2M conversations, mean 632 chars of
# dialogue, ~190M tokens) - it is the only English set that can carry a weight.  It is synthetic
# (InstructGPT distilled on commonsense triples, `language_creators:machine-generated`), which is why
# it gets 5% and not more: spot-checking found the assistant tone mild, but keep an eye on it and drop
# the weight if Chinese chat regresses.  The human-written English sets add up to only ~14M tokens
# (doc/datasets.md §1) and ride along in the same bucket for style diversity; each needs cleaning
# (DailyDialog spaces before punctuation, EmpatheticDialogues `_comma_`, Topical-Chat `<s>[agent_1]`).
# Two things about the specs above, both of which bite silently:
#   * LCCC / DailyDialog / EmpatheticDialogues / PersonaChat only ship a loading script, and `datasets`
#     >= 4 refuses scripts.  The `parquet:hf://...@refs/convert/parquet/...:field` form reads the Hub's
#     auto-converted parquet branch instead - keep the `@refs/...` revision, it is not a document limit.
#     MNBVC is the exception: it has no parquet branch, so it is unreachable either way (see progress.md TODO).
#   * LCCC's `dialog` is word-segmented (`你 去 那儿 竟然`), so everything that takes it is tokenised
#     with --desegment, or the model learns to answer with spaces between characters.
# This project is non-commercial, so the NC-licensed sets are fine to use; `doc/datasets.md` §1 flags the
# sets with no licence at all (Zhihu-KOL) - those are the ones worth double-checking before publishing.
CHAT="jsonl:data/chat/*.jsonl:text"          # human-to-human chat / forum / subtitles (zh)
EN="jsonl:data/en_chat/*.jsonl:text"         # English conversation (soda + the human-written sets)
FICTION="jsonl:data/fiction/*.jsonl:text"    # novels / scripts (dialogue-heavy)
LONG="jsonl:data/long/*.jsonl:text"          # long-form non-dialogue: articles, books, long answers
ROLEPLAY="jsonl:data/roleplay/*.jsonl:text"  # persona cards + multi-turn role-play
WIKI="jsonl:data/wiki/*.jsonl:text"          # general knowledge, keeps trivia answerable

# 0. tokenizer (32k BPE).  Four things to get right here, because redoing it means redoing every bucket:
#    * every bucket with a real token share must appear, or its vocabulary gets byte-split;
#    * `@N` is a per-source budget, so it is picked from the mean document length, not by feel - a chat
#      turn is ~50 chars and an article ~6300, so equal-looking N values would starve the target
#      distribution.  The numbers below aim for roughly chat 40% / fiction 20% / wiki 15% / long 15% / en 10%;
#    * word-segmented corpora (LCCC) need --desegment or the merges are learned on spaced-out text;
#    * numbers stay whole (`\p{N}+`), not digit-by-digit as in Qwen2.
python scripts/train_tokenizer.py --out tokenizer --desegment \
    --source "$CHAT@5000000" --source "$EN@90000" --source "$FICTION@20000" \
    --source "$WIKI@14000" --source "$LONG@100000"

# 1. tokenise each source separately so the mixture can be re-weighted without re-tokenising.
#    --max_tokens caps a bucket at the weight it is about to get; check the `train_tokens` each run
#    prints (also in <out>/meta.json) - a bucket that stops short of its cap is a bucket the mixer
#    will repeat, i.e. the weight is too high for the data that exists.
python scripts/prepare_data.py --tokenizer tokenizer --desegment --source "$CHAT"     --out data/bin/chat     --max_tokens 1350000000 --val_tokens 4000000
python scripts/prepare_data.py --tokenizer tokenizer             --source "$EN"       --out data/bin/en_chat  --max_tokens  150000000 --val_tokens 1000000
python scripts/prepare_data.py --tokenizer tokenizer             --source "$FICTION"  --out data/bin/fiction  --max_tokens  600000000 --val_tokens 2000000
python scripts/prepare_data.py --tokenizer tokenizer             --source "$LONG"     --out data/bin/long     --max_tokens  450000000 --val_tokens 2000000
python scripts/prepare_data.py --tokenizer tokenizer --desegment --source "$ROLEPLAY" --out data/bin/roleplay --max_tokens   15000000 --val_tokens   200000
python scripts/prepare_data.py --tokenizer tokenizer             --source "$WIKI"     --out data/bin/wiki     --max_tokens  435000000 --val_tokens 2000000

# 2. text pre-training, 3B tokens, dialogue-dominant mixture (doc/training.md 2).  Everything defaults
#    to the fast path: --precision auto (bf16 on Ampere+, else fp16 + GradScaler), --gdn_impl auto
#    (flash-linear-attention Triton kernels once `pip install flash-linear-attention` has been run) and
#    --grad_ckpt auto / --ce_ckpt auto, which probe the configured shape once and skip both activation
#    checkpointing and the CE's head-matmul recompute when they fit (~1.6x together).  micro_batch is
#    therefore kept small: what matters is that a micro-batch fits WITHOUT checkpointing, and
#    --grad_accum makes up the 524k tokens per step.
#    Measured on an 8 GB card at 524k tokens/step (doc/architecture.md 3.2, absolute numbers differ per session):
#       mb16 x accum32                   6938 tok/s   3.9 GB
#       mb2  x accum256                  13042 tok/s  5.1 GB   <- this command
#       mb4  x accum128 --compile        18143 tok/s  5.2 GB   <- what --compile adds (~1.39x)
#    Re-measure with:
#      python tools/measure_batch.py --stage pretrain [--gdn_impl fla|torch] [--precision bf16|fp16|fp32]
#                                    [--no_grad_ckpt] [--compile]
#    Weights are a design choice, not a free parameter: each one is a repeat factor over that bucket's
#    own token count (weight x 3e9 / train_tokens).  Chat and role-play are style sources, so ~1-2
#    epochs is intended; the big corpora are diversity sources and should stay under 1.  Role-play is
#    ~6M tokens in total (11.7 MB of parquet), so even 1% would be 5 passes - it gets 0.5%, a format
#    signal for the system turn, not a capability.  Raise it only by writing more persona data.
#    en_chat is 5% = 150M of the ~190M soda has, i.e. just under one pass; Chinese chat drops to 45%
#    so the English slice does not come out of the Chinese style budget twice.
#    Non-dialogue is already 49.5% (fiction 20 + long 15 + wiki 14.5) and every metric that matters is a
#    dialogue metric - do NOT raise those percentages.  To buy more long-form, raise --tokens instead
#    (3e9 -> 4e9) and keep chat's ABSOLUTE tokens fixed; that is the only way it does not dilute chat.
#    TokenMixer normalises the weights, so only their ratios matter.
python scripts/train.py --stage pretrain --tokenizer tokenizer --out_dir runs/pretrain \
    --data data/bin/chat:0.45,data/bin/en_chat:0.05,data/bin/fiction:0.20,data/bin/long:0.15,data/bin/wiki:0.145,data/bin/roleplay:0.005 \
    --tokens 3e9 --seq_len 1024 --micro_batch 2 --grad_accum 256 --lr 1e-3 --mtp_weight 0.3 --mtp_weight_final 0.1 \
    --compile --resume

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
