#!/usr/bin/env bash
# End-to-end recipe for the on-device chat / role-play model (see doc/training.md).
# Edit the DATA section for real corpora: the goal is human-like IM dialogue, not task solving.
set -euo pipefail
cd "$(dirname "$0")/.."

# ---------------- DATA (placeholders; doc/datasets.md lists the open datasets that fill each bucket) -----
# Each variable is a source spec, so a Hub dataset can be dropped in as-is - no local jsonl needed:
#   CHAT     hf:silver/lccc:-:base:train                              # human-to-human chat (MIT)
#   CHAT     hf:chenrm/anime-subtitles:data:default:train             # spoken-style subtitles
#   CHAT     hf:liwu/MNBVC:主题+回复.回复:forum:train                   # forum threads (MIT)
#   FICTION  hf:wdndev/webnovel-chinese:text:default:train            # web novels (apache-2.0)
#   ROLEPLAY hf:shibing624/roleplay-zh-sharegpt-gpt4-data:-:sharegpt_formatted_data-evol-gpt4:train
#   WIKI     hf:wikimedia/wikipedia:text:20231101.zh:train            # general knowledge (CC-BY-SA)
# This project is non-commercial, so the NC-licensed sets are fine to use; doc/datasets.md §6 lists
# which sets have no licence at all - those are the ones worth double-checking before publishing.
CHAT="jsonl:data/chat/*.jsonl:text"          # human-to-human chat / forum / subtitles
FICTION="jsonl:data/fiction/*.jsonl:text"    # novels / scripts (dialogue-heavy)
ROLEPLAY="jsonl:data/roleplay/*.jsonl:text"  # persona cards + multi-turn role-play
WIKI="jsonl:data/wiki/*.jsonl:text"          # general knowledge, keeps trivia answerable

# 0. tokenizer (32k BPE, dialogue + general knowledge), ~10 min
python scripts/train_tokenizer.py --out tokenizer --source "$CHAT@300000" --source "$WIKI@100000"

# 1. tokenise each source separately so the mixture can be re-weighted without re-tokenising
python scripts/prepare_data.py --tokenizer tokenizer --source "$CHAT"     --out data/bin/chat     --max_tokens 1500000000 --val_tokens 4000000
python scripts/prepare_data.py --tokenizer tokenizer --source "$FICTION"  --out data/bin/fiction  --max_tokens  600000000 --val_tokens 2000000
python scripts/prepare_data.py --tokenizer tokenizer --source "$ROLEPLAY" --out data/bin/roleplay --max_tokens  400000000 --val_tokens 2000000
python scripts/prepare_data.py --tokenizer tokenizer --source "$WIKI"     --out data/bin/wiki     --max_tokens  400000000 --val_tokens 2000000

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
python scripts/train.py --stage pretrain --tokenizer tokenizer --out_dir runs/pretrain \
    --data data/bin/chat:0.55,data/bin/fiction:0.20,data/bin/roleplay:0.12,data/bin/wiki:0.13 \
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
#    --mm_data takes source specs too, and remote ones are spooled into --mm_cache (data/cache) once:
#      --mm_data "hf:silver/lccc:-:base:train,hf:shibing624/roleplay-zh-sharegpt-gpt4-data:-:sharegpt_formatted_data-evol-gpt4:train,hf:Johnson8187/role-play-chinese:-:default:train"
#    The refusal slice (2-8%) has no off-the-shelf source: write it yourself per doc/training.md 4.4.
python scripts/train.py --stage sft --tokenizer tokenizer --init_from runs/align/final --out_dir runs/sft \
    --train llm --mm_data data/sft/human_dialog.jsonl,data/sft/persona.jsonl,data/sft/refuse.jsonl --epochs 2 \
    --micro_batch 8 --grad_accum 16 --resume

python scripts/sample.py --model runs/sft/final --prompt "在吗" --max_new_tokens 32
