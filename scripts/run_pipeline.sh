#!/usr/bin/env bash
# End-to-end recipe.  Edit the DATA section for real corpora; run the smoke test first (see README §6).
set -euo pipefail
cd "$(dirname "$0")/.."

# ---------------- DATA (examples; replace with your own sources) ----------------
EN="hf:HuggingFaceFW/fineweb-edu:text:sample-10BT:train"
ZH="jsonl:data/zh/*.jsonl:text"
CODE="jsonl:data/code/*.jsonl:content"
MATH="jsonl:data/math/*.jsonl:text"

# 0. tokenizer (32k BPE, zh+en+code), ~10 min
python scripts/train_tokenizer.py --out tokenizer --source "$EN@300000" --source "$ZH@300000" --source "$CODE@100000"

# 1. tokenise each source separately so the mixture can be re-weighted without re-tokenising
python scripts/prepare_data.py --tokenizer tokenizer --source "$EN"   --out data/bin/en   --max_tokens 3000000000 --val_tokens 4000000
python scripts/prepare_data.py --tokenizer tokenizer --source "$ZH"   --out data/bin/zh   --max_tokens 1800000000 --val_tokens 4000000
python scripts/prepare_data.py --tokenizer tokenizer --source "$CODE" --out data/bin/code --max_tokens  700000000 --val_tokens 2000000
python scripts/prepare_data.py --tokenizer tokenizer --source "$MATH" --out data/bin/math --max_tokens  500000000 --val_tokens 2000000

# 2. text pre-training, 6B tokens.  Everything defaults to the fast path: --precision auto (bf16 on
#    Ampere+, else fp16 + GradScaler), --gdn_impl auto (flash-linear-attention Triton kernels once
#    `pip install flash-linear-attention` has been run) and --grad_ckpt auto / --ce_ckpt auto, which probe
#    the configured shape once and skip both activation checkpointing and the CE's head-matmul recompute
#    when they fit (~1.6x together).  micro_batch is therefore kept small: what matters is that a
#    micro-batch fits WITHOUT checkpointing, and --grad_accum makes up the 524k tokens per step.
#    Measured on an 8 GB card at 524k tokens/step (README 2.1/2.2, absolute numbers differ per session):
#       mb16 x accum32                   6938 tok/s   3.9 GB
#       mb2  x accum256                  13042 tok/s  5.1 GB   <- this command
#       mb4  x accum128 --compile        18143 tok/s  5.2 GB   <- what --compile adds (~1.39x)
#    Re-measure with:
#      python tools/measure_batch.py --stage pretrain [--gdn_impl fla|torch] [--precision bf16|fp16|fp32]
#                                    [--no_grad_ckpt] [--compile]
python scripts/train.py --stage pretrain --tokenizer tokenizer --out_dir runs/pretrain \
    --data data/bin/en:0.50,data/bin/zh:0.30,data/bin/code:0.12,data/bin/math:0.08 \
    --tokens 6e9 --seq_len 1024 --micro_batch 2 --grad_accum 256 --lr 1e-3 --mtp_weight 0.3 --mtp_weight_final 0.1 \
    --compile --resume

# 3. vision alignment: LLM frozen, ViT + projector learn from captions (e.g. LLaVA-Pretrain 558k)
python scripts/train.py --stage align --tokenizer tokenizer --init_from runs/pretrain/final --out_dir runs/align \
    --mm_data data/mm/caption.jsonl --image_root data/mm/images --epochs 1 --micro_batch 8 --grad_accum 32 --resume

# 4. instruction / VQA SFT: everything trainable (mix image data with text-only chat so text ability is kept)
python scripts/train.py --stage sft --tokenizer tokenizer --init_from runs/align/final --out_dir runs/sft \
    --mm_data data/mm/sft_vqa.jsonl,data/text/sft_chat.jsonl --image_root data/mm/images --epochs 2 \
    --micro_batch 8 --grad_accum 16 --resume

python scripts/sample.py --model runs/sft/final --prompt "Describe this image." --image example.jpg
