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
#   EN       hf:noanabeshima/TinyStoriesV2:text:default:train          # en, ~550M tok; GPT-4 short stories
#            # (CDLA-Sharing-1.0), deliberately simple narrative.  Listed FIRST: it is what a cap that binds
#            # reads first, and it is what val.bin is taken from (see the EN paragraph below).
#   EN       hf:allenai/soda:dialogue:default:train                   # en, ~190M tok; INSTRUCTGPT-GENERATED
#   EN       hf:ParlAI/blended_skill_talk:free_messages:default:train # en, crowd-sourced: persona + empathy + knowledge
#   EN       hf:Fishball02/topical-chat:-:default:train               # en, MTurk; strip the <s>[agent_1] markup
#   EN       parquet:hf://datasets/li2017dailydialog/daily_dialog@refs/convert/parquet/default/train/*.parquet:dialog
#   EN       parquet:hf://datasets/facebook/empathetic_dialogues@refs/convert/parquet/default/train/*.parquet:prompt+utterance
#   FICTION  hf:wdndev/webnovel-chinese:text:default:train            # web novels (apache-2.0)
#   FICTION  hf:Miaowuawa/ChineseNovels:-:default:train               # one book per record, 623 MB; median
#            # 222k chars/book, so prepare_data's 200k-char cut keeps only the head of 56% of the books
#            # unless they are split per chapter (~254M of ~529M tokens survive the cut)
#   LONG     hf:wangrui6/Zhihu-KOL:RESPONSE:default:train             # human long-form answers
#   ROLEPLAY hf:shibing624/roleplay-zh-sharegpt-gpt4-data:-:sharegpt_formatted_data-evol-gpt4:train
#   ROLEPLAY hf:Johnson8187/role-play-chinese:-:default:train
#   ROLEPLAY hf:LooksJuicy/Chinese-Roleplay-SingleTurn:-:default:train  # character cards + one-turn replies
#            # (apache-2.0, ~3.3M tok).  `instruction` is not in normalize_messages' _PERSONA, so for SFT it
#            # has to be renamed `system_prompt` before the jsonl is written, or the card repeats in the
#            # user turn; for pre-training the whole record is read as text
#   CHAR     parquet:hf://datasets/silk-road/ChatHaruhi-54K-Role-Playing-Dialogue@refs/convert/parquet/default/train/*.parquet:user_question+agent_response
#            # novel/film characters; only the dialogue pair, the file's other 49% is metadata
#   CATGIRL  hf download kxdw2580/catgirl-dataset --include "v2/*"  ->  clean to jsonl in data/roleplay_catgirl
#            # gated (accept the terms first), the v2 files are JSON arrays rather than JSONL, and the
#            # answer's chain of thought is inline (<think>...</think>) - field selection cannot remove
#            # it, the text has to be cleaned before tokenising.  v2 only: the author rules out v1
#            # (identity leaks), v2/common-v2-extra (irrelevant Q&A) and the whole v2-qwen3 series.
#   WIKI     parquet:hf://datasets/lars1234/baidu-baike-dataset@refs/convert/parquet/default/partial-train/*.parquet:-
#                                                                     # 简体白话百科；parquet 3.2 GB，
#                                                                     # `hf:` 直连会拉 16.8 GB 的原始 json
#   WIKI     hf:mrzjy/chinese_moegirl_wiki_corpus_raw:text:default:train  # ACG trivia; RAW wikitext (99.6%
#            # of rows carry [[, 91.5% {{), so strip templates/tables/comments and write jsonl into
#            # data/wiki/ before tokenising; 291M chars is pre-cleaning, re-measure the token count
#   ---- Limour/b-corpus, in full (cc-by-nc-sa-4.0 + not-for-all-audiences) ----
#   Every corpus in that repo is one .txt.gz per work, mixed in with the uploader's own extractor
#   scripts, so each directory needs the same pre-step - decompress, split per line/chapter, drop the
#   `角色名：` / `system：` prefixes, write jsonl into the matching local dir (doc/datasets.md 1 has the
#   measured size of each one).  Bucket mapping:
#     v-corpus-zh   (820 gz / 188 MB)  -> CHAT      data/chat/          visual-novel dialogue
#     v-corpus-en   (135 gz /  69 MB)  -> EN        data/en_chat/       same, English side
#     emoji         (  1 gz / 2.1 MB)  -> EN        data/en_chat/       `user：`/`gpt：` chat log
#     ChatHaruhi    ( 38 gz / 2.5 MB)  -> CHAR      data/roleplay_char/ one file per character
#     q-corpus      ( 61 gz / 3.8 MB)  -> ROLEPLAY  data/roleplay/      R18 scenario text, filter first
#     近现代小说 / 翻译名著 / 轻小说 -> FICTION data/fiction/ (published works, provenance unclear)
#     提高逼格 / 畅销书 / 散文集 / 古籍 -> LONG data/long/ (古籍 is 文言, style pollution)
#     互动小说 is mrzjy/Chinese_interactive_novels_3k; read it from the Hub instead of unpacking the
#     7z parts - `parquet:hf://datasets/mrzjy/Chinese_interactive_novels_3k@refs/convert/parquet/default/train/*.parquet:book_chapter`
#     - and drop `content_tag`/`role` before writing, the field expression cannot.  00-preview/ and the
#     per-work *.py are statistics and extractors, not corpora.
#   With b-corpus, TinyStoriesV2, ChineseNovels, moegirl and LooksJuicy in, a bin's pool grows past the
#   `expected_tokens` recorded in configs/pretrain_plan.json (EN already carries its raised 860M cap,
#   because its new first source is what a low cap would cut).  Update those estimates, raise the step-1
#   --max_tokens of any bin the recipe now reads in full above its new pool, and re-run
#   `tools/plan_mix.py --plan configs/pretrain_plan.json` against the fresh meta.json before training
#   (doc/training.md 2).
# The EN bucket exists because allenai/soda is large (measured: 1.2M conversations, mean 632 chars of
# dialogue, ~191M tokens) - it is the only English set that can carry a weight.  It is synthetic
# (InstructGPT distilled on commonsense triples, `language_creators:machine-generated`); the human-written
# sets add up to ~17M tokens (doc/datasets.md §1.2) and ride along in the same bucket for style diversity,
# each needing cleaning (DailyDialog spaces before punctuation, EmpatheticDialogues `_comma_`,
# Topical-Chat `<s>[agent_1]`).  Order: TinyStoriesV2, then soda, then the human-written sets.
# The bucket also takes b-corpus v-corpus-en + the kajiwoto log (~52M) and TinyStoriesV2 (~550M), so its
# pool is ~810M.  Source order is a priority and validation knob, not a training order (TokenMixer
# shuffles every window of a bin): prepare_data reads the list in order, so a --max_tokens below
# TinyStoriesV2's own ~550M drops everything after it, and the first --val_tokens tokens - the bucket's
# val.bin, which is what train.py scores ppl on - come from the head of the list, i.e. TinyStoriesV2.
# Two things about the specs above, both of which bite silently:
#   * LCCC / DailyDialog / EmpatheticDialogues only ship a loading script, and `datasets`
#     >= 4 refuses scripts.  The `parquet:hf://...@refs/convert/parquet/...:field` form reads the Hub's
#     auto-converted parquet branch instead - keep the `@refs/...` revision, it is not a document limit.
#   * LCCC's `dialog` is word-segmented (`你 去 那儿 竟然`), so everything that takes it is tokenised
#     with --desegment, or the model learns to answer with spaces between characters.
# This project is non-commercial, so the NC-licensed sets are fine to use; `doc/datasets.md` §1.4 flags the
# sets with no licence at all (Zhihu-KOL) - those are the ones worth double-checking before publishing.
CHAT="jsonl:data/chat/*.jsonl:text"                  # human-to-human chat / forum / subtitles (zh)
EN="jsonl:data/en_chat/*.jsonl:text"                 # English: TinyStoriesV2, soda, then the human sets
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
#      i.e. roughly chat 19% / fiction 30% / wiki 26% / long 10% / en 15%;
#    * word-segmented corpora (LCCC) need --desegment or the merges are learned on spaced-out text;
#    * numbers stay whole (`\p{N}+`), not digit-by-digit as in Qwen2.
python scripts/train_tokenizer.py --out tokenizer --desegment \
    --source "$CHAT@2700000" --source "$EN@350000" --source "$FICTION@40000" \
    --source "$WIKI@34000" --source "$LONG@85000"

# 1. tokenise each source separately so the mixture can be re-weighted without re-tokenising.
#    The cap is a safety limit, not a budget: it sits at or above what the recipe allocates that bin, so
#    `train_tokens` in <out>/meta.json is the pool every phase samples from.  A cap that binds stops the
#    bin mid-source, which is a corpus left untrained, not a mixture correction - the weights come from
#    those pools (tools/plan_mix.py) and no bin is asked for more than the recipe allocates it.
#    EN is the one bin whose list order makes its cap load-bearing: TinyStoriesV2 is listed first and is
#    ~550M tokens on its own, so the 860M below (the estimated ~810M pool plus headroom) is what keeps
#    soda and the human-written sets from being cut off entirely.
#    WIKI is capped below its pool on purpose: the recipe allocates it 1.226B of the 1.425B, because
#    baidu-baike is ordered common entity -> rare entity and those 200M tokens are worth less to a 117M
#    model than the chat replay they pay for (doc/training.md 2).  Re-derive every number with:
#      python tools/plan_mix.py --plan configs/pretrain_plan.json
python scripts/prepare_data.py --tokenizer tokenizer --desegment --source "$CHAT"     --out data/bin/chat     --max_tokens 1020000000 --val_tokens 4000000
python scripts/prepare_data.py --tokenizer tokenizer             --source "$EN"       --out data/bin/en_chat  --max_tokens  860000000 --val_tokens 1000000
python scripts/prepare_data.py --tokenizer tokenizer             --source "$FICTION"  --out data/bin/fiction  --max_tokens 1610000000 --val_tokens 2500000
python scripts/prepare_data.py --tokenizer tokenizer             --source "$LONG"     --out data/bin/long     --max_tokens  530000000 --val_tokens 2000000
python scripts/prepare_data.py --tokenizer tokenizer --desegment --source "$ROLEPLAY" --out data/bin/roleplay         --max_tokens   18000000 --val_tokens 200000
python scripts/prepare_data.py --tokenizer tokenizer --desegment --source "$CHAR"     --out data/bin/roleplay_char    --max_tokens    5200000 --val_tokens 100000
python scripts/prepare_data.py --tokenizer tokenizer --desegment --source "$CATGIRL" --out data/bin/roleplay_catgirl --max_tokens     180000 --val_tokens  20000
python scripts/prepare_data.py --tokenizer tokenizer             --source "$WIKI"     --out data/bin/wiki     --max_tokens 1228050000 --val_tokens 2500000

# 2. text pre-training: configs/pretrain_plan.json is the recipe, and plan_mix.py is what turns it into
#    runnable phases.  The mixture is a schedule, not a constant - phase A spends the first 1.21B on
#    long-form only at seq 2048, phase B brings dialogue in for 3.10B, phase C decays on chat + role-play
#    for the last 1.08B - and the weights come out of the bins' measured pools, so they are derived
#    rather than typed.  chat is the one bin read more than once (allocate 1.212B for a 1.010B pool):
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

# 3. (optional) vision alignment: only when the deployment really sees images (stickers / memes /
#    chat screenshots).  LLM frozen, ViT + projector learn from IN-CONVERSATION images, not academic
#    VQA captions - VQA data pulls the model back towards assistant behaviour.  Skip this stage for a
#    text-only chat model and point step 4 at runs/pretrain/final instead.
#    Open image-chat sets keep their images inside the dataset, so they still have to be exported to
#    --image_root by hand; this line expects an already-written jsonl.
python scripts/train.py --stage align --plan runs/pretrain/plan.resolved.json --phase align \
    --tokenizer tokenizer --init_from runs/pretrain/final --out_dir runs/align \
    --mm_data data/mm/chat_image.jsonl --image_root data/mm/images --resume

# 4. dialogue / role-play SFT: human-to-human conversations + personas + a small out-of-scope refusal
#    set ("I don't know, ask an AI").  Only the imitated side (the `self` role in the encoder)
#    is scored; `--train llm` matches the text-only path (use runs/pretrain/final if step 3 was skipped).
#    --mm_weights is what turns the recipe's 74 / 21 / 5 slice into an actual mixture: without it a
#    source's share of the batches is whatever its line count happens to be, which is how a 500-row
#    refusal set disappears next to a million-row chat dump.
#    --mm_data takes source specs too, and remote ones are spooled into --mm_cache (data/cache) once.
#    LCCC comes through its converted parquet branch and needs --desegment here as well, or the SFT
#    targets are word-segmented:
#      --mm_data "parquet:hf://datasets/silver/lccc@refs/convert/parquet/base/train/*.parquet:dialog,hf:shibing624/roleplay-zh-sharegpt-gpt4-data:-:sharegpt_formatted_data-evol-gpt4:train,hf:Johnson8187/role-play-chinese:-:default:train" --desegment
#    LooksJuicy/Chinese-Roleplay-SingleTurn can join SFT too, but rename its `instruction` to
#    `system_prompt` when writing the jsonl, or the card repeats in every user turn (doc/datasets.md 1.6).
#    The refusal slice (2-8%) has no off-the-shelf source: write it yourself per doc/training.md 4.4.
python scripts/train.py --stage sft --plan runs/pretrain/plan.resolved.json --phase sft \
    --tokenizer tokenizer --init_from runs/align/final --out_dir runs/sft \
    --mm_data data/sft/human_dialog.jsonl,data/sft/persona.jsonl,data/sft/refuse.jsonl \
    --mm_weights 0.74,0.21,0.05 --desegment --resume

python scripts/sample.py --model runs/sft/final --prompt "在吗" --max_new_tokens 32
