#!/usr/bin/env python
"""Torch-free checks of the offline data path: the snapshot converters and what run_pipeline.sh points at.

Covers `scripts/prepare_local_data.py`'s text cleaners (the part of the pipeline with no other test), the
per-job snapshot inputs, and the recipe's wiring - every `$SNAP/...` source has to resolve on disk and no
`hf:` spec may survive in `run_pipeline.sh`, which is what keeps a training run off the network.

`--lengths` instead samples every source named in the recipe and prints its mean document length, which
is what the tokenizer's per-source `@N` budgets are derived from.
"""
import argparse
import glob
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from common_io import iter_texts, split_limit  # noqa: E402
from prepare_local_data import (JOBS, DEFAULT_SNAPSHOT, chunk_texts, clean_cornell,  # noqa: E402
                               clean_daily_dialog, clean_empathetic, clean_topical, han_ratio,
                               strip_speaker, strip_wikitext)

ROOT = os.path.join(os.path.dirname(__file__), "..")
ok = True


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")


def lengths(snap, samples=200):
    """Mean/median document length of every `--source` in the recipe (the tokenizer budgets' input)."""
    recipe = open(os.path.join(ROOT, "scripts", "run_pipeline.sh"), encoding="utf-8").read()
    code = "\n".join(ln.split("#")[0] for ln in recipe.splitlines()).replace("$SNAP", snap)
    seen = []
    for spec in re.findall(r'--source\s+"([^"]+)"', code):
        spec, limit = split_limit(spec)
        if spec in [s for s, _ in seen]:
            continue
        seen.append((spec, limit))
    print(f"{'source':<78}{'docs':>6}{'mean':>9}{'median':>9}")
    for spec, limit in seen:
        n = min(samples, limit) if limit else samples
        try:
            lens = [len(t) for t in iter_texts(spec, max_docs=n)]
        except FileNotFoundError as e:
            print(f"{spec[:77]:<78}{'-':>6}  {e}")
            continue
        if not lens:
            print(f"{spec[:77]:<78}{0:>6}  empty")
            continue
        print(f"{spec[:77]:<78}{len(lens):>6}{statistics.mean(lens):>9.0f}{statistics.median(lens):>9.0f}")


ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--lengths", action="store_true", help="sample the recipe's sources instead of checking")
ap.add_argument("--samples", type=int, default=200, help="documents to sample per source in --lengths")
args = ap.parse_args()
snap = os.environ.get("CIRNO_SNAPSHOT", DEFAULT_SNAPSHOT)
if args.lengths:
    lengths(snap, args.samples)
    sys.exit(0)


# ---- wikitext ------------------------------------------------------------------------------------
wiki = """蓝坝娘
{{到伪基百科|蓝坝娘}}
'''蓝坝——网络引航'''软件，由[http://example.org 伪音姬]设定
{| class="wikitable"
|-
! 年份 !! 事件
|-
| 2010 || 走红
|}
== 外部链接 ==
[[萌娘百科|百科条目]]，另见 [[分类:人物]] 与 [[File:x.png|缩略图]]
<ref name=a>注释</ref>
<!-- 隐藏注释 -->
"""
clean = strip_wikitext(wiki)
check("templates, tables, refs and comments are gone",
      all(t not in clean for t in ("{{", "}}", "{|", "|}", "<!--", "<ref", "class=")))
check("template content goes with the template", "伪基百科" not in clean)
check("prose, link labels and section titles survive",
      all(t in clean for t in ("蓝坝——网络引航", "百科条目", "外部链接", "伪音姬")))
check("media and category links are dropped", "File" not in clean and "分类" not in clean)
check("emphasis markup is dropped but its text stays", "'''" not in clean and "网络引航" in clean)

# ---- speaker prefixes ----------------------------------------------------------------------------
check("a quoted turn loses its character name",
      strip_speaker("秦绢：「是，不过莫大师伯可真的这么说。」") == "「是，不过莫大师伯可真的这么说。」")
check("the user/gpt chat-log prefixes are dropped",
      strip_speaker("user：Everything its end.") == "Everything its end."
      and strip_speaker("gpt：Pfft, evils won't cross.") == "Pfft, evils won't cross.")
check("narration keeps a colon that is not a speaker tag",
      strip_speaker("相信我：这就是蒙古民歌的起源。") == "相信我：这就是蒙古民歌的起源。")
check("a bare line is unchanged", strip_speaker("「哈呜，不需要照相机～」") == "「哈呜，不需要照相机～」")

# ---- chunking ------------------------------------------------------------------------------------
lines = [f"第{i}行" * 10 for i in range(20)]  # 40 chars each
docs = list(chunk_texts(lines, target=100))
check("chunks stay near the target and are made of whole lines",
      all(len(d) <= 100 + 41 and set(d.split("\n")) <= set(lines) for d in docs))
check("every line ends up in exactly one chunk", sum(d.count("\n") + 1 for d in docs) == len(lines))
check("a single over-long line is kept whole rather than cut",
      list(chunk_texts(["x" * 500], target=100)) == ["x" * 500])

check("Han ratio splits the ChatHaruhi files by language",
      han_ratio("令狐冲：「我做你们掌门」") > 0.5 > han_ratio("Harry: \"Hello there, Hermione.\""))

# ---- English small sets --------------------------------------------------------------------------
check("DailyDialog loses the space before its punctuation",
      clean_daily_dialog(["Say , Jim , how about going for a few beers ? "])
      == "Say, Jim, how about going for a few beers?")
check("EmpatheticDialogues restores the escaped comma",
      clean_empathetic("a lot of people_comma_ but it was fine", "that sounds rough")
      == "a lot of people, but it was fine\nthat sounds rough")
check("Topical-Chat markers become turn breaks",
      clean_topical("<s>[agent_1]Hello.</s><s>[agent_2]Hi!</s>") == "Hello.\nHi!")
check("Cornell loses its character prefix and CRLF",
      clean_cornell("BIANCA   They do not!\r\n") == "They do not!")

# ---- snapshot inputs -----------------------------------------------------------------------------
snap = os.environ.get("CIRNO_SNAPSHOT", DEFAULT_SNAPSHOT)
if not os.path.isdir(snap):
    print(f"[skip] snapshot jobs: {snap} is not mounted (CIRNO_SNAPSHOT points at it)")
else:
    for name, spec in JOBS.items():
        missing = [pat for pat in spec["sources"]
                   if not glob.glob(os.path.join(snap, pat), recursive=True)]
        check(f"snapshot inputs for '{name}' resolve", not missing, f"({missing})" if missing else "")

# ---- recipe wiring -------------------------------------------------------------------------------
recipe = open(os.path.join(ROOT, "scripts", "run_pipeline.sh"), encoding="utf-8").read()
code = "\n".join(ln.split("#")[0] for ln in recipe.splitlines())
check("the recipe never dereferences an hf: spec", not re.search(r"\bhf:", code) and "hf://" not in code)
check("the recipe forces the offline switches",
      all(f"{v}=1" in code for v in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")))
check("the recipe runs the snapshot converter",
      "prepare_local_data.py" in code and "CIRNO_SNAPSHOT" in recipe)

# A ref is the path alone: `$SNAP/dir/file.parquet:field@N` and comma-separated --mm_data lists both end
# the path at the first `:` / `,` / `@`.
refs = sorted({m for m in re.findall(r"\$SNAP/([^\s\"'):,@]+)", code)})
if not os.path.isdir(snap):
    print(f"[skip] {len(refs)} $SNAP source(s): snapshot not mounted")
else:
    missing = [r for r in refs if not glob.glob(os.path.join(snap, r), recursive=True)]
    check(f"all {len(refs)} $SNAP source(s) in the recipe exist", not missing, f"({missing})" if missing else "")

for glob_pat in sorted(set(re.findall(r"jsonl:(data/[^\"\s:]+)", code))):
    if not glob.glob(os.path.join(ROOT, glob_pat)):
        print(f"[note] {glob_pat} is empty - run scripts/prepare_local_data.py --out data")

print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
