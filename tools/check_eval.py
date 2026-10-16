#!/usr/bin/env python
"""Torch-free checks of hvlm/eval_core.py and the committed eval sets (evals/*.jsonl, configs/eval.json)."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hvlm import eval_core as ec  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
ok = True


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")


# ---- text normalisation and matching -----------------------------------------------------------
check("full-width digits and punctuation normalise onto ASCII",
      ec.matches("一共１２个月。", ["12"]) and ec.matches("It's 24.", ["24"]))
check("a Chinese numeral is matched by its own spelling, not converted",
      ec.matches("二十四岁", ["24", "二十四"]) and not ec.matches("二十四岁", ["24"]))
check("rejected keywords block a match", not ec.matches("是12个月", ["12"], ["12个月"])
      and ec.matches("是12个月", ["12"], ["13"]))
check("an unknown answer is recognised", ec.is_unknown("我也不知道诶") and ec.is_unknown("这个说不准")
      and not ec.is_unknown("是12个月"))
check("assistant tone is detected", ec.tone_hits("## 总结\n1. 先这样") == ["markdown_heading", "ordered_list"]
      and ec.tone_hits("很高兴为您服务") == ["service_formula"])
check("a normal IM reply trips no tone pattern", not ec.tone_hits("在的在的 咋啦") and not ec.tone_hits("哈哈哈哈"))
check("claiming to be an AI is distinguished from a persona name",
      ec.is_self_reference("我是一个AI助手") and not ec.is_self_reference("我是小雨"))
check("4-gram repetition is flagged, short text is not",
      ec.has_repeat("哈哈哈哈哈哈哈哈哈哈") and not ec.has_repeat("在吗") and not ec.has_repeat("今天天气不错啊"))

stats = ec.length_stats([1, 2, 3, 4, 200])
check("length stats report the IM share", stats["n"] == 5 and stats["max"] == 200 and stats["long_share"] == 0.2,
      f"({stats})")
check("an empty length distribution is not a crash", ec.length_stats([])["n"] == 0)


# ---- item scoring ------------------------------------------------------------------------------
fact = dict(kind="knowledge", id="k1", category="fact", prompt="一年有多少个月？", any_of=["12", "十二"],
            expect_unknown=False)
unknown = dict(kind="knowledge", id="k2", category="unknown", prompt="我昨天几点睡的？", expect_unknown=True)
samples = lambda *texts: [dict(text=t, n_tokens=3) for t in texts]  # noqa: E731
check("a correct answer scores a hit", ec.score_knowledge(fact, samples("十二个月。"))["hit"])
check("a wrong answer does not", not ec.score_knowledge(fact, samples("十个月吧"))["hit"])
check("an unanswerable probe answered with a guess is flagged",
      ec.score_knowledge(unknown, samples("你昨晚11点睡的"))["over_answer"])
check("an unanswerable probe answered with uncertainty passes",
      ec.score_knowledge(unknown, samples("我哪知道呀"))["unknown"])

persona = dict(kind="persona", id="p1", persona="你是小雨，24岁。",
               probes=[dict(field="name", question="你叫什么？", any_of=["小雨"]),
                       dict(field="age", question="你多大？", any_of=["24"])])
same = ec.score_persona(persona, 0, samples("我叫小雨", "小雨呀", "我是小雨"))
diff = ec.score_persona(persona, 0, samples("我叫小雨", "我叫小美", "我是小雨"))
check("persona consistency needs every sample to agree", same["consistent"] and not diff["consistent"])
check("persona accuracy uses the card's keywords",
      ec.score_persona(persona, 0, samples("我叫小雨"))["hit"] and
      not ec.score_persona(persona, 1, samples("我今年30"))["hit"])
no_persona = dict(kind="persona", id="np1", persona=None,
                  probes=[dict(field="name", question="你叫什么？")])
check("a no-persona probe compares samples with each other",
      ec.score_persona(no_persona, 0, samples("小雪", "小雪"))["consistent"]
      and not ec.score_persona(no_persona, 0, samples("小雪", "小冰"))["consistent"])

rp = dict(kind="roleplay", id="rp1", persona="猫娘", history=[dict(role="user", content="在吗")],
          rubric=[dict(name="口癖", any_of=["喵"]), dict(name="不出戏", not_any_of=["作为一个AI"])])
verdict = ec.score_roleplay(rp, samples("喵～主人好", "在呢"))
check("a rubric item passes when any sample satisfies it",
      verdict["rubrics"]["口癖"] and verdict["rubrics"]["不出戏"] and verdict["all_pass"])
check("a negative rubric item fails on an out-of-character reply",
      not ec.score_roleplay(rp, samples("作为一个AI，我可以帮你"))["all_pass"])


# ---- aggregation, gates, rendering -------------------------------------------------------------
kn_items = [ec.score_knowledge(fact, samples("12个月")), ec.score_knowledge(unknown, samples("猜不到"))]
kn_samples = {"k1": samples("12个月"), "k2": samples("猜不到")}
metrics = ec.aggregate("knowledge", kn_items, kn_samples)
check("knowledge aggregate reports accuracy and unknown handling",
      metrics["accuracy"] == 1.0 and metrics["unknown_rate"] == 1.0 and metrics["over_answer_rate"] == 0.0,
      f"({ {k: metrics[k] for k in ('accuracy', 'unknown_rate', 'over_answer_rate')} })")
check("the text block rides along on every set", metrics["tone_rate"] == 0.0 and metrics["n_samples"] == 2)

roll = ec.aggregate("roleplay", [verdict], {"rp1": samples("喵～主人好")})
check("roleplay aggregate reports per-item and overall rates",
      roll["rubric_pass_rate"] == {"口癖": 1.0, "不出戏": 1.0} and roll["overall_pass_rate"] == 1.0)
im = ec.aggregate("imform", [dict(id="im1")], {"im1": samples("在呢", "咋啦")},
                  prior=dict(tokens=dict(median=8, p90=20)))
check("imform aggregate compares against the human prior",
      im["length"]["median"] == 3.0 and abs(im["median_ratio"] - 0.375) < 1e-9 and im["prior"]["tokens"]["median"] == 8)

gates = ec.evaluate_gates({"imform": dict(tone_rate=0.0, empty_rate=0.5, median_ratio=1.2)},
                          {"imform.tone_rate": {"max": 0.02}, "imform.empty_rate": {"max": 0.01},
                           "imform.median_ratio": {"min": 0.5}, "imform.repeat_rate": {"max": 0.1}})
verdicts = {c["metric"]: c["ok"] for c in gates["checks"]}
check("gates pass, fail and treat a missing metric as a failure",
      gates["passed"] is False and verdicts == {"imform.tone_rate": True, "imform.empty_rate": False,
                                                "imform.median_ratio": True, "imform.repeat_rate": False},
      f"({verdicts})")
partial = ec.evaluate_gates({"imform": dict(tone_rate=0.0)}, {"imform.tone_rate": {"max": 0.02},
                                                              "persona.self_reference_rate": {"max": 0.02}})
check("a gate whose set was not evaluated is skipped, not failed",
      partial["passed"] and {c["metric"]: c.get("skipped") for c in partial["checks"]} ==
      {"imform.tone_rate": False, "persona.self_reference_rate": True},
      f"({[(c['metric'], c.get('skipped')) for c in partial['checks']]})")
check("metric lookup ignores non-numeric nodes", ec.metric_value({"a": dict(b="x")}, "a.b") is None)

md = ec.render_markdown(dict(tag="t", model="m", sets={"imform": dict(kind="imform", file="f", n_items=1,
                                                                    n_generations=1, skipped=0,
                                                                    metrics=dict(tone_rate=0.0))},
                             gates=gates, cost=dict(prefill_ms=3.0), blind=None))
check("the markdown report carries the set, the gate verdict and the cost",
      "imform" in md and "FAIL" in md and "prefill_ms" in md)
check("an empty report renders", "评测报告" in ec.render_markdown({}))


# ---- blind test --------------------------------------------------------------------------------
items = [("b0", [dict(role="user", content="在吗")]), ("b1", [dict(role="user", content="吃了吗")])]
pairs, key = ec.make_blind_pairs(items, ["喵喵喵", "吃了吃了"], ["在的在的", "刚吃完"], seed=7)
check("blind pairs carry both arms and the key keeps the model's side",
      len(pairs) == 2 and {p["id"] for p in pairs} == {"b0", "b1"} and
      all(key[i]["model_arm"] in ("a", "b") for i in key))
picks = [dict(id="b0", pick=key["b0"]["model_arm"]), dict(id="b1", pick="b" if key["b1"]["model_arm"] == "a" else "a")]
blind = ec.score_blind_picks(picks, key, pairs)
check("blind scoring splits model/human wins", blind["n"] == 2 and blind["model_win_rate"] in (0.0, 0.5, 1.0),
      f"({blind})")
parrot_pairs = [dict(id="b0", messages=[], a="在的在的", b="在的在的")]
parrot = ec.score_blind_picks([dict(id="b0", pick="tie")], {"b0": {"model_arm": "a"}}, parrot_pairs)
check("an identical human arm is counted as parroting", parrot["parrot_rate"] == 1.0)


# ---- set schema --------------------------------------------------------------------------------
def rejects(rec, why):
    try:
        ec.validate_record(rec, "t")
    except ec.EvalError:
        return True
    print(f"       (accepted {why})")
    return False


check("schema rejects a category/expect_unknown mismatch",
      rejects(dict(kind="knowledge", id="x", category="fact", prompt="p", expect_unknown=True), "mismatch"))
check("schema rejects an answerable probe without keywords",
      rejects(dict(kind="knowledge", id="x", category="fact", prompt="p", expect_unknown=False), "no any_of"))
check("schema rejects a keyworded no-persona probe",
      rejects(dict(kind="persona", id="x", persona=None, probes=[dict(field="name", question="q", any_of=["a"])]),
              "no-persona + any_of"))
check("schema rejects an imform record that does not end on a user turn",
      rejects(dict(kind="imform", id="x", tags=["t"], messages=[dict(role="self", content="hi")]), "self-last"))
check("schema rejects a rubric item with no keywords",
      rejects(dict(kind="roleplay", id="x", persona="p", history=[dict(role="user", content="q")],
                   rubric=[dict(name="n")]), "empty rubric item"))
check("schema rejects a roleplay history that ends on a self turn",
      rejects(dict(kind="roleplay", id="x", persona="p", history=[dict(role="self", content="q")],
                   rubric=[dict(name="n", any_of=["a"])]), "self-last history"))

with tempfile.TemporaryDirectory() as tmp:
    dup = os.path.join(tmp, "dup.jsonl")
    with open(dup, "w", encoding="utf-8") as f:
        f.write(json.dumps(fact) + "\n" + json.dumps(fact) + "\n")
    try:
        ec.load_set(dup)
        check("duplicate ids are rejected", False)
    except ec.EvalError:
        check("duplicate ids are rejected", True)

check("prompt shapes per kind",
      ec.prompt_messages(fact) == [dict(role="user", content="一年有多少个月？")] and
      ec.prompt_messages(persona, persona["probes"][0])[0]["role"] == "system" and
      ec.prompt_messages(rp) == [dict(role="system", content="猫娘"), dict(role="user", content="在吗")])

# ---- the committed sets and config -------------------------------------------------------------
config_path = os.path.join(ROOT, "configs", "eval.json")
with open(config_path, encoding="utf-8") as f:
    config = json.load(f)
totals = {}
for name, path in config["sets"].items():
    records = ec.load_set(os.path.join(ROOT, path))
    kind = ec.set_kind(records)
    totals[name] = len(records)
    check(f"{path} parses and its records match their schema", True, f"({kind}, {len(records)} records)")
    check(f"{name} has a sampling block", kind in config["sampling"])
for path in config["gates"]:
    set_name = path.split(".")[0]
    check(f"gate {path} refers to a configured set", set_name in config["sets"])
check("the config names a human length prior", isinstance(config.get("prior"), str) and config["prior"])
prior_path = os.path.join(ROOT, config["prior"])
if os.path.exists(prior_path):
    with open(prior_path, encoding="utf-8") as f:
        prior = json.load(f)
    check("a built human length prior has the fields imform needs",
          prior["tokens"]["median"] > 0 and prior["n"] > 0)
else:
    check("no prior built yet; imform length ratios stay empty (scripts/eval.py --length-prior)", True)
print(f"\nsets: {totals}")
print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
