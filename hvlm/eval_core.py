"""Generation-based evaluation: eval-set schema, scoring, aggregation, gates and report rendering.

Torch-free on purpose.  ``scripts/eval.py`` owns the model and turns each record into decoded samples;
everything that decides *what a sample means* lives here, so the scoring contract is unit-testable
without a GPU (``tools/check_eval.py``) and the committed eval sets are validated by the same code that
scores them.

A set file is jsonl, one record per line, each carrying its own ``kind``:

    knowledge  {"kind","id","category","prompt", "any_of"?,"not_any_of"?,"expect_unknown",
                "context"?, "image"?}
    persona    {"kind","id","persona"|null,"probes":[{"field","question","any_of"?}]}
    imform     {"kind","id","tags","messages":[{"role","content"}],"image"?}
    roleplay   {"kind","id","persona","history":[{"role","content"}],
                "rubric":[{"name","any_of"?,"not_any_of"?}]}

Matching is substring based on normalised text (:func:`normalize`), which is what makes a 130M model's
answers scorable without a judge model.  The lexicons below are the negative-asset list of
doc/training.md 4.1: assistant tone and AI self-description are the two things this positioning treats
as failures, so they are counted separately from the set-specific metrics.
"""
import json
import re
import statistics
import unicodedata
from collections import Counter

KINDS = ("knowledge", "persona", "imform", "roleplay")
KNOWLEDGE_CATEGORIES = ("fact", "weather", "everyday", "unknown")
ROLES = ("system", "user", "self")
REPLY_TOKENS_IM = 128  # a reply longer than this is outside the IM distribution (doc/training.md 4.1)

# "I do not know" in the registers this model is meant to use; used to score the expect_unknown probes.
UNCERTAIN = ("不知道", "不清楚", "不晓得", "不记得", "想不起来", "说不好", "说不准", "没法知道",
             "无法知道", "猜不到", "不敢说", "不知道诶", "我哪知道", "没有听说", "没听说过")

# Assistant tone: structural markdown, service formulas and report-style enumeration.
TONE_PATTERNS = (
    ("markdown_heading", re.compile(r"(?m)^\s{0,3}#{1,6}\s")),
    ("ordered_list", re.compile(r"(?m)^\s*\d{1,2}[.、)]\s")),
    ("bullet_list", re.compile(r"(?m)^\s*[-*\u2022]\s")),
    ("service_formula", re.compile("很高兴(为您|为你)服务|我可以(帮你|帮您|协助你|为你|为您)"
                                   "|有什么(可以|能)帮|希望(能|可以)帮到你|请问还有什么"
                                   "|随时(可以)?找我|还有什么(可以|需要)帮")),
    ("report_style", re.compile(r"(?m)^\s*(总结|综上|首先|其次|最后|另外)[:：，,]\s*")),
)
# Claiming to be an assistant / AI is a failure of the positioning itself, not just of style.
SELF_REFERENCE = re.compile("我是(一个|一名)?\\s*(AI|人工智能|语言模型|大语言模型|机器人|智能助手|助手|智能体)"
                            "|作为(一个|一名)?\\s*(AI|人工智能|语言模型|助手|智能助手)"
                            "|我是一个?(AI|人工智能)助手")
# Characters kept by normalize(): letters, digits and marks.  Punctuation/symbols/separators are dropped so
# "24。" and "24" match, and NFKC folds full-width forms onto their ASCII equivalents.
_PUNCT = ("P", "S", "Z", "C")


def normalize(text):
    """Case/width/punctuation-insensitive form used by every matcher and by the parrot check."""
    if not isinstance(text, str):
        text = str(text)
    out = []
    for ch in unicodedata.normalize("NFKC", text).lower():
        if unicodedata.category(ch)[0] not in _PUNCT:
            out.append(ch)
    return "".join(out)


def hits(text, patterns):
    """Substrings of `patterns` present in the normalised `text`."""
    n = normalize(text)
    return [p for p in patterns if normalize(p) in n]


def matches(text, any_of=(), not_any_of=()):
    """True when any accepted keyword occurs and no rejected one does."""
    if not any_of:
        return False
    if not hits(text, any_of):
        return False
    return not hits(text, not_any_of)


def extract(text, any_of):
    """The accepted keyword the reply contains (canonical spelling), or None."""
    found = hits(text, any_of)
    return found[0] if found else None


def is_unknown(text):
    return bool(hits(text, UNCERTAIN))


def tone_hits(text):
    """Names of the assistant-tone patterns a reply trips."""
    return [name for name, rx in TONE_PATTERNS if rx.search(text or "")]


def is_self_reference(text):
    return bool(SELF_REFERENCE.search(text or ""))


def has_repeat(text, n=4, min_ngrams=8):
    """True when a reply repeats an n-gram: the observed shape of degenerate decoding."""
    s = normalize(text)
    if len(s) < max(n, min_ngrams):
        return False
    grams = [s[i:i + n] for i in range(len(s) - n + 1)]
    return len(set(grams)) < len(grams)


class EvalError(ValueError):
    """A malformed eval set: reported with the file and line so it can be fixed at the source."""


# ------------------------------------------------------------------------------------------------
# schema
# ------------------------------------------------------------------------------------------------
def _need(rec, key, where, type_=None):
    if key not in rec or rec[key] is None:
        raise EvalError(f"{where}: missing {key!r}")
    if type_ is not None and not isinstance(rec[key], type_):
        raise EvalError(f"{where}: {key!r} must be {type_.__name__}, got {type(rec[key]).__name__}")
    return rec[key]


def _keywords(value, key, where, required):
    if value is None:
        if required:
            raise EvalError(f"{where}: {key!r} is required here")
        return []
    if not isinstance(value, list) or not value or not all(isinstance(x, str) and x.strip() for x in value):
        raise EvalError(f"{where}: {key!r} must be a non-empty list of strings")
    return value


def _messages(value, where, key="messages", roles=ROLES, non_empty=True):
    if not isinstance(value, list) or (non_empty and not value):
        raise EvalError(f"{where}: {key!r} must be a non-empty list")
    for i, m in enumerate(value):
        if not isinstance(m, dict) or m.get("role") not in roles or not str(m.get("content") or "").strip():
            raise EvalError(f"{where}: {key}[{i}] must be {{'role': one of {roles}, 'content': str}}")
    return value


def validate_record(rec, where):
    """Raise EvalError unless `rec` satisfies the schema of its own `kind`."""
    if not isinstance(rec, dict):
        raise EvalError(f"{where}: record is not an object")
    kind = rec.get("kind")
    if kind not in KINDS:
        raise EvalError(f"{where}: kind must be one of {KINDS}, got {kind!r}")
    _need(rec, "id", where, str)
    if rec.get("image") is not None and not isinstance(rec["image"], str):
        raise EvalError(f"{where}: 'image' must be a path string")

    if kind == "knowledge":
        cat = _need(rec, "category", where, str)
        if cat not in KNOWLEDGE_CATEGORIES:
            raise EvalError(f"{where}: category must be one of {KNOWLEDGE_CATEGORIES}, got {cat!r}")
        _need(rec, "prompt", where, str)
        unknown = _need(rec, "expect_unknown", where, bool)
        if unknown != (cat == "unknown"):
            raise EvalError(f"{where}: category {cat!r} and expect_unknown={unknown} disagree")
        _keywords(rec.get("any_of"), "any_of", where, required=not unknown)
        _keywords(rec.get("not_any_of"), "not_any_of", where, required=False)
        if rec.get("context") is not None:
            _messages(rec["context"], where, key="context", roles=("user", "self"))
    elif kind == "persona":
        persona = rec.get("persona")
        if persona is not None and not (isinstance(persona, str) and persona.strip()):
            raise EvalError(f"{where}: 'persona' must be a non-empty string or null")
        probes = _need(rec, "probes", where, list)
        if not probes:
            raise EvalError(f"{where}: 'probes' is empty")
        for i, p in enumerate(probes):
            w = f"{where}: probes[{i}]"
            if not isinstance(p, dict):
                raise EvalError(f"{w} is not an object")
            _need(p, "field", w, str)
            _need(p, "question", w, str)
            if persona is None and p.get("any_of"):
                raise EvalError(f"{w}: a no-persona probe must not carry any_of")
            if persona is not None:
                _keywords(p.get("any_of"), "any_of", w, required=True)
    elif kind == "imform":
        msgs = _messages(rec.get("messages"), where)
        if msgs[-1]["role"] != "user":
            raise EvalError(f"{where}: the last message must be a user turn")
        if rec.get("tags") is not None:
            _keywords(rec["tags"], "tags", where, required=False)
    else:  # roleplay
        _need(rec, "persona", where, str)
        hist = _messages(rec.get("history"), where, key="history", roles=("user", "self"))
        if hist[-1]["role"] != "user":
            raise EvalError(f"{where}: history must end on a user turn")
        rubric = _need(rec, "rubric", where, list)
        if not rubric:
            raise EvalError(f"{where}: 'rubric' is empty")
        for i, it in enumerate(rubric):
            w = f"{where}: rubric[{i}]"
            if not isinstance(it, dict):
                raise EvalError(f"{w} is not an object")
            _need(it, "name", w, str)
            any_of = _keywords(it.get("any_of"), "any_of", w, required=False)
            not_any_of = _keywords(it.get("not_any_of"), "not_any_of", w, required=False)
            if not any_of and not not_any_of:
                raise EvalError(f"{w}: needs any_of or not_any_of")
    return kind


def load_set(path, limit=None):
    """Parse and validate a jsonl eval set; duplicates and schema errors fail at load time."""
    records, seen = [], set()
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvalError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
            validate_record(rec, f"{path}:{lineno}")
            if rec["id"] in seen:
                raise EvalError(f"{path}:{lineno}: duplicate id {rec['id']!r}")
            seen.add(rec["id"])
            records.append(rec)
            if limit is not None and len(records) >= limit:
                break
    if not records:
        raise EvalError(f"{path}: no records")
    return records


def set_kind(records):
    kinds = {r["kind"] for r in records}
    if len(kinds) != 1:
        raise EvalError(f"a set must hold one kind, got {sorted(kinds)}")
    return kinds.pop()


def prompt_messages(rec, probe=None):
    """The chat context the model continues; the caller appends the opening `self` turn head."""
    kind = rec["kind"]
    if kind == "knowledge":
        return list(rec.get("context") or []) + [{"role": "user", "content": rec["prompt"]}]
    if kind == "persona":
        if probe is None:
            raise EvalError("a persona record needs a probe")
        out = [{"role": "system", "content": rec["persona"]}] if rec.get("persona") else []
        return out + [{"role": "user", "content": probe["question"]}]
    if kind == "imform":
        return list(rec["messages"])
    if kind == "roleplay":
        return [{"role": "system", "content": rec["persona"]}] + list(rec["history"])
    raise EvalError(f"unknown kind {kind!r}")


# ------------------------------------------------------------------------------------------------
# scoring
# ------------------------------------------------------------------------------------------------
def length_stats(counts):
    """Reply-length distribution in tokens; `long_share` is the part outside the IM distribution."""
    counts = sorted(int(c) for c in counts)
    if not counts:
        return dict(n=0, mean=0.0, median=0.0, p90=0.0, p99=0.0, max=0, long_share=0.0)
    at = lambda q: counts[min(len(counts) - 1, int(q * len(counts)))]  # noqa: E731
    return dict(n=len(counts), mean=round(statistics.fmean(counts), 2), median=float(at(0.5)),
                p90=float(at(0.9)), p99=float(at(0.99)), max=counts[-1],
                long_share=round(sum(c > REPLY_TOKENS_IM for c in counts) / len(counts), 4))


def text_block(samples):
    """Style metrics that apply to every generative set (samples: [{"text","n_tokens"}])."""
    texts = [s["text"] for s in samples]
    n = len(texts) or 1
    return dict(
        n_samples=len(texts),
        empty_rate=round(sum(not t.strip() for t in texts) / n, 4),
        tone_rate=round(sum(bool(tone_hits(t)) for t in texts) / n, 4),
        tone_hits=dict(Counter(h for t in texts for h in tone_hits(t))),
        self_reference_rate=round(sum(is_self_reference(t) for t in texts) / n, 4),
        repeat_rate=round(sum(has_repeat(t) for t in texts) / n, 4),
        length=length_stats([s["n_tokens"] for s in samples]),
    )


def score_knowledge(rec, samples):
    """Item-level result for one knowledge probe (a hit in any sample counts as answered)."""
    answers = [s["text"] for s in samples]
    if rec["expect_unknown"]:
        unknown = any(is_unknown(a) for a in answers)
        return dict(id=rec["id"], category=rec["category"], expect_unknown=True,
                    unknown=unknown, over_answer=not unknown)
    hit = any(matches(a, rec.get("any_of", []), rec.get("not_any_of", [])) for a in answers)
    return dict(id=rec["id"], category=rec["category"], expect_unknown=False, hit=hit)


def score_persona(rec, probe_index, samples):
    """Item-level result for one (record, probe) pair; `values` feeds the consistency metric."""
    probe = rec["probes"][probe_index]
    values = [extract(s["text"], probe.get("any_of", [])) or normalize(s["text"]) for s in samples]
    return dict(id=f"{rec['id']}.{probe['field']}", field=probe["field"], persona=rec.get("persona"),
                expected=probe.get("any_of") or [], values=values,
                hit=(extract(values[0], probe["any_of"]) if probe.get("any_of") else None),
                consistent=len(set(values)) == 1)


def rubric_pass(item, text):
    any_of, not_any_of = item.get("any_of", []), item.get("not_any_of", [])
    if any_of and not hits(text, any_of):
        return False
    return not (not_any_of and hits(text, not_any_of))


def score_roleplay(rec, samples):
    """One rubric verdict per item (a sample passing is enough: the model is sampled, not deterministic)."""
    verdicts = {it["name"]: any(rubric_pass(it, s["text"]) for s in samples) for it in rec["rubric"]}
    return dict(id=rec["id"], rubrics=verdicts, all_pass=all(verdicts.values()))


def aggregate(kind, items, samples_by_id, prior=None):
    """Turn per-item results into the metric block stored in report.json."""
    all_samples = [s for samples in samples_by_id.values() for s in samples]
    metrics = dict(n_items=len(items), **text_block(all_samples))
    if kind == "knowledge":
        answerable = [i for i in items if not i["expect_unknown"]]
        unknown = [i for i in items if i["expect_unknown"]]
        metrics["accuracy"] = round(sum(i["hit"] for i in answerable) / len(answerable), 4) if answerable else None
        metrics["unknown_rate"] = round(sum(i["unknown"] for i in unknown) / len(unknown), 4) if unknown else None
        metrics["over_answer_rate"] = round(sum(i["over_answer"] for i in unknown) / len(unknown), 4) if unknown else None
        metrics["by_category"] = {c: round(sum(i["hit"] for i in answerable if i["category"] == c)
                                           / max(1, sum(1 for i in answerable if i["category"] == c)), 4)
                                  for c in sorted({i["category"] for i in answerable})}
        metrics["missed"] = [i["id"] for i in answerable if not i["hit"]]
        metrics["over_answered"] = [i["id"] for i in unknown if i["over_answer"]]
    elif kind == "persona":
        with_expectation = [i for i in items if i["expected"]]
        metrics["identity_accuracy"] = round(sum(i["hit"] is not None for i in with_expectation)
                                             / len(with_expectation), 4) if with_expectation else None
        metrics["consistency"] = round(sum(i["consistent"] for i in items) / len(items), 4)
        no_persona = [i["values"][0] for i in items if not i["persona"]]
        metrics["identity_distribution"] = dict(Counter(no_persona).most_common(8))
        metrics["by_field"] = {f: round(sum(i["consistent"] for i in items if i["field"] == f)
                                        / max(1, sum(1 for i in items if i["field"] == f)), 4)
                               for f in sorted({i["field"] for i in items})}
    elif kind == "roleplay":
        names = sorted({n for i in items for n in i["rubrics"]})
        metrics["rubric_pass_rate"] = {n: round(sum(i["rubrics"][n] for i in items) / len(items), 4) for n in names}
        metrics["overall_pass_rate"] = round(sum(i["all_pass"] for i in items) / len(items), 4)
        metrics["failed"] = [i["id"] for i in items if not i["all_pass"]]
    else:  # imform
        metrics["prior"] = prior
        if prior and prior.get("tokens", {}).get("median"):
            metrics["median_ratio"] = round(metrics["length"]["median"] / prior["tokens"]["median"], 4)
        if prior and prior.get("tokens", {}).get("p90"):
            metrics["p90_ratio"] = round(metrics["length"]["p90"] / prior["tokens"]["p90"], 4)
    return metrics


# ------------------------------------------------------------------------------------------------
# gates and report
# ------------------------------------------------------------------------------------------------
def metric_value(sets_metrics, path):
    """Resolve `set.metric` (or `set.metric.sub`) from the per-set metric blocks."""
    parts = path.split(".")
    node = sets_metrics
    for p in parts:
        if not isinstance(node, dict) or p not in node:
            return None
        node = node[p]
    return node if isinstance(node, (int, float)) and not isinstance(node, bool) else None


def evaluate_gates(sets_metrics, gates):
    """Apply `{"set.metric": {"max"|"min": x, "note"?: str}}`.

    A gate whose set this run did not evaluate is skipped (`--sets imform` must not fail the persona gate);
    a gate whose set *was* evaluated but produced no such metric fails, because that is a real gap between
    configs/eval.json and what the scorer reports."""
    checks = []
    for path, spec in sorted(gates.items()):
        lo, hi, note = spec.get("min"), spec.get("max"), spec.get("note")
        if path.split(".")[0] not in sets_metrics:
            checks.append(dict(metric=path, value=None, min=lo, max=hi, ok=True, skipped=True,
                               detail="set not evaluated in this run", note=note))
            continue
        value = metric_value(sets_metrics, path)
        if value is None:
            ok, why = False, "metric not produced by this run"
        elif lo is not None and value < lo:
            ok, why = False, f"{value} < min {lo}"
        elif hi is not None and value > hi:
            ok, why = False, f"{value} > max {hi}"
        else:
            ok, why = True, f"{value}"
        checks.append(dict(metric=path, value=value, min=lo, max=hi, ok=ok, skipped=False, detail=why,
                           note=note))
    return dict(passed=all(c["ok"] for c in checks), checks=checks)


def _fmt(value, digits=4):
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    if isinstance(value, dict):
        return ", ".join(f"{k}={_fmt(v, digits)}" for k, v in value.items()) or "-"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value) or "-"
    return str(value)


def render_markdown(report):
    """Human-readable report: one metric table per set, then gates, cost and blind test."""
    lines = [f"# 评测报告 {report.get('tag', '')}", ""]
    meta = [("model", report.get("model")), ("tokenizer", report.get("tokenizer")),
            ("created", report.get("created")), ("device", report.get("device")),
            ("precision", report.get("precision")), ("gdn_impl", report.get("gdn_impl")),
            ("git", report.get("git"))]
    lines += [f"- {k}: {v}" for k, v in meta if v]
    for name, block in (report.get("sets") or {}).items():
        lines += ["", f"## {name} ({block.get('kind')}, {block.get('file')})", "",
                  f"{block.get('n_items')} items, {block.get('n_generations')} generations, "
                  f"{block.get('skipped')} skipped", "",
                  "| 指标 | 值 |", "|---|---|"]
        for key, value in (block.get("metrics") or {}).items():
            lines.append(f"| {key} | {_fmt(value)} |")
    checks = (report.get("gates") or {}).get("checks") or []
    if checks:
        lines += ["", "## 门限", "", "| 指标 | 值 | 约束 | 结果 |", "|---|---|---|---|"]
        for c in checks:
            limit = " ".join(x for x in (f"min {c['min']}" if c["min"] is not None else "",
                                         f"max {c['max']}" if c["max"] is not None else "") if x)
            verdict = "SKIP" if c.get("skipped") else ("PASS" if c["ok"] else "FAIL")
            lines.append(f"| {c['metric']} | {_fmt(c['value'])} | {limit} | {verdict} |")
        lines += ["", f"gate: {'PASS' if report['gates']['passed'] else 'FAIL'}"]
    cost = report.get("cost") or {}
    if cost:
        lines += ["", "## 端侧成本（无缓存解码，见 doc/evaluation.md）", "", "| 指标 | 值 |", "|---|---|"]
        lines += [f"| {k} | {_fmt(v)} |" for k, v in cost.items()]
    blind = report.get("blind") or {}
    if blind.get("n"):
        lines += ["", "## 人工盲测", "", "| 指标 | 值 |", "|---|---|"]
        lines += [f"| {k} | {_fmt(v)} |" for k, v in blind.items()]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------
# blind test
# ------------------------------------------------------------------------------------------------
def make_blind_pairs(items, model_replies, human_replies, seed=0):
    """Build the shuffled A/B view and the answer key.

    `items` are (id, messages) contexts; one arm holds the model reply, the other a human one.  Which arm
    is the model's is decided per pair from `seed` and recorded only in the key, so the labeling pass sees
    two unlabelled replies."""
    import random

    rng = random.Random(seed)
    pairs, key = [], {}
    for (id_, messages), model, human in zip(items, model_replies, human_replies):
        if rng.random() < 0.5:
            a, b, arm = model, human, "a"
        else:
            a, b, arm = human, model, "b"
        pairs.append(dict(id=id_, messages=messages, a=a, b=b))
        key[id_] = dict(model_arm=arm)
    return pairs, key


def score_blind_picks(picks, key, pairs):
    """Win/tie rates plus the parroting signal (near-identical arms: the model memorised the reply)."""
    by_id = {p["id"]: p for p in pairs}
    model_wins = human_wins = ties = skipped = parrots = 0
    for pick in picks:
        if pick.get("pick") == "skip":
            skipped += 1
            continue
        arm = key[pick["id"]]["model_arm"]
        if pick.get("pick") == "tie":
            ties += 1
        elif pick.get("pick") == arm:
            model_wins += 1
        else:
            human_wins += 1
        pair = by_id.get(pick["id"]) or {}
        model, human = normalize(pair.get(arm, "")), normalize(pair.get("b" if arm == "a" else "a", ""))
        if model and human and (model == human or (len(model) >= 8 and (model in human or human in model))):
            parrots += 1
    judged = model_wins + human_wins + ties
    return dict(n=judged, model_win_rate=round(model_wins / judged, 4) if judged else None,
                human_win_rate=round(human_wins / judged, 4) if judged else None,
                tie_rate=round(ties / judged, 4) if judged else None, skipped=skipped,
                parrot_rate=round(parrots / judged, 4) if judged else None,
                note="0.5 win rate means a human cannot tell the model's replies from human ones")
