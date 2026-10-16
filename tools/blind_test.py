#!/usr/bin/env python
"""Human blind test over the pairs scripts/eval.py --blind writes.

    python tools/eval_table.py ...                      # (comparison view; see --help)
    python tools/blind_test.py --dir runs/eval/sft      # label pairs: for each one pick a / b / tie / skip
    python tools/blind_test.py --score runs/eval/sft    # fold the win rate back into report.json

The two replies of a pair come from the same context; one is the model's sample and the other a real human
reply.  Nothing on screen says which is which (the mapping lives in blind.key.json), and picks are appended
to picks.jsonl after every answer, so quitting midway keeps the work done so far.  The win rate is the
release-relevant number: a rate near 0.5 means a person cannot tell the two apart.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hvlm import eval_core as ec  # noqa: E402

PICKS = "picks.jsonl"


def read_json(path):
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}: run `python scripts/eval.py ... --blind N` first")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_picks(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()] if os.path.exists(path) else []


def label(pairs, picks_path):
    done = {p["id"] for p in read_picks(picks_path)}
    todo = [p for p in pairs if p["id"] not in done]
    if not todo:
        print(f"all {len(pairs)} pairs are labelled; score with --score")
        return 0
    print(f"{len(todo)} of {len(pairs)} pairs left.  a = A is better, b = B is better, "
          f"t = tie, s = skip, q = quit\n")
    with open(picks_path, "a", encoding="utf-8") as out:
        for pair in todo:
            print("--- " + pair["id"])
            for m in pair["messages"]:
                print(f"[{m['role']}] {m['content']}")
            print(f"A: {pair['a']}")
            print(f"B: {pair['b']}")
            while True:
                answer = input("pick (a/b/t/s/q): ").strip().lower()
                if answer in ("a", "b", "t", "s", "q"):
                    break
            if answer == "q":
                print("stopped; run this command again to continue")
                return 0
            pick = {"a": "a", "b": "b", "t": "tie", "s": "skip"}[answer]
            out.write(json.dumps({"id": pair["id"], "pick": pick}, ensure_ascii=False) + "\n")
            out.flush()
    print(f"done: {len(pairs)} pairs labelled -> {picks_path}")
    print("score with --score")
    return 0


def score(run_dir):
    pairs = read_json(os.path.join(run_dir, "blind.json"))
    key = read_json(os.path.join(run_dir, "blind.key.json"))
    picks = read_picks(os.path.join(run_dir, PICKS))
    if not picks:
        raise SystemExit(f"no picks in {os.path.join(run_dir, PICKS)}: label the pairs first")
    result = ec.score_blind_picks(picks, key, pairs)
    report_path = os.path.join(run_dir, "report.json")
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    report["blind"] = dict(result, n_pairs=len(pairs), labelled=len(picks),
                           pairs=os.path.join(run_dir, "blind.json"), key=os.path.join(run_dir, "blind.key.json"))
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(ec.render_markdown(report))
    print(f"blind: {result['n']} judged, model win rate {result['model_win_rate']}, "
          f"human {result['human_win_rate']}, tie {result['tie_rate']}, parrot {result['parrot_rate']}")
    print(f"written to {report_path}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default=None, help="eval run directory (same as --dir)")
    ap.add_argument("--dir", default=None, help="eval run directory holding blind.json / blind.key.json")
    ap.add_argument("--score", action="store_true", help="score picks.jsonl into report.json instead of labelling")
    ap.add_argument("--pairs", default=None, help="blind.json (defaults to --dir/blind.json)")
    ap.add_argument("--key", default=None, help="blind.key.json (defaults to --dir/blind.key.json)")
    ap.add_argument("--picks", default=None, help="picks.jsonl (defaults to --dir/picks.jsonl)")
    a = ap.parse_args()
    run_dir = a.run_dir or a.dir or (os.path.dirname(a.pairs) if a.pairs else None)
    if not run_dir:
        raise SystemExit("an eval run directory is required (positional, or --dir / --pairs with --key)")
    if a.score:
        return score(run_dir)
    if not sys.stdin.isatty():
        raise SystemExit("labelling needs a terminal; use --score to fold existing picks into the report")
    pairs_path = a.pairs or os.path.join(run_dir, "blind.json")
    a.pairs, a.key = pairs_path, a.key or os.path.join(run_dir, "blind.key.json")
    read_json(a.key)  # fail early if the key is missing
    return label(read_json(pairs_path), a.picks or os.path.join(run_dir, PICKS))


if __name__ == "__main__":
    sys.exit(main())
