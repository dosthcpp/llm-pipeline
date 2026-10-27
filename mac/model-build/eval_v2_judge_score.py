#!/usr/bin/env python3
"""TON-2397 P0.3.1 — De-anonymize judge scores and build the per-model table.

Inputs:
  --scores judge_scores.jsonl : {blind_id, disposition_correct, grounded, actionable,
                                  calibrated, overall, note}  (from blinded judge)
  --key    judge_key.jsonl    : {blind_id, model, run_id, issue_key, gold}
Output: per-model mean of each rubric dimension + overall, with n. Prints a markdown table.
"""
import json, argparse
from collections import defaultdict

DIMS = ["disposition_correct", "grounded", "actionable", "calibrated", "overall"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="judge_scores.jsonl")
    ap.add_argument("--key", default="judge_key.jsonl")
    args = ap.parse_args()

    key = {}
    for line in open(args.key):
        k = json.loads(line)
        key[k["blind_id"]] = k

    agg = defaultdict(lambda: {d: [] for d in DIMS})
    n_missing = 0
    for line in open(args.scores):
        s = json.loads(line)
        bid = s["blind_id"]
        if bid not in key:
            n_missing += 1
            continue
        model = key[bid]["model"]
        for d in DIMS:
            if d in s and s[d] is not None:
                agg[model][d].append(float(s[d]))

    order = ["teacher", "base", "v0", "v1", "baseline"]
    models = [m for m in order if m in agg] + [m for m in agg if m not in order]

    print("| model | n | disposition | grounded | actionable | calibrated | **overall** |")
    print("|---|---|---|---|---|---|---|")
    rows = {}
    for m in models:
        a = agg[m]
        n = max(len(a[d]) for d in DIMS)
        means = {d: (sum(a[d]) / len(a[d]) if a[d] else float("nan")) for d in DIMS}
        rows[m] = means
        print(f"| {m} | {n} | {means['disposition_correct']:.2f} | {means['grounded']:.2f} "
              f"| {means['actionable']:.2f} | {means['calibrated']:.2f} | **{means['overall']:.2f}** |")

    if "teacher" in rows and "baseline" in rows:
        gap = rows["teacher"]["overall"] - rows["baseline"]["overall"]
        print(f"\nDISCRIMINATION (teacher overall - baseline overall): {gap:+.2f}")
        print("PASS bar: teacher clearly > baseline." if gap >= 1.0 else "WEAK: gap < 1.0 — judge not discriminating.")
    if n_missing:
        print(f"\n[warn] {n_missing} scores had no matching blind_id")


if __name__ == "__main__":
    main()
