#!/usr/bin/env python3
"""TON-2813 — aggregate blinded 8B judge scores, place v2-8B on the frozen scale.

Merges judge_8b_chunks/scores_*.jsonl, joins judge_8b_key.jsonl, prints per-model
means + discrimination, and validates the teacher calibration anchor reproduces its
frontier-batch value (~2.89). If teacher drifts, an anchor-corrected 8B is also shown.
"""
import json, glob, argparse
from collections import defaultdict

DIMS = ["disposition_correct", "grounded", "actionable", "calibrated", "overall"]
TEACHER_FRONTIER = 2.89  # from EVAL_V2_FRONTIER_RESULT.md (same rubric, prior batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="judge_8b_key.jsonl")
    ap.add_argument("--scores-glob", default="judge_8b_chunks/scores_*.jsonl")
    args = ap.parse_args()

    key = {}
    for line in open(args.key):
        k = json.loads(line)
        key[k["blind_id"]] = k

    scores = {}
    for path in sorted(glob.glob(args.scores_glob)):
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                s = json.loads(line)
            except Exception:
                continue
            if "blind_id" in s:
                scores[s["blind_id"]] = s  # last wins; blind_ids unique

    agg = defaultdict(lambda: {d: [] for d in DIMS})
    matched = 0
    for bid, s in scores.items():
        if bid not in key:
            continue
        matched += 1
        m = key[bid]["model"]
        for d in DIMS:
            if s.get(d) is not None:
                try:
                    agg[m][d].append(float(s[d]))
                except (TypeError, ValueError):
                    pass

    total_expected = len(key)
    print(f"scored {len(scores)} / {total_expected} blind_ids | matched-to-key {matched}")
    print()
    print("| model | n | disposition | grounded | actionable | calibrated | **overall** |")
    print("|---|---|---|---|---|---|---|")
    means = {}
    for m in ["v2_8b", "teacher", "baseline"]:
        if m not in agg:
            continue
        a = agg[m]
        n = max(len(a[d]) for d in DIMS)
        mm = {d: (sum(a[d]) / len(a[d]) if a[d] else float("nan")) for d in DIMS}
        means[m] = mm
        print(f"| {m} | {n} | {mm['disposition_correct']:.2f} | {mm['grounded']:.2f} "
              f"| {mm['actionable']:.2f} | {mm['calibrated']:.2f} | **{mm['overall']:.2f}** |")

    print()
    if "teacher" in means and "baseline" in means:
        disc = means["teacher"]["overall"] - means["baseline"]["overall"]
        print(f"DISCRIMINATION teacher-baseline = {disc:+.2f} "
              f"({'VALID >=1.0' if disc >= 1.0 else 'WEAK <1.0'})")
    if "teacher" in means:
        t = means["teacher"]["overall"]
        drift = t - TEACHER_FRONTIER
        print(f"TEACHER ANCHOR this batch = {t:.2f} vs frontier-batch {TEACHER_FRONTIER:.2f} "
              f"(drift {drift:+.2f})")
        if "v2_8b" in means:
            raw = means["v2_8b"]["overall"]
            corrected = raw - drift  # shift 8B by the anchor drift
            print(f"v2_8b RAW overall = {raw:.2f}")
            print(f"v2_8b ANCHOR-CORRECTED (to frontier scale) = {corrected:.2f}")


if __name__ == "__main__":
    main()
