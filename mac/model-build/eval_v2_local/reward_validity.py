#!/usr/bin/env python3
"""TON-2440 Phase 0 — reward-validity analysis.

Correlate the LOCAL 35B teacher judge (candidate GRPO reward) against the blinded
Opus judge (ground-truth quality metric) on the H1 eval set.

Pre-registered kill criterion: if the local-teacher reward does NOT correlate with
the Opus judge, reward option (a) is dead.

Usage:
  python3 reward_validity.py --opus judge_scores_h1.jsonl \
      --local judge_scores_h1_local.jsonl --key judge_key_h1.jsonl
"""
import argparse, json, math
from collections import defaultdict

def load(path):
    d = {}
    for l in open(path):
        if not l.strip():
            continue
        o = json.loads(l)
        d[o["blind_id"]] = o
    return d

def spearman(x, y):
    n = len(x)
    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j+1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j+1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    return pearson(rx, ry)

def pearson(x, y):
    n = len(x)
    mx, my = sum(x)/n, sum(y)/n
    num = sum((a-mx)*(b-my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a-mx)**2 for a in x))
    dy = math.sqrt(sum((b-my)**2 for b in y))
    return num/(dx*dy) if dx and dy else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opus", default="judge_scores_h1.jsonl")
    ap.add_argument("--local", default="judge_scores_h1_local.jsonl")
    ap.add_argument("--key", default="judge_key_h1.jsonl")
    ap.add_argument("--dim", default="overall")
    args = ap.parse_args()

    opus = load(args.opus)
    local = load(args.local)
    key = load(args.key)

    dim = args.dim
    pairs, by_model = [], defaultdict(list)
    n_local_err = sum(1 for v in local.values() if "overall" not in v)
    for bid, o in opus.items():
        l = local.get(bid)
        if not l or dim not in o or dim not in l:
            continue
        ov, lv = o[dim], l[dim]
        pairs.append((bid, ov, lv))
        m = key.get(bid, {}).get("model", "?")
        by_model[m].append((ov, lv))

    print(f"items: opus={len(opus)} local={len(local)} local_parse_err={n_local_err} paired={len(pairs)}")
    if len(pairs) < 5:
        print("NOT ENOUGH PAIRED ITEMS — local judge run incomplete.")
        return
    xo = [p[1] for p in pairs]
    xl = [p[2] for p in pairs]
    sp = spearman(xo, xl)
    pe = pearson(xo, xl)
    # exact-agreement & within-1
    exact = sum(1 for _, a, b in pairs if a == b) / len(pairs)
    within1 = sum(1 for _, a, b in pairs if abs(a-b) <= 1) / len(pairs)
    print(f"\n=== {dim}: local-teacher vs Opus (n={len(pairs)}) ===")
    print(f"  Spearman rho = {sp:.3f}")
    print(f"  Pearson  r   = {pe:.3f}")
    print(f"  exact agreement = {exact:.1%}   within-1 = {within1:.1%}")
    print(f"  mean Opus={sum(xo)/len(xo):.2f}  mean local={sum(xl)/len(xl):.2f}")

    # per-model mean ranking agreement (does local rank base/v0/teacher like Opus?)
    print("\n=== per-model mean (does local preserve the model ranking?) ===")
    print(f"  {'model':<10} {'n':>3} {'opus_mean':>10} {'local_mean':>11}")
    rows = []
    for m, vals in by_model.items():
        om = sum(v[0] for v in vals)/len(vals)
        lm = sum(v[1] for v in vals)/len(vals)
        rows.append((m, len(vals), om, lm))
    for m, n, om, lm in sorted(rows, key=lambda r: -r[2]):
        print(f"  {m:<10} {n:>3} {om:>10.2f} {lm:>11.2f}")
    opus_rank = [r[0] for r in sorted(rows, key=lambda r: -r[2])]
    local_rank = [r[0] for r in sorted(rows, key=lambda r: -r[3])]
    print(f"\n  Opus  model ranking: {opus_rank}")
    print(f"  Local model ranking: {local_rank}")
    print(f"  ranking match: {opus_rank == local_rank}")

    # verdict
    print("\n=== VERDICT (pre-registered) ===")
    if sp >= 0.5 and within1 >= 0.6:
        print(f"  PASS: local-teacher reward correlates with Opus (rho={sp:.2f}, within1={within1:.0%}).")
        print("  Reward option (a) viable -> GRPO P1 can use local-teacher-as-judge.")
    elif sp >= 0.3:
        print(f"  WEAK: partial correlation (rho={sp:.2f}). Risky as sole reward; consider blending rule-based (c).")
    else:
        print(f"  FAIL: local-teacher does NOT correlate with Opus (rho={sp:.2f}). KILL reward (a).")

if __name__ == "__main__":
    main()
