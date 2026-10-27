#!/usr/bin/env python3
"""TON-3144: uncertainty quantification for the frozen N=116 benchmark.

Inputs (frozen eval v2 artifacts, this directory):
  eval_v2_frontier_judge_scores_all.jsonl  — blinded judge scores per blind_id
                                             (has duplicate blind_id lines from
                                             re-judged chunks; 36 lines lack
                                             `overall`)
  eval_v2_frontier_judge_key_all.jsonl     — blind_id -> (model, run_id/case, gold)

Aggregation policy (case-level): one value per (model, run_id) = mean of all
`overall` scores judged for that blind_id. This de-duplicates re-judged lines
(the legacy eval_v2_judge_score.py counted duplicate lines twice — that is why
the published table shows sonnet n=117 / opus n=115).

Outputs:
  per-model mean overall, SD, SE, t-based 95% CI
  paired teacher(sovereign)-vs-haiku analysis on shared scored cases:
    mean diff, 95% CI, paired t-test p, exact sign-test p
  legacy line-level means (to reconcile with the published 2.89 / 2.80)
  ton3144_significance.json

Pure stdlib: Student-t p via exact incomplete beta, t quantile via bisection.
"""
import json, math, statistics
from collections import defaultdict

BASE = "/Volumes/Data/tony_ai_lab/model-build"

def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]

def betacf(a, b, x):
    MAXIT, EPS, FPMIN = 200, 3e-9, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < FPMIN: d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN: d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN: c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN: d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN: c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS: break
    return h

def betai(a, b, x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * betacf(a, b, x) / a
    return 1.0 - bt * betacf(b, a, 1.0 - x) / b

def t_sf2(t, df):
    x = df / (df + t * t)
    return betai(df / 2.0, 0.5, x)

def t_ppf975(df):
    lo, hi = 1.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_sf2(mid, df) > 0.05:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

def binom_two_sided(k, n):
    def pmf(i):
        return math.comb(n, i) * 0.5 ** n
    pk = pmf(k)
    return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= pk + 1e-12))

score_lines = load(f"{BASE}/eval_v2_frontier_judge_scores_all.jsonl")
keys = load(f"{BASE}/eval_v2_frontier_judge_key_all.jsonl")
key_by_bid = {k["blind_id"]: k for k in keys}

# blind_id -> list of overall values (duplicates = re-judged lines)
overall_by_bid = defaultdict(list)
for s in score_lines:
    if s.get("overall") is not None:
        overall_by_bid[s["blind_id"]].append(float(s["overall"]))

# case-level: model -> {run_id: mean overall}   (baseline key rows repeat
# run_ids — 58 unique cases listed twice — so collect per run_id then mean)
case_vals = defaultdict(lambda: defaultdict(list))
unscored = defaultdict(int)
for k in keys:
    vals = overall_by_bid.get(k["blind_id"])
    if not vals:
        unscored[k["model"]] += 1
        continue
    case_vals[k["model"]][k["run_id"]].extend(vals)
by_model = {m: {r: statistics.mean(v) for r, v in d.items()} for m, d in case_vals.items()}

# legacy line-level means (published-table reconciliation)
legacy = defaultdict(list)
for s in score_lines:
    k = key_by_bid.get(s["blind_id"])
    if k and s.get("overall") is not None:
        legacy[k["model"]].append(float(s["overall"]))

print("== Reconciliation: legacy line-level mean (published table) vs case-level ==")
for m in by_model:
    lv = legacy[m]
    print(f"{m:12s} legacy n={len(lv):3d} mean={statistics.mean(lv):.3f} | "
          f"case-level n={len(by_model[m]):3d} mean={statistics.mean(by_model[m].values()):.3f} "
          f"(unscored key rows: {unscored.get(m,0)})")

print()
print("== Per-model overall (1-5), case-level, t-based 95% CI ==")
stats_out = {}
for m in sorted(by_model, key=lambda m: -statistics.mean(by_model[m].values())):
    vals = list(by_model[m].values())
    n = len(vals)
    mean = statistics.mean(vals)
    sd = statistics.stdev(vals)
    se = sd / math.sqrt(n)
    ci = t_ppf975(n - 1) * se
    stats_out[m] = dict(n=n, mean=mean, sd=sd, se=se, ci95_half=ci,
                        ci95=[mean - ci, mean + ci])
    print(f"{m:12s} n={n:3d} mean={mean:.3f} sd={sd:.3f} se={se:.3f} "
          f"95%CI ±{ci:.3f} -> [{mean-ci:.2f}, {mean+ci:.2f}]")

print()
print("== Paired: teacher (sovereign) vs haiku-4-5, shared scored cases ==")
t_map, h_map = by_model["teacher"], by_model["haiku-4-5"]
shared = sorted(set(t_map) & set(h_map))
diffs = [t_map[r] - h_map[r] for r in shared]
n = len(diffs)
mean_d = statistics.mean(diffs)
sd_d = statistics.stdev(diffs)
se_d = sd_d / math.sqrt(n)
ci = t_ppf975(n - 1) * se_d
tstat = mean_d / se_d
p_t = t_sf2(abs(tstat), n - 1)
wins = sum(1 for d in diffs if d > 0)
losses = sum(1 for d in diffs if d < 0)
ties = n - wins - losses
p_sign = binom_two_sided(min(wins, losses), wins + losses) if wins + losses else 1.0
print(f"shared cases: {n} (teacher scored {len(t_map)}, haiku scored {len(h_map)})")
print(f"mean diff (teacher - haiku) = {mean_d:+.3f}  sd={sd_d:.3f}  se={se_d:.3f}")
print(f"95% CI of diff = [{mean_d-ci:+.3f}, {mean_d+ci:+.3f}]")
print(f"paired t = {tstat:.3f}, df={n-1}, two-sided p = {p_t:.3f}")
print(f"case-level: wins={wins} losses={losses} ties={ties}; exact sign-test p = {p_sign:.3f}")
verdict = "SIGNIFICANT" if p_t < 0.05 else "NOT significant"
print(f"VERDICT: teacher-haiku difference is {verdict} at alpha=0.05")

out = dict(
    policy="case-level: mean of duplicate judged lines per (model, run_id); cases with no overall dropped",
    per_model=stats_out,
    paired=dict(n=n, teacher_scored=len(t_map), haiku_scored=len(h_map),
                mean_diff=mean_d, sd=sd_d, se=se_d,
                ci95=[mean_d - ci, mean_d + ci], t=tstat, df=n - 1,
                p_paired_t=p_t, wins=wins, losses=losses, ties=ties,
                p_sign=p_sign, significant_at_05=p_t < 0.05),
    legacy_line_level={m: dict(n=len(v), mean=statistics.mean(v)) for m, v in legacy.items()},
)
with open(f"{BASE}/ton3144_significance.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nwrote {BASE}/ton3144_significance.json")
