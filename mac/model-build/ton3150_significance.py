#!/usr/bin/env python3
"""TON-3150: re-run TON-3144 significance with the API-mode Haiku re-judge.

Why: the original haiku-4-5 rows were generated via `claude -p` (decode:"cli-default",
EVAL_FROZEN_SPEC §2) — the only candidate that entered through a different door. 5/116
responses self-reveal the responder's identity/environment, breaking the blind exactly
on the model we claim parity with, and CLI-flavored refusals/questions plausibly
depressed the 2.80.

Leak count is 5, not 4: a bare /Claude Code/ regex finds only 4. The 5th (run_id
669e89ef…, TON-436) leaks as "Session is sandboxed to model-build directory only" —
no "Claude Code" string. The authoritative triaged list is ton3258_sensitivity.py
LEAKED; keep this docstring, TON3258_SENSITIVITY.md and TON-3150 all saying 5.
(A 6th case, TON-1366, regex-hits "Anthropic" but that is issue-content — a channel
list — not self-identification, and is correctly excluded. gen_sonnet.jsonl hits the
same word the same benign way, which is why "Anthropic" alone is not a leak marker.)

Fix: gen_haiku_api.jsonl (decode:"api-default", same frozen subset + prompt + parser)
was re-judged blind (same rubric, same Opus judge, seed 2397) into
judge_scores_ton3150.jsonl / judge_key_ton3150.jsonl. This script rebuilds the
case-level table with the ORIGINAL scores for every other model, haiku-4-5 replaced
by the API re-run, and reports old-vs-new haiku delta + teacher-vs-haiku paired tests.

Case-level policy (same as ton3144_significance.py): one value per (model, run_id) =
mean of all judged `overall` lines for that case. This is also the published-table fix
for the n=117>N=116 artifact (duplicate re-judged lines double-counted line-level).
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

def case_level(keys, score_lines, drop_models=(), rename=None):
    overall_by_bid = defaultdict(list)
    for s in score_lines:
        if s.get("overall") is not None:
            overall_by_bid[s["blind_id"]].append(float(s["overall"]))
    case_vals = defaultdict(lambda: defaultdict(list))
    for k in keys:
        m = k["model"]
        if m in drop_models:
            continue
        if rename:
            m = rename.get(m, m)
        vals = overall_by_bid.get(k["blind_id"])
        if not vals:
            continue
        case_vals[m][k["run_id"]].extend(vals)
    return {m: {r: statistics.mean(v) for r, v in d.items()} for m, d in case_vals.items()}

# original run, haiku (cli-mode) EXCLUDED
by_model = case_level(load(f"{BASE}/eval_v2_frontier_judge_key_all.jsonl"),
                      load(f"{BASE}/eval_v2_frontier_judge_scores_all.jsonl"),
                      drop_models=("haiku-4-5",))
# keep the old cli-mode haiku separately, for the delta report
old_haiku = case_level(load(f"{BASE}/eval_v2_frontier_judge_key_all.jsonl"),
                       load(f"{BASE}/eval_v2_frontier_judge_scores_all.jsonl")
                       ).get("haiku-4-5", {})
# API-mode haiku re-judge
new = case_level(load(f"{BASE}/judge_key_ton3150.jsonl"),
                 load(f"{BASE}/judge_scores_ton3150.jsonl"))
assert set(new) <= {"haiku-4-5"}, f"unexpected models in re-judge key: {set(new)}"
by_model["haiku-4-5"] = new["haiku-4-5"]

print("== Per-model overall (1-5), case-level, t-based 95% CI ==")
print("   (haiku-4-5 = API-mode re-run decode:api-default; others = original judged run)")
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
print("== Haiku old (cli-default) vs new (api-default), shared scored cases ==")
shared_h = sorted(set(old_haiku) & set(by_model["haiku-4-5"]))
dh = [by_model["haiku-4-5"][r] - old_haiku[r] for r in shared_h]
old_mean = statistics.mean(old_haiku.values())
new_mean = statistics.mean(by_model["haiku-4-5"].values())
print(f"old n={len(old_haiku)} mean={old_mean:.3f}  ->  new n={len(by_model['haiku-4-5'])} mean={new_mean:.3f}"
      f"  (paired shared={len(shared_h)}, mean paired delta={statistics.mean(dh):+.3f})" if shared_h else "no shared cases")
haiku_delta = dict(old_n=len(old_haiku), old_mean=old_mean,
                   new_n=len(by_model["haiku-4-5"]), new_mean=new_mean,
                   shared=len(shared_h),
                   paired_delta=statistics.mean(dh) if shared_h else None)

def paired(a_map, b_map, a_name, b_name):
    shared = sorted(set(a_map) & set(b_map))
    diffs = [a_map[r] - b_map[r] for r in shared]
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
    print(f"\n== Paired: {a_name} vs {b_name}, shared scored cases ==")
    print(f"shared cases: {n} ({a_name} scored {len(a_map)}, {b_name} scored {len(b_map)})")
    print(f"mean diff ({a_name} - {b_name}) = {mean_d:+.3f}  sd={sd_d:.3f}  se={se_d:.3f}")
    print(f"95% CI of diff = [{mean_d-ci:+.3f}, {mean_d+ci:+.3f}]")
    print(f"paired t = {tstat:.3f}, df={n-1}, two-sided p = {p_t:.3f}")
    print(f"case-level: wins={wins} losses={losses} ties={ties}; exact sign-test p = {p_sign:.3f}")
    verdict = "SIGNIFICANT" if p_t < 0.05 else "NOT significant"
    print(f"VERDICT: {a_name}-{b_name} difference is {verdict} at alpha=0.05")
    return dict(n=n, mean_diff=mean_d, sd=sd_d, se=se_d, ci95=[mean_d - ci, mean_d + ci],
                t=tstat, df=n - 1, p_paired_t=p_t, wins=wins, losses=losses, ties=ties,
                p_sign=p_sign, significant_at_05=p_t < 0.05)

paired_th = paired(by_model["teacher"], by_model["haiku-4-5"], "teacher", "haiku-4-5(api)")

out = dict(
    policy="case-level; haiku-4-5 replaced by API-mode re-run (TON-3150); others from original judged run",
    per_model=stats_out,
    haiku_old_vs_new=haiku_delta,
    paired_teacher_vs_haiku_api=paired_th,
)
with open(f"{BASE}/ton3150_significance.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nwrote {BASE}/ton3150_significance.json")
