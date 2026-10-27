#!/usr/bin/env python3
"""TON-3258: worst-case sensitivity of the parity claim to the 5 CLI blind-leak cases.

The frozen N=116 eval generated haiku-4-5 via `claude -p` (decode:"cli-default").
5/116 haiku responses self-reveal the responder's identity/environment (leak list
below, re-derived from gen_haiku.jsonl by regex + manual triage; matches the two
verbatim quotes in TON-3150's description). Question: if all 5 had been scored a
perfect 5.0 — i.e. the contamination was maximally against us — does the published
"teacher(2.89) ~ haiku(2.80), p=0.79 parity" conclusion still hold?

Pure recomputation on eval_v2_frontier_judge_scores_all.jsonl — zero API calls.
Case-level policy identical to ton3144_significance.py (mean of duplicate judged
lines per (model, run_id); unscored cases dropped, except scenario (b) where an
unscored leaked case would enter at 5.0).
"""
import json, math, statistics
from collections import defaultdict

BASE = "/Volumes/Data/tony_ai_lab/model-build"

# The 5 leaked haiku cases (run_id, issue_key). TON-1366 also regex-hits "Anthropic"
# but that is issue-content (a channel list), not self-identification — excluded.
LEAKED = [
    ("20c291c1-2cd3-43e7-bfa3-da12ec26b24b", "TON-1659"),  # "I'm Claude Code running in a sandbox environment"
    ("669e89ef-0abf-4e66-8c51-07b235c4d72d", "TON-436"),   # "Session is sandboxed to model-build directory only"
    ("6cf08ac9-43e4-4040-a8ab-5744a1fd4b9a", "TON-2183"),  # "a separate Claude Code adapter project?"
    ("e719b8f1-38a9-420b-9faf-6217d3766817", "TON-1954"),  # "Requires active Claude Code session"
    ("f2c600ea-96a7-4b1a-a3b4-72e9191b7f97", "TON-2156"),  # "a different Claude Code session"
]
LEAKED_RUNS = {r for r, _ in LEAKED}

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

overall_by_bid = defaultdict(list)
lines_by_bid = defaultdict(list)
for s in score_lines:
    lines_by_bid[s["blind_id"]].append(s)
    if s.get("overall") is not None:
        overall_by_bid[s["blind_id"]].append(float(s["overall"]))

case_vals = defaultdict(lambda: defaultdict(list))
for k in keys:
    vals = overall_by_bid.get(k["blind_id"])
    if vals:
        case_vals[k["model"]][k["run_id"]].extend(vals)
by_model = {m: {r: statistics.mean(v) for r, v in d.items()} for m, d in case_vals.items()}

print("== 1. The 5 leaked haiku cases: blind_id + current judged scores ==")
haiku_bids = {k["run_id"]: k["blind_id"] for k in keys if k["model"] == "haiku-4-5"}
leak_report = []
for run_id, ik in LEAKED:
    bid = haiku_bids.get(run_id)
    lines = lines_by_bid.get(bid, [])
    case = by_model["haiku-4-5"].get(run_id)
    axes = [{a: l.get(a) for a in ("disposition_correct", "grounded", "actionable", "calibrated", "overall")}
            for l in lines]
    leak_report.append(dict(issue_key=ik, run_id=run_id, blind_id=bid,
                            case_overall=case, judged_lines=axes))
    print(f"{ik:9s} blind_id={bid} case_overall={case} lines={axes}")

scored_leaked = [r for r, _ in LEAKED if r in by_model["haiku-4-5"]]
print(f"\nleaked scored: {len(scored_leaked)}/5 "
      f"(haiku scored cases total: {len(by_model['haiku-4-5'])})")

def scenario(name, h_map):
    t_map = by_model["teacher"]
    vals = list(h_map.values())
    n_h = len(vals)
    mean_h = statistics.mean(vals)
    sd_h = statistics.stdev(vals)
    ci_h = t_ppf975(n_h - 1) * sd_h / math.sqrt(n_h)
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
    sig = p_t < 0.05
    direction = "teacher ahead" if mean_d > 0 else "haiku ahead"
    # unpaired Welch over ALL scored cases — covers the 3 leaked cases that the
    # paired subset structurally cannot see (teacher unscored on them)
    tv = list(t_map.values())
    n_t, mean_t = len(tv), statistics.mean(tv)
    var_t, var_h = statistics.variance(tv), statistics.variance(vals)
    se_w = math.sqrt(var_t / n_t + var_h / n_h)
    dw = mean_t - mean_h
    df_w = (var_t / n_t + var_h / n_h) ** 2 / (
        (var_t / n_t) ** 2 / (n_t - 1) + (var_h / n_h) ** 2 / (n_h - 1))
    t_w = dw / se_w
    p_w = t_sf2(abs(t_w), df_w)
    print(f"\n== {name} ==")
    print(f"haiku: n={n_h} mean={mean_h:.3f} 95%CI [{mean_h-ci_h:.2f}, {mean_h+ci_h:.2f}]")
    print(f"paired (n={n}): diff(teacher-haiku)={mean_d:+.3f} "
          f"95%CI [{mean_d-ci:+.3f}, {mean_d+ci:+.3f}] t={tstat:.3f} p={p_t:.3f}")
    print(f"wins/losses/ties = {wins}/{losses}/{ties}, sign-test p={p_sign:.3f}")
    print(f"Welch (all scored, {n_t} vs {n_h}): diff={dw:+.3f} t={t_w:.3f} df={df_w:.1f} p={p_w:.3f}")
    sig_any = sig or p_w < 0.05
    print(f"-> {'SIGNIFICANT (' + direction + ')' if sig_any else 'NOT significant -> parity holds'}")
    return dict(name=name, haiku_n=n_h, haiku_mean=mean_h,
                haiku_ci95=[mean_h - ci_h, mean_h + ci_h],
                paired_n=n, mean_diff=mean_d, diff_ci95=[mean_d - ci, mean_d + ci],
                t=tstat, p_paired_t=p_t, wins=wins, losses=losses, ties=ties,
                p_sign=p_sign,
                welch=dict(diff=dw, t=t_w, df=df_w, p=p_w),
                significant_at_05=sig_any, direction=direction)

h_cur = dict(by_model["haiku-4-5"])

# (a) as published
res_a = scenario("(a) current (as published)", h_cur)

# (b) worst case against us: every leaked case rescored to a perfect 5.0
#     (unscored leaked cases enter at 5.0 too — maximally adversarial)
h_max = dict(h_cur)
for r in LEAKED_RUNS:
    h_max[r] = 5.0
res_b = scenario("(b) leaked 5 -> 5.0 (worst case)", h_max)

# (c) leaked cases dropped entirely
h_drop = {r: v for r, v in h_cur.items() if r not in LEAKED_RUNS}
res_c = scenario("(c) leaked 5 dropped", h_drop)

holds = not (res_b["significant_at_05"] and res_b["mean_diff"] < 0)
print("\n== VERDICT ==")
print("parity conclusion HOLDS in the worst case" if holds
      else "parity conclusion DOES NOT HOLD in the worst case -> ESCALATE TO CEO")

out = dict(
    policy="case-level, identical to ton3144_significance.py; scenario (b) forces leaked cases to 5.0 (unscored leaked cases enter at 5.0)",
    leaked_cases=leak_report,
    scenarios=dict(current=res_a, leaked_to_5=res_b, leaked_dropped=res_c),
    parity_holds_worst_case=holds,
)
with open(f"{BASE}/ton3258_sensitivity.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nwrote {BASE}/ton3258_sensitivity.json")
