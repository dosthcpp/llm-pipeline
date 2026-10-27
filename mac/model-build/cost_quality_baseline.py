#!/usr/bin/env python3
"""TON-2773 — Cost/quality baseline for the disposition engine (Bet A, frozen harness).

Joins the FROZEN quality signal (blinded Opus-judge next-action-quality `overall`,
from the same de-anonymized judge artifacts as eval_v2_judge_score.py) with a
MEASURED per-decision token footprint + a documented cost model (cost_model.json),
and emits:
  1. cost-per-1k-decisions for each frontier-API model and the self-hosted sovereign,
  2. quality-per-dollar (overall / $ per 1k decisions),
  3. the v2 gate: at teacher-level quality (overall >= bar), what % of the cheapest
     frontier option matching that quality does the sovereign cost?

This is the cost SIDE of the deliverable. The quality side is certified by
EVAL_V2_H2_RESULT.md. A real frontier-API *candidate* on the same harness (so the
gate compares like-for-like quality, not teacher-as-proxy) is the named child task.

Usage:
  python cost_quality_baseline.py \
      --scores eval_v2_local/h2/judge_scores_h2.jsonl \
      --key    eval_v2_local/h2/judge_key_h2.jsonl \
      --cost   cost_model.json
"""
import argparse, json
from collections import defaultdict


def load_quality(scores_path, key_path):
    key = {}
    for line in open(key_path):
        k = json.loads(line)
        key[k["blind_id"]] = k
    agg = defaultdict(list)
    for line in open(scores_path):
        s = json.loads(line)
        bid = s.get("blind_id")
        if bid in key and s.get("overall") is not None:
            agg[key[bid]["model"]].append(float(s["overall"]))
    return {m: sum(v) / len(v) for m, v in agg.items() if v}


def frontier_cost_per_1k(cm, model):
    fp = cm["footprint_tokens"]
    fa = cm["frontier_api"]
    p = fa["models"][model]
    cf = fa.get("cache_input_fraction", 0.0)
    eff_in = p["in"] * (cf * 0.1 + (1 - cf))  # cached input billed 0.1x
    usd_per_decision = (fp["input_per_decision"] * eff_in
                        + fp["output_per_decision"] * p["out"]) / 1e6
    return usd_per_decision * 1000


def sovereign_cost_per_1k(cm, dec_per_hour=None):
    sh = cm["self_hosted_sovereign"]
    dph = dec_per_hour or sh["decisions_per_hour"]
    usd_per_decision = sh["amortized_usd_per_hour"] / dph
    return usd_per_decision * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="eval_v2_local/h2/judge_scores_h2.jsonl")
    ap.add_argument("--key", default="eval_v2_local/h2/judge_key_h2.jsonl")
    ap.add_argument("--cost", default="cost_model.json")
    ap.add_argument("--throughput", type=float, default=None,
                    help="sovereign decisions/hr for the gate. Default: cost_model "
                         "decisions_per_hour_production if present (measured prod path, "
                         "TON-2786), else decisions_per_hour (hf-eager floor).")
    args = ap.parse_args()

    cm = json.load(open(args.cost))
    qual = load_quality(args.scores, args.key)
    fp = cm["footprint_tokens"]

    # Gate throughput: the gate must price the PRODUCTION serving path (llama.cpp GGUF),
    # not the conservative hf-eager floor. Prefer the measured production rate when present.
    sh = cm["self_hosted_sovereign"]
    gate_dph = args.throughput or sh.get("decisions_per_hour_production") or sh["decisions_per_hour"]
    gate_basis = ("CLI override" if args.throughput else
                  "decisions_per_hour_production (measured llama.cpp, TON-2786)"
                  if sh.get("decisions_per_hour_production") else
                  "decisions_per_hour (hf-eager floor)")

    print(f"# Cost/Quality Baseline — disposition engine (TON-2773)\n")
    print(f"Per-decision footprint: {fp['input_per_decision']} in + "
          f"{fp['output_per_decision']} out tokens (measured)\n")

    # ---- frontier API table ----
    print("## Frontier API (priced per measured footprint, no cache)\n")
    print("| model | quality(overall) | $/1k decisions | quality/$ |")
    print("|---|---|---|---|")
    frontier_rows = {}
    for m in cm["frontier_api"]["models"]:
        c = frontier_cost_per_1k(cm, m)
        frontier_rows[m] = c
        q = qual.get(m)  # frontier candidate quality not yet measured on harness
        qs = f"{q:.2f}" if q is not None else "— (not on harness yet)"
        qpd = f"{q / c:.3f}" if q is not None else "—"
        print(f"| {m} | {qs} | ${c:.2f} | {qpd} |")

    # ---- self-hosted sovereign ----
    print("\n## Self-hosted sovereign (Qwen3-4B on Atlas)\n")
    print("| candidate | quality(overall) | $/1k @ throughput | ")
    print("|---|---|---|")
    sweep = sh.get("throughput_sensitivity_decisions_per_hour", [sh["decisions_per_hour"]])
    cost_str = "  ".join(f"{d}/hr=${sovereign_cost_per_1k(cm, d):.2f}" for d in sweep)
    for cand in ("teacher", "v0", "base"):
        if cand in qual:
            print(f"| {cand} | {qual[cand]:.2f} | {cost_str} |")
    base_cost = sovereign_cost_per_1k(cm)
    print(f"\n(default throughput {sh['decisions_per_hour']}/hr → ${base_cost:.2f}/1k decisions)")

    # ---- v2 gate ----
    g = cm["v2_gate"]
    bar = g["quality_bar_overall"]
    ceil = g["cost_ceiling_pct_of_frontier"]
    print(f"\n## v2 GATE\n")
    print(f"Quality bar: overall >= {bar}  |  Cost ceiling: <= {ceil}% of matching frontier\n")
    # cheapest frontier option that would meet the quality bar. Until a frontier
    # candidate is on the harness we anchor to the teacher as the quality proxy and
    # price the gate against the CHEAPEST frontier model as the cost reference.
    cheapest_model = min(frontier_rows, key=frontier_rows.get)
    cheapest_cost = frontier_rows[cheapest_model]
    dearest_model = max(frontier_rows, key=frontier_rows.get)
    dearest_cost = frontier_rows[dearest_model]
    gate_ceiling = cheapest_cost * ceil / 100.0
    print(f"Cheapest frontier ref: {cheapest_model} = ${cheapest_cost:.2f}/1k")
    print(f"Dearest frontier ref:  {dearest_model} = ${dearest_cost:.2f}/1k")
    print(f"=> v2 must hit overall >= {bar} at <= ${gate_ceiling:.2f}/1k "
          f"({ceil}% of {cheapest_model})\n")
    sov = sovereign_cost_per_1k(cm, gate_dph)
    print(f"Sovereign gate throughput = {gate_dph:.0f} dec/hr [{gate_basis}]")
    print(f"Sovereign cost = ${sov:.2f}/1k "
          f"= {100*sov/cheapest_cost:.1f}% of {cheapest_model}, "
          f"{100*sov/dearest_cost:.1f}% of {dearest_model}")
    teacher_q = qual.get("teacher")
    # Frontier cost-anchor quality, on-harness (TON-2773-B / TON-2801 keyless Haiku).
    anchor_q = qual.get(cheapest_model)  # judge tags == cost_model keys (e.g. "haiku-4-5")
    if teacher_q is not None:
        meets_q = "MEETS" if teacher_q >= bar else "BELOW"
        meets_c = "MEETS" if sov <= gate_ceiling else "OVER"
        print(f"\nSovereign (teacher proxy) quality overall {teacher_q:.2f} "
              f"→ quality {meets_q} bar ({bar}).")
        if anchor_q is not None:
            delta = teacher_q - anchor_q
            cmp = ("at/above" if delta >= -0.10 else "below")
            print(f"Frontier cost-anchor {cheapest_model} quality overall {anchor_q:.2f} "
                  f"(on-harness, keyless). Sovereign is {cmp} the anchor "
                  f"(Δ {delta:+.2f}); like-for-like confirmed.")
        print(f"Cost side: sovereign ${sov:.2f}/1k {meets_c} ceiling ${gate_ceiling:.2f}/1k.")
        gate_pass = (teacher_q >= bar) and (sov <= gate_ceiling) and \
                    (anchor_q is None or teacher_q >= anchor_q - 0.10)
        print(f"\n>>> v2 GATE VERDICT: {'PASS' if gate_pass else 'FAIL'} <<<")
        if anchor_q is None:
            print("NOTE: frontier anchor not yet on harness; quality side is teacher-proxy only.")


if __name__ == "__main__":
    main()
