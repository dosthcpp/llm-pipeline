#!/usr/bin/env python3
"""TON-2837 — smoke test: Claude Sonnet as RAFT reward judge vs local 27B teacher.
Reads ANTHROPIC_API_KEY from ~/.hermes/.env (run_2806_api.py pattern).
Same rubric + user format as local_teacher_judge.judge_one / run_opus_judge_8b.py.
Sonnet (not Opus) on purpose: the frozen eval judge is claude-opus-4-8 — keeping
the reward judge in a different tier limits training directly against the gate judge.
"""
import concurrent.futures, json, os, statistics, sys, time
from pathlib import Path

for line in open(Path.home() / ".hermes" / ".env"):
    if line.startswith("ANTHROPIC_API_KEY="):
        os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit("no ANTHROPIC_API_KEY in ~/.hermes/.env")

import anthropic

MODEL = "claude-sonnet-5"
SYSTEM = open("/tmp/JUDGE_RUBRIC.md").read().strip()
client = anthropic.Anthropic()

def judge_one(item):
    prompt = (f"Context:\n{item['context']}\n\n"
              f"Gold disposition: {item['gold']}\n\n"
              f"Agent response:\n{item['response']}")
    t0 = time.time()
    for attempt in range(4):
        try:
            msg = client.messages.create(
                model=MODEL, max_tokens=256, system=SYSTEM,
                thinking={"type": "disabled"},
                messages=[{"role": "user", "content": prompt}],
            )
            raw = next(b.text for b in msg.content if b.type == "text").strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            scores = json.loads(raw)
            scores["_latency_s"] = round(time.time() - t0, 1)
            scores["_in_tok"] = msg.usage.input_tokens
            scores["_out_tok"] = msg.usage.output_tokens
            return {"blind_id": item["blind_id"], **scores}
        except (anthropic.RateLimitError, anthropic.APIStatusError) as e:
            time.sleep(2 ** attempt * 5)
            last = str(e)
        except json.JSONDecodeError as e:
            return {"blind_id": item["blind_id"], "error": f"json:{e}", "raw": raw[:120]}
    return {"blind_id": item["blind_id"], "error": f"max_retries:{last}"}

items = [json.loads(l) for l in open("/tmp/claude_reward_smoke_sample.jsonl") if l.strip()]
t_start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    results = list(pool.map(judge_one, items))
wall = time.time() - t_start

by_id = {r["blind_id"]: r for r in results}
with open("/tmp/claude_reward_smoke_scores.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---- analysis ----
def spearman(x, y):
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and v[s[j + 1]] == v[s[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                rk[s[k]] = avg
            i = j + 1
        return rk
    rx, ry = rank(x), rank(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")

paired = [(it, by_id[it["blind_id"]]) for it in items
          if "overall" in by_id.get(it["blind_id"], {})]
n_err = len(items) - len(paired)
t27 = [it["teacher27b"]["overall"] for it, _ in paired]
tcl = [r["overall"] for _, r in paired]
print(f"\n=== smoke result: n={len(items)} scored={len(paired)} err={n_err} wall={wall:.0f}s ===")
print(f"spearman(27B, sonnet) overall: {spearman(t27, tcl):.3f}")
lat = [r["_latency_s"] for _, r in paired]
itok = sum(r["_in_tok"] for _, r in paired)
otok = sum(r["_out_tok"] for _, r in paired)
print(f"latency p50={statistics.median(lat):.1f}s mean={statistics.mean(lat):.1f}s "
      f"(27B iter-0: ~57s serial)")
print(f"tokens: in={itok} out={otok} -> per-item in={itok//len(paired)} out={otok//len(paired)}")
# Sonnet 5 intro pricing through 2026-08-31: $2/M in, $10/M out (list $3/$15)
c_inr, c_out = itok / 1e6 * 2, otok / 1e6 * 10
per_item = (c_inr + c_out) / len(paired)
print(f"cost this run: ${c_inr + c_out:.3f} -> extrapolated NP400xK4=1600 items: "
      f"${per_item * 1600:.2f} (intro pricing)")
print("\nstratum detail (27B vs sonnet overall, resp_len):")
for it, r in paired:
    tag = ("HI_LONG" if it["teacher27b"]["overall"] >= 3 and it["resp_len"] > 4000
           else "HI_SHORT" if it["teacher27b"]["overall"] >= 3 else "LO")
    d27, dcl = it["teacher27b"], r
    print(f"  {tag:8s} len={it['resp_len']:5d} 27B ov={d27['overall']} gr={d27.get('grounded')} "
          f"| sonnet ov={dcl['overall']} gr={dcl.get('grounded')} cal={dcl.get('calibrated')}")
