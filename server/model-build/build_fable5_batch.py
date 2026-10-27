#!/usr/bin/env python3
"""TON-2586 — emit run_id-keyed fable5 scoring batch for items not yet scored.
Stable blind_id = run_id (UUID, no shuffle dependency) so scoring is incremental/resumable.
Aggregates later via key_fable5.jsonl (run_id -> fable5)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_v2_judge_prep import render_context
MB = os.path.expanduser("~/model-build")
gen = [json.loads(l) for l in open(f"{MB}/gen_qwen27b_fable5.jsonl") if l.strip()]
ctx = {}
for l in open(f"{MB}/dataset/eval.jsonl"):
    r = json.loads(l); ctx[r["run_id"]] = render_context(r.get("input"))
scored = set()
sf = f"{MB}/judge_scores_fable5.jsonl"
if os.path.exists(sf):
    for l in open(sf):
        try:
            scored.add(json.loads(l)["blind_id"])
        except Exception:
            pass
existing_keys = set()
kf_path = f"{MB}/key_fable5.jsonl"
if os.path.exists(kf_path):
    for l in open(kf_path):
        try:
            existing_keys.add(json.loads(l)["blind_id"])
        except Exception:
            pass
batch = []
new_keys = []
for g in gen:
    rid = g["run_id"]
    if rid in scored:
        continue
    batch.append({"blind_id": rid, "context": ctx.get(rid, "(missing)"),
                  "gold": g.get("gold"), "response": g.get("response", "")})
    if rid not in existing_keys:
        new_keys.append({"blind_id": rid, "model": "fable5", "run_id": rid,
                         "issue_key": g.get("issue_key"), "gold": g.get("gold")})
json.dump(batch, open(f"{MB}/batch_fable5_pending.json", "w"))
with open(kf_path, "a") as kf:
    for k in new_keys:
        kf.write(json.dumps(k) + "\n")
print(f"pending unscored fable5 items: {len(batch)} (total gen {len(gen)}, already scored {len(scored)})")
