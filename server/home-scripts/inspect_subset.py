import json, sys, os
sys.path.insert(0, os.path.expanduser("~/model-build"))
from eval_v2_gen import build_eval_subset, make_prompt, gold_disp
rows = build_eval_subset(os.path.expanduser("~/model-build/dataset/eval.jsonl"), 30, 30)[:40]
sizes = []
for i, r in enumerate(rows):
    p = make_prompt(r)
    sizes.append((i, len(p)))
for i in range(15, 30):
    r = rows[i]
    print("idx", i, "run_id", r.get("run_id"), "gold", gold_disp(r), "prompt_chars", len(make_prompt(r)))
print("--- top 5 largest prompts ---")
for i, n in sorted(sizes, key=lambda x: -x[1])[:5]:
    print("idx", i, "chars", n)
