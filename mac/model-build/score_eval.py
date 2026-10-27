#!/usr/bin/env python3
"""Compute disposition match rates for teacher / base / v0.  TON-2395.
Preds files are JSON arrays with records: {run_id, issue_key, gold, pred, raw_tail}
"""
import json
from collections import defaultdict

PREDS = {
    "teacher": "/home/tony/model-build/preds_teacher.jsonl",
    "base":    "/home/tony/model-build/preds_base.jsonl",
    "v0":      "/home/tony/model-build/preds_v0.jsonl",
}
OUT = "/home/tony/model-build/eval_results.json"


def load(path):
    raw = open(path).read().strip()
    if raw.startswith("["):
        return json.loads(raw)
    return [json.loads(l) for l in raw.splitlines() if l.strip()]


def score(preds):
    hits = wrong = no_pred = 0
    for p in preds:
        gold = p.get("gold", "unknown")
        pred = p.get("pred", "unknown")
        if not pred or pred == "unknown":
            no_pred += 1
        elif pred == gold:
            hits += 1
        else:
            wrong += 1
    n = len(preds)
    return dict(n=n, hits=hits, wrong=wrong, no_pred=no_pred,
                match_pct=round(100 * hits / n, 1),
                coverage=round(100 * (hits + wrong) / n, 1))


def per_class(preds):
    cls = defaultdict(lambda: {"hit": 0, "total": 0})
    for p in preds:
        g = p.get("gold", "unknown")
        pr = p.get("pred", "unknown")
        cls[g]["total"] += 1
        if pr == g:
            cls[g]["hit"] += 1
    return {k: f"{v['hit']}/{v['total']}" for k, v in sorted(cls.items())}


results = {}
for name, path in PREDS.items():
    preds = load(path)
    s = score(preds)
    results[name] = s
    print(f"{name.upper():8}: match={s['match_pct']}%  coverage={s['coverage']}%  "
          f"hits={s['hits']}/{s['n']}  no_pred={s['no_pred']}")

v0_preds = load(PREDS["v0"])
pc = per_class(v0_preds)
print("V0 per-class:", pc)
results["v0_per_class"] = pc

json.dump(results, open(OUT, "w"), indent=2)
print(f"Written {OUT}")
