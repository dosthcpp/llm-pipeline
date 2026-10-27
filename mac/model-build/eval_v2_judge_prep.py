#!/usr/bin/env python3
"""TON-2397 P0.3.1 — Blinded judge batch builder.

Joins native-format generations (gen_<model>.jsonl: run_id,issue_key,gold,pred,response)
to their wake/issue INPUT context (dataset/eval.jsonl, by run_id), injects a degenerate
baseline responder as a floor anchor, anonymizes + shuffles, and emits:
  - judge_batch.jsonl : {blind_id, context, gold, response}   <- what the judge sees
  - judge_key.jsonl   : {blind_id, model, run_id, issue_key}  <- de-anon key (judge never sees)

The judge (blinded Opus, via CTO Agent subagents) scores each blind_id on JUDGE_RUBRIC.md.
Deterministic shuffle (fixed seed) so reruns are stable and reviewable.
"""
import json, sys, argparse, random, hashlib

DEGENERATE = "I'll continue working on this issue. Disposition: in_progress"


def render_context(input_obj, max_chars=1600):
    """Compact, judge-readable rendering of the wake context the agent saw."""
    if isinstance(input_obj, str):
        try:
            input_obj = json.loads(input_obj)
        except Exception:
            return input_obj[:max_chars]
    o = input_obj
    parts = []
    wr = o.get("wake_reason") or o.get("wake_source")
    if wr:
        parts.append(f"WAKE: {wr}")
    iss = o.get("issue") or {}
    if iss:
        parts.append(f"ISSUE {iss.get('key','?')}: {iss.get('title','')}")
        st = iss.get("status") or iss.get("state")
        if st:
            parts.append(f"STATUS: {st}")
        desc = (iss.get("description") or "").strip()
        if desc:
            parts.append("DESCRIPTION:\n" + desc)
    # latest comment / pending comments if present
    for key in ("latest_comment", "latestComment", "pending_comments", "comments"):
        v = o.get(key)
        if v:
            s = v if isinstance(v, str) else json.dumps(v)[:600]
            parts.append(f"{key.upper()}: {s}")
            break
    txt = "\n".join(parts).strip()
    if len(txt) > max_chars:
        txt = txt[:max_chars] + "\n…[truncated]"
    return txt or "(no context rendered)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", nargs="+", required=True,
                    help="model=path pairs, e.g. teacher=gen_teacher.jsonl base=gen_base.jsonl")
    ap.add_argument("--eval", default="dataset/eval.jsonl")
    ap.add_argument("--baseline-frac", type=float, default=0.5,
                    help="fraction of contexts that also get a degenerate-baseline row")
    ap.add_argument("--limit", type=int, default=0, help="cap items per model (0=all)")
    ap.add_argument("--seed", type=int, default=2397)
    ap.add_argument("--out-batch", default="judge_batch.jsonl")
    ap.add_argument("--out-key", default="judge_key.jsonl")
    args = ap.parse_args()

    # index input contexts by run_id
    ctx = {}
    for line in open(args.eval):
        r = json.loads(line)
        ctx[r["run_id"]] = render_context(r.get("input"))

    rows = []  # (model, run_id, issue_key, gold, response, context)
    seen_runs = []
    for pair in args.gen:
        model, path = pair.split("=", 1)
        data = json.load(open(path)) if path.endswith(".jsonl") and open(path).read(1) == "[" else [json.loads(l) for l in open(path)]
        if args.limit:
            data = data[:args.limit]
        for x in data:
            rid = x["run_id"]
            rows.append((model, rid, x.get("issue_key"), x.get("gold"),
                         x.get("response", ""), ctx.get(rid, "(context missing)")))
            seen_runs.append((rid, x.get("issue_key"), x.get("gold")))

    # degenerate baseline rows over a deterministic subset of distinct contexts
    rng = random.Random(args.seed)
    distinct = {}
    for rid, ik, gold in seen_runs:
        distinct[rid] = (ik, gold)
    dlist = sorted(distinct.items())
    rng.shuffle(dlist)
    n_base = int(len(dlist) * args.baseline_frac)
    for rid, (ik, gold) in dlist[:n_base]:
        rows.append(("baseline", rid, ik, gold, DEGENERATE, ctx.get(rid, "(context missing)")))

    # anonymize + shuffle
    rng.shuffle(rows)
    batch, key = [], []
    for i, (model, rid, ik, gold, resp, context) in enumerate(rows):
        bid = "B" + hashlib.sha1(f"{args.seed}:{i}:{rid}:{model}".encode()).hexdigest()[:10]
        batch.append({"blind_id": bid, "context": context, "gold": gold, "response": resp})
        key.append({"blind_id": bid, "model": model, "run_id": rid, "issue_key": ik, "gold": gold})

    with open(args.out_batch, "w") as f:
        for b in batch:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    with open(args.out_key, "w") as f:
        for k in key:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"wrote {args.out_batch} ({len(batch)} items), {args.out_key}")
    print("model mix:", dict(Counter(k["model"] for k in key)))


if __name__ == "__main__":
    main()
