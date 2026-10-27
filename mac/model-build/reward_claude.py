#!/usr/bin/env python3
"""TON-2837 R1-B — RAFT reward phase with a Claude teacher-judge.

Drop-in replacement for `raft.py reward` (same rollout input, same reward-jsonl
output schema, same JUDGE_RUBRIC.md system prompt, --resume compatible), but the
judge is a Claude model via the Anthropic API instead of the local 27B teacher.

Model choice (--model): claude-fable-5 (default, boss-confirmed direction) or
claude-sonnet-5. 18-item iter-0 smoke (2026-07-03): both catch the fabricated
candidates the 27B rated 5/5; Fable is stricter and more uniform (grounded 1-2 on
every hacked candidate; also flags think-section leakage), ~$65/cycle vs ~$9,
p50 8.9s vs 3.3s per judgment. Fable requires no `thinking` param (always on)
and a large max_tokens; Sonnet runs thinking-disabled at max_tokens=256.

Goodhart guard: the frozen eval harness judge is claude-opus-4-8 — either choice
keeps the reward judge a different model from the gate judge. Rubric text is
unchanged; output schema is enforced with structured outputs (smoke showed
Sonnet otherwise omits `overall` on ~40% of items).

Ops profile vs local teacher: no GPU contention (prod 27B stays parallel-1 UP),
1600 judgments in ~20-40 min at 6 workers.

Key: ANTHROPIC_API_KEY env, else ~/model-build/.anthropic_key (chmod 600).

Usage:
  python reward_claude.py --rollouts roll_8b_iter1.jsonl --out reward_8b_iter1.jsonl \
      [--model claude-fable-5] [--workers 6] [--resume] [--limit N]
"""
import argparse, concurrent.futures, json, os, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "eval_v2_local"))
from eval_v2_judge_prep import render_context  # noqa: E402
SCHEMA = {
    "type": "object",
    "properties": {
        "disposition_correct": {"type": "integer"},
        "grounded": {"type": "integer"},
        "actionable": {"type": "integer"},
        "calibrated": {"type": "integer"},
        "overall": {"type": "integer"},
        "note": {"type": "string"},
    },
    "required": ["disposition_correct", "grounded", "actionable",
                 "calibrated", "overall", "note"],
    "additionalProperties": False,
}


def load_key():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    keyfile = os.path.join(HERE, ".anthropic_key")
    if os.path.exists(keyfile):
        os.environ["ANTHROPIC_API_KEY"] = open(keyfile).read().strip()
        return
    sys.exit("ERROR: no ANTHROPIC_API_KEY in env and no ~/model-build/.anthropic_key")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rubric", default=os.path.join(HERE, "eval_v2_local", "JUDGE_RUBRIC.md"))
    ap.add_argument("--model", default="claude-fable-5",
                    choices=["claude-fable-5", "claude-sonnet-5"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    load_key()
    import anthropic
    client = anthropic.Anthropic(max_retries=4)
    system = open(args.rubric).read().strip()

    cands = [json.loads(l) for l in open(args.rollouts) if l.strip()]
    if args.limit:
        cands = cands[: args.limit]
    done = set()
    if args.resume and os.path.exists(args.out):
        for l in open(args.out):
            if l.strip():
                o = json.loads(l)
                if "overall" in o:
                    done.add(o["id"])
    todo = [c for c in cands if f"{c['run_id']}#{c['cand_idx']}" not in done]
    print(f"reward(claude): scoring {len(todo)} candidates ({len(done)} already done)",
          flush=True)

    write_lock = threading.Lock()
    counter = [0, 0]  # [ok, done]

    # Fable 5: thinking is always on (an explicit `thinking` param 400s) and
    # thinking tokens count toward max_tokens, so give it headroom. Sonnet 5:
    # run thinking-disabled at a tight cap (matches the Opus frozen-judge setup).
    if args.model == "claude-fable-5":
        req_kwargs = {"max_tokens": 8000}
    else:
        req_kwargs = {"max_tokens": 256, "thinking": {"type": "disabled"}}

    def judge_one(item):
        prompt = (f"Context:\n{item['context']}\n\n"
                  f"Gold disposition: {item['gold']}\n\n"
                  f"Agent response:\n{item['response']}")
        t0 = time.time()
        last = None
        for attempt in range(5):
            try:
                msg = client.messages.create(
                    model=args.model, system=system,
                    messages=[{"role": "user", "content": prompt}],
                    extra_body={"output_config": {"format": {
                        "type": "json_schema", "schema": SCHEMA}}},
                    **req_kwargs,
                )
                if msg.stop_reason == "refusal":
                    cat = msg.stop_details.category if msg.stop_details else None
                    return {"error": f"refusal:{cat}"}
                raw = next(b.text for b in msg.content if b.type == "text")
                scores = json.loads(raw)
                scores["_latency_s"] = round(time.time() - t0, 1)
                return scores
            except anthropic.RateLimitError as e:
                last = f"rate_limit:{e}"
                time.sleep(2 ** attempt * 10)
            except anthropic.APIStatusError as e:
                last = f"api:{e.status_code}"
                time.sleep(2 ** attempt * 5)
            except Exception as e:  # noqa: BLE001 — resumable batch: record & continue
                last = f"exc:{e}"
                time.sleep(2 ** attempt * 2)
        return {"error": f"max_retries:{last}"}

    def score_one(c):
        cid = f"{c['run_id']}#{c['cand_idx']}"
        item = {"context": render_context(c["input"]), "gold": c.get("gold"),
                "response": c.get("response", "")}
        sc = judge_one(item)
        out = {"id": cid, "run_id": c["run_id"], "cand_idx": c["cand_idx"]}
        for k in ("disposition_correct", "grounded", "actionable", "calibrated",
                  "overall", "note", "error", "_latency_s"):
            if k in sc:
                out[k] = sc[k]
        with write_lock:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
            counter[1] += 1
            if "overall" in out:
                counter[0] += 1
            print(f"  [{counter[1]}/{len(todo)}] {cid} overall={out.get('overall', 'ERR')} "
                  f"lat={out.get('_latency_s', '?')}s", flush=True)
        return out

    with open(args.out, "a") as f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(score_one, todo))
    print(f"reward(claude) done: {counter[0]}/{len(todo)} scored OK -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
