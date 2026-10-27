#!/usr/bin/env python3
"""TON-2893 — Blinded judge runner on judge_batch.jsonl using Claude API.

Takes the output of eval_v2_judge_prep.py (blind_id, context, gold, response)
and scores each item on JUDGE_RUBRIC.md using a Claude model (default: claude-opus-4-8).
Output schema matches reward_claude.py: {blind_id, disposition_correct, grounded,
actionable, calibrated, overall, note, _latency_s}.
--resume: skips already-scored blind_ids. Safe to restart.
"""
import argparse, concurrent.futures, json, os, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))

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
    sys.exit("ERROR: no ANTHROPIC_API_KEY in env and no .anthropic_key")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, help="judge_batch.jsonl from eval_v2_judge_prep.py")
    ap.add_argument("--out", required=True, help="output judge_scores.jsonl")
    ap.add_argument("--rubric", default=os.path.join(HERE, "eval_v2_local", "JUDGE_RUBRIC.md"))
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--filter-model", default="", help="only judge rows matching this model tag (empty=all)")
    ap.add_argument("--key-file", default="", help="judge_key.jsonl to filter by model tag")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    load_key()
    import anthropic
    client = anthropic.Anthropic(max_retries=4)
    system = open(args.rubric).read().strip()

    items = [json.loads(l) for l in open(args.batch) if l.strip()]

    # optional: only judge rows whose blind_id maps to filter_model
    if args.filter_model and args.key_file:
        key_map = {}
        for l in open(args.key_file):
            k = json.loads(l)
            key_map[k["blind_id"]] = k["model"]
        items = [it for it in items if key_map.get(it["blind_id"]) == args.filter_model]

    done = set()
    if args.resume and os.path.exists(args.out):
        for l in open(args.out):
            if l.strip():
                o = json.loads(l)
                if "overall" in o:
                    done.add(o["blind_id"])
    todo = [it for it in items if it["blind_id"] not in done]
    print(f"[judge_batch_claude] scoring {len(todo)} items ({len(done)} already done) "
          f"model={args.model} filter={args.filter_model or 'all'}", flush=True)

    lock = threading.Lock()
    counter = [0, 0]  # [ok, err]

    def score_one(it):
        t0 = time.time()
        user_msg = (
            f"Wake/issue context:\n{it['context']}\n\n"
            f"Gold disposition: {it['gold']}\n\n"
            f"Agent response:\n{it['response']}"
        )
        for attempt in range(4):
            try:
                msg = client.messages.create(
                    model=args.model,
                    system=system,
                    messages=[{"role": "user", "content": user_msg}],
                    max_tokens=512,
                    tools=[{
                        "name": "score_response",
                        "description": "Output structured quality scores",
                        "input_schema": SCHEMA,
                    }],
                    tool_choice={"type": "tool", "name": "score_response"},
                )
                scores = next(b.input for b in msg.content if getattr(b, "type", "") == "tool_use")
                rec = {"blind_id": it["blind_id"], **scores, "_latency_s": round(time.time() - t0, 1)}
                with lock:
                    with open(args.out, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    counter[0] += 1
                    if counter[0] % 10 == 0:
                        print(f"  [{counter[0]+counter[1]}/{len(todo)}] ok={counter[0]} err={counter[1]}", flush=True)
                return
            except Exception as e:
                wait = 10 * (2 ** attempt)
                print(f"  [retry {attempt+1}] {it['blind_id']}: {e} (wait {wait}s)", flush=True)
                time.sleep(wait)
        with lock:
            counter[1] += 1
            print(f"  [SKIP] {it['blind_id']} failed after 4 attempts", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(score_one, todo))

    print(f"[judge_batch_claude] done: ok={counter[0]} err={counter[1]} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
