#!/usr/bin/env python3
"""TON-2563 — Blinded Opus judge runner for v3 eval batch.

Reads judge_batch_v3.jsonl (754 blinded items from prep_v3_judge.sh),
scores each with claude-opus-4-8 against JUDGE_RUBRIC.md, writes
judge_scores_v3.jsonl: {blind_id, disposition_correct, grounded,
actionable, calibrated, overall, note}.

Usage:
    python run_judge_v3.py                          # score all 754 items
    python run_judge_v3.py --batch judge_batch_v3.jsonl --out judge_scores_v3.jsonl
    python run_judge_v3.py --limit 10               # smoke test first 10 items
    python run_judge_v3.py --resume                 # skip already-scored blind_ids

Concurrency: --workers N (default 8) parallel Anthropic API calls.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import anthropic
except ImportError:
    sys.exit("ERROR: pip install anthropic")

JUDGE_MODEL = "claude-opus-4-8"
MAX_TOKENS = 256

def load_rubric(path: str) -> str:
    return Path(path).read_text().strip()

def score_item(client, rubric: str, item: dict, retries: int = 3) -> dict:
    """Call Opus to score one blinded judge item. Returns score dict."""
    prompt = f"""{rubric}

---
CONTEXT (what the agent saw):
{item['context']}

GOLD DISPOSITION: {item['gold']}

AGENT RESPONSE:
{item['response']}
---

Score the response above. Output STRICT JSON only, no markdown:
{{"disposition_correct": int, "grounded": int, "actionable": int, "calibrated": int, "overall": int, "note": "<=20 words"}}"""

    for attempt in range(retries):
        try:
            msg = client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text.strip()
            # strip markdown fences if model wraps anyway
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            scores = json.loads(text)
            scores["blind_id"] = item["blind_id"]
            return scores
        except json.JSONDecodeError as e:
            if attempt == retries - 1:
                print(f"  [WARN] {item['blind_id']}: JSON parse failed after {retries} attempts: {e}")
                return {"blind_id": item["blind_id"], "error": str(e)}
        except Exception as e:
            if attempt == retries - 1:
                print(f"  [WARN] {item['blind_id']}: API error after {retries} attempts: {e}")
                return {"blind_id": item["blind_id"], "error": str(e)}
            time.sleep(2 ** attempt)
    return {"blind_id": item["blind_id"], "error": "max retries"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="judge_batch_v3.jsonl")
    ap.add_argument("--rubric", default="JUDGE_RUBRIC.md")
    ap.add_argument("--out", default="judge_scores_v3.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true", help="skip already-scored blind_ids in --out")
    args = ap.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ERROR: ANTHROPIC_API_KEY not set")

    rubric = load_rubric(args.rubric)
    items = [json.loads(l) for l in open(args.batch)]
    if args.limit:
        items = items[:args.limit]

    already_scored = set()
    if args.resume and Path(args.out).exists():
        for line in open(args.out):
            r = json.loads(line)
            already_scored.add(r["blind_id"])
        print(f"Resuming: {len(already_scored)} already scored, skipping")
        items = [x for x in items if x["blind_id"] not in already_scored]

    print(f"Model: {JUDGE_MODEL} | Items to score: {len(items)} | Workers: {args.workers}")
    if not items:
        print("Nothing to score.")
        return

    client = anthropic.Anthropic(api_key=api_key)
    done = 0
    errors = 0
    start = time.time()

    out_file = open(args.out, "a")
    futures = {}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for item in items:
            fut = pool.submit(score_item, client, rubric, item)
            futures[fut] = item["blind_id"]

        for fut in as_completed(futures):
            bid = futures[fut]
            result = fut.result()
            out_file.write(json.dumps(result) + "\n")
            out_file.flush()
            done += 1
            if "error" in result:
                errors += 1
            elapsed = time.time() - start
            rate = done / elapsed
            remaining = (len(items) - done) / rate if rate > 0 else 0
            print(f"  [{done}/{len(items)}] {bid} overall={result.get('overall','ERR')} "
                  f"| {rate:.1f}/s | ETA {remaining:.0f}s")

    out_file.close()
    elapsed = time.time() - start
    print(f"\n=== DONE: {done} scored, {errors} errors in {elapsed:.0f}s ===")
    print(f"Scores written to {args.out}")
    print(f"\nNext: python eval_v2_judge_score.py --scores {args.out} --key judge_key_v3.jsonl")


if __name__ == "__main__":
    main()
