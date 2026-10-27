#!/usr/bin/env python3
"""TON-2425 — Opus judge pass for sovereign-v2-8b eval.
Reads judge_batch_8b.jsonl from Atlas (via SSH), calls claude-opus-4-8 for each item,
writes judge_scores_8b.jsonl, then runs eval_v2_judge_score.py and prints table.

Usage:
  python3 run_opus_judge_8b.py [--batch PATH] [--out PATH] [--workers N] [--resume]
"""
import argparse, json, os, sys, time, subprocess, concurrent.futures
from pathlib import Path
import anthropic

RUBRIC = Path(__file__).parent / "JUDGE_RUBRIC.md"

SYSTEM = RUBRIC.read_text().strip()

MODEL = "claude-opus-4-8"
MAX_TOKENS = 256

def judge_one(client, item: dict) -> dict:
    prompt = (
        f"Context:\n{item['context']}\n\n"
        f"Gold disposition: {item['gold']}\n\n"
        f"Agent response:\n{item['response']}"
    )
    for attempt in range(4):
        try:
            msg = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text.strip()
            # strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            scores = json.loads(raw)
            return {"blind_id": item["blind_id"], **scores}
        except (anthropic.RateLimitError, anthropic.APIStatusError) as e:
            wait = 2 ** attempt * 10
            print(f"  [retry {attempt+1}] {e} — sleeping {wait}s", flush=True)
            time.sleep(wait)
        except json.JSONDecodeError as e:
            print(f"  [json-err] blind_id={item['blind_id']} raw={raw[:80]} err={e}", flush=True)
            return {"blind_id": item["blind_id"], "error": str(e), "raw": raw}
    return {"blind_id": item["blind_id"], "error": "max_retries"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="judge_batch_8b.jsonl")
    ap.add_argument("--out", default="judge_scores_8b.jsonl")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--resume", action="store_true", help="Skip already-scored blind_ids")
    args = ap.parse_args()

    batch_path = Path(args.batch)
    out_path = Path(args.out)

    if not batch_path.exists():
        print(f"ERROR: {batch_path} not found. Is judge_batch_8b.jsonl ready?", file=sys.stderr)
        sys.exit(1)

    items = [json.loads(l) for l in batch_path.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(items)} items from {batch_path}", flush=True)

    done_ids = set()
    if args.resume and out_path.exists():
        for l in out_path.read_text().splitlines():
            if l.strip():
                done_ids.add(json.loads(l)["blind_id"])
        print(f"Resuming: {len(done_ids)} already scored, {len(items)-len(done_ids)} remaining", flush=True)
        items = [i for i in items if i["blind_id"] not in done_ids]

    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("_ANTHROPIC_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(items)
    done = [0]

    with open(out_path, "a") as fout:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(judge_one, client, item): item for item in items}
            for fut in concurrent.futures.as_completed(futures):
                result = fut.result()
                fout.write(json.dumps(result) + "\n")
                fout.flush()
                done[0] += 1
                if done[0] % 10 == 0 or done[0] == total:
                    print(f"  {done[0]}/{total} scored", flush=True)

    print(f"\nDone. {total} scores written to {out_path}")

    # run scoring table
    key_path = Path(args.batch.replace("batch", "key"))
    if key_path.exists():
        print("\n=== EVAL TABLE ===")
        subprocess.run([
            sys.executable, str(Path(__file__).parent / "eval_v2_judge_score.py"),
            "--scores", str(out_path),
            "--key", str(key_path),
        ])
    else:
        print(f"Key file {key_path} not found — run eval_v2_judge_score.py manually")


if __name__ == "__main__":
    main()
