#!/usr/bin/env python3
"""TON-2440 Phase 0 — reward-validity check.

Run the LOCAL 35B teacher (serving at 127.0.0.1:8020, OpenAI-compatible) as an
LLM-judge over the SAME H1 eval items the blinded Opus judge scored, using the
SAME rubric and prompt format. Output local_teacher overall scores so we can
correlate them against judge_scores_h1.jsonl (Opus). If they DON'T correlate,
reward (a) (local-teacher-as-judge) is dead and the GRPO bet's reward must change
or the bet is killed (pre-registered kill criterion).

No GPU blackout: teacher is already serving production. We only send inference.

Usage:
  python3 local_teacher_judge.py [--batch judge_batch_h1.jsonl]
      [--out judge_scores_h1_local.jsonl] [--rubric JUDGE_RUBRIC.md]
      [--url http://127.0.0.1:8020] [--limit N] [--resume]
"""
import argparse, json, os, re, sys, time, urllib.request

MODEL = "Qwopus3.6-35B-A3B-v1-MTP-Q4_K_M.gguf"

def extract_json(raw: str):
    """Pull the last {...} JSON object out of possibly-think-wrapped text."""
    if raw is None:
        return None
    # drop <think>...</think>
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    # markdown fences
    if "```" in raw:
        parts = raw.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                raw = p
                break
    # find last balanced {...}
    cands = re.findall(r"\{[^{}]*\}", raw, flags=re.S)
    for c in reversed(cands):
        try:
            return json.loads(c)
        except Exception:
            continue
    try:
        return json.loads(raw.strip())
    except Exception:
        return None

def judge_one(url, system, item, timeout=180):
    user = (f"Context:\n{item['context']}\n\n"
            f"Gold disposition: {item['gold']}\n\n"
            f"Agent response:\n{item['response']}")
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": 2048,  # reasoning model: needs budget to finish <think> then emit JSON
        "temperature": 0,
        "top_p": 1,
    }
    data = json.dumps(payload).encode()
    t0 = time.time()
    last = None
    for attempt in range(5):  # server is flappy (watchdog restarts) -> retry w/ backoff
        try:
            req = urllib.request.Request(url + "/v1/chat/completions", data=data,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resp = json.loads(r.read().decode())
            break
        except Exception as e:
            last = e
            time.sleep(min(5 * (attempt + 1), 30))
    else:
        raise last
    dt = time.time() - t0
    ch = resp["choices"][0]["message"]
    raw = (ch.get("content") or "") or (ch.get("reasoning_content") or "")
    parsed = extract_json(raw)
    out = {"blind_id": item["blind_id"], "_latency_s": round(dt, 1)}
    if parsed and "overall" in parsed:
        for k in ("disposition_correct", "grounded", "actionable", "calibrated", "overall", "note"):
            if k in parsed:
                out[k] = parsed[k]
    else:
        out["error"] = "parse_fail"
        out["raw"] = raw[:300]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="judge_batch_h1.jsonl")
    ap.add_argument("--out", default="judge_scores_h1_local.jsonl")
    ap.add_argument("--rubric", default="JUDGE_RUBRIC.md")
    ap.add_argument("--url", default="http://127.0.0.1:8020")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    system = open(args.rubric).read().strip()
    items = [json.loads(l) for l in open(args.batch) if l.strip()]
    if args.limit:
        items = items[:args.limit]

    done = set()
    if args.resume and os.path.exists(args.out):
        for l in open(args.out):
            if l.strip():
                o = json.loads(l)
                if "overall" in o:  # only skip successfully-scored items; retry errors
                    done.add(o["blind_id"])
        items = [i for i in items if i["blind_id"] not in done]
    print(f"judging {len(items)} items ({len(done)} already done)", flush=True)

    n_ok = 0
    with open(args.out, "a") as f:
        for i, item in enumerate(items, 1):
            try:
                out = judge_one(args.url, system, item)
            except Exception as e:
                out = {"blind_id": item["blind_id"], "error": f"exc:{e}"}
            f.write(json.dumps(out) + "\n")
            f.flush()
            if "overall" in out:
                n_ok += 1
            print(f"  [{i}/{len(items)}] {item['blind_id']} "
                  f"overall={out.get('overall','ERR')} lat={out.get('_latency_s','?')}s",
                  flush=True)
            time.sleep(0.5)  # be gentle on the production teacher server
    print(f"done: {n_ok}/{len(items)} parsed OK -> {args.out}", flush=True)

if __name__ == "__main__":
    main()
