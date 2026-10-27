#!/usr/bin/env python3
"""TON-2586 — Resumable, incremental-write native gen for Fable-5 (Q5_K_M :8030).
Writes one JSONL record per item (flush+fsync) so a window timeout never loses data.
Resumes by skipping run_ids already present in --out. Judge prep accepts JSONL.
"""
import argparse, json, os, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sft_train import SYSTEM_PROMPT
from eval_v2_gen import build_eval_subset, gold_disp, make_prompt, parse_disposition

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/model-build/dataset/eval.jsonl"))
    ap.add_argument("--url", default="http://127.0.0.1:8030/v1/chat/completions")
    ap.add_argument("--model", default="fable5")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--done-cap", type=int, default=30)
    ap.add_argument("--blocked-cap", type=int, default=30)
    ap.add_argument("--out", default=os.path.expanduser("~/model-build/gen_qwen27b_fable5.jsonl"))
    args = ap.parse_args()

    rows = build_eval_subset(args.data, args.done_cap, args.blocked_cap)
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            line = line.strip()
            if not line or line[0] == "[":
                continue
            try:
                done.add(json.loads(line)["run_id"])
            except Exception:
                pass
    print(f"subset N={len(rows)}, already done={len(done)}", flush=True)
    f = open(args.out, "a")
    t0 = time.time(); n = 0
    for r in rows:
        rid = r.get("run_id")
        if rid in done:
            continue
        body = json.dumps({"model": args.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": make_prompt(r)}],
            "temperature": 0.0, "max_tokens": args.max_tokens, "stream": False}).encode()
        text = ""
        for attempt in range(4):
            req = urllib.request.Request(args.url, data=body, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    j = json.loads(resp.read()); msg = j["choices"][0]["message"]
                    text = msg.get("content") or ""
                    if not text.strip():
                        text = msg.get("reasoning_content") or ""
                break
            except Exception as e:
                text = f"__ERR__ {e}"
                if attempt < 3:
                    time.sleep(20)
        rec = {"run_id": rid, "issue_key": r.get("issue_key"), "gold": gold_disp(r),
               "pred": parse_disposition(text), "response": text}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush(); os.fsync(f.fileno())
        n += 1
        print(f"  {len(done)+n}/{len(rows)} ({(time.time()-t0)/n:.1f}s/ex) rid={rid}", flush=True)
    f.close()
    print(f"DONE wrote {args.out} (+{n} new, total target {len(rows)})", flush=True)

if __name__ == "__main__":
    main()
