#!/usr/bin/env python3
"""TON-2806 launcher: strict frontier cert (Opus+Sonnet) via --mode api.

Reads ANTHROPIC_API_KEY straight from ~/.hermes/.env into the process env (avoids shell/rtk
mangling of the secret), then runs the frozen-spec greedy api generation with checkpoint+resume
so a teardown never re-pays. Never prints the key. Boss topped up credit (TON-2783 thread) —
use sparingly; checkpointing + bounded max_tokens=2048 keep spend at the ~$2.6 one-time floor.
"""
import os, sys

ENV = os.path.expanduser("~/.hermes/.env")
for line in open(ENV):
    if line.startswith("ANTHROPIC_API_KEY="):
        os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
        break
if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit("no ANTHROPIC_API_KEY in ~/.hermes/.env")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_v2_gen as e

DATA = "dataset/eval.jsonl"
rows = e.build_eval_subset(DATA, 30, 30)
targets = sys.argv[1:] or ["opus-4-8", "sonnet-4-6"]
outmap = {"opus-4-8": "gen_opus.jsonl", "sonnet-4-6": "gen_sonnet.jsonl", "haiku-4-5": "gen_haiku.jsonl"}
for tag in targets:
    out = outmap[tag]
    print(f"== {tag} -> {out} (mode=api, greedy, N={len(rows)}) ==", flush=True)
    e.run_api(rows, tag, 2048, out, concurrency=6)
print("DONE", flush=True)
