#!/usr/bin/env bash
# TON-2873 — finish RAFT iter1 once claude -p (sonnet-5) reward lands.
# Reuses run_raft_r1.sh EXACT prereg select/train (reward phase no-ops on the
# complete reward file). Guards: abort if the iter1 adapter was not produced,
# so build_gguf never runs on a failed train. Science levers UNCHANGED.
set -uo pipefail
cd ~/model-build
SON=reward_8b_iter1_sonnet.jsonl
ADP=out/raft-v2-8b-iter1-adapter/adapter_model.safetensors
N=$(grep -c '"overall"' "$SON" 2>/dev/null || echo 0)
if [ "$N" -lt 1600 ]; then echo "[finish] reward incomplete ($N/1600) — abort"; exit 1; fi
echo "[finish $(date -u +%FT%TZ)] reward complete ($N rows). staging -> reward_8b_iter1.jsonl"
cp -f reward_8b_iter1.jsonl reward_8b_iter1.jsonl.prev.bak 2>/dev/null || true
cp -f "$SON" reward_8b_iter1.jsonl
echo "[finish] running driver (rollout skip, reward no-op, SELECT + TRAIN warm-start) ..."
bash run_raft_r1.sh
if [ ! -f "$ADP" ]; then
  echo "[finish] TRAIN FAILED — no iter1 adapter at $ADP — aborting before GGUF"; exit 1
fi
echo "[finish] adapter present. building GGUF Q4_K_M ..."
bash build_gguf_iter1.sh
echo "[finish $(date -u +%FT%TZ)] GGUF DONE. NEXT: frozen blinded eval via claude -p opus-4-8 -> 2.80 gate."
