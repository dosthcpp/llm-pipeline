#!/usr/bin/env bash
# TON-2537 — final P2 policy-adapter eval-gen over H1 native prompts.
# Frees GPU (stop raw llama-server), generates from raft-v2-iter2-adapter,
# then ALWAYS restores the teacher endpoint (trap on EXIT).
set -u
cd "$HOME/model-build"
source /home/tony/sovereign-model-env/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=0
ADAPTER="$HOME/model-build/out/raft-v2-iter2-adapter"
LIMIT="${LIMIT:-40}"
MT="${MT:-2048}"

restore_llama() {
  echo "[$(date +%T)] restoring llama-server (teacher :8020) ..."
  if pgrep -f "llama-server .*8020" >/dev/null 2>&1; then
    echo "[$(date +%T)] llama-server already up"; return 0
  fi
  bash "$HOME/start-llama.sh"
  sleep 10
  for i in $(seq 1 20); do
    curl -s -m 5 http://127.0.0.1:8020/health 2>/dev/null | grep -q ok && { echo "[$(date +%T)] llama HEALTHY"; return 0; }
    sleep 6
  done
  echo "[$(date +%T)] WARNING: llama not healthy after restart — CHECK MANUALLY"
}
trap restore_llama EXIT

echo "[$(date +%T)] === Phase 1: free GPU (kill raw llama-server) ==="
pkill -f "llama-server .*--port 8020" 2>/dev/null || true
sleep 8
for i in $(seq 1 15); do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  echo "[$(date +%T)] free VRAM: ${FREE} MiB"
  [ "${FREE:-0}" -gt 18000 ] && break
  sleep 4
done

echo "[$(date +%T)] === Phase 2: HF gen — policy adapter (limit=$LIMIT) ==="
python eval_v2_gen.py --mode hf --adapter "$ADAPTER" --max-tokens "$MT" --limit "$LIMIT" --out gen_p2.jsonl 2>&1
RC=$?
echo "[$(date +%T)] gen rc=$RC, gen_p2.jsonl: $(python -c 'import json;print(len(json.load(open("gen_p2.jsonl"))))' 2>/dev/null || echo '?') records"
echo "[$(date +%T)] === disposition dist (p2 vs base vs teacher) ==="
python eval_v2_gen.py --dist gen_p2.jsonl gen_base.jsonl gen_teacher.jsonl 2>&1 || true
echo "[$(date +%T)] P2_EVAL_GEN_DONE rc=$RC"
