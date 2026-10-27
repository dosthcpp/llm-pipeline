#!/usr/bin/env bash
# TON-2893 — neg-signal DPO driver. Offline pairs already built (no rollout/reward GPU).
# Only the DPO train step needs the GPU: guarded stop/start of prod llama (27B), same
# non-destructive bracket as run_raft_r1.sh. Short job (186 pairs x 2 ep).
set -uo pipefail
cd "$HOME/model-build"
source /home/tony/sovereign-model-env/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT="$HOME/model-build/out/neg-v2-8b-dpo-adapter"
LOG="$HOME/model-build/dpo_neg.log"
PAUSE_FILE="$HOME/watchdog.pause"
PAUSE_SECS=3600
DONE="$HOME/model-build/dpo_neg.DONE"; FAIL="$HOME/model-build/dpo_neg.FAILED"
rm -f "$DONE" "$FAIL"

stop_llama() {
  echo "[$(date +%T)] pause watchdog(${PAUSE_SECS}s) + stop llama ..."
  echo $(( $(date +%s) + PAUSE_SECS )) > "$PAUSE_FILE"
  for p in $(pgrep llama-server); do kill "$p" 2>/dev/null; done
  for i in $(seq 1 20); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ "${free:-0}" -ge 8000 ] && { echo "[$(date +%T)] VRAM free=${free}MiB"; return 0; }
    for p in $(pgrep llama-server); do kill -9 "$p" 2>/dev/null; done
    sleep 3
  done
  echo "[$(date +%T)] WARNING: VRAM not freed"
}
restore_prod() {
  echo "[$(date +%T)] restore prod llama (parallel 1)"
  rm -f "$PAUSE_FILE"; pkill llama-server 2>/dev/null; sleep 5
  bash ~/start-llama.sh & sleep 20
  curl -s -m5 http://127.0.0.1:8020/health | grep -q ok && echo "[$(date +%T)] prod restored HEALTHY" || echo "[$(date +%T)] WARN llama not healthy"
}
trap restore_prod EXIT

exec > >(tee -a "$LOG") 2>&1
echo "[$(date +%T)] ===== TON-2893 neg-signal DPO START ====="
python build_dpo_pairs.py || { echo FAIL_PAIRS; touch "$FAIL"; exit 1; }
stop_llama
python dpo_train.py \
  --model Qwen/Qwen3-8B \
  --pairs dpo_pairs_iter1.jsonl \
  --init-adapter "$HOME/model-build/out/raft-v2-8b-iter0-adapter" \
  --output-dir "$OUT" \
  --beta 0.1 --lr 5e-6 --epochs 2 --grad-accum 8 --max-length 1024 --seed 42
RC=$?
if [ $RC -eq 0 ] && [ -f "$OUT/policy/adapter_model.safetensors" ]; then
  touch "$DONE"; echo "[$(date +%T)] ===== DPO TRAIN COMPLETE rc=0 ====="
  echo "[$(date +%T)] NEXT: build GGUF from $OUT/policy -> frozen eval_v2_gen (N=116) -> gate §4"
else
  touch "$FAIL"; echo "[$(date +%T)] ===== DPO TRAIN FAILED rc=$RC ====="
fi
