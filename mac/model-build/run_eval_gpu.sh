#!/usr/bin/env bash
# P0.3 TON-2395 — GPU eval phase: free VRAM (stop llama-server), run base + v0,
# then ALWAYS bring llama-server back (trap on EXIT). The teacher endpoint goes
# down for the duration of this script — keep it tight.
set -u
cd "$HOME/model-build"
source /home/tony/sovereign-model-env/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=0
ADAPTER="$HOME/model-build/out/sovereign-v0-adapter"
MT="${MT:-768}"
PY=python

restore_llama() {
  echo "[$(date +%T)] restarting llama.service ..."
  sudo -n /usr/bin/systemctl start llama.service
  sleep 8
  for i in $(seq 1 20); do
    if curl -s -m 5 http://127.0.0.1:8020/health 2>/dev/null | grep -q ok; then
      echo "[$(date +%T)] llama-server HEALTHY"; return 0
    fi
    sleep 6
  done
  echo "[$(date +%T)] WARNING: llama-server not healthy after restart — CHECK MANUALLY"
}
trap restore_llama EXIT

echo "[$(date +%T)] stopping llama.service to free VRAM ..."
sudo -n /usr/bin/systemctl stop llama.service
sleep 6
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

echo "[$(date +%T)] === BASE eval (Qwen3-4B, no adapter) ==="
$PY eval_harness.py --mode hf --adapter "" --max-tokens "$MT" --out preds_base.jsonl
echo "[$(date +%T)] === V0 eval (Qwen3-4B + sovereign-v0-adapter) ==="
$PY eval_harness.py --mode hf --adapter "$ADAPTER" --max-tokens "$MT" --out preds_v0.jsonl
echo "[$(date +%T)] GPU eval phase done."
