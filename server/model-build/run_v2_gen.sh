#!/usr/bin/env bash
# P0.3.1 TON-2397 — Redesigned eval generation. Phase 1: teacher (llama up).
# Phase 2: stop llama, native base + v0 gen, restart llama (trap). Background w/ nohup.
set -u
cd "$HOME/model-build"
source /home/tony/sovereign-model-env/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=0
ADAPTER="$HOME/model-build/out/sovereign-v0-adapter"
MT="${MT:-2048}"
DC="${DC:-30}"; BC="${BC:-30}"
PY=python

echo "[$(date +%T)] === Phase 1: TEACHER native gen (live llama-server :8020) ==="
$PY eval_v2_gen.py --mode teacher --max-tokens "$MT" --done-cap "$DC" --blocked-cap "$BC" \
    --out gen_teacher.jsonl 2>&1

restore_llama() {
  echo "[$(date +%T)] restarting llama.service ..."
  sudo -n /usr/bin/systemctl start llama.service
  sleep 8
  for i in $(seq 1 20); do
    curl -s -m 5 http://127.0.0.1:8020/health 2>/dev/null | grep -q ok && { echo "[$(date +%T)] llama HEALTHY"; return 0; }
    sleep 6
  done
  echo "[$(date +%T)] WARNING: llama not healthy — CHECK MANUALLY"
}
trap restore_llama EXIT

echo "[$(date +%T)] === Phase 2: stop llama, native hf gen ==="
sudo -n /usr/bin/systemctl stop llama.service
sleep 6
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

echo "[$(date +%T)] --- BASE native gen ---"
$PY eval_v2_gen.py --mode hf --adapter "" --max-tokens "$MT" --done-cap "$DC" --blocked-cap "$BC" --out gen_base.jsonl 2>&1
echo "[$(date +%T)] --- V0 native gen (no triage directive) ---"
$PY eval_v2_gen.py --mode hf --adapter "$ADAPTER" --max-tokens "$MT" --done-cap "$DC" --blocked-cap "$BC" --out gen_v0.jsonl 2>&1

echo "[$(date +%T)] === gen done — disposition distributions (v0-collapse diagnosis) ==="
$PY eval_v2_gen.py --dist gen_v0.jsonl gen_base.jsonl gen_teacher.jsonl 2>&1
echo "[$(date +%T)] ALL DONE"
