#!/usr/bin/env bash
# TON-2410 option 3 — orchestrate 8B retrain. Waits for Qwen3-8B download, then runs
# run_sft_8b.sh (stops 35B -> QLoRA train -> native gen -> restarts 35B via trap).
# Detached + robust: survives heartbeat cycles. Blackout = training duration only.
set -uo pipefail
cd ~/model-build
LOG=~/model-build/deploy_8b_master.log
: > "$LOG"
echo "[$(date -u +%H:%M:%S)] master start — waiting for Qwen3-8B download" | tee -a "$LOG"
# 1) wait for download to finish
for i in $(seq 1 240); do
  if ! pgrep -f "hf download" >/dev/null 2>&1; then break; fi
  sleep 15
done
SNAP=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/ 2>/dev/null | head -1)
NSAFE=$(ls "$SNAP"*.safetensors 2>/dev/null | wc -l)
SZ=$(du -s ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B 2>/dev/null | cut -f1)
echo "[$(date -u +%H:%M:%S)] download done: snapshot=$SNAP safetensors=$NSAFE size_kb=$SZ" | tee -a "$LOG"
if [ "${NSAFE:-0}" -lt 1 ] || [ "${SZ:-0}" -lt 14000000 ]; then
  echo "[$(date -u +%H:%M:%S)] ABORT: 8B download incomplete (<14GB or no safetensors). 35B untouched." | tee -a "$LOG"
  exit 1
fi
# 2) run the self-contained train script (handles 35B stop/restart via trap)
echo "[$(date -u +%H:%M:%S)] === LAUNCHING run_sft_8b.sh (35B BLACKOUT BEGINS) ===" | tee -a "$LOG"
bash ~/model-build/run_sft_8b.sh >> "$LOG" 2>&1
RC=$?
echo "[$(date -u +%H:%M:%S)] run_sft_8b.sh exited rc=$RC (35B should be restored by trap)" | tee -a "$LOG"
# 3) safety: ensure 35B is back regardless
sleep 5
if ! curl -s -m 5 http://127.0.0.1:8020/health 2>/dev/null | grep -q ok; then
  echo "[$(date -u +%H:%M:%S)] SAFETY: 35B not healthy, forcing start" | tee -a "$LOG"
  sudo -n /usr/bin/systemctl start llama.service
fi
echo "[$(date -u +%H:%M:%S)] MASTER DONE rc=$RC" | tee -a "$LOG"
