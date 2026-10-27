#!/bin/bash
# DISABLED 2026-07-13 per TON-3160 (CEO: training/RAFT line parked - stock open-weights + serving optimization). Read DECISION_STOP_TRAINING.md before re-enabling.
echo "DISABLED per TON-3160: training/RAFT parked. See model-build/DECISION_STOP_TRAINING.md" >&2; exit 86
# TON-2394 — sovereign-v0 QLoRA run on Atlas using the NOPASSWD llama.service grant.
# llama.service is already stopped by the caller; this trains and ALWAYS restarts
# the service afterward (even on crash/kill) so the fleet's inference is restored.
set -u
cd /home/tony/model-build || exit 99
mkdir -p out
LOG=out/train-v0.log
DONE=out/train-v0.done
rm -f "$DONE" 2>/dev/null || true
log(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }

restore() {
    log "RESTORE: sudo systemctl start llama.service"
    sudo -n systemctl start llama.service 2>>"$LOG" || log "RESTORE WARN: start returned non-zero"
    for i in $(seq 1 60); do
        sleep 5
        curl -sf http://127.0.0.1:8020/health >/dev/null 2>&1 && { log "RESTORE: llama /health OK (${i}x5s)"; break; }
    done
}
trap 'restore; echo "TRAIN_RC=${RC:-130} at $(date -Is)" > "$DONE"' EXIT

# Safety: ensure the GPU is actually free before training.
for i in $(seq 1 12); do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    FREE=${FREE:-0}
    [ "$FREE" -ge 18000 ] && break
    log "waiting for VRAM (free=${FREE} MiB)"; sleep 5
done

log "TRAIN: starting QLoRA (Qwen3-4B), free VRAM=${FREE} MiB"
source /home/tony/sovereign-model-env/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=0
python sft_train.py --max-length 2048 --io-cap 800 --epochs 1 --batch-size 1 --grad-accum 16 >> "$LOG" 2>&1
RC=$?
log "TRAIN: finished RC=$RC"
