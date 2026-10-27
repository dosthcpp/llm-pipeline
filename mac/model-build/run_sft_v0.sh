#!/bin/bash
# DISABLED 2026-07-13 per TON-3160 (CEO: training/RAFT line parked - stock open-weights + serving optimization). Read DECISION_STOP_TRAINING.md before re-enabling.
echo "DISABLED per TON-3160: training/RAFT parked. See model-build/DECISION_STOP_TRAINING.md" >&2; exit 86
# TON-2394 — self-contained sovereign-v0 QLoRA run on Atlas.
# Frees the GPU from the production llama-server, trains, and ALWAYS restores
# inference afterward (even on failure), so the fleet is never left without LLM.
set -u
cd /home/tony/model-build || exit 99
mkdir -p out
LOG="out/train-v0.log"
DONE="out/train-v0.done"
: > "$DONE.tmp" 2>/dev/null || true
rm -f "$DONE" 2>/dev/null || true

log(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }

restore_inference() {
    log "RESTORE: relaunching llama-server"
    bash /home/tony/start-llama.sh
    for i in $(seq 1 48); do
        sleep 5
        if curl -sf http://127.0.0.1:8020/health >/dev/null 2>&1; then
            log "RESTORE: llama-server /health OK after ${i}x5s"
            break
        fi
    done
    # Re-arm the watchdog tmux session (it will see llama already up and just monitor)
    if ! tmux has-session -t llama-server 2>/dev/null; then
        tmux new-session -d -s llama-server "bash ~/watchdog.sh > ~/watchdog.log 2>&1"
        log "RESTORE: watchdog tmux session re-armed"
    fi
}
# Guarantee restore on ANY exit (crash, kill, error).
trap 'restore_inference; echo "TRAIN_RC=${RC:-130} (trap) at $(date -Is)" > "$DONE"' EXIT

# 1) Free the GPU: stop watchdog first (else it respawns llama), then llama.
log "STOP: killing watchdog + llama-server"
tmux kill-session -t llama-server 2>/dev/null || true
pkill -f 'bash .*watchdog.sh' 2>/dev/null || true
pkill -x llama-server 2>/dev/null || true
pkill -f 'llama.cpp/build/bin/llama-server' 2>/dev/null || true

# 2) Wait for VRAM to actually free (>=18 GB free) before training.
for i in $(seq 1 24); do
    sleep 5
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    FREE=${FREE:-0}
    log "VRAM free=${FREE} MiB (wait ${i})"
    [ "$FREE" -ge 18000 ] && break
done

# 3) Train. RC captured for the sentinel; trap restores inference regardless.
log "TRAIN: starting QLoRA (Qwen3-4B)"
source /home/tony/sovereign-model-env/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=0
python sft_train.py >> "$LOG" 2>&1
RC=$?
log "TRAIN: finished RC=$RC"
# trap (EXIT) runs restore_inference + writes the final sentinel.
