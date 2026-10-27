#!/bin/bash
# DISABLED 2026-07-13 per TON-3160 (CEO: training/RAFT line parked - stock open-weights + serving optimization). Read DECISION_STOP_TRAINING.md before re-enabling.
echo "DISABLED per TON-3160: training/RAFT parked. See model-build/DECISION_STOP_TRAINING.md" >&2; exit 86
# TON-2394 — no-root cooperative GPU window for sovereign-v0 QLoRA.
# Adds a temporary lock-gate to ~/watchdog.sh (a file we own) so the watchdog
# itself keeps llama-server down while TRAIN_LOCK exists — cooperating with
# systemd (the watchdog process stays alive → no Restart cascade) instead of
# fighting it. Fully reversible: backs up watchdog.sh, restores it at the end.
set -u
WD=/home/tony/watchdog.sh
BAK=/home/tony/watchdog.sh.bak-TON2394
LOCK=/home/tony/TRAIN_LOCK
cd /home/tony/model-build || exit 99
mkdir -p out
LOG=out/train-v0.log
DONE=out/train-v0.done
rm -f "$DONE" 2>/dev/null || true
log(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }

cleanup() {
    log "CLEANUP: removing lock + restoring watchdog.sh"
    rm -f "$LOCK" 2>/dev/null || true
    [ -f "$BAK" ] && cp "$BAK" "$WD"
    # Wait for the watchdog to cold-start llama again (lock gone) and go healthy.
    for i in $(seq 1 48); do
        sleep 5
        curl -sf http://127.0.0.1:8020/health >/dev/null 2>&1 && { log "RESTORE: llama /health OK (${i}x5s)"; break; }
    done
    echo "TRAIN_RC=${RC:-130} at $(date -Is)" > "$DONE"
}
trap cleanup EXIT

# 1) Back up + insert the lock-gate right after `while true; do` (idempotent).
cp "$WD" "$BAK"
if ! grep -q 'TON-2394 cooperative GPU yield' "$WD"; then
    python3 - "$WD" <<'PY'
import sys
p=sys.argv[1]
s=open(p).read()
hook=('while true; do\n'
      '    # [TON-2394 cooperative GPU yield] While TRAIN_LOCK exists, keep\n'
      '    # llama-server down so the GPU is free for training. The watchdog\n'
      '    # process stays alive (systemd sees it running -> no restart cascade).\n'
      '    # Inert when no lock file is present.\n'
      '    if [ -f /home/tony/TRAIN_LOCK ]; then\n'
      '        pkill -x llama-server 2>/dev/null\n'
      '        sleep 10\n'
      '        continue\n'
      '    fi\n')
assert 'while true; do\n' in s, "anchor not found"
s=s.replace('while true; do\n', hook, 1)
open(p,'w').write(s)
print("patched")
PY
fi
log "PATCH: lock-gate present in watchdog.sh"

# 2) Create the lock + restart the watchdog so it runs the patched code.
touch "$LOCK"
log "LOCK created; restarting watchdog to load patched code"
# Kill the running watchdog (NOT our own shell); systemd Restart=on-failure
# brings it back with the patched file within RestartSec(30s).
for pid in $(pgrep -f 'bash .*watchdog.sh'); do kill "$pid" 2>/dev/null; done

# 3) Wait for the GPU to free (patched watchdog kills llama while locked).
for i in $(seq 1 30); do
    sleep 5
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    FREE=${FREE:-0}
    log "VRAM free=${FREE} MiB (wait ${i})"
    [ "$FREE" -ge 18000 ] && break
done

# 4) Train. cleanup() (EXIT trap) removes lock + restores watchdog regardless.
log "TRAIN: starting QLoRA (Qwen3-4B)"
source /home/tony/sovereign-model-env/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=0
python sft_train.py >> "$LOG" 2>&1
RC=$?
log "TRAIN: finished RC=$RC"
