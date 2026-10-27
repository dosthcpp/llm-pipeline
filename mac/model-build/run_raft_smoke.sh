#!/usr/bin/env bash
# DISABLED 2026-07-13 per TON-3160 (CEO: training/RAFT line parked - stock open-weights + serving optimization). Read DECISION_STOP_TRAINING.md before re-enabling.
echo "DISABLED per TON-3160: training/RAFT parked. See model-build/DECISION_STOP_TRAINING.md" >&2; exit 86
# TON-2441 — RAFT loop SMOKE iteration (parent TON-2440, GO-P1 accepted 915d819c).
#
# Proves the offline reward-ranked-FT loop end-to-end on a few prompts, NO full run:
#   rollout (llama DOWN) -> reward (llama UP, NO blackout) -> select (CPU) -> 1 train step (llama DOWN)
# Sequential GPU phases only — never 35B reward + policy train concurrent on the single 3090.
# Each phase artifact is resumable; the trap guarantees llama-server (prod teacher) is restored.
#
# Tunables (env): K, NP (num prompts), MAXNEW, EPOCHS, ADAPTER (warm-start policy, "" = base 4B).
set -u
cd "$HOME/model-build"
source /home/tony/sovereign-model-env/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=0

ITER="${ITER:-0}"
K="${K:-4}"; NP="${NP:-5}"; MAXNEW="${MAXNEW:-1024}"
EPOCHS="${EPOCHS:-1}"; ADAPTER="${ADAPTER:-}"
ROLL="roll_iter${ITER}.jsonl"; REWARD="reward_iter${ITER}.jsonl"
RAFT="raft_iter${ITER}.jsonl"; OUT="$HOME/model-build/out/raft-v2-iter${ITER}-adapter"
PY=python
PAUSE_FILE="$HOME/watchdog.pause"
PAUSE_SECS="${PAUSE_SECS:-1800}"   # self-expiring; watchdog auto-resumes if driver dies

# Free the 3090 for a policy GPU phase. The ~/watchdog.sh keeper-alive resurrects
# llama-server within 10s of any kill, so we MUST pause it first (self-expiring sentinel),
# then kill every llama-server pid (owned by tony — no sudo), then wait for VRAM to drain.
stop_llama() {
  echo "[$(date +%T)] pausing watchdog (${PAUSE_SECS}s) + stopping llama-server to free VRAM ..."
  echo $(( $(date +%s) + PAUSE_SECS )) > "$PAUSE_FILE"
  for p in $(pgrep llama-server); do kill "$p" 2>/dev/null; done
  for i in $(seq 1 20); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ "${free:-0}" -ge 8000 ] && { echo "[$(date +%T)] VRAM free=${free}MiB"; return 0; }
    for p in $(pgrep llama-server); do kill -9 "$p" 2>/dev/null; done
    sleep 3
  done
  echo "[$(date +%T)] WARNING: VRAM not freed (free=${free:-?}MiB) — CHECK MANUALLY"; return 1
}

# Bring the prod teacher back and resume watchdog protection. Idempotent: if already
# healthy just lift the pause. Only remove the pause AFTER llama is healthy so the
# watchdog can't race us into a double-start.
start_llama() {
  if curl -s -m 5 http://127.0.0.1:8020/health 2>/dev/null | grep -q ok; then
    rm -f "$PAUSE_FILE"; echo "[$(date +%T)] llama already HEALTHY — watchdog resumed"; return 0
  fi
  echo "[$(date +%T)] starting llama-server (prod teacher) ..."
  bash "$HOME/start-llama.sh"
  for i in $(seq 1 40); do
    sleep 6
    curl -s -m 5 http://127.0.0.1:8020/health 2>/dev/null | grep -q ok && {
      rm -f "$PAUSE_FILE"; echo "[$(date +%T)] llama HEALTHY — watchdog resumed"; return 0; }
  done
  rm -f "$PAUSE_FILE"  # let watchdog take over recovery
  echo "[$(date +%T)] WARNING: llama not healthy in time — watchdog resumed to recover"; return 1
}
trap start_llama EXIT  # whatever happens, restore prod teacher + watchdog

echo "[$(date +%T)] ===== RAFT smoke iter${ITER}: K=$K prompts=$NP max_new=$MAXNEW epochs=$EPOCHS adapter='${ADAPTER:-base}' ====="

# ---- Phase 1: rollout (GPU = policy, llama DOWN) ----
echo "[$(date +%T)] === Phase 1/4 ROLLOUT ==="
stop_llama
$PY raft.py rollout --adapter "$ADAPTER" --k "$K" --limit "$NP" --max-new "$MAXNEW" --out "$ROLL"
RC=$?; [ $RC -ne 0 ] && { echo "[$(date +%T)] ROLLOUT FAILED rc=$RC"; exit $RC; }

# ---- Phase 2: reward (GPU = teacher @ :8020, llama UP, NO blackout) ----
echo "[$(date +%T)] === Phase 2/4 REWARD (teacher judge, no blackout) ==="
start_llama || { echo "[$(date +%T)] teacher not healthy — cannot score"; exit 1; }
$PY raft.py reward --rollouts "$ROLL" --out "$REWARD" --resume
RC=$?; [ $RC -ne 0 ] && { echo "[$(date +%T)] REWARD FAILED rc=$RC"; exit $RC; }

# ---- Phase 3: select (CPU) ----
echo "[$(date +%T)] === Phase 3/4 SELECT ==="
$PY raft.py select --rollouts "$ROLL" --rewards "$REWARD" --out "$RAFT"
RC=$?; [ $RC -ne 0 ] && { echo "[$(date +%T)] SELECT FAILED rc=$RC"; exit $RC; }
NRAFT=$(grep -c . "$RAFT" 2>/dev/null || echo 0)
if [ "$NRAFT" -lt 1 ]; then echo "[$(date +%T)] no RAFT rows selected — abort train"; exit 1; fi

# ---- Phase 4: one QLoRA train step on the selected set (GPU = policy, llama DOWN) ----
echo "[$(date +%T)] === Phase 4/4 TRAIN ($NRAFT RAFT rows) ==="
stop_llama
$PY sft_train.py \
  --data "$RAFT" --io-cap 0 --eval-data /tmp/_no_eval_$$ \
  --epochs "$EPOCHS" --max-length 2048 --batch-size 1 --grad-accum 4 \
  --lora-r 16 --lora-alpha 32 --lr 2e-4 \
  --output-dir "$OUT"
RC=$?; [ $RC -ne 0 ] && { echo "[$(date +%T)] TRAIN FAILED rc=$RC"; exit $RC; }

echo "[$(date +%T)] === RAFT smoke iter${ITER} COMPLETE ==="
ls -la "$OUT" 2>&1
echo "[$(date +%T)] artifacts: $ROLL $REWARD $RAFT  adapter: $OUT"
# trap restarts llama on exit
