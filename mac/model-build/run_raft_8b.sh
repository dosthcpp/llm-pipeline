#!/usr/bin/env bash
# DISABLED 2026-07-13 per TON-3160 (CEO: training/RAFT line parked - stock open-weights + serving optimization). Read DECISION_STOP_TRAINING.md before re-enabling.
echo "DISABLED per TON-3160: training/RAFT parked. See model-build/DECISION_STOP_TRAINING.md" >&2; exit 86
# TON-2816 (S3) — v2-8B RAFT/GRPO iteration driver. Parent TON-2814 Option-1 (board-approved).
#
# Goal: close the +0.69 gap to the 2.80 frontier gate by directly targeting the residual
# hallucination dimensions — grounded (1.97) + calibrated (2.08) — with offline iterated
# reward-ranked FT (RAFT ~= offline GRPO), reusing the P0-validated teacher-as-reward (ρ=0.68).
#
# Differences vs run_raft_smoke.sh (which proved the loop at 4B):
#   * POLICY = Qwen3-8B + sovereign-v2-8b-adapter (the deployed v2-8B), warm-started so the
#     RAFT step REFINES the same adapter lineage (rollout adapter-N -> train -> adapter-N+1).
#   * ROLLOUT DATA = dataset/train.jsonl  (NOT eval.jsonl) — the frozen eval set stays HELD OUT
#     so the TON-2813 blinded harness remains an honest, uncontaminated re-measure.
#   * SHARDED rollout: the bulk GPU phase is split into NP/ROLL_CHUNK short windows; the prod
#     35B is restored between chunks (trap + start_llama), so no single long fleet blackout.
#   * SELECT uses the S3 default weights (overall primary; grounded+calibrated tiebreak).
#
# GPU residency (single RTX 3090, 35B reward server owns ~22GB when up):
#   rollout : 35B DOWN (policy on GPU)      <- the disruptive window; sharded + resumable
#   reward  : 35B UP   (teacher judge)      <- NO blackout (teacher already serves)
#   select  : CPU only
#   train   : 35B DOWN (policy on GPU)      <- short (~few hundred rows, 1 epoch)
#
# Env tunables: ITER K NP MAXNEW EPOCHS ROLL_CHUNK MIN_REWARD (see defaults below).
set -u
cd "$HOME/model-build"
source /home/tony/sovereign-model-env/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

ITER="${ITER:-0}"
BASE="${BASE:-Qwen/Qwen3-8B}"
V2_ADAPTER="${V2_ADAPTER:-$HOME/model-build/out/sovereign-v2-8b-adapter}"
# On iter>0, warm-start from the previous RAFT adapter instead of the v2-8b baseline.
if [ "$ITER" -gt 0 ]; then
  PREV=$(( ITER - 1 ))
  POLICY_ADAPTER="${POLICY_ADAPTER:-$HOME/model-build/out/raft-v2-8b-iter${PREV}-adapter}"
else
  POLICY_ADAPTER="${POLICY_ADAPTER:-$V2_ADAPTER}"
fi
K="${K:-4}"; NP="${NP:-200}"; MAXNEW="${MAXNEW:-1024}"; EPOCHS="${EPOCHS:-1}"
ROLL_CHUNK="${ROLL_CHUNK:-50}"      # prompts per rollout sub-window (short blackout each)
MIN_REWARD="${MIN_REWARD:-2.0}"
DATA="${DATA:-$HOME/model-build/dataset/train.jsonl}"

ROLL="roll_8b_iter${ITER}.jsonl"; REWARD="reward_8b_iter${ITER}.jsonl"
RAFT="raft_8b_iter${ITER}.jsonl";  OUT="$HOME/model-build/out/raft-v2-8b-iter${ITER}-adapter"
PY=python
PAUSE_FILE="$HOME/watchdog.pause"
PAUSE_SECS="${PAUSE_SECS:-2400}"    # self-expiring; watchdog auto-restores prod if driver dies

stop_llama() {
  echo "[$(date +%T)] pause watchdog (${PAUSE_SECS}s) + stop llama-server (free VRAM) ..."
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
  rm -f "$PAUSE_FILE"
  echo "[$(date +%T)] WARNING: llama not healthy in time — watchdog resumed to recover"; return 1
}
trap start_llama EXIT   # whatever happens, restore prod teacher + watchdog

echo "[$(date +%T)] ===== RAFT-8B iter${ITER}: base=$BASE policy_adapter=$(basename "$POLICY_ADAPTER")"
echo "[$(date +%T)]       K=$K NP=$NP max_new=$MAXNEW chunk=$ROLL_CHUNK epochs=$EPOCHS data=$(basename "$DATA") ====="

# ---- Phase 1: rollout (GPU=policy, 35B DOWN), SHARDED into short resumable windows ----
echo "[$(date +%T)] === Phase 1/4 ROLLOUT (sharded, $ROLL_CHUNK prompts/window) ==="
: > "$ROLL"
off=0
while [ "$off" -lt "$NP" ]; do
  lim=$(( NP - off )); [ "$lim" -gt "$ROLL_CHUNK" ] && lim="$ROLL_CHUNK"
  CH="roll_8b_iter${ITER}_off${off}.jsonl"
  echo "[$(date +%T)] --- rollout window: offset=$off limit=$lim -> $CH ---"
  stop_llama
  $PY raft.py rollout --data "$DATA" --base-model "$BASE" --adapter "$POLICY_ADAPTER" \
      --k "$K" --offset "$off" --limit "$lim" --max-new "$MAXNEW" --out "$CH"
  RC=$?; [ $RC -ne 0 ] && { echo "[$(date +%T)] ROLLOUT window failed rc=$RC"; exit $RC; }
  cat "$CH" >> "$ROLL"
  start_llama   # restore prod 35B between windows so the fleet is only briefly disrupted
  off=$(( off + lim ))
done
echo "[$(date +%T)] rollout complete: $(grep -c . "$ROLL") candidates -> $ROLL"

# ---- Phase 2: reward (GPU=teacher @ :8020, 35B UP, NO blackout), resumable ----
echo "[$(date +%T)] === Phase 2/4 REWARD (teacher judge, no blackout) ==="
start_llama || { echo "[$(date +%T)] teacher not healthy — cannot score"; exit 1; }
$PY raft.py reward --rollouts "$ROLL" --out "$REWARD" --resume
RC=$?; [ $RC -ne 0 ] && { echo "[$(date +%T)] REWARD FAILED rc=$RC"; exit $RC; }

# ---- Phase 3: select (CPU) — S3 default weights: overall primary, grounded+calibrated tiebreak ----
echo "[$(date +%T)] === Phase 3/4 SELECT (S3 grounded+calibrated shaping) ==="
$PY raft.py select --rollouts "$ROLL" --rewards "$REWARD" --min-reward "$MIN_REWARD" --out "$RAFT"
RC=$?; [ $RC -ne 0 ] && { echo "[$(date +%T)] SELECT FAILED rc=$RC"; exit $RC; }
NRAFT=$(grep -c . "$RAFT" 2>/dev/null || echo 0)
if [ "$NRAFT" -lt 1 ]; then echo "[$(date +%T)] no RAFT rows selected — abort train"; exit 1; fi

# ---- Phase 4: QLoRA train continuing the v2-8b adapter (GPU=policy, 35B DOWN) ----
echo "[$(date +%T)] === Phase 4/4 TRAIN ($NRAFT RAFT rows, warm-start $(basename "$POLICY_ADAPTER")) ==="
stop_llama
$PY sft_train.py \
  --model "$BASE" --init-adapter "$POLICY_ADAPTER" \
  --data "$RAFT" --io-cap 0 --eval-data /tmp/_no_eval_$$ \
  --epochs "$EPOCHS" --max-length 2048 --batch-size 1 --grad-accum 4 \
  --lr 1e-4 --output-dir "$OUT"
RC=$?; [ $RC -ne 0 ] && { echo "[$(date +%T)] TRAIN FAILED rc=$RC"; exit $RC; }

echo "[$(date +%T)] === RAFT-8B iter${ITER} COMPLETE ==="
ls -la "$OUT" 2>&1
echo "[$(date +%T)] artifacts: $ROLL $REWARD $RAFT  adapter: $OUT"
echo "[$(date +%T)] NEXT: build GGUF (build_measure_v2_8b.sh w/ ADAPTER=$OUT) -> re-measure on FROZEN harness"
echo "[$(date +%T)]       (eval_v2_gen.py --mode teacher + 8 blinded Opus judges + aggregate_8b_scores.py)"
# trap restarts llama on exit
