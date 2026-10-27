#!/usr/bin/env bash
# DISABLED 2026-07-13 per TON-3160 (CEO: training/RAFT line parked - stock open-weights + serving optimization). Read DECISION_STOP_TRAINING.md before re-enabling.
echo "DISABLED per TON-3160: training/RAFT parked. See model-build/DECISION_STOP_TRAINING.md" >&2; exit 86
# TON-2837 R1-B — RAFT iter-1 driver, Claude reward variant.
# v2 (TON-2860, 2026-07-03): ops-robustness patch after shard off=100 died with
# "CUDA error: unspecified launch failure" and the loop silently moved on:
#   - per-shard completeness check + ONE retry, then hard-fail (watcher catches FAILED)
#   - pre-merge total row-count guard (never feed a short rollout into reward/select/train)
#   - PAUSE_SECS 2400->4500: shards now run ~50s/prompt (~46min) and were outliving the
#     watchdog pause, risking a llama restart mid-rollout on the same GPU.
# Science levers (NP/K/MIN_REWARD/weights/guards/reward model) UNCHANGED from prereg R1-B.
# Identical to run_raft_r1.sh except Phase 2: reward is scored by a Claude model
# via API (reward_claude.py, default claude-fable-5; override with REWARD_MODEL)
# instead of the local 27B teacher. Consequences:
#   - NO llama-server restart bracket in Phase 2: prod 27B stays UP at parallel 1
#     the whole time (removes the riskiest ops step of the R0 plan).
#   - Reward wall-clock ~20 min (6 workers, ~3s/judgment) vs ~4-5h parallel-3 plan.
#   - Reward judge is fabrication-robust (iter-0 smoke: catches candidates the 27B
#     scored 5/5 that fabricate IDs/paths/outputs). R1-A select guards stay ON as
#     defense in depth.
# Requires: anthropic SDK in sovereign-model-env + ~/model-build/.anthropic_key.
set -uo pipefail
cd "$HOME/model-build"
source /home/tony/sovereign-model-env/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
ITER=1
BASE="Qwen/Qwen3-8B"
V2_ADAPTER="$HOME/model-build/out/sovereign-v2-8b-adapter"
ITER0_ADAPTER="$HOME/model-build/out/raft-v2-8b-iter0-adapter"
POLICY_ADAPTER="$ITER0_ADAPTER"  # warm-start from iter-0
K=4; NP=400; MAXNEW=1024; EPOCHS=1
ROLL_CHUNK=50
MIN_REWARD=3.0
DATA="$HOME/model-build/dataset/train.jsonl"
ROLL="roll_8b_iter${ITER}.jsonl"
REWARD="reward_8b_iter${ITER}.jsonl"
RAFT="raft_8b_iter${ITER}.jsonl"
OUT="$HOME/model-build/out/raft-v2-8b-iter${ITER}-adapter"
LOG="$HOME/model-build/raft_8b_iter${ITER}.log"
PAUSE_FILE="$HOME/watchdog.pause"
PAUSE_SECS=4500
stop_llama() {
  echo "[$(date +%T)] pause watchdog (${PAUSE_SECS}s) + stop llama-server ..."
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
start_llama() {
  echo "[$(date +%T)] resuming watchdog + starting llama-server (parallel 1) ..."
  rm -f "$PAUSE_FILE"
  bash ~/start-llama.sh &
  for i in $(seq 1 40); do
    curl -s -m4 http://127.0.0.1:8020/health 2>/dev/null | grep -q ok && {
      echo "[$(date +%T)] llama HEALTHY"; return 0; }
    sleep 5
  done
  echo "[$(date +%T)] WARNING: llama not healthy after restart"
}
restore_prod() {
  echo "[$(date +%T)] TRAP: restore prod llama (parallel 1)"
  pkill llama-server 2>/dev/null; sleep 5
  rm -f "$PAUSE_FILE"
  bash ~/start-llama.sh &
  sleep 20
  curl -s -m5 http://127.0.0.1:8020/health | grep -q ok && echo "[$(date +%T)] prod restored HEALTHY"
}
shard_done() {
  [ -f "$1" ] && [ "$(wc -l < "$1")" -ge $(( ROLL_CHUNK * K )) ]
}
exec > >(tee -a "$LOG") 2>&1
echo "[$(date +%T)] ===== RAFT-8B R1-B(iter${ITER}, claude reward) START [driver v2] ====="
echo "[$(date +%T)] policy_adapter=$POLICY_ADAPTER NP=$NP K=$K MIN_REWARD=$MIN_REWARD reward=claude-sonnet-5"
# ── Phase 1/4 ROLLOUT (sharded, prod 27B DOWN per shard) ─────────────────────
echo "[$(date +%T)] === Phase 1/4 ROLLOUT (sharded NP=$NP, ROLL_CHUNK=$ROLL_CHUNK) ==="
if [ -f "$ROLL" ] && [ "$(wc -l < "$ROLL")" -ge $(( NP * K )) ]; then
  echo "[$(date +%T)] rollout already complete ($(wc -l < "$ROLL") lines), skipping"
else
  PY=python
  for OFFSET in $(seq 0 $ROLL_CHUNK $(( NP - 1 ))); do
    OFF_TAG="_off${OFFSET}"
    SHARD_OUT="${ROLL%.jsonl}${OFF_TAG}.jsonl"
    if shard_done "$SHARD_OUT"; then
      echo "[$(date +%T)] shard off=${OFFSET} already done, skipping"
      continue
    fi
    for ATTEMPT in 1 2; do
      stop_llama
      echo "[$(date +%T)] rollout shard off=${OFFSET} limit=${ROLL_CHUNK} (attempt ${ATTEMPT}) ..."
      $PY raft.py rollout \
        --data "$DATA" --base-model "$BASE" --adapter "$POLICY_ADAPTER" \
        --k $K --offset $OFFSET --limit $ROLL_CHUNK --max-new $MAXNEW \
        --out "$SHARD_OUT"
      start_llama
      sleep 10
      shard_done "$SHARD_OUT" && break
      echo "[$(date +%T)] WARNING: shard off=${OFFSET} incomplete ($(wc -l < "$SHARD_OUT" 2>/dev/null || echo 0) rows) after attempt ${ATTEMPT}"
    done
    if ! shard_done "$SHARD_OUT"; then
      echo "[$(date +%T)] ROLLOUT FAILED: shard off=${OFFSET} still incomplete after 2 attempts — aborting driver"
      exit 1
    fi
  done
  # merge shards
  cat "${ROLL%.jsonl}"_off*.jsonl > "$ROLL"
  echo "[$(date +%T)] rollout merged: $(wc -l < "$ROLL") rows"
  if [ "$(wc -l < "$ROLL")" -lt $(( NP * K )) ]; then
    echo "[$(date +%T)] ROLLOUT FAILED: merged $(wc -l < "$ROLL") < $(( NP * K )) rows — aborting driver"
    exit 1
  fi
fi
# ── Phase 2/4 REWARD (Claude API — prod 27B stays UP, no restart) ────────────
REWARD_MODEL="${REWARD_MODEL:-claude-fable-5}"
echo "[$(date +%T)] === Phase 2/4 REWARD ($REWARD_MODEL, workers 6, no GPU contention) ==="
trap restore_prod EXIT
python reward_claude.py \
  --rollouts "$ROLL" \
  --model "$REWARD_MODEL" \
  --workers 6 --resume \
  --out "$REWARD"
RC=$?; [ $RC -ne 0 ] && { echo "[$(date +%T)] REWARD FAILED rc=$RC"; exit $RC; }
# ── Phase 3/4 SELECT (R1-A guards, defense in depth) ─────────────────────────
echo "[$(date +%T)] === Phase 3/4 SELECT (R1-A guards) ==="
python raft.py select \
  --rollouts "$ROLL" --rewards "$REWARD" \
  --min-reward $MIN_REWARD \
  --w-grounded 0.7 --w-calibrated 0.3 \
  --len-penalty-threshold 4000 --len-penalty 0.5 \
  --grounded-floor 3.0 \
  --out "$RAFT"
echo "[$(date +%T)] selected $(wc -l < "$RAFT") RAFT training rows"
# ── Phase 4/4 TRAIN ──────────────────────────────────────────────────────────
echo "[$(date +%T)] === Phase 4/4 TRAIN (warm-start from iter0-adapter) ==="
stop_llama
python sft_train.py \
  --data "$RAFT" --out "$OUT" \
  --init-adapter "$POLICY_ADAPTER" \
  --base-model "$BASE" \
  --epochs $EPOCHS
start_llama
echo "[$(date +%T)] === RAFT-8B R1-B(iter${ITER}) COMPLETE ==="
echo "[$(date +%T)] NEXT: build GGUF (build_gguf_iter1.sh) -> frozen re-measure -> 2.80 gate"