#!/usr/bin/env bash
# DISABLED 2026-07-13 per TON-3160 (CEO: training/RAFT line parked - stock open-weights + serving optimization). Read DECISION_STOP_TRAINING.md before re-enabling.
echo "DISABLED per TON-3160: training/RAFT parked. See model-build/DECISION_STOP_TRAINING.md" >&2; exit 86
# TON-2442 — P2 full iterated RAFT run (GATED: only run inside a CEO-approved GPU blackout window).
#
# Orchestrates N iterations of the TON-2441-validated per-iteration driver (run_raft_smoke.sh),
# chaining each iteration warm-start from the previous adapter, over the FULL balanced H1 subset
# (NP=0 => no --limit => done_cap/blocked_cap balanced ~60 prompts). Each iteration is:
#   rollout (blackout) -> reward (teacher @:8020, NO blackout) -> select (CPU) -> train (blackout)
# Strictly sequential GPU phases; the inner driver self-expiring watchdog.pause + EXIT trap
# guarantee the prod 35B teacher is always restored, even on a hang/kill (no v3 blind-blackout).
#
# RESUMABLE: a completed iteration (out/raft-v2-iter<i>-adapter/adapter_config.json present) is
# skipped. Re-run the same command to continue after any interruption.
#
# Tunables (env): NITERS (default 3), K (candidates/prompt, default 8), MAXNEW (default 1024),
#                 ITER_TIMEOUT (per-iteration hard cap seconds, default 9000 = 2.5h hang-guard).
set -u
cd "$HOME/model-build"
NITERS="${NITERS:-3}"; K="${K:-8}"; MAXNEW="${MAXNEW:-1024}"
ITER_TIMEOUT="${ITER_TIMEOUT:-9000}"
LOG="$HOME/model-build/raft_p2.log"
echo "[$(date -u +%FT%TZ)] ===== P2 RAFT full run: NITERS=$NITERS K=$K MAXNEW=$MAXNEW full-set ITER_TIMEOUT=${ITER_TIMEOUT}s =====" | tee -a "$LOG"

ADAPTER=""
for ITER in $(seq 0 $((NITERS-1))); do
  OUT="$HOME/model-build/out/raft-v2-iter${ITER}-adapter"
  if [ -f "$OUT/adapter_config.json" ]; then
    echo "[$(date -u +%FT%TZ)] iter${ITER}: adapter already present -> SKIP (resume), chain from it" | tee -a "$LOG"
    ADAPTER="$OUT"; continue
  fi
  echo "[$(date -u +%FT%TZ)] iter${ITER}: START warm-start adapter='${ADAPTER:-base}'" | tee -a "$LOG"
  # Per-iteration hard timeout = hang-guard (TON-2436 v3 hung step 39/171). On timeout the
  # inner driver EXIT trap still restores the prod teacher; the partial iter is resumable.
  ITER="$ITER" K="$K" NP=0 MAXNEW="$MAXNEW" ADAPTER="$ADAPTER" \
    timeout --signal=INT "${ITER_TIMEOUT}" bash run_raft_smoke.sh 2>&1 | tee -a "$LOG"
  RC=${PIPESTATUS[0]}
  if [ "$RC" -eq 124 ]; then echo "[$(date -u +%FT%TZ)] iter${ITER}: TIMEOUT ${ITER_TIMEOUT}s -> hang-guard fired, ABORT (resumable)" | tee -a "$LOG"; exit 124; fi
  if [ "$RC" -ne 0 ]; then echo "[$(date -u +%FT%TZ)] iter${ITER}: FAILED rc=$RC -> ABORT (resumable)" | tee -a "$LOG"; exit "$RC"; fi
  if [ ! -f "$OUT/adapter_config.json" ]; then echo "[$(date -u +%FT%TZ)] iter${ITER}: no adapter produced -> ABORT" | tee -a "$LOG"; exit 1; fi
  ADAPTER="$OUT"
  echo "[$(date -u +%FT%TZ)] iter${ITER}: DONE -> $OUT" | tee -a "$LOG"
done

echo "[$(date -u +%FT%TZ)] ===== P2 RAFT full run COMPLETE. Final policy adapter: $ADAPTER =====" | tee -a "$LOG"
echo "Next: eval-gen final policy on H1 native prompts + blinded Opus judge (TON-2397) vs base 2.52 / teacher 3.01." | tee -a "$LOG"
