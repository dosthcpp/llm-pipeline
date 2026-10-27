#!/usr/bin/env bash
# DISABLED 2026-07-13 per TON-3160 (CEO: training/RAFT line parked - stock open-weights + serving optimization). Read DECISION_STOP_TRAINING.md before re-enabling.
echo "DISABLED per TON-3160: training/RAFT parked. See model-build/DECISION_STOP_TRAINING.md" >&2; exit 86
# TON-2563 hang guard (supersedes TON-2438 version): kill v3 training if the GPU
# goes idle for 8 min (the step-39 stall signature) so run_sft_v3.sh's trap
# restores prod 35B. NEW: before killing, capture a py-spy native stack dump +
# dmesg tail so a recurrence yields a DEFINITIVE diagnosis (the issue's "재발 시
# 즉시 abort + 진단" mandate), not just another rc=137.
LOG="$HOME/model-build/train_watchdog.log"
DLOG="$HOME/model-build/deploy_v3.log"
DUMP="$HOME/model-build/hang_diag_$(date -u +%Y%m%dT%H%M%SZ).txt"
PYSPY="/home/tony/sovereign-model-env/bin/py-spy"
{
  echo "[wd] start $(date -u)"
  PID=""
  for i in $(seq 1 60); do
    PID=$(pgrep -f "[s]ft_train.py" | head -1)
    [ -n "$PID" ] && break
    sleep 5
  done
  [ -z "$PID" ] && { echo "[wd] no training proc after 5min, exit"; exit 0; }
  echo "[wd] training PID=$PID; waiting for first step marker (gate out load/tokenize)"
  for i in $(seq 1 80); do   # up to 40 min for load+tokenize of huge records
    kill -0 "$PID" 2>/dev/null || { echo "[wd] PID gone before first step"; exit 0; }
    tr "\r" "\n" < "$DLOG" 2>/dev/null | grep -qE "[0-9]+/171 \[" && break
    sleep 30
  done
  echo "[wd] training loop detected $(date -u); monitoring GPU idle"
  idle=0
  while kill -0 "$PID" 2>/dev/null; do
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d " ")
    if [ "${util:-0}" -lt 5 ]; then idle=$((idle+1)); else idle=0; fi
    # At 6min idle (12 ticks) grab a live diagnosis while the process is still stuck.
    if [ "$idle" -eq 12 ]; then
      laststep=$(tr "\r" "\n" < "$DLOG" 2>/dev/null | grep -oE "[0-9]+/171 \[" | tail -1)
      echo "[wd] GPU idle 6min — capturing hang diagnosis (last step: $laststep) $(date -u)" | tee -a "$DUMP"
      echo "=== py-spy dump PID=$PID ===" >> "$DUMP"
      timeout 60 "$PYSPY" dump --pid "$PID" --native >> "$DUMP" 2>&1 || echo "[wd] py-spy dump failed" >> "$DUMP"
      echo "=== dmesg tail ===" >> "$DUMP"
      (sudo -n dmesg 2>/dev/null | tail -40 >> "$DUMP") || echo "(no dmesg access)" >> "$DUMP"
      echo "=== nvidia-smi ===" >> "$DUMP"
      nvidia-smi >> "$DUMP" 2>&1
      echo "[wd] diagnosis written to $DUMP"
    fi
    if [ "$idle" -ge 16 ]; then
      echo "[wd] GPU idle 8min while train alive — KILLING $PID $(date -u)"
      kill -9 "$PID" 2>/dev/null
      break
    fi
    sleep 30
  done
  echo "[wd] exit $(date -u)"
} >> "$LOG" 2>&1
