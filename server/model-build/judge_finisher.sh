#!/usr/bin/env bash
# TON-2565 finisher: after V3_DONE+gen_v3.jsonl, build the blinded 6-model judge batch.
# Idempotent + detached. 35B restore is handled by run_sft_v3.sh's trap (port 8020).
set -u
cd "$HOME/model-build"
LOG="$HOME/model-build/judge_finisher.log"
{
  echo "[$(date -u +%FT%TZ)] finisher start (pid $$)"
  if [ -f "$HOME/model-build/V3_JUDGE_READY" ]; then
    echo "[$(date -u +%FT%TZ)] V3_JUDGE_READY already present — nothing to do"; exit 0
  fi
  # wait up to ~110 min for the gen to finish
  for i in $(seq 1 220); do
    if [ -f "$HOME/model-build/V3_DONE" ] && [ -s "$HOME/model-build/gen_v3.jsonl" ]; then
      echo "[$(date -u +%FT%TZ)] V3_DONE present; gen_v3 lines=$(wc -l < gen_v3.jsonl)"; break
    fi
    sleep 30
  done
  if [ ! -f "$HOME/model-build/V3_DONE" ] || [ ! -s "$HOME/model-build/gen_v3.jsonl" ]; then
    echo "[$(date -u +%FT%TZ)] TIMEOUT: V3_DONE/gen_v3.jsonl not ready — aborting prep"; exit 2
  fi
  echo "[$(date -u +%FT%TZ)] running prep_v3_judge.sh"
  if bash prep_v3_judge.sh; then
    touch "$HOME/model-build/V3_JUDGE_READY"
    echo "[$(date -u +%FT%TZ)] V3_JUDGE_READY: batch=$(wc -l < judge_batch_v3.jsonl) key=$(wc -l < judge_key_v3.jsonl)"
  else
    echo "[$(date -u +%FT%TZ)] prep_v3_judge.sh FAILED rc=$?"; exit 3
  fi
  echo "[$(date -u +%FT%TZ)] finisher done"
} >> "$LOG" 2>&1
