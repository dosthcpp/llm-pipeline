#!/usr/bin/env bash
set -uo pipefail
cd "$HOME/model-build"
L=out/await_prep.log
log(){ echo "[$(date -u +%H:%M:%S)] $*" >> "$L"; }
log "waiting for gen_qwen27b_fable5.jsonl (FABLE5_GEN_DONE) ..."
for i in $(seq 1 720); do  # up to 2h
  if grep -q "FABLE5_GEN_DONE" out/fable5_window.log 2>/dev/null && [ -s gen_qwen27b_fable5.jsonl ]; then
    n=$(wc -l < gen_qwen27b_fable5.jsonl)
    log "gen done ($n lines); running prep ..."
    bash prep_fable5_judge.sh >> "$L" 2>&1
    log "prep rc=$? -> judge_batch_fable5.jsonl ready"
    break
  fi
  sleep 15
done
