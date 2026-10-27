#!/usr/bin/env bash
# TON-2565 resumable runner: fresh start, then restart gen_v3_resumable.py until V3_DONE
# (rc 0) or attempt cap. Each restart resumes from gen_v3.jsonl + skiplist.
cd "$HOME/model-build"
source /home/tony/sovereign-model-env/bin/activate
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
LOG="$HOME/model-build/gen_v3_run.log"
{
  echo "[$(date -u +%T)] runner start; fresh state"
  rm -f gen_v3.jsonl gen_v3.skip V3_DONE
  for attempt in $(seq 1 40); do
    [ -f V3_DONE ] && { echo "[$(date -u +%T)] V3_DONE present"; break; }
    echo "[$(date -u +%T)] attempt $attempt launch"
    python gen_v3_resumable.py
    rc=$?
    echo "[$(date -u +%T)] attempt $attempt rc=$rc lines=$(wc -l < gen_v3.jsonl 2>/dev/null) skip=$(tr '\n' ' ' < gen_v3.skip 2>/dev/null)"
    [ $rc -eq 0 ] && break
    sleep 4
  done
  echo "[$(date -u +%T)] runner finished; V3_DONE=$([ -f V3_DONE ] && echo yes || echo no) lines=$(wc -l < gen_v3.jsonl 2>/dev/null)"
} >> "$LOG" 2>&1
