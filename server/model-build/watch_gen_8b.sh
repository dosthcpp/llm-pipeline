#!/usr/bin/env bash
# TON-2429 fix: fire prep_8b_judge.sh ONLY when gen_v2_8b.jsonl is COMPLETE.
# Completion = gen file non-empty AND no eval_v2_gen.py running AND train PID dead AND
#              line count stable across a 30s settle (guards incremental-write race).
# Idempotent + 2h cap. Logs to auto_judge_8b.log.
set -u
cd "$HOME/model-build"
LOG="$HOME/model-build/auto_judge_8b.log"
TRAIN_PID=2363340
echo "[$(date -u +%F\ %T)] watcher v2 start (pid $$) — gate=gen-complete" >> "$LOG"
DEADLINE=$(( $(date +%s) + 9000 ))   # 2.5h cap
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  if [ -f judge_batch_8b.jsonl ]; then
    echo "[$(date -u +%F\ %T)] judge_batch_8b.jsonl already exists; nothing to do" >> "$LOG"; exit 0
  fi
  if [ -s gen_v2_8b.jsonl ] && ! kill -0 "$TRAIN_PID" 2>/dev/null && ! pgrep -f eval_v2_gen.py >/dev/null 2>&1; then
    n1=$(wc -l < gen_v2_8b.jsonl)
    sleep 30
    n2=$(wc -l < gen_v2_8b.jsonl)
    if [ "$n1" = "$n2" ] && [ "$n1" -ge 300 ] && ! pgrep -f eval_v2_gen.py >/dev/null 2>&1; then
      echo "[$(date -u +%F\ %T)] gen COMPLETE ($n2 lines, stable, no gen proc) -> prep_8b_judge.sh" >> "$LOG"
      bash ~/model-build/prep_8b_judge.sh >> "$LOG" 2>&1
      PREP_RC=$?; echo "[$(date -u +%F %T)] prep_8b_judge rc=$PREP_RC" >> "$LOG"; bash ~/model-build/notify_batch_ready.sh >> "$LOG" 2>&1 && echo "[$(date -u +%F %T)] notified Paperclip" >> "$LOG"
      exit 0
    else
      echo "[$(date -u +%F\ %T)] gen not stable yet (n1=$n1 n2=$n2) — keep waiting" >> "$LOG"
    fi
  fi
  sleep 30
done
echo "[$(date -u +%F\ %T)] watcher TIMEOUT without complete gen_v2_8b.jsonl" >> "$LOG"
exit 1
