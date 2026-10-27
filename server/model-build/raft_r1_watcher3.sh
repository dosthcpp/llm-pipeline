#!/usr/bin/env bash
# DISABLED 2026-07-13 per TON-3160 (CEO: training/RAFT line parked - stock open-weights + serving optimization). Read DECISION_STOP_TRAINING.md before re-enabling.
echo "DISABLED per TON-3160: training/RAFT parked. See model-build/DECISION_STOP_TRAINING.md" >&2; exit 86
# TON-2867 atlas-side watcher v4 (2026-07-03).
# v3→v4: crash signatures are NO LONGER terminal — driver v2/v3 auto-retries shards,
# so a mid-log CUDA traceback is expected retry noise; v3 exited on one at 05:10Z and
# left the retry hang unwatched. Terminal states now: COMPLETE, driver process gone,
# log stall >3000s (catches CUDA hangs). Posts to TON-2867 (live issue; TON-2860 is
# stranded-blocked). Token read at post time from .pc_token (run-scoped JWT, ~1h exp).
LOG="$HOME/model-build/raft_8b_iter1.log"
WLOG="$HOME/model-build/raft_r1_watcher.log"
STATUS="$HOME/model-build/raft_r1_status.txt"
TOKENF="$HOME/model-build/.pc_token"
API="http://100.93.196.36:3101"
ISSUE="TON-2867"
post_until_ok() {
  for i in $(seq 1 288); do
    KEY=$(cat "$TOKENF" 2>/dev/null)
    body=$(python3 -c 'import json,sys; print(json.dumps({"body": sys.argv[1]}))' "$1")
    code=$(curl -s -m 20 -o /tmp/pc_post_out -w "%{http_code}" -X POST "$API/api/issues/$ISSUE/comments" \
      -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" --data "$body")
    echo "$(date -u) post attempt $i http=$code $(head -c 120 /tmp/pc_post_out 2>/dev/null)" >> "$WLOG"
    case "$code" in 2*) return 0;; esac
    sleep 300
  done
  return 1
}
terminal() {
  echo "$1" > "$STATUS"
  echo "$(date -u) terminal: $1" >> "$WLOG"
  post_until_ok "$2"
  touch "$HOME/model-build/raft_r1_watcher.done"
  exit 0
}
echo "$(date -u) watcher4 started (pid $$)" >> "$WLOG"
echo "running (watcher4 start $(date -u +%H:%MZ))" > "$STATUS"
while true; do
  now=$(date +%s); mt=$(stat -c %Y "$LOG" 2>/dev/null || echo "$now"); age=$((now-mt))
  last=$(tail -1 "$LOG" | tr -d '\r' | cut -c1-100)
  if tail -c 4000 "$LOG" | grep -q "R1-B(iter1) COMPLETE"; then
    terminal "complete" "**[atlas watcher] RAFT R1 driver COMPLETE ($(date -u +%H:%MZ))** — rollout→reward→select→train 완료. 다음 단계: GGUF Q4_K_M 빌드 → frozen blinded eval → 2.80 게이트 판정."
  fi
  if ! pgrep -f "[r]un_raft_r1.sh" >/dev/null; then
    sig=$(grep -aE "ROLLOUT FAILED|REWARD FAILED|CUDA error|Traceback" "$LOG" | tail -1 | cut -c1-150)
    terminal "driver_gone: $last" "**[atlas watcher] RAFT R1 driver가 COMPLETE 없이 종료되었습니다 ($(date -u +%H:%MZ))** — last: ${last} / sig: ${sig} — CTO 개입이 필요합니다."
  fi
  if [ "$age" -gt 4800 ]; then
    terminal "stalled ${age}s: $last" "**[atlas watcher] RAFT R1 stall 감지 ($(date -u +%H:%MZ))**: ${age}s 동안 로그 무갱신 (CUDA hang 의심) — last: ${last} — CTO 개입이 필요합니다."
  fi
  sleep 120
done
