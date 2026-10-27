#!/usr/bin/env bash
# DISABLED 2026-07-13 per TON-3160 (CEO: training/RAFT line parked - stock open-weights + serving optimization). Read DECISION_STOP_TRAINING.md before re-enabling.
echo "DISABLED per TON-3160: training/RAFT parked. See model-build/DECISION_STOP_TRAINING.md" >&2; exit 86
# TON-2860 atlas-side watcher v3 (2026-07-03).
# Fixes vs v2:
#  - Paperclip JWTs are run-scoped (~1h exp), so the token is READ AT POST TIME from
#    ~/model-build/.pc_token (CTO heartbeats refresh this file). Failed posts retry
#    every 5 min until a fresh token lands — never exit on a failed post (v2 posted
#    once, got 401 Unauthorized, and died silently with the .done marker set).
#  - Failure signatures are only matched in log bytes appended AFTER watcher start
#    (v2 false-fired on the stale 03:47Z CUDA traceback still in its tail window).
#  - Writes terminal state to ~/model-build/raft_r1_status.txt so heartbeat wakes can
#    pull state instantly even if the comment post is still pending.
LOG="$HOME/model-build/raft_8b_iter1.log"
WLOG="$HOME/model-build/raft_r1_watcher.log"
STATUS="$HOME/model-build/raft_r1_status.txt"
TOKENF="$HOME/model-build/.pc_token"
API="http://100.93.196.36:3101"
post_until_ok() {
  for i in $(seq 1 288); do
    KEY=$(cat "$TOKENF" 2>/dev/null)
    body=$(python3 -c 'import json,sys; print(json.dumps({"body": sys.argv[1]}))' "$1")
    code=$(curl -s -m 20 -o /tmp/pc_post_out -w "%{http_code}" -X POST "$API/api/issues/TON-2860/comments" \
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
echo "$(date -u) watcher3 started (pid $$)" >> "$WLOG"
echo "running (watcher3 start $(date -u +%H:%MZ))" > "$STATUS"
LASTSZ=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
while true; do
  now=$(date +%s); mt=$(stat -c %Y "$LOG" 2>/dev/null || echo "$now"); age=$((now-mt))
  last=$(tail -1 "$LOG" | tr -d '\r' | cut -c1-100)
  sz=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
  if tail -c 4000 "$LOG" | grep -q "R1-B(iter1) COMPLETE"; then
    terminal "complete" "**[atlas watcher] RAFT R1 driver COMPLETE ($(date -u +%H:%MZ))** — rollout→reward→select→train 전 단계 종료. 다음 단계: GGUF 빌드 → frozen 재측정 → 2.80 게이트 판정을 진행해 주십시오."
  fi
  if [ "$sz" -gt "$LASTSZ" ]; then
    sig=$(tail -c +"$((LASTSZ+1))" "$LOG" | grep -aE "FAILED|Traceback|CUDA error|OOM" | tail -1 | cut -c1-150)
    if [ -n "$sig" ]; then
      terminal "failed: $sig" "**[atlas watcher] RAFT R1 실패 시그니처 감지 ($(date -u +%H:%MZ))**: ${sig} — CTO 개입이 필요합니다."
    fi
  fi
  LASTSZ=$sz
  if ! pgrep -f "[r]un_raft_r1.sh" >/dev/null; then
    terminal "driver_gone: $last" "**[atlas watcher] RAFT R1 driver 프로세스가 COMPLETE 없이 종료되었습니다 ($(date -u +%H:%MZ))** — 마지막 로그: ${last} — CTO 개입이 필요합니다."
  fi
  if [ "$age" -gt 3000 ]; then
    terminal "stalled ${age}s: $last" "**[atlas watcher] RAFT R1 stall 감지 ($(date -u +%H:%MZ))**: ${age}s 동안 로그 무갱신 — 마지막 로그: ${last} — wedge 재발 가능성, CTO 확인 필요."
  fi
  sleep 120
done
