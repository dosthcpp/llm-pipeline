#!/usr/bin/env bash
# TON-2893 — Atlas-side watcher: posts gate result to Paperclip when measure_dpo_neg.sh completes.
# Runs as a loop from tmux or cron. Posts to TON-2893, TON-2867, TON-2837.
# Token read at post time from .pc_token (run-scoped JWT). Uses retry-on-failure pattern.
set -uo pipefail
MB="$HOME/model-build"
TOKENF="$MB/.pc_token"
API="http://100.93.196.36:3101"
LOG="$MB/measure_neg_watcher.log"
STATUS_FILE="$MB/measure_neg_watcher.status"

post_comment() {
  local issue="$1" body="$2"
  for i in $(seq 1 144); do
    KEY=$(cat "$TOKENF" 2>/dev/null)
    payload=$(python3 -c 'import json,sys; print(json.dumps({"body": sys.argv[1]}))' "$body")
    code=$(curl -s -m 30 -o /tmp/pc_post_out -w "%{http_code}" -X POST "$API/api/issues/$issue/comments" \
      -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" --data "$payload")
    echo "$(date -u) post $issue attempt $i http=$code $(head -c 120 /tmp/pc_post_out 2>/dev/null)" >> "$LOG"
    case "$code" in 2*) return 0;; esac
    sleep 300
  done
  return 1
}

echo "$(date -u) watcher start (pid $$)" >> "$LOG"
# poll every 2 min until measure_dpo_neg.sh signals done
while true; do
  if grep -q "MEASURE DONE" "$MB/measure_dpo_neg.log" 2>/dev/null; then
    echo "$(date -u) MEASURE_DONE detected" >> "$LOG"
    # extract gate result from the log
    result=$(awk '/GATE RESULT/,/MEASURE DONE/' "$MB/measure_dpo_neg.log" | head -40 | tr '\n' '|' | sed 's/|/\\n/g')
    # build message
    msg="## TON-2893 neg-signal DPO 게이트 판정 (자동 보고, Atlas watcher)

측정 완료. 사전등록 §4 기준: calibrated≥2.50 AND overall≥2.30.

\`\`\`
$(awk '/GATE RESULT/,/MEASURE DONE/' "$MB/measure_dpo_neg.log" | head -35)
\`\`\`

모델 파일: \`~/models/neg-v2-8b-dpo-Q4_K_M.gguf\`
judge 로그: \`~/model-build/judge_scores_neg_dpo.jsonl\`

결과에 따른 다음 단계 (PREREG §4):
- PASS → CEO 게이트 재관여 (쇼케이스/Version B 리프레시 + 보스 보고)
- FAIL → 즉시 중단, 보스 에스컬레이션 (자동 14B/다음 사이클 금지)"
    post_comment "TON-2893" "$msg"
    post_comment "TON-2867" "## neg-signal DPO 게이트 판정 (TON-2893 결과, Atlas watcher 자동 보고)

TON-2893 neg-DPO 측정 완료. 상세 결과 및 게이트 판정은 TON-2893 코멘트 참조."
    post_comment "TON-2837" "## neg-signal DPO 게이트 판정 (TON-2893 완료, Atlas watcher)

neg-signal 사이클(TON-2893) 측정 완료. 상세 결과 및 게이트 판정은 TON-2893 코멘트 참조. CEO 재관여 시점."
    echo "$(date -u) watcher done — posted to TON-2893/2867/2837" >> "$LOG"
    echo "done" > "$STATUS_FILE"
    exit 0
  fi
  # check for failure
  if grep -q "FAILED\|exit 1\|Error\|Traceback" "$MB/measure_dpo_neg.log" 2>/dev/null; then
    last=$(tail -5 "$MB/measure_dpo_neg.log")
    msg="**[Atlas watcher] TON-2893 measure_dpo_neg.sh FAILED** — 마지막 로그:\n\`\`\`\n${last}\n\`\`\`\nCTO 개입 필요."
    post_comment "TON-2893" "$msg"
    echo "$(date -u) watcher: FAILURE posted" >> "$LOG"
    echo "failed" > "$STATUS_FILE"
    exit 1
  fi
  sleep 120
done
