#!/usr/bin/env bash
# TON-3090: wait for the 8B triage endpoint (:8020) to come up, then run the
# escalate-only ensemble pipeline. GPU box is brought up by the CEO out-of-band.
set -u
cd /Volumes/Data/tony_ai_lab/model-build/ton3079
EP="http://127.0.0.1:8020"
DEADLINE=$(( $(date +%s) + 1500 ))   # wait up to 25 min for the CEO to fire it up

echo "[wait] polling ${EP} until up (deadline 25m)..."
while true; do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 4 "${EP}/health" 2>/dev/null)
  if [ "$code" != "000" ] && [ -n "$code" ]; then
    echo "[up] ${EP} answered HTTP ${code} — starting pipeline"
    break
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "[timeout] ${EP} still down after 25m — still blocked on GPU window"
    exit 7
  fi
  sleep 15
done

set -x
python3 sec_eval.py --backend openai:${EP} --model sovereign-v2-8b --label student_8b \
  && python3 sec_eval.py --ensemble --left det_baseline --right student_8b \
        --right-escalate-only --label ensemble_det_union_8besc \
  && python3 sec_eval.py --report
rc=$?
set +x
echo "[done] pipeline exit rc=${rc}"
exit $rc
