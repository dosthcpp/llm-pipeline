#!/usr/bin/env bash
# TON-2537 — auto-resume policy gen, skipping any stochastic CUDA stall.
set -u
cd "$HOME/model-build"
source /home/tony/sovereign-model-env/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ADAPTER="$HOME/model-build/out/raft-v2-iter2-adapter"
END=40
OUT=gen_p2.jsonl

maxidx() {
  python - "$OUT" <<'PYEOF' 2>/dev/null || echo -1
import json,sys
try:
    xs=[json.loads(l)["idx"] for l in open(sys.argv[1]) if l.strip()]
    print(max(xs) if xs else -1)
except Exception:
    print(-1)
PYEOF
}

pkill -9 -f gen_p2_safe 2>/dev/null; sleep 3
NEXT=$(( $(maxidx) + 1 ))
while [ "$NEXT" -lt "$END" ]; do
  ASTART=$NEXT
  echo "[$(date +%T)] launching segment start=$ASTART"
  setsid nohup python -u "$HOME/gen_p2_safe.py" --adapter "$ADAPTER" --out "$OUT" \
     --max-tokens 2048 --max-time 90 --start "$ASTART" --end "$END" \
     > "$HOME/model-build/gen_seg.log" 2>&1 < /dev/null &
  sleep 8
  last=$(wc -l < "$OUT" 2>/dev/null || echo 0); stall=0
  while pgrep -f gen_p2_safe >/dev/null 2>&1; do
    sleep 15
    n=$(wc -l < "$OUT" 2>/dev/null || echo 0)
    if [ "$n" = "$last" ]; then stall=$((stall+1)); else stall=0; last=$n; fi
    if [ "$stall" -ge 11 ]; then
      echo "[$(date +%T)] STALL (~165s no write) at $n lines — killing segment"
      pkill -9 -f gen_p2_safe; sleep 3
      break
    fi
  done
  NEWMAX=$(maxidx)
  if [ "$NEWMAX" -ge "$ASTART" ]; then
    NEXT=$((NEWMAX+1))
  else
    echo "[$(date +%T)] idx $ASTART stalled immediately — skipping it"
    NEXT=$((ASTART+1))
  fi
done
echo "[$(date +%T)] RESUME_DRIVER_DONE maxidx=$(maxidx) lines=$(wc -l < $OUT)"
