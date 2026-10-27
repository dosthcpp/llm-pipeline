#!/usr/bin/env bash
# TON-2586 — resumable gen finisher: gen all 116 -> teardown server -> restore prod.
set -uo pipefail
MB="$HOME/model-build"
PAUSE="$HOME/watchdog.pause"
LOG="$MB/out/fable5_gen2.log"
OUT="$MB/gen_qwen27b_fable5.jsonl"
mkdir -p "$MB/out"
source "$HOME/sovereign-model-env/bin/activate" 2>/dev/null || true
echo "[start $(date -u +%FT%TZ)]" >> "$LOG"
python "$MB/gen_fable5_resumable.py" --out "$OUT" >> "$LOG" 2>&1
rc=$?
N=$(grep -c . "$OUT" 2>/dev/null || echo 0)
echo "GEN2_DONE rc=$rc n=$N $(date -u +%FT%TZ)" >> "$LOG"
# teardown: free GPU + restore prod teacher
pkill -f "llama-server.*--port 8030" 2>/dev/null
rm -f "$PAUSE"
sleep 20
if curl -s -m 8 http://127.0.0.1:8020/health 2>/dev/null | grep -qi ok; then
  echo "prod teacher :8020 restored" >> "$LOG"
else
  echo "prod :8020 not yet up (watchdog will resurrect within interval)" >> "$LOG"
fi
