#!/bin/bash
set -uo pipefail
BIN=~/llama.cpp/build/bin/llama-server
MODEL=~/models/sovereign-v1-128k-Q4_K_M.gguf
LOG=~/model-build/debug_v1_run.log
PORT=8021
: > "$LOG"
run_test() {
  local label="$1"; shift
  echo "=== $label start $(date -u +%H:%M:%S) ===" | tee -a "$LOG"
  "$BIN" --model "$MODEL" --ctx-size 8192 -ngl 0 --parallel 1 --jinja \
    --temp 0.6 --top-p 0.95 --top-k 20 "$@" \
    --port $PORT --host 127.0.0.1 >> "$LOG" 2>&1 &
  local SRV=$!
  local ok=0
  for i in $(seq 1 60); do sleep 2; curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { ok=1; break; }; done
  if [ $ok -eq 1 ]; then
    python3 /tmp/cplx_test.py "$label" "$PORT" 2>&1 | tee -a "$LOG"
  else
    echo "RESULT[$label] SERVER_FAILED_TO_START" | tee -a "$LOG"
  fi
  kill -9 $SRV 2>/dev/null; sleep 3
}
run_test "A_budget2048" --reasoning-budget 2048
run_test "B_nobudget"
echo "ALL_DONE $(date -u +%H:%M:%S)" | tee -a "$LOG"
