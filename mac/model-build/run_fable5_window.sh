#!/usr/bin/env bash
# TON-2586 P0.3.3 — OSS substitution test: serve TeichAI Qwen3.6-27B-Fable-5 (Q5_K_M GGUF)
# in a self-restoring GPU window and native-gen the certified TON-2397 eval set.
#
# Safety design (production teacher is on :8020 via ~/watchdog.sh keepalive):
#   1. WAIT for the GGUF download to finish BEFORE pausing prod (no idle downtime).
#   2. Self-expiring watchdog pause (TON-2441): even if this script dies, prod auto-resumes.
#   3. trap EXIT ALWAYS: kill our server, rm pause -> watchdog resurrects the teacher.
#   4. Hard gen timeout caps the window; pause expiry is the backstop.
set -uo pipefail

MB="$HOME/model-build"
GGUF="$HOME/models/Qwen3.6-27B-Fable-5-GGUF/Qwen3.6-27B-Fable-5-Distill.q5_k_m.gguf"
EXPECTED_BYTES=20616186112   # ~19.2 GB Q5_K_M; accept >= 19.0 GB as complete
MIN_BYTES=19500000000
PORT=8030
URL="http://127.0.0.1:${PORT}/v1/chat/completions"
OUT="$MB/gen_qwen27b_fable5.jsonl"
LOG="$MB/out/fable5_window.log"
SRV_LOG="$MB/out/fable5_server.log"
PAUSE="$HOME/watchdog.pause"
WINDOW_SECS=5400          # 90 min self-expiring pause backstop
GEN_TIMEOUT=4800          # 80 min hard cap on generation
mkdir -p "$MB/out"

log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

SRV_PID=""
restore(){
  log "RESTORE: killing fable5 server (pid=${SRV_PID:-none}); removing pause -> watchdog resurrects prod"
  [ -n "$SRV_PID" ] && kill -9 "$SRV_PID" 2>/dev/null
  pkill -f "llama-server.*--port ${PORT}" 2>/dev/null
  rm -f "$PAUSE"
  # give watchdog a cycle to bring :8020 back, then report
  sleep 20
  if curl -s --max-time 8 http://127.0.0.1:8020/health 2>/dev/null | grep -qi ok; then
    log "RESTORE: prod teacher :8020 healthy again"
  else
    log "RESTORE: prod :8020 not yet healthy — watchdog will restart it within its interval"
  fi
}
trap restore EXIT

# --- 1. wait for download (GPU-free; do NOT pause prod yet) ---
log "waiting for download to reach >= ${MIN_BYTES} bytes ..."
for i in $(seq 1 720); do   # up to 2h
  sz=$(stat -c %s "$GGUF" 2>/dev/null || echo 0)
  if grep -q "done dl" "$MB/out/pull_fable5.log" 2>/dev/null && [ "$sz" -ge "$MIN_BYTES" ]; then
    log "download complete: $sz bytes"; break
  fi
  if [ "$sz" -ge "$EXPECTED_BYTES" ]; then log "download size >= expected: $sz bytes"; break; fi
  [ $((i % 12)) -eq 0 ] && log "  ... dl $sz bytes"
  sleep 10
done
sz=$(stat -c %s "$GGUF" 2>/dev/null || echo 0)
if [ "$sz" -lt "$MIN_BYTES" ]; then log "ABORT: download incomplete ($sz bytes)"; exit 1; fi

# --- 2. open self-expiring GPU window ---
exp=$(( $(date +%s) + WINDOW_SECS ))
echo "$exp" > "$PAUSE"
log "watchdog pause set until $exp (window ${WINDOW_SECS}s)"
# kill the production teacher to free the 3090 (watchdog is paused, will not resurrect)
pkill -f "llama-server.*Qwopus3.6-35B" 2>/dev/null
log "stopped prod teacher; waiting for VRAM to free ..."
for i in $(seq 1 30); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  log "  free VRAM ${free} MiB"
  [ "${free:-0}" -ge 20000 ] && break
  sleep 5
done
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "${free:-0}" -lt 20000 ]; then log "ABORT: only ${free} MiB free, need >=20000"; exit 1; fi

# --- 3. serve Fable-5 Q5_K_M ---
log "starting llama-server for Fable-5 on :${PORT}"
/home/tony/llama.cpp/build/bin/llama-server \
  --model "$GGUF" \
  --ctx-size 8192 --parallel 1 \
  --cache-type-k f16 --cache-type-v f16 \
  -ngl 99 --jinja --flash-attn off \
  --temp 0.0 \
  --port ${PORT} --host 127.0.0.1 > "$SRV_LOG" 2>&1 &
SRV_PID=$!
log "server pid $SRV_PID; waiting for ready ..."
ready=0
for i in $(seq 1 120); do
  if curl -s --max-time 5 http://127.0.0.1:${PORT}/health 2>/dev/null | grep -qi ok; then ready=1; break; fi
  if ! kill -0 "$SRV_PID" 2>/dev/null; then log "ABORT: server died during load"; tail -20 "$SRV_LOG" | tee -a "$LOG"; exit 1; fi
  sleep 5
done
[ "$ready" -ne 1 ] && { log "ABORT: server not ready in time"; tail -20 "$SRV_LOG" | tee -a "$LOG"; exit 1; }
log "server ready; arch lines:"; grep -iE "arch|n_layer|n_embd|model type|params" "$SRV_LOG" | head -8 | tee -a "$LOG"

# --- 4. native generation over the certified eval subset ---
source /home/tony/sovereign-model-env/bin/activate 2>/dev/null || true
log "generating -> $OUT (timeout ${GEN_TIMEOUT}s)"
timeout ${GEN_TIMEOUT} python "$MB/eval_v2_gen.py" \
  --mode teacher \
  --url "$URL" \
  --teacher-model "fable5" \
  --max-tokens 2048 \
  --out "$OUT" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
log "generation rc=$rc"
if [ "$rc" -eq 0 ] && [ -f "$OUT" ]; then
  n=$(python -c "import json;print(len(json.load(open('$OUT'))))" 2>/dev/null || echo '?')
  log "DONE: wrote $OUT ($n items)"
  echo "FABLE5_GEN_DONE $n" >> "$MB/out/fable5_window.log"
else
  log "FAIL: generation rc=$rc (out may be partial)"
fi
# trap restores prod
