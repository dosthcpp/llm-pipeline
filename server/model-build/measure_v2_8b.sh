#!/usr/bin/env bash
# TON-2786 — MEASURE phase only (v2-8B Q4_K_M GGUF already built).
# Pause watchdog, free GPU, serve v2-8B on llama.cpp, run eval gen, read print_timing.
# trap ALWAYS unpauses watchdog + restores the 35B teacher (llama.service :8020).
set -uo pipefail
cd ~/model-build
PY=~/sovereign-model-env/bin/python
Q4=~/models/sovereign-v2-8b-Q4_K_M.gguf
GENOUT=~/model-build/gen_v2_8b_gguf_throughput.jsonl
SRVLOG=~/sov_gguf_v2_8b_server.log
RESLOG=~/model-build/throughput_v2_8b_measure.log
PORT=8021
PAUSE_FILE=~/watchdog.pause
: > "$SRVLOG"
exec > >(tee "$RESLOG") 2>&1
echo "[$(date +%T)] ===== TON-2786 v2-8B GGUF MEASURE ====="
ls -lh "$Q4" || { echo "Q4 MISSING — abort"; exit 1; }

cleanup() {
  echo "[$(date +%T)] cleanup: stop v2 server, unpause watchdog, restore teacher"
  [ -n "${SRVPID:-}" ] && kill "$SRVPID" 2>/dev/null || true
  pkill -f "llama-server.*--port $PORT" 2>/dev/null || true
  sleep 3
  rm -f "$PAUSE_FILE"
  sudo -n /usr/bin/systemctl start llama.service
  sleep 8
  for i in $(seq 1 25); do
    curl -s -m5 http://127.0.0.1:8020/health 2>/dev/null | grep -q ok && { echo "[$(date +%T)] teacher(35B :8020) HEALTHY"; return; }
    sleep 6
  done
  echo "[$(date +%T)] WARNING: teacher not healthy — CHECK MANUALLY"
}
trap cleanup EXIT

echo "[$(date +%T)] pause watchdog 30m + stop llama.service (free GPU)"
echo $(( $(date +%s) + 1800 )) > "$PAUSE_FILE"
sudo -n /usr/bin/systemctl stop llama.service
for i in $(seq 1 30); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  echo "[$(date +%T)] gpu free=${free} MiB"
  [ "${free:-0}" -ge 18000 ] && { echo "[$(date +%T)] GPU freed"; break; }
  sleep 4
done

echo "[$(date +%T)] launch llama-server v2-8B (-fa on -ngl 99 -c 8192) :$PORT"
~/llama.cpp/build/bin/llama-server -m "$Q4" --host 127.0.0.1 --port "$PORT" \
    -ngl 99 -fa on -c 8192 > "$SRVLOG" 2>&1 &
SRVPID=$!
HEALTHY=0
for i in $(seq 1 50); do
  curl -s -m3 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q '"status":"ok"' && { echo "[$(date +%T)] v2 server HEALTHY"; HEALTHY=1; break; }
  kill -0 "$SRVPID" 2>/dev/null || { echo "[$(date +%T)] v2 server DIED — see $SRVLOG"; break; }
  sleep 3
done
echo "[$(date +%T)] --- offload check ---"
grep -aiE "offload|layers to GPU|CUDA0 model buffer|load_tensors|n_ctx" "$SRVLOG" | head -15
[ "$HEALTHY" = 1 ] || { echo "[$(date +%T)] server not healthy — abort measure"; exit 1; }

echo "[$(date +%T)] eval_v2_gen --mode teacher (cap 30/30 ~60 decisions, max-tokens 2048)"
$PY eval_v2_gen.py --mode teacher \
   --url "http://127.0.0.1:$PORT/v1/chat/completions" \
   --teacher-model "sovereign-v2-8b-Q4_K_M.gguf" \
   --max-tokens 2048 --done-cap 30 --blocked-cap 30 \
   --out "$GENOUT" 2>&1

echo "[$(date +%T)] === print_timing: per-request gen rate (eval time lines) ==="
grep -aE "eval time" "$SRVLOG" | tail -40
echo "[$(date +%T)] === aggregate gen tok/s (mean of 'eval time' generation lines) ==="
grep -aE "^.*eval time =.*tokens per second" "$SRVLOG" | grep -aiv "prompt" | \
  sed -E 's/.*\(([0-9.]+) ms per token, *([0-9.]+) tokens per second.*/\2/' | \
  $PY -c "import sys; v=[float(x) for x in sys.stdin if x.strip()]; print(f'n={len(v)} mean_gen_tok_s={sum(v)/len(v):.2f} min={min(v):.2f} max={max(v):.2f}') if v else print('NO TIMING LINES PARSED')"
echo "[$(date +%T)] gen rows: $(wc -l < $GENOUT 2>/dev/null)"
echo "[$(date +%T)] ALL_DONE_TON2786"
