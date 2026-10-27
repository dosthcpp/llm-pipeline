#!/usr/bin/env bash
# TON-2873 — iter1 frozen-harness generation (mirrors build_measure_raft_iter0.sh
# Phase 4 EXACTLY: same eval_v2_gen params — --mode teacher, max-tokens 4096,
# done-cap 30, blocked-cap 30, dataset/eval.jsonl). GGUF already built by
# build_gguf_iter1.sh, so build phases are skipped. Serves iter1 GGUF on :8021,
# runs gen, ALWAYS restores prod 27B via trap.
set -uo pipefail
cd ~/model-build
PY=~/sovereign-model-env/bin/python
Q4=~/models/raft-v2-8b-iter1-Q4_K_M.gguf
GENOUT=~/model-build/gen_raft_8b_iter1.jsonl
SRVLOG=~/sov_gguf_raft_iter1_server.log
PORT=8021
PAUSE_FILE="$HOME/watchdog.pause"
PAUSE_SECS=10800
[ -f "$Q4" ] || { echo "MISSING $Q4"; exit 1; }
exec > >(tee -a ~/model-build/measure_raft_iter1.log) 2>&1
echo "[$(date +%T)] ===== iter1 frozen-harness gen START ====="

restore_llama() {
  echo "[$(date +%T)] cleanup: kill :$PORT server, restart prod 27B teacher ..."
  pkill -f "llama-server.*$PORT" 2>/dev/null || true
  sleep 3
  bash "$HOME/start-llama.sh" &
  for i in $(seq 1 40); do
    sleep 6
    curl -s -m 5 http://127.0.0.1:8020/health 2>/dev/null | grep -q ok && {
      rm -f "$PAUSE_FILE"; echo "[$(date +%T)] prod 27B HEALTHY — watchdog resumed"; return; }
  done
  rm -f "$PAUSE_FILE"; echo "[$(date +%T)] WARNING: 27B not healthy — CHECK"
}
trap restore_llama EXIT

echo $(( $(date +%s) + PAUSE_SECS )) > "$PAUSE_FILE"
echo "[$(date +%T)] watchdog paused; stopping prod llama-server to free GPU"
for p in $(pgrep llama-server); do kill "$p" 2>/dev/null; done
for i in $(seq 1 20); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ "${free:-0}" -ge 8000 ] && break
  for p in $(pgrep llama-server); do kill -9 "$p" 2>/dev/null; done
  sleep 3
done
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

echo "[$(date +%T)] launch llama-server iter1 GGUF (-ngl 99 -fa on) on :$PORT"
~/llama.cpp/build/bin/llama-server -m "$Q4" --host 127.0.0.1 --port "$PORT" \
    -ngl 99 -fa on -c 8192 --no-webui > "$SRVLOG" 2>&1 &
ok=0
for i in $(seq 1 40); do
  curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q "\"status\":\"ok\"" && { ok=1; echo "[$(date +%T)] iter1 server HEALTHY"; break; }
  sleep 3
done
[ "$ok" -ne 1 ] && { echo "[$(date +%T)] iter1 server FAILED to start"; exit 1; }

echo "[$(date +%T)] run eval_v2_gen --mode teacher (frozen params: done-cap 30 blocked-cap 30 max-tokens 4096)"
$PY eval_v2_gen.py --mode teacher \
    --url "http://127.0.0.1:$PORT/v1/chat/completions" \
    --teacher-model "raft-v2-8b-iter1-Q4_K_M.gguf" \
    --max-tokens 4096 --done-cap 30 --blocked-cap 30 \
    --out "$GENOUT" 2>&1
echo "[$(date +%T)] gen rows: $(wc -l < $GENOUT 2>/dev/null)"
echo "[$(date +%T)] ALL_DONE_ITER1_GEN"
