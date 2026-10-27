#!/usr/bin/env bash
set -uo pipefail
cd ~/model-build
PORT=8030
PY=~/sovereign-model-env/bin/python
LOG=~/gen_v2_8b_full.log
cleanup(){ echo "[$(date +%T)] cleanup: stop 8B :$PORT server"; pkill -f "llama-server.*--port $PORT" 2>/dev/null || true; }
trap cleanup EXIT
echo "[$(date +%T)] === FULL 116 gen (8B CPU :$PORT, --done-cap 30 --blocked-cap 30) ==="
# ensure server up
if ! curl -s -m3 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q '"status":"ok"'; then
  nohup ~/llama.cpp/build/bin/llama-server -m ~/models/sovereign-v2-8b-Q4_K_M.gguf --host 127.0.0.1 --port $PORT -ngl 0 -t 8 -c 8192 --alias sovereign-v2-8b-Q4_K_M.gguf > ~/sov_8b_cpu_$PORT.log 2>&1 &
  for i in $(seq 1 40); do curl -s -m3 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q '"status":"ok"' && break; sleep 3; done
fi
$PY eval_v2_gen.py --mode teacher \
   --url "http://127.0.0.1:$PORT/v1/chat/completions" \
   --teacher-model "sovereign-v2-8b-Q4_K_M.gguf" \
   --done-cap 30 --blocked-cap 30 --max-tokens 2048 \
   --out ~/model-build/gen_v2_8b.jsonl
RC=$?
echo "[$(date +%T)] gen exit=$RC ; rows=$(python3 -c "import json;print(len(json.load(open(\"$HOME/model-build/gen_v2_8b.jsonl\"))))" 2>/dev/null || grep -c . ~/model-build/gen_v2_8b.jsonl)"
echo "[$(date +%T)] 35B health: $(curl -s -m5 http://127.0.0.1:8020/health 2>/dev/null | head -c 40)"
echo "[$(date +%T)] === DONE FULL 8B GEN ==="
