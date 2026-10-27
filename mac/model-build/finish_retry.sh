#!/usr/bin/env bash
# TON-2542 retry — skip-quant fix for the Qwen3_5 hybrid VL+SSM model.
# Waits for kernel pip build, frees GPU, SMOKE-tests 1 item (caps downtime if still
# broken), runs full gen only if smoke passes, ALWAYS restores the production teacher.
set -u
cd "$HOME/model-build"
source /home/tony/sovereign-model-env/bin/activate
LOG=/home/tony/model-build/out/retry.log
SENT=/home/tony/model-build/out/retry.DONE
MODEL_DIR="$HOME/models/Qwopus3.6-27B-Coder"
rm -f "$SENT"
say(){ echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }

say "=== retry start: waiting for kernel pip build ==="
for i in $(seq 1 80); do pgrep -f "pip install" >/dev/null || break; sleep 15; done
KERN=$(pip list 2>/dev/null | grep -iE "causal-conv1d|mamba-ssm|flash-linear-attention" | tr '\n' ' ')
say "kernels: ${KERN:-NONE}"

say "freeing GPU (user level)"
pkill -f watchdog.sh 2>/dev/null
tmux kill-session -t llama-server 2>/dev/null
pkill -f "llama.cpp/build/bin/llama-server" 2>/dev/null
for i in $(seq 1 24); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  say "free VRAM ${free} MiB"; [ "${free:-0}" -ge 18000 ] && break; sleep 5
done

say "=== SMOKE test (1 item) ==="
SMOKE=$(python gen_27b.py --model "$MODEL_DIR" --smoke 2>>"$LOG")
say "smoke result: $SMOKE"
if echo "$SMOKE" | grep -q "SMOKE OK"; then
  say "=== smoke PASSED -> full gen (40, MT1536) ==="
  T=$(date +%s)
  python gen_27b.py --model "$MODEL_DIR" --limit 40 --max-tokens 1536 --out gen_qwopus27b.jsonl >> "$LOG" 2>&1
  GENRC=$?
  say "full gen rc=$GENRC elapsed=$(( ($(date +%s)-T)/60 ))min"
else
  GENRC=99
  say "=== smoke FAILED -> abort gen, restore teacher (minimal downtime) ==="
fi

say "restoring production teacher"
bash ~/start-llama.sh >> "$LOG" 2>&1
for i in $(seq 1 30); do curl -s -m 5 http://127.0.0.1:8020/health 2>/dev/null | grep -q ok && { say "teacher HEALTHY"; break; }; sleep 6; done
HEALTH=$(curl -s -m 5 http://127.0.0.1:8020/health 2>/dev/null | grep -o ok || echo DOWN)
G=$(python -c "import json;print(len(json.load(open('gen_qwopus27b.jsonl'))))" 2>/dev/null || echo 0)
say "FINISH gen_rc=$GENRC gens=$G teacher=$HEALTH kernels=${KERN:-NONE}"
echo "gen_rc=$GENRC gens=$G teacher=$HEALTH" > "$SENT"
