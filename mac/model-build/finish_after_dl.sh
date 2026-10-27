#!/usr/bin/env bash
# TON-2542 — waits for dl_loop.DONE, then runs the CEO-approved GPU window:
# free 3090 (user level) -> certified nf4 gen -> restore production teacher.
# Robust: runs entirely on Atlas, detached. Writes final sentinel for the judge step.
set -u
cd "$HOME/model-build"
source /home/tony/sovereign-model-env/bin/activate
LOG=/home/tony/model-build/out/gpu_window.log
SENT=/home/tony/model-build/out/gpu_window.DONE
DLDONE=/home/tony/model-build/out/dl_loop.DONE
MODEL_DIR="$HOME/models/Qwopus3.6-27B-Coder"
rm -f "$SENT"
say(){ echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }

say "=== finish_after_dl: waiting for download ==="
for i in $(seq 1 240); do          # up to 120 min
  [ -f "$DLDONE" ] && { say "download DONE detected"; break; }
  sleep 30
done
n=$(ls "$MODEL_DIR"/model-*-of-00015.safetensors 2>/dev/null | wc -l)
if [ "$n" -lt 15 ] || [ ! -f "$MODEL_DIR/config.json" ]; then
  say "ABORT: shards $n/15 incomplete"; echo "FAIL download $n/15" > "$SENT"; exit 2
fi
say "shards complete ($n/15)"

say "freeing GPU (user level: watchdog + llama-server)"
pkill -f watchdog.sh 2>/dev/null
tmux kill-session -t llama-server 2>/dev/null
pkill -f "llama.cpp/build/bin/llama-server" 2>/dev/null
for i in $(seq 1 24); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  say "free VRAM ${free} MiB"; [ "${free:-0}" -ge 18000 ] && break; sleep 5
done

say "=== gen start (nf4, MT=1536, caps 10/10, limit 40) ==="
T=$(date +%s)
python eval_v2_gen.py --mode hf --base-model "$MODEL_DIR" --adapter "" \
  --max-tokens 1536 --done-cap 10 --blocked-cap 10 --limit 40 \
  --out gen_qwopus27b.jsonl >> "$LOG" 2>&1
GENRC=$?
say "gen rc=$GENRC elapsed=$(( ($(date +%s)-T)/60 ))min"

say "restoring production teacher"
bash ~/start-llama.sh >> "$LOG" 2>&1
for i in $(seq 1 30); do curl -s -m 5 http://127.0.0.1:8020/health 2>/dev/null | grep -q ok && { say "teacher HEALTHY"; break; }; sleep 6; done
HEALTH=$(curl -s -m 5 http://127.0.0.1:8020/health 2>/dev/null | grep -o ok || echo DOWN)
G=$(python -c "import json;print(len(json.load(open('gen_qwopus27b.jsonl'))))" 2>/dev/null || echo 0)
say "FINISH gen_rc=$GENRC gens=$G teacher=$HEALTH"
echo "gen_rc=$GENRC gens=$G teacher=$HEALTH" > "$SENT"
