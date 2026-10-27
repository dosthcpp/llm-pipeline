#!/usr/bin/env bash
# TON-2542 v2 — self-healing GPU window. Resumes a stalled download, then frees the
# 3090 (user level), runs certified nf4 gen, restores the production teacher.
# CEO-approved (approval 0777eb81). No sudo.
set -u
cd "$HOME/model-build"
source /home/tony/sovereign-model-env/bin/activate
LOG=/home/tony/model-build/out/orchestrate_qwopus27b.log
SENT=/home/tony/model-build/out/orchestrate_qwopus27b.DONE
MODEL_DIR="$HOME/models/Qwopus3.6-27B-Coder"
rm -f "$SENT"
say(){ echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }
shards(){ ls "$MODEL_DIR"/model-*-of-00015.safetensors 2>/dev/null | wc -l; }

say "=== orchestrator v2 start (self-healing download) ==="
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT=60
for i in $(seq 1 360); do          # up to 180 min
  n=$(shards)
  if [ "$n" -ge 15 ] && [ -f "$MODEL_DIR/config.json" ]; then say "download complete ($n/15)"; break; fi
  if ! pgrep -f "hf download" >/dev/null; then
    say "download not running (have $n/15) -> (re)launch resumable"
    nohup hf download Jackrong/Qwopus3.6-27B-Coder --local-dir "$MODEL_DIR" >> "$LOG" 2>&1 &
  fi
  sleep 30
done
n=$(shards)
if [ "$n" -lt 15 ]; then say "ABORT: download incomplete ($n/15)"; echo "FAIL download $n/15" > "$SENT"; exit 2; fi

say "freeing GPU"
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

say "restoring teacher"
bash ~/start-llama.sh >> "$LOG" 2>&1
for i in $(seq 1 30); do curl -s -m 5 http://127.0.0.1:8020/health 2>/dev/null | grep -q ok && { say "teacher HEALTHY"; break; }; sleep 6; done
HEALTH=$(curl -s -m 5 http://127.0.0.1:8020/health 2>/dev/null | grep -o ok || echo DOWN)
G=$(python -c "import json;print(len(json.load(open(\"gen_qwopus27b.jsonl\")))) " 2>/dev/null || echo 0)
say "FINISH gen_rc=$GENRC gens=$G teacher=$HEALTH"
echo "gen_rc=$GENRC gens=$G teacher=$HEALTH" > "$SENT"
