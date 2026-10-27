#!/usr/bin/env bash
# TON-2816 — Build RAFT iter0 GGUF (merge raft-v2-8b-iter0-adapter into Qwen3-8B, Q4_K_M)
# then run the FROZEN harness generation (eval_v2_gen.py, identical params to TON-2813).
# Phases 1-3 pure CPU (35B stays up). Phase 4 pauses watchdog, frees GPU, serves iter0
# GGUF on :8021, runs eval gen, ALWAYS restores 35B via trap (raft-style pause+restore).
set -uo pipefail
cd ~/model-build
PY=~/sovereign-model-env/bin/python
ADAPTER=~/model-build/out/raft-v2-8b-iter0-adapter
MERGED=~/model-build/out/merged-raft-8b-iter0
OUT=~/models
F16="$OUT/sovereign-raft-8b-iter0-f16.gguf"
Q4="$OUT/sovereign-raft-8b-iter0-Q4_K_M.gguf"
GENOUT=~/model-build/gen_raft_8b_iter0.jsonl
SRVLOG=~/sov_gguf_raft_iter0_server.log
RESLOG=~/model-build/build_raft_iter0.log
PORT=8021
PAUSE_FILE="$HOME/watchdog.pause"
PAUSE_SECS=10800
mkdir -p "$OUT"
exec > >(tee -a "$RESLOG") 2>&1
echo "[$(date +%T)] ===== TON-2816 raft-iter0 GGUF build + frozen-harness gen ====="

if [ ! -f "$Q4" ]; then
  echo "[$(date +%T)] [1/4] merge LoRA -> base Qwen3-8B (CPU, bf16)"
  $PY - <<PYEOF
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map="cpu")
tok=AutoTokenizer.from_pretrained("/home/tony/model-build/out/raft-v2-8b-iter0-adapter")
m=PeftModel.from_pretrained(base, "/home/tony/model-build/out/raft-v2-8b-iter0-adapter")
m=m.merge_and_unload()
m.save_pretrained("/home/tony/model-build/out/merged-raft-8b-iter0", safe_serialization=True)
tok.save_pretrained("/home/tony/model-build/out/merged-raft-8b-iter0")
print("merge done")
PYEOF
  [ $? -ne 0 ] && { echo "[$(date +%T)] MERGE FAILED"; exit 1; }
  echo "[$(date +%T)] [2/4] convert HF -> GGUF f16"
  $PY ~/llama.cpp/convert_hf_to_gguf.py "$MERGED" --outfile "$F16" --outtype f16 || exit 1
  echo "[$(date +%T)] [3/4] quantize -> Q4_K_M"
  ~/llama.cpp/build/bin/llama-quantize "$F16" "$Q4" Q4_K_M || exit 1
else
  echo "[$(date +%T)] [1-3/4] Q4 already present, skipping build"
fi
ls -lh "$OUT"/sovereign-raft-8b-iter0-*.gguf

echo "[$(date +%T)] [4/4] === frozen-harness gen: pause watchdog, free GPU, serve iter0 GGUF ==="
restore_llama() {
  echo "[$(date +%T)] cleanup: kill v2 server :$PORT, restart 35B teacher ..."
  pkill -f "llama-server.*$PORT" 2>/dev/null || true
  sleep 3
  bash "$HOME/start-llama.sh"
  for i in $(seq 1 40); do
    sleep 6
    curl -s -m 5 http://127.0.0.1:8020/health 2>/dev/null | grep -q ok && { rm -f "$PAUSE_FILE"; echo "[$(date +%T)] llama(35B) HEALTHY — watchdog resumed"; return; }
  done
  rm -f "$PAUSE_FILE"
  echo "[$(date +%T)] WARNING: 35B not healthy — watchdog resumed to recover"
}
trap restore_llama EXIT

echo $(( $(date +%s) + PAUSE_SECS )) > "$PAUSE_FILE"
echo "[$(date +%T)] watchdog paused ${PAUSE_SECS}s; stopping llama-server"
for p in $(pgrep llama-server); do kill "$p" 2>/dev/null; done
for i in $(seq 1 20); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ "${free:-0}" -ge 8000 ] && break
  for p in $(pgrep llama-server); do kill -9 "$p" 2>/dev/null; done
  sleep 3
done
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

echo "[$(date +%T)] launch llama-server iter0 GGUF (-ngl 99 -fa) on :$PORT"
~/llama.cpp/build/bin/llama-server -m "$Q4" --host 127.0.0.1 --port "$PORT" \
    -ngl 99 -fa on -c 8192 --no-webui > "$SRVLOG" 2>&1 &
SRVPID=$!
ok=0
for i in $(seq 1 40); do
  curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q "\"status\":\"ok\"" && { ok=1; echo "[$(date +%T)] iter0 server HEALTHY"; break; }
  sleep 3
done
[ "$ok" -ne 1 ] && { echo "[$(date +%T)] iter0 server FAILED to start"; exit 1; }

echo "[$(date +%T)] run eval_v2_gen --mode teacher (frozen harness params: done-cap 30 blocked-cap 30 max-tokens 4096)"
$PY eval_v2_gen.py --mode teacher \
    --url "http://127.0.0.1:$PORT/v1/chat/completions" \
    --teacher-model "sovereign-raft-8b-iter0-Q4_K_M.gguf" \
    --max-tokens 4096 --done-cap 30 --blocked-cap 30 \
    --out "$GENOUT" 2>&1
echo "[$(date +%T)] gen rows: $(wc -l < $GENOUT 2>/dev/null)"
echo "[$(date +%T)] ALL_DONE_TON2816_ITER0_GEN"
