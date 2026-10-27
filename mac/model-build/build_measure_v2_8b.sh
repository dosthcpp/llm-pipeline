#!/usr/bin/env bash
# TON-2786 — Build v2-8B GGUF (merge sovereign-v2-8b-adapter into Qwen3-8B, Q4_K_M)
# then MEASURE production llama.cpp throughput. Converts cost_model estimate -> measured.
# Phases 1-3 are pure CPU/disk (non-disruptive). Phase 4 needs the GPU: it stops
# llama.service (35B teacher :8020), serves the v2-8B GGUF on :8021, runs eval gen,
# then ALWAYS restores llama.service via trap.
set -uo pipefail
cd ~/model-build
PY=~/sovereign-model-env/bin/python
ADAPTER=~/model-build/out/sovereign-v2-8b-adapter
MERGED=~/model-build/out/merged-v2-8b
OUT=~/models
F16="$OUT/sovereign-v2-8b-f16.gguf"
Q4="$OUT/sovereign-v2-8b-Q4_K_M.gguf"
GENOUT=~/model-build/gen_v2_8b_gguf_throughput.jsonl
SRVLOG=~/sov_gguf_v2_8b_server.log
RESLOG=~/model-build/throughput_v2_8b.log
PORT=8021
mkdir -p "$OUT"
exec > >(tee -a "$RESLOG") 2>&1
echo "[$(date +%T)] ===== TON-2786 v2-8B GGUF build + throughput ====="

if [ ! -f "$Q4" ]; then
  echo "[$(date +%T)] [1/4] merge LoRA -> base Qwen3-8B (CPU, bf16)"
  $PY - <<PY
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base_id="Qwen/Qwen3-8B"
print("loading base on CPU...")
base=AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16, device_map="cpu")
tok=AutoTokenizer.from_pretrained("$ADAPTER")
print("attaching adapter + merge...")
m=PeftModel.from_pretrained(base, "$ADAPTER")
m=m.merge_and_unload()
print("saving merged ->", "$MERGED")
m.save_pretrained("$MERGED", safe_serialization=True)
tok.save_pretrained("$MERGED")
print("merge done")
PY
  echo "[$(date +%T)] [2/4] convert HF -> GGUF f16"
  $PY ~/llama.cpp/convert_hf_to_gguf.py "$MERGED" --outfile "$F16" --outtype f16
  echo "[$(date +%T)] [3/4] quantize -> Q4_K_M"
  ~/llama.cpp/build/bin/llama-quantize "$F16" "$Q4" Q4_K_M
else
  echo "[$(date +%T)] [1-3/4] Q4 GGUF already present, skipping build"
fi
ls -lh "$OUT"/sovereign-v2-8b-*.gguf

echo "[$(date +%T)] [4/4] === MEASURE: free GPU, serve v2-8B GGUF, run eval gen ==="
restore_llama() {
  echo "[$(date +%T)] cleanup: kill v2 server, restart llama.service ..."
  [ -n "${SRVPID:-}" ] && kill "$SRVPID" 2>/dev/null || true
  pkill -f "llama-server.*$PORT" 2>/dev/null || true
  sleep 3
  sudo -n /usr/bin/systemctl start llama.service
  sleep 8
  for i in $(seq 1 20); do
    curl -s -m 5 http://127.0.0.1:8020/health 2>/dev/null | grep -q ok && { echo "[$(date +%T)] llama(35B teacher) HEALTHY"; return; }
    sleep 6
  done
  echo "[$(date +%T)] WARNING: llama.service not healthy — CHECK MANUALLY"
}
trap restore_llama EXIT

echo "[$(date +%T)] stop llama.service (free GPU)"
sudo -n /usr/bin/systemctl stop llama.service
sleep 6
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

echo "[$(date +%T)] launch llama-server v2-8B (flash-attn, -ngl 99) on :$PORT"
~/llama.cpp/build/bin/llama-server -m "$Q4" --host 127.0.0.1 --port "$PORT" \
    -ngl 99 -fa -c 8192 --no-webui > "$SRVLOG" 2>&1 &
SRVPID=$!
for i in $(seq 1 40); do
  curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q '"status":"ok"' && { echo "[$(date +%T)] v2-8B server HEALTHY"; break; }
  sleep 3
done

echo "[$(date +%T)] run eval_v2_gen --mode teacher (N=full eval.jsonl, max-tokens 2048)"
$PY eval_v2_gen.py --mode teacher \
    --url "http://127.0.0.1:$PORT/v1/chat/completions" \
    --teacher-model "sovereign-v2-8b-Q4_K_M.gguf" \
    --max-tokens 2048 --done-cap 30 --blocked-cap 30 \
    --out "$GENOUT" 2>&1

echo "[$(date +%T)] === llama-server print_timing (gen tok/s source) ==="
grep -E "eval time|tokens per second|prompt eval|generation eval|n_eval" "$SRVLOG" | tail -30
echo "[$(date +%T)] gen rows: $(wc -l < $GENOUT 2>/dev/null)"
echo "[$(date +%T)] ALL_DONE_TON2786"
