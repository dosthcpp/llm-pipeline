#!/usr/bin/env bash
# TON-2410 — merge sovereign-v1 LoRA into Qwen3-4B, convert to GGUF, quantize Q4_K_M.
# Non-disruptive: pure CPU/disk work; does NOT touch the production GPU llama-server (8020).
set -euo pipefail
cd ~/model-build
PY=~/sovereign-model-env/bin/python
ADAPTER=~/model-build/out/sovereign-v1-adapter
MERGED=~/model-build/out/merged-v1
OUT=~/models
mkdir -p "$OUT"
echo "[1/3] merge LoRA -> base (CPU, bf16)"
$PY - <<PY
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base_id="Qwen/Qwen3-4B"
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
echo "[2/3] convert HF -> GGUF f16"
$PY ~/llama.cpp/convert_hf_to_gguf.py "$MERGED" \
    --outfile "$OUT/sovereign-v1-f16.gguf" --outtype f16
echo "[3/3] quantize -> Q4_K_M"
~/llama.cpp/build/bin/llama-quantize "$OUT/sovereign-v1-f16.gguf" "$OUT/sovereign-v1-Q4_K_M.gguf" Q4_K_M
ls -la "$OUT"/sovereign-v1-*.gguf
echo "ALL_DONE"
