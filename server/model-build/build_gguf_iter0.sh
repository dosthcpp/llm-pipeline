#!/usr/bin/env bash
# TON-2816 S3 — build GGUF for RAFT iter0 adapter. CPU/disk ONLY (no llama.service stop).
# merge raft-v2-8b-iter0-adapter -> Qwen3-8B, convert f16, quantize Q4_K_M.
set -uo pipefail
cd ~/model-build
PY=~/sovereign-model-env/bin/python
ADAPTER=~/model-build/out/raft-v2-8b-iter0-adapter
MERGED=~/model-build/out/merged-raft-8b-iter0
F16=~/models/raft-v2-8b-iter0-f16.gguf
Q4=~/models/raft-v2-8b-iter0-Q4_K_M.gguf
echo "[$(date +%T)] [1/3] merge LoRA -> base Qwen3-8B (CPU, bf16)"
$PY - <<PYEOF
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map="cpu")
tok=AutoTokenizer.from_pretrained("$ADAPTER")
m=PeftModel.from_pretrained(base, "$ADAPTER")
m=m.merge_and_unload()
m.save_pretrained("$MERGED", safe_serialization=True)
tok.save_pretrained("$MERGED")
print("merge done")
PYEOF
echo "[$(date +%T)] [2/3] convert HF -> GGUF f16"
$PY ~/llama.cpp/convert_hf_to_gguf.py "$MERGED" --outfile "$F16" --outtype f16
echo "[$(date +%T)] [3/3] quantize -> Q4_K_M"
~/llama.cpp/build/bin/llama-quantize "$F16" "$Q4" Q4_K_M
ls -lh "$F16" "$Q4"
echo "[$(date +%T)] ALL_DONE_BUILD_ITER0"
