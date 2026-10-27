#!/usr/bin/env bash
# TON-2893 — build GGUF for neg-signal DPO adapter. CPU/disk only.
set -uo pipefail
cd ~/model-build
PY=~/sovereign-model-env/bin/python
ADAPTER=~/model-build/out/neg-v2-8b-dpo-adapter/policy
MERGED=~/model-build/out/merged-neg-v2-8b-dpo
F16=~/models/neg-v2-8b-dpo-f16.gguf
Q4=~/models/neg-v2-8b-dpo-Q4_K_M.gguf
LOG=~/model-build/build_gguf_dpo_neg.log
exec > >(tee -a "$LOG") 2>&1
echo "[$(date +%T)] [1/3] merge DPO policy LoRA -> base Qwen3-8B (CPU, bf16)"
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
echo "[$(date +%T)] ALL_DONE_BUILD_DPO_NEG"
