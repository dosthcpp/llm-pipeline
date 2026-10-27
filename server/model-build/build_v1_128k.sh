#!/usr/bin/env bash
# TON-2410 — rebuild v1 GGUF with YaRN baked into metadata for 128K context.
# Native train 40960 -> factor 3.2 -> 131072 (within Qwen3 officially-supported 4x YaRN).
# Writes NEW filenames so the live 8020 server (old mmap) is undisturbed.
set -euo pipefail
cd ~/model-build
PY=~/sovereign-model-env/bin/python
MERGED=~/model-build/out/merged-v1
OUT=~/models
LOG=~/model-build/build_v1_128k.log
echo "=== build_v1_128k start $(date -u +%H:%M:%S) ===" | tee "$LOG"
echo "[1/3] patch config.json with YaRN rope_scaling + max_position 131072" | tee -a "$LOG"
$PY - <<PY 2>>"$LOG"
import json
p="$MERGED/config.json"
c=json.load(open(p))
c["max_position_embeddings"]=131072
c["rope_scaling"]={"rope_type":"yarn","factor":3.2,"original_max_position_embeddings":40960}
json.dump(c, open(p,"w"), indent=2)
print("patched:", c["rope_scaling"], "max_pos", c["max_position_embeddings"])
PY
echo "[2/3] convert HF -> GGUF f16 (128k)" | tee -a "$LOG"
$PY ~/llama.cpp/convert_hf_to_gguf.py "$MERGED" \
    --outfile "$OUT/sovereign-v1-128k-f16.gguf" --outtype f16 >>"$LOG" 2>&1
echo "[3/3] quantize -> Q4_K_M (128k)" | tee -a "$LOG"
~/llama.cpp/build/bin/llama-quantize "$OUT/sovereign-v1-128k-f16.gguf" "$OUT/sovereign-v1-128k-Q4_K_M.gguf" Q4_K_M >>"$LOG" 2>&1
ls -la "$OUT"/sovereign-v1-128k-*.gguf | tee -a "$LOG"
echo "BUILD_128K_DONE $(date -u +%H:%M:%S)" | tee -a "$LOG"
