#!/usr/bin/env bash
set -euo pipefail
DEST=~/models/Qwen3.6-27B-Fable-5-GGUF
mkdir -p "$DEST"
cd "$DEST"
URL="https://huggingface.co/TeichAI/Qwen3.6-27B-Fable-5-Experimental-GGUF/resolve/main/Qwen3.6-27B-Fable-5-Distill.q5_k_m.gguf?download=true"
echo "[$(date)] start dl q5_k_m" >> ~/model-build/out/pull_fable5.log
# resumeable download
for i in $(seq 1 20); do
  curl -L -C - -o Qwen3.6-27B-Fable-5-Distill.q5_k_m.gguf "$URL" && break
  echo "[$(date)] retry $i" >> ~/model-build/out/pull_fable5.log
  sleep 5
done
echo "[$(date)] done dl: $(ls -la Qwen3.6-27B-Fable-5-Distill.q5_k_m.gguf)" >> ~/model-build/out/pull_fable5.log
