#!/usr/bin/env bash
# TON-2410 — serve sovereign-v1 on Atlas (CPU, port 8022, non-disruptive).
# Production Qwopus-35B stays on the GPU (8020) untouched.
set -euo pipefail
MODEL="${MODEL:-$HOME/models/sovereign-v1-Q4_K_M.gguf}"
PORT="${PORT:-8022}"
NGL="${NGL:-0}"
exec ~/llama.cpp/build/bin/llama-server --model "$MODEL" \
  --ctx-size 4096 --parallel 1 --jinja -ngl "$NGL" \
  --temp 0.6 --top-p 0.95 --top-k 20 --port "$PORT" --host 0.0.0.0
