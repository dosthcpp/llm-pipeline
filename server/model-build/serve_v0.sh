#!/usr/bin/env bash
# TON-2396 Phase 2 — serve sovereign-v0 on Atlas (CPU, separate port, non-disruptive).
# Production Qwopus-35B stays on the GPU untouched. Default CPU (-ngl 0); set NGL>0 only if GPU has room.
set -euo pipefail
MODEL="${MODEL:-$HOME/models/sovereign-v0-Q4_K_M.gguf}"
PORT="${PORT:-8021}"
NGL="${NGL:-0}"
exec ~/llama.cpp/build/bin/llama-server --model "$MODEL" \
  --ctx-size 4096 --parallel 1 --jinja -ngl "$NGL" \
  --temp 0.6 --top-p 0.95 --top-k 20 --port "$PORT" --host 0.0.0.0
