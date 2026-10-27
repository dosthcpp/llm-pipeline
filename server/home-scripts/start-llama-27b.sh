#!/bin/bash
# TON-2747 (2026-07-01) — boss "로컬 모델 qwen 3.6 27B (순정)으로 바꿀래".
#   Swap the demo-gen MoA local proposer from Qwen3.6-35B-A3B UD quant to the
#   STOCK (순정) dense Qwen3.6-27B, bartowski straight Q4_K_M of Qwen's official
#   weights (no distill, no Unsloth-Dynamic). Prior 35B launcher backed up as
#   start-llama.sh.bak-35b-*. Dense 27B ~16GB on the 24GB 3090.
/home/tony/llama.cpp/build/bin/llama-server \
  --model /home/tony/models/Qwen3.6-27B-Q4_K_M.gguf \
  --ctx-size 131072 \
  --parallel 1 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --flash-attn on \
  -ngl 99 \
  --jinja \
  --temp 0.7 --top-p 0.95 --top-k 40 --repeat-penalty 1.1 \
  --reasoning-budget 8192 \
  --batch-size 1024 --ubatch-size 1024 \
  --cache-ram 16384 \
  --alias Qwen3.6-27B-Q4_K_M.gguf \
  --port 8020 --host 0.0.0.0 \
  >> /home/tony/llama-server.log 2>&1 &
