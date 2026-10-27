#!/bin/bash
# TON-2747: fetch stock (순정) Qwen3.6-27B Q4_K_M GGUF for the demo-gen MoA local proposer.
cd /home/tony/models || exit 1
pkill -f 'aria2c.*Qwen3.6-27B' 2>/dev/null
rm -f Qwen3.6-27B-Q4_K_M.gguf Qwen3.6-27B-Q4_K_M.gguf.aria2
URL="https://huggingface.co/bartowski/Qwen_Qwen3.6-27B-GGUF/resolve/main/Qwen_Qwen3.6-27B-Q4_K_M.gguf"
exec aria2c -x16 -s16 -k1M --summary-interval=5 --console-log-level=notice \
  -o Qwen3.6-27B-Q4_K_M.gguf "$URL"
