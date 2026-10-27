#!/usr/bin/env bash
set -u
source /home/tony/sovereign-model-env/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DOWNLOAD_TIMEOUT=60
M=/home/tony/models/Qwopus3.6-27B-Coder
L=/home/tony/model-build/out/dl_loop.log
say(){ echo "[$(date -u +%FT%TZ)] $*" >> "$L"; }
say "dl_loop start ($(ls $M/model-*-of-00015.safetensors 2>/dev/null|wc -l)/15)"
for i in $(seq 1 60); do
  n=$(ls $M/model-*-of-00015.safetensors 2>/dev/null | wc -l)
  if [ "$n" -ge 15 ]; then say "COMPLETE $n/15"; echo done > /home/tony/model-build/out/dl_loop.DONE; break; fi
  say "attempt $i: have $n/15, downloading..."
  hf download Jackrong/Qwopus3.6-27B-Coder --local-dir "$M" >> "$L" 2>&1
  say "attempt $i exit rc=$?"
  sleep 5
done
