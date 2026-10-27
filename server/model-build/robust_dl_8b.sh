#!/usr/bin/env bash
# Robust resumable Qwen3-8B download: per-attempt timeout + retry until complete.
export HF_HUB_ENABLE_HF_TRANSFER=1
HF=~/sovereign-model-env/bin/hf
DIR=~/.cache/huggingface/hub/models--Qwen--Qwen3-8B
LOG=~/model-build/dl_8b.log
: > "$LOG"
for attempt in $(seq 1 40); do
  echo "[$(date -u +%H:%M:%S)] download attempt $attempt" >> "$LOG"
  timeout 900 "$HF" download Qwen/Qwen3-8B >> "$LOG" 2>&1
  SNAP=$(ls -d $DIR/snapshots/*/ 2>/dev/null | head -1)
  NSAFE=$(ls "$SNAP"*.safetensors 2>/dev/null | wc -l)
  SZ=$(du -s "$DIR" 2>/dev/null | cut -f1)
  echo "[$(date -u +%H:%M:%S)] attempt $attempt done: safetensors=$NSAFE size_kb=$SZ" >> "$LOG"
  if [ "${NSAFE:-0}" -ge 1 ] && [ "${SZ:-0}" -ge 14000000 ]; then
    echo "[$(date -u +%H:%M:%S)] DOWNLOAD_COMPLETE" >> "$LOG"; exit 0
  fi
  sleep 5
done
echo "[$(date -u +%H:%M:%S)] DOWNLOAD_GAVE_UP after 40 attempts" >> "$LOG"; exit 1
