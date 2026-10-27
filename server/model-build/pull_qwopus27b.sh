#!/usr/bin/env bash
set -u
source /home/tony/sovereign-model-env/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=1
LOG=/home/tony/model-build/out/pull_qwopus27b.log
echo "[$(date -u +%FT%TZ)] start pull Jackrong/Qwopus3.6-27B-Coder" >> "$LOG"
hf download Jackrong/Qwopus3.6-27B-Coder --local-dir /home/tony/models/Qwopus3.6-27B-Coder >> "$LOG" 2>&1
echo "[$(date -u +%FT%TZ)] pull exit rc=$? du=$(du -sh /home/tony/models/Qwopus3.6-27B-Coder 2>/dev/null | cut -f1)" >> "$LOG"
