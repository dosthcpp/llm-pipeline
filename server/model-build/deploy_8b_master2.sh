#!/usr/bin/env bash
# TON-2410 opt3 master v2 — wait for robust download COMPLETE marker, then train.
set -uo pipefail
cd ~/model-build
LOG=~/model-build/deploy_8b_master.log
DIR=~/.cache/huggingface/hub/models--Qwen--Qwen3-8B
echo "[$(date -u +%H:%M:%S)] master2 start — waiting for DOWNLOAD_COMPLETE" | tee "$LOG"
for i in $(seq 1 360); do
  grep -q DOWNLOAD_COMPLETE ~/model-build/dl_8b.log 2>/dev/null && break
  grep -q DOWNLOAD_GAVE_UP ~/model-build/dl_8b.log 2>/dev/null && { echo "[$(date -u +%H:%M:%S)] ABORT: download gave up" | tee -a "$LOG"; exit 1; }
  sleep 15
done
SZ=$(du -s "$DIR" 2>/dev/null | cut -f1)
if [ "${SZ:-0}" -lt 14000000 ]; then echo "[$(date -u +%H:%M:%S)] ABORT: size ${SZ}kb <14GB" | tee -a "$LOG"; exit 1; fi
echo "[$(date -u +%H:%M:%S)] download OK (${SZ}kb). === LAUNCH run_sft_8b.sh — 35B BLACKOUT BEGINS ===" | tee -a "$LOG"
bash ~/model-build/run_sft_8b.sh >> "$LOG" 2>&1
RC=$?
echo "[$(date -u +%H:%M:%S)] run_sft_8b rc=$RC" | tee -a "$LOG"
sleep 5
curl -s -m 5 http://127.0.0.1:8020/health 2>/dev/null | grep -q ok || { echo "[$(date -u +%H:%M:%S)] SAFETY restart 35B" | tee -a "$LOG"; sudo -n /usr/bin/systemctl start llama.service; }
echo "[$(date -u +%H:%M:%S)] MASTER2 DONE rc=$RC" | tee -a "$LOG"
