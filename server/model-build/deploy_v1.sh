#!/usr/bin/env bash
# TON-2410 — full v1 deploy: build GGUF, then serve on 8022 (CPU), health-check.
set -uo pipefail
cd ~/model-build
LOG=~/model-build/deploy_v1.log
echo "=== deploy_v1 start $(date -u +%H:%M:%S) ===" | tee "$LOG"
# 1) build (merge/convert/quantize)
bash ~/model-build/build_gguf_v1.sh >>"$LOG" 2>&1
if ! grep -q ALL_DONE "$LOG"; then
  echo "BUILD_FAILED" | tee -a "$LOG"; exit 1
fi
echo "[build ok] $(date -u +%H:%M:%S)" | tee -a "$LOG"
# 2) kill any stale 8022 server, launch fresh detached
pkill -f "llama-server.*8022" 2>/dev/null || true
sleep 2
setsid bash ~/model-build/serve_v1.sh >~/model-build/serve_v1.log 2>&1 < /dev/null &
echo "[serve launched pid via setsid] $(date -u +%H:%M:%S)" | tee -a "$LOG"
# 3) wait for health
for i in $(seq 1 30); do
  sleep 4
  if curl -sf http://127.0.0.1:8022/health >/dev/null 2>&1; then
    echo "SERVE_HEALTHY after $((i*4))s" | tee -a "$LOG"
    break
  fi
done
curl -s http://127.0.0.1:8022/health | tee -a "$LOG"
echo "" ; echo "DEPLOY_COMPLETE $(date -u +%H:%M:%S)" | tee -a "$LOG"
