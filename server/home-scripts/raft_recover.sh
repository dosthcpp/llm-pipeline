#!/bin/bash
# TON-2515 CTO recovery: clean llama restart + detached orchestration relaunch
set -u
LOG=~/model-build/raft_recover.log
exec >> "$LOG" 2>&1
echo "===== RECOVER $(date -u) ====="

# 1. Kill any stale orchestration + llama-server
pkill -f raft_llama_watchdog_v2 2>/dev/null && echo "killed watchdog"
pkill -f run_raft_p2 2>/dev/null && echo "killed run_raft_p2"
pkill -f run_raft_smoke 2>/dev/null && echo "killed run_raft_smoke"
pkill -f sft_train 2>/dev/null && echo "killed sft_train"
pkill -f llama-server 2>/dev/null && echo "killed llama-server"
rm -f ~/watchdog.pause && echo "removed pause"

# 2. Drain VRAM
for i in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  echo "drain check $i: used=${used}MiB"
  [ "$used" -lt 1500 ] && { echo "VRAM drained"; break; }
  sleep 2
done

# 3. Fresh llama-server
bash ~/start-llama.sh
echo "llama-server launched"

# 4. Wait for /health then sane probe
for i in $(seq 1 36); do
  sleep 5
  h=$(curl -s -m 4 http://127.0.0.1:8020/health 2>/dev/null)
  echo "health check $i: $h"
  echo "$h" | grep -q '"status":"ok"' && { echo "HEALTHY after ${i}x5s"; break; }
done

# sane probe (reasoning model: allow time, cap reasoning)
probe=$(curl -s -m 90 http://127.0.0.1:8020/completion -H "Content-Type: application/json" \
  -d '{"prompt":"What is 2+2? Answer with just the number.","n_predict":32,"temperature":0.2}' 2>/dev/null)
echo "PROBE: $probe"

# 5. Relaunch orchestration FULLY DETACHED (setsid = survives session teardown)
cd ~/model-build
setsid nohup env NITERS=3 K=8 MAXNEW=1024 ITER_TIMEOUT=9000 bash run_raft_p2.sh >> raft_p2_boot.log 2>&1 < /dev/null &
echo "run_raft_p2 relaunched (setsid) pid $!"
sleep 3
setsid nohup bash ~/raft_llama_watchdog_v2.sh >> ~/model-build/llama_watchdog.log 2>&1 < /dev/null &
echo "watchdog relaunched (setsid) pid $!"
sleep 5
echo "=== FINAL PROCS ==="
pgrep -af "run_raft_p2|raft_llama_watchdog_v2|llama-server" | grep -v "bash -c"
echo "===== RECOVER DONE $(date -u) ====="
