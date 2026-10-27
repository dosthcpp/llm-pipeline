#!/usr/bin/env bash
# TON-2512 — HARDENED llama-server watchdog for the RAFT P2 reward phase.
# TON-2524 — REWRITE of the trigger logic. The teacher runs with --parallel 1, so a
#            concurrent inference probe ALWAYS times out while reward holds the single
#            slot (false "degenerate" -> harmful restart that kills reward). And the old
#            stall trigger watched the ROLLOUT log, idle by design during reward. Both
#            false-positive modes are removed. The reliable liveness signal during reward
#            is the REWARD OUTPUT FILE growing (one row per scored candidate). We restart
#            ONLY if that file is frozen far longer than any single 1024-token generation.
#
# Original TON-2512 fix retained: DRAIN VRAM (memory.used < 1500MiB) before reloading so a
# reload never corrupts weights on the single 3090; EXIT-trapped run script restores the
# teacher; respect ~/watchdog.pause so the watchdog never fights intentional blackouts.
set -u
URL="http://127.0.0.1:8020"
MB="$HOME/model-build"
WLOG="$MB/llama_watchdog.log"
PAUSE="$HOME/watchdog.pause"
STALL_SECS=1200         # reward-output freeze threshold (>> one 1024-tok generation @ --parallel 1)
log(){ echo "[$(date +%T)] $*" | tee -a "$WLOG"; }
paused(){ [ -f "$PAUSE" ] && [ "$(cat "$PAUSE" 2>/dev/null || echo 0)" -gt "$(date +%s)" ]; }
# Post-reload sanity: with the slot now free (reward request was killed) a probe is reliable.
probe_sane(){
  curl -s -m 30 "$URL/completion" -H 'Content-Type: application/json' \
    -d '{"prompt":"The capital of France is","n_predict":12,"temperature":0}' 2>/dev/null \
  | python3 -c '
import sys,json,re
try:
    c=json.load(sys.stdin).get("content","") or ""
except Exception:
    print("BAD"); sys.exit()
s=c.strip()
bad=(len(s)==0) or (len(set(s.replace(" ","")))<3) or (not re.search(r"[A-Za-z]",s))
print("BAD" if bad else "OK")
' | grep -q OK
}
# Resolve the reward output file from the live reward process (handles any iter); fall back
# to the newest reward_iter*.jsonl. Returns an absolute path or empty.
reward_out(){
  local o
  o=$(pgrep -af "raft.py reward" | grep -oE -- "--out [^ ]+" | head -1 | awk '{print $2}')
  [ -z "$o" ] && o=$(ls -t "$MB"/reward_iter*.jsonl 2>/dev/null | head -1)
  [ -z "$o" ] && return 1
  case "$o" in /*) echo "$o";; *) echo "$MB/$o";; esac
}
drain_and_restart(){
  log "RESTART: pause+kill+DRAIN-VRAM then reload (TON-2512 safe path)"
  expr $(date +%s) + 300 > "$PAUSE"
  kill -9 $(pgrep llama-server) 2>/dev/null
  for i in $(seq 1 40); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ "${used:-99999}" -lt 1500 ] && { log "  VRAM drained used=${used}MiB"; break; }
    sleep 2
  done
  nohup bash "$HOME/start-llama.sh" >> "$HOME/llama-server.log" 2>&1 &
  for i in $(seq 1 60); do
    sleep 5
    curl -s -m 4 "$URL/health" 2>/dev/null | grep -q ok && { log "  /health ok after ${i}x5s"; break; }
  done
  for i in $(seq 1 12); do
    if probe_sane; then log "  inference SANE -> recovery complete"; rm -f "$PAUSE"; return 0; fi
    sleep 5
  done
  log "  WARNING: inference still NOT sane after reload — will retry next cycle"
  rm -f "$PAUSE"; return 1
}
log "hardened watchdog started (reward-output stall=${STALL_SECS}s, probe DISABLED under --parallel 1) [TON-2524 v2]"
while true; do
  sleep 30
  # liveness: stop when the whole P2 run is gone
  pgrep -f "run_raft_p2.sh" >/dev/null || { log "run_raft_p2.sh gone — watchdog exiting"; exit 0; }
  # only act during the reward phase + never during an intentional blackout
  pgrep -f "raft.py reward" >/dev/null || continue
  paused && continue
  now=$(date +%s)
  RW=$(reward_out) || { log "WARN: cannot resolve reward output file — skip"; continue; }
  [ -f "$RW" ] || continue
  mtime=$(stat -c %Y "$RW" 2>/dev/null || echo "$now")
  age=$(( now - mtime ))
  if [ "$age" -gt "$STALL_SECS" ]; then
    log "STALL: reward output $(basename "$RW") frozen ${age}s (> ${STALL_SECS}s) — teacher hung, restart"
    drain_and_restart
  fi
done
