#!/bin/bash
# TON-2747 — auto-swap the Atlas :8020 llama-server from Qwen3.6-35B-A3B (UD quant)
# to the STOCK dense Qwen3.6-27B Q4_K_M, ONCE the download finishes. Self-verifying
# with automatic rollback: if the 27B does not come up healthy on :8020, restore the
# 35B launcher so the shared endpoint is never left dead. Detached via setsid.
set -u
cd /home/tony/models || exit 1
LOG=/home/tony/swap27b.log
GGUF=/home/tony/models/Qwen3.6-27B-Q4_K_M.gguf
say(){ echo "$(date '+%F %T') $*" >>"$LOG"; }
say "=== swap27b started ==="

# 1) wait for the download to finish (aria2c gone + no .aria2 control + full size)
for i in $(seq 1 300); do
  if ! pgrep -f 'aria2c.*Qwen3.6-27B' >/dev/null 2>&1 && [ ! -f "$GGUF.aria2" ]; then break; fi
  sleep 60
done
sz=$(stat -c%s "$GGUF" 2>/dev/null || echo 0)
if [ "$sz" -lt 17984872960 ]; then say "FAIL: download incomplete size=$sz"; exit 2; fi
if [ "$(head -c4 "$GGUF")" != "GGUF" ]; then say "FAIL: bad GGUF magic"; exit 3; fi
say "download OK size=$sz magic=GGUF"

# 2) back up the live 35B launcher and swap in the staged 27B launcher
TS=$(date +%Y%m%d-%H%M%S)
BAK=/home/tony/start-llama.sh.bak-35b-$TS
cp -a /home/tony/start-llama.sh "$BAK" || { say "FAIL: cannot back up start-llama.sh"; exit 4; }
cp -a /home/tony/start-llama-27b.sh /home/tony/start-llama.sh || { say "FAIL: cannot stage 27B launcher"; exit 5; }
say "start-llama.sh -> 27B (backup $BAK)"

# 3) stop the running 35B; the tmux watchdog restarts via start-llama.sh (now 27B)
pkill -9 -f 'llama-server.*--port 8020' 2>/dev/null
say "killed old llama-server; waiting for watchdog to bring up 27B"

# 4) wait up to ~12 min for a healthy :8020 serving the 27B alias
ok=0; mid=""
for i in $(seq 1 48); do
  sleep 15
  code=$(curl -s -o /dev/null -m 8 -w '%{http_code}' http://localhost:8020/health 2>/dev/null)
  if [ "$code" = "200" ]; then
    mid=$(curl -s -m 8 http://localhost:8020/v1/models 2>/dev/null | grep -o 'Qwen3.6-27B-Q4_K_M.gguf' | head -1)
    [ -n "$mid" ] && { ok=1; break; }
  fi
done

if [ "$ok" = "1" ]; then
  say "SUCCESS: :8020 healthy serving $mid"
  exit 0
fi

# 5) rollback — restore the 35B launcher and restart so :8020 is never left dead
say "FAIL: 27B not healthy (code=$code mid=$mid) -> ROLLBACK to 35B"
cp -a "$BAK" /home/tony/start-llama.sh
pkill -9 -f 'llama-server.*--port 8020' 2>/dev/null
say "rolled back start-llama.sh to 35B; watchdog will restart it"
exit 9
