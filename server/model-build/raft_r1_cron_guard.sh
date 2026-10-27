#!/usr/bin/env bash
# DISABLED 2026-07-13 per TON-3160 (CEO: training/RAFT line parked - stock open-weights + serving optimization). Read DECISION_STOP_TRAINING.md before re-enabling.
echo "DISABLED per TON-3160: training/RAFT parked. See model-build/DECISION_STOP_TRAINING.md" >&2; exit 86
# TON-2873 credit-gated auto-resume guard (v2, 2026-07-03 CTO).
# Prior v1 only relaunched the *watcher* (which merely monitors) and fired even
# while Anthropic credits were exhausted -> immediate re-pause + prod-llama churn.
# v2: when the pipeline is paused on credit exhaustion, probe claude-fable-5;
# ONLY when credits are restored do we clear the pause markers and relaunch the
# actual DRIVER (run_raft_r1.sh, which resumes reward --resume -> select ->
# train). No science levers touched: liveness/transport only.
set -uo pipefail
GD="$HOME/model-build"
LOCK="$GD/.cron_guard.lock"
exec 9>"$LOCK" || exit 0
flock -n 9 || exit 0

WLOG="$GD/raft_r1_watcher.log"
STATUS="$GD/raft_r1_status.txt"

# already live? (driver or watcher running) -> nothing to do
pgrep -f "[r]un_raft_r1.sh"      >/dev/null && exit 0
pgrep -f "[r]aft_r1_watcher3.sh" >/dev/null && exit 0

# Only auto-resume when we are in the credit-pause state. Any other terminal
# state (genuine COMPLETE, or a non-credit failure needing CTO eyes) is left
# alone so a human/CTO decides.
if ! grep -qi "credits exhausted" "$STATUS" 2>/dev/null; then
  exit 0
fi

# Probe Fable-5 credit balance (cheap 3-token call).
source /home/tony/sovereign-model-env/bin/activate 2>/dev/null
OK=$(cd "$GD" && python - <<'PY' 2>/dev/null
import os
kf=os.path.expanduser("~/model-build/.anthropic_key")
if os.path.exists(kf): os.environ["ANTHROPIC_API_KEY"]=open(kf).read().strip()
try:
    import anthropic
    anthropic.Anthropic(max_retries=0).messages.create(
        model="claude-fable-5", max_tokens=3,
        messages=[{"role":"user","content":"ping"}])
    print("OK")
except Exception as e:
    print("NOCREDIT" if "credit balance" in str(e).lower() else "ERR:"+type(e).__name__)
PY
)
if [ "$OK" != "OK" ]; then
  echo "$(date -u) cron_guard: credits still low ($OK), holding" >> "$WLOG"
  exit 0
fi

# Credits restored -> clear pause markers, rotate log, relaunch DRIVER + watcher.
echo "$(date -u) cron_guard: CREDITS RESTORED -> clearing pause, relaunching driver" >> "$WLOG"
rm -f "$GD/raft_r1_watcher.done"
TS=$(date -u +%Y%m%dT%H%M%SZ)
[ -f "$GD/raft_8b_iter1.log" ] && cp "$GD/raft_8b_iter1.log" "$GD/raft_8b_iter1.log.$TS.bak"
echo "resuming (cron_guard credit-restore $(date -u +%H:%MZ))" > "$STATUS"
setsid bash "$GD/run_raft_r1.sh"      >/dev/null 2>&1 &
sleep 3
setsid bash "$GD/raft_r1_watcher3.sh" >/dev/null 2>&1 &
echo "$(date -u) cron_guard: driver(pid $!) + watcher relaunched" >> "$WLOG"
