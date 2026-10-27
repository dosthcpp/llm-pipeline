#!/usr/bin/env bash
# TON-2438 off-hours v3 retry (board chose retry_offhours). Self-contained:
# clean stale -> launch GPU-idle watchdog -> train (new seed 20438 + use_reentrant=False)
# -> on V3_DONE build blinded judge batch. Trap in run_sft_v3.sh restores prod 35B on exit.
LOG="$HOME/model-build/run_v3_offhours.log"
cd "$HOME/model-build" || exit 1
{
  echo "[offhours] START $(date -u)"
  rm -f V3_DONE gen_v3.jsonl judge_batch_v3.jsonl judge_key_v3.jsonl train_watchdog.log
  setsid nohup bash train_watchdog.sh >/dev/null 2>&1 </dev/null &
  echo "[offhours] watchdog launched"
  # run_sft_v3.sh writes to deploy_v3.log (watchdog reads it for step markers); SEED breaks the
  # deterministic step-39 batch. run_sft_v3.sh blocks until V3_DONE and restores llama via trap.
  SEED=20438 bash run_sft_v3.sh > "$HOME/model-build/deploy_v3.log" 2>&1
  RC=$?
  echo "[offhours] run_sft_v3.sh rc=$RC $(date -u)"
  echo "[offhours] llama: $(systemctl is-active llama.service)"
  if [ -f V3_DONE ]; then
    echo "[offhours] V3_DONE present -> building blinded judge batch"
    bash prep_v3_judge.sh
    echo "[offhours] judge_batch_v3 lines: $(wc -l < judge_batch_v3.jsonl 2>/dev/null)"
    echo "[offhours] gen_v3 lines: $(wc -l < gen_v3.jsonl 2>/dev/null)"
  else
    echo "[offhours] NO V3_DONE -> training failed/hung (watchdog or trap restored prod). See train_watchdog.log + deploy_v3.log"
  fi
  echo "[offhours] DONE $(date -u)"
} >> "$LOG" 2>&1
