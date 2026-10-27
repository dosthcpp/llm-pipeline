#!/usr/bin/env bash
# TON-2438 durable watcher: wait for V3_DONE, then build blinded judge batch.
# Detached via nohup so it survives the operators heartbeat session.
cd "$HOME/model-build"
{
  echo "[watch] start $(date -u)"
  for i in $(seq 1 120); do
    if [ -f V3_DONE ]; then
      echo "[watch] V3_DONE present $(date -u) (loop $i)"
      echo "[watch] llama: $(systemctl is-active llama.service)"
      echo "[watch] gen_v3 lines: $(wc -l < gen_v3.jsonl 2>/dev/null)"
      if [ ! -f judge_batch_v3.jsonl ]; then
        echo "[watch] running prep_v3_judge.sh"
        bash prep_v3_judge.sh
      else
        echo "[watch] judge_batch_v3.jsonl already exists, skip prep"
      fi
      echo "[watch] DONE $(date -u)"
      exit 0
    fi
    sleep 60
  done
  echo "[watch] TIMEOUT no V3_DONE after 120m $(date -u)"
  exit 1
} >> watch_v3_prep.log 2>&1
