#!/usr/bin/env bash
# TON-2410 — v1 native gen RE-RUN after first attempt hung on a runaway greedy decode.
# Fix: eval_v2_gen.py generate() now has max_time=90 (caps per-example wall clock).
# Self-contained GPU window: stop llama -> gen -> restart llama (EXIT trap).
set -u
cd "$HOME/model-build"
source /home/tony/sovereign-model-env/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=0
OUT="$HOME/model-build/out/sovereign-v1-adapter"
restore_llama() {
  echo "[$(date +%T)] restarting llama.service ..."
  sudo -n /usr/bin/systemctl start llama.service
  sleep 8
  for i in $(seq 1 20); do
    curl -s -m 5 http://127.0.0.1:8020/health 2>/dev/null | grep -q ok && { echo "[$(date +%T)] llama HEALTHY"; return; }
    sleep 6
  done
  echo "[$(date +%T)] WARNING: llama not healthy — CHECK MANUALLY"
}
trap restore_llama EXIT
echo "[$(date +%T)] === ensure llama stopped, free VRAM ==="
sudo -n /usr/bin/systemctl stop llama.service
sleep 4
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
echo "[$(date +%T)] === V1 native gen (max_time=90 per-example guard) ==="
python eval_v2_gen.py --mode hf --adapter "$OUT" --max-tokens 2048 \
  --done-cap 30 --blocked-cap 30 --out gen_v1.jsonl 2>&1
RC=$?
echo "[$(date +%T)] gen rc=$RC"
ls -la gen_v1.jsonl 2>/dev/null
if [ $RC -eq 0 ] && [ -f gen_v1.jsonl ]; then
  echo "[$(date +%T)] === disposition distribution: v1 vs v0 vs teacher vs base ==="
  python eval_v2_gen.py --dist gen_v1.jsonl gen_v0.jsonl gen_teacher.jsonl gen_base.jsonl 2>&1
fi
echo "[$(date +%T)] ALL DONE rc=$RC"
