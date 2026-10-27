#!/usr/bin/env bash
# DISABLED 2026-07-13 per TON-3160 (CEO: training/RAFT line parked - stock open-weights + serving optimization). Read DECISION_STOP_TRAINING.md before re-enabling.
echo "DISABLED per TON-3160: training/RAFT parked. See model-build/DECISION_STOP_TRAINING.md" >&2; exit 86
# TON-2410 P0.3.1 — sovereign-v2-8b retrain (grounding + under-training fix).
#
# Diagnosis basis (TON-2397 H1 CONFIRMED): v0's "blocked collapse" was a PROMPT
# ARTIFACT of the off-distribution triage directive, NOT a learned bias — in native
# format v0's blocked share is 7.8% (vs 56% triaged) and v0 already beats base and
# approaches teacher on the blinded next-action-quality judge. The ONE real residual
# is `grounded` (v0 2.47 vs teacher 3.63). v0 was also UNDER-TRAINED: 1 epoch, 800
# Tier-B IO-only (CoT-free) rows mixed in.
#
# v1 recipe = clean single-cluster delta vs v0 (epochs 1->3, io-cap 800->0); LoRA
# r16/a32, lr 2e-4, max_length 2048, batch1 x grad-accum16 all held constant so the
# eval delta is attributable to (more Tier-A CoT exposure) + (no CoT-free dilution).
#
# One GPU window: stop llama -> train v1 -> v1 native gen -> restart llama (trap).
set -u
cd "$HOME/model-build"
source /home/tony/sovereign-model-env/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=0
OUT="$HOME/model-build/out/sovereign-v2-8b-adapter"
MT="${MT:-2048}"
DC="${DC:-30}"; BC="${BC:-30}"
PY=python

restore_llama() {
  echo "[$(date +%T)] restarting llama.service ..."
  sudo -n /usr/bin/systemctl start llama.service
  sleep 8
  for i in $(seq 1 20); do
    curl -s -m 5 http://127.0.0.1:8020/health 2>/dev/null | grep -q ok && { echo "[$(date +%T)] llama HEALTHY"; return 0; }
    sleep 6
  done
  echo "[$(date +%T)] WARNING: llama not healthy — CHECK MANUALLY"
}
trap restore_llama EXIT

echo "[$(date +%T)] === Stop llama.service, free VRAM for QLoRA ==="
sudo -n /usr/bin/systemctl stop llama.service
sleep 6
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

echo "[$(date +%T)] === Train sovereign-v2-8b (epochs=3, io-cap=0, vs v0 epochs=1 io-cap=800) ==="
$PY sft_train.py \
  --model Qwen/Qwen3-8B \
  --epochs 3 --io-cap 0 \
  --max-length 2048 --batch-size 1 --grad-accum 16 \
  --lora-r 16 --lora-alpha 32 --lr 2e-4 \
  --output-dir "$OUT" 2>&1
TRAIN_RC=$?
if [ $TRAIN_RC -ne 0 ]; then
  echo "[$(date +%T)] TRAIN FAILED rc=$TRAIN_RC — aborting gen, llama will restart via trap"
  exit $TRAIN_RC
fi
echo "[$(date +%T)] === Train OK. adapter at $OUT ==="
ls -la "$OUT"

echo "[$(date +%T)] === V1 native gen (no triage directive) ==="
$PY eval_v2_gen.py --mode hf --base-model Qwen/Qwen3-8B --adapter "$OUT" --max-tokens "$MT" \
  --done-cap "$DC" --blocked-cap "$BC" --out gen_v2_8b.jsonl 2>&1

echo "[$(date +%T)] === disposition distribution: v1 vs v0 vs teacher vs base ==="
$PY eval_v2_gen.py --dist gen_v2_8b.jsonl gen_v0.jsonl gen_teacher.jsonl gen_base.jsonl 2>&1
echo "[$(date +%T)] ALL DONE"
