#!/usr/bin/env bash
# DISABLED 2026-07-13 per TON-3160 (CEO: training/RAFT line parked - stock open-weights + serving optimization). Read DECISION_STOP_TRAINING.md before re-enabling.
echo "DISABLED per TON-3160: training/RAFT parked. See model-build/DECISION_STOP_TRAINING.md" >&2; exit 86
# TON-2436 P0.3.2 — sovereign-v3 contrastive-grounding retrain.
#
# Clean single-variable delta vs v1: SAME recipe (epochs 3, io-cap 0, max-length
# 2048, batch1 x grad-accum16, LoRA r16/a32, lr 2e-4) — ONLY the data changes to
# train_contrastive.jsonl (v1 Tier-A + a rule-grounded "Boundary check" <think>
# block per record, prepended so it survives truncation). Calibration = use clean
# Tier-A prior, no blocked-heavy Tier-B dilution (io-cap 0).
#
# Hard kill-criterion (TON-2436): v3 must clear base (>2.52) on the TON-2397 §3
# blinded next-action judge or we PARK the sovereign bet.
#
# One GPU window: stop llama -> train v3 -> v3 native gen -> restart llama (trap).
set -u
cd "$HOME/model-build"
source /home/tony/sovereign-model-env/bin/activate
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_ENABLE_HF_TRANSFER=0
OUT="$HOME/model-build/out/sovereign-v3-contrastive"
DATA="$HOME/model-build/dataset/train_contrastive.jsonl"
MT="${MT:-2048}"
DC="${DC:-30}"; BC="${BC:-30}"
PY=python
# TON-2563: prod 35B (Qwopus3.6-35B-A3B) no longer runs under systemd — it is kept
# alive by ~/watchdog.sh (tmux session "llama-server"), which resurrects llama-server
# via ~/start-llama.sh on health-fail. The OLD systemctl stop/start teardown is stale
# (stop=no-op -> VRAM never freed; start=tmux collision). Correct mechanism is the
# sanctioned self-expiring pause file (TON-2441): while it holds a FUTURE unix expiry
# the keepalive does NOT resurrect llama, so the GPU frees. Self-expiry is the safety
# net: if this run dies and never cleans up, the keepalive auto-restores prod at expiry.
GPU_PAUSE="$HOME/watchdog.pause"

restore_llama() {
  echo "[$(date +%T)] restoring prod 35B (rm pause + start-llama) ..."
  rm -f "$GPU_PAUSE"                                   # let keepalive resume guarding prod
  pgrep -o llama-server >/dev/null 2>&1 || bash /home/tony/start-llama.sh  # immediate restart if down
  sleep 8
  for i in $(seq 1 30); do
    curl -s -m 5 http://127.0.0.1:8020/health 2>/dev/null | grep -q ok && { echo "[$(date +%T)] llama HEALTHY"; return 0; }
    sleep 6
  done
  echo "[$(date +%T)] WARNING: llama not healthy — pause removed, keepalive will retry; CHECK MANUALLY"
}
trap restore_llama EXIT

echo "[$(date +%T)] === v3 preflight: data render check ==="
$PY - <<PYEOF
import json, sft_train as s
r=json.loads(open("$DATA").readline())
t=s.build_text(r); a=t.split("<|im_start|>assistant")[1]
assert "Boundary check" in a.split("</think>")[0], "boundary not inside <think>"
print("preflight OK: boundary block inside <think>;", r.get("contrastive"))
PYEOF

echo "[$(date +%T)] === Pause keepalive + stop llama-server, free VRAM for QLoRA ==="
echo "$(( $(date +%s) + 9000 ))" > "$GPU_PAUSE"        # pause ~2.5h (covers train+gen); self-expires
pkill -f "/llama.cpp/build/bin/llama-server" 2>/dev/null || true
sleep 10
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

echo "[$(date +%T)] === Train sovereign-v3 (contrastive; epochs=3 io-cap=0 max=$MT) ==="
$PY sft_train.py \
  --data "$DATA" \
  --epochs 3 --io-cap 0 \
  --max-length "$MT" --batch-size 1 --grad-accum 16 \
  --lora-r 16 --lora-alpha 32 --lr 2e-4 \
  --seed "${SEED:-20438}" \
  --output-dir "$OUT" 2>&1
TRAIN_RC=$?
if [ $TRAIN_RC -ne 0 ]; then
  echo "[$(date +%T)] TRAIN FAILED rc=$TRAIN_RC — aborting gen, llama restarts via trap"
  exit $TRAIN_RC
fi
echo "[$(date +%T)] === Train OK. adapter at $OUT ==="
ls -la "$OUT"

echo "[$(date +%T)] === V3 native gen (no triage directive) ==="
$PY eval_v2_gen.py --mode hf --adapter "$OUT" --max-tokens "$MT" \
  --done-cap "$DC" --blocked-cap "$BC" --out gen_v3.jsonl 2>&1

echo "[$(date +%T)] === disposition distribution: v3 vs v0 vs teacher vs base ==="
$PY eval_v2_gen.py --dist gen_v3.jsonl gen_v0.jsonl gen_teacher.jsonl gen_base.jsonl 2>&1
echo "[$(date +%T)] V3 PIPELINE DONE — touch done-marker"
touch "$HOME/model-build/V3_DONE"
