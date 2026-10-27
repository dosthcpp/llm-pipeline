#!/usr/bin/env bash
# TON-2542 P0.3.2 — GPU generation phase for Qwopus3.6-27B-Coder zero-shot bench.
# Reuses the CERTIFIED TON-2397 native-format harness (eval_v2_gen.py run_hf, nf4 4-bit).
# JUDGING is done OFF-GPU by blinded Opus-judge subagents (no production interruption).
#
# PRECONDITION (board/CEO-gated): the 3090 must be FREE. The production teacher
# llama.service holds ~22.5/24 GB; an operator with sudo must stop it first:
#     sudo systemctl stop llama.service     # frees VRAM
#     bash run_eval_qwopus27b.sh            # ~10-20 min for N~90 native gens
#     sudo systemctl start llama.service    # restore production teacher
# This account currently has NO passwordless systemctl (SUDO_SYSTEMCTL_DENIED),
# so it cannot free/restore the GPU itself.
set -u
cd "$HOME/model-build"
source /home/tony/sovereign-model-env/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=0
MODEL_DIR="$HOME/models/Qwopus3.6-27B-Coder"
MT="${MT:-2048}"
[ -f "$MODEL_DIR/config.json" ] || { echo "ERROR: $MODEL_DIR not downloaded"; exit 2; }
FREEMB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "[$(date +%T)] free VRAM ${FREEMB} MiB"
if [ "${FREEMB:-0}" -lt 16000 ]; then
  echo "ERROR: <16 GB free. Stop llama.service (needs sudo/operator) before running."; exit 3
fi
echo "[$(date +%T)] === Qwopus3.6-27B-Coder zero-shot native-format gen (nf4 4-bit) ==="
python eval_v2_gen.py --mode hf --base-model "$MODEL_DIR" --adapter "" \
  --max-tokens "$MT" --out gen_qwopus27b.jsonl
echo "[$(date +%T)] done -> gen_qwopus27b.jsonl. Next: blinded Opus-judge vs certified anchors (teacher 2.90 / v0 2.63 / base 2.43 / baseline 1.13)."
