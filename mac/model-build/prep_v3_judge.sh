#!/usr/bin/env bash
# TON-2438/TON-2563 — Build blinded judge batch adding v3 (contrastive) to the pool.
# TON-2563: v2_8b added so the table matches the deliverable
# (teacher/base/v0/v1/v2_8b/v3). Run AFTER gen_v3.jsonl exists (V3_DONE flag present).
# Output: judge_batch_v3.jsonl + judge_key_v3.jsonl (CTO Opus subagents score).
set -eu
cd "$HOME/model-build"
source /home/tony/sovereign-model-env/bin/activate
GEN_V3="gen_v3.jsonl"
if [ ! -f "$GEN_V3" ]; then
    echo "ERROR: $GEN_V3 not found — wait for run_sft_v3.sh / V3_DONE"
    exit 1
fi
echo "gen_v3.jsonl found ($(wc -l < $GEN_V3) lines)"
python eval_v2_judge_prep.py \
    --gen teacher=gen_teacher.jsonl base=gen_base.jsonl v0=gen_v0.jsonl v1=gen_v1.jsonl v2_8b=gen_v2_8b.jsonl v3="$GEN_V3" \
    --eval dataset/eval.jsonl \
    --baseline-frac 0.5 \
    --seed 2438 \
    --out-batch judge_batch_v3.jsonl \
    --out-key   judge_key_v3.jsonl
echo "=== batch contents ==="
python -c "
import json
from collections import Counter
mc = Counter(json.loads(l)[\"model\"] for l in open(\"judge_key_v3.jsonl\"))
print(\"model mix:\", dict(mc))
print(\"total items:\", sum(mc.values()))
"
echo "=== PREP DONE — ready for CTO Opus judge pass ==="
