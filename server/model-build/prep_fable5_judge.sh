#!/usr/bin/env bash
# TON-2586 — Build blinded judge batch adding fable5 (Qwen3.6-27B-Fable-5) to the
# canonical 6-model anchor pool (teacher/base/v0/v1/v2_8b/v3). Same seed/frac as
# prep_v3_judge.sh so teacher~2.99 / base~2.44 anchors reproduce and fable5 is comparable.
# Output: judge_batch_fable5.jsonl + judge_key_fable5.jsonl (CTO Opus subagents score).
set -eu
cd "$HOME/model-build"
source /home/tony/sovereign-model-env/bin/activate
GEN_F="gen_qwen27b_fable5.jsonl"
if [ ! -f "$GEN_F" ]; then echo "ERROR: $GEN_F not found"; exit 1; fi
echo "$GEN_F found ($(wc -l < $GEN_F) lines)"
python eval_v2_judge_prep.py \
    --gen teacher=gen_teacher.jsonl base=gen_base.jsonl v0=gen_v0.jsonl v1=gen_v1.jsonl v2_8b=gen_v2_8b.jsonl v3=gen_v3.jsonl fable5="$GEN_F" \
    --eval dataset/eval.jsonl \
    --baseline-frac 0.5 \
    --seed 2438 \
    --out-batch judge_batch_fable5.jsonl \
    --out-key   judge_key_fable5.jsonl
python -c "
import json
from collections import Counter
mc = Counter(json.loads(l)[\"model\"] for l in open(\"judge_key_fable5.jsonl\"))
print(\"model mix:\", dict(mc)); print(\"total items:\", sum(mc.values()))
"
echo "PREP_FABLE5_DONE"
