#!/usr/bin/env bash
# TON-2410 — Build blinded judge batch that adds v1 to the H2 pool (v0+teacher+base).
# Run AFTER gen_v1.jsonl exists (produced by run_sft_v1.sh).
# Output: judge_batch_v1.jsonl + judge_key_v1.jsonl (CTO Opus subagents score the batch).
set -eu
cd "$HOME/model-build"
source /home/tony/sovereign-model-env/bin/activate

GEN_V1="gen_v1.jsonl"
if [ ! -f "$GEN_V1" ]; then
    echo "ERROR: $GEN_V1 not found — wait for run_sft_v1.sh to complete"
    exit 1
fi
echo "gen_v1.jsonl found ($(wc -l < $GEN_V1) lines)"

python eval_v2_judge_prep.py \
    --gen teacher=gen_teacher.jsonl base=gen_base.jsonl v0=gen_v0.jsonl v1="$GEN_V1" \
    --eval dataset/eval.jsonl \
    --baseline-frac 0.5 \
    --seed 2410 \
    --out-batch judge_batch_v1.jsonl \
    --out-key   judge_key_v1.jsonl

echo "=== batch contents ==="
python -c "
import json
from collections import Counter
c = Counter(json.loads(l)['blind_id'][:1]+'...' for l in open('judge_key_v1.jsonl'))
mc = Counter(json.loads(l)['model'] for l in open('judge_key_v1.jsonl'))
print('model mix:', dict(mc))
print('total items:', sum(mc.values()))
"
echo "=== PREP DONE — ready for CTO Opus judge pass ==="
