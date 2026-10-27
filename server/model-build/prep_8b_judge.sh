#!/usr/bin/env bash
# TON-2410 opt3 — Build blinded judge batch adding v2_8b to the pool (teacher/base/v0/v1).
# Run AFTER gen_v2_8b.jsonl exists (produced by run_sft_8b.sh).
set -eu
cd "$HOME/model-build"
source /home/tony/sovereign-model-env/bin/activate
GEN="gen_v2_8b.jsonl"
if [ ! -f "$GEN" ]; then echo "ERROR: $GEN not found — wait for run_sft_8b.sh"; exit 1; fi
echo "$GEN found ($(wc -l < $GEN) lines)"
python eval_v2_judge_prep.py \
    --gen teacher=gen_teacher.jsonl base=gen_base.jsonl v0=gen_v0.jsonl v1=gen_v1.jsonl v2_8b="$GEN" \
    --eval dataset/eval.jsonl \
    --baseline-frac 0.5 \
    --seed 2410 \
    --out-batch judge_batch_8b.jsonl \
    --out-key   judge_key_8b.jsonl
python -c "
import json
from collections import Counter
mc = Counter(json.loads(l)[chr(39)+chr(109)+chr(111)+chr(100)+chr(101)+chr(108)+chr(39)] for l in open(chr(39)+chr(106)+chr(117)+chr(100)+chr(103)+chr(101)+chr(95)+chr(107)+chr(101)+chr(121)+chr(95)+chr(56)+chr(98)+chr(46)+chr(106)+chr(115)+chr(111)+chr(110)+chr(108)+chr(39)))
print(chr(109)+chr(111)+chr(100)+chr(101)+chr(108)+chr(32)+chr(109)+chr(105)+chr(120)+chr(58), dict(mc), chr(116)+chr(111)+chr(116)+chr(97)+chr(108), sum(mc.values()))
"
echo "=== PREP DONE — ready for Opus judge pass on judge_batch_8b.jsonl ==="
