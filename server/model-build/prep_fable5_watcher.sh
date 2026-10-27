#!/usr/bin/env bash
# TON-2586 — waits for gen=116, builds fable5-only batch (+degenerate baseline floor).
# Anchors (teacher/base) already scored separately. Seed 2586 for stable blind_ids.
set -uo pipefail
MB="$HOME/model-build"
LOG="$MB/out/prep_fable5.log"
OUT="$MB/gen_qwen27b_fable5.jsonl"
source "$HOME/sovereign-model-env/bin/activate" 2>/dev/null || true
echo "[start $(date -u +%FT%TZ)] waiting for 116 gen items" >> "$LOG"
for i in $(seq 1 360); do
  n=$(grep -c . "$OUT" 2>/dev/null || echo 0)
  if [ "$n" -ge 116 ] || grep -q "GEN2_DONE" "$MB/out/fable5_gen2.log" 2>/dev/null; then
    echo "[$(date -u +%FT%TZ)] gen ready: $n items" >> "$LOG"; break
  fi
  sleep 30
done
n=$(grep -c . "$OUT" 2>/dev/null || echo 0)
echo "[$(date -u +%FT%TZ)] building fable5 batch from $n items (+baseline floor)" >> "$LOG"
python "$MB/eval_v2_judge_prep.py" \
  --gen fable5="$OUT" \
  --eval dataset/eval.jsonl --baseline-frac 0.5 --seed 2586 \
  --out-batch "$MB/batch_fable5.jsonl" --out-key "$MB/key_fable5.jsonl" >> "$LOG" 2>&1
rc=$?
echo "PREP_FABLE5_DONE rc=$rc $(date -u +%FT%TZ)" >> "$LOG"
python3 -c "
import json
from collections import Counter
mc=Counter(json.loads(l)[\"model\"] for l in open(\"$MB/key_fable5.jsonl\"))
print(\"model mix:\", dict(mc), \"total:\", sum(mc.values()))
" >> "$LOG" 2>&1
