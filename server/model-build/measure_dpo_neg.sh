#!/usr/bin/env bash
# TON-2893 — frozen blinded eval for neg-signal DPO model.
# FROZEN HARNESS (TON-2814): DO NOT modify eval_v2_gen.py / eval_v2_judge_prep.py / eval_v2_judge_score.py.
# Gen params IDENTICAL to iter-1: mode teacher, max-tokens 4096, done-cap 30, blocked-cap 30, N=116.
# Judge: opus-4-8 metered (apples-to-apples with iter-1 2.01 baseline). Requires topup TON-2878.
set -uo pipefail
cd ~/model-build
source ~/sovereign-model-env/bin/activate
PY=~/sovereign-model-env/bin/python
Q4=~/models/neg-v2-8b-dpo-Q4_K_M.gguf
GENOUT=~/model-build/gen_neg_dpo.jsonl
BATCHOUT=~/model-build/judge_batch_neg_dpo.jsonl
KEYOUT=~/model-build/judge_key_neg_dpo.jsonl
SCORESOUT=~/model-build/judge_scores_neg_dpo.jsonl
LOG=~/model-build/measure_dpo_neg.log
SRVLOG=~/sov_gguf_dpo_neg_server.log
PORT=8021
PAUSE_FILE="$HOME/watchdog.pause"
PAUSE_SECS=10800
[ -f "$Q4" ] || { echo "MISSING $Q4"; exit 1; }
exec > >(tee -a "$LOG") 2>&1
echo "[$(date +%T)] ===== TON-2893 neg-DPO frozen-harness eval START ====="

restore_llama() {
  echo "[$(date +%T)] cleanup: kill :$PORT, restart prod 27B ..."
  pkill -f "llama-server.*$PORT" 2>/dev/null || true
  sleep 3
  bash "$HOME/start-llama.sh" &
  for i in $(seq 1 40); do
    sleep 6
    curl -s -m 5 http://127.0.0.1:8020/health 2>/dev/null | grep -q ok && {
      rm -f "$PAUSE_FILE"; echo "[$(date +%T)] prod 27B HEALTHY"; return; }
  done
  rm -f "$PAUSE_FILE"; echo "[$(date +%T)] WARNING: 27B not healthy"
}
trap restore_llama EXIT

# ── Phase 1/4 GEN (frozen harness, neg-DPO GGUF on :8021) ──
if [ -f "$GENOUT" ] && [ "$(wc -l < "$GENOUT")" -ge 116 ]; then
  echo "[$(date +%T)] gen already complete ($(wc -l < "$GENOUT") rows), skipping"
else
  echo $(( $(date +%s) + PAUSE_SECS )) > "$PAUSE_FILE"
  for p in $(pgrep llama-server); do kill "$p" 2>/dev/null; done
  for i in $(seq 1 20); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ "${free:-0}" -ge 8000 ] && break
    for p in $(pgrep llama-server); do kill -9 "$p" 2>/dev/null; done
    sleep 3
  done
  nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
  echo "[$(date +%T)] launch neg-DPO GGUF on :$PORT"
  GGML_CUDA_DISABLE_GRAPHS=1 ~/llama.cpp/build/bin/llama-server -m "$Q4" --host 127.0.0.1 --port "$PORT" \
      -ngl 99 -fa on -c 8192 -np 1 --no-webui > "$SRVLOG" 2>&1 &
  ok=0
  for i in $(seq 1 40); do
    curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q "\"status\":\"ok\"" && { ok=1; echo "[$(date +%T)] neg-DPO server HEALTHY"; break; }
    sleep 3
  done
  [ "$ok" -ne 1 ] && { echo "[$(date +%T)] server FAILED"; exit 1; }
  echo "[$(date +%T)] run eval_v2_gen (frozen: mode teacher, max-tokens 4096, done-cap 30, blocked-cap 30)"
  $PY eval_v2_gen.py --mode teacher \
      --url "http://127.0.0.1:$PORT/v1/chat/completions" \
      --teacher-model "neg-v2-8b-dpo-Q4_K_M.gguf" \
      --max-tokens 4096 --done-cap 30 --blocked-cap 30 \
      --out "$GENOUT" 2>&1
  echo "[$(date +%T)] gen rows: $(wc -l < $GENOUT 2>/dev/null)"
fi

# ── Phase 2/4 BLINDED BATCH (neg_dpo only; baseline degenerate floor auto-added) ──
if [ -f "$BATCHOUT" ] && [ "$(wc -l < "$BATCHOUT")" -ge 50 ]; then
  echo "[$(date +%T)] batch already built, skipping prep"
else
  echo "[$(date +%T)] === Phase 2/4: blinded judge batch ==="
  $PY eval_v2_judge_prep.py \
      --gen neg_dpo="$GENOUT" \
      --eval dataset/eval.jsonl \
      --baseline-frac 0.5 --seed 2893 \
      --out-batch "$BATCHOUT" --out-key "$KEYOUT" 2>&1
  echo "[$(date +%T)] batch: $(wc -l < "$BATCHOUT") items"
fi

# ── Phase 3/4 JUDGE (opus-4-8, 4 workers, blinded, rubric identical to iter-1) ──
echo "[$(date +%T)] === Phase 3/4: judge neg_dpo with opus-4-8 (--resume safe) ==="
ANTHROPIC_API_KEY=$(cat ~/model-build/.anthropic_key 2>/dev/null || echo "")
export ANTHROPIC_API_KEY
$PY judge_batch_claude.py \
    --batch "$BATCHOUT" \
    --key-file "$KEYOUT" \
    --filter-model neg_dpo \
    --out "$SCORESOUT" \
    --model claude-opus-4-8 \
    --workers 4 --resume 2>&1
echo "[$(date +%T)] judged: $(wc -l < "$SCORESOUT" 2>/dev/null) rows"

# ── Phase 4/4 AGGREGATE + GATE (PREREG §4) ──
echo "[$(date +%T)] === Phase 4/4: aggregate → gate ==="
echo "[$(date +%T)] neg_dpo judge_scores (N=$(wc -l < "$SCORESOUT" 2>/dev/null)):"
# Combine neg_dpo scores with iter-1 anchors for context (re-use existing scores)
$PY - <<'PYEOF'
import json, statistics, collections
dims = ["disposition_correct","grounded","actionable","calibrated","overall"]

# load neg_dpo scores
scores = [json.loads(l) for l in open("judge_scores_neg_dpo.jsonl") if l.strip()]
neg_dpo = [s for s in scores]  # all are neg_dpo (filtered at judge step)

# load key to confirm model tag
key = {json.loads(l)["blind_id"]: json.loads(l)["model"] for l in open("judge_key_neg_dpo.jsonl") if l.strip()}

results = {}
for model_name, rows in [("neg_dpo", neg_dpo)]:
    n = len(rows)
    if not n:
        continue
    means = {d: statistics.mean(s[d] for s in rows if d in s) for d in dims}
    results[model_name] = {"n": n, **means}

# comparison anchors from iter-1 (baseline: 2.01)
anchors = {
    "iter-1 (RAFT)": {"calibrated": 2.03, "overall": 2.01, "n": 116},
    "baseline (iter-0)": {"calibrated": 2.06, "overall": 1.99, "n": 116},
}

print("\n=== GATE RESULT (PREREG TON-2893 §4) ===")
for m, r in results.items():
    cal = r.get("calibrated", 0)
    ov = r.get("overall", 0)
    pass_cal = cal >= 2.50
    pass_ov = ov >= 2.30
    gate = "PASS ✓" if (pass_cal and pass_ov) else "FAIL ✗"
    print(f"\n  {m} (n={r['n']})")
    for d in dims:
        bar = "← TARGET" if d in ("calibrated","overall") else ""
        print(f"    {d:22s}: {r.get(d,0):.2f} {bar}")
    print(f"    --- gate: calibrated≥2.50={pass_cal} ({cal:.2f}) | overall≥2.30={pass_ov} ({ov:.2f}) → {gate}")
print("\n  -- anchors (frozen) --")
for m, a in anchors.items():
    print(f"  {m}: calibrated={a['calibrated']:.2f} overall={a['overall']:.2f}")
print("\nPRERER §4 rule: calibrated≥2.50 AND overall≥2.30 → PASS. Else → STOP+escalate.")
PYEOF
echo "[$(date +%T)] ===== MEASURE DONE. See gate result above. ====="
