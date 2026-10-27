#!/usr/bin/env bash
# TON-2773-B — Frontier-API candidate on the frozen disposition harness (EVAL_FROZEN_SPEC v1).
#
# TWO keyless/keyed paths (boss TON-2783: "claude -p 로 하면 안되남" → yes):
#   * NO board key needed: uses `claude -p` (local authenticated Claude Code session). DEFAULT.
#     Decode = CLI default sampling (NOT frozen greedy temp=0) — directional like-for-like.
#   * WITH a board-gated key in env (ANTHROPIC_API_KEY/FRONTIER_API_KEY): strict frozen greedy
#     decode via the Anthropic Messages API. Auto-selected when a key is present.
#
#   ./run_frontier_candidate.sh                  # all three (auto: api if key, else cli)
#   ./run_frontier_candidate.sh haiku-4-5        # one tag
#   FORCE_MODE=cli ./run_frontier_candidate.sh   # force claude -p even if a key is set
#
# Produces gen_<short>.jsonl (same schema as gen_teacher/gen_v0), ready for:
#   python eval_v2_judge_prep.py --gen teacher=gen_teacher.jsonl v0=gen_v0.jsonl \
#       opus-4-8=gen_opus.jsonl sonnet-4-6=gen_sonnet.jsonl haiku-4-5=gen_haiku.jsonl
#   (then blinded Opus judge -> eval_v2_judge_score.py -> cost_quality_baseline.py)
set -euo pipefail
cd "$(dirname "$0")"

DATA="${DATA:-dataset/eval.jsonl}"
PY="${PYTHON:-python3}"
MODE="${FORCE_MODE:-}"
if [[ -z "$MODE" ]]; then
  if [[ -n "${ANTHROPIC_API_KEY:-}" || -n "${FRONTIER_API_KEY:-}" ]]; then
    MODE=api; echo "[mode] api — frontier key found, strict frozen greedy decode"
  else
    MODE=cli; echo "[mode] cli — no key, using 'claude -p' (decode=cli-default, keyless)"
  fi
fi

TAGS=( "$@" )
[[ ${#TAGS[@]} -eq 0 ]] && TAGS=( opus-4-8 sonnet-4-6 haiku-4-5 )

for tag in "${TAGS[@]}"; do
  case "$tag" in            # bash 3.2 (macOS) safe — no associative arrays
    opus-4-8)   out=gen_opus.jsonl ;;
    sonnet-4-6) out=gen_sonnet.jsonl ;;
    haiku-4-5)  out=gen_haiku.jsonl ;;
    *) echo "unknown tag: $tag (use opus-4-8|sonnet-4-6|haiku-4-5)" >&2; exit 1 ;;
  esac
  echo "== generating frontier candidate: $tag -> $out  (mode=$MODE) =="
  "$PY" eval_v2_gen.py --mode "$MODE" --api-model "$tag" --data "$DATA" --out "$out" \
        --concurrency "${CONCURRENCY:-5}" --claude-bin "${CLAUDE_BIN:-claude}"
done
echo "done. next: eval_v2_judge_prep.py with these gen files, then blinded judge + cost_quality_baseline.py"
