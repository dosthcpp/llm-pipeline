#!/usr/bin/env python3
"""TON-2893 — Build offline DPO preference pairs from the already-rewarded
iter-1 RAFT rollouts (no new rollout, no re-reward, $0).

Directly targets the 73% over-claim / fabricated-done failure mode diagnosed in
TON-2880: chosen = grounded+calibrated response, rejected = fabricated-done
response from the SAME prompt. This is the negative signal positive-only RAFT
cannot produce.

Selection rule (PREREG_TON2893 §2 — FIXED before build):
  chosen   = grounded>=3 AND overall>=3   ; tie-break: calibrated desc, overall desc
  rejected = overall<=2  OR  grounded<=1  ; tie-break: calibrated asc,  overall asc
  keep one pair per prompt (run_id) where both exist.

Prompt text uses sft_train.render_user() so it is identical to the SFT/RAFT/eval prompt.

Inputs (Atlas ~/model-build/):
  --rollouts roll_8b_iter1.jsonl   (run_id, cand_idx, response, input)
  --rewards  reward_8b_iter1.jsonl (run_id, cand_idx, grounded, calibrated, overall, note)
Output:
  dpo_pairs_iter1.jsonl  {run_id, issue_key, prompt, chosen, rejected, meta}

Usage:
  python build_dpo_pairs.py                      # build
  python build_dpo_pairs.py --report            # build + print distribution report
"""
import argparse, json, sys, collections

# reuse the exact prompt renderer used by training/eval
from sft_train import render_user, SYSTEM_PROMPT, _as_dict


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def chat_prompt(inp: dict) -> str:
    """Same ChatML frame as sft_train.build_text, minus the assistant body."""
    user = render_user(_as_dict(inp))
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default="roll_8b_iter1.jsonl")
    ap.add_argument("--rewards", default="reward_8b_iter1.jsonl")
    ap.add_argument("--out", default="dpo_pairs_iter1.jsonl")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    rolls = load_jsonl(args.rollouts)
    rewards = load_jsonl(args.rewards)

    # index rollouts by (run_id, cand_idx) -> response/input/issue_key
    roll_by = {}
    for r in rolls:
        roll_by[(r["run_id"], r["cand_idx"])] = r

    # attach reward subscores to each rollout candidate
    by_run = collections.defaultdict(list)
    missing_text = 0
    for rw in rewards:
        key = (rw["run_id"], rw["cand_idx"])
        rr = roll_by.get(key)
        if rr is None or not (rr.get("response") or "").strip():
            missing_text += 1
            continue
        by_run[rw["run_id"]].append({
            "cand_idx": rw["cand_idx"],
            "response": rr["response"].strip(),
            "input": rr.get("input", {}),
            "issue_key": rr.get("issue_key"),
            "grounded": rw.get("grounded"),
            "calibrated": rw.get("calibrated"),
            "overall": rw.get("overall"),
            "disposition_correct": rw.get("disposition_correct"),
            "note": rw.get("note", ""),
        })

    pairs = []
    skip_no_chosen = skip_no_rejected = 0
    for run_id, cands in by_run.items():
        chosen_pool = [c for c in cands if (c["grounded"] or 0) >= 3 and (c["overall"] or 0) >= 3]
        rejected_pool = [c for c in cands if (c["overall"] or 5) <= 2 or (c["grounded"] or 5) <= 1]
        if not chosen_pool:
            skip_no_chosen += 1
            continue
        if not rejected_pool:
            skip_no_rejected += 1
            continue
        chosen = max(chosen_pool, key=lambda c: (c["calibrated"] or 0, c["overall"] or 0))
        rejected = min(rejected_pool, key=lambda c: (c["calibrated"] or 0, c["overall"] or 0))
        if chosen["cand_idx"] == rejected["cand_idx"]:
            continue
        pairs.append({
            "run_id": run_id,
            "issue_key": chosen.get("issue_key"),
            "prompt": chat_prompt(chosen["input"]),
            "chosen": chosen["response"],
            "rejected": rejected["response"],
            "meta": {
                "chosen": {k: chosen[k] for k in ("cand_idx", "grounded", "calibrated", "overall", "disposition_correct")},
                "rejected": {k: rejected[k] for k in ("cand_idx", "grounded", "calibrated", "overall", "disposition_correct")},
                "rejected_note": (rejected["note"] or "")[:200],
            },
        })

    with open(args.out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"[build_dpo_pairs] rollouts={len(rolls)} rewards={len(rewards)} "
          f"missing_text={missing_text} prompts={len(by_run)}")
    print(f"[build_dpo_pairs] skipped: no_chosen={skip_no_chosen} no_rejected={skip_no_rejected}")
    print(f"[build_dpo_pairs] wrote {len(pairs)} DPO pairs -> {args.out}")

    if args.report and pairs:
        def dist(side, axis):
            c = collections.Counter(p["meta"][side][axis] for p in pairs)
            return dict(sorted(c.items()))
        print("\n== pair quality report ==")
        for axis in ("grounded", "calibrated", "overall"):
            print(f"  chosen.{axis:10s}: {dist('chosen', axis)}")
            print(f"  reject.{axis:10s}: {dist('rejected', axis)}")
        # calibrated margin (the axis we target)
        margins = [(p["meta"]["chosen"]["calibrated"] or 0) - (p["meta"]["rejected"]["calibrated"] or 0) for p in pairs]
        print(f"  calibrated margin (chosen-rejected): mean={sum(margins)/len(margins):.2f} "
              f"min={min(margins)} max={max(margins)}")
        print("\n== sample pair 0 (truncated) ==")
        p = pairs[0]
        print(f"  issue={p['issue_key']} chosen_meta={p['meta']['chosen']} reject_meta={p['meta']['rejected']}")
        print(f"  reject_note: {p['meta']['rejected_note']}")
        print(f"  chosen[:200]:   {p['chosen'][:200]!r}")
        print(f"  rejected[:200]: {p['rejected'][:200]!r}")


if __name__ == "__main__":
    main()
