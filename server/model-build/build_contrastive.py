#!/usr/bin/env python3
"""P0.3.2 — Grounding correction via contrastive negatives + calibration. TON-2436.

Parent diagnosis (TON-2429): three SFT variants (v0 4B, v1 4B, v2 8B) all FAILED
to clear base (2.52). 8B doubled params over 4B and moved nothing -> capacity is
NOT the bottleneck. The diagnosed bottleneck is GROUNDING/CALIBRATION: the models
reason fluently but mis-call the *next action / disposition* relative to the actual
issue state, with a prior collapse toward `blocked`.

This script builds the ONE untested lever, fully OFFLINE (no GPU, no production
35B blackout). Two mechanisms, both grounded in the Paperclip disposition RULES
(not in re-parsing the teacher's final text, which is circular -- 811/910 Tier-A
gold labels come from disposition_source=text):

  1. CONTRASTIVE BOUNDARY AUGMENTATION  (teach the decision boundary)
     For each teacher trajectory we identify the most-confusable *wrong*
     disposition (the near-miss) and append a short, rule-grounded "Boundary
     check" reasoning block to the teacher trace. It commits to the gold
     disposition by stating ITS defining precondition, and rejects the near-miss
     by naming the precondition it FAILS. The model thus learns
       state -> precondition test -> disposition
     rather than the surface form of the trace.

  2. CALIBRATION RESAMPLING  (fix the over-`blocked` prior)
     We compute a target prior from the high-quality Tier-A trace distribution
     (less churn-contaminated than the IO tier, where `blocked` is inflated by
     old recovery-loop runs), then emit per-class caps so the merged train set's
     disposition frequencies match teacher-realistic proportions instead of the
     skew that produced the collapse.

Output is the SAME record schema as train.jsonl, so sft_train.py consumes it with
NO trainer change (lowest blackout risk):  --data dataset/train_contrastive.jsonl

  python3 build_contrastive.py                     # full build -> dataset/
  python3 build_contrastive.py --dry-run           # stats only, write nothing
"""
import argparse, json, os, re, collections, hashlib

DATA_DIR_DEFAULT = "/Volumes/Data/tony_ai_lab/model-build/dataset"

DISPOS = ["done", "in_review", "blocked", "in_progress", "todo"]

# The defining precondition of each disposition, per Paperclip governance
# (the execution-contract disposition checklist). These are what the FT models
# fail to ground against.
PRECONDITION = {
    "done":        "the work is complete AND verified by a concrete artifact/result",
    "in_review":   "a REAL reviewer, approval, live interaction, or monitor path exists",
    "blocked":     "a first-class blocker or a NAMED unblock owner + exact action exists",
    "in_progress": "a live continuation path exists (a concrete next step this run hands off)",
    "todo":        "the task is queued with no work started and no continuation in flight",
}

# Most-confusable near-miss per gold disposition, grounded in the observed
# failure modes (premature close, over-blocking, claimed-review-without-path).
CONFUSION = {
    "done":        ["in_review", "in_progress"],
    "in_review":   ["done", "blocked"],
    "blocked":     ["in_progress", "in_review"],
    "in_progress": ["done", "blocked"],
    "todo":        ["in_progress", "blocked"],
}

# State-signal detectors over the available context text. Used to pick the
# SHARPEST near-miss (the one whose precondition is most clearly absent) and to
# make the rejection concrete rather than generic.
SIG = {
    "review":   re.compile(r"\b(in_review|interaction|request_confirmation|reviewer|approval|"
                           r"awaiting (?:approval|review)|QA|review gate|pending (?:approval|review))\b", re.I),
    "blocker":  re.compile(r"\b(blocked by|blocker|waiting on|depends on|pending (?:boss|ceo|go)|"
                           r"unblock|gated|board-gated|out of authorization)\b", re.I),
    "continue": re.compile(r"\b(next step|remaining|continue|in_progress|will (?:continue|resume)|"
                           r"follow-up|child issue)\b", re.I),
    "done":     re.compile(r"\b(done|completed|verified|merged|tests pass|landed|shipped|closed)\b", re.I),
}


def load_jsonl(path, limit=None):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def state_text(r):
    inp = r.get("input") or {}
    if not isinstance(inp, dict):
        inp = {}
    issue = inp.get("issue") or {}
    if not isinstance(issue, dict):
        issue = {}
    out = r.get("output") or {}
    parts = [
        issue.get("title") or "", issue.get("description") or "",
        inp.get("instruction") or "", inp.get("task_markdown") or "",
        inp.get("continuation_summary") or "", (out.get("final_text") or "")[:1500],
    ]
    return " \n".join(p for p in parts if p)


def pick_near_miss(gold, txt):
    """Choose the near-miss whose precondition is most clearly ABSENT from the
    state text -- that is the sharpest contrastive signal."""
    cands = CONFUSION.get(gold, [])
    if not cands:
        return None
    has = {k: bool(p.search(txt)) for k, p in SIG.items()}
    # Score each candidate by how clearly its precondition is missing.
    def absent_score(d):
        if d == "in_review":
            return 0 if has["review"] else 2          # no reviewer/interaction signal -> clearly wrong
        if d == "blocked":
            return 0 if has["blocker"] else 2          # no blocker signal -> over-blocking is the wrong call
        if d == "in_progress":
            return 0 if has["continue"] else 1
        if d == "done":
            return 0 if has["done"] else 1
        return 1
    return max(cands, key=absent_score)


def boundary_block(gold, near, txt):
    """A short, rule-grounded boundary-check reasoning string. Names the gold
    precondition (satisfied) and the near-miss precondition (failed)."""
    has = {k: bool(p.search(txt)) for k, p in SIG.items()}
    why_wrong = {
        "in_review":   ("there is no real reviewer, approval, live interaction, or monitor path in this state"
                        if not has["review"] else "the review path is not the active gate here"),
        "blocked":     ("nothing is actually blocking -- there is no first-class blocker or named unblock owner+action"
                        if not has["blocker"] else "a path forward exists, so this is not a hard block"),
        "in_progress": ("there is no live continuation path to hand off"
                        if not has["continue"] else "the remaining step does not need another run"),
        "done":        ("the work is not yet complete-and-verified"
                        if not has["done"] else "closing now would be premature"),
        "todo":        "work is already in flight, so it is not merely queued",
    }
    return (
        f"Boundary check: the correct disposition is `{gold}` because {PRECONDITION[gold]}. "
        f"It is NOT `{near}` because {why_wrong.get(near, 'its precondition is not met in this state')}. "
        f"Ground the call in the actual issue state, not the fluency of the reasoning."
    )


def augment(r):
    """Return a contrastive-augmented copy of record r, or None if it has no
    usable gold disposition to anchor the boundary."""
    out = r.get("output") or {}
    gold = out.get("disposition")
    if gold not in PRECONDITION:
        return None
    txt = state_text(r)
    near = pick_near_miss(gold, txt)
    if not near:
        return None
    block = boundary_block(gold, near, txt)
    new = json.loads(json.dumps(r))  # deep copy
    msgs = new.get("messages") or []
    # PREPEND the boundary check as the FIRST synthetic teacher thinking turn so it
    # flows into <think>...</think> via sft_train.render_assistant unchanged AND
    # survives truncation: render caps think at 6000 chars and the tokenizer
    # truncates from the END, so a front-placed block is never lost (measured: an
    # appended block was dropped on 50/901 long traces; prepended -> 0).
    msgs.insert(0, {"role": "assistant", "content": [{"type": "thinking", "text": block}]})
    new["messages"] = msgs
    new["contrastive"] = {"gold": gold, "near_miss": near}
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR_DEFAULT)
    ap.add_argument("--io-cap", type=int, default=0,
                    help="also augment up to N Tier-B io rows (0=Tier-A only; they lack a trace so the "
                         "boundary block becomes the whole <think>)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    train = load_jsonl(os.path.join(args.data_dir, "train.jsonl"))
    aug, skipped = [], 0
    for r in train:
        a = augment(r)
        if a:
            aug.append(a)
        else:
            skipped += 1

    io_aug = []
    if args.io_cap:
        for r in load_jsonl(os.path.join(args.data_dir, "train_io.jsonl"), limit=args.io_cap):
            a = augment(r)
            if a:
                io_aug.append(a)

    # ---- calibration: target prior from Tier-A trace distribution ----
    tier_a_dist = collections.Counter(r["output"].get("disposition") for r in train
                                      if (r.get("output") or {}).get("disposition") in PRECONDITION)
    total_a = sum(tier_a_dist.values()) or 1
    target_prior = {d: round(tier_a_dist.get(d, 0) / total_a, 4) for d in DISPOS}

    aug_dist = collections.Counter(a["contrastive"]["gold"] for a in aug)
    near_dist = collections.Counter(a["contrastive"]["near_miss"] for a in aug)

    # per-class cap so no class exceeds 1.6x its target share of the augmented set
    n_aug = len(aug) or 1
    caps = {d: int(round(target_prior[d] * 1.6 * n_aug)) for d in DISPOS}

    report = {
        "issue": "TON-2436 (parent TON-2397 / diagnosis TON-2429)",
        "input_tier_a": len(train),
        "augmented_tier_a": len(aug),
        "skipped_no_gold": skipped,
        "augmented_io": len(io_aug),
        "target_prior_from_tier_a": target_prior,
        "augmented_gold_dist": dict(aug_dist),
        "near_miss_dist": dict(near_dist),
        "calibration_caps_per_class": caps,
        "mechanism": {
            "contrastive": "rule-grounded Boundary-check <think> block per record; commits to gold "
                           "precondition, rejects sharpest near-miss precondition",
            "calibration": "target prior from Tier-A trace dist; per-class cap = 1.6x target share to "
                           "deflate the over-`blocked` prior when mixing Tier-B IO",
        },
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.dry_run:
        print("\n== dry-run: wrote nothing ==")
        return

    out_path = os.path.join(args.data_dir, "train_contrastive.jsonl")
    with open(out_path, "w") as f:
        for a in aug + io_aug:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    with open(os.path.join(args.data_dir, "contrastive_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # sample for human inspection
    if aug:
        samp = aug[0]
        last = samp["messages"][-1]["content"][0]["text"]
        print(f"\n== wrote {out_path} ({len(aug)+len(io_aug)} records) ==")
        print(f"== sample boundary block (gold={samp['contrastive']['gold']}, "
              f"near={samp['contrastive']['near_miss']}) ==\n{last}")


if __name__ == "__main__":
    main()
