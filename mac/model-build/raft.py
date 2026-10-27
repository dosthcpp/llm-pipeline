#!/usr/bin/env python3
"""TON-2441 — RAFT (reward-ranked FT) loop for the sovereign v2 RL bet (parent TON-2440).

Offline iterated reward-ranked fine-tuning (RAFT / rejection-sampling policy gradient).
Per the P0 hardware finding (single RTX 3090; the 35B teacher reward server occupies
~22 GB while serving production) the GPU phases are STRICTLY SEQUENTIAL and never run
concurrently with the reward server:

  rollout  (GPU = policy, llama DOWN)
    -> reward  (GPU = teacher @ :8020, llama UP, **NO blackout** — teacher already serves)
    -> select  (CPU only)
    -> train   (GPU = policy, llama DOWN)   [via sft_train.py in the driver]

This approximates the GRPO objective offline: keep the top-reward rollout per prompt and
take a gradient step toward it. It never needs both models resident at once.

Subcommands (each a separate process for a clean VRAM lifecycle + resumability):
  rollout  K candidate next-actions per H1 prompt from the 4B policy (base or base+adapter)
  reward   score every candidate with the local 35B teacher-as-judge (JUDGE_RUBRIC; P0 ρ=0.68)
  select   keep the top-reward candidate per prompt -> RAFT training set (sft_train-consumable)

Every phase writes a JSONL artifact and is resumable (already-done items are skipped).
The train step reuses sft_train.py unchanged (`--data raft_train.jsonl`, raft_response branch).

Usage (on Atlas, venv /home/tony/sovereign-model-env):
  # phase 1 — llama DOWN
  python raft.py rollout --adapter "" --k 4 --limit 5 --max-new 1024 --out roll_iter0.jsonl
  # phase 2 — llama UP (no blackout)
  python raft.py reward  --rollouts roll_iter0.jsonl --out reward_iter0.jsonl --resume
  # phase 3 — CPU
  python raft.py select  --rollouts roll_iter0.jsonl --rewards reward_iter0.jsonl \
                         --out raft_iter0.jsonl
  # phase 4 — llama DOWN: python sft_train.py --data raft_iter0.jsonl ...
"""
import argparse, json, os, re, sys, time, concurrent.futures

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "eval_v2_local"))

from sft_train import SYSTEM_PROMPT, render_user, _as_dict  # noqa: E402
from eval_v2_gen import build_eval_subset, gold_disp, parse_disposition  # noqa: E402
from eval_v2_judge_prep import render_context  # noqa: E402


# ---------------------------------------------------------------------------
# Phase 1 — rollout: 4B policy samples K candidate next-actions per H1 prompt.
# GPU = policy. The production llama-server MUST be stopped first (driver does this).
# ---------------------------------------------------------------------------
def cmd_rollout(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    rows = build_eval_subset(args.data, args.done_cap, args.blocked_cap)
    # TON-2816: --offset/--limit slice a stable-ordered prompt window so the driver can
    # shard the rollout into SHORT resumable GPU sub-windows (each restores the prod 35B
    # between chunks) instead of one long fleet-disrupting blackout. build_eval_subset is
    # deterministic, so offset N on the same --data always returns the same prompts.
    if args.offset:
        rows = rows[args.offset:]
    if args.limit:
        rows = rows[: args.limit]
    from collections import Counter
    print(f"rollout: {len(rows)} prompts (offset {args.offset}) x K={args.k}  "
          f"dist={dict(Counter(gold_disp(r) for r in rows))}", flush=True)

    free, total = torch.cuda.mem_get_info()
    print(f"free VRAM {free/1e9:.1f}/{total/1e9:.1f} GB", flush=True)
    if free < 6e9:
        sys.exit("ERROR: <6 GB free VRAM — stop llama.service before rollout.")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Force the ENTIRE model onto GPU0. device_map="auto" can strand the 152k-vocab
    # lm_head on CPU, making every decoded token a CPU matmul (GPU 0% / CPU 100% / ~5 tok/s).
    # The 4B 4-bit fits in ~3 GB, far under the freed 24 GB, so a single-device map is safe.
    model = AutoModelForCausalLM.from_pretrained(args.base_model, quantization_config=bnb,
                                                 device_map={"": 0}, trust_remote_code=True)
    tag = "base"
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        tag = os.path.basename(args.adapter.rstrip("/"))
    model.eval()
    torch.manual_seed(args.seed)

    n_out = 0
    with open(args.out, "w") as f:
        t0 = time.time()
        for i, r in enumerate(rows):
            inp = _as_dict(r.get("input"))
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": render_user(inp)}]
            enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                          return_tensors="pt", return_dict=True).to(model.device)
            in_len = enc["input_ids"].shape[1]
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new, do_sample=True,
                                     temperature=args.temperature, top_p=args.top_p,
                                     num_return_sequences=args.k,
                                     pad_token_id=tok.pad_token_id)
            for k in range(args.k):
                text = tok.decode(gen[k][in_len:], skip_special_tokens=True)
                rec = {"run_id": r.get("run_id"), "issue_key": r.get("issue_key"),
                       "gold": gold_disp(r), "cand_idx": k,
                       "pred": parse_disposition(text), "response": text,
                       "input": inp, "policy": tag}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_out += 1
            f.flush()
            print(f"  prompt {i+1}/{len(rows)} ({tag}) "
                  f"({(time.time()-t0)/(i+1):.1f}s/prompt)", flush=True)
    print(f"wrote {args.out} ({n_out} candidates from {len(rows)} prompts)", flush=True)


# ---------------------------------------------------------------------------
# Phase 2 — reward: local 35B teacher-as-judge scores each candidate.
# GPU = teacher @ :8020 (already serving production — NO blackout). Resumable.
# ---------------------------------------------------------------------------
def cmd_reward(args):
    from local_teacher_judge import judge_one  # reuse P0 judge harness verbatim

    system = open(args.rubric).read().strip()
    cands = [json.loads(l) for l in open(args.rollouts) if l.strip()]
    if args.limit:
        cands = cands[: args.limit]

    done = set()
    if args.resume and os.path.exists(args.out):
        for l in open(args.out):
            if l.strip():
                o = json.loads(l)
                if "overall" in o:
                    done.add(o["id"])
    todo = [c for c in cands if f"{c['run_id']}#{c['cand_idx']}" not in done]
    print(f"reward: scoring {len(todo)} candidates ({len(done)} already done)", flush=True)

    import threading
    write_lock = threading.Lock()
    counter = [0, 0]  # [ok, done]

    def score_one(c):
        cid = f"{c['run_id']}#{c['cand_idx']}"
        item = {"blind_id": cid, "context": render_context(c["input"]),
                "gold": c.get("gold"), "response": c.get("response", "")}
        try:
            sc = judge_one(args.url, system, item)
        except Exception as e:
            sc = {"blind_id": cid, "error": f"exc:{e}"}
        out = {"id": cid, "run_id": c["run_id"], "cand_idx": c["cand_idx"]}
        for k in ("disposition_correct", "grounded", "actionable", "calibrated",
                  "overall", "note", "error", "_latency_s"):
            if k in sc:
                out[k] = sc[k]
        with write_lock:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
            counter[1] += 1
            if "overall" in out:
                counter[0] += 1
            print(f"  [{counter[1]}/{len(todo)}] {cid} overall={out.get('overall','ERR')} "
                  f"lat={out.get('_latency_s','?')}s", flush=True)
        return out

    n_ok = 0
    with open(args.out, "a") as f:
        if args.workers <= 1:
            for c in todo:
                score_one(c)
                time.sleep(0.5)  # gentle on the production teacher
        else:
            # Parallel reward scoring: llama-server parallel slots must be ≥ workers.
            # TON-2837 R0: workers=3 to target ~3× reward throughput (56.5s → ~20s effective).
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(score_one, todo))
    n_ok = counter[0]
    print(f"reward done: {n_ok}/{len(todo)} scored OK -> {args.out}", flush=True)


# ---------------------------------------------------------------------------
# Phase 3 — select: keep the top-reward candidate per prompt -> RAFT train set.
# CPU only. Emits sft_train-consumable rows ({input, raft_response, ...}).
# ---------------------------------------------------------------------------
def _reward_key(s, w, response_len=0, len_penalty_threshold=4000, len_penalty=0.5):
    # TON-2816 S3 lever: overall stays STRICTLY PRIMARY; grounded+calibrated tiebreak shapes
    # WITHIN ties toward the residual gap. GRPO uses relative advantage within the group, so
    # only the induced rank matters.
    #
    # TON-2837 R1-A amendment (reward-hacking guard, 2026-07-03):
    #   iter-0 showed the 35B reward judge cannot verify fabricated claims → RAFT learned to
    #   produce long, confident-but-fabricated responses (mean 2.5k→7.3k chars). Two guards
    #   applied in the SORT KEY (not in the reward signal — judge prompt unchanged):
    #   1. Length penalty: responses >len_penalty_threshold chars get overall docked by
    #      len_penalty in the sort key (not stored, only affects rank within a group).
    #      Same-overall group → shorter grounded response wins over longer fabricated one.
    #   2. Grounded hard floor applied at the prompt level in cmd_select (not here).
    overall = s.get("overall", 0)
    if response_len > len_penalty_threshold:
        overall = max(0, overall - len_penalty)
    shaping = (w["grounded"] * s.get("grounded", 0)
               + w["calibrated"] * s.get("calibrated", 0)
               + w["actionable"] * s.get("actionable", 0)
               + w["disp"] * s.get("disposition_correct", 0))
    return (overall, shaping, s.get("grounded", 0), s.get("calibrated", 0))


def cmd_select(args):
    w = {"grounded": args.w_grounded, "calibrated": args.w_calibrated,
         "actionable": args.w_actionable, "disp": args.w_disp}
    print(f"select: overall primary; grounded+calibrated tiebreak weights (S3): {w}", flush=True)
    print(f"  R1-A guards: len_penalty_threshold={args.len_penalty_threshold} "
          f"len_penalty={args.len_penalty} grounded_floor={args.grounded_floor}", flush=True)
    cands = [json.loads(l) for l in open(args.rollouts) if l.strip()]
    scores = {}
    for l in open(args.rewards):
        if not l.strip():
            continue
        o = json.loads(l)
        if "overall" in o:
            scores[o["id"]] = o

    by_prompt = {}
    for c in cands:
        cid = f"{c['run_id']}#{c['cand_idx']}"
        s = scores.get(cid)
        if not s:
            continue
        by_prompt.setdefault(c["run_id"], []).append((c, s))

    kept, all_best, all_mean, dropped = [], [], [], 0
    dropped_grounded = 0
    kept_grounded, kept_calibrated = [], []
    for run_id, pairs in by_prompt.items():
        pairs.sort(key=lambda p: _reward_key(
            p[1], w,
            response_len=len(p[0].get("response", "")),
            len_penalty_threshold=args.len_penalty_threshold,
            len_penalty=args.len_penalty,
        ), reverse=True)
        best_c, best_s = pairs[0]
        rewards = [p[1].get("overall", 0) for p in pairs]
        all_best.append(best_s.get("overall", 0))
        all_mean.append(sum(rewards) / len(rewards))
        if best_s.get("overall", 0) < args.min_reward:
            dropped += 1  # no candidate cleared the floor — skip this prompt
            continue
        # TON-2837 R1-A: grounded hard floor — fabricated-done responses score low on
        # grounded even when overall is high; this floor rejects them before they enter train.
        if args.grounded_floor and best_s.get("grounded", 0) < args.grounded_floor:
            dropped_grounded += 1
            continue
        kept_grounded.append(best_s.get("grounded", 0))
        kept_calibrated.append(best_s.get("calibrated", 0))
        kept.append({
            "input": best_c["input"],
            "raft_response": best_c["response"],
            "run_id": run_id,
            "issue_key": best_c.get("issue_key"),
            "gold": best_c.get("gold"),
            "reward": best_s.get("overall"),
            "reward_breakdown": {k: best_s.get(k) for k in
                                 ("disposition_correct", "grounded", "actionable", "calibrated")},
            "group_rewards": sorted(rewards, reverse=True),
        })

    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(all_best) or 1
    mb = sum(all_best) / n
    mm = sum(all_mean) / n
    print(f"select: {len(by_prompt)} prompts scored, kept {len(kept)} "
          f"(dropped {dropped} below min_reward={args.min_reward}, "
          f"{dropped_grounded} below grounded_floor={args.grounded_floor})", flush=True)
    print(f"  mean BEST reward {mb:.2f} vs mean candidate reward {mm:.2f} "
          f"-> RAFT advantage = +{mb-mm:.2f}  (positive => reward separates good rollouts)",
          flush=True)
    if kept_grounded:
        kg = sum(kept_grounded) / len(kept_grounded)
        kc = sum(kept_calibrated) / len(kept_calibrated)
        # S3 shaping check: kept-set grounded/calibrated should sit ABOVE the v2-8B
        # residual floor (grounded 1.97 / calibrated 2.08) or the RAFT set won't teach
        # the gap. If these are ~floor, raise --w-grounded/--w-calibrated or --min-reward.
        print(f"  kept-set mean grounded={kg:.2f} calibrated={kc:.2f} "
              f"(v2-8B residual floor: grounded 1.97 / calibrated 2.08 — kept must exceed)",
              flush=True)
    print(f"wrote {args.out} ({len(kept)} RAFT training rows)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("rollout")
    r.add_argument("--data", default=os.path.expanduser("~/model-build/dataset/eval.jsonl"))
    r.add_argument("--base-model", default="Qwen/Qwen3-4B")
    r.add_argument("--adapter", default="")
    r.add_argument("--k", type=int, default=4)
    r.add_argument("--limit", type=int, default=5)
    r.add_argument("--offset", type=int, default=0,
                   help="skip the first N prompts (for sharded resumable rollout windows)")
    r.add_argument("--max-new", type=int, default=1024)
    r.add_argument("--temperature", type=float, default=0.7)
    r.add_argument("--top-p", type=float, default=0.95)
    r.add_argument("--done-cap", type=int, default=30)
    r.add_argument("--blocked-cap", type=int, default=30)
    r.add_argument("--seed", type=int, default=2441)
    r.add_argument("--out", default="roll_iter0.jsonl")
    r.set_defaults(func=cmd_rollout)

    w = sub.add_parser("reward")
    w.add_argument("--rollouts", default="roll_iter0.jsonl")
    w.add_argument("--rubric", default=os.path.join(HERE, "eval_v2_local", "JUDGE_RUBRIC.md"))
    w.add_argument("--url", default="http://127.0.0.1:8020")
    w.add_argument("--limit", type=int, default=0)
    w.add_argument("--resume", action="store_true")
    w.add_argument("--workers", type=int, default=1,
                   help="parallel reward workers (TON-2837 R0: use 3 to target ~3x throughput; "
                        "llama-server --parallel must be >= this value)")
    w.add_argument("--out", default="reward_iter0.jsonl")
    w.set_defaults(func=cmd_reward)

    s = sub.add_parser("select")
    s.add_argument("--rollouts", default="roll_iter0.jsonl")
    s.add_argument("--rewards", default="reward_iter0.jsonl")
    s.add_argument("--min-reward", type=float, default=2.0,
                   help="drop prompts whose best candidate did not clear this floor")
    # S3 (TON-2816) tiebreak weights: overall is always primary; these shape selection
    # among equal-overall rollouts toward grounded+calibrated (the residual gap to the
    # 2.80 gate). Set g/c weights to 0 to recover the legacy grounded>actionable tiebreak.
    s.add_argument("--w-grounded", type=float, default=0.5)
    s.add_argument("--w-calibrated", type=float, default=0.5)
    s.add_argument("--w-actionable", type=float, default=0.0)
    s.add_argument("--w-disp", type=float, default=0.0)
    # TON-2837 R1-A: length-penalty guard (response >threshold gets overall docked in sort key)
    s.add_argument("--len-penalty-threshold", type=int, default=4000,
                   help="responses longer than this (chars) get a sort-key penalty")
    s.add_argument("--len-penalty", type=float, default=0.5,
                   help="overall docked by this amount for over-length responses (sort key only)")
    # TON-2837 R1-A: grounded hard floor (fabricated-done responses score low on grounded)
    s.add_argument("--grounded-floor", type=float, default=0.0,
                   help="drop prompts whose best candidate grounded score < this (0=disabled). "
                        "R1 uses 3.0 to block fabricated-done from entering training.")
    s.add_argument("--out", default="raft_iter0.jsonl")
    s.set_defaults(func=cmd_select)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
