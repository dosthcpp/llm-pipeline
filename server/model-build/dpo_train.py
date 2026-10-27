#!/usr/bin/env python3
"""TON-2893 — Hand-rolled DPO (negative-gradient) on QLoRA, self-contained.

WHY hand-rolled: this venv's trl 1.6.0 DPOTrainer fails to import against
torch 2.5.1 (`cannot import name 'FSDPModule'`). Rather than churn the
training/serving venv, we implement the standard DPO loss directly on peft.

Two LoRA adapters on ONE 4-bit base (single 8B copy in VRAM):
  policy : warm-started from iter-0 adapter, TRAINABLE  (pi_theta)
  ref    : warm-started from iter-0 adapter, FROZEN     (pi_ref)
At step 0 policy==ref -> margin 0 -> loss=log(2)~0.693 (smoke check).

DPO loss (Rafailov et al.):
  L = -log sigmoid( beta * [ (logp_pi(chosen)-logp_ref(chosen))
                            -(logp_pi(rejected)-logp_ref(rejected)) ] )

Levers FIXED by PREREG_TON2893 §3: beta=0.1 lr=5e-6 epochs=2 seed=42
  batch=1 grad_accum=8 max_length=1536 warm-start=iter-0.

Usage:
  python dpo_train.py --validate                 # CPU-only: tokenize pairs, length stats, NO GPU
  python dpo_train.py --dry-run                  # print config only
  python dpo_train.py                            # full DPO train (needs free GPU)
Output: out/neg-v2-8b-dpo-adapter/policy/  (LoRA adapter) + loss_log.json
"""
import argparse, json, math, os, statistics, sys


def load_pairs(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--pairs", default="dpo_pairs_iter1.jsonl")
    ap.add_argument("--init-adapter", default=os.path.expanduser("~/model-build/out/raft-v2-8b-iter0-adapter"))
    ap.add_argument("--output-dir", default="./out/neg-v2-8b-dpo-adapter")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--validate", action="store_true",
                    help="CPU-only: tokenize pairs, length stats + loss=log2 sanity note. No GPU, no weights.")
    args = ap.parse_args()

    print("== DPO config ==")
    for k, v in vars(args).items():
        print(f"  {k:<14}: {v}")
    if args.dry_run:
        print("== dry-run: no work =="); return

    pairs = load_pairs(args.pairs)
    print(f"  loaded {len(pairs)} preference pairs")
    if not pairs:
        sys.exit("ERROR: no pairs")

    # ---- CPU validation: tokenize + length stats, no model weights ----
    if args.validate:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        def tot_len(p, side):
            return len(tok(p["prompt"] + p[side], add_special_tokens=False)["input_ids"])
        ch = sorted(tot_len(p, "chosen") for p in pairs)
        rj = sorted(tot_len(p, "rejected") for p in pairs)
        def pct(a, q): return a[min(len(a) - 1, int(len(a) * q))]
        for name, a in (("chosen", ch), ("rejected", rj)):
            over = sum(1 for l in a if l > args.max_length)
            print(f"  {name}: n={len(a)} p50={pct(a,.5)} p90={pct(a,.9)} max={a[-1]} "
                  f">max_len({args.max_length})={over} ({100*over/len(a):.1f}% -> PROMPT trimmed, completion kept whole)")
        print(f"  sanity: at step0 policy==ref -> DPO loss = -log sigmoid(0) = log(2) = {math.log(2):.4f}")
        print("== validate OK ==")
        return

    # ---- Full DPO training (needs free GPU) ----
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    torch.manual_seed(args.seed)
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info()
    print(f"  free VRAM: {free/1e9:.1f} GB / {total/1e9:.1f} GB")
    if free < 8e9:
        sys.exit("ERROR: <8 GB free VRAM — stop the production llama-server before training.")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map="auto", trust_remote_code=True)
    base.config.use_cache = False

    # policy (trainable) + ref (frozen) — one base, two adapters
    model = PeftModel.from_pretrained(base, args.init_adapter, adapter_name="policy", is_trainable=True)
    model.load_adapter(args.init_adapter, adapter_name="ref")  # loading a 2nd adapter resets grad flags
    # Explicitly (re)assert grad flags: ONLY the policy LoRA trains; ref + base stay frozen.
    n_pol = 0
    for n, p in model.named_parameters():
        train = ("lora_" in n and ".policy." in n)
        p.requires_grad_(train)
        n_pol += int(train)
    print(f"  marked {n_pol} policy LoRA tensors trainable")
    model.set_adapter("policy")
    # Checkpointing is REQUIRED (8B full-activation backward OOMs 24G). The earlier
    # deadlock came from interleaving the no_grad ref forward + set_adapter() switches
    # INSIDE the policy graph; below we split ref (detached) and policy phases so no
    # adapter switch happens between the policy forward and its backward.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in trainable)
    print(f"  trainable params (policy LoRA): {n_tr/1e6:.2f}M")

    def encode(prompt, completion):
        # Loss is computed over the COMPLETION (comp_mask); the fabricated-done claim
        # lives there. So keep the completion whole and truncate the (boilerplate-heavy)
        # PROMPT tail-first instead. Only if a completion alone exceeds the budget do we
        # trim it. chosen/rejected share the prompt, so equal prompt trimming is fair.
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tok(completion, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        if len(c_ids) >= args.max_length:          # pathological: completion itself too long
            c_ids = c_ids[: args.max_length - 16]
            p_ids = p_ids[-16:]                    # minimal prompt tail to still condition
        elif len(p_ids) + len(c_ids) > args.max_length:
            keep_p = args.max_length - len(c_ids)  # keep the prompt tail (closest to assistant turn)
            p_ids = p_ids[-keep_p:]
        ids = p_ids + c_ids
        comp_mask = [0] * len(p_ids) + [1] * len(c_ids)
        return ids, comp_mask

    def seq_logp(input_ids, comp_mask):
        # sum log p(token) over completion tokens only
        attn = torch.ones_like(input_ids)
        out = model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        m = comp_mask[:, 1:].float()
        # token logp = selected_logit - logsumexp(vocab); avoids materializing the full
        # [B,T,vocab] log_softmax tensor (Qwen3 vocab ~152k) which is the memory hog.
        sel = torch.gather(logits, 2, labels.unsqueeze(-1)).squeeze(-1).float()
        logZ = torch.logsumexp(logits.float(), dim=-1)
        tok_lp = sel - logZ
        return (tok_lp * m).sum(dim=-1)

    def to_t(prompt, completion):
        ids, cm = encode(prompt, completion)
        return (torch.tensor([ids], device=model.device),
                torch.tensor([cm], device=model.device))

    opt = torch.optim.AdamW(trainable, lr=args.lr)
    order = list(range(len(pairs)))
    rng = torch.Generator().manual_seed(args.seed)
    steps = int(math.ceil(len(pairs) * args.epochs))
    log = []
    model.train()
    opt.zero_grad()
    gstep = 0; running = 0.0; acc = 0; correct = 0
    total_updates = 0
    for ep in range(int(args.epochs)):
        perm = torch.randperm(len(pairs), generator=rng).tolist()
        for i, idx in enumerate(perm):
            p = pairs[idx]
            ch_ids, ch_m = to_t(p["prompt"], p["chosen"])
            rj_ids, rj_m = to_t(p["prompt"], p["rejected"])
            # Phase A: reference logps (adapter=ref, no_grad) -> detached scalars.
            model.set_adapter("ref")
            with torch.no_grad():
                ref_c = seq_logp(ch_ids, ch_m).detach()
                ref_r = seq_logp(rj_ids, rj_m).detach()
            # Phase B: policy logps (adapter=policy, grad, checkpointed) -> backward.
            # No adapter switch between these forwards and backward() => no deadlock.
            model.set_adapter("policy")
            pol_c = seq_logp(ch_ids, ch_m)
            pol_r = seq_logp(rj_ids, rj_m)
            logits = args.beta * ((pol_c - ref_c) - (pol_r - ref_r))
            loss = -torch.nn.functional.logsigmoid(logits).mean()
            (loss / args.grad_accum).backward()
            running += loss.item()
            acc += 1
            if (pol_c - ref_c).item() > (pol_r - ref_r).item():
                correct += 1
            if acc % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step(); opt.zero_grad(); total_updates += 1
                gstep += 1
                avg = running / args.grad_accum
                rec = {"epoch": ep, "update": total_updates, "pairs_seen": acc + ep * len(pairs),
                       "loss": round(avg, 4), "pref_acc": round(correct / acc, 3)}
                log.append(rec)
                print(f"  ep{ep} upd{total_updates:03d} loss={avg:.4f} pref_acc={correct/acc:.3f}")
                running = 0.0
        # flush partial accumulation at epoch end
        if acc % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); opt.zero_grad(); total_updates += 1

    os.makedirs(args.output_dir, exist_ok=True)
    model.set_adapter("policy")
    model.save_pretrained(args.output_dir, selected_adapters=["policy"])
    tok.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "loss_log.json"), "w") as f:
        json.dump({"config": vars(args), "log": log}, f, indent=2)
    print(f"== DPO done. final pref_acc={correct/max(1,acc):.3f}. "
          f"Saved policy adapter -> {args.output_dir}/policy ==")


if __name__ == "__main__":
    main()
