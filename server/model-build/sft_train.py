#!/usr/bin/env python3
"""P0.2 — Domain-specialized sovereign model SFT via QLoRA. TON-2394.

Base: Qwen/Qwen3-4B (default — hybrid reasoning model, native <think> support
matches our Tier-A CoT traces; Apache-2.0; fits QLoRA on a single RTX 3090).
Alternatives: Qwen/Qwen3-8B (tighter VRAM @ seq 4096 but feasible),
Qwen/Qwen3-4B-Instruct-2507 (non-thinking variant).

Data: P0.1 curation (TON-2393). Real record schema (see DATASHEET.md):
    input:    {wake_reason, instruction, issue:{key,title,description,status,priority},
               valid_disposition_options, task_markdown, continuation_summary}
    messages: teacher trace [{role, content:[{type:thinking|text|tool_use,...}]}] ([] for Tier B)
    output:   {final_text, disposition, disposition_source, ...}
We render (system + user-context) -> assistant(<think>teacher reasoning</think> + final action).

Hardware target: RTX 3090, 24 GB VRAM. QLoRA (4-bit nf4) fits comfortably,
BUT the production llama-server holds ~22 GB — free the GPU before training.

Usage:
    python sft_train.py --validate-data        # CPU-only: format + tokenize stats, NO GPU, NO model
    python sft_train.py --dry-run              # print resolved config, no work
    python sft_train.py                        # full QLoRA train (needs free GPU)
    python sft_train.py --model Qwen/Qwen3-8B  # bigger base

Outputs: ./out/sovereign-v0-adapter/ (LoRA adapter + training args + trainer_state.json loss log)
"""
import argparse
import json
import os
import statistics
import sys

SYSTEM_PROMPT = (
    "You are a Paperclip agent operating on a heartbeat. Given a wake/issue "
    "context and the valid disposition options, reason about the situation and "
    "produce the correct next action and a clear final disposition."
)


def load_jsonl(path, limit=None):
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _as_dict(v):
    """Records store input/output/meta as nested JSON objects; be defensive
    in case any arrived as a JSON-encoded string."""
    if isinstance(v, dict):
        return v
    if isinstance(v, str) and v.strip():
        try:
            d = json.loads(v)
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}
    return {}


def render_user(inp: dict) -> str:
    """Render the distilled task context as the user turn."""
    parts = []
    wake = inp.get("wake_reason") or inp.get("wake_source")
    if wake:
        parts.append(f"Wake: {wake}")
    issue = _as_dict(inp.get("issue"))
    if issue:
        head = f"Issue {issue.get('key','?')}: {issue.get('title','').strip()}"
        meta = []
        if issue.get("status"):
            meta.append(f"status={issue['status']}")
        if issue.get("priority"):
            meta.append(f"priority={issue['priority']}")
        if meta:
            head += f" ({', '.join(meta)})"
        parts.append(head)
        desc = (issue.get("description") or "").strip()
        if desc:
            parts.append(f"Description:\n{desc[:1800]}")
    instr = (inp.get("instruction") or "").strip()
    if instr:
        parts.append(f"Instruction:\n{instr[:1200]}")
    tm = (inp.get("task_markdown") or "").strip()
    if tm and tm not in (desc if issue else ""):
        parts.append(f"Task:\n{tm[:1500]}")
    cs = (inp.get("continuation_summary") or "").strip()
    if cs:
        parts.append(f"Continuation so far:\n{cs[:1000]}")
    opts = inp.get("valid_disposition_options")
    if opts:
        if isinstance(opts, list):
            opts = ", ".join(map(str, opts))
        parts.append(f"Valid dispositions: {opts}")
    return "\n\n".join(parts).strip() or "(no context)"


def render_assistant(messages, output: dict) -> str:
    """Teacher reasoning (<think>) + final action. Tier B has empty messages."""
    think_chunks = []
    for m in messages or []:
        if m.get("role") != "assistant":
            continue
        for c in m.get("content", []):
            if c.get("type") == "thinking" and c.get("text"):
                think_chunks.append(c["text"].strip())
    final = (output.get("final_text") or "").strip()
    disp = output.get("disposition")
    body = final
    if disp and disp not in ("none", None):
        body = f"{final}\n\nDisposition: {disp}"
    if think_chunks:
        think = "\n\n".join(think_chunks)[:6000]
        return f"<think>\n{think}\n</think>\n\n{body}"
    return body


def build_text(r: dict) -> str:
    inp = _as_dict(r.get("input"))
    user = render_user(inp)
    # RAFT (TON-2441): when a precomputed assistant target is present — a reward-ranked
    # policy rollout selected by the teacher-judge — train toward it directly instead of
    # the teacher CoT trace. Backward-compatible: SFT rows have no `raft_response`.
    raft = r.get("raft_response")
    if raft:
        assistant = raft.strip()
    else:
        out = _as_dict(r.get("output"))
        assistant = render_assistant(r.get("messages") or [], out)
    # Qwen3 ChatML; rendered explicitly for version-independent control over <think>.
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}<|im_end|>"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--data", default=os.path.expanduser("~/model-build/dataset/train.jsonl"))
    ap.add_argument("--eval-data", default=os.path.expanduser("~/model-build/dataset/eval.jsonl"))
    ap.add_argument("--io-data", default=os.path.expanduser("~/model-build/dataset/train_io.jsonl"),
                    help="Tier-B IO-only rows to mix in (optional)")
    ap.add_argument("--io-cap", type=int, default=2000, help="max Tier-B rows to mix in (0=none)")
    ap.add_argument("--output-dir", default="./out/sovereign-v0-adapter")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8, help="effective batch = batch_size * grad_accum")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--init-adapter", default="",
                    help="warm-start from an existing LoRA (continue-train it) instead of a fresh one; "
                         "TON-2816 RAFT uses this to keep the v2-8b adapter lineage across iterations")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42,
                    help="trainer + data shuffle seed (TON-2563: was missing -> run_sft_v3.sh --seed crashed argparse)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--validate-data", action="store_true",
                    help="CPU-only: format + tokenize, print length stats and samples. No GPU, no model weights.")
    args = ap.parse_args()

    print("== SFT config ==")
    for k, v in vars(args).items():
        print(f"  {k:<16}: {v}")

    if args.dry_run:
        print("== dry-run: no work ==")
        return

    if not os.path.exists(args.data):
        sys.exit(f"ERROR: data not found: {args.data}")

    # ---- Build training texts (Tier A CoT + optional Tier B IO) ----
    train_rows = load_jsonl(args.data)
    texts = [build_text(r) for r in train_rows]
    n_tier_a = len(texts)
    if args.io_cap and os.path.exists(args.io_data):
        io_rows = load_jsonl(args.io_data, limit=args.io_cap)
        texts += [build_text(r) for r in io_rows]
    print(f"  tier-A(CoT): {n_tier_a}  tier-B(IO): {len(texts)-n_tier_a}  total: {len(texts)}")

    # TON-2563 degenerate-record guard: drop rows whose assistant body renders empty
    # (no <think>, no final_text). With full-sequence loss these add no signal and a
    # near-empty / structurally odd batch is a plausible step-39 stall trigger. Cheap
    # insurance + logged so the drop is never silent.
    def _assistant_seg(t):
        return t.split("<|im_start|>assistant\n", 1)[-1].rsplit("<|im_end|>", 1)[0].strip()
    before = len(texts)
    texts = [t for t in texts if _assistant_seg(t)]
    dropped = before - len(texts)
    if dropped:
        print(f"  [guard] dropped {dropped} record(s) with empty assistant body")

    eval_texts = []
    if os.path.exists(args.eval_data):
        eval_texts = [build_text(r) for r in load_jsonl(args.eval_data)]
    print(f"  eval: {len(eval_texts)}")

    # ---- Validation mode: tokenizer only, no GPU, no model weights ----
    if args.validate_data:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        lens = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
        lens.sort()
        def pct(p):
            return lens[min(len(lens) - 1, int(len(lens) * p))]
        over = sum(1 for l in lens if l > args.max_length)
        print("\n== token-length stats (train) ==")
        print(f"  n={len(lens)}  min={lens[0]}  p50={pct(.5)}  p90={pct(.9)}  "
              f"p99={pct(.99)}  max={lens[-1]}  mean={statistics.mean(lens):.0f}")
        print(f"  > max_length({args.max_length}): {over} ({100*over/len(lens):.1f}%) will be truncated")
        print("\n== sample formatted example (first Tier-A) ==")
        print(texts[0][:2500])
        print("\n== validate-data OK ==")
        return

    # ---- Full QLoRA training (requires free GPU) ----
    import torch
    from datasets import Dataset
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    )
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig

    print(f"\nGPU: {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    free, total = torch.cuda.mem_get_info()
    print(f"  free VRAM: {free/1e9:.1f} GB / {total/1e9:.1f} GB")
    if free < 8e9:
        sys.exit("ERROR: <8 GB free VRAM — stop the production llama-server before training.")

    train_dataset = Dataset.from_list([{"text": t} for t in texts])
    eval_dataset = Dataset.from_list([{"text": t} for t in eval_texts]) if eval_texts else None

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb_cfg,
        device_map="auto", trust_remote_code=True,
    )

    # TON-2816 RAFT: --init-adapter continues an EXISTING LoRA (e.g. sovereign-v2-8b)
    # instead of training a fresh one, so the iterated reward-ranked step refines the
    # SAME adapter lineage the rollout policy used (adapter-N -> adapter-N+1). When set,
    # the model is already a PeftModel, so SFTTrainer must NOT re-wrap it (peft_config=None).
    if args.init_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        lora_cfg = None
        print(f"== warm-start: continuing LoRA from {args.init_adapter} (r/alpha inherited) ==")
    else:
        lora_cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )

    # trl 1.6 / transformers 5.12: SFTConfig holds max_length + dataset_text_field;
    # eval_strategy (not evaluation_strategy); tokenizer passed as processing_class.
    sft_cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True, fp16=False,
        logging_steps=10,
        # TON-2563: eval at step 50 OOM'd on contrastive records (max-len 2048 + boundary
        # block). The blinded judge IS the real score gate, so skip eval during training.
        eval_strategy="no",
        per_device_eval_batch_size=1,   # kept for completeness; eval_strategy=no means unused
        save_steps=100, save_total_limit=2,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        # TON-2563 hang fix: reentrant grad-checkpoint is the documented intermittent
        # single-GPU stall (process blocked CPU-side, GPU idle — the exact step-39
        # signature). use_reentrant=False uses the non-reentrant autograd path.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Rule out dataloader-worker / pin-memory deadlock as the stall source.
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        seed=args.seed,
        data_seed=args.seed,
        report_to="none",
        max_length=args.max_length,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_cfg,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    # Persist loss curve explicitly for the deliverable.
    with open(os.path.join(args.output_dir, "loss_log.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print(f"== Saved adapter + loss_log.json to {args.output_dir} ==")


if __name__ == "__main__":
    main()
