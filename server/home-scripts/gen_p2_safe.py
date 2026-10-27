#!/usr/bin/env python3
"""TON-2537 — robust policy-adapter native gen with INCREMENTAL write + resumable --start.
Reuses eval_v2_gen helpers so the subset/prompt are byte-identical to base/teacher gens.
Writes one JSONL line per example (flushed) so a CUDA stall never loses prior work; on a
per-example Python exception it records __ERR__ and continues. If generate() hard-stalls
(CUDA), kill the proc and re-run with --start <next idx> (already-written lines are kept).
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.expanduser("~/model-build"))
from eval_v2_gen import build_eval_subset, make_prompt, gold_disp, parse_disposition
from sft_train import SYSTEM_PROMPT

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen3-4B")
    ap.add_argument("--data", default=os.path.expanduser("~/model-build/dataset/eval.jsonl"))
    ap.add_argument("--out", default="gen_p2.jsonl")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-time", type=int, default=70)
    ap.add_argument("--done-cap", type=int, default=30)
    ap.add_argument("--blocked-cap", type=int, default=30)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=40)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    free, total = torch.cuda.mem_get_info()
    print(f"free VRAM {free/1e9:.1f}/{total/1e9:.1f} GB", flush=True)
    if free < 6e9:
        sys.exit("ERROR: <6 GB free VRAM — stop llama-server first.")

    rows = build_eval_subset(args.data, args.done_cap, args.blocked_cap)
    rows = rows[:args.end]
    print(f"eval subset full N={len(rows)}; generating idx [{args.start}:{args.end})", flush=True)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, quantization_config=bnb,
                                                 device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    mode = "a" if args.start > 0 and os.path.exists(args.out) else "w"
    fout = open(args.out, mode)
    t0 = time.time()
    for i in range(args.start, min(args.end, len(rows))):
        r = rows[i]
        rec = {"idx": i, "run_id": r.get("run_id"), "issue_key": r.get("issue_key"),
               "gold": gold_disp(r)}
        try:
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": make_prompt(r)}]
            enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                          return_dict=True).to(model.device)
            in_len = enc["input_ids"].shape[1]
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_tokens, do_sample=False,
                                     max_time=args.max_time, pad_token_id=tok.pad_token_id)
            text = tok.decode(gen[0][in_len:], skip_special_tokens=True)
        except Exception as e:
            text = f"__ERR__ {type(e).__name__}: {e}"
        rec["pred"] = parse_disposition(text)
        rec["response"] = text
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        os.fsync(fout.fileno())
        done = i - args.start + 1
        print(f"  idx {i} done ({(time.time()-t0)/done:.1f}s/ex avg)", flush=True)
    fout.close()
    print(f"GEN_SEGMENT_DONE wrote {args.out} idx[{args.start}:{args.end})", flush=True)

if __name__ == "__main__":
    main()
