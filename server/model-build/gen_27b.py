#!/usr/bin/env python3
"""TON-2542 — Qwopus3.6-27B-Coder (Qwen3_5 hybrid VL+SSM) zero-shot gen.

Same certified subset/prompt/parse as eval_v2_gen, but the model is a hybrid
linear-attention + Mamba/SSM VL model. bitsandbytes 4-bit corrupts the conv/SSM/
gated-norm kernels (CUDA launch failure), so we SKIP-quantize those modules
(keep bf16) and only 4-bit the standard projections. Smoke-tests 1 item first so a
failed retry costs minimal teacher downtime.

Usage (GPU free): python gen_27b.py --model ~/models/Qwopus3.6-27B-Coder --smoke
                  python gen_27b.py --model ~/models/Qwopus3.6-27B-Coder --limit 40 --out gen_qwopus27b.jsonl
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_v2_gen import build_eval_subset, make_prompt, parse_disposition, gold_disp
from sft_train import SYSTEM_PROMPT

SKIP = ["linear_attn", "conv", "mamba", "A_log", "dt", "norm", "visual", "vision",
        "mtp", "lm_head", "embed_tokens"]

def load(model_dir):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    free, total = torch.cuda.mem_get_info()
    print(f"free VRAM {free/1e9:.1f}/{total/1e9:.1f} GB", flush=True)
    if free < 16e9:
        sys.exit("ERROR: <16 GB free VRAM — teacher still up?")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True,
                             llm_int8_skip_modules=SKIP)
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="eager")
    model.eval()
    return tok, model

def gen_one(tok, model, r, max_tokens):
    import torch
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_prompt(r)}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True).to(model.device)
    in_len = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][in_len:], skip_special_tokens=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=os.path.expanduser("~/model-build/dataset/eval.jsonl"))
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--done-cap", type=int, default=10)
    ap.add_argument("--blocked-cap", type=int, default=10)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="gen_qwopus27b.jsonl")
    a = ap.parse_args()
    tok, model = load(a.model)
    if a.smoke:
        rows = build_eval_subset(a.data, a.done_cap, a.blocked_cap)[:1]
        t = time.time()
        txt = gen_one(tok, model, rows[0], 32)
        print(f"SMOKE OK ({time.time()-t:.1f}s) pred={parse_disposition(txt)!r} sample={txt[:160]!r}", flush=True)
        return
    rows = build_eval_subset(a.data, a.done_cap, a.blocked_cap)
    if a.limit:
        rows = rows[:a.limit]
    print(f"gen N={len(rows)}", flush=True)
    preds, t0 = [], time.time()
    for i, r in enumerate(rows):
        try:
            txt = gen_one(tok, model, r, a.max_tokens)
        except Exception as e:
            txt = f"__ERR__ {e}"
        preds.append({"run_id": r.get("run_id"), "issue_key": r.get("issue_key"),
                      "gold": gold_disp(r), "pred": parse_disposition(txt), "response": txt})
        if (i+1) % 5 == 0:
            print(f"  {i+1}/{len(rows)} ({(time.time()-t0)/(i+1):.1f}s/ex)", flush=True)
    json.dump(preds, open(a.out, "w"), indent=0)
    print(f"wrote {a.out} ({len(preds)})", flush=True)

if __name__ == "__main__":
    main()
