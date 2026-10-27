#!/usr/bin/env python3
# TON-2565 hardened v3 native gen. Root cause of the hang: eval_v2_gen.py used
# device_map="auto" which CPU-offloaded the 4B base (only ~3.8GB GPU used), and
# generate(max_time=90) was not enforced -> a single greedy item ground on CPU for
# 20+ min. Fix: pin fully to GPU (device_map={"":0}), write incrementally, and add a
# hard per-item SIGALRM backstop so no item can stall the batch.
import os, sys, json, time, signal
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
sys.path.insert(0, os.path.expanduser("~/model-build"))
import eval_v2_gen as E

DATA = os.path.expanduser("~/model-build/dataset/eval.jsonl")
ADAPTER = os.path.expanduser("~/model-build/out/sovereign-v3-contrastive")
OUT = os.path.expanduser("~/model-build/gen_v3.jsonl")
MAXTOK, PER_ITEM = 2048, 150

rows = E.build_eval_subset(DATA, 30, 30)
print(f"subset N={len(rows)}", flush=True)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", quantization_config=bnb,
                                             device_map={"": 0}, trust_remote_code=True)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()
devs = set(str(p.device) for p in model.parameters())
print("param devices:", devs, flush=True)

class TO(Exception):
    pass
signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TO()))

preds = []
t0 = time.time()
with open(OUT, "w") as fout:
    for i, r in enumerate(rows):
        msgs = [{"role": "system", "content": E.SYSTEM_PROMPT},
                {"role": "user", "content": E.make_prompt(r)}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True).to(model.device)
        in_len = enc["input_ids"].shape[1]
        text = ""
        signal.alarm(PER_ITEM)
        try:
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=MAXTOK, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            text = tok.decode(gen[0][in_len:], skip_special_tokens=True)
        except TO:
            text = "[TIMEOUT]"
            print(f"  idx{i} TIMEOUT {PER_ITEM}s key={r.get('issue_key')}", flush=True)
        finally:
            signal.alarm(0)
        rec = {"run_id": r.get("run_id"), "issue_key": r.get("issue_key"),
               "gold": E.gold_disp(r), "pred": E.parse_disposition(text), "response": text}
        preds.append(rec)
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(rows)} ({(time.time()-t0)/(i+1):.1f}s/ex)", flush=True)
print(f"wrote {OUT} ({len(preds)}); timeouts={sum(1 for p in preds if p['response']=='[TIMEOUT]')}", flush=True)
open(os.path.expanduser("~/model-build/V3_DONE"), "w").close()
print("V3_DONE touched", flush=True)
# restore prod 35B (mirror run_sft_v3.sh restore_llama): free pause + relaunch keepalive target
os.system("rm -f ~/watchdog.pause; pgrep -o llama-server >/dev/null 2>&1 || bash ~/start-llama.sh >/dev/null 2>&1 &")
print("restore_llama issued", flush=True)
