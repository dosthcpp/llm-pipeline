#!/usr/bin/env python3
# TON-2565 resumable v3 native gen. Uses the ORIGINAL proven run_hf config
# (device_map="auto", max_time=90 — generated 20 items fine before one item hung).
# Hang-proofing the original's two weaknesses:
#   1) all-or-nothing output  -> append each item immediately (resume across restarts).
#   2) a C/CUDA-level hang that SIGALRM can't interrupt -> a WATCHDOG THREAD that
#      os._exit()s on stall (runs independently of the blocked main thread), records
#      the stuck index to a skiplist; the bash runner restarts and resumes past it.
import os, sys, json, time, threading
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
sys.path.insert(0, os.path.expanduser("~/model-build"))
import eval_v2_gen as E

HOME = os.path.expanduser("~")
DATA = f"{HOME}/model-build/dataset/eval.jsonl"
ADAPTER = f"{HOME}/model-build/out/sovereign-v3-contrastive"
OUT = f"{HOME}/model-build/gen_v3.jsonl"
SKIP = f"{HOME}/model-build/gen_v3.skip"
MAXTOK, STALL, LOAD_TIMEOUT = 2048, 200, 480

rows = E.build_eval_subset(DATA, 30, 30)
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        try: done.add(json.loads(l)["run_id"])
        except Exception: pass
skip = set()
if os.path.exists(SKIP):
    skip = {int(x) for x in open(SKIP).read().split() if x.strip().isdigit()}
print(f"subset N={len(rows)} done={len(done)} skip={sorted(skip)}", flush=True)

state = {"ready": False, "cur": -1, "last": time.time()}
def watchdog():
    while True:
        time.sleep(10)
        now = time.time()
        if not state["ready"]:
            if now - state["last"] > LOAD_TIMEOUT:
                print("WATCHDOG: load timeout -> exit 8", flush=True); os._exit(8)
            continue
        if state["cur"] >= 0 and now - state["last"] > STALL:
            with open(SKIP, "a") as s: s.write(f"{state['cur']}\n")
            print(f"WATCHDOG: stall idx{state['cur']} >{STALL}s -> skip + exit 7", flush=True)
            os._exit(7)
threading.Thread(target=watchdog, daemon=True).start()

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", quantization_config=bnb,
                                             device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()
state["last"] = time.time(); state["ready"] = True
print("model ready", flush=True)

fout = open(OUT, "a")
t0, n_new = time.time(), 0
for i, r in enumerate(rows):
    if r.get("run_id") in done or i in skip:
        continue
    state["cur"] = i; state["last"] = time.time()
    msgs = [{"role": "system", "content": E.SYSTEM_PROMPT},
            {"role": "user", "content": E.make_prompt(r)}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True).to(model.device)
    in_len = enc["input_ids"].shape[1]
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=MAXTOK, do_sample=False, max_time=90,
                             pad_token_id=tok.pad_token_id)
    text = tok.decode(gen[0][in_len:], skip_special_tokens=True)
    rec = {"run_id": r.get("run_id"), "issue_key": r.get("issue_key"),
           "gold": E.gold_disp(r), "pred": E.parse_disposition(text), "response": text}
    fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
    done.add(r.get("run_id")); n_new += 1; state["cur"] = -1; state["last"] = time.time()
    if n_new % 5 == 0:
        print(f"  +{n_new} idx{i} total={len(done)} ({(time.time()-t0)/n_new:.1f}s/new)", flush=True)
fout.close()
remaining = [i for i, r in enumerate(rows) if r.get("run_id") not in done and i not in skip]
if remaining:
    print(f"INCOMPLETE remaining={remaining[:6]} -> exit 1", flush=True); os._exit(1)
print(f"ALL DONE total={len(done)} skipped={sorted(skip)}", flush=True)
open(f"{HOME}/model-build/V3_DONE", "w").close()
os.system("rm -f ~/watchdog.pause; pgrep -o llama-server >/dev/null 2>&1 || (bash ~/start-llama.sh >/dev/null 2>&1 &)")
print("V3_DONE + restore_llama issued", flush=True)
os._exit(0)
