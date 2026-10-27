import time, os, sys
sys.path.insert(0, os.path.expanduser("~/model-build"))
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
print("START load", flush=True)
t = time.time()
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", quantization_config=bnb,
                                             device_map="auto", trust_remote_code=True)
print("base loaded %.1fs" % (time.time() - t), flush=True)
model = PeftModel.from_pretrained(model, os.path.expanduser("~/model-build/out/raft-v2-iter2-adapter"))
model.eval()
print("adapter loaded %.1fs" % (time.time() - t), flush=True)
msgs = [{"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in one short sentence."}]
enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
print("encoded, generating...", flush=True)
t2 = time.time()
with torch.no_grad():
    gen = model.generate(**enc, max_new_tokens=64, do_sample=False, max_time=60, pad_token_id=tok.pad_token_id)
print("GEN OK %.1fs" % (time.time() - t2), flush=True)
print("OUT:", tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)[:200], flush=True)
