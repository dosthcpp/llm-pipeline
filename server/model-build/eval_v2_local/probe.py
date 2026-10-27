import json,urllib.request
item=json.loads(open("judge_batch_h1.jsonl").read().splitlines()[0])
system=open("JUDGE_RUBRIC.md").read().strip()
user="Context:\n%s\n\nGold disposition: %s\n\nAgent response:\n%s" % (item["context"],item["gold"],item["response"])
p={"model":"Qwopus3.6-35B-A3B-v1-MTP-Q4_K_M.gguf","messages":[{"role":"system","content":system},{"role":"user","content":user}],"max_tokens":600,"temperature":0}
req=urllib.request.Request("http://127.0.0.1:8020/v1/chat/completions",data=json.dumps(p).encode(),headers={"Content-Type":"application/json"})
r=json.loads(urllib.request.urlopen(req,timeout=180).read())
ch=r["choices"][0]
print("FINISH:",ch.get("finish_reason"))
print("CONTENT_LEN:",len(ch["message"].get("content") or ""))
print("REASONING_LEN:",len(ch["message"].get("reasoning_content") or ""))
print("=== CONTENT ===")
print((ch["message"].get("content") or "")[:1500])
print("=== REASONING tail ===")
print((ch["message"].get("reasoning_content") or "")[-500:])
