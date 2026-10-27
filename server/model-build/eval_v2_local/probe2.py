import json,urllib.request,time
item=json.loads(open("judge_batch_h1.jsonl").read().splitlines()[0])
system=open("JUDGE_RUBRIC.md").read().strip()
user="Context:\n%s\n\nGold disposition: %s\n\nAgent response:\n%s" % (item["context"],item["gold"],item["response"])
for mt in [2048]:
    p={"model":"Qwopus3.6-35B-A3B-v1-MTP-Q4_K_M.gguf","messages":[{"role":"system","content":system},{"role":"user","content":user}],"max_tokens":mt,"temperature":0}
    req=urllib.request.Request("http://127.0.0.1:8020/v1/chat/completions",data=json.dumps(p).encode(),headers={"Content-Type":"application/json"})
    t0=time.time();r=json.loads(urllib.request.urlopen(req,timeout=300).read());dt=time.time()-t0
    ch=r["choices"][0]
    print("max_tokens=%d FINISH=%s lat=%.1fs reasoning_len=%d content_len=%d"%(mt,ch.get("finish_reason"),dt,len(ch["message"].get("reasoning_content") or ""),len(ch["message"].get("content") or "")))
    print("CONTENT:",(ch["message"].get("content") or "")[:600])
