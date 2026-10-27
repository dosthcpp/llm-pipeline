import json
p=json.load(open("/home/tony/model-build/preds_v0.jsonl"))
print("preds len:", len(p))
print("preds[0] keys:", list(p[0].keys()))
import pprint; pprint.pprint({k:str(v)[:120] for k,v in p[0].items()})
rows=[json.loads(l) for l in open("/home/tony/model-build/dataset/eval.jsonl") if l.strip()]
print("\neval len:", len(rows))
print("eval[0] top keys:", list(rows[0].keys()))
out=rows[0].get("output",{})
if isinstance(out,dict):
    print("output keys:", list(out.keys()))
    print("disposition:", out.get("disposition"))
else:
    print("output type:", type(out), str(out)[:200])
