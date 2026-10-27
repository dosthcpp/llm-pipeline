#!/usr/bin/env python3
"""P0.3 — Domain eval harness: sovereign-v0 vs base vs teacher. TON-2395 (parent TON-2391).

Honest, held-out disposition/triage eval. We measure: given a real wake/issue
context from the P0.1 eval split (frozen issues, ZERO run_id overlap with train),
does each model pick the SAME final disposition the production teacher actually took?

Gold label = output.disposition recorded on the held-out run (the teacher's real
decision). Each model is scored on agreement with that gold label. Same prompt,
same decoding, same parser for all three — no per-model tuning, no benchmark gaming.

Models:
  base    : Qwen/Qwen3-4B, no adapter            (--mode hf --adapter "")
  v0      : Qwen/Qwen3-4B + sovereign-v0-adapter (--mode hf --adapter <path>)
  teacher : live Qwopus served by llama-server    (--mode teacher)  [the distill source]

The teacher is the ceiling reference: re-querying it on held-out inputs measures
its self-agreement with its own recorded decisions (not trivially 100% — sampling,
format drift). v0's job is to approach teacher-level agreement while beating base.

Usage:
  python eval_harness.py --mode teacher --out preds_teacher.jsonl
  python eval_harness.py --mode hf --adapter ""  --out preds_base.jsonl   # needs free GPU
  python eval_harness.py --mode hf --adapter ~/model-build/out/sovereign-v0-adapter --out preds_v0.jsonl
  python eval_harness.py --score preds_teacher.jsonl preds_base.jsonl preds_v0.jsonl
"""
import argparse, json, os, re, sys, urllib.request, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sft_train import render_user, SYSTEM_PROMPT, _as_dict, load_jsonl  # faithful to training

VALID = ["done", "in_progress", "blocked", "in_review", "todo"]
EXCLUDE_ISSUES = {"TON-2050"}  # the single train/eval issue_key leak — dropped for honesty
# Symmetric triage-classifier instruction appended to ALL models (no per-model tuning).
# Forbids agentic execution: the recorded disposition is a triage judgment, and the
# reasoning models otherwise burn the token budget "doing the task" before deciding.
FINAL_INSTR = (
    "\n\n---\nThis is a TRIAGE decision, not an execution task. Do NOT write code, "
    "run commands, or perform the work. Based only on the context above, decide the "
    "single most likely correct disposition for this issue right now. Reply with ONLY "
    "one final line, exactly: Disposition: <one of the valid options>")

def gold_disp(r):
    return _as_dict(r.get("output")).get("disposition")

def build_eval_subset(path, done_cap=45, blocked_cap=45, seed_sort=True):
    rows = load_jsonl(path)
    rows = [r for r in rows if (r.get("issue_key") not in EXCLUDE_ISSUES)]
    rows = [r for r in rows if gold_disp(r) in VALID]  # concrete gold only
    if seed_sort:
        rows.sort(key=lambda r: r.get("run_id") or "")  # deterministic, reproducible
    kept, n_done, n_blk = [], 0, 0
    for r in rows:
        d = gold_disp(r)
        if d == "done":
            if n_done >= done_cap: continue
            n_done += 1
        elif d == "blocked":
            if n_blk >= blocked_cap: continue
            n_blk += 1
        kept.append(r)
    return kept

def make_prompt(r):
    user = render_user(_as_dict(r.get("input")))
    opts = ", ".join(VALID)
    if "Valid dispositions:" not in user:
        user += f"\n\nValid dispositions: {opts}"
    return user + FINAL_INSTR

def parse_disposition(text):
    if not text:
        return None
    m = list(re.finditer(r"disposition\s*[:=]\s*`?([a-z_]+)`?", text, re.I))
    if m:
        cand = m[-1].group(1).lower().strip()
        if cand in VALID:
            return cand
    # fallback: first valid label keyword appearing in the tail of the reply
    tail = text[-400:].lower()
    hits = [(tail.rfind(v), v) for v in VALID if v in tail]
    hits = [(i, v) for i, v in hits if i >= 0]
    if hits:
        return max(hits)[1]
    return None

# ---------- teacher (llama-server HTTP) ----------
def run_teacher(rows, url, model, max_tokens, out):
    preds = []
    for i, r in enumerate(rows):
        body = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": make_prompt(r)},
            ],
            "temperature": 0.0, "max_tokens": max_tokens, "stream": False,
        }).encode()
        text = ""
        for attempt in range(4):  # survive watchdog kill + systemd restart of llama-server
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=240) as resp:
                    j = json.loads(resp.read())
                    msg = j["choices"][0]["message"]
                    text = msg.get("content") or ""
                    if not parse_disposition(text):  # content empty/truncated -> use think channel
                        rc = msg.get("reasoning_content") or ""
                        if parse_disposition(rc):
                            text = rc
                break
            except Exception as e:
                text = f"__ERR__ {e}"
                if attempt < 3:
                    time.sleep(20)  # wait for systemd to bring llama-server back
        pred = parse_disposition(text)
        preds.append({"run_id": r.get("run_id"), "issue_key": r.get("issue_key"),
                      "gold": gold_disp(r), "pred": pred, "raw_tail": text[-200:]})
        if (i + 1) % 10 == 0:
            print(f"  teacher {i+1}/{len(rows)}", flush=True)
    json.dump(preds, open(out, "w"), indent=0)
    print(f"wrote {out} ({len(preds)})")

# ---------- hf (base / v0 via transformers+peft) ----------
def run_hf(rows, base_model, adapter, max_tokens, out):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    free, total = torch.cuda.mem_get_info()
    print(f"free VRAM {free/1e9:.1f}/{total/1e9:.1f} GB")
    if free < 6e9:
        sys.exit("ERROR: <6 GB free VRAM — stop llama.service before hf eval.")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb,
                                                 device_map="auto", trust_remote_code=True)
    tag = "base"
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        tag = "v0"
    model.eval()
    preds = []
    t0 = time.time()
    for i, r in enumerate(rows):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": make_prompt(r)}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True).to(model.device)
        in_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        text = tok.decode(gen[0][in_len:], skip_special_tokens=True)
        pred = parse_disposition(text)
        preds.append({"run_id": r.get("run_id"), "issue_key": r.get("issue_key"),
                      "gold": gold_disp(r), "pred": pred, "raw_tail": text[-200:]})
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  {tag} {i+1}/{len(rows)} ({el/(i+1):.1f}s/ex)", flush=True)
    json.dump(preds, open(out, "w"), indent=0)
    print(f"wrote {out} ({len(preds)})")

# ---------- scoring ----------
def score(files):
    from collections import Counter, defaultdict
    sets = {}
    for f in files:
        name = re.sub(r".*preds_|\.jsonl", "", f)
        sets[name] = {p["run_id"]: p for p in json.load(open(f))}
    common = set.intersection(*[set(s) for s in sets.values()])
    gold = {rid: next(iter(sets.values()))[rid]["gold"] for rid in common}
    maj = Counter(gold.values()).most_common(1)[0]
    N = len(common)
    print(f"\n== EVAL: N={N} held-out runs (frozen issues, 0 run_id leak, TON-2050 excluded) ==")
    print(f"   majority-class baseline: always '{maj[0]}' = {maj[1]}/{N} = {maj[1]/N:.1%}\n")
    def f1(name):
        s = sets[name]
        tp = sum(1 for rid in common if s[rid]["pred"] == gold[rid])
        per = defaultdict(lambda: [0, 0, 0])  # tp, pred_pos, gold_pos
        for rid in common:
            g, p = gold[rid], s[rid]["pred"]
            per[g][2] += 1
            if p in VALID: per[p][1] += 1
            if p == g: per[g][0] += 1
        f1s = []
        for c in VALID:
            t, pp, gp = per[c]
            prec = t / pp if pp else 0.0
            rec = t / gp if gp else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
        unp = sum(1 for rid in common if s[rid]["pred"] is None)
        return tp / N, sum(f1s) / len(VALID), unp
    print(f"{'model':<10}{'accuracy':>10}{'macro-F1':>10}{'unparsed':>10}")
    accs = {}
    for name in sets:
        a, mf1, unp = f1(name)
        accs[name] = a
        print(f"{name:<10}{a:>10.1%}{mf1:>10.3f}{unp:>10}")
    # win-rate (pairwise per-example agreement advantage)
    print("\n== win-rate (per-example correctness) ==")
    def winrate(a, b):
        sa, sb = sets[a], sets[b]
        aw = bw = tie = 0
        for rid in common:
            ca = sa[rid]["pred"] == gold[rid]
            cb = sb[rid]["pred"] == gold[rid]
            if ca and not cb: aw += 1
            elif cb and not ca: bw += 1
            else: tie += 1
        print(f"  {a} vs {b}: {a} wins {aw}, {b} wins {bw}, tie {tie}  -> {a} net {aw-bw:+d}")
    if "v0" in sets and "base" in sets: winrate("v0", "base")
    if "v0" in sets and "teacher" in sets: winrate("v0", "teacher")
    return accs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["teacher", "hf"])
    ap.add_argument("--data", default=os.path.expanduser("~/model-build/dataset/eval.jsonl"))
    ap.add_argument("--adapter", default="")
    ap.add_argument("--base-model", default="Qwen/Qwen3-4B")
    ap.add_argument("--url", default="http://127.0.0.1:8020/v1/chat/completions")
    ap.add_argument("--teacher-model", default="Qwopus3.6-35B-A3B-v1-MTP-Q4_K_M.gguf")
    ap.add_argument("--max-tokens", type=int, default=1024)  # reasoning model needs think+answer room
    ap.add_argument("--done-cap", type=int, default=45)
    ap.add_argument("--blocked-cap", type=int, default=45)
    ap.add_argument("--out", default="preds.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="smoke-test: only first N rows")
    ap.add_argument("--score", nargs="+")
    args = ap.parse_args()
    if args.score:
        score(args.score); return
    rows = build_eval_subset(args.data, args.done_cap, args.blocked_cap)
    if args.limit:
        rows = rows[:args.limit]
    from collections import Counter
    print(f"eval subset N={len(rows)}  dist={dict(Counter(gold_disp(r) for r in rows))}")
    if args.mode == "teacher":
        run_teacher(rows, args.url, args.teacher_model, args.max_tokens, args.out)
    elif args.mode == "hf":
        run_hf(rows, args.base_model, args.adapter, args.max_tokens, args.out)
    else:
        sys.exit("need --mode or --score")

if __name__ == "__main__":
    main()
