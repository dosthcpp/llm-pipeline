#!/usr/bin/env python3
"""P0.3.1 — Redesigned eval: NATIVE-format next-action generation. TON-2397.

Unlike P0.3's eval_harness.py, this does NOT append a triage directive and does NOT
feed final_text. Each model gets the same native prompt it was trained/served on
(system + rendered wake/issue context) and emits its natural next-action response.
Two purposes:
  1. Produce candidate responses for the blinded LLM-judge next-action-quality eval
     (the redesigned target — see EVAL_REDESIGN_P031.md).
  2. v0 disposition-collapse diagnosis (fix #2): report the disposition distribution
     v0 emits in its NATIVE format (no triage directive) vs the P0.3 triaged run, to
     separate prompt-off-distribution (H1) from a real learned bias (H2).

Faithful to training: imports render_user / SYSTEM_PROMPT from sft_train (build_text
format), same eval subset selection logic as eval_harness (deterministic, capped).

Usage (on Atlas, venv /home/tony/sovereign-model-env):
  python eval_v2_gen.py --mode teacher --out gen_teacher.jsonl            # live llama-server :8020
  python eval_v2_gen.py --mode hf --adapter "" --out gen_base.jsonl       # needs free GPU
  python eval_v2_gen.py --mode hf --adapter ~/model-build/out/sovereign-v0-adapter --out gen_v0.jsonl
  python eval_v2_gen.py --dist gen_v0.jsonl gen_teacher.jsonl gen_base.jsonl   # disposition dist only
"""
import argparse, json, os, re, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sft_train import render_user, SYSTEM_PROMPT, _as_dict, load_jsonl

VALID = ["done", "in_progress", "blocked", "in_review", "todo"]
EXCLUDE_ISSUES = {"TON-2050"}  # the single train/eval issue_key leak

# TON-2773-B: frontier-API candidates. Tags MUST equal cost_model.json frontier_api keys
# so judge_key model tag -> quality dict lines up with the priced rows in cost_quality_baseline.
FRONTIER_MODELS = {
    "opus-4-8":   "claude-opus-4-8",
    "sonnet-4-6": "claude-sonnet-4-6",
    "haiku-4-5":  "claude-haiku-4-5-20251001",
}


def gold_disp(r):
    return _as_dict(r.get("output")).get("disposition")


def build_eval_subset(path, done_cap, blocked_cap):
    """Same deterministic, stratified selection as P0.3 eval_harness for comparability."""
    rows = [r for r in load_jsonl(path) if r.get("issue_key") not in EXCLUDE_ISSUES]
    rows = [r for r in rows if gold_disp(r) in VALID]
    rows.sort(key=lambda r: r.get("run_id") or "")
    kept, n_done, n_blk = [], 0, 0
    for r in rows:
        d = gold_disp(r)
        if d == "done":
            if n_done >= done_cap:
                continue
            n_done += 1
        elif d == "blocked":
            if n_blk >= blocked_cap:
                continue
            n_blk += 1
        kept.append(r)
    return kept


def make_prompt(r):
    """NATIVE: render_user only — NO triage directive, NO final_text. Same as train user turn."""
    return render_user(_as_dict(r.get("input")))


def parse_disposition(text):
    if not text:
        return None
    m = list(re.finditer(r"disposition\s*[:=]\s*`?([a-z_]+)`?", text, re.I))
    if m:
        cand = m[-1].group(1).lower().strip()
        if cand in VALID:
            return cand
    tail = text[-500:].lower()
    hits = [(tail.rfind(v), v) for v in VALID if v in tail]
    hits = [(i, v) for i, v in hits if i >= 0]
    return max(hits)[1] if hits else None


def run_teacher(rows, url, model, max_tokens, out, repeat_penalty=None):
    # TON-2814 S1 (decode-only, frozen-faithful): SYSTEM_PROMPT/make_prompt are UNCHANGED so this
    # stays comparable to the teacher anchor + frontier table. Only the decode budget/params move:
    #   --max-tokens 4096   recovers truncated tails (agentic output emits disposition LAST; 11/13
    #                       no-parse cases in gen_v2_8b.jsonl were cap-truncated, not genuine).
    #   --repeat-penalty R  (llama.cpp server) kills the repetition loops seen in the truncated tails.
    # repeat_penalty=None keeps STRICT frozen greedy (temp=0.0, no penalty). When set, records are
    # tagged decode:"s1-decode-relaxed" so the deviation is never silent (EVAL_FROZEN_SPEC §2).
    decode_tag = "frozen-greedy" if repeat_penalty is None else "s1-decode-relaxed"
    preds = []
    for i, r in enumerate(rows):
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": make_prompt(r)},
            ],
            "temperature": 0.0, "max_tokens": max_tokens, "stream": False,
        }
        if repeat_penalty is not None:
            payload["repeat_penalty"] = repeat_penalty  # llama.cpp OpenAI-compat server extension
        body = json.dumps(payload).encode()
        text = ""
        for attempt in range(4):  # survive watchdog kill + systemd restart of llama-server
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    j = json.loads(resp.read())
                    msg = j["choices"][0]["message"]
                    text = msg.get("content") or ""
                    if not text.strip():  # reasoning models sometimes empty content channel
                        text = msg.get("reasoning_content") or ""
                break
            except Exception as e:
                text = f"__ERR__ {e}"
                if attempt < 3:
                    time.sleep(20)
        preds.append({"run_id": r.get("run_id"), "issue_key": r.get("issue_key"),
                      "gold": gold_disp(r), "pred": parse_disposition(text), "response": text,
                      "decode": decode_tag})
        if (i + 1) % 10 == 0:
            print(f"  teacher {i+1}/{len(rows)}", flush=True)
    json.dump(preds, open(out, "w"), indent=0)
    print(f"wrote {out} ({len(preds)})")


def run_cli(rows, api_model, max_tokens, out, claude_bin="claude", concurrency=1):
    """TON-2773-B (boss: "claude -p 로 하면 안되남") — frontier candidate via the Claude Code CLI
    in headless print mode. Uses the LOCAL authenticated session, so it needs NO board-gated API
    key. Same frozen native prompt (SYSTEM_PROMPT + render_user) and same parser; output schema
    identical to run_teacher/run_hf/run_api.

    DECODE CAVEAT (honest, documented in EVAL_FROZEN_SPEC §2): `claude -p` exposes no temperature
    flag, so this is the CLI default sampling, NOT the frozen greedy temp=0.0. Results carry a
    `decode:"cli-default"` tag; treat CLI-mode numbers as a directional like-for-like read, and
    prefer --mode api (greedy) for a strict frozen-spec certification when a key is available.
    """
    import subprocess
    if api_model not in FRONTIER_MODELS:
        sys.exit(f"ERROR: --api-model must be one of {list(FRONTIER_MODELS)}")
    full_model = FRONTIER_MODELS[api_model]
    sys_path = out + ".sysprompt.txt"
    with open(sys_path, "w") as f:
        f.write(SYSTEM_PROMPT)

    # CHECKPOINT + RESUME: long cli runs (348 claude -p calls) outlive a single heartbeat and
    # get torn down. Persist after EVERY item (atomic tmp+rename) and skip run_ids already done
    # with a non-error response, so a restart continues instead of losing all work.
    by_id = {}
    if os.path.exists(out):
        try:
            for x in json.load(open(out)):
                by_id[x.get("run_id")] = x
        except Exception:
            by_id = {}

    def is_done(x):
        # "done" = has a real (non-error) response. pred (parsed disposition) is NOT required:
        # frontier models on the native prompt often don't emit a literal "Disposition:" keyword,
        # but the blinded judge scores the RESPONSE text, so an unparsed-pred row is still usable.
        r = str(x.get("response", "")) if x else ""
        return bool(r) and not r.startswith("__ERR__")

    def flush():
        tmp = out + ".tmp"
        json.dump([by_id[r.get("run_id")] for r in rows if r.get("run_id") in by_id],
                  open(tmp, "w"), indent=0)
        os.replace(tmp, out)

    done0 = sum(1 for r in rows if is_done(by_id.get(r.get("run_id"))))
    if done0:
        print(f"  resume: {done0}/{len(rows)} already done in {out}", flush=True)
    todo = [r for r in rows if not is_done(by_id.get(r.get("run_id")))]

    def gen_one(r):
        # --setting-sources "" + --strict-mcp-config: do NOT load the session's user/project
        # settings or MCP servers (serena/unity/gmail/…) on each invocation — that startup was
        # the bottleneck (minutes/call → ~4s/call). Pure single-shot generation needs neither.
        cmd = [claude_bin, "-p", "--model", full_model,
               "--exclude-dynamic-system-prompt-sections",
               "--setting-sources", "", "--strict-mcp-config",
               "--system-prompt-file", sys_path]
        text = ""
        for attempt in range(3):
            try:
                p = subprocess.run(cmd, input=make_prompt(r), capture_output=True,
                                   text=True, timeout=180)
                text = (p.stdout or "").strip()
                if not text and p.returncode != 0:
                    text = f"__ERR__ rc={p.returncode} {(p.stderr or '')[:200]}"
                    if attempt < 2:
                        time.sleep(10); continue
                break
            except Exception as e:
                text = f"__ERR__ {e}"
                if attempt < 2:
                    time.sleep(10)
        return {"run_id": r.get("run_id"), "issue_key": r.get("issue_key"),
                "gold": gold_disp(r), "pred": parse_disposition(text),
                "response": text, "decode": "cli-default"}

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    lock = threading.Lock()
    n = 0
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        futs = {ex.submit(gen_one, r): r for r in todo}
        for fut in as_completed(futs):
            rec = fut.result()
            with lock:
                by_id[rec["run_id"]] = rec
                flush()  # checkpoint after each completion
                n += 1
                if n % 10 == 0 or n == len(todo):
                    nd = sum(1 for rr in rows if is_done(by_id.get(rr.get("run_id"))))
                    print(f"  {api_model}/cli {n}/{len(todo)} new (done={nd}/{len(rows)})", flush=True)
    flush()
    nd = sum(1 for r in rows if is_done(by_id.get(r.get("run_id"))))
    print(f"wrote {out} ({nd}/{len(rows)} done)  [decode=cli-default, model={full_model}]")


def run_api(rows, api_model, max_tokens, out, concurrency=1):
    """TON-2773-B: frontier candidate on the SAME frozen native prompt/decode/parser.

    Anthropic Messages API. Key from env only (ANTHROPIC_API_KEY, fallback FRONTIER_API_KEY)
    — never hardcoded; fail-closed if absent. greedy (temperature=0.0) to match the frozen
    decode of every other candidate. Zero-code-change unblock: provision the board-gated key,
    set the env var, run. Output schema identical to run_teacher/run_hf so it flows straight
    into eval_v2_judge_prep.py (--gen <tag>=gen_<tag>.jsonl).
    """
    if api_model not in FRONTIER_MODELS:
        sys.exit(f"ERROR: --api-model must be one of {list(FRONTIER_MODELS)}")
    key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("FRONTIER_API_KEY")
    if not key:
        sys.exit("ERROR: no frontier API key in env (ANTHROPIC_API_KEY / FRONTIER_API_KEY). "
                 "BLOCKED on board-gated key — TON-2783. Set the env var and re-run; no code change.")
    full_model = FRONTIER_MODELS[api_model]
    url = "https://api.anthropic.com/v1/messages"
    headers = {"Content-Type": "application/json", "x-api-key": key,
               "anthropic-version": "2023-06-01"}

    # CHECKPOINT + RESUME (same as run_cli): heartbeat teardown must NOT re-pay for completed
    # items — boss said use the key sparingly. Persist after each item; skip done run_ids.
    by_id = {}
    if os.path.exists(out):
        try:
            for x in json.load(open(out)):
                by_id[x.get("run_id")] = x
        except Exception:
            by_id = {}

    def is_done(x):
        r = str(x.get("response", "")) if x else ""
        return bool(r) and not r.startswith("__ERR__")

    def flush():
        tmp = out + ".tmp"
        json.dump([by_id[r.get("run_id")] for r in rows if r.get("run_id") in by_id],
                  open(tmp, "w"), indent=0)
        os.replace(tmp, out)

    def gen_one(r):
        # NOTE: newer models (Opus 4.8 / Sonnet 4.6) DEPRECATED the `temperature` param — the API
        # 400s if it's sent. So we can't pin greedy temp=0 here; decode is the model default,
        # still bounded by max_tokens. Tagged decode:"api-default" (honest; less off-spec than cli
        # since it IS token-bounded). See EVAL_FROZEN_SPEC §2.
        body = json.dumps({
            "model": full_model,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": make_prompt(r)}],
            "max_tokens": max_tokens,
        }).encode()
        text = ""
        for attempt in range(4):  # survive 429/5xx with backoff
            req = urllib.request.Request(url, data=body, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    j = json.loads(resp.read())
                    text = "".join(b.get("text", "") for b in j.get("content", [])
                                   if b.get("type") == "text")
                break
            except Exception as e:
                text = f"__ERR__ {e}"
                if attempt < 3:
                    time.sleep(20 * (attempt + 1))
        return {"run_id": r.get("run_id"), "issue_key": r.get("issue_key"),
                "gold": gold_disp(r), "pred": parse_disposition(text), "response": text,
                "decode": "api-default"}

    todo = [r for r in rows if not is_done(by_id.get(r.get("run_id")))]
    done0 = len(rows) - len(todo)
    if done0:
        print(f"  resume: {done0}/{len(rows)} already done in {out}", flush=True)
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    lock = threading.Lock()
    n = 0
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        futs = {ex.submit(gen_one, r): r for r in todo}
        for fut in as_completed(futs):
            rec = fut.result()
            with lock:
                by_id[rec["run_id"]] = rec
                flush()
                n += 1
                if n % 10 == 0 or n == len(todo):
                    nd = sum(1 for rr in rows if is_done(by_id.get(rr.get("run_id"))))
                    print(f"  {api_model}/api {n}/{len(todo)} new (done={nd}/{len(rows)})", flush=True)
    flush()
    nd = sum(1 for r in rows if is_done(by_id.get(r.get("run_id"))))
    print(f"wrote {out} ({nd}/{len(rows)} done)  [decode=api-default, model={full_model}]")


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
    preds, t0 = [], time.time()
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
        preds.append({"run_id": r.get("run_id"), "issue_key": r.get("issue_key"),
                      "gold": gold_disp(r), "pred": parse_disposition(text), "response": text})
        if (i + 1) % 10 == 0:
            print(f"  {tag} {i+1}/{len(rows)} ({(time.time()-t0)/(i+1):.1f}s/ex)", flush=True)
    json.dump(preds, open(out, "w"), indent=0)
    print(f"wrote {out} ({len(preds)})")


def dist(files):
    """Disposition distribution per model + gold — the v0-collapse diagnosis."""
    from collections import Counter
    for f in files:
        preds = json.load(open(f))
        name = re.sub(r".*gen_|\.jsonl", "", f)
        pc = Counter(p["pred"] for p in preds)
        gc = Counter(p["gold"] for p in preds)
        n = len(preds)
        blk = pc.get("blocked", 0)
        print(f"\n== {name}  (N={n}) ==")
        print(f"   pred dist: {dict(pc)}")
        print(f"   gold dist: {dict(gc)}")
        print(f"   blocked share: pred {blk}/{n}={blk/n:.1%}  vs gold "
              f"{gc.get('blocked',0)}/{n}={gc.get('blocked',0)/n:.1%}")
        none = sum(1 for p in preds if not p["pred"])
        print(f"   unparsed: {none}/{n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["teacher", "hf", "api", "cli"])
    ap.add_argument("--api-model", default="opus-4-8",
                    help="frontier tag: opus-4-8 | sonnet-4-6 | haiku-4-5 (TON-2773-B)")
    ap.add_argument("--claude-bin", default="claude", help="claude CLI path for --mode cli")
    ap.add_argument("--concurrency", type=int, default=1, help="parallel claude -p workers (cli mode)")
    ap.add_argument("--data", default=os.path.expanduser("~/model-build/dataset/eval.jsonl"))
    ap.add_argument("--adapter", default="")
    ap.add_argument("--base-model", default="Qwen/Qwen3-4B")
    ap.add_argument("--url", default="http://127.0.0.1:8020/v1/chat/completions")
    ap.add_argument("--teacher-model", default="Qwopus3.6-35B-A3B-v1-MTP-Q4_K_M.gguf")
    ap.add_argument("--max-tokens", type=int, default=2048)  # native agentic output needs room
    ap.add_argument("--repeat-penalty", type=float, default=None,
                    help="TON-2814 S1: llama.cpp repeat_penalty for local GGUF (--mode teacher). "
                         "Default None = strict frozen greedy; e.g. 1.1 kills truncation-tail loops.")
    ap.add_argument("--done-cap", type=int, default=30)
    ap.add_argument("--blocked-cap", type=int, default=30)
    ap.add_argument("--out", default="gen.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dist", nargs="+")
    args = ap.parse_args()
    if args.dist:
        dist(args.dist); return
    rows = build_eval_subset(args.data, args.done_cap, args.blocked_cap)
    if args.limit:
        rows = rows[:args.limit]
    from collections import Counter
    print(f"eval subset N={len(rows)}  dist={dict(Counter(gold_disp(r) for r in rows))}")
    if args.mode == "teacher":
        run_teacher(rows, args.url, args.teacher_model, args.max_tokens, args.out, args.repeat_penalty)
    elif args.mode == "hf":
        run_hf(rows, args.base_model, args.adapter, args.max_tokens, args.out)
    elif args.mode == "api":
        run_api(rows, args.api_model, args.max_tokens, args.out, args.concurrency)
    elif args.mode == "cli":
        run_cli(rows, args.api_model, args.max_tokens, args.out, args.claude_bin, args.concurrency)
    else:
        sys.exit("need --mode or --dist")


if __name__ == "__main__":
    main()
