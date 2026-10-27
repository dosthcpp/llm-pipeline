#!/bin/bash
# TON-2410 option-1 isolated debug: does --reasoning-budget bound v1 runaway?
# CPU on port 8021 so production 35B (GPU 8020) is untouched. A/B: budget vs no-budget.
set -uo pipefail
BIN=~/llama.cpp/build/bin/llama-server
MODEL=~/models/sovereign-v1-128k-Q4_K_M.gguf
LOG=~/model-build/debug_v1_isolated.log
PORT=8021
: > "$LOG"

cat > /tmp/cplx_prompt.json <<JSON
{"messages":[{"role":"user","content":"You are CTO of an autonomous software company. Heartbeat wake.\n\nWAKE: issue_assigned\nISSUE TON-1234: Implement rate limiting on the public API gateway\nDESCRIPTION: The gateway has no rate limiting. A single client sent 50k req/min and degraded service. Add a token-bucket limiter: 100 req/min per API key, 429 on exceed, Retry-After header. Tests required. Code Supervisor review before merge.\nParent: TON-1200. Status: todo.\nComments:\n- [PM]: coordinate with infra on limiter state (Redis vs in-process).\n- [CEO]: high priority, customer-facing.\n\nDecide the disposition and the concrete next action. Be decisive and concise."}],"max_tokens":4000,"temperature":0.6}
JSON

run_test() {
  local label="$1"; shift
  echo "=== START $label $(date -u +%H:%M:%S) ===" >> "$LOG"
  "$BIN" --model "$MODEL" --ctx-size 8192 -ngl 0 --parallel 1 --jinja \
    --temp 0.6 --top-p 0.95 --top-k 20 "$@" \
    --port $PORT --host 127.0.0.1 >> "$LOG" 2>&1 &
  local SRV=$!
  for i in $(seq 1 40); do sleep 2; curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; done
  python3 - "$label" <<PY >> "$LOG" 2>&1
import json,urllib.request,time,sys
label=sys.argv[1]
body=open("/tmp/cplx_prompt.json","rb").read()
req=urllib.request.Request("http://127.0.0.1:$PORT/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
t0=time.time()
d=json.load(urllib.request.urlopen(req,timeout=600))
dt=time.time()-t0
c=d["choices"][0]; u=d.get("usage",{})
print(f"RESULT[{label}] finish={c.get(\"finish_reason\")} completion_tokens={u.get(\"completion_tokens\")} wall={dt:.0f}s reasoning_len={len(c[\"message\"].get(\"reasoning_content\") or \"\")} content_len={len(c[\"message\"][\"content\"] or \"\")}")
print("CONTENT:", (c["message"]["content"] or "")[:300])
PY
  kill -9 $SRV 2>/dev/null; sleep 3
  echo "=== END $label ===" >> "$LOG"
}

run_test "A_with_budget2048" --reasoning-budget 2048
run_test "B_no_budget" 
echo "ALL_DEBUG_DONE $(date -u +%H:%M:%S)" >> "$LOG"
grep "RESULT\|CONTENT\|ALL_DEBUG_DONE" "$LOG"
