#!/usr/bin/env python3
"""P0.1 — distillation dataset extractor (v0 scaffold).  TON-2393.

Pulls succeeded heartbeat runs from the embedded Paperclip PG and emits JSONL
distillation candidates. v0 uses the *structured* signal (context_snapshot +
result_json) which is clean; the raw terminal transcript (run-logs/*.ndjson) is
noisy TUI output and is referenced by path for an optional later reasoning-trace pass.

Source confirmed 2026-06-15: heartbeat_runs has 10,058 succeeded rows; transcripts at
  <runtime>/instances/default/data/run-logs/<companyId>/<agentId>/<runId>.ndjson  (=log_ref)

Usage:
  python extract_dataset.py --limit 50 --out /tmp/smoke.jsonl        # smoke test
  python extract_dataset.py --out train_raw.jsonl                    # full pull

TODO (tracked on TON-2393): join run->issue->produced-comment for a clean
(issue context -> agent comment/disposition) target; quality filter (drop churn /
looping / empty); secret redaction; train/eval split.
"""
import argparse
import json
import os
import subprocess

PG = dict(host="127.0.0.1", port="54329", user="paperclip", db="paperclip", pw="paperclip")
LOG_ROOT = "/Volumes/Data/tony_ai_lab/paperclip-runtime/instances/default/data/run-logs"


def psql(sql: str) -> str:
    env = dict(os.environ, PGPASSWORD=PG["pw"])
    cmd = ["psql", "-h", PG["host"], "-p", PG["port"], "-U", PG["user"],
           "-d", PG["db"], "-t", "-A", "-F", "\x1f", "-c", sql]
    return subprocess.run(cmd, env=env, capture_output=True, text=True, check=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lim = f"limit {args.limit}" if args.limit else ""
    sql = (
        "select id, agent_id, started_at, "
        "coalesce(context_snapshot::text,''), coalesce(result_json::text,''), "
        "coalesce(next_action,''), coalesce(log_ref,'') "
        "from heartbeat_runs "
        "where status='succeeded' and context_snapshot is not null "
        f"order by started_at desc {lim};"
    )
    rows = [r for r in psql(sql).splitlines() if r.strip()]
    n_ok = n_trace = 0
    with open(args.out, "w") as f:
        for r in rows:
            parts = r.split("\x1f")
            if len(parts) < 7:
                continue
            rid, aid, started, ctx, res, na, log_ref = parts[:7]
            trace_path = os.path.join(LOG_ROOT, log_ref) if log_ref else ""
            has_trace = bool(trace_path and os.path.exists(trace_path))
            rec = {
                "run_id": rid, "agent_id": aid, "started_at": started,
                "input_context": _loads(ctx), "result": _loads(res),
                "next_action": na or None,
                "transcript_path": trace_path or None,
                "has_transcript": has_trace,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += 1
            n_trace += int(has_trace)
    print(f"wrote {n_ok} records to {args.out}  ({n_trace} with on-disk transcript)")


def _loads(s: str):
    try:
        return json.loads(s) if s else None
    except Exception:
        return {"_raw": s[:2000]}


if __name__ == "__main__":
    main()
