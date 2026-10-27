#!/usr/bin/env bash
set -uo pipefail
source /home/tony/sovereign-model-env/bin/activate 2>/dev/null || true
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DOWNLOAD_TIMEOUT=60
DEST=/home/tony/models/Qwen3.6-27B-Fable-5-GGUF
FILE=Qwen3.6-27B-Fable-5-Distill.q5_k_m.gguf
WANT=19231099040
L=/home/tony/model-build/out/pull_fable5.log
say(){ echo "[$(date -u +%FT%TZ)] $*" >> "$L"; }
mkdir -p "$DEST"
say "=== hf re-download start (want $WANT bytes) ==="
for i in $(seq 1 40); do
  sz=$(stat -c %s "$DEST/$FILE" 2>/dev/null || echo 0)
  if [ "$sz" -eq "$WANT" ]; then say "COMPLETE $sz bytes"; echo "done dl complete $sz" >> "$L"; echo done > /home/tony/model-build/out/pull_fable5.DONE; break; fi
  say "attempt $i: have $sz/$WANT, hf download ..."
  hf download TeichAI/Qwen3.6-27B-Fable-5-Experimental-GGUF "$FILE" --local-dir "$DEST" >> "$L" 2>&1
  say "attempt $i rc=$?"
  sleep 5
done
