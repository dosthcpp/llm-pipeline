#!/bin/bash
# TON-4143 (2026-08-15): serve Qwen3.8-27B on the RTX 3090.
# Source: ggml-org/Qwen3.8-27B-GGUF@0669b98607d47046c7c2b3f801011d54a08cfccf
# Main GGUF SHA-256: 31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34
# mmproj SHA-256: 2e968a6af97ce35d8971890b257b9b7edabf20ad91450501fa53162a19ee33eb
# Rollback: cp ~/start-llama.sh.bak-ton4143-pre-qwen38 ~/start-llama.sh
#           kill "$(pgrep -o llama-server)"  # watchdog relaunches the backup

LOG=/home/tony/llama-server.log
GUARD_LOG=/home/tony/llama-guard.log
MARKER=/home/tony/.llama_gpu_guard_kills
MODEL_DIR=/home/tony/models/Qwen3.8-27B-GGUF
MODEL="$MODEL_DIR/Qwen3.8-27B-Q4_K_M.gguf"
MMPROJ="$MODEL_DIR/mmproj-Qwen3.8-27B-Q8_0.gguf"
GPU_MIN_MIB=18500

# Keep the lock open in llama-server so overlapping watchdog/manual launches
# cannot race each other into CPU fallback.
LOCK_DIR=/run/user/$(id -u)
[ -d "$LOCK_DIR" ] || LOCK_DIR=/home/tony
exec 9>"$LOCK_DIR/llama-launch.lock"
if ! flock -n 9; then
    echo "$(date) [gpu-guard] another launcher/server owns the launch lock -- aborting" >> "$GUARD_LOG"
    exit 1
fi

# Wait for a previous process to release VRAM so llama.cpp cannot silently
# fall back to CPU. This path is shared by both watchdog relaunch paths.
GPU_READY=0
for _i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    if ! [[ "$used" =~ ^[0-9]+$ ]]; then
        echo "$(date) [gpu-guard] unable to read numeric VRAM usage before launch -- aborting" >> "$GUARD_LOG"
        exit 1
    fi
    if [ "$used" -lt 1024 ]; then
        GPU_READY=1
        break
    fi
    sleep 1
done
if [ "$GPU_READY" -ne 1 ]; then
    echo "$(date) [gpu-guard] VRAM did not drain below 1024 MiB in 60s -- aborting" >> "$GUARD_LOG"
    exit 1
fi

# Preserve the established LAN/Tailscale API binding for existing consumers;
# the unauthenticated Web UI and slot-monitoring endpoints remain disabled.
/home/tony/llama.cpp/build/bin/llama-server \
  --model "$MODEL" \
  --mmproj "$MMPROJ" \
  --image-min-tokens 1024 \
  --ctx-size 131072 --parallel 1 \
  --cache-type-k q4_0 --cache-type-v q4_0 --flash-attn on -ngl 99 --jinja \
  --fit off --cache-ram 0 --no-cache-idle-slots --no-webui --no-slots \
  --reasoning auto --reasoning-format deepseek --reasoning-preserve \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 \
  --batch-size 512 --ubatch-size 512 \
  --alias Qwen3.8-27B --port 8020 --host 0.0.0.0 \
  >> "$LOG" 2>&1 &
LLAMA_PID=$!

# Reject a load that returned healthy only after spilling weights to CPU.
(
    code=000
    for _i in $(seq 1 240); do
        code=$(curl -s -o /dev/null -m 3 -w '%{http_code}' http://localhost:8020/health 2>/dev/null)
        [ "$code" = "200" ] && break
        kill -0 "$LLAMA_PID" 2>/dev/null || exit 0
        sleep 1
    done
    [ "$code" = "200" ] || exit 0
    used=$(nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null |
        awk -F, -v pid="$LLAMA_PID" '{gsub(/[[:space:]]/, "", $1); gsub(/[[:space:]]/, "", $2); if ($1 == pid) {print $2; exit}}')
    if ! [[ "$used" =~ ^[0-9]+$ ]]; then
        used=0
    fi
    if [ "$used" -lt "$GPU_MIN_MIB" ]; then
        now=$(date +%s)
        touch "$MARKER"
        awk -v now="$now" 'now - $1 < 1800' "$MARKER" > "$MARKER.tmp" && mv "$MARKER.tmp" "$MARKER"
        kills=$(wc -l < "$MARKER")
        if [ "$kills" -ge 2 ]; then
            echo "$(date) [gpu-guard] CPU fallback detected (VRAM ${used} MiB < ${GPU_MIN_MIB}) but kill budget exhausted (${kills}/30min) -- leaving process up" >> "$GUARD_LOG"
            exit 0
        fi
        echo "$now" >> "$MARKER"
        echo "$(date) [gpu-guard] CPU fallback detected (process VRAM ${used} MiB < ${GPU_MIN_MIB}) -- terminating llama-server for clean GPU relaunch" >> "$GUARD_LOG"
        kill "$LLAMA_PID" 2>/dev/null
        for _i in $(seq 1 15); do
            kill -0 "$LLAMA_PID" 2>/dev/null || exit 0
            sleep 1
        done
        kill -9 "$LLAMA_PID" 2>/dev/null
    else
        echo "$(date) [gpu-guard] Qwen3.8 GPU residency OK (process VRAM ${used} MiB)" >> "$GUARD_LOG"
    fi
) &
