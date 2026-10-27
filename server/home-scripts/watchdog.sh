#!/bin/bash
# watchdog.sh
# CPU spill / hang 감지 후 llama-server 재시작.
# 핵심: CPU는 반드시 "순간 사용률"을 봐야 한다.
#       ps %cpu 는 프로세스 생애 전체 평균이라 hang을 제때 감지하지 못한다.
#
# [TON-2381] GPU 사용률을 단일 순간 샘플 대신 윈도우 MAX로 측정.
#   기존: nvidia-smi 1회 샘플이 토큰 커널 사이에서 0%로 읽혀, 활발한 긴 생성 중에도
#         CPU 100%/GPU 0%로 오인 -> 3사이클 후 llama-server를 mid-stream으로 kill,
#         클라이언트 연결 끊김 -> Atlas "empty response" 유발.
#   변경: 0.5s 간격 4회 샘플의 MAX를 사용. 진짜 CPU-spill 행은 윈도우 전체에서 GPU가
#         지속적으로 0%이므로 여전히 감지되고, 생성 중 GPU 버스트는 idle로 오인되지 않음.

RESTART_CMD="/home/tony/start-llama.sh"
CHECK_INTERVAL=10   # 10초마다 체크
CPU_THRESHOLD=90    # 단일코어 100% 기준. 스레드가 통째로 스핀하면 수백%가 됨
GPU_THRESHOLD=10
HANG_COUNT=0
HANG_LIMIT=3        # 3번 연속 감지시 재시작
SAMPLE=2            # 순간 CPU 측정용 샘플링 구간(초)
CLK=$(getconf CLK_TCK)

HEALTH_URL="http://localhost:8020/health"
HEALTH_TIMEOUT=5    # curl 최대 대기(초). 행 걸린 서버에서 watchdog이 같이 멈추지 않도록.
HEALTH_FAIL=0
HEALTH_LIMIT=5      # 5번 연속 무응답/비정상시 재시작 (로딩 중 503 오탐 방지용 여유)

# /health는 HTTP 루프만 살아 있어도 200을 반환한다. 취소된 생성이 slot을
# 해제하지 못해 실제 요청만 영원히 대기하는 경우를 잡기 위해, idle 상태에서만
# 저빈도로 1-token 생성을 검증한다. 2회 연속 실패해야 재시작해 일시적 혼잡을 허용한다.
PROBE_EVERY=5
PROBE_TIMEOUT=15
PROBE_FAIL=0
PROBE_FAIL_LIMIT=2
PROBE_TICK=0

# 이 호스트에는 :8020 생성용과 :8021 CPU 임베딩용 llama-server가 같이 있다.
# pgrep -o/-n은 시작 순서만 보므로 정확한 --port 8020 프로세스를 선택해야 한다.
find_server_pid() {
    local pid cmdline
    while read -r pid; do
        [ -n "$pid" ] || continue
        cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null) || continue
        case " $cmdline " in
            *" --port 8020 "*) echo "$pid"; return 0 ;;
        esac
    done < <(pgrep -x llama-server)
    return 1
}

# 공통 재시작 루틴
restart_server() {
    local pid=$1
    echo "$(date) 🔄 재시작 진행! (PID=$pid)"
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null
    # [TON-2971] relaunch 전에 VRAM 실제 해제 대기. kill 직후 sleep 5만으로는
    #   20.8GB VRAM이 다 반환되지 않아 새 llama-server가 cudaMalloc OOM -> -ngl 99
    #   인데도 CPU 폴백하는 레이스가 있었다. <1GiB로 떨어질 때까지(최대 30s) 대기.
    for _i in $(seq 1 30); do
        _used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
        [ -z "$_used" ] && break
        [ "$_used" -lt 1024 ] && break
        sleep 1
    done
    sleep 3
    bash $RESTART_CMD &
    HANG_COUNT=0
    HEALTH_FAIL=0
    sleep 45  # 재시작 후 로딩 시간 확보
}

# /proc/<pid>/stat 델타로 순간 CPU 사용률(%)을 계산.
# comm 필드에 괄호/공백이 있어도 안전하도록 마지막 ') ' 이후만 파싱.
get_instant_cpu() {
    local pid=$1
    local line u1 s1 u2 s2
    line=$(sed 's/.*) //' /proc/$pid/stat 2>/dev/null) || { echo 0; return; }
    read -r _ _ _ _ _ _ _ _ _ _ u1 s1 _ <<< "$line"
    sleep "$SAMPLE"
    line=$(sed 's/.*) //' /proc/$pid/stat 2>/dev/null) || { echo 0; return; }
    read -r _ _ _ _ _ _ _ _ _ _ u2 s2 _ <<< "$line"
    if [[ -z "$u1" || -z "$u2" ]]; then echo 0; return; fi
    echo $(( ( (u2 + s2) - (u1 + s1) ) * 100 / (CLK * SAMPLE) ))
}

# [TON-2381] GPU 사용률을 윈도우(0.5s x 4)에서 MAX로 측정.
# 단일 순간 샘플은 토큰 커널 사이의 0%를 잡아 활발한 생성을 idle로 오인한다.
# 윈도우 MAX는 생성 중 GPU 버스트를 포착하므로 오탐을 제거하고,
# 진짜 CPU-spill(윈도우 전체 GPU 0%)은 그대로 감지한다.
get_max_gpu() {
    local g maxg=0
    for _ in 1 2 3 4; do
        g=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
        if [[ "$g" =~ ^[0-9]+$ ]] && [ "$g" -gt "$maxg" ]; then
            maxg=$g
        fi
        sleep 0.5
    done
    echo "$maxg"
}

PAUSE_FILE="$HOME/watchdog.pause"
while true; do
    # [TON-2441] Self-expiring pause: a GPU-policy phase (RAFT rollout/train) needs the
    # 3090 free, so it stops llama-server. While paused, do NOT resurrect it. The pause
    # file holds a unix expiry; if it is stale/invalid the watchdog auto-resumes guarding
    # prod — a crashed driver can never leave the teacher down indefinitely.
    if [ -f "$PAUSE_FILE" ]; then
        exp=$(cat "$PAUSE_FILE" 2>/dev/null)
        now=$(date +%s)
        if [[ "$exp" =~ ^[0-9]+$ ]] && [ "$now" -lt "$exp" ]; then
            echo "$(date) ⏸️  watchdog paused (until $exp) — skipping restart"
            sleep $CHECK_INTERVAL
            continue
        else
            echo "$(date) ⏸️  pause expired/invalid — resuming prod protection"
            rm -f "$PAUSE_FILE"
        fi
    fi
    # :8021 임베딩 서버를 잘못 종료하지 않도록 :8020 PID만 선택.
    PID=$(find_server_pid)

    if [ -z "$PID" ]; then
        echo "$(date) llama-server 죽어있음, 재시작..."
        bash $RESTART_CMD &
        sleep 45  # 모델 로딩 대기 (로딩 중 오탐 방지)
        HANG_COUNT=0
        HEALTH_FAIL=0
        continue
    fi

    CPU=$(get_instant_cpu "$PID")
    CPU=${CPU:-0}

    # [TON-2381] 윈도우 MAX GPU 사용률 (단일 순간 샘플 오탐 제거)
    GPU=$(get_max_gpu)
    GPU=${GPU:-0}

    # /health 무응답 또는 비정상(200 아님) 감지. --max-time으로 행 걸린 서버 방어.
    HTTP=$(curl -s -o /dev/null -m "$HEALTH_TIMEOUT" -w '%{http_code}' "$HEALTH_URL" 2>/dev/null)
    HTTP=${HTTP:-000}

    echo "$(date) CPU: ${CPU}% GPU(max): ${GPU}% HEALTH: ${HTTP}"

    # 조건 1: CPU만 펄펄 끓고 GPU는 (윈도우 전체에서) 거의 안 쓰는 상태 = CPU spill / hang
    if [ "$CPU" -ge $CPU_THRESHOLD ] && [ "$GPU" -le $GPU_THRESHOLD ]; then
        HANG_COUNT=$((HANG_COUNT + 1))
        echo "⚠️  CPU spill 감지 ${HANG_COUNT}/${HANG_LIMIT} (CPU ${CPU}% / GPU(max) ${GPU}%)"
    else
        HANG_COUNT=0
    fi

    # 조건 2: /health 무응답 = HTTP 행/데드락
    if [ "$HTTP" != "200" ]; then
        HEALTH_FAIL=$((HEALTH_FAIL + 1))
        echo "⚠️  /health 비정상 ${HEALTH_FAIL}/${HEALTH_LIMIT} (code=${HTTP})"
    else
        HEALTH_FAIL=0
    fi

    # 서버가 idle이고 /health가 정상일 때만 실제 생성 경로를 검증한다.
    # 생성 중인 서버에 probe를 줄 세우지 않아 정상적인 긴 요청을 방해하지 않는다.
    PROBE_TICK=$((PROBE_TICK + 1))
    if [ "$HTTP" = "200" ] && [ "$CPU" -lt "$CPU_THRESHOLD" ] && \
       [ "$GPU" -le "$GPU_THRESHOLD" ] && [ "$PROBE_TICK" -ge "$PROBE_EVERY" ]; then
        PROBE_TICK=0
        PROBE_HTTP=$(curl -s -o /dev/null -m "$PROBE_TIMEOUT" -w '%{http_code}' \
            -H 'Content-Type: application/json' \
            -d '{"prompt":"ping","n_predict":1,"temperature":0,"cache_prompt":false}' \
            http://localhost:8020/completion 2>/dev/null)
        PROBE_HTTP=${PROBE_HTTP:-000}
        if [ "$PROBE_HTTP" = "200" ]; then
            PROBE_FAIL=0
            echo "✅ generation probe: 200"
        else
            PROBE_FAIL=$((PROBE_FAIL + 1))
            echo "⚠️  generation probe 비정상 ${PROBE_FAIL}/${PROBE_FAIL_LIMIT} (code=${PROBE_HTTP})"
        fi
    fi

    # 둘 중 하나라도 한계 도달시 재시작
    if [ $HANG_COUNT -ge $HANG_LIMIT ]; then
        echo "🔄 CPU spill 한계 도달 -> 재시작"
        restart_server "$PID"
        continue
    fi
    if [ $HEALTH_FAIL -ge $HEALTH_LIMIT ]; then
        echo "🔄 /health 무응답 한계 도달 -> 재시작"
        restart_server "$PID"
        continue
    fi
    if [ $PROBE_FAIL -ge $PROBE_FAIL_LIMIT ]; then
        echo "🔄 generation slot 무응답 한계 도달 -> 재시작"
        restart_server "$PID"
        PROBE_FAIL=0
        PROBE_TICK=0
        continue
    fi

    sleep $CHECK_INTERVAL
done
