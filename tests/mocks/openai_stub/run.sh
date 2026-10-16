#!/usr/bin/env bash
# OpenAI-compatible 协议替身启动脚本（后台）
#
# 用法：
#   bash tests/mocks/openai_stub/run.sh start     # 后台启动，PID 写到 .openai_stub.pid
#   bash tests/mocks/openai_stub/run.sh stop      # 停掉
#   bash tests/mocks/openai_stub/run.sh status    # 看存活
#   bash tests/mocks/openai_stub/run.sh restart   # stop + start
#
# 端口默认 18888（与 .env.example OPENAI_STUB_URL 默认一致）。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PID_FILE="$PROJECT_DIR/.openai_stub.pid"
LOG_FILE="$PROJECT_DIR/logs/openai_stub.log"
PORT="${OPENAI_STUB_PORT:-18888}"

mkdir -p "$PROJECT_DIR/logs"

start() {
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "[openai_stub] 已在运行 pid=$(cat "$PID_FILE")"
        return 0
    fi
    cd "$PROJECT_DIR"
    nohup python -m tests.mocks.openai_stub.main --host 127.0.0.1 --port "$PORT" \
        >> "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    sleep 0.5
    if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "[openai_stub] 已启动 pid=$(cat "$PID_FILE") port=$PORT log=$LOG_FILE"
    else
        echo "[openai_stub] 启动失败，查看日志：$LOG_FILE" >&2
        return 1
    fi
}

stop() {
    if [[ -f "$PID_FILE" ]]; then
        local pid
        pid="$(cat "$PID_FILE")"
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            sleep 0.5
            kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$PID_FILE"
        echo "[openai_stub] 已停止"
    else
        echo "[openai_stub] 未运行"
    fi
}

status() {
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "[openai_stub] 存活 pid=$(cat "$PID_FILE") port=$PORT"
        return 0
    fi
    echo "[openai_stub] 未运行"
    return 1
}

case "${1:-start}" in
    start) start ;;
    stop) stop ;;
    status) status ;;
    restart) stop; start ;;
    *) echo "[openai_stub] 用法：$0 {start|stop|status|restart}" >&2; exit 2 ;;
esac