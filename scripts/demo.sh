#!/usr/bin/env bash
# 单病种指标因果归因系统 — 一键启动脚本（本地开发与 e2e 演示）
#
# 用法：
#   bash scripts/demo.sh                  # 全跑：venv → pip install → pytest 收窄 → uvicorn → curl /healthz
#   bash scripts/demo.sh --no-tests       # 跳过 pytest
#   bash scripts/demo.sh --stop           # 停掉 demo.sh 之前启的 uvicorn
#
# 依赖：Python ≥3.10；启动前确保 .env 已就位（cp .env.example .env）

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PORT="${APP_PORT:-8000}"
PID_FILE="$PROJECT_DIR/.uvicorn.pid"
LOG_DIR="$PROJECT_ROOT_DIR/logs"
mkdir -p "$PROJECT_DIR/logs"

RUN_TESTS=1
STOP_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --no-tests) RUN_TESTS=0 ;;
        --stop) STOP_ONLY=1 ;;
        *) echo "[demo.sh] 未知参数：$arg" >&2; exit 2 ;;
    esac
done

stop_server() {
    if [[ -f "$PID_FILE" ]]; then
        pid="$(cat "$PID_FILE" 2>/dev/null || true)"
        if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
            echo "[demo.sh] 停掉 uvicorn pid=$pid"
            kill "$pid" 2>/dev/null || true
            sleep 1
            kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$PID_FILE"
    fi
    echo "[demo.sh] done --stop"
}

if [[ "$STOP_ONLY" == "1" ]]; then
    stop_server
    exit 0
fi

# ---- 1. venv ----
if [[ ! -d ".venv" ]]; then
    echo "[demo.sh] 建虚拟环境 .venv"
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
echo "[demo.sh] Python: $(python --version)  pip: $(pip --version)"

# ---- 2. 依赖 ----
echo "[demo.sh] 装依赖（pyproject.toml）"
pip install --quiet --upgrade pip
pip install --quiet -e ".[dev]"

# ---- 3. .env 兜底 ----
if [[ ! -f ".env" ]]; then
    echo "[demo.sh] .env 不存在 → cp .env.example .env（请按需覆盖凭据）"
    cp .env.example .env
fi

# ---- 4. 收窄跑测试（仅受影响模块；按 Phase 4 Token 纪律） ----
if [[ "$RUN_TESTS" == "1" ]]; then
    TARGETS=("$@")
    if [[ ${#TARGETS[@]} -eq 0 ]]; then
        # 默认 demo 跑：仅配置 + 核心入口相关测试
        echo "[demo.sh] pytest -q tests/test_config.py（收窄跑）"
        python -m pytest tests/test_config.py -q || {
            echo "[demo.sh] pytest 失败，但 demo.sh 继续启服务（按 Phase 4 失败恢复纪律）"
        }
    fi
fi

# ---- 5. 启 uvicorn ----
stop_server  # 清掉残留
echo "[demo.sh] 启 uvicorn (port=$PORT, log=logs/uvicorn.log)"
nohup python -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT" \
    > "$PROJECT_DIR/logs/uvicorn.log" 2>&1 &
echo $! > "$PID_FILE"
sleep 2

# ---- 6. 验证 /healthz ----
echo "[demo.sh] curl http://127.0.0.1:${PORT}/healthz"
if curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/healthz"; then
    echo
    echo "[demo.sh] ✓ 服务已就绪  →  http://127.0.0.1:${PORT}/"
    echo "[demo.sh] 停服：bash scripts/demo.sh --stop"
else
    echo "[demo.sh] ✗ healthz 未达标；tail logs/uvicorn.log"
    tail -n 50 "$PROJECT_DIR/logs/uvicorn.log" || true
    exit 3
fi