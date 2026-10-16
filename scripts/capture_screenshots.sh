#!/usr/bin/env bash
# 单病种指标因果归因系统 — 采集 4 张真实运行截图
#
# 用法：
#   bash scripts/capture_screenshots.sh
#   bash scripts/capture_screenshots.sh --no-tests
#
# 行为：
#   1. venv → pip install（按需） → 设置测试环境变量（spec §8 凭据位 → 占位值）
#   2. 后台启 uvicorn → 等 /healthz 200
#   3. 用 playwright + chromium headless 采集 4 张真实运行截图到 docs/screenshots/
#      - 01_startup.png        启动首屏（empty 三态）
#      - 02_causal_graph.png   建模完成后因果图渲染（Plotly 节点 + 边）
#      - 03_counterfactual.png 反事实 modal 打开（节点 + 假设值输入框 + 预测结果）
#      - 04_review_confirm.png 评审 confirm 按钮可点 + toast 通知
#   4. 停掉 uvicorn + 写日志
#
# **纪律**：截图是真实浏览器渲染（playwright.chromium）→ /tmp/.png → 文档，
# 禁止用概念图/设计稿冒充实测。

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PORT="${APP_PORT:-8000}"
SCREENSHOTS_DIR="$PROJECT_DIR/docs/screenshots"
LOG_DIR="$PROJECT_DIR/logs"
PID_FILE="$PROJECT_DIR/.capture.pid"
LOG_FILE="$LOG_DIR/capture_screenshots-$(date -u +%Y%m%dT%H%M%SZ).log"

mkdir -p "$SCREENSHOTS_DIR" "$LOG_DIR"

RUN_TESTS=0
for arg in "$@"; do
    case "$arg" in
        --no-tests) RUN_TESTS=0 ;;
        --help|-h)
            echo "[capture_screenshots.sh] 用法：bash scripts/capture_screenshots.sh [--no-tests]"
            exit 0
            ;;
        *) echo "[capture_screenshots.sh] 未知参数：$arg" >&2; exit 2 ;;
    esac
done

# ---- 0. 清理旧截图 ----
rm -f "$SCREENSHOTS_DIR"/*.png 2>/dev/null || true

# ---- 1. venv ----
if [[ ! -d ".venv" ]]; then
    echo "[capture_screenshots.sh] 建虚拟环境 .venv"
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# ---- 2. 凭据位（spec §8 测试占位）----
# 仅用于启动 uvicorn + 让 FastAPI lifespan 通过校验；不连真实系统
export APP_ENV="${APP_ENV:-test}"
export APP_HOST="${APP_HOST:-127.0.0.1}"
export APP_PORT="$PORT"
export APP_LOG_LEVEL="${APP_LOG_LEVEL:-WARNING}"
export DATABASE_URL="sqlite:///./local_state.db"
export HIS_DB_URL="postgresql+psycopg2://his_user:test_pass@127.0.0.1:5432/his_discharge"
export HIS_DB_USER="his_user"
export HIS_DB_PASS="test_pass"
export HIS_DB_POOL_RECYCLE=1800
export DIPCASE_REPORTER_URL="https://dipcase.example-hospital.cn/api/v1"
export DIPCASE_REPORTER_TOKEN="test_bearer_token_for_screenshot_capture_only"
export DIPCASE_REPORTER_TIMEOUT=15
export OPENAI_BASE_URL="http://127.0.0.1:18888/v1"
export OPENAI_API_KEY="test_openai_key_for_screenshot_capture_only"
export OPENAI_MODEL="gpt-4o-mini"
export OPENAI_TIMEOUT=30
export QUALITY_ROLE_HEADER="X-Quality-Role"
export MIN_SAMPLE_SIZE=30
export MAX_CI_WIDTH=0.6
export REFUTATION_N_JOBS=1   # 截图场景下 1 即可（避免 CI 卡顿）

# ---- 3. 启 uvicorn ----
if [[ -f "$PID_FILE" ]]; then
    old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "${old_pid:-}" ]] && kill -0 "$old_pid" 2>/dev/null; then
        echo "[capture_screenshots.sh] 停掉残留 uvicorn pid=$old_pid"
        kill "$old_pid" 2>/dev/null || true
        sleep 1
        kill -9 "$old_pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
fi

echo "[capture_screenshots.sh] 启 uvicorn (port=$PORT, log=$LOG_FILE)"
nohup python -m uvicorn app.main:app --host 127.0.0.1 --port "$PORT" \
    >> "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
UVICORN_PID="$(cat "$PID_FILE")"

# 等服务就绪（最多 30s）
for i in $(seq 1 30); do
    if curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/api/healthz" >/dev/null 2>&1; then
        echo "[capture_screenshots.sh] uvicorn 已就绪（${i}s）"
        break
    fi
    sleep 1
done

if ! curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/api/healthz" >/dev/null 2>&1; then
    echo "[capture_screenshots.sh] ✗ uvicorn 未就绪；tail $LOG_FILE"
    tail -n 80 "$LOG_FILE" || true
    kill "$UVICORN_PID" 2>/dev/null || true
    exit 3
fi

# ---- 4. 采集截图（python + playwright）----
echo "[capture_screenshots.sh] 采集 4 张真实运行截图"
set +e
python scripts/_capture_screenshots.py \
    --url "http://127.0.0.1:${PORT}/" \
    --out-dir "$SCREENSHOTS_DIR" \
    >> "$LOG_FILE" 2>&1
capture_rc=$?
set -e
if [[ "$capture_rc" -ne 0 ]]; then
    echo "[capture_screenshots.sh] ✗ 截图采集失败 rc=$capture_rc；tail $LOG_FILE"
    tail -n 120 "$LOG_FILE" || true
    kill "$UVICORN_PID" 2>/dev/null || true
    rm -f "$PID_FILE"
    exit "$capture_rc"
fi

# ---- 5. 验收：4 张 png 全部存在 + 大小 > 5KB ----
missing=0
for f in 01_startup.png 02_causal_graph.png 03_counterfactual.png 04_review_confirm.png; do
    fp="$SCREENSHOTS_DIR/$f"
    if [[ ! -f "$fp" ]]; then
        echo "[capture_screenshots.sh] ✗ 缺失 $fp"
        missing=$((missing+1))
        continue
    fi
    size=$(stat -f%z "$fp" 2>/dev/null || stat -c%s "$fp" 2>/dev/null || echo 0)
    if [[ "${size:-0}" -lt 5120 ]]; then
        echo "[capture_screenshots.sh] ✗ $fp 大小仅 ${size}B（<5KB 视为 placeholder）"
        missing=$((missing+1))
    else
        echo "[capture_screenshots.sh] ✓ $fp  ${size}B"
    fi
done

# ---- 6. 停 uvicorn ----
kill "$UVICORN_PID" 2>/dev/null || true
sleep 1
kill -9 "$UVICORN_PID" 2>/dev/null || true
rm -f "$PID_FILE"

if [[ "$missing" -gt 0 ]]; then
    echo "[capture_screenshots.sh] ✗ $missing 张截图不合格；日志：$LOG_FILE"
    exit 4
fi

echo "[capture_screenshots.sh] ✓ 4 张真实运行截图已落盘：$SCREENSHOTS_DIR"
echo "[capture_screenshots.sh] 日志：$LOG_FILE"