#!/usr/bin/env bash
# 单病种指标因果归因系统 — 端到端集成测试一键脚本
#
# 用法：
#   bash scripts/run_e2e.sh             # 全跑：venv → pytest tests/e2e/ → 断言摘要
#   bash scripts/run_e2e.sh --no-deps   # 跳过 pip install（CI / 已装环境）
#   bash scripts/run_e2e.sh --verbose   # 显示完整 pytest 输出
#
# 跑什么：
#   tests/e2e/test_e2e.py — 端到端集成测试（HIS SQLite 替身 + 上报 in-memory 替身 +
#   ModelGateway stub + AuditLogger in-memory → 完整建模 → 反事实 → 起草 → review →
#   confirm/reject/rollback 链路）。
#
# 依赖：Python ≥3.10；启动前不需 .env（conftest 自动注入测试凭据位）。

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

INSTALL_DEPS=1
VERBOSE=0
E2E_LOG="$LOG_DIR/run_e2e-$(date -u +%Y%m%dT%H%M%SZ).log"

for arg in "$@"; do
    case "$arg" in
        --no-deps)   INSTALL_DEPS=0 ;;
        --verbose|-v) VERBOSE=1 ;;
        --help|-h)
            echo "[run_e2e.sh] 用法：bash scripts/run_e2e.sh [--no-deps] [--verbose]"
            exit 0
            ;;
        *) echo "[run_e2e.sh] 未知参数：$arg" >&2; exit 2 ;;
    esac
done

# ---- 1. venv ----
if [[ ! -d ".venv" ]]; then
    echo "[run_e2e.sh] 建虚拟环境 .venv"
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
echo "[run_e2e.sh] Python: $(python --version)  pip: $(pip --version)"

# ---- 2. 依赖（按需）----
if [[ "$INSTALL_DEPS" == "1" ]]; then
    echo "[run_e2e.sh] 装依赖（pyproject.toml）"
    pip install --quiet --upgrade pip
    pip install --quiet -e ".[dev]"
fi

# ---- 3. 跑端到端测试 ----
echo "[run_e2e.sh] 跑 tests/e2e/test_e2e.py（spec §11 收尾最后一档 e2e 集成验收）"
PYTEST_ARGS=("tests/e2e/test_e2e.py" "-v")
if [[ "$VERBOSE" == "1" ]]; then
    PYTEST_ARGS+=("--tb=short")
else
    PYTEST_ARGS+=("--tb=line")
fi

# 跑 + 摘要
if python -m pytest "${PYTEST_ARGS[@]}" 2>&1 | tee "$E2E_LOG"; then
    echo
    echo "[run_e2e.sh] ✓ 端到端测试通过"
    echo "[run_e2e.sh] 完整日志：$E2E_LOG"
    exit 0
else
    rc=$?
    echo
    echo "[run_e2e.sh] ✗ 端到端测试失败（rc=$rc）"
    echo "[run_e2e.sh] 完整日志：$E2E_LOG"
    echo "[run_e2e.sh] tail -n 80 $E2E_LOG  查看失败摘要"
    exit "$rc"
fi
