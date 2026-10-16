#!/usr/bin/env bash
# 一键跑全 perf benchmark（spec §11 第 5 步 / T18）。
#
# 与 ``python benchmarks/run_perf_benchmark.py`` 等价；这里额外做的：
#   1. 设 PYTHONPATH=. 让 ``app.*`` 与 ``data.*`` 可直接 import
#   2. 跑前打印环境信息（python / dowhy 版本）
#   3. 跑后打印 perf_records/ 下生成文件数 + 写 summary log
#
# Usage
# -----
#   $ bash benchmarks/run_parallel.sh           # 全跑
#   $ bash benchmarks/run_parallel.sh --dry-run # 仅打印计划

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
PERF_RECORDS_DIR="${PROJECT_ROOT}/perf_records"
mkdir -p "${LOG_DIR}" "${PERF_RECORDS_DIR}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "=== perf benchmark 一键跑全 ==="
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "PYTHON=$(command -v python3 || command -v python)"
python -V
python -c "import dowhy; print('dowhy version:', dowhy.__version__)" 2>/dev/null \
    || python -c "import sys; print('dowhy 不可 import；请确认依赖已装', file=sys.stderr)"

LOG_FILE="${LOG_DIR}/perf_benchmark-$(date -u +%Y%m%dT%H%M%SZ).log"
echo "Log: ${LOG_FILE}"

if python "${SCRIPT_DIR}/run_perf_benchmark.py" "$@" 2>&1 | tee "${LOG_FILE}"; then
    echo "--- benchmark 完成 ---"
    echo "perf_records 落盘文件数：$(ls -1 "${PERF_RECORDS_DIR}"/benchmark_*.json 2>/dev/null | wc -l | tr -d ' ')"
    echo "perf_records/benchmark_summary.json:"
    if [[ -f "${PERF_RECORDS_DIR}/benchmark_summary.json" ]]; then
        python -c "import json,sys; s=json.load(open('${PERF_RECORDS_DIR}/benchmark_summary.json')); print(f\"  total={s['n_combos']} success={s['n_success']} error_rate={s['error_rate']:.3f} (threshold={s['error_rate_threshold']}) passes={s['passes_error_rate']}\")"
    fi
    exit 0
else
    echo "!!! benchmark 失败，详情见 ${LOG_FILE}" >&2
    exit 1
fi
