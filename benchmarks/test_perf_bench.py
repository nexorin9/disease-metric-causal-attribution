"""test_perf_bench — 性能与稳定性基准测试（spec.md §11 第 5 步 / T18 验收）。

测试策略：
    1. **不重跑全 benchmark**（避免 60s+ 长测污染 pytest 反馈），
       而是先 ``subprocess`` 调一次 ``run_perf_benchmark.py`` 把 48 个 JSON
       写到 ``perf_records/``，再逐文件断言验收项。
    2. 若 ``perf_records/benchmark_*.json`` 已存在则**复用**（CI / 本地反复跑时
       不重复耗时；T34 发布前清理不删 perf_records/benchmark_*）。
    3. 验收维度（spec §11 第 5 步）：
       - 48 个组合 JSON 全部落盘（3 病种 × 4 outcome × 4 样本量 = 48）
       - n=2000 单组合 total_elapsed_ms < 60s（spec §11 硬约束）
       - 全组合 error_rate < 5%（spec §11 硬约束）
       - 每组合 stages 全列齐全（identify / estimate / refute / counterfactual）
       - 每组合 counterfactual.scenarios 至少 3 条且 scenario_id 非空
       - 每组合 refute_verdict ∈ {robust / partial / unstable}

源产品能力参考（README 不写源仓库名）：
- 与 T6–T17 共用 dowhy CausalModel 四步流水线与 BatchCounterfactualRunner
- benchmark 只度量性能 / 稳定性，不引入新算法

⚠️ 本模块为 benchmark 验收层；不修改产品行为。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import pytest

# ----------------------------------------------------------------------------------
# 路径与常量
# ----------------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PERF_RECORDS_DIR = PROJECT_ROOT / "perf_records"
RUNNER = PROJECT_ROOT / "benchmarks" / "run_perf_benchmark.py"

EXPECTED_DISEASES = ("I21", "I63", "Z96.10")
EXPECTED_OUTCOMES = ("cmi", "los_days", "readmit_flag", "complication_flag")
EXPECTED_N_GRID = (200, 500, 1000, 2000)
EXPECTED_COMBO_COUNT = len(EXPECTED_DISEASES) * len(EXPECTED_OUTCOMES) * len(EXPECTED_N_GRID)

# spec §11 第 5 步硬约束
N2000_BUDGET_SECONDS = 60.0
ERROR_RATE_THRESHOLD = 0.05

REQUIRED_STAGES = ("data_prep", "identify_effect", "estimate_effect", "refute", "counterfactual")
ALLOWED_VERDICTS = ("robust", "partial", "unstable")


# ----------------------------------------------------------------------------------
# fixture：跑一次全 benchmark（仅当 perf_records/benchmark_*.json 不全时）
# ----------------------------------------------------------------------------------


def _existing_combo_count() -> int:
    if not PERF_RECORDS_DIR.exists():
        return 0
    return sum(1 for _ in PERF_RECORDS_DIR.glob("benchmark_*_n*.json"))


@pytest.fixture(scope="session", autouse=True)
def ensure_benchmark_ran():
    """session 级 fixture：保证 48 个 JSON 落盘（已存在则跳过）。"""
    if _existing_combo_count() >= EXPECTED_COMBO_COUNT:
        # 已存在 48 个 JSON 文件 → 复用（避免重跑）
        yield "reused"
        return
    # 否则 subprocess 跑一次；超时 600s（48 组合可能花 ~5min）
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    PERF_RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, str(RUNNER)],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"run_perf_benchmark.py 失败（rc={proc.returncode}）\n"
            f"stdout tail:\n{proc.stdout[-2000:]}\n"
            f"stderr tail:\n{proc.stderr[-2000:]}"
        )
    yield "fresh"


# ----------------------------------------------------------------------------------
# helper
# ----------------------------------------------------------------------------------


def _load_all_combos() -> Dict[str, dict]:
    """加载 perf_records/benchmark_*.json 全集 → {filename: dict}。"""
    out: Dict[str, dict] = {}
    if not PERF_RECORDS_DIR.exists():
        return out
    for path in sorted(PERF_RECORDS_DIR.glob("benchmark_*_n*.json")):
        try:
            out[path.name] = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            # 解析失败的文件跳过；下个测试会单独断言
            continue
    return out


def _combo_filename(disease: str, outcome: str, n: int) -> str:
    return f"benchmark_{disease}_{outcome}_n{n}.json"


# ----------------------------------------------------------------------------------
# 1. 落盘数量
# ----------------------------------------------------------------------------------


def test_benchmark_files_count():
    """48 个组合 JSON 全部存在（3 病种 × 4 outcome × 4 样本量 = 48）。"""
    combos = _load_all_combos()
    assert len(combos) == EXPECTED_COMBO_COUNT, (
        f"perf_records/benchmark_*.json 数量 {len(combos)}；"
        f"期望 {EXPECTED_COMBO_COUNT}（{EXPECTED_DISEASES}×{EXPECTED_OUTCOMES}×{EXPECTED_N_GRID}）"
    )


@pytest.mark.parametrize(
    "disease,outcome,n",
    [
        (d, o, n)
        for d in EXPECTED_DISEASES
        for o in EXPECTED_OUTCOMES
        for n in EXPECTED_N_GRID
    ],
)
def test_benchmark_file_exists(disease, outcome, n):
    """每个 (disease, outcome, n) 组合对应 JSON 文件存在。"""
    fname = _combo_filename(disease, outcome, n)
    path = PERF_RECORDS_DIR / fname
    assert path.exists(), f"缺失 {fname}"
    # 文件大小 > 1KB（确认非空 placeholder）
    assert path.stat().st_size > 1024, f"{fname} 太小（{path.stat().st_size}B），疑似空文件"


# ----------------------------------------------------------------------------------
# 2. schema 与字段完整性
# ----------------------------------------------------------------------------------


def test_benchmark_summary_present():
    """perf_records/benchmark_summary.json 存在且含 spec §11 第 5 步硬约束字段。"""
    summary_path = PERF_RECORDS_DIR / "benchmark_summary.json"
    assert summary_path.exists(), "缺失 perf_records/benchmark_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for key in (
        "schema_version",
        "n_combos",
        "n_success",
        "n_failure",
        "error_rate",
        "error_rate_threshold",
        "passes_error_rate",
        "n2000_total_ms",
        "n2000_passes_budget",
        "n2000_budget_seconds",
    ):
        assert key in summary, f"summary 缺字段 {key!r}"
    assert summary["schema_version"] == "perf_benchmark/1.0"
    assert summary["n_combos"] == EXPECTED_COMBO_COUNT
    assert summary["error_rate_threshold"] == ERROR_RATE_THRESHOLD
    assert summary["n2000_budget_seconds"] == N2000_BUDGET_SECONDS


@pytest.mark.parametrize(
    "disease,outcome,n",
    [
        (d, o, n)
        for d in EXPECTED_DISEASES
        for o in EXPECTED_OUTCOMES
        for n in EXPECTED_N_GRID
    ],
)
def test_benchmark_combo_schema(disease, outcome, n):
    """单组合 JSON schema 完整（spec §11 第 5 步）。"""
    fname = _combo_filename(disease, outcome, n)
    path = PERF_RECORDS_DIR / fname
    if not path.exists():
        pytest.skip(f"{fname} 缺失；上层 test 已记录")
    data = json.loads(path.read_text(encoding="utf-8"))

    # 顶层必填字段
    for key in (
        "schema_version",
        "ts_utc",
        "disease_code",
        "outcome",
        "sample_size",
        "overall_success",
        "total_elapsed_ms",
        "refute_verdict",
        "stages",
        "counterfactual",
    ):
        assert key in data, f"{fname} 缺顶层字段 {key!r}"

    assert data["schema_version"] == "perf_benchmark/1.0"
    assert data["disease_code"] == disease
    assert data["outcome"] == outcome
    assert data["sample_size"] == n
    assert data["refute_verdict"] in ALLOWED_VERDICTS

    # stages 列表覆盖完整流水线
    stage_names = {s["name"] for s in data["stages"]}
    for required in REQUIRED_STAGES:
        assert required in stage_names, (
            f"{fname} stages 缺 {required!r}；实际 {sorted(stage_names)}"
        )


@pytest.mark.parametrize(
    "disease,outcome,n",
    [
        (d, o, n)
        for d in EXPECTED_DISEASES
        for o in EXPECTED_OUTCOMES
        for n in EXPECTED_N_GRID
    ],
)
def test_benchmark_counterfactual_payload(disease, outcome, n):
    """反事实 payload 至少 3 条 scenario 且 scenario_id 非空。"""
    fname = _combo_filename(disease, outcome, n)
    path = PERF_RECORDS_DIR / fname
    if not path.exists():
        pytest.skip(f"{fname} 缺失")
    data = json.loads(path.read_text(encoding="utf-8"))
    cf = data.get("counterfactual", {})
    scenarios = cf.get("scenarios", [])
    assert len(scenarios) >= 3, (
        f"{fname} counterfactual scenarios 仅 {len(scenarios)} 条；spec 期望 ≥3"
    )
    for sc in scenarios:
        sid = sc.get("scenario_id")
        assert isinstance(sid, str) and sid, (
            f"{fname} scenario 缺 scenario_id 或为空字符串: {sc}"
        )


# ----------------------------------------------------------------------------------
# 3. n=2000 耗时验收（spec §11 第 5 步硬约束 < 60s）
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "disease,outcome",
    [(d, o) for d in EXPECTED_DISEASES for o in EXPECTED_OUTCOMES],
)
def test_n2000_under_60s_per_combo(disease, outcome):
    """n=2000 单组合 total_elapsed_ms < 60s（spec §11 第 5 步硬约束）。"""
    fname = _combo_filename(disease, outcome, 2000)
    path = PERF_RECORDS_DIR / fname
    if not path.exists():
        pytest.skip(f"{fname} 缺失")
    data = json.loads(path.read_text(encoding="utf-8"))
    elapsed_ms = data["total_elapsed_ms"]
    assert elapsed_ms < N2000_BUDGET_SECONDS * 1000, (
        f"{fname} 耗时 {elapsed_ms:.0f}ms ≥ {N2000_BUDGET_SECONDS}s 上限"
    )


def test_n2000_budget_summary_field():
    """benchmark_summary.json n2000_passes_budget 字段为 True。"""
    summary_path = PERF_RECORDS_DIR / "benchmark_summary.json"
    if not summary_path.exists():
        pytest.skip("summary 缺失")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["n2000_passes_budget"] is True, (
        f"n2000 至少 1 组合超时：{summary['n2000_total_ms']}"
    )


# ----------------------------------------------------------------------------------
# 4. error_rate < 5%（spec §11 第 5 步硬约束）
# ----------------------------------------------------------------------------------


def test_error_rate_under_5pct():
    """全组合 error_rate < 5%（spec §11 第 5 步硬约束）。"""
    summary_path = PERF_RECORDS_DIR / "benchmark_summary.json"
    if not summary_path.exists():
        pytest.skip("summary 缺失")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["error_rate"] < ERROR_RATE_THRESHOLD, (
        f"benchmark error_rate={summary['error_rate']:.3f} ≥ {ERROR_RATE_THRESHOLD}"
    )
    assert summary["passes_error_rate"] is True


# ----------------------------------------------------------------------------------
# 5. 同一组合二次跑耗时稳定性（防止单测重跑所有 48 组合；仅跑 1 个代表性组合）
# ----------------------------------------------------------------------------------


def test_repeated_run_stability_smoke():
    """二次跑同一组合耗时比 ≤ 5×（防止极端抖动；smoke 级单测）。"""
    # 选 n=200 的 I21/cmi 组合（最快档位）
    disease, outcome, n = "I21", "cmi", 200
    fname = _combo_filename(disease, outcome, n)
    path = PERF_RECORDS_DIR / fname
    if not path.exists():
        pytest.skip(f"{fname} 缺失")

    data1 = json.loads(path.read_text(encoding="utf-8"))
    elapsed1 = data1["total_elapsed_ms"]

    # 二次跑（不落盘，仅统计耗时）
    from benchmarks.run_perf_benchmark import run_combo  # noqa: WPS433 — 测试内 import

    t0 = time.monotonic()
    combo2 = run_combo(disease, outcome, n)
    elapsed2 = (time.monotonic() - t0) * 1000

    ratio = max(elapsed1, elapsed2) / max(1.0, min(elapsed1, elapsed2))
    assert ratio < 5.0, (
        f"二次耗时比 {ratio:.2f}× 过大（{elapsed1:.0f}ms vs {elapsed2:.0f}ms）"
    )
