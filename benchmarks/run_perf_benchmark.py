"""性能与稳定性实测 benchmark（spec.md §11 第 5 步 / T18 主路径）。

对每个 (disease_code, outcome, sample_size) 组合跑完因果推理四步流水线 +
反事实 + 反驳，记录每档耗时与稳健性判定到
``perf_records/benchmark_<disease>_<outcome>_n<n>.json``。

参数扫描（spec §11 第 5 步）：
    病种：I21（急性心梗）/ I63（脑梗）/ Z96.10（髋置换）
    outcome：cmi / los_days / readmit_flag / complication_flag
    样本量：n = 200 / 500 / 1000 / 2000
    总组合数：3 × 4 × 4 = 48

每个组合独立跑一次主路径（流水线 → 反事实 → 反驳），与单病种评审会上
跑一次的现实使用形态一致；benchmark 输出供后续性能 / 稳定性回归对照。

Usage
-----
    $ cd disease-metric-causal-attribution
    $ PYTHONPATH=. python benchmarks/run_perf_benchmark.py           # 全跑（48 组合）
    $ PYTHONPATH=. python benchmarks/run_perf_benchmark.py --dry-run  # 仅打印计划不跑
    $ PYTHONPATH=. python benchmarks/run_perf_benchmark.py --n 500    # 只跑 n=500 档
    $ PYTHONPATH=. python benchmarks/run_perf_benchmark.py --disease I21 --outcome cmi

源产品能力参考（README 不写源仓库名）：
- CausalModel 四步流水线（data → identify → estimate → refute）
- DoSampler.do_sample(x) 与 Pearl do() 操作
- GraphRefuter 条件独立性检验 + 反驳稳定性判定
- gcm/equation_parser 改写封装为 SCM 自动生成

⚠️ 本 benchmark 不写源项目名；只描述方法族。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

# ----------------------------------------------------------------------------------
# 路径处理：让 ``python benchmarks/run_perf_benchmark.py`` 直接可跑
# ----------------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.dag_builder import DISEASE_OUTCOME_VALUES, SUPPORTED_DISEASE_CODES  # noqa: E402
from app.core.dosampler import CounterfactualRunner  # noqa: E402
from app.core.dosampler_batch import (  # noqa: E402
    BATCH_SIZE_DEFAULT,
    BATCH_TOTAL_TIMEOUT_SECONDS,
    BatchCounterfactualRunner,
    Intervention,
)
from app.core.pipeline import CausalPipeline, EstimandNotIdentifiedError  # noqa: E402
from app.core.refuter import RefuterPipeline  # noqa: E402
from data.seed_loader import load_fixture_csv  # noqa: E402

LOGGER = logging.getLogger("benchmark.perf")

# ----------------------------------------------------------------------------------
# 常量
# ----------------------------------------------------------------------------------

# spec §11 第 5 步硬约束：4 档样本量
SAMPLE_SIZE_GRID: tuple = (200, 500, 1000, 2000)

# 误差率上限（spec §11 第 5 步）：benchmark 内部组合失败率 < 5%
ERROR_RATE_THRESHOLD: float = 0.05

# 单组合总耗时上限（spec §11 第 5 步）：n=2000 时 ≤ 60s
N2000_BUDGET_SECONDS: float = 60.0

# 每组合反事实场景数（面板「一次提交 5 条」用例）
COUNTERFACTUAL_SCENARIO_COUNT: int = 5

# benchmark 报告 schema 版本
SCHEMA_VERSION: str = "perf_benchmark/1.0"

# perf_records 目录（与 sample_size_gate 一致路径）
PERF_RECORDS_DIR: Path = PROJECT_ROOT / "perf_records"

# ----------------------------------------------------------------------------------
# Benchmark 输出 dataclass
# ----------------------------------------------------------------------------------


@dataclass
class StageRecord:
    """流水线单阶段耗时 + 结果摘要。"""

    name: str
    elapsed_ms: float
    success: bool
    degraded: bool = False
    degraded_reasons: List[str] = field(default_factory=list)
    error: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComboResult:
    """单个 (disease × outcome × n) 组合的 benchmark 结果。"""

    disease_code: str
    outcome: str
    sample_size: int
    total_elapsed_ms: float
    overall_success: bool
    stages: List[StageRecord]
    counterfactual: Dict[str, Any]
    refute_verdict: str
    notes: List[str] = field(default_factory=list)


# ----------------------------------------------------------------------------------
# 数据准备 helper
# ----------------------------------------------------------------------------------


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """fixture → pipeline-ready DataFrame（spec §3.1 派生列）。"""
    df = df.copy()
    # complication_count 派生自 secondary_diagnosis_list（与现有测试一致）
    df["complication_count"] = df["secondary_diagnosis_list"].fillna("").apply(
        lambda v: 0 if not v or v == "[]" else min(len(str(v).split("|")), 2)
    )
    # age_band 是字符串区间（fixture 实际值：35-44/45-54/55-64/65-74/75+/75-84/85+），
    # 转换为数值（mid-point 或端点代理），便于 dowhy 数值列；与年龄段枚举 (1/2/3/4) 对齐。
    # - 35-44 / 18-44 → 1（青年段）
    # - 45-54 / 55-64 → 2（中年段）
    # - 65-74 → 3（年轻老年段）
    # - 75+ / 75-84 / 85+ / >=75 → 4（高龄段）
    age_band_map = {
        "18-44": 1,
        "35-44": 1,
        "45-54": 2,
        "55-64": 2,
        "65-74": 3,
        "75-84": 4,
        "75+": 4,
        "85+": 4,
        ">=75": 4,
    }
    mapped = df["age_band"].map(age_band_map)
    # 未映射值兜底：抓字符串首段做 int（fixture 全部覆盖，应到不了这条路径）
    fallback = pd.to_numeric(
        df["age_band"].astype(str).str.extract(r"(\d+)")[0],
        errors="coerce",
    ).fillna(0).astype(int).clip(lower=1, upper=4)
    df["age_band"] = mapped.where(mapped.notna(), fallback).astype(int)
    return df


def _bootstrap_to_size(df: pd.DataFrame, n: int, *, seed: int = 20261016) -> pd.DataFrame:
    """bootstrap-resample 把 df 扩到 n 行（保留因果结构基本一致）。"""
    if n <= len(df):
        out = df.head(n).copy()
    else:
        n_extra = n - len(df)
        extra = df.sample(n=n_extra, replace=True, random_state=seed).reset_index(drop=True)
        base = df.reset_index(drop=True)
        out = pd.concat([base, extra], ignore_index=True)
    # 重新生成 encounter_id 保持全局唯一（spec §3.1 必填：enc_<12hex>）
    out["encounter_id"] = [f"enc_{uuid.uuid4().hex[:12]}" for _ in range(len(out))]
    return out.reset_index(drop=True)


def _build_counterfactual_scenarios(disease_code: str) -> List[Intervention]:
    """为给定 disease_code 构造 5 条反事实场景（覆盖主要干预维度）。

    干预节点命名遵循 spec §11 第 4 步 / dosampler 校验：
        - treatment:<取值>           治疗方式（取值随 disease_code 枚举变化）
        - age_band:<1/2/3/4>          年龄段（与 dag_builder.AGE_BAND_VALUES 对齐）
        - admission_route:<1/2/3/9>   入院途径（与 dag_builder.ADMISSION_ROUTE_VALUES 对齐）
        - complication_count:<0/1/2>  并发症数（派生自 secondary_diagnosis_list）
    """
    from app.core.dag_builder import TREATMENT_VALUES

    treatments = TREATMENT_VALUES[disease_code]
    return [
        Intervention(
            intervention_node=f"treatment:{treatments[0]}",
            hypothetical_value=treatments[0],
            label=f"全部改用 {treatments[0]}",
        ),
        Intervention(
            intervention_node="age_band:3",
            hypothetical_value=3,
            label="年龄段统一到 65-74",
        ),
        Intervention(
            intervention_node="admission_route:1",
            hypothetical_value=1,
            label="入院途径统一为 1（急诊）",
        ),
        Intervention(
            intervention_node="complication_count:0",
            hypothetical_value=0,
            label="并发症数清零",
        ),
        Intervention(
            intervention_node=f"treatment:{treatments[-1]}",
            hypothetical_value=treatments[-1],
            label=f"全部改用 {treatments[-1]}",
        ),
    ]


# ----------------------------------------------------------------------------------
# 单组合 benchmark 主路径
# ----------------------------------------------------------------------------------


def run_combo(
    disease_code: str,
    outcome: str,
    sample_size: int,
    *,
    counterfactual_timeout: float = BATCH_TOTAL_TIMEOUT_SECONDS,
) -> ComboResult:
    """跑单个 (disease × outcome × n) 组合的完整主路径并返回结果。

    主路径：流水线 identify → estimate → 反驳（4 档顺序）→ 反事实（5 场景批量）。
    """
    overall_start = time.monotonic()
    stages: List[StageRecord] = []
    notes: List[str] = []

    # ---- 数据准备 ----
    t0 = time.monotonic()
    try:
        raw = load_fixture_csv(disease_code)
        prepared = _prepare_df(raw)
        df = _bootstrap_to_size(prepared, sample_size)
        if sample_size > len(prepared):
            notes.append(
                f"n={sample_size} > fixture 基线 {len(prepared)}，bootstrap-resample 扩样"
            )
        if df.empty or len(df) < 30:
            raise ValueError(
                f"样本量不足：实际 {len(df)} 行 < 30（spec §6 最低样本量）"
            )
        data_stage = StageRecord(
            name="data_prep",
            elapsed_ms=(time.monotonic() - t0) * 1000,
            success=True,
            extra={"rows": len(df), "columns": len(df.columns)},
        )
    except Exception as exc:  # noqa: BLE001
        data_stage = StageRecord(
            name="data_prep",
            elapsed_ms=(time.monotonic() - t0) * 1000,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
        )
        stages.append(data_stage)
        return ComboResult(
            disease_code=disease_code,
            outcome=outcome,
            sample_size=sample_size,
            total_elapsed_ms=(time.monotonic() - overall_start) * 1000,
            overall_success=False,
            stages=stages,
            counterfactual={"scenarios": [], "elapsed_ms": 0.0, "success": False},
            refute_verdict="unstable",
            notes=notes,
        )
    stages.append(data_stage)

    # ---- 流水线 identify ----
    t0 = time.monotonic()
    try:
        pipeline = CausalPipeline(disease_code=disease_code, outcome=outcome, data=df)
        estimand = pipeline.call_identify_effect()
        identify_stage = StageRecord(
            name="identify_effect",
            elapsed_ms=(time.monotonic() - t0) * 1000,
            success=estimand is not None,
            extra={
                "estimand_type": getattr(estimand, "estimand_type", None),
                "backdoor_count": len(getattr(estimand, "backdoor_variables", []) or []),
            },
        )
    except EstimandNotIdentifiedError as exc:
        identify_stage = StageRecord(
            name="identify_effect",
            elapsed_ms=(time.monotonic() - t0) * 1000,
            success=False,
            error=f"EstimandNotIdentifiedError: {exc}",
        )
        stages.append(identify_stage)
        return ComboResult(
            disease_code=disease_code,
            outcome=outcome,
            sample_size=sample_size,
            total_elapsed_ms=(time.monotonic() - overall_start) * 1000,
            overall_success=False,
            stages=stages,
            counterfactual={"scenarios": [], "elapsed_ms": 0.0, "success": False},
            refute_verdict="unstable",
            notes=notes,
        )
    except Exception as exc:  # noqa: BLE001
        identify_stage = StageRecord(
            name="identify_effect",
            elapsed_ms=(time.monotonic() - t0) * 1000,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
            degraded=True,
            degraded_reasons=[f"{type(exc).__name__}: {exc}"],
        )
        stages.append(identify_stage)
        return ComboResult(
            disease_code=disease_code,
            outcome=outcome,
            sample_size=sample_size,
            total_elapsed_ms=(time.monotonic() - overall_start) * 1000,
            overall_success=False,
            stages=stages,
            counterfactual={"scenarios": [], "elapsed_ms": 0.0, "success": False},
            refute_verdict="unstable",
            notes=notes,
        )
    stages.append(identify_stage)

    # ---- 流水线 estimate ----
    t0 = time.monotonic()
    try:
        estimate = pipeline.call_estimate_effect("backdoor.linear_regression")
        estimate_ok = estimate is not None and getattr(estimate, "ate", None) is not None
        estimate_stage = StageRecord(
            name="estimate_effect",
            elapsed_ms=(time.monotonic() - t0) * 1000,
            success=estimate_ok,
            degraded=(not estimate_ok),
            degraded_reasons=([] if estimate_ok else ["estimate_returned_none"]),
            extra={
                "method": getattr(estimate, "method_name", None) if estimate else None,
                "ate": getattr(estimate, "ate", None) if estimate else None,
                "ci_lower": getattr(estimate, "ci_lower", None) if estimate else None,
                "ci_upper": getattr(estimate, "ci_upper", None) if estimate else None,
                "ci_unstable": getattr(estimate, "estimate_ci_unstable", None)
                if estimate
                else None,
            },
        )
    except Exception as exc:  # noqa: BLE001
        estimate_stage = StageRecord(
            name="estimate_effect",
            elapsed_ms=(time.monotonic() - t0) * 1000,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
            degraded=True,
            degraded_reasons=[f"{type(exc).__name__}: {exc}"],
        )
        stages.append(estimate_stage)
        estimate = None
    stages.append(estimate_stage)

    # ---- 反驳 4 档 ----
    t0 = time.monotonic()
    refute_verdict = "unstable"
    try:
        causal_model = pipeline._build_causal_model()  # noqa: SLF001 — 内部接口复用
        refuter = RefuterPipeline(
            causal_model=causal_model,
            estimate=estimate if estimate is not None else _DummyEstimate(),
            estimand=estimand,
            data=df,
            num_simulations=20,  # benchmark 仿真实测性能，不求高仿真度
        )
        refute_results = refuter.run()
        refute_verdict = refuter.verdict(refute_results)
        refute_stage = StageRecord(
            name="refute",
            elapsed_ms=(time.monotonic() - t0) * 1000,
            success=True,
            extra={
                "verdict": refute_verdict,
                "n_methods": len(refute_results),
                "n_passed": sum(1 for r in refute_results if getattr(r, "passed", False)),
                "n_unstable": sum(
                    1 for r in refute_results if getattr(r, "refutation_unstable", False)
                ),
                "n_degraded": sum(1 for r in refute_results if getattr(r, "degraded", False)),
            },
        )
    except Exception as exc:  # noqa: BLE001
        refute_stage = StageRecord(
            name="refute",
            elapsed_ms=(time.monotonic() - t0) * 1000,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
            degraded=True,
            degraded_reasons=[f"{type(exc).__name__}: {exc}"],
            extra={"verdict": "unstable"},
        )
    stages.append(refute_stage)

    # ---- 反事实（5 场景批量）----
    t0 = time.monotonic()
    try:
        scenarios = _build_counterfactual_scenarios(disease_code)
        batch_runner = BatchCounterfactualRunner(
            disease_code=disease_code,
            outcome=outcome,
            data=df,
            batch_size=BATCH_SIZE_DEFAULT,
        )
        cf_results = batch_runner.run(scenarios)
        cf_success = all(
            getattr(s, "predicted_outcome", None) is not None for s in cf_results
        )
        cf_stage = StageRecord(
            name="counterfactual",
            elapsed_ms=(time.monotonic() - t0) * 1000,
            success=bool(cf_results),
            degraded=(not cf_success) and bool(cf_results),
            degraded_reasons=(
                []
                if cf_success
                else [f"predicted_outcome_none_in_{sum(1 for s in cf_results if getattr(s, 'predicted_outcome', None) is None)}_scenarios"]
            ),
            extra={
                "scenario_count": len(cf_results),
                "n_timeout": sum(1 for s in cf_results if getattr(s, "timeout", False)),
                "n_exception": sum(
                    1
                    for s in cf_results
                    if any("exception" in str(op).lower() for op in (getattr(s, "do_operations", []) or []))
                ),
            },
        )
    except Exception as exc:  # noqa: BLE001
        cf_stage = StageRecord(
            name="counterfactual",
            elapsed_ms=(time.monotonic() - t0) * 1000,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
            degraded=True,
            degraded_reasons=[f"{type(exc).__name__}: {exc}"],
        )
        cf_results = []
    stages.append(cf_stage)

    overall_success = all(s.success for s in stages)
    return ComboResult(
        disease_code=disease_code,
        outcome=outcome,
        sample_size=sample_size,
        total_elapsed_ms=(time.monotonic() - overall_start) * 1000,
        overall_success=overall_success,
        stages=stages,
        counterfactual={
            "scenarios": [
                {
                    "scenario_id": getattr(s, "scenario_id", None),
                    "intervention_node": getattr(s, "intervention_node", None),
                    "predicted_outcome": getattr(s, "predicted_outcome", None),
                    "confidence": getattr(s, "confidence", None),
                    "timeout": getattr(s, "timeout", False),
                    "elapsed_ms": getattr(s, "elapsed_ms", 0),
                }
                for s in cf_results
            ],
            "elapsed_ms": cf_stage.elapsed_ms,
            "success": cf_stage.success,
        },
        refute_verdict=refute_verdict,
        notes=notes,
    )


# 内部 dummy（estimate 为 None 时让 refute 不抛）
class _DummyEstimate:
    """RefuterPipeline 要求 estimate 非 None；estimate 失败时用 dummy 占位。"""

    ate = None
    ci_lower = None
    ci_upper = None
    p_value = None
    method_name = "dummy"


# ----------------------------------------------------------------------------------
# 序列化与落盘
# ----------------------------------------------------------------------------------


def _combo_to_dict(combo: ComboResult) -> Dict[str, Any]:
    """ComboResult → dict（可 JSON 序列化）。"""
    return {
        "schema_version": SCHEMA_VERSION,
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "disease_code": combo.disease_code,
        "outcome": combo.outcome,
        "sample_size": combo.sample_size,
        "overall_success": combo.overall_success,
        "total_elapsed_ms": combo.total_elapsed_ms,
        "refute_verdict": combo.refute_verdict,
        "stages": [
            {
                "name": s.name,
                "elapsed_ms": s.elapsed_ms,
                "success": s.success,
                "degraded": s.degraded,
                "degraded_reasons": list(s.degraded_reasons),
                "error": s.error,
                "extra": dict(s.extra),
            }
            for s in combo.stages
        ],
        "counterfactual": dict(combo.counterfactual),
        "notes": list(combo.notes),
    }


def _write_combo(combo: ComboResult, perf_dir: Path = PERF_RECORDS_DIR) -> Path:
    """落盘单个组合的 JSON 报告；返回写入路径。"""
    perf_dir.mkdir(parents=True, exist_ok=True)
    out = perf_dir / f"benchmark_{combo.disease_code}_{combo.outcome}_n{combo.sample_size}.json"
    out.write_text(
        json.dumps(_combo_to_dict(combo), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out


# ----------------------------------------------------------------------------------
# 全跑入口
# ----------------------------------------------------------------------------------


def run_all(
    *,
    sample_sizes: Sequence[int] = SAMPLE_SIZE_GRID,
    diseases: Sequence[str] = SUPPORTED_DISEASE_CODES,
    outcomes: Sequence[str] = DISEASE_OUTCOME_VALUES,
    dry_run: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """跑全部 (disease × outcome × n) 组合；返回总览报告 dict。"""
    log = logger or LOGGER
    combos = [
        (d, o, n)
        for d in diseases
        for o in outcomes
        for n in sample_sizes
    ]
    log.info("perf benchmark 计划：%d 个组合", len(combos))
    if dry_run:
        for d, o, n in combos:
            log.info("  - disease=%s outcome=%s n=%d", d, o, n)
        return {
            "dry_run": True,
            "n_combos": len(combos),
            "combos": [
                {"disease": d, "outcome": o, "n": n} for d, o, n in combos
            ],
        }

    results: List[ComboResult] = []
    failures: List[Dict[str, Any]] = []
    for d, o, n in combos:
        log.info(
            "→ 跑 %s/%s/n=%d (剩余 %d)",
            d,
            o,
            n,
            len(combos) - len(results),
        )
        try:
            combo = run_combo(d, o, n)
        except Exception as exc:  # noqa: BLE001 — 组合层兜底
            log.error("组合 %s/%s/n=%d 顶层异常: %s", d, o, n, exc)
            log.debug("traceback:\n%s", traceback.format_exc())
            failures.append(
                {"disease": d, "outcome": o, "n": n, "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        out_path = _write_combo(combo)
        log.info(
            "  ✓ 落盘 %s (%.1fms, success=%s)",
            out_path.name,
            combo.total_elapsed_ms,
            combo.overall_success,
        )
        if not combo.overall_success:
            failures.append(
                {
                    "disease": d,
                    "outcome": o,
                    "n": n,
                    "error": next(
                        (s.error for s in combo.stages if not s.success), "unknown"
                    ),
                }
            )
        results.append(combo)

    total = len(combos)
    success = sum(1 for r in results if r.overall_success)
    error_rate = (total - success) / total if total else 0.0
    summary = {
        "schema_version": SCHEMA_VERSION,
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "n_combos": total,
        "n_success": success,
        "n_failure": total - success,
        "error_rate": error_rate,
        "error_rate_threshold": ERROR_RATE_THRESHOLD,
        "passes_error_rate": error_rate < ERROR_RATE_THRESHOLD,
        "n2000_total_ms": [
            r.total_elapsed_ms for r in results if r.sample_size == 2000
        ],
        "n2000_passes_budget": all(
            r.total_elapsed_ms < N2000_BUDGET_SECONDS * 1000
            for r in results
            if r.sample_size == 2000
        ),
        "n2000_budget_seconds": N2000_BUDGET_SECONDS,
        "failures": failures,
    }
    # 全局 summary 落盘（便于面板与 CI 复用）
    summary_path = PERF_RECORDS_DIR / "benchmark_summary.json"
    PERF_RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info(
        "✓ perf benchmark 完成：%d/%d 成功，error_rate=%.3f（阈值 %.2f）；n=2000 总耗时 %s ms",
        success,
        total,
        error_rate,
        ERROR_RATE_THRESHOLD,
        summary["n2000_total_ms"],
    )
    return summary


# ----------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_perf_benchmark",
        description="性能与稳定性实测（spec §11 第 5 步）",
    )
    parser.add_argument(
        "--n",
        type=int,
        action="append",
        default=None,
        help="只跑指定 n 档（可多次）；默认全跑 200/500/1000/2000",
    )
    parser.add_argument("--disease", action="append", default=None, help="只跑指定病种")
    parser.add_argument("--outcome", action="append", default=None, help="只跑指定 outcome")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="DEBUG 级日志输出"
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    args = _parse_args(argv)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    sample_sizes = tuple(args.n) if args.n else SAMPLE_SIZE_GRID
    diseases = tuple(args.disease) if args.disease else SUPPORTED_DISEASE_CODES
    outcomes = tuple(args.outcome) if args.outcome else DISEASE_OUTCOME_VALUES

    summary = run_all(
        sample_sizes=sample_sizes,
        diseases=diseases,
        outcomes=outcomes,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return 0
    return 0 if summary.get("passes_error_rate", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
