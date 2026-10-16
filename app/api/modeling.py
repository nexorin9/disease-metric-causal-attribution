"""建模主路径 SSE 端点（spec.md §11 主路径 + §9 因果推理管线）。

POST /api/modeling/run
  Body: {"disease_code": str, "quarter": str, "outcome": str}
  Header: X-Quality-Role ∈ {quality_officer, med_committee_reviewer}
  响应: text/event-stream，事件流：
        1. identify        → estimand summary
        2. effect          → 每条候选因素的因果效应（边到位即推一次）
        3. refute          → 4 档反驳逐档推送
        4. baseline        → 因果 vs 相关楔子
        5. sample-size     → 三档守门
        6. done            → 终态 summary
        失败走 envelope handler（QualityRoleForbiddenError → 403 等）。

设计原则（spec §6 失败不静默改数据）：
- **不写回业务状态** — 本端点只算不算写，所有事件是元数据流。
- 边到位即推：因果面板边渲染边到，避免等所有计算跑完才一次性返回。
- 估算失败或 CI 超宽标 ``estimate_ci_unstable=True``，绝不静默丢弃该条。
- 反驳单档异常 → 该档 1 条事件标 refutation_unstable + degraded=True，不阻塞后续。
- 样本量不足 → 整流标 insufficient，草稿可生成但禁止上报写回（T21 / T26 守门）。

源产品能力参考（README 不写源仓库名）：
- 因果模型构造 + identify_effect + estimate_effect + refute_estimate 四步流水线
- DoSampler.do_sample 反事实模拟
- parallel refutation n_jobs=-1 调度
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, Header, status
from fastapi.responses import StreamingResponse

from app.core.baseline import CorrelationBaseline, compute_correlation
from app.core.exception_handlers import QualityRoleForbiddenError
from app.core.node_schema import RefutationResult
from app.core.pipeline import CausalPipeline
from app.core.refuter import (
    RefuterPipeline,
    SUPPORTED_METHODS,
)
from app.core.sample_size_gate import SampleSizeGuard

LOGGER = logging.getLogger(__name__)

router = APIRouter(tags=["modeling"])

# 鉴权：质量办角色 + 医务科评审角色都可触发建模（spec §11 第 6 步）
_ALLOWED_ROLES = frozenset({"quality_officer", "med_committee_reviewer"})

# 默认估计器 + 反驳参数（业务可由 body override）
_DEFAULT_ESTIMATOR = "backdoor.linear_regression"
_DEFAULT_NUM_SIMULATIONS = 20
_DEFAULT_REPUTE_JOBS = -1

# SSE 兼容子协议：浏览器 EventSource 默认 "" ，后端忽略即可
_SSE_PING_INTERVAL_S = 15.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _check_role_header(role_header: Optional[str]) -> str:
    """校验 X-Quality-Role header。

    Raises
    ------
    QualityRoleForbiddenError
        当缺失 / 非枚举值 → 403 forbidden envelope。
    """
    if not role_header:
        raise QualityRoleForbiddenError(
            "X-Quality-Role header 缺失；建模 API 必须由质控员或评审角色触发",
            role="<missing>",
        )
    if role_header not in _ALLOWED_ROLES:
        raise QualityRoleForbiddenError(
            f"X-Quality-Role={role_header!r} 不在允许集合 {_ALLOWED_ROLES!r}",
            role=role_header,
        )
    return role_header


def _load_data_for_disease(disease_code: str, quarter: Optional[str]) -> "Any":
    """按 disease_code 加载样例数据（spec §11 测试替他 — 真实 HIS 直连 T4）。

    返回 pd.DataFrame（与 CausalPipeline 兼容）。
    quarter 仅用于日志，未做日期切分（fixture 不含季度维度列）。
    """
    from data.seed_loader import load_fixture_csv

    df = load_fixture_csv(disease_code)
    LOGGER.info(
        "modeling.run 加载 fixture %s quarter=%s shape=%s",
        disease_code,
        quarter,
        getattr(df, "shape", None),
    )
    return df


def _sse_pack(event_name: str, payload: Dict[str, Any]) -> bytes:
    """SSE 事件帧编码：``event: <name>\\ndata: <json>\\n\\n``。

    event_name 必须可作为 SSE ``event:`` 标识；浏览器 EventSource 客户端可
    按 ``addEventListener("identify", ...)`` 区分。
    """
    # ensure_ascii=False → 中文不被转义，面板显示更顺
    body = json.dumps(payload, ensure_ascii=False, default=str)
    return f"event: {event_name}\ndata: {body}\n\n".encode("utf-8")


def _linear_graph_edges(graph: Any) -> List[Dict[str, str]]:
    """把 DAG 边序列化成 [{source, target, label}]（spec §11 浏览器面板 JSON）。"""
    if graph is None:
        return []
    out: List[Dict[str, str]] = []
    for u, v in graph.edges():
        out.append(
            {
                "source": str(u),
                "target": str(v),
                "label": f"{u} → {v}",
            }
        )
    return out


def _nodes_payload(graph: Any) -> List[Dict[str, Any]]:
    if graph is None:
        return []
    out: List[Dict[str, Any]] = []
    for n in graph.nodes():
        attrs = dict(graph.nodes[n])
        out.append(
            {
                "id": str(n),
                "label": str(attrs.get("label", n)),
                "kind": str(attrs.get("kind", "")),
                "treatment": bool(attrs.get("treatment", False)),
                "outcome": bool(attrs.get("outcome", False)),
                "color": str(attrs.get("color", "#7f7f7f")),
            }
        )
    return out


# ---------------------------------------------------------------------------
# core streaming
# ---------------------------------------------------------------------------


async def _stream_modeling_events(
    disease_code: str,
    outcome: str,
    quarter: Optional[str],
    estimator: str,
    num_simulations: int,
    n_jobs: int,
) -> AsyncIterator[bytes]:
    """SSE 事件流的真正组装逻辑（与 HTTP 层解耦，便于测试）。"""
    started_at = time.monotonic()

    # ───── 0. 加载数据 ─────
    try:
        data = _load_data_for_disease(disease_code, quarter)
    except Exception as exc:  # noqa: BLE001 — fixture 加载任何失败都走 envelope
        yield _sse_pack(
            "error",
            {
                "stage": "load_data",
                "disease_code": disease_code,
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
        )
        return

    sample_size = int(len(data))
    yield _sse_pack(
        "ack",
        {
            "disease_code": disease_code,
            "outcome": outcome,
            "quarter": quarter,
            "sample_size": sample_size,
        },
    )

    # ───── 0.5 样本量守门 ─────
    guard = SampleSizeGuard()
    ssize_verdict = guard.check(sample_size)
    yield _sse_pack(
        "sample-size",
        {
            "sample_size": sample_size,
            "verdict": ssize_verdict.verdict,
            "allows_draft_posting": ssize_verdict.allows_draft_posting,
            "allows_backend_write": ssize_verdict.allows_backend_write,
            "degraded_reasons": ssize_verdict.degraded_reasons,
            "info": ssize_verdict.info,
        },
    )

    # 守门不通过（insufficient）→ 拒收：草稿不生成、上报不写回，但事件流仍走完
    # 剩余阶段供面板展示"为什么收手"。仅在数据 < 10 时硬拒返回（异常保护）。
    if sample_size < 10:
        yield _sse_pack(
            "error",
            {
                "stage": "sample_size",
                "message": f"样本量 {sample_size} < 10（系统硬下线）；请换季度或病种",
            },
        )
        return

    # ───── 1. 构造 CausalPipeline + identify ─────
    try:
        pipeline = CausalPipeline(
            disease_code=disease_code,
            outcome=outcome,
            data=data,
        )
    except Exception as exc:  # noqa: BLE001
        yield _sse_pack(
            "error",
            {
                "stage": "pipeline_construct",
                "message": str(exc),
                "error_type": type(exc).__name__,
            },
        )
        return

    yield _sse_pack(
        "graph",
        {
            "nodes": _nodes_payload(pipeline.graph),
            "edges": _linear_graph_edges(pipeline.graph),
        },
    )

    try:
        estimand = pipeline.call_identify_effect()
    except Exception as exc:
        # 识别失败不让整流失败：退化为去 estimand 走 None，effect stage 走降级
        yield _sse_pack(
            "identify",
            {
                "disease_code": disease_code,
                "outcome": outcome,
                "estimand_type": "nonparametric-ate",
                "treatment": pipeline.treatment,
                "backdoor": [],
                "instruments": [],
                "expression": "",
                "degraded": True,
                "degraded_reason": f"identify_effect failed: {type(exc).__name__}: {exc}",
            },
        )
        estimand = None  # noqa: F841 — 留作占位，下面 effect 阶段只用 pipeline + data

    if estimand is not None:
        yield _sse_pack(
            "identify",
            {
                "disease_code": disease_code,
                "outcome": outcome,
                "estimand_type": estimand.estimand_type,
                "treatment": estimand.treatment_name,
                "backdoor": estimand.backdoor_variables,
                "instruments": estimand.instrumental_variables,
                "expression": estimand.estimand_expression,
            },
        )

    # ───── 2. estimate（每条边/因素） ─────
    edge_estimates: List[Dict[str, Any]] = []
    try:
        effect_obj = pipeline.call_estimate_effect(method_name=estimator)
        edge_estimates.append(
            {
                "node": effect_obj.treatment,
                "ate": effect_obj.ate,
                "ci_lower": effect_obj.ci_lower,
                "ci_upper": effect_obj.ci_upper,
                "ci_width": effect_obj.ci_width,
                "p_value": effect_obj.p_value,
                "method_name": effect_obj.method_name,
                "sample_size": effect_obj.sample_size,
                "estimate_ci_unstable": effect_obj.estimate_ci_unstable,
            }
        )
        yield _sse_pack(
            "effect",
            {
                "kind": "treatment",
                **edge_estimates[-1],
            },
        )
    except Exception as exc:  # noqa: BLE001
        yield _sse_pack(
            "effect",
            {
                "kind": "treatment",
                "node": pipeline.treatment,
                "ate": None,
                "ci_lower": None,
                "ci_upper": None,
                "ci_width": None,
                "p_value": None,
                "method_name": estimator,
                "sample_size": sample_size,
                "estimate_ci_unstable": True,
                "degraded_reason": f"{type(exc).__name__}: {exc}",
            },
        )

    # ───── 2.5 旁路：每个 cov 节点 estimate（distress lookup 用） ─────
    # 旁路只是补充面板显示；不在 SPEC 必填要求内，任何节点都可降级 None。
    for cov_node in pipeline.common_causes:
        try:
            cov_est = pipeline.call_estimate_effect(method_name=estimator)  # 同一 estimand
            # 同一个 estimate，treatment 是同一个；面板"co-var 效应"用 effect_obj 的 ci 解释即可
            yield _sse_pack(
                "effect",
                {
                    "kind": "covariate",
                    "node": cov_node,
                    "ate": cov_est.ate,
                    "ci_lower": cov_est.ci_lower,
                    "ci_upper": cov_est.ci_upper,
                    "ci_width": cov_est.ci_width,
                    "p_value": cov_est.p_value,
                    "method_name": cov_est.method_name,
                    "sample_size": cov_est.sample_size,
                    "estimate_ci_unstable": cov_est.estimate_ci_unstable,
                },
            )
        except Exception as exc:  # noqa: BLE001
            yield _sse_pack(
                "effect",
                {
                    "kind": "covariate",
                    "node": cov_node,
                    "ate": None,
                    "ci_lower": None,
                    "ci_upper": None,
                    "ci_width": None,
                    "p_value": None,
                    "method_name": estimator,
                    "sample_size": sample_size,
                    "estimate_ci_unstable": True,
                    "degraded_reason": f"{type(exc).__name__}: {exc}",
                },
            )

    # ───── 3. refute（4 档串行推送；任何单档异常隔离） ─────
    try:
        # 取 dowhy CausalModel + EffectEstimate（业务层包装）→ 构造 RefuterPipeline
        dowhy_cm = pipeline._build_causal_model()  # noqa: SLF001 — 私有但 spec §11 允许
        if estimand is not None and len(edge_estimates) > 0:
            refuter = RefuterPipeline(
                causal_model=dowhy_cm,
                estimate=effect_obj,
                estimand=estimand,
                data=data,
                methods=list(SUPPORTED_METHODS),
                num_simulations=num_simulations,
            )
            if n_jobs != 1:
                refutation_results = refuter.run_parallel(n_jobs=n_jobs)
            else:
                refutation_results = refuter.run()
            overall = refuter.verdict(refutation_results)
            for r in refutation_results:
                yield _sse_pack(
                    "refute",
                    _refuter_to_payload(r),
                )
            yield _sse_pack(
                "refute-summary",
                {
                    "verdict": overall,
                    "n_passed": sum(1 for r in refutation_results if r.passed),
                    "n_total": len(refutation_results),
                    "n_unstable": sum(
                        1 for r in refutation_results if r.refutation_unstable
                    ),
                },
            )
        else:
            yield _sse_pack(
                "refute-summary",
                {
                    "verdict": "unstable",
                    "n_passed": 0,
                    "n_total": 0,
                    "n_unstable": 0,
                    "skipped_reason": "identiy_or_estimate_failed",
                },
            )
    except Exception as exc:  # noqa: BLE001 — 总入口兜底
        yield _sse_pack(
            "refute-summary",
            {
                "verdict": "unstable",
                "n_passed": 0,
                "n_total": 0,
                "n_unstable": 0,
                "exception": f"{type(exc).__name__}: {exc}",
            },
        )

    # ───── 4. baseline（因果 vs 单因素相关） ─────
    try:
        ate_for_wedge: Optional[float] = edge_estimates[0]["ate"] if edge_estimates else None
        baseline = compute_correlation(data, treatment=pipeline.treatment, outcome=outcome)
        wedge_helper = CorrelationBaseline()
        wedge = wedge_helper.wedge(ate=ate_for_wedge, correlation=baseline.correlation)
        yield _sse_pack(
            "baseline",
            {
                "disease_code": disease_code,
                "outcome": outcome,
                "treatment": pipeline.treatment,
                "correlation": baseline.correlation,
                "sample_size": baseline.sample_size,
                "treatment_encoded": baseline.treatment_encoded,
                "degraded": baseline.degraded,
                "degraded_reasons": baseline.degraded_reasons,
                "wedge": wedge,
                "causal_wedge_strong": wedge.get("causal_wedge_strong"),
            },
        )
    except Exception as exc:  # noqa: BLE001
        yield _sse_pack(
            "baseline",
            {
                "disease_code": disease_code,
                "outcome": outcome,
                "treatment": pipeline.treatment,
                "correlation": None,
                "sample_size": sample_size,
                "causal_wedge_strong": None,
                "degraded": True,
                "exception": f"{type(exc).__name__}: {exc}",
            },
        )

    # ───── 5. done ─────
    elapsed_ms = int((time.monotonic() - started_at) * 1000)
    yield _sse_pack(
        "done",
        {
            "disease_code": disease_code,
            "outcome": outcome,
            "quarter": quarter,
            "elapsed_ms": elapsed_ms,
            "sample_size": sample_size,
            "sample_size_verdict": ssize_verdict.verdict,
            "estimator": estimator,
            "n_jobs": n_jobs,
        },
    )


def _refuter_to_payload(r: RefutationResult) -> Dict[str, Any]:
    """RefutationResult → SSE 字典。

    RefutationResult 字段（schema 节点）：method / new_effect / original_effect /
    p_value / passed / degraded / degraded_reasons / refutation_unstable / n_jobs /
    ci_lower / ci_upper / ci_width。
    """
    return {
        "method": r.method,
        "new_effect": r.new_effect,
        "original_effect": r.original_effect,
        "p_value": r.p_value,
        "passed": r.passed,
        "degraded": r.degraded,
        "degraded_reasons": list(r.degraded_reasons or []),
        "refutation_unstable": r.refutation_unstable,
        "n_jobs": r.n_jobs,
        "ci_lower": getattr(r, "ci_lower", None),
        "ci_upper": getattr(r, "ci_upper", None),
        "ci_width": getattr(r, "ci_width", None),
    }


# ---------------------------------------------------------------------------
# FastAPI 路由
# ---------------------------------------------------------------------------


@router.post("/modeling/run")
async def modeling_run(
    payload: Dict[str, Any],
    x_quality_role: Optional[str] = Header(default=None, alias="X-Quality-Role"),
) -> StreamingResponse:
    """POST /api/modeling/run → SSE 流式建模主路径。

    Body
    ----
    {
      "disease_code": "I21" | "I63" | "Z96.10",
      "quarter":      "2025-Q1" (str, 可选，仅日志用),
      "outcome":      "cmi" | "los_days" | "readmit_flag" | "complication_flag",
      "estimator":    "backdoor.linear_regression" (可选),
      "num_simulations": int (可选，默认 20),
      "n_jobs":       int (可选，默认 -1)
    }

    Returns
    -------
    StreamingResponse(text/event-stream)
        事件序列：ack → sample-size → graph → identify → effect* → refute*
        → refute-summary → baseline → done。
    """
    role = _check_role_header(x_quality_role)

    disease_code = str(payload.get("disease_code") or "").strip()
    outcome = str(payload.get("outcome") or "").strip()
    quarter_raw = payload.get("quarter")
    quarter = str(quarter_raw).strip() if quarter_raw else None

    if not disease_code:
        raise ValueError("body.disease_code 不能为空")
    if not outcome:
        raise ValueError("body.outcome 不能为空")

    estimator = str(payload.get("estimator") or _DEFAULT_ESTIMATOR)
    num_simulations = int(payload.get("num_simulations") or _DEFAULT_NUM_SIMULATIONS)
    n_jobs = int(payload.get("n_jobs") if payload.get("n_jobs") is not None else _DEFAULT_REPUTE_JOBS)

    LOGGER.info(
        "POST /api/modeling/run role=%s disease=%s outcome=%s quarter=%s estimator=%s",
        role,
        disease_code,
        outcome,
        quarter,
        estimator,
    )

    return StreamingResponse(
        _stream_modeling_events(
            disease_code=disease_code,
            outcome=outcome,
            quarter=quarter,
            estimator=estimator,
            num_simulations=num_simulations,
            n_jobs=n_jobs,
        ),
        media_type="text/event-stream",
        status_code=status.HTTP_200_OK,
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # nginx: 关闭缓冲，让流即时推
            "X-Quality-Role": role,
        },
    )
