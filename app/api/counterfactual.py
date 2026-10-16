"""反事实批量端点（spec.md §11 第 4 步批量反事实 + §9 因果推理管线）。

POST /api/counterfactual/run
  Body: {
    "disease_code": "I21" | "I63" | "Z96.10",
    "outcome":      "cmi" | "los_days" | "readmit_flag" | "complication_flag",
    "quarter":      "2025-Q1" (可选, 仅日志),
    "scenarios": [
      {"intervention_node": "treatment:PCI", "hypothetical_value": "PCI"},
      {"intervention_node": "age_band:3",    "hypothetical_value": 3},
      ...
    ]
  }
  Header: X-Quality-Role ∈ {quality_officer, med_committee_reviewer}
  响应: JSON {
    "disease_code", "outcome", "quarter",
    "n_scenarios": int,
    "n_completed": int,
    "n_timed_out": int,
    "n_errored": int,
    "scenarios": [ CounterfactualScenario, ... ]   # 与输入 scenarios 等长
  }

失败降级（spec §6）：
- 单条场景异常 → 该条 scenario.timeout=True 或 do_operations 含 exception 标记；
  其它场景不受影响（spec §6 失败不静默改数据）。
- 总超时（30s）由 BatchCounterfactualRunner 内置。
- 鉴权失败 → QualityRoleForbiddenError → 403 envelope。

源产品能力参考（README 不写源仓库名）：
- BatchCounterfactualRunner 沿用 T12 CounterfactualRunner 的同一拟合路径；
  差异只在并发调度 + 总量门。
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, status
from fastapi.responses import JSONResponse

from app.core.dosampler_batch import (
    BATCH_SIZE_DEFAULT,
    BatchCounterfactualRunner,
    Intervention,
)
from app.core.exception_handlers import QualityRoleForbiddenError

LOGGER = logging.getLogger(__name__)

router = APIRouter(tags=["counterfactual"])

# 与 modeling 共用鉴权矩阵（spec §11 第 6 步）
_ALLOWED_ROLES = frozenset({"quality_officer", "med_committee_reviewer"})

# 批量场景并发上限（spec §11 第 4 步）
BATCH_SCENARIOS_LIMIT = 50


def _check_role_header(role_header: Optional[str]) -> str:
    if not role_header:
        raise QualityRoleForbiddenError(
            "X-Quality-Role header 缺失；反事实 API 必须由质控员或评审角色触发",
            role="<missing>",
        )
    if role_header not in _ALLOWED_ROLES:
        raise QualityRoleForbiddenError(
            f"X-Quality-Role={role_header!r} 不在允许集合 {_ALLOWED_ROLES!r}",
            role=role_header,
        )
    return role_header


def _normalize_scenarios(raw: Any) -> List[Intervention]:
    """校验 + 转换 body.scenarios 为 List[Intervention]。

    校验规则：
      - 必须是 list（非 list → ValueError envelope）
      - 元素必须是 dict（含 intervention_node / hypothetical_value）
      - intervention_node: 非空字符串
      - hypothetical_value: 非 None（值类型由 Intervention 自带校验兜底）
    """
    if raw is None:
        raise ValueError("body.scenarios 不能为 None；至少传 1 条场景")
    if not isinstance(raw, list):
        raise ValueError("body.scenarios 必须是 list")
    if len(raw) == 0:
        raise ValueError("body.scenarios 至少 1 条场景")
    if len(raw) > BATCH_SCENARIOS_LIMIT:
        raise ValueError(
            f"body.scenarios 长度 {len(raw)} > 上限 {BATCH_SCENARIOS_LIMIT}"
        )

    out: List[Intervention] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(
                f"scenarios[{i}] 必须是 dict；实际 {type(item).__name__}"
            )
        node = item.get("intervention_node")
        hyp = item.get("hypothetical_value")
        label = str(item.get("label") or "")
        if not isinstance(node, str) or not node:
            raise ValueError(f"scenarios[{i}].intervention_node 必须是非空字符串")
        if hyp is None:
            raise ValueError(f"scenarios[{i}].hypothetical_value 不能为 None")
        out.append(
            Intervention(
                intervention_node=node,
                hypothetical_value=hyp,
                label=label,
            )
        )
    return out


def _scenario_to_dict(s) -> Dict[str, Any]:
    """CounterfactualScenario → 字典（面板可序列化）。

    防御性处理 numpy / pandas 类型（避免 int64 / float64 不能 JSON 序列化）。
    """
    def _to_native(v):
        # numpy scalar → Python 原生类型
        try:
            import numpy as np
            if isinstance(v, np.integer):
                return int(v)
            if isinstance(v, np.floating):
                return float(v)
            if isinstance(v, np.bool_):
                return bool(v)
        except ImportError:
            pass
        return v

    return {
        "scenario_id": _to_native(s.scenario_id),
        "disease_code": _to_native(s.disease_code),
        "outcome": _to_native(s.outcome),
        "intervention_node": _to_native(s.intervention_node),
        "hypothetical_value": _to_native(s.hypothetical_value),
        "baseline_value": _to_native(s.baseline_value),
        "predicted_outcome": _to_native(s.predicted_outcome),
        "confidence": _to_native(s.confidence),
        "do_operations": [_to_native(op) for op in (s.do_operations or [])],
        "elapsed_ms": _to_native(s.elapsed_ms),
        "timeout": bool(s.timeout),
        "baseline_std": _to_native(s.baseline_std),
    }


@router.post("/counterfactual/run")
async def counterfactual_run(
    payload: Dict[str, Any],
    x_quality_role: Optional[str] = Header(default=None, alias="X-Quality-Role"),
) -> JSONResponse:
    """POST /api/counterfactual/run → 批量反事实 JSON 响应。

    Body
    ----
    {
      "disease_code": "I21" | "I63" | "Z96.10",
      "outcome":      "cmi" | "los_days" | "readmit_flag" | "complication_flag",
      "quarter":      "2025-Q1" (可选),
      "scenarios": [
        {"intervention_node": "treatment:PCI", "hypothetical_value": "PCI", "label": "全部 PCI"},
        ...
      ],
      "batch_size": int (可选; 默认 10)
    }

    Returns
    -------
    JSONResponse
        {
          "schema_version", "disease_code", "outcome", "quarter",
          "n_scenarios", "n_completed", "n_timed_out", "n_errored",
          "scenarios": [...]   # 与 body.scenarios 等长
        }
    """
    role = _check_role_header(x_quality_role)

    disease_code = str(payload.get("disease_code") or "").strip()
    outcome = str(payload.get("outcome") or "").strip()
    quarter_raw = payload.get("quarter")
    quarter = str(quarter_raw).strip() if quarter_raw else None
    batch_size = int(payload.get("batch_size") or BATCH_SIZE_DEFAULT)

    if not disease_code:
        raise ValueError("body.disease_code 不能为空")
    if not outcome:
        raise ValueError("body.outcome 不能为空")
    if batch_size < 1:
        raise ValueError("body.batch_size 必须 >= 1")

    scenarios_in = _normalize_scenarios(payload.get("scenarios"))

    LOGGER.info(
        "POST /api/counterfactual/run role=%s disease=%s outcome=%s quarter=%s n_scenarios=%d",
        role,
        disease_code,
        outcome,
        quarter,
        len(scenarios_in),
    )

    # 加载数据（沿用 modeling 端点策略：spec §11 测试替他 — 真实 HIS 直连由 T4 完成）
    from data.seed_loader import load_fixture_csv

    started = time.monotonic()
    try:
        data = load_fixture_csv(disease_code)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("加载 fixture 失败 %s: %s", disease_code, exc)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "schema_version": "counterfactual_response/1.0",
                "ok": False,
                "error_type": type(exc).__name__,
                "error_code": "load_data_failed",
                "message": str(exc),
                "disease_code": disease_code,
                "outcome": outcome,
            },
        )

    # 跑批量反事实
    runner = BatchCounterfactualRunner(
        disease_code=disease_code,
        outcome=outcome,
        data=data,
        batch_size=batch_size,
    )
    scenarios_out = runner.run(scenarios_in)

    # 聚合指标
    n_total = int(len(scenarios_out))
    n_timed_out = int(sum(1 for s in scenarios_out if bool(getattr(s, "timeout", False))))
    n_errored = int(sum(
        1
        for s in scenarios_out
        if any(
            (isinstance(op, str) and op.startswith("exception:"))
            or (isinstance(op, str) and op.startswith("validation_error:"))
            for op in (s.do_operations or [])
        )
    ))
    n_completed = int(n_total - n_timed_out - n_errored)

    response_body: Dict[str, Any] = {
        "schema_version": "counterfactual_response/1.0",
        "ok": True,
        "disease_code": disease_code,
        "outcome": outcome,
        "quarter": quarter,
        "n_scenarios": n_total,
        "n_completed": n_completed,
        "n_timed_out": n_timed_out,
        "n_errored": n_errored,
        "elapsed_ms": int((time.monotonic() - started) * 1000),
        "batch_size": batch_size,
        "scenarios": [_scenario_to_dict(s) for s in scenarios_out],
    }
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=response_body,
        headers={"X-Quality-Role": role},
    )
