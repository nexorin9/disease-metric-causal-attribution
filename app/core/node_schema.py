"""Pydantic schema：节点 / 边 / 效应 / 反驳 / 反事实 / 审计事件。

这些 schema 是面板 SSE 流式推送 + 数据库草稿 + ModelGateway 输入/输出的
统一接口契约。所有路径共用同一组类型，便于跨模块序列化。

约定：
- 数值字段允许 None（不可用 / 不可信 / 样本不足时显式标 null）
- 时间戳统一 ISO8601（panel 端按本地时区显示）
- 严禁 PII 字段（spec §6）— schema 中不出现 patient_name / id_card / phone / address
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _StrictModel(BaseModel):
    """统一配置：禁止未知字段 + 验证赋值。"""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )


# ----------------------------------------------------------------------------------
# 因果图节点 / 边（面板 Plotly.js 数据契约）
# ----------------------------------------------------------------------------------


class Node(_StrictModel):
    """因果图节点。"""

    id: str = Field(..., description="节点 id（与 DAG nx.DiGraph 节点一致）")
    label: str = Field(..., description="面板显示名")
    kind: Literal["treatment", "covariate", "outcome"] = Field(...)
    color: str = Field(..., description="节点颜色（hex / CSS color）")
    semantic: str = Field("", description="业务语义说明（spec §3.1 节点语义）")
    spec_section: str = Field("", description="spec 章节定位")

    @field_validator("id", "label")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("id / label 不可为空")
        return v


class Edge(_StrictModel):
    """因果图边。"""

    source: str = Field(...)
    target: str = Field(...)
    kind: Literal["treatment_to_outcome", "covariate_to_outcome", "background"] = Field(...)
    color: str = Field(...)
    label: str = Field("")
    spec_section: str = Field("")


# ----------------------------------------------------------------------------------
# 效应估计（estimate_effect 产出）
# ----------------------------------------------------------------------------------


class EffectEstimate(_StrictModel):
    """estimate_effect 单边结果。"""

    node: str = Field(..., description="候选因子节点 id")
    treatment: str = Field(..., description="treatment 节点 id（从 → outcome 边的源节点）")
    outcome: str = Field(..., description="outcome 节点 id")
    method_name: str = Field(..., description="估计器名（如 backdoor.linear_regression）")
    ate: Optional[float] = Field(None, description="Average Treatment Effect")
    ci_lower: Optional[float] = Field(None, description="置信区间下界")
    ci_upper: Optional[float] = Field(None, description="置信区间上界")
    p_value: Optional[float] = Field(None, description="p-value；None 表示不可计算")
    ci_width: Optional[float] = Field(None, description="ci_upper - ci_lower")
    estimate_ci_unstable: bool = Field(
        False, description="True = CI 超宽（>spec.max_ci_width），结论不可信"
    )
    sample_size: int = Field(..., ge=0, description="实际用于估计的样本量")
    control_value: Any = Field(None, description="对照值（treatment baseline）")
    treatment_value: Any = Field(None, description="处理值（treatment intervention）")

    @field_validator("method_name")
    @classmethod
    def _known_method(cls, v: str) -> str:
        known = {
            "backdoor.propensity_score_matching",
            "backdoor.linear_regression",
            "backdoor.propensity_score_weighting",
            "iv.instrumental_variable",
        }
        if v not in known:
            raise ValueError(f"未知估计器: {v!r}；已知: {sorted(known)}")
        return v


# ----------------------------------------------------------------------------------
# 反驳结果（refute_estimate 产出）
# ----------------------------------------------------------------------------------


class RefutationResult(_StrictModel):
    """refute_estimate 单档反驳结果。"""

    method: Literal[
        "random_common_cause",
        "placebo_treatment",
        "bootstrap",
        "data_subset",
        "dummy_outcome",
    ] = Field(...)
    new_effect: Optional[float] = Field(None, description="反驳后的效应估计值")
    original_effect: Optional[float] = Field(None, description="原始效应估计值")
    p_value: Optional[float] = Field(None, description="反驳 p-value")
    # T10: 显式记录置信区间字段，供 is_degraded() 判断 CI 超宽（ci_upper - ci_lower > 0.6）。
    # 三者均为可选；dowhy 部分版本不暴露 ci_upper/ci_lower 时留空即可（仍可走 p_value 阈值判定）。
    ci_lower: Optional[float] = Field(None, description="反驳后效应置信区间下界")
    ci_upper: Optional[float] = Field(None, description="反驳后效应置信区间上界")
    ci_width: Optional[float] = Field(
        None, description="置信区间宽度（ci_upper - ci_lower）；None 表示不可用"
    )
    passed: Optional[bool] = Field(None, description="True = 反驳通过（结论稳健）")
    degraded: bool = Field(False, description="True = 反驳降级（CI 超宽 / 拒收敛）")
    degraded_reasons: List[str] = Field(
        default_factory=list, description="降级原因列表（如 ci_width_too_wide / no_convergence）"
    )
    refutation_unstable: bool = Field(False, description="反驳本身失败")
    n_jobs: int = Field(1, description="并行 worker 数；-1 表示全部 CPU")


# ----------------------------------------------------------------------------------
# 反事实场景（do_sample 产出）
# ----------------------------------------------------------------------------------


class CounterfactualScenario(_StrictModel):
    """单条反事实预测结果。"""

    scenario_id: str = Field(..., description="场景 id（UUID / 哈希）")
    disease_code: str = Field(...)
    outcome: str = Field(...)
    intervention_node: str = Field(..., description="被干预的节点 id")
    hypothetical_value: Any = Field(..., description="干预后的取值")
    baseline_value: Any = Field(None, description="原始观测值")
    predicted_outcome: Optional[float] = Field(None, description="do() 后的预测 outcome")
    confidence: Optional[float] = Field(None, description="预测置信度（0-1）")
    do_operations: List[str] = Field(
        default_factory=list, description="do() 操作序列（用于可解释性展示）"
    )
    elapsed_ms: int = Field(0, ge=0, description="耗时（毫秒）")
    timeout: bool = Field(False, description="True = do() 超时")
    baseline_std: Optional[float] = Field(None, description="基线 outcome 标准差")


# ----------------------------------------------------------------------------------
# 审计事件（audit_logs 表 + 跨模块统一）
# ----------------------------------------------------------------------------------


class AuditEvent(_StrictModel):
    """单条审计事件。

    event_type 取值见 spec §6 维度 + audit/logger.py 注册列表。
    payload 字段在写入数据库前由 strip_pii() 强制剔除 patient_name / id_card / phone / address。
    """

    event_type: Literal[
        "modeling/run",
        "draft/post",
        "state/transition",
        "model/call",
        "his/query",
        "refutation/degraded",
        "counterfactual/run",
        "sample/insufficient",
    ] = Field(...)
    operator_id: str = Field(..., description="操作人 / 服务标识")
    operator_role: str = Field("", description="角色（med_committee_reviewer / system / …）")
    disease_code: str = Field("", description="关联病种；跨病种任务用空串")
    quarter: str = Field("", description="关联季度；如 2026Q3")
    outcome: str = Field("", description="关联 outcome；如 cmi / los_days")
    payload: Dict[str, Any] = Field(default_factory=dict, description="事件负载（已脱敏）")
    ts: datetime = Field(default_factory=datetime.utcnow, description="UTC 时间戳")
    state_not_advanced: bool = Field(
        True, description="True = 事件未推进业务状态（默认审计事件不静默改数据）"
    )
