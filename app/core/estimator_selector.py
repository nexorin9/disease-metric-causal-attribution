"""estimator 自动切换策略器（spec.md §11 estimator 切换策略器）。

业务动机：
- 流水线层不强制写死估计器；当样本量 / 工具变量可用性变化时，自动选最合适的估计器：
    * n < SMALL_SAMPLE_THRESHOLD            → backdoor.linear_regression（OLS 在小样本下最稳定）
    * n >= SMALL_SAMPLE_THRESHOLD + has_iv  → iv.instrumental_variable（IV 可用时优先，未观测混杂更稳健）
    * 其它                                  → backdoor.propensity_score_matching（中等样本下 PSM 偏差/方差平衡）
- 面板「建模」按钮触发 CausalPipeline 之前先选 estimator，再走 identify/estimate/refute。
- 这是 (a) 深度：策略决策器让流水线能根据数据特征自适应。

数据契约：
- EstimatorChoice{ method_name, sample_size, has_iv, rationale, alternatives_considered }
- 决策表与 config 完全对齐 spec.md §11 第 6 步硬约束；可通过 SPEC_OVERRIDE_ESTIMATOR env 强制覆盖（开发态）

源产品能力参考（README 不写源 repo 名）：
- CausalModel.identify_effect + estimate_effect(method_name=...) 来自方法族实现的统一接口；
- 本模块仅为策略选择层，不直接调用方法族实现（保持与 pipeline.py 的低耦合）。

⚠️ README 不得写源项目名（dowhy / pywhy）；本模块仅描述方法族。
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------
# 常量
# ----------------------------------------------------------------------------------

# 估计器候选池（与 pipeline.SUPPORTED_ESTIMATORS 对齐；spec §11 第 6 步）。
ESTIMATOR_LINEAR_REGRESSION: str = "backdoor.linear_regression"
ESTIMATOR_PROPENSITY_MATCHING: str = "backdoor.propensity_score_matching"
ESTIMATOR_IV: str = "iv.instrumental_variable"

# 决策阈值：样本量 < SMALL_SAMPLE_THRESHOLD 选 LR（OLS 在小样本下最稳定）。
SMALL_SAMPLE_THRESHOLD: int = 100

# 已知估计器集合（用于 ValidationError；不可静默接受未知方法）。
_KNOWN_ESTIMATORS: frozenset = frozenset(
    {
        ESTIMATOR_LINEAR_REGRESSION,
        ESTIMATOR_PROPENSITY_MATCHING,
        ESTIMATOR_IV,
        "backdoor.propensity_score_weighting",  # 业务备用
    }
)

# 环境变量：开发态强制覆盖（spec §6 可观测；default=None 即不覆盖）。
SPEC_OVERRIDE_ENV: str = "SPEC_OVERRIDE_ESTIMATOR"


# ----------------------------------------------------------------------------------
# 数据契约
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class EstimatorChoice:
    """策略器输出（spec §11 第 6 步）。

    字段：
      - method_name: 选中的估计器方法名（与 pipeline.call_estimate_effect(method_name=...) 对齐）
      - sample_size: 输入样本量（透传，便于审计）
      - has_iv: 输入是否声明有工具变量可用（透传）
      - rationale: 决策理由文本（供面板展示 / 审计日志）
      - alternatives_considered: 决策表里所有候选及是否被选中（供审计）
    """

    method_name: str
    sample_size: int
    has_iv: bool
    rationale: str
    alternatives_considered: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.method_name not in _KNOWN_ESTIMATORS:
            raise ValueError(
                f"未知估计器: {self.method_name!r}；已知: {sorted(_KNOWN_ESTIMATORS)}"
            )
        if self.sample_size < 0:
            raise ValueError(f"sample_size 必须 >= 0；got {self.sample_size}")
        if not isinstance(self.has_iv, bool):
            raise TypeError(
                f"has_iv 必须是 bool；got {type(self.has_iv).__name__}"
            )
        if not self.rationale:
            raise ValueError("rationale 不可为空")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_name": self.method_name,
            "sample_size": self.sample_size,
            "has_iv": self.has_iv,
            "rationale": self.rationale,
            "alternatives_considered": list(self.alternatives_considered),
        }


# ----------------------------------------------------------------------------------
# 决策表
# ----------------------------------------------------------------------------------


def _decision_table(sample_size: int, has_iv: bool) -> List[Dict[str, Any]]:
    """返回决策表（每条候选 + 是否被选中）。

    顺序：
      1. n<100                    → LR（小样本）
      2. n>=100 + has_iv          → IV（工具变量可用）
      3. 其它                     → PSM（中等样本）

    每条候选含：
      - method_name
      - selected: bool
      - reason:   决策原因（被选中的理由；其它候选说明为什么没被选）
    """
    rows: List[Dict[str, Any]] = []

    if sample_size < SMALL_SAMPLE_THRESHOLD:
        # 规则 1：小样本 → LR
        rows.append(
            {
                "method_name": ESTIMATOR_LINEAR_REGRESSION,
                "selected": True,
                "reason": (
                    f"sample_size={sample_size} < SMALL_SAMPLE_THRESHOLD="
                    f"{SMALL_SAMPLE_THRESHOLD} → 选择 LR（小样本下 OLS 最稳定）"
                ),
            }
        )
        rows.append(
            {
                "method_name": ESTIMATOR_IV,
                "selected": False,
                "reason": (
                    f"sample_size={sample_size} < "
                    f"{SMALL_SAMPLE_THRESHOLD}：IV 在小样本下偏弱（first-stage "
                    "F 检验可能不显著），优先 LR"
                ),
            }
        )
        rows.append(
            {
                "method_name": ESTIMATOR_PROPENSITY_MATCHING,
                "selected": False,
                "reason": (
                    f"sample_size={sample_size} < "
                    f"{SMALL_SAMPLE_THRESHOLD}：PSM 需要更大样本做匹配，"
                    "小样本下倾向得分估计不准；优先 LR"
                ),
            }
        )
        return rows

    # sample_size >= SMALL_SAMPLE_THRESHOLD
    if has_iv:
        # 规则 2：大样本 + IV 可用 → IV
        rows.append(
            {
                "method_name": ESTIMATOR_IV,
                "selected": True,
                "reason": (
                    f"sample_size={sample_size} >= {SMALL_SAMPLE_THRESHOLD} "
                    "且 has_iv=True → 选择 IV（未观测混杂下更稳健）"
                ),
            }
        )
        rows.append(
            {
                "method_name": ESTIMATOR_LINEAR_REGRESSION,
                "selected": False,
                "reason": (
                    f"sample_size={sample_size}，LR 仍可用但 IV 在 has_iv=True "
                    "下能控制未观测混杂，IV 优先"
                ),
            }
        )
        rows.append(
            {
                "method_name": ESTIMATOR_PROPENSITY_MATCHING,
                "selected": False,
                "reason": (
                    "PSM 仅控制观测到的混杂；has_iv=True 时 IV 能进一步控制未观测混杂，PSM 退而求其次"
                ),
            }
        )
        return rows

    # 规则 3：大样本 + 无 IV → PSM
    rows.append(
        {
            "method_name": ESTIMATOR_PROPENSITY_MATCHING,
            "selected": True,
            "reason": (
                f"sample_size={sample_size} >= {SMALL_SAMPLE_THRESHOLD} "
                "且 has_iv=False → 选择 PSM（中等样本下偏差/方差平衡）"
            ),
        }
    )
    rows.append(
        {
            "method_name": ESTIMATOR_IV,
            "selected": False,
            "reason": "has_iv=False：无可用工具变量，跳过 IV"
        },
    )
    rows.append(
        {
            "method_name": ESTIMATOR_LINEAR_REGRESSION,
            "selected": False,
            "reason": (
                f"sample_size={sample_size}，LR 假设线性可加效应，PSM 在非线性 "
                "treatment 效应下偏差更小，PSM 优先"
            ),
        }
    )
    return rows


# ----------------------------------------------------------------------------------
# 策略器主体
# ----------------------------------------------------------------------------------


class EstimatorSelector:
    """根据样本量 + 工具变量可用性自动选估计器（spec §11 第 6 步 / T17）。

    Usage
    -----
    >>> EstimatorSelector().select(sample_size=80, has_iv=False)
    EstimatorChoice(method_name='backdoor.linear_regression', ...)
    >>> EstimatorSelector().select(sample_size=500, has_iv=True)
    EstimatorChoice(method_name='iv.instrumental_variable', ...)
    """

    def __init__(self, *, override_method: Optional[str] = None) -> None:
        """构造器。

        Parameters
        ----------
        override_method
            可选：强制覆盖决策结果（开发 / 调试态使用）；默认 None 时按决策表走。
            也可从环境变量 SPEC_OVERRIDE_ESTIMATOR 读取（若 override_method 未显式传入）。
        """
        self.override_method = override_method
        if self.override_method is None:
            env_override = os.environ.get(SPEC_OVERRIDE_ENV)
            if env_override:
                self.override_method = env_override

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def select(
        self,
        sample_size: int,
        has_iv: bool,
        *,
        outcome: Optional[str] = None,
    ) -> EstimatorChoice:
        """根据 sample_size + has_iv 选择估计器。

        Parameters
        ----------
        sample_size
            用于估计的样本量（已 n==spec 数据过滤后实际样本）。
        has_iv
            是否声明有工具变量可用（业务 DAG 中 IV 候选列表非空即 True）。
        outcome
            可选：业务 outcome 名（cmi / los_days / ...）；仅用于 rationale 文案，
            不参与决策（决策只依赖 sample_size + has_iv）。

        Returns
        -------
        EstimatorChoice
            含 method_name + rationale + alternatives_considered。

        Raises
        ------
        ValueError
            sample_size < 0 / has_iv 非 bool
        """
        if sample_size < 0:
            raise ValueError(f"sample_size 必须 >= 0；got {sample_size}")
        if not isinstance(has_iv, bool):
            raise TypeError(
                f"has_iv 必须是 bool；got {type(has_iv).__name__}"
            )

        # 强制覆盖（开发 / 调试）
        if self.override_method is not None:
            if self.override_method not in _KNOWN_ESTIMATORS:
                raise ValueError(
                    f"override_method 未知: {self.override_method!r}；"
                    f"已知: {sorted(_KNOWN_ESTIMATORS)}"
                )
            rationale = (
                f"强制覆盖（override={self.override_method}）；"
                f"忽略决策表（sample_size={sample_size}, has_iv={has_iv}"
                f"{', outcome=' + outcome if outcome else ''}）"
            )
            LOGGER.warning(
                "EstimatorSelector 被强制覆盖：method=%s（sample_size=%d, has_iv=%s）",
                self.override_method,
                sample_size,
                has_iv,
            )
            return EstimatorChoice(
                method_name=self.override_method,
                sample_size=sample_size,
                has_iv=has_iv,
                rationale=rationale,
                alternatives_considered=[
                    {
                        "method_name": self.override_method,
                        "selected": True,
                        "reason": "强制覆盖（override_method=...",
                    }
                ],
            )

        # 走决策表
        rows = _decision_table(sample_size, has_iv)
        chosen = next((r for r in rows if r["selected"]), None)
        if chosen is None:
            # 决策表永远应有 1 个 selected；缺失视为实现 bug
            raise RuntimeError(
                "EstimatorSelector 决策表无 selected 行；"
                f"rows={rows!r}"
            )

        outcome_suffix = f"（outcome={outcome}）" if outcome else ""
        rationale = (
            f"{chosen['reason']}{outcome_suffix}"
        )

        return EstimatorChoice(
            method_name=chosen["method_name"],
            sample_size=sample_size,
            has_iv=has_iv,
            rationale=rationale,
            alternatives_considered=list(rows),
        )


# ----------------------------------------------------------------------------------
# 便捷函数
# ----------------------------------------------------------------------------------


def select_estimator(
    sample_size: int,
    has_iv: bool,
    *,
    outcome: Optional[str] = None,
    override_method: Optional[str] = None,
) -> EstimatorChoice:
    """EstimatorSelector().select() 的最简封装。"""
    return EstimatorSelector(override_method=override_method).select(
        sample_size=sample_size,
        has_iv=has_iv,
        outcome=outcome,
    )


# ----------------------------------------------------------------------------------
# 模块常量导出（供 pipeline / api 引用）
# ----------------------------------------------------------------------------------

__all__ = [
    "EstimatorChoice",
    "EstimatorSelector",
    "ESTIMATOR_LINEAR_REGRESSION",
    "ESTIMATOR_PROPENSITY_MATCHING",
    "ESTIMATOR_IV",
    "SMALL_SAMPLE_THRESHOLD",
    "select_estimator",
]