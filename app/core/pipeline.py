"""因果推理四步流水线入口（spec.md §11 主路径）。

CausalPipeline 封装 dowhy 的 CausalModel（model → identify → estimate → refute 四步），
为医院单病种评审场景提供：
  - CausalPipeline(disease_code, outcome, data)  →  一行构造
  - call_identify_effect()                         →  estimand（spec §9 流水线入口）
  - call_estimate_effect(method_name)              →  EffectEstimate（T7 扩展）
  - call_refute_estimate(method_name, n_jobs=-1)    →  RefutationResult（T8-T10 扩展）

异常（业务层捕获 → 走 spec §6 失败降级）：
  - EstimandNotIdentifiedError：identify_effect 返回 None / 不可识别
  - InsufficientSampleError：n < spec.min_sample_size（默认 30）

源产品能力参考：
- CausalModel 构造（data + treatment + outcome + graph）见 causal_model.py:18-67
- identify_effect 返回 estimand（backdoor / IV / front-door 等识别方法）
- 见 GITHUB_CODE_REF.md → causal_model.py + causal_identifier/

⚠️ README 不得写源项目名（dowhy / pywhy）；本模块仅描述方法族。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

import networkx as nx
import pandas as pd

LOGGER = logging.getLogger(__name__)

# 故意延迟导入 dowhy 内的 CausalModel（避免模块加载时强依赖副作用）


class EstimandNotIdentifiedError(RuntimeError):
    """identify_effect 未能从图 + 数据中识别出可估的因果 estimand。

    业务处理：spec §6 失败降级 — 标 refutation_unstable，草稿可生成但不触发写回。
    """


class InsufficientSampleError(RuntimeError):
    """样本量不足以跑因果估计（spec §6 样本量阈值）。"""

    def __init__(self, sample_size: int, min_required: int) -> None:
        super().__init__(
            f"样本量不足: actual={sample_size}, required>={min_required}（spec §6）"
        )
        self.sample_size = sample_size
        self.min_required = min_required


@dataclass
class Estimand:
    """identify_effect 包装对象（业务层与下游 estimate_effect 共用）。"""

    treatment_name: Any
    outcome_name: Any
    estimand_type: str
    estimand_expression: str
    backdoor_variables: List[str]
    instrumental_variables: List[str]
    raw_estimand: Any  # dowhy 原始 estimand 对象（avoid serialisation；estimate 时再传）

    def summary(self) -> dict:
        return {
            "treatment": self.treatment_name,
            "outcome": self.outcome_name,
            "estimand_type": self.estimand_type,
            "expression": self.estimand_expression,
            "backdoor": self.backdoor_variables,
            "instruments": self.instrumental_variables,
        }


# 支持的估计器（spec §9 / T7 扩展）
SUPPORTED_ESTIMATORS: tuple = (
    "backdoor.propensity_score_matching",
    "backdoor.linear_regression",
    "backdoor.propensity_score_weighting",
    "iv.instrumental_variable",
)


class CausalPipeline:
    """单病种因果四步流水线入口（spec.md §11 融合后主路径）。

    Usage
    -----
    >>> pipeline = CausalPipeline(disease_code="I21", outcome="cmi", data=df)
    >>> estimand = pipeline.call_identify_effect()
    >>> estimate = pipeline.call_estimate_effect("backdoor.linear_regression")
    """

    def __init__(
        self,
        disease_code: str,
        outcome: str,
        data: pd.DataFrame,
        *,
        treatment: str = "treatment_strategy",
        graph: Optional[nx.DiGraph] = None,
        min_sample_size: int = 30,
        common_causes: Optional[List[str]] = None,
        instruments: Optional[List[str]] = None,
    ) -> None:
        from app.core.dag_builder import (
            SUPPORTED_DISEASE_CODES,
            DiseaseOutcome,
            build_dag,
        )

        if disease_code not in SUPPORTED_DISEASE_CODES:
            raise ValueError(
                f"未知 disease_code: {disease_code!r}；已知 {list(SUPPORTED_DISEASE_CODES)}"
            )
        try:
            DiseaseOutcome(outcome)
        except ValueError as exc:
            raise ValueError(
                f"未知 outcome: {outcome!r}；已知 {[o.value for o in DiseaseOutcome]}"
            ) from exc
        if data is None or data.empty:
            raise ValueError("data 不能为空")
        if len(data) < min_sample_size:
            raise InsufficientSampleError(
                sample_size=len(data), min_required=min_sample_size
            )
        if treatment not in data.columns:
            raise ValueError(
                f"treatment 列 {treatment!r} 不在 data 列中；实际: {list(data.columns)}"
            )
        if outcome not in data.columns:
            raise ValueError(
                f"outcome 列 {outcome!r} 不在 data 列中；实际: {list(data.columns)}"
            )

        self.disease_code = disease_code
        self.outcome = outcome
        self.treatment = treatment
        self.data = data.copy()
        self.min_sample_size = min_sample_size
        self.instruments: List[str] = list(instruments or [])

        # 默认 common causes：age_band + admission_route + complication_count（业务 DAG）
        self.common_causes: List[str] = list(common_causes or [
            "age_band",
            "admission_route",
            "complication_count",
        ])
        missing_cc = [c for c in self.common_causes if c not in self.data.columns]
        if missing_cc:
            LOGGER.warning(
                "common_causes 部分列缺失: %s（自动剔除；估计时按数据实际列）",
                missing_cc,
            )
            self.common_causes = [c for c in self.common_causes if c in self.data.columns]

        # DAG：业务模板（spec §11 §3.1）
        self.graph: nx.DiGraph = graph if graph is not None else build_dag(disease_code, outcome)
        self._causal_model: Any = None  # dowhy.CausalModel 延迟实例化

    # ------------------------------------------------------------------
    # dowhy CausalModel 实例化（按需）
    # ------------------------------------------------------------------

    def _build_causal_model(self) -> Any:
        """构造 dowhy.CausalModel。

        故意延迟到首次调用 identify / estimate 时才实例化，避免模块加载时副作用。

        路径选择：
          1. 优先 graph=DOT 字符串（dowhy 会调用 pydot / pygraphviz 解析）
          2. 若 graph 解析失败（环境缺 pygraphviz / pydot 与 networkx 兼容性 bug），
             **降级到 common_causes 路径**（spec §6 失败降级原则）

        数据列过滤：dowhy 的 linear_regression / propensity_score_matching 等数值估计器
        只接受数值列。我们从 self.data 中筛出 treatment + outcome + common_causes + instruments
        + numeric-only 派生列（spec §3.1），object 列（encounter_id / ICD 编码等标识列）
        留在原 data 但不出现在喂给 CausalModel 的列里。
        """
        if self._causal_model is not None:
            return self._causal_model
        try:
            from dowhy import CausalModel  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "无法 import dowhy.CausalModel；请确认已安装 'dowhy>=0.10,<0.11'"
            ) from exc

        # 数值列裁剪（dowhy 数值估计器需要 numeric dtypes）
        model_data = self._select_numeric_columns_for_dowhy()

        # 尝试 graph=DOT 路径；若解析异常立即降级到 common_causes
        graph_dot: Optional[str] = None
        try:
            graph_dot = self._to_dot(self.graph)
            cm = CausalModel(
                data=model_data,
                treatment=self.treatment,
                outcome=self.outcome,
                graph=graph_dot,
                instruments=self.instruments,
                effect_modifiers=None,
            )
            self._causal_model = cm
            return cm
        except Exception:  # noqa: BLE001 — pydot/pygraphviz 解析失败 / networkx 兼容性 bug
            LOGGER.warning(
                "graph=DOT 路径失败（pygraphviz/pydot 缺失或不兼容），"
                "降级到 common_causes 路径"
            )

        # 降级路径：common_causes（业务 DAG 中已知的协变量列表）
        self._causal_model = CausalModel(
            data=model_data,
            treatment=self.treatment,
            outcome=self.outcome,
            common_causes=self.common_causes,
            instruments=self.instruments or None,
        )
        return self._causal_model

    def _select_numeric_columns_for_dowhy(self) -> pd.DataFrame:
        """裁剪 self.data 为 dowhy 数值估计器可用的列子集。

        保留规则：
          - treatment 列（spec §9 默认 treatment_strategy）
          - outcome 列
          - common_causes（已校验存在）
          - instruments（已校验存在）
          - 其它数值列（int/float/bool）— encounter_id / ICD 编码等 object 列剔除

        类型转换：treatment + common_causes + instruments 中的 object 列做 category
        编码（astype("category").cat.codes），让 dowhy 的 OLS/PSM 能跑数值回归。
        返回新的 DataFrame，不修改 self.data。
        """
        keep: List[str] = []
        for col in self.data.columns:
            if col in (self.treatment, self.outcome):
                keep.append(col)
                continue
            if col in self.common_causes or col in self.instruments:
                keep.append(col)
                continue
            if pd.api.types.is_numeric_dtype(self.data[col]):
                keep.append(col)
        out = self.data[keep].copy()
        # treatment/common_causes/instruments 中的 object 列 → category 编码
        # （object 列不进 dowhy OLS；category codes 是 int）
        for col in out.columns:
            if col == self.outcome:
                continue  # outcome 保持原数值类型（默认 float）
            if out[col].dtype == object:
                out[col] = out[col].astype("category").cat.codes
        return out

    @staticmethod
    def _to_dot(g: nx.DiGraph) -> str:
        """nx.DiGraph → DOT 字符串（dowhy 可识别的最小子集）。"""
        lines = ["digraph {"]
        for n in g.nodes():
            attrs = g.nodes[n]
            label = attrs.get("label", str(n))
            lines.append(f'  "{n}" [label="{label}"];')
        for u, v in g.edges():
            lines.append(f'  "{u}" -> "{v}";')
        lines.append("}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 步骤 1：identify_effect
    # ------------------------------------------------------------------

    def call_identify_effect(self) -> Estimand:
        """跑 dowhy CausalModel.identify_effect() 并返回包装后的 Estimand。

        Raises
        ------
        EstimandNotIdentifiedError
            identify_effect 返回 None / 空 estimand（业务层走 spec §6 降级）
        """
        model = self._build_causal_model()
        try:
            raw_estimand = model.identify_effect(proceed_when_unidentifiable=False)
        except Exception as exc:  # noqa: BLE001 — dowhy 内部抛各类异常
            LOGGER.warning(
                "identify_effect 抛异常（disease=%s outcome=%s）: %s",
                self.disease_code,
                self.outcome,
                exc,
            )
            raise EstimandNotIdentifiedError(
                f"identify_effect failed: {exc}"
            ) from exc

        if raw_estimand is None:
            raise EstimandNotIdentifiedError("identify_effect 返回 None")

        # 业务包装（属性访问兼容 dowhy 0.10）
        backdoor_vars: List[str] = []
        instrument_vars: List[str] = []
        estimand_type = "nonparametric-ate"
        estimand_expr = ""
        try:
            backdoor_vars = list(getattr(raw_estimand, "backdoor_variables", []) or [])
            instrument_vars = list(getattr(raw_estimand, "instrumental_variables", []) or [])
            estimand_type = str(
                getattr(getattr(raw_estimand, "estimand_type", None), "name", None)
                or getattr(raw_estimand, "estimand_type", "nonparametric-ate")
            )
            estimand_expr = str(getattr(raw_estimand, "estimand_expression", "") or "")
        except Exception:  # pragma: no cover
            LOGGER.exception("estimand 字段访问失败；以空值兜底")

        if not backdoor_vars and not instrument_vars and not estimand_expr:
            raise EstimandNotIdentifiedError(
                "estimand 既无 backdoor 也无 instrument，且 expression 为空；"
                "当前 DAG 可能未指定共同原因或工具变量。"
            )

        return Estimand(
            treatment_name=self.treatment,
            outcome_name=self.outcome,
            estimand_type=estimand_type,
            estimand_expression=estimand_expr,
            backdoor_variables=backdoor_vars,
            instrumental_variables=instrument_vars,
            raw_estimand=raw_estimand,
        )

    # ------------------------------------------------------------------
    # 步骤 2：estimate_effect（T7 扩展入口；本任务仅占位 + 单元测）
    # ------------------------------------------------------------------

    def call_estimate_effect(
        self,
        method_name: str = "backdoor.linear_regression",
        *,
        estimand: Optional[Estimand] = None,
        control_value: Any = 0,
        treatment_value: Any = 1,
    ) -> "EffectEstimate":  # noqa: F821 — forward ref to node_schema.EffectEstimate
        """estimate_effect（spec §11 第 2 步）。

        完整 EffectEstimate 字段 + CI 超宽判断在 T7 落地；
        本任务实现最小返回（ate + method_name + sample_size）让 T6 测试可跑。
        """
        from app.core.node_schema import EffectEstimate

        if method_name not in SUPPORTED_ESTIMATORS:
            raise ValueError(
                f"未知估计器: {method_name!r}；已知: {list(SUPPORTED_ESTIMATORS)}"
            )

        model = self._build_causal_model()
        if estimand is None:
            estimand = self.call_identify_effect()

        ci_lower: Optional[float] = None
        ci_upper: Optional[float] = None
        ate: Optional[float] = None
        p_value: Optional[float] = None

        try:
            # PSM 仅支持二元 treatment；当 treatment_strategy 是多值时优雅降级到 LR
            effective_method = method_name
            if method_name == "backdoor.propensity_score_matching":
                n_unique_treatment = int(self.data[self.treatment].nunique())
                if n_unique_treatment > 2:
                    LOGGER.warning(
                        "PSM 仅支持二元 treatment；当前 %s 有 %d 个取值 → 自动降级到 "
                        "backdoor.linear_regression",
                        self.treatment,
                        n_unique_treatment,
                    )
                    effective_method = "backdoor.linear_regression"

            estimate = model.estimate_effect(
                estimand.raw_estimand,
                method_name=effective_method,
                control_value=control_value,
                treatment_value=treatment_value,
            )
            ate = float(getattr(estimate, "value", float("nan")))
            if ate != ate:  # NaN check
                ate = None

            # 置信区间：优先调 dowhy.get_confidence_intervals；不可用时从 SE + 1.96 兜底
            try:
                ci = estimate.get_confidence_intervals(alpha=0.05)
                if ci is not None:
                    arr = list(ci) if not isinstance(ci, (list, tuple)) else ci
                    if len(arr) >= 2:
                        ci_lower = float(arr[0])
                        ci_upper = float(arr[1])
            except Exception:  # noqa: BLE001 — 部分估计器无 CI
                LOGGER.debug("置信区间直接 API 不可用；尝试从 SE 兜底", exc_info=True)
                try:
                    se_obj = estimate.get_standard_error()
                    if se_obj is not None:
                        # se 可能是 CausalEstimate 对象（含 .value）或标量
                        se_val = float(getattr(se_obj, "value", se_obj))
                        if se_val == se_val and se_val >= 0:  # 非 NaN + 非负
                            ci_lower = ate - 1.96 * se_val
                            ci_upper = ate + 1.96 * se_val
                except Exception:  # pragma: no cover — SE 也不可用
                    LOGGER.debug("SE 兜底失败；保留 None", exc_info=True)

            # p-value：dowhy 0.10 test_stat_significance() 返回 dict{'p_value': array([...])}
            try:
                ts_p = estimate.test_stat_significance()
                if isinstance(ts_p, dict) and "p_value" in ts_p:
                    pv = ts_p["p_value"]
                    if hasattr(pv, "__len__") and not isinstance(pv, str):
                        pv = pv[0] if len(pv) > 0 else None
                    if pv is not None:
                        p_value = float(pv)
                        if p_value != p_value:  # NaN
                            p_value = None
                elif ts_p is not None:
                    # 部分估计器返回 (test_stat, p_value) tuple
                    arr = list(ts_p) if not isinstance(ts_p, (list, tuple)) else ts_p
                    if len(arr) >= 2 and arr[1] is not None:
                        p_value = float(arr[1])
                        if p_value != p_value:
                            p_value = None
            except Exception:  # noqa: BLE001
                LOGGER.debug("test_stat_significance 不可用；p_value 留 None", exc_info=True)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "estimate_effect 抛异常（method=%s）: %s", method_name, exc
            )
            ate = None

        ci_width: Optional[float] = None
        if ci_lower is not None and ci_upper is not None:
            ci_width = ci_upper - ci_lower

        # 不静默：ate 为 None 时（估计器抛异常 / model 数据不兼容）必须显式标
        # estimate_ci_unstable=True；这是 spec §6 失败不静默改数据的契约。
        # CI 宽度 > 0.6 仅用于「成功估计但 CI 太宽」场景。
        estimate_ci_unstable = bool(
            ate is None or (ci_width is not None and ci_width > 0.6)
        )

        return EffectEstimate(
            node=self.treatment,
            treatment=self.treatment,
            outcome=self.outcome,
            method_name=method_name,  # 用户请求的估计器（即使内部降级也保持原值）
            ate=ate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_width=ci_width,
            p_value=p_value,
            estimate_ci_unstable=estimate_ci_unstable,
            sample_size=len(self.data),
            control_value=control_value,
            treatment_value=treatment_value,
        )
