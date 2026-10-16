"""对照基线（简单相关分析）+ 因果楔子持久化（spec.md §11 主路径 + §6 失败不静默）。

本模块把单因素 Pearson 相关分析作为「弱基线」(weak baseline)，与因果估计的 ATE
（Average Treatment Effect）做对比，给出 10x 楔子强度报告 — 这是 spec §11「核心模块
深度」中的 (b) 维度「对照基线」。

业务含义：
- 因果效应（backdoor.linear_regression 控制混杂后）vs 单因素 Pearson 相关，
  二者差异越大 → 越说明混杂控制产生了额外价值（**因果楔子**）。
- |ATE| ≥ THRESHOLD 倍 |Pearson r| → 标 ``causal_wedge_strong=True`` 写入
  baseline_report.json，供面板在「**为什么用因果而不是只用相关**」一栏展示。

输入 / 输出契约：
  compute_correlation(data, treatment, outcome) -> BaselineResult
      返回 Pearson r、样本量、是否降级（数据形状 / 取值异常）。
  wedge_strength(ate, correlation) -> dict
      计算楔子比值、绝对差、是否 strong、direction = same / opposite / null。
  write_baseline_report(report, output_path) -> Path
      把报告 dict 落盘成 JSON；既有字段缺失不抛。

设计原则（spec §6）：
- 不静默改数据：causal ATE 不可用 / 相关计算失败 → 字段标 ``degraded=True`` + 原因列表。
- 楔子 ≠ 因果正确：strong wedge 仅表示因果估计量比单因素相关有额外价值，不代表因果
  方向正确；refuter + 反事实 + DAG 拓扑共同给出最终可信度评估（spec §11）。
- ⚠️ README 不得写源仓库名；本模块仅描述方法族（Pearson 相关 + 楔子比值）。

源产品能力参考：spec §10「对照基线」—— 楔子持久化是 spec §11 第 5 步「端到端对比」的
工程落地；与参考仓库无直接代码溯源（业务工程支撑，便于按医院调整）。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------
# 常量
# ----------------------------------------------------------------------------------

# 因果楔子触发阈值：|ATE| >= THRESHOLD * |Pearson r| 时视为因果估计量比单因素相关
# 强 1.5 倍以上（spec §11 第 5 步）。
CAUSAL_WEDGE_THRESHOLD: float = 1.5

# 相关性降级阈值：|Pearson r| 小于该值视为「无相关基线」，楔子比值不适用
# （避免 /0 与 noise-dominant 场景下虚假触发）。
MIN_CORRELATION_FLOOR: float = 1e-3

# 最小样本量（与 refuter 一致：spec §6 样本量阈值）
MIN_BASELINE_SAMPLES: int = 30

# 报告落盘 schema 版本（向前兼容）
BASELINE_REPORT_SCHEMA_VERSION: str = "baseline_report/1.0"


# ----------------------------------------------------------------------------------
# Dataclass
# ----------------------------------------------------------------------------------


@dataclass
class BaselineResult:
    """单因素 Pearson 相关基线结果。

    字段
    ----
    correlation : Optional[float]
        Pearson 相关系数；不可计算时为 None。
    sample_size : int
        实际用于计算相关的样本量（剔除缺失值后）。
    treatment_encoded : str
        实际计算时所用的 treatment 列名（数值化后缀 _num）。
    outcome_column : str
        实际计算时所用的 outcome 列名。
    degraded : bool
        True = 计算降级（样本不足 / 取值唯一 / 含 NaN）。
    degraded_reasons : List[str]
        降级原因列表（空列表 = 正常计算）。
    """

    correlation: Optional[float]
    sample_size: int
    treatment_encoded: str
    outcome_column: str
    degraded: bool = False
    degraded_reasons: List[str] = field(default_factory=list)


# ----------------------------------------------------------------------------------
# 1. compute_correlation：单因素 Pearson 相关
# ----------------------------------------------------------------------------------


def _encode_for_corr(
    data: pd.DataFrame, column: str, suffix: str = "_num"
) -> tuple[pd.Series, str]:
    """把列数值化为 pd.Series，便于 pd.DataFrame.corr() 计算 Pearson。

    规则：
      - 已经是数值 dtype（int / float / bool）→ 原样返回；
      - object / category dtype → pd.factorize → 数值 codes；
      - 返回新 Series，避免修改原 data。

    Returns
    -------
    encoded_series, encoded_column_name
    """
    if column not in data.columns:
        raise ValueError(
            f"{column!r} 不在 data 列中；实际: {list(data.columns)}"
        )
    if pd.api.types.is_numeric_dtype(data[column]):
        return data[column], column
    encoded = pd.factorize(data[column], sort=True)[0]
    # factorize 产出的是 int64 ndarray，包装成 Series 跟原索引对齐便于后续计算
    out = pd.Series(encoded, index=data.index, name=column + suffix)
    return out, column + suffix


def compute_correlation(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
) -> BaselineResult:
    """计算单因素 Pearson 相关作为弱基线。

    Parameters
    ----------
    data : pd.DataFrame
        含 treatment / outcome 列的样本数据。
    treatment : str
        treatment 列名（业务 DAG 中的 treatment_strategy）。
    outcome : str
        outcome 列名（cmi / los_days / readmit_flag / complication_flag）。

    Returns
    -------
    BaselineResult
        详见 dataclass 注释。
    """
    if data is None or data.empty:
        return BaselineResult(
            correlation=None,
            sample_size=0,
            treatment_encoded=treatment,
            outcome_column=outcome,
            degraded=True,
            degraded_reasons=["empty_data"],
        )

    # 数值化编码（业务 fixture 中 treatment 是中文 / object）
    try:
        treatment_series, treatment_col = _encode_for_corr(data, treatment)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("treatment 编码失败: %s", exc)
        return BaselineResult(
            correlation=None,
            sample_size=len(data),
            treatment_encoded=treatment,
            outcome_column=outcome,
            degraded=True,
            degraded_reasons=[f"treatment_encode_failed:{exc}"],
        )

    # outcome 是 cmi / los_days / readmit_flag / complication_flag，本身是数值；
    # 留接口给布尔 outcome（直接由 pd.api.types.is_numeric_dtype 接受 bool）
    if not pd.api.types.is_numeric_dtype(data[outcome]):
        try:
            outcome_series, outcome_col = _encode_for_corr(data, outcome)
        except Exception as exc:  # noqa: BLE001
            return BaselineResult(
                correlation=None,
                sample_size=len(data),
                treatment_encoded=treatment_col,
                outcome_column=outcome,
                degraded=True,
                degraded_reasons=[f"outcome_encode_failed:{exc}"],
            )
    else:
        outcome_series = data[outcome]
        outcome_col = outcome

    # 构造临时 DataFrame 并 dropna
    pair = pd.concat([treatment_series, outcome_series], axis=1).dropna()
    n = len(pair)
    degraded_reasons: List[str] = []

    if n < MIN_BASELINE_SAMPLES:
        degraded_reasons.append(
            f"sample_size_too_small:n={n}<{MIN_BASELINE_SAMPLES}"
        )

    # 取值唯一检查：treatment 或 outcome 全相等 → 相关无法定义
    if pair.iloc[:, 0].nunique() < 2:
        degraded_reasons.append("treatment_constant")
    if pair.iloc[:, 1].nunique() < 2:
        degraded_reasons.append("outcome_constant")

    if not degraded_reasons:
        corr_matrix = pair.corr(method="pearson")
        # pair.corr() 是 2x2 对称矩阵，[0,1] 元素即 treatment-outcome 相关
        corr_value = float(corr_matrix.iloc[0, 1])
        if corr_value != corr_value:  # NaN
            degraded_reasons.append("correlation_is_nan")
            corr_value = None
    else:
        corr_value = None

    return BaselineResult(
        correlation=corr_value,
        sample_size=n,
        treatment_encoded=treatment_col,
        outcome_column=outcome_col,
        degraded=bool(degraded_reasons),
        degraded_reasons=degraded_reasons,
    )


# ----------------------------------------------------------------------------------
# 2. wedge_strength：因果楔子计算
# ----------------------------------------------------------------------------------


def wedge_strength(
    ate: Optional[float],
    correlation: Optional[float],
    threshold: float = CAUSAL_WEDGE_THRESHOLD,
) -> Dict[str, Any]:
    """计算因果楔子强度（spec §11 第 5 步 + §6 不静默降级）。

    Parameters
    ----------
    ate : Optional[float]
        因果估计 ATE（来自 EffectEstimate.ate；不可用时为 None）。
    correlation : Optional[float]
        单因素 Pearson 相关系数（BaselineResult.correlation）。
    threshold : float
        楔子触发阈值（默认 1.5）。

    Returns
    -------
    dict
        {
          "ratio": Optional[float],          # |ATE| / |correlation|；相关性接近 0 时 None
          "ratio_display": str,              # "N/A" / "∞" / 数字字符串
          "ate": Optional[float],
          "correlation": Optional[float],
          "threshold": float,
          "causal_wedge_strong": bool,       # True = 达到阈值
          "direction": str,                  # "same" / "opposite" / "ate_only" / "corr_only" / "none"
          "degraded": bool,                  # True = ATE / 相关不可用导致无法定楔子
          "degraded_reasons": List[str],
        }
    """
    reasons: List[str] = []
    if ate is None:
        reasons.append("ate_unavailable")
    if correlation is None:
        reasons.append("correlation_unavailable")

    if not reasons and (abs(ate) < MIN_CORRELATION_FLOOR and abs(correlation) < MIN_CORRELATION_FLOOR):
        reasons.append("both_zero_magnitude")

    degraded = bool(reasons)
    if degraded:
        return {
            "ratio": None,
            "ratio_display": "N/A",
            "ate": ate,
            "correlation": correlation,
            "threshold": threshold,
            "causal_wedge_strong": False,
            "direction": "none" if not reasons else _direction(ate, correlation),
            "degraded": True,
            "degraded_reasons": reasons,
        }

    abs_corr = abs(correlation)
    if abs_corr < MIN_CORRELATION_FLOOR:
        # 相关约等于 0 → 任一有意义的 ATE 都视为楔子强（即「相关看不见，因果看见」）
        strong = ate is not None and abs(ate) > 0
        ratio: Optional[float] = None
        ratio_display = "∞"
        direction = _direction(ate, correlation)
        # 同时如果 corr == 0 但 ate 不为 0，是经典「混杂抑制了相关，因果解开」场景，
        # 逻辑上 ratio 应该是 inf，方向由 ATE 单边定义。
        direction = direction if direction != "none" else "ate_only"
        reasons.append("correlation_floor")
        return {
            "ratio": ratio,
            "ratio_display": ratio_display,
            "ate": ate,
            "correlation": correlation,
            "threshold": threshold,
            "causal_wedge_strong": strong,
            "direction": direction,
            "degraded": False,
            "degraded_reasons": reasons,
        }

    ratio = abs(ate) / abs_corr
    ratio_display = f"{ratio:.4f}"
    direction = _direction(ate, correlation)
    strong = bool(ratio >= threshold)

    return {
        "ratio": ratio,
        "ratio_display": ratio_display,
        "ate": ate,
        "correlation": correlation,
        "threshold": threshold,
        "causal_wedge_strong": strong,
        "direction": direction,
        "degraded": False,
        "degraded_reasons": reasons,
    }


def _direction(ate: Optional[float], correlation: Optional[float]) -> str:
    """推断 ATE 与 correlation 方向是否一致。

    Returns
    -------
    "same" / "opposite" / "ate_only" / "corr_only" / "none"
    """
    ate_pos = ate is not None and abs(ate) > MIN_CORRELATION_FLOOR
    corr_pos = correlation is not None and abs(correlation) > MIN_CORRELATION_FLOOR
    if not ate_pos and not corr_pos:
        return "none"
    if ate_pos and not corr_pos:
        return "ate_only"
    if corr_pos and not ate_pos:
        return "corr_only"
    # both
    return "same" if (ate > 0) == (correlation > 0) else "opposite"


# ----------------------------------------------------------------------------------
# 3. write_baseline_report：楔子持久化
# ----------------------------------------------------------------------------------


def write_baseline_report(
    *,
    disease_code: str,
    outcome: str,
    treatment: str,
    baseline: BaselineResult,
    wedge: Dict[str, Any],
    effect_estimate: Optional[Dict[str, Any]] = None,
    dag_summary: Optional[Dict[str, Any]] = None,
    output_path: Path,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """把基线对比报告写入 JSON 文件（spec §11 第 5 步楔子持久化）。

    Parameters
    ----------
    disease_code / outcome / treatment
        标识符（业务含义）。
    baseline : BaselineResult
        compute_correlation() 的返回。
    wedge : dict
        wedge_strength() 的返回。
    effect_estimate : Optional[dict]
        EffectEstimate.summary() 输出；不可用可为 None。
    dag_summary : Optional[dict]
        dag_builder.dag_summary() 输出；不可用可为 None。
    output_path : Path
        落盘目标路径；父目录自动创建。
    extra : Optional[dict]
        附加元数据（如 fixture_seed / sample_size 等）。

    Returns
    -------
    Path
        实际写入的路径（便于调用方读取验证）。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "schema_version": BASELINE_REPORT_SCHEMA_VERSION,
        "disease_code": disease_code,
        "outcome": outcome,
        "treatment": treatment,
        "baseline": {
            "correlation": baseline.correlation,
            "sample_size": baseline.sample_size,
            "treatment_encoded": baseline.treatment_encoded,
            "outcome_column": baseline.outcome_column,
            "degraded": baseline.degraded,
            "degraded_reasons": baseline.degraded_reasons,
        },
        "wedge": wedge,
        "effect_estimate": effect_estimate,
        "dag_summary": dag_summary,
        "extra": extra or {},
    }

    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOGGER.info(
        "baseline_report 落盘: %s（wedge_strong=%s）",
        output_path,
        wedge.get("causal_wedge_strong"),
    )
    return output_path


# ----------------------------------------------------------------------------------
# 4. CorrelationBaseline：高层便捷类（spec §11 描述）
# ----------------------------------------------------------------------------------


class CorrelationBaseline:
    """对照基线高层接口（spec §11 第 5 步 + §6 失败不静默）。

    Usage
    -----
    >>> baseline = CorrelationBaseline()
    >>> result = baseline.compute(data, treatment="treatment_strategy", outcome="cmi")
    >>> result.correlation          # Pearson r
    >>> wedge = baseline.wedge(ate=0.3, correlation=result.correlation)
    >>> wedge["causal_wedge_strong"]
    """

    def compute(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
    ) -> BaselineResult:
        return compute_correlation(data, treatment, outcome)

    def wedge(
        self,
        ate: Optional[float],
        correlation: Optional[float],
        threshold: float = CAUSAL_WEDGE_THRESHOLD,
    ) -> Dict[str, Any]:
        return wedge_strength(ate, correlation, threshold=threshold)

    def report(
        self,
        *,
        disease_code: str,
        outcome: str,
        treatment: str,
        baseline: BaselineResult,
        effect_estimate: Optional[Dict[str, Any]] = None,
        output_path: Path,
        dag_summary: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """便捷：compute → wedge → write 三步走。"""
        ate: Optional[float] = None
        if effect_estimate is not None:
            ate = effect_estimate.get("ate")
        wedge = self.wedge(ate=ate, correlation=baseline.correlation)
        return write_baseline_report(
            disease_code=disease_code,
            outcome=outcome,
            treatment=treatment,
            baseline=baseline,
            wedge=wedge,
            effect_estimate=effect_estimate,
            dag_summary=dag_summary,
            output_path=output_path,
            extra=extra,
        )
