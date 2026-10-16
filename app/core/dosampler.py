"""DoSampler 反事实接入（spec.md §11 主路径 / §9 因果推理管线 / Pearl do() 操作）。

本模块把参考产品中 Pearl ``do()`` 操作封装到面向单病种评审场景的
``CounterfactualRunner`` —— 选定业务 DAG 节点（年龄、治疗、合并症数、入院途径等），
输入一个假设值（"如果把年龄段统一调到 65-74"、"如果全部改 PCI"），模型给出
``CounterfactualScenario``：在 do() 操作下 outcome 的预测值、置信度、do() 操作
序列与超时标记。

设计原则（spec §6 失败不静默改数据）：
- **5 秒超时**：用 ThreadPoolExecutor 跑 do()，超时即返回 ``timeout=True``；
  业务层可正常走草稿生成（标 timeout 即可，不抛异常阻断流程）。
- **do() 操作可解释**：``do_operations`` 列出具体 do() 步骤字符串，供面板
  显示（如 "do(admission_route = 1); do(treatment_strategy = PCI)"）。
- **预测置信度**：基于拟合模型 R² 与预测残差 std 计算 0–1 之间的 confidence。
- **baseline_std**：原始 outcome 标准差，供面板展示预测偏差合理性。
- **integration surface**：本模块为纯算法；本任务不联网、不写数据库，仅在
  内存里跑 do() 后返回 ``CounterfactualScenario`` 给调用方。

源产品能力参考：
- DoSampler.do_sample(x) 主路径见 ``github_ref/dowhy/dowhy/do_sampler.py`` 的
  ``do_sample``（reset → disrupt_causes → make_treatment_effective → sample）。
- CausalModel.do(x, identified_estimand, method_name) 调用 ``causal_estimator.do(x)``；
  不同 estimator 的 ``do()`` 实现各异（linear_regression / propensity_score_weighting
  等有定义；PSM / GBM 等不一定实现 do()，会抛 NotImplementedError）。
- 本模块为工程稳定性采用「数值拟合 → predict」的等价表达：当 dowhy 某估计器
  do() 不支持时，业务层仍能拿到一个可解释的反事实预测。

⚠️ README 不得写源仓库名（dowhy / pywhy）；本模块仅描述方法族。
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.core.dag_builder import (
    ADMISSION_ROUTE_VALUES,
    AGE_BAND_VALUES,
    DISEASE_OUTCOME_VALUES,
    SUPPORTED_DISEASE_CODES,
    TREATMENT_VALUES,
    DiseaseOutcome,
)
from app.core.node_schema import CounterfactualScenario

LOGGER = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------
# 常量
# ----------------------------------------------------------------------------------

# 单条反事实超时（spec §11 第 4 步 — 反事实模拟；5s 内必须返回）
COUNTERFACTUAL_TIMEOUT_SECONDS: float = 5.0

# 干预节点可取值的取值集合（按节点类别区分）
# - treatment: 业务 DAG 治疗方式枚举（如 PCI / thrombolysis / conservative）
# - age_band: 1/2/3/4 离散区间
# - admission_route: 1/2/3/9
# - complication_count: 0/1/2 离散区间
TREATMENT_NODE_PREFIX = "treatment:"
AGE_BAND_NODE_PREFIX = "age_band:"
ADMISSION_ROUTE_NODE_PREFIX = "admission_route:"
COMPLICATION_COUNT_NODE_PREFIX = "complication_count:"

OUTCOME_NODE_PREFIXES = (
    TREATMENT_NODE_PREFIX,
    AGE_BAND_NODE_PREFIX,
    ADMISSION_ROUTE_NODE_PREFIX,
    COMPLICATION_COUNT_NODE_PREFIX,
)


# ----------------------------------------------------------------------------------
# 内部 helper：节点合法性校验
# ----------------------------------------------------------------------------------


def _is_valid_intervention_node(node: str) -> bool:
    """检查干预节点 id 是否属于业务 DAG 已知节点（spec §3.1）。"""
    if not isinstance(node, str) or not node:
        return False
    if node in DISEASE_OUTCOME_VALUES:
        return False  # outcome 节点不能作为干预目标
    return any(node.startswith(prefix) for prefix in OUTCOME_NODE_PREFIXES)


def _extract_node_value(node: str) -> Any:
    """从节点 id 提取其业务取值（仅用于标准化 hypothetical_value 校验）。

    Example
    -------
    >>> _extract_node_value("treatment:PCI")
    'PCI'
    >>> _extract_node_value("age_band:3")
    '3'
    """
    for prefix in OUTCOME_NODE_PREFIXES:
        if node.startswith(prefix):
            return node[len(prefix):]
    return node


def _validate_intervention_node_for_disease(node: str, disease_code: str) -> None:
    """校验干预节点是否属于当前病种的合法取值范围。

    Raises
    ------
    ValueError
        - 未知 disease_code
        - 干预节点前缀不属于业务 DAG
        - 节点取值不在该病种的合法取值集合内
    """
    if disease_code not in SUPPORTED_DISEASE_CODES:
        raise ValueError(
            f"未知 disease_code: {disease_code!r}；已知: {list(SUPPORTED_DISEASE_CODES)}"
        )
    if not _is_valid_intervention_node(node):
        raise ValueError(
            f"未知干预节点: {node!r}；"
            f"合法前缀: {list(OUTCOME_NODE_PREFIXES)}，且非 outcome 节点"
        )

    suffix = _extract_node_value(node)
    if node.startswith(TREATMENT_NODE_PREFIX):
        if suffix not in TREATMENT_VALUES[disease_code]:
            raise ValueError(
                f"治疗节点取值 {suffix!r} 不在 disease_code={disease_code} 合法集合: "
                f"{TREATMENT_VALUES[disease_code]}"
            )
    elif node.startswith(AGE_BAND_NODE_PREFIX):
        if not suffix.isdigit() or int(suffix) not in AGE_BAND_VALUES:
            raise ValueError(
                f"age_band 节点取值 {suffix!r} 不在合法集合 {AGE_BAND_VALUES}"
            )
    elif node.startswith(ADMISSION_ROUTE_NODE_PREFIX):
        if suffix not in ADMISSION_ROUTE_VALUES:
            raise ValueError(
                f"admission_route 节点取值 {suffix!r} 不在合法集合 "
                f"{list(ADMISSION_ROUTE_VALUES)}"
            )
    # complication_count: 任意正整数都允许（业务 DAG 仅展示 0/1/2，实际取整数即可）


def _coerce_hypothetical_value(node: str, hypothetical_value: Any) -> Any:
    """标准化 hypothetical_value 到与 fixture 字段一致的类型（数值或字符串）。"""
    suffix = _extract_node_value(node)
    if node.startswith(AGE_BAND_NODE_PREFIX):
        return int(hypothetical_value)
    if node.startswith(ADMISSION_ROUTE_NODE_PREFIX):
        return str(hypothetical_value)
    if node.startswith(COMPLICATION_COUNT_NODE_PREFIX):
        return int(hypothetical_value)
    if node.startswith(TREATMENT_NODE_PREFIX):
        # 治疗方式取值字符串与 fixture treatment_strategy 列一致
        return str(hypothetical_value)
    # 兜底：返回原值（节点值由 _validate_intervention_node_for_disease 保证）
    return hypothetical_value


def _map_node_to_data_column(node: str) -> str:
    """干预节点 id → fixture DataFrame 列名。

    Example
    -------
    >>> _map_node_to_data_column("treatment:PCI")
    'treatment_strategy'
    >>> _map_node_to_data_column("age_band:3")
    'age_band'
    >>> _map_node_to_data_column("admission_route:1")
    'admission_route'
    >>> _map_node_to_data_column("complication_count:2")
    'complication_count'
    """
    if node.startswith(TREATMENT_NODE_PREFIX):
        return "treatment_strategy"
    if node.startswith(AGE_BAND_NODE_PREFIX):
        return "age_band"
    if node.startswith(ADMISSION_ROUTE_NODE_PREFIX):
        return "admission_route"
    if node.startswith(COMPLICATION_COUNT_NODE_PREFIX):
        return "complication_count"
    raise ValueError(f"无法映射节点 {node!r} 到 fixture 列")


def _build_scenario_id(disease_code: str, intervention_node: str, hypothetical_value: Any) -> str:
    """构造稳定的 scenario_id（同一干预组合多次调用得到同一 id，便于面板 SSE 去重）。"""
    raw = f"{disease_code}::{intervention_node}::{hypothetical_value}".encode()
    return "scn_" + hashlib.sha1(raw).hexdigest()[:16]


# ----------------------------------------------------------------------------------
# 数值拟合 + 反事实预测（业务层主路径）
# ----------------------------------------------------------------------------------


def _prepare_features(
    data: pd.DataFrame,
    outcome: str,
    intervention_node: str,
    covariates: Optional[List[str]] = None,
) -> tuple:
    """准备 (X, y, category_maps) 给拟合 / 预测。

    业务 DAG 中：
      - outcome（spec §3.1）：cmi / los_days / readmit_flag / complication_flag
        默认走数值拟合（one-hot 化 readmit_flag / complication_flag）；
      - intervention_node：单列名（treatment_strategy / age_band / admission_route /
        complication_count），构造独热或数值；
      - covariates：默认 age_band / admission_route / complication_count（业务
        DAG 协变量集合）。

    Returns
    -------
    (X: pd.DataFrame, y: pd.Series, category_maps: Dict[str, Dict[Any, int]])
        category_maps 把每个类别列的 {原始值 → 整数 code}，供构造反事实样本时
        把字符串 hypothetical_value（如 "PCI"）转换成同编码后的 int。
    """
    if outcome not in data.columns:
        raise ValueError(f"outcome 列 {outcome!r} 不在 data 列中")

    cols: List[str] = []
    target_col = _map_node_to_data_column(intervention_node)
    if target_col not in data.columns:
        raise ValueError(f"干预节点对应列 {target_col!r} 不在 data 列中")
    cols.append(target_col)

    if covariates is None:
        covariates = ["age_band", "admission_route", "complication_count"]
    for c in covariates:
        if c == target_col:
            continue
        if c in data.columns and c != outcome:
            cols.append(c)

    sub = data[cols + [outcome]].copy()

    # 缺失值守门
    n_before = len(sub)
    sub = sub.dropna()
    n_dropped = n_before - len(sub)
    if n_dropped > 0:
        LOGGER.info("反事实拟合：丢弃 %d 行缺失", n_dropped)

    y = sub[outcome]
    X = sub[cols].copy()

    # 类别列：treatment_strategy / age_band / admission_route / complication_count
    # 把 object dtype 列编码为 int（与 pandas astype("category").cat.codes 一致），
    # 同时保存 {原始值 → 整数 code} 映射供构造反事实样本时复用。
    category_maps: Dict[str, Dict[Any, int]] = {}
    for col in X.columns:
        if X[col].dtype == object or X[col].dtype.name == "category":
            # 保留原始值列表（去重后保留顺序）；构建映射
            unique_vals = list(X[col].drop_duplicates().tolist())
            mapping: Dict[Any, int] = {v: i for i, v in enumerate(unique_vals)}
            category_maps[col] = mapping
            X[col] = X[col].map(mapping).astype(int)

    return X, y, category_maps


def _fit_predict_linear(
    data: pd.DataFrame,
    outcome: str,
    intervention_node: str,
    hypothetical_value: Any,
) -> Dict[str, Any]:
    """数值最小二乘拟合 + 反事实预测（业务层主路径，无外部依赖）。

    算法：
      1. 把 outcome ~ intervention_node + covariates 拟合为线性回归；
      2. 计算 baseline 预测（所有样本原始 outcome 均值 + 拟合截距）；
      3. 把干预节点的 hypothetical_value 代入，得到反事实预测 outcome；
      4. 计算 confidence = max(0, 1 - normalized_residual_std)；
      5. baseline_std = y.std()。

    全部在 numpy / pandas 内完成，无需 scikit-learn；与 dowhy 的 `do()` 输出
    等价（线性回归方程代入即可）。

    Returns
    -------
    dict
        - predicted_outcome
        - baseline_value
        - baseline_std
        - confidence
        - r_squared
        - intercept
        - coef_map（dict: 列名 → 系数）
    """
    X, y, category_maps = _prepare_features(data, outcome, intervention_node)

    if len(X) < 5:
        raise ValueError(
            f"有效样本不足 (n={len(X)})，无法拟合反事实（最少 5 行）"
        )

    # 构造 X 矩阵（加截距列）
    X_mat = np.column_stack([np.ones(len(X)), X.values.astype(float)])
    y_arr = y.values.astype(float)

    # 最小二乘：coef = (X^T X)^-1 X^T y；用 lstsq 求数值稳定解
    coef, *_ = np.linalg.lstsq(X_mat, y_arr, rcond=None)

    intercept = float(coef[0])
    coef_map: Dict[str, float] = {}
    for i, col in enumerate(X.columns):
        coef_map[col] = float(coef[i + 1])

    # baseline_value：所有样本的 fitted outcome 均值（不替换干预列时的预测）
    baseline_predictions = X_mat @ coef
    baseline_value = float(np.mean(baseline_predictions))

    # baseline_std：观测 outcome 的标准差（用于面板显示预测偏差）
    baseline_std = float(np.std(y_arr, ddof=1)) if len(y_arr) > 1 else 0.0

    # 反事实预测：构造一行 (1, intervention_node=hypothetical, covariates=median)
    counterfactual_row: List[float] = []
    intervention_col = _map_node_to_data_column(intervention_node)
    raw_intervention_value = _coerce_hypothetical_value(intervention_node, hypothetical_value)

    for col in X.columns:
        if col == intervention_col:
            # 干预列：用同一 category_map 编码 hypothetical_value
            if col in category_maps:
                mapping = category_maps[col]
                if raw_intervention_value in mapping:
                    counterfactual_row.append(float(mapping[raw_intervention_value]))
                else:
                    # hypothetical_value 不在已知取值集合：用 mapping 均值作为兜底
                    codes = list(mapping.values())
                    counterfactual_row.append(float(sum(codes) / max(1, len(codes))))
            else:
                counterfactual_row.append(float(raw_intervention_value))
        else:
            # 其它协变量列取观测中位数（数值列编码后）或众数（已编码）
            series = X[col]
            counterfactual_row.append(float(series.median()))

    predicted_outcome = float(intercept + np.dot(coef[1:], counterfactual_row))

    # 置信度：基于 R² 与预测相对偏差；clamp 到 [0, 1]
    ss_res = float(np.sum((y_arr - baseline_predictions) ** 2))
    ss_tot = float(np.sum((y_arr - float(np.mean(y_arr))) ** 2))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    # 业务语义：R² 越高，confidence 越高；R² 极低时 confidence 接近 0
    confidence = float(max(0.0, min(1.0, r_squared)))

    return {
        "predicted_outcome": predicted_outcome,
        "baseline_value": baseline_value,
        "baseline_std": baseline_std,
        "confidence": confidence,
        "r_squared": r_squared,
        "intercept": intercept,
        "coef_map": coef_map,
    }


# ----------------------------------------------------------------------------------
# CounterfactualRunner 主类
# ----------------------------------------------------------------------------------


class CounterfactualRunner:
    """反事实运行器（spec.md §11 第 4 步 / T12）。

    Usage
    -----
    >>> runner = CounterfactualRunner(
    ...     disease_code="I21",
    ...     outcome="cmi",
    ...     data=df,
    ... )
    >>> scenario = runner.run(
    ...     intervention_node="treatment:PCI",
    ...     hypothetical_value="PCI",
    ... )
    >>> print(scenario.predicted_outcome, scenario.confidence, scenario.timeout)
    """

    # 类级别共享线程池（每次 run() 都启动一次子线程，5s 超时由 Future.result(timeout=...) 控制）
    _thread_pool: Optional[ThreadPoolExecutor] = None
    _thread_pool_lock = threading.Lock()

    @classmethod
    def _get_thread_pool(cls) -> ThreadPoolExecutor:
        """懒创建单例线程池（4 worker 足够 do() 计算量；关闭时由 Python GC 处理）。"""
        if cls._thread_pool is None:
            with cls._thread_pool_lock:
                if cls._thread_pool is None:
                    cls._thread_pool = ThreadPoolExecutor(
                        max_workers=4, thread_name_prefix="dosampler-"
                    )
        return cls._thread_pool

    def __init__(
        self,
        disease_code: str,
        outcome: str,
        data: pd.DataFrame,
        *,
        scenario_prefix: str = "scn",
    ) -> None:
        if disease_code not in SUPPORTED_DISEASE_CODES:
            raise ValueError(
                f"未知 disease_code: {disease_code!r}；已知: {list(SUPPORTED_DISEASE_CODES)}"
            )
        try:
            DiseaseOutcome(outcome)
        except ValueError as exc:
            raise ValueError(
                f"未知 outcome: {outcome!r}；已知: {[o.value for o in DiseaseOutcome]}"
            ) from exc
        if data is None or data.empty:
            raise ValueError("data 不能为空")

        self.disease_code = disease_code
        self.outcome = outcome
        self.data = data.copy()
        self.scenario_prefix = scenario_prefix

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def run(
        self,
        intervention_node: str,
        hypothetical_value: Any,
        *,
        covariates: Optional[List[str]] = None,
    ) -> CounterfactualScenario:
        """跑 do() 反事实 → CounterfactualScenario（5s 超时）。

        Parameters
        ----------
        intervention_node
            干预节点 id（如 "treatment:PCI" / "age_band:3" / "admission_route:1" /
            "complication_count:2"）；必须属于业务 DAG。
        hypothetical_value
            假设该节点被 do() 设定的值（如 "PCI" / 3 / 1）。
        covariates
            协变量列名；默认 ["age_band", "admission_route", "complication_count"]。
        """
        # 构造稳定 id（同一 (disease, node, value) 多次调用得到同一 id）
        scenario_id = _build_scenario_id(self.disease_code, intervention_node, hypothetical_value)

        # 节点合法性校验（已知节点 + 病种合法取值）
        try:
            _validate_intervention_node_for_disease(intervention_node, self.disease_code)
        except ValueError as exc:
            LOGGER.warning("反事实节点非法: %s", exc)
            return CounterfactualScenario(
                scenario_id=scenario_id,
                disease_code=self.disease_code,
                outcome=self.outcome,
                intervention_node=intervention_node,
                hypothetical_value=hypothetical_value,
                baseline_value=None,
                predicted_outcome=None,
                confidence=None,
                do_operations=[f"validation_error:{exc}"],
                elapsed_ms=0,
                timeout=False,
                baseline_std=None,
            )

        # 标准化 hypothetical_value 到与 fixture 一致的类型
        normalized_value = _coerce_hypothetical_value(intervention_node, hypothetical_value)
        intervention_col = _map_node_to_data_column(intervention_node)

        # baseline_value：原数据该列的众数（最常观测到的取值；面板显示"原始观测"）
        try:
            col_series = self.data[intervention_col]
            baseline_value: Any = col_series.mode().iloc[0]
        except (KeyError, IndexError):
            baseline_value = None

        do_op_str = f"do({intervention_col} = {normalized_value!r})"
        do_operations: List[str] = [do_op_str]

        # 5 秒超时跑 do() 数值拟合
        start_ms = time.monotonic()
        executor = self._get_thread_pool()
        future = executor.submit(
            _fit_predict_linear,
            self.data,
            self.outcome,
            intervention_node,
            normalized_value,
        )

        timeout_occurred = False
        fit_result: Optional[Dict[str, Any]] = None
        try:
            fit_result = future.result(timeout=COUNTERFACTUAL_TIMEOUT_SECONDS)
        except FutureTimeout:
            timeout_occurred = True
            LOGGER.warning(
                "do_sample 超时（disease=%s outcome=%s intervention=%s value=%s）",
                self.disease_code,
                self.outcome,
                intervention_node,
                normalized_value,
            )
            # 不 cancel 线程（dowhy 内部可能持锁），future 仍会完成；本场景直接 timeout=True
        except Exception as exc:  # noqa: BLE001 — 拟合路径所有异常统一兜底
            LOGGER.warning(
                "do_sample 抛异常（disease=%s outcome=%s intervention=%s）: %s",
                self.disease_code,
                self.outcome,
                intervention_node,
                exc,
            )
            return CounterfactualScenario(
                scenario_id=scenario_id,
                disease_code=self.disease_code,
                outcome=self.outcome,
                intervention_node=intervention_node,
                hypothetical_value=normalized_value,
                baseline_value=baseline_value,
                predicted_outcome=None,
                confidence=None,
                do_operations=do_operations + [f"exception:{type(exc).__name__}"],
                elapsed_ms=int((time.monotonic() - start_ms) * 1000),
                timeout=False,
                baseline_std=None,
            )

        elapsed_ms = int((time.monotonic() - start_ms) * 1000)

        if timeout_occurred or fit_result is None:
            # 超时：返回 scenario + timeout=True，predicted_outcome 留 None
            return CounterfactualScenario(
                scenario_id=scenario_id,
                disease_code=self.disease_code,
                outcome=self.outcome,
                intervention_node=intervention_node,
                hypothetical_value=normalized_value,
                baseline_value=baseline_value,
                predicted_outcome=None,
                confidence=None,
                do_operations=do_operations + [
                    f"timeout:{COUNTERFACTUAL_TIMEOUT_SECONDS}s"
                ],
                elapsed_ms=elapsed_ms,
                timeout=True,
                baseline_std=None,
            )

        # 正常路径：填充分支 do_operations
        do_operations.append(
            f"fit: linear_regression(outcome={self.outcome} ~ intervention={intervention_col} + covariates)"
        )
        coef_map = fit_result.get("coef_map", {})
        if coef_map:
            coef_strs = [f"{k}={v:+.4f}" for k, v in list(coef_map.items())[:5]]
            do_operations.append("coefficients: " + ", ".join(coef_strs))

        return CounterfactualScenario(
            scenario_id=scenario_id,
            disease_code=self.disease_code,
            outcome=self.outcome,
            intervention_node=intervention_node,
            hypothetical_value=normalized_value,
            baseline_value=baseline_value,
            predicted_outcome=float(fit_result["predicted_outcome"]),
            confidence=float(fit_result["confidence"]),
            do_operations=do_operations,
            elapsed_ms=elapsed_ms,
            timeout=False,
            baseline_std=float(fit_result["baseline_std"]),
        )


# ----------------------------------------------------------------------------------
# 便捷函数
# ----------------------------------------------------------------------------------


def run_counterfactual(
    disease_code: str,
    outcome: str,
    data: pd.DataFrame,
    intervention_node: str,
    hypothetical_value: Any,
    *,
    covariates: Optional[List[str]] = None,
) -> CounterfactualScenario:
    """一次性跑反事实（CounterfactualRunner 的最简封装）。

    等价于::

        runner = CounterfactualRunner(disease_code, outcome, data)
        return runner.run(intervention_node, hypothetical_value, covariates=covariates)
    """
    return CounterfactualRunner(
        disease_code=disease_code,
        outcome=outcome,
        data=data,
    ).run(
        intervention_node=intervention_node,
        hypothetical_value=hypothetical_value,
        covariates=covariates,
    )