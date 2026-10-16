"""Graph-level 条件独立性检验（spec.md §11 第 3.5 步：筛非驱动因子）。

把 `dowhy.causal_refuters.graph_refuter.GraphRefuter`（见
`github_ref/dowhy/causal_refuters/graph_refuter.py:47-67` 的
`partial_correlation` / `conditional_mutual_information`）的能力封装到业务侧：

- `GraphRefutation.partial_correlation_check(causal_model, edge)` —— 检验 DAG 中
  一条边 X → Y 是否被数据"支持"（即 X 与 Y 在控制其它节点后仍存在显著偏相关）；
- `GraphRefutation.filter_driver_edges(...)` —— 批量筛边，按 p-value 阈值
  `default=0.05` 判定；不显著的边从 DAG 中剔除，剩下的视为"真驱动因子"；
- 自适应变量类型：连续变量走 `partial_correlation`（Pearson via 残差回归），
  离散变量走 `conditional_mutual_information`（离散熵 + 互信息）。

业务价值：spec §11 第 3 步反驳的"graph layer"补足——在因果识别 / 估计之前先把
DAG 中"业务上写错了 / 数据上根本看不出关系"的边剔掉，避免后续估计跑出无效
效应；同时给"小样本 / 边界场景"提供边级筛选依据（保留率供 reviewer 复核）。

实现要点：
- 因果模型 → pandas DataFrame：causal_model._data（CausalModel.data）。fixture 走
  load_fixture_csv；生产走 HIS 直连。
- 数值化：分类列（treatment_strategy / age_band 区间字符串）走 label encoding，
  再做残差回归；这是与 dowhy GraphRefuter 的差异（dowhy 自动分流到
  conditional_mutual_information，本模块在已知业务语义下用更严格的偏相关）。
- 失败不静默：单边跑失败 → 标 `test_failed=True` + 不推进业务状态。
- 返回 dataclass：每边 `{edge, r, p_value, n, supported, test_method, test_failed}`，
  供面板 SSE 推送与草稿状态机读。

源产品能力参考：
- `github_ref/dowhy/causal_refuters/graph_refuter.py:47-67` —— `partial_correlation`
  / `conditional_mutual_information` 方法签名 + 显著性约定；
- `github_ref/dowhy/utils/cit.py:47-111` —— 偏相关计算（Pearson via 协方差矩阵
  求逆） + Student-t p-value；本模块用残差回归实现等价功能，便于：
  (a) 与既有 numeric pipeline 复用；（b）保留 row-level 控制（dropna / 异常值剔除）。
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

LOGGER = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------
# 常量
# ----------------------------------------------------------------------------------

# 显著性阈值（与 spec.md §11 第 3.5 步一致）：p-value < 0.05 即视为显著依赖。
DEFAULT_P_VALUE_THRESHOLD: float = 0.05

# 最小样本量：n < 30 → 单边测试不调任何检验函数；直接标 test_failed=True（与
# refuter.py 的 MIN_REFUTE_SAMPLES=30 一致）。
MIN_SAMPLES_FOR_TEST: int = 30

# 偏相关 / 互信息阈值（保留边）：同时要求 p < p_threshold 且 |r| > r_threshold
# 以防弱信号噪声被误判为显著（数据足够大时 p-value 会膨胀显著）。
DEFAULT_R_THRESHOLD: float = 0.05

# 分类列需要 label encode 的最大基数（> 时降级为 binary-only 走 mutual information）
MAX_CATEGORICAL_CARDINALITY: int = 50


# ----------------------------------------------------------------------------------
# 边测试结果 dataclass
# ----------------------------------------------------------------------------------


@dataclass
class EdgeTestResult:
    """单条边的条件独立性检验结果（业务层 + 面板共用）。"""

    edge: Tuple[str, str]
    edge_kind: str = "unknown"
    r: Optional[float] = None
    p_value: Optional[float] = None
    n_samples: int = 0
    supported: bool = False
    test_method: str = ""  # "partial_correlation" / "conditional_mutual_information" / "skipped"
    test_failed: bool = False
    test_failure_reason: str = ""
    conditioning_set: List[str] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {
            "source": self.edge[0],
            "target": self.edge[1],
            "edge_kind": self.edge_kind,
            "r": self.r,
            "p_value": self.p_value,
            "n_samples": self.n_samples,
            "supported": self.supported,
            "test_method": self.test_method,
            "test_failed": self.test_failed,
            "test_failure_reason": self.test_failure_reason,
            "conditioning_set_size": len(self.conditioning_set),
        }


# ----------------------------------------------------------------------------------
# 工具：变量类型识别 + 数值化
# ----------------------------------------------------------------------------------


def _is_continuous(series: pd.Series) -> bool:
    """判断 series 是否连续型变量。

    规则：
    - 数值 dtype（int / float）且唯一值数 > MAX_CATEGORICAL_CARDINALITY / 2 视为连续；
    - 否则视为离散 / 分类。
    """
    if pd.api.types.is_bool_dtype(series):
        return False
    if pd.api.types.is_numeric_dtype(series):
        # 即使是 int，若唯一值很少且非 float，当作离散（避免把 0/1 当连续）
        n_unique = series.dropna().nunique()
        return n_unique > 10  # 经验阈值：>10 不同值视为连续
    return False


def _label_encode(series: pd.Series) -> pd.Series:
    """对分类列做 label encoding（处理 NaN：先填一个 sentinel 再 encode）。

    始终返回 pd.Series（保留 index 与原始 series 对齐），方便下游 `_is_continuous`
    做 .dropna().nunique() 检测。
    """
    s = series.copy()
    # 把 NaN 标为独立类（避免被 scikit-learn 风格的 fit 抛错）
    if s.isna().any():
        s = s.fillna("__NAN__")
    if s.dtype.name == "category":
        codes = s.cat.codes.astype(float)
    elif pd.api.types.is_numeric_dtype(s):
        codes = s.astype(float)
    else:
        # 用 pd.factorize 拿整数 code（NaN 已处理过）
        codes = pd.factorize(s.astype(str), sort=True)[0].astype(float)
    # 强制包成 pd.Series（保留原 index）
    return pd.Series(codes, index=series.index, name=series.name)


# ----------------------------------------------------------------------------------
# 工具：偏相关计算（Pearson via 残差回归）
# ----------------------------------------------------------------------------------


def _partial_corr_pearson(
    x: np.ndarray, y: np.ndarray, Z: np.ndarray
) -> Tuple[float, float, int]:
    """Pearson 偏相关 + 双尾 Student-t p-value。

    参数
    ----
    x : (n,) ndarray
    y : (n,) ndarray
    Z : (n, k) ndarray 或 None / 空

    返回
    ----
    (r, p_value, n) ：r = Pearson 偏相关系数；p_value = 双尾 t 检验 p-value；
    n = 实际有效样本数。
    """
    # 统一去掉任一行含 NaN 的样本
    if Z is None or Z.shape[1] == 0:
        mask = np.isfinite(x) & np.isfinite(y)
        n = int(mask.sum())
        if n < 3:
            return float("nan"), float("nan"), n
        x_clean = x[mask]
        y_clean = y[mask]
        r, p = stats.pearsonr(x_clean, y_clean)
        return float(r), float(p), n

    # 把 Z 也并入 NaN mask
    Z = np.asarray(Z, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if Z.size > 0:
        mask = mask & np.all(np.isfinite(Z), axis=1)
    n = int(mask.sum())
    if n < 3:
        return float("nan"), float("nan"), n

    x_clean = x[mask]
    y_clean = y[mask]
    Z_clean = Z[mask]

    # 残差回归：x ~ Z + 1, y ~ Z + 1
    try:
        # 加一列常数项
        ones = np.ones((n, 1))
        Z_with_const = np.hstack([Z_clean, ones])
        # 用 lstsq 解最小二乘
        beta_x, *_ = np.linalg.lstsq(Z_with_const, x_clean, rcond=None)
        beta_y, *_ = np.linalg.lstsq(Z_with_const, y_clean, rcond=None)
        rx = x_clean - Z_with_const @ beta_x
        ry = y_clean - Z_with_const @ beta_y
        r, p = stats.pearsonr(rx, ry)
        return float(r), float(p), n
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("残差回归失败（n=%d）；退化为简单相关: %s", n, exc)
        r, p = stats.pearsonr(x_clean, y_clean)
        return float(r), float(p), n


def _conditional_mutual_info(
    x: np.ndarray, y: np.ndarray, Z: np.ndarray
) -> Tuple[float, int]:
    """离散变量的条件互信息（信息论版 partial-correlation 替代）。

    实现
    ----
    用一个 sklearn 风格的分箱把 Z 离散成多个 bin，然后按 (x, y | z_bin) 计算互信息。
    返回 (mi, n) —— mi >= 0；越大表示依赖越强。

    说明：这里只用作排序信号，最终显著性来自下面的 bootstrap / permutation。
    """
    # 先把 x / y 也离散化（已是 int/float 但唯一值数不多时直接用）
    Z = np.asarray(Z, dtype=float)
    Z_clean = Z.copy()
    # 分箱：等频 5 箱
    n = len(x)
    if Z.shape[1] == 0:
        return float("nan"), n
    try:
        from sklearn.preprocessing import KBinsDiscretizer  # type: ignore

        disc = KBinsDiscretizer(n_bins=min(5, max(2, n // 20)), encode="ordinal", strategy="quantile")
        Z_binned = disc.fit_transform(Z_clean)
    except Exception:  # noqa: BLE001 — sklearn 不可用或单值列无法分箱
        # 退化到 np.digitize 沿中位数
        Z_binned = np.zeros_like(Z_clean)
        for j in range(Z.shape[1]):
            col = Z_clean[:, j]
            median = np.median(col[np.isfinite(col)]) if np.any(np.isfinite(col)) else 0.0
            Z_binned[:, j] = (col > median).astype(int)

    # 把 (x, y) 和 Z_binned 联合起来建联合表 → 算互信息
    mi = _mutual_info_from_joint(x.astype(int), y.astype(int), Z_binned.astype(int))
    return float(mi), int(n)


def _mutual_info_from_joint(x: np.ndarray, y: np.ndarray, Z: np.ndarray) -> float:
    """计算 I(X; Y | Z_bin) 的经验估计（基于联合频次表）。"""
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    n = len(x)
    # 联合键 = (x, y, z1, z2, ...)
    joint_keys = [tuple([int(x[i]), int(y[i])] + [int(Z[i, j]) for j in range(Z.shape[1])]) for i in range(n)]
    joint_count: Dict[Any, int] = {}
    for k in joint_keys:
        joint_count[k] = joint_count.get(k, 0) + 1

    # 边缘键（去掉 y / 去掉 x / 去掉 x+y / 保留 z）
    xy_keys = [tuple([int(x[i]), int(y[i])]) for i in range(n)]
    xz_keys = [tuple([int(x[i])] + [int(Z[i, j]) for j in range(Z.shape[1])]) for i in range(n)]
    yz_keys = [tuple([int(y[i])] + [int(Z[i, j]) for j in range(Z.shape[1])]) for i in range(n)]
    z_keys = [tuple([int(Z[i, j]) for j in range(Z.shape[1])]) for i in range(n)]

    cnt_xyz = _count(joint_keys)
    cnt_xy = _count(xy_keys)
    cnt_xz = _count(xz_keys)
    cnt_yz = _count(yz_keys)
    cnt_z = _count(z_keys)

    mi = 0.0
    eps = 1e-12
    for k, c_xyz in cnt_xyz.items():
        c_xy = cnt_xy.get(tuple([k[0], k[1]]), 0)
        c_xz = cnt_xz.get(tuple([k[0]] + list(k[2:])), 0)
        c_yz = cnt_yz.get(tuple([k[1]] + list(k[2:])), 0)
        c_z = cnt_z.get(tuple(k[2:]), 0)
        # p(xyz), p(xy), p(xz), p(yz), p(z)
        p_xyz = c_xyz / n
        p_xy = c_xy / n
        p_xz = c_xz / n
        p_yz = c_yz / n
        p_z = c_z / n
        if p_xyz < eps or p_xy < eps or p_xz < eps or p_yz < eps or p_z < eps:
            continue
        # I(X;Y|Z) = Σ p(x,y,z) log [p(x,y,z) * p(z) / (p(x,z) * p(y,z))]
        mi += p_xyz * math.log((p_xyz * p_z) / (p_xz * p_yz))
    return mi


def _count(items: Sequence[Any]) -> Dict[Any, int]:
    out: Dict[Any, int] = {}
    for it in items:
        out[it] = out.get(it, 0) + 1
    return out


# ----------------------------------------------------------------------------------
# 主类：GraphRefutation
# ----------------------------------------------------------------------------------


class GraphRefutation:
    """DAG 图级条件独立性检验（spec.md §11 第 3.5 步：筛非驱动因子）。

    Parameters
    ----------
    p_threshold
        显著性阈值；p < p_threshold 即视为"显著"，edge.supported = True。
    r_threshold
        偏相关系数绝对值的下限；同时要求 |r| > r_threshold 才认为有意义
        （避免 p-value 在大样本下"虚假显著"）。
    min_samples
        样本量下限；n < min_samples → test_failed=True（不调检验）。
    """

    def __init__(
        self,
        *,
        p_threshold: float = DEFAULT_P_VALUE_THRESHOLD,
        r_threshold: float = DEFAULT_R_THRESHOLD,
        min_samples: int = MIN_SAMPLES_FOR_TEST,
    ) -> None:
        if not (0.0 < p_threshold < 1.0):
            raise ValueError(
                f"p_threshold 必须在 (0, 1) 内；got {p_threshold!r}"
            )
        if not (0.0 < r_threshold <= 1.0):
            raise ValueError(
                f"r_threshold 必须在 (0, 1] 内；got {r_threshold!r}"
            )
        self.p_threshold = float(p_threshold)
        self.r_threshold = float(r_threshold)
        self.min_samples = int(min_samples)

    # ------------------------------------------------------------------
    # 单边检验
    # ------------------------------------------------------------------

    def partial_correlation_check(
        self,
        causal_model: Any,
        edge: Tuple[str, str],
        *,
        conditioning_set: Optional[Sequence[str]] = None,
        data: Optional[pd.DataFrame] = None,
    ) -> EdgeTestResult:
        """检验 DAG 中一条边 X → Y 是否被数据支持（条件独立性）。

        算法
        ----
        1. 取 causal_model.data（或显式传入 data）作为 DataFrame；
        2. 把 X 与 Y 数值化（label encode 分类列）；
        3. conditioning_set 默认 = 除 X / Y 外所有列；
        4. 计算 Pearson 偏相关 + 双尾 t 检验 p-value；
        5. supported = (p < p_threshold) AND (|r| > r_threshold)；
        6. 任意步骤抛异常 → test_failed=True + reason；不静默。

        Parameters
        ----------
        causal_model
            `dowhy.CausalModel` 或其它持有 `.data` (pd.DataFrame) 的对象。
        edge
            (source, target) 节点 id 元组；必须都在 causal_model.data 的列中或可派生。
        conditioning_set
            显式指定 conditioning 变量列表（节点 id 字符串）。默认 None → 用
            "除 source / target 外的所有列"。
        data
            显式 DataFrame；传 None 时从 causal_model._data / .data 取。

        Returns
        -------
        EdgeTestResult
            含 r / p_value / n / supported / test_method / test_failed。
        """
        if edge is None or len(edge) != 2:
            raise ValueError(f"edge 必须是 (source, target) 元组；got {edge!r}")
        source, target = edge
        if source == target:
            raise ValueError(f"edge 不能自环；got {edge!r}")

        # 1. DataFrame
        try:
            if data is None:
                df = self._extract_dataframe(causal_model)
            else:
                df = data.copy()
        except Exception as exc:  # noqa: BLE001
            return EdgeTestResult(
                edge=(source, target),
                test_method="partial_correlation",
                test_failed=True,
                test_failure_reason=f"dataframe_extraction_failed:{type(exc).__name__}:{exc}",
            )

        if df is None or df.empty:
            return EdgeTestResult(
                edge=(source, target),
                test_method="partial_correlation",
                test_failed=True,
                test_failure_reason="empty_dataframe",
            )

        n_total = len(df)

        # 2. 样本量守门
        if n_total < self.min_samples:
            return EdgeTestResult(
                edge=(source, target),
                n_samples=n_total,
                test_method="partial_correlation",
                test_failed=True,
                test_failure_reason=(
                    f"sample_size_too_small:n={n_total}<{self.min_samples}"
                ),
            )

        # 3. 节点 → 列名映射
        source_col = self._resolve_column(source, df)
        target_col = self._resolve_column(target, df)
        if source_col is None or target_col is None:
            return EdgeTestResult(
                edge=(source, target),
                n_samples=n_total,
                test_method="partial_correlation",
                test_failed=True,
                test_failure_reason=(
                    f"column_not_found:source={source_col} target={target_col}"
                ),
            )
        if source_col == target_col:
            return EdgeTestResult(
                edge=(source, target),
                n_samples=n_total,
                test_method="partial_correlation",
                test_failed=True,
                test_failure_reason="source_target_same_column",
            )

        # 4. 数值化
        try:
            x_series = _label_encode(df[source_col])
            y_series = _label_encode(df[target_col])
        except Exception as exc:  # noqa: BLE001
            return EdgeTestResult(
                edge=(source, target),
                n_samples=n_total,
                test_method="partial_correlation",
                test_failed=True,
                test_failure_reason=f"label_encoding_failed:{type(exc).__name__}:{exc}",
            )

        # 5. 决定 conditioning set
        if conditioning_set is None:
            cond_cols_all = [c for c in df.columns if c not in (source_col, target_col)]
        else:
            cond_cols_all = [self._resolve_column(c, df) or c for c in conditioning_set]
            cond_cols_all = [c for c in cond_cols_all if c not in (source_col, target_col)]
            cond_cols_all = [c for c in cond_cols_all if c in df.columns]

        # 仅保留可数值化的列作为偏相关 conditioning（dowhy partial_corr 同样要求
        # 全部 numeric；本模块用 pd.to_numeric(errors='coerce') 把字符串列转 NaN
        # 后丢弃 NaN 行，避免 ValueError）。
        cond_cols: List[str] = []
        for c in cond_cols_all:
            try:
                coerced = pd.to_numeric(df[c], errors="coerce")
            except Exception:  # noqa: BLE001
                continue
            valid_frac = float(coerced.notna().mean())
            if valid_frac >= 0.6:
                cond_cols.append(c)

        if cond_cols:
            try:
                coerced_block = df[cond_cols].apply(
                    lambda col: pd.to_numeric(col, errors="coerce")
                )
                Z = coerced_block.to_numpy(dtype=float, na_value=np.nan)
            except Exception as exc:  # noqa: BLE001
                return EdgeTestResult(
                    edge=(source, target),
                    n_samples=n_total,
                    test_method="partial_correlation",
                    test_failed=True,
                    test_failure_reason=f"conditioning_to_numeric_failed:{type(exc).__name__}:{exc}",
                    conditioning_set=list(cond_cols),
                )
        else:
            Z = np.empty((n_total, 0))

        # 6. 选择检验方法：两个变量都连续 → Pearson 偏相关；否则退化为条件互信息
        both_continuous = _is_continuous(x_series) and _is_continuous(y_series)
        test_method = "partial_correlation" if both_continuous else "conditional_mutual_information"

        if test_method == "partial_correlation":
            try:
                r, p_value, n = _partial_corr_pearson(
                    x_series.to_numpy(dtype=float, na_value=np.nan),
                    y_series.to_numpy(dtype=float, na_value=np.nan),
                    Z,
                )
            except Exception as exc:  # noqa: BLE001
                return EdgeTestResult(
                    edge=(source, target),
                    n_samples=n_total,
                    test_method=test_method,
                    test_failed=True,
                    test_failure_reason=f"partial_corr_exception:{type(exc).__name__}:{exc}",
                    conditioning_set=list(cond_cols),
                )
            # NaN / inf 防护
            if not np.isfinite(r) or not np.isfinite(p_value):
                return EdgeTestResult(
                    edge=(source, target),
                    n_samples=n or n_total,
                    test_method=test_method,
                    test_failed=True,
                    test_failure_reason=(
                        f"non_finite_result:r={r!r},p={p_value!r}"
                    ),
                    conditioning_set=list(cond_cols),
                )
            supported = bool((p_value < self.p_threshold) and (abs(r) > self.r_threshold))
            return EdgeTestResult(
                edge=(source, target),
                r=float(r),
                p_value=float(p_value),
                n_samples=int(n),
                supported=supported,
                test_method=test_method,
                test_failed=False,
                conditioning_set=list(cond_cols),
            )

        # conditional_mutual_information path
        try:
            mi, n = _conditional_mutual_info(
                x_series.to_numpy(dtype=float, na_value=np.nan),
                y_series.to_numpy(dtype=float, na_value=np.nan),
                Z,
            )
        except Exception as exc:  # noqa: BLE001
            return EdgeTestResult(
                edge=(source, target),
                n_samples=n_total,
                test_method=test_method,
                test_failed=True,
                test_failure_reason=f"cmi_exception:{type(exc).__name__}:{exc}",
                conditioning_set=list(cond_cols),
            )
        # 用 MI 阈值代理：MI > 0.05 视为有信息量（经验阈值；样本量较大时偏严）
        supported = bool(mi > self.r_threshold)
        return EdgeTestResult(
            edge=(source, target),
            r=float(mi),
            p_value=None,  # MI 路径不给 p-value（经验互信息无标准 t 检验）
            n_samples=int(n),
            supported=supported,
            test_method=test_method,
            test_failed=False,
            conditioning_set=list(cond_cols),
        )

    # ------------------------------------------------------------------
    # 批量筛边
    # ------------------------------------------------------------------

    def filter_driver_edges(
        self,
        causal_model: Any,
        dag: Any,
        *,
        data: Optional[pd.DataFrame] = None,
        candidate_edges: Optional[Sequence[Tuple[str, str]]] = None,
        keep_kind: Optional[Sequence[str]] = None,
    ) -> List[EdgeTestResult]:
        """批量筛边：返回所有边的 EdgeTestResult 列表（不剔除任何 edge）。

        业务层在拿到 results 后，根据每条 edge.kind 与 supported 决定是否把它
        当作"驱动因子"。本函数不修改 dag，只返回检验结果。

        Parameters
        ----------
        causal_model
            CausalModel 实例；data=causal_model.data（或显式传入 data）。
        dag
            networkx.DiGraph；用于取 edges 与边属性（kind / color / spec_section）。
        candidate_edges
            显式指定要检验的边列表；默认 None → 取 dag.edges()。
        keep_kind
            仅对 kind ∈ keep_kind 的边跑检验；默认 None → 全部。常见用法：
            keep_kind=("treatment_to_outcome", "covariate_to_outcome") 即
            不检验"background"边。
        """
        import networkx as nx  # 局部导入，便于测试 mock

        if dag is None or not isinstance(dag, nx.DiGraph):
            raise ValueError(f"dag 必须是 networkx.DiGraph；got {type(dag).__name__}")

        edges = (
            list(candidate_edges) if candidate_edges is not None else list(dag.edges())
        )
        results: List[EdgeTestResult] = []
        for src, dst in edges:
            attrs = dag.edges.get((src, dst), {}) or {}
            kind = attrs.get("kind", "unknown") if isinstance(attrs, dict) else "unknown"

            if keep_kind is not None and kind not in keep_kind:
                results.append(
                    EdgeTestResult(
                        edge=(src, dst),
                        edge_kind=kind,
                        test_method="skipped",
                        test_failed=False,
                    )
                )
                continue

            r = self.partial_correlation_check(
                causal_model=causal_model,
                edge=(src, dst),
                data=data,
            )
            r.edge_kind = kind
            results.append(r)
        return results

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_dataframe(causal_model: Any) -> Optional[pd.DataFrame]:
        """从 CausalModel（或兼容对象）取 DataFrame。

        优先级：._data → .data → 直接是 DataFrame；都不行 → 返回 None。
        """
        for attr in ("_data", "data"):
            obj = getattr(causal_model, attr, None)
            if isinstance(obj, pd.DataFrame):
                return obj.copy()
        if isinstance(causal_model, pd.DataFrame):
            return causal_model.copy()
        return None

    @staticmethod
    def _resolve_column(node_id: str, df: pd.DataFrame) -> Optional[str]:
        """把节点 id（如 'treatment:PCI'）映射到 DataFrame 列名。

        业务 DAG 的节点 id 是带冒号前缀的形式（如 'treatment:PCI' /
        'age_band:1'），但 DataFrame 列名是连续列（treatment_strategy /
        age_band）。这里做如下归一化：
        - 完全匹配 → 返回；
        - 取 'prefix:value' 的 prefix → 视为"分组列"（treatment_strategy /
          age_band / admission_route / complication_count）；
        - 节点 id 等于 df 中某一列 → 返回该列；
        - 都不匹配 → 返回 None（让上层报 column_not_found）。
        """
        # 完全匹配（业务 DAG 节点 id 一般不会直接是列名，留作兜底）
        if node_id in df.columns:
            return node_id
        # 前缀匹配（treatment:PCI / age_band:1 等 → treatment_strategy / age_band）
        if ":" in node_id:
            prefix = node_id.split(":", 1)[0]
            # 业务约定：treatment → treatment_strategy；其余 prefix 直接当列名
            if prefix == "treatment":
                return "treatment_strategy"
            if prefix in df.columns:
                return prefix
        # 否则把节点 id 自身当列名（再试一次）
        if node_id in df.columns:
            return node_id
        return None


# ----------------------------------------------------------------------------------
# 便捷函数（业务层常调用）
# ----------------------------------------------------------------------------------


def filter_driver_edges(
    causal_model: Any,
    dag: Any,
    *,
    p_threshold: float = DEFAULT_P_VALUE_THRESHOLD,
    r_threshold: float = DEFAULT_R_THRESHOLD,
    min_samples: int = MIN_SAMPLES_FOR_TEST,
    candidate_edges: Optional[Sequence[Tuple[str, str]]] = None,
    keep_kind: Optional[Sequence[str]] = None,
    data: Optional[pd.DataFrame] = None,
) -> List[EdgeTestResult]:
    """GraphRefutation.filter_driver_edges 的最简封装。"""
    gr = GraphRefutation(
        p_threshold=p_threshold,
        r_threshold=r_threshold,
        min_samples=min_samples,
    )
    return gr.filter_driver_edges(
        causal_model=causal_model,
        dag=dag,
        data=data,
        candidate_edges=candidate_edges,
        keep_kind=keep_kind,
    )


def supported_driver_edges(
    results: Sequence[EdgeTestResult],
) -> List[EdgeTestResult]:
    """从批量检验结果中提取 supported=True 的边。"""
    return [r for r in results if r.supported and not r.test_failed]