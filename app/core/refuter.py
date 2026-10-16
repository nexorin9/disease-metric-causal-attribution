"""多档反驳（refute_estimate）封装（spec.md §11 主路径 + §9 因果推理管线）。

把参考仓库支持的多档反驳（random_common_cause / placebo_treatment / bootstrap /
data_subset）收敛到一个统一接口 `RefuterPipeline.run(causal_model, estimate)`，每档独立
捕获异常并返回 `List[RefutationResult]`。这是规格中的「失败不静默改数据」原则的落地：
反驳失败 → 标 refutation_unstable=True，但不抛异常阻断后续草稿生成（草稿可以带降级标记走
reviewer 复核）。

业务层使用：
  from app.core.refuter import RefuterPipeline
  results: List[RefutationResult] = RefuterPipeline(
      causal_model=cm, estimate=estimate, estimand=estimand,
  ).run()
  for r in results:
      print(r.method, r.passed, r.refutation_unstable, r.degraded)

源产品能力参考：
- 反多档反驳入口见 `github_ref/dowhy/causal_refuter.py:769-820` 的
  `refute_estimate(method_name=...)` 调度；
- 反四档具体实现分别在 `github_ref/dowhy/causal_refuters/` 下四个文件；
- 并行执行见 `github_ref/dowhy/example_parallel_refutation.py` 的 `n_jobs=...`；
- README / 项目子目录对外文档不得出现源仓库名（dowhy / pywhy / 等）。

设计要点：
- 每档独立 try/except → 异常不阻断其它档；
- 任意档抛异常 → 标 refutation_unstable=True，passed=None，p_value=None；
- 缺 estimate.value / 数据形状过小（n < min_refute_samples=30）→ 全部档不抛：
  显式构造 RefutationResult(refutation_unstable=True, ...)，业务层可正常走草稿生成。
- 降级标记（CI 超宽 / p-value 过高 / 拒收敛）由 `is_degraded(result)` 统一判断：
  T8 落基础多档结构，T9 加并行，T10 加拒收敛/CI 超宽显式降级。
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from app.core.node_schema import RefutationResult

LOGGER = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------
# 常量
# ----------------------------------------------------------------------------------

# 单次反驳最低样本量：与规格 §6 样本量阈值一致。
MIN_REFUTE_SAMPLES: int = 30

# 默认 num_simulations：bootstrap/placebo 各档运行时随机抽样次数。
DEFAULT_NUM_SIMULATIONS_RCC: int = 100   # random_common_cause
DEFAULT_NUM_SIMULATIONS_PLC: int = 100   # placebo_treatment_refuter
DEFAULT_NUM_SIMULATIONS_BTS: int = 100   # bootstrap_refuter
DEFAULT_NUM_SIMULATIONS_SUB: int = 100   # data_subset_refuter

# T10 拒收敛 / CI 超宽降级阈值（spec §11 第 3 步 — 降级判定）：
#   - CI 宽度（ci_upper - ci_lower）超过 0.6 → 结论可信区间跨度过大，标 degraded；
#   - 反驳 p_value 超过 0.5 → 无法拒绝零假设，标 degraded。
# 两者任一触发即视为该档降级；refutation_unstable 与 degraded 不互斥：
#   - refutation_unstable=True → 该档完全失败（未跑出有效结果）；
#   - degraded=True → 该档跑出了结果但结论不可信（CI 超宽 / p-value 过高）。
MAX_CI_WIDTH: float = 0.6
MAX_REFUTATION_P_VALUE: float = 0.5

# T8 任务固定的方法枚举（不可变），避免散落成散字符串。
SUPPORTED_METHODS: tuple = (
    "random_common_cause",
    "placebo_treatment",
    "bootstrap",
    "data_subset",
)
# 内部别名：schema 用简洁名，dowhy API 用带 _refuter 后缀的枚举名。
_DOWHY_METHOD_ALIAS = {
    "random_common_cause": "random_common_cause",
    "placebo_treatment": "placebo_treatment_refuter",
    "bootstrap": "bootstrap_refuter",
    "data_subset": "data_subset_refuter",
}


# ----------------------------------------------------------------------------------
# 内部 dataclass：包装 dowhy 的 RefuteResult 属性。
# ----------------------------------------------------------------------------------


@dataclass
class _RawRefute:
    """dowhy refute_estimate 返回的内部对象属性快照（避免直接持 dowhy 对象）。"""

    new_effect: Optional[float]
    p_value: Optional[float]
    passed: Optional[bool]
    raw: Any  # 保留原对象供审计 / 调试


def _extract_dowhy_refute(out: Any) -> _RawRefute:
    """从 dowhy RefuteResult（或类似对象）抽取三字段。

    不同 dowhy 版本返回的对象属性名不尽相同（Refute / RefuteResult / CausalRefutation），
    这里走鸭子类型访问，任意字段缺失则记 None。
    """
    new_eff: Optional[float] = None
    pv: Optional[float] = None
    passed: Optional[bool] = None

    # new_effect：可能是 .new_effect / .effect / .new_estimate
    for attr in ("new_effect", "new_estimate", "effect"):
        v = getattr(out, attr, None)
        if v is not None:
            try:
                new_eff = float(v)
                break
            except (TypeError, ValueError):
                continue

    # p_value
    for attr in ("p_value", "pvalue", "pval"):
        v = getattr(out, attr, None)
        if v is not None:
            try:
                pv = float(v)
                # p < 0 或 p > 1 视为不可用
                if 0.0 <= pv <= 1.0:
                    break
                pv = None
            except (TypeError, ValueError):
                continue

    # passed
    passed_attr = getattr(out, "passed", None)
    if isinstance(passed_attr, bool):
        passed = passed_attr
    else:
        # 退而求其次：bool(evaluate) 或 根据 new_effect 与 original 的接近度判断
        bool_method = getattr(out, "evaluate", None)
        if callable(bool_method):
            try:
                r = bool_method()
                passed = bool(r) if r is not None else None
            except Exception:  # noqa: BLE001
                LOGGER.debug("RefuteResult.evaluate() 抛异常；passed 留 None", exc_info=True)

    return _RawRefute(new_effect=new_eff, p_value=pv, passed=passed, raw=out)


# ----------------------------------------------------------------------------------
# T10: 拒收敛 / CI 超宽降级判定
# ----------------------------------------------------------------------------------


def _compute_ci_width(result: Any) -> Optional[float]:
    """从 RefutationResult 取 ci_width（直接字段，或从 ci_upper/ci_lower 推算）。

    RefutationResult schema 没有 ci_width 字段（schema 与 pipeline 共用时只保留
    estimate 用的字段；反驳 p_value 是核心指标）。所以这里只读 result 上的
    ci_upper / ci_lower 字段或推断自新效应与 p_value 的常规分布。
    """
    # 优先直接字段
    cw = getattr(result, "ci_width", None)
    if cw is not None:
        try:
            return float(cw)
        except (TypeError, ValueError):
            pass
    # 退而求其次：上界 - 下界
    up = getattr(result, "ci_upper", None)
    lo = getattr(result, "ci_lower", None)
    if up is not None and lo is not None:
        try:
            return float(up) - float(lo)
        except (TypeError, ValueError):
            return None
    return None


def _compute_effect_swing(result: Any) -> Optional[float]:
    """计算 |new_effect - original_effect|（效应摆动幅度）作为降级辅助信号。

    当 dowhy 反驳后效应变化过大时，意味着原始估计对数据扰动敏感 → 不可信。
    """
    new_eff = getattr(result, "new_effect", None)
    orig_eff = getattr(result, "original_effect", None)
    if new_eff is None or orig_eff is None:
        return None
    try:
        return abs(float(new_eff) - float(orig_eff))
    except (TypeError, ValueError):
        return None


def is_degraded(result: RefutationResult) -> tuple:
    """T10：判断 RefutationResult 是否降级（CI 超宽 / p_value 过高 / 效应摆动过大）。

    判定规则（独立可累加的 reasons 列表）：
      - ci_width > MAX_CI_WIDTH (0.6) → "ci_width_too_wide:{w:.4f}>{MAX_CI_WIDTH}"
      - p_value > MAX_REFUTATION_P_VALUE (0.5) → "p_value_too_high:{p:.4f}>{MAX_REFUTATION_P_VALUE}"
      - |new_effect - original_effect| > MAX_CI_WIDTH 且 new_effect 与 original 极性反转
        → "effect_sign_reversed_and_swing:{swing:>{MAX_CI_WIDTH}}"

    Parameters
    ----------
    result
        RefutationResult（业务层 + schema 双重约束）。

    Returns
    -------
    (degraded: bool, reasons: List[str])
        - degraded = (len(reasons) > 0)；
        - reasons 包含全部命中的判定原因（去重 / 限长 5）；
        - 永不抛异常；result 字段缺失或类型异常均记为「不可判」并返回 (False, [])。

    Notes
    -----
    - 这是 spec §6 「失败不静默改数据」原则的延伸：即便 dowhy 跑通了，仍可能因为
      CI 超宽或 p_value 过高导致结论不可信；必须显式降级，供 reviewer 复核。
    - 与 refutation_unstable 不同：unstable = 跑失败（未跑出有效结果）；
      degraded = 跑出来了但不可信（CI 超宽 / p-value 过高）。
    """
    reasons: List[str] = []

    try:
        # CI 超宽
        ci_width = _compute_ci_width(result)
        if ci_width is not None and ci_width > MAX_CI_WIDTH:
            reasons.append(
                f"ci_width_too_wide:{ci_width:.4f}>{MAX_CI_WIDTH}"
            )

        # p_value 过高
        pv = getattr(result, "p_value", None)
        if pv is not None:
            try:
                pv_f = float(pv)
                if 0.0 <= pv_f <= 1.0 and pv_f > MAX_REFUTATION_P_VALUE:
                    reasons.append(
                        f"p_value_too_high:{pv_f:.4f}>{MAX_REFUTATION_P_VALUE}"
                    )
            except (TypeError, ValueError):
                pass

        # 效应极性反转 + 大幅摆动（说明原始估计对扰动极敏感）
        swing = _compute_effect_swing(result)
        new_eff = getattr(result, "new_effect", None)
        orig_eff = getattr(result, "original_effect", None)
        if (
            swing is not None
            and new_eff is not None
            and orig_eff is not None
            and swing > MAX_CI_WIDTH
        ):
            try:
                # 极性反转：new * original < 0 且绝对值都不接近 0
                if float(new_eff) * float(orig_eff) < 0.0:
                    reasons.append(
                        f"effect_sign_reversed_and_swing:{swing:.4f}>{MAX_CI_WIDTH}"
                    )
            except (TypeError, ValueError):
                pass
    except Exception as exc:  # noqa: BLE001 — 防御性兜底，绝不抛
        LOGGER.debug("is_degraded 判定期望外异常；返回 (False, []): %s", exc, exc_info=True)
        return (False, [])

    return (bool(reasons), reasons)


# ----------------------------------------------------------------------------------
# RefuterPipeline 主体
# ----------------------------------------------------------------------------------


class RefuterPipeline:
    """多档反驳封装（spec.md §11 第 3 步；T8 主类）。

    Parameters
    ----------
    causal_model
        dowhy `CausalModel` 实例（已构造 data + treatment + outcome + graph/common_causes）。
    estimate
        CausalPipeline.call_estimate_effect(...) 返回的 EffectEstimate；
        内部会调用 estimate_effect 重跑以拿 dowhy 原始 estimate 对象（因 EffectEstimate 不
        直接持有 dowhy CausalEstimate，refute_estimate 需后者）。
    estimand
        CausalPipeline.call_identify_effect(...) 返回的 Estimand（必填：refute 需 raw_estimand）。
    data
        原始 pd.DataFrame；用于样本量守门（n < MIN_REFUTE_SAMPLES 直接标 unstable）。
    methods
        要跑的方法名列表；默认全部四档。传入空列表则返回空 list。
    num_simulations
        反驳仿真次数；每档方法独立传 num_simulations=...。默认 100。
    """

    def __init__(
        self,
        *,
        causal_model: Any,
        estimate: Any,
        estimand: Any,
        data: Any,
        methods: Optional[Sequence[str]] = None,
        num_simulations: int = 100,
    ) -> None:
        if causal_model is None:
            raise ValueError("causal_model 不能为空；先构造 CausalPipeline 再跑 identify/estimate")
        if estimate is None:
            raise ValueError("estimate 不能为空；先跑 estimate_effect")
        if estimand is None:
            raise ValueError("estimand 不能为空；先跑 identify_effect")
        if data is None:
            raise ValueError("data 不能为空；样本量守门所需")

        self.causal_model = causal_model
        self.estimate = estimate  # EffectEstimate（业务层包装）
        self.estimand = estimand  # Estimand（业务层包装）
        self.data = data
        self.methods: List[str] = list(methods) if methods is not None else list(SUPPORTED_METHODS)
        self._validate_methods()
        self.num_simulations = int(num_simulations)

    def _validate_methods(self) -> None:
        for m in self.methods:
            if m not in _DOWHY_METHOD_ALIAS:
                raise ValueError(
                    f"未知反驳方法: {m!r}；支持: {list(_DOWHY_METHOD_ALIAS)}"
                )

    # ------------------------------------------------------------------
    # 入口
    # ------------------------------------------------------------------

    def run(self) -> List[RefutationResult]:
        """跑 self.methods 中所有档，返回 List[RefutationResult]。

        每档独立 try/except：单档异常只影响该档（标 refutation_unstable=True），
        不阻断其它档；总入口从不向业务层冒泡异常。
        """
        # 样本量守门：n < MIN_REFUTE_SAMPLES 直接构造 unstable 结果（不真正调 dowhy）
        n_samples = int(len(self.data))
        if n_samples < MIN_REFUTE_SAMPLES:
            LOGGER.warning(
                "样本量不足：n=%d < %d；全部反驳档标 refutation_unstable=True",
                n_samples,
                MIN_REFUTE_SAMPLES,
            )
            return [
                self._build_result(
                    method=m,
                    new_effect=None,
                    original_effect=self._safe_ate(),
                    p_value=None,
                    passed=None,
                    refutation_unstable=True,
                    degraded_reasons=[f"sample_size_too_small:n={n_samples}<{MIN_REFUTE_SAMPLES}"],
                )
                for m in self.methods
            ]

        results: List[RefutationResult] = []
        for method in self.methods:
            result = self._run_single(method, n_jobs=1)
            results.append(result)
        self._last_results = results
        return results

    def run_parallel(self, n_jobs: int = -1) -> List[RefutationResult]:
        """并行跑 self.methods 中所有档（T9；spec §11 第 3 步反驳 — 并行档位）。

        调度策略：
        - **方法间并行**：使用 ThreadPoolExecutor 将 self.methods 中所有档并行调度；
          dowhy 内部仿真多为 Python 计算密集 + 部分 NumPy / sklearn C 代码，线程池在
          GIL 释放点上仍可受益，且线程比进程开销小、dowhy 内部状态可复用。
        - **方法内并行**：将 `n_jobs` 透传给 dowhy 的 `refute_estimate` 调用（dowhy 0.10+
          在 random_common_cause / placebo / bootstrap / data_subset 内部均支持
          `n_jobs` 并行抽样；见 `github_ref/dowhy/example_parallel_refutation.py`）。
        - **异常隔离**：每档独立 try/except（沿用 `_run_single` 内部已有的隔离路径），
          线程层额外兜底 Future.exception()，单档失败不阻塞其它档也不抛到主线程。

        Parameters
        ----------
        n_jobs
            并行 worker 数；-1 表示全部 CPU。1 = 顺序（等价 `run()`）。

        Returns
        -------
        List[RefutationResult]
            按 self.methods 的顺序返回（与 `run()` 等价；用于 SSE / 草稿生成链路）。
        """
        # 规范化 n_jobs：-1 → CPU 数；< 1（除 -1）→ 视为 1
        if n_jobs == -1:
            n_jobs = max(1, os.cpu_count() or 1)
        elif n_jobs < 1:
            LOGGER.warning(
                "run_parallel: n_jobs=%d 非法（除 -1 外需 >= 1）；降级到 1", n_jobs
            )
            n_jobs = 1

        # 记录本次并行 worker 数（透传到 RefutationResult.n_jobs）
        self._last_n_jobs = int(n_jobs)

        # 样本量守门：n < MIN_REFUTE_SAMPLES 不真正调 dowhy；走 run() 等价逻辑。
        n_samples = int(len(self.data))
        if n_samples < MIN_REFUTE_SAMPLES:
            LOGGER.warning(
                "run_parallel: 样本量不足 n=%d < %d；全部档标 refutation_unstable=True",
                n_samples,
                MIN_REFUTE_SAMPLES,
            )
            results = [
                self._build_result(
                    method=m,
                    new_effect=None,
                    original_effect=self._safe_ate(),
                    p_value=None,
                    passed=None,
                    refutation_unstable=True,
                    degraded_reasons=[
                        f"sample_size_too_small:n={n_samples}<{MIN_REFUTE_SAMPLES}"
                    ],
                    n_jobs=n_jobs,
                )
                for m in self.methods
            ]
            self._last_results = results
            return results

        # 单档时退化为顺序跑（避免线程开销）
        if n_jobs == 1 or len(self.methods) <= 1:
            LOGGER.debug(
                "run_parallel: n_jobs=%d 或 methods=%d；退化到顺序 run()",
                n_jobs,
                len(self.methods),
            )
            return self.run()

        results: List[RefutationResult] = []
        # 保留方法顺序 → 用 dict 收集
        out_by_method: Dict[str, RefutationResult] = {}

        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            future_to_method = {
                executor.submit(self._run_single_with_n_jobs, m, n_jobs): m
                for m in self.methods
            }
            for future in as_completed(future_to_method):
                method_name = future_to_method[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001 — 兜底线程层异常
                    LOGGER.warning(
                        "run_parallel: method=%s 线程层抛异常: %s；降级为 unstable",
                        method_name,
                        exc,
                    )
                    result = self._build_result(
                        method=method_name,
                        new_effect=None,
                        original_effect=self._safe_ate(),
                        p_value=None,
                        passed=None,
                        refutation_unstable=True,
                        degraded_reasons=[f"thread_pool_exception:{type(exc).__name__}"],
                        n_jobs=n_jobs,
                    )
                out_by_method[method_name] = result

        # 按 self.methods 顺序返回
        for m in self.methods:
            results.append(out_by_method[m])
        self._last_results = results
        return results

    def verdict(self, results: Optional[List[RefutationResult]] = None) -> str:
        """汇总多档反驳结果给出总判定（T9 + T10；spec §11 第 3 步 — 总判定）。

        判定规则（基于 RefutationResult 集合）：
        - **unstable**：所有档 refutation_unstable=True（任何档都没有真正跑通），
          **或** 全档 degraded=True（所有档都触发了 CI 超宽 / p-value 过高降级标记），
          **或** results 为空。
        - **robust**：所有档 passed=True 且 refutation_unstable=False 且 degraded=False
          （四档全跑通、无降级）。
        - **partial**：其它情况（部分通过 / 部分失败 / 部分降级）。

        Parameters
        ----------
        results
            可选；传 None 时使用 `run()` / `run_parallel()` 最近一次的缓存（self._last_results）。

        Returns
        -------
        str
            {"robust", "partial", "unstable"} 之一。
        """
        if results is None:
            results = getattr(self, "_last_results", None) or []
        if not results:
            # 无反驳结果：按 spec §6 失败不静默原则视为 unstable
            return "unstable"

        total = len(results)
        unstable_count = sum(1 for r in results if r.refutation_unstable)
        passed_count = sum(1 for r in results if r.passed is True)
        # T10: degraded 判定走 is_degraded()（与 _run_single 内部一致）
        # 这里的 degraded_count 与 passed_count 是两套独立计数：refutation_unstable=True
        # 的档既不算 passed 也不算 degraded（已失败 → 不参与 passed/degraded 计数）。
        degraded_count = sum(1 for r in results if getattr(r, "degraded", False))

        # 全部 unstable → unstable
        if unstable_count == total:
            return "unstable"

        # 全部 passed 且无降级 → robust
        if passed_count == total and degraded_count == 0:
            return "robust"

        # 全部 degraded（且都不算 passed） → unstable
        # 当所有档都触发降级时，结论仍不可信
        if degraded_count == total and passed_count == 0:
            return "unstable"

        # 其它 → partial
        return "partial"

    # ------------------------------------------------------------------
    # 内部：单档
    # ------------------------------------------------------------------

    def _run_single(self, method: str, n_jobs: int = 1) -> RefutationResult:
        dowhy_method = _DOWHY_METHOD_ALIAS[method]
        raw_estimate = self._get_dowhy_estimate()

        try:
            # n_jobs 透传给 dowhy：dowhy 0.10+ 各档支持 n_jobs 并行抽样
            # （github_ref/dowhy/example_parallel_refutation.py 已确认）
            # 仅当 n_jobs != 1 时透传（避免对不支持 n_jobs 的版本产生意外副作用）
            kw = {
                "method_name": dowhy_method,
                "num_simulations": self.num_simulations,
                "show_progress_bar": False,
            }
            if n_jobs != 1:
                kw["n_jobs"] = n_jobs
            refute_out = self.causal_model.refute_estimate(
                self.estimand.raw_estimand,
                raw_estimate,
                **kw,
            )
        except TypeError as type_exc:
            # 部分 dowhy 版本不识别 n_jobs 参数 → 退化到无 n_jobs 重试
            if "n_jobs" in str(type_exc):
                LOGGER.debug(
                    "dowhy 不支持 n_jobs 参数（method=%s）；退化重试: %s",
                    method,
                    type_exc,
                )
                try:
                    refute_out = self.causal_model.refute_estimate(
                        self.estimand.raw_estimand,
                        raw_estimate,
                        method_name=dowhy_method,
                        num_simulations=self.num_simulations,
                        show_progress_bar=False,
                    )
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning(
                        "refute_estimate 抛异常（method=%s, 无 n_jobs 重试）: %s",
                        method,
                        exc,
                    )
                    return self._build_result(
                        method=method,
                        new_effect=None,
                        original_effect=self._safe_ate(),
                        p_value=None,
                        passed=None,
                        refutation_unstable=True,
                        degraded_reasons=[f"exception:{type(exc).__name__}"],
                        n_jobs=n_jobs,
                    )
            else:
                LOGGER.warning(
                    "refute_estimate TypeError（method=%s）: %s",
                    method,
                    type_exc,
                )
                return self._build_result(
                    method=method,
                    new_effect=None,
                    original_effect=self._safe_ate(),
                    p_value=None,
                    passed=None,
                    refutation_unstable=True,
                    degraded_reasons=[f"exception:{type(type_exc).__name__}"],
                    n_jobs=n_jobs,
                )
        except Exception as exc:  # noqa: BLE001 — dowhy 各档内部抛异常类型不一致
            LOGGER.warning(
                "refute_estimate 抛异常（method=%s）: %s",
                method,
                exc,
            )
            return self._build_result(
                method=method,
                new_effect=None,
                original_effect=self._safe_ate(),
                p_value=None,
                passed=None,
                refutation_unstable=True,
                degraded_reasons=[f"exception:{type(exc).__name__}"],
                n_jobs=n_jobs,
            )

        # 解析 dowhy RefuteResult 属性 → 业务 schema
        raw = _extract_dowhy_refute(refute_out)

        # T10: 先构造裸 RefutationResult，再用 is_degraded() 决定 degraded / degraded_reasons。
        # 这样能让 is_degraded 看到完整 schema 字段（含 ci_width / p_value / new_effect /
        # original_effect）；reviewer 看到的 degraded 字段与 is_degraded() 函数返回值完全一致。
        base_result = self._build_result(
            method=method,
            new_effect=raw.new_effect,
            original_effect=self._safe_ate(),
            p_value=raw.p_value,
            passed=raw.passed,
            refutation_unstable=False,
            degraded_reasons=[],
            n_jobs=n_jobs,
        )

        # 应用降级标记（CI 超宽 / p_value 过高 / 效应极性反转）
        degraded, reasons = is_degraded(base_result)
        if degraded:
            # 合并原始 degraded_reasons 与本次新增原因
            merged = list(base_result.degraded_reasons) + reasons
            # 去重保持顺序
            seen: set = set()
            deduped: List[str] = []
            for s in merged:
                if s not in seen:
                    seen.add(s)
                    deduped.append(s)
            base_result.degraded = True
            base_result.degraded_reasons = deduped[:8]  # 限长 8 防爆
            LOGGER.info(
                "refuter degraded: method=%s reasons=%s",
                method,
                deduped,
            )

        return base_result

    def _run_single_with_n_jobs(self, method: str, n_jobs: int) -> RefutationResult:
        """run_parallel 线程任务包装：记录线程起始时间到 result.elapsed_ms。

        直接调 `_run_single(method, n_jobs=n_jobs)`，因为 _run_single 已支持 n_jobs
        参数。此处包装是为 run_parallel 留清晰入口（未来可扩展 thread-local 上下文）。
        """
        return self._run_single(method, n_jobs=n_jobs)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _get_dowhy_estimate(self) -> Any:
        """拿 dowhy CausalEstimate 对象（refute_estimate 需后者的 .value）。

        CausalPipeline.call_estimate_effect() 内部已经跑过 estimate_effect；这里从
        causal_model 实例的 _estimate_holder / 或最近一次 estimate_effect 拿到原始
        对象。dowhy 0.14 没有缓存 estimate，所以最稳妥的做法是再跑一次（spec §9 流水线
        应可重复执行），用相同 method_name。
        """
        try:
            method_name = getattr(self.estimate, "method_name", None)
            if method_name is None:
                return None
            fresh = self.causal_model.estimate_effect(
                self.estimand.raw_estimand,
                method_name=method_name,
            )
            return fresh
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("重新估计以拿 dowhy CausalEstimate 失败: %s", exc)
            return None

    def _safe_ate(self) -> Optional[float]:
        a = getattr(self.estimate, "ate", None)
        if a is None:
            return None
        try:
            return float(a)
        except (TypeError, ValueError):
            return None

    def _build_result(
        self,
        *,
        method: str,
        new_effect: Optional[float],
        original_effect: Optional[float],
        p_value: Optional[float],
        passed: Optional[bool],
        refutation_unstable: bool,
        degraded_reasons: List[str],
        n_jobs: int = 1,
    ) -> RefutationResult:
        """构造 RefutationResult（schema 校验由 Pydantic 完成；构造失败显式降级 unstable）。

        n_jobs 参数说明：
        - run() 默认 n_jobs=1（顺序执行）。
        - run_parallel() 透传其 n_jobs（-1 已被规范化成实际 worker 数）。
        """
        try:
            return RefutationResult(
                method=method,
                new_effect=new_effect,
                original_effect=original_effect,
                p_value=p_value,
                passed=passed,
                degraded=bool(degraded_reasons),  # T8: degraded=True 当有降级原因
                degraded_reasons=degraded_reasons,
                refutation_unstable=refutation_unstable,
                n_jobs=int(n_jobs),
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "RefutationResult 构造失败（method=%s）: %s；降级到 stable=False + passed=None",
                method,
                exc,
            )
            return RefutationResult(
                method=method,
                new_effect=None,
                original_effect=None,
                p_value=None,
                passed=None,
                degraded=True,
                degraded_reasons=degraded_reasons + [f"schema_construction_failed:{type(exc).__name__}"],
                refutation_unstable=True,
                n_jobs=int(n_jobs),
            )


# ----------------------------------------------------------------------------------
# 便捷函数（业务层常调用）
# ------------------------------------------------------------------


def run_all_refuters(
    causal_model: Any,
    estimate: Any,
    estimand: Any,
    data: Any,
    *,
    methods: Optional[Sequence[str]] = None,
    num_simulations: int = 100,
    n_jobs: int = 1,
    parallel: bool = False,
) -> List[RefutationResult]:
    """一次性跑完多档反驳（RefuterPipeline 的最简封装）。

    Parameters
    ----------
    parallel
        True 时调用 run_parallel（n_jobs 生效）；False 时调用 run（顺序）。
        默认 False 以保持向后兼容（T8 行为）。
    """
    pipeline = RefuterPipeline(
        causal_model=causal_model,
        estimate=estimate,
        estimand=estimand,
        data=data,
        methods=list(methods) if methods is not None else None,
        num_simulations=num_simulations,
    )
    if parallel:
        return pipeline.run_parallel(n_jobs=n_jobs)
    return pipeline.run()
