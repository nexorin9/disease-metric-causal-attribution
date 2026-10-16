"""样本量守门（spec.md §6 样本量阈值 + §11 草稿生成守门）。

核心职责：把样本量 ``n`` 划成三档，控制草稿生成 + 上报写回的开关（spec §6 失败不静默改数据）。

三档定义（硬阈值，由用户在初始化时可覆盖）：
  - ``insufficient``   : n < INSUFFICIENT_THRESHOLD   → **不触发写回**（spec §6 失败降级），
                          标 ``sample_insufficient``；草稿评审表不可见；
  - ``marginal``       : INSUFFICIENT_THRESHOLD <= n < MARGINAL_THRESHOLD
                          → 草稿可生成但显式标 marginal；上报写回仍触发（业务可决策）；
  - ``sufficient``     : n >= MARGINAL_THRESHOLD         → 草稿正常生成 + 上报正常走。

为什么三档而非两档：临床评审会场景下 n=20 时仍允许医生手动补充归纳，但 n=10 时
因果估计的方差会爆掉；分段让业务面板能在 UI 上明确告诉质控员「这份草稿基于
marginal 数据，请人工重点核查」。

性能记录（spec §11 第 11 步 — 估算 n=30/100/500 性能）：
  estimate_runtime(n, fn, *args, **kwargs)
      计时调用 fn(...)；同时把 elapsed_ms + 档位 + n + 取样时间戳写入
      ``perf_records/sample_size_runtime_n<n>.json``，供面板展示「不同样本量
      全流水线耗时对比」(T18 benchmark 已落 perf_records/benchmark_*.json)。

⚠️ 本模块与样本量本身相关 — 阈值常量来源 spec §6 + §11 第 11 步。
⚠️ README 不得写源仓库名；本模块仅描述方法族（阈值分段 + 运行时长记录）。
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------
# 常量（spec §6 样本量阈值 + §11 第 11 步 — 估算性能档位）
# ----------------------------------------------------------------------------------

# 不可接受的最小样本量：n<30 时因果估计方差爆掉，不触发上报写回
INSUFFICIENT_THRESHOLD: int = 30

# 草稿可生成但需显式标 marginal 的上限：n<100 时下游应核对显著性 / 置信区间
MARGINAL_THRESHOLD: int = 100

# perf 估算（spec §11 第 11 步）— 与 INSUFFICIENT / MARGINAL 边界对齐
PERF_SAMPLE_SIZES: Tuple[int, ...] = (30, 100, 500)

# 性能记录 schema 版本（向前兼容）
PERF_RECORD_SCHEMA_VERSION: str = "sample_size_runtime/1.0"


# ----------------------------------------------------------------------------------
# 三档枚举（panel UI 与上报开关共用）
# ----------------------------------------------------------------------------------

VERDICT_INSUFFICIENT: str = "insufficient"
VERDICT_MARGINAL: str = "marginal"
VERDICT_SUFFICIENT: str = "sufficient"

ALL_VERDICTS: Tuple[str, ...] = (VERDICT_INSUFFICIENT, VERDICT_MARGINAL, VERDICT_SUFFICIENT)


# ----------------------------------------------------------------------------------
# 数据契约
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleSizeVerdict:
    """样本量三档判定（spec §6 + §11 草稿守门）。"""

    verdict: str  # 'insufficient' / 'marginal' / 'sufficient'
    sample_size: int
    insufficient_threshold: int
    marginal_threshold: int
    allows_draft_posting: bool
    allows_backend_write: bool
    degraded_reasons: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "sample_size": self.sample_size,
            "insufficient_threshold": self.insufficient_threshold,
            "marginal_threshold": self.marginal_threshold,
            "allows_draft_posting": self.allows_draft_posting,
            "allows_backend_write": self.allows_backend_write,
            "degraded_reasons": list(self.degraded_reasons),
            "info": dict(self.info),
        }


# ----------------------------------------------------------------------------------
# 1. SampleSizeGuard：核心守门类
# ----------------------------------------------------------------------------------


class SampleSizeGuard:
    """样本量守门（spec §6 + §11 草稿守门）。

    Usage
    -----
    >>> guard = SampleSizeGuard()
    >>> v = guard.check(25)
    >>> v.verdict             # 'insufficient'
    >>> v.allows_draft_posting  # False
    >>> v.allows_backend_write  # False
    >>> guard.check(60).verdict       # 'marginal'
    >>> guard.check(150).verdict      # 'sufficient'
    """

    def __init__(
        self,
        insufficient_threshold: int = INSUFFICIENT_THRESHOLD,
        marginal_threshold: int = MARGINAL_THRESHOLD,
    ) -> None:
        if insufficient_threshold < 1:
            raise ValueError(
                f"insufficient_threshold 必须 >= 1，实际 {insufficient_threshold}"
            )
        if marginal_threshold <= insufficient_threshold:
            raise ValueError(
                f"marginal_threshold 必须 > insufficient_threshold（spec §6）；"
                f"实际 {marginal_threshold} <= {insufficient_threshold}"
            )
        self.insufficient_threshold = int(insufficient_threshold)
        self.marginal_threshold = int(marginal_threshold)

    # -----------------------------------------------------------------
    # 1.1 三档判定
    # -----------------------------------------------------------------

    def classify(self, sample_size: int) -> str:
        """样本量 → 三档标签（spec §6 阈值分段）。"""
        self._validate_sample_size(sample_size)
        if sample_size < self.insufficient_threshold:
            return VERDICT_INSUFFICIENT
        if sample_size < self.marginal_threshold:
            return VERDICT_MARGINAL
        return VERDICT_SUFFICIENT

    def check(self, sample_size: int) -> SampleSizeVerdict:
        """样本量 → SampleSizeVerdict（spec §11 §6 草稿守门主入口）。

        Rules
        -----
        - insufficient : draft 不可生成 / 上报写回不可触发；
        - marginal     : draft 可生成（标 marginal）+ 上报写回可触发；
        - sufficient   : draft 正常生成 + 上报写回正常走。
        """
        self._validate_sample_size(sample_size)
        verdict = self.classify(sample_size)

        if verdict == VERDICT_INSUFFICIENT:
            allows_draft = False
            allows_write = False
            reasons: List[str] = []
            reasons.append(
                f"sample_insufficient:n={sample_size}<{self.insufficient_threshold}"
            )
        elif verdict == VERDICT_MARGINAL:
            allows_draft = True
            allows_write = True
            reasons = [
                f"sample_marginal:{self.insufficient_threshold}<=n={sample_size}<{self.marginal_threshold}",
            ]
        else:  # sufficient
            allows_draft = True
            allows_write = True
            reasons = []

        info: Dict[str, Any] = {
            "insufficient_threshold": self.insufficient_threshold,
            "marginal_threshold": self.marginal_threshold,
        }

        return SampleSizeVerdict(
            verdict=verdict,
            sample_size=sample_size,
            insufficient_threshold=self.insufficient_threshold,
            marginal_threshold=self.marginal_threshold,
            allows_draft_posting=allows_draft,
            allows_backend_write=allows_write,
            degraded_reasons=reasons,
            info=info,
        )

    # -----------------------------------------------------------------
    # 1.2 perf_records 性能记录（spec §11 第 11 步）
    # -----------------------------------------------------------------

    def estimate_runtime(
        self,
        n: int,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[SampleSizeVerdict, Any, float]:
        """跑 ``fn(*args, **kwargs)`` 并记时落盘到 ``perf_records/``。

        Parameters
        ----------
        n : int
            样本量；先过 check() 判定后再跑函数（spec §6 样本量阈值守卫）。
            当 verdict == insufficient 时仍会跑完（用于采集 perf 数值），
            但 ``perf_record.degraded=True`` 会被写入，便于面板标 marginal。
        fn : Callable
            待测函数；通常传入 ``lambda: pipeline.call_identify_effect()``
            之类无状态调用。
        *args, **kwargs
            透传给 fn。

        Returns
        -------
        (verdict, result, elapsed_seconds)
            verdict : SampleSizeVerdict
                三档判定结果。
            result : Any
                fn(*args, **kwargs) 的返回值；若 fn 抛异常则保留异常对象。
            elapsed_seconds : float
                实测耗时（秒，float）。
        """
        verdict = self.check(n)

        start = time.perf_counter()
        error: Optional[BaseException] = None
        result: Any = None
        try:
            result = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 — perf 阶段吞所有异常
            error = exc
            result = exc
        elapsed = time.perf_counter() - start
        elapsed_ms = elapsed * 1000.0

        # 即便样本量不足以走业务流，也把 perf 数值落盘（spec §11 第 11 步要求）
        record: Dict[str, Any] = {
            "schema_version": PERF_RECORD_SCHEMA_VERSION,
            "sample_size": n,
            "verdict": verdict.verdict,
            "elapsed_ms": elapsed_ms,
            "elapsed_seconds": elapsed,
            "ts": datetime.now(timezone.utc).isoformat(),
            "insufficient_threshold": self.insufficient_threshold,
            "marginal_threshold": self.marginal_threshold,
            "result_type": type(result).__name__ if error is None else "Exception",
            "exception_type": type(error).__name__ if error is not None else None,
            "exception_message": str(error) if error is not None else None,
            "allows_draft_posting": verdict.allows_draft_posting,
            "allows_backend_write": verdict.allows_backend_write,
            "fn_qualname": getattr(fn, "__qualname__", repr(fn)),
        }

        try:
            self._write_perf_record(record, n)
        except Exception as exc:  # noqa: BLE001 — 落盘失败不阻断业务
            LOGGER.warning("perf_records 落盘失败（n=%s）: %s", n, exc)

        if error is not None:
            LOGGER.info(
                "estimate_runtime(n=%d) fn 抛异常 (%s): %s；elapsed=%.4fs；verdict=%s",
                n, type(error).__name__, error, elapsed, verdict.verdict,
            )

        return verdict, result, elapsed

    def _write_perf_record(self, record: Dict[str, Any], n: int) -> Path:
        """写单条 perf 记录到 ``perf_records/sample_size_runtime_n<n>.json``。"""
        project_root = Path(__file__).resolve().parent.parent.parent
        target_dir = project_root / "perf_records"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"sample_size_runtime_n{n}.json"
        target_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return target_path

    # -----------------------------------------------------------------
    # 1.3 内部 helper
    # -----------------------------------------------------------------

    @staticmethod
    def _validate_sample_size(sample_size: Any) -> None:
        if not isinstance(sample_size, int) or isinstance(sample_size, bool):
            # bool 是 int 的子类但语义不该被当作 sample size
            raise ValueError(
                f"sample_size 必须是 int（不含 bool），实际类型 {type(sample_size).__name__}"
            )
        if sample_size < 0:
            raise ValueError(
                f"sample_size 必须 >= 0，实际 {sample_size}"
            )

    # -----------------------------------------------------------------
    # 1.4 便捷函数
    # -----------------------------------------------------------------

    def should_allow_draft(self, n: int) -> bool:
        """便捷：是否允许草稿评审单生成。"""
        return self.check(n).allows_draft_posting

    def should_allow_write(self, n: int) -> bool:
        """便捷：是否允许上游写回（DIPCASE 上报等）。"""
        return self.check(n).allows_backend_write


# ----------------------------------------------------------------------------------
# 2. 模块级便捷函数（不依赖实例配置）
# ------------------------------------------------------------------


def get_default_guard() -> SampleSizeGuard:
    """模块默认 SampleSizeGuard（30 / 100 阈值）。"""
    return SampleSizeGuard()


def classify_sample_size(
    sample_size: int,
    *,
    insufficient_threshold: int = INSUFFICIENT_THRESHOLD,
    marginal_threshold: int = MARGINAL_THRESHOLD,
) -> str:
    """无实例化分类（spec §6 三档）。"""
    return SampleSizeGuard(
        insufficient_threshold=insufficient_threshold,
        marginal_threshold=marginal_threshold,
    ).classify(sample_size)


def evaluate_sample_size(
    sample_size: int,
    *,
    insufficient_threshold: int = INSUFFICIENT_THRESHOLD,
    marginal_threshold: int = MARGINAL_THRESHOLD,
) -> SampleSizeVerdict:
    """无实例化完整 verdict（spec §11 草稿守门主入口）。"""
    return SampleSizeGuard(
        insufficient_threshold=insufficient_threshold,
        marginal_threshold=marginal_threshold,
    ).check(sample_size)
