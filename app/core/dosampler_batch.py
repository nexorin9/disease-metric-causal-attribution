"""多假设批量反事实（spec.md §11 主路径 / §9 因果推理管线）。

把 T12 单条 CounterfactualRunner.run() 升级为 BatchCounterfactualRunner.run()：
一次性提交多条假设干预（每条 = (intervention_node, hypothetical_value)），在
``batch_size`` 内分批并行调度，全部场景在 ``BATCH_TOTAL_TIMEOUT_SECONDS`` 内
返回 ``List[CounterfactualScenario]``。这是因果面板「批量反事实实验」按钮的
后端主路径：质控员一次提交 10 条干预假设，面板上一次性看到 10 条预测 outcome
与置信区间，便于横向对比。

设计原则（spec §6 失败不静默改数据）：
- **批量并发**：用 ThreadPoolExecutor 跑每个场景；每条场景独立 try/except，
  失败只影响该条（标 timeout 或 do_operations 含 exception），不阻断其它场景。
- **总超时门**：30 秒内必须全部返回；超过即剩余未跑场景返回 timeout=True。
- **单条超时仍由 CounterfactualRunner 自身控制（5s）**；批量层只做并发调度
  与总量控制，不重复引入超时。
- **scenario_id 稳定**：每条场景沿用 CounterfactualRunner._build_scenario_id
  同一哈希，便于面板 SSE 流式去重与 cache 命中。

源产品能力参考（README 不写源仓库名）：
- 批量并发思路与源 DoSampler 在 ``do_sample(x)`` 之上调度多次 do() 等价。
- 本模块不引入新的因果算法；每条场景仍走 T12 的数值拟合（线性回归 +
  do()），差异只在调度层。
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import pandas as pd

from app.core.dosampler import (
    COUNTERFACTUAL_TIMEOUT_SECONDS,
    CounterfactualRunner,
    _build_scenario_id,
)
from app.core.node_schema import CounterfactualScenario

LOGGER = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------
# 常量
# ----------------------------------------------------------------------------------

# 批量场景并发上限（spec §11 第 4 步批量反事实；与面板「一次提交 10 条」对齐）。
BATCH_SIZE_DEFAULT: int = 10

# 批量场景总超时：10 条场景 × 单条 5s 超时，串行理论上界 50s；并发下 30s 足够。
BATCH_TOTAL_TIMEOUT_SECONDS: float = 30.0

# 线程池 worker 数：默认 batch_size 上限（10），足以一次跑完所有场景。
_BATCH_THREAD_POOL_MAX_WORKERS: int = 10


# ----------------------------------------------------------------------------------
# 输入数据契约
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Intervention:
    """单条批量反事实干预（spec §11 第 4 步批量反事实）。

    Attributes
    ----------
    intervention_node
        干预节点 id（如 "treatment:PCI" / "age_band:3"）。
    hypothetical_value
        假设取值（如 "PCI" / 3 / 1）。
    label
        可读标签（供面板显示；如 "全部用 PCI"）；不参与 scenario_id 计算。
    """

    intervention_node: str
    hypothetical_value: Any
    label: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.intervention_node, str) or not self.intervention_node:
            raise ValueError("intervention_node 不能为空")
        if self.hypothetical_value is None:
            raise ValueError("hypothetical_value 不能为 None")


# ----------------------------------------------------------------------------------
# BatchCounterfactualRunner 主体
# ----------------------------------------------------------------------------------


class BatchCounterfactualRunner:
    """批量反事实运行器（spec.md §11 第 4 步 / T13）。

    Usage
    -----
    >>> runner = BatchCounterfactualRunner(
    ...     disease_code="I21", outcome="cmi", data=df,
    ... )
    >>> scenarios = runner.run([
    ...     Intervention("treatment:PCI", "PCI", "全部 PCI"),
    ...     Intervention("age_band:3", 3, "年龄段统一 65-74"),
    ...     ...
    ... ])
    >>> for s in scenarios:
    ...     print(s.scenario_id, s.predicted_outcome, s.timeout)
    """

    def __init__(
        self,
        disease_code: str,
        outcome: str,
        data: pd.DataFrame,
        *,
        batch_size: int = BATCH_SIZE_DEFAULT,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size 必须 >= 1；got {batch_size}")
        # 立即校验 disease_code / outcome / data：走 CounterfactualRunner.__init__
        # 触发 ValueError（未知 disease_code / outcome / 空 data）。
        # 这样调用方在构造期就能发现问题，不必等到 run()。
        self.disease_code = disease_code
        self.outcome = outcome
        self.data = data.copy() if data is not None else data
        self.batch_size = int(batch_size)
        # 触发 CounterfactualRunner 构造期校验（仅校验，不持有）
        try:
            CounterfactualRunner(
                disease_code=self.disease_code,
                outcome=self.outcome,
                data=self.data,
            )
        except ValueError:
            raise  # 透传给调用方（test 期望 in __init__）

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def run(self, scenarios: Sequence[Intervention]) -> List[CounterfactualScenario]:
        """跑批量反事实场景 → List[CounterfactualScenario]（30s 总超时）。

        Parameters
        ----------
        scenarios
            干预场景列表；长度超过 batch_size 时按 batch_size 分批并行，
            批内并发用 ThreadPoolExecutor，批间顺序（避免线程开销爆炸）。

        Returns
        -------
        List[CounterfactualScenario]
            与输入 scenarios 等长；每条场景结果按 scenarios 顺序返回。
            任何单条场景异常 → 该条 scenario.timeout=True 或 do_operations
            含 exception 标记；不阻断其它场景（spec §6 失败不静默改数据）。
        """
        if not scenarios:
            return []

        # 单条 runner（沿用 T12 校验路径：disease_code/outcome/data）
        single_runner = CounterfactualRunner(
            disease_code=self.disease_code,
            outcome=self.outcome,
            data=self.data,
        )

        # 预构造结果列表（按 scenarios 顺序填充），先用 timeout 占位，
        # 总超时门耗尽时未跑场景保留 timeout=True。
        results: List[CounterfactualScenario] = [
            self._build_batch_timeout_placeholder(single_runner, sc)
            for sc in scenarios
        ]

        # 总超时门：30s 内必须全部返回
        total_deadline = time.monotonic() + BATCH_TOTAL_TIMEOUT_SECONDS

        # 分批：每批 batch_size 条；批内并发
        for batch_start in range(0, len(scenarios), self.batch_size):
            batch = list(scenarios[batch_start: batch_start + self.batch_size])
            remaining_budget = max(0.0, total_deadline - time.monotonic())
            if remaining_budget <= 0:
                LOGGER.warning(
                    "BatchCounterfactualRunner 总超时（%ss）已耗尽；剩余 %d 条全部标 timeout",
                    BATCH_TOTAL_TIMEOUT_SECONDS,
                    len(batch),
                )
                # 全部 placeholder 已 timeout；跳出循环
                break

            self._run_batch(
                runner=single_runner,
                batch=batch,
                offset=batch_start,
                results=results,
                per_batch_timeout=remaining_budget,
            )

        return results

    # ------------------------------------------------------------------
    # 内部：单批并发
    # ------------------------------------------------------------------

    def _run_batch(
        self,
        *,
        runner: CounterfactualRunner,
        batch: List[Intervention],
        offset: int,
        results: List[CounterfactualScenario],
        per_batch_timeout: float,
    ) -> None:
        """单批并发跑：ThreadPoolExecutor 调度；异常场景替换为 exception 占位。"""
        max_workers = min(self.batch_size, _BATCH_THREAD_POOL_MAX_WORKERS)
        with ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="batch-cf-"
        ) as executor:
            future_to_idx = {
                executor.submit(self._safe_run_single, runner, sc): idx
                for idx, sc in enumerate(batch)
            }
            for future in as_completed(future_to_idx, timeout=per_batch_timeout):
                idx = future_to_idx[future]
                try:
                    scenario = future.result(timeout=COUNTERFACTUAL_TIMEOUT_SECONDS)
                except Exception as exc:  # noqa: BLE001 — 线程层兜底
                    LOGGER.warning(
                        "BatchCounterfactualRunner 线程层异常（idx=%d intervention=%s）: %s",
                        offset + idx,
                        batch[idx],
                        exc,
                    )
                    scenario = self._build_exception_placeholder(
                        runner, batch[idx], exc
                    )
                results[offset + idx] = scenario

    @staticmethod
    def _safe_run_single(
        runner: CounterfactualRunner,
        sc: Intervention,
    ) -> CounterfactualScenario:
        """线程任务：跑单条 CounterfactualRunner.run；异常兜底不抛。"""
        try:
            return runner.run(
                intervention_node=sc.intervention_node,
                hypothetical_value=sc.hypothetical_value,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "BatchCounterfactualRunner 单条异常: intervention=%s exc=%s",
                sc,
                exc,
            )
            return BatchCounterfactualRunner._build_exception_placeholder_static(
                runner, sc, exc
            )

    # ------------------------------------------------------------------
    # 占位 scenario 构造
    # ------------------------------------------------------------------

    @staticmethod
    def _build_batch_timeout_placeholder(
        runner: CounterfactualRunner,
        sc: Intervention,
    ) -> CounterfactualScenario:
        """构造 batch 层 timeout 占位 scenario（scenario_id 走 _build_scenario_id）。

        区别于 CounterfactualRunner.run() 内部的单条 timeout（5s），
        此处是 batch 层总超时（30s）触发后未跑场景的兜底。
        """
        return CounterfactualScenario(
            scenario_id=_build_scenario_id(
                runner.disease_code, sc.intervention_node, sc.hypothetical_value
            ),
            disease_code=runner.disease_code,
            outcome=runner.outcome,
            intervention_node=sc.intervention_node,
            hypothetical_value=sc.hypothetical_value,
            baseline_value=None,
            predicted_outcome=None,
            confidence=None,
            do_operations=[f"batch_timeout:{BATCH_TOTAL_TIMEOUT_SECONDS}s"],
            elapsed_ms=0,
            timeout=True,
            baseline_std=None,
        )

    @staticmethod
    def _build_exception_placeholder(
        runner: CounterfactualRunner,
        sc: Intervention,
        exc: Exception,
    ) -> CounterfactualScenario:
        """构造 exception 占位 scenario（线程层 / 单条异常兜底）。"""
        return BatchCounterfactualRunner._build_exception_placeholder_static(
            runner, sc, exc
        )

    @staticmethod
    def _build_exception_placeholder_static(
        runner: CounterfactualRunner,
        sc: Intervention,
        exc: Exception,
    ) -> CounterfactualScenario:
        return CounterfactualScenario(
            scenario_id=_build_scenario_id(
                runner.disease_code, sc.intervention_node, sc.hypothetical_value
            ),
            disease_code=runner.disease_code,
            outcome=runner.outcome,
            intervention_node=sc.intervention_node,
            hypothetical_value=sc.hypothetical_value,
            baseline_value=None,
            predicted_outcome=None,
            confidence=None,
            do_operations=[f"exception:{type(exc).__name__}"],
            elapsed_ms=0,
            timeout=False,
            baseline_std=None,
        )


# ----------------------------------------------------------------------------------
# 便捷函数
# ----------------------------------------------------------------------------------


def run_batch_counterfactual(
    disease_code: str,
    outcome: str,
    data: pd.DataFrame,
    scenarios: Sequence[Intervention],
    *,
    batch_size: int = BATCH_SIZE_DEFAULT,
) -> List[CounterfactualScenario]:
    """批量跑反事实（BatchCounterfactualRunner 的最简封装）。

    等价于::

        runner = BatchCounterfactualRunner(disease_code, outcome, data, batch_size=batch_size)
        return runner.run(scenarios)
    """
    return BatchCounterfactualRunner(
        disease_code=disease_code,
        outcome=outcome,
        data=data,
        batch_size=batch_size,
    ).run(scenarios)
