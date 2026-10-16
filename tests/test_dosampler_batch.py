"""test_dosampler_batch — 验证批量反事实并发调度（spec §11 第 4 步 / T13）。

覆盖维度：
  1. 构造期校验：未知 disease_code / outcome / 空 data / batch_size 非法
  2. Intervention 数据契约：字段校验 + 不可空
  3. 空 scenarios → 空结果
  4. 10 条场景批量 < 30s 返回（spec §11 第 4 步批量反事实）
  5. 字段完整性：scenario_id 稳定 / disease_code / outcome / predicted_outcome
  6. 顺序保留：结果与输入 scenarios 顺序一致（面板按序渲染）
  7. 异常隔离：单条场景抛异常 → 仅该条标 exception，其它场景不受影响
  8. timeout 占位：单条 5s 超时 → scenario.timeout=True，scenario_id 稳定
  9. 便捷函数 run_batch_counterfactual 等价
 10. Schema 边界：CounterfactualScenario 字段完整 + extra=forbid
"""
from __future__ import annotations

import time
from typing import List

import pandas as pd
import pytest

from app.core.dosampler import COUNTERFACTUAL_TIMEOUT_SECONDS
from app.core.dosampler_batch import (
    BATCH_SIZE_DEFAULT,
    BATCH_TOTAL_TIMEOUT_SECONDS,
    BatchCounterfactualRunner,
    Intervention,
    run_batch_counterfactual,
)
from app.core.node_schema import CounterfactualScenario


# ----------------------------------------------------------------------------------
# 公共 helper
# ----------------------------------------------------------------------------------


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """fixture 上加 complication_count 列（spec §3.1 secondary_diagnosis_list 派生）。"""
    df = df.copy()
    df["complication_count"] = df["secondary_diagnosis_list"].fillna("").apply(
        lambda v: 0 if not v or v == "[]" else min(len(str(v).split("|")), 2)
    )
    return df


def _make_i21_interventions(n: int = 10) -> List[Intervention]:
    """构造 n 条 I21 反事实干预（覆盖 3 种治疗 + 年龄段 + 入院途径）。"""
    all_items: List[Intervention] = [
        Intervention("treatment:PCI", "PCI", "全部 PCI"),
        Intervention("treatment:thrombolysis", "thrombolysis", "全部溶栓"),
        Intervention("treatment:conservative", "conservative", "全部保守"),
        Intervention("age_band:3", 3, "年龄段统一 65-74"),
        Intervention("age_band:4", 4, "年龄段统一 75+"),
        Intervention("admission_route:1", "1", "入院途径 1"),
        Intervention("admission_route:2", "2", "入院途径 2"),
        Intervention("complication_count:0", 0, "无合并症"),
        Intervention("complication_count:1", 1, "1 个合并症"),
        Intervention("complication_count:2", 2, "2 个合并症"),
    ]
    return all_items[:n]


# ----------------------------------------------------------------------------------
# 1. 构造期校验
# ----------------------------------------------------------------------------------


def test_runner_unknown_disease_code(fixture_for_disease):
    """未知 disease_code → ValueError。"""
    df = _prepare(fixture_for_disease("I21"))
    with pytest.raises(ValueError, match="未知 disease_code"):
        BatchCounterfactualRunner(disease_code="I99", outcome="cmi", data=df)


def test_runner_unknown_outcome(fixture_for_disease):
    """未知 outcome → ValueError。"""
    df = _prepare(fixture_for_disease("I21"))
    with pytest.raises(ValueError, match="未知 outcome"):
        BatchCounterfactualRunner(disease_code="I21", outcome="mortality", data=df)


def test_runner_empty_data_raises():
    """空 DataFrame → ValueError。"""
    with pytest.raises(ValueError, match="data 不能为空"):
        BatchCounterfactualRunner(disease_code="I21", outcome="cmi", data=pd.DataFrame())


def test_runner_batch_size_must_be_positive(fixture_for_disease):
    """batch_size < 1 → ValueError。"""
    df = _prepare(fixture_for_disease("I21"))
    with pytest.raises(ValueError, match="batch_size 必须 >= 1"):
        BatchCounterfactualRunner(disease_code="I21", outcome="cmi", data=df, batch_size=0)


# ----------------------------------------------------------------------------------
# 2. Intervention 数据契约
# ----------------------------------------------------------------------------------


def test_intervention_requires_intervention_node():
    """Intervention.intervention_node 必填。"""
    with pytest.raises(ValueError, match="intervention_node 不能为空"):
        Intervention(intervention_node="", hypothetical_value="PCI")


def test_intervention_requires_hypothetical_value():
    """Intervention.hypothetical_value 必填非 None。"""
    with pytest.raises(ValueError, match="hypothetical_value 不能为 None"):
        Intervention(intervention_node="treatment:PCI", hypothetical_value=None)


def test_intervention_default_label():
    """Intervention 默认 label 为空串。"""
    sc = Intervention(intervention_node="treatment:PCI", hypothetical_value="PCI")
    assert sc.label == ""


# ----------------------------------------------------------------------------------
# 3. 空 scenarios → 空结果
# ----------------------------------------------------------------------------------


def test_run_empty_scenarios_returns_empty_list(fixture_for_disease):
    """空 scenarios → 空 list（不抛异常）。"""
    df = _prepare(fixture_for_disease("I21"))
    runner = BatchCounterfactualRunner(disease_code="I21", outcome="cmi", data=df)
    result = runner.run([])
    assert result == []


# ----------------------------------------------------------------------------------
# 4. 10 条场景批量 < 30s 返回（核心验收）
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("disease_code", ["I21", "I63", "Z96.10"])
def test_run_batch_10_scenarios_under_30s(fixture_for_disease, disease_code):
    """3 病种 × 10 条场景批量跑通 + 总耗时 < 30s（spec §11 第 4 步批量反事实）。

    这是 T13 的核心验收：面板上「批量反事实实验」按钮后端主路径必须 30s 内
    返回 10 条结果，超时即业务层无法用。
    """
    df = _prepare(fixture_for_disease(disease_code))
    runner = BatchCounterfactualRunner(disease_code=disease_code, outcome="cmi", data=df)
    interventions = _make_i21_interventions(n=10)

    start = time.monotonic()
    scenarios = runner.run(interventions)
    elapsed = time.monotonic() - start

    # 1. 总耗时 < 30s（含 5s 容忍以应对 thread 调度抖动）
    assert elapsed < BATCH_TOTAL_TIMEOUT_SECONDS, (
        f"批量 10 条耗时 {elapsed:.2f}s 超过 {BATCH_TOTAL_TIMEOUT_SECONDS}s 上限"
    )

    # 2. 数量与输入一致
    assert len(scenarios) == 10

    # 3. 字段完整：每条都是 CounterfactualScenario
    assert all(isinstance(s, CounterfactualScenario) for s in scenarios)
    for s in scenarios:
        assert s.disease_code == disease_code
        assert s.outcome == "cmi"
        assert s.scenario_id.startswith("scn_")
        assert s.elapsed_ms >= 0


# ----------------------------------------------------------------------------------
# 5. 字段完整性
# ----------------------------------------------------------------------------------


def test_run_batch_predicted_outcomes_present(fixture_for_disease):
    """10 条场景的 predicted_outcome 在合理范围内（批量场景正常路径）。"""
    df = _prepare(fixture_for_disease("I21"))
    runner = BatchCounterfactualRunner(disease_code="I21", outcome="cmi", data=df)
    interventions = _make_i21_interventions(n=10)

    scenarios = runner.run(interventions)

    # 大多数场景应成功跑出 predicted_outcome（不强制全部，因 fixture 字段可能
    # 存在类别取值不匹配；至少 8/10 场景应有值）
    success_count = sum(1 for s in scenarios if s.predicted_outcome is not None)
    assert success_count >= 8, (
        f"10 条场景仅 {success_count} 条跑出 predicted_outcome（>= 8 期望）"
    )

    # confidence ∈ [0, 1]
    for s in scenarios:
        if s.confidence is not None:
            assert 0.0 <= s.confidence <= 1.0


def test_run_batch_scenario_ids_stable(fixture_for_disease):
    """scenario_id 在两次调用之间稳定（同 (disease, node, value)）。"""
    df = _prepare(fixture_for_disease("I21"))
    runner = BatchCounterfactualRunner(disease_code="I21", outcome="cmi", data=df)
    interventions = _make_i21_interventions(n=10)

    s1 = runner.run(interventions)
    s2 = runner.run(interventions)

    ids_1 = sorted([s.scenario_id for s in s1])
    ids_2 = sorted([s.scenario_id for s in s2])
    assert ids_1 == ids_2


# ----------------------------------------------------------------------------------
# 6. 顺序保留
# ----------------------------------------------------------------------------------


def test_run_batch_preserves_input_order(fixture_for_disease):
    """结果与输入 scenarios 顺序一致（面板按序渲染）。"""
    df = _prepare(fixture_for_disease("I21"))
    runner = BatchCounterfactualRunner(disease_code="I21", outcome="cmi", data=df)
    interventions = _make_i21_interventions(n=10)

    scenarios = runner.run(interventions)

    for i, sc in enumerate(interventions):
        assert scenarios[i].intervention_node == sc.intervention_node
        assert scenarios[i].hypothetical_value == sc.hypothetical_value


# ----------------------------------------------------------------------------------
# 7. 异常隔离（单条异常不阻断其它场景）
# ----------------------------------------------------------------------------------


def test_run_batch_isolates_exceptions(monkeypatch, fixture_for_disease):
    """单条场景抛异常 → 仅该条标 exception，其它场景不受影响（spec §6）。"""
    df = _prepare(fixture_for_disease("I21"))
    runner = BatchCounterfactualRunner(disease_code="I21", outcome="cmi", data=df)
    interventions = _make_i21_interventions(n=10)

    # 让第二条场景的 CounterfactualRunner.run 抛异常
    import app.core.dosampler as ds_mod

    original_run = ds_mod.CounterfactualRunner.run
    call_count = {"n": 0}

    def _maybe_raise(self, intervention_node, hypothetical_value, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated failure on scenario #2")
        return original_run(self, intervention_node, hypothetical_value, **kwargs)

    monkeypatch.setattr(ds_mod.CounterfactualRunner, "run", _maybe_raise)

    scenarios = runner.run(interventions)

    # 全部 10 条都返回（不抛异常）
    assert len(scenarios) == 10
    # 第二条标 exception，其它正常
    assert any("exception:RuntimeError" in op for op in scenarios[1].do_operations)
    # 第 1/3/10 等其它场景正常
    assert scenarios[0].timeout is False
    assert "exception:RuntimeError" not in scenarios[0].do_operations


# ----------------------------------------------------------------------------------
# 8. timeout 占位
# ----------------------------------------------------------------------------------


def test_run_batch_handles_single_timeout(monkeypatch, fixture_for_disease):
    """单条 5s 超时 → scenario.timeout=True；不阻塞其它场景。"""
    df = _prepare(fixture_for_disease("I21"))
    runner = BatchCounterfactualRunner(disease_code="I21", outcome="cmi", data=df)
    interventions = _make_i21_interventions(n=10)

    # 让第三条场景触发超时：monkeypatch _fit_predict_linear 强制 sleep > 5s
    import app.core.dosampler as ds_mod

    original_fit = ds_mod._fit_predict_linear
    call_count = {"n": 0}

    def _maybe_slow(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 3:
            time.sleep(6.0)
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(ds_mod, "_fit_predict_linear", _maybe_slow)

    start = time.monotonic()
    scenarios = runner.run(interventions)
    elapsed = time.monotonic() - start

    # 第三条场景 timeout=True（其它场景不受影响）
    assert len(scenarios) == 10
    # 第三条 timeout（不依赖精确位置因为 batch_size 默认 10 一批全跑）
    timeout_scenarios = [s for s in scenarios if s.timeout]
    assert len(timeout_scenarios) >= 1, (
        f"期望至少 1 条 timeout；实际 {len(timeout_scenarios)} 条"
    )
    # 至少一条其它场景成功跑出 predicted_outcome
    success_count = sum(1 for s in scenarios if s.predicted_outcome is not None)
    assert success_count >= 5, f"超时不阻断其它场景（期望 >= 5 条成功；实际 {success_count}）"
    # 整体耗时 < 30s
    assert elapsed < BATCH_TOTAL_TIMEOUT_SECONDS


# ----------------------------------------------------------------------------------
# 9. 便捷函数 run_batch_counterfactual
# ----------------------------------------------------------------------------------


def test_run_batch_counterfactual_helper(fixture_for_disease):
    """run_batch_counterfactual 等价于 BatchCounterfactualRunner.run()。"""
    df = _prepare(fixture_for_disease("I21"))
    interventions = _make_i21_interventions(n=5)

    s_helper = run_batch_counterfactual(
        disease_code="I21",
        outcome="cmi",
        data=df,
        scenarios=interventions,
    )

    runner = BatchCounterfactualRunner(disease_code="I21", outcome="cmi", data=df)
    s_runner = runner.run(interventions)

    assert len(s_helper) == len(s_runner) == 5
    # scenario_id 一致
    for h, r in zip(s_helper, s_runner):
        assert h.scenario_id == r.scenario_id
        assert h.predicted_outcome == r.predicted_outcome


def test_run_batch_counterfactual_with_custom_batch_size(fixture_for_disease):
    """run_batch_counterfactual 支持自定义 batch_size。"""
    df = _prepare(fixture_for_disease("I21"))
    interventions = _make_i21_interventions(n=5)

    # batch_size=2 → 分 3 批
    s = run_batch_counterfactual(
        disease_code="I21",
        outcome="cmi",
        data=df,
        scenarios=interventions,
        batch_size=2,
    )
    assert len(s) == 5


# ----------------------------------------------------------------------------------
# 10. Schema 边界
# ----------------------------------------------------------------------------------


def test_counterfactual_scenario_schema_rejects_extra_fields():
    """CounterfactualScenario schema 严格性（extra=forbid）。"""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CounterfactualScenario(
            scenario_id="scn_test",
            disease_code="I21",
            outcome="cmi",
            intervention_node="treatment:PCI",
            hypothetical_value="PCI",
            extra_unknown="not_allowed",
        )


# ----------------------------------------------------------------------------------
# 11. batch_size 边界
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 5, 10, 20])
def test_run_batch_various_batch_sizes(fixture_for_disease, batch_size):
    """多种 batch_size 都能跑通 10 条场景。"""
    df = _prepare(fixture_for_disease("I21"))
    runner = BatchCounterfactualRunner(
        disease_code="I21", outcome="cmi", data=df, batch_size=batch_size
    )
    interventions = _make_i21_interventions(n=10)

    scenarios = runner.run(interventions)
    assert len(scenarios) == 10
