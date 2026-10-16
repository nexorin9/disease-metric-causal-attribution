"""test_dosampler — 验证 CounterfactualRunner 反事实接入（spec §11 第 4 步）。

覆盖维度（spec §11 反事实模拟 + §9 因果推理管线）：
  1. 构造期校验：未知 disease_code / outcome / 空 data
  2. 节点合法性：未知节点 / outcome 节点 / 病种非法取值
  3. 反事实预测：3 病种 × treatment/age_band/admission_route 节点 × cmi outcome
  4. 5s 超时：注入 sleep 强制超时 → scenario.timeout=True + predicted_outcome=None
  5. predicted_outcome 在 baseline ± 3σ 范围内（合理范围断言）
  6. do_operations 字段完整（含 do() 操作字符串 + 拟合说明）
  7. scenario_id 稳定：同一干预组合多次调用得到同一 id
  8. 便捷函数 run_counterfactual 等价
"""
from __future__ import annotations

import time

import pandas as pd
import pytest

from app.core.dosampler import (
    ADMISSION_ROUTE_NODE_PREFIX,
    AGE_BAND_NODE_PREFIX,
    COMPLICATION_COUNT_NODE_PREFIX,
    COUNTERFACTUAL_TIMEOUT_SECONDS,
    TREATMENT_NODE_PREFIX,
    CounterfactualRunner,
    _coerce_hypothetical_value,
    _extract_node_value,
    _fit_predict_linear,
    _is_valid_intervention_node,
    _map_node_to_data_column,
    run_counterfactual,
)
from app.core.node_schema import CounterfactualScenario


# ----------------------------------------------------------------------------------
# 公共 helper
# ----------------------------------------------------------------------------------


def _prepare(df):
    """在 fixture 上加 complication_count 列（spec §3.1 secondary_diagnosis_list 派生）。

    反事实拟合需要 complication_count 作为协变量之一。
    """
    df = df.copy()
    df["complication_count"] = df["secondary_diagnosis_list"].fillna("").apply(
        lambda v: 0 if not v or v == "[]" else min(len(str(v).split("|")), 2)
    )
    return df


# ----------------------------------------------------------------------------------
# 1. 构造期校验
# ----------------------------------------------------------------------------------


def test_runner_unknown_disease_code(fixture_for_disease):
    """未知 disease_code → ValueError。"""
    df = _prepare(fixture_for_disease("I21"))
    with pytest.raises(ValueError, match="未知 disease_code"):
        CounterfactualRunner(disease_code="I99", outcome="cmi", data=df)


def test_runner_unknown_outcome(fixture_for_disease):
    """未知 outcome → ValueError。"""
    df = _prepare(fixture_for_disease("I21"))
    with pytest.raises(ValueError, match="未知 outcome"):
        CounterfactualRunner(disease_code="I21", outcome="mortality", data=df)


def test_runner_empty_data_raises():
    """空 DataFrame → ValueError。"""
    with pytest.raises(ValueError, match="data 不能为空"):
        CounterfactualRunner(disease_code="I21", outcome="cmi", data=pd.DataFrame())


# ----------------------------------------------------------------------------------
# 2. 节点合法性校验
# ----------------------------------------------------------------------------------


def test_is_valid_intervention_node_true_for_known_prefixes():
    """已知前缀节点 → True。"""
    assert _is_valid_intervention_node("treatment:PCI")
    assert _is_valid_intervention_node("age_band:3")
    assert _is_valid_intervention_node("admission_route:1")
    assert _is_valid_intervention_node("complication_count:2")


def test_is_valid_intervention_node_false_for_outcome_nodes():
    """outcome 节点（cmi/los_days/...）→ False（不能作为干预目标）。"""
    assert not _is_valid_intervention_node("cmi")
    assert not _is_valid_intervention_node("los_days")
    assert not _is_valid_intervention_node("readmit_flag")
    assert not _is_valid_intervention_node("complication_flag")


def test_is_valid_intervention_node_false_for_empty_or_unknown():
    """空字符串 / 未知节点 → False。"""
    assert not _is_valid_intervention_node("")
    assert not _is_valid_intervention_node("unknown_node")
    assert not _is_valid_intervention_node("BMI:30")  # 不在业务 DAG


def test_extract_node_value():
    """节点 id → 业务取值字符串。"""
    assert _extract_node_value("treatment:PCI") == "PCI"
    assert _extract_node_value("age_band:3") == "3"
    assert _extract_node_value("admission_route:2") == "2"
    assert _extract_node_value("complication_count:1") == "1"


def test_map_node_to_data_column():
    """节点 id → fixture 列名。"""
    assert _map_node_to_data_column("treatment:PCI") == "treatment_strategy"
    assert _map_node_to_data_column("age_band:3") == "age_band"
    assert _map_node_to_data_column("admission_route:1") == "admission_route"
    assert _map_node_to_data_column("complication_count:2") == "complication_count"


def test_coerce_hypothetical_value_type_normalization():
    """hypothetical_value → 与 fixture 字段类型一致。"""
    assert _coerce_hypothetical_value("age_band:3", "3") == 3
    assert _coerce_hypothetical_value("age_band:3", 3) == 3
    assert _coerce_hypothetical_value("admission_route:1", "1") == "1"
    assert _coerce_hypothetical_value("treatment:PCI", "PCI") == "PCI"
    assert _coerce_hypothetical_value("complication_count:2", 2) == 2


# ----------------------------------------------------------------------------------
# 3. 反事实预测：3 病种 × 多种节点 × cmi outcome
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("disease_code", ["I21", "I63", "Z96.10"])
def test_run_treatment_intervention_on_fixtures(fixture_for_disease, disease_code):
    """3 病种 × treatment 节点反事实预测 < 5s 完成。

    spec §11：选定业务 DAG 治疗节点 → 输入假设值 → 5s 内返回 predicted_outcome。
    """
    df = _prepare(fixture_for_disease(disease_code))
    runner = CounterfactualRunner(disease_code=disease_code, outcome="cmi", data=df)

    # 选 disease_code 的第一种治疗方式作为干预目标
    if disease_code == "I21":
        node, value = "treatment:PCI", "PCI"
    elif disease_code == "I63":
        node, value = "treatment:thrombolysis", "thrombolysis"
    else:
        node, value = "treatment:total_hip_replacement", "total_hip_replacement"

    start = time.monotonic()
    scenario = runner.run(intervention_node=node, hypothetical_value=value)
    elapsed = time.monotonic() - start

    # 1. 字段完整性
    assert isinstance(scenario, CounterfactualScenario)
    assert scenario.disease_code == disease_code
    assert scenario.outcome == "cmi"
    assert scenario.intervention_node == node
    assert scenario.hypothetical_value == value
    assert scenario.scenario_id.startswith("scn_")
    assert scenario.timeout is False
    assert scenario.elapsed_ms >= 0
    assert scenario.elapsed_ms < COUNTERFACTUAL_TIMEOUT_SECONDS * 1000

    # 2. 数值字段
    assert scenario.predicted_outcome is not None
    assert scenario.confidence is not None
    assert 0.0 <= scenario.confidence <= 1.0
    assert scenario.baseline_std is not None
    assert scenario.baseline_std > 0  # 真实 fixture 有方差

    # 3. do_operations 至少含 do() 操作字符串
    assert len(scenario.do_operations) >= 1
    assert any("do(" in op for op in scenario.do_operations)

    # 4. 5s 内返回（含测试延迟）
    assert elapsed < COUNTERFACTUAL_TIMEOUT_SECONDS


@pytest.mark.parametrize(
    "node,value",
    [
        ("age_band:3", 3),
        ("age_band:4", 4),
        ("admission_route:1", "1"),
        ("admission_route:2", "2"),
        ("complication_count:0", 0),
        ("complication_count:2", 2),
    ],
)
def test_run_covariate_intervention_on_i21(fixture_for_disease, node, value):
    """协变量节点（age_band / admission_route / complication_count）反事实预测。"""
    df = _prepare(fixture_for_disease("I21"))
    runner = CounterfactualRunner(disease_code="I21", outcome="cmi", data=df)

    scenario = runner.run(intervention_node=node, hypothetical_value=value)

    assert scenario.timeout is False
    assert scenario.predicted_outcome is not None
    assert scenario.confidence is not None
    assert 0.0 <= scenario.confidence <= 1.0
    assert scenario.baseline_value is not None  # baseline = 数据众数


@pytest.mark.parametrize("outcome", ["cmi", "los_days", "readmit_flag", "complication_flag"])
def test_run_on_each_outcome(fixture_for_disease, outcome):
    """4 outcome 都可跑（spec §11：cmi / los_days / readmit_flag / complication_flag）。"""
    df = _prepare(fixture_for_disease("I21"))
    runner = CounterfactualRunner(disease_code="I21", outcome=outcome, data=df)

    scenario = runner.run(intervention_node="treatment:PCI", hypothetical_value="PCI")

    assert scenario.outcome == outcome
    assert scenario.timeout is False
    assert scenario.predicted_outcome is not None


# ----------------------------------------------------------------------------------
# 4. 5s 超时（mock _fit_predict_lin 强制 sleep > 5s）
# ----------------------------------------------------------------------------------


def test_run_timeout_marker(monkeypatch, fixture_for_disease):
    """超时场景：scenario.timeout=True + predicted_outcome=None。

    monkeypatch _fit_predict_linear 强制 sleep 6s，验证 CounterfactualRunner
    在 5s 超时后正常返回 scenario + timeout=True（不抛异常、不阻塞业务层）。
    """
    import app.core.dosampler as ds_mod

    def _slow(*args, **kwargs):
        time.sleep(6.0)  # > COUNTERFACTUAL_TIMEOUT_SECONDS
        return {"predicted_outcome": 0.0, "baseline_value": 0.0, "baseline_std": 0.0,
                "confidence": 0.0, "r_squared": 0.0, "intercept": 0.0, "coef_map": {}}

    monkeypatch.setattr(ds_mod, "_fit_predict_linear", _slow)

    df = _prepare(fixture_for_disease("I21"))
    runner = CounterfactualRunner(disease_code="I21", outcome="cmi", data=df)

    start = time.monotonic()
    scenario = runner.run(intervention_node="treatment:PCI", hypothetical_value="PCI")
    elapsed = time.monotonic() - start

    # 5s 超时即返回；不阻塞（elapsed 应 ≤ 6s — 多 1s 容忍 future.result 的内部调度）
    assert elapsed < 6.5
    assert scenario.timeout is True
    assert scenario.predicted_outcome is None
    assert scenario.confidence is None
    # do_operations 含 timeout 标记
    assert any("timeout" in op for op in scenario.do_operations)


# ----------------------------------------------------------------------------------
# 5. predicted_outcome 合理范围断言
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("disease_code", ["I21", "I63", "Z96.10"])
def test_predicted_outcome_within_baseline_range(fixture_for_disease, disease_code):
    """predicted_outcome 在 baseline ± 3σ 范围内（合理反事实预测）。"""
    df = _prepare(fixture_for_disease(disease_code))
    runner = CounterfactualRunner(disease_code=disease_code, outcome="cmi", data=df)

    if disease_code == "I21":
        node, value = "treatment:PCI", "PCI"
    elif disease_code == "I63":
        node, value = "treatment:thrombolysis", "thrombolysis"
    else:
        node, value = "treatment:total_hip_replacement", "total_hip_replacement"

    scenario = runner.run(intervention_node=node, hypothetical_value=value)

    assert scenario.predicted_outcome is not None
    assert scenario.baseline_std is not None
    # outcome 基线均值 ± 3σ 覆盖预测（线性拟合 + 中位数协变量代入结果应在此范围内）
    baseline_mean = df["cmi"].mean()
    lower = baseline_mean - 3 * scenario.baseline_std
    upper = baseline_mean + 3 * scenario.baseline_std
    assert lower <= scenario.predicted_outcome <= upper, (
        f"predicted_outcome={scenario.predicted_outcome:.4f} "
        f"不在 [{lower:.4f}, {upper:.4f}]"
    )


# ----------------------------------------------------------------------------------
# 6. do_operations 字段完整性
# ----------------------------------------------------------------------------------


def test_do_operations_contains_do_string(fixture_for_disease):
    """do_operations 第一项必含 do(<column> = <value>) 字符串。"""
    df = _prepare(fixture_for_disease("I21"))
    runner = CounterfactualRunner(disease_code="I21", outcome="cmi", data=df)

    scenario = runner.run(intervention_node="treatment:PCI", hypothetical_value="PCI")

    assert len(scenario.do_operations) >= 1
    first_op = scenario.do_operations[0]
    # do(treatment_strategy = 'PCI') 格式
    assert first_op.startswith("do(")
    assert "treatment_strategy" in first_op
    assert "PCI" in first_op


def test_do_operations_includes_fit_description(fixture_for_disease):
    """do_operations 第二项含拟合方法描述。"""
    df = _prepare(fixture_for_disease("I21"))
    runner = CounterfactualRunner(disease_code="I21", outcome="cmi", data=df)

    scenario = runner.run(intervention_node="age_band:3", hypothetical_value=3)

    # 至少含 "fit:" 描述项
    fit_ops = [op for op in scenario.do_operations if op.startswith("fit:")]
    assert len(fit_ops) >= 1
    assert "linear_regression" in fit_ops[0]


# ----------------------------------------------------------------------------------
# 7. scenario_id 稳定
# ----------------------------------------------------------------------------------


def test_scenario_id_stable_for_same_intervention(fixture_for_disease):
    """同一 (disease, node, value) 多次调用得到同一 id（面板 SSE 去重）。"""
    df = _prepare(fixture_for_disease("I21"))
    runner = CounterfactualRunner(disease_code="I21", outcome="cmi", data=df)

    s1 = runner.run(intervention_node="treatment:PCI", hypothetical_value="PCI")
    s2 = runner.run(intervention_node="treatment:PCI", hypothetical_value="PCI")

    assert s1.scenario_id == s2.scenario_id


def test_scenario_id_differs_for_different_intervention(fixture_for_disease):
    """不同 (node, value) → 不同 scenario_id。"""
    df = _prepare(fixture_for_disease("I21"))
    runner = CounterfactualRunner(disease_code="I21", outcome="cmi", data=df)

    s1 = runner.run(intervention_node="treatment:PCI", hypothetical_value="PCI")
    s2 = runner.run(intervention_node="treatment:thrombolysis", hypothetical_value="thrombolysis")

    assert s1.scenario_id != s2.scenario_id


# ----------------------------------------------------------------------------------
# 8. 便捷函数 run_counterfactual
# ----------------------------------------------------------------------------------


def test_run_counterfactual_helper(fixture_for_disease):
    """run_counterfactual 等价于 CounterfactualRunner.run()。"""
    df = _prepare(fixture_for_disease("I21"))

    s_helper = run_counterfactual(
        disease_code="I21",
        outcome="cmi",
        data=df,
        intervention_node="treatment:PCI",
        hypothetical_value="PCI",
    )

    runner = CounterfactualRunner(disease_code="I21", outcome="cmi", data=df)
    s_runner = runner.run(intervention_node="treatment:PCI", hypothetical_value="PCI")

    assert s_helper.scenario_id == s_runner.scenario_id
    assert s_helper.predicted_outcome == s_runner.predicted_outcome


# ----------------------------------------------------------------------------------
# 9. 非法干预节点的降级行为
# ----------------------------------------------------------------------------------


def test_unknown_intervention_node_returns_validation_error(fixture_for_disease):
    """未知干预节点 → scenario 含 validation_error 标记 + 字段为 None（不抛异常）。"""
    df = _prepare(fixture_for_disease("I21"))
    runner = CounterfactualRunner(disease_code="I21", outcome="cmi", data=df)

    # 不抛异常，直接返回 scenario
    scenario = runner.run(intervention_node="BMI:30", hypothetical_value=30)

    assert isinstance(scenario, CounterfactualScenario)
    assert scenario.predicted_outcome is None
    assert scenario.confidence is None
    assert scenario.timeout is False
    assert any("validation_error" in op for op in scenario.do_operations)


def test_outcome_node_not_allowed_as_intervention(fixture_for_disease):
    """outcome 节点（cmi/los_days/...）不允许作为干预目标。"""
    df = _prepare(fixture_for_disease("I21"))
    runner = CounterfactualRunner(disease_code="I21", outcome="cmi", data=df)

    # 把 cmi 当干预节点 → validation_error
    scenario = runner.run(intervention_node="cmi", hypothetical_value=5.0)

    assert scenario.predicted_outcome is None
    assert any("validation_error" in op for op in scenario.do_operations)


def test_invalid_treatment_value_for_disease(fixture_for_disease):
    """治疗节点取值不在该病种合法集合 → validation_error。"""
    df = _prepare(fixture_for_disease("I21"))
    runner = CounterfactualRunner(disease_code="I21", outcome="cmi", data=df)

    # I21 合法治疗：PCI / thrombolysis / conservative；"foo" 不合法
    scenario = runner.run(intervention_node="treatment:foo", hypothetical_value="foo")

    assert scenario.predicted_outcome is None
    assert any("validation_error" in op for op in scenario.do_operations)


# ----------------------------------------------------------------------------------
# 10. _fit_predict_linear 直接调用
# ----------------------------------------------------------------------------------


def test_fit_predict_linear_basic(fixture_for_disease):
    """_fit_predict_linear 直接调用：返回完整 dict。"""
    df = _prepare(fixture_for_disease("I21"))
    result = _fit_predict_linear(
        data=df,
        outcome="cmi",
        intervention_node="treatment:PCI",
        hypothetical_value="PCI",
    )

    assert "predicted_outcome" in result
    assert "baseline_value" in result
    assert "baseline_std" in result
    assert "confidence" in result
    assert "r_squared" in result
    assert "intercept" in result
    assert "coef_map" in result
    assert isinstance(result["coef_map"], dict)
    assert "treatment_strategy" in result["coef_map"]


def test_fit_predict_linear_insufficient_samples():
    """样本不足 → ValueError。"""
    df = pd.DataFrame({
        "treatment_strategy": ["PCI", "thrombolysis", "conservative", "PCI"],
        "age_band": [1, 2, 3, 4],
        "admission_route": [1, 2, 3, 9],
        "complication_count": [0, 1, 2, 0],
        "cmi": [1.0, 2.0, 3.0, 4.0],
    })
    with pytest.raises(ValueError, match="有效样本不足"):
        _fit_predict_linear(
            data=df,
            outcome="cmi",
            intervention_node="treatment:PCI",
            hypothetical_value="PCI",
        )


# ----------------------------------------------------------------------------------
# 11. Schema 边界
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


def test_counterfactual_scenario_id_required():
    """scenario_id 必填（spec §11）。"""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CounterfactualScenario(
            # 缺 scenario_id
            disease_code="I21",
            outcome="cmi",
            intervention_node="treatment:PCI",
            hypothetical_value="PCI",
        )