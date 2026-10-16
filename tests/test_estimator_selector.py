"""test_estimator_selector — 验证估计器切换策略器（spec §11 第 6 步 / T17）。

覆盖维度：
  1. 构造期校验
  2. 决策表：4 触发条件
       a) n<100 → LR
       b) n>=100 + has_iv=True → IV
       c) n>=100 + has_iv=False → PSM
       d) 边界：n=99 (LR) / n=100 (走 IV 或 PSM 分支)
  3. alternatives_considered 字段完整性（含 selected / reason）
  4. override_method（构造期）+ 环境变量 SPEC_OVERRIDE_ESTIMATOR
  5. EstimatorChoice 数据契约（method_name 合法 + rationale 非空）
  6. 便捷函数 select_estimator
  7. 样本量守门一致性：与 sample_size_gate.SMALL_SAMPLE_THRESHOLD 阈值对齐
"""
from __future__ import annotations

import os

import pytest

from app.core.estimator_selector import (
    ESTIMATOR_IV,
    ESTIMATOR_LINEAR_REGRESSION,
    ESTIMATOR_PROPENSITY_MATCHING,
    SMALL_SAMPLE_THRESHOLD,
    EstimatorChoice,
    EstimatorSelector,
    select_estimator,
)
from app.core.sample_size_gate import (
    INSUFFICIENT_THRESHOLD,
    MARGINAL_THRESHOLD,
)


# ----------------------------------------------------------------------------------
# 1. 构造期校验
# ----------------------------------------------------------------------------------


def test_estimator_selector_default_constructs():
    """默认构造器无 override。"""
    sel = EstimatorSelector()
    assert sel.override_method is None


def test_estimator_selector_explicit_override_constructs():
    """显式 override_method 可构造。"""
    sel = EstimatorSelector(override_method=ESTIMATOR_LINEAR_REGRESSION)
    assert sel.override_method == ESTIMATOR_LINEAR_REGRESSION


def test_estimator_choice_known_method_validation():
    """EstimatorChoice 未知 method_name → ValueError。"""
    with pytest.raises(ValueError, match="未知估计器"):
        EstimatorChoice(
            method_name="backdoor.invalid",
            sample_size=100,
            has_iv=False,
            rationale="x",
        )


def test_estimator_choice_empty_rationale_raises():
    """rationale 空 → ValueError。"""
    with pytest.raises(ValueError, match="rationale 不可为空"):
        EstimatorChoice(
            method_name=ESTIMATOR_LINEAR_REGRESSION,
            sample_size=100,
            has_iv=False,
            rationale="",
        )


def test_estimator_choice_negative_sample_size_raises():
    """sample_size < 0 → ValueError。"""
    with pytest.raises(ValueError, match="sample_size 必须 >= 0"):
        EstimatorChoice(
            method_name=ESTIMATOR_LINEAR_REGRESSION,
            sample_size=-1,
            has_iv=False,
            rationale="x",
        )


def test_estimator_choice_has_iv_type_check():
    """has_iv 非 bool → TypeError。"""
    with pytest.raises(TypeError, match="has_iv 必须是 bool"):
        EstimatorChoice(
            method_name=ESTIMATOR_LINEAR_REGRESSION,
            sample_size=100,
            has_iv="yes",
            rationale="x",
        )


# ----------------------------------------------------------------------------------
# 2. 决策表：4 触发条件
# ----------------------------------------------------------------------------------


def test_select_small_sample_chooses_linear_regression():
    """n<100 → backdoor.linear_regression（spec §11 第 6 步）。"""
    sel = EstimatorSelector()
    for n in (10, 30, 50, 80, 99):
        choice = sel.select(sample_size=n, has_iv=False)
        assert choice.method_name == ESTIMATOR_LINEAR_REGRESSION, (
            f"n={n} 应选 LR；got {choice.method_name}"
        )


def test_select_small_sample_with_iv_still_linear_regression():
    """n<100 + has_iv=True → 仍走 LR（IV 在小样本下偏弱）。"""
    sel = EstimatorSelector()
    for n in (30, 80, 99):
        choice = sel.select(sample_size=n, has_iv=True)
        assert choice.method_name == ESTIMATOR_LINEAR_REGRESSION


def test_select_large_sample_with_iv_chooses_iv():
    """n>=100 + has_iv=True → iv.instrumental_variable。"""
    sel = EstimatorSelector()
    for n in (100, 200, 500, 2000):
        choice = sel.select(sample_size=n, has_iv=True)
        assert choice.method_name == ESTIMATOR_IV, (
            f"n={n}, has_iv=True 应选 IV；got {choice.method_name}"
        )


def test_select_large_sample_without_iv_chooses_psm():
    """n>=100 + has_iv=False → backdoor.propensity_score_matching。"""
    sel = EstimatorSelector()
    for n in (100, 200, 500, 2000):
        choice = sel.select(sample_size=n, has_iv=False)
        assert choice.method_name == ESTIMATOR_PROPENSITY_MATCHING, (
            f"n={n}, has_iv=False 应选 PSM；got {choice.method_name}"
        )


def test_select_boundary_n_99_chooses_lr():
    """n=99（<100）→ LR。"""
    sel = EstimatorSelector()
    assert sel.select(sample_size=99, has_iv=False).method_name == ESTIMATOR_LINEAR_REGRESSION
    assert sel.select(sample_size=99, has_iv=True).method_name == ESTIMATOR_LINEAR_REGRESSION


def test_select_boundary_n_100_takes_iv_or_psm_branch():
    """n=100（>=100）→ 走 IV（has_iv=True）或 PSM（has_iv=False）分支。"""
    sel = EstimatorSelector()
    assert sel.select(sample_size=100, has_iv=True).method_name == ESTIMATOR_IV
    assert sel.select(sample_size=100, has_iv=False).method_name == ESTIMATOR_PROPENSITY_MATCHING


# ----------------------------------------------------------------------------------
# 3. alternatives_considered 字段完整性
# ----------------------------------------------------------------------------------


def test_alternatives_considered_has_exactly_one_selected():
    """alternatives_considered 必须恰好 1 个 selected=True。"""
    sel = EstimatorSelector()
    for n, has_iv in [(50, False), (50, True), (200, False), (200, True)]:
        choice = sel.select(sample_size=n, has_iv=has_iv)
        selected_count = sum(
            1 for r in choice.alternatives_considered if r.get("selected")
        )
        assert selected_count == 1, (
            f"n={n}, has_iv={has_iv}：selected 必须恰好 1 个；got {selected_count}"
        )


def test_alternatives_considered_includes_all_three_estimators_small_sample():
    """n<100：alternatives_considered 含 LR/IV/PSM 三档。"""
    sel = EstimatorSelector()
    choice = sel.select(sample_size=50, has_iv=False)
    methods = {r["method_name"] for r in choice.alternatives_considered}
    assert methods == {
        ESTIMATOR_LINEAR_REGRESSION,
        ESTIMATOR_IV,
        ESTIMATOR_PROPENSITY_MATCHING,
    }


def test_alternatives_considered_includes_all_three_estimators_large_sample_with_iv():
    """n>=100 + has_iv=True：alternatives_considered 含 LR/IV/PSM 三档。"""
    sel = EstimatorSelector()
    choice = sel.select(sample_size=500, has_iv=True)
    methods = {r["method_name"] for r in choice.alternatives_considered}
    assert methods == {
        ESTIMATOR_LINEAR_REGRESSION,
        ESTIMATOR_IV,
        ESTIMATOR_PROPENSITY_MATCHING,
    }


def test_alternatives_considered_has_reason_field():
    """每条 alternatives_considered 含 reason 字段且非空。"""
    sel = EstimatorSelector()
    choice = sel.select(sample_size=200, has_iv=False)
    for r in choice.alternatives_considered:
        assert "reason" in r
        assert r["reason"]  # 非空


# ----------------------------------------------------------------------------------
# 4. override_method + 环境变量
# ----------------------------------------------------------------------------------


def test_select_with_explicit_override_returns_override_method():
    """override_method 强制覆盖决策表。"""
    sel = EstimatorSelector(override_method=ESTIMATOR_IV)
    # 即使 n=50（应选 LR），override 后仍返回 IV
    choice = sel.select(sample_size=50, has_iv=False)
    assert choice.method_name == ESTIMATOR_IV
    assert "强制覆盖" in choice.rationale


def test_select_with_unknown_override_raises():
    """override_method 未知 → ValueError。"""
    sel = EstimatorSelector(override_method="backdoor.invalid")
    with pytest.raises(ValueError, match="override_method 未知"):
        sel.select(sample_size=100, has_iv=False)


def test_select_with_env_override(monkeypatch):
    """SPEC_OVERRIDE_ESTIMATOR 环境变量生效。"""
    monkeypatch.setenv("SPEC_OVERRIDE_ESTIMATOR", ESTIMATOR_PROPENSITY_MATCHING)
    sel = EstimatorSelector()
    assert sel.override_method == ESTIMATOR_PROPENSITY_MATCHING
    choice = sel.select(sample_size=50, has_iv=False)
    assert choice.method_name == ESTIMATOR_PROPENSITY_MATCHING


def test_select_without_env_override_does_not_use_env(monkeypatch):
    """无 env 时 override_method=None。"""
    monkeypatch.delenv("SPEC_OVERRIDE_ESTIMATOR", raising=False)
    sel = EstimatorSelector()
    assert sel.override_method is None
    # 走决策表
    assert sel.select(sample_size=50, has_iv=False).method_name == ESTIMATOR_LINEAR_REGRESSION


# ----------------------------------------------------------------------------------
# 5. EstimatorChoice 数据契约
# ----------------------------------------------------------------------------------


def test_estimator_choice_to_dict_includes_all_fields():
    """to_dict 含 5 字段。"""
    sel = EstimatorSelector()
    choice = sel.select(sample_size=200, has_iv=True)
    d = choice.to_dict()
    assert set(d.keys()) >= {
        "method_name",
        "sample_size",
        "has_iv",
        "rationale",
        "alternatives_considered",
    }
    assert d["sample_size"] == 200
    assert d["has_iv"] is True


def test_rationale_contains_sample_size_and_iv_state():
    """rationale 文案含 sample_size + has_iv 关键事实。"""
    sel = EstimatorSelector()
    choice = sel.select(sample_size=300, has_iv=True)
    assert "300" in choice.rationale
    assert "has_iv=True" in choice.rationale or "has_iv" in choice.rationale


def test_rationale_with_outcome_suffix():
    """outcome 参数加入 rationale 文案。"""
    sel = EstimatorSelector()
    choice = sel.select(sample_size=300, has_iv=True, outcome="cmi")
    assert "outcome=cmi" in choice.rationale


# ----------------------------------------------------------------------------------
# 6. 便捷函数 select_estimator
# ----------------------------------------------------------------------------------


def test_select_estimator_helper_small_sample():
    """select_estimator(50, False) == LR。"""
    choice = select_estimator(sample_size=50, has_iv=False)
    assert choice.method_name == ESTIMATOR_LINEAR_REGRESSION


def test_select_estimator_helper_large_sample_with_iv():
    """select_estimator(500, True) == IV。"""
    choice = select_estimator(sample_size=500, has_iv=True)
    assert choice.method_name == ESTIMATOR_IV


def test_select_estimator_helper_with_override():
    """select_estimator override_method 生效。"""
    choice = select_estimator(
        sample_size=50,
        has_iv=False,
        override_method=ESTIMATOR_PROPENSITY_MATCHING,
    )
    assert choice.method_name == ESTIMATOR_PROPENSITY_MATCHING


# ----------------------------------------------------------------------------------
# 7. 与 sample_size_gate 阈值一致性
# ----------------------------------------------------------------------------------


def test_small_sample_threshold_aligned_with_sample_size_gate():
    """SMALL_SAMPLE_THRESHOLD(100) 与 sample_size_gate.MARGINAL_THRESHOLD 一致
    （n=100 是 sufficient 与 marginal 的边界；策略器选 LR 的上界）。"""
    assert SMALL_SAMPLE_THRESHOLD == MARGINAL_THRESHOLD


def test_insufficient_samples_also_choose_lr():
    """n<INSUFFICIENT_THRESHOLD 也走 LR（流水线层 InsufficientSampleError 会先抛）。"""
    sel = EstimatorSelector()
    for n in (1, 10, 29):
        choice = sel.select(sample_size=n, has_iv=False)
        assert choice.method_name == ESTIMATOR_LINEAR_REGRESSION
    # INSUFFICIENT_THRESHOLD（30）仍 < SMALL_SAMPLE_THRESHOLD（100），LR
    assert (
        sel.select(sample_size=INSUFFICIENT_THRESHOLD, has_iv=False).method_name
        == ESTIMATOR_LINEAR_REGRESSION
    )


# ----------------------------------------------------------------------------------
# 8. 样本量校验
# ----------------------------------------------------------------------------------


def test_select_negative_sample_size_raises():
    """select(sample_size=-1) → ValueError。"""
    sel = EstimatorSelector()
    with pytest.raises(ValueError, match="sample_size 必须 >= 0"):
        sel.select(sample_size=-1, has_iv=False)


def test_select_has_iv_type_check():
    """select(has_iv=非 bool) → TypeError。"""
    sel = EstimatorSelector()
    with pytest.raises(TypeError, match="has_iv 必须是 bool"):
        sel.select(sample_size=100, has_iv="yes")  # type: ignore[arg-type]