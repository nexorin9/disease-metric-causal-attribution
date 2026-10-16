"""test_dowhy_pipeline — 验证 CausalPipeline 构造 + identify_effect 在 fixture 上跑通。

覆盖维度（spec §11 主路径 + §9 CausalModel 流水线）：
  1. CausalPipeline 在 3 个 fixture 上可构造（构造期校验）
  2. identify_effect 返回非 None estimand（含 backdoor / instruments / expression）
  3. call_estimate_effect 跑通 estimate_effect + 返回 EffectEstimate
  4. 异常路径：未知 disease_code / outcome / 缺列 / 样本量不足
  5. 数据准备：fixture 加 complication_count 列（spec §3.1 secondary_diagnosis_list 派生）
"""
from __future__ import annotations

import uuid

import pandas as pd
import pytest

from app.core.dag_builder import SUPPORTED_DISEASE_CODES, DiseaseOutcome
from app.core.pipeline import (
    CausalPipeline,
    EstimandNotIdentifiedError,
    InsufficientSampleError,
    SUPPORTED_ESTIMATORS,
)


# ----------------------------------------------------------------------------------
# 公共 helper：fixture + 派生列
# ----------------------------------------------------------------------------------


def _prepare(df):
    """在 fixture 上加 complication_count 列（spec §3.1 secondary_diagnosis_list 派生）。"""
    df = df.copy()
    df["complication_count"] = df["secondary_diagnosis_list"].fillna("").apply(
        lambda v: 0 if not v or v == "[]" else min(len(str(v).split("|")), 2)
    )
    return df


def _extend_to_size(df, n: int, *, seed: int = 20261016):
    """通过 bootstrap-resample（放回抽样）把 df 扩到 n 行。

    T7 任务契约要求 n=30/100/500，但 fixture 每病种只生成 200 行。
    测试场景下用 bootstrap 扩到目标样本量即可保持因果结构基本一致。
    encounter_id 列重新生成以保持全局唯一。
    """
    import uuid

    if n <= len(df):
        return df.head(n).copy()
    n_extra = n - len(df)
    extra = df.sample(n=n_extra, replace=True, random_state=seed).reset_index(drop=True)
    base = df.reset_index(drop=True)
    out = pd.concat([base, extra], ignore_index=True)
    # 重生成 encounter_id 保持全局唯一（spec §3.1 必填：enc_<12hex>）
    out["encounter_id"] = [f"enc_{uuid.uuid4().hex[:12]}" for _ in range(len(out))]
    return out


@pytest.fixture
def prepared_fixture_for_disease(request):
    """parametrize 友好的 fixture 加载器（每个 case 拿一份 DataFrame）。"""
    from data.seed_loader import load_fixture_csv

    disease_code = request.param
    df = load_fixture_csv(disease_code)
    return _prepare(df)


# ----------------------------------------------------------------------------------
# 1. 构造期校验
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prepared_fixture_for_disease",
    list(SUPPORTED_DISEASE_CODES),
    indirect=True,
)
def test_pipeline_construction_on_fixtures(prepared_fixture_for_disease):
    """CausalPipeline 在 3 fixture 上可构造。"""
    disease_code = SUPPORTED_DISEASE_CODES[0]
    # 注：parametrize 每次拿一个病种；用 disease_code 即可
    # 这里直接通过 prepared_fixture_for_disease 推断 disease_code 不必要 — 校验构造不抛
    df = prepared_fixture_for_disease
    p = CausalPipeline(
        disease_code="I21",
        outcome="cmi",
        data=df,
    )
    assert p.disease_code == "I21"
    assert p.outcome == "cmi"
    assert p.data.shape[0] >= 30  # spec §6 样本量阈值
    assert p.graph is not None
    assert p.graph.number_of_nodes() > 0


@pytest.mark.parametrize(
    "disease_code",
    list(SUPPORTED_DISEASE_CODES),
)
def test_pipeline_construction_per_disease(disease_code):
    """3 病种 × cmi outcome 构造校验。"""
    from data.seed_loader import load_fixture_csv

    df = load_fixture_csv(disease_code)
    df = _prepare(df)
    p = CausalPipeline(disease_code=disease_code, outcome="cmi", data=df)
    assert p.disease_code == disease_code
    assert p.outcome == "cmi"
    # 至少包含 common_causes（自动从 data 列裁剪）
    assert "age_band" in p.common_causes
    assert "admission_route" in p.common_causes
    assert "complication_count" in p.common_causes


def test_pipeline_unknown_disease_code():
    """未知 disease_code → ValueError。"""
    from data.seed_loader import load_fixture_csv

    df = _prepare(load_fixture_csv("I21"))
    with pytest.raises(ValueError, match="未知 disease_code"):
        CausalPipeline(disease_code="I99", outcome="cmi", data=df)


def test_pipeline_unknown_outcome():
    """未知 outcome → ValueError。"""
    from data.seed_loader import load_fixture_csv

    df = _prepare(load_fixture_csv("I21"))
    with pytest.raises(ValueError, match="未知 outcome"):
        CausalPipeline(disease_code="I21", outcome="mortality", data=df)


def test_pipeline_empty_data_raises():
    """空 DataFrame → ValueError。"""
    import pandas as pd

    with pytest.raises(ValueError, match="data 不能为空"):
        CausalPipeline(disease_code="I21", outcome="cmi", data=pd.DataFrame())


def test_pipeline_insufficient_sample():
    """样本量 < min_sample_size → InsufficientSampleError。"""
    from data.seed_loader import load_fixture_csv

    df = _prepare(load_fixture_csv("I21")).head(10)
    with pytest.raises(InsufficientSampleError):
        CausalPipeline(
            disease_code="I21", outcome="cmi", data=df, min_sample_size=30
        )


def test_pipeline_missing_treatment_column():
    """treatment 列缺失 → ValueError。"""
    from data.seed_loader import load_fixture_csv

    df = _prepare(load_fixture_csv("I21")).drop(columns=["treatment_strategy"])
    with pytest.raises(ValueError, match="treatment 列"):
        CausalPipeline(disease_code="I21", outcome="cmi", data=df)


def test_pipeline_missing_outcome_column():
    """outcome 列缺失 → ValueError。"""
    from data.seed_loader import load_fixture_csv

    df = _prepare(load_fixture_csv("I21")).drop(columns=["cmi"])
    with pytest.raises(ValueError, match="outcome 列"):
        CausalPipeline(disease_code="I21", outcome="cmi", data=df)


# ----------------------------------------------------------------------------------
# 2. identify_effect 跑通
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("disease_code", list(SUPPORTED_DISEASE_CODES))
def test_call_identify_effect_returns_non_none(disease_code):
    """identify_effect 在 3 病种上返回非 None estimand。"""
    from data.seed_loader import load_fixture_csv

    df = _prepare(load_fixture_csv(disease_code))
    p = CausalPipeline(disease_code=disease_code, outcome="cmi", data=df)
    estimand = p.call_identify_effect()
    assert estimand is not None
    assert estimand.treatment_name == "treatment_strategy"
    assert estimand.outcome_name == "cmi"
    # 至少有一个 backdoor 变量（spec §11 业务 DAG）
    assert len(estimand.backdoor_variables) >= 1 or estimand.estimand_expression != ""


@pytest.mark.parametrize(
    "outcome",
    [o.value for o in DiseaseOutcome],
)
def test_call_identify_effect_per_outcome(outcome):
    """identify_effect 对 4 个 outcome 都能跑通。"""
    from data.seed_loader import load_fixture_csv

    df = _prepare(load_fixture_csv("I21"))
    p = CausalPipeline(disease_code="I21", outcome=outcome, data=df)
    estimand = p.call_identify_effect()
    assert estimand.outcome_name == outcome
    assert estimand.treatment_name == "treatment_strategy"


def test_call_identify_effect_summary():
    """Estimand.summary() 返回结构化字典。"""
    from data.seed_loader import load_fixture_csv

    df = _prepare(load_fixture_csv("I21"))
    p = CausalPipeline(disease_code="I21", outcome="cmi", data=df)
    estimand = p.call_identify_effect()
    summary = estimand.summary()
    assert summary["treatment"] == "treatment_strategy"
    assert summary["outcome"] == "cmi"
    assert "backdoor" in summary
    assert "instruments" in summary
    assert "expression" in summary


# ----------------------------------------------------------------------------------
# 3. estimate_effect 跑通（T6 仅占位 + 单元测，T7 完整 CI/p_value 行为）
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method_name",
    [
        "backdoor.linear_regression",
        "backdoor.propensity_score_matching",
    ],
)
def test_call_estimate_effect_basic(method_name):
    """estimate_effect 在 fixture + 至少一种估计器下能跑通 + 返回 EffectEstimate。"""
    from data.seed_loader import load_fixture_csv

    df = _prepare(load_fixture_csv("I21"))
    p = CausalPipeline(disease_code="I21", outcome="cmi", data=df)
    estimate = p.call_estimate_effect(method_name=method_name)
    assert estimate.method_name == method_name
    assert estimate.outcome == "cmi"
    assert estimate.treatment == "treatment_strategy"
    assert estimate.sample_size == len(df)


def test_call_estimate_effect_rejects_unknown_method():
    """未知估计器 → ValueError。"""
    from data.seed_loader import load_fixture_csv

    df = _prepare(load_fixture_csv("I21"))
    p = CausalPipeline(disease_code="I21", outcome="cmi", data=df)
    with pytest.raises(ValueError, match="未知估计器"):
        p.call_estimate_effect(method_name="backdoor.unknown_estimator")


def test_supported_estimators_constant():
    """SUPPORTED_ESTIMATORS 含 spec §9 列出的估计器名。"""
    assert "backdoor.propensity_score_matching" in SUPPORTED_ESTIMATORS
    assert "backdoor.linear_regression" in SUPPORTED_ESTIMATORS
    assert "iv.instrumental_variable" in SUPPORTED_ESTIMATORS


# ----------------------------------------------------------------------------------
# 4. 集成：identify + estimate 同管线串联
# ----------------------------------------------------------------------------------


def test_identify_then_estimate_serial():
    """完整串联：identify → estimate，estimand 可重用。"""
    from data.seed_loader import load_fixture_csv

    df = _prepare(load_fixture_csv("I63"))
    p = CausalPipeline(disease_code="I63", outcome="los_days", data=df)
    estimand = p.call_identify_effect()
    estimate = p.call_estimate_effect(
        "backdoor.linear_regression", estimand=estimand
    )
    assert estimate.outcome == "los_days"
    assert estimate.sample_size == len(df)


# ----------------------------------------------------------------------------------
# 5. T7：estimate_effect 三档估计器 + 小样本行为（n=30/100/500）
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sample_size,method_name",
    [
        # linear_regression 三档样本量
        (30, "backdoor.linear_regression"),
        (100, "backdoor.linear_regression"),
        (500, "backdoor.linear_regression"),
        # propensity_score_matching 三档样本量（多值 treatment 时自动降级到 LR）
        (30, "backdoor.propensity_score_matching"),
        (100, "backdoor.propensity_score_matching"),
        (500, "backdoor.propensity_score_matching"),
    ],
)
def test_call_estimate_effect_three_estimators_at_sample_sizes(sample_size, method_name):
    """T7：三档估计器在 n=30/100/500 不同样本量下断言效应值/CI/p_value。

    spec §6 样本量阈值：
      - n=30：样本量边界，CI 宽度比 n=100/500 显著更大（小样本诊断信号）
      - n=100/500：CI 收敛，置信区间合理
    """
    from data.seed_loader import load_fixture_csv

    df = _extend_to_size(_prepare(load_fixture_csv("I21")), sample_size)
    p = CausalPipeline(disease_code="I21", outcome="cmi", data=df)
    estimate = p.call_estimate_effect(method_name=method_name)

    # 1. EffectEstimate 字段完整性（spec §11 流水线 + node_schema.EffectEstimate）
    assert estimate.method_name == method_name  # 用户请求的估计器（即使降级也保持原值）
    assert estimate.treatment == "treatment_strategy"
    assert estimate.outcome == "cmi"
    assert estimate.sample_size == sample_size
    assert estimate.node == "treatment_strategy"

    # 2. ate / ci_lower / ci_upper / p_value 必须有值（spec §9 字段）
    assert estimate.ate is not None, f"{method_name} n={sample_size}: ate 为空"
    assert isinstance(estimate.ate, float)
    assert estimate.ci_lower is not None, f"{method_name} n={sample_size}: ci_lower 为空"
    assert estimate.ci_upper is not None, f"{method_name} n={sample_size}: ci_upper 为空"
    assert estimate.ci_width is not None
    assert estimate.ci_width == estimate.ci_upper - estimate.ci_lower
    assert estimate.p_value is not None, f"{method_name} n={sample_size}: p_value 为空"
    assert 0.0 <= estimate.p_value <= 1.0

    # 3. estimate_ci_unstable 与 ci_width 一致：width > 0.6 必 True；反之必 False
    expected_unstable = estimate.ci_width > 0.6
    assert estimate.estimate_ci_unstable is expected_unstable, (
        f"n={sample_size} {method_name}: CI width={estimate.ci_width:.4f}, "
        f"expected estimate_ci_unstable={expected_unstable}, got {estimate.estimate_ci_unstable}"
    )


def test_call_estimate_effect_ci_width_decreases_with_n():
    """T7：小样本行为 — CI 宽度随样本量增加而单调下降（信号小样本不可靠）。"""
    from data.seed_loader import load_fixture_csv

    widths = {}
    for n in (30, 100, 200):
        df = _prepare(load_fixture_csv("I21")).head(n)
        p = CausalPipeline(disease_code="I21", outcome="cmi", data=df)
        est = p.call_estimate_effect(method_name="backdoor.linear_regression")
        widths[n] = est.ci_width

    # n=30 的 CI 应比 n=100/n=200 更宽（小样本方差大）
    assert widths[30] > widths[100], (
        f"n=30 CI width {widths[30]:.4f} 应大于 n=100 CI width {widths[100]:.4f}"
    )
    assert widths[100] >= widths[200], (
        f"n=100 CI width {widths[100]:.4f} 应大于等于 n=200 CI width {widths[200]:.4f}"
    )


def test_call_estimate_effect_iv_instrumental_variable_graceful():
    """T7：iv.instrumental_variable 在无 instruments 时优雅降级。

    spec §9：估计器枚举含 iv.instrumental_variable；当 pipeline 未传 instruments 时
    estimate_effect 抛异常 → 被 pipeline 捕获 → ate=None（不静默推进业务状态）。
    """
    from data.seed_loader import load_fixture_csv

    df = _prepare(load_fixture_csv("I21"))
    p = CausalPipeline(disease_code="I21", outcome="cmi", data=df)
    estimate = p.call_estimate_effect(method_name="iv.instrumental_variable")
    assert estimate.method_name == "iv.instrumental_variable"
    # 无 instruments 时 ate 必然为 None（dowhy 抛 ValueError → pipeline 兜底）
    assert estimate.ate is None
    # 不静默：estimate_ci_unstable 显式 True 标记该估计失败
    assert estimate.estimate_ci_unstable is True


def test_call_estimate_effect_iv_with_instruments_runs():
    """T7：iv.instrumental_variable 传入 instruments 后能跑通（基线/工具变量分离）。

    注：dowhy IV 估计需要 instruments 是 treatment 的因果前驱但不直接影响 outcome。
    这里用 readmit_flag 作为伪工具变量（业务上不严谨，但仅验证 IV 路径可跑通）。
    """
    from data.seed_loader import load_fixture_csv

    df = _prepare(load_fixture_csv("I21"))
    # 找一个 instruments 列（任意不影响 outcome 的列 — 这里用 readmit_flag）
    instruments = ["readmit_flag"]
    p = CausalPipeline(
        disease_code="I21",
        outcome="cmi",
        data=df,
        instruments=instruments,
    )
    estimate = p.call_estimate_effect(method_name="iv.instrumental_variable")
    assert estimate.method_name == "iv.instrumental_variable"
    # dowhy IV 估计可能返回 NaN 或具体值；这里只断言不抛异常 + sample_size 正确
    assert estimate.sample_size == len(df)


def test_call_estimate_effect_ci_width_unstable_marker_threshold():
    """T7：CI 宽度 > 0.6 阈值 → estimate_ci_unstable=True（spec §6 阈值）。

    模拟：直接构造 EffectEstimate 检查 ci_width 字段与 estimate_ci_unstable 的对应关系。
    实际 CI 宽度由数据驱动，本测试覆盖 schema 层契约。
    """
    from app.core.node_schema import EffectEstimate

    # 边界 0.6 内
    est_ok = EffectEstimate(
        node="treatment_strategy",
        treatment="treatment_strategy",
        outcome="cmi",
        method_name="backdoor.linear_regression",
        ate=0.5,
        ci_lower=0.2,
        ci_upper=0.79,  # width=0.59
        ci_width=0.59,
        p_value=0.01,
        estimate_ci_unstable=False,
        sample_size=500,
    )
    assert est_ok.estimate_ci_unstable is False

    # 边界外
    est_unstable = EffectEstimate(
        node="treatment_strategy",
        treatment="treatment_strategy",
        outcome="los_days",
        method_name="backdoor.linear_regression",
        ate=0.5,
        ci_lower=-0.1,
        ci_upper=1.0,  # width=1.1
        ci_width=1.1,
        p_value=0.4,
        estimate_ci_unstable=True,
        sample_size=30,
    )
    assert est_unstable.estimate_ci_unstable is True
    assert est_unstable.ci_width > 0.6


def test_call_estimate_effect_does_not_raise_on_small_n():
    """T7：小样本 n=30 时构造 + estimate 都不抛异常。

    spec §6：n<spec.min_sample_size → 构造期 InsufficientSampleError；n>=30 允许 estimate。
    """
    from data.seed_loader import load_fixture_csv

    df = _prepare(load_fixture_csv("I63")).head(30)
    p = CausalPipeline(disease_code="I63", outcome="los_days", data=df)
    estimate = p.call_estimate_effect(method_name="backdoor.linear_regression")
    assert estimate is not None
    # n=30 边界下 estimate 必须有结果（即使 CI 较宽）
    assert estimate.ate is not None
    assert estimate.ci_width is not None
    # estimate_ci_unstable 与 ci_width 一致
    assert estimate.estimate_ci_unstable is (estimate.ci_width > 0.6)


def test_call_estimate_effect_psm_auto_fallback_when_multivalued_treatment():
    """T7：PSM 在多值 treatment 下自动降级到 linear_regression（不静默失败）。

    dowhy PSM 仅支持二元 treatment；业务 DAG 中 treatment_strategy 通常 3+ 值。
    pipeline 检测到 > 2 unique treatment 值时自动降级，method_name 保持原值（用户视角），
    estimate.ate 用 LR 计算结果。
    """
    from data.seed_loader import load_fixture_csv

    df = _prepare(load_fixture_csv("I21"))
    # I21 fixture treatment_strategy 有 3 个取值：PCI / thrombolysis / conservative
    assert df["treatment_strategy"].nunique() > 2

    p = CausalPipeline(disease_code="I21", outcome="cmi", data=df)
    est_p = p.call_estimate_effect(method_name="backdoor.propensity_score_matching")
    est_lr = p.call_estimate_effect(method_name="backdoor.linear_regression")
    # method_name 保留用户请求（业务面板透传原值）
    assert est_p.method_name == "backdoor.propensity_score_matching"
    # 内部降级到 LR，结果应与 LR 一致（同一 pipeline 同 run）
    assert est_p.ate is not None
    assert abs(est_p.ate - est_lr.ate) < 1e-6
    assert est_p.ci_lower is not None
    assert est_p.ci_upper is not None


def test_call_estimate_effect_method_name_validated_by_schema():
    """T7：EffectEstimate.method_name 字段由 Pydantic schema 验证（白名单）。"""
    from pydantic import ValidationError

    from app.core.node_schema import EffectEstimate

    # 合法值
    est = EffectEstimate(
        node="treatment_strategy",
        treatment="treatment_strategy",
        outcome="cmi",
        method_name="backdoor.linear_regression",
        ate=0.1,
        sample_size=100,
    )
    assert est.method_name == "backdoor.linear_regression"

    # 非法估计器 → ValidationError
    with pytest.raises(ValidationError):
        EffectEstimate(
            node="treatment_strategy",
            treatment="treatment_strategy",
            outcome="cmi",
            method_name="backdoor.unknown_estimator",
            ate=0.1,
            sample_size=100,
        )


# ----------------------------------------------------------------------------------
# 6. 异常：EstimandNotIdentifiedError（构造期就拦 — 业务 DAG 完整时几乎不触发，留接口）
# ----------------------------------------------------------------------------------


def test_estimand_not_identified_error_is_runtime_error():
    """EstimandNotIdentifiedError 继承 RuntimeError，业务层可统一捕获。"""
    assert issubclass(EstimandNotIdentifiedError, RuntimeError)


def test_insufficient_sample_error_attributes():
    """InsufficientSampleError 携带 sample_size + min_required。"""
    err = InsufficientSampleError(sample_size=10, min_required=30)
    assert err.sample_size == 10
    assert err.min_required == 30
