"""test_dag_builder — 验证 3 病种 × 4 outcome 的业务 DAG 节点 / 边 / 颜色正确。

覆盖维度（spec §11 融合后主路径）：
  1. DiseaseOutcome 枚举值与 spec §3.1 outcome 候选对齐
  2. 3 个 SUPPORTED_DISEASE_CODES × 4 个 outcome = 12 个 DAG 全部可构造
  3. 节点 kind 分布（treatment / covariate / outcome）
  4. 节点颜色编码（强驱动 / 弱驱动 / outcome / 背景）
  5. 边类型（treatment_to_outcome / covariate_to_outcome / background）
  6. outcome_node / treatment_nodes / covariate_nodes / dag_summary 提取函数
  7. 未知 disease_code / outcome → ValueError
"""
from __future__ import annotations

import pytest

from app.core.dag_builder import (
    ADMISSION_ROUTE_VALUES,
    AGE_BAND_VALUES,
    DISEASE_OUTCOME_VALUES,
    NODE_COLOR_OUTCOME,
    NODE_COLOR_STRONG,
    NODE_COLOR_WEAK,
    SUPPORTED_DISEASE_CODES,
    TREATMENT_VALUES,
    DiseaseOutcome,
    build_dag,
    covariate_nodes,
    dag_summary,
    outcome_node,
    treatment_nodes,
)


def test_supported_disease_codes():
    """3 病种：急性心梗 / 脑梗 / 髋置换。"""
    assert SUPPORTED_DISEASE_CODES == ("I21", "I63", "Z96.10")


def test_disease_outcome_enum_matches_spec():
    """DiseaseOutcome 4 值：cmi / los_days / readmit_flag / complication_flag。"""
    assert DiseaseOutcome.CMI.value == "cmi"
    assert DiseaseOutcome.LOS_DAYS.value == "los_days"
    assert DiseaseOutcome.READMIT_FLAG.value == "readmit_flag"
    assert DiseaseOutcome.COMPLICATION_FLAG.value == "complication_flag"


def test_outcome_value_set_constant():
    """DISPATH / DiseaseOutcome 暴露的 outcome 字符串集合一致。"""
    assert DISEASE_OUTCOME_VALUES == (
        "cmi",
        "los_days",
        "readmit_flag",
        "complication_flag",
    )


@pytest.mark.parametrize("disease_code", list(SUPPORTED_DISEASE_CODES))
@pytest.mark.parametrize("outcome", list(DiseaseOutcome))
def test_build_dag_all_combinations(disease_code, outcome):
    """3 病种 × 4 outcome = 12 个 DAG 全部可构造。"""
    g = build_dag(disease_code, outcome)
    assert g.number_of_nodes() > 0
    assert g.number_of_edges() > 0
    assert g.graph["disease_code"] == disease_code
    assert g.graph["outcome"] == outcome.value


@pytest.mark.parametrize("disease_code", list(SUPPORTED_DISEASE_CODES))
@pytest.mark.parametrize("outcome", list(DiseaseOutcome))
def test_dag_node_kind_distribution(disease_code, outcome):
    """每个 DAG 必须有 treatment / covariate / outcome 三类节点。"""
    import networkx as nx

    g = build_dag(disease_code, outcome)
    kinds = {attrs.get("kind") for _, attrs in g.nodes(data=True)}
    assert "treatment" in kinds
    assert "covariate" in kinds
    assert "outcome" in kinds


@pytest.mark.parametrize("disease_code", list(SUPPORTED_DISEASE_CODES))
@pytest.mark.parametrize("outcome", list(DiseaseOutcome))
def test_dag_treatment_count_matches_spec(disease_code, outcome):
    """treatment 节点数 = TREATMENT_VALUES[disease_code] 的枚举数。"""
    g = build_dag(disease_code, outcome)
    expected = len(TREATMENT_VALUES[disease_code])
    assert len(treatment_nodes(g)) == expected


@pytest.mark.parametrize("disease_code", list(SUPPORTED_DISEASE_CODES))
@pytest.mark.parametrize("outcome", list(DiseaseOutcome))
def test_dag_covariate_count(disease_code, outcome):
    """covariate 节点数 = age_band + admission_route + complication_count = 4 + 4 + 3 = 11。"""
    g = build_dag(disease_code, outcome)
    expected = len(AGE_BAND_VALUES) + len(ADMISSION_ROUTE_VALUES) + 3  # comp_count: 0/1/2
    assert len(covariate_nodes(g)) == expected


@pytest.mark.parametrize("disease_code", list(SUPPORTED_DISEASE_CODES))
@pytest.mark.parametrize("outcome", list(DiseaseOutcome))
def test_dag_color_encoding(disease_code, outcome):
    """节点颜色按 kind 编码：treatment=strong / covariate=weak / outcome=blue。"""
    g = build_dag(disease_code, outcome)
    for n, attrs in g.nodes(data=True):
        kind = attrs.get("kind")
        color = attrs.get("color")
        if kind == "treatment":
            assert color == NODE_COLOR_STRONG
        elif kind == "covariate":
            assert color == NODE_COLOR_WEAK
        elif kind == "outcome":
            assert color == NODE_COLOR_OUTCOME
        else:  # pragma: no cover
            pytest.fail(f"未知 kind: {kind}")


@pytest.mark.parametrize("disease_code", list(SUPPORTED_DISEASE_CODES))
@pytest.mark.parametrize("outcome", list(DiseaseOutcome))
def test_dag_treatment_edges_to_outcome(disease_code, outcome):
    """每个 treatment 节点都有一条到 outcome 的边（kind=treatment_to_outcome）。"""
    g = build_dag(disease_code, outcome)
    out = outcome.value
    for t_node in treatment_nodes(g):
        edges = g.out_edges(t_node, data=True)
        edge_to_outcome = [e for e in edges if e[1] == out]
        assert len(edge_to_outcome) == 1
        assert edge_to_outcome[0][2]["kind"] == "treatment_to_outcome"


@pytest.mark.parametrize("disease_code", list(SUPPORTED_DISEASE_CODES))
@pytest.mark.parametrize("outcome", list(DiseaseOutcome))
def test_dag_covariate_edges_to_outcome(disease_code, outcome):
    """每个 covariate 节点都有一条到 outcome 的边（kind=covariate_to_outcome）。"""
    g = build_dag(disease_code, outcome)
    out = outcome.value
    for c_node in covariate_nodes(g):
        edges = g.out_edges(c_node, data=True)
        edge_to_outcome = [e for e in edges if e[1] == out]
        assert len(edge_to_outcome) == 1
        assert edge_to_outcome[0][2]["kind"] == "covariate_to_outcome"


@pytest.mark.parametrize("disease_code", list(SUPPORTED_DISEASE_CODES))
@pytest.mark.parametrize("outcome", list(DiseaseOutcome))
def test_outcome_node_extraction(disease_code, outcome):
    """outcome_node() 返回 outcome 值字符串。"""
    g = build_dag(disease_code, outcome)
    assert outcome_node(g) == outcome.value


@pytest.mark.parametrize("disease_code", list(SUPPORTED_DISEASE_CODES))
@pytest.mark.parametrize("outcome", list(DiseaseOutcome))
def test_dag_summary(disease_code, outcome):
    """dag_summary 返回节点 / 边 / kinds / disease_code / outcome。"""
    g = build_dag(disease_code, outcome)
    summary = dag_summary(g)
    assert summary["disease_code"] == disease_code
    assert summary["outcome"] == outcome.value
    assert summary["node_count"] == g.number_of_nodes()
    assert summary["edge_count"] == g.number_of_edges()
    assert summary["kinds"]["treatment"] == len(TREATMENT_VALUES[disease_code])
    assert summary["kinds"]["covariate"] == (
        len(AGE_BAND_VALUES) + len(ADMISSION_ROUTE_VALUES) + 3
    )
    assert summary["kinds"]["outcome"] == 1


def test_build_dag_unknown_disease_code():
    """未知 disease_code → ValueError。"""
    with pytest.raises(ValueError, match="未知 disease_code"):
        build_dag("I99", "cmi")


def test_build_dag_unknown_outcome_string():
    """未知 outcome 字符串 → ValueError。"""
    with pytest.raises(ValueError, match="未知 outcome"):
        build_dag("I21", "mortality_rate")


def test_build_dag_accepts_disease_outcome_enum():
    """build_dag 接受 DiseaseOutcome 枚举（非 str）。"""
    g = build_dag("I21", DiseaseOutcome.CMI)
    assert g.graph["outcome"] == "cmi"


def test_build_dag_rejects_invalid_outcome_type():
    """build_dag 拒绝非 str / DiseaseOutcome 的 outcome 参数。"""
    with pytest.raises(TypeError):
        build_dag("I21", 123)


def test_treatment_values_per_disease():
    """TREATMENT_VALUES 三病种各 3 个治疗方式。"""
    assert len(TREATMENT_VALUES["I21"]) == 3
    assert "PCI" in TREATMENT_VALUES["I21"]
    assert "thrombolysis" in TREATMENT_VALUES["I63"]
    assert "total_hip_replacement" in TREATMENT_VALUES["Z96.10"]
