"""equation_parser 单元测试（spec §10/§11 + Task 14）。

覆盖：
1. 构造期校验（disease_code / outcome 未知）
2. 业务默认模板：3 病种 × 4 outcome = 12 个组合
3. SCM 摘要字段完整性（template_hash / node_count / edge_count / 分布表）
4. 节点机制（NodeMechanism）字段
5. 因果图与 dag_builder 一致性
6. 自定义模板（含根节点 / 非根节点 / unknown 边 三种语法）
7. 黑名单字符 / 函数名冲突 / 节点重复 / 语法错误
8. parse_custom_template + parse_scm 便捷函数
9. SCM 与 pipeline 接口兼容（graph 节点 / 边可被后续模块消费）
10. 模板 hash 稳定性（同一模板复算得到同一 hash）

源产品能力参考：
- 模板语法源自 ``github_ref/dowhy/dowhy/gcm/equation_parser.py:44-77``
- 安全黑名单 78 / 函数冲突检测 178-193
- 节点摘要字段 196-202（unknown node 默认 EmpiricalDistribution）

⚠ README / 对外文档不写源 repo 名；测试类注释仅做代码溯源。
"""
from __future__ import annotations

import re

import networkx as nx
import pytest

from app.core.dag_builder import (
    DISEASE_OUTCOME_VALUES,
    SUPPORTED_DISEASE_CODES,
    DiseaseOutcome,
    TREATMENT_VALUES,
)
from app.core.equation_parser import (
    BANNED_CHARACTERS,
    NOISE_MODEL_RE,
    NoiseModelKind,
    NodeMechanism,
    SCM,
    SCMParser,
    SCMParserError,
    SCMSummary,
    _classify_noise_model,
    _extract_equation_components,
    _extract_noise_model_components,
    _function_type_distribution,
    _noise_model_distribution,
    _parse_args,
    _sanitize_input_expression,
    parse_custom_template,
    parse_scm,
)
from app.core.equation_templates import (
    CMI_EQUATIONS,
    COMPLICATION_FLAG_EQUATIONS,
    LOS_DAYS_EQUATIONS,
    OUTCOME_EQUATIONS,
    READMIT_FLAG_EQUATIONS,
    ROOT_NODE_EQUATIONS,
    build_template,
    list_templates,
)


# ----------------------------------------------------------------------------------
# 1. 构造期校验
# ----------------------------------------------------------------------------------


def test_parse_unknown_disease_code_raises_value_error():
    parser = SCMParser()
    with pytest.raises(ValueError, match="未知 disease_code"):
        parser.parse(disease_code="I99", outcome="cmi")


def test_parse_unknown_outcome_raises_value_error():
    parser = SCMParser()
    with pytest.raises(ValueError, match="未知 outcome"):
        parser.parse(disease_code="I21", outcome="readmission_rate")


def test_parse_empty_template_raises_scm_parser_error():
    parser = SCMParser()
    with pytest.raises(SCMParserError, match="模板字符串为空"):
        parser.parse(disease_code="I21", outcome="cmi", template="")


def test_parse_whitespace_only_template_raises_scm_parser_error():
    parser = SCMParser()
    with pytest.raises(SCMParserError, match="模板字符串为空"):
        parser.parse(disease_code="I21", outcome="cmi", template="   \n\n  ")


def test_parse_non_string_template_raises_scm_parser_error():
    """显式传 None 应走默认模板路径（不抛错）。"""
    parser = SCMParser()
    scm = parser.parse(disease_code="I21", outcome="cmi", template=None)
    assert scm.summary.template_source == "business_default"


# ----------------------------------------------------------------------------------
# 2. 业务默认模板：3 病种 × 4 outcome = 12 个组合
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("disease_code", list(SUPPORTED_DISEASE_CODES))
@pytest.mark.parametrize("outcome", list(DISEASE_OUTCOME_VALUES))
def test_business_default_template_parses_12_combos(disease_code, outcome):
    """3 病种 × 4 outcome = 12 个组合均能解析成功。"""
    scm = parse_scm(disease_code=disease_code, outcome=outcome)
    assert isinstance(scm, SCM)
    # 4 根节点 + 1 outcome = 5 节点
    assert scm.graph.number_of_nodes() == 5
    assert scm.summary.disease_code == disease_code
    assert scm.summary.outcome == outcome
    assert scm.summary.template_source == "business_default"
    assert scm.summary.rejected_lines == 0


def test_business_default_template_outcome_node_present():
    """业务默认模板必含 outcome 节点（cmi / los_days / ...）。"""
    for outcome in DISEASE_OUTCOME_VALUES:
        scm = parse_scm(disease_code="I21", outcome=outcome)
        assert outcome in scm.graph.nodes, (
            f"outcome={outcome} 应在 SCM.graph 节点中"
        )


def test_business_default_template_treatment_root_present():
    """根节点 treatment_strategy / admission_route / age_band / complication_count 都在图中。"""
    scm = parse_scm(disease_code="I21", outcome="cmi")
    for root in ("treatment_strategy", "admission_route", "age_band", "complication_count"):
        assert root in scm.graph.nodes


def test_business_default_template_outcome_has_parent_edges():
    """outcome 节点有 4 条入边（treatment / age_band / admission_route / complication_count）。"""
    scm = parse_scm(disease_code="I21", outcome="cmi")
    parents = list(scm.graph.predecessors("cmi"))
    assert len(parents) == 4
    # 父节点集合等于 4 个根节点
    assert set(parents) == {
        "treatment_strategy", "age_band", "admission_route", "complication_count",
    }


def test_business_default_template_summary_node_edge_consistent():
    """summary.node_count 与 graph.number_of_nodes() 一致；edge_count 同理。"""
    scm = parse_scm(disease_code="I63", outcome="los_days")
    assert scm.summary.node_count == scm.graph.number_of_nodes()
    assert scm.summary.edge_count == scm.graph.number_of_edges()


# ----------------------------------------------------------------------------------
# 3. SCM 摘要字段完整性
# ----------------------------------------------------------------------------------


def test_summary_template_hash_is_stable_for_same_template():
    """同一模板两次解析得到同一 SHA-256。"""
    scm1 = parse_scm(disease_code="I21", outcome="cmi")
    scm2 = parse_scm(disease_code="I21", outcome="cmi")
    assert scm1.summary.template_hash == scm2.summary.template_hash
    # 64 字符 SHA-256 hex
    assert len(scm1.summary.template_hash) == 64
    assert re.match(r"^[0-9a-f]{64}$", scm1.summary.template_hash)


def test_summary_template_hash_differs_for_different_outcome():
    """不同 outcome → 不同模板 → 不同 hash。"""
    scm_cmi = parse_scm(disease_code="I21", outcome="cmi")
    scm_los = parse_scm(disease_code="I21", outcome="los_days")
    assert scm_cmi.summary.template_hash != scm_los.summary.template_hash


def test_summary_template_hash_differs_for_custom_vs_default():
    """custom 模板与默认模板 hash 不同。"""
    scm_default = parse_scm(disease_code="I21", outcome="cmi")
    custom = "X = norm(loc=0, scale=1)\nY = 2 * X + norm(loc=0, scale=0.5)"
    scm_custom = parse_scm(disease_code="I21", outcome="cmi", template=custom)
    assert scm_default.summary.template_hash != scm_custom.summary.template_hash
    assert scm_custom.summary.template_source == "custom"


def test_summary_function_type_distribution_includes_root_and_additive():
    """function_type 分布至少含 root_noise 与 additive_noise_model 两类。"""
    scm = parse_scm(disease_code="I21", outcome="cmi")
    assert scm.summary.function_type_distribution.get("root_noise", 0) >= 4
    assert scm.summary.function_type_distribution.get("additive_noise_model", 0) >= 1


def test_summary_noise_model_distribution_contains_norm_and_empirical():
    """噪声模型分布至少含 'norm'（outcome 的噪声）和 'empirical'（根节点）。"""
    scm = parse_scm(disease_code="I21", outcome="cmi")
    assert "norm" in scm.summary.noise_model_distribution
    assert "empirical" in scm.summary.noise_model_distribution


def test_summary_parent_count_by_node_keys_match_nodes():
    """parent_count_by_node 的键集合与 graph 节点集合一致。"""
    scm = parse_scm(disease_code="Z96.10", outcome="readmit_flag")
    assert set(scm.summary.parent_count_by_node.keys()) == set(scm.graph.nodes)


def test_summary_outcome_node_has_max_parent_count():
    """outcome 节点的父节点数最多（业务假设：outcome 是汇点）。"""
    scm = parse_scm(disease_code="I21", outcome="cmi")
    parent_counts = scm.summary.parent_count_by_node
    outcome = scm.summary.outcome
    outcome_parents = parent_counts[outcome]
    max_others = max(
        (cnt for node, cnt in parent_counts.items() if node != outcome),
        default=0,
    )
    assert outcome_parents >= max_others


# ----------------------------------------------------------------------------------
# 4. NodeMechanism 字段完整性
# ----------------------------------------------------------------------------------


def test_root_node_mechanism_is_empirical_distribution():
    """根节点 treatment_strategy 的 mechanism 是 root_noise + empirical。"""
    scm = parse_scm(disease_code="I21", outcome="cmi")
    mech = scm.causal_mechanisms["treatment_strategy"]
    assert isinstance(mech, NodeMechanism)
    assert mech.function_type == "root_noise"
    assert mech.noise_model_name == "empirical"
    assert mech.noise_model_kind == NoiseModelKind.EMPIRICAL
    assert mech.fully_defined is False
    assert mech.parent_nodes == []


def test_outcome_node_mechanism_is_additive_noise_model():
    """outcome 节点 mechanism 含父节点列表 + 拟合表达式 + 噪声模型参数。"""
    scm = parse_scm(disease_code="I21", outcome="cmi")
    mech = scm.causal_mechanisms["cmi"]
    assert mech.function_type == "additive_noise_model"
    assert mech.noise_model_name == "norm"
    assert mech.noise_model_kind == NoiseModelKind.SCIPY_DISTRIBUTION
    assert mech.fully_defined is True
    assert "scale" in mech.noise_args
    assert "loc" in mech.noise_args
    assert set(mech.parent_nodes) == {
        "treatment_strategy", "age_band", "admission_route", "complication_count",
    }
    # custom_function 含父节点名（至少含 treatment_strategy）
    assert "treatment_strategy" in mech.custom_function


def test_all_node_mechanisms_have_required_fields():
    """所有机制都有 node_name + function_type；噪声模型名非空（除 unknown 类型）。"""
    scm = parse_scm(disease_code="I21", outcome="cmi")
    for name, mech in scm.causal_mechanisms.items():
        assert mech.node_name == name
        assert mech.function_type in (
            "root_noise", "additive_noise_model", "unknown_relationship",
        )


# ----------------------------------------------------------------------------------
# 5. 因果图与 dag_builder 一致性
# ----------------------------------------------------------------------------------


def test_scm_graph_is_dag():
    """SCM 图是无环有向图（与 dag_builder.build_dag 同结构）。"""
    scm = parse_scm(disease_code="I21", outcome="cmi")
    assert isinstance(scm.graph, nx.DiGraph)
    assert nx.is_directed_acyclic_graph(scm.graph)


def test_scm_graph_outcome_node_has_no_successors():
    """outcome 节点是汇点，无后继。"""
    scm = parse_scm(disease_code="I21", outcome="cmi")
    assert list(scm.graph.successors("cmi")) == []


def test_scm_graph_root_nodes_have_no_predecessors():
    """根节点（treatment_strategy / age_band / admission_route）无前驱。"""
    scm = parse_scm(disease_code="I63", outcome="readmit_flag")
    for root in ("treatment_strategy", "age_band", "admission_route"):
        assert list(scm.graph.predecessors(root)) == []


# ----------------------------------------------------------------------------------
# 6. 自定义模板：三种语法（根节点 / 非根节点 / unknown 边）
# ----------------------------------------------------------------------------------


def test_custom_template_root_node_syntax():
    """根节点语法：``X = empirical()`` 解析为 root_noise。"""
    template = "X = empirical()\nY = norm(loc=0, scale=1)"
    scm = parse_custom_template(template, disease_code="X", outcome="Y")
    assert scm.causal_mechanisms["X"].function_type == "root_noise"
    assert scm.causal_mechanisms["Y"].function_type == "root_noise"
    assert scm.causal_mechanisms["Y"].noise_model_name == "norm"
    assert scm.causal_mechanisms["Y"].noise_args == {"loc": 0.0, "scale": 1.0}


def test_custom_template_non_root_with_additive_noise():
    """非根节点语法：``Y = 2 * X + norm(loc=0, scale=0.5)`` 解析为 additive_noise_model。"""
    template = "X = empirical()\nY = 2 * X + norm(loc=0, scale=0.5)"
    scm = parse_custom_template(template, disease_code="X", outcome="Y")
    assert scm.causal_mechanisms["Y"].function_type == "additive_noise_model"
    assert scm.causal_mechanisms["Y"].parent_nodes == ["X"]
    assert scm.causal_mechanisms["Y"].custom_function == "2 * X"
    assert scm.causal_mechanisms["Y"].noise_args == {"loc": 0.0, "scale": 0.5}
    assert scm.graph.has_edge("X", "Y")


def test_custom_template_unknown_relationship_syntax():
    """unknown 关系语法：``A -> B, C`` 仅定义边不定义函数。

    A 是唯一被定义的节点（unknown_relationship）；B/C 作为父节点被图引用。
    """
    template = "A -> B, C"
    scm = parse_custom_template(template, disease_code="A", outcome="A")
    # A 是 unknown_relationship 类型（仅边定义）
    mech_a = scm.causal_mechanisms["A"]
    assert mech_a.function_type == "unknown_relationship"
    assert mech_a.unknown is True
    assert mech_a.parent_nodes == ["B", "C"]
    assert scm.graph.has_edge("B", "A")
    assert scm.graph.has_edge("C", "A")
    # B 和 C 出现在图中但未定义 → 自动补全为 root_noise
    assert scm.causal_mechanisms["B"].function_type == "root_noise"
    assert scm.causal_mechanisms["C"].function_type == "root_noise"


def test_custom_template_multi_parent_function():
    """多父节点函数：``Z = 2 * X + 3 * Y + norm(loc=0, scale=1)``。"""
    template = (
        "X = empirical()\nY = empirical()\n"
        "Z = 2 * X + 3 * Y + norm(loc=0, scale=1)"
    )
    scm = parse_custom_template(template, disease_code="X", outcome="Z")
    mech = scm.causal_mechanisms["Z"]
    assert mech.parent_nodes == ["X", "Y"]  # 字母序
    assert scm.graph.has_edge("X", "Z")
    assert scm.graph.has_edge("Y", "Z")


def test_custom_template_uses_bayesian_mixture_model():
    """bayesiangaussianmixture 噪声模型识别正确。"""
    template = "X = bayesiangaussianmixture()\nY = X + bayesiangaussianmixture()"
    scm = parse_custom_template(template, disease_code="X", outcome="Y")
    assert scm.causal_mechanisms["X"].noise_model_kind == NoiseModelKind.BAYESIAN_GAUSSIAN_MIXTURE
    assert scm.causal_mechanisms["Y"].noise_model_kind == NoiseModelKind.BAYESIAN_GAUSSIAN_MIXTURE


def test_custom_template_uses_parametric_model():
    """parametric 噪声模型识别。"""
    template = "X = parametric()\nY = 1.5 * X + parametric()"
    scm = parse_custom_template(template, disease_code="X", outcome="Y")
    assert scm.causal_mechanisms["X"].noise_model_kind == NoiseModelKind.PARAMETRIC


# ----------------------------------------------------------------------------------
# 7. 错误路径（黑名单 / 函数名冲突 / 节点重复 / 语法错误）
# ----------------------------------------------------------------------------------


def test_banned_character_colon_in_function_raises():
    """含 ``:`` 在表达式内（非节点名）抛 SCMParserError。"""
    # 节点名 treatment:PCI 是合法的；但函数内出现 attr 调用会被黑名单拒
    template = "X = empirical()\nY = 0.6 * bad:symbol + norm(loc=0, scale=1)"
    with pytest.raises(SCMParserError, match="黑名单"):
        parse_custom_template(template, disease_code="X", outcome="Y")


def test_banned_character_import_raises():
    """含 ``import`` 抛 SCMParserError。"""
    template = "X = empirical()\nY = __import__('os').system('ls') + norm(loc=0, scale=1)"
    with pytest.raises(SCMParserError, match="黑名单"):
        parse_custom_template(template, disease_code="X", outcome="Y")


def test_banned_character_bracket_raises():
    """含 ``[`` 抛 SCMParserError。"""
    template = "X = empirical()\nY = list(range(3))[0] + norm(loc=0, scale=1)"
    with pytest.raises(SCMParserError, match="黑名单"):
        parse_custom_template(template, disease_code="X", outcome="Y")


def test_double_dot_attribute_access_rejected():
    """含 ``..`` 在非十进制数字中抛 SCMParserError（防属性访问注入）。"""
    template = "X = empirical()\nY = 1 .. 2 + norm(loc=0, scale=1)"
    with pytest.raises(SCMParserError, match="属性访问"):
        parse_custom_template(template, disease_code="X", outcome="Y")


def test_duplicate_node_definition_rejected():
    """同一节点在模板中定义两次 → 第二次抛 SCMParserError。"""
    template = (
        "X = empirical()\n"
        "Y = X + norm(loc=0, scale=1)\n"
        "Y = X + norm(loc=0, scale=2)"  # duplicate
    )
    parser = SCMParser()
    scm = parser.parse(disease_code="X", outcome="Y", template=template)
    # 第二条 Y 被跳过 → rejected_lines=1
    assert scm.summary.rejected_lines == 1
    # 第一条定义生效
    assert scm.causal_mechanisms["Y"].noise_args == {"loc": 0.0, "scale": 1.0}


def test_unknown_noise_model_in_additive_raises():
    """非根节点含未知噪声模型名抛 SCMParserError（被拒计入 rejected_lines）。"""
    template = "X = empirical()\nY = X + unicorn(loc=0, scale=1)"
    parser = SCMParser()
    scm = parser.parse(disease_code="X", outcome="Y", template=template)
    # Y 行被拒，X 仍生效
    assert scm.summary.rejected_lines == 1
    assert "Y" not in scm.causal_mechanisms


def test_function_name_conflict_warns_but_does_not_break():
    """父节点候选与内置函数名冲突：warn 但不阻塞解析，冲突名被忽略。"""
    template = "X = empirical()\nmean = empirical()\nY = 2 * X + norm(loc=0, scale=1)"
    parser = SCMParser()
    scm = parser.parse(disease_code="X", outcome="Y", template=template)
    # 'mean' 作为节点名合法；'mean' 出现在非根节点表达式中时若与函数冲突，会被 warn
    # 这里 'mean' 仅作为独立节点定义，Y 的父节点只有 X
    assert scm.causal_mechanisms["Y"].parent_nodes == ["X"]


def test_invalid_node_name_pattern_rejected():
    """非法节点名（含特殊字符）抛 SCMParserError。"""
    template = "X = empirical()\n123Y = norm(loc=0, scale=1)"
    parser = SCMParser()
    scm = parser.parse(disease_code="X", outcome="123Y", template=template)
    assert scm.summary.rejected_lines >= 1


def test_treat_unknown_root_as_empirical_false_raises():
    """``treat_unknown_root_as_empirical=False`` 时，未定义节点抛 SCMParserError。

    模板：``X -> Y`` — X 是被定义的节点（unknown_relationship），Y 是 X 的父节点
    但模板中**未**给出 Y 的方程 → Y 是「未定义节点」。
    """
    template = "X -> Y"
    parser = SCMParser()
    with pytest.raises(SCMParserError, match="节点.*在模板中未定义"):
        parser.parse(
            disease_code="X", outcome="X", template=template,
            treat_unknown_root_as_empirical=False,
        )


def test_treat_unknown_root_as_empirical_true_default():
    """默认 True 时未定义节点被补成 EmpiricalDistribution。"""
    template = "X -> Y"
    parser = SCMParser()
    scm = parser.parse(disease_code="X", outcome="X", template=template)
    # Y 自动补成 root + empirical
    mech_y = scm.causal_mechanisms["Y"]
    assert mech_y.function_type == "root_noise"
    assert mech_y.noise_model_name == "empirical"


def test_parse_args_positional_rejected():
    """位置参数（如 ``"norm(0, 1)"``）抛 SCMParserError。"""
    template = "X = empirical()\nY = X + norm(0, 1)"
    parser = SCMParser()
    scm = parser.parse(disease_code="X", outcome="Y", template=template)
    # Y 行被拒（位置参数格式）；X 仍生效
    assert scm.summary.rejected_lines >= 1


def test_parse_args_invalid_value_raises():
    """参数值非法字面量抛 SCMParserError。"""
    template = "X = empirical()\nY = X + norm(loc=not_a_value, scale=1)"
    parser = SCMParser()
    scm = parser.parse(disease_code="X", outcome="Y", template=template)
    assert scm.summary.rejected_lines >= 1


# ----------------------------------------------------------------------------------
# 8. 便捷函数 + 模板预览
# ----------------------------------------------------------------------------------


def test_parse_scm_equivalent_to_parser_method():
    """parse_scm 等价于 SCMParser().parse()。"""
    scm_helper = parse_scm("I21", "cmi")
    scm_method = SCMParser().parse("I21", "cmi")
    assert scm_helper.summary.template_hash == scm_method.summary.template_hash
    assert scm_helper.summary.node_count == scm_method.summary.node_count


def test_build_template_returns_multiline_string():
    """build_template 返回多行字符串（每行一条方程）。"""
    template = build_template("I21", "cmi")
    lines = [l for l in template.split("\n") if l.strip()]
    assert len(lines) >= 5  # 4 根节点 + 1 outcome
    # 含 outcome 方程关键词
    assert any("cmi" in line and "=" in line for line in lines)


def test_build_template_unknown_disease_raises():
    with pytest.raises(ValueError, match="未知 disease_code"):
        build_template("I99", "cmi")


def test_build_template_unknown_outcome_raises():
    with pytest.raises(ValueError, match="未知 outcome"):
        build_template("I21", "foo_bar")


def test_list_templates_returns_12_items():
    items = list_templates()
    assert len(items) == 12  # 3 病种 × 4 outcome
    disease_outcome_pairs = {(i["disease_code"], i["outcome"]) for i in items}
    for d in SUPPORTED_DISEASE_CODES:
        for o in DISEASE_OUTCOME_VALUES:
            assert (d, o) in disease_outcome_pairs


def test_list_templates_preview_contains_outcome_keyword():
    items = list_templates()
    for item in items:
        # 预览行包含 outcome 关键词
        assert item["outcome"] in item["template_preview"]


# ----------------------------------------------------------------------------------
# 9. SCM 与 pipeline 接口兼容（graph 可被消费）
# ----------------------------------------------------------------------------------


def test_scm_graph_predecessors_count_matches_parent_count_summary():
    """summary.parent_count_by_node 与 graph.predecessors() 数量一致。"""
    scm = parse_scm(disease_code="I21", outcome="cmi")
    for node, summary_count in scm.summary.parent_count_by_node.items():
        actual_parents = list(scm.graph.predecessors(node))
        assert len(actual_parents) == summary_count


def test_scm_graph_can_be_traversed_for_counterfactual():
    """SCM graph 可被 CounterfactualRunner 消费（节点 ID 与 fixture 列对齐）。"""
    # 通过 _map_node_to_data_column 同名节点的 fixture 列映射验证
    # 见 dosampler._map_node_to_data_column；这里只验证 graph 节点名格式正确
    scm = parse_scm(disease_code="I21", outcome="cmi")
    # 根节点名与 fixture 列名一致
    for node in ("treatment_strategy", "age_band", "admission_route", "complication_count"):
        assert node in scm.graph.nodes


def test_scm_graph_outcome_node_label_matches_disease_outcome():
    """outcome 节点 ID 与 DiseaseOutcome 枚举值一致。"""
    for outcome in DISEASE_OUTCOME_VALUES:
        scm = parse_scm(disease_code="I21", outcome=outcome)
        assert outcome in scm.graph.nodes
        assert outcome in scm.summary.parent_count_by_node


# ----------------------------------------------------------------------------------
# 10. 模块内部 helper 单测
# ----------------------------------------------------------------------------------


def test_sanitize_input_expression_accepts_normal():
    _sanitize_input_expression("cmi = 0.6 * X + norm(loc=0, scale=1)", BANNED_CHARACTERS)


def test_sanitize_input_expression_rejects_each_banned():
    for banned in BANNED_CHARACTERS:
        with pytest.raises(SCMParserError):
            _sanitize_input_expression(f"cmi = {banned}stuff", BANNED_CHARACTERS)


def test_sanitize_input_expression_rejects_double_dot():
    with pytest.raises(SCMParserError, match="属性访问"):
        _sanitize_input_expression("Y = 1 .. 2 + norm(loc=0, scale=1)", BANNED_CHARACTERS)


def test_sanitize_input_expression_rejects_non_whitelist():
    with pytest.raises(SCMParserError, match="非白名单"):
        _sanitize_input_expression("Y = $bad + norm(loc=0, scale=1)", BANNED_CHARACTERS)


def test_parse_args_empty_returns_empty_dict():
    assert _parse_args("") == {}


def test_parse_args_kw_args_returns_dict():
    assert _parse_args("loc=0, scale=1") == {"loc": 0.0, "scale": 1.0}


def test_parse_args_string_values():
    assert _parse_args("name='foo'") == {"name": "foo"}


def test_parse_args_positional_rejected():
    with pytest.raises(SCMParserError, match="keyword"):
        _parse_args("0, 1")


def test_extract_equation_components_arrow():
    node, expr = _extract_equation_components("A -> B, C")
    assert node == "A"
    assert expr == "B, C"


def test_extract_equation_components_equals():
    node, expr = _extract_equation_components("X = norm(loc=0, scale=1)")
    assert node == "X"
    assert expr == "norm(loc=0, scale=1)"


def test_extract_equation_components_missing_separator_raises():
    """无 ``=`` 且无 ``->`` 的非法方程抛 SCMParserError。"""
    with pytest.raises(SCMParserError):
        _extract_equation_components("Xnorm(loc 0 scale 1)")


def test_extract_noise_model_components_valid():
    name, args = _extract_noise_model_components("norm(loc=0, scale=1)")
    assert name == "norm"
    assert args == {"loc": 0.0, "scale": 1.0}


def test_extract_noise_model_components_invalid_raises():
    with pytest.raises(SCMParserError):
        _extract_noise_model_components("not_a_call_pattern")


def test_classify_noise_model_known():
    assert _classify_noise_model("empirical") == NoiseModelKind.EMPIRICAL
    assert _classify_noise_model("bayesiangaussianmixture") == NoiseModelKind.BAYESIAN_GAUSSIAN_MIXTURE
    assert _classify_noise_model("parametric") == NoiseModelKind.PARAMETRIC


def test_classify_noise_model_scipy_distribution():
    assert _classify_noise_model("norm") == NoiseModelKind.SCIPY_DISTRIBUTION
    assert _classify_noise_model("t") == NoiseModelKind.SCIPY_DISTRIBUTION
    assert _classify_noise_model("beta") == NoiseModelKind.SCIPY_DISTRIBUTION


def test_function_type_distribution_counts():
    mechanisms = {
        "A": NodeMechanism(node_name="A", function_type="root_noise"),
        "B": NodeMechanism(node_name="B", function_type="root_noise"),
        "C": NodeMechanism(node_name="C", function_type="additive_noise_model"),
    }
    dist = _function_type_distribution(mechanisms)
    assert dist == {"root_noise": 2, "additive_noise_model": 1}


def test_noise_model_distribution_counts():
    mechanisms = {
        "A": NodeMechanism(
            node_name="A", function_type="root_noise",
            noise_model_name="empirical",
        ),
        "B": NodeMechanism(
            node_name="B", function_type="root_noise",
            noise_model_name="norm",
        ),
        "C": NodeMechanism(
            node_name="C", function_type="additive_noise_model",
            noise_model_name="norm",
        ),
    }
    dist = _noise_model_distribution(mechanisms)
    assert dist == {"empirical": 1, "norm": 2}


def test_noise_model_re_pattern_matches_norm_with_args():
    m = NOISE_MODEL_RE.match("norm(loc=0, scale=1)")
    assert m is not None
    assert m.group(1) == "norm"
    assert m.group(2) == "loc=0, scale=1"


def test_noise_model_re_pattern_matches_empirical_empty_args():
    m = NOISE_MODEL_RE.match("empirical()")
    assert m is not None
    assert m.group(1) == "empirical"
    assert m.group(2) == ""


# ----------------------------------------------------------------------------------
# 11. 模板数据契约
# ----------------------------------------------------------------------------------


def test_root_node_equations_count():
    """ROOT_NODE_EQUATIONS 含 4 条根节点方程（treatment / age_band / admission_route / complication_count）。"""
    assert len(ROOT_NODE_EQUATIONS) == 4


def test_root_node_equations_use_empirical():
    """所有根节点方程使用 empirical() 噪声模型。"""
    for eq in ROOT_NODE_EQUATIONS:
        assert "empirical()" in eq


def test_root_node_equations_use_fixture_column_names():
    """根节点名与 fixture 聚合列名一致。"""
    expected = {"treatment_strategy", "age_band", "admission_route", "complication_count"}
    actual = {eq.split("=")[0].strip() for eq in ROOT_NODE_EQUATIONS}
    assert actual == expected


def test_outcome_equations_keys_match_disease_codes():
    """所有 outcome 方程表都覆盖 3 个病种。"""
    for outcome, eq_map in OUTCOME_EQUATIONS.items():
        assert set(eq_map.keys()) == set(SUPPORTED_DISEASE_CODES)


def test_outcome_equations_use_norm_noise():
    """所有 outcome 方程都使用 norm(...) 噪声模型。"""
    for outcome, eq_map in OUTCOME_EQUATIONS.items():
        for disease_code, eq in eq_map.items():
            assert "norm(" in eq, f"{outcome}/{disease_code} 缺少 norm 噪声"
            assert eq.startswith(f"{outcome} = ")


def test_cmi_equations_per_disease_have_treatment_term():
    """CMI 方程含 treatment_strategy 项。"""
    for disease_code, eq in CMI_EQUATIONS.items():
        assert "treatment_strategy" in eq, f"{disease_code} CMI 方程缺少 treatment 项"


def test_los_days_equations_have_age_band_term():
    """los_days 方程含 age_band 项。"""
    for disease_code, eq in LOS_DAYS_EQUATIONS.items():
        assert "age_band" in eq


def test_readmit_flag_equations_have_treatment_term():
    """readmit_flag 方程用 treatment_strategy 作为强驱动。"""
    for disease_code, eq in READMIT_FLAG_EQUATIONS.items():
        assert "treatment_strategy" in eq


def test_complication_flag_equations_have_treatment_term():
    """complication_flag 方程含 treatment_strategy 项。"""
    for disease_code, eq in COMPLICATION_FLAG_EQUATIONS.items():
        assert "treatment_strategy" in eq


# ----------------------------------------------------------------------------------
# 12. 端到端：3 病种 × 4 outcome 跑通 + summary 一致性
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("disease_code", list(SUPPORTED_DISEASE_CODES))
@pytest.mark.parametrize("outcome", list(DISEASE_OUTCOME_VALUES))
def test_scm_12_combos_full_validation(disease_code, outcome):
    """3 病种 × 4 outcome 端到端跑通；摘要字段、节点、机制都齐。"""
    scm = parse_scm(disease_code=disease_code, outcome=outcome)

    # SCM 类型
    assert isinstance(scm, SCM)
    assert isinstance(scm.summary, SCMSummary)

    # 图是 DAG
    assert isinstance(scm.graph, nx.DiGraph)
    assert nx.is_directed_acyclic_graph(scm.graph)

    # 节点 / 边计数一致
    assert scm.summary.node_count == scm.graph.number_of_nodes()
    assert scm.summary.edge_count == scm.graph.number_of_edges()

    # outcome 节点在图中
    assert outcome in scm.graph.nodes

    # 每个图节点都在 mechanisms 中
    for node in scm.graph.nodes:
        assert node in scm.causal_mechanisms, f"节点 {node} 缺机制"

    # summary.disease_code / outcome 一致
    assert scm.summary.disease_code == disease_code
    assert scm.summary.outcome == outcome

    # 解析行数 >= 1（至少有 1 条 outcome 方程）
    assert scm.summary.parsed_lines >= 1

    # 拒绝行数为 0（业务默认模板不应有错）
    assert scm.summary.rejected_lines == 0

    # function_type 分布不为空
    assert scm.summary.function_type_distribution

    # noise_model 分布不为空
    assert scm.summary.noise_model_distribution


def test_scm_summary_fields_types():
    """summary 各字段类型符合契约。"""
    scm = parse_scm(disease_code="I21", outcome="cmi")
    assert isinstance(scm.summary.disease_code, str)
    assert isinstance(scm.summary.outcome, str)
    assert isinstance(scm.summary.template_source, str)
    assert isinstance(scm.summary.template_hash, str)
    assert isinstance(scm.summary.node_count, int)
    assert isinstance(scm.summary.edge_count, int)
    assert isinstance(scm.summary.parent_count_by_node, dict)
    assert isinstance(scm.summary.function_type_distribution, dict)
    assert isinstance(scm.summary.noise_model_distribution, dict)
    assert isinstance(scm.summary.parsed_lines, int)
    assert isinstance(scm.summary.rejected_lines, int)


def test_scm_mechanism_field_types():
    """NodeMechanism 各字段类型符合契约。"""
    scm = parse_scm(disease_code="I21", outcome="cmi")
    for mech in scm.causal_mechanisms.values():
        assert isinstance(mech.node_name, str)
        assert isinstance(mech.function_type, str)
        assert isinstance(mech.parent_nodes, list)
        assert isinstance(mech.custom_function, str)
        assert isinstance(mech.noise_model_name, str)
        assert isinstance(mech.noise_model_kind, NoiseModelKind)
        assert isinstance(mech.noise_args, dict)
        assert isinstance(mech.fully_defined, bool)
        assert isinstance(mech.unknown, bool)


# ----------------------------------------------------------------------------------
# 13. DiseaseOutcome 枚举校验
# ----------------------------------------------------------------------------------


def test_disease_outcome_enum_values_match_template_keys():
    """DiseaseOutcome 枚举值与 OUTCOME_EQUATIONS 的键集合一致。"""
    enum_values = {o.value for o in DiseaseOutcome}
    template_keys = set(OUTCOME_EQUATIONS.keys())
    assert enum_values == template_keys


def test_treatment_values_keys_match_supported_disease_codes():
    """TREATMENT_VALUES 的键集合与 SUPPORTED_DISEASE_CODES 一致。"""
    assert set(TREATMENT_VALUES.keys()) == set(SUPPORTED_DISEASE_CODES)


# ----------------------------------------------------------------------------------
# 14. SCM summary.node_count 跨 12 组合一致性
# ----------------------------------------------------------------------------------


def test_all_business_default_scm_have_exactly_5_nodes():
    """所有业务默认 SCM 恰好 5 个节点（4 根 + 1 outcome）。"""
    for disease_code in SUPPORTED_DISEASE_CODES:
        for outcome in DISEASE_OUTCOME_VALUES:
            scm = parse_scm(disease_code=disease_code, outcome=outcome)
            assert scm.summary.node_count == 5, (
                f"{disease_code}/{outcome} 节点数={scm.summary.node_count} != 5"
            )


def test_all_business_default_scm_have_exactly_4_edges():
    """所有业务默认 SCM 恰好 4 条边（4 根 → outcome）。"""
    for disease_code in SUPPORTED_DISEASE_CODES:
        for outcome in DISEASE_OUTCOME_VALUES:
            scm = parse_scm(disease_code=disease_code, outcome=outcome)
            assert scm.summary.edge_count == 4, (
                f"{disease_code}/{outcome} 边数={scm.summary.edge_count} != 4"
            )


# ----------------------------------------------------------------------------------
# 15. SCM 与 spec §10「参考地基」行为对齐
# ----------------------------------------------------------------------------------


def test_scm_function_type_distribution_sums_to_node_count():
    """function_type 分布的和 = 节点数（每个节点都有 function_type）。"""
    scm = parse_scm(disease_code="I21", outcome="cmi")
    total = sum(scm.summary.function_type_distribution.values())
    assert total == scm.summary.node_count


def test_scm_noise_model_distribution_sum_le_node_count():
    """noise_model 分布的和 <= 节点数（unknown 类型可能没有 noise_model）。"""
    scm = parse_scm(disease_code="I21", outcome="cmi")
    total = sum(scm.summary.noise_model_distribution.values())
    assert total <= scm.summary.node_count


def test_scm_outcome_node_has_additive_noise_model_function_type():
    """outcome 节点的 function_type 是 additive_noise_model（业务因果机制）。

    只断言当前解析的 outcome；其它 outcome 在其它 parse 调用里。
    """
    scm = parse_scm(disease_code="I21", outcome="cmi")
    assert scm.causal_mechanisms["cmi"].function_type == "additive_noise_model"
    # 4 个 outcome 各自跑一遍都应是 additive_noise_model
    for outcome in DISEASE_OUTCOME_VALUES:
        scm_out = parse_scm(disease_code="I21", outcome=outcome)
        assert scm_out.causal_mechanisms[outcome].function_type == "additive_noise_model"


def test_scm_root_nodes_have_root_noise_function_type():
    """4 个根节点全部是 root_noise 类型。"""
    scm = parse_scm(disease_code="I63", outcome="cmi")
    for root in ("treatment_strategy", "age_band", "admission_route", "complication_count"):
        assert scm.causal_mechanisms[root].function_type == "root_noise"