"""test_path_traverse_extension — 验证 path_traverse 结果展平（T17）。

覆盖维度：
  1. 构造期校验
  2. flatten 主入口：返回 PanelNodeView 列表
  3. 节点字段完整性：node / role / color / semantic / in_paths / max_strength / children / intervenable
  4. 节点角色映射：DAG node.kind → PanelNodeView.role
  5. children 子节点列表：来自 nx.DiGraph.successors
  6. intervenable 标记：treatment / covariate → True；outcome → False
  7. in_paths / max_strength 聚合
  8. flatten_with_paths 整合 paths + views
  9. 便捷函数 flatten_traversal_for_panel / flatten_traversal_for_panel_dict
 10. flatten_traversal re-export（向前兼容 path_traverse.flatten_traversal）
 11. 空 result 行为
 12. 3 病种 × 4 outcome 跑通
"""
from __future__ import annotations

import pytest
import networkx as nx

from app.core.dag_builder import build_dag
from app.core.path_traverse import (
    PathTraverser,
    TraversalResult,
    flatten_traversal as _orig_flatten,
)
from app.core.path_traverse_extension import (
    PanelNodeView,
    PathTraversalExtension,
    flatten_traversal,
    flatten_traversal_for_panel,
    flatten_traversal_for_panel_dict,
)


# ----------------------------------------------------------------------------------
# 1. 构造期校验
# ----------------------------------------------------------------------------------


def test_extension_none_dag_raises():
    """dag=None → ValueError。"""
    with pytest.raises(ValueError, match="dag 不能为空"):
        PathTraversalExtension(dag=None)


def test_extension_wrong_type_raises():
    """非 nx.DiGraph → TypeError。"""
    with pytest.raises(TypeError, match="必须是 nx.DiGraph"):
        PathTraversalExtension(dag={"not": "a digraph"})


def test_extension_accepts_digraph():
    """nx.DiGraph 实例可构造。"""
    dag = build_dag("I21", "cmi")
    ext = PathTraversalExtension(dag=dag)
    assert ext.dag is dag


# ----------------------------------------------------------------------------------
# 2. flatten 主入口
# ----------------------------------------------------------------------------------


def test_flatten_returns_list_of_panel_node_views():
    """flatten 返回 PanelNodeView 列表。"""
    dag = build_dag("I21", "cmi")
    result = PathTraverser(dag).traverse("treatment:PCI", depth=3)
    views = PathTraversalExtension(dag).flatten(result)

    assert isinstance(views, list)
    assert len(views) >= 1
    for v in views:
        assert isinstance(v, PanelNodeView)


def test_flatten_does_not_include_unrelated_nodes():
    """flatten 仅包含出现在 result.paths 中的节点。"""
    dag = build_dag("I21", "cmi")
    result = PathTraverser(dag).traverse("treatment:PCI", depth=3)
    views = PathTraversalExtension(dag).flatten(result)

    node_set = {v.node for v in views}
    # treatment:PCI 必在；cmi 必在
    assert "treatment:PCI" in node_set
    assert "cmi" in node_set
    # 与 treatment:PCI 完全无路径关联的节点（如 I21 的 thrombolysis）不应出现在 views 中
    # 除非它在某条 path 里被访问到；这里 depth=3 应该不会溯到 thrombolysis
    assert "treatment:thrombolysis" not in node_set


# ----------------------------------------------------------------------------------
# 3. 节点字段完整性
# ----------------------------------------------------------------------------------


def test_panel_node_view_required_fields():
    """PanelNodeView 字段全：8 字段。"""
    dag = build_dag("I21", "cmi")
    result = PathTraverser(dag).traverse("treatment:PCI", depth=3)
    views = PathTraversalExtension(dag).flatten(result)

    for v in views:
        assert v.node
        assert v.role in ("treatment", "covariate", "outcome", "unknown")
        assert v.color.startswith("#")
        assert isinstance(v.in_paths, int)
        assert v.in_paths >= 1
        assert isinstance(v.max_strength, (int, float))
        assert v.max_strength >= 0.0
        assert isinstance(v.children, list)
        assert isinstance(v.intervenable, bool)


def test_panel_node_view_to_dict_includes_all_fields():
    """to_dict 含 8 字段。"""
    dag = build_dag("I21", "cmi")
    result = PathTraverser(dag).traverse("treatment:PCI", depth=3)
    views = PathTraversalExtension(dag).flatten(result)

    for v in views:
        d = v.to_dict()
        assert set(d.keys()) >= {
            "node",
            "role",
            "color",
            "semantic",
            "in_paths",
            "max_strength",
            "children",
            "intervenable",
        }


# ----------------------------------------------------------------------------------
# 4. 节点角色映射
# ----------------------------------------------------------------------------------


def test_role_mapping_for_treatment():
    """treatment 节点 → role='treatment'。"""
    dag = build_dag("I21", "cmi")
    result = PathTraverser(dag).traverse("treatment:PCI", depth=3)
    views = PathTraversalExtension(dag).flatten(result)

    pci = next(v for v in views if v.node == "treatment:PCI")
    assert pci.role == "treatment"


def test_role_mapping_for_covariate():
    """covariate 节点 → role='covariate'。

    业务 DAG：complication_count → outcome 路径上必有 covariate 节点。
    """
    dag = build_dag("I21", "cmi")
    result = PathTraverser(dag).traverse("treatment:PCI", depth=3)
    views = PathTraversalExtension(dag).flatten(result)

    comp_view = next(
        v for v in views if v.node.startswith("complication_count:")
    )
    assert comp_view.role == "covariate"


def test_role_mapping_for_age_band_via_traverse_all_sources():
    """traverse_all_sources 包含 age_band 节点 → role='covariate'。"""
    dag = build_dag("I21", "cmi")
    traverser = PathTraverser(dag)
    results = traverser.traverse_all_sources(depth=3)
    # 找 age_band:1 的 result
    age_band_result = results["age_band:1"]
    views = PathTraversalExtension(dag).flatten(age_band_result)

    age_band_view = next(v for v in views if v.node == "age_band:1")
    assert age_band_view.role == "covariate"


def test_role_mapping_for_outcome():
    """outcome 节点 → role='outcome'。"""
    dag = build_dag("I21", "cmi")
    result = PathTraverser(dag).traverse("treatment:PCI", depth=3)
    views = PathTraversalExtension(dag).flatten(result)

    cmi = next(v for v in views if v.node == "cmi")
    assert cmi.role == "outcome"


def test_color_from_dag_attribute():
    """颜色从 DAG 节点属性读。"""
    dag = build_dag("I21", "cmi")
    result = PathTraverser(dag).traverse("treatment:PCI", depth=3)
    views = PathTraversalExtension(dag).flatten(result)

    pci = next(v for v in views if v.node == "treatment:PCI")
    assert pci.color == "#d62728"  # NODE_COLOR_STRONG（红）


def test_semantic_from_dag_attribute():
    """semantic 从 DAG 节点属性读。"""
    dag = build_dag("I21", "cmi")
    result = PathTraverser(dag).traverse("treatment:PCI", depth=3)
    views = PathTraversalExtension(dag).flatten(result)

    pci = next(v for v in views if v.node == "treatment:PCI")
    assert "PCI" in pci.semantic


# ----------------------------------------------------------------------------------
# 5. children 子节点列表
# ----------------------------------------------------------------------------------


def test_children_for_treatment_node():
    """treatment:PCI 的 children 含 cmi 与各 complication_count。"""
    dag = build_dag("I21", "cmi")
    result = PathTraverser(dag).traverse("treatment:PCI", depth=3)
    views = PathTraversalExtension(dag).flatten(result)

    pci = next(v for v in views if v.node == "treatment:PCI")
    # DAG 中 treatment:PCI → cmi + complication_count:{0,1,2}（4 条 successor）
    assert "cmi" in pci.children
    assert all(
        f"complication_count:{i}" in pci.children for i in (0, 1, 2)
    )
    assert len(pci.children) == 4


def test_children_for_outcome_node():
    """outcome 节点 (cmi) 的 children 为空（无 successor）。"""
    dag = build_dag("I21", "cmi")
    result = PathTraverser(dag).traverse("treatment:PCI", depth=3)
    views = PathTraversalExtension(dag).flatten(result)

    cmi = next(v for v in views if v.node == "cmi")
    assert cmi.children == []


# ----------------------------------------------------------------------------------
# 6. intervenable 标记
# ----------------------------------------------------------------------------------


def test_intervenable_for_treatment():
    """treatment 节点 intervenable=True。"""
    dag = build_dag("I21", "cmi")
    result = PathTraverser(dag).traverse("treatment:PCI", depth=3)
    views = PathTraversalExtension(dag).flatten(result)

    pci = next(v for v in views if v.node == "treatment:PCI")
    assert pci.intervenable is True


def test_intervenable_for_covariate():
    """covariate 节点 intervenable=True（complication_count 在 treatment:PCI 路径上）。"""
    dag = build_dag("I21", "cmi")
    result = PathTraverser(dag).traverse("treatment:PCI", depth=3)
    views = PathTraversalExtension(dag).flatten(result)

    comp_view = next(v for v in views if v.node.startswith("complication_count:"))
    assert comp_view.intervenable is True


def test_intervenable_for_age_band_via_traverse_all_sources():
    """traverse_all_sources 中 age_band:1 的 view 也 intervenable=True。"""
    dag = build_dag("I21", "cmi")
    traverser = PathTraverser(dag)
    results = traverser.traverse_all_sources(depth=3)
    age_band_result = results["age_band:1"]
    views = PathTraversalExtension(dag).flatten(age_band_result)

    age_band_view = next(v for v in views if v.node == "age_band:1")
    assert age_band_view.intervenable is True


def test_intervenable_for_outcome():
    """outcome 节点 intervenable=False（do() outcome 会自指）。"""
    dag = build_dag("I21", "cmi")
    result = PathTraverser(dag).traverse("treatment:PCI", depth=3)
    views = PathTraversalExtension(dag).flatten(result)

    cmi = next(v for v in views if v.node == "cmi")
    assert cmi.intervenable is False


# ----------------------------------------------------------------------------------
# 7. in_paths / max_strength 聚合
# ----------------------------------------------------------------------------------


def test_in_paths_count_for_treatment_node():
    """treatment:PCI 出现在 ≥1 条路径里。"""
    dag = build_dag("I21", "cmi")
    result = PathTraverser(dag).traverse("treatment:PCI", depth=3)
    views = PathTraversalExtension(dag).flatten(result)

    pci = next(v for v in views if v.node == "treatment:PCI")
    assert pci.in_paths >= 1


def test_in_paths_aggregation_correct():
    """in_paths = 该节点出现在几条 path 里。"""
    dag = build_dag("I21", "cmi")
    result = PathTraverser(dag).traverse("treatment:PCI", depth=3)
    views = PathTraversalExtension(dag).flatten(result)

    # 手算 in_paths
    expected = {}
    for p in result.paths:
        for node in p.nodes:
            expected[node] = expected.get(node, 0) + 1
    # 对每个 view 验证
    for v in views:
        assert v.in_paths == expected[v.node], (
            f"节点 {v.node}：view.in_paths={v.in_paths}, "
            f"expected={expected[v.node]}"
        )


def test_max_strength_aggregation_correct():
    """max_strength = 该节点相关路径的最大 strength。"""
    dag = build_dag("I21", "cmi")
    result = PathTraverser(dag).traverse("treatment:PCI", depth=3)
    views = PathTraversalExtension(dag).flatten(result)

    expected = {}
    for p in result.paths:
        for node in p.nodes:
            expected[node] = max(expected.get(node, 0.0), p.strength)
    for v in views:
        assert v.max_strength == pytest.approx(expected[v.node])


# ----------------------------------------------------------------------------------
# 8. flatten_with_paths 整合
# ----------------------------------------------------------------------------------


def test_flatten_with_paths_returns_full_dict():
    """flatten_with_paths 返回 source/max_depth/truncated/paths/views 5 字段。"""
    dag = build_dag("I21", "cmi")
    result = PathTraverser(dag).traverse("treatment:PCI", depth=3)
    payload = PathTraversalExtension(dag).flatten_with_paths(result)

    assert set(payload.keys()) >= {
        "source",
        "max_depth",
        "truncated",
        "paths",
        "views",
    }
    assert payload["source"] == "treatment:PCI"
    assert payload["max_depth"] == 3
    assert isinstance(payload["paths"], list)
    assert isinstance(payload["views"], list)
    # 每条 path 含 nodes/strength/length/intervenable_count
    for p in payload["paths"]:
        assert "nodes" in p
        assert "strength" in p
        assert "length" in p
        assert "intervenable_count" in p
    # views 是 PanelNodeView dict 列表
    for v in payload["views"]:
        assert "node" in v
        assert "intervenable" in v


# ----------------------------------------------------------------------------------
# 9. 便捷函数
# ----------------------------------------------------------------------------------


def test_flatten_traversal_for_panel_helper():
    """flatten_traversal_for_panel = PathTraversalExtension(dag).flatten(result)。"""
    dag = build_dag("I21", "cmi")
    result = PathTraverser(dag).traverse("treatment:PCI", depth=3)
    views_helper = flatten_traversal_for_panel(result, dag)
    views_method = PathTraversalExtension(dag).flatten(result)

    assert len(views_helper) == len(views_method)
    for a, b in zip(views_helper, views_method):
        assert a.node == b.node
        assert a.role == b.role


def test_flatten_traversal_for_panel_dict_helper():
    """flatten_traversal_for_panel_dict 返回 dict 列表。"""
    dag = build_dag("I21", "cmi")
    result = PathTraverser(dag).traverse("treatment:PCI", depth=3)
    payload = flatten_traversal_for_panel_dict(result, dag)

    assert isinstance(payload, list)
    for entry in payload:
        assert isinstance(entry, dict)
        assert "node" in entry
        assert "intervenable" in entry


# ----------------------------------------------------------------------------------
# 10. flatten_traversal re-export 兼容
# ----------------------------------------------------------------------------------


def test_flatten_traversal_reexport_compatible():
    """path_traverse_extension.flatten_traversal 等价 path_traverse.flatten_traversal。"""
    dag = build_dag("I21", "cmi")
    result = PathTraverser(dag).traverse("treatment:PCI", depth=3)

    reexported = flatten_traversal(result)
    original = _orig_flatten(result)
    assert reexported == original


# ----------------------------------------------------------------------------------
# 11. 空 result
# ----------------------------------------------------------------------------------


def test_flatten_empty_result_returns_empty_list():
    """空 result → flatten 返回空 list（不抛）。"""
    dag = build_dag("I21", "cmi")
    result = TraversalResult(source="treatment:PCI", max_depth=3, paths=[])
    views = PathTraversalExtension(dag).flatten(result)

    assert views == []


def test_flatten_none_result_raises():
    """result=None → ValueError。"""
    dag = build_dag("I21", "cmi")
    with pytest.raises(ValueError, match="result 不能为空"):
        PathTraversalExtension(dag).flatten(None)


# ----------------------------------------------------------------------------------
# 12. 3 病种 × 4 outcome 跑通
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "disease_code,outcome",
    [
        ("I21", "cmi"),
        ("I21", "los_days"),
        ("I21", "readmit_flag"),
        ("I21", "complication_flag"),
        ("I63", "cmi"),
        ("I63", "los_days"),
        ("I63", "readmit_flag"),
        ("I63", "complication_flag"),
        ("Z96.10", "cmi"),
        ("Z96.10", "los_days"),
        ("Z96.10", "readmit_flag"),
        ("Z96.10", "complication_flag"),
    ],
)
def test_three_diseases_x_four_outcomes_flatten(disease_code, outcome):
    """3 病种 × 4 outcome = 12 组合 flatten 跑通。"""
    dag = build_dag(disease_code, outcome)
    # 从该病种第一个 treatment 节点出发
    treatment_nodes = [
        n for n, attrs in dag.nodes(data=True) if attrs.get("kind") == "treatment"
    ]
    source = treatment_nodes[0]
    result = PathTraverser(dag).traverse(source, depth=3)
    views = PathTraversalExtension(dag).flatten(result)

    # 至少 source + outcome 节点
    assert any(v.node == source for v in views)
    assert any(v.node == outcome for v in views)
    # 至少 1 条 view 的 children 非空（业务 DAG 中 treatment → outcome 必有）
    assert any(len(v.children) > 0 for v in views)
    # outcome 节点 intervenable=False
    outcome_views = [v for v in views if v.node == outcome]
    assert all(v.intervenable is False for v in outcome_views)


# ----------------------------------------------------------------------------------
# 13. PanelNodeView 数据契约
# ----------------------------------------------------------------------------------


def test_panel_node_view_empty_node_raises():
    """node 空字符串 → ValueError。"""
    with pytest.raises(ValueError, match="node 不可为空"):
        PanelNodeView(
            node="",
            role="treatment",
            color="#d62728",
            semantic="x",
            in_paths=1,
            max_strength=1.0,
        )


def test_panel_node_view_invalid_role_raises():
    """role 非法 → ValueError。"""
    with pytest.raises(ValueError, match="未知 role"):
        PanelNodeView(
            node="x",
            role="bad_role",
            color="#000",
            semantic="x",
            in_paths=1,
            max_strength=1.0,
        )


def test_panel_node_view_negative_in_paths_raises():
    """in_paths < 0 → ValueError。"""
    with pytest.raises(ValueError, match="in_paths 必须 >= 0"):
        PanelNodeView(
            node="x",
            role="treatment",
            color="#000",
            semantic="x",
            in_paths=-1,
            max_strength=1.0,
        )


def test_panel_node_view_negative_max_strength_raises():
    """max_strength < 0 → ValueError。"""
    with pytest.raises(ValueError, match="max_strength 必须 >= 0"):
        PanelNodeView(
            node="x",
            role="treatment",
            color="#000",
            semantic="x",
            in_paths=1,
            max_strength=-0.1,
        )