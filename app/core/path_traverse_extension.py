"""path_traverse 结果展平为可读节点列表（spec.md §11 第 7 步 / T17）。

业务动机：
- 面板点节点后希望看到一个「这个节点能看到的所有相关节点」列表，含：
    * 节点 id
    * 因果强度（来自沿路径的 effect 几何平均）
    * 出现在几条路径里（in_paths 计数）
    * 子节点（successors）列表
    * 是否可在反事实面板 do()
    * 节点角色（treatment / covariate / outcome）+ 颜色
- 这是 path_traverse.flatten_traversal 的扩展版（保留原函数不变，向前兼容），
  同时通过 PathTraversalExtension 类提供更结构化、可缓存、可被面板直接消费的输出。

数据契约：
- PanelNodeView{ node, role, color, in_paths, max_strength, children, intervenable, semantic }

源产品能力参考（README 不写源 repo 名）：
- PathTraverser.traverse / TraversalResult / CausalPath 与 path_traverse.py 同源
- nx.DiGraph.successors() 取子节点；本模块不引入新的图遍历逻辑
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import networkx as nx

from app.core.path_traverse import (
    CausalPath,
    TraversalResult,
    flatten_traversal as _flatten_traversal_basic,
)

LOGGER = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------
# 常量（与 path_traverse + dag_builder 对齐；不做硬编码业务值）
# ----------------------------------------------------------------------------------

# 可干预节点前缀（与 path_traverse.flatten_traversal 一致）
_INTERVENABLE_PREFIXES: tuple = (
    "treatment:",
    "age_band:",
    "admission_route:",
    "complication_count:",
)


# ----------------------------------------------------------------------------------
# 数据契约
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class PanelNodeView:
    """面板节点视图（spec §11 第 7 步 / T17）。

    字段：
      - node:         节点 id（与 DAG nx.DiGraph 节点一致）
      - role:         节点角色（treatment / covariate / outcome / unknown）
      - color:        节点颜色（hex / CSS color）；优先从 DAG 属性读
      - semantic:     业务语义说明（来自 DAG 节点属性）
      - in_paths:     该节点出现在几条路径里
      - max_strength: 该节点相关路径的最大 strength
      - children:     子节点 id 列表（nx.DiGraph.successors）
      - intervenable: 是否可在反事实面板 do()
    """

    node: str
    role: str
    color: str
    semantic: str
    in_paths: int
    max_strength: float
    children: List[str] = field(default_factory=list)
    intervenable: bool = False

    def __post_init__(self) -> None:
        if not self.node:
            raise ValueError("node 不可为空")
        if self.role not in ("treatment", "covariate", "outcome", "unknown"):
            raise ValueError(f"未知 role: {self.role!r}")
        if self.in_paths < 0:
            raise ValueError(f"in_paths 必须 >= 0；got {self.in_paths}")
        if self.max_strength < 0:
            raise ValueError(f"max_strength 必须 >= 0；got {self.max_strength}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node": self.node,
            "role": self.role,
            "color": self.color,
            "semantic": self.semantic,
            "in_paths": self.in_paths,
            "max_strength": self.max_strength,
            "children": list(self.children),
            "intervenable": self.intervenable,
        }


# ----------------------------------------------------------------------------------
# 展平器
# ----------------------------------------------------------------------------------


class PathTraversalExtension:
    """把 TraversalResult 展平为面板可直接消费的 PanelNodeView 列表（T17）。

    Usage
    -----
    >>> from app.core.dag_builder import build_dag
    >>> from app.core.path_traverse import PathTraverser
    >>> dag = build_dag("I21", "cmi")
    >>> result = PathTraverser(dag).traverse("treatment:PCI", depth=3)
    >>> views = PathTraversalExtension(dag).flatten(result)
    >>> for v in views:
    ...     print(v.node, v.role, v.in_paths, v.intervenable)
    """

    def __init__(self, dag: nx.DiGraph) -> None:
        if dag is None:
            raise ValueError("dag 不能为空")
        if not isinstance(dag, nx.DiGraph):
            raise TypeError(
                f"dag 必须是 nx.DiGraph；got {type(dag).__name__}"
            )
        self.dag = dag

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def flatten(self, result: TraversalResult) -> List[PanelNodeView]:
        """把 TraversalResult 展平为 PanelNodeView 列表。

        展平规则：
          - 节点出现顺序 = result.paths 中首次出现的顺序（DFS 路径遍历顺序）
          - 每个节点的角色 / 颜色 / semantic 从 DAG 节点属性读
          - children = list(self.dag.successors(node))，保持原顺序
          - intervenable：节点 role ∈ {treatment, covariate} 即 True
          - in_paths / max_strength 从 path 列表聚合
          - 仅出现在 path 中的节点会被纳入（panel 只看与 source 相关的节点）
        """
        if result is None:
            raise ValueError("result 不能为空")

        # 1. 节点 → 出现顺序 / in_paths / max_strength
        first_seen_order: List[str] = []
        seen: set = set()
        in_paths: Dict[str, int] = {}
        max_strength: Dict[str, float] = {}
        for path in result.paths:
            for node in path.nodes:
                if node not in seen:
                    first_seen_order.append(node)
                    seen.add(node)
                in_paths[node] = in_paths.get(node, 0) + 1
                cur = max_strength.get(node, 0.0)
                if path.strength > cur:
                    max_strength[node] = path.strength

        # 2. 把每个节点转成 PanelNodeView
        views: List[PanelNodeView] = []
        for node in first_seen_order:
            attrs = self.dag.nodes.get(node, {})
            role = attrs.get("kind", "unknown")
            color = attrs.get("color", "#7f7f7f")  # 默认背景灰
            semantic = attrs.get("semantic", "")
            children = list(self.dag.successors(node))
            intervenable = role in ("treatment", "covariate")
            views.append(
                PanelNodeView(
                    node=node,
                    role=role,
                    color=color,
                    semantic=semantic,
                    in_paths=in_paths[node],
                    max_strength=float(max_strength[node]),
                    children=children,
                    intervenable=intervenable,
                )
            )
        return views

    def flatten_with_paths(
        self, result: TraversalResult
    ) -> List[Dict[str, Any]]:
        """展平为 dict 列表（含每条路径元数据），供前端 React 直接消费。

        返回字段：
          - paths: List[CausalPath dict]（每条路径 nodes + strength + length + intervenable_count）
          - views: List[PanelNodeView dict]（节点视图列表）
          - source: 起点节点
          - max_depth: 最大深度
          - truncated: 是否被 MAX_PATHS_PER_SOURCE 截断
        """
        return {
            "source": result.source,
            "max_depth": result.max_depth,
            "truncated": result.truncated,
            "paths": [
                {
                    "nodes": list(p.nodes),
                    "strength": p.strength,
                    "length": p.length,
                    "intervenable_count": p.intervenable_count,
                }
                for p in result.paths
            ],
            "views": [v.to_dict() for v in self.flatten(result)],
        }


# ----------------------------------------------------------------------------------
# 便捷函数
# ----------------------------------------------------------------------------------


def flatten_traversal_for_panel(
    result: TraversalResult, dag: nx.DiGraph
) -> List[PanelNodeView]:
    """PathTraversalExtension(dag).flatten(result) 的最简封装。"""
    return PathTraversalExtension(dag).flatten(result)


def flatten_traversal_for_panel_dict(
    result: TraversalResult, dag: nx.DiGraph
) -> List[Dict[str, Any]]:
    """返回 List[dict]（PanelNodeView.to_dict），供 JSON 序列化（面板 SSE / API）。"""
    return [v.to_dict() for v in PathTraversalExtension(dag).flatten(result)]


def flatten_traversal(result: TraversalResult) -> List[Dict[str, Any]]:
    """path_traverse.flatten_traversal 的兼容 re-export（T17 向前兼容）。

    本函数等于 ``from app.core.path_traverse import flatten_traversal``；
    保留在这里是便于 panel 模块只 import path_traverse_extension 就能用旧 API。
    """
    return _flatten_traversal_basic(result)


__all__ = [
    "PanelNodeView",
    "PathTraversalExtension",
    "flatten_traversal_for_panel",
    "flatten_traversal_for_panel_dict",
    "flatten_traversal",
]