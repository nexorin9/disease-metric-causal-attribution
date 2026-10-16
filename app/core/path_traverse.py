"""因果路径遍历（spec.md §11 主路径 / §9 因果推理管线 / 因果图深度遍历）。

沿业务 DAG（nx.DiGraph，方向：treatment / covariate → outcome）从任一节点
出发做有限深度 DFS，列出所有从该节点出发、不超过 ``max_depth`` 跳的因果路径。
这是面板「点击节点 → 展开因果链」交互的后端主路径：质控员点击某个治疗方式
节点，面板显示该治疗到 outcome 的所有 ≤3 跳路径（治疗 → 合并症 → outcome /
治疗 → outcome / …），并标注每条路径的因果强度（由因果效应绝对值近似）。

设计原则（spec §6 失败不静默改数据）：
- **节点去重**：DFS 经过的节点不重复访问（防环）；若 DAG 存在回环则停在
  已访问节点集合里继续（不抛异常；跳过该方向即可）。
- **深度边界**：max_depth 默认 3（业务 DAG：treatment → complication_count
  → outcome 已经 2 跳；max_depth=3 留 1 跳冗余覆盖未来扩展）。
- **路径上限**：单源遍历路径数 ≤ 1000 防止指数爆炸；超出后停止 DFS 并返回
  已找到的路径（spec §6 防爆兜底）。
- **强度近似**：因果强度 = |效应绝对值|；未跑效应估计时默认 1.0。
- **intervenable 标记**：路径中所有 covariate 节点标记 ``intervenable=True``
  （这些节点可在反事实面板上 do()），outcome 节点 ``intervenable=False``。

源产品能力参考（README 不写源仓库名）：
- networkx.DiGraph 的 ``nx.all_simple_paths(G, source, target, cutoff=max_depth)``
  用于取简单路径；本模块在它基础上做了节点去重 + 路径上限保护。
- 沿用 CausalGraph 的 DiGraph 数据结构；effect_strength 由调用方传入。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import networkx as nx

from app.core.dag_builder import outcome_node as _dag_outcome_node

LOGGER = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------
# 常量
# ----------------------------------------------------------------------------------

# 默认遍历深度（业务 DAG：treatment → complication_count → outcome 已 2 跳）。
DEFAULT_MAX_DEPTH: int = 3

# 单源 DFS 路径上限（防指数爆炸）。
MAX_PATHS_PER_SOURCE: int = 1000

# 默认因果强度（未跑效应估计时）。
DEFAULT_EFFECT_STRENGTH: float = 1.0


# ----------------------------------------------------------------------------------
# 数据契约
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class CausalPath:
    """单条因果路径（节点序列 + 强度近似）。"""

    # 节点序列：必填（至少 1 个节点）
    nodes: Tuple[str, ...]
    # 因果强度 = 所有边 |effect| 的几何平均；未跑效应时取 1.0。
    strength: float = DEFAULT_EFFECT_STRENGTH
    # 路径长度（节点数）
    length: int = 0
    # 路径中可干预节点数（covariate / treatment 类节点）
    intervenable_count: int = 0

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ValueError("CausalPath.nodes 不能为空")
        if self.length <= 0:
            object.__setattr__(self, "length", len(self.nodes))
        if self.intervenable_count < 0:
            raise ValueError("intervenable_count 必须 >= 0")


@dataclass(frozen=True)
class TraversalResult:
    """节点遍历结果（spec §11 节点展示 / T13 主入口）。"""

    source: str
    max_depth: int
    paths: List[CausalPath] = field(default_factory=list)
    visited_nodes: Tuple[str, ...] = field(default_factory=tuple)
    truncated: bool = False  # 路径数超过 MAX_PATHS_PER_SOURCE 时 True

    def flat_nodes(self) -> List[str]:
        """展平所有路径节点去重 → 用于面板节点列表渲染。"""
        seen: List[str] = []
        seen_set: set = set()
        for path in self.paths:
            for node in path.nodes:
                if node not in seen_set:
                    seen.append(node)
                    seen_set.add(node)
        return seen


# ----------------------------------------------------------------------------------
# PathTraverser 主体
# ----------------------------------------------------------------------------------


class PathTraverser:
    """因果路径遍历器（spec.md §11 第 2 步 / T13）。

    Usage
    -----
    >>> from app.core.dag_builder import build_dag
    >>> dag = build_dag("I21", "cmi")
    >>> traverser = PathTraverser(dag)
    >>> result = traverser.traverse("treatment:PCI", depth=3)
    >>> for p in result.paths:
    ...     print(" → ".join(p.nodes), "strength=", p.strength)
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
    # 主入口：单源 DFS
    # ------------------------------------------------------------------

    def traverse(
        self,
        source: str,
        depth: int = DEFAULT_MAX_DEPTH,
        *,
        effect_strengths: Optional[Dict[str, float]] = None,
    ) -> TraversalResult:
        """从 source 出发，遍历所有长度 ≤ depth 的因果路径（DFS + 节点去重）。

        Parameters
        ----------
        source
            起点节点 id；必须在 self.dag 中。
        depth
            最大路径长度（节点数）；默认 3。必须 >= 1。
        effect_strengths
            可选：节点 → 边效应绝对值映射，供路径 strength 计算。
            缺省时全部边 strength=DEFAULT_EFFECT_STRENGTH。

        Returns
        -------
        TraversalResult
            paths / visited_nodes / truncated 三字段。
        """
        if depth < 1:
            raise ValueError(f"depth 必须 >= 1；got {depth}")
        if source not in self.dag.nodes:
            raise ValueError(
                f"未知 source 节点: {source!r}；"
                f"已知节点数: {self.dag.number_of_nodes()}"
            )

        effect_map = effect_strengths or {}

        paths: List[CausalPath] = []
        visited: List[str] = []
        visited_set: set = set()
        truncated = False

        # DFS 栈：每项 = (current_node, path_so_far, depth_so_far)
        stack: List[tuple] = [(source, [source], 1)]
        while stack:
            current, path, d = stack.pop()
            # 记录首个访问
            if current not in visited_set:
                visited.append(current)
                visited_set.add(current)

            # 路径收集条件：当前路径长度 >= 2（至少有 1 条边）
            if len(path) >= 2:
                # 已收集条数上限
                if len(paths) >= MAX_PATHS_PER_SOURCE:
                    truncated = True
                    break
                paths.append(
                    self._build_path(path, effect_map)
                )

            # 是否还能继续扩展
            if d >= depth:
                continue

            # 沿 successor 方向扩展（去重：仅访问不在 path_so_far 的节点）
            for succ in self.dag.successors(current):
                if succ in path:
                    # 已访问过（环）→ 跳过该方向
                    continue
                stack.append((succ, path + [succ], d + 1))

        return TraversalResult(
            source=source,
            max_depth=depth,
            paths=paths,
            visited_nodes=tuple(visited),
            truncated=truncated,
        )

    # ------------------------------------------------------------------
    # 主入口：批量起点（业务 DAG 中所有 treatment/covariate 节点）
    # ------------------------------------------------------------------

    def traverse_all_sources(
        self,
        *,
        depth: int = DEFAULT_MAX_DEPTH,
        effect_strengths: Optional[Dict[str, float]] = None,
    ) -> Dict[str, TraversalResult]:
        """对 DAG 中所有非 outcome 节点跑 traverse；返回 {source: TraversalResult}。"""
        result: Dict[str, TraversalResult] = {}
        for node, attrs in self.dag.nodes(data=True):
            if attrs.get("kind") == "outcome":
                continue  # outcome 节点本身不是遍历起点
            result[node] = self.traverse(
                source=node,
                depth=depth,
                effect_strengths=effect_strengths,
            )
        return result

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_intervenable(node: str, dag: nx.DiGraph) -> bool:
        """节点是否可作为反事实 do() 干预目标。

        - treatment / covariate 节点 → 可干预（业务 DAG 中这两类节点有合法取值）；
        - outcome 节点 → 不可干预（do() outcome 会导致自指悖论）。
        """
        attrs = dag.nodes.get(node, {})
        kind = attrs.get("kind", "")
        return kind in ("treatment", "covariate")

    def _build_path(
        self,
        node_sequence: List[str],
        effect_map: Dict[str, float],
    ) -> CausalPath:
        """构造 CausalPath：算 strength（边效应几何平均）+ intervenable_count。"""
        intervenable = sum(
            1 for n in node_sequence if self._is_intervenable(n, self.dag)
        )
        # strength：沿路径各边 effect 的几何平均（edge effect 用 source 节点 effect 表示）
        strengths: List[float] = []
        for n in node_sequence:
            s = effect_map.get(n, DEFAULT_EFFECT_STRENGTH)
            try:
                s_f = float(s)
            except (TypeError, ValueError):
                s_f = DEFAULT_EFFECT_STRENGTH
            strengths.append(abs(s_f))
        if strengths:
            # 几何平均 = exp(mean(log(x)))；退化处理：全 1 → 1
            import math
            log_sum = sum(math.log(max(s, 1e-9)) for s in strengths)
            strength = math.exp(log_sum / len(strengths))
        else:
            strength = DEFAULT_EFFECT_STRENGTH
        return CausalPath(
            nodes=tuple(node_sequence),
            strength=float(strength),
            length=len(node_sequence),
            intervenable_count=intervenable,
        )


# ----------------------------------------------------------------------------------
# 便捷函数
# ----------------------------------------------------------------------------------


def traverse_causal_paths(
    dag: nx.DiGraph,
    source: str,
    depth: int = DEFAULT_MAX_DEPTH,
    *,
    effect_strengths: Optional[Dict[str, float]] = None,
) -> TraversalResult:
    """从 source 遍历因果路径（PathTraverser 的最简封装）。"""
    return PathTraverser(dag).traverse(
        source=source,
        depth=depth,
        effect_strengths=effect_strengths,
    )


def flatten_traversal(result: TraversalResult) -> List[str]:
    """把 TraversalResult 展平为可读节点列表（带强度 / 子节点 / 可干预标记）。

    返回的每条 dict 字段：
      - node: 节点 id
      - in_paths: 该节点出现在几条路径里
      - max_strength: 该节点相关路径的最大 strength
      - intervenable: 该节点是否可 do()
    """
    flat: Dict[str, Dict[str, Any]] = {}
    for path in result.paths:
        for node in path.nodes:
            entry = flat.setdefault(
                node,
                {
                    "node": node,
                    "in_paths": 0,
                    "max_strength": 0.0,
                    "intervenable": False,
                },
            )
            entry["in_paths"] += 1
            if path.strength > entry["max_strength"]:
                entry["max_strength"] = path.strength
    # 标记可干预性（由 PathTraverser 推断；这里不再依赖 DAG 实例）
    for entry in flat.values():
        # 简化判断：节点以 treatment: / age_band: / admission_route: / complication_count: 开头 → 可干预
        nid = entry["node"]
        entry["intervenable"] = any(
            nid.startswith(prefix)
            for prefix in (
                "treatment:",
                "age_band:",
                "admission_route:",
                "complication_count:",
            )
        )
    return list(flat.values())
