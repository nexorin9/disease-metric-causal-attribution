"""业务因果 DAG 模板（spec.md §11 因果图 + §3 节点候选）。

为 3 个病种 × 4 个 outcome 预定义业务 DAG：
  - 节点候选：treatment_strategy / age_band / admission_route / complication_count
  - outcome 节点：cmi / los_days / readmit_flag / complication_flag

颜色编码（供 Plotly.js 面板按 effect 大小上色）：
  - 驱动因子（红 / strong）：treatment_strategy → outcome
  - 弱驱动（黄 / weak）：age_band / admission_route → outcome
  - 背景（灰 / background）：其它边

实现要点：
- 使用 networkx.DiGraph（与源产品 CausalGraph / do_sampler 共用数据结构）
- build_dag(disease_code, outcome) → nx.DiGraph；outcome 不在已知集 → ValueError
- 3 病种共享同一 DAG 骨架结构，但治疗方式枚举随 disease_code 变化
- DiseaseOutcome 枚举显式约束 outcome 字符串集合

源产品能力参考：nx.DiGraph 数据结构与 dot 转换（与 CausalGraph 同构）；
详见 spec.md §10「参考地基」。
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Tuple

import networkx as nx

# ----------------------------------------------------------------------------------
# 枚举与常量
# ----------------------------------------------------------------------------------

# 已知疾病编码（与 data/seed_loader.py FIXTURE_FILES 对齐；spec §3.1）
SUPPORTED_DISEASE_CODES: Tuple[str, ...] = ("I21", "I63", "Z96.10")


class DiseaseOutcome(str, Enum):
    """业务 outcome 枚举（spec.md §3.1 outcome 候选 + §2 触发场景）。

    字段值与 HIS 出院病案首页列名严格一致，便于直接用于 pd.DataFrame 列。
    """

    CMI = "cmi"
    LOS_DAYS = "los_days"
    READMIT_FLAG = "readmit_flag"
    COMPLICATION_FLAG = "complication_flag"


# outcome 字符串值常量（供面板 / 校验使用）
DISEASE_OUTCOME_VALUES: Tuple[str, ...] = tuple(o.value for o in DiseaseOutcome)


# 治疗方式枚举按 disease_code 区分（spec §3.1 by_disease_code 段）
TREATMENT_VALUES: Dict[str, List[str]] = {
    "I21": ["PCI", "thrombolysis", "conservative"],            # 急性心梗
    "I63": ["thrombolysis", "thrombectomy", "conservative"],   # 脑梗
    "Z96.10": ["total_hip_replacement", "revision", "conservative"],  # 髋置换
}

# 入院途径枚举（spec §3.1 admission_route；与 field_mapping.json 对齐）
ADMISSION_ROUTE_VALUES: Tuple[str, ...] = ("1", "2", "3", "9")

# 年龄段离散区间（与 fixture 一致：1=18-44 / 2=45-64 / 3=65-74 / 4=75+）
AGE_BAND_VALUES: Tuple[int, ...] = (1, 2, 3, 4)

# 节点角色 → 颜色编码（面板上色 + 反驳后高亮）
NODE_COLOR_STRONG = "#d62728"   # 强驱动（红）
NODE_COLOR_WEAK = "#ff7f0e"     # 弱驱动（黄）
NODE_COLOR_BACKGROUND = "#7f7f7f"  # 背景（灰）
NODE_COLOR_OUTCOME = "#1f77b4"  # outcome（蓝）

EDGE_COLOR_STRONG = "#d62728"
EDGE_COLOR_WEAK = "#ff7f0e"
EDGE_COLOR_BACKGROUND = "#cccccc"


# ----------------------------------------------------------------------------------
# DAG 构造
# ----------------------------------------------------------------------------------


def _outcome_node_meta(outcome: DiseaseOutcome) -> Dict[str, str]:
    """outcome 节点的元数据（label / semantic / spec_section）。"""
    semantic_map = {
        DiseaseOutcome.CMI: "病例组合指数",
        DiseaseOutcome.LOS_DAYS: "实际住院天数",
        DiseaseOutcome.READMIT_FLAG: "30 天再入院",
        DiseaseOutcome.COMPLICATION_FLAG: "并发症发生",
    }
    return {
        "label": outcome.value,
        "kind": "outcome",
        "color": NODE_COLOR_OUTCOME,
        "semantic": semantic_map[outcome],
        "spec_section": "§3.1 outcome 候选",
    }


def _treatment_node_meta(disease_code: str) -> Dict[str, List[Dict[str, str]]]:
    """治疗方式节点的元数据（按 disease_code 枚举展开）。"""
    treatments = TREATMENT_VALUES[disease_code]
    return {
        "values": [
            {
                "label": t,
                "kind": "treatment",
                "color": NODE_COLOR_STRONG,
                "semantic": f"{disease_code} 治疗方式: {t}",
                "spec_section": "§3.1 by_disease_code.treatment_strategy",
            }
            for t in treatments
        ]
    }


def _build_node_attributes(disease_code: str, outcome: DiseaseOutcome) -> List[Tuple[str, Dict[str, str]]]:
    """构造 DAG 节点列表（含属性）。"""
    nodes: List[Tuple[str, Dict[str, str]]] = []

    # 治疗方式节点（每病种 3 个）
    treatment_meta = _treatment_node_meta(disease_code)
    for t_node in treatment_meta["values"]:
        nodes.append((f"treatment:{t_node['label']}", t_node))

    # 年龄段节点（4 个）
    for band in AGE_BAND_VALUES:
        nodes.append(
            (
                f"age_band:{band}",
                {
                    "label": f"age_band:{band}",
                    "kind": "covariate",
                    "color": NODE_COLOR_WEAK,
                    "semantic": f"年龄段 {band}",
                    "spec_section": "§3.1 age_band 必填",
                },
            )
        )

    # 入院途径节点（4 个）
    for route in ADMISSION_ROUTE_VALUES:
        nodes.append(
            (
                f"admission_route:{route}",
                {
                    "label": f"admission_route:{route}",
                    "kind": "covariate",
                    "color": NODE_COLOR_WEAK,
                    "semantic": f"入院途径 {route}",
                    "spec_section": "§3.1 admission_route 必填",
                },
            )
        )

    # 合并症数量节点（按 encounter 离散计 0/1/2+）
    for comp in (0, 1, 2):
        nodes.append(
            (
                f"complication_count:{comp}",
                {
                    "label": f"complication_count:{comp}",
                    "kind": "covariate",
                    "color": NODE_COLOR_WEAK,
                    "semantic": f"合并症数 {comp}",
                    "spec_section": "§3.1 secondary_diagnosis_list 派生",
                },
            )
        )

    # outcome 节点
    nodes.append((outcome.value, _outcome_node_meta(outcome)))

    return nodes


def _build_edge_attributes(disease_code: str, outcome: DiseaseOutcome) -> List[Tuple[str, str, Dict[str, str]]]:
    """构造 DAG 边列表（含属性）。

    主边：treatment → outcome（强驱动，红色）
    协边：age_band / admission_route / complication_count → outcome（弱驱动，黄色）
    内部：treatment ↔ complication_count（治疗与合并症相关，背景灰）
    """
    edges: List[Tuple[str, str, Dict[str, str]]] = []
    out = outcome.value

    # treatment → outcome（每个治疗枚举一条）
    for t in TREATMENT_VALUES[disease_code]:
        edges.append(
            (
                f"treatment:{t}",
                out,
                {
                    "kind": "treatment_to_outcome",
                    "color": EDGE_COLOR_STRONG,
                    "label": "treatment→outcome",
                    "spec_section": "§3.1 treatment_strategy → outcome",
                },
            )
        )

    # age_band → outcome（每个年龄段一条）
    for band in AGE_BAND_VALUES:
        edges.append(
            (
                f"age_band:{band}",
                out,
                {
                    "kind": "covariate_to_outcome",
                    "color": EDGE_COLOR_WEAK,
                    "label": "age_band→outcome",
                    "spec_section": "§3.1 age_band 必填",
                },
            )
        )

    # admission_route → outcome
    for route in ADMISSION_ROUTE_VALUES:
        edges.append(
            (
                f"admission_route:{route}",
                out,
                {
                    "kind": "covariate_to_outcome",
                    "color": EDGE_COLOR_WEAK,
                    "label": "admission_route→outcome",
                    "spec_section": "§3.1 admission_route 必填",
                },
            )
        )

    # complication_count → outcome
    for comp in (0, 1, 2):
        edges.append(
            (
                f"complication_count:{comp}",
                out,
                {
                    "kind": "covariate_to_outcome",
                    "color": EDGE_COLOR_WEAK,
                    "label": "complication_count→outcome",
                    "spec_section": "§3.1 secondary_diagnosis_list 派生",
                },
            )
        )

    # treatment ↔ complication_count（治疗选择与合并症相关；背景灰）
    for t in TREATMENT_VALUES[disease_code]:
        for comp in (0, 1, 2):
            edges.append(
                (
                    f"treatment:{t}",
                    f"complication_count:{comp}",
                    {
                        "kind": "background",
                        "color": EDGE_COLOR_BACKGROUND,
                        "label": "treatment↔complication",
                        "spec_section": "§11 因果图结构",
                    },
                )
            )

    return edges


def build_dag(disease_code: str, outcome: str | DiseaseOutcome) -> nx.DiGraph:
    """按 disease_code 与 outcome 构建业务因果 DAG。

    Parameters
    ----------
    disease_code : str
        主诊断 ICD-10 前缀；当前支持 'I21' / 'I63' / 'Z96.10'
    outcome : str | DiseaseOutcome
        目标 outcome；与 DiseaseOutcome 枚举值一致（cmi / los_days / readmit_flag / complication_flag）

    Returns
    -------
    nx.DiGraph
        带节点 / 边属性（kind / color / label / semantic / spec_section）的 DAG

    Raises
    ------
    ValueError
        未知 disease_code 或 outcome
    """
    if disease_code not in SUPPORTED_DISEASE_CODES:
        raise ValueError(
            f"未知 disease_code: {disease_code!r}；"
            f"已知: {list(SUPPORTED_DISEASE_CODES)}"
        )
    if isinstance(outcome, str):
        try:
            outcome_enum = DiseaseOutcome(outcome)
        except ValueError as exc:
            raise ValueError(
                f"未知 outcome: {outcome!r}；"
                f"已知: {[o.value for o in DiseaseOutcome]}"
            ) from exc
    elif isinstance(outcome, DiseaseOutcome):
        outcome_enum = outcome
    else:
        raise TypeError(
            f"outcome 必须是 str 或 DiseaseOutcome；got {type(outcome).__name__}"
        )

    g = nx.DiGraph()
    g.graph["disease_code"] = disease_code
    g.graph["outcome"] = outcome_enum.value

    for node_id, attrs in _build_node_attributes(disease_code, outcome_enum):
        g.add_node(node_id, **attrs)

    for src, dst, attrs in _build_edge_attributes(disease_code, outcome_enum):
        g.add_edge(src, dst, **attrs)

    return g


def outcome_node(g: nx.DiGraph) -> str:
    """从 DAG 中提取 outcome 节点 id（=outcome_value 字段）。"""
    outcome = g.graph.get("outcome")
    if outcome is None:
        raise ValueError("DAG 未设置 outcome 字段；非业务 DAG")
    return outcome


def treatment_nodes(g: nx.DiGraph) -> List[str]:
    """提取所有 treatment 节点 id（kind=treatment）。"""
    return [n for n, attrs in g.nodes(data=True) if attrs.get("kind") == "treatment"]


def covariate_nodes(g: nx.DiGraph) -> List[str]:
    """提取所有 covariate 节点 id（kind=covariate）。"""
    return [n for n, attrs in g.nodes(data=True) if attrs.get("kind") == "covariate"]


def dag_summary(g: nx.DiGraph) -> Dict[str, int]:
    """DAG 摘要：节点 / 边 / 各 kind 计数。"""
    kinds: Dict[str, int] = {}
    for _, attrs in g.nodes(data=True):
        kind = attrs.get("kind", "unknown")
        kinds[kind] = kinds.get(kind, 0) + 1
    return {
        "node_count": g.number_of_nodes(),
        "edge_count": g.number_of_edges(),
        "kinds": kinds,
        "disease_code": g.graph.get("disease_code"),
        "outcome": g.graph.get("outcome"),
    }
