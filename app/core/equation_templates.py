"""业务 SCM 模板字符串（spec.md §10「参考地基」+ §11 产品主路径核验）。

本模块按 ref 源产品 ``gcm/equation_parser.py`` 的方程语法，为 3 个病种 × 4 个 outcome
预定义业务模板字符串。每条模板遵循源产品支持的 2 种语法：

1. 根节点方程（噪声先验）::

     <node> = <noise_model>(<args>)

   例：``treatment_strategy = empirical()``

2. 非根节点方程（结构因果机制）::

     <node> = <function_expression> + <noise_model>(<args>)

   例：``cmi = 0.6 * treatment_strategy + (-0.15) * age_band + norm(loc=0, scale=0.4)``

注意：本模板**不**使用 ref 源 ``Node -> <node1>, <node2>`` 第三种语法，因为业务 DAG
中根节点之间无强因果耦合（治疗与合并症数相关性由 fixture 数据驱动估计，不需要在
SCM 模板里硬编码）。

模板设计原则：
- 根节点是 4 个**聚合**列名（与 fixture 列对齐）：treatment_strategy / age_band /
  admission_route / complication_count。每个根节点作为一个 SCM 节点 — 而非按取值
  拆分（``treatment:PCI`` / ``treatment:thrombolysis``），原因是 ref 源 ``equation_parser.py``
  禁 ``:`` 字符（来源 78 的 ``banned_characters``），且 DAG 拆分节点是面板拓扑
  用途，与 SCM 因果机制无关。两者解耦：DAG 走 ``dag_builder.build_dag``；
  SCM 走本模板。**这是有意识的架构分离**：DAG 面板展示拆分节点，SCM 走聚合方程。
- outcome 节点方程按业务因果强度系数组装；系数符号反映业务假设（治疗 → 主效应、
  年龄/入院途径 → 弱效应、合并症 → 强效应）。
- 噪声分布：根节点用 ``empirical()``（直方图估计）；outcome 用 ``norm(loc=0, scale=...)``
  （带可调尺度因子，便于 benchmark 注入扰动）。
- 模板遵循源 ``equation_parser.py:218-222`` 的 ``_NOISE_MODEL_PATTERN``：
  ``^\s*([\w]+)\(([^)]*)\)\s*$``（只允许 ASCII 字母/数字节点名）。

源产品能力参考：
- 模板语法源自 ``github_ref/dowhy/dowhy/gcm/equation_parser.py:44-77`` 的
  ``create_causal_model_from_equations``；本模块是改写后的业务模板入口。
- 业务 DAG 拓扑来自 ``app/core/dag_builder.py``（3 病种 × 4 outcome 节点/边）。
- 与 ``equation_parser.py`` 的 ``banned_characters = [":", ";", "[", "__", "import", "lambda"]``
  保持一致 — 不引入新字符集。

⚠ README 不得写源仓库名（dowhy / pywhy / equation_parser）；仅描述方法族。
"""
from __future__ import annotations

from typing import Dict, List


# 业务根节点方程（噪声先验）。3 病种共用同一组根节点定义。
# 根节点 = fixture 聚合列名（与 dag_builder 的 TREATMENT_VALUES / AGE_BAND_VALUES 等
# 枚举一致）；每个根节点在 SCM 中是一个独立变量。
ROOT_NODE_EQUATIONS: List[str] = [
    "treatment_strategy = empirical()",
    "age_band = empirical()",
    "admission_route = empirical()",
    "complication_count = empirical()",
]


# ----------------------------------------------------------------------------------
# outcome 节点方程：按 disease_code × outcome 给出结构因果机制
# ----------------------------------------------------------------------------------

# 系数符号反映业务因果假设（与 spec.md §11 因果链一致）：
# - treatment_strategy → CMI：正向（介入治疗病种 CMI 通常高于保守）
# - age_band → CMI：负向（高龄并发症多、CMI 折扣）
# - admission_route → CMI：负向（急诊入院病情重、CMI 折扣）
# - complication_count → CMI：正向（合并症多 → CMI 提升）
# - los_days：与各因子影响相对均衡
# - readmit_flag / complication_flag：treatment 是主驱动，协变量是弱驱动

CMI_EQUATIONS: Dict[str, str] = {
    "I21": "cmi = 0.6 * treatment_strategy + (-0.15) * age_band + (-0.20) * admission_route + 0.30 * complication_count + norm(loc=0, scale=0.4)",
    "I63": "cmi = 0.5 * treatment_strategy + (-0.18) * age_band + (-0.15) * admission_route + 0.32 * complication_count + norm(loc=0, scale=0.4)",
    "Z96.10": "cmi = 0.7 * treatment_strategy + (-0.10) * age_band + 0.05 * admission_route + 0.25 * complication_count + norm(loc=0, scale=0.4)",
}

LOS_DAYS_EQUATIONS: Dict[str, str] = {
    "I21": "los_days = 0.4 * treatment_strategy + 0.20 * age_band + 0.30 * admission_route + 0.50 * complication_count + norm(loc=2, scale=1.2)",
    "I63": "los_days = 0.5 * treatment_strategy + 0.25 * age_band + 0.20 * admission_route + 0.45 * complication_count + norm(loc=3, scale=1.5)",
    "Z96.10": "los_days = 0.6 * treatment_strategy + 0.15 * age_band + (-0.10) * admission_route + 0.40 * complication_count + norm(loc=5, scale=2.0)",
}

READMIT_FLAG_EQUATIONS: Dict[str, str] = {
    "I21": "readmit_flag = 0.5 * treatment_strategy + 0.20 * age_band + 0.25 * admission_route + 0.40 * complication_count + norm(loc=0, scale=0.5)",
    "I63": "readmit_flag = 0.45 * treatment_strategy + 0.25 * age_band + 0.20 * admission_route + 0.50 * complication_count + norm(loc=0, scale=0.5)",
    "Z96.10": "readmit_flag = 0.40 * treatment_strategy + 0.30 * age_band + 0.10 * admission_route + 0.45 * complication_count + norm(loc=0, scale=0.5)",
}

COMPLICATION_FLAG_EQUATIONS: Dict[str, str] = {
    "I21": "complication_flag = 0.45 * treatment_strategy + 0.20 * age_band + 0.30 * admission_route + 0.55 * complication_count + norm(loc=0, scale=0.5)",
    "I63": "complication_flag = 0.50 * treatment_strategy + 0.25 * age_band + 0.25 * admission_route + 0.60 * complication_count + norm(loc=0, scale=0.5)",
    "Z96.10": "complication_flag = 0.55 * treatment_strategy + 0.20 * age_band + 0.10 * admission_route + 0.50 * complication_count + norm(loc=0, scale=0.5)",
}


# 业务 outcome 节点方程聚合表（disease_code × outcome → 方程字符串）
OUTCOME_EQUATIONS: Dict[str, Dict[str, str]] = {
    "cmi": CMI_EQUATIONS,
    "los_days": LOS_DAYS_EQUATIONS,
    "readmit_flag": READMIT_FLAG_EQUATIONS,
    "complication_flag": COMPLICATION_FLAG_EQUATIONS,
}


# ----------------------------------------------------------------------------------
# 模板组装（业务主入口）
# ----------------------------------------------------------------------------------


def build_template(disease_code: str, outcome: str) -> str:
    """按 disease_code + outcome 组装完整的方程模板字符串。

    模板包含：
      - ROOT_NODE_EQUATIONS（4 条根节点方程）
      - outcome 的结构因果方程（按 disease_code 选）

    Returns
    -------
    str
        多行方程字符串（每行一条方程），可直接喂给 ``SCMParser.parse()``。
        与源 ``create_causal_model_from_equations(node_equations)`` 输入格式一致。

    Raises
    ------
    ValueError
        未知 disease_code 或 outcome（与 dag_builder.py 的 DiseaseOutcome 枚举对齐）
    """
    from app.core.dag_builder import DISEASE_OUTCOME_VALUES, SUPPORTED_DISEASE_CODES

    if disease_code not in SUPPORTED_DISEASE_CODES:
        raise ValueError(
            f"未知 disease_code: {disease_code!r}；已知: {list(SUPPORTED_DISEASE_CODES)}"
        )
    if outcome not in DISEASE_OUTCOME_VALUES:
        raise ValueError(
            f"未知 outcome: {outcome!r}；已知: {list(DISEASE_OUTCOME_VALUES)}"
        )

    outcome_eq_map = OUTCOME_EQUATIONS.get(outcome)
    if outcome_eq_map is None:
        raise ValueError(f"outcome {outcome!r} 无模板")

    outcome_equation = outcome_eq_map.get(disease_code)
    if outcome_equation is None:
        raise ValueError(
            f"disease_code={disease_code} outcome={outcome} 无模板"
        )

    lines: List[str] = []
    lines.extend(ROOT_NODE_EQUATIONS)
    lines.append(outcome_equation)
    return "\n".join(lines)


def list_templates() -> List[Dict[str, str]]:
    """列出所有 (disease_code, outcome) 模板（供面板 / 健康检查展示）。"""
    from app.core.dag_builder import DISEASE_OUTCOME_VALUES, SUPPORTED_DISEASE_CODES

    items: List[Dict[str, str]] = []
    for d in SUPPORTED_DISEASE_CODES:
        for o in DISEASE_OUTCOME_VALUES:
            items.append(
                {
                    "disease_code": d,
                    "outcome": o,
                    "template_preview": _preview_template(d, o),
                }
            )
    return items


def _preview_template(disease_code: str, outcome: str) -> str:
    """模板预览（仅取 outcome 方程行，不含根节点，便于面板展示）。"""
    outcome_eq_map = OUTCOME_EQUATIONS.get(outcome, {})
    eq = outcome_eq_map.get(disease_code, "")
    return eq