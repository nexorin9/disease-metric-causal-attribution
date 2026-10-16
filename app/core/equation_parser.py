"""SCM 自动生成（spec.md §10「参考地基」+ §11 产品主路径核验）。

本模块把 ref 源产品 ``dowhy/gcm/equation_parser.py`` 的方程解析能力封装到面向
单病种评审场景的 ``SCMParser`` —— 业务模板字符串（如
``"admission_route = empirical()\ncmi = 0.6 * treatment:PCI + norm(loc=0, scale=0.4)"``）
→ ``SCM`` 对象（含 nx.DiGraph 因果图 + 节点因果机制 + SCM 摘要）。

业务侧：
- 调用方传入 ``disease_code`` + ``outcome``，从 ``equation_templates.build_template`` 拿到模板
- 调用 ``SCMParser.parse(template)`` → ``SCM``（graph + mechanism + summary）
- 下游 CounterfactualRunner / RefuterPipeline / BatchRunner 都从 ``SCM.graph`` 拿节点拓扑

源产品能力参考（与源码等价的语法约束）：
- 根节点方程：``<node> = <noise_model>(<args>)``（来源 ``equation_parser.py:48-54``）
- 非根节点方程：``<node> = <expr> + <noise_model>(<args>)``（来源 60-63）
- 未知因果关系：``Node -> <node1>, <node2>``（来源 64-66）
- 安全黑名单：``: ; [ __ import lambda``（来源 78；防注入）
- 节点名仅允许字母数字下划线（来源 29 的 ``_NODE_NAME_PATTERN = r"[a-zA-Z_]\\w*"``）

设计决定：
- ``SCM`` 是包含 ``graph: nx.DiGraph`` + ``causal_mechanisms: Dict[str, NodeMechanism]`` +
  ``summary: SCMSummary`` 的 dataclass。``graph`` 可被下游 do_sample / refuter 复用。
- ``NodeMechanism`` 记录父节点列表、函数表达式、噪声模型名、噪声参数、unknown 标记 —
  与源 ``causal_nodes_info`` 字段对齐（来源 79-115）。
- ``SCMSummary`` 包含：节点数、边数、函数类型分布、噪声分布列表、模板来源 —
  供 spec §5 输出闭环的「SCM 摘要字段」。
- **失败不静默**：未识别噪声模型 / 节点重复 / 含禁字符 / 函数名冲突
  都抛 ``SCMParserError``，不写空 summary 蒙混。

⚠ README 不得写源仓库名（dowhy / pywhy / equation_parser）；仅描述方法族。
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from app.core.dag_builder import DISEASE_OUTCOME_VALUES, SUPPORTED_DISEASE_CODES
from app.core.equation_templates import build_template

LOGGER = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------
# 常量与异常
# ----------------------------------------------------------------------------------

# 与源 equation_parser.py:46 + 78 一致的黑名单
BANNED_CHARACTERS: Tuple[str, ...] = (":", ";", "[", "__", "import", "lambda")

# 与源 equation_parser.py:29 一致的节点名正则
_NODE_NAME_PATTERN = r"[a-zA-Z_]\w*"
NODE_NAME_RE = re.compile(rf"^{_NODE_NAME_PATTERN}$")

# 与源 equation_parser.py:28 一致的噪声模型调用正则
# 形如 "norm(loc=0, scale=1)" 或 "empirical()"
_NOISE_MODEL_PATTERN = r"^\s*([A-Za-z_]\w*)\(([^)]*)\)\s*$"
NOISE_MODEL_RE = re.compile(_NOISE_MODEL_PATTERN)

# 与源 equation_parser.py:30 一致的未知关系正则
_UNKNOWN_MODEL_PATTERN = rf"^\s*\b{_NODE_NAME_PATTERN}(?:\s*,\s*{_NODE_NAME_PATTERN})*\b$"

# 与源 equation_parser.py:23 一致的噪声模型类型
SUPPORTED_NOISE_MODELS: Tuple[str, ...] = (
    "empirical",
    "bayesiangaussianmixture",
    "parametric",
)
# 注：scipy.stats.* 任何分布也允许，但面板摘要只展示 SUPPORTED_NOISE_MODELS 之内的

# 安全字符白名单（用于 _sanitize_input_expression 的补充校验）
_SAFE_CHARS_RE = re.compile(r"^[A-Za-z0-9_+\-*/().,= \t\n:>-]+$")


class SCMParserError(ValueError):
    """SCM 解析失败（节点重复 / 黑名单字符 / 未知噪声模型 / 函数冲突）。

    与 dag_builder.ValueError / pipeline.EstimandNotIdentifiedError 同级别；
    业务层捕获后标 ``scm_parse_failed`` 写入审计日志（spec §6 失败不静默改数据）。
    """


class NoiseModelKind(str, Enum):
    """噪声模型类型（与源 equation_parser.py:23 一致）。"""

    EMPIRICAL = "empirical"
    BAYESIAN_GAUSSIAN_MIXTURE = "bayesiangaussianmixture"
    PARAMETRIC = "parametric"
    SCIPY_DISTRIBUTION = "scipy_distribution"
    UNKNOWN = "unknown"


# ----------------------------------------------------------------------------------
# 数据契约：NodeMechanism / SCMSummary / SCM
# ----------------------------------------------------------------------------------


@dataclass
class NodeMechanism:
    """单个节点的因果机制（与源 causal_nodes_info 字段对齐）。

    Attributes
    ----------
    node_name
        节点 id（如 ``"cmi"`` / ``"treatment:PCI"`` / ``"admission_route"``）
    function_type
        ``"root_noise"`` / ``"additive_noise_model"`` / ``"unknown_relationship"``
    parent_nodes
        父节点列表（升序字母序，与源 ``get_ordered_predecessors`` 一致）
    custom_function
        拟合表达式（如 ``"0.6 * treatment:PCI + (-0.15) * age_band:3"``），
        仅 ``additive_noise_model`` 时非空
    noise_model_name
        噪声模型名（如 ``"norm"`` / ``"empirical"``）
    noise_model_kind
        噪声模型类别（NoiseModelKind 枚举）
    noise_args
        噪声模型参数（如 ``{"loc": 0.0, "scale": 0.4}``）
    fully_defined
        True = 噪声参数已显式给定；False = 仅声明模型名（如 ``empirical()`` 无参数）
    unknown
        True = ``Node -> ...`` 语法，仅定义边不定义函数（来源 103）
    """

    node_name: str
    function_type: str
    parent_nodes: List[str] = field(default_factory=list)
    custom_function: str = ""
    noise_model_name: str = ""
    noise_model_kind: NoiseModelKind = NoiseModelKind.UNKNOWN
    noise_args: Dict[str, Any] = field(default_factory=dict)
    fully_defined: bool = False
    unknown: bool = False


@dataclass
class SCMSummary:
    """SCM 摘要（供面板 SSE 推送 + 草稿生成使用）。

    Attributes
    ----------
    disease_code
    outcome
    template_source
        ``"business_default"`` 表示来自 ``equation_templates.build_template``；
        ``"custom"`` 表示调用方自带模板
    template_hash
        模板字符串 SHA-256（用于审计：同一 SCM 复算得到同一 hash）
    node_count
    edge_count
    parent_count_by_node
        ``{node_name: len(parent_nodes)}`` — 便于面板筛选叶节点
    function_type_distribution
        ``{"root_noise": N, "additive_noise_model": M, "unknown_relationship": K}``
    noise_model_distribution
        ``{"norm": N, "empirical": M, "t": K, ...}``
    parsed_lines
        解析成功的方程行数（不计空行 / 注释行）
    rejected_lines
        解析失败被跳过的行数（健康检查用；非零需人工核对模板）
    """

    disease_code: str = ""
    outcome: str = ""
    template_source: str = "business_default"
    template_hash: str = ""
    node_count: int = 0
    edge_count: int = 0
    parent_count_by_node: Dict[str, int] = field(default_factory=dict)
    function_type_distribution: Dict[str, int] = field(default_factory=dict)
    noise_model_distribution: Dict[str, int] = field(default_factory=dict)
    parsed_lines: int = 0
    rejected_lines: int = 0


@dataclass
class SCM:
    """结构因果模型（与源 ``StructuralCausalModel`` 同构的轻量封装）。

    Attributes
    ----------
    graph
        nx.DiGraph：节点带 ``kind`` / ``label`` / ``color`` / ``spec_section`` 属性；
        边带 ``kind`` / ``label`` 属性（与 dag_builder 的 ``build_dag`` 同结构）
    causal_mechanisms
        ``{node_name: NodeMechanism}`` — 每个节点的因果机制（含噪声模型）
    summary
        SCMSummary 摘要
    """

    graph: nx.DiGraph
    causal_mechanisms: Dict[str, NodeMechanism] = field(default_factory=dict)
    summary: SCMSummary = field(default_factory=SCMSummary)


# ----------------------------------------------------------------------------------
# SCMParser 主类
# ----------------------------------------------------------------------------------


class SCMParser:
    """SCM 解析器。

    Usage
    -----
    >>> parser = SCMParser()
    >>> scm = parser.parse(disease_code="I21", outcome="cmi")  # 用业务默认模板
    >>> scm = parser.parse(template=custom_template, disease_code="I21", outcome="cmi")
    >>> print(scm.summary.node_count, scm.summary.noise_model_distribution)
    """

    def __init__(self) -> None:
        # 函数名黑名单（与源 _allowed_callables 对齐 — 含 numpy/scipy/builtin；
        # 解析时遇到这些名字的 token 不当作 parent node）
        self._builtin_callable_names = {
            "len", "abs", "min", "max", "sum", "mean", "var", "std",
            "norm", "beta", "gamma", "t", "uniform", "exponential",
            "log", "exp", "sqrt", "sin", "cos", "tan", "round", "int",
            "float", "str", "list", "dict", "tuple", "set",
        }

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def parse(
        self,
        disease_code: str,
        outcome: str,
        template: Optional[str] = None,
        *,
        treat_unknown_root_as_empirical: bool = True,
    ) -> SCM:
        """解析方程模板 → SCM。

        Parameters
        ----------
        disease_code
            病种编码（I21 / I63 / Z96.10）
        outcome
            outcome 字符串（cmi / los_days / readmit_flag / complication_flag）
        template
            可选自定义模板；为 None 时从 ``equation_templates.build_template`` 取业务默认
        treat_unknown_root_as_empirical
            True = 父节点未定义时按根节点（EmpiricalDistribution）处理；
            False = 抛 SCMParserError（更严格，测试用）

        Returns
        -------
        SCM
            含 graph / causal_mechanisms / summary 的 SCM 对象

        Raises
        ------
        ValueError
            未知 disease_code / outcome
        SCMParserError
            模板字符串非法（含黑名单字符 / 节点重复 / 未知噪声模型 / 函数名冲突）
        """
        # 当显式提供 template 时跳过业务默认构建（允许任意 disease_code/outcome 字符串）
        # 当 disease_code 或 outcome 为空时强制要求 template
        if template is None:
            if not disease_code:
                raise ValueError(
                    "使用业务默认模板时必须传入 disease_code"
                )
            if not outcome:
                raise ValueError(
                    "使用业务默认模板时必须传入 outcome"
                )
            if disease_code not in SUPPORTED_DISEASE_CODES:
                raise ValueError(
                    f"未知 disease_code: {disease_code!r}；已知: {list(SUPPORTED_DISEASE_CODES)}"
                )
            if outcome not in DISEASE_OUTCOME_VALUES:
                raise ValueError(
                    f"未知 outcome: {outcome!r}；已知: {list(DISEASE_OUTCOME_VALUES)}"
                )
            template = build_template(disease_code, outcome)
            template_source = "business_default"
        else:
            template_source = "custom"

        if not isinstance(template, str) or not template.strip():
            raise SCMParserError("模板字符串为空")

        # 模板 hash 用于审计
        template_hash = hashlib.sha256(template.encode("utf-8")).hexdigest()

        # 解析：每行一条方程
        causal_nodes_info: Dict[str, NodeMechanism] = {}
        graph = nx.DiGraph()
        parsed_lines = 0
        rejected_lines = 0

        for raw_line in template.split("\n"):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                # 空行 / 注释
                continue

            # 安全校验（黑名单字符 / 属性访问）— 安全错误立即抛出，不静默跳过
            # 这是源 equation_parser.py 的契约（78 行 banned_characters + 210-215 sanitize）
            _check_security(line)

            try:
                _check_whitelist(line)
                node_name, expression = _extract_equation_components(line)
                if not NODE_NAME_RE.match(node_name):
                    raise SCMParserError(f"节点名 {node_name!r} 不合法（仅字母/数字/下划线）")
                # 重复检查放在机制创建之前：避免后置 rollback 误删首次合法定义
                if node_name in causal_nodes_info:
                    raise SCMParserError(f"节点 {node_name!r} 重复定义（每节点只能出现一次）")
                # 解析阶段不立即加入 causal_nodes_info / graph；待完整解析后再提交
                pending_node_name = node_name
                pending_added_to_graph = False

                graph.add_node(pending_node_name)
                pending_added_to_graph = True
                mechanism = NodeMechanism(node_name=pending_node_name, function_type="root_noise")
                # 暂时存到局部变量，待完整解析后再 commit 到 causal_nodes_info

                root_node_match = NOISE_MODEL_RE.match(expression)
                unknown_model_match = _check_if_model_is_unknown(line, expression)

                if root_node_match:
                    causal_mechanism_name = root_node_match.group(1)
                    args_str = root_node_match.group(2)
                    parsed_args = _parse_args(args_str)
                    mechanism.function_type = "root_noise"
                    mechanism.noise_model_name = causal_mechanism_name
                    mechanism.noise_model_kind = _classify_noise_model(causal_mechanism_name)
                    mechanism.noise_args = parsed_args
                    mechanism.fully_defined = bool(parsed_args)
                elif unknown_model_match:
                    parent_node_candidates = [
                        p.strip() for p in expression.split(",") if p.strip()
                    ]
                    parent_nodes = self._get_sorted_parent_nodes(parent_node_candidates)
                    for parent in parent_nodes:
                        graph.add_edge(parent, node_name)
                    mechanism.function_type = "unknown_relationship"
                    mechanism.parent_nodes = parent_nodes
                    mechanism.unknown = True
                else:
                    # 非根节点方程：<node> = <expr> + <noise_model>(<args>)
                    custom_func, noise_eq = expression.rsplit("+", 1)
                    custom_func = custom_func.strip()
                    noise_eq = noise_eq.strip()
                    parent_node_candidates = re.findall(
                        _NODE_NAME_PATTERN, custom_func
                    )
                    parent_nodes = self._get_sorted_parent_nodes(parent_node_candidates)
                    for parent in parent_nodes:
                        graph.add_edge(parent, node_name)
                    noise_model_name, parsed_args = _extract_noise_model_components(noise_eq)
                    mechanism.function_type = "additive_noise_model"
                    mechanism.parent_nodes = parent_nodes
                    mechanism.custom_function = custom_func
                    mechanism.noise_model_name = noise_model_name
                    mechanism.noise_model_kind = _classify_noise_model(noise_model_name)
                    mechanism.noise_args = parsed_args
                    mechanism.fully_defined = True
                # 解析完整成功：commit 到 causal_nodes_info
                causal_nodes_info[pending_node_name] = mechanism
                parsed_lines += 1
            except SCMParserError as exc:
                LOGGER.warning("SCM 解析跳过 1 行: %s | line=%r", exc, line)
                # 回滚本行加入的 graph 节点（半成品机制未 commit 进 causal_nodes_info）
                if pending_added_to_graph and graph.has_node(pending_node_name):
                    graph.remove_node(pending_node_name)
                rejected_lines += 1

        # 补全：图中出现但未定义的节点（与源 _add_undefined_nodes_info 等价；
        # treat_unknown_root_as_empirical=True 时补成根节点 EmpiricalDistribution）
        for present_node in list(graph.nodes):
            if present_node not in causal_nodes_info:
                if not treat_unknown_root_as_empirical:
                    raise SCMParserError(
                        f"节点 {present_node!r} 在模板中未定义"
                    )
                LOGGER.warning(
                    "节点 %s 在模板中未定义；按根节点处理（EmpiricalDistribution）",
                    present_node,
                )
                causal_nodes_info[present_node] = NodeMechanism(
                    node_name=present_node,
                    function_type="root_noise",
                    noise_model_name="empirical",
                    noise_model_kind=NoiseModelKind.EMPIRICAL,
                    noise_args={},
                    fully_defined=False,
                )

        # 摘要统计
        summary = SCMSummary(
            disease_code=disease_code,
            outcome=outcome,
            template_source=template_source,
            template_hash=template_hash,
            node_count=graph.number_of_nodes(),
            edge_count=graph.number_of_edges(),
            parent_count_by_node={
                n: len(list(graph.predecessors(n))) for n in graph.nodes
            },
            function_type_distribution=_function_type_distribution(causal_nodes_info),
            noise_model_distribution=_noise_model_distribution(causal_nodes_info),
            parsed_lines=parsed_lines,
            rejected_lines=rejected_lines,
        )

        return SCM(
            graph=graph,
            causal_mechanisms=causal_nodes_info,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # 内部 helper
    # ------------------------------------------------------------------

    def _get_sorted_parent_nodes(self, parent_node_candidates: List[str]) -> List[str]:
        """过滤掉与函数名冲突的候选，按字母序返回真正的父节点列表。

        与源 ``_get_sorted_parent_nodes`` 一致：含 ``logger.warning`` 提示冲突。
        """
        parent_nodes: List[str] = []
        for candidate in parent_node_candidates:
            candidate = candidate.strip()
            if not candidate:
                continue
            if candidate in self._builtin_callable_names:
                LOGGER.warning(
                    "'%s' 与内置函数名冲突；不当作 parent node。"
                    "若需作为父节点，请改名避免冲突。",
                    candidate,
                )
                continue
            parent_nodes.append(candidate)
        parent_nodes = sorted(set(parent_nodes))
        return parent_nodes


# ----------------------------------------------------------------------------------
# 模块级便捷函数
# ------------------------------------------------------------------


def parse_scm(
    disease_code: str,
    outcome: str,
    template: Optional[str] = None,
) -> SCM:
    """一次性调用（SCMParser 的最简封装）。

    等价于::

        parser = SCMParser()
        return parser.parse(disease_code, outcome, template)
    """
    return SCMParser().parse(disease_code, outcome, template)


def parse_custom_template(template: str, disease_code: str = "", outcome: str = "") -> SCM:
    """解析自定义模板（不走业务默认值；disease_code / outcome 仅写入 summary）。"""
    return SCMParser().parse(disease_code=disease_code, outcome=outcome, template=template)


# ----------------------------------------------------------------------------------
# 内部 helper（与源 equation_parser.py 私有函数等价）
# ----------------------------------------------------------------------------------


def _sanitize_input_expression(expression: str, banned_characters: Tuple[str, ...]) -> None:
    """模板字符串安全校验（与源 210-215 等价，向后兼容的合并接口）。

    Raises
    ------
    SCMParserError
        含黑名单字符 / 含 ``..``（防属性访问）/ 含非白名单字符
    """
    _check_security(expression)
    _check_whitelist(expression)


def _check_security(expression: str) -> None:
    """安全校验（黑名单字符 + 属性访问）— 立即抛出 SCMParserError。

    与源 ``_sanitize_input_expression`` 的黑名单 + ``..`` 检测对齐。
    业务层希望安全错误立即失败，而非计入 rejected_lines。
    """
    for char in BANNED_CHARACTERS:
        if char in expression:
            raise SCMParserError(
                f"字符 {char!r} 在表达式中不允许（安全黑名单）"
            )
    if re.search(r"[^0-9]\.[^0-9]", expression):
        raise SCMParserError(
            "字符 '.' 仅允许出现在十进制数字中（防属性访问注入）"
        )


def _check_whitelist(expression: str) -> None:
    """白名单校验（非安全等级）— 失败抛 SCMParserError（被 parse() 计入 rejected_lines）。

    与源 ``_sanitize_input_expression`` 的白名单兜底对齐。
    """
    if not _SAFE_CHARS_RE.match(expression):
        raise SCMParserError(
            f"表达式 {expression!r} 含非白名单字符（仅允许字母/数字/算符/括号/等号/逗号/冒号/箭头/空格）"
        )


def _parse_args(args: str) -> Dict[str, Any]:
    """解析噪声模型参数字符串（与源 126-138 等价）。

    仅支持 keyword args：``"loc=0, scale=1"`` → ``{"loc": 0.0, "scale": 1.0}``
    不支持 positional（与源一致；无数字解析时抛 SCMParserError）。
    """
    kwargs: Dict[str, Any] = {}
    if not args.strip():
        return kwargs
    for arg in args.split(","):
        arg = arg.strip()
        if not arg:
            continue
        if "=" not in arg:
            raise SCMParserError(
                f"噪声参数 {arg!r} 必须使用 keyword 形式（如 'loc=0, scale=1'）"
            )
        key, value = arg.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            # 与源 ast.literal_eval 等价：仅解析字面量
            import ast

            kwargs[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            raise SCMParserError(
                f"无法解析噪声参数 {arg!r}: {exc}"
            ) from exc
    return kwargs


def _extract_equation_components(equation: str) -> Tuple[str, str]:
    """拆解方程：``A -> B, C`` → ``(A, "B, C")``；``A = ...`` → ``(A, "...")``。

    与源 168-175 等价（先 ``->`` 后 ``=``）。
    """
    if "->" in equation:
        node_name, expression = equation.split("->", 1)
    else:
        if "=" not in equation:
            raise SCMParserError(
                f"方程 {equation!r} 缺少 '=' 或 '->'"
            )
        node_name, expression = equation.split("=", 1)
    return node_name.strip(), expression.strip()


def _extract_noise_model_components(noise_eq: str) -> Tuple[str, Dict[str, Any]]:
    """拆解噪声模型表达式（与源 157-165 等价）。"""
    match = NOISE_MODEL_RE.match(noise_eq)
    if not match:
        raise SCMParserError(
            f"无法识别噪声模型 {noise_eq!r}（须形如 'norm(loc=0, scale=1)'）"
        )
    name = match.group(1)
    args_str = match.group(2)
    return name, _parse_args(args_str)


def _check_if_model_is_unknown(equation: str, expression: str) -> bool:
    """判断是否走 ``Node -> <node1>, <node2>`` 未知关系语法（与源 218-222 等价）。"""
    if "->" in equation:
        if re.match(_UNKNOWN_MODEL_PATTERN, expression):
            return True
    return False


def _classify_noise_model(name: str) -> NoiseModelKind:
    """把噪声模型名分类（与源 _STOCHASTIC_MODEL_TYPES + scipy.stats 对齐）。

    已知类别：
      - SUPPORTED_NOISE_MODELS 之一（empirical / bayesiangaussianmixture / parametric）
      - scipy.stats.* 任意合法分布（如 norm / t / beta）
      - 其它：抛 SCMParserError（与源 equation_parser.py:154 一致）
    """
    if name in SUPPORTED_NOISE_MODELS:
        try:
            return NoiseModelKind(name)
        except ValueError:
            pass
    # scipy.stats.* 任意分布
    try:
        import scipy.stats

        if getattr(scipy.stats, name, None) is not None:
            return NoiseModelKind.SCIPY_DISTRIBUTION
    except ImportError:
        pass
    # 未知噪声模型（源 152-154 的行为）
    raise SCMParserError(f"无法识别噪声模型: {name!r}")


def _function_type_distribution(mechanisms: Dict[str, NodeMechanism]) -> Dict[str, int]:
    """汇总 function_type 分布。"""
    dist: Dict[str, int] = {}
    for mech in mechanisms.values():
        dist[mech.function_type] = dist.get(mech.function_type, 0) + 1
    return dist


def _noise_model_distribution(mechanisms: Dict[str, NodeMechanism]) -> Dict[str, int]:
    """汇总噪声模型名分布。"""
    dist: Dict[str, int] = {}
    for mech in mechanisms.values():
        if mech.noise_model_name:
            dist[mech.noise_model_name] = dist.get(mech.noise_model_name, 0) + 1
    return dist