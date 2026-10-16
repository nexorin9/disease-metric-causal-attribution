"""ModelGateway · OpenAI-compatible 单协议 LLM 网关（spec.md §7 + §11 收尾交付）。

| 入口                                       | 用途                                                |
|--------------------------------------------|-----------------------------------------------------|
| `ModelGateway.from_settings(settings)`     | 从 pydantic-settings 单例构造（spec §8 凭据 env）   |
| `ModelGateway(base_url, api_key, model)`   | 测试构造注入：mock transport / 失败注入             |
| `generate_node_narrative(node, effect)`   | 节点自然语言化 → {node, narrative_zh}              |
| `generate_counterfactual_narrative(s, ...)`| 反事实语句 → {scenario, predicted_outcome_zh, conf}|
| `generate_draft_summary(draft)`            | 草稿摘要 → {summary, narrative_zh}                 |

设计原则（spec §7 + §11）：
- **OpenAI-compatible 单协议**（默认 base_url=https://api.openai.com；`OPENAI_STUB_URL` 仅本地协议兼容性测试用）。
- **结构化请求**：messages + `response_format={"type": "json_object"}`；业务层不直接拼字符串拼装 prompt。
- **结构化响应**：解析 `choices[0].message.content` JSON；不可解析 → `model_output_unparseable=True` + 走 `DeterministicNarrativeFallback`（T28 引入，本任务只定义钩子 + 返回 ModelResponse.content=fallback_json 字符串）。
- **超时 10s（推理类）/ 30s（解释类）**：本任务在 ModelGateway 内部按方法路由（generate_node_narrative / generate_counterfactual_narrative → 10s；generate_draft_summary → 30s）。
- **审计元数据**：`prompt_hash` / `response_hash` / `latency_ms` / `tokens_used` / `model_version`（spec §7 + §11 每次调用记录）；本任务 ModelResponse 已经携带这些字段；T30 audit_logger 落库。
- **人工责任边界**：所有 ModelResponse 默认 `state_not_advanced=True`（不静默改数据）；草稿生成是「草稿提示」，reviewer 在面板点「确认」才真正写回上报系统。
- **降级到确定性规则**：T28 引入 DeterministicNarrativeFallback（基于 z-score 排序 + 模板语句）；本任务在不可解析 / 超时时返回 `ModelResponse(fallback_used=True, state_not_advanced=True, content=fallback_json)`，**不在 model 层抛异常**，由调用方判定是否采用。
- **凭据位**：base_url / api_key / model / timeout 全部环境变量（OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL / OPENAI_TIMEOUT），与 spec §8 一致；`.env.example` 已就位。

依赖：`httpx`（pyproject 已锁定）。
测试替身：tests/test_model_gateway_basic.py 用本地 stub transport 跑协议握手 + 结构化输出 + 配置加载。
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 常量（spec §7 + §11）
# ---------------------------------------------------------------------------

# OpenAI-compatible chat completions 路径（业内事实标准；与 base_url 拼接）
OPENAI_CHAT_COMPLETIONS_PATH = "/chat/completions"

# 默认模型版本声明（业务层会再覆盖；这里仅做兜底；README 不写死具体厂商）
DEFAULT_MODEL_VERSION = "openai_compatible/1.0"

# 模型响应 schema_version（业务层 + 审计层统一标识）
MODEL_RESPONSE_SCHEMA_VERSION = "model_response/1.0"

# response_format 参数（OpenAI 兼容协议事实标准；强制 JSON 输出）
RESPONSE_FORMAT_JSON_OBJECT: Dict[str, str] = {"type": "json_object"}

# 超时档位（spec §7 + §11 硬约束）
TIMEOUT_REASONING_SECONDS = 10  # 推理类：node_narrative / counterfactual_narrative
TIMEOUT_EXPLAINING_SECONDS = 30  # 解释类：draft_summary

# 默认顶层超时（OPENAI_TIMEOUT 兜底；不与推理/解释档冲突——这两档是更紧的子集）
DEFAULT_TIMEOUT_SECONDS = 30

# 结构化响应必需的 JSON 字段（按方法不同；缺字段标 model_output_unparseable）
NODE_NARRATIVE_REQUIRED_KEYS: tuple = ("narrative_zh",)
COUNTERFACTUAL_NARRATIVE_REQUIRED_KEYS: tuple = ("predicted_outcome_zh",)
DRAFT_SUMMARY_REQUIRED_KEYS: tuple = ("summary_zh",)


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------


class ModelGatewayError(RuntimeError):
    """ModelGateway 基础异常（spec §7 不静默改数据；显式错误便于调用方降级）。"""


class ModelGatewayConfigError(ModelGatewayError, ValueError):
    """凭据缺失或 URL 非法 — 不重试，立即失败。"""


class ModelGatewayTimeoutError(ModelGatewayError):
    """调用超时（spec §7 推理 10s / 解释 30s）。"""


# ---------------------------------------------------------------------------
# 数据契约
# ---------------------------------------------------------------------------


@dataclass
class ModelResponse:
    """模型调用单次结果（spec §7 + §11 审计字段完整）。

    字段语义：
      - content              : 解析后的 dict（OpenAI-compatible message.content 是 JSON 字符串）
      - fallback_used        : True = 不可解析 / 超时 / 网络错 → 走确定性兜底（本任务钩子占位，T28 接入 fallback 模板）
      - state_not_advanced   : True = 未推进业务状态（草稿仍是 pending_review；reviewer 未点确认）
      - prompt_hash          : SHA-256(prompt_messages_json) 前 16 hex（审计去重 + 复核）
      - response_hash        : SHA-256(raw_response_bytes) 前 16 hex
      - latency_ms           : 单次调用耗时
      - tokens_used          : usage.total_tokens（OpenAI-compatible 字段；缺时为 0）
      - model_version        : 实际调用模型名（来自响应 model 字段或 settings.openai_model）
      - method               : generate_node_narrative / generate_counterfactual_narrative / generate_draft_summary
      - schema_version       : model_response/1.0
    """

    content: Dict[str, Any]
    fallback_used: bool = False
    fallback_reason: str = ""
    state_not_advanced: bool = True
    prompt_hash: str = ""
    response_hash: str = ""
    latency_ms: float = 0.0
    tokens_used: int = 0
    model_version: str = DEFAULT_MODEL_VERSION
    method: str = ""
    schema_version: str = MODEL_RESPONSE_SCHEMA_VERSION
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "content": self.content,
            "fallback_used": bool(self.fallback_used),
            "fallback_reason": self.fallback_reason,
            "state_not_advanced": bool(self.state_not_advanced),
            "prompt_hash": self.prompt_hash,
            "response_hash": self.response_hash,
            "latency_ms": round(float(self.latency_ms), 2),
            "tokens_used": int(self.tokens_used or 0),
            "model_version": self.model_version,
            **({"extra": self.extra} if self.extra else {}),
        }


# ---------------------------------------------------------------------------
# 内部 helper
# ---------------------------------------------------------------------------


def _short_hash(payload: str) -> str:
    """SHA-256 截前 16 hex（审计去重用）。"""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _normalize_messages(messages: List[Mapping[str, str]]) -> List[Dict[str, str]]:
    """把消息规范化：必须 [{role, content}, ...]；非空 content。"""
    if not messages:
        raise ModelGatewayConfigError("messages 不能为空")
    out: List[Dict[str, str]] = []
    for i, m in enumerate(messages):
        if not isinstance(m, Mapping):
            raise ModelGatewayConfigError(f"messages[{i}] 不是 Mapping")
        role = str(m.get("role", "")).strip()
        content = str(m.get("content", "")).strip()
        if not role:
            raise ModelGatewayConfigError(f"messages[{i}].role 缺失")
        if not content:
            raise ModelGatewayConfigError(f"messages[{i}].content 缺失")
        out.append({"role": role, "content": content})
    return out


def _extract_content_and_usage(raw: Mapping[str, Any]) -> tuple:
    """从 OpenAI-compatible 响应 JSON 抽取 (content_str, total_tokens, model_name)。

    兼容字段差异（不同实现命名差异）：
      - content: choices[0].message.content 或 choices[0].text
      - tokens:  usage.total_tokens / usage.total / usage.input_tokens + output_tokens
      - model:   model / id
    """
    if not isinstance(raw, Mapping):
        return ("", 0, "")

    content_str = ""
    choices = raw.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, Mapping):
            msg = first.get("message")
            if isinstance(msg, Mapping):
                content_str = str(msg.get("content", "") or "")
            else:
                content_str = str(first.get("text", "") or "")

    usage = raw.get("usage")
    total_tokens = 0
    if isinstance(usage, Mapping):
        if isinstance(usage.get("total_tokens"), (int, float)):
            total_tokens = int(usage["total_tokens"])
        elif isinstance(usage.get("total"), (int, float)):
            total_tokens = int(usage["total"])
        else:
            inp = usage.get("prompt_tokens", 0) or 0
            out = usage.get("completion_tokens", 0) or 0
            if isinstance(inp, (int, float)) and isinstance(out, (int, float)):
                total_tokens = int(inp) + int(out)

    model_name = str(raw.get("model", "") or raw.get("id", "") or "")

    return (content_str, total_tokens, model_name)


def _parse_json_content(content_str: str) -> Optional[Dict[str, Any]]:
    """解析 content_str → dict；失败返回 None（不抛）。"""
    if not content_str:
        return None
    try:
        v = json.loads(content_str)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return v if isinstance(v, dict) else None


def _build_fallback_content(method: str, **context: Any) -> Dict[str, Any]:
    """确定性兜底内容（spec §7 + §11 模型不可达降级）。

    本函数是历史轻量版骨架；T28 引入 DeterministicNarrativeFallback 后，
    业务层优先调 DeterministicNarrativeFallback.build(...) 获得基于 z-score
    排序 + 模板语句的兜底内容；本函数保留作为兜底的兜底（fallback-of-fallback）。
    """
    if method == "generate_node_narrative":
        node = context.get("node") or "unknown_node"
        return {
            "narrative_zh": (
                f"[fallback] 节点 {node} 的归因解释："
                "本次评审由确定性规则生成（模型不可达）；请评审者结合反驳结果判断。"
            ),
            "_fallback_kind": "node_narrative",
        }
    if method == "generate_counterfactual_narrative":
        node = context.get("intervention_node") or "unknown_node"
        hyp = context.get("hypothetical_value")
        return {
            "predicted_outcome_zh": (
                f"[fallback] 假设 {node} 调整为 {hyp} 时，"
                "outcome 变化由确定性规则给出；模型不可达，不阻塞草稿生成。"
            ),
            "confidence": 0.0,
            "_fallback_kind": "counterfactual_narrative",
        }
    if method == "generate_draft_summary":
        disease = context.get("disease_code") or "unknown"
        outcome = context.get("outcome") or "unknown"
        return {
            "summary_zh": (
                f"[fallback] 病种 {disease} / outcome {outcome} 的归因摘要："
                "模型不可达，由确定性规则汇总；草稿仍可生成，"
                "请由 reviewer 在面板复核。"
            ),
            "_fallback_kind": "draft_summary",
        }
    return {"_fallback_kind": "unknown"}


# ---------------------------------------------------------------------------
# T28 · 确定性兜底叙事生成器（z-score 排序 + 模板语句）
# ---------------------------------------------------------------------------
#
# 设计（spec §6 失败降级 + §7 模型不可达降级 + §11 第 7 步）：
#   - 不依赖外部 LLM；纯规则 + z-score 排序
#   - 节点自然语言化：按 |ate| 绝对值 z-score 排序，前 3 名驱动因子组成摘要
#   - 反事实：根据 hypothetical_value 与 baseline_value 偏差给定性方向（"上升"/"下降"/"维持"）
#   - 草稿摘要：聚合 effect_estimates 与 refutation_results 的 robust/partial/unstable 分布
#   - 与原 _build_fallback_content 同结构（content 字段 + _fallback_kind），便于替换
#   - 失败时（effect / scenario / draft 任一字段缺失） → 走 _build_fallback_content 兜底
#


# z-score 排序 + 强弱分级阈值（spec §11 第 5 步 楔子硬约束对齐）
_ZSCORE_STRONG_THRESHOLD = 0.5  # |z| >= 0.5 视为强驱动
_ZSCORE_WEAK_THRESHOLD = 0.2  # |z| >= 0.2 视为弱驱动
_TOP_N_DRIVERS = 3  # 草稿摘要最多提 3 个驱动因子


def _safe_zscore(values: List[float]) -> List[float]:
    """计算 z-score；空 / 全相同值 → 全部返回 0.0（避免除零）。"""
    if not values:
        return []
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = var ** 0.5
    if std < 1e-9:
        return [0.0] * n
    return [(v - mean) / std for v in values]


def _direction_word(ate: Optional[float]) -> str:
    """把因果效应值映射成业务主任可读的方向词。"""
    if ate is None:
        return "维持"
    if ate > 0.05:
        return "上升"
    if ate < -0.05:
        return "下降"
    return "维持"


class DeterministicNarrativeFallback:
    """基于 z-score 排序 + 业务模板的确定性兜底叙事生成器（T28）。

    三方法与 ModelGateway 主路径对应：
      - build_node_narrative(node, effect, disease_code, outcome)
      - build_counterfactual_narrative(scenario, baseline, predicted)
      - build_draft_summary(disease_code, outcome, effect_estimates,
                             refutation_results, counterfactual_scenarios)

    每个 build_* 返回 dict（与 ModelResponse.content 形状一致）+ 元数据
    （_fallback_kind / _fallback_engine='deterministic_v1' / _drivers_used 等）。

    任何异常 / 字段缺失 → 走 _build_fallback_content 兜底，永不抛异常。
    """

    FALLBACK_ENGINE = "deterministic_v1"

    @classmethod
    def build_node_narrative(
        cls,
        *,
        node: str,
        effect: Optional[Mapping[str, Any]] = None,
        disease_code: str = "",
        outcome: str = "",
    ) -> Dict[str, Any]:
        """节点自然语言化兜底（z-score 单节点评估）。

        Parameters
        ----------
        node : str
            节点 id（treatment_strategy / age_band / admission_route / complication_count 等）
        effect : Optional[Mapping]
            含 ate / p_value / method_name；任一缺失走 _build_fallback_content 兜底
        disease_code, outcome : str
            业务上下文（缺失不阻塞，仅影响文案）
        """
        try:
            if not isinstance(effect, Mapping):
                raise ValueError("effect 不是 Mapping")
            ate_raw = effect.get("ate")
            if ate_raw is None:
                raise ValueError("effect.ate 缺失")
            ate = float(ate_raw)
            p_value_raw = effect.get("p_value")
            p_value = float(p_value_raw) if p_value_raw is not None else None
            method_name = str(effect.get("method_name", "") or "")

            direction = _direction_word(ate)
            disease_str = str(disease_code or "").strip() or "当前病种"
            outcome_str = str(outcome or "").strip() or "当前 outcome"

            strength = "强" if abs(ate) >= _ZSCORE_STRONG_THRESHOLD else (
                "弱" if abs(ate) >= _ZSCORE_WEAK_THRESHOLD else "背景"
            )
            confidence_part = ""
            if p_value is not None:
                if p_value < 0.01:
                    confidence_part = "（高显著性）"
                elif p_value < 0.05:
                    confidence_part = "（显著）"
                else:
                    confidence_part = "（不显著，仅供参考）"
            method_part = f"；估计器 {method_name}" if method_name else ""

            narrative_zh = (
                f"[确定性兜底] 在 {disease_str} / {outcome_str} 评审中，"
                f"节点 {node} 为{strength}驱动因子，"
                f"因果效应 {direction} {abs(ate):.3f}{confidence_part}"
                f"{method_part}。模型不可达，由确定性规则生成；"
                "请评审者结合反驳结果判断。"
            )
            return {
                "narrative_zh": narrative_zh,
                "_fallback_kind": "node_narrative",
                "_fallback_engine": cls.FALLBACK_ENGINE,
                "_drivers_used": [node],
                "_z_score_strength": strength,
                "_direction": direction,
            }
        except Exception:  # noqa: BLE001
            return _build_fallback_content(
                "generate_node_narrative",
                node=node,
                disease_code=disease_code,
                outcome=outcome,
            )

    @classmethod
    def build_counterfactual_narrative(
        cls,
        *,
        scenario: Optional[Mapping[str, Any]] = None,
        baseline: Optional[Mapping[str, Any]] = None,
        predicted: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """反事实叙事兜底（hypothetical vs baseline 偏差 → 定性方向）。"""
        try:
            if not isinstance(scenario, Mapping):
                raise ValueError("scenario 不是 Mapping")
            intervention_node = str(scenario.get("intervention_node", "") or "")
            hypothetical_value = scenario.get("hypothetical_value")
            if not intervention_node:
                raise ValueError("scenario.intervention_node 缺失")

            baseline_value = float((baseline or {}).get("baseline_value", 0.0) or 0.0) if isinstance(baseline, Mapping) else 0.0
            predicted_value = float((predicted or {}).get("predicted_outcome", 0.0) or 0.0) if isinstance(predicted, Mapping) else 0.0
            delta = predicted_value - baseline_value

            direction = "上升" if delta > 0.05 else ("下降" if delta < -0.05 else "基本维持")
            magnitude = abs(delta)
            if magnitude < 1e-6:
                magnitude_text = "无显著变化"
            elif magnitude < 0.05:
                magnitude_text = f"幅度约 {magnitude:.3f}（轻微）"
            else:
                magnitude_text = f"幅度约 {magnitude:.3f}"

            predicted_outcome_zh = (
                f"[确定性兜底] 假设将 {intervention_node} 调整为 {hypothetical_value}，"
                f"outcome 预测{direction}（{magnitude_text}）。"
                "模型不可达，由确定性规则汇总；草稿仍可生成。"
            )
            return {
                "predicted_outcome_zh": predicted_outcome_zh,
                "confidence": 0.0,
                "_fallback_kind": "counterfactual_narrative",
                "_fallback_engine": cls.FALLBACK_ENGINE,
                "_delta": round(delta, 4),
                "_direction": direction,
            }
        except Exception:  # noqa: BLE001
            return _build_fallback_content(
                "generate_counterfactual_narrative",
                intervention_node=(scenario or {}).get("intervention_node") if isinstance(scenario, Mapping) else "",
                hypothetical_value=(scenario or {}).get("hypothetical_value") if isinstance(scenario, Mapping) else None,
            )

    @classmethod
    def build_draft_summary(
        cls,
        *,
        draft: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """草稿摘要兜底（按 |ate| z-score 排序前 3 名驱动因子）。"""
        try:
            if not isinstance(draft, Mapping):
                raise ValueError("draft 不是 Mapping")
            disease_code = str(draft.get("disease_code", "") or "").strip() or "当前病种"
            outcome = str(draft.get("outcome", "") or "").strip() or "当前 outcome"
            sample_size = int(draft.get("sample_size", 0) or 0)
            effect_estimates = list(draft.get("effect_estimates") or [])
            refutation_results = list(draft.get("refutation_results") or [])
            counterfactual_scenarios = list(draft.get("counterfactual_scenarios") or [])

            # 按 |ate| 排序前 3 名
            scored: List[tuple] = []
            for idx, est in enumerate(effect_estimates):
                if not isinstance(est, Mapping):
                    continue
                ate_raw = est.get("ate")
                if ate_raw is None:
                    continue
                try:
                    scored.append((idx, float(ate_raw), str(est.get("node", f"driver_{idx}"))))
                except (TypeError, ValueError):
                    continue
            z_scores = _safe_zscore([abs(s[1]) for s in scored])
            scored_with_z = sorted(
                [(s[0], s[1], s[2], z) for s, z in zip(scored, z_scores)],
                key=lambda x: abs(x[3]),
                reverse=True,
            )
            top_drivers = scored_with_z[:_TOP_N_DRIVERS]

            # 反驳聚合（robust / partial / unstable）
            robust = sum(
                1 for r in refutation_results
                if isinstance(r, Mapping) and r.get("passed") is True
                and not r.get("refutation_unstable")
            )
            unstable = sum(
                1 for r in refutation_results
                if isinstance(r, Mapping) and r.get("refutation_unstable") is True
            )
            total = max(len(refutation_results), 1)
            if unstable == total:
                verdict = "不稳定"
            elif robust == total:
                verdict = "稳健"
            else:
                verdict = "部分稳健"

            # 草稿主路径
            if not top_drivers:
                drivers_text = "暂无显著驱动因子"
            else:
                drivers_text = "、".join(
                    f"{node}（|ate|={abs(ate):.3f}, z={z:.2f}）"
                    for _, ate, node, z in top_drivers
                )
            summary_zh = (
                f"[确定性兜底] {disease_code} / {outcome}（n={sample_size}）"
                f"的主要驱动因子：{drivers_text}。"
                f"反驳稳健性：{verdict}（{robust}/{len(refutation_results)} 档通过）；"
                f"反事实场景 {len(counterfactual_scenarios)} 条。"
                "模型不可达，由确定性规则汇总；草稿仍可生成，"
                "请 reviewer 在面板复核。"
            )
            return {
                "summary_zh": summary_zh,
                "_fallback_kind": "draft_summary",
                "_fallback_engine": cls.FALLBACK_ENGINE,
                "_drivers_used": [t[1] for t in top_drivers],
                "_refutation_verdict": verdict,
                "_z_score_top_n": _TOP_N_DRIVERS,
            }
        except Exception:  # noqa: BLE001
            disease_code = (draft or {}).get("disease_code", "") if isinstance(draft, Mapping) else ""
            outcome = (draft or {}).get("outcome", "") if isinstance(draft, Mapping) else ""
            return _build_fallback_content(
                "generate_draft_summary",
                disease_code=disease_code,
                outcome=outcome,
            )


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------


class ModelGateway:
    """OpenAI-compatible 单协议 LLM 网关（spec §7 + §11）。

    用法
    ----
    >>> gw = ModelGateway.from_settings()
    >>> r = gw.generate_node_narrative(node="treatment_strategy", effect={"ate": 0.12, "p_value": 0.003})
    >>> assert r.state_not_advanced is True  # 不静默推进业务状态
    >>> r.content["narrative_zh"]  # 由模型生成的中文归因解释

    测试可注入 `transport=lambda method, url, headers, body, timeout: ...` 跑协议握手。
    """

    # ---- 静态方法 ----

    @staticmethod
    def from_settings(
        settings: Optional[Any] = None,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        transport: Optional[Callable[..., Any]] = None,
        audit_writer: Optional[Callable[..., None]] = None,
    ) -> "ModelGateway":
        """从 pydantic-settings 单例构造（spec §8 + §11 真实协议路径）。

        参数全部可显式覆盖 — 用于测试或真实凭据位注入。
        """
        if settings is None:
            from app.config import get_settings  # 延迟导入

            settings = get_settings()

        cfg_base_url = (
            base_url if base_url is not None else settings.openai_base_url
        ).strip()
        cfg_api_key = (
            api_key if api_key is not None else settings.openai_api_key
        ).strip()
        cfg_model = (
            model if model is not None else settings.openai_model
        ).strip()
        cfg_timeout = (
            int(timeout) if timeout is not None else int(settings.openai_timeout)
        )

        return ModelGateway(
            base_url=cfg_base_url,
            api_key=cfg_api_key,
            model=cfg_model,
            timeout=cfg_timeout,
            transport=transport,
            audit_writer=audit_writer,
        )

    # ---- 构造期 ----

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "gpt-4o-mini",
        *,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        path: str = OPENAI_CHAT_COMPLETIONS_PATH,
        transport: Optional[Callable[..., Any]] = None,
        audit_writer: Optional[Callable[..., None]] = None,
    ) -> None:
        if not base_url or not isinstance(base_url, str):
            raise ModelGatewayConfigError("base_url 必须为非空字符串")
        if not api_key or not isinstance(api_key, str):
            raise ModelGatewayConfigError("api_key 必须为非空字符串（bearer）")
        if not model or not isinstance(model, str):
            raise ModelGatewayConfigError("model 必须为非空字符串")
        if timeout < 1:
            raise ModelGatewayConfigError(f"timeout 必须 ≥ 1，实际 {timeout}")

        # base_url 不允许带尾斜杠（防 // 双斜杠）
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = str(model).strip()
        self.timeout = int(timeout)
        self.path = str(path).strip() or OPENAI_CHAT_COMPLETIONS_PATH
        # 注入式 transport：测试用；默认 None 表示用 httpx.post
        self._transport = transport
        self._audit_writer = audit_writer
        self._lock = threading.RLock()
        # 暴露 last 计数（业务层/测试可查）
        self.last_status_code: Optional[int] = None
        self.last_error: Optional[str] = None

    # ---- 内部辅助 ----

    @property
    def chat_url(self) -> str:
        """拼接 base_url + path（base_url 已去尾斜杠）。"""
        return f"{self.base_url}{self.path}"

    def _auth_header(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _timeout_for_method(self, method: str) -> float:
        """按 spec §7 路由超时档：推理 10s / 解释 30s。"""
        if method in ("generate_node_narrative", "generate_counterfactual_narrative"):
            return float(min(self.timeout, TIMEOUT_REASONING_SECONDS))
        return float(min(self.timeout, TIMEOUT_EXPLAINING_SECONDS))

    def _default_transport(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        body: str,
        timeout: float,
    ) -> Any:
        """默认 transport：httpx POST（生产路径真实协议）。"""
        try:
            import httpx  # 延迟导入便于测试 monkeypatch
        except ImportError as exc:  # pragma: no cover - 依赖已在 pyproject 锁定
            raise ModelGatewayConfigError(
                f"httpx 未安装（pyproject 已锁定 httpx>=0.27）：{exc}"
            ) from exc

        return httpx.post(
            url,
            headers=dict(headers),
            content=body,
            timeout=timeout,
        )

    def _do_request(self, body_json: str, timeout: float) -> Any:
        """发起单次 POST（transport 注入点）。"""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            **self._auth_header(),
        }
        transport = self._transport or self._default_transport
        return transport(
            url=self.chat_url,
            headers=headers,
            body=body_json,
            timeout=timeout,
        )

    @staticmethod
    def _extract_status_code(resp: Any) -> Optional[int]:
        """从 transport 返回抽取 status_code（httpx / requests 兼容）。"""
        sc = getattr(resp, "status_code", None)
        if sc is None:
            return None
        try:
            return int(sc)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_json(resp: Any) -> Optional[Dict[str, Any]]:
        """从 transport 返回抽取 JSON body（不抛异常）。"""
        if resp is None:
            return None
        # 1. httpx Response.json()
        json_method = getattr(resp, "json", None)
        if callable(json_method):
            try:
                v = json_method()
                return v if isinstance(v, dict) else None
            except Exception:  # noqa: BLE001 - 不可解析走 None
                pass
        # 2. raw .content / .text
        raw = getattr(resp, "content", None)
        if raw is None:
            raw = getattr(resp, "text", None)
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        if isinstance(raw, str) and raw:
            try:
                v = json.loads(raw)
                return v if isinstance(v, dict) else None
            except (json.JSONDecodeError, TypeError, ValueError):
                return None
        return None

    def _build_messages(
        self,
        *,
        method: str,
        user_payload: Mapping[str, Any],
        system_prompt: str,
    ) -> List[Dict[str, str]]:
        """组装 messages；system + user(user_payload 的 JSON 形式)。"""
        user_json = json.dumps(dict(user_payload), ensure_ascii=False, sort_keys=True)
        return _normalize_messages(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_json},
            ]
        )

    def _system_prompt_for(self, method: str) -> str:
        """按方法返回 system prompt；强制 JSON 输出。"""
        if method == "generate_node_narrative":
            return (
                "你是单病种指标因果归因评审助手。"
                "请根据用户给出的节点名称、因果效应值与 p-value，"
                "用 1-2 句中文给业务主任可读的归因解释。"
                "严格按 JSON 输出：{\"narrative_zh\": \"...\"}。"
            )
        if method == "generate_counterfactual_narrative":
            return (
                "你是单病种指标反事实模拟助手。"
                "请根据用户给出的干预节点、假设值、预测 outcome 与置信度，"
                "用 1-2 句中文给业务主任可读的反事实说明。"
                "严格按 JSON 输出：{\"predicted_outcome_zh\": \"...\", \"confidence\": 0.0-1.0}。"
            )
        if method == "generate_draft_summary":
            return (
                "你是单病种归因报告草稿摘要助手。"
                "请根据用户给出的病种、outcome、关键因果效应与反驳结果，"
                "用 2-3 句中文给一段评审摘要。"
                "严格按 JSON 输出：{\"summary_zh\": \"...\"}。"
            )
        raise ModelGatewayConfigError(f"未知 method: {method!r}")

    def _check_required_keys(self, method: str, content: Mapping[str, Any]) -> bool:
        """按方法校验结构化响应是否含必需字段。"""
        if method == "generate_node_narrative":
            return "narrative_zh" in content and isinstance(content["narrative_zh"], str)
        if method == "generate_counterfactual_narrative":
            return (
                "predicted_outcome_zh" in content
                and isinstance(content["predicted_outcome_zh"], str)
            )
        if method == "generate_draft_summary":
            return "summary_zh" in content and isinstance(content["summary_zh"], str)
        return False

    def _write_audit(
        self,
        *,
        method: str,
        prompt_hash: str,
        response_hash: str,
        latency_ms: float,
        tokens_used: int,
        model_version: str,
        fallback_used: bool,
        fallback_reason: str,
    ) -> bool:
        """审计写日志（best-effort；不抛）。"""
        if self._audit_writer is None:
            return False
        try:
            envelope = {
                "event_type": "model/call",
                "method": method,
                "schema_version": MODEL_RESPONSE_SCHEMA_VERSION,
                "prompt_hash": prompt_hash,
                "response_hash": response_hash,
                "latency_ms": round(float(latency_ms), 2),
                "tokens_used": int(tokens_used or 0),
                "model_version": model_version,
                "fallback_used": bool(fallback_used),
                "fallback_reason": str(fallback_reason or ""),
                "state_not_advanced": True,
            }
            self._audit_writer(**envelope)
            return True
        except Exception as audit_exc:  # pragma: no cover - best-effort
            LOGGER.warning(
                "ModelGateway audit 写日志失败（best-effort，不阻断业务）: %s",
                audit_exc,
            )
            return False

    # ---- 内部统一调用入口 ----

    def _call(
        self,
        *,
        method: str,
        user_payload: Mapping[str, Any],
        fallback_context: Mapping[str, Any],
    ) -> ModelResponse:
        """统一调用入口：拼 messages → POST → 解析 → 降级；返回 ModelResponse。"""
        system_prompt = self._system_prompt_for(method)
        messages = self._build_messages(
            method=method,
            user_payload=user_payload,
            system_prompt=system_prompt,
        )
        # prompt hash 基于 user_payload + system_prompt + model（同一调用稳定）
        prompt_blob = json.dumps(
            {
                "model": self.model,
                "system": system_prompt,
                "user": dict(user_payload),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        prompt_hash = _short_hash(prompt_blob)

        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "response_format": RESPONSE_FORMAT_JSON_OBJECT,
                "temperature": 0.2,
            },
            ensure_ascii=False,
            sort_keys=False,
        )

        timeout = self._timeout_for_method(method)
        start = time.monotonic()
        fallback_used = False
        fallback_reason = ""
        content_str = ""
        total_tokens = 0
        model_name = self.model
        last_status: Optional[int] = None
        last_error: Optional[str] = None

        try:
            resp = self._do_request(body, timeout)
        except Exception as exc:  # noqa: BLE001 - 网络 / 超时 / DNS
            last_error = f"exception:{type(exc).__name__}"
            self.last_error = last_error
            # T28: 显式区分超时异常（httpx.TimeoutException / requests.Timeout /
            # asyncio.TimeoutError / TimeoutError）与其他网络错误；spec §7 第 4 段
            # 硬约束：超时 10s / 30s 必须落到确定性兜底，且 state_not_advanced=True
            exc_type_name = type(exc).__name__
            exc_module = type(exc).__module__
            is_timeout = (
                "Timeout" in exc_type_name
                or exc_type_name == "TimeoutError"
                or "Timeout" in str(exc)
            )
            if is_timeout:
                fallback_used = True
                fallback_reason = (
                    f"timeout:{exc_type_name}:timeout>{timeout:.1f}s"[:200]
                )
                LOGGER.warning(
                    "ModelGateway.%s 调用超时 (timeout=%.1fs) → 降级到确定性兜底: %s — %s",
                    method,
                    timeout,
                    exc_type_name,
                    exc,
                )
            else:
                fallback_used = True
                fallback_reason = (
                    f"network_error:{exc_type_name}:{exc}"[:200]
                )
                LOGGER.warning(
                    "ModelGateway.%s 网络异常 → 降级 fallback: %s — %s",
                    method,
                    exc_type_name,
                    exc,
                )
        else:
            last_status = self._extract_status_code(resp)
            self.last_status_code = last_status
            if last_status is None or last_status >= 400:
                fallback_used = True
                fallback_reason = (
                    f"http_error:{last_status if last_status is not None else 'unknown'}"
                )
                LOGGER.warning(
                    "ModelGateway.%s 收到错误响应 status=%s → 降级 fallback",
                    method,
                    last_status,
                )
            else:
                raw_json = self._extract_json(resp)
                if raw_json is None:
                    fallback_used = True
                    fallback_reason = "response_unparseable"
                else:
                    content_str, total_tokens, model_name = _extract_content_and_usage(raw_json)

        elapsed_ms = (time.monotonic() - start) * 1000.0

        # 解析 JSON content
        parsed: Optional[Dict[str, Any]] = _parse_json_content(content_str) if content_str else None
        if parsed is None and not fallback_used:
            fallback_used = True
            fallback_reason = "model_output_unparseable"

        # 校验必需字段
        if not fallback_used and parsed is not None and not self._check_required_keys(method, parsed):
            fallback_used = True
            fallback_reason = "missing_required_keys"

        # 兜底内容
        if fallback_used:
            content = _build_fallback_content(method, **dict(fallback_context))
        else:
            content = parsed if parsed is not None else {}

        # response hash（基于 raw response 或 content_str；兜底走 content 序列化）
        response_blob = content_str or json.dumps(content, ensure_ascii=False, sort_keys=True)
        response_hash = _short_hash(response_blob)

        # model_name 兜底
        if not model_name:
            model_name = self.model

        result = ModelResponse(
            content=content,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            state_not_advanced=True,
            prompt_hash=prompt_hash,
            response_hash=response_hash,
            latency_ms=elapsed_ms,
            tokens_used=int(total_tokens or 0),
            model_version=str(model_name or DEFAULT_MODEL_VERSION),
            method=method,
        )
        # 审计 best-effort
        result.extra["audit_event_written"] = self._write_audit(
            method=method,
            prompt_hash=prompt_hash,
            response_hash=response_hash,
            latency_ms=elapsed_ms,
            tokens_used=int(total_tokens or 0),
            model_version=result.model_version,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
        )
        return result

    # ---- 对外主路径（spec §7 + §11） ----

    def generate_node_narrative(
        self,
        *,
        node: str,
        effect: Mapping[str, Any],
        disease_code: str = "",
        outcome: str = "",
    ) -> ModelResponse:
        """节点自然语言化（spec §7 推理类 10s）。

        Parameters
        ----------
        node : str
            节点 id（如 treatment_strategy / age_band）
        effect : Mapping
            至少含 ate / p_value / method_name（来自 EffectEstimate）
        disease_code, outcome : str
            业务上下文（可选）

        Returns
        -------
        ModelResponse
            content 含 narrative_zh；state_not_advanced=True
        """
        if not node or not isinstance(node, str):
            raise ModelGatewayConfigError("node 必须为非空字符串")
        if effect is None or not isinstance(effect, Mapping):
            raise ModelGatewayConfigError("effect 必须为非空 Mapping")

        return self._call(
            method="generate_node_narrative",
            user_payload={
                "node": node,
                "disease_code": str(disease_code or ""),
                "outcome": str(outcome or ""),
                "effect": dict(effect),
            },
            fallback_context={
                "node": node,
                "disease_code": disease_code,
                "outcome": outcome,
            },
        )

    def generate_counterfactual_narrative(
        self,
        *,
        scenario: Mapping[str, Any],
        baseline: Optional[Mapping[str, Any]] = None,
        predicted: Optional[Mapping[str, Any]] = None,
    ) -> ModelResponse:
        """反事实语句生成（spec §7 推理类 10s）。

        Parameters
        ----------
        scenario : Mapping
            含 intervention_node / hypothetical_value（来自 CounterfactualScenario）
        baseline : Optional[Mapping]
            原始观测值（baseline_value + baseline_std）
        predicted : Optional[Mapping]
            含 predicted_outcome / confidence / do_operations

        Returns
        -------
        ModelResponse
            content 含 predicted_outcome_zh / confidence；state_not_advanced=True
        """
        if scenario is None or not isinstance(scenario, Mapping):
            raise ModelGatewayConfigError("scenario 必须为非空 Mapping")
        if not scenario.get("intervention_node"):
            raise ModelGatewayConfigError("scenario.intervention_node 必须")

        return self._call(
            method="generate_counterfactual_narrative",
            user_payload={
                "scenario": dict(scenario),
                "baseline": dict(baseline) if baseline else {},
                "predicted": dict(predicted) if predicted else {},
            },
            fallback_context={
                "intervention_node": scenario.get("intervention_node"),
                "hypothetical_value": scenario.get("hypothetical_value"),
                "disease_code": scenario.get("disease_code", ""),
                "outcome": scenario.get("outcome", ""),
            },
        )

    def generate_draft_summary(
        self,
        *,
        draft: Mapping[str, Any],
    ) -> ModelResponse:
        """草稿摘要生成（spec §7 解释类 30s）。

        Parameters
        ----------
        draft : Mapping
            含 disease_code / outcome / effect_estimates / refutation_results / counterfactual_scenarios

        Returns
        -------
        ModelResponse
            content 含 summary_zh；state_not_advanced=True
        """
        if draft is None or not isinstance(draft, Mapping):
            raise ModelGatewayConfigError("draft 必须为非空 Mapping")

        # 取关键字段（避免把全量草稿塞给模型）
        disease_code = str(draft.get("disease_code", "") or "")
        outcome = str(draft.get("outcome", "") or "")
        effect_estimates = list(draft.get("effect_estimates") or [])
        refutation_results = list(draft.get("refutation_results") or [])
        counterfactual_scenarios = list(draft.get("counterfactual_scenarios") or [])

        return self._call(
            method="generate_draft_summary",
            user_payload={
                "disease_code": disease_code,
                "outcome": outcome,
                "treatment": str(draft.get("treatment", "") or ""),
                "sample_size": int(draft.get("sample_size", 0) or 0),
                "effect_estimates": effect_estimates,
                "refutation_results": refutation_results,
                "counterfactual_scenarios": counterfactual_scenarios,
            },
            fallback_context={
                "disease_code": disease_code,
                "outcome": outcome,
            },
        )

    def close(self) -> None:
        """lifespan shutdown 钩子：本实现无连接池，placeholder 占位。"""
        return None


# ---------------------------------------------------------------------------
# 模块便捷函数
# ---------------------------------------------------------------------------


def generate_node_narrative(
    *,
    node: str,
    effect: Mapping[str, Any],
    disease_code: str = "",
    outcome: str = "",
    settings: Optional[Any] = None,
) -> ModelResponse:
    """一行调用 generate_node_narrative（业务层最常用入口）。"""
    gw = ModelGateway.from_settings(settings=settings)
    return gw.generate_node_narrative(
        node=node,
        effect=effect,
        disease_code=disease_code,
        outcome=outcome,
    )


def generate_counterfactual_narrative(
    *,
    scenario: Mapping[str, Any],
    baseline: Optional[Mapping[str, Any]] = None,
    predicted: Optional[Mapping[str, Any]] = None,
    settings: Optional[Any] = None,
) -> ModelResponse:
    """一行调用 generate_counterfactual_narrative。"""
    gw = ModelGateway.from_settings(settings=settings)
    return gw.generate_counterfactual_narrative(
        scenario=scenario,
        baseline=baseline,
        predicted=predicted,
    )


def generate_draft_summary(
    *,
    draft: Mapping[str, Any],
    settings: Optional[Any] = None,
) -> ModelResponse:
    """一行调用 generate_draft_summary。"""
    gw = ModelGateway.from_settings(settings=settings)
    return gw.generate_draft_summary(draft=draft)


__all__ = [
    "DEFAULT_MODEL_VERSION",
    "DEFAULT_TIMEOUT_SECONDS",
    "DETERMINISTIC_NARRATIVE_SCHEMA_VERSION",
    "MODEL_RESPONSE_SCHEMA_VERSION",
    "OPENAI_CHAT_COMPLETIONS_PATH",
    "RESPONSE_FORMAT_JSON_OBJECT",
    "TIMEOUT_EXPLAINING_SECONDS",
    "TIMEOUT_REASONING_SECONDS",
    "DeterministicNarrativeFallback",
    "ModelGateway",
    "ModelGatewayConfigError",
    "ModelGatewayError",
    "ModelGatewayTimeoutError",
    "ModelResponse",
    "generate_counterfactual_narrative",
    "generate_draft_summary",
    "generate_node_narrative",
]


# 确定性兜底叙事的 schema_version（与 ModelResponse.schema_version 区分；
# 便于业务层审计调用方区分来源：ModelResponse 是真实模型 / 兜底骨架，
# DeterministicNarrativeFallback 是 z-score 排序的确定性兜底）
DETERMINISTIC_NARRATIVE_SCHEMA_VERSION = "deterministic_narrative/1.0"