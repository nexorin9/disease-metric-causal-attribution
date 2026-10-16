"""test_model_gateway_degradation — ModelGateway 降级与兜底路径验证（T28）。

覆盖维度（spec §6 + §7 + §11）：

1. **超时降级**：显式 timeout 异常 → fallback_used=True + reason 含 "timeout:" 前缀 +
   走确定性兜底（narrative_zh / predicted_outcome_zh / summary_zh 仍可用）。
2. **不可解析降级**：HTTP 200 但 content 不是 JSON → fallback_used=True +
   reason="response_unparseable" / "model_output_unparseable"；业务层草稿可生成。
3. **DeterministicNarrativeFallback（z-score 排序模板）**：
   - build_node_narrative：强/弱/背景驱动因子 + ate 正负 + p_value 显著性档位
   - build_counterfactual_narrative：delta 正负 → 上升/下降/维持
   - build_draft_summary：|ate| z-score 排序前 3 名 + 反驳 robust/partial/unstable 聚合
4. **Fallback-of-fallback**：effect / scenario / draft 任一字段缺失 →
   _build_fallback_content 兜底（带 [fallback] 前缀 + narrative_zh 仍可用）
5. **Audit 元数据完整性**：prompt_hash / response_hash / latency_ms /
   tokens_used / model_version + state_not_advanced=True 全部就位。
6. **state_not_advated=True 永远**：3 方法 × 4 降级路径 parametrize 全部断言。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

# 让 `from app.xxx import ...` 工作
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("APP_ENV", "test")

from app.adapters.model_gateway import (  # noqa: E402
    DETERMINISTIC_NARRATIVE_SCHEMA_VERSION,
    MODEL_RESPONSE_SCHEMA_VERSION,
    ModelGateway,
    ModelResponse,
    DeterministicNarrativeFallback,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _StubResp:
    """OpenAI-compatible 替身响应对象（httpx.Response 兼容：status_code + json() + content）。"""

    def __init__(self, status_code: int, body: Dict[str, Any]) -> None:
        self.status_code = int(status_code)
        self._body = body
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.content = raw
        self.text = raw.decode("utf-8")

    def json(self) -> Dict[str, Any]:
        return self._body


class _TimeoutException(Exception):
    """模拟 httpx.TimeoutException / requests.Timeout 的测试替身。"""

    def __init__(self, msg: str = "Read timed out"):
        super().__init__(msg)


def _openai_compat_response(
    *,
    content_obj: Dict[str, Any],
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    return {
        "id": "chatcmpl-test-stub",
        "object": "chat.completion",
        "created": 1700000000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps(content_obj, ensure_ascii=False),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


# ---------------------------------------------------------------------------
# 1. 超时降级
# ---------------------------------------------------------------------------


class TestTimeoutFallback:
    def test_timeout_exception_falls_back_with_timeout_reason(self):
        """transport 抛超时异常 → fallback_used=True + reason 含 'timeout:' 前缀。"""

        def timeout_transport(**kwargs):
            raise _TimeoutException("Read timed out")

        gw = ModelGateway("https://x", "k", "m", transport=timeout_transport)
        r = gw.generate_node_narrative(node="treatment_strategy", effect={"ate": 0.12})
        assert r.fallback_used is True
        assert "timeout:" in r.fallback_reason
        assert "TimeoutException" in r.fallback_reason
        # 业务层仍能拿到 narrative_zh（确定性兜底）
        assert "narrative_zh" in r.content
        assert r.state_not_advanced is True

    def test_httpx_style_timeout_subclass_detected(self):
        """httpx.TimeoutException 子类（含模块名 'httpx'）也走 timeout 降级路径。"""

        class _HttpxLikeTimeout(_TimeoutException):
            pass

        def timeout_transport(**kwargs):
            raise _HttpxLikeTimeout("timeout")

        gw = ModelGateway("https://x", "k", "m", transport=timeout_transport)
        r = gw.generate_node_narrative(node="x", effect={"ate": 0.0})
        assert r.fallback_used is True
        assert "timeout:" in r.fallback_reason

    def test_timeout_for_counterfactual_method(self):
        """反事实方法（10s 档）的超时也走降级路径。"""

        def timeout_transport(**kwargs):
            raise _TimeoutException("timeout")

        gw = ModelGateway("https://x", "k", "m", transport=timeout_transport)
        r = gw.generate_counterfactual_narrative(
            scenario={"intervention_node": "treatment_strategy", "hypothetical_value": "PCI"}
        )
        assert r.fallback_used is True
        assert "timeout:" in r.fallback_reason
        assert r.state_not_advanced is True

    def test_timeout_for_draft_summary_method(self):
        """草稿摘要方法（30s 档）的超时也走降级路径。"""

        def timeout_transport(**kwargs):
            raise _TimeoutException("timeout")

        gw = ModelGateway("https://x", "k", "m", transport=timeout_transport)
        r = gw.generate_draft_summary(draft={"disease_code": "I21", "outcome": "cmi"})
        assert r.fallback_used is True
        assert "timeout:" in r.fallback_reason
        assert r.state_not_advanced is True

    def test_timeout_distinguished_from_other_network_errors(self):
        """超时与一般网络异常走不同 fallback_reason。"""
        # 超时
        def timeout_transport(**kwargs):
            raise _TimeoutException("Read timed out")

        gw = ModelGateway("https://x", "k", "m", transport=timeout_transport)
        r1 = gw.generate_node_narrative(node="x", effect={"ate": 0.0})
        assert "timeout:" in r1.fallback_reason

        # 普通网络错
        def network_transport(**kwargs):
            raise ConnectionError("dns-fail")

        gw2 = ModelGateway("https://x", "k", "m", transport=network_transport)
        r2 = gw2.generate_node_narrative(node="x", effect={"ate": 0.0})
        assert "network_error:" in r2.fallback_reason
        assert "timeout:" not in r2.fallback_reason


# ---------------------------------------------------------------------------
# 2. 不可解析降级
# ---------------------------------------------------------------------------


class TestUnparseableFallback:
    def test_http_200_with_non_json_content_falls_back(self):
        """HTTP 200 但 content 不是 JSON → 降级到兜底 narrative_zh。"""
        stub = _StubResp(200, {"choices": [{"message": {"content": "not-json"}}]})

        class _T:
            def __call__(self, **kwargs):
                return stub

        gw = ModelGateway("https://x", "k", "m", transport=_T())
        r = gw.generate_node_narrative(node="x", effect={"ate": 0.0})
        assert r.fallback_used is True
        assert r.fallback_reason in ("response_unparseable", "model_output_unparseable")
        assert "narrative_zh" in r.content
        assert r.state_not_advanced is True

    def test_http_500_falls_back(self):
        stub = _StubResp(500, {"error": "internal"})

        class _T:
            def __call__(self, **kwargs):
                return stub

        gw = ModelGateway("https://x", "k", "m", transport=_T())
        r = gw.generate_node_narrative(node="x", effect={"ate": 0.0})
        assert r.fallback_used is True
        assert "http_error" in r.fallback_reason
        assert "500" in r.fallback_reason

    def test_missing_required_keys_falls_back(self):
        """响应 JSON 解析成功但缺必需字段 → 降级。"""
        body = _openai_compat_response(content_obj={"wrong_key": "value"})
        stub = _StubResp(200, body)

        class _T:
            def __call__(self, **kwargs):
                return stub

        gw = ModelGateway("https://x", "k", "m", transport=_T())
        r = gw.generate_node_narrative(node="x", effect={"ate": 0.0})
        assert r.fallback_used is True
        assert r.fallback_reason == "missing_required_keys"
        assert "narrative_zh" in r.content

    def test_fallback_content_has_narrative_zh_for_node_method(self):
        """节点方法 fallback 必含 narrative_zh（业务层无须特殊判断）。"""

        def timeout_transport(**kwargs):
            raise _TimeoutException()

        gw = ModelGateway("https://x", "k", "m", transport=timeout_transport)
        r = gw.generate_node_narrative(node="treatment_strategy", effect={"ate": 0.12})
        assert r.fallback_used is True
        assert "narrative_zh" in r.content
        assert isinstance(r.content["narrative_zh"], str)
        assert len(r.content["narrative_zh"]) > 0

    def test_fallback_content_has_predicted_outcome_zh_for_counterfactual(self):
        def timeout_transport(**kwargs):
            raise _TimeoutException()

        gw = ModelGateway("https://x", "k", "m", transport=timeout_transport)
        r = gw.generate_counterfactual_narrative(
            scenario={"intervention_node": "treatment_strategy", "hypothetical_value": "PCI"}
        )
        assert r.fallback_used is True
        assert "predicted_outcome_zh" in r.content
        assert "confidence" in r.content

    def test_fallback_content_has_summary_zh_for_draft_method(self):
        def timeout_transport(**kwargs):
            raise _TimeoutException()

        gw = ModelGateway("https://x", "k", "m", transport=timeout_transport)
        r = gw.generate_draft_summary(draft={"disease_code": "I21", "outcome": "cmi"})
        assert r.fallback_used is True
        assert "summary_zh" in r.content


# ---------------------------------------------------------------------------
# 3. DeterministicNarrativeFallback（z-score 排序模板）
# ---------------------------------------------------------------------------


class TestDeterministicNarrativeFallbackNode:
    """build_node_narrative：单节点 z-score + 强/弱/背景 + ate 方向 + p_value 显著性。"""

    def test_strong_driver_with_high_significance(self):
        """ate ≥ 0.5 + p_value < 0.01 → 强驱动 + 高显著性。"""
        result = DeterministicNarrativeFallback.build_node_narrative(
            node="treatment_strategy",
            effect={"ate": 0.8, "p_value": 0.003, "method_name": "backdoor.linear_regression"},
            disease_code="I21",
            outcome="cmi",
        )
        assert result["_fallback_kind"] == "node_narrative"
        assert result["_fallback_engine"] == DeterministicNarrativeFallback.FALLBACK_ENGINE
        assert result["_z_score_strength"] == "强"
        assert result["_direction"] == "上升"
        assert "强驱动因子" in result["narrative_zh"]
        assert "高显著性" in result["narrative_zh"]
        assert "treatment_strategy" in result["narrative_zh"]

    def test_weak_driver_with_marginal_significance(self):
        """0.2 ≤ |ate| < 0.5 + 0.01 ≤ p < 0.05 → 弱驱动 + 显著。"""
        result = DeterministicNarrativeFallback.build_node_narrative(
            node="age_band",
            effect={"ate": -0.3, "p_value": 0.04},
            disease_code="I63",
            outcome="los_days",
        )
        assert result["_z_score_strength"] == "弱"
        assert result["_direction"] == "下降"
        assert "弱驱动因子" in result["narrative_zh"]
        assert "显著" in result["narrative_zh"]

    def test_background_driver_with_high_p_value(self):
        """ate 接近 0 + p_value > 0.05 → 背景驱动 + 不显著。"""
        result = DeterministicNarrativeFallback.build_node_narrative(
            node="admission_route",
            effect={"ate": 0.05, "p_value": 0.42},
            disease_code="Z96.10",
            outcome="readmit_flag",
        )
        assert result["_z_score_strength"] == "背景"
        assert "不显著" in result["narrative_zh"]

    def test_missing_effect_falls_back_to_legacy(self):
        """effect 为 None → 走 _build_fallback_content 兜底。"""
        result = DeterministicNarrativeFallback.build_node_narrative(
            node="x",
            effect=None,
        )
        assert "[fallback]" in result["narrative_zh"]

    def test_effect_missing_ate_falls_back(self):
        """effect 不含 ate → 兜底。"""
        result = DeterministicNarrativeFallback.build_node_narrative(
            node="x",
            effect={"p_value": 0.04},
        )
        assert "[fallback]" in result["narrative_zh"]

    def test_empty_disease_outcome_still_renders(self):
        """disease_code / outcome 空字符串 → 走"当前病种" / "当前 outcome"占位，不阻塞。"""
        result = DeterministicNarrativeFallback.build_node_narrative(
            node="x",
            effect={"ate": 0.12, "p_value": 0.04},
            disease_code="",
            outcome="",
        )
        assert "当前病种" in result["narrative_zh"]
        assert "当前 outcome" in result["narrative_zh"]

    def test_method_name_appended_to_narrative(self):
        result = DeterministicNarrativeFallback.build_node_narrative(
            node="x",
            effect={"ate": 0.5, "p_value": 0.003, "method_name": "backdoor.propensity_score_matching"},
        )
        assert "backdoor.propensity_score_matching" in result["narrative_zh"]


class TestDeterministicNarrativeFallbackCounterfactual:
    """build_counterfactual_narrative：delta 正负 → 上升/下降/维持。"""

    def test_positive_delta_renders_up(self):
        """predicted > baseline → 上升。"""
        result = DeterministicNarrativeFallback.build_counterfactual_narrative(
            scenario={"intervention_node": "treatment_strategy", "hypothetical_value": "PCI"},
            baseline={"baseline_value": 1.0, "baseline_std": 0.1},
            predicted={"predicted_outcome": 1.3},
        )
        assert result["_fallback_kind"] == "counterfactual_narrative"
        assert result["_direction"] == "上升"
        assert "上升" in result["predicted_outcome_zh"]
        assert "treatment_strategy" in result["predicted_outcome_zh"]
        assert "PCI" in result["predicted_outcome_zh"]
        assert result["_delta"] == pytest.approx(0.3, abs=1e-4)
        assert result["confidence"] == 0.0

    def test_negative_delta_renders_down(self):
        result = DeterministicNarrativeFallback.build_counterfactual_narrative(
            scenario={"intervention_node": "age_band", "hypothetical_value": 3},
            baseline={"baseline_value": 1.5, "baseline_std": 0.2},
            predicted={"predicted_outcome": 1.2},
        )
        assert result["_direction"] == "下降"

    def test_near_zero_delta_renders_maintain(self):
        """delta 接近 0 → 基本维持。"""
        result = DeterministicNarrativeFallback.build_counterfactual_narrative(
            scenario={"intervention_node": "complication_count", "hypothetical_value": 0},
            baseline={"baseline_value": 1.0, "baseline_std": 0.05},
            predicted={"predicted_outcome": 1.005},
        )
        assert result["_direction"] == "基本维持"

    def test_missing_scenario_falls_back(self):
        result = DeterministicNarrativeFallback.build_counterfactual_narrative(scenario=None)
        assert "[fallback]" in result["predicted_outcome_zh"]

    def test_missing_intervention_node_falls_back(self):
        result = DeterministicNarrativeFallback.build_counterfactual_narrative(
            scenario={"hypothetical_value": 1},
        )
        assert "[fallback]" in result["predicted_outcome_zh"]

    def test_missing_baseline_predicted_still_renders(self):
        """baseline / predicted 缺失 → 默认 0.0 → delta = 0 → 维持。"""
        result = DeterministicNarrativeFallback.build_counterfactual_narrative(
            scenario={"intervention_node": "x", "hypothetical_value": 1},
        )
        assert "基本维持" in result["predicted_outcome_zh"]


class TestDeterministicNarrativeFallbackDraftSummary:
    """build_draft_summary：|ate| z-score 排序前 3 名 + 反驳 robust 聚合。"""

    def test_top_n_drivers_sorted_by_abs_ate(self):
        """驱动因子按 |ate| z-score 排序，前 3 名进 summary。"""
        draft = {
            "disease_code": "I21",
            "outcome": "cmi",
            "sample_size": 200,
            "effect_estimates": [
                {"node": "treatment_strategy", "ate": 0.8, "p_value": 0.003},
                {"node": "age_band", "ate": 0.05, "p_value": 0.7},
                {"node": "complication_count", "ate": 0.6, "p_value": 0.01},
                {"node": "admission_route", "ate": 0.2, "p_value": 0.1},
            ],
            "refutation_results": [
                {"method": "random_common_cause", "passed": True},
                {"method": "placebo_treatment", "passed": True},
                {"method": "bootstrap", "passed": True},
                {"method": "data_subset", "passed": True},
            ],
            "counterfactual_scenarios": [{"scenario_id": "scn_1"}, {"scenario_id": "scn_2"}],
        }
        result = DeterministicNarrativeFallback.build_draft_summary(draft=draft)
        assert result["_fallback_kind"] == "draft_summary"
        assert result["_refutation_verdict"] == "稳健"
        assert len(result["_drivers_used"]) <= 3
        # treatment_strategy (0.8) 是最强驱动
        assert "treatment_strategy" in result["summary_zh"]
        # 反事实场景数
        assert "反事实场景 2 条" in result["summary_zh"]
        # 4 档反驳全 passed → 稳健
        assert "稳健" in result["summary_zh"]

    def test_partial_refutation_verdict(self):
        """部分 passed → 部分稳健。"""
        draft = {
            "disease_code": "I21",
            "outcome": "cmi",
            "sample_size": 100,
            "effect_estimates": [{"node": "x", "ate": 0.1, "p_value": 0.5}],
            "refutation_results": [
                {"method": "random_common_cause", "passed": True},
                {"method": "bootstrap", "passed": False},
            ],
            "counterfactual_scenarios": [],
        }
        result = DeterministicNarrativeFallback.build_draft_summary(draft=draft)
        assert result["_refutation_verdict"] == "部分稳健"

    def test_unstable_refutation_verdict(self):
        """refutation_unstable=True 全部 → 不稳定。"""
        draft = {
            "disease_code": "I21",
            "outcome": "cmi",
            "sample_size": 100,
            "effect_estimates": [{"node": "x", "ate": 0.1, "p_value": 0.5}],
            "refutation_results": [
                {"method": "random_common_cause", "passed": False, "refutation_unstable": True},
                {"method": "bootstrap", "passed": False, "refutation_unstable": True},
            ],
            "counterfactual_scenarios": [],
        }
        result = DeterministicNarrativeFallback.build_draft_summary(draft=draft)
        assert result["_refutation_verdict"] == "不稳定"

    def test_empty_draft_falls_back(self):
        result = DeterministicNarrativeFallback.build_draft_summary(draft=None)
        assert "[fallback]" in result["summary_zh"]

    def test_no_effect_estimates_still_renders(self):
        """effect_estimates 空 → "暂无显著驱动因子"占位。"""
        draft = {
            "disease_code": "I21",
            "outcome": "cmi",
            "sample_size": 100,
            "effect_estimates": [],
            "refutation_results": [],
            "counterfactual_scenarios": [],
        }
        result = DeterministicNarrativeFallback.build_draft_summary(draft=draft)
        assert "暂无显著驱动因子" in result["summary_zh"]

    def test_missing_p_value_in_estimates_still_counted(self):
        """effect_estimates 含 ate 无 p_value → 仍参与排序。"""
        draft = {
            "disease_code": "I21",
            "outcome": "cmi",
            "sample_size": 100,
            "effect_estimates": [
                {"node": "x", "ate": 0.5},
                {"node": "y", "ate": 0.3},
            ],
            "refutation_results": [{"method": "x", "passed": True}],
            "counterfactual_scenarios": [],
        }
        result = DeterministicNarrativeFallback.build_draft_summary(draft=draft)
        assert "x" in result["summary_zh"] or "y" in result["summary_zh"]

    def test_top_n_limit_is_three(self):
        """即使 5 个 driver 也只输出前 3 名（z-score 排序）。"""
        draft = {
            "disease_code": "I21",
            "outcome": "cmi",
            "sample_size": 200,
            "effect_estimates": [
                {"node": f"d{i}", "ate": float(i) * 0.1} for i in range(1, 6)
            ],
            "refutation_results": [{"method": "x", "passed": True}],
            "counterfactual_scenarios": [],
        }
        result = DeterministicNarrativeFallback.build_draft_summary(draft=draft)
        assert len(result["_drivers_used"]) == 3


# ---------------------------------------------------------------------------
# 4. Fallback-of-fallback（effect / scenario / draft 任一字段缺失）
# ---------------------------------------------------------------------------


class TestFallbackOfFallback:
    """任一字段缺失 → _build_fallback_content 兜底（带 [fallback] 前缀 + narrative_zh 可用）。"""

    def test_node_missing_effect(self):
        result = DeterministicNarrativeFallback.build_node_narrative(
            node="x",
            effect=None,
        )
        assert "[fallback]" in result["narrative_zh"]
        assert "_fallback_kind" in result

    def test_node_effect_ate_non_numeric(self):
        result = DeterministicNarrativeFallback.build_node_narrative(
            node="x",
            effect={"ate": "not-a-number"},
        )
        assert "[fallback]" in result["narrative_zh"]

    def test_counterfactual_scenario_none(self):
        result = DeterministicNarrativeFallback.build_counterfactual_narrative(scenario=None)
        assert "[fallback]" in result["predicted_outcome_zh"]

    def test_draft_none(self):
        result = DeterministicNarrativeFallback.build_draft_summary(draft=None)
        assert "[fallback]" in result["summary_zh"]

    def test_draft_non_mapping(self):
        result = DeterministicNarrativeFallback.build_draft_summary(draft="not-a-mapping")
        assert "[fallback]" in result["summary_zh"]


# ---------------------------------------------------------------------------
# 5. Audit 元数据完整性
# ---------------------------------------------------------------------------


class TestAuditMetadataCompleteness:
    """5 字段全部就位：prompt_hash / response_hash / latency_ms / tokens_used / model_version。"""

    def test_all_five_audit_fields_present_on_success(self):
        body = _openai_compat_response(
            content_obj={"narrative_zh": "ok"},
            model="custom-v2",
        )
        stub = _StubResp(200, body)

        class _T:
            def __call__(self, **kwargs):
                return stub

        gw = ModelGateway("https://x", "k", "m", transport=_T())
        r = gw.generate_node_narrative(node="x", effect={"ate": 0.0})
        assert isinstance(r.prompt_hash, str) and len(r.prompt_hash) == 16
        assert isinstance(r.response_hash, str) and len(r.response_hash) == 16
        assert r.latency_ms >= 0.0
        assert r.tokens_used == 15  # 10 + 5
        assert r.model_version == "custom-v2"

    def test_all_five_audit_fields_present_on_timeout(self):
        def timeout_transport(**kwargs):
            raise _TimeoutException()

        gw = ModelGateway("https://x", "k", "m", transport=timeout_transport)
        r = gw.generate_node_narrative(node="x", effect={"ate": 0.0})
        # 即便降级，5 字段仍写
        assert isinstance(r.prompt_hash, str) and len(r.prompt_hash) == 16
        assert isinstance(r.response_hash, str) and len(r.response_hash) == 16
        assert r.latency_ms >= 0.0
        assert r.tokens_used == 0  # 超时无 usage
        assert r.model_version  # 兜底走 settings.openai_model

    def test_audit_writer_called_with_five_fields(self):
        captured: list = []

        def audit_writer(**kwargs):
            captured.append(kwargs)

        def timeout_transport(**kwargs):
            raise _TimeoutException()

        gw = ModelGateway(
            "https://x", "k", "m", transport=timeout_transport, audit_writer=audit_writer
        )
        r = gw.generate_node_narrative(node="x", effect={"ate": 0.0})
        assert len(captured) == 1
        ev = captured[0]
        assert ev["event_type"] == "model/call"
        for k in ("prompt_hash", "response_hash", "latency_ms", "tokens_used", "model_version"):
            assert k in ev
        assert ev["fallback_used"] is True
        assert "timeout:" in ev["fallback_reason"]
        assert ev["state_not_advanced"] is True

    def test_model_response_to_dict_contains_audit_metadata(self):
        def timeout_transport(**kwargs):
            raise _TimeoutException()

        gw = ModelGateway("https://x", "k", "m", transport=timeout_transport)
        r = gw.generate_node_narrative(node="x", effect={"ate": 0.0})
        d = r.to_dict()
        for k in (
            "schema_version",
            "method",
            "content",
            "fallback_used",
            "fallback_reason",
            "state_not_advanced",
            "prompt_hash",
            "response_hash",
            "latency_ms",
            "tokens_used",
            "model_version",
        ):
            assert k in d
        assert d["schema_version"] == MODEL_RESPONSE_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# 6. state_not_advanced=True 永远（4 降级路径 × 3 方法 parametrize）
# ---------------------------------------------------------------------------


class TestStateNotAdvancedAlwaysTrue:
    """3 方法 × 4 降级路径（成功 / timeout / unparseable / http_error）
    全部断言 state_not_advanced=True（spec §7 不静默推进业务状态契约）。"""

    @pytest.mark.parametrize(
        "method_name,call_kwargs",
        [
            ("generate_node_narrative", {"node": "x", "effect": {"ate": 0.0}}),
            (
                "generate_counterfactual_narrative",
                {"scenario": {"intervention_node": "y", "hypothetical_value": 1}},
            ),
            ("generate_draft_summary", {"draft": {"disease_code": "I21", "outcome": "cmi"}}),
        ],
    )
    def test_state_not_advanced_on_success(self, method_name, call_kwargs):
        body = _openai_compat_response(content_obj={"narrative_zh": "ok"})
        stub = _StubResp(200, body)

        class _T:
            def __call__(self, **kwargs):
                return stub

        gw = ModelGateway("https://x", "k", "m", transport=_T())
        r = getattr(gw, method_name)(**call_kwargs)
        assert r.state_not_advanced is True

    @pytest.mark.parametrize(
        "method_name,call_kwargs",
        [
            ("generate_node_narrative", {"node": "x", "effect": {"ate": 0.0}}),
            (
                "generate_counterfactual_narrative",
                {"scenario": {"intervention_node": "y", "hypothetical_value": 1}},
            ),
            ("generate_draft_summary", {"draft": {"disease_code": "I21", "outcome": "cmi"}}),
        ],
    )
    def test_state_not_advanced_on_timeout(self, method_name, call_kwargs):
        def timeout_transport(**kwargs):
            raise _TimeoutException()

        gw = ModelGateway("https://x", "k", "m", transport=timeout_transport)
        r = getattr(gw, method_name)(**call_kwargs)
        assert r.state_not_advanced is True

    @pytest.mark.parametrize(
        "method_name,call_kwargs",
        [
            ("generate_node_narrative", {"node": "x", "effect": {"ate": 0.0}}),
            (
                "generate_counterfactual_narrative",
                {"scenario": {"intervention_node": "y", "hypothetical_value": 1}},
            ),
            ("generate_draft_summary", {"draft": {"disease_code": "I21", "outcome": "cmi"}}),
        ],
    )
    def test_state_not_advanced_on_unparseable(self, method_name, call_kwargs):
        stub = _StubResp(200, {"choices": [{"message": {"content": "not-json"}}]})

        class _T:
            def __call__(self, **kwargs):
                return stub

        gw = ModelGateway("https://x", "k", "m", transport=_T())
        r = getattr(gw, method_name)(**call_kwargs)
        assert r.state_not_advanced is True

    @pytest.mark.parametrize(
        "method_name,call_kwargs",
        [
            ("generate_node_narrative", {"node": "x", "effect": {"ate": 0.0}}),
            (
                "generate_counterfactual_narrative",
                {"scenario": {"intervention_node": "y", "hypothetical_value": 1}},
            ),
            ("generate_draft_summary", {"draft": {"disease_code": "I21", "outcome": "cmi"}}),
        ],
    )
    def test_state_not_advanced_on_http_error(self, method_name, call_kwargs):
        stub = _StubResp(503, {"error": "service-unavailable"})

        class _T:
            def __call__(self, **kwargs):
                return stub

        gw = ModelGateway("https://x", "k", "m", transport=_T())
        r = getattr(gw, method_name)(**call_kwargs)
        assert r.state_not_advanced is True


# ---------------------------------------------------------------------------
# 7. 模块契约（schema_version / 常量 / __all__）
# ---------------------------------------------------------------------------


class TestModuleContracts:
    def test_deterministic_schema_version_exists(self):
        assert isinstance(DETERMINISTIC_NARRATIVE_SCHEMA_VERSION, str)
        assert DETERMINISTIC_NARRATIVE_SCHEMA_VERSION.startswith("deterministic_narrative/")

    def test_fallback_engine_name(self):
        assert DeterministicNarrativeFallback.FALLBACK_ENGINE == "deterministic_v1"

    def test_module_all_exports_degradation(self):
        from app.adapters import model_gateway as mg_module

        for name in (
            "DETERMINISTIC_NARRATIVE_SCHEMA_VERSION",
            "DeterministicNarrativeFallback",
            "ModelGateway",
            "ModelResponse",
        ):
            assert name in mg_module.__all__, f"missing __all__ entry: {name}"
            assert hasattr(mg_module, name), f"missing attribute: {name}"

    def test_fallback_response_has_all_three_methods_supported(self):
        """3 方法（node / counterfactual / draft）都应可调用 DeterministicNarrativeFallback。"""
        r1 = DeterministicNarrativeFallback.build_node_narrative(
            node="x", effect={"ate": 0.1, "p_value": 0.5}
        )
        r2 = DeterministicNarrativeFallback.build_counterfactual_narrative(
            scenario={"intervention_node": "x", "hypothetical_value": 1},
            baseline={"baseline_value": 1.0},
            predicted={"predicted_outcome": 1.2},
        )
        r3 = DeterministicNarrativeFallback.build_draft_summary(
            draft={"disease_code": "I21", "outcome": "cmi"}
        )
        assert r1["_fallback_kind"] == "node_narrative"
        assert r2["_fallback_kind"] == "counterfactual_narrative"
        assert r3["_fallback_kind"] == "draft_summary"