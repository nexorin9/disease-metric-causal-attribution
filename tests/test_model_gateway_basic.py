"""test_model_gateway_basic — ModelGateway OpenAI-compatible 单协议基础测试。

覆盖维度（spec §7 + §11）：
  1. 构造期校验：缺 base_url / 缺 api_key / 缺 model / 非法 timeout
  2. from_settings 工厂：从 pydantic-settings 单例读 OPENAI_BASE_URL/API_KEY/MODEL/TIMEOUT
  3. 协议握手：mock transport 收到 POST /chat/completions + Authorization Bearer + response_format
  4. 结构化请求：messages 至少含 system + user，且 user JSON 形式
  5. 结构化响应：ModelResponse.content 是 dict；narrative_zh / predicted_outcome_zh / summary_zh 抽取正确
  6. 审计元数据：prompt_hash / response_hash / latency_ms / tokens_used / model_version 全部存在
  7. state_not_advanced 永远 True（不静默推进业务状态）
  8. 三方法 3 路 timeout 路由：node_narrative 10s / counterfactual_narrative 10s / draft_summary 30s
  9. 一行调用入口：generate_node_narrative / generate_counterfactual_narrative / generate_draft_summary
 10. 模块常量 + SchemaVersion + __all__ 完整性
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

# 让 `from app.xxx import ...` 工作
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("APP_ENV", "test")

from app.adapters.model_gateway import (  # noqa: E402
    DEFAULT_MODEL_VERSION,
    DEFAULT_TIMEOUT_SECONDS,
    MODEL_RESPONSE_SCHEMA_VERSION,
    OPENAI_CHAT_COMPLETIONS_PATH,
    RESPONSE_FORMAT_JSON_OBJECT,
    TIMEOUT_EXPLAINING_SECONDS,
    TIMEOUT_REASONING_SECONDS,
    ModelGateway,
    ModelGatewayConfigError,
    ModelResponse,
    generate_counterfactual_narrative,
    generate_draft_summary,
    generate_node_narrative,
)
from app.config import get_settings, reset_settings_cache  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _StubResp:
    """OpenAI-compatible 替身响应对象（httpx.Response 兼容：status_code + json() + content）。"""

    def __init__(
        self,
        status_code: int,
        body: Dict[str, Any],
    ) -> None:
        self.status_code = int(status_code)
        self._body = body
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.content = raw
        self.text = raw.decode("utf-8")

    def json(self) -> Dict[str, Any]:
        return self._body


class _SpyTransport:
    """捕获最近一次请求 + 返回预设响应（OpenAI-compatible 协议握手测试用）。"""

    def __init__(self, resp: _StubResp) -> None:
        self.resp = resp
        self.calls: list = []

    def __call__(self, *, url: str, headers: Dict[str, str], body: str, timeout: float) -> _StubResp:
        self.calls.append({"url": url, "headers": headers, "body": body, "timeout": timeout})
        return self.resp


def _openai_compat_response(
    *,
    content_obj: Dict[str, Any],
    model: str = "gpt-4o-mini",
    prompt_tokens: int = 12,
    completion_tokens: int = 8,
) -> Dict[str, Any]:
    """构造一个 OpenAI-compatible 协议层响应 JSON。"""
    return {
        "id": "chatcmpl-test-stub",
        "object": "chat.completion",
        "created": 1700000000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": json.dumps(content_obj, ensure_ascii=False)},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_settings():
    """每个测试前重置 pydantic-settings 单例，避免跨测试污染。"""
    reset_settings_cache()
    yield
    reset_settings_cache()


# ---------------------------------------------------------------------------
# 1. 构造期校验
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_missing_base_url_raises(self):
        with pytest.raises(ModelGatewayConfigError):
            ModelGateway(base_url="", api_key="k", model="m")

    def test_missing_api_key_raises(self):
        with pytest.raises(ModelGatewayConfigError):
            ModelGateway(base_url="https://x", api_key="", model="m")

    def test_missing_model_raises(self):
        with pytest.raises(ModelGatewayConfigError):
            ModelGateway(base_url="https://x", api_key="k", model="")

    def test_invalid_timeout_raises(self):
        with pytest.raises(ModelGatewayConfigError):
            ModelGateway(base_url="https://x", api_key="k", model="m", timeout=0)

    def test_negative_timeout_raises(self):
        with pytest.raises(ModelGatewayConfigError):
            ModelGateway(base_url="https://x", api_key="k", model="m", timeout=-1)

    def test_default_path_used_when_empty(self):
        gw = ModelGateway(base_url="https://x", api_key="k", model="m", path="")
        assert gw.path == OPENAI_CHAT_COMPLETIONS_PATH
        assert gw.chat_url == f"https://x{OPENAI_CHAT_COMPLETIONS_PATH}"

    def test_trailing_slash_stripped_from_base_url(self):
        gw = ModelGateway(base_url="https://x.example.com/", api_key="k", model="m")
        assert gw.base_url == "https://x.example.com"
        assert gw.chat_url == f"https://x.example.com{OPENAI_CHAT_COMPLETIONS_PATH}"


# ---------------------------------------------------------------------------
# 2. from_settings 工厂
# ---------------------------------------------------------------------------


class TestFromSettings:
    def test_loads_from_pydantic_settings(self, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.com")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        monkeypatch.setenv("OPENAI_MODEL", "custom-model")
        monkeypatch.setenv("OPENAI_TIMEOUT", "25")
        reset_settings_cache()
        s = get_settings()
        gw = ModelGateway.from_settings(s)
        assert gw.base_url == "https://api.example.com"
        assert gw.api_key == "sk-from-env"
        assert gw.model == "custom-model"
        assert gw.timeout == 25

    def test_explicit_overrides_take_precedence(self, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.com")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        monkeypatch.setenv("OPENAI_MODEL", "env-model")
        monkeypatch.setenv("OPENAI_TIMEOUT", "30")
        reset_settings_cache()
        s = get_settings()
        gw = ModelGateway.from_settings(
            s,
            base_url="https://override.com",
            api_key="override-key",
            model="override-model",
            timeout=15,
        )
        assert gw.base_url == "https://override.com"
        assert gw.api_key == "override-key"
        assert gw.model == "override-model"
        assert gw.timeout == 15

    def test_default_settings_used_when_none(self, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "https://auto.example.com")
        monkeypatch.setenv("OPENAI_API_KEY", "auto-key")
        reset_settings_cache()
        gw = ModelGateway.from_settings()
        assert gw.base_url == "https://auto.example.com"
        assert gw.api_key == "auto-key"


# ---------------------------------------------------------------------------
# 3 + 4. 协议握手 + 结构化请求
# ---------------------------------------------------------------------------


class TestProtocolHandshake:
    def test_post_chat_completions_with_bearer(self):
        resp_body = _openai_compat_response(
            content_obj={"narrative_zh": "PCI 治疗组的 CMI 显著高于保守治疗（p=0.003）。"}
        )
        stub = _StubResp(200, resp_body)
        spy = _SpyTransport(stub)

        gw = ModelGateway(
            base_url="https://api.example.com",
            api_key="sk-test",
            model="gpt-4o-mini",
            transport=spy,
        )
        r = gw.generate_node_narrative(
            node="treatment_strategy",
            effect={"ate": 0.12, "p_value": 0.003, "method_name": "backdoor.linear_regression"},
        )

        # 1 次调用
        assert len(spy.calls) == 1
        call = spy.calls[0]
        # URL 拼接
        assert call["url"] == f"https://api.example.com{OPENAI_CHAT_COMPLETIONS_PATH}"
        # Bearer 鉴权
        assert call["headers"]["Authorization"] == "Bearer sk-test"
        assert call["headers"]["Content-Type"] == "application/json"
        # body JSON 含 model + messages + response_format
        body_json = json.loads(call["body"])
        assert body_json["model"] == "gpt-4o-mini"
        assert body_json["response_format"] == RESPONSE_FORMAT_JSON_OBJECT
        assert isinstance(body_json["messages"], list)
        assert len(body_json["messages"]) >= 2
        assert body_json["messages"][0]["role"] == "system"
        assert body_json["messages"][1]["role"] == "user"
        # user content 是 JSON 字符串
        user_content = body_json["messages"][1]["content"]
        assert isinstance(user_content, str)
        user_dict = json.loads(user_content)
        assert user_dict["node"] == "treatment_strategy"
        assert user_dict["effect"]["ate"] == 0.12
        assert user_dict["effect"]["p_value"] == 0.003

        # 响应解析
        assert isinstance(r, ModelResponse)
        assert r.content["narrative_zh"] == "PCI 治疗组的 CMI 显著高于保守治疗（p=0.003）。"
        assert r.method == "generate_node_narrative"
        assert r.fallback_used is False

    def test_timeout_routing_per_method(self):
        resp_body = _openai_compat_response(content_obj={"narrative_zh": "ok"})
        stub = _StubResp(200, resp_body)
        spy = _SpyTransport(stub)

        gw = ModelGateway(
            base_url="https://api.example.com",
            api_key="sk-test",
            model="m",
            timeout=60,  # 上限较高，让方法路由决定
            transport=spy,
        )
        # generate_node_narrative → 10s
        gw.generate_node_narrative(node="x", effect={"ate": 0.0})
        assert spy.calls[-1]["timeout"] == TIMEOUT_REASONING_SECONDS  # 10s

        # generate_counterfactual_narrative → 10s
        gw.generate_counterfactual_narrative(
            scenario={"intervention_node": "y", "hypothetical_value": 1}
        )
        assert spy.calls[-1]["timeout"] == TIMEOUT_REASONING_SECONDS  # 10s

        # generate_draft_summary → 30s
        gw.generate_draft_summary(
            draft={"disease_code": "I21", "outcome": "cmi", "treatment": "PCI"}
        )
        assert spy.calls[-1]["timeout"] == TIMEOUT_EXPLAINING_SECONDS  # 30s

    def test_timeout_clamped_by_settings_timeout(self):
        """即使 OPENAI_TIMEOUT=5（低于默认档位），实际 timeout 仍是 5。"""
        resp_body = _openai_compat_response(content_obj={"narrative_zh": "ok"})
        stub = _StubResp(200, resp_body)
        spy = _SpyTransport(stub)

        gw = ModelGateway(
            base_url="https://api.example.com",
            api_key="sk",
            model="m",
            timeout=5,
            transport=spy,
        )
        gw.generate_node_narrative(node="x", effect={"ate": 0.0})
        assert spy.calls[-1]["timeout"] == 5.0

    def test_post_url_format(self):
        resp_body = _openai_compat_response(content_obj={"narrative_zh": "ok"})
        stub = _StubResp(200, resp_body)
        spy = _SpyTransport(stub)

        gw = ModelGateway(
            base_url="https://api.example.com",
            api_key="sk",
            model="m",
            path="/v1/chat/completions",  # 自定义 path
            transport=spy,
        )
        gw.generate_node_narrative(node="x", effect={"ate": 0.0})
        assert spy.calls[-1]["url"] == "https://api.example.com/v1/chat/completions"


# ---------------------------------------------------------------------------
# 5. 结构化响应（3 方法）
# ---------------------------------------------------------------------------


class TestStructuredOutput:
    def test_generate_node_narrative_parses_content(self):
        stub = _StubResp(
            200,
            _openai_compat_response(content_obj={"narrative_zh": "节点 X 的归因解释"}),
        )
        spy = _SpyTransport(stub)
        gw = ModelGateway("https://x", "k", "m", transport=spy)

        r = gw.generate_node_narrative(
            node="age_band",
            effect={"ate": -0.05, "p_value": 0.04, "method_name": "backdoor.linear_regression"},
            disease_code="I21",
            outcome="cmi",
        )
        assert r.content["narrative_zh"] == "节点 X 的归因解释"
        assert r.fallback_used is False
        assert r.method == "generate_node_narrative"

    def test_generate_counterfactual_narrative_parses_content(self):
        stub = _StubResp(
            200,
            _openai_compat_response(
                content_obj={
                    "predicted_outcome_zh": "若 PCI 占比升至 70%，CMI 预测上升 0.18",
                    "confidence": 0.82,
                }
            ),
        )
        spy = _SpyTransport(stub)
        gw = ModelGateway("https://x", "k", "m", transport=spy)

        r = gw.generate_counterfactual_narrative(
            scenario={
                "intervention_node": "treatment_strategy",
                "hypothetical_value": "PCI",
                "disease_code": "I21",
                "outcome": "cmi",
            },
            baseline={"baseline_value": 1.20, "baseline_std": 0.10},
            predicted={"predicted_outcome": 1.38, "do_operations": ["do(treatment_strategy=PCI)"]},
        )
        assert r.content["predicted_outcome_zh"] == "若 PCI 占比升至 70%，CMI 预测上升 0.18"
        assert r.content["confidence"] == 0.82
        assert r.fallback_used is False
        assert r.method == "generate_counterfactual_narrative"

    def test_generate_draft_summary_parses_content(self):
        stub = _StubResp(
            200,
            _openai_compat_response(
                content_obj={"summary_zh": "本季度 I21 / cmi 主要由治疗方式驱动（ate=0.12, p=0.003）。"}
            ),
        )
        spy = _SpyTransport(stub)
        gw = ModelGateway("https://x", "k", "m", transport=spy)

        r = gw.generate_draft_summary(
            draft={
                "disease_code": "I21",
                "outcome": "cmi",
                "treatment": "treatment_strategy",
                "sample_size": 200,
                "effect_estimates": [{"node": "treatment_strategy", "ate": 0.12, "p_value": 0.003}],
                "refutation_results": [{"method": "random_common_cause", "passed": True}],
                "counterfactual_scenarios": [],
            }
        )
        assert r.content["summary_zh"] == "本季度 I21 / cmi 主要由治疗方式驱动（ate=0.12, p=0.003）。"
        assert r.fallback_used is False
        assert r.method == "generate_draft_summary"


# ---------------------------------------------------------------------------
# 6. 审计元数据
# ---------------------------------------------------------------------------


class TestAuditMetadata:
    def test_prompt_response_hash_present(self):
        stub = _StubResp(
            200,
            _openai_compat_response(content_obj={"narrative_zh": "ok"}),
        )
        spy = _SpyTransport(stub)
        gw = ModelGateway("https://x", "k", "m", transport=spy)

        r = gw.generate_node_narrative(node="x", effect={"ate": 0.0})
        # prompt_hash / response_hash：16 hex
        assert isinstance(r.prompt_hash, str) and len(r.prompt_hash) == 16
        assert isinstance(r.response_hash, str) and len(r.response_hash) == 16
        assert all(c in "0123456789abcdef" for c in r.prompt_hash)
        assert all(c in "0123456789abcdef" for c in r.response_hash)

    def test_latency_ms_recorded(self):
        stub = _StubResp(200, _openai_compat_response(content_obj={"narrative_zh": "ok"}))
        spy = _SpyTransport(stub)
        gw = ModelGateway("https://x", "k", "m", transport=spy)

        r = gw.generate_node_narrative(node="x", effect={"ate": 0.0})
        assert r.latency_ms >= 0.0

    def test_tokens_used_from_usage(self):
        stub = _StubResp(
            200,
            _openai_compat_response(
                content_obj={"narrative_zh": "ok"}, prompt_tokens=42, completion_tokens=13
            ),
        )
        spy = _SpyTransport(stub)
        gw = ModelGateway("https://x", "k", "m", transport=spy)

        r = gw.generate_node_narrative(node="x", effect={"ate": 0.0})
        assert r.tokens_used == 55  # 42 + 13

    def test_model_version_recorded(self):
        stub = _StubResp(
            200,
            _openai_compat_response(content_obj={"narrative_zh": "ok"}, model="custom-model-v2"),
        )
        spy = _SpyTransport(stub)
        gw = ModelGateway("https://x", "k", "m", transport=spy)

        r = gw.generate_node_narrative(node="x", effect={"ate": 0.0})
        assert r.model_version == "custom-model-v2"

    def test_audit_writer_invoked(self):
        captured: list = []

        def audit_writer(**kwargs):
            captured.append(kwargs)

        stub = _StubResp(200, _openai_compat_response(content_obj={"narrative_zh": "ok"}))
        spy = _SpyTransport(stub)
        gw = ModelGateway("https://x", "k", "m", transport=spy, audit_writer=audit_writer)
        r = gw.generate_node_narrative(node="x", effect={"ate": 0.0})

        assert len(captured) == 1
        ev = captured[0]
        assert ev["event_type"] == "model/call"
        assert ev["method"] == "generate_node_narrative"
        assert ev["schema_version"] == MODEL_RESPONSE_SCHEMA_VERSION
        assert ev["prompt_hash"] == r.prompt_hash
        assert ev["response_hash"] == r.response_hash
        assert ev["tokens_used"] == r.tokens_used
        assert ev["model_version"] == r.model_version
        assert ev["state_not_advanced"] is True
        assert r.extra["audit_event_written"] is True

    def test_audit_writer_failure_does_not_break_call(self):
        def bad_audit(**kwargs):
            raise RuntimeError("audit-sink-down")

        stub = _StubResp(200, _openai_compat_response(content_obj={"narrative_zh": "ok"}))
        spy = _SpyTransport(stub)
        gw = ModelGateway("https://x", "k", "m", transport=spy, audit_writer=bad_audit)
        r = gw.generate_node_narrative(node="x", effect={"ate": 0.0})
        assert r.fallback_used is False
        assert r.extra["audit_event_written"] is False


# ---------------------------------------------------------------------------
# 7. state_not_advanced 永远 True
# ---------------------------------------------------------------------------


class TestStateNotAdvanced:
    @pytest.mark.parametrize("method_name,call_kwargs", [
        ("generate_node_narrative", {"node": "x", "effect": {"ate": 0.0}}),
        (
            "generate_counterfactual_narrative",
            {"scenario": {"intervention_node": "y", "hypothetical_value": 1}},
        ),
        ("generate_draft_summary", {"draft": {"disease_code": "I21", "outcome": "cmi"}}),
    ])
    def test_state_not_advanced_always_true(self, method_name, call_kwargs):
        stub = _StubResp(200, _openai_compat_response(content_obj={"narrative_zh": "ok"}))
        spy = _SpyTransport(stub)
        gw = ModelGateway("https://x", "k", "m", transport=spy)
        r = getattr(gw, method_name)(**call_kwargs)
        assert r.state_not_advanced is True


# ---------------------------------------------------------------------------
# 8. 降级路径（最小：模型不可达 / 不可解析 / 缺必需字段）
# ---------------------------------------------------------------------------


class TestFallbackPath:
    def test_unparseable_response_falls_back(self):
        """响应 body 不是 JSON → 降级到 fallback_content（narrative_zh 兜底）。"""
        stub = _StubResp(200, {"choices": [{"message": {"content": "not-json"}}]})
        spy = _SpyTransport(stub)
        gw = ModelGateway("https://x", "k", "m", transport=spy)

        r = gw.generate_node_narrative(node="x", effect={"ate": 0.0})
        assert r.fallback_used is True
        assert r.fallback_reason in ("model_output_unparseable", "response_unparseable")
        # fallback content 必须含 narrative_zh（业务层无须特殊判断）
        assert "narrative_zh" in r.content
        assert "[fallback]" in r.content["narrative_zh"]

    def test_http_error_falls_back(self):
        stub = _StubResp(500, {"error": "internal"})
        spy = _SpyTransport(stub)
        gw = ModelGateway("https://x", "k", "m", transport=spy)

        r = gw.generate_node_narrative(node="x", effect={"ate": 0.0})
        assert r.fallback_used is True
        assert "http_error" in r.fallback_reason

    def test_network_error_falls_back(self):
        def bad_transport(**kwargs):
            raise ConnectionError("network-down")

        gw = ModelGateway("https://x", "k", "m", transport=bad_transport)
        r = gw.generate_node_narrative(node="x", effect={"ate": 0.0})
        assert r.fallback_used is True
        assert "network_error" in r.fallback_reason
        # 兜底仍含 narrative_zh
        assert "narrative_zh" in r.content

    def test_missing_required_keys_falls_back(self):
        """响应 JSON 解析成功但缺 narrative_zh → 降级。"""
        stub = _StubResp(200, _openai_compat_response(content_obj={"wrong_key": "value"}))
        spy = _SpyTransport(stub)
        gw = ModelGateway("https://x", "k", "m", transport=spy)

        r = gw.generate_node_narrative(node="x", effect={"ate": 0.0})
        assert r.fallback_used is True
        assert r.fallback_reason == "missing_required_keys"
        assert "narrative_zh" in r.content  # 兜底内容仍有 narrative_zh


# ---------------------------------------------------------------------------
# 9. 便捷函数入口
# ---------------------------------------------------------------------------


class TestModuleFunctions:
    def test_generate_node_narrative_helper(self, monkeypatch):
        """便捷函数走 ModelGateway.from_settings(settings)。"""
        monkeypatch.setenv("OPENAI_BASE_URL", "https://x")
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        reset_settings_cache()

        # 通过 monkeypatch ModelGateway 来 spy
        called: list = []

        class _GW:
            def generate_node_narrative(self, **kw):
                called.append(kw)
                return ModelResponse(
                    content={"narrative_zh": "from helper"},
                    method="generate_node_narrative",
                )

        from app.adapters import model_gateway as mg_module

        original = mg_module.ModelGateway.from_settings
        monkeypatch.setattr(mg_module.ModelGateway, "from_settings", staticmethod(lambda settings=None: _GW()))

        r = generate_node_narrative(
            node="x", effect={"ate": 0.0}, disease_code="I21", outcome="cmi"
        )
        assert r.content["narrative_zh"] == "from helper"
        assert called[0]["node"] == "x"
        assert called[0]["disease_code"] == "I21"

        # restore
        mg_module.ModelGateway.from_settings = original

    def test_generate_counterfactual_narrative_helper(self, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "https://x")
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        reset_settings_cache()

        from app.adapters import model_gateway as mg_module

        class _GW:
            def generate_counterfactual_narrative(self, **kw):
                return ModelResponse(
                    content={"predicted_outcome_zh": "from helper"},
                    method="generate_counterfactual_narrative",
                )

        monkeypatch.setattr(
            mg_module.ModelGateway, "from_settings", staticmethod(lambda settings=None: _GW())
        )

        r = generate_counterfactual_narrative(
            scenario={"intervention_node": "y", "hypothetical_value": 1}
        )
        assert r.content["predicted_outcome_zh"] == "from helper"

    def test_generate_draft_summary_helper(self, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "https://x")
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        reset_settings_cache()

        from app.adapters import model_gateway as mg_module

        class _GW:
            def generate_draft_summary(self, **kw):
                return ModelResponse(
                    content={"summary_zh": "from helper"},
                    method="generate_draft_summary",
                )

        monkeypatch.setattr(
            mg_module.ModelGateway, "from_settings", staticmethod(lambda settings=None: _GW())
        )

        r = generate_draft_summary(draft={"disease_code": "I21", "outcome": "cmi"})
        assert r.content["summary_zh"] == "from helper"


# ---------------------------------------------------------------------------
# 10. 模块常量 + SchemaVersion 完整性
# ---------------------------------------------------------------------------


class TestContracts:
    def test_constants_exist(self):
        assert isinstance(DEFAULT_MODEL_VERSION, str) and DEFAULT_MODEL_VERSION
        assert isinstance(MODEL_RESPONSE_SCHEMA_VERSION, str) and MODEL_RESPONSE_SCHEMA_VERSION
        assert isinstance(OPENAI_CHAT_COMPLETIONS_PATH, str) and OPENAI_CHAT_COMPLETIONS_PATH.startswith("/")
        assert isinstance(RESPONSE_FORMAT_JSON_OBJECT, dict)
        assert RESPONSE_FORMAT_JSON_OBJECT == {"type": "json_object"}
        assert TIMEOUT_REASONING_SECONDS == 10
        assert TIMEOUT_EXPLAINING_SECONDS == 30
        assert DEFAULT_TIMEOUT_SECONDS >= TIMEOUT_EXPLAINING_SECONDS

    def test_model_response_schema_version(self):
        r = ModelResponse(content={"x": "y"})
        assert r.schema_version == MODEL_RESPONSE_SCHEMA_VERSION

    def test_model_response_to_dict_has_all_fields(self):
        r = ModelResponse(
            content={"narrative_zh": "ok"},
            fallback_used=False,
            fallback_reason="",
            state_not_advanced=True,
            prompt_hash="abc123",
            response_hash="def456",
            latency_ms=12.3,
            tokens_used=10,
            model_version="m",
            method="generate_node_narrative",
        )
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
        assert d["state_not_advanced"] is True
        assert d["latency_ms"] == 12.3
        assert d["tokens_used"] == 10

    def test_module_all_exports(self):
        from app.adapters import model_gateway as mg_module

        for name in (
            "DEFAULT_MODEL_VERSION",
            "DEFAULT_TIMEOUT_SECONDS",
            "MODEL_RESPONSE_SCHEMA_VERSION",
            "OPENAI_CHAT_COMPLETIONS_PATH",
            "RESPONSE_FORMAT_JSON_OBJECT",
            "TIMEOUT_EXPLAINING_SECONDS",
            "TIMEOUT_REASONING_SECONDS",
            "ModelGateway",
            "ModelGatewayConfigError",
            "ModelGatewayError",
            "ModelGatewayTimeoutError",
            "ModelResponse",
            "generate_counterfactual_narrative",
            "generate_draft_summary",
            "generate_node_narrative",
        ):
            assert name in mg_module.__all__, f"missing __all__ entry: {name}"
            assert hasattr(mg_module, name), f"missing attribute: {name}"

    def test_input_validation_node_required(self):
        gw = ModelGateway("https://x", "k", "m", transport=_SpyTransport(_StubResp(200, _openai_compat_response(content_obj={"narrative_zh": "x"}))))
        with pytest.raises(ModelGatewayConfigError):
            gw.generate_node_narrative(node="", effect={"ate": 0.0})

    def test_input_validation_intervention_node_required(self):
        gw = ModelGateway("https://x", "k", "m", transport=_SpyTransport(_StubResp(200, _openai_compat_response(content_obj={"predicted_outcome_zh": "x"}))))
        with pytest.raises(ModelGatewayConfigError):
            gw.generate_counterfactual_narrative(scenario={"intervention_node": ""})

    def test_input_validation_effect_required(self):
        gw = ModelGateway("https://x", "k", "m", transport=_SpyTransport(_StubResp(200, _openai_compat_response(content_obj={"narrative_zh": "x"}))))
        with pytest.raises(ModelGatewayConfigError):
            gw.generate_node_narrative(node="x", effect=None)
        with pytest.raises(ModelGatewayConfigError):
            gw.generate_node_narrative(node="x", effect="not-a-mapping")


# ---------------------------------------------------------------------------
# 11. close() 钩子 + lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_close_is_noop(self):
        gw = ModelGateway("https://x", "k", "m")
        assert gw.close() is None  # 不抛错