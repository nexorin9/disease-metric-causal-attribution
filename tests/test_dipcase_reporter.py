"""DipcaseReporter 测试（spec §3.2 + §11 收尾交付）。

覆盖：
  1. 构造期校验 — 凭据 / URL / timeout / max_retries
  2. payload 校验 — 缺字段 / 类型错 / PII 白名单 / schema_version 注入
  3. 协议握手 — Authorization Bearer + Content-Type + URL 拼接
  4. retry 3 次 — 5xx 触发重试 / 4xx 不重试 / 重试耗尽降级
  5. 失败回退 — 重试耗尽 degraded=True + ok=False + status_code=None；4xx degraded=False
  6. audit 钩子 — audit_writer 注入 + strip_pii 净化 + 不抛
  7. 端到端 — from_default_settings + 默认 transport 走协议握手
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import MagicMock

import pytest
import requests

# 本地 conftest 已把 sys.path 注入；显式 import 项目模块
from app.adapters.dipcase_reporter import (
    CONTENT_TYPE_JSON,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_BASE_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    DIPCASE_PAYLOAD_SCHEMA_VERSION,
    DIPCASE_REPORTER_PATH,
    DipcaseConfigError,
    DipcasePayloadValidationError,
    DipcaseReporter,
    DipcaseReporterError,
    PostResult,
    _strip_pii,
    _validate_payload,
    post_attribution_draft,
)


# =============================================================================
# 测试夹具
# =============================================================================


class _FakeResp:
    """替身 transport 返回 — 与 requests.Response 鸭子类型兼容。"""

    def __init__(self, status_code: int, body: Any = None) -> None:
        self.status_code = int(status_code)
        # body 仅用于诊断；spec §11 不写响应正文进 audit
        self._body = body or {}

    def json(self) -> Dict[str, Any]:
        return dict(self._body)


def _ok_transport(sc: int = 200, body: Optional[Dict[str, Any]] = None) -> Callable[..., Any]:
    """返回 2xx 的 transport 工厂。"""
    def _transport(*, url, headers, body, timeout):
        return _FakeResp(sc, body or {"ok": True, "draft_id": "drf_test"})
    return _transport


def _seq_transport(
    statuses: List[int],
    body: Optional[Dict[str, Any]] = None,
) -> Callable[..., Any]:
    """返回序列响应的 transport — 第一/第二/... 次返回对应 status。"""
    state = {"i": 0}

    def _transport(*, url, headers, body, timeout):
        idx = min(state["i"], len(statuses) - 1)
        state["i"] += 1
        return _FakeResp(statuses[idx], body or {"ok": False})
    return _transport


def _raising_transport(exc_cls: Optional[Exception] = None) -> Callable[..., Any]:
    """返回抛异常的 transport — 网络错模拟。"""
    cls = exc_cls or ConnectionError

    def _transport(*, url, headers, body, timeout):
        raise cls("dipcase 替身端点不可达")
    return _transport


def _slow_transport(sleep_seconds: float, sc: int = 200) -> Callable[..., Any]:
    """返回会超时的 transport — 时间跳变慢。"""
    def _transport(*, url, headers, body, timeout):
        time.sleep(sleep_seconds)
        return _FakeResp(sc)
    return _transport


def _audit_recorder() -> tuple[Callable[..., None], List[Dict[str, Any]]]:
    """返回 audit_writer + 收件箱。"""
    inbox: List[Dict[str, Any]] = []

    def writer(**kwargs):
        inbox.append(dict(kwargs))
    return writer, inbox


def _full_payload() -> Dict[str, Any]:
    """spec §3.2 完整 payload（11 必填字段 + 3 列数组都有内容）。"""
    return {
        "disease_code": "I21",
        "quarter": "2026Q3",
        "outcome": "cmi",
        "treatment": "treatment_strategy",
        "graph_snapshot_dot": "digraph { a -> b }",
        "effect_estimates": [
            {"node": "treatment_strategy", "ate": 0.12, "ci": [0.04, 0.20], "p_value": 0.003},
        ],
        "refutation_results": [
            {"method": "random_common_cause", "passed": True, "p_value": 0.81},
        ],
        "counterfactual_scenarios": [
            {"node": "treatment_strategy", "hypothetical": 0.5, "predicted": 0.78},
        ],
        "narrative_zh": "某季度某单病种 CMI 因果归因说明草稿文本。",
        "reviewer_id": "rev_test_001",
        "model_version": "causal_pipeline/0.1.0",
    }


# =============================================================================
# A. 构造期校验
# =============================================================================


class TestConstructor:
    def test_minimal_construct_ok(self):
        r = DipcaseReporter(
            base_url="https://dipcase.example-hospital.cn/api/v1",
            token="tk_test_bearer",
        )
        assert r.base_url == "https://dipcase.example-hospital.cn/api/v1"
        assert r.token == "tk_test_bearer"
        assert r.timeout == DEFAULT_TIMEOUT_SECONDS
        assert r.max_retries == DEFAULT_MAX_RETRIES
        assert r.path == DIPCASE_REPORTER_PATH

    def test_trailing_slash_stripped(self):
        r = DipcaseReporter(
            base_url="https://example.com/api/",
            token="tk",
        )
        assert r.base_url == "https://example.com/api"
        assert r.post_url == "https://example.com/api" + DIPCASE_REPORTER_PATH

    def test_empty_base_url_raises(self):
        with pytest.raises(DipcaseConfigError):
            DipcaseReporter(base_url="", token="tk")

    def test_empty_token_raises(self):
        with pytest.raises(DipcaseConfigError):
            DipcaseReporter(base_url="https://x", token="")

    def test_negative_timeout_raises(self):
        with pytest.raises(DipcaseConfigError):
            DipcaseReporter(base_url="https://x", token="tk", timeout=-1)

    def test_negative_max_retries_raises(self):
        with pytest.raises(DipcaseConfigError):
            DipcaseReporter(base_url="https://x", token="tk", max_retries=-1)

    def test_retry_base_zero_raises(self):
        with pytest.raises(DipcaseConfigError):
            DipcaseReporter(
                base_url="https://x",
                token="tk",
                retry_base_seconds=0,
            )

    def test_post_url_concatenation(self):
        r = DipcaseReporter(
            base_url="https://a.example.cn",
            token="tk",
            path="/custom/draft",
        )
        assert r.post_url == "https://a.example.cn/custom/draft"


# =============================================================================
# B. from_default_settings
# =============================================================================


class TestFromDefaultSettings:
    def test_uses_env_when_no_override(self, monkeypatch):
        monkeypatch.setenv("DIPCASE_REPORTER_URL", "https://env.example.cn/v1")
        monkeypatch.setenv("DIPCASE_REPORTER_TOKEN", "tk_env_bearer")
        monkeypatch.setenv("DIPCASE_REPORTER_TIMEOUT", "22")

        from app.config import get_settings, reset_settings_cache
        reset_settings_cache()
        try:
            r = DipcaseReporter.from_default_settings()
            assert r.base_url == "https://env.example.cn/v1"
            assert r.token == "tk_env_bearer"
            assert r.timeout == 22
        finally:
            reset_settings_cache()

    def test_override_args_take_precedence(self, monkeypatch):
        monkeypatch.setenv("DIPCASE_REPORTER_URL", "https://env.example.cn")
        monkeypatch.setenv("DIPCASE_REPORTER_TOKEN", "tk_env")
        monkeypatch.setenv("DIPCASE_REPORTER_TIMEOUT", "10")

        from app.config import reset_settings_cache
        reset_settings_cache()
        try:
            r = DipcaseReporter.from_default_settings(
                url="https://explicit.example.cn",
                token="tk_explicit",
                timeout=42,
            )
            assert r.base_url == "https://explicit.example.cn"
            assert r.token == "tk_explicit"
            assert r.timeout == 42
        finally:
            reset_settings_cache()


# =============================================================================
# C. payload 校验 + PII 剔除
# =============================================================================


class TestPayloadValidation:
    def test_full_payload_validates_and_returns_cleaned(self):
        payload = _full_payload()
        cleaned = _validate_payload(payload)
        # schema_version 注入
        assert cleaned["schema_version"] == DIPCASE_PAYLOAD_SCHEMA_VERSION
        # 必填字段全
        for f in (
            "disease_code", "quarter", "outcome", "treatment",
            "graph_snapshot_dot", "effect_estimates", "refutation_results",
            "counterfactual_scenarios", "narrative_zh", "reviewer_id",
            "model_version",
        ):
            assert f in cleaned

    def test_pii_fields_stripped(self):
        payload = _full_payload()
        payload["patient_name"] = "<EXAMPLE_PATIENT>"
        payload["id_card"] = "1234xxxx"
        payload["phone"] = "+86xxxxxxxx"
        payload["address"] = "<EXAMPLE_ADDR>"
        payload["some_safe_field"] = "safe_value"
        cleaned = _validate_payload(payload)
        for blocked in ("patient_name", "id_card", "phone", "address"):
            assert blocked not in cleaned
        assert cleaned["some_safe_field"] == "safe_value"

    def test_pii_nested_in_list_stripped(self):
        payload = _full_payload()
        payload["counterfactual_scenarios"] = [
            {"node": "x", "patient_name": "ABC", "hypothetical": 0.5},
        ]
        cleaned = _validate_payload(payload)
        assert "patient_name" not in cleaned["counterfactual_scenarios"][0]

    def test_missing_disease_code_raises(self):
        payload = _full_payload()
        payload.pop("disease_code")
        with pytest.raises(DipcasePayloadValidationError):
            _validate_payload(payload)

    def test_missing_quarter_raises(self):
        payload = _full_payload()
        payload.pop("quarter")
        with pytest.raises(DipcasePayloadValidationError):
            _validate_payload(payload)

    def test_missing_narrative_zh_raises(self):
        payload = _full_payload()
        payload.pop("narrative_zh")
        with pytest.raises(DipcasePayloadValidationError):
            _validate_payload(payload)

    def test_missing_multiple_fields_lists_all(self):
        payload = _full_payload()
        payload.pop("disease_code")
        payload.pop("outcome")
        with pytest.raises(DipcasePayloadValidationError) as exc_info:
            _validate_payload(payload)
        msg = str(exc_info.value)
        assert "disease_code" in msg
        assert "outcome" in msg

    def test_bad_type_effect_estimates_raises(self):
        payload = _full_payload()
        payload["effect_estimates"] = "not-a-list"
        with pytest.raises(DipcasePayloadValidationError):
            _validate_payload(payload)

    def test_bad_type_narrative_zh_raises(self):
        payload = _full_payload()
        payload["narrative_zh"] = 12345  # int instead of str
        with pytest.raises(DipcasePayloadValidationError):
            _validate_payload(payload)

    def test_none_payload_raises(self):
        with pytest.raises(DipcasePayloadValidationError):
            _validate_payload(None)

    def test_non_mapping_payload_raises(self):
        with pytest.raises(DipcasePayloadValidationError):
            _validate_payload("not-a-mapping")

    def test_strip_pii_handles_nested(self):
        obj = {
            "a": 1,
            "patient_name": "should go",
            "nested": {"phone": "should go", "ok": 1},
            "list": [{"id_card": "xxx", "y": 2}],
            "tuple": ({"phone": "in tuple"}, {"ok_kept": True}),
        }
        cleaned = _strip_pii(obj)
        assert "patient_name" not in cleaned
        assert "phone" not in cleaned["nested"]
        assert "id_card" not in cleaned["list"][0]
        # tuple 路径：剥完 PII 仍是 tuple
        assert isinstance(cleaned["tuple"], tuple)
        assert len(cleaned["tuple"]) == 2
        assert "phone" not in cleaned["tuple"][0]
        assert cleaned["tuple"][1] == {"ok_kept": True}
        assert cleaned["a"] == 1
        assert cleaned["nested"]["ok"] == 1
        assert cleaned["list"][0]["y"] == 2


# =============================================================================
# D. 协议握手 — Headers + URL + JSON body
# =============================================================================


class TestProtocolHandshake:
    def test_post_url_appends_path(self):
        r = DipcaseReporter(
            base_url="https://dipcase.example-hospital.cn/api/v1",
            token="tk_test",
            transport=lambda **kw: _FakeResp(200, {"ok": True}),
        )
        result = r.post_attribution_draft(_full_payload())
        assert result.ok is True
        assert result.status_code == 200

    def test_auth_header_passed_to_transport(self):
        captured: Dict[str, Any] = {}

        def t(**kw):
            captured.update(kw)
            return _FakeResp(200, {"ok": True})

        r = DipcaseReporter(
            base_url="https://example.com",
            token="secret_bearer_xyz",
            transport=t,
        )
        result = r.post_attribution_draft(_full_payload())
        assert result.ok is True
        assert captured["headers"]["Authorization"] == "Bearer secret_bearer_xyz"

    def test_content_type_passed(self):
        captured: Dict[str, Any] = {}

        def t(**kw):
            captured.update(kw)
            return _FakeResp(200, {"ok": True})

        r = DipcaseReporter(base_url="https://x", token="tk", transport=t)
        r.post_attribution_draft(_full_payload())
        assert captured["headers"]["Content-Type"] == CONTENT_TYPE_JSON

    def test_body_is_json_serialized(self):
        captured: Dict[str, Any] = {}

        def t(**kw):
            captured.update(kw)
            return _FakeResp(200, {"ok": True})

        r = DipcaseReporter(base_url="https://x", token="tk", transport=t)
        payload = _full_payload()
        r.post_attribution_draft(payload)
        body = json.loads(captured["body"])
        assert body["disease_code"] == "I21"
        assert body["quarter"] == "2026Q3"
        assert body["outcome"] == "cmi"
        assert body["schema_version"] == DIPCASE_PAYLOAD_SCHEMA_VERSION

    def test_timeout_passed_through(self):
        captured: Dict[str, Any] = {}

        def t(**kw):
            captured.update(kw)
            return _FakeResp(200)

        r = DipcaseReporter(base_url="https://x", token="tk", timeout=42, transport=t)
        r.post_attribution_draft(_full_payload())
        assert captured["timeout"] == 42


# =============================================================================
# E. retry 策略
# =============================================================================


class TestRetryStrategy:
    def test_first_attempt_200_returns_ok(self):
        calls = {"n": 0}

        def t(**kw):
            calls["n"] += 1
            return _FakeResp(200, {"ok": True})

        r = DipcaseReporter(
            base_url="https://x",
            token="tk",
            transport=t,
            max_retries=3,
            retry_base_seconds=0.01,  # 加快测试
        )
        result = r.post_attribution_draft(_full_payload())
        assert result.ok is True
        assert calls["n"] == 1
        assert result.attempt == 1

    def test_retry_on_5xx_until_success(self):
        calls = {"n": 0}

        def t(**kw):
            calls["n"] += 1
            # 前 2 次 5xx，第 3 次 200
            if calls["n"] < 3:
                return _FakeResp(503, {"err": "temporarily unavailable"})
            return _FakeResp(200, {"ok": True})

        r = DipcaseReporter(
            base_url="https://x",
            token="tk",
            transport=t,
            max_retries=3,
            retry_base_seconds=0.01,
        )
        result = r.post_attribution_draft(_full_payload())
        assert result.ok is True
        assert calls["n"] == 3
        assert result.attempt == 3
        assert result.status_code == 200

    def test_4xx_no_retry(self):
        calls = {"n": 0}

        def t(**kw):
            calls["n"] += 1
            return _FakeResp(404, {"err": "draft not found"})

        r = DipcaseReporter(
            base_url="https://x",
            token="tk",
            transport=t,
            max_retries=3,
            retry_base_seconds=0.01,
        )
        result = r.post_attribution_draft(_full_payload())
        assert result.ok is False
        assert calls["n"] == 1
        assert result.status_code == 404
        assert result.degraded is False  # 4xx 不视降级

    def test_5xx_exhaust_returns_degraded(self):
        calls = {"n": 0}

        def t(**kw):
            calls["n"] += 1
            return _FakeResp(500, {"err": "internal"})

        r = DipcaseReporter(
            base_url="https://x",
            token="tk",
            transport=t,
            max_retries=3,
            retry_base_seconds=0.01,
        )
        result = r.post_attribution_draft(_full_payload())
        assert result.ok is False
        assert calls["n"] == 4  # 1 original + 3 retries
        assert result.status_code == 500
        assert result.degraded is True
        assert "retries_exhausted" in (result.error or "") or "http_500" in (result.error or "")

    def test_network_exception_triggers_retry_until_exhaust(self):
        calls = {"n": 0}

        def t(**kw):
            calls["n"] += 1
            raise ConnectionError("替身不可达")

        r = DipcaseReporter(
            base_url="https://x",
            token="tk",
            transport=t,
            max_retries=2,
            retry_base_seconds=0.01,
        )
        result = r.post_attribution_draft(_full_payload())
        assert result.ok is False
        assert calls["n"] == 3  # 1 + 2 retries
        assert result.status_code is None
        assert result.degraded is True
        assert "exception:ConnectionError" in (result.error or "")

    def test_retry_eventually_recovers(self):
        # 1次失败 → 重试 → 成功
        calls = {"n": 0}

        def t(**kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return _FakeResp(502, {"err": "bad gateway"})
            return _FakeResp(200, {"ok": True})

        r = DipcaseReporter(
            base_url="https://x",
            token="tk",
            transport=t,
            max_retries=3,
            retry_base_seconds=0.01,
        )
        result = r.post_attribution_draft(_full_payload())
        assert result.ok is True
        assert calls["n"] == 2
        assert result.status_code == 200


# =============================================================================
# F. 失败回退 + post_result 结构
# =============================================================================


class TestFailureRollback:
    def test_post_result_to_dict_schema(self):
        r = PostResult(
            ok=True,
            status_code=200,
            attempt=1,
            elapsed_ms=12.5,
            degraded=False,
        )
        d = r.to_dict()
        assert d["ok"] is True
        assert d["status_code"] == 200
        assert d["attempt"] == 1
        assert d["elapsed_ms"] == 12.5
        assert d["degraded"] is False
        assert d["schema_version"] == DIPCASE_PAYLOAD_SCHEMA_VERSION

    def test_post_result_failure_fields(self):
        r = PostResult(
            ok=False,
            status_code=None,
            error="retries_exhausted",
            error_reason="4 次尝试失败",
            attempt=4,
            elapsed_ms=500.0,
            degraded=True,
        )
        d = r.to_dict()
        assert d["ok"] is False
        assert d["status_code"] is None
        assert d["error"] == "retries_exhausted"
        assert d["error_reason"] == "4 次尝试失败"
        assert d["degraded"] is True

    def test_exhausted_retry_status_code_preserved(self):
        calls = {"n": 0}

        def t(**kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return _FakeResp(500)
            if calls["n"] == 2:
                return _FakeResp(503)
            return _FakeResp(500)

        r = DipcaseReporter(
            base_url="https://x",
            token="tk",
            transport=t,
            max_retries=2,
            retry_base_seconds=0.01,
        )
        result = r.post_attribution_draft(_full_payload())
        assert result.ok is False
        # 最后一次 status code (500) 应保留
        assert result.status_code == 500

    def test_payload_validation_error_does_not_call_transport(self):
        calls = {"n": 0}

        def t(**kw):
            calls["n"] += 1
            return _FakeResp(200)

        r = DipcaseReporter(
            base_url="https://x",
            token="tk",
            transport=t,
            max_retries=3,
        )
        bad = _full_payload()
        bad.pop("disease_code")
        with pytest.raises(DipcasePayloadValidationError):
            r.post_attribution_draft(bad)
        # 没发请求
        assert calls["n"] == 0


# =============================================================================
# G. Audit 钩子 + best-effort 行为
# =============================================================================


class TestAuditHook:
    def test_audit_writer_invoked_on_success(self):
        writer, inbox = _audit_recorder()
        r = DipcaseReporter(
            base_url="https://x",
            token="tk",
            transport=_ok_transport(),
            audit_writer=writer,
        )
        result = r.post_attribution_draft(_full_payload())
        assert result.ok is True
        assert len(inbox) == 1
        env = inbox[0]
        assert env["event_type"] == "draft/post"
        assert env["operator"] == "rev_test_001"
        assert env["model_version"] == "causal_pipeline/0.1.0"
        assert env["draft_disease"] == "I21"
        assert env["draft_quarter"] == "2026Q3"
        assert env["result"]["ok"] is True
        assert env["result"]["status_code"] == 200

    def test_audit_writer_invoked_on_failure(self):
        writer, inbox = _audit_recorder()

        def t(**kw):
            return _FakeResp(500)

        r = DipcaseReporter(
            base_url="https://x",
            token="tk",
            transport=t,
            audit_writer=writer,
            max_retries=2,
            retry_base_seconds=0.01,
        )
        result = r.post_attribution_draft(_full_payload())
        assert result.ok is False
        assert len(inbox) == 1
        assert inbox[0]["result"]["ok"] is False

    def test_audit_event_strip_pii(self):
        writer, inbox = _audit_recorder()
        r = DipcaseReporter(
            base_url="https://x",
            token="tk",
            transport=_ok_transport(),
            audit_writer=writer,
        )
        payload = _full_payload()
        payload["counterfactual_scenarios"] = [
            {"node": "treatment_strategy", "patient_name": "<EXAMPLE_PATIENT>", "predicted": 0.78},
        ]
        r.post_attribution_draft(payload)
        env = inbox[0]
        # audit envelope 不应含 patient_name
        assert "patient_name" not in json.dumps(env, ensure_ascii=False)
        assert "patient_name" not in env

    def test_audit_failure_does_not_block_business(self):
        def writer(**kw):  # 抛异常
            raise RuntimeError("audit 系统挂了")

        r = DipcaseReporter(
            base_url="https://x",
            token="tk",
            transport=_ok_transport(),
            audit_writer=writer,
        )
        result = r.post_attribution_draft(_full_payload())
        assert result.ok is True  # 上报成功（audit 失败不影响）

    def test_no_audit_writer_no_write(self):
        r = DipcaseReporter(
            base_url="https://x",
            token="tk",
            transport=_ok_transport(),
            audit_writer=None,
        )
        result = r.post_attribution_draft(_full_payload())
        assert result.ok is True
        assert result.audit_event_written is False

    def test_audit_writer_called_once_per_call(self):
        writer, inbox = _audit_recorder()
        r = DipcaseReporter(
            base_url="https://x",
            token="tk",
            transport=_seq_transport([500, 502, 200]),
            audit_writer=writer,
            max_retries=3,
            retry_base_seconds=0.01,
        )
        result = r.post_attribution_draft(_full_payload())
        assert result.ok is True
        # 成功时只写一次 audit envelope（最后一次成功）
        assert len(inbox) == 1
        assert inbox[0]["result"]["ok"] is True


# =============================================================================
# H. 便捷函数 post_attribution_draft
# =============================================================================


class TestConvenienceFunction:
    def test_post_attribution_draft_uses_default_settings(self, monkeypatch):
        monkeypatch.setenv("DIPCASE_REPORTER_URL", "https://env.example.cn/v1")
        monkeypatch.setenv("DIPCASE_REPORTER_TOKEN", "tk_env")
        monkeypatch.setenv("DIPCASE_REPORTER_TIMEOUT", "10")

        from app.config import reset_settings_cache
        reset_settings_cache()
        try:
            captured: Dict[str, Any] = {}

            def t(**kw):
                captured.update(kw)
                return _FakeResp(200, {"ok": True})

            result = post_attribution_draft(
                _full_payload(),
                transport=t,
            )
            assert result.ok is True
            assert "Bearer tk_env" in captured["headers"]["Authorization"]
        finally:
            reset_settings_cache()


# =============================================================================
# I. 端到端冒烟（真实 transport 路径）
# =============================================================================


class TestEndToEndSmoke:
    def test_request_real_protocol_with_mocked_requests_post(self, monkeypatch):
        """Patch requests.post → 验证真实协议调用（method/post/headers/data/timeout）。"""
        # 由于 _default_transport 用 `import requests` 内嵌导入，因此 monkeypatch sys.modules
        mock_post = MagicMock(return_value=_FakeResp(200, {"ok": True, "id": "drf_real"}))
        fake_requests = MagicMock()
        fake_requests.post = mock_post
        monkeypatch.setitem(sys.modules, "requests", fake_requests)

        # 配置 Settings（即便用 from_default_settings 也需要 env 兜底）
        monkeypatch.setenv("DIPCASE_REPORTER_URL", "https://real.example.cn/v1")
        monkeypatch.setenv("DIPCASE_REPORTER_TOKEN", "tk_real")
        monkeypatch.setenv("DIPCASE_REPORTER_TIMEOUT", "10")

        from app.config import reset_settings_cache
        reset_settings_cache()
        try:
            r = DipcaseReporter.from_default_settings()
            result = r.post_attribution_draft(_full_payload())
            assert result.ok is True
            assert result.status_code == 200
            # 真实 requests.post 被调用 1 次
            assert mock_post.call_count == 1
            call_args = mock_post.call_args
            # url 是 positional，headers/data/timeout 是 kwargs
            assert call_args.args[0] == "https://real.example.cn/v1" + DIPCASE_REPORTER_PATH
            kwargs = call_args.kwargs
            assert kwargs["headers"]["Authorization"] == "Bearer tk_real"
            assert kwargs["headers"]["Content-Type"] == "application/json"
            assert kwargs["timeout"] == 10
            # body 是 JSON，可解析
            body = json.loads(kwargs["data"])
            assert body["disease_code"] == "I21"
        finally:
            reset_settings_cache()

    def test_backoff_sleeps_total_within_bounds(self):
        """retry 实际等待时间上限（避免测试等太久）。"""
        # 走 retry_base=0.01 → 每次等待 < 0.05s；3 次重试总 < 0.2s
        def t(**kw):
            return _FakeResp(500)

        r = DipcaseReporter(
            base_url="https://x",
            token="tk",
            transport=t,
            max_retries=2,
            retry_base_seconds=0.01,
        )
        start = time.monotonic()
        result = r.post_attribution_draft(_full_payload())
        elapsed = time.monotonic() - start
        # 留点余量
        assert result.ok is False
        assert elapsed < 2.0


# =============================================================================
# J. 模块常量 + 契约
# =============================================================================


class TestModuleConstants:
    def test_content_type_json(self):
        assert CONTENT_TYPE_JSON == "application/json"

    def test_default_max_retries_three(self):
        assert DEFAULT_MAX_RETRIES == 3

    def test_path_default(self):
        assert DIPCASE_REPORTER_PATH == "/reports/attribution/draft"

    def test_schema_version(self):
        assert DIPCASE_PAYLOAD_SCHEMA_VERSION == "dipcase_attribution/1.0"

    def test_default_timeout_fifteen(self):
        assert DEFAULT_TIMEOUT_SECONDS == 15

    def test_retry_base_half_second(self):
        assert DEFAULT_RETRY_BASE_SECONDS == 0.5

    def test_pii_blocked_fields_constant(self):
        from app.adapters.dipcase_reporter import PII_BLOCKED_FIELDS
        for f in ("patient_name", "id_card", "phone", "address"):
            assert f in PII_BLOCKED_FIELDS

    def test_required_payload_fields_constant(self):
        from app.adapters.dipcase_reporter import REQUIRED_PAYLOAD_FIELDS
        assert "disease_code" in REQUIRED_PAYLOAD_FIELDS
        assert "model_version" in REQUIRED_PAYLOAD_FIELDS
        assert len(REQUIRED_PAYLOAD_FIELDS) == 11

    def test_exception_inheritance(self):
        assert issubclass(DipcaseConfigError, DipcaseReporterError)
        assert issubclass(DipcaseConfigError, ValueError)
        assert issubclass(DipcasePayloadValidationError, DipcaseReporterError)
        assert issubclass(DipcasePayloadValidationError, ValueError)


# =============================================================================
# K. drafts.py 上报钩子兼容（T21 已留 degrade stub；T26 接入真模块）
# =============================================================================


class TestDraftsAPIBridgeCompat:
    """验证 drafts._post_to_dipcase 现在可调到真实模块而不是 stub 降级。"""

    def test_module_importable_and_bridge_uses_real_path(self, monkeypatch):
        """最简路径：drafts.py 内部 `from app.adapters.dipcase_reporter import DipcaseReporter`
        能 import 成功（不再走到 T26 之前的 fallback stub 路径）。"""
        from app.adapters.dipcase_reporter import DipcaseReporter as _DR
        assert _DR is not None

        # drafts.py 的 _post_to_dipcase 内部 `from app.adapters.dipcase_reporter import DipcaseReporter`
        # 是函数内 import；monkeypatch sys.modules 让它拿到 FakeReporter。
        captured = {}

        class FakeReporter:
            @staticmethod
            def from_default_settings(*, url=None, token=None, timeout=None, max_retries=3, transport=None, audit_writer=None):
                captured["factory_called"] = True
                captured["factory_kwargs"] = {
                    "url": url, "token": token, "timeout": timeout, "max_retries": max_retries,
                }
                return FakeReporter()

            def post_attribution_draft(self, payload):
                captured["called"] = True
                captured["payload_keys"] = list(payload.keys())
                return PostResult(
                    ok=True, status_code=200, attempt=1,
                    elapsed_ms=2.0, degraded=False,
                )

        # 造一个真实模块替身（含全部本模块属性 + 替换 DipcaseReporter）
        import types as _types
        import app.adapters.dipcase_reporter as _real_mod
        fake_mod = _types.ModuleType("app.adapters.dipcase_reporter")
        for attr in dir(_real_mod):
            if attr.startswith("_") and attr not in ("_strip_pii", "_validate_payload", "_default_transport", "_do_request", "_is_retryable", "_extract_status_code", "_wait_seconds_for_attempt", "_write_audit", "_auth_header", "_post_to_dipcase"):
                continue
            setattr(fake_mod, attr, getattr(_real_mod, attr))
        fake_mod.DipcaseReporter = FakeReporter
        monkeypatch.setitem(sys.modules, "app.adapters.dipcase_reporter", fake_mod)

        # 设 env 满足 Settings 加载（from_default_settings 内调用 get_settings）
        monkeypatch.setenv("DIPCASE_REPORTER_URL", "https://env.example.cn/v1")
        monkeypatch.setenv("DIPCASE_REPORTER_TOKEN", "tk_env")
        monkeypatch.setenv("DIPCASE_REPORTER_TIMEOUT", "10")

        from app.api import drafts as drafts_mod
        from app.drafts import Draft, DraftStatus, get_default_repository

        repo = get_default_repository()
        repo.clear()
        d = repo.create(
            disease_code="I21",
            outcome="cmi",
            treatment="treatment_strategy",
            quarter="2026Q3",
            narrative_zh="桥接测试草稿",
        )
        d.reviewer_id = "rev_test_001"
        d.reviewed_at = "2026-10-17T00:00:00Z"
        d.status = DraftStatus.REVIEWED.value
        repo.save(d)

        result = drafts_mod._post_to_dipcase(d, action="confirm")
        # 现在走真实路径：ok=True（已不再降级 stub）
        assert result["ok"] is True
        assert captured.get("factory_called") is True
        assert captured.get("called") is True
        # payload 走 draft.to_dict() → spec §3.2 字段
        assert "disease_code" in captured["payload_keys"]
        # drafts.py 内部 from_default_settings() 不显式传 url/token/timeout（从 Settings 取）
        assert captured["factory_kwargs"]["url"] is None
        assert captured["factory_kwargs"]["token"] is None
        assert captured["factory_kwargs"]["timeout"] is None
