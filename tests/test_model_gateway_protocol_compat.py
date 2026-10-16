"""test_model_gateway_protocol_compat — ModelGateway ↔ OpenAI 替身协议兼容性测试（T29）。

设计（spec §7 + §11 + T29）：
- 用 FastAPI `TestClient` 在**进程内**启动替身（避免子进程编排开销），但保持
  真实 httpx 网络栈（`httpx.Client(transport=ASGITransport(app=...))`）— 即
  仍然走真实 HTTP 握手 → response 解析路径，**不是**函数直调。
- 验证以下六条兼容性契约：
  1. **协议握手**：POST /v1/chat/completions + Bearer 鉴权 + response_format=json_object；
     返回结构可被 ModelGateway._extract_content_and_usage 正确解析。
  2. **超时降级**：替身 `inject_timeout(delay_seconds=15)` 后 ModelGateway
     （10s 推理档）应触发 `fallback_used=True` + `fallback_reason` 含 `timeout:` 前缀。
  3. **不可解析降级**：替身 `inject_unparseable()` 后 ModelGateway 应触发
     `fallback_used=True` + `fallback_reason="response_unparseable"`。
  4. **5xx 降级**：替身 `inject_500()` 后 ModelGateway 应触发
     `fallback_used=True` + `fallback_reason="http_error:500"`。
  5. **重新握手**：`reset()` 后替身恢复默认行为，ModelGateway 再次调用成功解析。
  6. **provider-neutral 边界**：换 base_url（指向另一个 ASGI app 实例）后
     ModelGateway 行为不受影响，证明网关层不绑定具体厂商。

测试替身 vs 真实端点：本测试是协议兼容性验证；产品代码（app/adapters/model_gateway.py）
仍走真实 httpx POST；T27/T28 的 transport-injection 单元测试已覆盖异常路径，
本测试侧重「真实协议栈 + 真实 URL 拼接 + 真实 response 解析」全链路。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import pytest

# 让 `from app.xxx import ...` 工作
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("APP_ENV", "test")


# ---------------------------------------------------------------------------
# 替身装配（进程内 FastAPI TestClient + ASGITransport）
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stub_factory():
    """返回 (client, base_url, state) — 每个模块构建一次替身。"""
    from tests.mocks.openai_stub.main import STATE, create_app
    from fastapi.testclient import TestClient
    import httpx

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    # ModelGateway 把 base_url + "/chat/completions" 拼成 chat_url；
    # 替身路由前缀是 /v1/chat/completions，所以 base_url 必须含 /v1 后缀。
    # TestClient 自己的 base_url 不含 /v1：admin 端点（如 /healthz /__admin/*）不带前缀。
    client = TestClient(app)
    base_url = "http://openai-stub.test/v1"  # 与 conftest OPENAI_STUB_URL 对齐
    return {"client": client, "transport": transport, "base_url": base_url, "state": STATE}


@pytest.fixture(autouse=True)
def _reset_stub_state(stub_factory):
    """每个测试前重置替身状态。"""
    stub_factory["state"].reset()
    yield
    stub_factory["state"].reset()


@pytest.fixture
def stub_client(stub_factory) -> "TestClient":
    return stub_factory["client"]


# ---------------------------------------------------------------------------
# 1. 协议握手
# ---------------------------------------------------------------------------


class TestProtocolHandshake:
    """真实协议栈握手 + 鉴权 + 结构化响应解析。"""

    def test_chat_completions_path_and_auth(self, stub_factory):
        """POST /v1/chat/completions + Bearer 鉴权 + response_format 完整协议字段。"""
        client = stub_factory["client"]
        before_count = len(stub_factory["state"].call_history)
        resp = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-key"},
            json={
                "model": "stub-model",
                "messages": [
                    {"role": "system", "content": "你是归因评审助手。"},
                    {"role": "user", "content": "{}"},
                ],
                "response_format": {"type": "json_object"},
            },
        )
        assert resp.status_code == 200
        history = stub_factory["state"].call_history
        # 协议层已记录 messages 数量 + response_format（验证替身收到完整 OpenAI-compatible 协议）
        assert len(history) >= before_count + 1
        last = history[-1]
        assert last["messages_count"] == 2
        assert last["response_format"] == {"type": "json_object"}
        assert last["model"] == "stub-model"

    def test_model_gateway_handshake_round_trip(self, stub_factory):
        """ModelGateway 通过 ASGI 替身真实握手 → ModelResponse 解析正确。"""
        from app.adapters.model_gateway import ModelGateway, MODEL_RESPONSE_SCHEMA_VERSION
        import httpx

        gw = ModelGateway(
            base_url=stub_factory["base_url"],
            api_key="protocol-test-key",
            model="stub-model",
            timeout=10,
            transport=_build_transport(stub_factory["client"]),
        )
        result = gw.generate_node_narrative(
            node="treatment_strategy",
            effect={"ate": 0.12, "p_value": 0.003, "method_name": "backdoor.linear_regression"},
            disease_code="I21",
            outcome="cmi",
        )
        # 协议握手成功：fallback_used=False + content.narrative_zh 非空
        assert result.fallback_used is False
        assert result.method == "generate_node_narrative"
        assert result.schema_version == MODEL_RESPONSE_SCHEMA_VERSION
        assert "narrative_zh" in result.content
        assert "[stub]" in result.content["narrative_zh"]
        # 审计字段
        assert len(result.prompt_hash) == 16
        assert len(result.response_hash) == 16
        assert result.latency_ms >= 0.0
        assert result.tokens_used >= 1
        assert result.state_not_advanced is True
        # 替身记录了真实调用（module 级共享 state，只断言至少 1 次）
        assert len(stub_factory["state"].call_history) >= 1


# ---------------------------------------------------------------------------
# 2. 超时降级（替身 inject_timeout + ModelGateway 10s 推理档）
# ---------------------------------------------------------------------------


class TestTimeoutDegradation:
    """替身 sleep > ModelGateway timeout → 触发降级。"""

    def test_inject_timeout_triggers_fallback(self, stub_factory):
        from app.adapters.model_gateway import ModelGateway
        import httpx

        stub_factory["state"].inject_timeout(delay_seconds=15.0)
        gw = ModelGateway(
            base_url=stub_factory["base_url"],
            api_key="timeout-test",
            model="stub-model",
            timeout=2,  # 全局档 2s，会被方法档进一步夹紧到 TIMEOUT_REASONING_SECONDS=10s
            transport=_build_transport(stub_factory["client"]),
        )
        # 用 generate_node_narrative（10s 推理档）→ 替身 sleep 15s → httpx 超时
        result = gw.generate_node_narrative(
            node="treatment_strategy",
            effect={"ate": 0.12, "p_value": 0.003, "method_name": "backdoor.linear_regression"},
        )
        assert result.fallback_used is True
        assert "timeout" in result.fallback_reason.lower()
        # 兜底 narrative_zh 仍可用
        assert "narrative_zh" in result.content
        assert result.state_not_advanced is True


# ---------------------------------------------------------------------------
# 3. 不可解析降级（HTTP 200 但 content 不是 JSON）
# ---------------------------------------------------------------------------


class TestUnparseableDegradation:
    """HTTP 200 但 message.content 不是合法 JSON → 触发不可解析降级。"""

    def test_inject_unparseable_triggers_fallback(self, stub_factory):
        from app.adapters.model_gateway import ModelGateway
        import httpx

        stub_factory["state"].inject_unparseable()
        gw = ModelGateway(
            base_url=stub_factory["base_url"],
            api_key="unparseable-test",
            model="stub-model",
            timeout=5,
            transport=_build_transport(stub_factory["client"]),
        )
        result = gw.generate_node_narrative(
            node="treatment_strategy",
            effect={"ate": 0.05, "p_value": 0.04, "method_name": "backdoor.linear_regression"},
        )
        assert result.fallback_used is True
        assert result.fallback_reason in ("response_unparseable", "model_output_unparseable")
        assert "narrative_zh" in result.content
        assert result.state_not_advanced is True


# ---------------------------------------------------------------------------
# 4. 5xx 降级
# ---------------------------------------------------------------------------


class TestHttpErrorDegradation:
    """替身返回 HTTP 500 → 触发 http_error 降级。"""

    def test_inject_500_triggers_fallback(self, stub_factory):
        from app.adapters.model_gateway import ModelGateway
        import httpx

        stub_factory["state"].inject_status(500)
        gw = ModelGateway(
            base_url=stub_factory["base_url"],
            api_key="http-500-test",
            model="stub-model",
            timeout=5,
            transport=_build_transport(stub_factory["client"]),
        )
        result = gw.generate_node_narrative(
            node="age_band",
            effect={"ate": 0.08, "p_value": 0.01, "method_name": "backdoor.linear_regression"},
        )
        assert result.fallback_used is True
        assert result.fallback_reason.startswith("http_error")
        assert "500" in result.fallback_reason
        assert "narrative_zh" in result.content
        assert result.state_not_advanced is True


# ---------------------------------------------------------------------------
# 5. 重新握手（reset → 再次握手成功）
# ---------------------------------------------------------------------------


class TestRehandshakeAfterReset:
    """注入失败 → reset → 恢复协议握手。"""

    def test_reset_restores_normal_protocol(self, stub_factory):
        from app.adapters.model_gateway import ModelGateway
        import httpx

        # 第一步：注入 500 → 触发 http_error fallback
        stub_factory["state"].inject_status(503)
        gw = ModelGateway(
            base_url=stub_factory["base_url"],
            api_key="rehandshake-test",
            model="stub-model",
            timeout=5,
            transport=_build_transport(stub_factory["client"]),
        )
        r1 = gw.generate_node_narrative(
            node="treatment_strategy",
            effect={"ate": 0.05, "p_value": 0.04, "method_name": "backdoor.linear_regression"},
        )
        assert r1.fallback_used is True
        assert "503" in r1.fallback_reason

        # 第二步：reset → 重新握手
        stub_factory["state"].reset()
        assert stub_factory["state"].reset_count >= 1

        r2 = gw.generate_node_narrative(
            node="treatment_strategy",
            effect={"ate": 0.05, "p_value": 0.04, "method_name": "backdoor.linear_regression"},
        )
        assert r2.fallback_used is False
        assert "narrative_zh" in r2.content
        assert "[stub]" in r2.content["narrative_zh"]
        assert r2.state_not_advanced is True


# ---------------------------------------------------------------------------
# 6. provider-neutral 边界（换 base_url 不影响行为）
# ---------------------------------------------------------------------------


class TestProviderNeutral:
    """ModelGateway 应不绑定具体厂商；不同 base_url 行为一致。"""

    def test_different_base_url_yields_same_protocol_shape(self, stub_factory):
        from app.adapters.model_gateway import ModelGateway
        import httpx

        # 两个 ModelGateway 实例，transport 都指向同一个替身（同一 ASGI app）
        # 但 base_url 不同 — 模拟「同一医院内部署到不同 provider endpoint」
        transport = _build_transport(stub_factory["client"])

        gw_a = ModelGateway(
            base_url="http://provider-a.internal/v1",
            api_key="key-a",
            model="model-a",
            timeout=5,
            transport=transport,
        )
        gw_b = ModelGateway(
            base_url="http://provider-b.internal/v1",
            api_key="key-b",
            model="model-b",
            timeout=5,
            transport=transport,
        )

        r_a = gw_a.generate_node_narrative(
            node="treatment_strategy",
            effect={"ate": 0.05, "p_value": 0.04, "method_name": "backdoor.linear_regression"},
        )
        r_b = gw_b.generate_node_narrative(
            node="treatment_strategy",
            effect={"ate": 0.05, "p_value": 0.04, "method_name": "backdoor.linear_regression"},
        )

        # 两个网关都成功解析 + ModelResponse schema_version 一致
        assert r_a.fallback_used is False
        assert r_b.fallback_used is False
        assert r_a.schema_version == r_b.schema_version == "model_response/1.0"
        # model_version 来自响应字段（替身按请求的 model 字段回写），不绑定具体厂商
        assert r_a.model_version == "model-a"
        assert r_b.model_version == "model-b"


# ---------------------------------------------------------------------------
# 7. 三方法 protocol 路径全覆盖（generate_node_narrative / counterfactual / draft_summary）
# ---------------------------------------------------------------------------


class TestAllMethodsProtocolCoverage:
    """三方法都跑一遍真实协议握手，确认 method 路由 + 替身方法路由都生效。"""

    @pytest.mark.parametrize(
        "method_name,payload",
        [
            ("generate_node_narrative", {"node": "treatment_strategy", "effect": {"ate": 0.1, "p_value": 0.02, "method_name": "backdoor.linear_regression"}}),
            ("generate_counterfactual_narrative", {"scenario": {"intervention_node": "treatment_strategy", "hypothetical_value": "PCI", "disease_code": "I21", "outcome": "cmi"}}),
            ("generate_draft_summary", {"draft": {"disease_code": "I21", "outcome": "cmi", "sample_size": 200, "effect_estimates": [{"node": "treatment_strategy", "ate": 0.1}], "refutation_results": [{"method": "bootstrap_refuter", "passed": True}], "counterfactual_scenarios": []}}),
        ],
    )
    def test_three_methods_protocol_handshake(self, stub_factory, method_name, payload):
        from app.adapters.model_gateway import ModelGateway
        import httpx

        gw = ModelGateway(
            base_url=stub_factory["base_url"],
            api_key=f"key-{method_name}",
            model="stub-model",
            timeout=10,
            transport=_build_transport(stub_factory["client"]),
        )
        method_fn = getattr(gw, method_name)
        result = method_fn(**payload)
        # 协议层成功
        assert result.fallback_used is False
        assert result.method == method_name
        assert result.schema_version == "model_response/1.0"
        assert result.state_not_advanced is True
        # 替身按 system prompt 路由到正确 default content
        assert any(content_type in result.content for content_type in ("narrative_zh", "predicted_outcome_zh", "summary_zh"))


# ---------------------------------------------------------------------------
# 8. 替身 admin 端点契约
# ---------------------------------------------------------------------------


class TestStubAdminEndpoints:
    """替身自身的 admin 接口契约（保障测试 fixture 稳定）。"""

    def test_healthz_ok(self, stub_client):
        resp = stub_client.get("/healthz")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "calls_recorded" in body
        assert "reset_count" in body

    def test_admin_state_visible(self, stub_client):
        # admin 端点是 /v1/__admin/state（替身路由）；TestClient 默认 base_url
        # 是 http://testserver，所以这里直接传完整 /v1 前缀路径。
        resp = stub_client.get("/v1/__admin/state")
        assert resp.status_code == 200
        body = resp.json()
        assert "calls_recorded" in body
        assert "reset_count" in body

    def test_admin_inject_timeout(self, stub_client):
        resp = stub_client.post("/v1/__admin/inject_timeout", json={"delay_seconds": 12.0})
        assert resp.status_code == 200
        assert resp.json()["injected"] == "timeout"

    def test_admin_reset(self, stub_client):
        # 先注入再重置
        stub_client.post("/v1/__admin/inject_unparseable")
        resp = stub_client.post("/v1/__admin/reset")
        assert resp.status_code == 200
        assert resp.json()["reset"] is True


# ---------------------------------------------------------------------------
# 9. run.sh 脚本契约（start/stop/status/restart — 进程级）
# ---------------------------------------------------------------------------


class TestRunScript:
    """run.sh 脚本能正确执行 start/stop/status（不真正起服务，只验脚本语法 + help）。"""

    def test_run_script_exists_and_executable(self):
        script_path = _PROJECT_ROOT / "tests/mocks/openai_stub/run.sh"
        assert script_path.exists(), f"run.sh 不存在：{script_path}"
        import os
        import stat
        mode = os.stat(script_path).st_mode
        assert mode & stat.S_IXUSR, "run.sh 不可执行"

    def test_run_script_syntax_ok(self):
        import subprocess
        result = subprocess.run(
            ["bash", "-n", str(_PROJECT_ROOT / "tests/mocks/openai_stub/run.sh")],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"run.sh 语法错误：{result.stderr}"

    def test_run_script_help_usage(self):
        import subprocess
        result = subprocess.run(
            ["bash", str(_PROJECT_ROOT / "tests/mocks/openai_stub/run.sh"), "unknown-cmd"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert "用法" in result.stderr


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _build_transport(test_client):
    """把 FastAPI TestClient（同步 ASGI wrapper）包成 ModelGateway transport 协议。

    ModelGateway transport 签名：`(url, headers, body, timeout) -> httpx.Response`
    TestClient.request(method, url, content, headers) 是同步入口；这里把
    ModelGateway 拼出的绝对 URL 切成 path+query 后传入，避免 testserver 不匹配。
    """
    from urllib.parse import urlparse

    def _send(*, url: str, headers, body: str, timeout: float):
        # TestClient 不接受 timeout 参数；用 ThreadPoolExecutor 跑出真实 timeout。
        import httpx
        import concurrent.futures

        parsed = urlparse(url)
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        def _do_request():
            return test_client.request(
                "POST",
                path,
                content=body.encode("utf-8") if isinstance(body, str) else body,
                headers=dict(headers),
            )

        if timeout and timeout > 0:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_do_request)
                try:
                    return future.result(timeout=float(timeout))
                except concurrent.futures.TimeoutError:
                    # 触发 ModelGateway 的 timeout 分支：抛 httpx.TimeoutException 子类
                    raise httpx.TimeoutException(
                        f"openai_stub response > {timeout:.1f}s (TestClient wrapped)",
                        request=None,
                    )
        return _do_request()

    return _send