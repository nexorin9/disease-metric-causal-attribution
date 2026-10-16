"""test_app_lifespan — FastAPI 入口 + 异常 handler 兜底契约测试（spec §11 收尾）。

覆盖维度：
  1. create_app() 工厂：构造完成 + lifespan 触发 + 元数据
  2. lifespan startup 顺序：Settings → ExceptionHandlers → DB → ModelGateway → Audit
  3. lifespan 启动审计：write_audit_event lifecycle/started 被触发
  4. OperatorIdMiddleware：X-Operator-Id 回落到 settings.local_reviewer_id
  5. /api/info 鉴权与 metadata
  6. EstimandNotIdentifiedError → 422 envelope + audit
  7. InsufficientSampleError → 422 + audit
  8. RefutationUnstableError → 200 envelope（业务降级 ≠ HTTP 失败）
  9. ModelingFailedError → 500 + audit
 10. DipcasePostFailedError → 502 + audit
 11. QualityRoleForbiddenError → 403 + audit
 12. RequestValidationError → 422 + audit
 13. HTTPException(404) 透传 status
 14. 通用 Exception 兜底 500 + audit
 15. envelope schema_version + X-Audit-Event-Id header 一致
 16. 多次 create_app() 互不污染（隔离性）
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
# lifespan 跳过 validate_for_runtime（避免 OPENAI_API_KEY 校验失败阻断 startup）
os.environ["APP_ENV"] = "test"

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from starlette.exceptions import HTTPException as StarletteHTTPException  # noqa: E402

from app.config import get_settings, reset_settings_cache  # noqa: E402
from app.core.exception_handlers import (  # noqa: E402
    ERROR_ENVELOPE_SCHEMA_VERSION,
    DipcasePostFailedError,
    ModelingFailedError,
    QualityRoleForbiddenError,
    RefutationUnstableError,
    install_exception_handlers,
)


# ---------- fixtures ----------

@pytest.fixture
def clean_settings():
    """重置 pydantic-settings lru_cache。"""
    reset_settings_cache()
    get_settings()  # force re-parse env
    yield
    reset_settings_cache()


@pytest.fixture
def in_memory_audit_logger():
    """替身 AuditLogger，写入不落 DB，仅存到 self.events 列表。

    通过 patch app.main._build_audit_logger 在 lifespan 内被注入到 app.state.audit_logger。
    """
    events: list = []

    class _MemAudit:
        def write_audit_event(self, *, event_type: str, operator: str, payload: dict) -> str:
            ev_id = f"mem_{len(events):04d}"
            events.append(
                {"event_type": event_type, "operator": operator, "payload": payload, "id": ev_id}
            )
            return ev_id

    return _MemAudit(), events


@pytest.fixture
def app_and_audit(monkeypatch, in_memory_audit_logger, clean_settings):
    """构造应用 + 在 startup 后用 in-memory 审计 logger 替换 audit_logger。

    替换方式：先把 lifespan 跑完，再把 state.audit_logger 替换为 _MemAudit，
    这样：(a) startup 审计走真实路径；(b) 业务异常审计走 mem 替身。
    """
    audit, events = in_memory_audit_logger

    import app.main as app_main

    # 把 audit 与 model_gateway 的构造都替成安全降级，避免对 DB / 真实基座连线的副作用
    monkeypatch.setattr(app_main, "_build_audit_logger", lambda settings: audit)
    monkeypatch.setattr(app_main, "_build_model_gateway", lambda settings: None)
    monkeypatch.setattr(app_main, "_build_session_factory", lambda settings: None)

    app = app_main.create_app()

    # raise_server_exceptions=False 让异常经 envelope handler 处理后正常返回 4xx/5xx，
    # 否则 TestClient 会把异常 re-raise 到测试侧。
    with TestClient(app, raise_server_exceptions=False) as client:
        # lifespan 已触发；现在替换 audit_logger 让 handler 写进 mem
        client.app.state.audit_logger = audit
        yield client, audit, events

    # 退出后清 settings 单例
    reset_settings_cache()


# ---------- 1. 工厂与元数据 ----------

def test_create_app_returns_fastapi_instance(clean_settings):
    """create_app() 返回 FastAPI 实例 + 元数据齐。"""
    import app.main as app_main

    monkeypatch_attr = [
        ("_build_audit_logger", lambda s: None),
        ("_build_model_gateway", lambda s: None),
        ("_build_session_factory", lambda s: None),
    ]

    import unittest.mock as mock
    with mock.patch.multiple(app_main, **{k: v for k, v in monkeypatch_attr}):
        app = app_main.create_app()
    assert isinstance(app, FastAPI)
    assert app.title == "单病种指标因果归因系统"
    assert app.version == "0.1.0"
    assert app.openapi_url == "/api/openapi.json"
    assert app.docs_url == "/api/docs"


def test_app_module_level_app_is_singleton(clean_settings):
    """模块顶层的 app: FastAPI = create_app() 是单例。"""
    import unittest.mock as mock
    import app.main as app_main

    with mock.patch.multiple(
        app_main,
        _build_audit_logger=lambda s: None,
        _build_model_gateway=lambda s: None,
        _build_session_factory=lambda s: None,
    ):
        from importlib import reload
        reload(app_main)
        from app.main import app as top_level_app

        assert isinstance(top_level_app, FastAPI)
        assert top_level_app.title == "单病种指标因果归因系统"


# ---------- 2. lifespan startup 顺序与状态 ----------

def test_lifespan_startup_attaches_state(app_and_audit):
    """lifespan 必须把 settings/db_session_factory/model_gateway/audit_logger/registered_paths 注入 state。"""
    client, audit, events = app_and_audit
    app = client.app

    assert app.state.settings is not None
    # DB session 工厂为 None（fixture monkeypatch）
    assert app.state.db_session_factory is None
    # model_gateway 为 None（fixture monkeypatch）
    assert app.state.model_gateway is None
    # audit_logger 已替换为 _MemAudit
    assert app.state.audit_logger is audit
    # registered_paths 是 list
    assert isinstance(app.state.registered_paths, list)
    # startup_events 至少含一条 lifecycle/started
    started = [e for e in events if e["event_type"] == "lifecycle/started"]
    assert started, "lifespan 必须审计 lifecycle/started"


def test_lifespan_started_event_payload(app_and_audit):
    """lifecycle/started 审计 payload 含驱动、registered_paths、组件可用性。"""
    client, audit, events = app_and_audit
    started = [e for e in events if e["event_type"] == "lifecycle/started"]
    assert started
    payload = started[0]["payload"]
    assert "driver_kind" in payload
    assert "app_env" in payload
    assert "registered_paths" in payload
    assert isinstance(payload["has_db_session"], bool)
    assert isinstance(payload["has_model_gateway"], bool)


# ---------- 3. OperatorIdMiddleware 回退 ----------

def test_operator_id_middleware_falls_back_when_missing(app_and_audit):
    """未传 X-Operator-Id 时回落到 settings.local_reviewer_id。"""
    client, audit, _ = app_and_audit
    settings = client.app.state.settings

    resp = client.get("/api/info")
    assert resp.status_code == 200
    assert resp.headers.get("X-Operator-Id") == settings.local_reviewer_id


def test_operator_id_middleware_echoes_custom_header(app_and_audit):
    """X-Operator-Id 显式传入时原样回传。"""
    client, audit, _ = app_and_audit
    resp = client.get("/api/info", headers={"X-Operator-Id": "tester-007"})
    assert resp.status_code == 200
    assert resp.headers["X-Operator-Id"] == "tester-007"


# ---------- 4. /api/info metadata ----------

def test_info_endpoint_metadata(app_and_audit):
    """/api/info 返回 name/version/app_env/openai_model。"""
    client, audit, _ = app_and_audit
    settings = client.app.state.settings
    body = client.get("/api/info").json()
    assert body["name"] == "disease-metric-causal-attribution"
    assert body["version"] == "0.1.0"
    assert body["app_env"] == settings.app_env
    assert "openai_model" in body


# ---------- 5. 异常 handler 登记 ----------

def _make_route_that_raises(app: FastAPI, exc: Exception, path: str = "/_test/raise") -> None:
    """在 app 上注入一个抛异常的路由，专门用于异常 handler 测试。"""
    from fastapi import APIRouter

    test_router = APIRouter()

    @test_router.get(path, tags=["system"])
    async def _raise():
        raise exc

    app.include_router(test_router)


def _audit_events_matching(events, error_code: str):
    return [e for e in events if e["payload"].get("error_code") == error_code]


def test_estimand_not_identified_returns_422(app_and_audit):
    """EstimandNotIdentifiedError → 422 + envelope + audit。"""
    from app.core.pipeline import EstimandNotIdentifiedError

    client, audit, events = app_and_audit
    _make_route_that_raises(client.app, EstimandNotIdentifiedError("图与字段不一致"))

    resp = client.get("/_test/raise")
    assert resp.status_code == 422
    body = resp.json()
    assert body["schema_version"] == ERROR_ENVELOPE_SCHEMA_VERSION
    assert body["ok"] is False
    assert body["error_type"] == "EstimandNotIdentifiedError"
    assert body["error_code"] == "estimand_not_identified"
    assert "audit_event_id" in body and body["audit_event_id"].startswith("ev_")
    assert resp.headers.get("X-Audit-Event-Id") == body["audit_event_id"]
    # 审计被触发
    matched = _audit_events_matching(events, "estimand_not_identified")
    assert matched, "handler 必须写 audit；payload 含 error_code=estimand_not_identified"


def test_insufficient_sample_returns_422(app_and_audit):
    """InsufficientSampleError → 422。"""
    from app.core.pipeline import InsufficientSampleError

    client, audit, events = app_and_audit
    _make_route_that_raises(
        client.app, InsufficientSampleError(sample_size=20, min_required=30)
    )

    resp = client.get("/_test/raise")
    assert resp.status_code == 422
    body = resp.json()
    assert body["error_code"] == "insufficient_sample"
    assert "样本量不足" in body["message"]
    # 附加字段被抽出
    assert body["details"]["sample_size"] == 20
    assert body["details"]["min_required"] == 30
    # 审计
    assert _audit_events_matching(events, "insufficient_sample")


def test_refutation_unstable_returns_200(app_and_audit):
    """RefutationUnstableError → 200（业务降级 ≠ HTTP 失败）。"""
    client, audit, events = app_and_audit
    _make_route_that_raises(
        client.app, RefutationUnstableError("CI=0.7>0.6 触发降级")
    )

    resp = client.get("/_test/raise")
    assert resp.status_code == 200  # 业务降级仍 200
    body = resp.json()
    assert body["ok"] is False
    assert body["error_code"] == "refutation_unstable"
    assert _audit_events_matching(events, "refutation_unstable")


def test_modeling_failed_returns_500(app_and_audit):
    """ModelingFailedError → 500 + 审计。"""
    client, audit, events = app_and_audit
    _make_route_that_raises(
        client.app, ModelingFailedError("主路径中断", stage="estimate")
    )

    resp = client.get("/_test/raise")
    assert resp.status_code == 500
    body = resp.json()
    assert body["error_code"] == "modeling_failed"
    assert body["details"]["stage"] == "estimate"
    assert _audit_events_matching(events, "modeling_failed")


def test_dipcase_post_failed_returns_502(app_and_audit):
    """DipcasePostFailedError → 502 + 审计。"""
    client, audit, events = app_and_audit
    _make_route_that_raises(
        client.app, DipcasePostFailedError("3 次重试均失败", attempts=3)
    )

    resp = client.get("/_test/raise")
    assert resp.status_code == 502
    body = resp.json()
    assert body["error_code"] == "dipcase_post_failed"
    assert body["details"]["attempts"] == 3
    assert _audit_events_matching(events, "dipcase_post_failed")


def test_quality_role_forbidden_returns_403(app_and_audit):
    """QualityRoleForbiddenError → 403。"""
    client, audit, events = app_and_audit
    _make_route_that_raises(
        client.app, QualityRoleForbiddenError("角色非法", role="intern")
    )

    resp = client.get("/_test/raise")
    assert resp.status_code == 403
    body = resp.json()
    assert body["error_code"] == "quality_role_forbidden"
    assert body["details"]["role"] == "intern"
    assert _audit_events_matching(events, "quality_role_forbidden")


def test_request_validation_error_returns_422(app_and_audit):
    """RequestValidationError → 422 + envelope（FastAPI 内置异常，由 handler 统一包装）。"""
    from fastapi import APIRouter
    from pydantic import BaseModel

    # 注：在 fixture 的 app 上加一个期望 int 的路由；传 str 触发 422
    test_router = APIRouter()

    class _Params(BaseModel):
        n: int

    @test_router.get("/_test/validate", tags=["system"])
    async def _validate_endpoint(n: int = 0) -> dict:
        return {"n": n}

    client, audit, events = app_and_audit
    client.app.include_router(test_router)
    resp = client.get("/_test/validate?n=abc")
    assert resp.status_code == 422
    body = resp.json()
    assert body["error_type"] == "RequestValidationError"
    assert body["error_code"] == "request_validation_error"
    assert _audit_events_matching(events, "request_validation_error")


def test_http_exception_passes_through_status(app_and_audit):
    """HTTPException(404) 透传 status（不强制 envelope 化）。"""
    client, audit, events = app_and_audit
    _make_route_that_raises(client.app, StarletteHTTPException(status_code=404, detail="Not Found"))

    resp = client.get("/_test/raise")
    assert resp.status_code == 404
    # StarletteHTTPException handler 输出 envelope，但保留 status
    body = resp.json()
    assert body["error_code"] == "http_exception"
    assert body["details"]["status_code"] == 404
    assert _audit_events_matching(events, "http_exception")


def test_unhandled_exception_returns_500(app_and_audit):
    """未注册异常 → 兜底 500 + audit。"""
    client, audit, events = app_and_audit
    _make_route_that_raises(client.app, ValueError("未捕获业务异常"))

    resp = client.get("/_test/raise")
    assert resp.status_code == 500
    body = resp.json()
    assert body["error_code"] == "unhandled_exception"
    assert body["error_type"] == "ValueError"
    assert "exc_repr" in body["details"]
    assert _audit_events_matching(events, "unhandled_exception")


# ---------- 6. envelope shape 一致性 ----------

def test_envelope_schema_and_audit_event_id_consistent(app_and_audit):
    """envelope schema_version 固定 + X-Audit-Event-Id 与 body.audit_event_id 一致。"""
    from app.core.pipeline import EstimandNotIdentifiedError

    client, _, _ = app_and_audit
    _make_route_that_raises(client.app, EstimandNotIdentifiedError("x"))

    resp = client.get("/_test/raise")
    body = resp.json()
    assert body["schema_version"] == ERROR_ENVELOPE_SCHEMA_VERSION
    assert resp.headers["X-Audit-Event-Id"] == body["audit_event_id"]


def test_audit_best_effort_does_not_propagate(app_and_audit):
    """audit_logger.write_audit_event 抛错也不影响 envelope 返回。"""
    from app.core.pipeline import EstimandNotIdentifiedError

    client, _, _ = app_and_audit

    # 替换为抛错的 audit 替身
    class _Boom:
        def write_audit_event(self, **_):
            raise RuntimeError("audit sink dead")

    original = client.app.state.audit_logger
    client.app.state.audit_logger = _Boom()
    try:
        _make_route_that_raises(client.app, EstimandNotIdentifiedError("y"))
        resp = client.get("/_test/raise")
        assert resp.status_code == 422
        assert resp.json()["ok"] is False
    finally:
        client.app.state.audit_logger = original


# ---------- 7. install_exception_handlers 幂等 ----------

def test_install_exception_handlers_idempotent():
    """同一 app 上多次 install_exception_handlers 不会抛错。"""
    app = FastAPI()
    install_exception_handlers(app)
    install_exception_handlers(app)  # 第二轮不应抛错
    # handler 数 > 0
    assert len(app.exception_handlers) > 0


# ---------- 8. 隔离：create_app() 多次调用互不污染 ----------

def test_repeated_create_app_isolated(monkeypatch, clean_settings):
    """反复 create_app() 应得到互相独立的实例 + 互相独立的 state。"""
    import app.main as app_main

    monkeypatch.setattr(app_main, "_build_audit_logger", lambda s: None)
    monkeypatch.setattr(app_main, "_build_model_gateway", lambda s: None)
    monkeypatch.setattr(app_main, "_build_session_factory", lambda s: None)

    a = app_main.create_app()
    b = app_main.create_app()
    assert a is not b
    a.state.injected_tag = "tag-a"
    b.state.injected_tag = "tag-b"
    assert a.state.injected_tag != b.state.injected_tag


# ---------- 9. uvicorn entry / run 函数 ----------

def test_run_is_callable():
    """app.main.run 是 callable（不真启动 uvicorn）。"""
    import app.main as app_main

    assert callable(app_main.run)


# ---------- 10. module 顶层 app 已实例化 ----------

def test_module_top_level_app_built(clean_settings):
    """import app.main 即调用 create_app()；顶层 app 是 FastAPI。"""
    import importlib

    import app.main as app_main
    importlib.reload(app_main)
    assert isinstance(app_main.app, FastAPI)
    assert app_main.app.title == "单病种指标因果归因系统"
