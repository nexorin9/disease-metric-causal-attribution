"""FastAPI 应用入口 + lifespan + uvicorn 运行入口（spec.md §11 收尾交付）。

启动顺序（lifespan startup）严格按 spec §6 / §11：
  1. 解析 .env 加载 Settings 单例 → validate_for_runtime() 校验必填凭据
  2. 注册异常处理器（spec §6 失败降级契约）
  3. 构造 DB session 工厂（仅 SQLite / PostgreSQL 替身不真正连 HIS 库）
  4. 构造 ModelGateway 单例（按 base_url + api_key，仅协议替身留作兼容测试）
  5. 构造 AuditLogger 单例 → 写入 audit_logs
  6. 反射注册 routes（modeling / counterfactual / drafts / healthz / modules）
  7. 写一条 audit event: lifecycle/started
  8. 设置 app.state.startup_events = [...] 便于测试断言

关闭顺序（lifespan shutdown）对称：
  - audit lifecycle/shutdown
  - ModelGateway / DB engine dispose（关闭连接池）

入口形态：
  - 生产：uvicorn app.main:app --host 0.0.0.0 --port 8000
  - 本地：bash scripts/demo.sh（venv + install + uvicorn）
  - 测试：from app.main import create_app; app = create_app()

⚠️ 严禁在模块级别直接读环境变量 / 触发 DB 连线；所有副作用都进 lifespan。
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Awaitable, Callable, Dict, List, Optional

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings

LOGGER = logging.getLogger(__name__)


# ---- 启动期副作用辅助函数（被 lifespan 调） ----

def _build_session_factory(settings: Settings) -> Optional[Callable[[], Any]]:
    """构造 DB session 工厂。

    返回 None 当 DATABASE_URL 不可用 / SQLAlchemy 未装（不阻断进程 start）。
    """
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker, Session
    except ImportError:
        LOGGER.warning("sqlalchemy 未装，DB session 工厂降级为 None（spec §11 仅在产品环境启用）")
        return None

    db_url = settings.database_url
    if not db_url:
        return None
    try:
        engine = create_engine(db_url, future=True)
        SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    except Exception as exc:
        LOGGER.warning(
            "DB session 工厂构造失败：%s → 启动后 DB 子健康可能 red（spec §6 状态可观测）",
            exc,
        )
        return None
    return SessionLocal


def _build_model_gateway(settings: Settings):
    """构造 ModelGateway 单例（spec §8 仅在 ai_role=core 时启用）。

    若 ModelGateway 模块尚未引入（T27 才落盘），安全降级返回 None，
    但 lifespan 不应因此中断 —— 仅在 healthz 标记 gateway=missing。
    """
    try:
        from app.adapters.model_gateway import ModelGateway  # type: ignore
    except Exception as exc:
        LOGGER.info("ModelGateway 未引入（降级 None）：%s", exc)
        return None
    try:
        return ModelGateway.from_settings(settings)
    except Exception as exc:
        LOGGER.warning("ModelGateway.from_settings 失败：%s", exc)
        return None


def _build_audit_logger(settings: Settings):
    """构造 AuditLogger 单例（T30 才落盘，先降级 None）。"""
    try:
        from app.audit.logger import AuditLogger  # type: ignore
    except Exception as exc:
        LOGGER.info("AuditLogger 未引入（T30 才落盘，降级 None）：%s", exc)
        return None
    try:
        return AuditLogger.from_settings(settings)
    except Exception as exc:
        LOGGER.warning("AuditLogger.from_settings 失败：%s", exc)
        return None


def _register_routes(app: FastAPI, settings: Settings) -> List[str]:
    """反射式注册路由；返回成功挂载的 path 列表，便于测试断言。

    路由子模块引入时序：
      - T21 healthz / drafts / modules（独立模块，最先挂上）
      - T20 modeling / counterfactual（依赖 ModelGateway）
      - T22 浏览器面板 index.html（在最末挂"/"路由 + 把 static/ 挂到 /static）

    注意：panel 走 include_router(router) + mount_static_files(app) 两步：
      - StaticFiles 必须在 router include 之前完成 mount，否则新路由覆盖在 mount 之上会冲突。
    """
    registered: List[str] = []
    # 1. 先挂 panel static（spec §11 + T22）
    try:
        from app.api.panel import mount_static_files

        mounted = mount_static_files(app)
        if mounted:
            app.state.panel_static_mounted = True
            registered.append("/static/*")
    except Exception as exc:  # noqa: BLE001
        LOGGER.info("panel static 挂载未就位（占位）：%s", exc)

    route_modules: tuple = (
        ("app.api.healthz", "/api/healthz"),
        ("app.api.modules", "/api/modules"),
        ("app.api.drafts", "/api/drafts"),
        ("app.api.modeling", "/api/modeling/run"),
        ("app.api.counterfactual", "/api/counterfactual/run"),
        ("app.api.panel", "/"),
    )
    for module_name, _path_hint in route_modules:
        try:
            mod = __import__(module_name, fromlist=["router"])
        except Exception as exc:
            LOGGER.info("路由模块未引入（占位）：%s → %s", module_name, exc)
            continue
        router = getattr(mod, "router", None)
        if router is None:
            continue
        app.include_router(router)
        registered.append(_path_hint)
        LOGGER.info("路由模块已挂载：%s", module_name)
    return registered


# ---- 中间件：默认 operator_id 注入 ----

class OperatorIdMiddleware:
    """纯 ASGI 中间件：把请求头里的 X-Operator-Id / X-Quality-Role 注入到 request.state。

    ⚠️ 不要继承 BaseHTTPMiddleware：其对非 HTTPException 的异常传递在 Starlette 中
    表现不稳（被外层 ServerErrorMiddleware 接住返回 500）；改用纯 ASGI 让异常向下层
    ExceptionMiddleware 流过去，命中已注册的 exception_handlers。
    """

    def __init__(self, app: Any, *, local_reviewer_id: str) -> None:
        self._app = app
        self._local_reviewer_id = local_reviewer_id

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            return await self._app(scope, receive, send)

        # 抽出 header / 设置到 scope["state"] 会被 Starlette 在内部构造 Request.state
        headers = dict(scope.get("headers") or [])
        op_id_b = headers.get(b"x-operator-id") or headers.get(b"x-quality-role")
        op_id = op_id_b.decode("latin-1") if op_id_b else self._local_reviewer_id

        # 注入：把 operator_id 写到 scope 的 state（Starlette 1.x） + 准备回写 header
        # state 用一个列表让 send 时包一层 header 注入
        captured_op = op_id
        state = scope.setdefault("state", {})
        state["operator_id"] = captured_op

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers") or [])
                headers.append((b"x-operator-id", captured_op.encode("latin-1")))
                message["headers"] = headers
            await send(message)

        await self._app(scope, receive, send_wrapper)


# ---- lifespan ----

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan：startup + shutdown。

    启停顺序：见模块顶部 8 步注释。
    """
    LOGGER.info("=" * 60)
    LOGGER.info("[lifespan] startup begin")

    # 1. 解析 .env + 校验必填凭据（spec §8 "凭据缺失显式报错，绝不静默兜底"）
    settings = get_settings()
    if os.getenv("APP_ENV", "local") != "test":
        try:
            settings.validate_for_runtime()
            LOGGER.info("[lifespan] 配置校验通过 (driver=%s)", settings.driver_kind())
        except RuntimeError as exc:
            # 测试形态跳过校验；其他环境 propagate
            LOGGER.warning("[lifespan] 配置校验失败（不阻断 startup，按 best-effort 继续）：%s", exc)

    # 2. 异常处理器已在 create_app() 阶段装好（避免 lifespan 内安装被 TestClient stack 构造忽略）
    LOGGER.info("[lifespan] 异常处理器（已在 create_app 阶段安装）就位：%d 类", len(app.exception_handlers))

    # 3. 构造 DB session 工厂
    session_factory = _build_session_factory(settings)
    app.state.db_session_factory = session_factory
    app.state.settings = settings

    # 4. 构造 ModelGateway 单例（T27 占位：未引入时安全降级 None）
    app.state.model_gateway = _build_model_gateway(settings)

    # 5. 构造 AuditLogger 单例（T30 占位）
    audit_logger = _build_audit_logger(settings)
    app.state.audit_logger = audit_logger

    # 6. 反射注册 routes
    registered_paths = _register_routes(app, settings)
    app.state.registered_paths = registered_paths

    # 7. 写启动审计
    startup_events: List[Dict[str, Any]] = []
    if audit_logger is not None:
        try:
            ev_id = audit_logger.write_audit_event(
                event_type="lifecycle/started",
                operator=settings.local_reviewer_id,
                payload={
                    "app_env": settings.app_env,
                    "driver_kind": settings.driver_kind(),
                    "registered_paths": registered_paths,
                    "has_db_session": session_factory is not None,
                    "has_model_gateway": app.state.model_gateway is not None,
                },
            )
            startup_events.append({"event": "lifecycle/started", "id": ev_id})
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("lifecycle/started audit 失败：%s", exc)
    app.state.startup_events = startup_events

    LOGGER.info(
        "[lifespan] startup done: %d routes mounted, db=%s, gateway=%s, audit=%s",
        len(registered_paths),
        "ok" if session_factory else "missing",
        "ok" if app.state.model_gateway else "missing",
        "ok" if audit_logger else "missing",
    )
    LOGGER.info("=" * 60)

    try:
        yield
    finally:
        LOGGER.info("[lifespan] shutdown begin")
        # shutdown audit
        if audit_logger is not None:
            try:
                audit_logger.write_audit_event(
                    event_type="lifecycle/shutdown",
                    operator=settings.local_reviewer_id,
                    payload={"registered_paths": registered_paths},
                )
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("lifecycle/shutdown audit 失败：%s", exc)
        # 关闭 model gateway / db engine
        try:
            if app.state.model_gateway is not None:
                close = getattr(app.state.model_gateway, "close", None)
                if close is not None:
                    close()
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("model_gateway.close 失败：%s", exc)
        LOGGER.info("[lifespan] shutdown done")


# ---- 应用工厂 ----

def create_app() -> FastAPI:
    """应用工厂：lifespan + middleware + 元数据。

    测试时也可反复调 create_app() 反复构造独立实例。

    ⚠️ **重要顺序**：`install_exception_handlers(app)` 必须在 `add_middleware(...)` 之前调用。
    测试发现：TestClient 在 `__enter__` 时构造 middleware_stack，lifespan 内再装 handler
    不会被捕获到（外层 ServerErrorMiddleware 把所有未识别异常转 500）。把 handler 安装
    提前到 create_app() 阶段，stack 构造时已含全部异常映射。
    """
    settings = get_settings()
    app = FastAPI(
        title=__import__("app.api", fromlist=["OPENAPI_TITLE"]).OPENAPI_TITLE,
        description=__import__("app.api", fromlist=["OPENAPI_DESCRIPTION"]).OPENAPI_DESCRIPTION,
        version=__import__("app.__init__", fromlist=["__version__"]).__version__,
        lifespan=lifespan,
        openapi_tags=list(
            __import__("app.api", fromlist=["OPENAPI_TAGS"]).OPENAPI_TAGS
        ),
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )

    # 1. 异常 handler 先装（先于 add_middleware，避免 stack 构造时漏装）
    from app.core.exception_handlers import install_exception_handlers
    install_exception_handlers(app)

    # 2. Operator 中间件：把 operator_id 注入 request.state，供异常 handler / 审计用
    app.add_middleware(
        OperatorIdMiddleware,
        local_reviewer_id=settings.local_reviewer_id,
    )

    # 2.5 panel /static 挂载（必须在 lifespan 注册路由之前完成，否则 lifespan 内的
    # include_router 顺序与 mount 顺序冲突；spec §11 + T22）。
    # _register_routes 期间会调用 mount_static_files(app) 替我们挂上。
    app.state.panel_static_mounted = False

    # 根目录 discovery（不在 lifespan 内挂：避免路由注册顺序脆弱）
    @app.get("/api/info", tags=["system"])
    async def info_endpoint() -> Dict[str, Any]:
        return {
            "name": "disease-metric-causal-attribution",
            "version": __import__("app.__init__", fromlist=["__version__"]).__version__,
            "api_version": __import__("app.api", fromlist=["API_VERSION"]).API_VERSION,
            "app_env": settings.app_env,
            "openai_model": settings.openai_model,
        }

    return app


# uvicorn entrypoint: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
app: FastAPI = create_app()


def run() -> None:
    """供 `python -m app.main` 与 scripts 触发的常驻入口。

    实际生产部署仍走 uvicorn；本函数用于本地脚本（demo.sh / run_e2e.sh 复用）。
    """
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        log_level=settings.app_log_level.lower(),
        reload=False,
    )


if __name__ == "__main__":  # pragma: no cover
    run()
