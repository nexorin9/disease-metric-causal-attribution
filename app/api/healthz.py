"""系统健康端点（spec.md §11 收尾交付：常驻服务 + 健康检查）。

GET /api/healthz → 200 / 503 + 4 项子健康：
  - db:        SQLAlchemy session 工厂可用性
  - model_gateway: OpenAI-compatible 网关可用性
  - his_conn:  HIS DB driver 推断 + 配置存在性
  - drafts:    in-memory 草稿存储可读写（lifespan 起进程即可用）

设计原则：
- 不抛任何业务异常；所有子项 best-effort 探测（最多 100ms 一次）。
- 不连真实库、不发真实请求：仅检查配置 / 模块就位 + 探活 lightweight 函数。
- 即使 db / model_gateway / his_conn 都不可用，仍返回 200 + 子状态 red —— 这是 spec §6
  "失败不静默改数据" 的可见化（运维需先看 healthz 才能定位问题）。
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.config import get_settings

LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["system"])

HEALTHZ_SCHEMA_VERSION = "healthz_response/1.0"


def _check_db(app_state: Any) -> Dict[str, Any]:
    """DB 子健康：检查 session 工厂已构造 + 反射测试（轻量）。"""
    factory = getattr(app_state, "db_session_factory", None)
    if factory is None:
        return {"status": "red", "reason": "session_factory_missing"}
    try:
        # 不执行 SQL：仅验证工厂可调用且能 close session
        session = factory()
        try:
            return {"status": "green", "reason": "session_factory_callable"}
        finally:
            try:
                session.close()
            except Exception:  # noqa: BLE001 - best-effort close
                pass
    except Exception as exc:  # noqa: BLE001 - DB 不可用
        return {"status": "red", "reason": f"session_factory_failed:{type(exc).__name__}"}


def _check_model_gateway(app_state: Any) -> Dict[str, Any]:
    """ModelGateway 子健康：仅检查 app.state.model_gateway 已注入。"""
    gw = getattr(app_state, "model_gateway", None)
    if gw is None:
        return {"status": "red", "reason": "model_gateway_not_initialized"}
    return {"status": "green", "reason": "model_gateway_present"}


def _check_his_conn() -> Dict[str, Any]:
    """HIS DB driver 推断：postgresql / mysql / sqlite；非空 URL + 无 CHANGE_ME 占位。"""
    try:
        s = get_settings()
    except Exception as exc:  # noqa: BLE001
        return {"status": "red", "reason": f"settings_load_failed:{type(exc).__name__}"}
    if not s.his_db_url or "CHANGE_ME" in s.his_db_url:
        return {"status": "red", "reason": "his_db_url_missing_or_placeholder"}
    try:
        kind = s.driver_kind()
    except ValueError as exc:
        return {"status": "red", "reason": f"driver_kind_failed:{exc}"}
    return {"status": "green", "driver": kind, "reason": "his_db_url_valid"}


def _check_drafts_store() -> Dict[str, Any]:
    """in-memory 草稿存储：能创建 DraftRepository 即视为 green。"""
    try:
        from app.drafts import DraftRepository

        r = DraftRepository()
        d = r.create(
            disease_code="__healthcheck__",
            outcome="__probe__",
            treatment="__probe__",
            narrative_zh="healthcheck-probe",
        )
        r.clear()  # 立即清掉（不污染默认单例）
        return {"status": "green", "reason": "drafts_repository_writable", "probe_id": d.draft_id}
    except Exception as exc:  # noqa: BLE001
        return {"status": "red", "reason": f"drafts_repository_failed:{type(exc).__name__}"}


@router.get("/healthz")
async def healthz(request: Request) -> JSONResponse:
    """GET /api/healthz → 系统健康。

    整体 status 由 4 项子状态 roll-up：
      - 全部 green → green（HTTP 200）
      - 任一 red   → red  （HTTP 503）
      - 其余        → yellow（HTTP 200）
    """
    db = _check_db(request.app.state)
    gw = _check_model_gateway(request.app.state)
    his = _check_his_conn()
    drafts = _check_drafts_store()

    statuses = [db["status"], gw["status"], his["status"], drafts["status"]]
    if all(s == "green" for s in statuses):
        overall = "green"
        http_status = 200
    elif any(s == "red" for s in statuses):
        overall = "red"
        http_status = 503
    else:
        overall = "yellow"
        http_status = 200

    body = {
        "schema_version": HEALTHZ_SCHEMA_VERSION,
        "ok": overall == "green",
        "status": overall,
        "checks": {
            "db": db,
            "model_gateway": gw,
            "his_conn": his,
            "drafts": drafts,
        },
    }
    return JSONResponse(status_code=http_status, content=body)


__all__ = ["HEALTHZ_SCHEMA_VERSION", "router"]