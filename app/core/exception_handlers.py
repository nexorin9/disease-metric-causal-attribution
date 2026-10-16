"""全局异常处理层（spec.md §6 / §11 失败降级契约）。

把核心层 / 适配层抛出的业务异常统一映射到 FastAPI handler 上，返回
标准化 JSON envelope + 写审计日志（spec §6 关键动作："任何外发或写操作前
若缺少审计字段，必须先写审计再推进业务"）。

JSON envelope 形状（统一 schema_version='error_envelope/1.0'）：
  {
    "schema_version": "error_envelope/1.0",
    "ok": false,
    "error_type": "<exception class name>",
    "error_code": "<stable 机器可读码>",
    "message": "<中文简短>",   # 给业务方读
    "details": {...},           # 异常附加字段（可选）
    "audit_event_id": "<uuid>",
    "ts": "<ISO8601 UTC>"
  }

不静默改数据 → 所有 handler 一律：
  1. 构造 envelope
  2. 调 audit_logger.write_audit_event 记录（best-effort，audit 失败也不抛）
  3. 返回 JSONResponse（4xx/5xx 视业务）
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

LOGGER = logging.getLogger(__name__)

ERROR_ENVELOPE_SCHEMA_VERSION = "error_envelope/1.0"


class RefutationUnstableError(RuntimeError):
    """反驳 CI 超宽 / p>0.5 / 极性反转 → 标 refutation_unstable 后抛出。

    业务处理：草稿可生成，但禁止触发写回，状态停在 pending_review。
    """


class ModelingFailedError(RuntimeError):
    """建模主路径中断（DAG 缺边 / 节点不识别 / 拒绝收敛等不可恢复错误）。

    业务处理：草稿不生成；面板显示 error 重试入口。
    """

    def __init__(self, message: str, *, stage: Optional[str] = None) -> None:
        super().__init__(message)
        self.stage = stage


class DipcasePostFailedError(RuntimeError):
    """单病种上报端 HTTP POST 失败（即使经过 retry 也未成功）。

    业务处理：草稿退回 pending_review（保持人工落地 + 审计）；看板 toast 红色提示。
    """

    def __init__(self, message: str, *, attempts: int = 0) -> None:
        super().__init__(message)
        self.attempts = attempts


class QualityRoleForbiddenError(PermissionError):
    """X-Quality-Role 不在白名单（典型：reviewer 角色被错配给非业务用户）。"""

    def __init__(self, message: str, *, role: Optional[str] = None) -> None:
        super().__init__(message)
        self.role = role


# 错误码：保持稳定、机器可读；与 docs/api-errors.md 一致
ERROR_CODE_MAP: Dict[str, str] = {
    "EstimandNotIdentifiedError": "estimand_not_identified",
    "InsufficientSampleError": "insufficient_sample",
    "RefutationUnstableError": "refutation_unstable",
    "ModelingFailedError": "modeling_failed",
    "DipcasePostFailedError": "dipcase_post_failed",
    "QualityRoleForbiddenError": "quality_role_forbidden",
    "RequestValidationError": "request_validation_error",
    "HTTPException": "http_exception",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _audit_event_id() -> str:
    return f"ev_{uuid.uuid4().hex[:16]}"


def _build_envelope(
    exc: BaseException,
    *,
    error_code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    http_status: int,
) -> Dict[str, Any]:
    """构造统一的错误 envelope。"""
    return {
        "schema_version": ERROR_ENVELOPE_SCHEMA_VERSION,
        "ok": False,
        "error_type": type(exc).__name__,
        "error_code": error_code,
        "message": message,
        "details": details or {},
        "audit_event_id": _audit_event_id(),
        "ts": _now_iso(),
        "http_status": http_status,
    }


def _extract_details(exc: BaseException) -> Dict[str, Any]:
    """从异常对象抽出安全可序列化的字段。"""
    out: Dict[str, Any] = {}
    for attr in ("stage", "attempts", "role", "sample_size", "min_required"):
        if hasattr(exc, attr):
            v = getattr(exc, attr)
            if isinstance(v, (str, int, float, bool, type(None))):
                out[attr] = v
    return out


def _best_effort_audit(
    audit_logger: Optional[Any],
    *,
    event_type: str,
    operator: str,
    payload: Dict[str, Any],
) -> None:
    """Best-effort 写审计；audit 失败必须静默（不能反向抛业务异常）。

    audit_logger 可以是 None 或 None-like（未注入 lifespan 时）。
    """
    if audit_logger is None:
        return
    try:
        audit_logger.write_audit_event(
            event_type=event_type,
            operator=operator,
            payload=payload,
        )
    except Exception as audit_exc:  # pragma: no cover - best-effort
        LOGGER.warning(
            "audit 写入失败（best-effort，不反向抛业务异常）：%s",
            audit_exc,
        )


# ---- FastAPI handler 工厂 ----

def _generic_handler_factory(
    *,
    http_status: int,
    error_type_name: str,
    default_message_zh: str,
    audit_event_type: str,
    error_code_override: Optional[str] = None,
    details_builder: Optional[Callable[[BaseException], Dict[str, Any]]] = None,
) -> Callable[..., Awaitable[JSONResponse]]:
    """为一种异常类生成 FastAPI handler 闭包。

    闭包内捕获 audit_logger；为 None 时不写审计。
    """

    async def _handler(request: Request, exc: BaseException) -> JSONResponse:
        error_code = error_code_override or ERROR_CODE_MAP.get(error_type_name, error_type_name)
        details = details_builder(exc) if details_builder else _extract_details(exc)
        envelope = _build_envelope(
            exc,
            error_code=error_code,
            message=str(exc) or default_message_zh,
            details=details,
            http_status=http_status,
        )
        # 审计 best-effort：拿 lifespan 注入的 audit_logger
        audit_logger = getattr(request.app.state, "audit_logger", None)
        _best_effort_audit(
            audit_logger,
            event_type=audit_event_type,
            operator=str(getattr(request.state, "operator_id", "anonymous")),
            payload={
                "path": str(request.url.path),
                "method": request.method,
                "error_type": envelope["error_type"],
                "error_code": envelope["error_code"],
                "http_status": http_status,
                "audit_event_id": envelope["audit_event_id"],
                "details": details,
            },
        )
        return JSONResponse(
            content=envelope,
            status_code=http_status,
            headers={"X-Audit-Event-Id": envelope["audit_event_id"]},
        )

    return _handler


def install_exception_handlers(app: FastAPI) -> None:
    """把异常 handler 装到 FastAPI app 上。

    异常类型：
      - EstimandNotIdentifiedError → 422 unprocessable_entity
      - InsufficientSampleError   → 422 unprocessable_entity
      - RefutationUnstableError   → 200 OK（业务降级而非 HTTP 失败；草稿可生成但禁止写回）
      - ModelingFailedError       → 500 internal_server_error
      - DipcasePostFailedError    → 502 bad_gateway
      - QualityRoleForbiddenError → 403 forbidden
      - RequestValidationError    → 422 unprocessable_entity（FastAPI 内置）
      - HTTPException             → 透传原 status（Starlette 内置）
      - Exception                 → 兜底 500；记录到 LOGGER
    """
    # 局部延迟导入避免循环（pipeline.py 已 import app.config 等）
    from app.core.pipeline import EstimandNotIdentifiedError, InsufficientSampleError

    app.add_exception_handler(
        EstimandNotIdentifiedError,
        _generic_handler_factory(
            http_status=422,
            error_type_name="EstimandNotIdentifiedError",
            default_message_zh="因果 estimand 不可识别（图与字段不一致）",
            audit_event_type="modeling/unidentified_estimand",
        ),
    )
    app.add_exception_handler(
        InsufficientSampleError,
        _generic_handler_factory(
            http_status=422,
            error_type_name="InsufficientSampleError",
            default_message_zh="样本量不足，无法产出可信结果",
            audit_event_type="modeling/insufficient_sample",
        ),
    )
    app.add_exception_handler(
        RefutationUnstableError,
        _generic_handler_factory(
            http_status=200,
            error_type_name="RefutationUnstableError",
            default_message_zh="反驳结果不稳定，草稿可生成但禁止自动上报",
            audit_event_type="refutation/degraded",
        ),
    )
    app.add_exception_handler(
        ModelingFailedError,
        _generic_handler_factory(
            http_status=500,
            error_type_name="ModelingFailedError",
            default_message_zh="建模主路径中断（请重试或检查输入数据）",
            audit_event_type="modeling/failed",
        ),
    )
    app.add_exception_handler(
        DipcasePostFailedError,
        _generic_handler_factory(
            http_status=502,
            error_type_name="DipcasePostFailedError",
            default_message_zh="单病种上报接口不可达，草稿已退回待人工处理",
            audit_event_type="dipcase/post_failed",
        ),
    )
    app.add_exception_handler(
        QualityRoleForbiddenError,
        _generic_handler_factory(
            http_status=403,
            error_type_name="QualityRoleForbiddenError",
            default_message_zh="角色鉴权未通过",
            audit_event_type="authz/role_forbidden",
        ),
    )
    app.add_exception_handler(
        RequestValidationError,
        _generic_handler_factory(
            http_status=422,
            error_type_name="RequestValidationError",
            default_message_zh="请求参数校验失败",
            audit_event_type="api/validation_error",
        ),
    )

    # StarletteHTTPException 透传 status：详查 exc.status_code
    async def _http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        envelope = _build_envelope(
            exc,
            error_code=ERROR_CODE_MAP.get("HTTPException", "http_exception"),
            message=str(exc.detail) if getattr(exc, "detail", None) else "HTTP 错误",
            details={"status_code": getattr(exc, "status_code", 500)},
            http_status=getattr(exc, "status_code", 500) or 500,
        )
        audit_logger = getattr(request.app.state, "audit_logger", None)
        _best_effort_audit(
            audit_logger,
            event_type="api/http_exception",
            operator=str(getattr(request.state, "operator_id", "anonymous")),
            payload={
                "path": str(request.url.path),
                "method": request.method,
                "error_code": envelope["error_code"],
                "http_status": envelope["http_status"],
                "audit_event_id": envelope["audit_event_id"],
            },
        )
        return JSONResponse(
            content=envelope,
            status_code=envelope["http_status"],
            headers={"X-Audit-Event-Id": envelope["audit_event_id"]},
        )

    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)

    # 兜底：捕获一切未注册异常
    async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        LOGGER.exception("未注册异常被兜底拦截：%s", exc)
        envelope = _build_envelope(
            exc,
            error_code="unhandled_exception",
            message="服务器内部错误",
            details={"exc_repr": repr(exc)[:512]},
            http_status=500,
        )
        audit_logger = getattr(request.app.state, "audit_logger", None)
        _best_effort_audit(
            audit_logger,
            event_type="api/unhandled_exception",
            operator=str(getattr(request.state, "operator_id", "anonymous")),
            payload={
                "path": str(request.url.path),
                "method": request.method,
                "error_type": envelope["error_type"],
                "error_code": envelope["error_code"],
                "http_status": 500,
                "audit_event_id": envelope["audit_event_id"],
                "exc_repr": repr(exc)[:512],
            },
        )
        return JSONResponse(
            content=envelope,
            status_code=500,
            headers={"X-Audit-Event-Id": envelope["audit_event_id"]},
        )

    app.add_exception_handler(Exception, _unhandled_exception_handler)


# ---- 模块级便捷工具 ----

def serialize_error_envelope(envelope: Dict[str, Any]) -> str:
    """供测试 / 日志序列化 envelope。"""
    return json.dumps(envelope, ensure_ascii=False, sort_keys=True, default=str)


__all__ = [
    "DipcasePostFailedError",
    "ERROR_CODE_MAP",
    "ERROR_ENVELOPE_SCHEMA_VERSION",
    "ModelingFailedError",
    "QualityRoleForbiddenError",
    "RefutationUnstableError",
    "install_exception_handlers",
    "serialize_error_envelope",
]
