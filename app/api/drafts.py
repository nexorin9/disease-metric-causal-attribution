"""草稿评审 API（spec.md §3.2 + §11 收尾交付）。

| Endpoint                          | 用途                                |
|-----------------------------------|-------------------------------------|
| GET  /api/drafts/{draft_id}        | 单条草稿详情 + transitions 列表    |
| GET  /api/drafts                  | 列表（按 status / disease_code 过滤）|
| POST /api/drafts/{draft_id}/review| review action=confirm|reject → 状态机 + 上报 |

设计原则（spec §6）：
- **人工确认**：所有 review 操作必须有 X-Quality-Role=med_committee_reviewer。
- **状态机只接受合法转换**：pending_review → reviewed/rejected → archived；跳级拒绝。
- **失败回退到 pending_review**：若 confirm 时上报 POST 失败，状态机退回 + draft.notes 写入原因。
- **audit**：所有路径经 _best_effort_audit 落 audit_logs（best-effort，audit 不可用不影响业务）。

依赖：
- app.drafts.models.DraftRepository（in-memory store，T25 升级 SQLAlchemy）
- app.drafts.state_machine.transition（合法转换）
- app.adapters.dipcase_reporter（DipcaseReporter，T26 落盘；本端点留可注入钩子）
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, Query, Request, status
from fastapi.responses import JSONResponse

from app.core.exception_handlers import QualityRoleForbiddenError
from app.drafts import (
    DRAFT_SCHEMA_VERSION,
    DraftStatus,
    InvalidStateTransitionError,
    get_default_repository,
    transition,
)
from app.drafts.models import Draft, DraftRepository, StateTransitionRecord

LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/api/drafts", tags=["drafts"])

# 仅医务科评审可触发 review（spec §11 第 6 步；T25 增加角色 403 严格校验）
_REVIEWER_ROLE = "med_committee_reviewer"
_ALLOWED_VIEW_ROLES = frozenset({"quality_officer", "med_committee_reviewer"})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _resolve_repository(request: Request) -> DraftRepository:
    """优先用 app.state.drafts_repository（lifespan / 测试可注入）；否则取默认单例。"""
    r = getattr(request.app.state, "drafts_repository", None)
    if r is not None:
        return r
    return get_default_repository()


def _check_view_role(role_header: Optional[str]) -> str:
    """读路径（GET 单条 + 列表）的角色校验：允许质控 + 医务科评审。"""
    if not role_header:
        raise QualityRoleForbiddenError(
            "X-Quality-Role header 缺失；查看草稿必须由质控员或评审角色触发",
            role="<missing>",
        )
    if role_header not in _ALLOWED_VIEW_ROLES:
        raise QualityRoleForbiddenError(
            f"X-Quality-Role={role_header!r} 不在读视图允许集合",
            role=role_header,
        )
    return role_header


def _check_reviewer_role(role_header: Optional[str]) -> str:
    """写路径（POST review）角色校验：仅医务科评审。"""
    if not role_header:
        raise QualityRoleForbiddenError(
            "X-Quality-Role header 缺失；评审操作必须由医务科评审角色触发",
            role="<missing>",
        )
    if role_header != _REVIEWER_ROLE:
        raise QualityRoleForbiddenError(
            f"X-Quality-Role={role_header!r} ≠ {_REVIEWER_ROLE!r}；无权限执行评审",
            role=role_header,
        )
    return role_header


def _draft_to_payload(d: Draft) -> Dict[str, Any]:
    return d.to_dict()


def _transitions_to_payload(items: List[StateTransitionRecord]) -> List[Dict[str, Any]]:
    return [t.to_dict() for t in items]


def _best_effort_audit(
    request: Request,
    *,
    event_type: str,
    operator: str,
    operator_role: str,
    payload: Dict[str, Any],
) -> None:
    """Best-effort 写审计（与 app.core.exception_handlers 风格一致）。"""
    audit_logger = getattr(request.app.state, "audit_logger", None)
    if audit_logger is None:
        return
    try:
        audit_logger.write_audit_event(
            event_type=event_type,
            operator=operator,
            payload={
                "operator_role": operator_role,
                **payload,
            },
        )
    except Exception as audit_exc:  # pragma: no cover - best-effort
        LOGGER.warning(
            "drafts API 审计写入失败（best-effort，不反向抛业务异常）：%s",
            audit_exc,
        )


# ---------------------------------------------------------------------------
# 上报钩子（T26 落盘；本端点先打桩）
# ---------------------------------------------------------------------------


def _post_to_dipcase(draft: Draft, *, action: str) -> Dict[str, Any]:
    """上报到单病种上报系统；失败返回 dict 含 ok=False（业务层捕获并回退）。

    T21 阶段：尚未引入 DipcaseReporter；走安全降级，post_attribution_draft 不可用时
    返回 ok=False 不阻断 UI；T26 把这段替换为真实 HTTP POST + retry 3 + 失败回退。
    """
    try:
        from app.adapters.dipcase_reporter import DipcaseReporter  # type: ignore
    except Exception as exc:  # noqa: BLE001 — T26 之前模块未引入
        LOGGER.info("DipcaseReporter 未引入（T26 之前）：%s → 降级 stub", exc)
        return {
            "ok": False,
            "degraded": True,
            "reason": "dipcase_reporter_not_initialized",
            "status_code": None,
        }

    try:
        reporter = DipcaseReporter.from_default_settings()  # type: ignore[attr-defined]
        result = reporter.post_attribution_draft(draft.to_dict())  # type: ignore[attr-defined]
        # 真实 DipcaseReporter 返回 PostResult（dataclass）；统一收敛为 dict
        if hasattr(result, "to_dict") and callable(getattr(result, "to_dict")):
            return result.to_dict()
        if isinstance(result, dict):
            return dict(result)
        if result is None:
            return {"ok": True}
        return {"ok": False, "degraded": True, "reason": "unexpected_post_result_type"}
    except Exception as exc:  # noqa: BLE001 — T26 实际 retry 3 后仍未成功
        LOGGER.warning("上报草稿 %s 异常: %s", draft.draft_id, exc)
        return {
            "ok": False,
            "degraded": True,
            "reason": f"exception:{type(exc).__name__}",
            "message": str(exc),
            "status_code": None,
        }


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------


@router.get("/{draft_id}")
async def get_draft(
    draft_id: str,
    request: Request,
    x_quality_role: Optional[str] = Header(default=None, alias="X-Quality-Role"),
) -> JSONResponse:
    """GET /api/drafts/{draft_id} → 单条草稿 + transitions 列表。"""
    role = _check_view_role(x_quality_role)
    r = _resolve_repository(request)
    d = r.get(draft_id)
    if d is None:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "schema_version": DRAFT_SCHEMA_VERSION,
                "ok": False,
                "error_code": "draft_not_found",
                "message": f"draft_id={draft_id} 不存在",
                "draft_id": draft_id,
            },
        )

    transitions = _transitions_to_payload(r.transitions(draft_id))
    body = {
        "schema_version": DRAFT_SCHEMA_VERSION,
        "ok": True,
        "draft": _draft_to_payload(d),
        "transitions": transitions,
    }

    _best_effort_audit(
        request,
        event_type="draft/get",
        operator=str(getattr(d, "reviewer_id", "") or "viewer"),
        operator_role=role,
        payload={"draft_id": draft_id, "n_transitions": len(transitions)},
    )

    return JSONResponse(status_code=200, content=body, headers={"X-Quality-Role": role})


@router.get("")
async def list_drafts(
    request: Request,
    status_filter: Optional[str] = Query(default=None, alias="status"),
    disease_code: Optional[str] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=200),
    x_quality_role: Optional[str] = Header(default=None, alias="X-Quality-Role"),
) -> JSONResponse:
    """GET /api/drafts → 列表，按 status / disease_code 过滤。

    status 取值必须属于 DraftStatus；非法值返回 422。
    """
    role = _check_view_role(x_quality_role)

    if status_filter is not None and status_filter not in {s.value for s in DraftStatus}:
        return JSONResponse(
            status_code=422,
            content={
                "schema_version": DRAFT_SCHEMA_VERSION,
                "ok": False,
                "error_code": "invalid_status",
                "message": f"status={status_filter!r} 不在合法集合",
                "allowed": [s.value for s in DraftStatus],
            },
        )

    r = _resolve_repository(request)
    items = r.list(status=status_filter, disease_code=disease_code, limit=limit)
    return JSONResponse(
        status_code=200,
        content={
            "schema_version": DRAFT_SCHEMA_VERSION,
            "ok": True,
            "n_total": len(items),
            "status_filter": status_filter,
            "disease_code": disease_code,
            "drafts": [_draft_to_payload(d) for d in items],
        },
        headers={"X-Quality-Role": role},
    )


@router.post("/{draft_id}/review")
async def review_draft(
    draft_id: str,
    payload: Dict[str, Any],
    request: Request,
    x_quality_role: Optional[str] = Header(default=None, alias="X-Quality-Role"),
    x_operator_id: Optional[str] = Header(default=None, alias="X-Operator-Id"),
) -> JSONResponse:
    """POST /api/drafts/{draft_id}/review → 评审 action=confirm|reject。

    Body
    ----
    {
      "action":  "confirm" | "reject",
      "note":    "可空；confirm 时评审意见 / reject 时退回原因",
      "post_to_dipcase": bool (默认 true — confirm 时触发上报)
    }

    Returns
    -------
    JSONResponse
        confirm: 200 + {draft, transitions, post_result}
        reject:  200 + {draft, transitions}   # 不上报
        跳级: 422 + InvalidStateTransitionError envelope
        角色非 med_committee_reviewer: 403
    """
    role = _check_reviewer_role(x_quality_role)

    action = str(payload.get("action") or "").strip().lower()
    if action not in ("confirm", "reject"):
        return JSONResponse(
            status_code=422,
            content={
                "schema_version": DRAFT_SCHEMA_VERSION,
                "ok": False,
                "error_code": "invalid_action",
                "message": f"action={action!r} 必须为 confirm 或 reject",
            },
        )

    note = str(payload.get("note") or "")
    post_to_dipcase = bool(payload.get("post_to_dipcase", True))

    r = _resolve_repository(request)
    d = r.get(draft_id)
    if d is None:
        return JSONResponse(
            status_code=404,
            content={
                "schema_version": DRAFT_SCHEMA_VERSION,
                "ok": False,
                "error_code": "draft_not_found",
                "message": f"draft_id={draft_id} 不存在",
                "draft_id": draft_id,
            },
        )

    operator_id = x_operator_id or d.reviewer_id or "reviewer-anonymous"

    # ---- 1. 状态机转换 ----
    target_state = (
        DraftStatus.REVIEWED.value if action == "confirm" else DraftStatus.REJECTED.value
    )
    try:
        record = transition(
            r,
            draft_id=draft_id,
            to_state=target_state,
            operator_id=operator_id,
            operator_role=role,
            note=note,
        )
    except InvalidStateTransitionError as exc:
        LOGGER.info(
            "drafts review %s: 非法转换 %s → %s",
            draft_id,
            exc.from_state,
            exc.to_state,
        )
        return JSONResponse(
            status_code=422,
            content={
                "schema_version": DRAFT_SCHEMA_VERSION,
                "ok": False,
                "error_code": "invalid_state_transition",
                "message": str(exc),
                "from_state": exc.from_state,
                "to_state": exc.to_state,
                "allowed_transitions": _allowed_transitions_display(),
            },
        )
    except KeyError as exc:
        return JSONResponse(
            status_code=404,
            content={
                "schema_version": DRAFT_SCHEMA_VERSION,
                "ok": False,
                "error_code": "draft_not_found",
                "message": str(exc),
                "draft_id": draft_id,
            },
        )

    # ---- 2. confirm 触发上报；失败 → 回退到 pending_review ----
    post_result: Optional[Dict[str, Any]] = None
    rolled_back = False
    if action == "confirm" and post_to_dipcase:
        started = time.monotonic()
        post_result = _post_to_dipcase(d, action=action)
        post_result = {
            **post_result,
            "elapsed_ms": int((time.monotonic() - started) * 1000),
        }

        if not post_result.get("ok"):
            # 上报失败 → 回退到 pending_review（spec §11 第 5 步）
            LOGGER.warning(
                "上报草稿 %s 失败，回退到 pending_review：%s",
                draft_id,
                post_result.get("reason") or post_result.get("message"),
            )
            try:
                rollback = transition(
                    r,
                    draft_id=draft_id,
                    to_state=DraftStatus.PENDING_REVIEW.value,
                    operator_id="system:auto-rollback",
                    operator_role="system",
                    note=f"上报失败自动回退：{post_result.get('reason') or post_result.get('message')}",
                )
                d = r.get(draft_id)  # 刷新
                post_result["rolled_back"] = True
                post_result["rollback_transition_id"] = rollback.transition_id
                rolled_back = True
            except Exception as rb_exc:  # noqa: BLE001
                LOGGER.warning("回退到 pending_review 失败：%s", rb_exc)
                post_result["rolled_back"] = False
                post_result["rollback_error"] = f"{type(rb_exc).__name__}: {rb_exc}"

    # ---- 3. 审计 ----
    _best_effort_audit(
        request,
        event_type="draft/post",
        operator=operator_id,
        operator_role=role,
        payload={
            "draft_id": draft_id,
            "action": action,
            "from_state": record.from_state,
            "to_state": record.to_state,
            "transition_id": record.transition_id,
            "post_result": post_result,
            "rolled_back": rolled_back,
        },
    )

    body = {
        "schema_version": DRAFT_SCHEMA_VERSION,
        "ok": True,
        "action": action,
        "draft": _draft_to_payload(d),
        "transition": record.to_dict(),
        "post_result": post_result,
        "rolled_back": rolled_back,
    }
    return JSONResponse(status_code=200, content=body, headers={"X-Quality-Role": role})


def _allowed_transitions_display() -> List[str]:
    """运行时构造合法转换键列表（人类可读）。"""
    from app.drafts.state_machine import ALLOWED_TRANSITIONS as T

    return [f"{k.split('->')[0]} → {k.split('->')[1]}" for k in T]


__all__ = ["router"]