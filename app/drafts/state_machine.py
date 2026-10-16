"""草稿状态机（spec.md §3.2 + §11 收尾 · Task 25 角色 403 强校验）。

合法转换：
    pending_review → reviewed       （reviewer 必须 = med_committee_reviewer）
    pending_review → rejected       （reviewer 必须 = med_committee_reviewer）
    reviewed       → archived       （reviewer 必须 = med_committee_reviewer 或 quality_officer）
    rejected       → archived       （reviewer 必须 = med_committee_reviewer 或 quality_officer）
    reviewed       → pending_review  （**系统自动回退**；仅 operator_role='system' 允许）

非法转换一律拒绝（InvalidStateTransitionError）。
直接修改 Draft.status 视为非法（构造期锁定不变）。

角色策略（ROLE_POLICY）：
    transition  → 合法角色集合
    "*"           通配（开发/测试期可临时绕过，但生产路径仍需显式 med_committee_reviewer）

角色失配 → RoleMismatchError（PermissionError 子类；FastAPI handler 映射 403）。

⚠️ 严禁在状态机内直接调 audit_logger —— API 层负责落审计。
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, Optional, Set

from app.drafts.models import (
    Draft,
    DraftRepository,
    DraftStatus,
    StateTransitionRecord,
)


# 合法的状态转换表
ALLOWED_TRANSITIONS: Dict[str, str] = {
    f"{DraftStatus.PENDING_REVIEW.value}->{DraftStatus.REVIEWED.value}": DraftStatus.REVIEWED.value,
    f"{DraftStatus.PENDING_REVIEW.value}->{DraftStatus.REJECTED.value}": DraftStatus.REJECTED.value,
    f"{DraftStatus.REVIEWED.value}->{DraftStatus.ARCHIVED.value}": DraftStatus.ARCHIVED.value,
    f"{DraftStatus.REJECTED.value}->{DraftStatus.ARCHIVED.value}": DraftStatus.ARCHIVED.value,
    # 系统自动回退：confirm 时上报失败 → 回退 pending_review（spec §11 第 5 步）
    f"{DraftStatus.REVIEWED.value}->{DraftStatus.PENDING_REVIEW.value}": DraftStatus.PENDING_REVIEW.value,
}


# spec.md §6 + §11：review 操作必须由医务科评审（med_committee_reviewer）触发；
# 归档操作可由质控员（quality_officer）或医务科评审触发；rollback 仅系统。
REVIEWER_ROLE = "med_committee_reviewer"
QUALITY_ROLE = "quality_officer"
SYSTEM_ROLE = "system"


ROLE_POLICY: Dict[str, Set[str]] = {
    # pending_review → reviewed
    f"{DraftStatus.PENDING_REVIEW.value}->{DraftStatus.REVIEWED.value}": frozenset({REVIEWER_ROLE}),
    # pending_review → rejected
    f"{DraftStatus.PENDING_REVIEW.value}->{DraftStatus.REJECTED.value}": frozenset({REVIEWER_ROLE}),
    # reviewed → archived（归档可由质控员或评审触发）
    f"{DraftStatus.REVIEWED.value}->{DraftStatus.ARCHIVED.value}": frozenset(
        {REVIEWER_ROLE, QUALITY_ROLE}
    ),
    # rejected → archived
    f"{DraftStatus.REJECTED.value}->{DraftStatus.ARCHIVED.value}": frozenset(
        {REVIEWER_ROLE, QUALITY_ROLE}
    ),
    # 系统回退（仅 system 角色；spec §11 第 5 步）
    f"{DraftStatus.REVIEWED.value}->{DraftStatus.PENDING_REVIEW.value}": frozenset({SYSTEM_ROLE}),
}


class InvalidStateTransitionError(ValueError):
    """非法状态转换（跳级 / 同状态 / 不可达目标）。"""

    def __init__(self, message: str, *, from_state: str, to_state: str) -> None:
        super().__init__(message)
        self.from_state = from_state
        self.to_state = to_state


class RoleMismatchError(PermissionError):
    """角色失配（spec §6 + §11）。

    operator_role 不在该 transition 允许的角色集合内。
    API 层在 exception_handlers 注册 → 403 + audit event。
    """

    def __init__(
        self,
        message: str,
        *,
        operator_role: str,
        required_roles: Set[str],
        from_state: str,
        to_state: str,
    ) -> None:
        super().__init__(message)
        self.operator_role = operator_role
        self.required_roles = set(required_roles)
        self.from_state = from_state
        self.to_state = to_state


def is_legal_transition(from_state: str, to_state: str) -> bool:
    """判断 from_state → to_state 是否在合法转换表中。"""
    return f"{from_state}->{to_state}" in ALLOWED_TRANSITIONS


def allowed_roles_for(from_state: str, to_state: str) -> Set[str]:
    """返回该 transition 允许的角色集合（用于 403 envelope 显示）。"""
    key = f"{from_state}->{to_state}"
    return set(ROLE_POLICY.get(key, frozenset()))


def transition(
    repository: DraftRepository,
    *,
    draft_id: str,
    to_state: str,
    operator_id: str,
    operator_role: str = "",
    note: str = "",
    enforce_role_policy: bool = True,
) -> StateTransitionRecord:
    """执行一次状态转换（含角色 403 校验）。

    Parameters
    ----------
    repository : DraftRepository
        草稿仓库（in-memory 或 SQLAlchemy，T25 升级）。
    draft_id : str
        目标草稿 id。
    to_state : str
        目标状态（必须是 DraftStatus 的值）。
    operator_id : str
        操作人 / 服务标识。
    operator_role : str
        操作人角色（API 层从 X-Quality-Role 注入；system 用于自动回退）。
    note : str
        备注（如 confirm 时填 reviewer 复核意见；reject 时填退回原因）。
    enforce_role_policy : bool
        T25 默认 True：开启角色 403 校验；仅测试时显式 False 可绕过。

    Returns
    -------
    StateTransitionRecord
        新增的转换记录（已写入 repository）。

    Raises
    ------
    KeyError
        draft_id 不存在。
    ValueError
        to_state 不是合法状态值 / operator_id 为空。
    InvalidStateTransitionError
        跳级 / 同状态 / 不可达目标。
    RoleMismatchError
        operator_role 不在 ROLE_POLICY 允许集合。
    """
    d = repository.get(draft_id)
    if d is None:
        raise KeyError(f"draft_id={draft_id} 不存在")

    # 校验 to_state 是合法值
    valid_states = {s.value for s in DraftStatus}
    if to_state not in valid_states:
        raise ValueError(
            f"to_state={to_state!r} 不是合法状态；允许: {sorted(valid_states)}"
        )

    # 校验转换合法性
    if not is_legal_transition(d.status, to_state):
        raise InvalidStateTransitionError(
            f"非法转换：{d.status} → {to_state}；合法路径: {sorted(ALLOWED_TRANSITIONS.keys())}",
            from_state=d.status,
            to_state=to_state,
        )

    from_state = d.status
    key = f"{from_state}->{to_state}"

    # 角色 403 校验（Task 25 新增；defense-in-depth，API 层 + state machine 层双校验）
    if enforce_role_policy:
        allowed = allowed_roles_for(from_state, to_state)
        if operator_role not in allowed:
            raise RoleMismatchError(
                f"角色 {operator_role!r} 无法触发 {from_state} → {to_state}；"
                f"允许角色: {sorted(allowed) or '(none)'}",
                operator_role=operator_role,
                required_roles=allowed,
                from_state=from_state,
                to_state=to_state,
            )

    now_ts = _now_iso()

    # 修改 draft 字段
    d.status = to_state
    if to_state in (DraftStatus.REVIEWED.value, DraftStatus.REJECTED.value):
        d.reviewer_id = operator_id
        d.reviewer_role = operator_role
        d.reviewed_at = now_ts
        if note:
            d.notes = note
    elif to_state == DraftStatus.ARCHIVED.value:
        if note:
            d.notes = (d.notes + " | " if d.notes else "") + note
    elif to_state == DraftStatus.PENDING_REVIEW.value:
        # 回退到 pending_review 时清空 reviewer_* 字段（不视为评审结果）
        d.reviewer_id = ""
        d.reviewer_role = ""
        d.reviewed_at = None
        if note:
            d.notes = (d.notes + " | " if d.notes else "") + note
    d.updated_at = now_ts
    repository.save(d)

    rec = StateTransitionRecord(
        transition_id=f"tr_{uuid.uuid4().hex[:12]}",
        draft_id=draft_id,
        from_state=from_state,
        to_state=to_state,
        operator_id=operator_id,
        operator_role=operator_role,
        note=note,
        ts=now_ts,
    )
    repository.append_transition(rec)
    return rec


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = [
    "ALLOWED_TRANSITIONS",
    "InvalidStateTransitionError",
    "QUALITY_ROLE",
    "REVIEWER_ROLE",
    "ROLE_POLICY",
    "RoleMismatchError",
    "SYSTEM_ROLE",
    "allowed_roles_for",
    "is_legal_transition",
    "transition",
]