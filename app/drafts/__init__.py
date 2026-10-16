"""归因报告草稿子包（spec.md §3.2 + §11 收尾交付）。

| 模块                  | 引入 task | 说明                                       |
|-----------------------|-----------|--------------------------------------------|
| app.drafts.models     | T21       | Draft 数据类 + in-memory store + 状态枚举  |
| app.drafts.state_machine | T21    | pending_review → reviewed/rejected 转换    |
| app.drafts.repository | T25       | SQLAlchemy 持久化（Task 25 落盘）          |

T21 实现 in-memory 版便于面板联调与 SSE 流式末尾生成草稿；
T25 将 store 升级为 SQLAlchemy + 角色 403 + 跳级拒绝的完整校验。

⚠️ **模块边界**：drafts 包不调 audit / API；它对外只暴露给路由层调用。
API 层负责鉴权 / envelope / audit 写日志（spec §6）。
"""
from __future__ import annotations

from app.drafts.models import (
    DRAFT_SCHEMA_VERSION,
    Draft,
    DraftRepository,
    DraftStatus,
    StateTransitionRecord,
    get_default_repository,
    reset_default_repository_for_tests,
)
from app.drafts.state_machine import (
    ALLOWED_TRANSITIONS,
    InvalidStateTransitionError,
    QUALITY_ROLE,
    REVIEWER_ROLE,
    ROLE_POLICY,
    RoleMismatchError,
    SYSTEM_ROLE,
    allowed_roles_for,
    is_legal_transition,
    transition,
)

__all__ = [
    "ALLOWED_TRANSITIONS",
    "DRAFT_SCHEMA_VERSION",
    "Draft",
    "DraftRepository",
    "DraftStatus",
    "InvalidStateTransitionError",
    "QUALITY_ROLE",
    "REVIEWER_ROLE",
    "ROLE_POLICY",
    "RoleMismatchError",
    "SYSTEM_ROLE",
    "StateTransitionRecord",
    "allowed_roles_for",
    "get_default_repository",
    "is_legal_transition",
    "reset_default_repository_for_tests",
    "transition",
]