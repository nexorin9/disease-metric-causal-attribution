"""审计子包（spec.md §6 安全审计契约 + §11 收尾交付 · Task 30）。

| 模块                  | 引入 task | 说明                                       |
|-----------------------|-----------|--------------------------------------------|
| app.audit.logger      | T30       | AuditLogger + event_type 枚举 + strip_pii  |

⚠️ **模块边界**：audit 包不调任何业务核心 / 模型网关；它对外只暴露
``AuditLogger`` 给 API 层 / ModelGateway / DipcaseReporter 调 ``write_audit_event``；
字段脱敏（``strip_pii``）独立成函数，业务层亦可调用以做 pre-write 自检。
"""
from __future__ import annotations

from app.audit.logger import (
    AUDIT_LOG_SCHEMA_VERSION,
    AUDIT_PAYLOAD_PII_FIELDS,
    AuditEventType,
    AuditLogRow,
    AuditLogger,
    AuditLoggerError,
    AuditPayloadError,
    InvalidEventTypeError,
    strip_pii,
)

__all__ = [
    "AUDIT_LOG_SCHEMA_VERSION",
    "AUDIT_PAYLOAD_PII_FIELDS",
    "AuditEventType",
    "AuditLogRow",
    "AuditLogger",
    "AuditLoggerError",
    "AuditPayloadError",
    "InvalidEventTypeError",
    "strip_pii",
]