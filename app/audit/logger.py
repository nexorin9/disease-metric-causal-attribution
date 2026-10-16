"""审计日志（spec.md §6 + §11 收尾交付 · Task 30）。

约束（spec §6 + §8）：
  - **PII 字段强制剔除**：patient_name / id_card / phone / address / name / email
    等敏感字段在写入 ``audit_logs.payload`` 前由 ``strip_pii()`` 递归剔除；
    业务层调用 ``write_audit_event()`` 后落库的 payload 永远不含 PII（白名单校验）。
  - **event_type 受控枚举**：仅允许 ``AuditEventType`` 枚举值；非法 event_type 显式
    ``InvalidEventTypeError``，绝不静默默认归类。
  - **缺凭据显式报错**：未配置 DATABASE_URL / 未注入 session_factory → ``AuditLoggerError``，
    绝不静默默认到本地文件 / 内存。
  - **best-effort 写入**：写失败不阻断业务流程；调用方拿到 ``None`` 表示写失败，
    但日志层面 warning 记录。``write_audit_event`` 不抛业务异常。

落盘形态：
  - 表 ``audit_logs``（app.db 启动期 create_all 自动建表）。
  - 字段：event_id / event_type / operator / operator_role / ts(UTC ISO)
    / payload(JSON) / source_module / schema_version / state_not_advanced。
"""
from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Index,
    JSON,
    String,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Mapped, mapped_column, sessionmaker

from app.db.schema import Base


LOGGER = logging.getLogger(__name__)


# ---- schema_version + PII 白名单（spec §6 + §11） ----

AUDIT_LOG_SCHEMA_VERSION = "audit_log/1.0"

# PII 白名单：业务层即使传入，strip_pii 也会在写入前剔除。
# 集合结构便于 O(1) 查询；大小写不敏感（递归时同步小写化 key 后比对）。
AUDIT_PAYLOAD_PII_FIELDS = frozenset({
    "patient_name",
    "id_card",
    "phone",
    "address",
    "name",          # 兼容：包含患者姓名场景的通用 "name" 字段
    "email",
    "mobile",
    "tel",
    "ssn",
    "birth_date",    # 明文生日 + 性别可重识别患者
    "birthdate",
})


# ---- event_type 枚举（spec §6 安全审计 — 11 类） ----

class AuditEventType(str, Enum):
    """审计事件类型枚举（spec §6 安全审计契约 + T26/T27/T29 引入）。

    命名遵循 ``<domain>/<verb>`` 风格，便于 grep 与面板按域过滤。
    """

    # ---- 业务主路径 ----
    MODELING_RUN = "modeling/run"
    COUNTERFACTUAL_RUN = "counterfactual/run"
    DRAFT_POST = "draft/post"

    # ---- 状态变更 ----
    STATE_TRANSITION = "state/transition"

    # ---- 模型 / 上报 ----
    MODEL_CALL = "model/call"
    DIPCASE_REPORTER_POST = "dipcase/post"

    # ---- HIS 查询 ----
    HIS_QUERY = "his/query"

    # ---- 降级 / 失败 ----
    REFUTATION_DEGRADED = "refutation/degraded"
    MODEL_DEGRADED = "model/degraded"

    # ---- 系统生命周期 ----
    LIFECYCLE_STARTED = "lifecycle/started"
    LIFECYCLE_SHUTDOWN = "lifecycle/shutdown"


# event_type 字符串集合（供 CHECK 约束 + Pydantic-style 校验）
_EVENT_TYPES_TUPLE = tuple(e.value for e in AuditEventType)


# ---- 自定义异常 ----

class AuditLoggerError(RuntimeError):
    """AuditLogger 配置/构造期错误（缺凭据 / 缺 session_factory 等）。"""


class InvalidEventTypeError(ValueError):
    """event_type 不在 AuditEventType 枚举内。"""


class AuditPayloadError(ValueError):
    """payload 不是 Mapping（必须是 Mapping）。"""


# ---- PII 脱敏函数（业务层可独立调用） ----

def strip_pii(
    payload: Any,
    *,
    pii_fields: Optional[Sequence[str]] = None,
    case_insensitive: bool = True,
) -> Any:
    """递归剔除 dict / list 内的 PII 字段。

    Parameters
    ----------
    payload : Any
        待清洗对象；dict / list 递归处理；其它原样。
    pii_fields : Sequence[str] | None
        自定义 PII 字段集合；默认 ``AUDIT_PAYLOAD_PII_FIELDS``。
    case_insensitive : bool
        是否大小写不敏感；默认 True（dict key 既匹配 ``patient_name`` 也匹配 ``PATIENT_NAME``）。

    Returns
    -------
    Any
        清洗后的对象；dict / list 返回新对象（不修改原对象），其它原样。

    Notes
    -----
    - 任何嵌套 dict 中若 key 命中 PII 白名单（大小写不敏感），整 key:value 对被丢弃。
    - 列表/元组保持顺序；list 元素递归清洗。
    - 不对 value 做值替换/打码 —— 直接剔除字段，避免误打码导致审计模糊。
    """
    blocked = frozenset(
        {(p.lower() if case_insensitive else p) for p in (pii_fields or AUDIT_PAYLOAD_PII_FIELDS)}
    )

    def _clean(obj: Any) -> Any:
        if isinstance(obj, dict):
            out: Dict[str, Any] = {}
            for k, v in obj.items():
                key_norm = k.lower() if case_insensitive else k
                if key_norm in blocked:
                    continue
                out[k] = _clean(v)
            return out
        if isinstance(obj, list):
            return [_clean(x) for x in obj]
        if isinstance(obj, tuple):
            return tuple(_clean(x) for x in obj)
        return obj

    return _clean(payload)


# ---- ORM model：audit_logs 表 ----

class AuditLogRow(Base):
    """SQLAlchemy ORM 表：审计日志。

    字段：
      - event_id          主键（短 hex）
      - event_type        AuditEventType 枚举值
      - operator          操作人 / 服务标识
      - operator_role     操作人角色（med_committee_reviewer / quality_officer / system 等）
      - ts                UTC ISO8601
      - payload           JSON（已经过 strip_pii；不含 PII）
      - source_module     写入方模块（app.api.modeling / app.audit.logger 等）
      - schema_version    audit_log/1.0
      - state_not_advanced 业务状态是否未被推进（True = 仅日志，False = 已推进业务状态）
    """

    __tablename__ = "audit_logs"

    event_id: Mapped[str] = mapped_column(String(24), primary_key=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    operator: Mapped[str] = mapped_column(String(64), nullable=False)
    operator_role: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    ts: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    payload: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    source_module: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False, default=AUDIT_LOG_SCHEMA_VERSION)
    state_not_advanced: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    __table_args__ = (
        CheckConstraint(
            f"event_type IN {_EVENT_TYPES_TUPLE!r}".replace("'", "'"),
            name="ck_audit_logs_event_type_enum",
        ),
        Index("ix_audit_logs_operator_ts", "operator", "ts"),
        Index("ix_audit_logs_event_type_ts", "event_type", "ts"),
        {"comment": "审计日志（spec §6 · app.audit.logger · 字段已 strip_pii 脱敏）"},
    )

    def to_dict(self) -> Dict[str, Any]:
        return _row_to_dict(self)


# ---- 写入结果 dataclass ----

@dataclass
class AuditWriteResult:
    """一次 write_audit_event 的结果（best-effort 写入）。

    ok=True 表示成功落库；ok=False 表示失败（如 params 缺失或 DB 不可达），
    调用方应记录日志但不应阻断业务流程。
    """

    ok: bool
    event_id: Optional[str]
    event_type: str
    operator: str
    ts: str
    error: Optional[str] = None
    degraded: bool = False
    audit_event_written: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "operator": self.operator,
            "ts": self.ts,
            "error": self.error,
            "degraded": self.degraded,
            "audit_event_written": self.audit_event_written,
        }


# ---- AuditLogger 主体 ----

class AuditLogger:
    """审计日志器（线程安全 + best-effort 写入）。

    接口：
      - ``write_audit_event(event_type, operator, payload, operator_role='', source_module='', state_not_advanced=True)``
        → ``event_id``（成功）；None（失败 —— 不抛异常，业务层 best-effort 继续）。
      - ``from_settings(settings)`` 工厂：从 app.config.Settings 读 DATABASE_URL 构造。

    PII 字段由 ``strip_pii`` 在写入前强制剔除（白名单 = AUDIT_PAYLOAD_PII_FIELDS）。
    """

    def __init__(self, *, session_factory: sessionmaker) -> None:
        if session_factory is None:
            raise AuditLoggerError("session_factory 不可为 None")
        self._session_factory = session_factory
        self._lock = threading.RLock()

    # ---- 工厂 ----

    @classmethod
    def from_settings(cls, settings: Any) -> "AuditLogger":
        """从 app.config.Settings 构造 AuditLogger（lifespan 调用）。"""
        try:
            from app.db import get_default_session_factory  # 局部 import 防循环
        except Exception as exc:
            raise AuditLoggerError(f"app.db 不可用：{exc}") from exc
        try:
            sf = get_default_session_factory()
        except Exception as exc:
            raise AuditLoggerError(
                f"get_default_session_factory 失败（DATABASE_URL 缺失？）：{exc}"
            ) from exc
        return cls(session_factory=sf)

    @classmethod
    def from_engine(cls, engine: Engine) -> "AuditLogger":
        """从 SQLAlchemy engine 构造（测试 / 直连场景）。"""
        from app.db import make_session_factory

        return cls(session_factory=make_session_factory(engine))

    # ---- 写入 ----

    def write_audit_event(
        self,
        event_type: Union[str, AuditEventType],
        operator: str,
        payload: Optional[Mapping[str, Any]] = None,
        *,
        operator_role: str = "",
        source_module: str = "",
        state_not_advanced: bool = True,
    ) -> Optional[str]:
        """写一条审计事件。

        Returns
        -------
        Optional[str]
            event_id（成功）；None（失败 —— 不抛异常，业务层 best-effort 继续）。

        Notes
        -----
        - event_type 必须是 ``AuditEventType`` 枚举值；非法 ``InvalidEventTypeError``
          立即抛，**绝不**默认归类（spec §6 审计契约）。
        - payload 非 Mapping → ``AuditPayloadError`` 抛错；非 Mapping 视为 bug 必须显式报错。
        - payload 经 ``strip_pii`` 清洗后落库（patient_name / id_card / phone / address
          等字段强制剔除）。
        - 写库失败：warning 日志 + 返回 None + 不抛异常（best-effort）。
        """
        # 1. 校验 event_type
        if isinstance(event_type, AuditEventType):
            et_value = event_type.value
        elif isinstance(event_type, str):
            if event_type not in _EVENT_TYPES_TUPLE:
                raise InvalidEventTypeError(
                    f"event_type={event_type!r} 不在 AuditEventType 枚举内 "
                    f"({_EVENT_TYPES_TUPLE})"
                )
            et_value = event_type
        else:
            raise InvalidEventTypeError(
                f"event_type 类型错误：{type(event_type).__name__}"
            )

        # 2. 校验 payload
        if payload is None:
            payload = {}
        if not isinstance(payload, Mapping):
            raise AuditPayloadError(
                f"payload 必须是 Mapping；实测 {type(payload).__name__}"
            )

        # 3. 校验 operator
        op = str(operator or "").strip()
        if not op:
            raise AuditPayloadError("operator 不可为空")

        # 4. PII 脱敏（spec §6 强制）
        cleaned_payload = strip_pii(dict(payload))

        # 5. 构造 envelope
        ev_id = f"ev_{uuid.uuid4().hex[:20]}"
        now = datetime.now(timezone.utc)

        # 6. 写入 DB（best-effort）
        try:
            with self._lock:
                with self._session_factory() as session:
                    row = AuditLogRow(
                        event_id=ev_id,
                        event_type=et_value,
                        operator=op,
                        operator_role=str(operator_role or ""),
                        ts=now,
                        payload=cleaned_payload,
                        source_module=str(source_module or ""),
                        schema_version=AUDIT_LOG_SCHEMA_VERSION,
                        state_not_advanced=bool(state_not_advanced),
                    )
                    session.add(row)
                    session.commit()
        except Exception as exc:
            LOGGER.warning(
                "[audit] write_audit_event 失败：event_type=%s operator=%s err=%s",
                et_value,
                op,
                exc,
            )
            return None

        LOGGER.debug(
            "[audit] event written: %s type=%s op=%s",
            ev_id,
            et_value,
            op,
        )
        return ev_id

    # ---- 读取 ----

    def list_recent(
        self,
        *,
        event_type: Optional[str] = None,
        operator: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """查询最近审计事件（按 ts DESC）。仅供 healthz / 监控 / 测试用。"""
        from sqlalchemy import select

        with self._lock:
            with self._session_factory() as session:
                stmt = select(AuditLogRow)
                if event_type is not None:
                    stmt = stmt.where(AuditLogRow.event_type == event_type)
                if operator is not None:
                    stmt = stmt.where(AuditLogRow.operator == operator)
                stmt = stmt.order_by(AuditLogRow.ts.desc()).limit(max(1, int(limit or 1)))
                rows = session.execute(stmt).scalars().all()
                return [_row_to_dict(r) for r in rows]

    def count(self, *, event_type: Optional[str] = None) -> int:
        """统计审计事件条数（供 healthz / 测试）。"""
        from sqlalchemy import func, select

        with self._lock:
            with self._session_factory() as session:
                stmt = select(func.count()).select_from(AuditLogRow)
                if event_type is not None:
                    stmt = stmt.where(AuditLogRow.event_type == event_type)
                return int(session.execute(stmt).scalar() or 0)

    def clear(self) -> None:
        """测试用：清空所有审计事件（线程安全）。"""
        with self._lock:
            with self._session_factory() as session:
                session.query(AuditLogRow).delete()
                session.commit()


# ---- 辅助 ----

def _row_to_dict(row: AuditLogRow) -> Dict[str, Any]:
    return {
        "event_id": row.event_id,
        "event_type": row.event_type,
        "operator": row.operator,
        "operator_role": row.operator_role,
        "ts": _isoformat(row.ts),
        "payload": dict(row.payload or {}),
        "source_module": row.source_module,
        "schema_version": row.schema_version,
        "state_not_advanced": bool(row.state_not_advanced),
    }


def _isoformat(dt: Optional[datetime]) -> str:
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# 模块入口（__main__）已移除：app.audit.logger 是子模块；
# 以 `python -m app.audit.logger` 运行会触发 audit_logs 表的二次注册，
# 与 SQLAlchemy 2.x DeclarativeBase 单例模型冲突。
# 自检请直接执行 `pytest tests/test_audit.py`。


__all__ = [
    "AUDIT_LOG_SCHEMA_VERSION",
    "AUDIT_PAYLOAD_PII_FIELDS",
    "AuditEventType",
    "AuditLogRow",
    "AuditLogger",
    "AuditLoggerError",
    "AuditPayloadError",
    "AuditWriteResult",
    "InvalidEventTypeError",
    "strip_pii",
]