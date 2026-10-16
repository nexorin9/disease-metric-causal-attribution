"""评审草稿 + 状态机审计 SQLAlchemy ORM 模型（spec.md §3.2 + §11 收尾 · Task 25 升级）。

两张表：
    - ``drafts``                  草稿主表（字段严格对齐 spec.md §3.2 草稿 payload 契约）
    - ``draft_state_transitions`` 状态变更审计表（FK→drafts.draft_id）

约束：
    - **PII 字段严禁落库**（patient_name / id_card / phone / address）—— schema 不定义。
    - 状态转换必须经 app.drafts.state_machine.transition()；直接 UPDATE 视为非法
      （spec §6 + §8 audit 契约）。
    - JSON 字段（effect_estimates / refutation_results / counterfactual_scenarios）使用
      SQLAlchemy JSON 类型；SQLite / PostgreSQL 均支持。
    - state_transitions.from_state/to_state 必须是 DraftStatus 枚举值（CHECK 在业务层做）。

兼容：
    - 与 in-memory DraftRepository 共享 DraftStatus / DRAFT_SCHEMA_VERSION / 转换语义。
    - drafts.draft_id 与 in-memory Draft.draft_id 一致（short hex）；保持面板 URL 兼容。
"""
from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import (
    Mapped,
    Session,
    mapped_column,
    sessionmaker,
)

from app.db.schema import Base
from app.drafts.models import (
    DRAFT_SCHEMA_VERSION,
    Draft,
    DraftStatus,
    StateTransitionRecord,
)


# DraftStatus 取值（SQLAlchemy Enum 约束）
_DRAFT_STATUS_VALUES = tuple(s.value for s in DraftStatus)

# 表名 / 索引前缀（spec.md §3.2：表名与 in-memory store 行为一致；按需前缀命名）
DRAFTS_TABLE = "drafts"
DRAFT_TRANSITIONS_TABLE = "draft_state_transitions"


class DraftRow(Base):
    """SQLAlchemy ORM 表：评审草稿主表。

    字段集（spec.md §3.2）严格对齐 in-memory Draft dataclass；
    JSON 字段用 SQLAlchemy JSON（SQLite/PostgreSQL 都支持）。
    """

    __tablename__ = DRAFTS_TABLE

    draft_id: Mapped[str] = mapped_column(String(36), nullable=False, primary_key=True)
    disease_code: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    quarter: Mapped[str] = mapped_column(String(8), nullable=False, default="")
    outcome: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    treatment: Mapped[str] = mapped_column(String(128), nullable=False)
    graph_snapshot_dot: Mapped[str] = mapped_column(Text, nullable=False, default="")

    effect_estimates: Mapped[List[Dict[str, Any]]] = mapped_column(
        # SQLAlchemy JSON：Python list[dict] 直接落库
        # type_ignore: SQLAlchemy 2.x JSON 在不同 DBAPI 下用 JSON/JSON 字段
        __import__("sqlalchemy").JSON,  # type: ignore[arg-type]
        nullable=False,
        default=list,
    )
    refutation_results: Mapped[List[Dict[str, Any]]] = mapped_column(
        __import__("sqlalchemy").JSON,  # type: ignore[arg-type]
        nullable=False,
        default=list,
    )
    counterfactual_scenarios: Mapped[List[Dict[str, Any]]] = mapped_column(
        __import__("sqlalchemy").JSON,  # type: ignore[arg-type]
        nullable=False,
        default=list,
    )

    narrative_zh: Mapped[str] = mapped_column(Text, nullable=False, default="")
    sample_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    status: Mapped[str] = mapped_column(
        SAEnum(*_DRAFT_STATUS_VALUES, name="draft_status_enum", native_enum=False),
        nullable=False,
        default=DraftStatus.PENDING_REVIEW.value,
        index=True,
    )

    reviewer_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    reviewer_role: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    notes: Mapped[str] = mapped_column(Text, nullable=False, default="")

    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # ---- 反范式化：方便按业务列表快照读 ----
    schema_version: Mapped[str] = mapped_column(
        String(32), nullable=False, default=DRAFT_SCHEMA_VERSION
    )

    __table_args__ = (
        # 业务约束：状态枚举值
        CheckConstraint(
            f"status IN {_DRAFT_STATUS_VALUES!r}".replace("'", "'"),
            name="ck_drafts_status_enum",
        ),
        Index("ix_drafts_disease_quarter", "disease_code", "quarter"),
        Index("ix_drafts_status_created", "status", "created_at"),
        {"comment": "归因报告评审草稿（spec.md §3.2 · app.drafts.models）"},
    )

    def to_draft(self) -> Draft:
        """转 in-memory Draft dataclass（供 API 层统一序列化）。"""
        return Draft(
            draft_id=self.draft_id,
            disease_code=self.disease_code,
            outcome=self.outcome,
            treatment=self.treatment,
            quarter=self.quarter or "",
            graph_snapshot_dot=self.graph_snapshot_dot or "",
            effect_estimates=list(self.effect_estimates or []),
            refutation_results=list(self.refutation_results or []),
            counterfactual_scenarios=list(self.counterfactual_scenarios or []),
            narrative_zh=self.narrative_zh or "",
            sample_size=int(self.sample_size or 0),
            status=self.status,
            reviewer_id=self.reviewer_id or "",
            reviewer_role=self.reviewer_role or "",
            created_at=_isoformat(self.created_at),
            reviewed_at=_isoformat(self.reviewed_at) if self.reviewed_at else None,
            updated_at=_isoformat(self.updated_at),
            notes=self.notes or "",
        )


class DraftStateTransitionRow(Base):
    """SQLAlchemy ORM 表：状态变更审计。

    字段集（spec.md §3.2 + §6 审计契约）：
      - transition_id  变更主键（short hex）
      - draft_id       FK → drafts.draft_id（CASCADE）
      - from_state     转换前状态
      - to_state       转换后状态
      - operator_id    操作人 / 服务标识
      - operator_role  角色（med_committee_reviewer / quality_officer / system）
      - note           备注
      - ts             转换时间（UTC）
    """

    __tablename__ = DRAFT_TRANSITIONS_TABLE

    transition_id: Mapped[str] = mapped_column(String(24), nullable=False, primary_key=True)
    draft_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(f"{DRAFTS_TABLE}.draft_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    from_state: Mapped[str] = mapped_column(String(32), nullable=False)
    to_state: Mapped[str] = mapped_column(String(32), nullable=False)
    operator_id: Mapped[str] = mapped_column(String(64), nullable=False)
    operator_role: Mapped[str] = mapped_column(String(32), nullable=False)
    note: Mapped[str] = mapped_column(Text, nullable=False, default="")
    ts: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)

    __table_args__ = (
        CheckConstraint(
            f"from_state IN {_DRAFT_STATUS_VALUES!r}".replace("'", "'"),
            name="ck_draft_transitions_from_enum",
        ),
        CheckConstraint(
            f"to_state IN {_DRAFT_STATUS_VALUES!r}".replace("'", "'"),
            name="ck_draft_transitions_to_enum",
        ),
        Index("ix_draft_transitions_draft_ts", "draft_id", "ts"),
        {"comment": "草稿状态变更审计（spec.md §3.2 · §6 审计契约）"},
    )

    def to_record(self) -> StateTransitionRecord:
        return StateTransitionRecord(
            transition_id=self.transition_id,
            draft_id=self.draft_id,
            from_state=self.from_state,
            to_state=self.to_state,
            operator_id=self.operator_id,
            operator_role=self.operator_role,
            note=self.note or "",
            ts=_isoformat(self.ts),
        )


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat(dt: Optional[datetime]) -> str:
    if dt is None:
        return ""
    if dt.tzinfo is None:
        # 兼容 SQLite 不存 tz：假定 UTC
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _gen_draft_id() -> str:
    return f"dr_{uuid.uuid4().hex[:16]}"


def _gen_transition_id() -> str:
    return f"tr_{uuid.uuid4().hex[:12]}"


def _parse_iso(ts: str) -> datetime:
    """ISO8601 字符串 → datetime（带 UTC tz）。"""
    if not ts:
        return _now_utc()
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return _now_utc()


class SQLAlchemyDraftRepository:
    """SQLAlchemy 版 DraftRepository：把 in-memory DraftRepository 接口暴露给状态机。

    接口签名与 DraftRepository 完全一致（get / save / list / has_draft / clear / transitions /
    append_transition），便于 state_machine.transition() 透明切换 in-memory ↔ SQLAlchemy。
    """

    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory
        self._lock = threading.RLock()

    # ---- 会话管理 ----

    def _session(self) -> Session:
        return self._session_factory()

    # ---- 写入 ----

    def create(
        self,
        *,
        disease_code: str,
        outcome: str,
        treatment: str,
        quarter: str = "",
        graph_snapshot_dot: str = "",
        effect_estimates: Optional[List[Dict[str, Any]]] = None,
        refutation_results: Optional[List[Dict[str, Any]]] = None,
        counterfactual_scenarios: Optional[List[Dict[str, Any]]] = None,
        narrative_zh: str = "",
        sample_size: int = 0,
    ) -> Draft:
        with self._lock:
            now = _now_utc()
            with self._session() as s:
                row = DraftRow(
                    draft_id=_gen_draft_id(),
                    disease_code=str(disease_code or "").strip(),
                    outcome=str(outcome or "").strip(),
                    treatment=str(treatment or "").strip(),
                    quarter=str(quarter or "").strip(),
                    graph_snapshot_dot=str(graph_snapshot_dot or ""),
                    effect_estimates=list(effect_estimates or []),
                    refutation_results=list(refutation_results or []),
                    counterfactual_scenarios=list(counterfactual_scenarios or []),
                    narrative_zh=str(narrative_zh or ""),
                    sample_size=int(sample_size or 0),
                    status=DraftStatus.PENDING_REVIEW.value,
                    created_at=now,
                    updated_at=now,
                )
                s.add(row)
                s.commit()
                s.refresh(row)
                return row.to_draft()

    def save(self, draft: Draft) -> None:
        with self._lock:
            with self._session() as s:
                row = s.get(DraftRow, draft.draft_id)
                if row is None:
                    raise KeyError(f"draft_id={draft.draft_id} 不存在；不可 save")
                row.disease_code = draft.disease_code
                row.outcome = draft.outcome
                row.treatment = draft.treatment
                row.quarter = draft.quarter
                row.graph_snapshot_dot = draft.graph_snapshot_dot
                row.effect_estimates = list(draft.effect_estimates or [])
                row.refutation_results = list(draft.refutation_results or [])
                row.counterfactual_scenarios = list(draft.counterfactual_scenarios or [])
                row.narrative_zh = draft.narrative_zh
                row.sample_size = int(draft.sample_size or 0)
                row.status = draft.status
                row.reviewer_id = draft.reviewer_id
                row.reviewer_role = draft.reviewer_role
                row.notes = draft.notes
                row.created_at = _parse_iso(draft.created_at) or _now_utc()
                row.updated_at = _now_utc()
                row.reviewed_at = _parse_iso(draft.reviewed_at) if draft.reviewed_at else None
                s.commit()

    def append_transition(self, record: StateTransitionRecord) -> None:
        with self._lock:
            with self._session() as s:
                # 验证 draft 存在（FK 已保证，但显式拒绝更好）
                if s.get(DraftRow, record.draft_id) is None:
                    raise KeyError(f"draft_id={record.draft_id} 不存在；不可追加 transition")
                row = DraftStateTransitionRow(
                    transition_id=record.transition_id or _gen_transition_id(),
                    draft_id=record.draft_id,
                    from_state=record.from_state,
                    to_state=record.to_state,
                    operator_id=record.operator_id,
                    operator_role=record.operator_role,
                    note=record.note or "",
                    ts=_parse_iso(record.ts),
                )
                s.add(row)
                s.commit()

    # ---- 读取 ----

    def get(self, draft_id: str) -> Optional[Draft]:
        with self._lock:
            with self._session() as s:
                row = s.get(DraftRow, draft_id)
                return row.to_draft() if row is not None else None

    def list(
        self,
        *,
        status: Optional[str] = None,
        disease_code: Optional[str] = None,
        limit: int = 50,
    ) -> List[Draft]:
        with self._lock:
            with self._session() as s:
                stmt = select(DraftRow)
                if status is not None:
                    stmt = stmt.where(DraftRow.status == status)
                if disease_code is not None:
                    stmt = stmt.where(DraftRow.disease_code == disease_code)
                stmt = stmt.order_by(DraftRow.created_at.desc()).limit(max(1, int(limit or 1)))
                rows = s.execute(stmt).scalars().all()
                return [r.to_draft() for r in rows]

    def transitions(self, draft_id: str) -> List[StateTransitionRecord]:
        with self._lock:
            with self._session() as s:
                stmt = (
                    select(DraftStateTransitionRow)
                    .where(DraftStateTransitionRow.draft_id == draft_id)
                    .order_by(DraftStateTransitionRow.ts.asc())
                )
                rows = s.execute(stmt).scalars().all()
                return [r.to_record() for r in rows]

    def has_draft(self, draft_id: str) -> bool:
        with self._lock:
            with self._session() as s:
                return s.get(DraftRow, draft_id) is not None

    def clear(self) -> None:
        """测试用：清空所有 draft + transitions。"""
        with self._lock:
            with self._session() as s:
                s.query(DraftStateTransitionRow).delete()
                s.query(DraftRow).delete()
                s.commit()


# ---- Engine / 启动期自动建表 ----


def make_engine(database_url: str) -> Engine:
    """构造 SQLAlchemy engine；SQLite 需加 check_same_thread=False 便于跨线程使用。"""
    connect_args: Dict[str, Any] = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    return create_engine(database_url, future=True, connect_args=connect_args)


def ensure_draft_tables(engine: Engine) -> None:
    """启动期确保 draft + state_transition 表存在。"""
    Base.metadata.create_all(engine, tables=[DraftRow.__table__, DraftStateTransitionRow.__table__])


def make_session_factory(engine: Engine) -> sessionmaker:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


__all__ = [
    "DRAFTS_TABLE",
    "DRAFT_TRANSITIONS_TABLE",
    "DraftRow",
    "DraftStateTransitionRow",
    "SQLAlchemyDraftRepository",
    "ensure_draft_tables",
    "make_engine",
    "make_session_factory",
]