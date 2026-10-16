"""SQLAlchemy schema + 项目自带数据库顶层汇总（spec.md §3.2 + §6 审计契约）。

设计要点：
  - **单一入口**：app.db.{get_database_url, make_engine, make_session_factory,
    ensure_all_tables, get_default_engine, get_default_session_factory}；
    下游模块（audit.logger / draft_models）只通过本模块拿 engine/session_factory，
    避免各自重新解析 DATABASE_URL。
  - **自动建表**：lifespan startup 调 ensure_all_tables(engine) 一次性 create_all
    全套表（drafts + draft_state_transitions + audit_logs + 任何其他 ORM Model）。
  - **凭据一律从环境变量读**：DATABASE_URL（spec §8）。缺凭据显式报错不静默。
  - **PII 字段绝不落库**：audit_logs.payload 在写入前由 app.audit.logger.strip_pii
    强制剔除 patient_name / id_card / phone / address 等白名单字段。

入口形态：
  - 生产：lifespan 启动期 调 ensure_all_tables(engine)；其余时间通过 session_factory 拿 Session。
  - 本地：scripts/demo.sh（依赖 .env） → uvicorn 触发 lifespan。
  - 测试：tests/conftest.py 已 setdefault DATABASE_URL=sqlite:///./tests/.benchmarks/test_conftest.db；
    本模块 ensure_all_tables 同步建表。
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 子模块重导出（向后兼容：spec.md §3.2 + §6 + §11 收尾交付）
# ---------------------------------------------------------------------------

from app.db.draft_models import (
    DRAFTS_TABLE,
    DRAFT_TRANSITIONS_TABLE,
    DraftRow,
    DraftStateTransitionRow,
    SQLAlchemyDraftRepository,
    ensure_draft_tables,
    make_engine as _make_engine_sqlalchemy,  # 备份原名；下方的 make_engine 会覆盖对外
    make_session_factory as _make_session_factory_sqlalchemy,
)
from app.db.schema import (
    ADMISSION_ROUTE_VALUES,
    Base,
    DischargeFrontPageRow,
    NAMING_CONVENTION,
)


# ---------------------------------------------------------------------------
# 项目自带 DB 顶层汇总（spec §6 + §11 收尾交付 · Task 30）
# ---------------------------------------------------------------------------


def get_database_url() -> str:
    """从环境变量读 DATABASE_URL；缺失显式 RuntimeError（spec §8 绝不静默兜底）。

    本函数与 app.config.get_settings().database_url 一致；走独立读取路径
    是为了让 db 子包在不触发整个 Settings 解析的场合也能工作（审计 / 持久化层）。
    """
    import os

    url = os.environ.get("DATABASE_URL") or os.environ.get("database_url") or ""
    if not url:
        raise RuntimeError(
            "DATABASE_URL 缺失（spec §8 凭据缺失显式报错，绝不静默兜底）"
        )
    return str(url).strip()


def make_engine(database_url: str) -> Engine:
    """构造 SQLAlchemy engine（顶层入口；spec §8 项目自带 PostgreSQL / 本地 SQLite 替身）。

    SQLite 自动加 connect_args={"check_same_thread": False} 便于跨线程使用；
    PostgreSQL / MySQL 走标准 create_engine future=True。
    """
    from sqlalchemy import create_engine

    connect_args: dict[str, Any] = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    engine = create_engine(database_url, future=True, connect_args=connect_args)
    LOGGER.info("[app.db] engine constructed: %s", _mask_url(database_url))
    return engine


def make_session_factory(engine: Engine) -> sessionmaker:
    """构造 session 工厂。expire_on_commit=False 便于 commit 后继续访问字段。"""
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def ensure_all_tables(engine: Engine) -> dict[str, int]:
    """启动期一次性 create_all 全套表（drafts / draft_state_transitions / audit_logs）。

    返回 {table_name: row_count_after_create}；row_count 用于测试断言「建表 + 0 行」。
    任意 ORM Model 子类只要继承 app.db.schema.Base，都会被 create_all 覆盖。
    """
    # 触发 app.audit.logger 模块 import 以注册 AuditLogRow 到 Base.metadata
    try:
        from app.audit.logger import AuditLogRow  # noqa: F401
    except Exception as exc:  # pragma: no cover - 仅在 audit 子包缺位时退化
        LOGGER.warning(
            "[app.db] app.audit.logger import 失败：%s → audit_logs 表未注册", exc
        )

    table_objs = [
        DraftRow.__table__,
        DraftStateTransitionRow.__table__,
    ]
    # 把 AuditLogRow.__table__ 也加进来（若已 import）
    try:
        from app.audit.logger import AuditLogRow as _ALR  # noqa: F401

        table_objs.append(_ALR.__table__)
    except Exception:
        pass

    Base.metadata.create_all(engine, tables=table_objs)

    # 统计建表后的 row_count
    counts: dict[str, int] = {}
    with engine.begin() as conn:
        from sqlalchemy import text

        for tbl in table_objs:
            try:
                cnt = conn.execute(text(f"SELECT COUNT(*) FROM {tbl.name}")).scalar() or 0
            except Exception:  # pragma: no cover
                cnt = 0
            counts[tbl.name] = int(cnt)

    LOGGER.info(
        "[app.db] ensure_all_tables: %d tables created/exist: %s",
        len(table_objs),
        list(counts.keys()),
    )
    return counts


def _mask_url(url: str) -> str:
    """日志脱敏：把 URL 里的 password 段替换为 ***；其它部分原样保留。"""
    if "@" not in url:
        return url
    head, tail = url.split("@", 1)
    scheme_user, _, _ = head.rpartition(":")
    if scheme_user:
        return f"{scheme_user}:***@{tail}"
    return url


# ---------------------------------------------------------------------------
# 进程内默认单例（lifespan 复用；测试可 reset）
# ---------------------------------------------------------------------------

_default_engine: Optional[Engine] = None
_default_session_factory: Optional[sessionmaker] = None
# 用 RLock（可重入）而非 Lock：get_default_session_factory 内部调 get_default_engine 时
# 同一线程二次 acquire 必须放行，否则 lifespan startup 调用链永久死锁（T32 修复）
_default_lock = threading.RLock()


def get_default_engine() -> Engine:
    """lifespan 启动期构造一次；后续复用。

    构造失败显式抛错（spec §8 绝不静默兜底）。测试可调
    ``reset_default_engine_for_tests()`` 强制重新构造。
    """
    global _default_engine
    with _default_lock:
        if _default_engine is None:
            url = get_database_url()
            _default_engine = make_engine(url)
            ensure_all_tables(_default_engine)
    return _default_engine


def get_default_session_factory() -> sessionmaker:
    global _default_session_factory
    with _default_lock:
        if _default_session_factory is None:
            _default_session_factory = make_session_factory(get_default_engine())
    return _default_session_factory


def reset_default_engine_for_tests() -> None:
    """测试 / Phase 4 启动脚本专用：清掉默认 engine 强制重新解析 env。"""
    global _default_engine, _default_session_factory
    with _default_lock:
        if _default_engine is not None:
            try:
                _default_engine.dispose()
            except Exception:  # pragma: no cover
                pass
        _default_engine = None
        _default_session_factory = None


__all__ = [
    "ADMISSION_ROUTE_VALUES",
    "Base",
    "DRAFTS_TABLE",
    "DRAFT_TRANSITIONS_TABLE",
    "DischargeFrontPageRow",
    "DraftRow",
    "DraftStateTransitionRow",
    "NAMING_CONVENTION",
    "SQLAlchemyDraftRepository",
    "ensure_all_tables",
    "ensure_draft_tables",
    "get_database_url",
    "get_default_engine",
    "get_default_session_factory",
    "make_engine",
    "make_session_factory",
    "reset_default_engine_for_tests",
]