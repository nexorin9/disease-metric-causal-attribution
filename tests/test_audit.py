"""AuditLogger + strip_pii 测试（spec §6 + §11 收尾交付 · Task 30）。

覆盖：
  1. strip_pii —— PII 白名单字段（patient_name / id_card / phone / address /
     name / email / mobile / tel / ssn / birth_date / birthdate）递归剔除
  2. strip_pii —— 大小写不敏感 + 不改原对象 + dict/list/tuple/原样值
  3. AuditLogger 构造期校验（缺 session_factory）
  4. AuditEventType 枚举完整性（spec §6 6 类 + 5 类衍生共 11 类）
  5. write_audit_event 校验（event_type / payload / operator）
  6. write_audit_event 落库后字段完整（event_id / ts / payload 已脱敏 / schema_version /
     state_not_advanced）
  7. write_audit_event best-effort：注入会失败的 session_factory → 写返回 None 不抛
  8. list_recent + count + clear
  9. from_engine 工厂 + from_settings 工厂
 10. audit_logs 表自动建表（app.db.ensure_all_tables 集成）
"""
from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest


# ---- 模块导入 ----

from app.audit import (
    AUDIT_LOG_SCHEMA_VERSION,
    AUDIT_PAYLOAD_PII_FIELDS,
    AuditEventType,
    AuditLogger,
    AuditLoggerError,
    AuditPayloadError,
    InvalidEventTypeError,
    strip_pii,
)
from app.audit.logger import (
    AuditLogRow,
    AuditWriteResult,
    _row_to_dict,
)


# =============================================================================
# 测试夹具：SQLite in-memory + AuditLogger
# =============================================================================


@pytest.fixture
def sqlite_engine():
    """SQLite in-memory engine（每个测试独立；线程安全）。"""
    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    yield engine
    try:
        engine.dispose()
    except Exception:
        pass


@pytest.fixture
def audit_logger(sqlite_engine) -> AuditLogger:
    """构造 AuditLogger（自动 ensure audit_logs 表）。"""
    # 触发 ORM metadata 注册
    from app.db.schema import Base
    Base.metadata.create_all(sqlite_engine, tables=[AuditLogRow.__table__])
    return AuditLogger.from_engine(sqlite_engine)


# =============================================================================
# 1. strip_pii —— PII 白名单字段递归剔除
# =============================================================================


class TestStripPiiBasic:
    """strip_pii 基础功能。"""

    def test_dict_top_level_pii_removed(self):
        sample = {
            "patient_name": "张三",
            "id_card": "110101199001011234",
            "phone": "13800138000",
            "address": "北京市朝阳区",
            "encounter_id": "enc_abc123",
            "disease_code": "I21",
        }
        cleaned = strip_pii(sample)
        # PII 全清
        assert "patient_name" not in cleaned
        assert "id_card" not in cleaned
        assert "phone" not in cleaned
        assert "address" not in cleaned
        # 业务字段保留
        assert cleaned["encounter_id"] == "enc_abc123"
        assert cleaned["disease_code"] == "I21"

    def test_nested_dict_pii_removed(self):
        sample = {
            "operator": "user_001",
            "patient_info": {
                "patient_name": "李四",
                "id_card": "320101199002022345",
                "encounter_id": "enc_xyz",
            },
            "audit_meta": {
                "operator_id": "user_001",
                "patient_contact": {
                    "phone": "13900139000",
                    "address": "上海市浦东新区",
                },
            },
        }
        cleaned = strip_pii(sample)
        # 顶层 operator 保留
        assert cleaned["operator"] == "user_001"
        # patient_info 内的 PII 被剔除，encounter_id 保留
        assert "patient_name" not in cleaned["patient_info"]
        assert "id_card" not in cleaned["patient_info"]
        assert cleaned["patient_info"]["encounter_id"] == "enc_xyz"
        # audit_meta 内 patient_contact 全清
        assert "patient_contact" not in cleaned["audit_meta"] or \
            "phone" not in cleaned["audit_meta"].get("patient_contact", {})
        # patient_contact 整个对象的所有 PII 字段被剥离
        pc = cleaned["audit_meta"]["patient_contact"]
        assert "phone" not in pc
        assert "address" not in pc

    def test_list_of_dicts_pii_removed(self):
        sample = {
            "events": [
                {"event_type": "modeling/run", "operator": "u1", "patient_name": "患者甲"},
                {"event_type": "his/query", "operator": "u2", "phone": "13800138001"},
            ]
        }
        cleaned = strip_pii(sample)
        assert len(cleaned["events"]) == 2
        for ev in cleaned["events"]:
            assert "patient_name" not in ev
            assert "phone" not in ev
            assert "event_type" in ev
            assert "operator" in ev

    def test_all_pii_fields_removed(self):
        """11 个 PII 字段全部命中。"""
        sample = {f: f"v_{f}" for f in AUDIT_PAYLOAD_PII_FIELDS}
        sample["encounter_id"] = "enc_keep"
        cleaned = strip_pii(sample)
        for pii_field in AUDIT_PAYLOAD_PII_FIELDS:
            assert pii_field not in cleaned, f"{pii_field} should be removed"
        assert cleaned["encounter_id"] == "enc_keep"


class TestStripPiiEdgeCases:
    """strip_pii 边界场景。"""

    def test_case_insensitive(self):
        sample = {
            "PATIENT_NAME": "x",
            "Id_Card": "y",
            "PHONE": "z",
            "encounter_id": "enc_keep",
        }
        cleaned = strip_pii(sample, case_insensitive=True)
        assert "PATIENT_NAME" not in cleaned
        assert "Id_Card" not in cleaned
        assert "PHONE" not in cleaned
        assert cleaned["encounter_id"] == "enc_keep"

    def test_case_sensitive_default_removes_lower(self):
        """默认大小写不敏感；显式 case_insensitive=False 时仅匹配小写 key。"""
        sample = {
            "patient_name": "x",  # 仅小写会被剥离
            "PATIENT_NAME": "y",  # 大小写敏感时保留
            "encounter_id": "e1",
        }
        # 默认（case_insensitive=True）
        cleaned_default = strip_pii(sample)
        assert "patient_name" not in cleaned_default
        assert "PATIENT_NAME" not in cleaned_default

        # 显式 case_sensitive
        cleaned_strict = strip_pii(sample, case_insensitive=False)
        assert "patient_name" not in cleaned_strict
        assert "PATIENT_NAME" in cleaned_strict

    def test_does_not_mutate_original(self):
        sample = {
            "patient_name": "x",
            "encounter_id": "enc_keep",
        }
        original_snapshot = dict(sample)
        _ = strip_pii(sample)
        assert sample == original_snapshot, "strip_pii must not mutate original"

    def test_non_dict_non_list_pass_through(self):
        """int / str / None 原样返回。"""
        assert strip_pii(42) == 42
        assert strip_pii("hello") == "hello"
        assert strip_pii(None) is None

    def test_empty_dict(self):
        assert strip_pii({}) == {}

    def test_empty_list(self):
        assert strip_pii([]) == []

    def test_tuple_of_dicts(self):
        sample = ({"patient_name": "x", "k": 1}, {"phone": "y", "k": 2})
        cleaned = strip_pii(sample)
        assert isinstance(cleaned, tuple)
        assert len(cleaned) == 2
        for item in cleaned:
            assert "patient_name" not in item
            assert "phone" not in item

    def test_custom_pii_fields(self):
        sample = {"my_secret": "x", "encounter_id": "e1"}
        cleaned = strip_pii(sample, pii_fields=["my_secret"])
        assert "my_secret" not in cleaned
        assert cleaned["encounter_id"] == "e1"


# =============================================================================
# 2. AuditLogger 构造期校验
# =============================================================================


class TestAuditLoggerConstruction:
    """AuditLogger 构造 / 工厂。"""

    def test_construct_without_session_factory_raises(self):
        with pytest.raises(AuditLoggerError):
            AuditLogger(session_factory=None)  # type: ignore[arg-type]

    def test_from_engine_constructs(self, sqlite_engine):
        logger = AuditLogger.from_engine(sqlite_engine)
        assert logger is not None
        assert logger._session_factory is not None

    def test_from_settings_constructs(self, sqlite_engine, monkeypatch):
        """from_settings 走 app.db.get_default_session_factory 路径。"""
        from app.db import make_session_factory

        monkeypatch.setattr(
            "app.db.get_default_session_factory",
            lambda: make_session_factory(sqlite_engine),
        )

        # 构造 fake settings
        settings = MagicMock()
        logger = AuditLogger.from_settings(settings)
        assert logger is not None


# =============================================================================
# 3. AuditEventType 枚举完整性
# =============================================================================


class TestAuditEventTypeEnum:
    """AuditEventType 枚举完整性（spec §6 6 类 + 衍生 5 类共 11）。"""

    def test_required_event_types_present(self):
        # spec §6 列出的 6 类
        required = {
            "modeling/run",
            "draft/post",
            "state/transition",
            "model/call",
            "his/query",
            "refutation/degraded",
        }
        actual = {e.value for e in AuditEventType}
        for et in required:
            assert et in actual, f"missing required event_type={et}"

    def test_extended_event_types_present(self):
        extended = {
            "counterfactual/run",
            "dipcase/post",
            "model/degraded",
            "lifecycle/started",
            "lifecycle/shutdown",
        }
        actual = {e.value for e in AuditEventType}
        for et in extended:
            assert et in actual, f"missing extended event_type={et}"

    def test_enum_is_string_subtype(self):
        """AuditEventType 继承 str；可直接当字符串用。"""
        assert AuditEventType.MODELING_RUN == "modeling/run"
        # str + Enum：可 hash，可 in set
        s = {AuditEventType.MODELING_RUN, "draft/post"}
        assert "modeling/run" in s


# =============================================================================
# 4. write_audit_event 校验
# =============================================================================


class TestWriteAuditEventValidation:
    """write_audit_event 入参校验。"""

    def test_invalid_event_type_string_raises(self, audit_logger):
        with pytest.raises(InvalidEventTypeError):
            audit_logger.write_audit_event(
                event_type="unknown/event",
                operator="u1",
                payload={"k": 1},
            )

    def test_invalid_event_type_other_type_raises(self, audit_logger):
        with pytest.raises(InvalidEventTypeError):
            audit_logger.write_audit_event(
                event_type=12345,  # type: ignore[arg-type]
                operator="u1",
                payload={"k": 1},
            )

    def test_payload_must_be_mapping(self, audit_logger):
        with pytest.raises(AuditPayloadError):
            audit_logger.write_audit_event(
                event_type=AuditEventType.MODELING_RUN,
                operator="u1",
                payload=[1, 2, 3],  # type: ignore[arg-type]
            )

    def test_payload_string_raises(self, audit_logger):
        with pytest.raises(AuditPayloadError):
            audit_logger.write_audit_event(
                event_type=AuditEventType.MODELING_RUN,
                operator="u1",
                payload="not a mapping",  # type: ignore[arg-type]
            )

    def test_operator_required(self, audit_logger):
        with pytest.raises(AuditPayloadError):
            audit_logger.write_audit_event(
                event_type=AuditEventType.MODELING_RUN,
                operator="",
                payload={"k": 1},
            )

    def test_none_payload_treated_as_empty(self, audit_logger):
        ev_id = audit_logger.write_audit_event(
            event_type=AuditEventType.MODELING_RUN,
            operator="u1",
            payload=None,
        )
        assert ev_id is not None
        assert ev_id.startswith("ev_")


# =============================================================================
# 5. write_audit_event 落库完整性
# =============================================================================


class TestWriteAuditEventPersist:
    """write_audit_event 落库后字段完整 + PII 已脱敏。"""

    def test_returns_event_id(self, audit_logger):
        ev_id = audit_logger.write_audit_event(
            event_type=AuditEventType.MODELING_RUN,
            operator="u1",
            payload={"disease_code": "I21"},
        )
        assert ev_id is not None
        assert ev_id.startswith("ev_")
        assert len(ev_id) >= 12

    def test_event_persisted_with_all_fields(self, audit_logger):
        ev_id = audit_logger.write_audit_event(
            event_type=AuditEventType.MODELING_RUN,
            operator="u1",
            payload={"disease_code": "I21", "sample_size": 200},
            operator_role="quality_officer",
            source_module="app.api.modeling",
            state_not_advanced=True,
        )
        # 查回
        rows = audit_logger.list_recent(limit=1)
        assert len(rows) == 1
        row = rows[0]
        assert row["event_id"] == ev_id
        assert row["event_type"] == "modeling/run"
        assert row["operator"] == "u1"
        assert row["operator_role"] == "quality_officer"
        assert row["source_module"] == "app.api.modeling"
        assert row["schema_version"] == AUDIT_LOG_SCHEMA_VERSION
        assert row["state_not_advanced"] is True
        assert row["payload"]["disease_code"] == "I21"
        assert row["payload"]["sample_size"] == 200
        # ts 是 ISO8601 字符串
        assert row["ts"].endswith("Z")
        assert "T" in row["ts"]

    def test_pii_stripped_before_persist(self, audit_logger):
        ev_id = audit_logger.write_audit_event(
            event_type=AuditEventType.HIS_QUERY,
            operator="u1",
            payload={
                "patient_name": "张三",
                "id_card": "110101199001011234",
                "phone": "13800138000",
                "address": "北京市朝阳区",
                "encounter_id": "enc_keep",
                "disease_code": "I21",
            },
        )
        rows = audit_logger.list_recent(event_type="his/query")
        assert len(rows) == 1
        payload = rows[0]["payload"]
        # PII 全清
        for pii_field in ("patient_name", "id_card", "phone", "address"):
            assert pii_field not in payload, f"{pii_field} should be stripped"
        # 业务字段保留
        assert payload["encounter_id"] == "enc_keep"
        assert payload["disease_code"] == "I21"

    def test_pii_stripped_in_nested_payload(self, audit_logger):
        ev_id = audit_logger.write_audit_event(
            event_type=AuditEventType.MODELING_RUN,
            operator="u1",
            payload={
                "samples": [
                    {"encounter_id": "e1", "patient_name": "甲", "cmi": 1.2},
                    {"encounter_id": "e2", "phone": "13800138000", "cmi": 0.9},
                ]
            },
        )
        rows = audit_logger.list_recent(limit=1)
        payload = rows[0]["payload"]
        assert len(payload["samples"]) == 2
        for s in payload["samples"]:
            assert "patient_name" not in s
            assert "phone" not in s
            assert s["cmi"] in (1.2, 0.9)

    def test_schema_version_default(self, audit_logger):
        audit_logger.write_audit_event(
            event_type=AuditEventType.LIFECYCLE_STARTED,
            operator="system",
            payload={"app_env": "test"},
        )
        rows = audit_logger.list_recent(event_type="lifecycle/started")
        assert rows[0]["schema_version"] == AUDIT_LOG_SCHEMA_VERSION

    def test_state_not_advanced_default_true(self, audit_logger):
        audit_logger.write_audit_event(
            event_type=AuditEventType.MODEL_CALL,
            operator="u1",
            payload={"model_version": "test"},
        )
        rows = audit_logger.list_recent(event_type="model/call")
        assert rows[0]["state_not_advanced"] is True

    def test_state_not_advanced_can_be_false(self, audit_logger):
        audit_logger.write_audit_event(
            event_type=AuditEventType.STATE_TRANSITION,
            operator="reviewer_001",
            payload={"from": "pending_review", "to": "reviewed"},
            state_not_advanced=False,
        )
        rows = audit_logger.list_recent(event_type="state/transition")
        assert rows[0]["state_not_advanced"] is False


# =============================================================================
# 6. write_audit_event best-effort
# =============================================================================


class TestWriteAuditEventBestEffort:
    """DB 写入失败 → 返回 None 不抛异常。"""

    def test_db_failure_returns_none(self):
        """构造一个会失败的 session_factory（commit 抛异常）。"""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        engine = create_engine("sqlite:///:memory:", future=True)

        # session_factory 返回 session，但 session.commit 抛
        bad_session = MagicMock()
        bad_session.add = MagicMock()
        bad_session.commit = MagicMock(side_effect=RuntimeError("simulated DB failure"))
        bad_session.__enter__ = MagicMock(return_value=bad_session)
        bad_session.__exit__ = MagicMock(return_value=False)

        sf = MagicMock(return_value=bad_session)
        logger = AuditLogger(session_factory=sf)  # type: ignore[arg-type]

        result = logger.write_audit_event(
            event_type=AuditEventType.MODELING_RUN,
            operator="u1",
            payload={"k": 1},
        )
        assert result is None  # 不抛，返回 None


# =============================================================================
# 7. list_recent / count / clear
# =============================================================================


class TestAuditLoggerRead:
    """list_recent / count / clear。"""

    def test_list_recent_orders_by_ts_desc(self, audit_logger):
        for i in range(3):
            audit_logger.write_audit_event(
                event_type=AuditEventType.MODELING_RUN,
                operator=f"u{i}",
                payload={"i": i},
            )
        rows = audit_logger.list_recent(limit=10)
        assert len(rows) == 3
        # 时间倒序：最新在前（i=2 应在 i=0 之前）
        ts_seq = [r["ts"] for r in rows]
        assert ts_seq == sorted(ts_seq, reverse=True)

    def test_list_recent_filter_by_event_type(self, audit_logger):
        audit_logger.write_audit_event(
            event_type=AuditEventType.MODELING_RUN, operator="u1", payload={}
        )
        audit_logger.write_audit_event(
            event_type=AuditEventType.HIS_QUERY, operator="u1", payload={}
        )
        audit_logger.write_audit_event(
            event_type=AuditEventType.HIS_QUERY, operator="u2", payload={}
        )
        rows = audit_logger.list_recent(event_type="his/query")
        assert len(rows) == 2
        assert all(r["event_type"] == "his/query" for r in rows)

    def test_list_recent_filter_by_operator(self, audit_logger):
        audit_logger.write_audit_event(
            event_type=AuditEventType.MODELING_RUN, operator="u1", payload={}
        )
        audit_logger.write_audit_event(
            event_type=AuditEventType.MODELING_RUN, operator="u2", payload={}
        )
        rows = audit_logger.list_recent(operator="u1")
        assert len(rows) == 1
        assert rows[0]["operator"] == "u1"

    def test_list_recent_limit(self, audit_logger):
        for i in range(5):
            audit_logger.write_audit_event(
                event_type=AuditEventType.MODELING_RUN, operator=f"u{i}", payload={}
            )
        rows = audit_logger.list_recent(limit=2)
        assert len(rows) == 2

    def test_count_returns_total(self, audit_logger):
        assert audit_logger.count() == 0
        for _ in range(3):
            audit_logger.write_audit_event(
                event_type=AuditEventType.MODELING_RUN, operator="u1", payload={}
            )
        assert audit_logger.count() == 3

    def test_count_filter_by_event_type(self, audit_logger):
        audit_logger.write_audit_event(
            event_type=AuditEventType.MODELING_RUN, operator="u1", payload={}
        )
        audit_logger.write_audit_event(
            event_type=AuditEventType.HIS_QUERY, operator="u1", payload={}
        )
        audit_logger.write_audit_event(
            event_type=AuditEventType.HIS_QUERY, operator="u2", payload={}
        )
        assert audit_logger.count(event_type="his/query") == 2
        assert audit_logger.count(event_type="modeling/run") == 1
        assert audit_logger.count() == 3

    def test_clear_removes_all(self, audit_logger):
        for _ in range(3):
            audit_logger.write_audit_event(
                event_type=AuditEventType.MODELING_RUN, operator="u1", payload={}
            )
        assert audit_logger.count() == 3
        audit_logger.clear()
        assert audit_logger.count() == 0


# =============================================================================
# 8. audit_logs 表自动建表（与 app.db.ensure_all_tables 集成）
# =============================================================================


class TestAuditTablesIntegration:
    """app.db.ensure_all_tables 自动建 audit_logs 表。"""

    def test_ensure_all_tables_creates_audit_logs(self, sqlite_engine, tmp_path):
        """app.db.ensure_all_tables 必须能注册 AuditLogRow。"""
        # 触发 ORM 注册
        from app.audit.logger import AuditLogRow  # noqa: F401
        from app.db import ensure_all_tables
        from app.db.schema import Base

        # ensure_all_tables 应能成功构造（不要求 session_factory 默认单例）
        counts = ensure_all_tables(sqlite_engine)
        # audit_logs 必须存在
        assert "audit_logs" in counts

        # 反射验证
        from sqlalchemy import inspect
        insp = inspect(sqlite_engine)
        tables = insp.get_table_names()
        assert "audit_logs" in tables
        assert "drafts" in tables
        assert "draft_state_transitions" in tables

    def test_audit_logs_table_schema(self, sqlite_engine):
        """audit_logs 表字段齐全。"""
        from app.audit.logger import AuditLogRow  # noqa: F401
        from app.db.schema import Base

        Base.metadata.create_all(sqlite_engine, tables=[AuditLogRow.__table__])

        from sqlalchemy import inspect
        insp = inspect(sqlite_engine)
        cols = {c["name"]: c for c in insp.get_columns("audit_logs")}
        for required in (
            "event_id", "event_type", "operator", "operator_role",
            "ts", "payload", "source_module", "schema_version",
            "state_not_advanced",
        ):
            assert required in cols, f"audit_logs.{required} missing"


# =============================================================================
# 9. 端到端集成：6 类 event_type 全部入库
# =============================================================================


class TestEndToEndAllEventTypes:
    """spec §6 6 类 event_type 全部注册并可写入。"""

    @pytest.mark.parametrize("event_type", [
        "modeling/run",
        "draft/post",
        "state/transition",
        "model/call",
        "his/query",
        "refutation/degraded",
        "counterfactual/run",
        "dipcase/post",
        "model/degraded",
        "lifecycle/started",
        "lifecycle/shutdown",
    ])
    def test_all_event_types_writable(self, audit_logger, event_type):
        ev_id = audit_logger.write_audit_event(
            event_type=event_type,
            operator="system",
            payload={"event_type": event_type, "sample": 1},
        )
        assert ev_id is not None
        rows = audit_logger.list_recent(event_type=event_type)
        assert len(rows) == 1
        assert rows[0]["event_type"] == event_type
        assert rows[0]["payload"]["sample"] == 1


# =============================================================================
# 10. 模块契约
# =============================================================================


class TestModuleContracts:
    """模块对外契约（常量 / __all__ / 函数签名）。"""

    def test_audit_log_schema_version_format(self):
        """schema_version 必须形如 <name>/<major>.<minor>。"""
        assert AUDIT_LOG_SCHEMA_VERSION == "audit_log/1.0"

    def test_pii_fields_contains_required(self):
        """spec §6 必须的 PII 字段都在白名单。"""
        for required in ("patient_name", "id_card", "phone", "address"):
            assert required in AUDIT_PAYLOAD_PII_FIELDS

    def test_audit_logger_exports(self):
        from app import audit
        for required in (
            "AUDIT_LOG_SCHEMA_VERSION",
            "AUDIT_PAYLOAD_PII_FIELDS",
            "AuditEventType",
            "AuditLogger",
            "strip_pii",
        ):
            assert hasattr(audit, required), f"app.audit.{required} not found"

    def test_audit_logger_class_signatures(self):
        """构造 / 工厂方法签名稳定。"""
        import inspect
        sig = inspect.signature(AuditLogger.__init__)
        assert "session_factory" in sig.parameters
        assert sig.parameters["session_factory"].kind == inspect.Parameter.KEYWORD_ONLY

        sig = inspect.signature(AuditLogger.write_audit_event)
        for p in ("event_type", "operator", "payload"):
            assert p in sig.parameters

    def test_audit_write_result_dataclass(self):
        result = AuditWriteResult(
            ok=True,
            event_id="ev_test",
            event_type="modeling/run",
            operator="u1",
            ts="2026-01-01T00:00:00Z",
        )
        d = result.to_dict()
        assert d["ok"] is True
        assert d["event_id"] == "ev_test"
        assert d["event_type"] == "modeling/run"

    def test_strip_pii_function_callable(self):
        """strip_pii 可独立调用（spec §6 PII 强制剔除）。"""
        result = strip_pii({"patient_name": "x", "encounter_id": "e1"})
        assert "patient_name" not in result
        assert result["encounter_id"] == "e1"


# =============================================================================
# 11. 线程安全（并发写不丢不乱）
# =============================================================================


class TestThreadSafety:
    """AuditLogger 线程安全（spec §11 多线程最佳实践）。"""

    def test_concurrent_writes_dont_lose_events(self, audit_logger):
        """10 线程 × 5 次 = 50 条审计，应全部落库。"""
        errors: List[BaseException] = []
        event_ids: List[Optional[str]] = []

        def worker(idx: int) -> None:
            try:
                for j in range(5):
                    ev_id = audit_logger.write_audit_event(
                        event_type=AuditEventType.MODELING_RUN,
                        operator=f"u_{idx}",
                        payload={"idx": idx, "j": j},
                    )
                    event_ids.append(ev_id)
            except BaseException as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"unexpected errors: {errors}"
        assert len(event_ids) == 50
        # 全部成功
        assert all(e is not None for e in event_ids)
        # event_id 唯一
        unique_ids = set(event_ids)
        assert len(unique_ids) == 50
        # 全部可查
        assert audit_logger.count() == 50


# =============================================================================
# 12. _row_to_dict 辅助
# =============================================================================


class TestRowToDict:
    """_row_to_dict 工具函数。"""

    def test_row_to_dict_returns_9_fields(self, audit_logger):
        ev_id = audit_logger.write_audit_event(
            event_type=AuditEventType.MODELING_RUN,
            operator="u1",
            payload={"disease_code": "I21"},
        )
        # 直接查 row
        from sqlalchemy import select

        with audit_logger._session_factory() as session:
            row = session.execute(select(AuditLogRow).where(AuditLogRow.event_id == ev_id)).scalar_one()
            d = _row_to_dict(row)
        assert set(d.keys()) == {
            "event_id",
            "event_type",
            "operator",
            "operator_role",
            "ts",
            "payload",
            "source_module",
            "schema_version",
            "state_not_advanced",
        }