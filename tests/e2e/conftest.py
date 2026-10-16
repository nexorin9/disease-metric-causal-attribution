"""端到端测试 conftest — 一键准备替身 fixture。

替身策略（spec §6 / §11）：
  - HIS DB：SQLite（spec §8 替身）→ `HIS_DB_URL=sqlite:///<e2e db>/his_e2e.db`
  - 上报：tests.mocks.dipcase_stub（FastAPI + ASGITransport，capture POST requests）
  - ModelGateway：直接 inject 一个 stub ModelGateway（不走 httpx 真实协议栈 → 无挂死）
  - AuditLogger：in-memory 替身（不连真实 DB → 无挂死）
  - DB session_factory：in-memory SQLite（AuditLogger 也用同一 SQLite → 业务层 audit 真实落库）

⚠️ 不要把这些 fixture 用到 tests/ 根目录的单元测试里——它们启动 FastAPI lifespan + 注入多个替身，
对单元测试过重。单元测试应继续走单测替身（如 test_drafts_api.py）。
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

# 让 `from app.xxx import ...` 与 `from data.xxx import ...` 在测试时能工作
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# 环境变量兜底（必须在 conftest import 时就生效，否则 Settings 单例会缓存错值）
# ---------------------------------------------------------------------------

os.environ["APP_ENV"] = "test"
os.environ.setdefault("DATABASE_URL", "sqlite:///./tests/.benchmarks/dmca_e2e.db")
os.environ.setdefault("HIS_DB_URL", "sqlite:///./tests/.benchmarks/his_e2e.db")
os.environ.setdefault("HIS_DB_USER", "test_user_e2e")
os.environ.setdefault("HIS_DB_PASS", "test_pass_only_for_e2e")
os.environ.setdefault("DIPCASE_REPORTER_URL", "http://localhost:18899/v1")
os.environ.setdefault("DIPCASE_REPORTER_TOKEN", "test_token_only_for_e2e")
os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:18888/v1")
os.environ.setdefault("OPENAI_API_KEY", "test_openai_key_only_for_e2e")


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def project_root() -> Path:
    return _PROJECT_ROOT


@pytest.fixture(scope="session")
def e2e_db_dir(project_root: Path) -> Path:
    """e2e SQLite 数据库目录（自动创建）。"""
    db_dir = project_root / "tests" / ".benchmarks"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir


@pytest.fixture
def reset_settings():
    """重置 pydantic-settings lru_cache + draft repo 单例。"""
    from app.config import reset_settings_cache
    from app.drafts import reset_default_repository_for_tests

    reset_settings_cache()
    reset_default_repository_for_tests()
    yield
    reset_settings_cache()
    reset_default_repository_for_tests()


# ---------------------------------------------------------------------------
# HIS SQLite 替身（spec §3.1 + spec §8 允许 SQLite）
# ---------------------------------------------------------------------------


@pytest.fixture
def his_sqlite_fixture(e2e_db_dir, monkeypatch):
    """把 data/fixture_*.csv 导入到 his_e2e.db 的 discharge_front_page 表。

    Returns
    -------
    dict[str, int]
        {disease_code: row_count} 导入统计。
    """
    import pandas as pd

    db_path = e2e_db_dir / "his_e2e.db"
    if db_path.exists():
        db_path.unlink()

    columns = [
        "encounter_id",
        "primary_diagnosis_code",
        "treatment_strategy",
        "age_band",
        "admission_route",
        "los_days",
        "cmi",
        "readmit_flag",
        "complication_flag",
        "admission_date",
        "discharge_date",
        "quarter_tag",
        "created_at",
    ]

    def _age_band_to_int(v: Any) -> int:
        if isinstance(v, (int, float)):
            return int(v)
        s = str(v).strip()
        if s.endswith("+"):
            base = int(s.rstrip("+").split("-")[0] or "75")
        elif "-" in s:
            parts = s.split("-")
            base = int(parts[0])
        else:
            try:
                return int(float(s))
            except (ValueError, TypeError):
                return 0
        if base < 45:
            return 1
        if base < 65:
            return 2
        if base < 75:
            return 3
        return 4

    counts: Dict[str, int] = {}
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "CREATE TABLE discharge_front_page ("
            "  encounter_id TEXT PRIMARY KEY,"
            "  primary_diagnosis_code TEXT,"
            "  treatment_strategy TEXT,"
            "  age_band INTEGER,"
            "  admission_route TEXT,"
            "  los_days INTEGER,"
            "  cmi REAL,"
            "  readmit_flag INTEGER,"
            "  complication_flag INTEGER,"
            "  admission_date TEXT,"
            "  discharge_date TEXT,"
            "  quarter_tag TEXT,"
            "  created_at TEXT"
            ")"
        )

        fixtures = {
            "I21": _PROJECT_ROOT / "data" / "fixture_acute_mi.csv",
            "I63": _PROJECT_ROOT / "data" / "fixture_stroke.csv",
            "Z96.10": _PROJECT_ROOT / "data" / "fixture_hip_replacement.csv",
        }
        for disease_code, csv_path in fixtures.items():
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            cols = [c for c in columns if c in df.columns]
            df = df[cols].copy()
            df["age_band"] = df["age_band"].apply(_age_band_to_int).astype(int)
            df["admission_route"] = df["admission_route"].astype(str)
            df["readmit_flag"] = df["readmit_flag"].astype(int)
            df["complication_flag"] = df["complication_flag"].astype(int)
            df["los_days"] = df["los_days"].astype(int)
            df["cmi"] = df["cmi"].astype(float)
            if "quarter_tag" not in df.columns:
                df["quarter_tag"] = "2026-Q1"
            if "admission_date" not in df.columns:
                df["admission_date"] = "2026-01-01"
            if "discharge_date" not in df.columns:
                df["discharge_date"] = "2026-01-15"
            if "created_at" not in df.columns:
                df["created_at"] = "2026-01-15T12:00:00Z"
            df.to_sql("discharge_front_page", conn, if_exists="append", index=False)
            counts[disease_code] = int(len(df))

        conn.commit()
    finally:
        conn.close()

    monkeypatch.setenv("HIS_DB_URL", f"sqlite:///{db_path}")
    return counts


# ---------------------------------------------------------------------------
# 上报 mock（FastAPI 替身 + 进程内 PostResult capture）
# ---------------------------------------------------------------------------


class _InMemoryDipcaseClient:
    """替身 DipcaseReporter —— 同步 capture payload，不走真实 HTTP。

    Returns
    -------
    PostResult 兼容对象，含 to_dict() / ok / status_code / degraded / reason / attempt / audit_event_written
    """

    def __init__(self) -> None:
        self.captured_payloads: List[Dict[str, Any]] = []
        self._failure_response: Optional[Dict[str, Any]] = None
        # 不真访问网络；接替的真实 DipcaseReporter 行为（HTTP POST + retry 3）
        # 由 e2e 测试通过 monkeypatch 注入失败而非真实触网。

    def post_attribution_draft(self, payload: Dict[str, Any], *, return_payload: bool = False):
        from app.adapters.dipcase_reporter import PostResult

        # 走 strip_pii 模拟真实客户端（PII 字段剔除）
        try:
            from app.audit.logger import strip_pii
            safe_payload = strip_pii(dict(payload))
        except Exception:
            safe_payload = dict(payload)

        self.captured_payloads.append(safe_payload)

        if self._failure_response is not None:
            return PostResult(
                ok=False,
                status_code=self._failure_response.get("status_code", 503),
                error=self._failure_response.get("error", "e2e_injected_failure"),
                error_reason=self._failure_response.get("reason", "e2e_injected"),
                attempt=3,
                elapsed_ms=0,
                degraded=True,
                audit_event_written=False,
            )

        return PostResult(
            ok=True,
            status_code=200,
            error="",
            error_reason="",
            attempt=1,
            elapsed_ms=12,
            degraded=False,
            audit_event_written=True,
            extra={"report_id": f"rpt_e2e_{len(self.captured_payloads):04d}"},
        )


@pytest.fixture
def dipcase_capturer():
    """in-memory DipcaseReporter 替身 + captured payloads 容器。

    Returns
    -------
    _InMemoryDipcaseClient
        .captured_payloads 列出所有已捕获的 payload dict。
    """
    return _InMemoryDipcaseClient()


@pytest.fixture
def install_dipcase_mock(monkeypatch, dipcase_capturer):
    """把 DipcaseReporter.post_attribution_draft 替成 capture-only 版本。"""
    from app.adapters import dipcase_reporter

    def _patched_post(self, payload, *, return_payload=False):
        return dipcase_capturer.post_attribution_draft(payload, return_payload=return_payload)

    monkeypatch.setattr(
        dipcase_reporter.DipcaseReporter,
        "post_attribution_draft",
        _patched_post,
    )
    return dipcase_capturer


# ---------------------------------------------------------------------------
# ModelGateway 替身（直接构造 stub，避免 monkeypatching httpx 引发 lifespan 挂死）
# ---------------------------------------------------------------------------


class _StubModelResponse:
    """ModelGateway.ModelResponse 兼容对象。"""

    def __init__(self, content: Dict[str, Any]) -> None:
        self.content = content
        self.latency_ms = 5
        self.tokens_used = 42
        self.model_version = "e2e_stub/v1"
        self.prompt_hash = "e2e_stub_prompt_hash"
        self.response_hash = "e2e_stub_response_hash"
        self.fallback_used = False
        self.fallback_reason = ""
        self.state_not_advanced = True
        self.schema_version = "model_response/1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "content": self.content,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_used,
            "model_version": self.model_version,
            "prompt_hash": self.prompt_hash,
            "response_hash": self.response_hash,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "state_not_advanced": self.state_not_advanced,
        }


class _StubModelGateway:
    """替身 ModelGateway —— 三方法都返回合法 response（不连真实 LLM）。"""

    def __init__(self) -> None:
        self.schema_version = "model_response/1.0"

    def generate_node_narrative(self, node, effect, disease_code, outcome):
        return _StubModelResponse({
            "narrative_zh": f"[e2e_stub] 节点 {node} 的因果效应 {effect.ate if effect else 'N/A'}（业务演示）",
        })

    def generate_counterfactual_narrative(self, scenario, baseline, predicted):
        return _StubModelResponse({
            "predicted_outcome_zh": f"[e2e_stub] 反事实场景 {scenario.intervention_node} 预测 {predicted}",
            "confidence": 0.85,
        })

    def generate_draft_summary(self, draft):
        return _StubModelResponse({
            "summary_zh": f"[e2e_stub] 草稿摘要：{draft.disease_code}/{draft.outcome} 演示",
        })

    def close(self):
        pass


@pytest.fixture
def stub_model_gateway():
    return _StubModelGateway()


# ---------------------------------------------------------------------------
# AuditLogger in-memory 替身（避免 lifespan 期间 connect 真实 DB）
# ---------------------------------------------------------------------------


class _InMemoryAuditLogger:
    """替身 AuditLogger —— 写入 self.events 列表，不连真实 DB。"""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def write_audit_event(self, *, event_type: str, operator: str, payload=None,
                          operator_role: str = "", source_module: str = "",
                          state_not_advanced: bool = True) -> str:
        from app.audit.logger import strip_pii
        ev_id = f"mem_{len(self.events):04d}"
        self.events.append({
            "event_id": ev_id,
            "event_type": event_type,
            "operator": operator,
            "operator_role": operator_role,
            "source_module": source_module,
            "schema_version": "audit_log/1.0",
            "state_not_advanced": state_not_advanced,
            "payload": strip_pii(dict(payload or {})),
        })
        return ev_id

    def list_recent(self, event_type=None, operator=None, limit=50):
        out = [e for e in self.events
               if (event_type is None or e["event_type"] == event_type)
               and (operator is None or e["operator"] == operator)]
        return out[-limit:]

    def count(self, event_type=None):
        return sum(1 for e in self.events
                   if event_type is None or e["event_type"] == event_type)

    def clear(self):
        self.events.clear()


@pytest.fixture
def in_memory_audit_logger():
    return _InMemoryAuditLogger()


# ---------------------------------------------------------------------------
# DB session factory 替身（指向真实 SQLite 但表由 ensure_all_tables 一次性建）
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_audit_db(e2e_db_dir, monkeypatch):
    """确保 dmca_e2e.db 存在 + 3 表（drafts / draft_state_transitions / audit_logs）建好。"""
    db_path = e2e_db_dir / "dmca_e2e.db"
    if db_path.exists():
        db_path.unlink()
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    # 重置 settings 单例后再 ensure_all_tables
    from app.config import reset_settings_cache
    from app.db import (
        ensure_all_tables,
        get_default_engine,
        reset_default_engine_for_tests,
    )

    reset_settings_cache()
    reset_default_engine_for_tests()
    ensure_all_tables(get_default_engine())

    return db_path


# ---------------------------------------------------------------------------
# FastAPI app + TestClient（lifespan 已启动 + 替身全部接入）
# ---------------------------------------------------------------------------


@pytest.fixture
def fastapi_app_with_stubs(
    monkeypatch,
    reset_settings,
    his_sqlite_fixture,
    sqlite_audit_db,
    install_dipcase_mock,
    stub_model_gateway,
    in_memory_audit_logger,
):
    """构造 FastAPI app + 全部替身 + lifespan 已启动。

    Returns
    -------
    TestClient
        与 lifespan + 路由 + 替身 + 单例全部就位。
    """
    import app.main as app_main
    from fastapi.testclient import TestClient

    # 接管 lifespan 内 _build_* 入口（避免 ModelGateway/AuditLogger 真实连接）
    monkeypatch.setattr(app_main, "_build_audit_logger", lambda settings: in_memory_audit_logger)
    monkeypatch.setattr(app_main, "_build_model_gateway", lambda settings: stub_model_gateway)
    monkeypatch.setattr(app_main, "_build_session_factory", lambda settings: None)

    # 给 app.state 注入 drafts_repository（lifespan 不注入，避免被 routes fallback 到默认单例）
    from app.drafts.models import DraftRepository

    app = app_main.create_app()
    app.state.drafts_repository = DraftRepository()

    with TestClient(app, raise_server_exceptions=False) as client:
        # 强制把 in-memory audit 写回 app.state 以便被 best_effort_audit 抓到
        client.app.state.audit_logger = in_memory_audit_logger
        client.app.state.model_gateway = stub_model_gateway
        yield client


@pytest.fixture
def fresh_drafts_repository(fastapi_app_with_stubs):
    """取 lifespan 注入的 drafts_repository（test isolation）。"""
    r = getattr(fastapi_app_with_stubs.app.state, "drafts_repository", None)
    if r is None:
        from app.drafts import get_default_repository
        r = get_default_repository()
    return r
