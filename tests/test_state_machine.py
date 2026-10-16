"""草稿评审单状态机测试（spec.md §3.2 + §11 收尾 · Task 25）。

覆盖维度（11 维度）：
    1. 构造期校验：to_state 非法 / operator_id 缺失 / draft 不存在
    2. 合法转换：pending_review → reviewed / rejected
    3. 合法转换：reviewed / rejected → archived
    4. 合法转换：reviewed → pending_review（系统自动回退）
    5. 非法跳级：pending_review → archived / pending_review → pending_review / reviewed → pending_review（人类角色）
    6. 角色 403：非 med_committee_reviewer 触发 review 转换 → RoleMismatchError
    7. 角色 403：非 system 触发 rollback → RoleMismatchError
    8. 角色 403：归档允许 med_committee_reviewer + quality_officer 两种角色
    9. 审计记录：每次转换写一条 StateTransitionRecord，字段完整（含 transition_id / from / to / operator_id / operator_role / note / ts）
   10. SQLAlchemy 持久化：DraftRow + DraftStateTransitionRow 双表 round-trip；
       状态机接口签名兼容，可透明替换 in-memory store
   11. ALLOWED_TRANSITIONS / ROLE_POLICY 一致性：转换表里所有键都对应一条策略
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Iterator, List

import pytest
from sqlalchemy.engine import Engine
from sqlalchemy import inspect

from app.db import (
    DraftRow,
    DraftStateTransitionRow,
    SQLAlchemyDraftRepository,
    ensure_draft_tables,
    make_engine,
    make_session_factory,
)
from app.drafts import (
    ALLOWED_TRANSITIONS,
    DRAFT_SCHEMA_VERSION,
    QUALITY_ROLE,
    REVIEWER_ROLE,
    ROLE_POLICY,
    SYSTEM_ROLE,
    DraftStatus,
    InvalidStateTransitionError,
    RoleMismatchError,
    allowed_roles_for,
    transition,
)
from app.drafts.models import DraftRepository


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_in_memory_repo() -> DraftRepository:
    """每次返回全新的 in-memory DraftRepository（线程安全）。"""
    return DraftRepository()


def _seed_draft(repo: DraftRepository, *, status: DraftStatus = DraftStatus.PENDING_REVIEW):
    """创建一条 pending_review 草稿；如需特定状态，强制走合法路径转换。"""
    d = repo.create(
        disease_code="I21",
        outcome="cmi",
        treatment="PCI",
        quarter="2025-Q1",
        graph_snapshot_dot="digraph { a -> b }",
        effect_estimates=[{"ate": 0.5, "ci_lower": 0.1, "ci_upper": 0.9}],
        refutation_results=[{"method": "random_common_cause", "passed": True}],
        counterfactual_scenarios=[{"scenario": "if_PCI_then_0.85"}],
        narrative_zh="示例叙述",
        sample_size=120,
    )
    # 如需非 pending_review 起点，先走合法 review 转换
    if status == DraftStatus.REVIEWED:
        transition(
            repo,
            draft_id=d.draft_id,
            to_state=DraftStatus.REVIEWED.value,
            operator_id="reviewer-001",
            operator_role=REVIEWER_ROLE,
            note="人工通过",
        )
    elif status == DraftStatus.REJECTED:
        transition(
            repo,
            draft_id=d.draft_id,
            to_state=DraftStatus.REJECTED.value,
            operator_id="reviewer-001",
            operator_role=REVIEWER_ROLE,
            note="退回原因",
        )
    elif status == DraftStatus.ARCHIVED:
        # pending_review → reviewed → archived
        transition(
            repo,
            draft_id=d.draft_id,
            to_state=DraftStatus.REVIEWED.value,
            operator_id="reviewer-001",
            operator_role=REVIEWER_ROLE,
            note="先 review",
        )
        transition(
            repo,
            draft_id=d.draft_id,
            to_state=DraftStatus.ARCHIVED.value,
            operator_id="archivist-001",
            operator_role=REVIEWER_ROLE,
            note="归档",
        )
    return d


# ---------------------------------------------------------------------------
# 1. 构造期校验
# ---------------------------------------------------------------------------


class TestTransitionConstruction:
    """构造期校验：to_state 非法 / operator_id 缺失 / draft 不存在。"""

    def test_to_state_invalid_value_raises_valueerror(self):
        repo = _make_in_memory_repo()
        d = _seed_draft(repo)
        with pytest.raises(ValueError, match="to_state"):
            transition(
                repo,
                draft_id=d.draft_id,
                to_state="not_a_state",
                operator_id="op-1",
                operator_role=REVIEWER_ROLE,
            )

    def test_unknown_draft_id_raises_keyerror(self):
        repo = _make_in_memory_repo()
        with pytest.raises(KeyError, match="不存在"):
            transition(
                repo,
                draft_id="dr_does_not_exist",
                to_state=DraftStatus.REVIEWED.value,
                operator_id="op-1",
                operator_role=REVIEWER_ROLE,
            )

    def test_empty_operator_id_acceptable_in_state_machine(self):
        """state machine 不强校 operator_id 非空（API 层填；此处只校验业务状态）。"""
        repo = _make_in_memory_repo()
        d = _seed_draft(repo)
        rec = transition(
            repo,
            draft_id=d.draft_id,
            to_state=DraftStatus.REVIEWED.value,
            operator_id="",  # 故意为空
            operator_role=REVIEWER_ROLE,
        )
        assert rec.operator_id == ""


# ---------------------------------------------------------------------------
# 2. 合法转换：pending_review → reviewed / rejected
# ---------------------------------------------------------------------------


class TestLegalReviewTransitions:
    """pending_review → reviewed / rejected。"""

    def test_pending_review_to_reviewed_succeeds(self):
        repo = _make_in_memory_repo()
        d = _seed_draft(repo)
        rec = transition(
            repo,
            draft_id=d.draft_id,
            to_state=DraftStatus.REVIEWED.value,
            operator_id="reviewer-001",
            operator_role=REVIEWER_ROLE,
            note="复核通过",
        )
        assert rec.from_state == DraftStatus.PENDING_REVIEW.value
        assert rec.to_state == DraftStatus.REVIEWED.value
        assert rec.operator_role == REVIEWER_ROLE
        # draft 字段同步
        d2 = repo.get(d.draft_id)
        assert d2.status == DraftStatus.REVIEWED.value
        assert d2.reviewer_id == "reviewer-001"
        assert d2.reviewer_role == REVIEWER_ROLE
        assert d2.reviewed_at  # ISO 时间戳已写
        assert d2.notes == "复核通过"

    def test_pending_review_to_rejected_succeeds(self):
        repo = _make_in_memory_repo()
        d = _seed_draft(repo)
        rec = transition(
            repo,
            draft_id=d.draft_id,
            to_state=DraftStatus.REJECTED.value,
            operator_id="reviewer-001",
            operator_role=REVIEWER_ROLE,
            note="数据口径不齐",
        )
        assert rec.from_state == DraftStatus.PENDING_REVIEW.value
        assert rec.to_state == DraftStatus.REJECTED.value
        d2 = repo.get(d.draft_id)
        assert d2.status == DraftStatus.REJECTED.value
        assert d2.reviewer_id == "reviewer-001"
        assert d2.notes == "数据口径不齐"

    def test_reviewed_draft_clears_reviewer_on_rollback(self):
        """reviewed → pending_review（系统回退）应清空 reviewer_id/reviewer_role/reviewed_at。"""
        repo = _make_in_memory_repo()
        d = _seed_draft(repo, status=DraftStatus.REVIEWED)
        # 此时 reviewer 字段已写
        assert repo.get(d.draft_id).reviewer_id == "reviewer-001"
        rec = transition(
            repo,
            draft_id=d.draft_id,
            to_state=DraftStatus.PENDING_REVIEW.value,
            operator_id="system:auto-rollback",
            operator_role=SYSTEM_ROLE,
            note="上报失败自动回退",
        )
        assert rec.to_state == DraftStatus.PENDING_REVIEW.value
        d2 = repo.get(d.draft_id)
        assert d2.status == DraftStatus.PENDING_REVIEW.value
        assert d2.reviewer_id == ""
        assert d2.reviewer_role == ""
        assert d2.reviewed_at is None
        assert "上报失败自动回退" in d2.notes


# ---------------------------------------------------------------------------
# 3. 合法转换：reviewed / rejected → archived
# ---------------------------------------------------------------------------


class TestLegalArchiveTransitions:
    """reviewed / rejected → archived（归档）。"""

    def test_reviewed_to_archived_by_reviewer(self):
        repo = _make_in_memory_repo()
        d = _seed_draft(repo, status=DraftStatus.REVIEWED)
        rec = transition(
            repo,
            draft_id=d.draft_id,
            to_state=DraftStatus.ARCHIVED.value,
            operator_id="archivist-001",
            operator_role=REVIEWER_ROLE,
            note="季度归档",
        )
        assert rec.to_state == DraftStatus.ARCHIVED.value
        assert repo.get(d.draft_id).status == DraftStatus.ARCHIVED.value

    def test_rejected_to_archived_by_reviewer(self):
        repo = _make_in_memory_repo()
        d = _seed_draft(repo, status=DraftStatus.REJECTED)
        rec = transition(
            repo,
            draft_id=d.draft_id,
            to_state=DraftStatus.ARCHIVED.value,
            operator_id="archivist-001",
            operator_role=REVIEWER_ROLE,
            note="拒后归档",
        )
        assert rec.to_state == DraftStatus.ARCHIVED.value

    def test_reviewed_to_archived_by_quality_officer_allowed(self):
        """归档允许 quality_officer 触发（policy 允许）。"""
        repo = _make_in_memory_repo()
        d = _seed_draft(repo, status=DraftStatus.REVIEWED)
        rec = transition(
            repo,
            draft_id=d.draft_id,
            to_state=DraftStatus.ARCHIVED.value,
            operator_id="quality-001",
            operator_role=QUALITY_ROLE,
            note="质控归档",
        )
        assert rec.to_state == DraftStatus.ARCHIVED.value
        assert rec.operator_role == QUALITY_ROLE


# ---------------------------------------------------------------------------
# 4. 合法转换：reviewed → pending_review（系统自动回退）
# ---------------------------------------------------------------------------


class TestLegalRollbackTransition:
    """reviewed → pending_review（系统自动回退，spec §11 第 5 步）。"""

    def test_rollback_by_system_succeeds(self):
        repo = _make_in_memory_repo()
        d = _seed_draft(repo, status=DraftStatus.REVIEWED)
        rec = transition(
            repo,
            draft_id=d.draft_id,
            to_state=DraftStatus.PENDING_REVIEW.value,
            operator_id="system:auto-rollback",
            operator_role=SYSTEM_ROLE,
            note="上报失败自动回退",
        )
        assert rec.from_state == DraftStatus.REVIEWED.value
        assert rec.to_state == DraftStatus.PENDING_REVIEW.value
        assert rec.operator_role == SYSTEM_ROLE
        # 此时 reviewer 字段已清
        assert repo.get(d.draft_id).reviewer_id == ""


# ---------------------------------------------------------------------------
# 5. 非法跳级
# ---------------------------------------------------------------------------


class TestIllegalTransitions:
    """非法跳级：pending_review → archived / reviewed → pending_review（人类角色）/ 同一状态自环。"""

    def test_pending_review_to_archived_skip_rejected(self):
        repo = _make_in_memory_repo()
        d = _seed_draft(repo)
        with pytest.raises(InvalidStateTransitionError) as exc_info:
            transition(
                repo,
                draft_id=d.draft_id,
                to_state=DraftStatus.ARCHIVED.value,
                operator_id="op-1",
                operator_role=REVIEWER_ROLE,
            )
        assert exc_info.value.from_state == DraftStatus.PENDING_REVIEW.value
        assert exc_info.value.to_state == DraftStatus.ARCHIVED.value

    def test_pending_review_to_pending_review_self_loop_rejected(self):
        repo = _make_in_memory_repo()
        d = _seed_draft(repo)
        with pytest.raises(InvalidStateTransitionError):
            transition(
                repo,
                draft_id=d.draft_id,
                to_state=DraftStatus.PENDING_REVIEW.value,
                operator_id="op-1",
                operator_role=REVIEWER_ROLE,
            )

    def test_rollback_to_pending_review_by_human_rejected(self):
        """人类角色不能触发 reviewed → pending_review（仅 system）。"""
        repo = _make_in_memory_repo()
        d = _seed_draft(repo, status=DraftStatus.REVIEWED)
        # 即便绕过 role policy 也要走 is_legal_transition 拒绝；但 spec 里这条是合法的，
        # 所以失败点应在 role policy（见 TestRoleMismatch）。
        # 此处换跳级：reviewed → rejected 不在合法表里，应被拒绝。
        with pytest.raises(InvalidStateTransitionError):
            transition(
                repo,
                draft_id=d.draft_id,
                to_state=DraftStatus.REJECTED.value,
                operator_id="reviewer-001",
                operator_role=REVIEWER_ROLE,
            )

    def test_rejected_to_pending_review_rejected(self):
        repo = _make_in_memory_repo()
        d = _seed_draft(repo, status=DraftStatus.REJECTED)
        with pytest.raises(InvalidStateTransitionError):
            transition(
                repo,
                draft_id=d.draft_id,
                to_state=DraftStatus.PENDING_REVIEW.value,
                operator_id="op-1",
                operator_role=REVIEWER_ROLE,
            )

    def test_archived_to_anything_rejected(self):
        """终态 archived 不能再去任何状态。"""
        repo = _make_in_memory_repo()
        d = _seed_draft(repo, status=DraftStatus.ARCHIVED)
        for target in (DraftStatus.PENDING_REVIEW.value, DraftStatus.REVIEWED.value, DraftStatus.REJECTED.value):
            with pytest.raises(InvalidStateTransitionError):
                transition(
                    repo,
                    draft_id=d.draft_id,
                    to_state=target,
                    operator_id="op-1",
                    operator_role=REVIEWER_ROLE,
                )


# ---------------------------------------------------------------------------
# 6 & 7 & 8. 角色 403 校验
# ---------------------------------------------------------------------------


class TestRoleMismatch:
    """RoleMismatchError 强制：operator_role 不在 ROLE_POLICY 允许集合 → 拒绝。"""

    def test_quality_officer_cannot_review_pending_draft(self):
        """quality_officer 不能触发 pending_review → reviewed/rejected。"""
        repo = _make_in_memory_repo()
        d = _seed_draft(repo)
        with pytest.raises(RoleMismatchError) as exc_info:
            transition(
                repo,
                draft_id=d.draft_id,
                to_state=DraftStatus.REVIEWED.value,
                operator_id="quality-001",
                operator_role=QUALITY_ROLE,
            )
        assert exc_info.value.operator_role == QUALITY_ROLE
        assert REVIEWER_ROLE in exc_info.value.required_roles
        assert exc_info.value.from_state == DraftStatus.PENDING_REVIEW.value
        assert exc_info.value.to_state == DraftStatus.REVIEWED.value

    def test_empty_role_cannot_review(self):
        """operator_role 为空字符串 → 403。"""
        repo = _make_in_memory_repo()
        d = _seed_draft(repo)
        with pytest.raises(RoleMismatchError):
            transition(
                repo,
                draft_id=d.draft_id,
                to_state=DraftStatus.REVIEWED.value,
                operator_id="op-1",
                operator_role="",
            )

    def test_unknown_role_cannot_review(self):
        repo = _make_in_memory_repo()
        d = _seed_draft(repo)
        with pytest.raises(RoleMismatchError):
            transition(
                repo,
                draft_id=d.draft_id,
                to_state=DraftStatus.REJECTED.value,
                operator_id="op-1",
                operator_role="random_visitor",
            )

    def test_reviewer_cannot_trigger_system_rollback(self):
        """med_committee_reviewer 不能触发 reviewed → pending_review（仅 system）。"""
        repo = _make_in_memory_repo()
        d = _seed_draft(repo, status=DraftStatus.REVIEWED)
        with pytest.raises(RoleMismatchError) as exc_info:
            transition(
                repo,
                draft_id=d.draft_id,
                to_state=DraftStatus.PENDING_REVIEW.value,
                operator_id="reviewer-001",
                operator_role=REVIEWER_ROLE,
                note="人工想撤审",
            )
        assert exc_info.value.operator_role == REVIEWER_ROLE
        assert SYSTEM_ROLE in exc_info.value.required_roles

    def test_system_cannot_perform_review(self):
        """system 角色不能执行人工 review（policy 不允许）。"""
        repo = _make_in_memory_repo()
        d = _seed_draft(repo)
        with pytest.raises(RoleMismatchError) as exc_info:
            transition(
                repo,
                draft_id=d.draft_id,
                to_state=DraftStatus.REVIEWED.value,
                operator_id="system:bot",
                operator_role=SYSTEM_ROLE,
            )
        assert REVIEWER_ROLE in exc_info.value.required_roles
        assert SYSTEM_ROLE not in exc_info.value.required_roles

    def test_role_403_does_not_modify_draft(self):
        """RoleMismatchError 后草稿状态保持不变；audit 也未写入。"""
        repo = _make_in_memory_repo()
        d = _seed_draft(repo)
        original_status = d.status
        original_updated_at = d.updated_at
        with pytest.raises(RoleMismatchError):
            transition(
                repo,
                draft_id=d.draft_id,
                to_state=DraftStatus.REVIEWED.value,
                operator_id="op-1",
                operator_role="random_visitor",
            )
        d2 = repo.get(d.draft_id)
        assert d2.status == original_status
        assert d2.updated_at == original_updated_at
        # 没有 transition 记录
        assert repo.transitions(d.draft_id) == []

    def test_enforce_role_policy_false_bypasses_403_for_legacy_or_tests(self):
        """enforce_role_policy=False 关闭校验（仅测试/迁移用）。"""
        repo = _make_in_memory_repo()
        d = _seed_draft(repo)
        rec = transition(
            repo,
            draft_id=d.draft_id,
            to_state=DraftStatus.REVIEWED.value,
            operator_id="op-1",
            operator_role="legacy_role",
            enforce_role_policy=False,
        )
        assert rec.to_state == DraftStatus.REVIEWED.value


# ---------------------------------------------------------------------------
# 9. 审计记录
# ---------------------------------------------------------------------------


class TestAuditTrail:
    """每次 transition 写一条 StateTransitionRecord，字段完整。"""

    def test_every_transition_appends_one_record(self):
        repo = _make_in_memory_repo()
        d = _seed_draft(repo)
        transition(
            repo,
            draft_id=d.draft_id,
            to_state=DraftStatus.REVIEWED.value,
            operator_id="reviewer-001",
            operator_role=REVIEWER_ROLE,
            note="first review",
        )
        transition(
            repo,
            draft_id=d.draft_id,
            to_state=DraftStatus.ARCHIVED.value,
            operator_id="archivist-001",
            operator_role=REVIEWER_ROLE,
            note="archive",
        )
        recs = repo.transitions(d.draft_id)
        assert len(recs) == 2

    def test_audit_record_fields_complete(self):
        repo = _make_in_memory_repo()
        d = _seed_draft(repo)
        rec = transition(
            repo,
            draft_id=d.draft_id,
            to_state=DraftStatus.REVIEWED.value,
            operator_id="reviewer-001",
            operator_role=REVIEWER_ROLE,
            note="复核通过",
        )
        # 字段完整
        assert rec.transition_id.startswith("tr_")
        assert rec.draft_id == d.draft_id
        assert rec.from_state == DraftStatus.PENDING_REVIEW.value
        assert rec.to_state == DraftStatus.REVIEWED.value
        assert rec.operator_id == "reviewer-001"
        assert rec.operator_role == REVIEWER_ROLE
        assert rec.note == "复核通过"
        assert rec.ts  # ISO 时间戳

    def test_audit_record_serializable_to_dict(self):
        repo = _make_in_memory_repo()
        d = _seed_draft(repo)
        rec = transition(
            repo,
            draft_id=d.draft_id,
            to_state=DraftStatus.REVIEWED.value,
            operator_id="reviewer-001",
            operator_role=REVIEWER_ROLE,
        )
        d_dict = rec.to_dict()
        # JSON 序列化往返一致
        s = json.dumps(d_dict, ensure_ascii=False, sort_keys=True)
        d2 = json.loads(s)
        assert d2 == d_dict

    def test_failed_transition_writes_no_audit(self):
        """非法/失败转换（InvalidStateTransition / RoleMismatch）不写 audit。"""
        repo = _make_in_memory_repo()
        d = _seed_draft(repo)
        # 非法跳级
        with pytest.raises(InvalidStateTransitionError):
            transition(
                repo,
                draft_id=d.draft_id,
                to_state=DraftStatus.ARCHIVED.value,
                operator_id="op-1",
                operator_role=REVIEWER_ROLE,
            )
        # 角色 403
        with pytest.raises(RoleMismatchError):
            transition(
                repo,
                draft_id=d.draft_id,
                to_state=DraftStatus.REVIEWED.value,
                operator_id="op-1",
                operator_role="random",
            )
        assert repo.transitions(d.draft_id) == []


# ---------------------------------------------------------------------------
# 10. SQLAlchemy 持久化
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_engine() -> Iterator[Engine]:
    """独立 SQLite 文件替身（不污染默认测试 DB）。"""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test_state_machine.db"
        engine = make_engine(f"sqlite:///{db_path}")
        ensure_draft_tables(engine)
        yield engine
        engine.dispose()


@pytest.fixture
def sqlalchemy_repo(sqlite_engine: Engine) -> Iterator[SQLAlchemyDraftRepository]:
    repo = SQLAlchemyDraftRepository(make_session_factory(sqlite_engine))
    repo.clear()
    yield repo
    repo.clear()


class TestSQLAlchemyPersistence:
    """SQLAlchemy 持久化：DraftRow + DraftStateTransitionRow 双表 round-trip。"""

    def test_create_and_get_roundtrip(self, sqlalchemy_repo: SQLAlchemyDraftRepository):
        d = sqlalchemy_repo.create(
            disease_code="I21",
            outcome="cmi",
            treatment="PCI",
            quarter="2025-Q1",
            graph_snapshot_dot="digraph { a -> b }",
            effect_estimates=[{"ate": 0.5}],
            refutation_results=[{"method": "random_common_cause", "passed": True}],
            counterfactual_scenarios=[{"scenario": "if_PCI_then_0.85"}],
            narrative_zh="示例叙述",
            sample_size=120,
        )
        assert d.draft_id.startswith("dr_")
        assert d.status == DraftStatus.PENDING_REVIEW.value
        d2 = sqlalchemy_repo.get(d.draft_id)
        assert d2 is not None
        assert d2.disease_code == "I21"
        assert d2.outcome == "cmi"
        assert d2.treatment == "PCI"
        assert d2.sample_size == 120
        assert d2.effect_estimates == [{"ate": 0.5}]
        assert d2.refutation_results == [{"method": "random_common_cause", "passed": True}]
        assert d2.counterfactual_scenarios == [{"scenario": "if_PCI_then_0.85"}]
        assert d2.narrative_zh == "示例叙述"
        assert d2.created_at
        assert d2.updated_at

    def test_save_modifies_existing_draft(self, sqlalchemy_repo: SQLAlchemyDraftRepository):
        d = sqlalchemy_repo.create(
            disease_code="I21",
            outcome="cmi",
            treatment="PCI",
        )
        d.narrative_zh = "新叙述"
        d.effect_estimates = [{"ate": 0.7}]
        sqlalchemy_repo.save(d)
        d2 = sqlalchemy_repo.get(d.draft_id)
        assert d2.narrative_zh == "新叙述"
        assert d2.effect_estimates == [{"ate": 0.7}]

    def test_append_transition_persists_separately(self, sqlalchemy_repo: SQLAlchemyDraftRepository):
        d = sqlalchemy_repo.create(
            disease_code="I21",
            outcome="cmi",
            treatment="PCI",
        )
        from app.drafts.models import StateTransitionRecord

        rec = StateTransitionRecord(
            transition_id=f"tr_{uuid.uuid4().hex[:12]}",
            draft_id=d.draft_id,
            from_state=DraftStatus.PENDING_REVIEW.value,
            to_state=DraftStatus.REVIEWED.value,
            operator_id="reviewer-001",
            operator_role=REVIEWER_ROLE,
            note="复核通过",
            ts="2026-01-01T00:00:00Z",
        )
        sqlalchemy_repo.append_transition(rec)
        recs = sqlalchemy_repo.transitions(d.draft_id)
        assert len(recs) == 1
        assert recs[0].transition_id == rec.transition_id
        assert recs[0].from_state == DraftStatus.PENDING_REVIEW.value
        assert recs[0].to_state == DraftStatus.REVIEWED.value
        assert recs[0].operator_role == REVIEWER_ROLE
        assert recs[0].note == "复核通过"

    def test_list_filters_by_status(self, sqlalchemy_repo: SQLAlchemyDraftRepository):
        d1 = sqlalchemy_repo.create(disease_code="I21", outcome="cmi", treatment="PCI")
        d2 = sqlalchemy_repo.create(disease_code="I63", outcome="los_days", treatment="thrombolysis")
        d3 = sqlalchemy_repo.create(disease_code="Z96.10", outcome="readmit_flag", treatment="THA")
        # 全部 pending_review
        all_items = sqlalchemy_repo.list()
        assert len(all_items) == 3

        # d1 → reviewed
        sqlalchemy_repo.append_transition(
            _mk_transition_record(d1.draft_id, "pending_review", "reviewed")
        )
        d1.status = DraftStatus.REVIEWED.value
        sqlalchemy_repo.save(d1)

        reviewed_items = sqlalchemy_repo.list(status=DraftStatus.REVIEWED.value)
        assert len(reviewed_items) == 1
        assert reviewed_items[0].draft_id == d1.draft_id

        pending_items = sqlalchemy_repo.list(status=DraftStatus.PENDING_REVIEW.value)
        assert len(pending_items) == 2

        i21_items = sqlalchemy_repo.list(disease_code="I21")
        assert len(i21_items) == 1
        assert i21_items[0].draft_id == d1.draft_id

    def test_has_draft_and_clear(self, sqlalchemy_repo: SQLAlchemyDraftRepository):
        d = sqlalchemy_repo.create(disease_code="I21", outcome="cmi", treatment="PCI")
        assert sqlalchemy_repo.has_draft(d.draft_id) is True
        assert sqlalchemy_repo.has_draft("dr_nope") is False
        sqlalchemy_repo.clear()
        assert sqlalchemy_repo.has_draft(d.draft_id) is False

    def test_full_workflow_via_state_machine(self, sqlalchemy_repo: SQLAlchemyDraftRepository):
        """完整工作流：create → review → archive → transitions 列表正确。"""
        d = sqlalchemy_repo.create(
            disease_code="I21",
            outcome="cmi",
            treatment="PCI",
        )
        # pending_review → reviewed
        rec1 = transition(
            sqlalchemy_repo,
            draft_id=d.draft_id,
            to_state=DraftStatus.REVIEWED.value,
            operator_id="reviewer-001",
            operator_role=REVIEWER_ROLE,
            note="通过",
        )
        assert rec1.transition_id.startswith("tr_")
        # reviewed → archived
        rec2 = transition(
            sqlalchemy_repo,
            draft_id=d.draft_id,
            to_state=DraftStatus.ARCHIVED.value,
            operator_id="archivist-001",
            operator_role=QUALITY_ROLE,
            note="归档",
        )
        # 验证状态机 + repo 持久化
        d_final = sqlalchemy_repo.get(d.draft_id)
        assert d_final.status == DraftStatus.ARCHIVED.value
        assert d_final.reviewer_id == "reviewer-001"
        assert d_final.reviewer_role == REVIEWER_ROLE
        recs = sqlalchemy_repo.transitions(d.draft_id)
        assert len(recs) == 2
        states = [(r.from_state, r.to_state) for r in recs]
        assert (DraftStatus.PENDING_REVIEW.value, DraftStatus.REVIEWED.value) in states
        assert (DraftStatus.REVIEWED.value, DraftStatus.ARCHIVED.value) in states

    def test_role_403_via_sqlalchemy_repo(self, sqlalchemy_repo: SQLAlchemyDraftRepository):
        """SQLAlchemy repo 也走 state machine 角色校验。"""
        d = sqlalchemy_repo.create(
            disease_code="I21",
            outcome="cmi",
            treatment="PCI",
        )
        with pytest.raises(RoleMismatchError):
            transition(
                sqlalchemy_repo,
                draft_id=d.draft_id,
                to_state=DraftStatus.REVIEWED.value,
                operator_id="op-1",
                operator_role="random",
            )
        # 状态未变
        d2 = sqlalchemy_repo.get(d.draft_id)
        assert d2.status == DraftStatus.PENDING_REVIEW.value
        assert sqlalchemy_repo.transitions(d.draft_id) == []

    def test_orm_tables_exist(self, sqlite_engine: Engine):
        """启动期 ensure_draft_tables 已建好两张表。"""
        insp = inspect(sqlite_engine)
        # SQLAlchemy 默认也会建 alembic_version 之类；这里只关心两张
        assert "drafts" in insp.get_table_names()
        assert "draft_state_transitions" in insp.get_table_names()

    def test_draft_table_columns_match_spec(self, sqlite_engine: Engine):
        """表结构对齐 spec §3.2 字段。"""
        insp = inspect(sqlite_engine)
        cols = {c["name"] for c in insp.get_columns("drafts")}
        expected = {
            "draft_id",
            "disease_code",
            "quarter",
            "outcome",
            "treatment",
            "graph_snapshot_dot",
            "effect_estimates",
            "refutation_results",
            "counterfactual_scenarios",
            "narrative_zh",
            "sample_size",
            "status",
            "reviewer_id",
            "reviewer_role",
            "notes",
            "created_at",
            "updated_at",
            "reviewed_at",
            "schema_version",
        }
        assert expected.issubset(cols)


def _mk_transition_record(draft_id: str, from_state: str, to_state: str):
    """构造测试用 StateTransitionRecord（绕过 state machine 的角色校验）。"""
    from app.drafts.models import StateTransitionRecord

    return StateTransitionRecord(
        transition_id=f"tr_{uuid.uuid4().hex[:12]}",
        draft_id=draft_id,
        from_state=from_state,
        to_state=to_state,
        operator_id="reviewer-001",
        operator_role=REVIEWER_ROLE,
        note="test",
        ts="2026-01-01T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# 11. ALLOWED_TRANSITIONS / ROLE_POLICY 一致性
# ---------------------------------------------------------------------------


class TestRolePolicyConsistency:
    """ALLOWED_TRANSITIONS 与 ROLE_POLICY 一致性 + 便捷函数 allowed_roles_for。"""

    def test_all_allowed_transitions_have_policy(self):
        """ALLOWED_TRANSITIONS 的每个键都必须在 ROLE_POLICY 中。"""
        for key in ALLOWED_TRANSITIONS:
            assert key in ROLE_POLICY, f"{key} 缺少 ROLE_POLICY 条目"

    def test_all_policy_entries_are_legal_transitions(self):
        """ROLE_POLICY 中不允许出现 ALLOWED_TRANSITIONS 没有的键。"""
        for key in ROLE_POLICY:
            assert key in ALLOWED_TRANSITIONS, f"ROLE_POLICY 多余条目: {key}"

    def test_allowed_roles_for_returns_correct_set(self):
        roles = allowed_roles_for(
            DraftStatus.PENDING_REVIEW.value, DraftStatus.REVIEWED.value
        )
        assert roles == {REVIEWER_ROLE}

        roles_arch = allowed_roles_for(
            DraftStatus.REVIEWED.value, DraftStatus.ARCHIVED.value
        )
        assert roles_arch == {REVIEWER_ROLE, QUALITY_ROLE}

        roles_rollback = allowed_roles_for(
            DraftStatus.REVIEWED.value, DraftStatus.PENDING_REVIEW.value
        )
        assert roles_rollback == {SYSTEM_ROLE}

    def test_allowed_roles_for_unknown_transition_returns_empty(self):
        roles = allowed_roles_for("ghost_state", DraftStatus.REVIEWED.value)
        assert roles == set()

    def test_policy_constants_have_expected_values(self):
        """REVIEWER_ROLE / QUALITY_ROLE / SYSTEM_ROLE 字段值稳定。"""
        assert REVIEWER_ROLE == "med_committee_reviewer"
        assert QUALITY_ROLE == "quality_officer"
        assert SYSTEM_ROLE == "system"

    def test_is_legal_transition_helper(self):
        from app.drafts.state_machine import is_legal_transition

        assert is_legal_transition(
            DraftStatus.PENDING_REVIEW.value, DraftStatus.REVIEWED.value
        ) is True
        assert is_legal_transition(
            DraftStatus.PENDING_REVIEW.value, DraftStatus.ARCHIVED.value
        ) is False
        assert is_legal_transition(
            DraftStatus.REVIEWED.value, DraftStatus.REJECTED.value
        ) is False