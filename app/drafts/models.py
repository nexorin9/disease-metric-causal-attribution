"""Draft 数据模型 + in-memory 存储（spec.md §3.2 + §11 收尾交付）。

字段集（spec §3.2 严格对齐）：
    draft_id              草稿主键（UUID/short hash）
    disease_code          I21 / I63 / Z96.10
    quarter               2025-Q1（仅日志用）
    outcome               cmi / los_days / readmit_flag / complication_flag
    treatment             treatment_strategy 值
    graph_snapshot_dot    因果图 DOT 字符串（面板复现用）
    effect_estimates      List[EffectEstimate]（来自 estimate_effect）
    refutation_results    List[RefutationResult]
    counterfactual_scenarios  List[CounterfactualScenario]
    narrative_zh          中文叙述草稿（业务可读；ModelGateway 生成）
    sample_size           建模样本量
    status                pending_review / reviewed / rejected / archived
    reviewer_id           评审人 id（仅 reviewed/rejected 时非空）
    reviewer_role         评审人角色（仅 reviewed/rejected 时非空）
    created_at / reviewed_at / updated_at

约束（spec §6 + §8）：
- **PII 字段严禁落库**（patient_name / id_card / phone / address）—— schema 不定义。
- 状态转换必须经 state_machine.transition()；直接 set status 视为非法（构造期锁定）。
- 状态变更记录到 transitions（in-memory 阶段为 list；T25 升级为 SQLAlchemy）。

存储形态（T21）：**in-memory 进程内单例**（DraftRepository）；
测试可用 DraftRepository 构造独立实例做隔离。
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

# spec §3.2 schema_version：草稿 JSON 序列化契约
DRAFT_SCHEMA_VERSION = "attribution_draft/1.0"


class DraftStatus(str, Enum):
    """草稿状态枚举（spec §3.2 + §11 收尾）。

    合法路径：pending_review → reviewed / rejected → archived。
    任何非法跳级由 state_machine.transition() 拒绝。
    """

    PENDING_REVIEW = "pending_review"
    REVIEWED = "reviewed"
    REJECTED = "rejected"
    ARCHIVED = "archived"


# spec §11 第 5 步：状态变更时把变更事件写入 audit_logs；本层只构造 envelope，
# 写日志由 API 层 / audit_logger 负责（best-effort）。
@dataclass
class StateTransitionRecord:
    """状态变更审计记录（in-memory；T25 升级为 SQLAlchemy）。"""

    transition_id: str
    draft_id: str
    from_state: str
    to_state: str
    operator_id: str
    operator_role: str
    note: str
    ts: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "draft_id": self.draft_id,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "operator_id": self.operator_id,
            "operator_role": self.operator_role,
            "note": self.note,
            "ts": self.ts,
        }


@dataclass
class Draft:
    """归因报告草稿（spec §3.2）。"""

    draft_id: str
    disease_code: str
    outcome: str
    treatment: str
    quarter: str = ""
    graph_snapshot_dot: str = ""
    effect_estimates: List[Dict[str, Any]] = field(default_factory=list)
    refutation_results: List[Dict[str, Any]] = field(default_factory=list)
    counterfactual_scenarios: List[Dict[str, Any]] = field(default_factory=list)
    narrative_zh: str = ""
    sample_size: int = 0
    status: str = DraftStatus.PENDING_REVIEW.value
    reviewer_id: str = ""
    reviewer_role: str = ""
    created_at: str = ""
    reviewed_at: Optional[str] = None
    updated_at: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """序列化为可 JSON 化的字典（spec §3.2 payload 契约）。"""
        return {
            "schema_version": DRAFT_SCHEMA_VERSION,
            "draft_id": self.draft_id,
            "disease_code": self.disease_code,
            "quarter": self.quarter,
            "outcome": self.outcome,
            "treatment": self.treatment,
            "graph_snapshot_dot": self.graph_snapshot_dot,
            "effect_estimates": list(self.effect_estimates or []),
            "refutation_results": list(self.refutation_results or []),
            "counterfactual_scenarios": list(self.counterfactual_scenarios or []),
            "narrative_zh": self.narrative_zh,
            "sample_size": int(self.sample_size or 0),
            "status": self.status,
            "reviewer_id": self.reviewer_id,
            "reviewer_role": self.reviewer_role,
            "created_at": self.created_at,
            "reviewed_at": self.reviewed_at,
            "updated_at": self.updated_at,
            "notes": self.notes,
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _gen_draft_id() -> str:
    return f"drf_{uuid.uuid4().hex[:16]}"


class DraftRepository:
    """in-memory 草稿存储（线程安全 + 测试可独立实例化）。

    设计要点：
      - 构造可指定 in_memory=True / 留单例；测试用 DraftRepository() 自管。
      - create / get / list / save / append_transition 全部带锁，避免并发脏写。
      - list(status=[]) 按状态过滤；status=None → 全部。
    """

    def __init__(self) -> None:
        self._drafts: Dict[str, Draft] = {}
        self._transitions: Dict[str, List[StateTransitionRecord]] = {}
        self._lock = threading.RLock()

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
        """新建 pending_review 草稿。"""
        with self._lock:
            now = _now_iso()
            d = Draft(
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
            self._drafts[d.draft_id] = d
            self._transitions[d.draft_id] = []
            return d

    def save(self, draft: Draft) -> None:
        """更新已存在草稿（仅 in-place 修改字段；不增 transition 记录）。"""
        with self._lock:
            if draft.draft_id not in self._drafts:
                raise KeyError(f"draft_id={draft.draft_id} 不存在；不可 save")
            draft.updated_at = _now_iso()
            self._drafts[draft.draft_id] = draft

    def append_transition(self, record: StateTransitionRecord) -> None:
        """追加一条状态变更记录。"""
        with self._lock:
            if record.draft_id not in self._drafts:
                raise KeyError(f"draft_id={record.draft_id} 不存在；不可追加 transition")
            self._transitions.setdefault(record.draft_id, []).append(record)

    # ---- 读取 ----

    def get(self, draft_id: str) -> Optional[Draft]:
        with self._lock:
            return self._drafts.get(draft_id)

    def list(
        self,
        *,
        status: Optional[str] = None,
        disease_code: Optional[str] = None,
        limit: int = 50,
    ) -> List[Draft]:
        """按状态过滤 + 病种过滤 + 上限截断。"""
        with self._lock:
            out: List[Draft] = []
            for d in self._drafts.values():
                if status is not None and d.status != status:
                    continue
                if disease_code is not None and d.disease_code != disease_code:
                    continue
                out.append(d)
            # 最新优先
            out.sort(key=lambda x: x.created_at, reverse=True)
            return out[: max(1, int(limit or 1))]

    def transitions(self, draft_id: str) -> List[StateTransitionRecord]:
        with self._lock:
            return list(self._transitions.get(draft_id, []))

    def has_draft(self, draft_id: str) -> bool:
        with self._lock:
            return draft_id in self._drafts

    def clear(self) -> None:
        """测试用：清空所有数据。"""
        with self._lock:
            self._drafts.clear()
            self._transitions.clear()


# ---- 进程内默认单例（lifespan 复用；测试可另建 DraftRepository()） ----

_default_repo: Optional[DraftRepository] = None
_default_repo_lock = threading.Lock()


def get_default_repository() -> DraftRepository:
    """取得进程内默认 DraftRepository 单例（懒加载）。"""
    global _default_repo
    with _default_repo_lock:
        if _default_repo is None:
            _default_repo = DraftRepository()
        return _default_repo


def reset_default_repository_for_tests() -> None:
    """测试用：清掉默认单例。"""
    global _default_repo
    with _default_repo_lock:
        _default_repo = None


__all__ = [
    "DRAFT_SCHEMA_VERSION",
    "Draft",
    "DraftRepository",
    "DraftStatus",
    "StateTransitionRecord",
    "get_default_repository",
    "reset_default_repository_for_tests",
]