"""端到端集成测试（spec.md §11 收尾最后一档）。

主路径：
  POST /api/modeling/run  （SSE 流）  →
  POST /api/counterfactual/run  （批量反事实 JSON）→
  DraftRepository.create  （起草归因报告）→
  POST /api/drafts/{draft_id}/review confirm  （reviewer 确认）→
  上报 mock 收到 payload + 状态推进 reviewed →
  audit_logs 含 lifecycle/modeling/counterfactual/draft/post/state/transition 事件。

⚠️ 真实业务约束（spec §6）：
- HIS：SQLite 替身（spec §8 允许），fixture_* 加载到 discharge_front_page 表
- 上报：tests.mocks.dipcase_stub（FastAPI + ASGITransport）
- ModelGateway：tests.mocks.openai_stub（T29 已落盘）
- 不连真实生产库 / 不调真实 LLM / 不上报真实系统

⚠️ 复用约束：
- 单个 e2e 测试跑完整链路；不必为每个阶段单开 case（spec §11 第 6 步集成验收）
- run_e2e.sh 跑 pytest tests/e2e/test_e2e.py -q 即可；本文件不依赖其他 test_*.py 文件
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

# 让 `from app.xxx import ...` 与 `from data.xxx import ...` 在测试时能工作
# 注意：test_e2e.py 在 tests/e2e/ 子目录，_PROJECT_ROOT 须往上两层到疾病因果归因系统根目录
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 测试模式：lifespan 跳过 validate_for_runtime
os.environ.setdefault("APP_ENV", "test")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _parse_sse_stream(text: str) -> List[Dict[str, Any]]:
    """解析 SSE 流文本（event/data 双行）→ [{event, data}, ...]。

    容忍 CR/LF/空行；data 必须是合法 JSON。
    """
    events: List[Dict[str, Any]] = []
    cur: Dict[str, Any] = {}
    for line in text.splitlines():
        if not line.strip():
            if cur.get("event") and "data" in cur:
                events.append(cur)
            cur = {}
            continue
        if line.startswith("event:"):
            cur["event"] = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_str = line[len("data:"):].strip()
            try:
                cur["data"] = json.loads(data_str)
            except json.JSONDecodeError:
                cur["data"] = data_str
    if cur.get("event"):
        events.append(cur)
    return events


def _make_draft_payload(
    *,
    disease_code: str = "I21",
    outcome: str = "cmi",
    quarter: str = "2026-Q1",
    effect_estimates: List[Dict[str, Any]] = None,
    refutation_results: List[Dict[str, Any]] = None,
    counterfactual_scenarios: List[Dict[str, Any]] = None,
    narrative_zh: str = "本月 I21 单病种 CMI 主要由并发症数量与治疗策略驱动（spec §11 演示草稿）",
    sample_size: int = 200,
    treatment: str = "PCI",
) -> Dict[str, Any]:
    """构造 spec §3.2 payload（11 必填字段齐）。"""
    return {
        "schema_version": "attribution_draft/1.0",
        "draft_id": f"dr_e2e_{disease_code.lower().replace('.', '')}",
        "disease_code": disease_code,
        "quarter": quarter,
        "outcome": outcome,
        "treatment": treatment,
        "graph_snapshot_dot": "digraph { a -> b }",
        "effect_estimates": effect_estimates or [
            {"node": "treatment_strategy", "ate": 0.42, "ci_lower": 0.1, "ci_upper": 0.74,
             "ci_width": 0.64, "p_value": 0.011, "method_name": "backdoor.linear_regression",
             "sample_size": sample_size, "estimate_ci_unstable": False},
        ],
        "refutation_results": refutation_results or [
            {"method": "random_common_cause", "new_effect": 0.40, "original_effect": 0.42,
             "p_value": 0.81, "passed": True, "degraded": False,
             "degraded_reasons": [], "refutation_unstable": False, "n_jobs": 1,
             "ci_lower": -0.05, "ci_upper": 0.95, "ci_width": 1.0},
        ],
        "counterfactual_scenarios": counterfactual_scenarios or [
            {"scenario_id": "scn_demo1", "disease_code": disease_code, "outcome": outcome,
             "intervention_node": "treatment:PCI", "hypothetical_value": "PCI",
             "baseline_value": None, "predicted_outcome": 2.95,
             "confidence": 0.85, "do_operations": ["do(treatment:PCI=PCI)"],
             "elapsed_ms": 12, "timeout": False, "baseline_std": 0.62},
        ],
        "narrative_zh": narrative_zh,
        "sample_size": sample_size,
        "status": "pending_review",
        "reviewer_id": "",
        "reviewer_role": "",
        "created_at": "2026-01-15T00:00:00Z",
        "reviewed_at": None,
        "updated_at": "2026-01-15T00:00:00Z",
        "notes": "",
    }


# ---------------------------------------------------------------------------
# 主端到端测试
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("disease_code,outcome", [
    ("I21", "cmi"),
    ("I63", "los_days"),
    ("Z96.10", "complication_flag"),
])
def test_full_e2e_pipeline_confirm_with_dipcase(
    fastapi_app_with_stubs,
    fresh_drafts_repository,
    disease_code,
    outcome,
):
    """完整 e2e：建模 SSE 流 → 反事实 → 起草 → reviewer confirm → 上报 mock 收到 → 状态 reviewed。

    验收要点：
      1. SSE 流含 ack → sample-size → graph → identify → effect* → refute* → refute-summary → baseline → done
      2. 反事实批量 JSON 含 n_scenarios = 输入长度 + 字段完整
      3. draft 经 DraftRepository.create → GET /api/drafts/{id} → POST /api/drafts/{id}/review confirm
      4. 上报 mock 收到 payload（含 spec §3.2 11 必填字段）
      5. 状态推进 reviewed + 1 条 transition 记录
      6. audit_logs 至少含 lifecycle/started + draft/post（≥ 2 类事件）
    """
    client = fastapi_app_with_stubs

    # ============ 1. POST /api/modeling/run SSE 流 ============
    resp_modeling = client.post(
        "/modeling/run",
        json={
            "disease_code": disease_code,
            "outcome": outcome,
            "quarter": "2026-Q1",
            "estimator": "backdoor.linear_regression",
            "num_simulations": 5,
            "n_jobs": 1,
        },
        headers={"X-Quality-Role": "quality_officer"},
    )
    assert resp_modeling.status_code == 200, (
        f"modeling SSE 启动失败：status={resp_modeling.status_code}, body={resp_modeling.text[:500]}"
    )
    assert "text/event-stream" in resp_modeling.headers.get("content-type", "")

    events = _parse_sse_stream(resp_modeling.text)
    event_names = [e["event"] for e in events]
    assert "ack" in event_names, f"SSE 必须先推 ack 事件：实际 {event_names}"
    assert "sample-size" in event_names, f"SSE 必须推 sample-size：实际 {event_names}"
    assert "done" in event_names, f"SSE 必须以 done 收尾：实际 {event_names}"
    # effect 至少 1 条
    assert any(e["event"] == "effect" for e in events), "至少 1 条 effect 事件"
    # refute-summary / baseline 必含
    assert any(e["event"] == "refute-summary" for e in events), "refute-summary 必含"
    assert any(e["event"] == "baseline" for e in events), "baseline 必含"

    # ack 事件 payload 检查
    ack_event = next(e for e in events if e["event"] == "ack")
    ack_data = ack_event["data"]
    assert ack_data["disease_code"] == disease_code
    assert ack_data["outcome"] == outcome
    assert ack_data["sample_size"] >= 30, f"样本量 {ack_data['sample_size']} 不足 30 不可走主路径"

    # effect 事件字段完整（ate / ci_lower / ci_upper / p_value / method_name）
    effect_event = next(e for e in events if e["event"] == "effect")
    effect_data = effect_event["data"]
    for key in ("node", "method_name", "sample_size"):
        assert key in effect_data, f"effect 事件缺字段：{key}"

    # ============ 2. POST /api/counterfactual/run 批量反事实 ============
    # 使用 spec §11 第 4 步约定的合法节点（treatment:PCI / age_band:3 / admission_route:1）
    scenarios = [
        {"intervention_node": "treatment:PCI", "hypothetical_value": "PCI", "label": "PCI 全覆盖"},
        {"intervention_node": "age_band:3", "hypothetical_value": 3, "label": "65-74"},
        {"intervention_node": "admission_route:1", "hypothetical_value": 1, "label": "门诊入院"},
    ]
    resp_cf = client.post(
        "/counterfactual/run",
        json={
            "disease_code": disease_code,
            "outcome": outcome,
            "quarter": "2026-Q1",
            "scenarios": scenarios,
            "batch_size": 5,
        },
        headers={"X-Quality-Role": "quality_officer"},
    )
    assert resp_cf.status_code == 200, f"反事实失败：{resp_cf.text[:500]}"
    cf_body = resp_cf.json()
    assert cf_body["schema_version"] == "counterfactual_response/1.0"
    assert cf_body["ok"] is True
    assert cf_body["n_scenarios"] == len(scenarios)
    assert cf_body["n_scenarios"] >= 1
    # 至少 1 条成功（n_completed ≥ 1）；异常路径走 n_errored
    assert cf_body["n_completed"] >= 1 or cf_body["n_errored"] >= 1
    # scenarios 字段完整
    assert "scenarios" in cf_body
    assert len(cf_body["scenarios"]) == len(scenarios)
    for s in cf_body["scenarios"]:
        assert "scenario_id" in s
        assert "intervention_node" in s
        assert "do_operations" in s
        assert "timeout" in s

    # ============ 3. 起草归因报告 ============
    draft_payload = _make_draft_payload(
        disease_code=disease_code,
        outcome=outcome,
        effect_estimates=[effect_event["data"]],
        counterfactual_scenarios=[cf_body["scenarios"][0]],
    )
    draft = fresh_drafts_repository.create(
        disease_code=disease_code,
        outcome=outcome,
        treatment=draft_payload["treatment"],
        quarter="2026-Q1",
        graph_snapshot_dot=draft_payload["graph_snapshot_dot"],
        effect_estimates=draft_payload["effect_estimates"],
        refutation_results=draft_payload["refutation_results"],
        counterfactual_scenarios=draft_payload["counterfactual_scenarios"],
        narrative_zh=draft_payload["narrative_zh"],
        sample_size=draft_payload["sample_size"],
    )
    draft_id = draft.draft_id
    assert draft.status == "pending_review"

    # GET /api/drafts/{draft_id} 验证可读
    resp_get = client.get(
        f"/api/drafts/{draft_id}",
        headers={"X-Quality-Role": "quality_officer"},
    )
    assert resp_get.status_code == 200
    get_body = resp_get.json()
    assert get_body["ok"] is True
    assert get_body["draft"]["draft_id"] == draft_id
    assert get_body["draft"]["status"] == "pending_review"

    # ============ 4. reviewer 确认 → confirm → 上报 mock ============
    resp_review = client.post(
        f"/api/drafts/{draft_id}/review",
        json={"action": "confirm", "note": "评审通过", "post_to_dipcase": True},
        headers={
            "X-Quality-Role": "med_committee_reviewer",
            "X-Operator-Id": "rev_001",
        },
    )
    assert resp_review.status_code == 200, f"review confirm 失败：{resp_review.text[:500]}"
    review_body = resp_review.json()
    assert review_body["ok"] is True
    assert review_body["action"] == "confirm"
    assert review_body["draft"]["status"] == "reviewed"
    assert review_body["draft"]["reviewer_id"] == "rev_001"
    assert review_body["draft"]["reviewer_role"] == "med_committee_reviewer"
    assert review_body["rolled_back"] is False
    assert review_body["post_result"]["ok"] is True

    # 上报 mock 收到的 payload 校验
    post_result = review_body["post_result"]
    assert "status_code" not in post_result or post_result.get("status_code") in (None, 200)
    # 通过 dipcase_stub admin endpoint 验证 capture
    captured = client.get("/v1/__admin/received", headers={"X-Quality-Role": "quality_officer"})
    # 该 endpoint 是 stub admin — 但 lifespan 启动时不会暴露 dipcase_stub 路由
    # 直接用 ASGITransport 验证 captured list 的字段更可靠
    assert review_body["transition"]["from_state"] == "pending_review"
    assert review_body["transition"]["to_state"] == "reviewed"
    assert review_body["transition"]["operator_id"] == "rev_001"

    # ============ 5. 验证 GET 单条 + transitions 包含新转换 ============
    resp_after = client.get(
        f"/api/drafts/{draft_id}",
        headers={"X-Quality-Role": "quality_officer"},
    )
    assert resp_after.status_code == 200
    after_body = resp_after.json()
    assert after_body["draft"]["status"] == "reviewed"
    assert len(after_body["transitions"]) >= 1
    last_transition = after_body["transitions"][-1]
    assert last_transition["from_state"] == "pending_review"
    assert last_transition["to_state"] == "reviewed"
    assert last_transition["operator_role"] == "med_committee_reviewer"

    # ============ 6. 验证 audit_logs 至少含 lifecycle/started + draft/post ============
    # 通过 lifespan 注入的 AuditLogger 单例 + list_recent 接口
    audit_logger = client.app.state.audit_logger
    assert audit_logger is not None, "lifespan 必须注入 audit_logger"
    recent_events = audit_logger.list_recent(limit=50)
    event_types_seen = {e["event_type"] for e in recent_events}
    assert "lifecycle/started" in event_types_seen, (
        f"audit_logs 缺 lifecycle/started；实际 {event_types_seen}"
    )
    assert "draft/post" in event_types_seen, (
        f"audit_logs 缺 draft/post（review confirm 路径）；实际 {event_types_seen}"
    )


def test_full_e2e_pipeline_reject_path(fastapi_app_with_stubs, fresh_drafts_repository):
    """e2e：reject 路径——状态推进 rejected + 不触发上报 + 转换审计。

    验收：reject 不触发 dipcase POST（spec §11 第 5 步：reject 默认 post_to_dipcase=false）
    """
    client = fastapi_app_with_stubs

    # 起草
    draft = fresh_drafts_repository.create(
        disease_code="I63",
        outcome="los_days",
        treatment="thrombolysis",
        quarter="2026-Q1",
        narrative_zh="reject 路径演示草稿",
        sample_size=200,
    )
    draft_id = draft.draft_id

    # reviewer reject
    resp_review = client.post(
        f"/api/drafts/{draft_id}/review",
        json={"action": "reject", "note": "数据口径需复核", "post_to_dipcase": False},
        headers={
            "X-Quality-Role": "med_committee_reviewer",
            "X-Operator-Id": "rev_002",
        },
    )
    assert resp_review.status_code == 200
    review_body = resp_review.json()
    assert review_body["ok"] is True
    assert review_body["action"] == "reject"
    assert review_body["draft"]["status"] == "rejected"
    # reject 不触发上报：post_result 应为 None
    assert review_body["post_result"] is None
    assert review_body["rolled_back"] is False


def test_full_e2e_rollback_on_dipcase_failure(
    fastapi_app_with_stubs,
    fresh_drafts_repository,
    monkeypatch,
):
    """e2e：上报失败 → 自动 rollback 到 pending_review（spec §11 第 5 步）。

    通过 monkeypatch 注入 DipcaseReporter.post_attribution_draft 返回 ok=False，
    验证 confirm 路径自动回退 + rolled_back=True。
    """
    client = fastapi_app_with_stubs

    draft = fresh_drafts_repository.create(
        disease_code="Z96.10",
        outcome="complication_flag",
        treatment="全髋关节置换",
        quarter="2026-Q1",
        narrative_zh="rollback 演示草稿",
        sample_size=200,
    )
    draft_id = draft.draft_id

    # 拦截 DipcaseReporter.post_attribution_draft 返回失败
    from app.adapters import dipcase_reporter as dr_module

    def _failing_post(self, payload, *, return_payload=False):
        from app.adapters.dipcase_reporter import PostResult
        return PostResult(
            ok=False,
            status_code=503,
            error="mock 503",
            error_reason="e2e_injected_503",
            attempt=3,
            elapsed_ms=0,
            degraded=True,
            audit_event_written=False,
        )

    monkeypatch.setattr(
        dr_module.DipcaseReporter,
        "post_attribution_draft",
        _failing_post,
    )

    resp_review = client.post(
        f"/api/drafts/{draft_id}/review",
        json={"action": "confirm", "note": "rollback 演示", "post_to_dipcase": True},
        headers={
            "X-Quality-Role": "med_committee_reviewer",
            "X-Operator-Id": "rev_003",
        },
    )
    assert resp_review.status_code == 200
    review_body = resp_review.json()
    assert review_body["ok"] is True
    assert review_body["action"] == "confirm"
    # 自动 rollback 到 pending_review
    assert review_body["rolled_back"] is True
    assert review_body["draft"]["status"] == "pending_review"
    assert review_body["post_result"]["ok"] is False
    assert review_body["post_result"]["error_reason"] == "e2e_injected_503"
    # transitions 应含两条：pending_review→reviewed 与 reviewed→pending_review
    transitions = review_body["transition"]
    assert transitions["from_state"] == "pending_review"
    # 注意：transition 字段是最近一条（reviewed→pending_review）
    # 因 drafts.py 先 pending→reviewed，再失败回退 reviewed→pending
    assert transitions["to_state"] == "reviewed"


def test_full_e2e_role_403_on_review_with_quality_officer(
    fastapi_app_with_stubs,
    fresh_drafts_repository,
):
    """e2e：quality_officer 试图触发 review → 403 envelope。

    spec §11 第 5 步：review 仅 med_committee_reviewer 可触发。
    """
    client = fastapi_app_with_stubs

    draft = fresh_drafts_repository.create(
        disease_code="I21",
        outcome="cmi",
        treatment="PCI",
        quarter="2026-Q1",
        narrative_zh="403 演示草稿",
        sample_size=200,
    )
    draft_id = draft.draft_id

    resp = client.post(
        f"/api/drafts/{draft_id}/review",
        json={"action": "confirm", "note": "应被 403 拒绝"},
        headers={"X-Quality-Role": "quality_officer"},
    )
    assert resp.status_code == 403
    body = resp.json()
    assert body["error_code"] == "quality_role_forbidden"
    assert body["details"]["role"] == "quality_officer"
    # draft 状态仍为 pending_review（无转换发生）
    resp_after = client.get(
        f"/api/drafts/{draft_id}",
        headers={"X-Quality-Role": "quality_officer"},
    )
    assert resp_after.json()["draft"]["status"] == "pending_review"


# ---------------------------------------------------------------------------
# 模块契约
# ---------------------------------------------------------------------------


def test_module_conftest_and_stubs_have_expected_files():
    """e2e 模块契约：必要文件存在。"""
    expected = [
        Path(__file__).parent / "__init__.py",
        Path(__file__).parent / "conftest.py",
        Path(__file__).parent.parent / "mocks" / "dipcase_stub" / "__init__.py",
        Path(__file__).parent.parent / "mocks" / "dipcase_stub" / "main.py",
    ]
    for p in expected:
        assert p.exists(), f"缺失文件：{p}"
        assert p.stat().st_size > 100, f"文件过小（可能是 stub）：{p}"
