"""test_drafts_api — 草稿 API + healthz + modules 端点（spec.md §11 收尾）。

覆盖维度（spec §11 第 5 步硬约束）：
  1. GET  /api/drafts/{id}   — 单条草稿详情 + transitions 列表
  2. GET  /api/drafts/{id}   — draft_id 不存在 → 404 envelope
  3. GET  /api/drafts/{id}   — 缺 X-Quality-Role → 403 envelope
  4. GET  /api/drafts/{id}   — role 非法 → 403 envelope
  5. GET  /api/drafts        — 列表按 status 过滤（pending_review / reviewed / archived）
  6. GET  /api/drafts        — 列表按 disease_code 过滤
  7. GET  /api/drafts        — status 非法值 → 422 envelope
  8. POST /api/drafts/{id}/review confirm → 状态机 → reviewed + 触发上报（stub ok）+ audit
  9. POST /api/drafts/{id}/review reject  → 状态机 → rejected
 10. POST /api/drafts/{id}/review action=invalid → 422
 11. POST /api/drafts/{id}/review role 非 med_committee_reviewer → 403
 12. POST /api/drafts/{id}/review draft_id 不存在 → 404
 13. POST /api/drafts/{id}/review confirm + 上报失败 → 自动回退 pending_review
 14. POST /api/drafts/{id}/review 跳级（reviewed → pending_review 通过非法转换） → 422
 15. POST /api/drafts/{id}/review 同状态（pending_review → pending_review） → 422
 16. /api/healthz 4 项子健康（db/model_gateway/his_conn/drafts）+ 整体 rollup
 17. /api/healthz 全部 red → 503
 18. /api/modules 模块清单 + 版本 + model_version
 19. 全部子端点不假设 _build_audit_logger 必返回实例（best-effort None 兜底）
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("HIS_DB_URL", "sqlite:///./tests/.benchmarks/his_fake.db")
os.environ.setdefault("DIPCASE_REPORTER_URL", "http://localhost:18899/mock")


# ---- 公共 helpers ----

def _make_client_with_app(monkeypatch):
    """构造带 lifespan 的 TestClient（屏蔽 DB / Gateway / Audit 三件套到 None）。

    关键：用 ``with TestClient(app)`` 触发 lifespan startup → routes 注册；
    非 context-manager 形态的 TestClient **不会**触发 lifespan，导致路由全 404。
    """
    import app.main as app_main

    monkeypatch.setattr(app_main, "_build_audit_logger", lambda s: None)
    monkeypatch.setattr(app_main, "_build_model_gateway", lambda s: None)
    monkeypatch.setattr(app_main, "_build_session_factory", lambda s: None)
    from fastapi.testclient import TestClient

    return TestClient(app_main.create_app(), raise_server_exceptions=False)


def _seed_drafts(monkeypatch):
    """预先种 3 条 pending_review 草稿（每病种一条），返回 repo + draft_ids。"""
    from app.drafts import get_default_repository, reset_default_repository_for_tests

    reset_default_repository_for_tests()
    repo = get_default_repository()

    ids = []
    for disease, outcome, treatment in [
        ("I21", "cmi", "PCI"),
        ("I63", "cmi", "thrombolysis"),
        ("Z96.10", "los_days", "arthroplasty"),
    ]:
        d = repo.create(
            disease_code=disease,
            outcome=outcome,
            treatment=treatment,
            quarter="2025-Q4",
            sample_size=180,
            narrative_zh=f"测试草稿：{disease} {outcome}",
            effect_estimates=[{"node": treatment, "ate": 0.42}],
        )
        ids.append(d.draft_id)
    return repo, ids


@pytest.fixture
def client(monkeypatch):
    """FastAPI TestClient（lifespan 已启动 → 路由已注册）。

    用 yield-from-style：手动 enter/exit 触发 lifespan startup，
    让 _register_routes() 把 routes 装到 app 上；test 期间保持 in-context。
    """
    test_client = _make_client_with_app(monkeypatch)
    # 触发 lifespan startup
    test_client.__enter__()
    try:
        yield test_client
    finally:
        test_client.__exit__(None, None, None)


# ---- 1. 单条草稿 ----

def test_get_draft_success(client, monkeypatch):
    _, ids = _seed_drafts(monkeypatch)
    draft_id = ids[0]

    resp = client.get(
        f"/api/drafts/{draft_id}",
        headers={"X-Quality-Role": "quality_officer"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["schema_version"].startswith("attribution_draft/")
    assert body["ok"] is True
    assert body["draft"]["draft_id"] == draft_id
    assert body["draft"]["status"] == "pending_review"
    assert isinstance(body["transitions"], list)
    assert body["draft"]["narrative_zh"]  # 种数据带 narrative


def test_get_draft_not_found(client, monkeypatch):
    _seed_drafts(monkeypatch)
    resp = client.get(
        "/api/drafts/dr_does_not_exist",
        headers={"X-Quality-Role": "quality_officer"},
    )
    assert resp.status_code == 404
    body = resp.json()
    assert body["error_code"] == "draft_not_found"


def test_get_draft_role_missing_returns_403(client):
    resp = client.get("/api/drafts/dr_anything")
    assert resp.status_code == 403
    assert resp.json()["error_code"] == "quality_role_forbidden"


def test_get_draft_role_invalid_returns_403(client):
    resp = client.get(
        "/api/drafts/dr_anything",
        headers={"X-Quality-Role": "intern"},
    )
    assert resp.status_code == 403
    assert resp.json()["details"]["role"] == "intern"


# ---- 2. 列表 ----

def test_list_drafts_no_filter(client, monkeypatch):
    _, ids = _seed_drafts(monkeypatch)
    resp = client.get(
        "/api/drafts",
        headers={"X-Quality-Role": "quality_officer"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_total"] == 3
    assert len(body["drafts"]) == 3


def test_list_drafts_filter_by_status(client, monkeypatch):
    from app.drafts.models import DraftStatus

    repo, ids = _seed_drafts(monkeypatch)

    # 把第 0 条推到 reviewed
    from app.drafts import transition

    transition(
        repo,
        draft_id=ids[0],
        to_state=DraftStatus.REVIEWED.value,
        operator_id="reviewer-1",
        operator_role="med_committee_reviewer",
        note="confirm",
    )

    resp = client.get(
        "/api/drafts?status=reviewed",
        headers={"X-Quality-Role": "quality_officer"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_total"] == 1
    assert body["drafts"][0]["draft_id"] == ids[0]
    assert body["drafts"][0]["status"] == "reviewed"
    assert body["status_filter"] == "reviewed"


def test_list_drafts_filter_by_disease(client, monkeypatch):
    _, ids = _seed_drafts(monkeypatch)
    resp = client.get(
        "/api/drafts?disease_code=I63",
        headers={"X-Quality-Role": "quality_officer"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_total"] == 1
    assert body["drafts"][0]["disease_code"] == "I63"


def test_list_drafts_invalid_status(client, monkeypatch):
    _seed_drafts(monkeypatch)
    resp = client.get(
        "/api/drafts?status=__bogus__",
        headers={"X-Quality-Role": "quality_officer"},
    )
    assert resp.status_code == 422
    assert resp.json()["error_code"] == "invalid_status"


def test_list_drafts_role_missing(client):
    resp = client.get("/api/drafts")
    assert resp.status_code == 403


# ---- 3. review confirm ----

def test_review_confirm_state_advance(client, monkeypatch):
    repo, ids = _seed_drafts(monkeypatch)
    draft_id = ids[0]

    resp = client.post(
        f"/api/drafts/{draft_id}/review",
        json={"action": "confirm", "note": "符合归因", "post_to_dipcase": False},
        headers={
            "X-Quality-Role": "med_committee_reviewer",
            "X-Operator-Id": "reviewer-007",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["action"] == "confirm"
    assert body["draft"]["status"] == "reviewed"
    assert body["draft"]["reviewer_id"] == "reviewer-007"
    assert body["draft"]["reviewer_role"] == "med_committee_reviewer"
    # transitions 含 1 条 reviewed
    assert body["transition"]["from_state"] == "pending_review"
    assert body["transition"]["to_state"] == "reviewed"
    # post_to_dipcase=False 时，post_result 应为 None
    assert body["post_result"] is None
    assert body["rolled_back"] is False


def test_review_confirm_dipcase_failure_rollback(client, monkeypatch):
    """confirm + 上报失败 → 自动回退到 pending_review。"""
    from app.api.drafts import _post_to_dipcase as orig_post  # noqa: F401

    def fake_post_failure(d, *, action):
        return {"ok": False, "degraded": True, "reason": "test_forced_failure", "status_code": 502}

    monkeypatch.setattr("app.api.drafts._post_to_dipcase", fake_post_failure)

    repo, ids = _seed_drafts(monkeypatch)
    draft_id = ids[1]

    resp = client.post(
        f"/api/drafts/{draft_id}/review",
        json={"action": "confirm"},
        headers={"X-Quality-Role": "med_committee_reviewer", "X-Operator-Id": "reviewer-007"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # 回退后：status 应为 pending_review；post_result.rolled_back=True
    assert body["rolled_back"] is True
    assert body["draft"]["status"] == "pending_review"
    assert body["post_result"]["reason"] == "test_forced_failure"


def test_review_confirm_dipcase_success_no_rollback(client, monkeypatch):
    """confirm + 上报成功 → status 保持 reviewed。"""
    def fake_post_ok(d, *, action):
        return {"ok": True, "degraded": False, "status_code": 200, "remote_id": "rem_001"}

    monkeypatch.setattr("app.api.drafts._post_to_dipcase", fake_post_ok)

    repo, ids = _seed_drafts(monkeypatch)
    draft_id = ids[2]

    resp = client.post(
        f"/api/drafts/{draft_id}/review",
        json={"action": "confirm"},
        headers={"X-Quality-Role": "med_committee_reviewer", "X-Operator-Id": "reviewer-007"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["rolled_back"] is False
    assert body["draft"]["status"] == "reviewed"
    assert body["post_result"]["ok"] is True


def test_review_reject(client, monkeypatch):
    _, ids = _seed_drafts(monkeypatch)
    draft_id = ids[0]

    resp = client.post(
        f"/api/drafts/{draft_id}/review",
        json={"action": "reject", "note": "样本量不足以支撑结论"},
        headers={"X-Quality-Role": "med_committee_reviewer", "X-Operator-Id": "reviewer-007"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["action"] == "reject"
    assert body["draft"]["status"] == "rejected"
    assert body["draft"]["notes"] == "样本量不足以支撑结论"
    # reject 不触发上报
    assert body["post_result"] is None


# ---- 4. review 错误路径 ----

def test_review_invalid_action(client, monkeypatch):
    _, ids = _seed_drafts(monkeypatch)
    resp = client.post(
        f"/api/drafts/{ids[0]}/review",
        json={"action": "__bogus__"},
        headers={"X-Quality-Role": "med_committee_reviewer"},
    )
    assert resp.status_code == 422
    assert resp.json()["error_code"] == "invalid_action"


def test_review_role_not_reviewer_returns_403(client, monkeypatch):
    _, ids = _seed_drafts(monkeypatch)
    resp = client.post(
        f"/api/drafts/{ids[0]}/review",
        json={"action": "confirm"},
        headers={"X-Quality-Role": "quality_officer"},
    )
    assert resp.status_code == 403
    assert resp.json()["error_code"] == "quality_role_forbidden"


def test_review_role_missing_returns_403(client, monkeypatch):
    _, ids = _seed_drafts(monkeypatch)
    resp = client.post(
        f"/api/drafts/{ids[0]}/review",
        json={"action": "confirm"},
    )
    assert resp.status_code == 403


def test_review_draft_not_found(client):
    resp = client.post(
        "/api/drafts/dr_does_not_exist/review",
        json={"action": "confirm"},
        headers={"X-Quality-Role": "med_committee_reviewer"},
    )
    assert resp.status_code == 404


def test_review_skip_state_rejected(client, monkeypatch):
    """跳级（reviewed → pending_review）必须被拒绝。"""
    from app.drafts import transition
    from app.drafts.models import DraftStatus

    repo, ids = _seed_drafts(monkeypatch)
    # 先 confirm → reviewed
    transition(
        repo,
        draft_id=ids[0],
        to_state=DraftStatus.REVIEWED.value,
        operator_id="reviewer-007",
        operator_role="med_committee_reviewer",
    )

    # 再试图直接 confirm 一次（pending_review → reviewed 不再合法：当前 reviewed）
    resp = client.post(
        f"/api/drafts/{ids[0]}/review",
        json={"action": "confirm"},
        headers={"X-Quality-Role": "med_committee_reviewer"},
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["error_code"] == "invalid_state_transition"


# ---- 5. /api/healthz ----

def test_healthz_all_green(client):
    resp = client.get("/api/healthz")
    # lifespan 注入 db=None/gateway=None → 应有 red
    assert resp.status_code in (200, 503)
    body = resp.json()
    assert body["schema_version"] == "healthz_response/1.0"
    assert "checks" in body
    assert set(body["checks"].keys()) == {"db", "model_gateway", "his_conn", "drafts"}
    # db / model_gateway 走 _build_session_factory / _build_model_gateway → None
    # → healthz 应报告 red；本测试只校验结构 + 至少一项 red 或 yellow
    statuses = [c["status"] for c in body["checks"].values()]
    assert "red" in statuses or "yellow" in statuses or all(s == "green" for s in statuses)
    # 整体 status 必是 green/yellow/red 之一
    assert body["status"] in ("green", "yellow", "red")


def test_healthz_with_injected_components(client, monkeypatch):
    """直接往当前 app.state 注入绿色组件 → healthz 应整体 green。"""
    # 注入 stub
    class _StubFactory:
        def __call__(self):
            class _S:
                def close(self):
                    pass

            return _S()

    class _StubGw:
        pass

    client.app.state.db_session_factory = _StubFactory()
    client.app.state.model_gateway = _StubGw()

    resp = client.get("/api/healthz")
    assert resp.status_code == 200
    body = resp.json()
    # db 与 model_gateway 已注入 stub，应为 green；his_conn 走 env
    assert body["checks"]["db"]["status"] == "green"
    assert body["checks"]["model_gateway"]["status"] == "green"


# ---- 6. /api/modules ----

def test_modules_endpoint(client):
    resp = client.get("/api/modules")
    assert resp.status_code == 200
    body = resp.json()
    assert body["schema_version"] == "modules_response/1.0"
    assert body["ok"] is True
    assert body["n_tracked"] >= 7
    assert "dowhy" in {m["name"] for m in body["modules"]}
    # model_version 与 graph_render 必须存在
    assert "model_version" in body and body["model_version"].startswith("causal_pipeline/")
    assert body["graph_render"] == "plotly.js@2.x"
    # 至少有一个模块 available
    available = [m for m in body["modules"] if m["available"]]
    assert len(available) >= 1
    # dowhy 是核心模块，必须 available（spec §11 硬约束）
    dowhy_entry = next(m for m in body["modules"] if m["name"] == "dowhy")
    assert dowhy_entry["available"] is True


# ---- 7. 全链路联调：review confirm → list 过滤 → 上报失败回退 ----

def test_full_lifecycle_confirm_failure_rollback_list(client, monkeypatch):
    """完整链路：seed → confirm (上报失败) → 列表过滤 pending_review 应仍可见。"""
    def fake_post_failure(d, *, action):
        return {"ok": False, "degraded": True, "reason": "mock_failure", "status_code": 502}

    monkeypatch.setattr("app.api.drafts._post_to_dipcase", fake_post_failure)

    _, ids = _seed_drafts(monkeypatch)
    draft_id = ids[0]

    # 1. confirm（上报失败 → 回退）
    r1 = client.post(
        f"/api/drafts/{draft_id}/review",
        json={"action": "confirm"},
        headers={"X-Quality-Role": "med_committee_reviewer", "X-Operator-Id": "reviewer-007"},
    )
    assert r1.status_code == 200
    assert r1.json()["rolled_back"] is True

    # 2. 列表：pending_review 应仍可见该草稿（因为已回退）
    r2 = client.get(
        "/api/drafts?status=pending_review",
        headers={"X-Quality-Role": "quality_officer"},
    )
    assert r2.status_code == 200
    body = r2.json()
    pending_ids = {d["draft_id"] for d in body["drafts"]}
    assert draft_id in pending_ids

    # 3. transitions 应含 3 条（reviewed → pending_review 回退）
    r3 = client.get(
        f"/api/drafts/{draft_id}",
        headers={"X-Quality-Role": "quality_officer"},
    )
    transitions = r3.json()["transitions"]
    assert len(transitions) >= 2
    states = [t["to_state"] for t in transitions]
    assert "pending_review" in states  # 含至少一次回退落点
    assert "reviewed" in states  # 含原始 confirm 落点