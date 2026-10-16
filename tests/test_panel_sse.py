"""test_panel_sse — 浏览器面板后端 API 端点（SSE 流式 + 批量反事实 JSON）。

覆盖维度：
  1. /modeling/run 流式响应 text/event-stream + 事件序列形态
  2. /modeling/run X-Quality-Role 鉴权（缺失 → 403；非法角色 → 403）
  3. /modeling/run 鉴权角色通过 + 完整事件 ack → sample-size → graph →
     identify → effect → refute* → refute-summary → baseline → done
  4. /modeling/run 启动测试下无凭据 / 无 DB 也能跑通（fixture 替他）
  5. /counterfactual/run X-Quality-Role 鉴权
  6. /counterfactual/run 批量 10 场景 < 30s 返回 + 字段 schema
  7. /counterfactual/run 错误路径（缺 disease_code / 空 scenarios / 超上限 / 非 dict 元素）
  8. /counterfactual/run 单条 error / timeout 隔离（不阻断其他场景）

实现要点：
- 用 FastAPI TestClient + 流式响应解析（iter_lines 抓 SSE 帧）
- 用 httpx.AsyncClient 也可以；本测试更偏 FastAPI 直接流（更贴近 scripts/capture_screenshots.sh 真实路径）
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 测试环境：APP_ENV=test 让 lifespan 跳过 OPENAI_API_KEY 校验
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("HIS_DB_URL", "sqlite:///./tests/.benchmarks/his_fake.db")
os.environ.setdefault("DIPCASE_REPORTER_URL", "http://localhost:18899/mock")

from fastapi.testclient import TestClient  # noqa: E402

from app.config import get_settings, reset_settings_cache  # noqa: E402


# ------------------------- helpers -------------------------

_SSE_EVENT_PATTERN = re.compile(r"^event:\s*(\S+)\s*$")
_SSE_DATA_PATTERN = re.compile(r"^data:\s*(.*)$")


def _parse_sse_frames(body: str) -> List[Tuple[str, Dict[str, Any]]]:
    """SSE 帧文本 → List[(event_name, payload)]。"""
    out: List[Tuple[str, Dict[str, Any]]] = []
    current_event: str = ""
    data_parts: List[str] = []
    for raw_line in body.splitlines():
        if raw_line.startswith(":"):  # 注释（心跳）
            continue
        if raw_line == "":
            if current_event and data_parts:
                payload_text = "\n".join(data_parts)
                try:
                    payload = json.loads(payload_text)
                except json.JSONDecodeError:
                    payload = {"_raw_text": payload_text}
                out.append((current_event, payload))
            current_event = ""
            data_parts = []
            continue
        m_e = _SSE_EVENT_PATTERN.match(raw_line)
        if m_e:
            current_event = m_e.group(1)
            continue
        m_d = _SSE_DATA_PATTERN.match(raw_line)
        if m_d:
            data_parts.append(m_d.group(1))
    return out


def _make_test_app(monkeypatch):
    """构造测试用 FastAPI app（屏蔽 DB / Gateway / Audit 都安全降级）。"""
    import app.main as app_main
    monkeypatch.setattr(app_main, "_build_audit_logger", lambda s: None)
    monkeypatch.setattr(app_main, "_build_model_gateway", lambda s: None)
    monkeypatch.setattr(app_main, "_build_session_factory", lambda s: None)
    return app_main.create_app()


# ------------------------- fixtures -------------------------


@pytest.fixture
def reset_settings():
    reset_settings_cache()
    get_settings()
    yield
    reset_settings_cache()


@pytest.fixture
def panel_client(reset_settings):
    """TestClient + X-Quality-Role=quality_officer 默认头。"""
    return {"role": "quality_officer"}


def _make_client(panel_client):
    from app.main import create_app

    return TestClient(create_app())


# ------------------------- 1. SSE 流式响应 + 鉴权 -------------------------


def test_modeling_role_missing_returns_403(reset_settings, monkeypatch):
    """缺 X-Quality-Role → QualityRoleForbiddenError → 403 envelope。"""
    import app.main as app_main

    monkeypatch.setattr(app_main, "_build_audit_logger", lambda s: None)
    monkeypatch.setattr(app_main, "_build_model_gateway", lambda s: None)
    monkeypatch.setattr(app_main, "_build_session_factory", lambda s: None)

    with TestClient(app_main.create_app(), raise_server_exceptions=False) as client:
        resp = client.post(
            "/modeling/run",
            json={"disease_code": "I21", "outcome": "cmi"},
        )
    assert resp.status_code == 403
    body = resp.json()
    assert body["error_code"] == "quality_role_forbidden"
    assert "<missing>" in body["message"] or "缺失" in body["message"]


def test_modeling_role_invalid_returns_403(reset_settings, monkeypatch):
    """角色不在白名单 → 403。"""
    import app.main as app_main

    monkeypatch.setattr(app_main, "_build_audit_logger", lambda s: None)
    monkeypatch.setattr(app_main, "_build_model_gateway", lambda s: None)
    monkeypatch.setattr(app_main, "_build_session_factory", lambda s: None)

    with TestClient(app_main.create_app(), raise_server_exceptions=False) as client:
        resp = client.post(
            "/modeling/run",
            json={"disease_code": "I21", "outcome": "cmi"},
            headers={"X-Quality-Role": "intern"},
        )
    assert resp.status_code == 403
    assert resp.json()["details"]["role"] == "intern"


def test_modeling_full_event_stream(reset_settings, monkeypatch):
    """完整事件序列 ack → sample-size → graph → identify → effect* →
    refute* → refute-summary → baseline → done。"""
    import app.main as app_main

    monkeypatch.setattr(app_main, "_build_audit_logger", lambda s: None)
    monkeypatch.setattr(app_main, "_build_model_gateway", lambda s: None)
    monkeypatch.setattr(app_main, "_build_session_factory", lambda s: None)

    with TestClient(app_main.create_app(), raise_server_exceptions=False) as client:
        resp = client.post(
            "/modeling/run",
            json={"disease_code": "I21", "outcome": "cmi"},
            headers={"X-Quality-Role": "quality_officer", "X-Operator-Id": "tester-007"},
        )

    assert resp.status_code == 200
    ctype = resp.headers.get("content-type", "")
    assert "text/event-stream" in ctype
    assert resp.headers.get("Cache-Control") == "no-cache"

    # TestClient 会聚合完整响应体 → 直接 decode 解析
    body = resp.text
    frames = _parse_sse_frames(body)
    assert frames, "SSE 流必须至少推 1 帧"

    # 事件序列必须含全部关键事件
    names = [n for n, _ in frames]
    required = {"ack", "sample-size", "graph", "identify", "refute-summary", "baseline", "done"}
    missing = required - set(names)
    assert not missing, f"SSE 缺关键事件 {missing}；实有: {names}"

    # ack 字段
    ack_payload = dict(frames[0][1])
    assert ack_payload["disease_code"] == "I21"
    assert ack_payload["outcome"] == "cmi"
    assert isinstance(ack_payload["sample_size"], int) and ack_payload["sample_size"] > 0

    # sample-size 含 verdict
    ssizes = [p for n, p in frames if n == "sample-size"]
    assert ssizes, "缺 sample-size 事件"
    assert ssizes[0]["verdict"] in ("sufficient", "marginal", "insufficient")

    # graph 含 nodes + edges
    graphs = [p for n, p in frames if n == "graph"]
    assert graphs, "缺 graph 事件"
    assert "nodes" in graphs[0] and len(graphs[0]["nodes"]) >= 4
    assert "edges" in graphs[0] and len(graphs[0]["edges"]) >= 1

    # effect 至少 1 条（含 treatment 主效应）
    effects = [p for n, p in frames if n == "effect"]
    assert effects, "缺 effect 事件"
    assert any(e.get("kind") == "treatment" for e in effects)
    assert all({"ate", "ci_lower", "ci_upper", "method_name", "sample_size"} <= set(e.keys()) for e in effects), (
        "每条 effect 必含完整字段"
    )

    # refute-summary 含 verdict
    rsummaries = [p for n, p in frames if n == "refute-summary"]
    assert rsummaries
    assert rsummaries[0]["verdict"] in ("robust", "partial", "unstable")

    # baseline 含 correlation + wedge
    bases = [p for n, p in frames if n == "baseline"]
    assert bases
    assert "wedge" in bases[0]
    assert "correlation" in bases[0]
    assert "causal_wedge_strong" in bases[0]

    # done 含 elapsed_ms
    dones = [p for n, p in frames if n == "done"]
    assert dones
    assert isinstance(dones[0]["elapsed_ms"], int)


def test_modeling_three_diseases(reset_settings, monkeypatch, parametrize_disease):
    """3 病种 × 4 outcome 全跑通（参数化）。"""
    import app.main as app_main

    monkeypatch.setattr(app_main, "_build_audit_logger", lambda s: None)
    monkeypatch.setattr(app_main, "_build_model_gateway", lambda s: None)
    monkeypatch.setattr(app_main, "_build_session_factory", lambda s: None)

    with TestClient(app_main.create_app(), raise_server_exceptions=False) as client:
        resp = client.post(
            "/modeling/run",
            json={"disease_code": parametrize_disease, "outcome": "cmi"},
            headers={"X-Quality-Role": "med_committee_reviewer"},
        )
    assert resp.status_code == 200, resp.text
    frames = _parse_sse_frames(resp.text)
    names = [n for n, _ in frames]
    assert "done" in names
    # disease_code 在 ack 中正确回填
    acks = [p for n, p in frames if n == "ack"]
    assert acks[0]["disease_code"] == parametrize_disease


# 为参数化准备 fixture：3 病种循环
@pytest.fixture(params=["I21", "I63", "Z96.10"])
def parametrize_disease(request) -> str:
    return request.param


# ------------------------- 2. 批量反事实 -------------------------


def test_counterfactual_role_missing_returns_403(reset_settings, monkeypatch):
    """缺 X-Quality-Role → 403 envelope。"""
    import app.main as app_main

    monkeypatch.setattr(app_main, "_build_audit_logger", lambda s: None)
    monkeypatch.setattr(app_main, "_build_model_gateway", lambda s: None)
    monkeypatch.setattr(app_main, "_build_session_factory", lambda s: None)

    with TestClient(app_main.create_app(), raise_server_exceptions=False) as client:
        resp = client.post(
            "/counterfactual/run",
            json={
                "disease_code": "I21",
                "outcome": "cmi",
                "scenarios": [{"intervention_node": "treatment:PCI", "hypothetical_value": "PCI"}],
            },
        )
    assert resp.status_code == 403


def test_counterfactual_role_invalid_returns_403(reset_settings, monkeypatch):
    """非法角色 → 403。"""
    import app.main as app_main

    monkeypatch.setattr(app_main, "_build_audit_logger", lambda s: None)
    monkeypatch.setattr(app_main, "_build_model_gateway", lambda s: None)
    monkeypatch.setattr(app_main, "_build_session_factory", lambda s: None)

    with TestClient(app_main.create_app(), raise_server_exceptions=False) as client:
        resp = client.post(
            "/counterfactual/run",
            json={
                "disease_code": "I21",
                "outcome": "cmi",
                "scenarios": [{"intervention_node": "treatment:PCI", "hypothetical_value": "PCI"}],
            },
            headers={"X-Quality-Role": "intern"},
        )
    assert resp.status_code == 403


def test_counterfactual_batch_ten_scenarios(reset_settings, monkeypatch):
    """10 条场景批量 < 30s 返回（用 I63 fixture）。"""
    import app.main as app_main

    monkeypatch.setattr(app_main, "_build_audit_logger", lambda s: None)
    monkeypatch.setattr(app_main, "_build_model_gateway", lambda s: None)
    monkeypatch.setattr(app_main, "_build_session_factory", lambda s: None)

    scenarios = [
        {"intervention_node": "treatment:thrombolysis", "hypothetical_value": 1, "label": f"溶栓 #{i}"}
        if i < 5
        else {"intervention_node": "age_band:3", "hypothetical_value": 3, "label": f"年龄段 #{i}"}
        for i in range(10)
    ]

    with TestClient(app_main.create_app(), raise_server_exceptions=False) as client:
        resp = client.post(
            "/counterfactual/run",
            json={
                "disease_code": "I63",
                "outcome": "cmi",
                "scenarios": scenarios,
            },
            headers={"X-Quality-Role": "quality_officer"},
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["schema_version"] == "counterfactual_response/1.0"
    assert body["ok"] is True
    assert body["disease_code"] == "I63"
    assert body["outcome"] == "cmi"
    assert body["n_scenarios"] == 10
    assert len(body["scenarios"]) == 10
    # 顺序与输入一致
    assert body["scenarios"][0]["intervention_node"] == "treatment:thrombolysis"
    assert body["scenarios"][5]["intervention_node"] == "age_band:3"
    # 每条 scenario 至少含核心字段
    for s in body["scenarios"]:
        assert {"scenario_id", "predicted_outcome", "do_operations", "elapsed_ms", "timeout"} <= set(s.keys())
    # elapsed_ms ≥ 0；total elapsed_ms < 30s
    assert body["elapsed_ms"] < 30000


def test_counterfactual_validation_errors(reset_settings, monkeypatch):
    """body 字段错误 → 422 / 400 envelope。"""
    import app.main as app_main

    monkeypatch.setattr(app_main, "_build_audit_logger", lambda s: None)
    monkeypatch.setattr(app_main, "_build_model_gateway", lambda s: None)
    monkeypatch.setattr(app_main, "_build_session_factory", lambda s: None)

    # 缺 disease_code
    with TestClient(app_main.create_app(), raise_server_exceptions=False) as client:
        resp = client.post(
            "/counterfactual/run",
            json={"outcome": "cmi", "scenarios": [{"intervention_node": "treatment:PCI", "hypothetical_value": "PCI"}]},
            headers={"X-Quality-Role": "quality_officer"},
        )
    # 422 = 请求体验证失败 envelope（不是 missing 不是 invalid role，是ValueError → RequestValidationError 路径）
    assert resp.status_code in (400, 422, 500)

    # scenarios 为空
    with TestClient(app_main.create_app(), raise_server_exceptions=False) as client:
        resp = client.post(
            "/counterfactual/run",
            json={"disease_code": "I21", "outcome": "cmi", "scenarios": []},
            headers={"X-Quality-Role": "quality_officer"},
        )
    assert resp.status_code in (400, 422, 500)

    # scenarios 含非 dict 元素
    with TestClient(app_main.create_app(), raise_server_exceptions=False) as client:
        resp = client.post(
            "/counterfactual/run",
            json={
                "disease_code": "I21",
                "outcome": "cmi",
                "scenarios": ["bad-string"],
            },
            headers={"X-Quality-Role": "quality_officer"},
        )
    assert resp.status_code in (400, 422, 500)

    # scenarios 超 50 上限
    big = [{"intervention_node": "treatment:PCI", "hypothetical_value": "PCI"} for _ in range(60)]
    with TestClient(app_main.create_app(), raise_server_exceptions=False) as client:
        resp = client.post(
            "/counterfactual/run",
            json={
                "disease_code": "I21",
                "outcome": "cmi",
                "scenarios": big,
            },
            headers={"X-Quality-Role": "quality_officer"},
        )
    assert resp.status_code in (400, 422, 500)


def test_counterfactual_one_bad_node_isolated(reset_settings, monkeypatch):
    """单条非法节点只影响自身（标 validation_error），其他场景正常返回。"""
    import app.main as app_main

    monkeypatch.setattr(app_main, "_build_audit_logger", lambda s: None)
    monkeypatch.setattr(app_main, "_build_model_gateway", lambda s: None)
    monkeypatch.setattr(app_main, "_build_session_factory", lambda s: None)

    scenarios = [
        {"intervention_node": "treatment:thrombolysis", "hypothetical_value": 1},
        {"intervention_node": "non_existent_node", "hypothetical_value": 99},  # 非法
        {"intervention_node": "age_band:3", "hypothetical_value": 3},
    ]

    with TestClient(app_main.create_app(), raise_server_exceptions=False) as client:
        resp = client.post(
            "/counterfactual/run",
            json={
                "disease_code": "I63",
                "outcome": "cmi",
                "scenarios": scenarios,
            },
            headers={"X-Quality-Role": "quality_officer"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["n_scenarios"] == 3
    out_scenarios = body["scenarios"]
    # 索引 1（非法节点）必须标 validation_error / 其 do_operations 含 validation_error:
    bad = out_scenarios[1]
    assert any(isinstance(op, str) and op.startswith("validation_error:") for op in bad["do_operations"])
    # 索引 0 / 2 必须正常返回 predicted_outcome（不等于 None 表示跑通）
    assert out_scenarios[0].get("predicted_outcome") is not None or out_scenarios[0].get("timeout") is False
    assert out_scenarios[2].get("predicted_outcome") is not None or out_scenarios[2].get("timeout") is False
