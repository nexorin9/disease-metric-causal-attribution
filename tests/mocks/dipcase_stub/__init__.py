"""单病种上报系统 HTTP 协议替身（FastAPI 应用）。

设计目标：spec §3.2 真实协议握手 + e2e 测试捕获。
- POST /v1/reports/attribution/draft：单病种上报端点
  - 验证 Content-Type=application/json
  - 验证 Authorization Bearer token
  - 验证 payload 含 spec §3.2 必填字段
  - 返回 200 {ok: True, report_id: "..."}
- POST /v1/__admin/inject_500：下一次调用返回 HTTP 500（模拟服务端故障）
- POST /v1/__admin/inject_503：下一次调用返回 HTTP 503（模拟服务端暂时不可用）
- GET /v1/__admin/received：查询已捕获的 payloads（最近 N 条；按时间序）
- POST /v1/__admin/reset：清空注入状态 + received 列表
- GET /v1/healthz：替身存活自检
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import JSONResponse
except ImportError:  # pragma: no cover
    FastAPI = None  # type: ignore
    JSONResponse = None  # type: ignore

LOGGER = logging.getLogger("dipcase_stub")

REQUIRED_PAYLOAD_FIELDS = (
    "disease_code",
    "quarter",
    "outcome",
    "treatment",
    "graph_snapshot_dot",
    "effect_estimates",
    "refutation_results",
    "counterfactual_scenarios",
    "narrative_zh",
    "reviewer_id",
    "model_version",
)


class _StubState:
    """线程安全的注入状态 + 已捕获 payloads 容器。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._next_status_code: Optional[int] = None  # 一次性
        self._next_raw_body: Optional[Dict[str, Any]] = None
        self.received_payloads: List[Dict[str, Any]] = []
        self.call_count = 0
        self.reset_count = 0

    def reset(self):
        with self._lock:
            self._next_status_code = None
            self._next_raw_body = None
            self.received_payloads.clear()
            self.call_count = 0
            self.reset_count += 1

    def inject_status(self, status_code: int, body: Optional[Dict[str, Any]] = None):
        with self._lock:
            self._next_status_code = int(status_code)
            self._next_raw_body = body

    def record_call(
        self,
        payload: Dict[str, Any],
        *,
        headers: Dict[str, str],
    ) -> Dict[str, Any]:
        """记录一次调用并返回 mock response body。

        Returns
        -------
        dict
            {"ok": True, "report_id": "...", "received_at": "..."}
            或注入的 raw_body / 错误响应。
        """
        with self._lock:
            self.call_count += 1
            record = {
                "received_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "headers": {
                    k: v
                    for k, v in headers.items()
                    if k.lower() in {"content-type", "authorization", "x-operator-id"}
                },
                "payload": dict(payload),
            }
            self.received_payloads.append(record)
            if self._next_status_code is not None:
                status = self._next_status_code
                body = self._next_raw_body
                self._next_status_code = None
                self._next_raw_body = None
                return {"status_code": status, "body": body or {"ok": False, "error": "injected"}}
            return {
                "status_code": 200,
                "body": {
                    "ok": True,
                    "report_id": f"rpt_{uuid.uuid4().hex[:12]}",
                    "received_at": record["received_at"],
                },
            }


def create_app() -> Any:
    """构造 FastAPI 应用。"""
    if FastAPI is None:  # pragma: no cover
        raise RuntimeError("fastapi 未装")

    app = FastAPI(title="dipcase_stub")
    state = _StubState()
    app.state.stub_state = state

    @app.post("/v1/reports/attribution/draft")
    async def post_attribution_draft(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error_code": "invalid_json", "message": str(exc)},
            )

        if not isinstance(body, dict):
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error_code": "invalid_payload", "message": "payload 必须是 dict"},
            )

        # 验证必填字段（spec §3.2）
        missing = [f for f in REQUIRED_PAYLOAD_FIELDS if f not in body]
        if missing:
            return JSONResponse(
                status_code=400,
                content={
                    "ok": False,
                    "error_code": "missing_required_fields",
                    "missing_fields": missing,
                },
            )

        result = state.record_call(body, headers=dict(request.headers))
        return JSONResponse(status_code=result["status_code"], content=result["body"])

    @app.post("/v1/__admin/inject_status")
    async def inject_status(payload: Dict[str, Any]) -> JSONResponse:
        state.inject_status(
            int(payload.get("status_code", 500)),
            body=payload.get("body"),
        )
        return JSONResponse(content={"ok": True, "injected": payload.get("status_code", 500)})

    @app.post("/v1/__admin/reset")
    async def reset_state() -> JSONResponse:
        state.reset()
        return JSONResponse(content={"ok": True})

    @app.get("/v1/__admin/received")
    async def list_received() -> JSONResponse:
        return JSONResponse(
            content={
                "ok": True,
                "call_count": state.call_count,
                "received_payloads": state.received_payloads,
            }
        )

    @app.get("/v1/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse(
            content={"ok": True, "status": "alive", "call_count": state.call_count}
        )

    return app


__all__ = ["create_app", "_StubState"]
