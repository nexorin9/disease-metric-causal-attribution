"""OpenAI-compatible 协议替身（FastAPI 应用）。

设计（spec §7 + §11）：
- 跑在 18888 端口（`OPENAI_STUB_URL` 的默认本地端点）。
- `/v1/chat/completions`：标准 OpenAI-compatible 协议路径。POST body{model, messages, response_format, …} → 返回
  `{id, object, created, model, choices[{index, message, finish_reason}], usage}`，且 message.content
  是合法 JSON 字符串（满足 `response_format=json_object` 协议约定）。
- `/v1/__admin/inject_timeout`：注入下一次调用延时返回（用于触发 ModelGateway 超时降级）。
- `/v1/__admin/inject_unparseable`：返回 200 但 content 不是合法 JSON（触发不可解析降级）。
- `/v1/__admin/inject_500`：下一次调用返回 500（触发 http_error 降级）。
- `/v1/__admin/reset`：清空所有注入状态（用于「重新握手」测试）。
- `/healthz`：替身存活自检端点（测试 fixture 等待用）。

注入语义：每次注入**只生效一次**（单次一次性），这样可以让一个测试依次触发多条路径：
握手成功 → 注入 timeout → 下一次调用超时 → 重置 → 重新握手成功。
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import threading
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
except ImportError:  # pragma: no cover - 依赖已在 pyproject 锁定
    FastAPI = None  # type: ignore
    JSONResponse = None  # type: ignore


LOGGER = logging.getLogger("openai_stub")

# 替身默认返回的 content（按 method 路由；未知 method 给定 narrative_zh）
_DEFAULT_CONTENT_BY_METHOD = {
    "generate_node_narrative": {"narrative_zh": "[stub] 该节点为主驱动因子；模型由协议替身给出。"},
    "generate_counterfactual_narrative": {
        "predicted_outcome_zh": "[stub] 反事实 outcome 预测已生成；模型由协议替身给出。",
        "confidence": 0.85,
    },
    "generate_draft_summary": {"summary_zh": "[stub] 草稿摘要已生成；模型由协议替身给出。"},
}

_DEFAULT_FALLBACK = {"narrative_zh": "[stub] 通用 fallback 响应。"}


# ---------------------------------------------------------------------------
# 注入状态（线程安全）
# ---------------------------------------------------------------------------


class _InjectionState:
    """测试替身状态：单次一次性注入 + 计数 + 调用历史。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._next_timeout_sleep: Optional[float] = None  # 一次性：下一次调用延时秒数
        self._next_unparseable: bool = False  # 一次性：下一次调用返回不可解析
        self._next_status_code: Optional[int] = None  # 一次性：下一次调用返回 status_code
        self._next_raw_body: Optional[str] = None  # 一次性：下一次调用返回 raw body（覆盖 default）
        self.call_history: List[Dict[str, Any]] = []
        self.reset_count = 0

    def reset(self):
        with self._lock:
            self._next_timeout_sleep = None
            self._next_unparseable = False
            self._next_status_code = None
            self._next_raw_body = None
            self.reset_count += 1

    def inject_timeout(self, delay_seconds: float):
        with self._lock:
            self._next_timeout_sleep = float(delay_seconds)

    def inject_unparseable(self):
        with self._lock:
            self._next_unparseable = True

    def inject_status(self, status_code: int, raw_body: Optional[str] = None):
        with self._lock:
            self._next_status_code = int(status_code)
            self._next_raw_body = raw_body

    def take_next_action(self) -> Dict[str, Any]:
        """取出下一次调用的注入动作并重置。"""
        with self._lock:
            action = {
                "timeout_sleep": self._next_timeout_sleep,
                "unparseable": self._next_unparseable,
                "status_code": self._next_status_code,
                "raw_body": self._next_raw_body,
            }
            self._next_timeout_sleep = None
            self._next_unparseable = False
            self._next_status_code = None
            self._next_raw_body = None
            return action

    def record_call(self, request_summary: Dict[str, Any]):
        with self._lock:
            self.call_history.append(request_summary)


STATE = _InjectionState()


# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------


def _resolve_method_name(messages: List[Dict[str, Any]]) -> str:
    """从 system prompt 反推方法名（替身按方法路由默认响应）。"""
    for msg in messages:
        if msg.get("role") == "system":
            content = str(msg.get("content", "") or "")
            if "反事实" in content:
                return "generate_counterfactual_narrative"
            if "草稿摘要" in content:
                return "generate_draft_summary"
            if "归因评审" in content:
                return "generate_node_narrative"
            break
    return "unknown"


def _build_default_response(messages: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
    method_name = _resolve_method_name(messages)
    content_obj = _DEFAULT_CONTENT_BY_METHOD.get(method_name, _DEFAULT_FALLBACK)
    return {
        "id": f"chatcmpl-stub-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps(content_obj, ensure_ascii=False),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
    }


def create_app() -> "FastAPI":
    """构造 FastAPI 应用（供 main 入口 / 进程内 fixture 复用）。"""
    if FastAPI is None:
        raise RuntimeError("FastAPI 未安装；pyproject 已锁定 fastapi 依赖")

    app = FastAPI(title="openai_stub", version="1.0")

    @app.get("/healthz")
    async def healthz():
        """替身存活自检。"""
        return {"status": "ok", "calls_recorded": len(STATE.call_history), "reset_count": STATE.reset_count}

    @app.get("/v1/__admin/state")
    async def admin_state():
        """查看当前注入状态（测试调试用）。"""
        return {
            "calls_recorded": len(STATE.call_history),
            "reset_count": STATE.reset_count,
            "next_pending": STATE.take_next_action.__name__,  # 只用于验证方法存在
            "last_call": STATE.call_history[-1] if STATE.call_history else None,
        }

    @app.post("/v1/__admin/inject_timeout")
    async def admin_inject_timeout(payload: Dict[str, Any]):
        delay = float(payload.get("delay_seconds", 60.0))
        STATE.inject_timeout(delay)
        return {"injected": "timeout", "delay_seconds": delay}

    @app.post("/v1/__admin/inject_unparseable")
    async def admin_inject_unparseable():
        STATE.inject_unparseable()
        return {"injected": "unparseable"}

    @app.post("/v1/__admin/inject_500")
    async def admin_inject_500(payload: Optional[Dict[str, Any]] = None):
        raw = (payload or {}).get("raw_body", "Internal Server Error from openai_stub")
        STATE.inject_status(500, raw_body=raw)
        return {"injected": "status_500", "raw_body": raw}

    @app.post("/v1/__admin/reset")
    async def admin_reset():
        STATE.reset()
        return {"reset": True}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        """OpenAI-compatible 主路径。"""
        body: Dict[str, Any] = {}
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return JSONResponse(status_code=400, content={"error": "invalid_json"})

        messages = body.get("messages") or []
        model = str(body.get("model") or "stub-model")
        action = STATE.take_next_action()

        STATE.record_call(
            {
                "ts": time.time(),
                "model": model,
                "messages_count": len(messages),
                "response_format": body.get("response_format"),
                "timeout_requested": bool(action["timeout_sleep"]),
            }
        )

        # 1) 超时注入：sleep > ModelGateway timeout → 触发 httpx.TimeoutException
        if action["timeout_sleep"]:
            await asyncio.sleep(float(action["timeout_sleep"]))
            # 即使 sleep 完，仍返回 200（ModelGateway 这边已先超时抛异常）

        # 2) status_code 注入（500 / 其它）
        if action["status_code"] is not None and action["status_code"] >= 400:
            return JSONResponse(
                status_code=int(action["status_code"]),
                content={"error": action["raw_body"] or "stub_error"},
            )

        # 3) 不可解析注入：返回 200 但 content 不是 JSON
        if action["unparseable"]:
            return JSONResponse(
                status_code=200,
                content={
                    "id": "chatcmpl-stub-unparseable",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "<not valid json>{{garbage",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )

        # 4) 默认：正常协议响应
        return JSONResponse(status_code=200, content=_build_default_response(messages, model))

    return app


# 进程级默认 app（uvicorn openai_stub.main:app --port 18888）
app = create_app()


def main() -> None:  # pragma: no cover - 直接启动入口
    """uvicorn 入口。"""
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(f"uvicorn 未安装：{exc}") from exc
    uvicorn.run(
        "tests.mocks.openai_stub.main:app",
        host="127.0.0.1",
        port=18888,
        log_level="warning",
    )


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["STATE", "create_app", "main"]