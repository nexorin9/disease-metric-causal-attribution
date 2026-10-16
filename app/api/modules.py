"""已装载算法模块清单（spec.md §11 收尾交付：面板展示模型版本）。

GET /api/modules → 200 + {
  schema_version, ok, modules: {dowhy, econml, networkx, ...}, versions,
  model_version (内置默认因果模型版本)
}

用途：浏览器面板右侧栏 / 审计自检展示当前系统装载了哪些算法、什么版本。
spec §11 第 5 步硬约束：版本号必须能稳定获取到，便于运维核对。
"""
from __future__ import annotations

import importlib
import logging
from typing import Any, Dict

from fastapi import APIRouter
from fastapi.responses import JSONResponse

LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["system"])

MODULES_SCHEMA_VERSION = "modules_response/1.0"

# 期望系统装载的算法模块清单（按 spec §11 第 5 步）
_TRACKED_MODULES = (
    "dowhy",
    "networkx",
    "pandas",
    "numpy",
    "scipy",
    "fastapi",
    "pydantic",
    "httpx",
    "sqlalchemy",
)


def _safe_version(name: str) -> Dict[str, Any]:
    """import + 拿 __version__；缺失即 missing，不抛错。"""
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", None)
        return {
            "name": name,
            "available": True,
            "version": version or "<unknown>",
        }
    except ImportError as exc:
        return {
            "name": name,
            "available": False,
            "version": None,
            "reason": f"ImportError: {exc}",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "name": name,
            "available": False,
            "version": None,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def _internal_model_version() -> str:
    """内置因果模型版本号（spec §11 第 5 步 硬约束）。

    取 app.__version__ + 一个固定的 model_version 字符串（spec §3.2 payload 字段）。
    """
    try:
        from app import __version__ as v
    except Exception:  # noqa: BLE001
        v = "0.0.0"
    return f"causal_pipeline/{v}"


@router.get("/modules")
async def modules() -> JSONResponse:
    """GET /api/modules → 模块清单 + 版本号。"""
    modules_payload = [_safe_version(name) for name in _TRACKED_MODULES]
    available = sum(1 for m in modules_payload if m["available"])
    body = {
        "schema_version": MODULES_SCHEMA_VERSION,
        "ok": True,
        "n_tracked": len(_TRACKED_MODULES),
        "n_available": available,
        "modules": modules_payload,
        "model_version": _internal_model_version(),
        "graph_render": "plotly.js@2.x",  # spec §11 第 5 步 — 面板前端版本
    }
    return JSONResponse(status_code=200, content=body)


__all__ = ["MODULES_SCHEMA_VERSION", "router"]