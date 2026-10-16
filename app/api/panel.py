"""浏览器面板路由层（spec.md §11 浏览器面板 + T22 收尾交付）。

挂载形态：
  GET /              → Jinja2 渲染 app/panel/templates/index.html
  GET /static/*      → StaticFiles 挂载 app/panel/static/（plotly.min.js / panel.css / panel.js）

设计约束：
- 路径按 main.py::_register_routes 注入："app.api.panel" → "/"
- 不在路由层做鉴权（SSE / 反事实 / 评审 confirm/reject 在各自子端点验 X-Quality-Role）
- 模板渲染失败的兜底：返回 500 + 一段简短 HTML 文本（不静默）
- 不在 lifespan 内挂载（避免路由注册顺序脆弱）
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Request
from fastapi.exceptions import HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.panel import STATIC_DIR, TEMPLATES_DIR

LOGGER = logging.getLogger(__name__)

# 路由前缀空（与 main.py 路径 hint "/" 一致）；
# static 挂载由 main.py 中 _register_routes 注册本 router 之后，由下方 mount_static_files()
# 显式追加；这里只暴露 HTML 入口。
router = APIRouter(tags=["panel"])


# ---------------------------------------------------------------------------
# 模板渲染辅助
# ---------------------------------------------------------------------------

def _render_index_html(context: Dict[str, Any]) -> str:
    """渲染 templates/index.html（无外部 jinja 扩展变量，保持模板可独立 grep）。

    使用纯 stdlib jinja2.Environment（spec §11 主要使用 jinja2，本路由避免
    引入额外依赖）。模板里没有复杂 filter，仅做变量替换 + 简单遍历。
    """
    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape  # type: ignore
    except Exception as exc:  # pragma: no cover — jinja2 是项目依赖
        LOGGER.warning("jinja2 不可用：%s", exc)
        raise HTTPException(status_code=500, detail=f"jinja2 unavailable: {exc}") from exc

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("index.html")
    return template.render(**context)


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------

@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index(request: Request) -> HTMLResponse:
    """GET / → 浏览器面板主页。

    模板上下文目前只用静态值；后续 task 把"系统模块版本 / 默认 outcome"等
    通过 main.py::app.state 注入即可。
    """
    context = {
        "app_name": "单病种指标因果归因系统",
        "spec_section": "spec.md §11 浏览器面板",
        "phase": "Phase 4 T22 baseline",
    }
    try:
        body = _render_index_html(context)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 — 模板渲染任何失败都明示
        LOGGER.exception("index.html 渲染失败")
        raise HTTPException(status_code=500, detail=f"index render failed: {exc}") from exc

    return HTMLResponse(
        content=body,
        status_code=200,
        headers={
            "Cache-Control": "no-cache, must-revalidate",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/healthz/panel", include_in_schema=False)
async def panel_health() -> Dict[str, Any]:
    """panel 健康检查（不连真实数据库）便于运维 sanity check。

    验证 index.html 与 static 资源都存在；任何缺失都返回 503。
    """
    files = [
        TEMPLATES_DIR / "index.html",
        STATIC_DIR / "plotly.min.js",
        STATIC_DIR / "panel.css",
        STATIC_DIR / "panel.js",
    ]
    missing = [str(p) for p in files if not p.exists()]
    ok = not missing
    body = {
        "panel_ok": ok,
        "templates_dir": str(TEMPLATES_DIR),
        "static_dir": str(STATIC_DIR),
        "missing_files": missing,
    }
    return JSONResponse(
        status_code=200 if ok else 503,
        content=body,
    )


# ---------------------------------------------------------------------------
# StaticFiles 挂载（由 main 在 lifespan 内显式 add）
# ---------------------------------------------------------------------------

def mount_static_files(app: Any) -> bool:
    """把 app/panel/static/ 挂到 /static 路径。

    Returns
    -------
    bool
        True = 挂载成功；False = static 目录缺失（不阻断 runtime）。
    """
    if not STATIC_DIR.exists():
        LOGGER.warning("Static 目录不存在：%s（panel 仍可通过 / 打开但静态资源 404）", STATIC_DIR)
        return False
    try:
        app.mount(
            "/static",
            StaticFiles(directory=str(STATIC_DIR), html=False, check_dir=False),
            name="panel-static",
        )
        LOGGER.info("Static 已挂载：/static → %s", STATIC_DIR)
        return True
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Static 挂载失败：%s", exc)
        return False


__all__ = ["mount_static_files", "router"]
