"""浏览器面板（spec.md §11 收尾交付：FastAPI + 浏览器 SSE 实时面板）。

面板物理形态（Phase 1 视觉目标卡）：

  app/panel/
  ├── __init__.py
  ├── templates/
  │   └── index.html          # Jinja2 主入口
  └── static/
      ├── plotly.min.js       # Plotly.js 2.x minified（API-compatible 本地兜底）
      ├── panel.css           # 设计 token（深色背景 + 三色节点 + 8px 栅格）
      └── panel.js            # EventSource 订阅 + Plotly.init + 三态 UI

FastAPI 路由层在 ``app/api/panel.py``：
  GET  /                  → 渲染 templates/index.html
  /static/*               → 挂载 static/ 目录

设计约束（spec §11 浏览器面板 + §6 失败不静默改数据）：
- **不静默改数据**：面板只读 SSE 流 + 调 POST 接口，所有写回经 reviewer 确认。
- **三态 UI**：loading（建模中）/ empty（无因果图）/ error（建模失败重试）
- **设计 token 集中**：颜色 + 间距 + 字号一律走 CSS 变量，禁止散落硬编码。
- **拒绝浏览器默认样式**：原生 ``alert/prompt/confirm`` 全替为 toast 与模态。
"""
from __future__ import annotations

from pathlib import Path

# templates / static 物理路径解析（让 app/api/panel.py 导入直接用）
PANEL_PACKAGE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = PANEL_PACKAGE_DIR / "templates"
STATIC_DIR = PANEL_PACKAGE_DIR / "static"


def templates_dir() -> Path:
    """供 app/api/panel.py 调用的 templates 路径解析（含 deprecated 测试入口）。"""
    return TEMPLATES_DIR


def static_dir() -> Path:
    """供 app/api/panel.py 调用的 static 路径解析。"""
    return STATIC_DIR


__all__ = [
    "PANEL_PACKAGE_DIR",
    "STATIC_DIR",
    "TEMPLATES_DIR",
    "static_dir",
    "templates_dir",
]
