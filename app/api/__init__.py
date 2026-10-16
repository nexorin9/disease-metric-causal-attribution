"""HTTP API 路由汇总（spec.md §11 收尾交付形态：FastAPI 常驻 + 浏览器 SSE 面板）。

子模块在 Phase 4 后续 task 逐个引入并注册：

| Endpoint                          | 引入 task | 说明                            |
|-----------------------------------|-----------|--------------------------------|
| GET  /api/healthz                 | T21       | 进程 + DB + HisConn + Gateway 健康 |
| GET  /api/modules                 | T21       | 已装载算法模块与版本清单        |
| POST /api/modeling/run            | T20       | SSE 流式建模（建模主路径流式）  |
| POST /api/counterfactual/run      | T20       | 反事实批量（JSON）              |
| GET  /api/drafts/{draft_id}        | T21       | 单条草稿详情                    |
| GET  /api/drafts                  | T21       | 草稿列表（按 status 过滤）      |
| POST /api/drafts/{draft_id}/review| T21       | 评审 action=confirm|reject      |
| GET  /                            | T22       | 浏览器面板入口（Jinja2）        |

本文件本身只声明 API 元数据（VERSION / OPENAPI_TAGS / 已注册路由反射），不持业务逻辑；
具体路由挂载由各子模块在 lifespan 完成后再注入到 app 实例上。

⚠️ 模块边界：本包**只**做路由挂载。业务逻辑在 app.core 与 app.adapters；状态机在
app.drafts；持久化在 app.db；审计在 app.audit。路由层必须保持"无状态 + 仅参数校验 + 调业务层"。
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

API_VERSION = "v1"
API_PREFIX = "/api"
OPENAPI_TITLE = "单病种指标因果归因系统"
OPENAPI_DESCRIPTION = (
    "基于四步因果流水线的月度单病种评审辅助面板：流式建模 / 反事实 / "
    "草稿评审 / 自动上报。规格详见 spec.md。"
)
OPENAPI_TAGS: Tuple[Dict[str, Any], ...] = (
    {"name": "modeling", "description": "建模主路径（SSE 流式 + JSON 批量）"},
    {"name": "counterfactual", "description": "反事实模拟（批量 JSON）"},
    {"name": "drafts", "description": "归因报告草稿：GET 单条 / 列表 / POST review"},
    {"name": "system", "description": "进程健康 / 模块清单 / 错误诊断"},
)


__all__ = [
    "API_PREFIX",
    "API_VERSION",
    "OPENAPI_DESCRIPTION",
    "OPENAPI_TAGS",
    "OPENAPI_TITLE",
]
