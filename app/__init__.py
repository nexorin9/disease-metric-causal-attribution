"""单病种指标因果归因系统 — 应用包。

模块边界遵循 spec.md §2 / §8：
  - app.adapters     HIS 读 / 单病种上报 HTTP 写 / ModelGateway
  - app.core         因果推理流水线（CausalModel + 反事实 + 反驳 + SCM 自动生成）
  - app.api          FastAPI 路由（SSE / drafts / healthz）
  - app.drafts       评审草稿状态机
  - app.db           SQLAlchemy schema
  - app.audit        审计日志（字段脱敏）
  - app.panel        浏览器面板模板 + 静态资源

AI 角色（ai_role=core）：ModelGateway 走 OpenAI-compatible 单协议；
业务层不耦合厂商 SDK，凭据统一从环境变量读取。
"""

__version__ = "0.1.0"