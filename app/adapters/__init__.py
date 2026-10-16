"""外部系统适配层（spec.md §8）：
  - his_adapter         HIS 出院病案首页库直连（仅读；psycopg2 / pymysql 双 driver）
  - dipcase_reporter    单病种上报系统 HTTP POST 写回（reviewer 确认触发）
  - model_gateway       OpenAI-compatible 单协议 LLM 网关（结构化 + 降级 + 审计）

业务层（app.core / app.api / app.drafts）只能通过本层与外部系统交互；
不直接 import os.environ，也不直接依赖厂商 SDK。
"""

from __future__ import annotations