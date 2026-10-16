"""端到端集成测试（spec.md §11 收尾交付最后一档）。

覆盖主路径：
  POST /api/modeling/run  （SSE 完整流）  →
  POST /api/counterfactual/run （批量反事实 JSON）→
  DraftRepository.create  （起草归因报告）→
  POST /api/drafts/{draft_id}/review confirm （reviewer 确认）→
  上报 mock 收到 payload + 状态推进 reviewed →
  audit_logs 含全部 lifecycle/modeling/counterfactual/draft/post/state/transition 事件。

⚠️ 真实业务约束（spec §6）：
- HIS 用 SQLite 替身（环境变量注入 sqlite:// 路径）
- 上报用 tests/mocks/dipcase_stub/（FastAPI + ASGITransport）
- ModelGateway 用 tests/mocks/openai_stub/（T29 已落盘，FastAPI + ASGITransport）
- 不连真实生产库 / 不调真实 LLM / 不上报真实系统

测试范围：完整 e2e 链路 + 状态推进 + 审计落库 + 失败回退（rollback）。
"""
