"""tests.mocks.openai_stub — 本地 OpenAI-compatible 协议替身（T29）。

目的：协议兼容性测试需要一个能跑起来的 OpenAI 兼容端点（18888 端口），
覆盖：协议握手、超时注入、不可解析响应、5xx 注入、降级路径、provider-neutral 边界。

- **main.py**：FastAPI app，`/v1/chat/completions` 返回标准 OpenAI-compatible 协议响应；
  配套 3 个 admin 端点用于测试中动态注入失败场景。
- **run.sh**：后台启动替身脚本（端口 18888 / PID 文件 / 日志 / 优雅停机）。
- 全部路径在 `disease-metric-causal-attribution/` 内部，与生产凭据隔离。
"""
from __future__ import annotations

# 暴露 main 模块以支持 `from tests.mocks.openai_stub import main`
from . import main as main  # noqa: F401

__all__ = ["main"]