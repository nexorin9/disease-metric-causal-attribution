"""dipcase_stub FastAPI 应用入口 — 暴露 create_app 供 ASGITransport / uvicorn 复用。"""
from __future__ import annotations

from tests.mocks.dipcase_stub import create_app as _create

create_app = _create


def main() -> None:  # pragma: no cover - 仅供手动启动
    import uvicorn

    uvicorn.run("tests.mocks.dipcase_stub.main:app", host="127.0.0.1", port=18899, log_level="info")


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["create_app", "app"]
