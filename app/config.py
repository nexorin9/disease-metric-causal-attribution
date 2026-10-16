"""应用配置层（pydantic-settings）。

强制约束：
- **凭据位一律从环境变量读取**（spec.md §8）；缺关键凭据显式报错，绝不静默兜底。
- 业务层 / API 层只能通过 `get_settings()` 拿单例；不得 import os.environ 直读。
- 字段命名沿用 spec.md §8 + .env.example，禁止混用下划线 / 短横线导致凭据位对不上。
- 启动期 `validate_for_runtime()` 校验必填项；HIS_DB_URL / DATABASE_URL / DIPCASE_REPORTER_*
  OPENAI_BASE_URL 任何一项缺失即拒绝（本地 SQLite 可临时充当 DATABASE_URL 替身用于开发）。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """应用配置（pydantic-settings 单例）。"""

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE) if ENV_FILE.exists() else None,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- 应用基础 ----
    app_env: Literal["local", "test", "staging", "prod"] = "local"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # ---- 项目自带 PostgreSQL / SQLite 替身 ----
    database_url: str = "sqlite:///./local_state.db"

    # ---- HIS 出院病案首页库直连（仅读） ----
    his_db_url: str
    his_db_user: str
    his_db_pass: str
    his_db_pool_recycle: int = 1800

    # ---- 单病种上报系统 HTTP POST 写归因报告草稿 ----
    dipcase_reporter_url: str
    dipcase_reporter_token: str
    dipcase_reporter_timeout: int = 15

    # ---- ModelGateway · OpenAI-compatible 单协议 ----
    openai_base_url: str
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    openai_timeout: int = 30
    openai_stub_url: Optional[str] = None  # 仅本地协议兼容性测试用

    # ---- 业务门栏 ----
    min_sample_size: int = 30
    max_ci_width: float = 0.6
    refutation_n_jobs: int = -1

    # ---- 审计 ----
    audit_log_retention_days: int = 365

    # ---- 面板鉴权 ----
    quality_role_header: str = "X-Quality-Role"
    local_reviewer_id: str = "dev-reviewer-001"
    local_reviewer_role: str = "med_committee_reviewer"

    @field_validator("min_sample_size")
    @classmethod
    def _check_min_sample(cls, v: int) -> int:
        if v < 1:
            raise ValueError("min_sample_size 必须 ≥ 1（业务门栏；spec.md §6）")
        return v

    @field_validator("max_ci_width")
    @classmethod
    def _check_ci_width(cls, v: float) -> float:
        if not 0.0 < v < 5.0:
            raise ValueError("max_ci_width 应在 (0, 5) 区间；0.6 是默认值")
        return v

    def driver_kind(self) -> str:
        """根据 HIS_DB_URL 推断 SQLAlchemy driver：postgresql / mysql / sqlite。

        业务层只关心驱动类型（用于构造 driver-specific 选项）；
        真实 driver 由 SQLAlchemy + 依赖列表里的 psycopg2-binary / pymysql 解析。
        """
        u = self.his_db_url
        if u.startswith(("postgresql", "postgres")):
            return "postgresql"
        if u.startswith("mysql"):
            return "mysql"
        if u.startswith("sqlite"):
            return "sqlite"
        raise ValueError(
            f"HIS_DB_URL 必须以 postgresql / mysql / sqlite 开头（实测：{u[:24]}…）"
        )

    def validate_for_runtime(self) -> None:
        """启动期强制校验：缺关键凭据显式抛错，绝不静默兜底。

        由 FastAPI lifespan 调一次；本地 SQLite 替身允许在 app_env=test/local 时临时充当
        DATABASE_URL（spec.md §8 "真实接口 vs 测试替身"）。
        """
        problems: list[str] = []

        # HIS_DB_URL 不允许占位符
        if not self.his_db_url or "CHANGE_ME" in self.his_db_url:
            problems.append("HIS_DB_URL 缺失或仍是占位符（spec.md §8）")
        if not self.his_db_user:
            problems.append("HIS_DB_USER 缺失")
        if not self.his_db_pass or "CHANGE_ME" in self.his_db_pass:
            problems.append("HIS_DB_PASS 缺失或仍是占位符")

        # 单病种上报
        if not self.dipcase_reporter_url:
            problems.append("DIPCASE_REPORTER_URL 缺失")
        if not self.dipcase_reporter_token or "CHANGE_ME" in self.dipcase_reporter_token:
            problems.append("DIPCASE_REPORTER_TOKEN 缺失或仍是占位符")

        # ModelGateway
        if not self.openai_base_url:
            problems.append("OPENAI_BASE_URL 缺失")
        if not self.openai_api_key or "CHANGE_ME" in self.openai_api_key:
            problems.append("OPENAI_API_KEY 缺失或仍是占位符")

        # 项目自带 DB（仅本地 SQLite 替身允许使用 file://）
        if not self.database_url:
            problems.append("DATABASE_URL 缺失")
        elif (
            self.database_url.startswith("sqlite")
            and self.app_env not in ("local", "staging", "test")
        ):
            problems.append("DATABASE_URL=sqlite 仅 local / staging / test 允许")

        if problems:
            quoted = '缺关键凭据显式报错，绝不静默兜底'
            raise RuntimeError(
                "启动校验失败（spec.md §8 " + quoted + "）：\n  - "
                + "\n  - ".join(problems)
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """配置单例（lru_cache 保证只解析一次 env）。"""
    try:
        return Settings()
    except ValidationError as e:  # pragma: no cover
        # 让上游（lifespan / 测试夹具）拿到具体字段错误
        raise RuntimeError(
            f"配置解析失败：{e}\n"
            "请检查 .env 是否就位、字段是否拼写正确。"
        ) from e


def reset_settings_cache() -> None:
    """测试 / Phase 4 启动脚本专用：清掉 lru_cache 强制重新解析 env。"""
    get_settings.cache_clear()


# ---- 自检入口（供 tests/test_config.py 与 scripts 复用） ----
if __name__ == "__main__":  # pragma: no cover
    import json
    import sys

    try:
        s = get_settings()
        s.validate_for_runtime()
    except RuntimeError as e:
        print(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False))
        sys.exit(2)

    print(
        json.dumps(
            {
                "ok": True,
                "app_env": s.app_env,
                "his_driver": s.driver_kind(),
                "database_kind": "sqlite" if s.database_url.startswith("sqlite") else "postgresql",
                "openai_model": s.openai_model,
                "min_sample_size": s.min_sample_size,
            },
            ensure_ascii=False,
        )
    )