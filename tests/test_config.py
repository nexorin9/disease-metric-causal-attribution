"""app/config.py 单测：环境变量加载、字段校验、缺凭据报错、driver 推断。

凭据位一律从环境变量读取；本测试用 monkeypatch 注入 env，避免依赖真实 .env。
"""

from __future__ import annotations

import os

import pytest

from app import config as cfg


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    """每次测试都用全新 env；lru_cache 必须清掉。"""
    cfg.reset_settings_cache()
    yield
    cfg.reset_settings_cache()


def _set_min_env(monkeypatch):
    monkeypatch.setenv("HIS_DB_URL", "postgresql+psycopg2://his:pwd@10.0.0.5:5432/his")
    monkeypatch.setenv("HIS_DB_USER", "his")
    monkeypatch.setenv("HIS_DB_PASS", "pwd")
    monkeypatch.setenv("DIPCASE_REPORTER_URL", "https://dipcase.example.cn/api")
    monkeypatch.setenv("DIPCASE_REPORTER_TOKEN", "tok-xyz")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///./local_state.db")


def test_get_settings_min_env(monkeypatch):
    _set_min_env(monkeypatch)
    s = cfg.get_settings()
    assert s.his_db_user == "his"
    assert s.openai_model == "gpt-4o-mini"  # 默认值
    assert s.min_sample_size == 30
    assert s.max_ci_width == 0.6


def test_driver_kind_postgresql(monkeypatch):
    _set_min_env(monkeypatch)
    monkeypatch.setenv("HIS_DB_URL", "postgresql+psycopg2://his:pwd@h:5432/d")
    s = cfg.get_settings()
    assert s.driver_kind() == "postgresql"


def test_driver_kind_mysql(monkeypatch):
    _set_min_env(monkeypatch)
    monkeypatch.setenv("HIS_DB_URL", "mysql+pymysql://his:pwd@h:3306/d")
    s = cfg.get_settings()
    assert s.driver_kind() == "mysql"


def test_driver_kind_unknown_raises(monkeypatch):
    _set_min_env(monkeypatch)
    monkeypatch.setenv("HIS_DB_URL", "mongodb://nope")
    s = cfg.get_settings()
    with pytest.raises(ValueError, match="postgresql / mysql / sqlite"):
        s.driver_kind()


def test_validate_for_runtime_ok(monkeypatch):
    _set_min_env(monkeypatch)
    s = cfg.get_settings()
    s.validate_for_runtime()  # 不应抛


def test_validate_for_runtime_missing_his_pass(monkeypatch):
    _set_min_env(monkeypatch)
    monkeypatch.setenv("HIS_DB_PASS", "CHANGE_ME")
    s = cfg.get_settings()
    with pytest.raises(RuntimeError, match="HIS_DB_PASS"):
        s.validate_for_runtime()


def test_validate_for_runtime_missing_token(monkeypatch):
    _set_min_env(monkeypatch)
    monkeypatch.setenv("DIPCASE_REPORTER_TOKEN", "")
    s = cfg.get_settings()
    with pytest.raises(RuntimeError, match="DIPCASE_REPORTER_TOKEN"):
        s.validate_for_runtime()


def test_validate_for_runtime_missing_openai_key(monkeypatch):
    _set_min_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "CHANGE_ME")
    s = cfg.get_settings()
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        s.validate_for_runtime()


def test_validate_for_runtime_missing_his_url(monkeypatch):
    _set_min_env(monkeypatch)
    monkeypatch.setenv("HIS_DB_URL", "")
    s = cfg.get_settings()
    with pytest.raises(RuntimeError, match="HIS_DB_URL"):
        s.validate_for_runtime()


def test_validate_for_runtime_sqlite_blocked_in_prod(monkeypatch):
    _set_min_env(monkeypatch)
    monkeypatch.setenv("APP_ENV", "prod")
    s = cfg.get_settings()
    with pytest.raises(RuntimeError, match="sqlite.*local / staging / test"):
        s.validate_for_runtime()


def test_min_sample_size_validator():
    # field_validator 是 Settings 内部；通过构造实例触发
    with pytest.raises(ValueError, match="min_sample_size"):
        cfg.Settings(
            his_db_url="postgresql://x",
            his_db_user="x",
            his_db_pass="x",
            dipcase_reporter_url="https://x",
            dipcase_reporter_token="x",
            openai_base_url="https://x",
            openai_api_key="x",
            database_url="sqlite:///./x.db",
            min_sample_size=0,
        )


def test_get_settings_singleton(monkeypatch):
    _set_min_env(monkeypatch)
    a = cfg.get_settings()
    b = cfg.get_settings()
    assert a is b  # lru_cache 单例


def test_reset_settings_cache(monkeypatch):
    _set_min_env(monkeypatch)
    a = cfg.get_settings()
    cfg.reset_settings_cache()
    monkeypatch.setenv("HIS_DB_USER", "different")
    b = cfg.get_settings()
    assert a.his_db_user == "his"
    assert b.his_db_user == "different"


def test_optional_openai_stub_url(monkeypatch):
    _set_min_env(monkeypatch)
    monkeypatch.setenv("OPENAI_STUB_URL", "http://127.0.0.1:18888")
    s = cfg.get_settings()
    assert s.openai_stub_url == "http://127.0.0.1:18888"