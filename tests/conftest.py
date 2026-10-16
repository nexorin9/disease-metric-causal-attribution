"""测试 conftest：脱敏 fixture 加载器 + 关键测试环境配置。

供 tests/test_*.py 复用：
  - fixture_for_disease：按 disease_code 返回脱敏样例 DataFrame
  - all_fixtures：3 个病种 fixture 全部加载（性能 / 反驳 / baseline 用）
  - reset_settings_cache：T2 引入的 pydantic-settings 单例重置
  - autouse fixture：环境变量兜底，避免真实凭据泄漏到测试
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterator

import pandas as pd
import pytest

# 让 `from app.xxx import ...` 与 `from data.xxx import ...` 在测试时能工作
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 测试环境变量兜底（不依赖外部 .env；任何凭据位缺失时显式警告）
os.environ.setdefault("DATABASE_URL", "sqlite:///./tests/.benchmarks/test_conftest.db")
os.environ.setdefault("HIS_DB_URL", "sqlite:///./tests/.benchmarks/his_fake.db")
os.environ.setdefault("HIS_DB_USER", "test_user")
os.environ.setdefault("HIS_DB_PASS", "test_pass_only_for_tests")
os.environ.setdefault("DIPCASE_REPORTER_URL", "http://localhost:18899/mock")
os.environ.setdefault("DIPCASE_REPORTER_TOKEN", "test_token_only_for_tests")
os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:18888/v1")
os.environ.setdefault("OPENAI_API_KEY", "test_openai_key_only_for_tests")
# 注意：OPENAI_MODEL / OPENAI_TIMEOUT 不在模块级别 setdefault，避免覆盖测试期望的默认值
# （test_get_settings_min_env 等依赖 pydantic-settings 默认值；OPENAI_MODEL 由 ModelGateway 自身读取）


@pytest.fixture(scope="session")
def project_root() -> Path:
    """项目子目录根路径。"""
    return _PROJECT_ROOT


@pytest.fixture(scope="session")
def fixture_data_dir(project_root: Path) -> Path:
    """脱敏 fixture CSV 目录。"""
    return project_root / "data"


@pytest.fixture
def fixture_for_disease():
    """按 disease_code 返回对应 fixture DataFrame。

    Returns
    -------
    Callable[[str], pd.DataFrame]
        接受 disease_code（'I21' / 'I63' / 'Z96.10'），返回 DataFrame
    """
    from data.seed_loader import load_fixture_csv

    def _load(disease_code: str) -> pd.DataFrame:
        return load_fixture_csv(disease_code)

    return _load


@pytest.fixture
def all_fixtures() -> dict:
    """3 病种 fixture 全部加载，返回 {disease_code: DataFrame}。"""
    from data.seed_loader import load_fixture_csv

    return {
        code: load_fixture_csv(code)
        for code in ("I21", "I63", "Z96.10")
    }


@pytest.fixture
def reset_settings_cache():
    """重置 pydantic-settings lru_cache 单例（app/config.py）。"""
    try:
        from app.config import reset_settings_cache
    except ImportError:
        yield
        return
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture(autouse=True)
def _test_env_isolation(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """自动隔离测试环境：保护敏感凭据位 + 防真实触网。

    行为：
      - HIS_DB_PASS / DIPCASE_REPORTER_TOKEN / OPENAI_API_KEY：若测试未显式 setenv 则注入安全替身
      - OPENAI_BASE_URL：若测试未显式 setenv 则指向本地协议替身（18888 端口）
      - OPENAI_MODEL / OPENAI_TIMEOUT：保持测试期望的默认值不被本 fixture 覆盖
    """
    # 仅在测试未显式设置时注入；monkeypatch.setenv 总是覆盖，所以用 os.environ 检测
    if "OPENAI_BASE_URL" not in os.environ:
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:18888/v1")
    if "OPENAI_API_KEY" not in os.environ:
        monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key_only_for_tests")
    if "HIS_DB_PASS" not in os.environ:
        monkeypatch.setenv("HIS_DB_PASS", "test_pass_only_for_tests")
    if "DIPCASE_REPORTER_TOKEN" not in os.environ:
        monkeypatch.setenv("DIPCASE_REPORTER_TOKEN", "test_token_only_for_tests")
    yield