"""脱敏 fixture 加载器测试（data/seed_loader.py）。

覆盖：
  - 3 个 fixture CSV 可读 + 行数 = 200
  - 字段对齐 spec.md §3.1
  - encounter_id 格式（enc_<12hex>）唯一
  - 类型还原（los_days / cmi / admission_route / readmit_flag / complication_flag）
  - 不含 spec §8 display_blocked_fields（patient_name / id_card / phone / address）
  - 未知 disease_code / 缺失字段 / 缺 encounter_id 文件名 / 含禁字段 显式报错
  - list_available_diseases 返回磁盘已存在病种
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data.seed_loader import (
    DATA_DIR,
    DISPLAY_BLOCKED_FIELDS,
    FIXTURE_FILES,
    REQUIRED_COLUMNS,
    list_available_diseases,
    load_fixture_csv,
)


EXPECTED_DISEASES = ("I21", "I63", "Z96.10")
EXPECTED_ROW_COUNT = 200


# ---- 字段 / 行数 ---------------------------------------------------------

class TestFixtureStructure:
    """3 fixture CSV 的结构与 spec.md §3.1 字段映射。"""

    @pytest.mark.parametrize("disease_code", list(EXPECTED_DISEASES))
    def test_fixture_row_count(self, disease_code: str):
        df = load_fixture_csv(disease_code)
        assert df.shape[0] == EXPECTED_ROW_COUNT, (
            f"{disease_code} 行数应为 {EXPECTED_ROW_COUNT}, 实测 {df.shape[0]}"
        )

    @pytest.mark.parametrize("disease_code", list(EXPECTED_DISEASES))
    def test_fixture_required_columns(self, disease_code: str):
        df = load_fixture_csv(disease_code)
        for col in REQUIRED_COLUMNS:
            assert col in df.columns, f"{disease_code} 缺字段 {col}"

    @pytest.mark.parametrize("disease_code", list(EXPECTED_DISEASES))
    def test_fixture_disease_code_column_matches(
        self, disease_code: str
    ):
        df = load_fixture_csv(disease_code)
        assert (df["disease_code"] == disease_code).all(), (
            f"{disease_code} 的 disease_code 列出现异质值"
        )


# ---- encounter_id 格式 / 唯一性 -------------------------------------------

class TestEncounterId:
    """encounter_id 格式（enc_<12hex>） + 全局唯一 + 无真实住院号。"""

    @pytest.mark.parametrize("disease_code", list(EXPECTED_DISEASES))
    def test_encounter_id_format(self, disease_code: str):
        df = load_fixture_csv(disease_code)
        pattern = df["encounter_id"].str.match(r"^enc_[0-9a-f]{12}$")
        assert pattern.all(), (
            f"{disease_code} encounter_id 不符合 enc_<12hex> 格式"
        )

    def test_encounter_id_unique_across_diseases(
        self, all_fixtures: dict
    ):
        """跨 3 病种 encounter_id 必须全局唯一（即使 hash 输入带 disease_code 区分）。"""
        all_ids = pd.concat(
            [df["encounter_id"] for df in all_fixtures.values()],
            ignore_index=True,
        )
        assert all_ids.is_unique, "encounter_id 在 3 病种间不唯一"


# ---- 类型还原 -----------------------------------------------------------

class TestTypeCoercion:
    """DataFrame 类型从 string 还原为 int / float / bool。"""

    @pytest.mark.parametrize("disease_code", list(EXPECTED_DISEASES))
    def test_los_days_is_int(self, disease_code: str):
        df = load_fixture_csv(disease_code)
        assert df["los_days"].dtype.kind == "i", (
            f"{disease_code} los_days 应为 int"
        )

    @pytest.mark.parametrize("disease_code", list(EXPECTED_DISEASES))
    def test_cmi_is_float(self, disease_code: str):
        df = load_fixture_csv(disease_code)
        assert df["cmi"].dtype.kind == "f", (
            f"{disease_code} cmi 应为 float"
        )

    @pytest.mark.parametrize("disease_code", list(EXPECTED_DISEASES))
    def test_admission_route_is_int(self, disease_code: str):
        df = load_fixture_csv(disease_code)
        assert df["admission_route"].dtype.kind == "i"
        # 实际临床分布：入院途径 ∈ {1, 2, 3, 9}（spec §3.1 enum）
        assert set(df["admission_route"].unique()).issubset({1, 2, 3, 9})

    @pytest.mark.parametrize("disease_code", list(EXPECTED_DISEASES))
    def test_readmit_flag_is_bool(self, disease_code: str):
        df = load_fixture_csv(disease_code)
        assert df["readmit_flag"].dtype == bool

    @pytest.mark.parametrize("disease_code", list(EXPECTED_DISEASES))
    def test_complication_flag_is_bool(self, disease_code: str):
        df = load_fixture_csv(disease_code)
        assert df["complication_flag"].dtype == bool


# ---- 临床合理范围 ---------------------------------------------------------

class TestClinicalRanges:
    """los_days / cmi / 年龄段 / ICD-10 长度均在临床合理范围（spec §3.1）。"""

    @pytest.mark.parametrize("disease_code", list(EXPECTED_DISEASES))
    def test_los_days_in_range(self, disease_code: str):
        df = load_fixture_csv(disease_code)
        # 急性心梗：4-22；脑梗：7-28；髋置换：10-25（含并发症加成）
        assert (df["los_days"] >= 1).all() and (df["los_days"] <= 35).all()

    @pytest.mark.parametrize("disease_code", list(EXPECTED_DISEASES))
    def test_cmi_in_range(self, disease_code: str):
        df = load_fixture_csv(disease_code)
        assert (df["cmi"] >= 0.5).all() and (df["cmi"] <= 5.0).all()

    @pytest.mark.parametrize("disease_code", list(EXPECTED_DISEASES))
    def test_primary_diagnosis_code_length(self, disease_code: str):
        """spec §3.1 主要诊断 ICD-10 编码 string(6) — 不超过 6 字符。"""
        df = load_fixture_csv(disease_code)
        assert (df["primary_diagnosis_code"].str.len() <= 6).all(), (
            f"{disease_code} ICD-10 编码超过 6 字符"
        )


# ---- spec §8 反拟真纪律 --------------------------------------------------

class TestDeidentification:
    """fixture 数据集不含 spec §8 display_blocked_fields。"""

    @pytest.mark.parametrize("disease_code", list(EXPECTED_DISEASES))
    def test_no_pii_fields(self, disease_code: str):
        df = load_fixture_csv(disease_code)
        leaked = DISPLAY_BLOCKED_FIELDS & set(df.columns)
        assert not leaked, (
            f"{disease_code} 含禁字段: {leaked}"
        )

    @pytest.mark.parametrize("disease_code", list(EXPECTED_DISEASES))
    def test_no_real_patient_id_pattern(self, disease_code: str):
        """禁出现易被误认为真实住院号 / 处方号 / 批次号的连续 8 位以上数字串。"""
        df = load_fixture_csv(disease_code)
        # encounter_id 格式为 enc_<12hex>，不含连续 8+ 位纯数字
        for col in df.columns:
            if df[col].dtype == object:
                for value in df[col].dropna().astype(str):
                    # 排除 enc_ 前缀，其余列也基本不含纯长数字串
                    if col == "encounter_id":
                        continue
                    # 主要诊断 ICD-10 / 入院途径短数字均安全；住院天数是 int 不走这里
                    assert not (
                        value.isdigit() and len(value) >= 8
                    ), f"{disease_code} 列 {col} 含疑似真实编号: {value}"


# ---- 错误路径 -----------------------------------------------------------

class TestErrorPaths:
    """显式报错：未知 disease_code / fixture 缺必填字段 / 含禁字段。"""

    def test_unknown_disease_code_raises_value_error(self):
        with pytest.raises(ValueError, match="未知 disease_code"):
            load_fixture_csv("ZZZ.99")

    def test_list_available_diseases_returns_three(self):
        available = list_available_diseases()
        assert set(EXPECTED_DISEASES).issubset(set(available)), (
            f"应至少含 3 病种, 实测: {available}"
        )

    def test_missing_fixture_file_raises_file_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """当 fixture 文件缺失时显式 FileNotFoundError。"""
        # 把 DATA_DIR 指向临时空目录（模拟 fixture 缺失）
        import data.seed_loader as loader

        monkeypatch.setattr(loader, "DATA_DIR", tmp_path)
        with pytest.raises(FileNotFoundError, match="fixture 不存在"):
            load_fixture_csv("I21")

    def test_missing_required_columns_raises_value_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """fixture 缺必填字段时显式 ValueError。"""
        # 写一个缺少 encounter_id 的伪 fixture
        bad_csv = tmp_path / "fixture_acute_mi.csv"
        bad_csv.write_text(
            "primary_diagnosis_code,treatment_strategy,age_band,cmi\n"
            "I21.0,PCI,55-64,2.5\n",
            encoding="utf-8",
        )
        import data.seed_loader as loader

        monkeypatch.setattr(loader, "DATA_DIR", tmp_path)
        with pytest.raises(ValueError, match="缺少必填字段"):
            load_fixture_csv("I21")

    def test_pii_field_in_fixture_raises_value_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """fixture 含禁字段时显式 ValueError。"""
        bad_csv = tmp_path / "fixture_acute_mi.csv"
        bad_csv.write_text(
            ",".join(REQUIRED_COLUMNS + ["patient_name"]) + "\n"
            + ",".join(["x"] * (len(REQUIRED_COLUMNS) + 1)) + "\n",
            encoding="utf-8",
        )
        import data.seed_loader as loader

        monkeypatch.setattr(loader, "DATA_DIR", tmp_path)
        with pytest.raises(ValueError, match="严禁展示字段"):
            load_fixture_csv("I21")


# ---- 常量完整性 ---------------------------------------------------------

def test_required_columns_includes_disease_code_and_encounter():
    """REQUIRED_COLUMNS 含 encounter_id + disease_code（跨病种合并 / 唯一性键）。"""
    assert "encounter_id" in REQUIRED_COLUMNS
    assert "disease_code" in REQUIRED_COLUMNS


def test_fixture_files_map_has_three_diseases():
    assert set(FIXTURE_FILES.keys()) == set(EXPECTED_DISEASES)


def test_data_dir_resolves_to_project_data_subdir():
    """DATA_DIR 必须是 data/ 子目录，而非 _PROJECT_ROOT 本身。"""
    assert DATA_DIR.name == "data"
    assert DATA_DIR.is_dir()