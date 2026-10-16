"""字段映射配置层测试（spec §3.1 + §6）。

覆盖：
  - default / by_disease_code 加载合并逻辑
  - disease_code 专属字段覆盖 default
  - 必填字段映射存在 + column 不为空 + spec_section 标注
  - display_blocked_fields 集合（PII 白名单）含 patient_name / id_card / phone 等
  - 必填字段与 PII 白名单不冲突
  - 缺字段显式 FieldMappingConfigError（不静默）
  - is_blocked_field / filter_projection 工作正确
  - Type / Value / JSON 损坏 / 文件缺失 → 显式报错路径
  - 字段映射与 spec §3.1 spec_required_field_names 一致
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# 让 `from app.xxx import ...` 工作（与 conftest.py 同步）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.adapters.field_mapping import (  # noqa: E402
    FieldMapping,
    FieldMappingConfigError,
    FieldMappingRegistry,
    load_field_mapping_registry,
)

# ---- 真实配置路径（默认）----
DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "app" / "field_mapping_config" / "field_mapping.json"

# ---- spec §3.1 必填字段清单（来源 spec）----
SPEC_REQUIRED_FIELDS = [
    "encounter_id",
    "primary_diagnosis_code",
    "treatment_strategy",
    "age_band",
    "admission_route",
    "los_days",
]

# ---- spec §6 PII 白名单（不变量）----
EXPECTED_PII_BLOCKLIST = {"patient_name", "id_card", "phone"}


# ============================================================================
# 1. 真实配置加载
# ============================================================================

def test_load_default_registry_runs():
    """真实配置文件可加载 default 视图。"""
    reg = load_field_mapping_registry(disease_code=None, validate=True)
    assert isinstance(reg, FieldMappingRegistry)
    assert reg.disease_code is None
    assert reg.config_path == DEFAULT_CONFIG_PATH


def test_load_default_registry_required_fields_present():
    """default 视图必填字段映射齐。"""
    reg = load_field_mapping_registry(disease_code=None, validate=True)
    for name in SPEC_REQUIRED_FIELDS:
        assert name in reg.fields, f"必填字段 {name} 缺失"
        assert reg.fields[name].required is True, f"{name} 应标 required=true"


def test_default_display_blocked_fields_contains_spec_pii():
    """default PII 白名单必须包含 spec §6 红线字段。"""
    reg = load_field_mapping_registry(disease_code=None, validate=True)
    for pii in EXPECTED_PII_BLOCKLIST:
        assert reg.is_blocked_field(pii) is True, f"{pii} 应在白名单"


# ============================================================================
# 2. disease_code 视图
# ============================================================================

@pytest.mark.parametrize("disease_code", ["I21", "I63", "Z96.10"])
def test_load_per_disease_registry(disease_code):
    """3 病种逐一加载 + 必填校验。"""
    reg = load_field_mapping_registry(disease_code=disease_code, validate=True)
    assert reg.disease_code == disease_code
    for name in SPEC_REQUIRED_FIELDS:
        assert name in reg.fields
        assert reg.fields[name].required is True


def test_disease_specific_overrides_default():
    """by_disease_code 段字段覆盖 default 段。"""
    reg = load_field_mapping_registry(disease_code="I21", validate=True)
    f = reg.fields["treatment_strategy"]
    assert f.values == ("PCI", "thrombolysis", "conservative")


def test_disease_specific_keeps_default_fields_not_overridden():
    """默认字段未显式覆盖时仍保留。"""
    reg = load_field_mapping_registry(disease_code="I21", validate=True)
    assert "los_days" in reg.fields
    assert reg.fields["los_days"].required is True
    assert "cmi" in reg.fields


# ============================================================================
# 3. 校验与启动期报错
# ============================================================================

def test_validate_or_raise_passes_for_real_config():
    """真实配置 validate_or_raise 不抛异常。"""
    reg = load_field_mapping_registry(disease_code="I21", validate=True)
    reg.validate_or_raise()  # 二次校验


def test_missing_required_field_raises(tmp_path):
    """缺必填字段（删 encounter_id 映射）→ FieldMappingConfigError。"""
    cfg = {
        "default": {
            "fields": {
                # encounter_id missing on purpose
                "primary_diagnosis_code": {
                    "column": "primary_diagnosis_code",
                    "type": "string",
                    "required": True,
                },
                "treatment_strategy": {
                    "column": "treatment_strategy",
                    "type": "string",
                    "required": True,
                },
                "age_band": {
                    "column": "age_band",
                    "type": "int",
                    "required": True,
                },
                "admission_route": {
                    "column": "admission_route",
                    "type": "enum",
                    "values": ["1", "2", "3", "9"],
                    "required": True,
                },
                "los_days": {
                    "column": "los_days",
                    "type": "int",
                    "required": True,
                },
            },
            "display_blocked_fields": ["patient_name"],
        },
        "_required_field_summary": {
            "required_field_names": SPEC_REQUIRED_FIELDS,
        },
    }
    p = tmp_path / "no_enc.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(FieldMappingConfigError, match="encounter_id"):
        load_field_mapping_registry(
            disease_code=None,
            config_path=p,
            validate=True,
        )


def test_required_field_without_column_raises(tmp_path):
    """必填字段缺 column 字段映射 → FieldMappingConfigError。"""
    cfg = {
        "default": {
            "fields": {
                "encounter_id": {
                    # column missing on purpose
                    "type": "string",
                    "required": True,
                },
                **{name: {"column": name, "type": "string", "required": True}
                   for name in SPEC_REQUIRED_FIELDS[1:]},
            },
            "display_blocked_fields": [],
        },
        "_required_field_summary": {"required_field_names": SPEC_REQUIRED_FIELDS},
    }
    p = tmp_path / "no_col.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(FieldMappingConfigError, match="column"):
        load_field_mapping_registry(config_path=p, validate=True)


def test_pii_conflicts_with_required_raises(tmp_path):
    """必填字段被标进 display_blocked_fields → 启动期显式报错。"""
    cfg = {
        "default": {
            "fields": {
                **{name: {"column": name, "type": "string", "required": True,
                          "spec_section": "§3.1 必填"}
                   for name in SPEC_REQUIRED_FIELDS},
            },
            "display_blocked_fields": ["patient_name", "id_card", "encounter_id"],
        },
        "_required_field_summary": {"required_field_names": SPEC_REQUIRED_FIELDS},
    }
    p = tmp_path / "conflict.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(FieldMappingConfigError, match="encounter_id"):
        load_field_mapping_registry(config_path=p, validate=True)


def test_missing_spec_section_raises(tmp_path):
    """必填字段缺 spec_section → 启动期显式报错。"""
    cfg = {
        "default": {
            "fields": {
                **{name: {"column": name, "type": "string", "required": True}
                   for name in SPEC_REQUIRED_FIELDS},
            },
            "display_blocked_fields": [],
        },
        "_required_field_summary": {"required_field_names": SPEC_REQUIRED_FIELDS},
    }
    p = tmp_path / "no_spec.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(FieldMappingConfigError, match="spec_section"):
        load_field_mapping_registry(config_path=p, validate=True)


# ============================================================================
# 4. JSON / 文件系统异常
# ============================================================================

def test_missing_file_raises(tmp_path):
    """配置文件不存在 → FieldMappingConfigError。"""
    p = tmp_path / "absent.json"
    with pytest.raises(FieldMappingConfigError, match="不存在"):
        load_field_mapping_registry(config_path=p, validate=True)


def test_bad_json_raises(tmp_path):
    """JSON 损坏 → FieldMappingConfigError（不静默）。"""
    p = tmp_path / "broken.json"
    p.write_text("{ not json", encoding="utf-8")
    with pytest.raises(FieldMappingConfigError, match="JSON 解析失败"):
        load_field_mapping_registry(config_path=p, validate=True)


def test_top_level_not_object_raises(tmp_path):
    """顶层不是 object → FieldMappingConfigError。"""
    p = tmp_path / "list.json"
    p.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    with pytest.raises(FieldMappingConfigError, match="object"):
        load_field_mapping_registry(config_path=p, validate=True)


def test_default_section_missing_raises(tmp_path):
    """缺 default 段 → FieldMappingConfigError。"""
    p = tmp_path / "no_default.json"
    p.write_text(json.dumps({"by_disease_code": {}}), encoding="utf-8")
    with pytest.raises(FieldMappingConfigError, match="default"):
        load_field_mapping_registry(config_path=p, validate=True)


def test_field_with_bad_type_raises(tmp_path):
    """字段 type 非法 → FieldMappingConfigError。"""
    cfg = {
        "default": {
            "fields": {
                "encounter_id": {
                    "column": "encounter_id",
                    "type": "this_is_not_a_real_type",
                    "required": True,
                },
            },
            "display_blocked_fields": [],
        },
        "_required_field_summary": {"required_field_names": ["encounter_id"]},
    }
    p = tmp_path / "bad_type.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(FieldMappingConfigError, match="type"):
        load_field_mapping_registry(config_path=p, validate=True)


# ============================================================================
# 5. 字段查询与 PII 过滤
# ============================================================================

def test_required_columns_method():
    """required_columns() 返回必填字段对应的物理列名。"""
    reg = load_field_mapping_registry(disease_code="I21", validate=True)
    cols = reg.required_columns()
    assert set(cols) == set(SPEC_REQUIRED_FIELDS)


def test_is_blocked_field_logical_and_physical():
    """PII 白名单对 logical / physical name 两种入口都生效。"""
    reg = load_field_mapping_registry(disease_code="I21", validate=True)
    assert reg.is_blocked_field("patient_name") is True
    assert reg.is_blocked_field("id_card") is True
    assert reg.is_blocked_field("phone") is True
    # 必填字段不被拦截
    assert reg.is_blocked_field("encounter_id") is False
    assert reg.is_blocked_field("los_days") is False
    # 未知字段不被拦截
    assert reg.is_blocked_field("random_unknown") is False
    # 空值兜底
    assert reg.is_blocked_field("") is False


def test_filter_projection_strips_pii():
    """filter_projection() 按白名单剔除 PII 字段。"""
    reg = load_field_mapping_registry(disease_code="I21", validate=True)
    cols = [
        "encounter_id",
        "los_days",
        "patient_name",   # PII
        "id_card",        # PII
        "phone",          # PII
        "cmi",
    ]
    cleaned = reg.filter_projection(cols)
    assert "patient_name" not in cleaned
    assert "id_card" not in cleaned
    assert "phone" not in cleaned
    assert "encounter_id" in cleaned
    assert "los_days" in cleaned
    assert "cmi" in cleaned


def test_get_column_returns_physical_name():
    """get_column('encounter_id') 返回物理列名 'encounter_id'。"""
    reg = load_field_mapping_registry(disease_code="I21", validate=True)
    assert reg.get_column("encounter_id") == "encounter_id"
    assert reg.get_column("not_existing") is None


# ============================================================================
# 6. FieldMapping dataclass 自身约束
# ============================================================================

def test_field_mapping_invalid_type_raises():
    """FieldMapping.__post_init__ 对非法 type 显式报错。"""
    with pytest.raises(FieldMappingConfigError, match="type"):
        FieldMapping(
            logical_name="x",
            column="x",
            type="not_a_type",
            required=True,
        )


def test_field_mapping_required_without_column_raises():
    """required=True 但 column 为空 → FieldMappingConfigError。"""
    with pytest.raises(FieldMappingConfigError, match="column"):
        FieldMapping(
            logical_name="x",
            column="",
            type="string",
            required=True,
        )


# ============================================================================
# 7. data fixture / schema 对齐
# ============================================================================

@pytest.mark.parametrize("disease_code", ["I21", "I63", "Z96.10"])
def test_field_mapping_matches_fixture_columns(fixture_for_disease, disease_code):
    """字段映射列名与 fixture CSV 表头一致（防 spec ↔ fixture 偏移）。"""
    df = fixture_for_disease(disease_code)
    reg = load_field_mapping_registry(disease_code=disease_code, validate=True)

    # 必填字段对应的物理列必须出现在 fixture 表头里
    for col in reg.required_columns():
        assert col in df.columns, (
            f"{disease_code}: 必填列 {col} 在 fixture 中缺失（spec §3.1 ↔ fixture 偏移）"
        )

    # PII 字段不在 fixture 表头里（防 fixture 误带 PII）
    df_cols = set(df.columns)
    blocked = set(reg.display_blocked_fields)
    leak = df_cols & blocked
    assert not leak, f"{disease_code}: fixture 含 PII 列 {leak}（spec §6 违规）"
