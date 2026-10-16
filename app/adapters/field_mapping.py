"""字段映射配置层（spec.md §3.1 + §6 PII 白名单）。

设计约束：
  - 启动期校验必填字段映射缺失显式报错（spec §3.1 "字段映射以 JSON 配置文件维护，启动时校验必填字段；缺失项报错不静默"）。
  - 按 disease_code 加载独立映射；fallback 到 `default` 段。
  - `display_blocked_fields` 是 PII 白名单——HIS 库若含 patient_name / id_card / phone 等字段，
    HISAdapter 投影时强制剔除，绝不进 ORM / DataFrame / 审计日志。
  - 字段名不是列名——`fields[logical_name]` 才是 source-of-truth，
    `column` 是物理列，HIS 表头不一致时改 `column` 即可。

对外接口：
  - FieldMappingRegistry(disease_code) → 子集 + 默认合并视图
  - registry.fields → Dict[str, FieldMapping]
  - registry.required_field_names → List[str]
  - registry.display_blocked_fields → set[str]
  - registry.is_blocked_field(name) → bool
  - registry.get(logical_name) → FieldMapping | None
  - registry.validate_or_raise() → 校验启动期必填字段映射；缺字段 → FieldMappingConfigError
  - FieldMappingRegistry.load_default() → 加载 default 视图（无 disease_code 时）

config/field_mapping.json 是真源；本模块为加载 + 校验 + 拦截逻辑。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence


class FieldMappingConfigError(ValueError):
    """字段映射配置错误：缺字段、类型错、JSON 坏——启动期显式抛出，绝不静默兜底。"""


@dataclass(frozen=True)
class FieldMapping:
    """单个逻辑字段到物理列的映射声明（启动期从 JSON 加载）。"""

    logical_name: str
    column: str
    type: str
    required: bool
    max_length: Optional[int] = None
    values: Optional[Sequence[str]] = None
    value_prefix: Optional[str] = None
    semantic: str = ""
    ref: Optional[str] = None
    spec_section: Optional[str] = None

    def __post_init__(self) -> None:
        if self.type not in ("string", "int", "decimal", "bool", "enum", "string[]"):
            raise FieldMappingConfigError(
                f"字段 {self.logical_name} 的 type 非法: {self.type!r} "
                "（仅允许 string / int / decimal / bool / enum / string[]）"
            )
        if self.required and not self.column:
            raise FieldMappingConfigError(
                f"必填字段 {self.logical_name} 必须有 column 映射"
            )


@dataclass(frozen=True)
class FieldMappingRegistry:
    """字段映射视图（disease_code 专属 + default 合并）。

    frozen=True：实例构造后不可变；HISAdapter 调用本对象的字段视为只读。
    """

    disease_code: Optional[str]
    fields: Dict[str, FieldMapping] = field(default_factory=dict)
    required_field_names: List[str] = field(default_factory=list)
    display_blocked_fields: FrozenSet[str] = field(default_factory=frozenset)
    config_path: Optional[Path] = None

    def get(self, logical_name: str) -> Optional[FieldMapping]:
        return self.fields.get(logical_name)

    def get_column(self, logical_name: str) -> Optional[str]:
        f = self.get(logical_name)
        return f.column if f else None

    def required_columns(self) -> List[str]:
        """必填字段对应的物理列名（启动期 + 运行时校验共用）。"""
        return [f.column for f in self.fields.values() if f.required]

    def is_blocked_field(self, physical_or_logical_name: str) -> bool:
        """检测字段是否在 PII 白名单内。

        支持 logical name 与 physical column name 两种入口；
        HISAdapter 在投影阶段用本方法剔除 PII 字段。
        """
        if not physical_or_logical_name:
            return False
        return physical_or_logical_name in self.display_blocked_fields

    def filter_projection(self, columns: Sequence[str]) -> List[str]:
        """剔除 PII 字段，返回干净的列名列表（HISAdapter 投影层用）。"""
        return [c for c in columns if not self.is_blocked_field(c)]

    def required_field_names_spec(self) -> List[str]:
        """spec §3.1 必填字段清单（logical name 视角，**不**按 disease_code 个性化）。"""
        # 由 _load_registry 在合并时取 config._required_field_summary.required_field_names
        # 本方法仅在配置里"经验性"补足：上面 list 是兜底；正常配置已在 __init__ 注入
        return list(self.required_field_names)

    @staticmethod
    def _coerce_field(logical_name: str, raw: Dict[str, Any]) -> FieldMapping:
        """构造 FieldMapping；JSON 中缺失的关键字段显式报错。"""
        if "column" not in raw:
            raise FieldMappingConfigError(
                f"字段 {logical_name} 缺 column 配置（spec §3.1 字段映射）"
            )
        if "type" not in raw:
            raise FieldMappingConfigError(
                f"字段 {logical_name} 缺 type 配置"
            )
        try:
            return FieldMapping(
                logical_name=logical_name,
                column=raw["column"],
                type=raw["type"],
                required=bool(raw.get("required", False)),
                max_length=raw.get("max_length"),
                values=tuple(raw["values"]) if "values" in raw else None,
                value_prefix=raw.get("value_prefix"),
                semantic=raw.get("semantic", ""),
                ref=raw.get("ref"),
                spec_section=raw.get("spec_section"),
            )
        except FieldMappingConfigError:
            raise
        except Exception as e:
            raise FieldMappingConfigError(
                f"字段 {logical_name} 解析失败: {e}"
            ) from e

    @classmethod
    def _load_registry(
        cls,
        config_path: Path,
        disease_code: Optional[str] = None,
    ) -> "FieldMappingRegistry":
        """从 JSON 加载；disease_code 专属映射覆盖 default；缺字段显式报错。"""
        if not config_path.exists():
            raise FieldMappingConfigError(
                f"字段映射配置文件不存在: {config_path}（spec §3.1：字段映射以 JSON 配置文件维护）"
            )
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise FieldMappingConfigError(
                f"字段映射 JSON 解析失败: {e}"
            ) from e

        if not isinstance(raw, dict):
            raise FieldMappingConfigError(
                f"字段映射 JSON 顶层必须是 object，实际为 {type(raw).__name__}"
            )
        if "default" not in raw or not isinstance(raw["default"], dict):
            raise FieldMappingConfigError(
                "字段映射 JSON 缺 default 段（spec §3.1 '默认映射'）"
            )

        default_section = raw["default"]
        default_fields_raw = default_section.get("fields", {})
        if not isinstance(default_fields_raw, dict) or not default_fields_raw:
            raise FieldMappingConfigError(
                "字段映射 default.fields 必须是非空对象（spec §3.1）"
            )

        # ---- default 字段映射加载 ----
        merged_fields: Dict[str, FieldMapping] = {}
        for logical_name, field_raw in default_fields_raw.items():
            if not isinstance(field_raw, dict):
                raise FieldMappingConfigError(
                    f"字段 {logical_name} 配置项必须是 object"
                )
            merged_fields[logical_name] = cls._coerce_field(logical_name, field_raw)

        # ---- by_disease_code 覆盖 ----
        disease_section: Optional[Dict[str, Any]] = None
        if disease_code is not None and "by_disease_code" in raw:
            by_disease = raw["by_disease_code"]
            if isinstance(by_disease, dict) and disease_code in by_disease:
                disease_section = by_disease[disease_code]
                if not isinstance(disease_section, dict):
                    raise FieldMappingConfigError(
                        f"by_disease_code[{disease_code}] 必须是 object"
                    )
                disease_fields_raw = disease_section.get("fields", {})
                if not isinstance(disease_fields_raw, dict):
                    raise FieldMappingConfigError(
                        f"by_disease_code[{disease_code}].fields 必须是 object"
                    )
                # 默认未指定、且 _inherit_from_default=True 的字段保持 default；
                # disease_fields_raw 中显式给出的字段覆盖 default。
                for logical_name, field_raw in disease_fields_raw.items():
                    if logical_name.startswith("_"):
                        continue  # _comment / _inherit_from_default 等元数据
                    if not isinstance(field_raw, dict):
                        raise FieldMappingConfigError(
                            f"by_disease_code[{disease_code}].{logical_name} 必须是 object"
                        )
                    merged_fields[logical_name] = cls._coerce_field(logical_name, field_raw)

        # ---- 必填字段清单：来源 = spec §3.1（JSON _required_field_summary）----
        required_summary = raw.get("_required_field_summary", {})
        required_field_names = required_summary.get("required_field_names", [])
        if not isinstance(required_field_names, list) or not required_field_names:
            # 兜底：取 default 段 required=true 的字段
            required_field_names = [
                name for name, fm in merged_fields.items() if fm.required
            ]
        # 全部转成字符串（防 JSON 数字混入）
        required_field_names = [str(n) for n in required_field_names]

        # ---- display_blocked_fields：来源 = default + 可选 by_disease_code 追加 ----
        blocked_default = default_section.get("display_blocked_fields", [])
        if not isinstance(blocked_default, list):
            raise FieldMappingConfigError(
                "default.display_blocked_fields 必须是 array"
            )
        blocked_set = set(str(x) for x in blocked_default)
        if disease_section is not None:
            blocked_disease = disease_section.get("display_blocked_fields", [])
            if not isinstance(blocked_disease, list):
                raise FieldMappingConfigError(
                    f"by_disease_code[{disease_code}].display_blocked_fields 必须是 array"
                )
            blocked_set.update(str(x) for x in blocked_disease)
        display_blocked_fields = frozenset(blocked_set)

        return cls(
            disease_code=disease_code,
            fields=merged_fields,
            required_field_names=required_field_names,
            display_blocked_fields=display_blocked_fields,
            config_path=config_path,
        )

    @classmethod
    def for_disease(cls, disease_code: str, config_path: Optional[Path] = None) -> "FieldMappingRegistry":
        """加载 disease_code 专属 + default 合并视图。"""
        return cls._load_registry(
            config_path=cls._resolve_config_path(config_path),
            disease_code=disease_code,
        )

    @classmethod
    def load_default(cls, config_path: Optional[Path] = None) -> "FieldMappingRegistry":
        """仅加载 default 视图（无 disease_code 时）。"""
        return cls._load_registry(
            config_path=cls._resolve_config_path(config_path),
            disease_code=None,
        )

    @staticmethod
    def _resolve_config_path(config_path: Optional[Path]) -> Path:
        if config_path is not None:
            return config_path
        # 默认路径：app/field_mapping_config/field_mapping.json（独立目录避免与 app/config.py 重名）
        return Path(__file__).resolve().parent.parent / "field_mapping_config" / "field_mapping.json"

    def validate_or_raise(self) -> None:
        """启动期校验：必填字段映射齐 + display_blocked_fields 不与必填冲突。"""
        # 必填字段必须映射存在
        missing = [
            name for name in self.required_field_names
            if name not in self.fields
        ]
        if missing:
            raise FieldMappingConfigError(
                "字段映射缺失必填配置（spec §3.1）："
                + ", ".join(missing)
                + "。请检查 app/config/field_mapping.json 是否齐。"
            )

        # 必填字段必须有 column
        required_no_column = [
            name for name in self.required_field_names
            if self.fields[name].column in (None, "")
        ]
        if required_no_column:
            raise FieldMappingConfigError(
                "必填字段缺 column 映射："
                + ", ".join(required_no_column)
            )

        # PII 白名单与必填字段**不能冲突**（不允许把 encounter_id 等标成 blocked）
        conflict = set(self.required_field_names) & set(self.display_blocked_fields)
        if conflict:
            raise FieldMappingConfigError(
                "必填字段与 display_blocked_fields 冲突（spec §6）："
                + ", ".join(sorted(conflict))
            )

        # 每个必填字段必须有 spec_section（spec 追溯）
        missing_spec = [
            name for name in self.required_field_names
            if not self.fields[name].spec_section
        ]
        if missing_spec:
            raise FieldMappingConfigError(
                "必填字段缺 spec_section 标注："
                + ", ".join(missing_spec)
            )


def load_field_mapping_registry(
    disease_code: Optional[str] = None,
    config_path: Optional[Path] = None,
    validate: bool = True,
) -> FieldMappingRegistry:
    """便捷函数：加载并可选地启动校验。

    用法（启动期）：
        registry = load_field_mapping_registry("I21")
    用法（默认）：
        registry = load_field_mapping_registry(None)
    """
    if disease_code is None:
        reg = FieldMappingRegistry.load_default(config_path=config_path)
    else:
        reg = FieldMappingRegistry.for_disease(disease_code, config_path=config_path)
    if validate:
        reg.validate_or_raise()
    return reg
