"""HIS 出院病案首页只读 ORM schema（spec.md §3.1）。

约束（spec.md §8 + §6）：
  - **只读**：本模型仅用于 SELECT；never INSERT/UPSERT/DELETE。
  - **必填字段**：encounter_id / primary_diagnosis_code / treatment_strategy /
    age_band / admission_route / los_days 在病案首页表里必须存在；
    缺字段启动期 ValueError 显式报错。
  - **PII 白名单**：patient_name / id_card / phone / address 等字段**绝不**在 ORM
    模型中定义；HIS 真实库若含此类字段，HISAdapter 投影时剔除。
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Enum as SAEnum,
    Integer,
    MetaData,
    Numeric,
    String,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


# 命名约定：约束 / 索引前缀统一 'his_' 便于跨表定位
NAMING_CONVENTION = {
    "ix": "ix_his_%(column_0_label)s",
    "uq": "uq_his_%(table_name)s_%(column_0_name)s",
    "ck": "ck_his_%(table_name)s_%(constraint_name)s",
    "fk": "fk_his_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_his_%(table_name)s",
}


class Base(DeclarativeBase):
    """HIS 只读视图基类（命名约定统一）。"""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


# spec §3.1：入院途径枚举（1=急诊 / 2=门诊 / 3=其他机构转入 / 9=其他）
ADMISSION_ROUTE_VALUES = ("1", "2", "3", "9")


class DischargeFrontPageRow(Base):
    """HIS 出院病案首页行（spec §3.1 字段映射 · 仅读）。

    表名 `discharge_front_page` 与医院典型命名一致；
    HISAdapter 通过 SQLAlchemy 反射（reflect）只读访问该表，
    本 ORM 仅做字段定义与类型校验，**绝不**用于 INSERT/UPDATE/DELETE。
    """

    __tablename__ = "discharge_front_page"

    # ---- 唯一主键（与单病种上报系统对齐）----
    encounter_id: Mapped[str] = mapped_column(String(36), nullable=False, primary_key=True)

    # ---- 必填字段（spec §3.1）----
    primary_diagnosis_code: Mapped[str] = mapped_column(String(6), nullable=False, index=True)
    secondary_diagnosis_list: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)
    treatment_strategy: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    age_band: Mapped[int] = mapped_column(Integer, nullable=False)
    admission_route: Mapped[str] = mapped_column(
        SAEnum(*ADMISSION_ROUTE_VALUES, name="admission_route_enum", native_enum=False),
        nullable=False,
    )
    los_days: Mapped[int] = mapped_column(Integer, nullable=False)

    # ---- 可选 outcome 候选（spec §3.1）----
    cmi: Mapped[Optional[float]] = mapped_column(Numeric(5, 2), nullable=True)
    readmit_flag: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    complication_flag: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # ---- 时间字段（按入院 / 出院日期落入区间）----
    admission_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True, index=True)
    discharge_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True, index=True)
    quarter_tag: Mapped[Optional[str]] = mapped_column(String(8), nullable=True, index=True)

    # ---- 审计元数据 ----
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        # 防止同一 encounter 被重复建模：业务层按 encounter_id + quarter_tag 去重
        {"comment": "HIS 出院病案首页只读视图（app.adapters.his_adapter · spec §3.1）"},
    )

    def to_dict(self) -> dict:
        """转 dict（仍为 ORM 行；不脱敏——脱敏由 HISAdapter 层负责）。"""
        return {
            "encounter_id": self.encounter_id,
            "primary_diagnosis_code": self.primary_diagnosis_code,
            "secondary_diagnosis_list": self.secondary_diagnosis_list,
            "treatment_strategy": self.treatment_strategy,
            "age_band": self.age_band,
            "admission_route": self.admission_route,
            "los_days": self.los_days,
            "cmi": float(self.cmi) if self.cmi is not None else None,
            "readmit_flag": self.readmit_flag,
            "complication_flag": self.complication_flag,
            "admission_date": self.admission_date.isoformat() if self.admission_date else None,
            "discharge_date": self.discharge_date.isoformat() if self.discharge_date else None,
            "quarter_tag": self.quarter_tag,
        }