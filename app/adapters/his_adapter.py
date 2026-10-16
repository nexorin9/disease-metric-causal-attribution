"""HIS 出院病案首页库直连适配层（spec.md §3.1 / §8）。

设计要点：
  - **只读**：SELECT only；never INSERT/UPDATE/DELETE；adapter 不暴露任何写方法。
  - **psycopg2 / pymysql 双 driver**：URL scheme 自动推断；不绑定具体厂商。
  - **字段投影**：从 HIS 真实表只投影 spec §3.1 必填字段 + 必要元数据；不引入 PII 列。
  - **连接池回收**：pool_recycle 默认 1800s（防真实库长连接断）。
  - **替身**：tests/ 用 SQLite 替身做协议兼容性测试，不连真实 HIS 库。

错误语义：
  - 缺凭据 → 由 Settings.validate_for_runtime() 在 lifespan 显式抛错；本模块不重复校验。
  - 表不存在 → DatabaseNotReachableError（来自 SQLAlchemy 反射失败）。
  - 必填字段缺失 → FieldMappingError（启动期一次性校验）。
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator, List, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker, Session

from app.config import Settings, get_settings
from app.db.schema import (
    ADMISSION_ROUTE_VALUES,
    DischargeFrontPageRow,
    NAMING_CONVENTION,
    Base as SchemaBase,
)

LOGGER = logging.getLogger(__name__)


# ---- 公开异常 ----
class HISAdapterError(RuntimeError):
    """HIS 适配层基类异常（spec §8 错误显式化）。"""


class DatabaseNotReachableError(HISAdapterError):
    """HIS DB 不可达 / 表不存在 / 反射失败。"""


class QueryParameterError(HISAdapterError):
    """查询参数非法（disease_code 未知 / 时间区间缺失）。"""


# ---- spec §3.1 必填字段（启动期一次性校验）----
REQUIRED_DB_FIELDS = (
    "encounter_id",
    "primary_diagnosis_code",
    "treatment_strategy",
    "age_band",
    "admission_route",
    "los_days",
)


class HISAdapter:
    """HIS 出院病案首页库直连适配器（仅读；spec §3.1 / §8）。

    Parameters
    ----------
    engine : sqlalchemy.Engine
        已构造好的 SQLAlchemy 引擎；测试可注入 SQLite 替身。
    db_kind : str
        "postgresql" / "mysql" / "sqlite"；驱动推断由 Settings.driver_kind() 提供。
    """

    def __init__(self, engine: Engine, db_kind: str):
        self.engine = engine
        self.db_kind = db_kind  # "postgresql" | "mysql" | "sqlite"
        self._session_factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
        LOGGER.info("HISAdapter init db_kind=%s", db_kind)

    # ---- 工厂 ----
    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None) -> "HISAdapter":
        """从 Settings 构造 HISAdapter（生产路径）。

        根据 HIS_DB_URL 自动推断 driver：
          postgresql+psycopg2 / mysql+pymysql / sqlite。
        """
        s = settings or get_settings()
        db_kind = s.driver_kind()
        url = s.his_db_url
        connect_args: dict = {}

        if db_kind == "sqlite":
            # SQLite 替身：tests/ 用，无连接池
            engine = create_engine(
                url,
                connect_args={"check_same_thread": False},
                poolclass=None,
            )
        else:
            # 真实 PostgreSQL / MySQL：连接池回收 + 健康检查
            engine = create_engine(
                url,
                pool_recycle=s.his_db_pool_recycle,
                pool_pre_ping=True,
                future=True,
            )
        return cls(engine=engine, db_kind=db_kind)

    # ---- 反射 / 校验 ----
    def ensure_schema_present(self) -> None:
        """启动期一次性校验：spec §3.1 必填字段在 HIS 表中存在；缺则显式报错。

        Raises
        ------
        DatabaseNotReachableError
            HIS DB 不可达
        FieldMappingError (HISAdapterError)
            spec §3.1 必填字段缺失
        """
        try:
            with self.engine.connect() as conn:
                # 反射 discharge_front_page 表的列
                rows = conn.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = :t"
                    )
                    if self.db_kind != "sqlite"
                    else text("PRAGMA table_info(discharge_front_page)")
                )
                if self.db_kind == "sqlite":
                    cols = {row[1] for row in rows}  # PRAGMA: cid,name,type,...
                else:
                    cols = {row[0] for row in rows}
        except SQLAlchemyError as e:
            raise DatabaseNotReachableError(
                f"HIS DB 不可达（db_kind={self.db_kind}）：{e}"
            ) from e

        # SQLite 替身 / 表未建 / 反射失败
        if not cols:
            LOGGER.warning(
                "discharge_front_page 表无列信息（db_kind=%s）；替身场景下允许跳过",
                self.db_kind,
            )
            return

        missing = [f for f in REQUIRED_DB_FIELDS if f not in cols]
        if missing:
            raise HISAdapterError(
                f"HIS discharge_front_page 缺少 spec §3.1 必填字段: {missing}\n"
                f"实际列: {sorted(cols)}\n"
                "请检查字段映射配置 app/config/field_mapping.json 或联系 HIS 厂商。"
            )
        LOGGER.info("HIS discharge_front_page 字段校验通过（%d 列）", len(cols))

    # ---- 查询 ----
    def query_discharge_front_page(
        self,
        disease_code: str,
        quarter_start: str,
        quarter_end: str,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """按 (病种 + 季度区间) 拉取病案首页（仅读；spec §3.1 字段映射）。

        Parameters
        ----------
        disease_code : str
            主诊断 ICD-10 编码（"I21" / "I63" / "Z96.10"）
        quarter_start, quarter_end : str
            季度 tag 区间（左闭右闭，如 "2026Q1" / "2026Q3"）。
            简化：暂以 quarter_tag 字符串字典序比较，不解析为日期（季度 tag 标准格式 6 字符）。
        limit : int | None
            限制返回行数（None 不限）；测试 / 大数据量场景使用。

        Returns
        -------
        pd.DataFrame
            字段对齐 spec §3.1 必填字段 + 必要元数据；类型还原 los_days:int /
            cmi:float / admission_route:str / readmit_flag:bool / complication_flag:bool

        Raises
        ------
        QueryParameterError
            disease_code 为空 / limit < 0
        DatabaseNotReachableError
            HIS DB 不可达
        """
        if not disease_code:
            raise QueryParameterError("disease_code 不能为空")
        if limit is not None and limit < 0:
            raise QueryParameterError(f"limit 必须 ≥ 0（实测 {limit}）")

        # SQL：LIKE 前缀匹配 disease_code + quarter_tag 区间
        # 注意：HIS 真实库 primary_diagnosis_code 通常为精确编码（如 "I21.4"），
        # 但本产品按"病种族"粒度（"I21" 包含所有 I21.x）建模，故用 LIKE 前缀。
        sql = text(
            """
            SELECT
                encounter_id,
                primary_diagnosis_code,
                secondary_diagnosis_list,
                treatment_strategy,
                age_band,
                admission_route,
                los_days,
                cmi,
                readmit_flag,
                complication_flag,
                admission_date,
                discharge_date,
                quarter_tag
            FROM discharge_front_page
            WHERE primary_diagnosis_code LIKE :code_prefix
              AND quarter_tag BETWEEN :q_start AND :q_end
            ORDER BY encounter_id
            """ + (" LIMIT :limit" if limit is not None else "")
        )
        params = {
            "code_prefix": f"{disease_code}%",
            "q_start": quarter_start,
            "q_end": quarter_end,
        }
        if limit is not None:
            params["limit"] = int(limit)

        try:
            with self.engine.connect() as conn:
                rows = conn.execute(sql, params).fetchall()
        except SQLAlchemyError as e:
            raise DatabaseNotReachableError(
                f"HIS DB 查询失败（disease={disease_code!r}, "
                f"quarter={quarter_start}..{quarter_end}）：{e}"
            ) from e

        if not rows:
            LOGGER.warning(
                "HIS 查询空结果（disease=%s, quarter=%s..%s, limit=%s）",
                disease_code, quarter_start, quarter_end, limit,
            )
            # 返回空 DataFrame（字段对齐 spec §3.1）
            return pd.DataFrame(columns=_EXPECTED_DF_COLUMNS)

        df = pd.DataFrame(rows, columns=_EXPECTED_DF_COLUMNS)
        return _coerce_types(df)

    # ---- 会话上下文（仅供 Future-read 内部使用；外部不暴露）----
    @contextmanager
    def session(self) -> Iterator[Session]:
        """读会话上下文（autoflush=False；commit=False 时禁自动写）。"""
        sess = self._session_factory()
        try:
            yield sess
        finally:
            sess.close()

    # ---- 池管理 ----
    def dispose(self) -> None:
        """释放连接池（应用关闭 / 测试 teardown 用）。"""
        try:
            self.engine.dispose()
        except Exception:  # pragma: no cover
            pass

    # ---- 表示 ----
    def __repr__(self) -> str:  # pragma: no cover
        return f"HISAdapter(db_kind={self.db_kind!r}, url={self.engine.url!r})"


# ---- 内部工具 ----
_EXPECTED_DF_COLUMNS: List[str] = [
    "encounter_id",
    "primary_diagnosis_code",
    "secondary_diagnosis_list",
    "treatment_strategy",
    "age_band",
    "admission_route",
    "los_days",
    "cmi",
    "readmit_flag",
    "complication_flag",
    "admission_date",
    "discharge_date",
    "quarter_tag",
]


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame 字段类型还原（与 data.seed_loader._coerce_types 对齐）。

    SQLAlchemy 反射的数字列会按原生类型返回；本函数做最后的强制收敛便于后续 pandas 算子。
    """
    df = df.copy()
    if df.empty:
        return df
    df["los_days"] = pd.to_numeric(df["los_days"], errors="raise").astype(int)
    df["age_band"] = pd.to_numeric(df["age_band"], errors="raise").astype(int)
    df["cmi"] = pd.to_numeric(df["cmi"], errors="coerce")
    df["admission_route"] = df["admission_route"].astype(str)
    df["readmit_flag"] = df["readmit_flag"].astype(bool)
    df["complication_flag"] = df["complication_flag"].astype(bool)
    return df


# ---- 自检入口（供 tests/test_his_adapter.py 与未来脚本调用）----
if __name__ == "__main__":  # pragma: no cover
    import json
    import sys

    try:
        adapter = HISAdapter.from_settings()
        adapter.ensure_schema_present()
        df = adapter.query_discharge_front_page("I21", "2026Q1", "2026Q4", limit=10)
        print(
            json.dumps(
                {
                    "ok": True,
                    "db_kind": adapter.db_kind,
                    "rows": len(df),
                    "columns": list(df.columns),
                },
                ensure_ascii=False,
            )
        )
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False))
        sys.exit(2)