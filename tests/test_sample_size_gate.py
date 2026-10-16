"""test_sample_size_gate — 样本量守门（spec §6 + §11 第 11 步）。

覆盖维度：
  1. SampleSizeGuard 构造期校验：阈值非法 / 边界 / 类型校验
  2. classify() 三档触发：n<30 / 30<=n<100 / n>=100
  3. check() 返回 SampleSizeVerdict 完整字段 + 草稿/写回开关
  4. should_allow_draft / should_allow_write 便捷方法
  5. estimate_runtime() 计时 + 异常隔离 + perf_records/ 落盘
  6. 边界值（恰好 30 / 100 / 0 / 1）触发正确档位
  7. 类型校验：bool/sample_size<0/None/str 拒绝
  8. 模块便捷函数：get_default_guard / classify_sample_size / evaluate_sample_size
  9. 与 baseline / refuter 三档一致性（n=29 全部标 unstable）
 10. perf 记录字段完整性：schema_version / sample_size / verdict / elapsed_ms /
     thresholds / allows_* / fn_qualname / exception_*

上下游契约：
  - T16 给 T17 (estimator_selector) 提供 should_allow_draft 信号
  - T18 benchmark 用 estimate_runtime 记录 n=200/500/1000/2000 耗时
  - T30 audit 与 sample/insufficient AuditEvent 联动（事件类型定义在 node_schema）
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import pytest

# 让 tests/ 能 import app
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.core.sample_size_gate import (  # noqa: E402
    ALL_VERDICTS,
    INSUFFICIENT_THRESHOLD,
    MARGINAL_THRESHOLD,
    PERF_RECORD_SCHEMA_VERSION,
    PERF_SAMPLE_SIZES,
    VERDICT_INSUFFICIENT,
    VERDICT_MARGINAL,
    VERDICT_SUFFICIENT,
    SampleSizeGuard,
    SampleSizeVerdict,
    classify_sample_size,
    evaluate_sample_size,
    get_default_guard,
)


# ==================================================================================
# A. 构造期校验
# ==================================================================================


class TestConstructor:
    def test_default_constructor(self) -> None:
        guard = SampleSizeGuard()
        assert guard.insufficient_threshold == INSUFFICIENT_THRESHOLD  # 30
        assert guard.marginal_threshold == MARGINAL_THRESHOLD  # 100

    def test_custom_thresholds(self) -> None:
        guard = SampleSizeGuard(insufficient_threshold=50, marginal_threshold=200)
        assert guard.insufficient_threshold == 50
        assert guard.marginal_threshold == 200

    def test_insufficient_threshold_too_small(self) -> None:
        with pytest.raises(ValueError, match="insufficient_threshold"):
            SampleSizeGuard(insufficient_threshold=0)

    def test_marginal_le_insufficient_raises(self) -> None:
        # marginal 必须 > insufficient（spec §6 三档分段）
        with pytest.raises(ValueError, match="marginal_threshold"):
            SampleSizeGuard(insufficient_threshold=50, marginal_threshold=50)

    def test_marginal_lt_insufficient_raises(self) -> None:
        with pytest.raises(ValueError, match="marginal_threshold"):
            SampleSizeGuard(insufficient_threshold=50, marginal_threshold=30)


# ==================================================================================
# B. classify 三档触发
# ==================================================================================


class TestClassify:
    def test_insufficient_below_threshold(self) -> None:
        guard = SampleSizeGuard()
        assert guard.classify(0) == VERDICT_INSUFFICIENT
        assert guard.classify(15) == VERDICT_INSUFFICIENT
        assert guard.classify(29) == VERDICT_INSUFFICIENT

    def test_marginal_in_window(self) -> None:
        guard = SampleSizeGuard()
        assert guard.classify(30) == VERDICT_MARGINAL
        assert guard.classify(45) == VERDICT_MARGINAL
        assert guard.classify(99) == VERDICT_MARGINAL

    def test_sufficient_at_threshold(self) -> None:
        guard = SampleSizeGuard()
        assert guard.classify(100) == VERDICT_SUFFICIENT
        assert guard.classify(150) == VERDICT_SUFFICIENT
        assert guard.classify(2000) == VERDICT_SUFFICIENT

    def test_custom_thresholds(self) -> None:
        guard = SampleSizeGuard(insufficient_threshold=20, marginal_threshold=150)
        assert guard.classify(19) == VERDICT_INSUFFICIENT
        assert guard.classify(20) == VERDICT_MARGINAL
        assert guard.classify(149) == VERDICT_MARGINAL
        assert guard.classify(150) == VERDICT_SUFFICIENT


# ==================================================================================
# C. check() 完整 SampleSizeVerdict 字段
# ==================================================================================


class TestCheck:
    def test_insufficient_blocks_draft_and_write(self) -> None:
        guard = SampleSizeGuard()
        v = guard.check(15)
        assert v.verdict == VERDICT_INSUFFICIENT
        assert v.sample_size == 15
        assert v.insufficient_threshold == 30
        assert v.marginal_threshold == 100
        assert v.allows_draft_posting is False
        assert v.allows_backend_write is False
        assert any("sample_insufficient" in r for r in v.degraded_reasons)

    def test_marginal_allows_draft_and_write_with_marker(self) -> None:
        guard = SampleSizeGuard()
        v = guard.check(60)
        assert v.verdict == VERDICT_MARGINAL
        assert v.sample_size == 60
        assert v.allows_draft_posting is True
        assert v.allows_backend_write is True
        assert any("sample_marginal" in r for r in v.degraded_reasons)

    def test_sufficient_clean(self) -> None:
        guard = SampleSizeGuard()
        v = guard.check(200)
        assert v.verdict == VERDICT_SUFFICIENT
        assert v.sample_size == 200
        assert v.allows_draft_posting is True
        assert v.allows_backend_write is True
        assert v.degraded_reasons == []

    def test_boundary_n_equals_thresholds(self) -> None:
        guard = SampleSizeGuard()
        # 恰好 n=30 应该是 marginal（非 insufficient）
        assert guard.check(30).verdict == VERDICT_MARGINAL
        # 恰好 n=100 应该是 sufficient（非 marginal）
        assert guard.check(100).verdict == VERDICT_SUFFICIENT

    def test_verdict_to_dict_round_trip(self) -> None:
        v = SampleSizeGuard().check(60)
        d = v.to_dict()
        assert d["verdict"] == VERDICT_MARGINAL
        assert d["sample_size"] == 60
        assert d["allows_draft_posting"] is True
        assert d["allows_backend_write"] is True
        assert isinstance(d["degraded_reasons"], list)


# ==================================================================================
# D. 便捷方法
# ==================================================================================


class TestConvenienceMethods:
    def test_should_allow_draft(self) -> None:
        guard = SampleSizeGuard()
        assert guard.should_allow_draft(15) is False
        assert guard.should_allow_draft(60) is True
        assert guard.should_allow_draft(200) is True

    def test_should_allow_write(self) -> None:
        guard = SampleSizeGuard()
        assert guard.should_allow_write(15) is False
        assert guard.should_allow_write(60) is True
        assert guard.should_allow_write(200) is True


# ==================================================================================
# E. 类型与边界校验
# ==================================================================================


class TestTypeValidation:
    def test_bool_rejected(self) -> None:
        guard = SampleSizeGuard()
        with pytest.raises(ValueError, match="bool"):
            guard.classify(True)
        with pytest.raises(ValueError, match="bool"):
            guard.check(True)

    def test_negative_rejected(self) -> None:
        guard = SampleSizeGuard()
        with pytest.raises(ValueError, match=">= 0"):
            guard.classify(-1)
        with pytest.raises(ValueError, match=">= 0"):
            guard.check(-100)

    def test_non_int_rejected(self) -> None:
        guard = SampleSizeGuard()
        with pytest.raises(ValueError):
            guard.classify("60")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            guard.classify(None)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            guard.classify(60.5)  # type: ignore[arg-type]


# ==================================================================================
# F. estimate_runtime() 行为
# ==================================================================================


@pytest.fixture
def reset_perf_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把 perf_records/ 路径切到临时目录，避免污染项目目录。"""
    # 注：sample_size_gate 用 Path(__file__).resolve().parent.parent.parent 计算项目根，
    # 这里重写 _write_perf_record 的目标路径。
    perf_dir = tmp_path / "perf_records"
    perf_dir.mkdir(parents=True, exist_ok=True)

    def _patched_write(self: SampleSizeGuard, record: dict, n: int) -> Path:  # type: ignore[no-untyped-def]
        target_path = perf_dir / f"sample_size_runtime_n{n}.json"
        target_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return target_path

    monkeypatch.setattr(SampleSizeGuard, "_write_perf_record", _patched_write)
    return perf_dir


class TestEstimateRuntime:
    def test_sufficient_records_to_disk(self, reset_perf_records: Path) -> None:
        guard = SampleSizeGuard()
        verdict, result, elapsed = guard.estimate_runtime(
            n=150,
            fn=lambda: "test_result",
        )
        assert verdict.verdict == VERDICT_SUFFICIENT
        assert result == "test_result"
        assert elapsed >= 0.0
        record_path = reset_perf_records / "sample_size_runtime_n150.json"
        assert record_path.exists()
        record = json.loads(record_path.read_text(encoding="utf-8"))
        assert record["schema_version"] == PERF_RECORD_SCHEMA_VERSION
        assert record["sample_size"] == 150
        assert record["verdict"] == VERDICT_SUFFICIENT
        assert record["allows_draft_posting"] is True
        assert record["allows_backend_write"] is True
        assert record["result_type"] == "str"
        assert record["exception_type"] is None

    def test_marginal_records_to_disk(self, reset_perf_records: Path) -> None:
        guard = SampleSizeGuard()
        verdict, _, _ = guard.estimate_runtime(n=60, fn=lambda: 42)
        assert verdict.verdict == VERDICT_MARGINAL
        record_path = reset_perf_records / "sample_size_runtime_n60.json"
        assert record_path.exists()

    def test_insufficient_records_to_disk_but_blocks_business(
        self, reset_perf_records: Path
    ) -> None:
        guard = SampleSizeGuard()
        verdict, _, _ = guard.estimate_runtime(n=15, fn=lambda: "anything")
        # 即使 insufficient，perf 数值仍落盘（spec §11 第 11 步）
        assert verdict.verdict == VERDICT_INSUFFICIENT
        assert verdict.allows_draft_posting is False
        assert verdict.allows_backend_write is False
        record_path = reset_perf_records / "sample_size_runtime_n15.json"
        assert record_path.exists()

    def test_exception_isolated_and_recorded(
        self, reset_perf_records: Path
    ) -> None:
        guard = SampleSizeGuard()

        def boom() -> None:
            raise RuntimeError("intentional")

        verdict, result, _ = guard.estimate_runtime(n=200, fn=boom)
        assert verdict.verdict == VERDICT_SUFFICIENT
        assert isinstance(result, RuntimeError)
        record_path = reset_perf_records / "sample_size_runtime_n200.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        assert record["exception_type"] == "RuntimeError"
        assert "intentional" in record["exception_message"]
        assert record["result_type"] == "Exception"

    def test_elapsed_ms_is_positive_and_reasonable(
        self, reset_perf_records: Path
    ) -> None:
        guard = SampleSizeGuard()

        def slow() -> str:
            time.sleep(0.01)
            return "ok"

        verdict, result, elapsed = guard.estimate_runtime(n=200, fn=slow)
        assert result == "ok"
        assert elapsed >= 0.005  # 至少 5ms
        record_path = reset_perf_records / "sample_size_runtime_n200.json"
        # 注：上一用例已写入 n=200.json，会被这一条覆盖；这里只校验 schema_version 等字段
        record = json.loads(record_path.read_text(encoding="utf-8"))
        assert record["elapsed_ms"] >= 5.0
        assert record["elapsed_seconds"] == record["elapsed_ms"] / 1000.0

    def test_fn_qualname_recorded(self, reset_perf_records: Path) -> None:
        guard = SampleSizeGuard()

        def named_helper() -> int:
            return 1

        guard.estimate_runtime(n=200, fn=named_helper)
        record_path = reset_perf_records / "sample_size_runtime_n200.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        assert "named_helper" in record["fn_qualname"]


# ==================================================================================
# G. perf_records 三档落盘（spec §11 第 11 步 — n=30/100/500）
# ==================================================================================


class TestPerfRecordsThreeBuckets:
    def test_three_buckets_land_in_perf_records(
        self, reset_perf_records: Path
    ) -> None:
        guard = SampleSizeGuard()
        for n in PERF_SAMPLE_SIZES:
            guard.estimate_runtime(n=n, fn=lambda: f"result_{n}")
        # 三档（n=30 → marginal / n=100 → sufficient / n=500 → sufficient）
        for n in PERF_SAMPLE_SIZES:
            record_path = reset_perf_records / f"sample_size_runtime_n{n}.json"
            assert record_path.exists(), f"perf record missing for n={n}"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            assert record["sample_size"] == n
            assert record["verdict"] in ALL_VERDICTS

    def test_n30_is_marginal(self, reset_perf_records: Path) -> None:
        guard = SampleSizeGuard()
        guard.estimate_runtime(n=30, fn=lambda: 1)
        record = json.loads(
            (reset_perf_records / "sample_size_runtime_n30.json").read_text(
                encoding="utf-8"
            )
        )
        assert record["verdict"] == VERDICT_MARGINAL

    def test_n100_is_sufficient(self, reset_perf_records: Path) -> None:
        guard = SampleSizeGuard()
        guard.estimate_runtime(n=100, fn=lambda: 1)
        record = json.loads(
            (reset_perf_records / "sample_size_runtime_n100.json").read_text(
                encoding="utf-8"
            )
        )
        assert record["verdict"] == VERDICT_SUFFICIENT

    def test_n500_is_sufficient(self, reset_perf_records: Path) -> None:
        guard = SampleSizeGuard()
        guard.estimate_runtime(n=500, fn=lambda: 1)
        record = json.loads(
            (reset_perf_records / "sample_size_runtime_n500.json").read_text(
                encoding="utf-8"
            )
        )
        assert record["verdict"] == VERDICT_SUFFICIENT


# ==================================================================================
# H. 模块便捷函数
# ==================================================================================


class TestModuleLevelHelpers:
    def test_get_default_guard(self) -> None:
        guard = get_default_guard()
        assert isinstance(guard, SampleSizeGuard)
        assert guard.insufficient_threshold == INSUFFICIENT_THRESHOLD
        assert guard.marginal_threshold == MARGINAL_THRESHOLD

    def test_classify_sample_size_function(self) -> None:
        assert classify_sample_size(15) == VERDICT_INSUFFICIENT
        assert classify_sample_size(60) == VERDICT_MARGINAL
        assert classify_sample_size(200) == VERDICT_SUFFICIENT

    def test_classify_sample_size_custom(self) -> None:
        assert (
            classify_sample_size(50, insufficient_threshold=20, marginal_threshold=60)
            == VERDICT_MARGINAL
        )

    def test_evaluate_sample_size_returns_verdict(self) -> None:
        v = evaluate_sample_size(60)
        assert isinstance(v, SampleSizeVerdict)
        assert v.verdict == VERDICT_MARGINAL


# ==================================================================================
# I. 与 baseline / refuter 一致性（spec §6 全产品线样本量阈值对齐）
# ==================================================================================


class TestConsistencyWithRefuterBaseline:
    def test_insufficient_threshold_matches_refuter_min(self) -> None:
        """SampleSizeGuard 的 insufficient_threshold 必须与 RefuterPipeline.MIN_REFUTE_SAMPLES 一致。"""
        # 这两个常量都是 spec §6 共享；模块导入时已对齐，否则会冲突。
        from app.core.refuter import MIN_REFUTE_SAMPLES

        assert INSUFFICIENT_THRESHOLD == MIN_REFUTE_SAMPLES, (
            "SampleSizeGuard.insufficient_threshold 必须与 refuter.MIN_REFUTE_SAMPLES 一致"
        )

    def test_insufficient_threshold_matches_baseline_min(self) -> None:
        from app.core.baseline import MIN_BASELINE_SAMPLES

        assert INSUFFICIENT_THRESHOLD == MIN_BASELINE_SAMPLES, (
            "SampleSizeGuard.insufficient_threshold 必须与 baseline.MIN_BASELINE_SAMPLES 一致"
        )

    def test_n29_treated_as_insufficient_similarly(self) -> None:
        """n=29（在 refuter 看是 all_degraded，在 SampleSizeGuard 是 insufficient）— 边界一致。"""
        v = SampleSizeGuard().check(29)
        assert v.verdict == VERDICT_INSUFFICIENT
        assert v.allows_draft_posting is False
        assert v.allows_backend_write is False


# ==================================================================================
# J. 持久化路径在真实 perf_records/ 落盘（spec §11 第 11 步）
# ==================================================================================


class TestRealPerfRecordsPath:
    def test_real_perf_records_dir_is_written(self) -> None:
        """不在 monkeypatch 下，直接验证文件落盘到疾病-metric-causal-attribution/perf_records/。"""
        guard = SampleSizeGuard()
        # 先清理可能存在的文件，避免污染
        project_root = Path(__file__).resolve().parent.parent
        perf_dir = project_root / "perf_records"
        for n in PERF_SAMPLE_SIZES:
            p = perf_dir / f"sample_size_runtime_n{n}.json"
            if p.exists():
                p.unlink()

        for n in PERF_SAMPLE_SIZES:
            guard.estimate_runtime(n=n, fn=lambda: f"real_perf_n{n}")

        # 至少 3 个 benchmark json 已落盘
        written = list(perf_dir.glob("sample_size_runtime_n*.json"))
        assert len(written) >= len(PERF_SAMPLE_SIZES)

        # 每个文件 JSON 合法且含 perf_records/1.0 schema_version 等关键字段
        for p in written:
            data = json.loads(p.read_text(encoding="utf-8"))
            assert data["schema_version"] == PERF_RECORD_SCHEMA_VERSION
            assert data["sample_size"] in PERF_SAMPLE_SIZES
            assert data["verdict"] in ALL_VERDICTS
            assert isinstance(data["elapsed_ms"], float)
            assert data["elapsed_ms"] >= 0.0
            assert "fn_qualname" in data
