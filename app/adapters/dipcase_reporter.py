"""单病种上报系统 HTTP POST 客户端（spec.md §3.2 + §11 收尾交付）。

| 入口                              | 用途                                       |
|-----------------------------------|--------------------------------------------|
| `DipcaseReporter.from_default_settings()` | 从环境变量（spec §8）构造客户端 |
| `DipcaseReporter(base_url, token, ...)`  | 测试构造注入：mock transport / 失败注入   |
| `post_attribution_draft(payload, ...)`   | 推送归因报告草稿到 `<DIPCASE_REPORTER_URL>/reports/attribution/draft`  |

设计原则（spec §6 + §11）：
- **重试 3 次 + 指数退避**：retry 策略内置；任何 5xx / 网络错 / 超时都触发重试。
- **失败回退状态机到 `pending_review`**：调用方根据 `ok=False` 决定是否 rollback（drafts.py 已实现）。
- **审计记录元数据，不写响应正文**：payload 透传（spec §3.2 字段集），response 只摘 attempt/status_code/elapsed_ms。
- **PII 白名单**：recorder / payload 内 strip_pii 兜底，杜绝 patient_name / id_card / phone / address 字段。
- **凭据全部从环境变量来**：base_url / token / timeout；不允许硬编码真实凭据。
- **2xx 通过 / 4xx 不重试**：仅 5xx 与网络/超时错误走 retry；4xx 视为客户端问题立即失败。

依赖：`requests`（pyproject 已锁定）。
测试替身：tests/test_dipcase_reporter.py 用 `_Transport` Mock 跑协议握手 + 重试 + 失败回退。
"""
from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 常量（spec §3.2 + §11）
# ---------------------------------------------------------------------------


# spec §3.2 上报 URL 后缀（不变，与路径 service worker 对齐）
DIPCASE_REPORTER_PATH = "/reports/attribution/draft"

# spec §3.2 POST header
CONTENT_TYPE_JSON = "application/json"

# spec §6 + §11 重试策略：3 次重试，指数退避（base 0.5s，jitter 0-0.2s 防雷鸣群）
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BASE_SECONDS = 0.5
DEFAULT_RETRY_MAX_ELAPSED_SECONDS = 5.0
DEFAULT_TIMEOUT_SECONDS = 15

# spec §3.2 payload schema_version（必须出现在 payload 顶层）
DIPCASE_PAYLOAD_SCHEMA_VERSION = "dipcase_attribution/1.0"

# PII 白名单：禁用字段防御性剔除（payload 走 strip_pii）
PII_BLOCKED_FIELDS: frozenset = frozenset({
    "patient_name",
    "id_card",
    "phone",
    "address",
    "name",
})

# spec §3.2 必填字段集合（用于 payload 校验 + 缺字段显式报错）
REQUIRED_PAYLOAD_FIELDS: tuple = (
    "disease_code",
    "quarter",
    "outcome",
    "treatment",
    "graph_snapshot_dot",
    "effect_estimates",
    "refutation_results",
    "counterfactual_scenarios",
    "narrative_zh",
    "reviewer_id",
    "model_version",
)


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------


class DipcaseReporterError(RuntimeError):
    """上报客户端基础异常（spec §6 不静默改数据；显式错误便于业务层回退）。"""


class DipcaseConfigError(DipcaseReporterError, ValueError):
    """凭据缺失或 URL 非法 — 不重试，立即失败。"""


class DipcasePayloadValidationError(DipcaseReporterError, ValueError):
    """payload 字段缺失 / 类型错 — spec §3.2 校验失败。"""


# ---------------------------------------------------------------------------
# 数据契约
# ---------------------------------------------------------------------------


@dataclass
class PostResult:
    """单次 post 调用结果（业务层 + 审计日志共用）。

    字段由 post_attribution_draft 写；业务层据此决定回退（spec §11）：
      - ok=True                        → 上报已成功（含 2xx），状态机保留 reviewed
      - ok=False, degraded=True        → 重试耗尽 / 网络错 / 5xx，状态机回退 pending_review
      - ok=False, degraded=False       → 4xx 视为客户端错误，立即失败不回退（避免重复错误刷上报系统）
      - rolled_back                    → 由调用方写回，本类不预设
    """

    ok: bool
    status_code: Optional[int] = None
    error: Optional[str] = None
    error_reason: Optional[str] = None
    attempt: int = 0
    elapsed_ms: float = 0.0
    degraded: bool = False
    audit_event_written: bool = False
    schema_version: str = DIPCASE_PAYLOAD_SCHEMA_VERSION
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "ok": bool(self.ok),
            "status_code": self.status_code,
            "error": self.error,
            "error_reason": self.error_reason,
            "attempt": int(self.attempt),
            "elapsed_ms": round(float(self.elapsed_ms), 2),
            "degraded": bool(self.degraded),
            "audit_event_written": bool(self.audit_event_written),
            **({"extra": self.extra} if self.extra else {}),
        }


# ---------------------------------------------------------------------------
# 防御性 PII 剔除
# ---------------------------------------------------------------------------


def _strip_pii(obj: Any) -> Any:
    """递归剔除 PII 字段（patient_name / id_card / phone 等）。

    - dict：剔除 PII key；递归其余
    - list / tuple：递归元素
    - 其它原样返回
    """
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(k, str) and k in PII_BLOCKED_FIELDS:
                continue
            out[k] = _strip_pii(v)
        return out
    if isinstance(obj, list):
        return [_strip_pii(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_strip_pii(v) for v in obj)
    return obj


# ---------------------------------------------------------------------------
# payload 校验（spec §3.2）
# ---------------------------------------------------------------------------


def _validate_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """spec §3.2 必填字段校验 + 防御性 PII 剔除 + 类型归一化。

    返回净化后的 payload（dict[str, Any]），供 HTTP POST 直接发出去。
    缺字段 / 类型错抛 DipcasePayloadValidationError（不静默）。
    """
    if payload is None:
        raise DipcasePayloadValidationError("payload is None")
    if not isinstance(payload, Mapping):
        raise DipcasePayloadValidationError(
            f"payload 类型必须为 Mapping，实际为 {type(payload).__name__}"
        )

    cleaned = _strip_pii(dict(payload))

    missing: List[str] = []
    bad_types: List[str] = []
    for f in REQUIRED_PAYLOAD_FIELDS:
        if f not in cleaned:
            missing.append(f)
            continue
        v = cleaned[f]
        # spec §3.2 字段类型期望（不强制；存在即做基础类型校验）
        if f in ("effect_estimates", "refutation_results", "counterfactual_scenarios"):
            if not isinstance(v, list):
                bad_types.append(f"{f}:{type(v).__name__}")
        elif f == "narrative_zh":
            if not isinstance(v, str):
                bad_types.append(f"{f}:{type(v).__name__}")
        elif f == "model_version":
            if not isinstance(v, str):
                bad_types.append(f"{f}:{type(v).__name__}")

    if missing:
        raise DipcasePayloadValidationError(
            f"payload 缺必填字段 {missing}（spec §3.2）"
        )
    if bad_types:
        raise DipcasePayloadValidationError(
            f"payload 字段类型错 {bad_types}（spec §3.2）"
        )

    # 注入 schema_version（业务层若已提供则尊重）
    cleaned.setdefault("schema_version", DIPCASE_PAYLOAD_SCHEMA_VERSION)

    return cleaned


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------


class DipcaseReporter:
    """单病种上报 HTTP POST 客户端（spec §3.2 + §11 收尾交付）。

    用法
    ----
    >>> reporter = DipcaseReporter.from_default_settings()
    >>> result = reporter.post_attribution_draft(draft.to_dict())
    >>> if result.ok:
    ...     # 状态机保留 reviewed
    ... else:
    ...     # 回退到 pending_review（业务层写）
    ...     pass

    测试可注入 `transport=lambda method, url, headers, body, timeout: ...` 跑协议握手。
    """

    # ---- 静态方法 ----

    @staticmethod
    def from_default_settings(
        *,
        url: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        transport: Optional[Callable[..., Any]] = None,
        audit_writer: Optional[Callable[..., None]] = None,
    ) -> "DipcaseReporter":
        """从 pydantic-settings 单例构造（spec §8 + §11 真实协议路径）。

        参数全部可显式覆盖 — 用于测试或真实凭据位注入。
        """
        # 延迟导入 Settings；测试夹具已加载 sys.path
        from app.config import get_settings  # type: ignore

        settings = get_settings()
        base_url = (url if url is not None else settings.dipcase_reporter_url).strip()
        bearer = (token if token is not None else settings.dipcase_reporter_token).strip()
        cfg_timeout = (
            int(timeout) if timeout is not None else int(settings.dipcase_reporter_timeout)
        )
        return DipcaseReporter(
            base_url=base_url,
            token=bearer,
            timeout=cfg_timeout,
            max_retries=max_retries,
            transport=transport,
            audit_writer=audit_writer,
        )

    # ---- 构造期 ----

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_base_seconds: float = DEFAULT_RETRY_BASE_SECONDS,
        retry_max_elapsed_seconds: float = DEFAULT_RETRY_MAX_ELAPSED_SECONDS,
        path: str = DIPCASE_REPORTER_PATH,
        transport: Optional[Callable[..., Any]] = None,
        audit_writer: Optional[Callable[..., None]] = None,
    ) -> None:
        if not base_url or not isinstance(base_url, str):
            raise DipcaseConfigError("base_url 必须为非空字符串")
        if not token or not isinstance(token, str):
            raise DipcaseConfigError("token 必须为非空字符串（bearer）")
        if timeout < 1:
            raise DipcaseConfigError(f"timeout 必须 ≥ 1，实际 {timeout}")
        if max_retries < 0:
            raise DipcaseConfigError(f"max_retries 必须 ≥ 0，实际 {max_retries}")
        if retry_base_seconds <= 0:
            raise DipcaseConfigError("retry_base_seconds 必须 > 0")
        if retry_max_elapsed_seconds <= 0:
            raise DipcaseConfigError("retry_max_elapsed_seconds 必须 > 0")

        # base_url 不允许带尾斜杠（防 // 双斜杠）
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = int(timeout)
        self.max_retries = int(max_retries)
        self.retry_base_seconds = float(retry_base_seconds)
        self.retry_max_elapsed_seconds = float(retry_max_elapsed_seconds)
        self.path = str(path).strip() or DIPCASE_REPORTER_PATH
        # 注入式 transport：测试用；默认 None 表示用 requests.post
        self._transport = transport
        self._audit_writer = audit_writer
        self._lock = threading.RLock()
        # 暴露 last 计数（业务层/测试可查）
        self.last_attempt: int = 0
        self.last_status_code: Optional[int] = None

    # ---- 内部辅助 ----

    @property
    def post_url(self) -> str:
        return f"{self.base_url}{self.path}"

    def _auth_header(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    def _default_transport(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        body: str,
        timeout: float,
    ) -> Any:
        """默认 transport：requests POST（生产路径真实协议）。"""
        import requests  # 延迟导入便于测试 monkeypatch

        return requests.post(
            url,
            headers=dict(headers),
            data=body,
            timeout=timeout,
        )

    def _do_request(self, body_json: str) -> Any:
        """发起单次 POST（transport 注入点）。"""
        headers = {
            "Content-Type": CONTENT_TYPE_JSON,
            "Accept": CONTENT_TYPE_JSON,
            **self._auth_header(),
        }
        transport = self._transport or self._default_transport
        return transport(
            url=self.post_url,
            headers=headers,
            body=body_json,
            timeout=self.timeout,
        )

    @staticmethod
    def _is_retryable(status_code: Optional[int]) -> bool:
        """仅 5xx / 网络错 / 超时走重试；4xx 视为客户端错误不重试。"""
        if status_code is None:
            return True  # 网络错 / 超时
        if 500 <= status_code <= 599:
            return True
        return False

    @staticmethod
    def _extract_status_code(resp: Any) -> Optional[int]:
        """从 transport 返回抽取 status_code。"""
        sc = getattr(resp, "status_code", None)
        if sc is None:
            return None
        try:
            return int(sc)
        except (TypeError, ValueError):
            return None

    def _wait_seconds_for_attempt(self, attempt: int) -> float:
        """指数退避 + jitter，第 attempt 次失败前等待的秒数。"""
        base = self.retry_base_seconds * (2 ** max(0, attempt - 1))
        capped = min(base, self.retry_max_elapsed_seconds)
        jitter = random.uniform(0.0, self.retry_base_seconds * 0.4)
        return float(max(0.0, capped + jitter))

    def _write_audit(
        self,
        *,
        payload: Mapping[str, Any],
        result: PostResult,
    ) -> bool:
        """审计写日志（best-effort；不抛）。"""
        if self._audit_writer is None:
            # 默认 audit_writer：尝试从 Settings 推断端点；本阶段由调用方注入
            # 不暴露 os.environ 直读（spec §8 凭据一律 env，但 audit_writer
            # 在 lifespan 阶段由 audit.logger 注入；此处无 env 直读）
            return False
        try:
            # audit envelope：脱敏 payload + 仅 result 摘要（不写响应原文敏感字段）
            cleaned_payload = _strip_pii(dict(payload))
            envelope = {
                "event_type": "draft/post",
                "operator": str(cleaned_payload.get("reviewer_id", "") or ""),
                "schema_version": DIPCASE_PAYLOAD_SCHEMA_VERSION,
                "result": result.to_dict(),
                "draft_disease": str(cleaned_payload.get("disease_code", "")),
                "draft_quarter": str(cleaned_payload.get("quarter", "")),
                "draft_outcome": str(cleaned_payload.get("outcome", "")),
                "model_version": str(cleaned_payload.get("model_version", "")),
            }
            # 防 PII 二次污染（payload 内潜在 reviewer_id 等业务字段都过 strip_pii 净化）
            envelope = _strip_pii(envelope)
            self._audit_writer(**envelope)
            return True
        except Exception as audit_exc:  # pragma: no cover - best-effort
            LOGGER.warning(
                "DipcaseReporter audit 写日志失败（best-effort，不阻断业务）: %s",
                audit_exc,
            )
            return False

    # ---- 对外主路径 ----

    def post_attribution_draft(
        self,
        payload: Mapping[str, Any],
        *,
        return_payload: bool = False,
    ) -> PostResult:
        """POST 归因报告草稿（spec §3.2）。

        Parameters
        ----------
        payload : Mapping
            spec §3.2 字段集；缺字段显式报错，绝不静默；PII 白名单自动剔除。
        return_payload : bool
            仅用于测试回放；返回净化后的 payload 不影响生产路径。

        Returns
        -------
        PostResult
            含 ok / status_code / attempt / elapsed_ms / degraded 等。
            业务层据此决定 rollback（spec §11）。
        """
        # 1. payload 校验 + PII 剔除
        cleaned = _validate_payload(payload)

        # 2. 重试循环
        attempts = self.max_retries + 1  # 1 次原请求 + max_retries 次重试
        attempt = 0
        last_status: Optional[int] = None
        last_error: Optional[str] = None
        last_error_reason: Optional[str] = None
        elapsed_total_ms = 0.0
        start = time.monotonic()

        for attempt in range(1, attempts + 1):
            self.last_attempt = attempt
            try:
                body = json.dumps(cleaned, ensure_ascii=False, sort_keys=False)
                resp = self._do_request(body)
            except Exception as exc:  # noqa: BLE001 — 网络 / 超时 / DNS 等
                last_error = f"exception:{type(exc).__name__}"
                last_error_reason = str(exc)
                last_status = None
                elapsed_total_ms = (time.monotonic() - start) * 1000.0
                LOGGER.warning(
                    "DipcaseReporter 第 %d/%d 次尝试网络异常: %s — %s",
                    attempt,
                    attempts,
                    type(exc).__name__,
                    exc,
                )
                if attempt < attempts:
                    time.sleep(self._wait_seconds_for_attempt(attempt))
                    continue
                break  # 耗尽

            sc = self._extract_status_code(resp)
            last_status = sc
            self.last_status_code = sc
            if sc is not None and 200 <= sc <= 299:
                # 成功
                elapsed_total_ms = (time.monotonic() - start) * 1000.0
                result = PostResult(
                    ok=True,
                    status_code=sc,
                    error=None,
                    error_reason=None,
                    attempt=attempt,
                    elapsed_ms=elapsed_total_ms,
                    degraded=False,
                    audit_event_written=self._write_audit(
                        payload=cleaned,
                        result=PostResult(  # 占位 audit 字段
                            ok=True,
                            status_code=sc,
                            attempt=attempt,
                            elapsed_ms=elapsed_total_ms,
                        ),
                    ),
                )
                return result

            # 失败：4xx 不重试；5xx / None 重试
            retryable = self._is_retryable(sc)
            if not retryable:
                elapsed_total_ms = (time.monotonic() - start) * 1000.0
                result = PostResult(
                    ok=False,
                    status_code=sc,
                    error=f"http_{sc}" if sc is not None else "http_unknown",
                    error_reason=f"4xx 客户端错误不重试（status={sc}）",
                    attempt=attempt,
                    elapsed_ms=elapsed_total_ms,
                    degraded=False,  # 4xx 不视为降级；业务层据此可重试草稿
                    audit_event_written=self._write_audit(
                        payload=cleaned,
                        result=PostResult(
                            ok=False,
                            status_code=sc,
                            error=f"http_{sc}",
                            attempt=attempt,
                            elapsed_ms=elapsed_total_ms,
                            degraded=False,
                        ),
                    ),
                )
                return result

            # 5xx：尝试重试
            last_error = f"http_{sc}" if sc is not None else "http_unknown"
            last_error_reason = f"5xx server error，retry 等待"
            LOGGER.warning(
                "DipcaseReporter 第 %d/%d 次尝试失败 status=%s，retry",
                attempt,
                attempts,
                sc,
            )
            if attempt < attempts:
                time.sleep(self._wait_seconds_for_attempt(attempt))
                continue
            break  # 耗尽

        # 重试耗尽 → 失败（spec §6 不静默改数据；业务层据此回退）
        elapsed_total_ms = (time.monotonic() - start) * 1000.0
        result = PostResult(
            ok=False,
            status_code=last_status,
            error=last_error or "retries_exhausted",
            error_reason=last_error_reason or f"{attempts} 次重试后仍未成功",
            attempt=attempts,
            elapsed_ms=elapsed_total_ms,
            degraded=True,  # 重试耗尽 → 降级（业务层回退 pending_review）
            audit_event_written=self._write_audit(
                payload=cleaned,
                result=PostResult(
                    ok=False,
                    status_code=last_status,
                    error=last_error or "retries_exhausted",
                    attempt=attempts,
                    elapsed_ms=elapsed_total_ms,
                    degraded=True,
                ),
            ),
        )
        return result


# ---------------------------------------------------------------------------
# 模块便捷函数
# ---------------------------------------------------------------------------


def post_attribution_draft(
    payload: Mapping[str, Any],
    *,
    url: Optional[str] = None,
    token: Optional[str] = None,
    timeout: Optional[int] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    transport: Optional[Callable[..., Any]] = None,
    audit_writer: Optional[Callable[..., None]] = None,
) -> PostResult:
    """一行调用 DipcaseReporter 上报（业务层最常用入口）。

    Returns
    -------
    PostResult（dict 兼容：to_dict()）
    """
    r = DipcaseReporter.from_default_settings(
        url=url,
        token=token,
        timeout=timeout,
        max_retries=max_retries,
        transport=transport,
        audit_writer=audit_writer,
    )
    return r.post_attribution_draft(payload)
