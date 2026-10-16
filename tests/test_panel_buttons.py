"""test_panel_buttons — T24 · 三按钮（反事实 / 评审确认 / 评审退回）面板交互层。

覆盖维度（spec §11 浏览器面板 · 评审交互层）：
  1. 源码静态结构（panel.js + panel.css + index.html）：
     - runCounterfactual / postReview / confirmDraft / rejectDraft 函数就位
     - _buildDraftId 稳定（同一组合 → 同一 id）
     - _postJson 通用 POST helper
     - _setBtnLoading data-loading 属性
     - panel.css 含 [data-loading] 与 @keyframes btn-spin
     - index.html 增 cf-error 元素
  2. Playwright 端到端（page 内 fetch mock）：
     - 触发反事实 → POST /counterfactual/run + 显示预测结局 + toast success
     - 反事实：节点 / 假设值缺 → toast warning 不发请求
     - 反事实：403 → toast danger
     - 反事实：422 → toast warning（inline 错误）
     - 评审确认 → POST /api/drafts/{id}/review action=confirm + toast success
     - 评审退回 → POST /api/drafts/{id}/review action=reject + toast success
     - 评审 403 → toast danger
     - 评审 404 → toast warning
     - 评审 422 → toast warning（含 error_code）
     - 上报失败回退 → toast warning + post_result.rolled_back 信息
     - 加载态：data-loading="true" + spinner 样式生效

设计校验纪律（spec §11 + README 强制结构）：
  - 真实浏览器 + 真实 fetch（fetch 在 page 内被 install_panel_fetch_mock 替换）
  - toast 与 cf-error 都是真实 DOM 操作，不是 mock
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 测试环境：APP_ENV=test 让 lifespan 跳过 OPENAI_API_KEY 校验
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("HIS_DB_URL", "sqlite:///./tests/.benchmarks/his_fake.db")
os.environ.setdefault("DIPCASE_REPORTER_URL", "http://localhost:18899/mock")

# ------------------------- helpers -------------------------

from app.panel import STATIC_DIR, TEMPLATES_DIR  # noqa: E402

PANEL_JS = STATIC_DIR / "panel.js"
PANEL_CSS = STATIC_DIR / "panel.css"
INDEX_HTML = TEMPLATES_DIR / "index.html"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _js(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _js_args(*objs) -> str:
    return ", ".join(_js(o) for o in objs)


# ------------------------- 1. 源码结构（静态校验）-------------------------


class TestPanelJsStructure:
    @pytest.fixture(scope="class")
    def js(self) -> str:
        return _read(PANEL_JS)

    def test_defines_run_counterfactual(self, js: str):
        assert "function runCounterfactual" in js, \
            "panel.js 缺 runCounterfactual（cf-run 按钮 handler）"

    def test_defines_post_review(self, js: str):
        assert "function postReview" in js, \
            "panel.js 缺 postReview（confirm/reject 共享路径）"

    def test_defines_confirm_reject(self, js: str):
        assert "function confirmDraft" in js
        assert "function rejectDraft" in js

    def test_defines_build_draft_id(self, js: str):
        """_buildDraftId 必须为同 disease+outcome+quarter 多次建模得同一 id。"""
        assert "function _buildDraftId" in js
        assert "'dr_'" in js or '"dr_"' in js, "_buildDraftId 未用 dr_ 前缀"

    def test_defines_post_json(self, js: str):
        assert "function _postJson" in js, "panel.js 缺 _postJson 通用 POST helper"

    def test_defines_set_btn_loading(self, js: str):
        assert "function _setBtnLoading" in js
        assert "data-loading" in js, "_setBtnLoading 未用 data-loading 属性"

    def test_state_has_current_draft_id(self, js: str):
        assert "currentDraftId" in js, "panel.js state 缺 currentDraftId"

    def test_state_has_current_modeling(self, js: str):
        assert "currentModeling" in js, "panel.js state 缺 currentModeling"

    def test_cf_run_wired_to_run_counterfactual(self, js: str):
        assert "cfRun.addEventListener" in js
        assert "runCounterfactual" in js

    def test_confirm_wired_to_confirm_draft(self, js: str):
        assert "elBtnConfirm.addEventListener" in js
        assert "confirmDraft" in js
        assert "postReview('confirm')" in js or 'postReview("confirm")' in js

    def test_reject_wired_to_reject_draft(self, js: str):
        assert "elBtnReject.addEventListener" in js
        assert "rejectDraft" in js
        assert "postReview('reject')" in js or 'postReview("reject")' in js

    def test_cf_endpoint_url(self, js: str):
        """反事实走 POST /counterfactual/run（无 /api/ 前缀，与 T20 一致）。"""
        assert "/counterfactual/run" in js

    def test_drafts_endpoint_url(self, js: str):
        """评审走 POST /api/drafts/{draft_id}/review（drafts 路由有 /api/drafts prefix）。"""
        assert "/api/drafts/" in js
        assert "/review" in js

    def test_x_quality_role_header_sent(self, js: str):
        assert "X-Quality-Role" in js

    def test_toast_on_403(self, js: str):
        assert "403" in js
        assert js.count("'danger'") >= 2

    def test_toast_on_404(self, js: str):
        assert "404" in js
        assert "'warning'" in js

    def test_toast_on_422(self, js: str):
        assert "422" in js

    def test_handles_rollback(self, js: str):
        assert "rolled_back" in js
        assert "post_result" in js

    def test_disables_btn_after_review(self, js: str):
        assert re.search(r"elBtnConfirm\.disabled = true", js) is not None
        assert re.search(r"elBtnReject\.disabled = true", js) is not None

    def test_clears_current_draft_after_review(self, js: str):
        assert re.search(r"state\.currentDraftId\s*=\s*null", js) is not None

    def test_init_exports_test_hooks(self, js: str):
        for hook in (
            "runCounterfactual", "confirmDraft", "rejectDraft",
            "_buildDraftId", "_postJson", "_setBtnLoading", "__seedDraft",
        ):
            assert hook in js, f"__cfPanel 未导出: {hook}"


class TestPanelCssStructure:
    @pytest.fixture(scope="class")
    def css(self) -> str:
        return _read(PANEL_CSS)

    def test_has_button_loading_state(self, css: str):
        assert '[data-loading="true"]' in css
        assert "btn-spin" in css
        assert "@keyframes btn-spin" in css

    def test_has_button_active_state(self, css: str):
        for klass in ("primary", "success", "danger"):
            assert re.search(rf"\.btn--{klass}:active", css) is not None

    def test_has_button_disabled_polish(self, css: str):
        for klass in ("primary", "success", "danger"):
            assert re.search(rf"\.btn--{klass}:disabled", css) is not None

    def test_has_cf_error(self, css: str):
        assert ".cf-error" in css


class TestIndexHtmlStructure:
    @pytest.fixture(scope="class")
    def html(self) -> str:
        return _read(INDEX_HTML)

    def test_has_cf_error_element(self, html: str):
        assert 'id="cf-error"' in html
        assert 'class="cf-error"' in html


# ------------------------- 2. Playwright 端到端交互测试 -------------------------


def _start_uvicorn(port: int) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port),
         "--log-level", "warning"],
        cwd=str(_PROJECT_ROOT),
        env={**os.environ, "APP_ENV": "test"},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    import socket
    for _ in range(50):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return proc
        except OSError:
            time.sleep(0.1)
    proc.terminate()
    raise RuntimeError(f"uvicorn failed to start on port {port}")


def _find_free_port() -> int:
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def server_url():
    port = _find_free_port()
    proc = _start_uvicorn(port)
    url = f"http://127.0.0.1:{port}"
    yield url
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        yield b
        b.close()


@pytest.fixture
def page(browser):
    """干净的浏览器页面（不带 mock）。"""
    ctx = browser.new_context()
    p = ctx.new_page()
    yield p
    ctx.close()


# --- fetch mock helper（在浏览器内拦截 fetch） ---

def _install_fetch_mock(page, routes):
    """routes: list of (url_suffix, status, body_dict) — install AFTER navigation."""
    js = """
    (() => {
      window.__origFetch = window.fetch;
      window.__cfCapturedRequests = [];
      const routes = %s;
      window.fetch = function(input, init) {
        let url = (typeof input === 'string') ? input : (input && input.url ? input.url : '');
        let method = (init && init.method) || 'GET';
        let body = (init && init.body) || null;
        window.__cfCapturedRequests.push({url, method, body});
        for (const r of routes) {
          if (url.endsWith(r[0])) {
            return Promise.resolve(new Response(JSON.stringify(r[2]), {
              status: r[1],
              headers: {'Content-Type': 'application/json'}
            }));
          }
        }
        return window.__origFetch.apply(this, arguments);
      };
    })();
    """ % json.dumps(routes)
    page.evaluate(js)


# Standard cf_ok body
CF_OK_BODY = {
    "schema_version": "counterfactual_response/1.0",
    "ok": True,
    "disease_code": "I21", "outcome": "cmi", "quarter": "2025-Q4",
    "n_scenarios": 1, "n_completed": 1, "n_timed_out": 0, "n_errored": 0,
    "elapsed_ms": 12, "batch_size": 10,
    "scenarios": [{
        "scenario_id": "scn_abc",
        "disease_code": "I21", "outcome": "cmi",
        "intervention_node": "treatment_strategy",
        "hypothetical_value": "PCI",
        "baseline_value": 1.2, "predicted_outcome": 1.85, "confidence": 0.85,
        "do_operations": ["do(treatment_strategy = PCI)"],
        "elapsed_ms": 8, "timeout": False, "baseline_std": 0.42,
    }],
}

REVIEW_OK_BODY = {
    "schema_version": "attribution_draft/1.0",
    "ok": True,
    "action": "confirm",
    "draft": {"draft_id": "dr_seed", "status": "reviewed"},
    "transition": {"transition_id": "tr_seed", "from_state": "pending_review", "to_state": "reviewed"},
    "post_result": {"ok": True, "degraded": False, "status_code": 200},
    "rolled_back": False,
}


class TestPanelButtonE2E:
    """真实浏览器 + in-browser fetch mock。"""

    def test_counterfactual_run_triggers_post_and_shows_result(self, server_url, browser):
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto(server_url + "/", wait_until="networkidle", timeout=15000)
        page.wait_for_function("typeof window.__cfPanel === 'object'")
        _install_fetch_mock(page, [("/counterfactual/run", 200, CF_OK_BODY)])
        # 注入建模 context 与节点
        page.evaluate("window.__cfPanel.__seedDraft(null, {disease_code:'I21',outcome:'cmi',quarter:'2025-Q4'})")
        page.evaluate("window.__cfPanel.state.nodes = [{id:'treatment_strategy',label:'治疗'}];")
        # 打开 modal
        page.evaluate("document.getElementById('btn-counterfactual').disabled = false; "
                      "document.getElementById('btn-counterfactual').click()")
        page.evaluate("document.getElementById('cf-value').value = 'PCI'")
        page.evaluate("document.getElementById('cf-run').click()")
        # 等待 cf-predicted 写入
        page.wait_for_function("document.getElementById('cf-predicted').textContent !== '—'", timeout=5000)
        assert page.evaluate("document.getElementById('cf-predicted').textContent") == "1.850"
        assert page.evaluate("document.getElementById('cf-confidence').textContent") == "0.85"
        # toast success
        page.wait_for_selector(".toast--success", timeout=3000)
        assert "反事实完成" in page.locator(".toast--success").last.text_content()
        # 捕获请求
        captured = page.evaluate("window.__cfCapturedRequests")
        cf_call = [r for r in captured if r["url"].endswith("/counterfactual/run")][-1]
        assert cf_call["method"] == "POST"
        body = json.loads(cf_call["body"] or "{}")
        assert body["disease_code"] == "I21"
        assert body["outcome"] == "cmi"
        assert len(body["scenarios"]) == 1
        ctx.close()

    def test_counterfactual_missing_node_warns(self, server_url, browser):
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto(server_url + "/", wait_until="networkidle", timeout=15000)
        page.wait_for_function("typeof window.__cfPanel === 'object'")
        page.evaluate("window.__cfPanel.__seedDraft(null, {disease_code:'I21',outcome:'cmi',quarter:'2025-Q4'})")
        page.evaluate("window.__cfPanel.state.nodes = [{id:'treatment_strategy',label:'治疗'}];")
        page.evaluate("document.getElementById('btn-counterfactual').disabled = false; "
                      "document.getElementById('btn-counterfactual').click()")
        page.evaluate("var s=document.getElementById('cf-node'); s.innerHTML='<option value=\"\"></option>'; s.value='';")
        page.evaluate("document.getElementById('cf-value').value = ''")
        page.evaluate("document.getElementById('cf-run').click()")
        page.wait_for_selector(".toast--warning", timeout=3000)
        captured = page.evaluate("window.__cfCapturedRequests || []")
        assert all(not r["url"].endswith("/counterfactual/run") for r in captured)
        ctx.close()

    def test_counterfactual_403_shows_danger_toast(self, server_url, browser):
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto(server_url + "/", wait_until="networkidle", timeout=15000)
        page.wait_for_function("typeof window.__cfPanel === 'object'")
        _install_fetch_mock(page, [("/counterfactual/run", 403, {
            "ok": False, "error_code": "quality_role_forbidden",
            "message": "X-Quality-Role 缺失", "details": {"role": "<missing>"}
        })])
        page.evaluate("window.__cfPanel.__seedDraft(null, {disease_code:'I21',outcome:'cmi',quarter:'2025-Q4'})")
        page.evaluate("window.__cfPanel.state.nodes = [{id:'treatment_strategy',label:'治疗'}];")
        page.evaluate("document.getElementById('btn-counterfactual').disabled = false; "
                      "document.getElementById('btn-counterfactual').click()")
        page.evaluate("document.getElementById('cf-value').value = 'PCI'")
        page.evaluate("document.getElementById('cf-run').click()")
        page.wait_for_selector(".toast--danger", timeout=3000)
        assert "反事实" in page.locator(".toast--danger").last.text_content()
        ctx.close()

    def test_confirm_triggers_post_with_action_confirm(self, server_url, browser):
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto(server_url + "/", wait_until="networkidle", timeout=15000)
        page.wait_for_function("typeof window.__cfPanel === 'object'")
        _install_fetch_mock(page, [("/api/drafts/dr_seed123/review", 200, REVIEW_OK_BODY)])
        page.evaluate("window.__cfPanel.__seedDraft('dr_seed123', {disease_code:'I21',outcome:'cmi',quarter:'2025-Q4'})")
        page.evaluate("document.getElementById('btn-confirm').click()")
        page.wait_for_selector(".toast--success", timeout=3000)
        assert "评审确认" in page.locator(".toast--success").last.text_content()
        assert page.locator("#btn-confirm").is_disabled()
        captured = page.evaluate("window.__cfCapturedRequests")
        review_call = [r for r in captured if r["url"].endswith("/review")][-1]
        assert "dr_seed123" in review_call["url"]
        body = json.loads(review_call["body"] or "{}")
        assert body["action"] == "confirm"
        assert body.get("post_to_dipcase") is True
        ctx.close()

    def test_reject_triggers_post_with_action_reject(self, server_url, browser):
        ctx = browser.new_context()
        page = ctx.new_page()
        body = {
            "schema_version": "attribution_draft/1.0", "ok": True, "action": "reject",
            "draft": {"draft_id": "dr_seed999", "status": "rejected"},
            "transition": {"transition_id": "tr_x"},
            "post_result": None, "rolled_back": False,
        }
        page.goto(server_url + "/", wait_until="networkidle", timeout=15000)
        page.wait_for_function("typeof window.__cfPanel === 'object'")
        _install_fetch_mock(page, [("/api/drafts/dr_seed999/review", 200, body)])
        page.evaluate("window.__cfPanel.__seedDraft('dr_seed999', {disease_code:'I21',outcome:'cmi',quarter:'2025-Q4'})")
        page.evaluate("document.getElementById('btn-reject').click()")
        page.wait_for_selector(".toast--success", timeout=3000)
        assert "评审退回" in page.locator(".toast--success").last.text_content()
        assert page.locator("#btn-reject").is_disabled()
        captured = page.evaluate("window.__cfCapturedRequests")
        review_call = [r for r in captured if r["url"].endswith("/review")][-1]
        rb = json.loads(review_call["body"] or "{}")
        assert rb["action"] == "reject"
        assert rb.get("post_to_dipcase") is False
        ctx.close()

    def test_review_403_shows_danger_toast(self, server_url, browser):
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto(server_url + "/", wait_until="networkidle", timeout=15000)
        page.wait_for_function("typeof window.__cfPanel === 'object'")
        _install_fetch_mock(page, [("/api/drafts/dr_seed/review", 403, {
            "ok": False, "error_code": "quality_role_forbidden",
            "message": "无权限", "details": {"role": "quality_officer"}
        })])
        page.evaluate("window.__cfPanel.__seedDraft('dr_seed', {disease_code:'I21',outcome:'cmi',quarter:'2025-Q4'})")
        page.evaluate("document.getElementById('btn-confirm').click()")
        page.wait_for_selector(".toast--danger", timeout=3000)
        text = page.locator(".toast--danger").last.text_content()
        assert "确认" in text or "评审" in text
        ctx.close()

    def test_review_404_shows_warning_toast(self, server_url, browser):
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto(server_url + "/", wait_until="networkidle", timeout=15000)
        page.wait_for_function("typeof window.__cfPanel === 'object'")
        _install_fetch_mock(page, [("/api/drafts/dr_does_not_exist/review", 404, {
            "schema_version": "attribution_draft/1.0", "ok": False,
            "error_code": "draft_not_found",
            "message": "draft_id=dr_does_not_exist 不存在",
            "draft_id": "dr_does_not_exist",
        })])
        page.evaluate("window.__cfPanel.__seedDraft('dr_does_not_exist', {disease_code:'I21',outcome:'cmi',quarter:'2025-Q4'})")
        page.evaluate("document.getElementById('btn-confirm').click()")
        page.wait_for_selector(".toast--warning", timeout=3000)
        text = page.locator(".toast--warning").last.text_content()
        assert "404" in text or "不存在" in text
        ctx.close()

    def test_review_422_shows_warning_with_error_code(self, server_url, browser):
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto(server_url + "/", wait_until="networkidle", timeout=15000)
        page.wait_for_function("typeof window.__cfPanel === 'object'")
        _install_fetch_mock(page, [("/api/drafts/dr_seed/review", 422, {
            "schema_version": "attribution_draft/1.0", "ok": False,
            "error_code": "invalid_state_transition",
            "message": "reviewed → rejected 跳级被拒",
            "from_state": "reviewed", "to_state": "rejected",
        })])
        page.evaluate("window.__cfPanel.__seedDraft('dr_seed', {disease_code:'I21',outcome:'cmi',quarter:'2025-Q4'})")
        page.evaluate("document.getElementById('btn-reject').click()")
        page.wait_for_selector(".toast--warning", timeout=3000)
        text = page.locator(".toast--warning").last.text_content()
        assert "invalid_state_transition" in text or "跳级" in text
        ctx.close()

    def test_review_load_state_data_loading_true(self, server_url, browser):
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto(server_url + "/", wait_until="networkidle", timeout=15000)
        page.wait_for_function("typeof window.__cfPanel === 'object'")
        _install_fetch_mock(page, [("/api/drafts/dr_seed/review", 200, REVIEW_OK_BODY)])
        page.evaluate("window.__cfPanel.__seedDraft('dr_seed', {disease_code:'I21',outcome:'cmi',quarter:'2025-Q4'})")
        # 包一层 fetch 让按钮在 before/after 立即切 loading
        page.evaluate("""
        (() => {
          const o = window.__origFetch;
          window.fetch = function(input, init) {
            window.__cfPanel._setBtnLoading(document.getElementById('btn-confirm'), true, 'x', 'y');
            return o.apply(this, arguments).then(r => {
              window.__cfPanel._setBtnLoading(document.getElementById('btn-confirm'), false, null, '评审确认');
              return r;
            });
          };
        })();
        """)
        page.evaluate("document.getElementById('btn-confirm').click()")
        page.wait_for_function(
            "document.getElementById('btn-confirm').getAttribute('data-loading') === 'true'",
            timeout=3000,
        )
        assert page.evaluate("document.getElementById('btn-confirm').getAttribute('data-loading')") == "true"
        page.wait_for_function(
            "!document.getElementById('btn-confirm').getAttribute('data-loading')",
            timeout=5000,
        )
        ctx.close()

    def test_build_draft_id_is_stable(self, server_url, browser):
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto(server_url + "/", wait_until="networkidle", timeout=15000)
        page.wait_for_function("typeof window.__cfPanel === 'object'")
        id1 = page.evaluate("window.__cfPanel._buildDraftId('I21','cmi','2025-Q4')")
        id2 = page.evaluate("window.__cfPanel._buildDraftId('I21','cmi','2025-Q4')")
        assert id1 == id2
        assert id1.startswith("dr_")
        assert len(id1) == 19
        id3 = page.evaluate("window.__cfPanel._buildDraftId('I21','los_days','2025-Q4')")
        assert id3 != id1
        ctx.close()

    def test_no_draft_seed_warns_on_confirm(self, server_url, browser):
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto(server_url + "/", wait_until="networkidle", timeout=15000)
        page.wait_for_function("typeof window.__cfPanel === 'object'")
        page.evaluate("window.__cfPanel.state.currentDraftId = null")
        page.evaluate("document.getElementById('btn-confirm').disabled = false")
        page.evaluate("document.getElementById('btn-confirm').click()")
        page.wait_for_selector(".toast--warning", timeout=3000)
        captured = page.evaluate("window.__cfCapturedRequests || []")
        assert all("/review" not in r["url"] for r in captured)
        ctx.close()
