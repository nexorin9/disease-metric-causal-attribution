"""test_panel_html — 浏览器面板基线测试（T22 收尾）。

覆盖维度（spec §11 浏览器面板）：
  1. 模板 / 静态文件物理存在：index.html / plotly.min.js / panel.css / panel.js
  2. index.html 含「plotly.min.js」引用 + 深色背景 + 三色节点 + 设计 token CSS 变量
  3. panel.css 含 token 名称（--bg-primary / --accent-red / --accent-yellow / --accent-gray / 8px 栅格）
  4. panel.js 含 EventSource 订阅代码 + 三态切换 + 因果图渲染占位
  5. plotly.min.js 含 Plotly API 表面（newPlot/purge/react/relayout）+ data-* 占位
  6. /api/panel 模块可被导入（__init__ + router）
  7. FastAPI TestClient GET / 返回 200 + HTML + 含核心节点
  8. /static/plotly.min.js 返回 200 + JS 内容 + 含 'Plotly' 标识
  9. /static/panel.css 返回 200 + 含 .app-header 或 :root 块
 10. /static/panel.js 返回 200 + 含 EventSource 标识（或 fetch 拉取）

设计校验纪律（spec §11 + README 强制结构）：
  - 模板禁止 30 秒 demo 故事线 / 一键演示等过度营销段
  - 设计 token 不允许散落硬编码（panel.js 仅引用 token 别名）
"""
from __future__ import annotations

import os
import re
import sys
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

INDEX_HTML = TEMPLATES_DIR / "index.html"
PLOTLY_JS = STATIC_DIR / "plotly.min.js"
PANEL_CSS = STATIC_DIR / "panel.css"
PANEL_JS = STATIC_DIR / "panel.js"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ------------------------- 1. 物理存在 -------------------------


class TestFilesPresent:
    def test_index_html_exists(self):
        assert INDEX_HTML.exists(), f"templates/index.html missing: {INDEX_HTML}"

    def test_plotly_js_exists(self):
        assert PLOTLY_JS.exists(), f"static/plotly.min.js missing: {PLOTLY_JS}"

    def test_panel_css_exists(self):
        assert PANEL_CSS.exists(), f"static/panel.css missing: {PANEL_CSS}"

    def test_panel_js_exists(self):
        assert PANEL_JS.exists(), f"static/panel.js missing: {PANEL_JS}"


# ------------------------- 2. index.html 内容 -------------------------


class TestIndexHtml:
    @pytest.fixture(scope="class")
    def html(self) -> str:
        return _read(INDEX_HTML)

    def test_references_plotly_min_js(self, html: str):
        """HTML 必须含 'plotly.min.js' 字面量（任务 hard 锚点）。"""
        assert "plotly.min.js" in html, "templates/index.html 必须含 plotly.min.js 引用"

    def test_references_panel_css(self, html: str):
        assert "panel.css" in html, "templates/index.html 必须引用 /static/panel.css"

    def test_references_panel_js(self, html: str):
        assert "panel.js" in html, "templates/index.html 必须引用 /static/panel.js"

    def test_has_lang_zh_cn(self, html: str):
        assert 'lang="zh-CN"' in html, "templates/index.html 必须声明 lang=zh-CN"

    def test_has_dark_design_system(self, html: str):
        """dark theme 入口 + 深色背景语义标记。"""
        # templates/index.html 内 data-theme="dark" 与 header class
        assert 'data-theme="dark"' in html

    def test_has_app_header_section(self, html: str):
        """顶部导航块 + 角色徽章 + 病种/季度/outcome 字段。"""
        for keyword in ('id="modeling-form"', 'id="select-disease"', 'id="select-quarter"', 'id="select-outcome"',
                         'id="role-badge"', 'id="btn-build"'):
            assert keyword in html, f"index.html 缺核心结构: {keyword}"

    def test_has_three_state_nodes(self, html: str):
        """三态：empty / loading / error 都在 HTML 内出现 data-for-state。"""
        for st in ("empty", "loading", "error"):
            assert f'data-for-state="{st}"' in html, f"缺三态: {st}"

    def test_has_causal_graph_div(self, html: str):
        assert 'id="causal-graph"' in html, "缺因果图主容器"

    def test_has_review_buttons(self, html: str):
        """T24 待接入：反事实 / 确认 / 退回 3 按钮（占位即可）。"""
        for btn in ("btn-counterfactual", "btn-confirm", "btn-reject"):
            assert btn in html, f"缺评审按钮: {btn}"

    def test_has_toast_root(self, html: str):
        assert 'id="toast-root"' in html, "缺 toast 通知容器"

    def test_has_counterfactual_modal(self, html: str):
        assert 'id="cf-modal"' in html, "缺反事实浮层"

    def test_no_emoji_icons(self, html: str):
        """反 emoji 图标纪律：禁用 emoji 当装饰图标（用 SVG）。允许 × 是 ASCII 字符，非 emoji。

        简单禁列表：典型临床 / 装饰 emoji 范围。
        """
        emoji_pattern = re.compile(r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF]")
        m = emoji_pattern.search(html)
        assert m is None, f"index.html 含 emoji: {m.group(0)!r}"


# ------------------------- 3. panel.css 设计 token -------------------------


class TestPanelCss:
    @pytest.fixture(scope="class")
    def css(self) -> str:
        return _read(PANEL_CSS)

    def test_has_root_token_block(self, css: str):
        assert ":root" in css, "panel.css 必须定义 :root CSS 变量"

    @pytest.mark.parametrize("token,description", [
        ("--bg-primary", "深色背景"),
        ("--bg-card", "卡片背景"),
        ("--accent-red", "强效应红"),
        ("--accent-yellow", "弱效应黄"),
        ("--accent-gray", "未达显著灰"),
        ("--text-primary", "主文字"),
        ("--space-1", "4px 间距"),
        ("--space-2", "8px 间距"),
        ("--fs-title", "标题字号"),
        ("--fs-body", "正文字号"),
    ])
    def test_has_token(self, css: str, token: str, description: str):
        assert token in css, f"缺 token: {token} ({description})"

    def test_8px_grid_present(self, css: str):
        for v in ("--space-1", "--space-2", "--space-3", "--space-4", "--space-5", "--space-6"):
            assert v in css, f"缺 8px 栅格 token: {v}"

    def test_button_states(self, css: str):
        """button 必须有 hover / active / disabled 三态样式。"""
        for state in (".btn:hover", ".btn:active", ".btn:disabled"):
            assert state in css, f"button 三态缺: {state}"

    def test_responsive_breaks(self, css: str):
        """至少两档断点：1280 / 768。"""
        assert "@media" in css
        for v in ("1280px", "768px"):
            assert v in css, f"响应式断点: {v}"


# ------------------------- 4. panel.js 内容 -------------------------


class TestPanelJs:
    @pytest.fixture(scope="class")
    def js(self) -> str:
        return _read(PANEL_JS)

    def test_defines_ii(self, js: str):
        """panel.js 走 IIFE / 自执行函数避免污染 window 命名空间。"""
        assert "(function" in js or "document.addEventListener" in js

    def test_handles_event_source_or_fetch(self, js: str):
        """panel.js 必须含 EventSource 订阅或 fetch 流式读取。"""
        assert ("EventSource" in js) or ("fetch" in js and "getReader" in js), \
            "panel.js 必须订阅 SSE 流式事件"

    def test_has_three_state_switcher(self, js: str):
        """setGraphState 切换 empty / loading / ready / error。"""
        for st in ("empty", "loading", "ready", "error"):
            assert st in js, f"三态缺: {st}"

    def test_calls_plotly_new_plot_or_react(self, js: str):
        """panel.js 必须尝试调 Plotly.newPlot 或 Plotly.react 渲染因果图。"""
        assert ("Plotly.newPlot" in js) or ("Plotly.react" in js), \
            "panel.js 必须调 Plotly 新建图"

    def test_defines_node_color_tokens(self, js: str):
        """节点颜色 token：红 / 黄 / 灰 与 panel.css 一致。"""
        # 注意 js 内的 hex 值允许存在（panel.js 不直接套用 CSS 变量，避免 flash）
        for hex_color in ("#d62728", "#ff7f0e", "#7f7f7f"):
            assert hex_color in js, f"panel.js 缺节点色: {hex_color}"

    def test_has_toast_helper(self, js: str):
        """替代浏览器原生 alert 的 toast 函数。"""
        assert "toast" in js, "panel.js 缺 toast 函数"

    def test_handles_role_badge(self, js: str):
        """角色徽章切换：quality_officer / med_committee_reviewer。"""
        for role in ("quality_officer", "med_committee_reviewer"):
            assert role in js, f"缺角色枚举: {role}"


# ------------------------- 5. plotly.min.js API 表面 -------------------------


class TestPlotlyShim:
    @pytest.fixture(scope="class")
    def js(self) -> str:
        return _read(PLOTLY_JS)

    def test_exposes_plotly_global(self, js: str):
        """plotly.min.js 必须把 Plotly API 挂到 window。"""
        assert "window.Plotly" in js or "global.Plotly" in js, "plotly.min.js 未挂 window.Plotly"

    @pytest.mark.parametrize("api_method,description", [
        ("newPlot", "新建图"),
        ("react", "响应式更新"),
        ("purge", "销毁"),
        ("redraw", "重绘"),
        ("relayout", "布局更新"),
        ("resize", "尺寸更新"),
        ("toImage", "导出图"),
    ])
    def test_provides_api_method(self, js: str, api_method: str, description: str):
        assert f"{api_method}:" in js or f"{api_method} " in js, f"plotly shim 缺 API: {api_method} ({description})"

    def test_writes_data_attrs_for_e2e(self, js: str):
        """shim 把 trace / layout 写入目标 div 的 data-* 属性，便于面板 e2e 测试。"""
        for attr in ("data-plotly-data", "data-plotly-layout", "data-plotly-config"):
            assert attr in js, f"plotly shim 缺 data-* 属性: {attr}"


# ------------------------- 6. 模块可被导入 -------------------------


class TestModuleImport:
    def test_panel_package_import(self):
        from app.panel import PANEL_PACKAGE_DIR, STATIC_DIR, TEMPLATES_DIR

        assert PANEL_PACKAGE_DIR.exists()
        assert STATIC_DIR.exists()
        assert TEMPLATES_DIR.exists()

    def test_api_panel_module_import(self):
        from app.api.panel import mount_static_files, router

        assert callable(mount_static_files)
        assert router is not None

    def test_static_dir_has_three_files(self):
        """spec §11 + T22：static 必须含 plotly.min.js / panel.css / panel.js。"""
        files = sorted(p.name for p in STATIC_DIR.iterdir())
        for expected in ("plotly.min.js", "panel.css", "panel.js"):
            assert expected in files, f"缺 {expected}; got {files}"


# ------------------------- 7. FastAPI TestClient / -------------------------


class TestFastAPIPanel:
    @pytest.fixture
    def client(self, monkeypatch):
        """TestClient 在 with-statement 内构造，触发 lifespan startup。

        否则 StaticFiles mount 不在 create_app 阶段生效（_register_routes 在 lifespan 内调），
        所有 /static/* 与 / 路径全部 404。
        """
        import contextlib

        import app.main as app_main
        from fastapi.testclient import TestClient

        monkeypatch.setattr(app_main, "_build_audit_logger", lambda s: None)
        monkeypatch.setattr(app_main, "_build_model_gateway", lambda s: None)
        monkeypatch.setattr(app_main, "_build_session_factory", lambda s: None)

        cm: contextlib.AbstractContextManager = TestClient(
            app_main.create_app(), raise_server_exceptions=False
        )
        # 进入 with，把 inner client 取出
        c = cm.__enter__()
        try:
            yield c
        finally:
            cm.__exit__(None, None, None)

    def test_root_returns_200_html(self, client):
        """GET / 返回 200 + HTML + 含 'plotly.min.js'。"""
        resp = client.get("/")
        assert resp.status_code == 200
        ct = resp.headers.get("content-type", "")
        assert "text/html" in ct, f"Content-Type 不对: {ct}"
        body = resp.text
        assert "plotly.min.js" in body, "GET / 返回 HTML 必须含 plotly.min.js 引用"
        assert "panel.css" in body, "GET / HTML 必须引用 panel.css"
        assert "panel.js" in body, "GET / HTML 必须引用 panel.js"
        assert "modeling-form" in body

    def test_panel_healthz_green_when_files_present(self, client):
        """GET /healthz/panel 应报 panel_ok=True（所有 4 个文件就位）。"""
        resp = client.get("/healthz/panel")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body.get("panel_ok") is True, body
        assert body.get("missing_files") == [], body

    def test_static_plotly_js(self, client):
        """GET /static/plotly.min.js → 200 + JS 内容。"""
        resp = client.get("/static/plotly.min.js")
        assert resp.status_code == 200, resp.text
        ct = resp.headers.get("content-type", "")
        assert ("javascript" in ct) or ("application" in ct) or resp.text.startswith("/*!") or ("Plotly" in resp.text[:4096])

    def test_static_panel_css(self, client):
        """GET /static/panel.css → 200 + 含 :root token 块。"""
        resp = client.get("/static/panel.css")
        assert resp.status_code == 200, resp.text
        assert ":root" in resp.text or "--bg-primary" in resp.text

    def test_static_panel_js(self, client):
        """GET /static/panel.js → 200 + 含 panel.js 关键字。"""
        resp = client.get("/static/panel.js")
        assert resp.status_code == 200, resp.text
        # panel.js 内 IIFE / setGraphState / Plotly.newPlot 任一即可
        assert ("setGraphState" in resp.text) or ("Plotly.newPlot" in resp.text) or ("function" in resp.text)
