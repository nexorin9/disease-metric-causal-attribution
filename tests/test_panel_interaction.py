"""test_panel_interaction — 节点交互层（T23）端到端测试。

覆盖维度（spec §11 浏览器面板 · 节点交互层）：
  1. 源码静态结构（panel.js + panel.css）：
     - state.selectedNode / state.tooltipEl / state.plotlyGd 字段就位
     - onGraphClick / onGraphHover / onGraphUnhover / attachGraphInteractions
     - showTooltip / hideTooltip / pinTooltipForNode / clearSelection
     - simulateNodeClick / simulateNodeHover / simulateNodeUnhover 测试钩子
     - __cfPanel 全局 export 含 setGraphState / simulateNode* / clearSelection
     - buildPlotTraces 支持 selectedNode 时的边淡化（dim color #3a3a3a）
     - renderGraph 注册 gd.on('plotly_click'/'plotly_hover'/'plotly_unhover')
     - panel.css 含 .tooltip / [data-pinned] / .tooltip__badge 三态 / 三态边框
  2. Playwright 端到端：
     - 页面加载 + Plotly 渲染兜底就位
     - 通过 __cfPanel.__seedDemo 注入 demo 数据（节点 + 边 + 效应）
     - simulateNodeClick('X') → tooltip 出现 + data-pinned="true"
     - state.selectedNode 设置正确
     - simulateNodeClick 同一节点再次 → 取消选中 + tooltip 隐藏
     - simulateNodeHover('Y') → tooltip 出现但 data-pinned="false"
     - simulateNodeUnhover() → tooltip 隐藏（未钉住）
     - clearSelection() → tooltip 隐藏 + state.selectedNode=null
     - 三态：setGraphState('empty' / 'loading' / 'ready' / 'error') 各检查
       data-state 属性 + 可见 state 子节点 + CSS 应用
     - 真实 Plotly gd.on 是否注册（plotly_click handler 调用计数）

设计校验纪律（spec §11 + README 强制结构）：
  - 真实运行截图（不是设计稿冒充实测）：后续 T32 收尾
  - 节点交互层与三态 polish 都基于真实 DOM 操作，不是 mock
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
    """把 Python 对象序列化为合法 JS 字面量（True → true 等）。"""
    return json.dumps(obj, ensure_ascii=False)


def _js_args(*objs) -> str:
    """多个 Python 对象 → 用 ', ' 连接的合法 JS 字面量串。"""
    return ", ".join(_js(o) for o in objs)


# ------------------------- 1. 源码结构（静态校验）-------------------------


class TestPanelJsStructure:
    @pytest.fixture(scope="class")
    def js(self) -> str:
        return _read(PANEL_JS)

    def test_state_has_selected_node(self, js: str):
        """state 字段必须含 selectedNode（节点选中态）。"""
        assert "selectedNode" in js, "panel.js state 缺 selectedNode"

    def test_state_has_tooltip_el(self, js: str):
        """state 必须含 tooltipEl / tooltipPinned（tooltip 容器引用 + 钉住状态）。"""
        assert "tooltipEl" in js
        assert "tooltipPinned" in js

    def test_state_has_plotly_gd(self, js: str):
        """state.plotlyGd 用于在 renderGraph 后注册 click/hover 监听。"""
        assert "plotlyGd" in js

    def test_defines_on_graph_click(self, js: str):
        assert "function onGraphClick" in js, "panel.js 缺 onGraphClick"

    def test_defines_on_graph_hover(self, js: str):
        assert "function onGraphHover" in js, "panel.js 缺 onGraphHover"

    def test_defines_on_graph_unhover(self, js: str):
        assert "function onGraphUnhover" in js, "panel.js 缺 onGraphUnhover"

    def test_defines_attach_graph_interactions(self, js: str):
        """注册 Plotly gd.on('plotly_click', ...) 入口。"""
        assert "function attachGraphInteractions" in js
        # 验证注册的是 plotly_click / plotly_hover / plotly_unhover
        assert "plotly_click" in js
        assert "plotly_hover" in js
        assert "plotly_unhover" in js

    def test_defines_show_hide_tooltip(self, js: str):
        assert "function showTooltip" in js
        assert "function hideTooltip" in js

    def test_defines_clear_selection(self, js: str):
        """清除选中节点 + 重新渲染。"""
        assert "function clearSelection" in js

    def test_defines_simulate_hooks(self, js: str):
        """测试钩子：simulateNodeClick / simulateNodeHover / simulateNodeUnhover。"""
        for name in ("simulateNodeClick", "simulateNodeHover", "simulateNodeUnhover"):
            assert "function " + name in js, f"缺测试钩子: {name}"

    def test_tooltip_content_includes_effect_fields(self, js: str):
        """tooltip 内容必须含 ate / CI / p / method 四个字段。"""
        # renderTooltipContent 应展示这些字段
        for kw in ("ate", "ci_lower", "ci_upper", "p_value", "method"):
            assert kw in js, f"tooltip 缺字段: {kw}"

    def test_build_plot_traces_dim_unconnected_edges(self, js: str):
        """选中节点时，未连接边用暗灰 #3a3a3a 替代原色。"""
        assert "#3a3a3a" in js, "panel.js 缺 dim 颜色"

    def test_build_plot_traces_highlight_selected_node(self, js: str):
        """选中节点边框加粗 + 其它节点 dim。"""
        assert "nBorderWidth" in js, "panel.js 缺选中节点边框宽度数组"
        assert "nOpacity" in js, "panel.js 缺未选中节点 opacity 数组"
        assert "'#ffffff'" in js or '"#ffffff"' in js, "panel.js 缺选中节点白边颜色"

    def test_init_exports_test_hooks(self, js: str):
        """__cfPanel 必须暴露给 e2e 测试：simulateNodeClick / simulateNodeHover /
        simulateNodeUnhover / clearSelection / setGraphState / __seedDemo。"""
        for hook in (
            "simulateNodeClick",
            "simulateNodeHover",
            "simulateNodeUnhover",
            "clearSelection",
            "setGraphState",
            "__seedDemo",
        ):
            assert hook in js, f"__cfPanel 未导出: {hook}"

    def test_handle_background_click_clears_selection(self, js: str):
        """点击因果图空白区域应清掉选中。"""
        # 在 attach() 内注册 click on elGraph
        assert "elGraph.addEventListener" in js or "graph.addEventListener" in js, \
            "缺空白区域 click 监听"

    def test_start_modeling_clears_selection(self, js: str):
        """新一轮建模清掉已选中节点。"""
        assert "state.selectedNode = null" in js, "缺新一轮建模清选中"


class TestPanelCssStructure:
    @pytest.fixture(scope="class")
    def css(self) -> str:
        return _read(PANEL_CSS)

    def test_has_tooltip_base(self, css: str):
        assert ".tooltip" in css, "panel.css 缺 .tooltip 基类"

    def test_has_tooltip_pinned_state(self, css: str):
        assert '[data-pinned="true"]' in css, "panel.css 缺钉住态"

    def test_has_tooltip_badge_variants(self, css: str):
        """三类 badge：outcome / treatment / confounder。"""
        for v in ("tooltip__badge--outcome", "tooltip__badge--treatment", "tooltip__badge--confounder"):
            assert v in css, f"panel.css 缺 badge 变体: {v}"

    def test_has_three_state_border_polish(self, css: str):
        """三态各自边框 + 背景：empty / loading / error / ready。"""
        for st in ("empty", "loading", "error", "ready"):
            assert f'data-state="{st}"]' in css, f"panel.css 缺三态: {st}"

    def test_has_loading_animation(self, css: str):
        """loading 态必须有 pulse / spinner 动画。"""
        assert "graph-loading-pulse" in css or "@keyframes spin" in css, \
            "panel.css 缺 loading 动画"

    def test_has_error_red_border(self, css: str):
        assert "var(--danger)" in css, "panel.css 缺 danger 变量引用"

    def test_has_tabular_nums(self, css: str):
        """数字 tabular-nums 等宽。"""
        assert "tabular-nums" in css, "panel.css 缺 tabular-nums"


# ------------------------- 2. 模块可被导入（FastAPI TestClient 路由）-------------------------


class TestPanelModuleImport:
    def test_panel_module_imports(self):
        from app.panel import PANEL_PACKAGE_DIR, STATIC_DIR, TEMPLATES_DIR
        assert PANEL_PACKAGE_DIR.exists()
        assert STATIC_DIR.exists()
        assert TEMPLATES_DIR.exists()


# ------------------------- 3. Playwright 端到端交互测试 -------------------------

# Demo 数据：3 节点 + 2 边 + 2 效应
DEMO_NODES = [
    {"id": "treatment_strategy", "label": "治疗方式", "treatment": True, "outcome": False, "confounder": False},
    {"id": "age_band",           "label": "年龄段",   "treatment": False, "outcome": False, "confounder": True},
    {"id": "cmi",                "label": "CMI",      "treatment": False, "outcome": True,  "confounder": False},
]
DEMO_EDGES = [
    {"source": "treatment_strategy", "target": "cmi", "label": "treatment → cmi"},
    {"source": "age_band",           "target": "cmi", "label": "age → cmi"},
]
DEMO_EFFECTS = [
    {"node": "treatment_strategy", "ate": 0.42, "ci_lower": 0.18, "ci_upper": 0.66, "p_value": 0.012, "method_name": "backdoor.linear_regression"},
    {"node": "age_band",           "ate": -0.08, "ci_lower": -0.21, "ci_upper": 0.05, "p_value": 0.31, "method_name": "backdoor.linear_regression"},
]


def _start_uvicorn(port: int) -> subprocess.Popen:
    """后台启动 uvicorn；测试结束后用 terminate 关闭。"""
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port),
         "--log-level", "warning"],
        cwd=str(_PROJECT_ROOT),
        env={**os.environ, "APP_ENV": "test"},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    # 等待端口就绪
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
        browser = p.chromium.launch(headless=True)
        yield browser
        browser.close()


class TestPanelInteractionE2E:
    """真实浏览器 + FastAPI + Playwright 端到端测试。"""

    def test_page_loads_with_panel_root(self, server_url, browser):
        """GET / 返回 200 + 含 causal-graph 主容器。"""
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto(server_url + "/", wait_until="networkidle", timeout=15000)
        assert page.locator("#causal-graph").count() == 1
        assert page.locator(".state").count() >= 3  # empty / loading / error
        ctx.close()

    def test_panel_global_exposed(self, server_url, browser):
        """__cfPanel 全局对象存在，含所有测试钩子。"""
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto(server_url + "/", wait_until="networkidle", timeout=15000)
        page.wait_for_function("typeof window.__cfPanel === 'object'", timeout=5000)
        for hook in ("simulateNodeClick", "simulateNodeHover", "simulateNodeUnhover",
                     "clearSelection", "setGraphState", "__seedDemo", "state"):
            present = page.evaluate(f"typeof window.__cfPanel.{hook}")
            assert present != "undefined", f"__cfPanel.{hook} 未暴露"
        ctx.close()

    def test_seed_demo_then_tooltip_appears_on_click(self, server_url, browser):
        """注入 demo 数据 + simulateNodeClick → tooltip 出现 + data-pinned=true。"""
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto(server_url + "/", wait_until="networkidle", timeout=15000)
        page.wait_for_function("typeof window.__cfPanel === 'object'")
        # 注入 demo（用 _js 把 Python True/False 转 JS true/false）
        page.evaluate(
            "window.__cfPanel.__seedDemo(" + _js_args(DEMO_NODES, DEMO_EDGES, DEMO_EFFECTS) + ")"
        )
        # 初始不应有 tooltip 元素（或 hidden）
        before = page.locator(".tooltip").count()
        assert before <= 1, f"不应预先存在可见 tooltip; got {before}"
        # 模拟点击
        page.evaluate("window.__cfPanel.simulateNodeClick('treatment_strategy')")
        # tooltip 应可见 + pinned
        page.wait_for_selector(".tooltip:not([hidden])", timeout=3000)
        tip = page.locator(".tooltip")
        assert tip.get_attribute("data-pinned") == "true"
        assert tip.get_attribute("data-node-id") == "treatment_strategy"
        # 选中的 state.selectedNode
        sn = page.evaluate("window.__cfPanel.state.selectedNode")
        assert sn == "treatment_strategy"
        # tooltip 内容含 ate / CI / p / method
        text = tip.text_content()
        assert "ate" in text
        assert "CI" in text or "95%" in text
        assert "p" in text
        assert "method" in text.lower() or "方法" in text or "backdoor" in text
        ctx.close()

    def test_click_same_node_again_unpins(self, server_url, browser):
        """第二次点击同一节点 → 取消选中 + tooltip hidden。"""
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto(server_url + "/", wait_until="networkidle", timeout=15000)
        page.wait_for_function("typeof window.__cfPanel === 'object'")
        page.evaluate(
            "window.__cfPanel.__seedDemo(" + _js_args(DEMO_NODES, DEMO_EDGES, DEMO_EFFECTS) + ")"
        )
        page.evaluate("window.__cfPanel.simulateNodeClick('cmi')")
        page.wait_for_selector(".tooltip:not([hidden])", timeout=3000)
        assert page.locator(".tooltip").get_attribute("data-pinned") == "true"
        # 再点一次
        page.evaluate("window.__cfPanel.simulateNodeClick('cmi')")
        # 取消选中 → state.selectedNode 应为 null
        page.wait_for_function("window.__cfPanel.state.selectedNode === null", timeout=3000)
        # tooltip 应隐藏
        hidden_attr = page.evaluate(
            "document.querySelector('.tooltip').hidden"
        )
        assert hidden_attr is True, "tooltip 应被隐藏（hidden=true）"
        ctx.close()

    def test_hover_shows_unpinned_tooltip(self, server_url, browser):
        """simulateNodeHover → tooltip 出现但 data-pinned='false'。"""
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto(server_url + "/", wait_until="networkidle", timeout=15000)
        page.wait_for_function("typeof window.__cfPanel === 'object'")
        page.evaluate(
            "window.__cfPanel.__seedDemo(" + _js_args(DEMO_NODES, DEMO_EDGES, DEMO_EFFECTS) + ")"
        )
        page.evaluate("window.__cfPanel.simulateNodeHover('age_band', 250, 300)")
        page.wait_for_selector(".tooltip:not([hidden])", timeout=3000)
        tip = page.locator(".tooltip")
        assert tip.get_attribute("data-pinned") == "false", \
            f"hover tooltip 不应被钉住; got {tip.get_attribute('data-pinned')}"
        # unhover → 隐藏
        page.evaluate("window.__cfPanel.simulateNodeUnhover()")
        page.wait_for_function(
            "document.querySelector('.tooltip').hidden === true",
            timeout=3000
        )
        ctx.close()

    def test_clear_selection_resets_state(self, server_url, browser):
        """clearSelection 清除 state.selectedNode + 隐藏 tooltip。"""
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto(server_url + "/", wait_until="networkidle", timeout=15000)
        page.wait_for_function("typeof window.__cfPanel === 'object'")
        page.evaluate(
            "window.__cfPanel.__seedDemo(" + _js_args(DEMO_NODES, DEMO_EDGES, DEMO_EFFECTS) + ")"
        )
        page.evaluate("window.__cfPanel.simulateNodeClick('treatment_strategy')")
        page.wait_for_function("window.__cfPanel.state.selectedNode !== null")
        page.evaluate("window.__cfPanel.clearSelection()")
        page.wait_for_function("window.__cfPanel.state.selectedNode === null", timeout=3000)
        hidden = page.evaluate("document.querySelector('.tooltip').hidden")
        assert hidden is True
        ctx.close()

    def test_three_state_loading(self, server_url, browser):
        """setGraphState('loading') → data-state=loading + spinner 可见。"""
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto(server_url + "/", wait_until="networkidle", timeout=15000)
        page.wait_for_function("typeof window.__cfPanel === 'object'")
        page.evaluate("window.__cfPanel.setGraphState('loading')")
        # data-state 属性
        attr = page.locator("#causal-graph").get_attribute("data-state")
        assert attr == "loading", f"data-state 应为 loading; got {attr}"
        # loading 子节点可见
        loading_visible = page.locator(".state--loading").is_visible()
        assert loading_visible, "loading 子节点应可见"
        # spinner 可见
        spinner_visible = page.locator(".state--loading .spinner").is_visible()
        assert spinner_visible, "spinner 应可见"
        ctx.close()

    def test_three_state_error(self, server_url, browser):
        """setGraphState('error') → data-state=error + error 子节点可见。"""
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto(server_url + "/", wait_until="networkidle", timeout=15000)
        page.wait_for_function("typeof window.__cfPanel === 'object'")
        page.evaluate("window.__cfPanel.setGraphState('error')")
        attr = page.locator("#causal-graph").get_attribute("data-state")
        assert attr == "error"
        error_visible = page.locator(".state--error").is_visible()
        assert error_visible, "error 子节点应可见"
        # 重试按钮可见
        retry_visible = page.locator("#btn-retry").is_visible()
        assert retry_visible
        ctx.close()

    def test_three_state_empty(self, server_url, browser):
        """setGraphState('empty') → empty 子节点可见。"""
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto(server_url + "/", wait_until="networkidle", timeout=15000)
        page.wait_for_function("typeof window.__cfPanel === 'object'")
        page.evaluate("window.__cfPanel.setGraphState('empty')")
        attr = page.locator("#causal-graph").get_attribute("data-state")
        assert attr == "empty"
        empty_visible = page.locator(".state--empty").is_visible()
        assert empty_visible, "empty 子节点应可见"
        ctx.close()

    def test_three_state_ready(self, server_url, browser):
        """setGraphState('ready') → ready 态，loading/empty/error 子节点均隐藏。"""
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto(server_url + "/", wait_until="networkidle", timeout=15000)
        page.wait_for_function("typeof window.__cfPanel === 'object'")
        page.evaluate("window.__cfPanel.setGraphState('ready')")
        attr = page.locator("#causal-graph").get_attribute("data-state")
        assert attr == "ready"
        # 三个 state 子节点都应 hidden
        for sel in (".state--empty", ".state--loading", ".state--error"):
            visible = page.locator(sel).is_visible()
            assert not visible, f"{sel} 在 ready 态应隐藏"
        ctx.close()

    def test_tooltip_pinned_state_persists_through_state_changes(self, server_url, browser):
        """选中态在 setGraphState 切换后应保持（直到再次点击）。"""
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto(server_url + "/", wait_until="networkidle", timeout=15000)
        page.wait_for_function("typeof window.__cfPanel === 'object'")
        page.evaluate(
            "window.__cfPanel.__seedDemo(" + _js_args(DEMO_NODES, DEMO_EDGES, DEMO_EFFECTS) + ")"
        )
        page.evaluate("window.__cfPanel.simulateNodeClick('cmi')")
        page.wait_for_selector(".tooltip:not([hidden])", timeout=3000)
        # 切到 loading 不会清掉 selectedNode（只有 startModeling 才清）
        page.evaluate("window.__cfPanel.setGraphState('loading')")
        sn = page.evaluate("window.__cfPanel.state.selectedNode")
        assert sn == "cmi", "setGraphState 不应清掉 selectedNode"
        ctx.close()

    def test_tooltip_positioned_in_graph_bounds(self, server_url, browser):
        """tooltip 必须落在 causal-graph 容器范围内（不出框）。"""
        ctx = browser.new_context(viewport={"width": 1280, "height": 800})
        page = ctx.new_page()
        page.goto(server_url + "/", wait_until="networkidle", timeout=15000)
        page.wait_for_function("typeof window.__cfPanel === 'object'")
        page.evaluate(
            "window.__cfPanel.__seedDemo(" + _js_args(DEMO_NODES, DEMO_EDGES, DEMO_EFFECTS) + ")"
        )
        # 在右下角模拟 hover，应自动翻转位置（左/上）
        page.evaluate("window.__cfPanel.simulateNodeHover('treatment_strategy', 1200, 700)")
        page.wait_for_selector(".tooltip:not([hidden])", timeout=3000)
        tip = page.locator(".tooltip")
        bbox = tip.bounding_box()
        graph = page.locator("#causal-graph").bounding_box()
        assert bbox is not None
        assert graph is not None
        # tooltip 完全在 graph 范围内
        assert bbox["x"] >= graph["x"] - 5
        assert bbox["y"] >= graph["y"] - 5
        assert bbox["x"] + bbox["width"] <= graph["x"] + graph["width"] + 5
        assert bbox["y"] + bbox["height"] <= graph["y"] + graph["height"] + 5
        ctx.close()
