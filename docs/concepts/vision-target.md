# 视觉目标卡（结构化）

> 本环境无图像生成能力，本文件仅作为**结构化视觉目标卡**，不伪造概念图。
> 真实运行截图由 `scripts/capture_screenshots.sh` 在 Phase 4 收尾采集到 `docs/screenshots/`。

## 三个状态

### 起始态（首屏 / 加载前）

- 浏览器地址栏：`http://localhost:8000/`
- 顶部：水平导航条（产品名 + 当前病种下拉 + 季度选择器 + outcome 选择器）
- 中央：Plotly.js 因果图占位区（深色背景 `#1a1a1a`），显示「请选择病种/季度/outcome 后点建模」提示文字
- 右侧：边栏空（建模完成后才填充）
- 底部：主操作区，「建模」按钮（`btn-primary` 蓝/红色），左下角显示「样本量待估」

### 首个可用瞬间（SSE 流式推送第一条边）

- 因果图占位区出现 1 个节点（治疗方式）+ 1 条带箭头边
- 节点颜色按效应量编码：红 `#d62728`（强效应）、黄 `#ff7f0e`（弱效应）、灰 `#7f7f7f`（无显著）
- 边粗细按 |ate| 编码；p<0.05 实线、否则虚线
- 节点下方小字 12px 效应值（tabular-nums 等宽对齐）
- 顶部 toast：「建模进行中…已收到 1/30 条边」
- 「建模」按钮 disabled，「中止」按钮启用

### 成形态（建模完成 + 反驳 + 草稿就绪）

- 因果图完整渲染（节点按因果强度编码颜色，所有边带效应值文本）
- 顶部 toast：「建模完成，4 档反驳跑完，总判定 robust/partial/unstable（颜色编码）」
- 右侧栏：3 按钮（反事实 / 确认 / 退回）就绪，按钮 hover / active / disabled 三态可见
- 底部：「样本量守门：n=523，sufficient」「楔子强度：因果效应 / 单因素相关 = 1.8x，causal_wedge_strong」
- 反事实按钮点开浮层：节点选择 + 假设值输入 + 5s 内返回 predicted_outcome + 置信区间
- 确认按钮：reviewer 角色可见，pending_review → reviewed → toast「已上报单病种系统」

## 画面类型

- 终端会话：无（产品是常驻 FastAPI + 浏览器面板）
- 桌面窗口：无
- 浏览器界面：**主要形态**（Plotly.js 因果图 + SSE 流式 + 反事实按钮 + 评审三态）
- 服务运行日志：辅助（运维通道，本 README 不展开）
- 导出的报告页：辅助（草稿 JSON 导出 + 终稿 PDF 导出）

## 视觉语言（Design Tokens）

### 配色

| token | 值 | 用途 |
|-------|-----|------|
| `--bg-primary` | `#1a1a1a` | 主背景（深色，专业感） |
| `--bg-card` | `#262626` | 卡片背景 |
| `--bg-elevated` | `#333333` | 浮层 / 模态背景 |
| `--text-primary` | `#f5f5f5` | 主要文字 |
| `--text-secondary` | `#a0a0a0` | 次要文字 |
| `--text-muted` | `#666666` | 辅助说明 |
| `--accent-red` | `#d62728` | 强效应 / danger |
| `--accent-yellow` | `#ff7f0e` | 弱效应 / warning |
| `--accent-gray` | `#7f7f7f` | 无显著 / disabled |
| `--accent-primary` | `#3b82f6` | 主操作按钮 |
| `--success` | `#22c55e` | success toast / confirm |
| `--border` | `#3a3a3a` | 分隔线 |

### 间距栅格（8px）

4 / 8 / 12 / 16 / 24 / 32（px）

### 圆角与阴影

- 卡片：`border-radius: 8px` + `box-shadow: 0 2px 4px rgba(0,0,0,0.2)`
- 浮层：`border-radius: 12px` + `box-shadow: 0 8px 24px rgba(0,0,0,0.4)`
- 模态：`border-radius: 16px` + `box-shadow: 0 16px 48px rgba(0,0,0,0.6)`

### 字体

- 系统中文栈：`-apple-system, "PingFang SC", "Microsoft YaHei", sans-serif`
- 数字：tabular-nums 等宽对齐
- 层级：页面标题 24px / 区块标题 18px / 正文 14px / 辅助说明 12px

## 必须可达的约束（按当前依赖）

- **Plotly.js 因果图**：✅ 已包含 `plotly.min.js` 2.x minified
- **SSE 流式**：✅ FastAPI StreamingResponse + 浏览器 EventSource
- **深色背景 + 三色节点编码**：✅ 设计 token CSS
- **8px 间距栅格**：✅ CSS 变量
- **节点点击 + tooltip**：✅ JS EventSource 订阅 + DOM event listener
- **反事实浮层 + 假设值输入**：✅ 原生 HTML + JS fetch POST
- **三按钮（反事实/确认/退回）+ toast**：✅ 原生 JS + DOM 操作

## 不允许出现的元素（受限于当前依赖）

- ❌ 3D 渲染（WebGL / Three.js 未引入）
- ❌ 视频流（无摄像头需求）
- ❌ 复杂动画（仅 CSS transition 150–250ms）
- ❌ emoji 当图标（用 SVG）
- ❌ 浏览器原生 alert / prompt / confirm（用 toast / 模态）

## 诚实标注

视觉目标与实际实现必然有差距：

- 字体真实装载受操作系统影响（PingFang SC 仅 macOS；Linux 退回到 Microsoft YaHei）
- 节点拖拽、缩放等 Plotly 高级交互受 plotly.min.js 体积限制（默认 2.x minified 不含可选模块）
- 真实截图由 Playwright 采集后写入 `docs/screenshots/`，与本卡视觉目标**不做事后抹平**