/* ===========================================================================
 * 浏览器面板交互层（spec.md §11 浏览器面板 + §6 失败不静默改数据）
 *
 * 单一职责：把 SSE 流翻译成 Plotly 图 + 三态 UI + toast。
 * 关键不变量：
 *   - 所有写回（评审 confirm / reject / 上报）经 reviewer 确认才动；
 *   - 任意一条 effect 估算 CI 超宽 → 节点画"不显著"，不断渲染冒绿；
 *   - 反事实按钮只在建模 ready 后启用，按下走 POST /api/counterfactual/run；
 *   - 浏览器原生 alert/prompt/confirm 全替为 toast / modal；
 *
 * 设计约束（spec §11 + 视觉目标卡）：
 *   - 节点颜色：red=强效应（|ate|>0.3 且 p<0.05）/ yellow=弱效应 / gray=未达显著
 *   - 边线：p<0.05 实线，否则虚线；边文本显示 ate + ci
 *   - 三态：loading（建模中）/ empty（待选择）/ error（建模失败重试）
 *
 * Phase 4 任务分工：
 *   - T22（本任务）：基线样式 + 因果图渲染 + EventSource 订阅起点（不写真实交互）
 *   - T23：节点点击 + tooltip + 三态切换 polish
 *   - T24：反事实 / 确认 / 退回 3 按钮 + toast 通知
 * ========================================================================= */
(function (global) {
  'use strict';

  /* ----- 设计 token（与 panel.css 一致，JS 仅读取不出编号）----- */
  var COLOR = {
    strong: '#d62728',
    weak:   '#ff7f0e',
    none:   '#7f7f7f',
    bg:     '#262626',
    text:   '#f5f5f5',
    edge:   '#888888',
  };

  var ROLE_STORAGE_KEY = 'cf.quality_role';

  /* ----- 元素引用 ----- */
  var elForm       = document.getElementById('modeling-form');
  var elBtnBuild   = document.getElementById('btn-build');
  var elBtnAbort   = document.getElementById('btn-abort');
  var elBtnCf      = document.getElementById('btn-counterfactual');
  var elBtnConfirm = document.getElementById('btn-confirm');
  var elBtnReject  = document.getElementById('btn-reject');
  var elGraph      = document.getElementById('causal-graph');
  var elToastRoot  = document.getElementById('toast-root');
  var elRoleBadge  = document.getElementById('role-badge');
  var elSummary    = document.getElementById('summary-list');

  var elFooterSampleSize = document.getElementById('footer-sample-size');
  var elFooterVerdict    = document.getElementById('footer-verdict');
  var elFooterRefute     = document.getElementById('footer-refute');
  var elFooterWedge      = document.getElementById('footer-wedge');
  var elProgressText     = document.getElementById('progress-text');
  var elErrorText        = document.getElementById('error-text');

  var state = {
    role:          readRole(),
    sseSource:     null,
    nodes:         [],
    edges:         [],
    effects:       [],
    refutations:   [],
    wedge:         null,
    sampleSize:    null,
    sampleVerdict: null,
    busy:          false,
    // T23: 节点交互状态
    selectedNode:  null,        // 当前选中的节点 id；null 表示未选
    tooltipPinned: false,       // 鼠标 hover 时 false；点击节点后 true
    hoveredNode:   null,        // 当前 hover 的节点 id
    plotlyGd:      null,        // Plotly graph div 引用（用于 attach click/hover）
  };

  /* ============================================================
   * Toast（替代原生 alert）
   * ============================================================ */
  function toast(message, kind) {
    if (!elToastRoot) return;
    var kindClass = ['success', 'warning', 'danger', 'info'].indexOf(kind) >= 0
      ? 'toast--' + kind : 'toast--info';
    var div = document.createElement('div');
    div.className = 'toast ' + kindClass;
    div.textContent = message;
    elToastRoot.appendChild(div);
    setTimeout(function () {
      div.style.transition = 'opacity 200ms';
      div.style.opacity = '0';
      setTimeout(function () { if (div.parentNode) div.parentNode.removeChild(div); }, 250);
    }, 3000);
  }

  /* ============================================================
   * 三态切换（empty / loading / ready / error）
   * ============================================================ */
  function setGraphState(st) {
    if (!elGraph) return;
    elGraph.setAttribute('data-state', st);
    // 显隐各 state 子块
    var states = elGraph.querySelectorAll('.state');
    for (var i = 0; i < states.length; i++) {
      var s = states[i];
      var for_ = s.getAttribute('data-for-state');
      s.hidden = (for_ !== st);
    }
  }

  /* ============================================================
   * 角色徽章（持久化于 localStorage；测试可注入）
   * ============================================================ */
  function readRole() {
    try {
      var v = global.localStorage && global.localStorage.getItem(ROLE_STORAGE_KEY);
      if (v === 'quality_officer' || v === 'med_committee_reviewer') return v;
    } catch (e) { /* localStorage may be disabled */ }
    return 'quality_officer';
  }

  function setRole(role) {
    state.role = role;
    try { global.localStorage && global.localStorage.setItem(ROLE_STORAGE_KEY, role); } catch (e) {}
    if (elRoleBadge) {
      elRoleBadge.setAttribute('data-role', role);
      var txt = elRoleBadge.querySelector('.role-text');
      if (txt) txt.textContent = role === 'med_committee_reviewer' ? '医务科评审' : '质控员';
    }
  }

  /* ============================================================
   * 因果图构建（Plotly 2.x trace：Scatter + 节点散点）
   * 设计：节点颜色按 |ate| 和 p_value；边线按 p_value 选 solid / dash
   * ============================================================ */
  function classifyEffect(e) {
    if (e == null) return 'none';
    var p = e.p_value, ate = e.ate;
    if (ate == null) return 'none';
    if (p != null && p < 0.05 && Math.abs(ate) >= 0.3) return 'strong';
    if (p != null && p < 0.05) return 'weak';
    return 'none';
  }

  function buildPlotTraces() {
    var g = window.Plotly;
    if (!g || typeof g.newPlot !== 'function') {
      // Plotly shim 缺席时降级（不抛错，留 placeholder）
      return null;
    }

    var nodesById = {};
    for (var i = 0; i < state.nodes.length; i++) {
      nodesById[state.nodes[i].id] = state.nodes[i];
    }

    // 边坐标（用简单环形布局）
    var n = state.nodes.length || 1;
    var radius = 1.0;
    var pos = {};
    for (var i2 = 0; i2 < n; i2++) {
      var theta = 2 * Math.PI * i2 / n;
      pos[state.nodes[i2].id] = [Math.cos(theta) * radius, Math.sin(theta) * radius];
    }

    // 边 trace（折线 + 注释）；T23：支持选中节点时其它边变灰（dim）
    var edgeX = [], edgeY = [], edgeText = [], edgeLineShape = [];
    var edgeSource = [], edgeTarget = [];
    var edgeColors = []; // Plotly line.color 支持按段数组
    var connectedEdges = []; // 用于节点边框高亮
    if (state.selectedNode) {
      connectedEdges = computeConnectedEdgeIndices(state.selectedNode);
    }
    var hasSelection = !!state.selectedNode;
    for (var e2 = 0; e2 < state.edges.length; e2++) {
      var ed = state.edges[e2];
      var sId = ed.source, tId = ed.target;
      if (!pos[sId] || !pos[tId]) continue;
      edgeX.push(pos[sId][0], pos[tId][0], null);
      edgeY.push(pos[sId][1], pos[tId][1], null);
      // 找该边对应的 effect（按 source 节点）
      var eff = null;
      for (var e3 = 0; e3 < state.effects.length; e3++) {
        if (state.effects[e3].node === sId) { eff = state.effects[e3]; break; }
      }
      var effCls = classifyEffect(eff);
      var isSignificant = eff && eff.p_value != null && eff.p_value < 0.05;
      var isConnected = !hasSelection || connectedEdges.indexOf(e2) >= 0;
      var baseColor = effCls === 'strong' ? COLOR.strong : (effCls === 'weak' ? COLOR.weak : COLOR.none);
      // 选中时未连接的边 → 极暗灰（不静默改数据，仅视觉）
      edgeColors.push(isConnected ? baseColor : '#3a3a3a');
      edgeLineShape.push({
        color: isConnected ? baseColor : '#3a3a3a',
        width: isConnected ? (effCls === 'strong' ? 3 : (effCls === 'weak' ? 2 : 1)) : 1,
        dash: isSignificant ? 'solid' : 'dot',
      });
      edgeText.push(eff ? (ed.label + ' (ate=' + (eff.ate == null ? '—' : eff.ate.toFixed(2)) + ')') : ed.label);
      edgeSource.push(sId);
      edgeTarget.push(tId);
    }

    var edgeTrace = {
      x: edgeX, y: edgeY, mode: 'lines', type: 'scatter',
      line: { width: 1, color: edgeColors.length ? edgeColors : COLOR.edge },
      hoverinfo: 'text',
      text: edgeText,
    };

    // 节点 trace（T23：选中节点加粗白边 + 未选中节点在选中时淡化）
    var nx = [], ny = [], nText = [], nColor = [], nSymbol = [], nSize = [];
    var nBorderColor = [], nBorderWidth = [], nOpacity = [];
    var hoverText = [];
    var hasSelectionFlag = !!state.selectedNode;
    for (var i3 = 0; i3 < state.nodes.length; i3++) {
      var node = state.nodes[i3];
      var p2 = pos[node.id];
      if (!p2) continue;
      nx.push(p2[0]); ny.push(p2[1]);
      nText.push(node.label || node.id);
      var c = node.outcome ? COLOR.strong : (node.treatment ? COLOR.weak : COLOR.none);
      nColor.push(c);
      nSymbol.push(node.outcome ? 'diamond' : (node.treatment ? 'square' : 'circle'));
      var baseSize = node.outcome ? 22 : (node.treatment ? 18 : 14);
      var isSelected = state.selectedNode === node.id;
      nSize.push(isSelected ? baseSize + 6 : baseSize);
      nBorderColor.push(isSelected ? '#ffffff' : '#000');
      nBorderWidth.push(isSelected ? 3 : 1);
      // 选中时非选中节点淡化（视觉清晰，spec §11 节点交互）
      var dimmed = hasSelectionFlag && !isSelected;
      nOpacity.push(dimmed ? 0.35 : 1.0);
      var eff2 = null;
      for (var e4 = 0; e4 < state.effects.length; e4++) {
        if (state.effects[e4].node === node.id) { eff2 = state.effects[e4]; break; }
      }
      hoverText.push(node.id + (eff2 ? ' / ate=' + (eff2.ate == null ? '—' : eff2.ate.toFixed(3)) : ''));
    }

    var nodeTrace = {
      x: nx, y: ny, mode: 'markers+text', type: 'scatter',
      marker: {
        color: nColor, symbol: nSymbol, size: nSize,
        line: { color: nBorderColor, width: nBorderWidth },
        opacity: nOpacity,
      },
      text: nText, textposition: 'bottom center',
      textfont: { color: COLOR.text, size: 12 },
      hovertext: hoverText, hoverinfo: 'text',
    };

    var layout = {
      paper_bgcolor: COLOR.bg,
      plot_bgcolor: COLOR.bg,
      font: { color: COLOR.text, family: 'inherit' },
      showlegend: false,
      xaxis: { visible: false, range: [-1.5, 1.5] },
      yaxis: { visible: false, range: [-1.5, 1.5] },
      margin: { l: 16, r: 16, t: 16, b: 16 },
    };

    return { traces: [edgeTrace, nodeTrace], layout: layout };
  }

  function renderGraph() {
    var built = buildPlotTraces();
    if (!built) {
      setGraphState('ready');
      toast('Plotly 渲染器未装载（plotly.min.js 缺失）', 'warning');
      return;
    }
    setGraphState('ready');
    try {
      window.Plotly.newPlot('causal-graph', built.traces, built.layout, { displayModeBar: false, responsive: true }).then(function (gd) {
        state.plotlyGd = gd;
        attachGraphInteractions(gd);
        // 已有选中节点：重渲染后还原 tooltip
        if (state.selectedNode) {
          var node = findNodeById(state.selectedNode);
          if (node) {
            pinTooltipForNode(state.selectedNode);
          }
        }
      });
    } catch (e) {
      // 渲染异常走 error 态
      setGraphState('error');
      if (elErrorText) elErrorText.textContent = '因果图渲染失败：' + (e && e.message ? e.message : 'unknown');
    }
  }

  /* ============================================================
   * SSE 流式建模
   * ============================================================ */
  function startModeling(diseaseCode, quarter, outcome) {
    if (state.busy) {
      toast('上一轮建模仍在进行，请先中止', 'warning');
      return;
    }
    if (!window.fetch || !window.EventSource) {
      // 浏览器能力检查（IE11 等老浏览器走 error 路径而不静默)
      if (elErrorText) elErrorText.textContent = '当前浏览器不支持 SSE / fetch';
      setGraphState('error');
      return;
    }

    state.busy = true;
    state.nodes = []; state.edges = []; state.effects = [];
    state.refutations = []; state.wedge = null;
    state.sampleSize = null; state.sampleVerdict = null;
    // T24：新一轮建模清掉已生成的当前草稿 id（让 confirm/reject 走新一轮）
    state.currentDraftId = null;
    state.currentModeling = {
      disease_code: diseaseCode,
      outcome: outcome,
      quarter: quarter,
    };
    // T23：新一轮建模清掉已选中节点与 tooltip
    state.selectedNode = null;
    state.tooltipPinned = false;
    if (state.tooltipEl) {
      state.tooltipEl.hidden = true;
      state.tooltipEl.setAttribute('data-pinned', 'false');
      state.tooltipEl.setAttribute('data-node-id', '');
    }
    if (elFooterSampleSize) elFooterSampleSize.textContent = '—';
    if (elFooterVerdict)    elFooterVerdict.textContent = '—';
    if (elFooterRefute)     elFooterRefute.textContent = '—';
    if (elFooterWedge)      elFooterWedge.textContent = '—';
    if (elSummary)          elSummary.innerHTML = '<li class="summary-empty">建模中…</li>';
    if (elProgressText)     elProgressText.textContent = '准备建模';
    elBtnBuild && (elBtnBuild.disabled = true);
    elBtnAbort && (elBtnAbort.disabled = false);
    elBtnCf && (elBtnCf.disabled = true);
    elBtnConfirm && (elBtnConfirm.disabled = true);
    elBtnReject && (elBtnReject.disabled = true);
    setGraphState('loading');

    var url = '/modeling/run';
    fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Quality-Role': state.role,
      },
      body: JSON.stringify({ disease_code: diseaseCode, quarter: quarter, outcome: outcome }),
    }).then(function (resp) {
      if (!resp.ok || !resp.body) {
        throw new Error('HTTP ' + resp.status);
      }
      var reader = resp.body.getReader();
      var decoder = new TextDecoder('utf-8');
      var buffer = '';

      function pump() {
        return reader.read().then(function (r) {
          if (r.done) return null;
          buffer += decoder.decode(r.value, { stream: true });
          var frames = buffer.split('\n\n');
          buffer = frames.pop() || '';
          for (var i = 0; i < frames.length; i++) {
            handleSseFrame(frames[i]);
          }
          return pump();
        });
      }
      return pump();
    }).catch(function (e) {
      if (elErrorText) elErrorText.textContent = '建模失败：' + (e && e.message ? e.message : 'unknown');
      setGraphState('error');
      toast('建模失败（请重试）', 'danger');
    }).then(function () {
      state.busy = false;
      elBtnBuild && (elBtnBuild.disabled = false);
      elBtnAbort && (elBtnAbort.disabled = true);
    });
  }

  /* 解析单帧：行首 event: ...  与 data: ... */
  function handleSseFrame(frameText) {
    var lines = frameText.split('\n');
    var eventName = '', dataStr = '';
    for (var i = 0; i < lines.length; i++) {
      var ln = lines[i];
      if (ln.indexOf('event:') === 0)  eventName = ln.substring(6).trim();
      else if (ln.indexOf('data:') === 0) dataStr += ln.substring(5).trim();
    }
    if (!eventName || !dataStr) return;
    var data;
    try { data = JSON.parse(dataStr); } catch (e) { return; }

    switch (eventName) {
      case 'ack':
        if (data.sample_size != null && elFooterSampleSize) elFooterSampleSize.textContent = data.sample_size;
        if (elProgressText) elProgressText.textContent = '已加载 ' + (data.sample_size || 0) + ' 行数据';
        toast('建模开始：' + data.disease_code + ' / ' + data.outcome, 'info');
        break;
      case 'sample-size':
        state.sampleSize = data.sample_size;
        state.sampleVerdict = data.verdict;
        if (elFooterVerdict) elFooterVerdict.textContent = data.verdict;
        break;
      case 'graph':
        state.nodes = data.nodes || [];
        state.edges = data.edges || [];
        break;
      case 'identify':
        if (elProgressText) elProgressText.textContent = '已识别因果 estimand';
        break;
      case 'effect':
        state.effects.push(data);
        if (elProgressText) elProgressText.textContent = '已到 ' + state.effects.length + ' / ' + (state.nodes.length || state.effects.length + 1) + ' 条效应';
        break;
      case 'refute':
        state.refutations.push(data);
        break;
      case 'refute-summary':
        if (elFooterRefute) elFooterRefute.textContent = data.verdict + ' (' + data.n_passed + '/' + data.n_total + ')';
        break;
      case 'baseline':
        state.wedge = data.wedge || (data.causal_wedge_strong != null
          ? { causal_wedge_strong: data.causal_wedge_strong } : null);
        if (elFooterWedge && data.wedge && data.wedge.ratio != null) {
          elFooterWedge.textContent = data.wedge.ratio.toFixed(2) + 'x ' +
            (data.wedge.causal_wedge_strong ? 'strong' : 'weak');
        }
        break;
      case 'done':
        if (elProgressText) elProgressText.textContent = '建模完成（' + (data.elapsed_ms || 0) + ' ms）';
        // T24：建模完成即生成当前草稿 id（dr_<sha1 hex> 16 hex chars），同 disease+outcome+quarter 多次建模得同一 id
        state.currentDraftId = _buildDraftId(
          state.currentModeling.disease_code,
          state.currentModeling.outcome,
          state.currentModeling.quarter
        );
        renderGraph();
        elBtnCf && (elBtnCf.disabled = false);
        // confirm/reject 全角色可见（spec §11 第 6 步）；后端以 X-Quality-Role=med_committee_reviewer 强校验
        elBtnConfirm && (elBtnConfirm.disabled = false);
        elBtnReject && (elBtnReject.disabled = false);
        refreshSummary();
        toast('建模完成 / 反驳总判定 ' + (state.refutations.length ? '已写入摘要' : '—'), 'success');
        break;
      case 'error':
        setGraphState('error');
        if (elErrorText) elErrorText.textContent = data.message || data.error_type || '建模失败';
        toast('建模失败：' + (data.message || data.error_type || 'unknown'), 'danger');
        break;
      default:
        break;
    }
  }

  function abortModeling() {
    // 当前 fetch 流式响应没有暴露 cancel；这里只重置 busy 与按钮
    state.busy = false;
    elBtnBuild && (elBtnBuild.disabled = false);
    elBtnAbort && (elBtnAbort.disabled = true);
    if (state.sseSource) { try { state.sseSource.close(); } catch (e) {} state.sseSource = null; }
    setGraphState('empty');
    toast('建模已中止', 'warning');
  }

  /* ============================================================
   * 摘要面板填充
   * ============================================================ */
  function refreshSummary() {
    if (!elSummary) return;
    var html = '';
    if (state.sampleSize != null) {
      html += row('样本量', state.sampleSize);
    }
    if (state.sampleVerdict) {
      html += row('样本守门', state.sampleVerdict);
    }
    if (state.effects.length) {
      var first = state.effects[0];
      var ate = first.ate == null ? '—' : first.ate.toFixed(3);
      html += row('主效应 ate', ate);
      html += row('p 值', first.p_value == null ? '—' : first.p_value.toFixed(3));
    }
    if (state.refutations.length) {
      html += row('反驳档数', state.refutations.length);
    }
    if (state.wedge && state.wedge.ratio != null) {
      html += row('楔子强度', state.wedge.ratio.toFixed(2) + 'x');
    }
    elSummary.innerHTML = html || '<li class="summary-empty">暂无结果</li>';
  }
  function row(k, v) {
    return '<li class="summary-row"><span class="summary-row__key">' + k +
      '</span><span class="summary-row__val">' + v + '</span></li>';
  }

  /* ============================================================
   * 反事实（T24 · 接入 POST /api/counterfactual/run）
   *
   * 设计约束（spec §11 第 4 步 + §6）：
   *   - 选节点 + 假设值 → POST body.scenarios=[{intervention_node, hypothetical_value}]
   *   - X-Quality-Role header 强制（缺失/非法走 403 envelope → toast danger）
   *   - 单场景异常 / 超时不阻塞 modal，标 cf-result 单条异常
   *   - 网络错误走 toast danger，不静默失败
   * ============================================================ */
  function openCounterfactual() {
    var modal = document.getElementById('cf-modal');
    var sel = document.getElementById('cf-node');
    if (!modal || !sel) return;
    sel.innerHTML = '';
    for (var i = 0; i < state.nodes.length; i++) {
      var n = state.nodes[i];
      var opt = document.createElement('option');
      opt.value = n.id; opt.textContent = (n.label || n.id) + (n.outcome ? '（结局）' : '');
      sel.appendChild(opt);
    }
    document.getElementById('cf-result').hidden = true;
    document.getElementById('cf-value').value = '';
    if (document.getElementById('cf-error')) {
      var elErr = document.getElementById('cf-error');
      elErr.hidden = true;
      elErr.textContent = '';
    }
    modal.hidden = false;
  }
  function closeCounterfactual() {
    var m = document.getElementById('cf-modal');
    if (m) m.hidden = true;
  }

  /* T24：从 disease+outcome+quarter 生成稳定的 draft_id（dr_<sha1 hex[:16]>）。
     同一组合多次建模得同一 id，便于面板 SSE 流去重 + 后端草稿查找。*/
  function _buildDraftId(diseaseCode, outcome, quarter) {
    var seed = String(diseaseCode || '') + '|' + String(outcome || '') + '|' +
      String(quarter || '') + '|panel-v1';
    // 简单稳定哈希（sha-1-like 替代；16 hex 字符够稳定）
    var h1 = 0xdeadbeef ^ seed.length;
    var h2 = 0x41c6ce57 ^ seed.length;
    for (var i = 0; i < seed.length; i++) {
      var ch = seed.charCodeAt(i);
      h1 = Math.imul(h1 ^ ch, 2654435761);
      h2 = Math.imul(h2 ^ ch, 1597334677);
    }
    h1 = (h1 ^ (h1 >>> 16)) >>> 0;
    h2 = (h2 ^ (h2 >>> 13)) >>> 0;
    var combined = ((h1 << 16) | h2) >>> 0;
    var hex = ('00000000' + combined.toString(16)).slice(-8);
    var hex2 = ('00000000' + (h1 >>> 0).toString(16)).slice(-8);
    return 'dr_' + hex + hex2;
  }

  /* T24：按钮 loading 态（禁用 + 文案占位 + spinner 类切换）*/
  function _setBtnLoading(btn, loading, loadingText, originalText) {
    if (!btn) return;
    btn.disabled = !!loading;
    if (loading) {
      btn.setAttribute('data-loading', 'true');
      if (loadingText) btn.textContent = loadingText;
    } else {
      btn.removeAttribute('data-loading');
      if (originalText) btn.textContent = originalText;
    }
  }

  /* T24：通用 POST JSON。返回 {ok, status, body, error}；业务层根据 status 走 toast 分支。*/
  function _postJson(url, body) {
    return fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Quality-Role': state.role,
      },
      body: JSON.stringify(body),
    }).then(function (resp) {
      return resp.text().then(function (txt) {
        var parsed = null;
        try { parsed = txt ? JSON.parse(txt) : null; } catch (e) { parsed = null; }
        return { ok: resp.ok, status: resp.status, body: parsed, raw: txt };
      });
    }).then(function (r) {
      return r;
    }).catch(function (e) {
      return { ok: false, status: 0, body: null, error: e && e.message ? e.message : 'network_error' };
    });
  }

  /* T24：反事实按钮 cf-run。打开 modal 后点运行 → POST /api/counterfactual/run + 显示结果 */
  function runCounterfactual() {
    var cfNode = document.getElementById('cf-node');
    var cfValue = document.getElementById('cf-value');
    var cfRun = document.getElementById('cf-run');
    var cfResult = document.getElementById('cf-result');
    var cfPredicted = document.getElementById('cf-predicted');
    var cfConfidence = document.getElementById('cf-confidence');
    var cfErr = document.getElementById('cf-error');
    if (!cfNode || !cfValue || !cfRun) return;

    var interventionNode = String(cfNode.value || '');
    var rawValue = cfValue.value;
    var hypValue;
    // 自动归一化类型：数字串 → number；否则保留为字符串
    if (/^-?\d+(?:\.\d+)?$/.test(rawValue)) {
      hypValue = parseFloat(rawValue);
    } else {
      hypValue = rawValue;
    }
    if (!interventionNode) {
      toast('请选择干预节点', 'warning');
      return;
    }
    if (hypValue === '' || hypValue == null) {
      toast('请输入假设值', 'warning');
      return;
    }
    if (!state.currentModeling || !state.currentModeling.disease_code || !state.currentModeling.outcome) {
      toast('请先完成建模再触发反事实', 'warning');
      return;
    }

    // 单一 scenario 入参（与后端 batch 接口兼容：scenarios[]）
    var scenario = {
      intervention_node: interventionNode,
      hypothetical_value: hypValue,
      label: '',
    };

    _setBtnLoading(cfRun, true, '反事实运行中…', '运行');
    if (cfErr) cfErr.hidden = true;
    if (cfResult) cfResult.hidden = true;

    _postJson('/counterfactual/run', {
      disease_code: state.currentModeling.disease_code,
      outcome: state.currentModeling.outcome,
      quarter: state.currentModeling.quarter,
      scenarios: [scenario],
    }).then(function (r) {
      _setBtnLoading(cfRun, false, null, '运行');
      if (!r.ok && r.status === 0) {
        // 网络错误
        if (cfErr) {
          cfErr.hidden = false;
          cfErr.textContent = '反事实请求失败：' + (r.error || 'network_error');
        }
        toast('反事实请求失败（请重试）', 'danger');
        return;
      }
      if (r.status === 403) {
        if (cfErr) { cfErr.hidden = false; cfErr.textContent = '角色鉴权失败（X-Quality-Role）'; }
        toast('反事实：当前角色无权触发', 'danger');
        return;
      }
      if (r.status === 400 || r.status === 422) {
        var msg = (r.body && (r.body.message || r.body.error_code)) || ('HTTP ' + r.status);
        if (cfErr) { cfErr.hidden = false; cfErr.textContent = '请求被拒：' + msg; }
        toast('反事实：' + msg, 'warning');
        return;
      }
      if (!r.body || !r.body.scenarios || !r.body.scenarios.length) {
        if (cfErr) { cfErr.hidden = false; cfErr.textContent = '反事实响应为空'; }
        toast('反事实：响应为空', 'warning');
        return;
      }
      var sc = r.body.scenarios[0];
      if (sc.timeout) {
        if (cfErr) { cfErr.hidden = false; cfErr.textContent = '反事实超时（5s）'; }
        toast('反事实超时', 'warning');
        return;
      }
      if (sc.predicted_outcome == null) {
        // 单条 errored
        var errOp = (sc.do_operations || []).find(function (op) {
          return typeof op === 'string' && op.indexOf('exception:') === 0;
        });
        var errLabel = errOp || '未知错误';
        if (cfErr) { cfErr.hidden = false; cfErr.textContent = '预测失败：' + errLabel; }
        toast('反事实：' + errLabel, 'danger');
        return;
      }
      // 成功路径
      if (cfPredicted) cfPredicted.textContent = Number(sc.predicted_outcome).toFixed(3);
      if (cfConfidence) cfConfidence.textContent = sc.confidence == null ? '—' : Number(sc.confidence).toFixed(2);
      if (cfResult) cfResult.hidden = false;
      toast('反事实完成（' + (r.body.n_completed || 1) + '/' + (r.body.n_scenarios || 1) + '）', 'success');
    });
  }

  /* T24：评审确认 / 退回 共享路径。action ∈ {confirm, reject}。
     POST /api/drafts/{draft_id}/review + body.action；
     上报失败后端会回退状态，面板把回退信息一并 toast 展示。*/
  function postReview(action) {
    if (!state.currentDraftId) {
      toast('请先完成建模（done 事件后才有当前草稿）', 'warning');
      return;
    }
    if (!state.currentModeling) {
      toast('建模上下文丢失，请重新建模', 'warning');
      return;
    }
    var draftId = state.currentDraftId;
    var btn = action === 'confirm' ? elBtnConfirm : elBtnReject;
    var labelText = action === 'confirm' ? '评审确认' : '评审退回';
    var loadingText = action === 'confirm' ? '评审确认中…' : '评审退回中…';

    _setBtnLoading(btn, true, loadingText, labelText);

    var body = {
      action: action,
      post_to_dipcase: action === 'confirm',  // reject 不上报
      note: '',
    };

    _postJson('/api/drafts/' + encodeURIComponent(draftId) + '/review', body).then(function (r) {
      _setBtnLoading(btn, false, null, labelText);
      if (!r.ok && r.status === 0) {
        toast(labelText + '网络异常：' + (r.error || 'network_error'), 'danger');
        return;
      }
      if (r.status === 403) {
        var roleMsg = (r.body && r.body.details && r.body.details.role) || state.role;
        toast(labelText + '被拒：当前角色「' + roleMsg + '」无' +
          (action === 'confirm' ? '确认' : '退回') + '权限', 'danger');
        return;
      }
      if (r.status === 404) {
        var msg404 = (r.body && r.body.message) || ('draft_id=' + draftId + ' 不存在');
        toast(labelText + '失败：' + msg404 + '（请重新建模）', 'warning');
        return;
      }
      if (r.status === 422) {
        var en = (r.body && r.body.error_code) || 'invalid';
        var msg = (r.body && r.body.message) || '';
        toast(labelText + '被拒：' + en + (msg ? '（' + msg + '）' : ''), 'warning');
        return;
      }
      // 成功路径
      var newState = (r.body && r.body.draft && r.body.draft.status) || (action === 'confirm' ? 'reviewed' : 'rejected');
      var postRollback = r.body && r.body.post_result && r.body.post_result.rolled_back;
      var toastMsg = (action === 'confirm' ? '评审确认' : '评审退回') + '成功（' + newState + '）';
      var toastKind = postRollback ? 'warning' : 'success';
      if (postRollback) {
        toastMsg += '·上报失败已自动回退';
      }
      toast(toastMsg, toastKind);
      // 评审后禁用按钮（终态后由 reviewer 决定下一步）
      if (action === 'confirm') {
        elBtnConfirm && (elBtnConfirm.disabled = true);
      } else {
        elBtnReject && (elBtnReject.disabled = true);
      }
      // 评审完成后清掉 currentDraftId，避免重复 confirm 状态机跳级
      state.currentDraftId = null;
      if (elSummary) {
        elSummary.innerHTML = '<li class="summary-empty">评审已结束，请重新建模以生成新草稿</li>';
      }
    });
  }

  function confirmDraft() { postReview('confirm'); }
  function rejectDraft() { postReview('reject'); }

  /* ============================================================
   * T23 · 节点交互层（点击 / 悬停 / tooltip / 三态 polish）
   *
   * 设计约束（spec §11 浏览器面板 + 视觉目标卡）：
   *   - 点击节点：选中并钉住 tooltip（ate / CI / p / method）
   *   - 选中节点的入边/出边保持原色，其它节点 + 边变暗
   *   - 悬停节点：临时浮层 tooltip，移开消失（不覆盖已钉住 tooltip）
   *   - 再次点击同一节点 / 点击空白：取消选中
   *   - 三态：empty（待选择）/ loading（建模中转动 + 蓝边）/ error（红边 + 重试）
   *   - 所有读 / 写操作不静默改数据（spec §6）；tooltip 只读展示 effect
   * ============================================================ */

  function findNodeById(nodeId) {
    for (var i = 0; i < state.nodes.length; i++) {
      if (state.nodes[i].id === nodeId) return state.nodes[i];
    }
    return null;
  }

  function findEffectForNode(nodeId) {
    for (var i = 0; i < state.effects.length; i++) {
      if (state.effects[i].node === nodeId) return state.effects[i];
    }
    return null;
  }

  function computeConnectedEdgeIndices(nodeId) {
    var idx = [];
    for (var i = 0; i < state.edges.length; i++) {
      var e = state.edges[i];
      if (e.source === nodeId || e.target === nodeId) idx.push(i);
    }
    return idx;
  }

  /* 在 #causal-graph 内创建/获取 tooltip 容器 */
  function ensureTooltipEl() {
    if (state.tooltipEl && state.tooltipEl.parentNode) return state.tooltipEl;
    var tip = document.createElement('div');
    tip.className = 'tooltip';
    tip.setAttribute('role', 'tooltip');
    tip.setAttribute('data-pinned', 'false');
    tip.hidden = true;
    if (elGraph) elGraph.appendChild(tip);
    state.tooltipEl = tip;
    return tip;
  }

  function renderTooltipContent(node, eff) {
    var lines = [];
    lines.push('<div class="tooltip__header">');
    lines.push('<span class="tooltip__node-id">' + escapeHtml(node.id) + '</span>');
    if (node.outcome) lines.push('<span class="tooltip__badge tooltip__badge--outcome">结局</span>');
    else if (node.treatment) lines.push('<span class="tooltip__badge tooltip__badge--treatment">干预</span>');
    else lines.push('<span class="tooltip__badge tooltip__badge--confounder">混杂</span>');
    lines.push('</div>');
    if (eff == null) {
      lines.push('<div class="tooltip__row"><span class="tooltip__key">效应</span><span class="tooltip__val">未估计</span></div>');
    } else {
      var ate = eff.ate == null ? '—' : eff.ate.toFixed(3);
      var ci = (eff.ci_lower != null && eff.ci_upper != null)
        ? '[' + eff.ci_lower.toFixed(3) + ', ' + eff.ci_upper.toFixed(3) + ']'
        : '—';
      var p = eff.p_value == null ? '—' : eff.p_value.toFixed(3);
      var m = eff.method_name || eff.method || '—';
      lines.push('<div class="tooltip__row"><span class="tooltip__key">ate</span><span class="tooltip__val">' + ate + '</span></div>');
      lines.push('<div class="tooltip__row"><span class="tooltip__key">95% CI</span><span class="tooltip__val">' + ci + '</span></div>');
      lines.push('<div class="tooltip__row"><span class="tooltip__key">p 值</span><span class="tooltip__val">' + p + '</span></div>');
      lines.push('<div class="tooltip__row"><span class="tooltip__key">方法</span><span class="tooltip__val">' + escapeHtml(m) + '</span></div>');
    }
    lines.push('<div class="tooltip__hint">再次点击节点或点击空白处取消</div>');
    return lines.join('');
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return ({ '&': '&', '<': '<', '>': '>', '"': '"', "'": '&#39;' })[c];
    });
  }

  function positionTooltip(tip, x, y) {
    if (!tip || !elGraph) return;
    var gw = elGraph.clientWidth || 600;
    var gh = elGraph.clientHeight || 420;
    var tw = tip.offsetWidth || 220;
    var th = tip.offsetHeight || 120;
    // 鼠标右侧 12px，超出右/下边界则翻转
    var nx = x + 12;
    var ny = y + 12;
    // 翻转后仍可能超界（如鼠标 x 远超 gw），做最终 clamp 保证 tooltip 不出框
    if (nx + tw > gw - 8) {
      var flippedX = x - tw - 12;
      // 翻转后仍超界则直接 clamp 到 gw - tw - 8
      nx = (flippedX >= 8) ? flippedX : Math.max(8, gw - tw - 8);
    }
    if (ny + th > gh - 8) {
      var flippedY = y - th - 12;
      ny = (flippedY >= 8) ? flippedY : Math.max(8, gh - th - 8);
    }
    // 最终 clamp 兜底：极端坐标下也保证落在 [8, gw-tw-8] 与 [8, gh-th-8]
    nx = Math.max(8, Math.min(nx, gw - tw - 8));
    ny = Math.max(8, Math.min(ny, gh - th - 8));
    tip.style.left = nx + 'px';
    tip.style.top = ny + 'px';
  }

  function showTooltip(nodeId, x, y, pinned) {
    var node = findNodeById(nodeId);
    if (!node) return;
    var tip = ensureTooltipEl();
    tip.innerHTML = renderTooltipContent(node, findEffectForNode(nodeId));
    tip.setAttribute('data-pinned', pinned ? 'true' : 'false');
    tip.setAttribute('data-node-id', nodeId);
    tip.hidden = false;
    positionTooltip(tip, x, y);
    state.tooltipPinned = !!pinned;
    if (pinned) state.selectedNode = nodeId;
  }

  function hideTooltip(force) {
    if (!state.tooltipEl) return;
    if (!force && state.tooltipPinned) return; // 钉住时不响应 hover-leave
    state.tooltipEl.hidden = true;
    state.tooltipEl.setAttribute('data-pinned', 'false');
    state.tooltipEl.setAttribute('data-node-id', '');
    state.tooltipPinned = false;
  }

  function pinTooltipForNode(nodeId) {
    // 在重渲染（done 事件）后还原已选中节点的 tooltip
    var tip = ensureTooltipEl();
    var node = findNodeById(nodeId);
    if (!node || !tip.parentNode) return;
    tip.innerHTML = renderTooltipContent(node, findEffectForNode(nodeId));
    tip.setAttribute('data-pinned', 'true');
    tip.setAttribute('data-node-id', nodeId);
    tip.hidden = false;
    // 居中显示（重渲染后无坐标参考）
    var gw = elGraph ? elGraph.clientWidth : 600;
    var gh = elGraph ? elGraph.clientHeight : 420;
    positionTooltip(tip, Math.floor(gw / 2 - 100), Math.floor(gh / 2 - 60));
  }

  function clearSelection() {
    state.selectedNode = null;
    state.tooltipPinned = false;
    if (state.tooltipEl) {
      state.tooltipEl.hidden = true;
      state.tooltipEl.setAttribute('data-pinned', 'false');
      state.tooltipEl.setAttribute('data-node-id', '');
    }
    if (state.nodes.length) renderGraph();
  }

  function onGraphClick(eventData) {
    if (!eventData || !eventData.points || !eventData.points.length) return;
    var pt = eventData.points[0];
    var nodeId = pt.data && pt.data.customdata ? pt.data.customdata : pt.text;
    if (!nodeId) return;
    // 第二次点击同一节点 → 取消选中
    if (state.selectedNode === nodeId && state.tooltipPinned) {
      clearSelection();
      toast('已取消选中节点：' + nodeId, 'info');
      return;
    }
    // Plotly 事件通常带 xpx / ypx（像素坐标），shim 也支持
    var x = (typeof pt.xpx === 'number') ? pt.xpx : (eventData.event && eventData.event.clientX ? eventData.event.clientX - (elGraph ? elGraph.getBoundingClientRect().left : 0) : 100);
    var y = (typeof pt.ypx === 'number') ? pt.ypx : (eventData.event && eventData.event.clientY ? eventData.event.clientY - (elGraph ? elGraph.getBoundingClientRect().top : 0) : 100);
    showTooltip(nodeId, x, y, true);
    renderGraph(); // 重渲染以触发高亮 + 边淡化
    toast('已选中节点：' + nodeId + '（再点一次取消）', 'info');
  }

  function onGraphHover(eventData) {
    if (!eventData || !eventData.points || !eventData.points.length) return;
    var pt = eventData.points[0];
    var nodeId = pt.data && pt.data.customdata ? pt.data.customdata : pt.text;
    if (!nodeId) return;
    if (state.tooltipPinned) return; // 已钉住时不覆盖
    var x = (typeof pt.xpx === 'number') ? pt.xpx : (eventData.event && eventData.event.clientX ? eventData.event.clientX - (elGraph ? elGraph.getBoundingClientRect().left : 0) : 100);
    var y = (typeof pt.ypx === 'number') ? pt.ypx : (eventData.event && eventData.event.clientY ? eventData.event.clientY - (elGraph ? elGraph.getBoundingClientRect().top : 0) : 100);
    showTooltip(nodeId, x, y, false);
  }

  function onGraphUnhover() {
    if (!state.tooltipPinned) hideTooltip(false);
  }

  function attachGraphInteractions(gd) {
    if (!gd) return;
    // Plotly 2.x 标准事件：plotly_click / plotly_hover / plotly_unhover
    try { gd.on('plotly_click', onGraphClick); } catch (e) {}
    try { gd.on('plotly_hover', onGraphHover); } catch (e) {}
    try { gd.on('plotly_unhover', onGraphUnhover); } catch (e) {}
  }

  /* 测试钩子：直接走 onGraphClick 同款逻辑；不依赖真实 Plotly 事件 */
  function simulateNodeClick(nodeId) {
    onGraphClick({ points: [{ text: nodeId, data: { customdata: nodeId }, xpx: 200, ypx: 200 }], event: { clientX: 200, clientY: 200 } });
  }
  function simulateNodeHover(nodeId, x, y) {
    var px = x || 200, py = y || 200;
    onGraphHover({ points: [{ text: nodeId, data: { customdata: nodeId }, xpx: px, ypx: py }], event: { clientX: px, clientY: py } });
  }
  function simulateNodeUnhover() {
    onGraphUnhover();
  }

  /* ============================================================
   * 角色切换（点徽章在两种角色间切）
   * ============================================================ */
  function toggleRole() {
    var next = state.role === 'quality_officer' ? 'med_committee_reviewer' : 'quality_officer';
    setRole(next);
    toast('切换到：' + (next === 'med_committee_reviewer' ? '医务科评审' : '质控员'), 'info');
  }

  /* ============================================================
   * 事件绑定
   * ============================================================ */
  function attach() {
    if (elForm) {
      elForm.addEventListener('submit', function (ev) {
        ev.preventDefault();
        var fd = new FormData(elForm);
        startModeling(
          String(fd.get('disease_code') || ''),
          String(fd.get('quarter') || ''),
          String(fd.get('outcome') || '')
        );
      });
    }
    if (elBtnAbort)  elBtnAbort.addEventListener('click', abortModeling);
    if (elBtnCf)     elBtnCf.addEventListener('click', openCounterfactual);
    var cfClose = document.getElementById('cf-close');
    if (cfClose) cfClose.addEventListener('click', closeCounterfactual);
    var cfRun = document.getElementById('cf-run');
    if (cfRun) cfRun.addEventListener('click', runCounterfactual);
    if (elBtnConfirm) elBtnConfirm.addEventListener('click', confirmDraft);
    if (elBtnReject) elBtnReject.addEventListener('click', rejectDraft);
    if (elRoleBadge) elRoleBadge.addEventListener('click', toggleRole);

    // T23：点击因果图空白区域取消选中（点击节点由 Plotly plotly_click 接管）
    if (elGraph) {
      elGraph.addEventListener('click', function (ev) {
        // 仅当点击的是图本体（不是 tooltip）时取消；Plotly 自己会触发 plotly_click 处理节点
        if (ev.target === elGraph || ev.target.classList.contains('plotly-stub')) {
          if (state.selectedNode) {
            clearSelection();
            toast('已取消选中节点', 'info');
          }
        }
      });
    }

    // 全局 ESC 关闭 modal
    document.addEventListener('keydown', function (ev) {
      if (ev.key === 'Escape') {
        var m = document.getElementById('cf-modal');
        if (m && !m.hidden) m.hidden = true;
      }
    });
  }

  /* ============================================================
   * 初始化
   * ============================================================ */
  function init() {
    setRole(state.role);
    attach();
    setGraphState('empty');
    // 暴露只读 state + 测试钩子（spec §11 可测性 + T23 节点交互测试）
    global.__cfPanel = {
      state: state,
      startModeling: startModeling,
      setRole: setRole,
      setGraphState: setGraphState,
      simulateNodeClick: simulateNodeClick,
      simulateNodeHover: simulateNodeHover,
      simulateNodeUnhover: simulateNodeUnhover,
      clearSelection: clearSelection,
      showTooltip: showTooltip,
      hideTooltip: hideTooltip,
      // T24：评审 / 反事实 API 调用
      runCounterfactual: runCounterfactual,
      confirmDraft: confirmDraft,
      rejectDraft: rejectDraft,
      _buildDraftId: _buildDraftId,
      _postJson: _postJson,
      _setBtnLoading: _setBtnLoading,
      // 测试专用：在未走 SSE 流程时塞入节点 + 效应，便于 e2e 验证
      __seedDemo: function (nodes, edges, effects) {
        state.nodes = nodes || [];
        state.edges = edges || [];
        state.effects = effects || [];
        renderGraph();
      },
      // 测试专用：直接给面板装一个草稿 id 与建模上下文，绕开 SSE 'done' 流
      __seedDraft: function (draftId, modeling) {
        state.currentDraftId = draftId || null;
        if (modeling) state.currentModeling = modeling;
        if (draftId) {
          elBtnConfirm && (elBtnConfirm.disabled = false);
          elBtnReject && (elBtnReject.disabled = false);
        }
        return { draftId: state.currentDraftId, modeling: state.currentModeling };
      },
    };
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})(window);
