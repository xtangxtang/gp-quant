import argparse
import glob
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request


DEFAULT_SCAN_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "results" / "multitimeframe_resonance" / "live_market_scan"


DASHBOARD_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>gp-quant Resonance Dashboard</title>
  <style>
    :root {
      --paper: #f6f0e7;
      --ink: #172121;
      --muted: #58636b;
      --card: rgba(255,255,255,0.74);
      --line: rgba(23,33,33,0.12);
      --accent: #c84c2f;
      --accent-strong: #9f3219;
      --good: #1f7a4c;
      --warn: #ad7a14;
      --bad: #a63f3f;
      --shadow: 0 18px 60px rgba(41, 34, 24, 0.12);
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(245, 201, 168, 0.85), transparent 28%),
        radial-gradient(circle at right 20%, rgba(200, 76, 47, 0.12), transparent 22%),
        linear-gradient(180deg, #f8efe3 0%, #efe3d2 48%, #f5efe7 100%);
      font-family: Georgia, "Noto Serif SC", "Source Han Serif SC", serif;
      min-height: 100vh;
    }

    .shell {
      max-width: 1520px;
      margin: 0 auto;
      padding: 28px 20px 48px;
    }

    .hero {
      display: grid;
      grid-template-columns: 1.25fr 0.95fr;
      gap: 18px;
      margin-bottom: 18px;
    }

    .hero-card,
    .panel {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(12px);
    }

    .hero-card {
      padding: 24px;
      position: relative;
      overflow: hidden;
    }

    .hero-card::after {
      content: "";
      position: absolute;
      right: -24px;
      top: -24px;
      width: 180px;
      height: 180px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(200, 76, 47, 0.18), transparent 68%);
    }

    .eyebrow {
      font-size: 12px;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--accent-strong);
      margin-bottom: 10px;
    }

    h1 {
      margin: 0 0 10px;
      font-size: clamp(32px, 5vw, 56px);
      line-height: 0.96;
      font-weight: 700;
    }

    .hero-copy {
      max-width: 720px;
      color: var(--muted);
      font-size: 16px;
      line-height: 1.6;
      margin-bottom: 20px;
    }

    .summary-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }

    .metric {
      padding: 14px 14px 12px;
      background: rgba(255,255,255,0.58);
      border: 1px solid rgba(23,33,33,0.08);
      border-radius: 16px;
    }

    .metric-label {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }

    .metric-value {
      font-size: 26px;
      font-weight: 700;
      line-height: 1;
    }

    .hero-side {
      display: grid;
      gap: 18px;
    }

    .panel {
      padding: 18px;
    }

    .panel-title {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 14px;
    }

    .panel-title h2,
    .panel-title h3 {
      margin: 0;
      font-size: 18px;
    }

    .stamp {
      font-size: 12px;
      color: var(--accent-strong);
      background: rgba(245, 201, 168, 0.45);
      padding: 6px 10px;
      border-radius: 999px;
      white-space: nowrap;
    }

    .toolbar {
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }

    label {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
    }

    input, select {
      width: 100%;
      border: 1px solid rgba(23,33,33,0.14);
      background: rgba(255,255,255,0.84);
      color: var(--ink);
      border-radius: 12px;
      padding: 10px 12px;
      font: inherit;
    }

    .toolbar-actions {
      display: flex;
      align-items: end;
      gap: 10px;
    }

    .preset-card {
      margin-top: 12px;
      padding: 14px 16px;
      border-radius: 16px;
      border: 1px solid rgba(23,33,33,0.10);
      background: rgba(255,255,255,0.52);
    }

    .preset-title {
      font-size: 12px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--accent-strong);
      margin-bottom: 8px;
    }

    .preset-copy {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
      margin-bottom: 10px;
    }

    .preset-tags {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .preset-tag {
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 12px;
      color: var(--ink);
      background: rgba(245, 201, 168, 0.42);
      border: 1px solid rgba(200, 76, 47, 0.16);
    }

    button {
      border: 0;
      border-radius: 999px;
      padding: 11px 16px;
      cursor: pointer;
      font: inherit;
      transition: transform 180ms ease, background 180ms ease;
    }

    button:hover { transform: translateY(-1px); }
    .btn-primary { background: var(--accent); color: #fff8f1; }
    .btn-secondary {
      background: rgba(255,255,255,0.74);
      color: var(--ink);
      border: 1px solid rgba(23,33,33,0.12);
    }

    .content-grid {
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 18px;
    }

    .table-wrap {
      overflow: auto;
      border-radius: 18px;
      border: 1px solid rgba(23,33,33,0.08);
      background: rgba(255,255,255,0.45);
    }

    table {
      width: 100%;
      border-collapse: collapse;
      min-width: 1160px;
      font-size: 13px;
    }

    th, td {
      padding: 10px 12px;
      border-bottom: 1px solid rgba(23,33,33,0.08);
      text-align: left;
      white-space: nowrap;
    }

    thead th {
      position: sticky;
      top: 0;
      background: #f3eadf;
      z-index: 1;
    }

    tbody tr:hover { background: rgba(245, 201, 168, 0.18); }

    .tabs {
      display: flex;
      gap: 8px;
      margin-bottom: 12px;
      flex-wrap: wrap;
    }

    .tab {
      padding: 9px 14px;
      border-radius: 999px;
      border: 1px solid rgba(23,33,33,0.1);
      background: rgba(255,255,255,0.72);
      cursor: pointer;
      color: var(--muted);
    }

    .tab.active {
      background: var(--ink);
      color: #f8efe3;
      border-color: var(--ink);
    }

    .list {
      display: grid;
      gap: 10px;
    }

    .list-item {
      padding: 12px 14px;
      border-radius: 16px;
      background: rgba(255,255,255,0.56);
      border: 1px solid rgba(23,33,33,0.08);
    }

    .kicker {
      display: inline-block;
      padding: 4px 8px;
      border-radius: 999px;
      font-size: 11px;
      margin-bottom: 8px;
    }

    .kicker.good { background: rgba(31,122,76,0.12); color: var(--good); }
    .kicker.warn { background: rgba(173,122,20,0.14); color: var(--warn); }
    .kicker.bad { background: rgba(166,63,63,0.12); color: var(--bad); }

    .chart-wrap {
      border-radius: 18px;
      border: 1px solid rgba(23,33,33,0.08);
      background: rgba(255,255,255,0.42);
      padding: 12px;
    }

    svg { width: 100%; height: auto; display: block; }

    .chart-caption {
      color: var(--muted);
      font-size: 12px;
      margin-top: 6px;
    }

    .empty {
      padding: 22px;
      border-radius: 16px;
      background: rgba(255,255,255,0.5);
      border: 1px dashed rgba(23,33,33,0.18);
      color: var(--muted);
      text-align: center;
    }

    @media (max-width: 1180px) {
      .hero,
      .content-grid { grid-template-columns: 1fr; }
      .toolbar { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .summary-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }

    @media (max-width: 720px) {
      .shell { padding: 18px 14px 32px; }
      .toolbar { grid-template-columns: 1fr; }
      .summary-grid { grid-template-columns: 1fr 1fr; }
      .toolbar-actions { align-items: stretch; }
      .toolbar-actions button { width: 100%; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <article class="hero-card">
        <div class="eyebrow">Multitimeframe Resonance</div>
        <h1>Live Scan<br/>Control Room</h1>
        <div class="hero-copy">直接查看实时扫描结果、按行业 / 市场 / 地域和流动性二次过滤、检查当天入选组合，并联动查看前瞻回测净值曲线和逐笔交易结果。当前推荐参数基线为成交额 ≥ 50 万、换手率 ≥ 1%、共振支撑数 ≥ 2、最多持有 10 只且单行业不超过 2 只。</div>
        <div class="summary-grid">
          <div class="metric"><div class="metric-label">扫描日期</div><div class="metric-value" id="metric-scan-date">-</div></div>
          <div class="metric"><div class="metric-label">候选数</div><div class="metric-value" id="metric-candidates">0</div></div>
          <div class="metric"><div class="metric-label">入选数</div><div class="metric-value" id="metric-selected">0</div></div>
          <div class="metric"><div class="metric-label">组合净值</div><div class="metric-value" id="metric-nav">1.00</div></div>
        </div>
      </article>

      <div class="hero-side">
        <section class="panel">
          <div class="panel-title">
            <h2>过滤器</h2>
            <div class="stamp" id="status-stamp">Ready</div>
          </div>
          <div class="toolbar">
            <div>
              <label for="scan-date">扫描日期</label>
              <select id="scan-date"></select>
            </div>
            <div>
              <label for="name-query">名称 / 代码</label>
              <input id="name-query" placeholder="如 600000 或 银行" />
            </div>
            <div>
              <label for="industry-filter">行业</label>
              <select id="industry-filter"><option value="">全部行业</option></select>
            </div>
            <div>
              <label for="market-filter">市场</label>
              <select id="market-filter"><option value="">全部市场</option></select>
            </div>
            <div>
              <label for="area-filter">地域</label>
              <select id="area-filter"><option value="">全部地域</option></select>
            </div>
            <div>
              <label for="include-st">ST 股票</label>
              <select id="include-st">
                <option value="0">排除</option>
                <option value="1">包含</option>
              </select>
            </div>
          </div>
          <div class="toolbar">
            <div>
              <label for="min-score">最低共振分</label>
              <input id="min-score" type="number" step="0.01" value="0" />
            </div>
            <div>
              <label for="min-amount">最低成交额</label>
              <input id="min-amount" type="number" step="100000" value="500000" />
            </div>
            <div>
              <label for="min-turnover">最低换手率</label>
              <input id="min-turnover" type="number" step="0.01" value="1.0" />
            </div>
            <div>
              <label for="support-count">最低共振支撑数</label>
              <select id="support-count">
                <option value="0">不限</option>
                <option value="2" selected>2</option>
                <option value="3">3</option>
              </select>
            </div>
            <div>
              <label for="table-limit">显示行数</label>
              <input id="table-limit" type="number" min="10" step="10" value="100" />
            </div>
            <div class="toolbar-actions">
              <button class="btn-primary" id="refresh-btn">刷新</button>
              <button class="btn-secondary" id="preset-btn">恢复推荐参数</button>
              <button class="btn-secondary" id="reset-btn">清空过滤</button>
            </div>
          </div>
          <div class="preset-card">
            <div class="preset-title">Recommended Preset</div>
            <div class="preset-copy">当前推荐基线来自最近短窗和 3 个月前瞻回测的共同结果，优先兼顾候选数量、行业分散和组合回撤控制。扫描脚本默认值也已同步到这组参数。</div>
            <div class="preset-tags">
              <span class="preset-tag">top_n = 30</span>
              <span class="preset-tag">min_amount = 500000</span>
              <span class="preset-tag">min_turnover = 1.0</span>
              <span class="preset-tag">support_count ≥ 2</span>
              <span class="preset-tag">hold_days = 5</span>
              <span class="preset-tag">max_positions = 10</span>
              <span class="preset-tag">per_industry = 2</span>
            </div>
          </div>
        </section>

        <section class="panel">
          <div class="panel-title">
            <h3>前瞻回测</h3>
            <div class="stamp" id="backtest-range">-</div>
          </div>
          <div class="list" id="backtest-summary"></div>
          <div class="chart-wrap" style="margin-top:12px;">
            <svg id="nav-chart" viewBox="0 0 520 220" preserveAspectRatio="none"></svg>
            <div class="chart-caption" id="nav-caption">暂无净值曲线</div>
          </div>
        </section>
      </div>
    </section>

    <section class="content-grid">
      <section class="panel">
        <div class="panel-title">
          <h2>扫描结果</h2>
          <div class="stamp" id="table-stamp">selected</div>
        </div>
        <div class="tabs">
          <button class="tab active" data-view="selected">入选组合</button>
          <button class="tab" data-view="candidates">候选池</button>
          <button class="tab" data-view="market">全市场快照</button>
        </div>
        <div class="table-wrap">
          <table>
            <thead id="table-head"></thead>
            <tbody id="table-body"></tbody>
          </table>
        </div>
        <div class="empty" id="table-empty" hidden>当前过滤条件下没有记录。</div>
      </section>

      <section class="panel">
        <div class="panel-title">
          <h2>交易明细</h2>
          <div class="stamp" id="trade-count">0 trades</div>
        </div>
        <div class="table-wrap">
          <table>
            <thead id="trade-head"></thead>
            <tbody id="trade-body"></tbody>
          </table>
        </div>
        <div class="empty" id="trade-empty" hidden>当前扫描日期没有回测交易明细。</div>
      </section>
    </section>

    <section class="panel" style="margin-top:18px;">
      <div class="panel-title">
        <h2>行业分组统计</h2>
        <div class="stamp" id="industry-stamp">0 industries</div>
      </div>
      <div class="table-wrap">
        <table>
          <thead id="industry-head"></thead>
          <tbody id="industry-body"></tbody>
        </table>
      </div>
      <div class="chart-caption">行业净值贡献按每笔交易等权占用一个持仓槽位近似计算，即 `sum(return_pct / max_positions)`。</div>
      <div class="empty" id="industry-empty" hidden>当前过滤条件下没有行业统计结果。</div>
    </section>
  </div>

  <script>
    const recommendedPreset = {
      nameQuery: '',
      industry: '',
      market: '',
      area: '',
      minScore: '0',
      minAmount: '500000',
      minTurnover: '1.0',
      supportCount: '2',
      includeSt: '0',
      tableLimit: '100'
    };

    const state = { view: 'selected', scanDate: '', dates: [], options: { industries: [], markets: [], areas: [] } };

    const tableColumns = {
      selected: ['selected_rank', 'symbol', 'name', 'industry', 'market', 'resonance_score', 'support_count', 'amount', 'turnover_rate', 'daily_score', 'weekly_score', 'monthly_score', 'index_regime'],
      candidates: ['symbol', 'name', 'industry', 'market', 'resonance_score', 'support_count', 'amount', 'turnover_rate', 'daily_score', 'weekly_score', 'monthly_score', 'index_regime'],
      market: ['symbol', 'name', 'industry', 'market', 'area', 'resonance_state', 'support_count', 'resonance_score', 'amount', 'turnover_rate', 'daily_state', 'weekly_state', 'monthly_state', 'index_regime'],
      trades: ['scan_date', 'symbol', 'name', 'industry', 'entry_date', 'exit_date', 'entry_price', 'exit_price', 'return_pct', 'max_runup_pct'],
      industryStats: ['industry', 'universe_count', 'candidate_count', 'selected_count', 'avg_resonance_score', 'trade_count', 'avg_trade_return_pct', 'nav_contribution_pct']
    };

    function fmtValue(value) {
      if (value === null || value === undefined || value === '') return '-';
      if (typeof value === 'number') {
        if (!Number.isFinite(value)) return '-';
        if (Math.abs(value) >= 1000) return value.toLocaleString('zh-CN', { maximumFractionDigits: 2 });
        return value.toLocaleString('zh-CN', { maximumFractionDigits: 4 });
      }
      return String(value);
    }

    function setStatus(text) {
      document.getElementById('status-stamp').textContent = text;
    }

    function applyRecommendedPreset() {
      document.getElementById('name-query').value = recommendedPreset.nameQuery;
      document.getElementById('industry-filter').value = recommendedPreset.industry;
      document.getElementById('market-filter').value = recommendedPreset.market;
      document.getElementById('area-filter').value = recommendedPreset.area;
      document.getElementById('min-score').value = recommendedPreset.minScore;
      document.getElementById('min-amount').value = recommendedPreset.minAmount;
      document.getElementById('min-turnover').value = recommendedPreset.minTurnover;
      document.getElementById('support-count').value = recommendedPreset.supportCount;
      document.getElementById('include-st').value = recommendedPreset.includeSt;
      document.getElementById('table-limit').value = recommendedPreset.tableLimit;
    }

    function clearFilters() {
      document.getElementById('name-query').value = '';
      document.getElementById('industry-filter').value = '';
      document.getElementById('market-filter').value = '';
      document.getElementById('area-filter').value = '';
      document.getElementById('min-score').value = '0';
      document.getElementById('min-amount').value = '0';
      document.getElementById('min-turnover').value = '0';
      document.getElementById('support-count').value = '0';
      document.getElementById('include-st').value = '0';
      document.getElementById('table-limit').value = '100';
    }

    function filtersToQuery() {
      const params = new URLSearchParams();
      params.set('scan_date', state.scanDate);
      params.set('view', state.view);
      params.set('name_query', document.getElementById('name-query').value.trim());
      params.set('industry', document.getElementById('industry-filter').value || '');
      params.set('market', document.getElementById('market-filter').value || '');
      params.set('area', document.getElementById('area-filter').value || '');
      params.set('min_resonance_score', document.getElementById('min-score').value || '0');
      params.set('min_amount', document.getElementById('min-amount').value || '0');
      params.set('min_turnover', document.getElementById('min-turnover').value || '0');
      params.set('support_count_min', document.getElementById('support-count').value || '0');
      params.set('include_st', document.getElementById('include-st').value || '0');
      params.set('limit', document.getElementById('table-limit').value || '100');
      return params;
    }

    function renderTable(headId, bodyId, emptyId, columns, rows) {
      const head = document.getElementById(headId);
      const body = document.getElementById(bodyId);
      const empty = document.getElementById(emptyId);
      head.innerHTML = '';
      body.innerHTML = '';
      if (!rows || rows.length === 0) {
        empty.hidden = false;
        return;
      }
      empty.hidden = true;

      const trHead = document.createElement('tr');
      columns.forEach((column) => {
        const th = document.createElement('th');
        th.textContent = column;
        trHead.appendChild(th);
      });
      head.appendChild(trHead);

      rows.forEach((row) => {
        const tr = document.createElement('tr');
        columns.forEach((column) => {
          const td = document.createElement('td');
          td.textContent = fmtValue(row[column]);
          tr.appendChild(td);
        });
        body.appendChild(tr);
      });
    }

    function renderBacktestSummary(summary) {
      const host = document.getElementById('backtest-summary');
      host.innerHTML = '';
      const items = [
        ['回测区间', summary.backtest_start_date && summary.backtest_end_date ? `${summary.backtest_start_date} -> ${summary.backtest_end_date}` : '-'],
        ['持有天数', summary.hold_days ?? '-'],
        ['最大持仓', summary.max_positions ?? '-'],
        ['行业上限', summary.max_positions_per_industry ?? 0],
        ['交易笔数', summary.n_trades ?? 0],
        ['胜率', summary.win_rate == null ? '-' : `${(summary.win_rate * 100).toFixed(2)}%`],
        ['累计收益', summary.total_return_pct == null ? '-' : `${summary.total_return_pct.toFixed(2)}%`],
        ['最大回撤', summary.max_drawdown_pct == null ? '-' : `${summary.max_drawdown_pct.toFixed(2)}%`],
        ['最终净值', summary.final_nav == null ? '-' : summary.final_nav.toFixed(4)],
      ];
      items.forEach(([label, value], idx) => {
        const div = document.createElement('div');
        div.className = 'list-item';
        const tone = idx < 3 ? 'good' : (idx < 6 ? 'warn' : 'bad');
        div.innerHTML = `<div class="kicker ${tone}">${label}</div><div>${fmtValue(value)}</div>`;
        host.appendChild(div);
      });
      document.getElementById('backtest-range').textContent = summary.backtest_start_date && summary.backtest_end_date ? `${summary.backtest_start_date} - ${summary.backtest_end_date}` : 'No backtest';
      document.getElementById('metric-nav').textContent = summary.final_nav == null ? '1.00' : Number(summary.final_nav).toFixed(2);
    }

    function renderNavChart(dailyRows) {
      const svg = document.getElementById('nav-chart');
      const caption = document.getElementById('nav-caption');
      svg.innerHTML = '';
      if (!dailyRows || dailyRows.length === 0) {
        caption.textContent = '暂无净值曲线';
        return;
      }

      const validRows = dailyRows.filter((row) => Number.isFinite(Number(row.nav)));
      if (validRows.length === 0) {
        caption.textContent = '暂无净值曲线';
        return;
      }

      const width = 520;
      const height = 220;
      const padL = 42;
      const padR = 18;
      const padT = 16;
      const padB = 34;
      const navValues = validRows.map((row) => Number(row.nav));
      let minNav = Math.min(...navValues);
      let maxNav = Math.max(...navValues);
      if (minNav === maxNav) {
        minNav -= 0.02;
        maxNav += 0.02;
      }

      const xAt = (idx) => padL + (idx / Math.max(1, validRows.length - 1)) * (width - padL - padR);
      const yAt = (nav) => height - padB - ((nav - minNav) / (maxNav - minNav)) * (height - padT - padB);

      const bg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      bg.setAttribute('x', '0');
      bg.setAttribute('y', '0');
      bg.setAttribute('width', width);
      bg.setAttribute('height', height);
      bg.setAttribute('rx', '18');
      bg.setAttribute('fill', 'rgba(255,255,255,0.35)');
      svg.appendChild(bg);

      [minNav, (minNav + maxNav) / 2, maxNav].forEach((value) => {
        const y = yAt(value);
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', padL);
        line.setAttribute('x2', width - padR);
        line.setAttribute('y1', y);
        line.setAttribute('y2', y);
        line.setAttribute('stroke', 'rgba(23,33,33,0.10)');
        line.setAttribute('stroke-dasharray', '3 5');
        svg.appendChild(line);

        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', 6);
        text.setAttribute('y', y + 4);
        text.setAttribute('font-size', '11');
        text.setAttribute('fill', '#58636b');
        text.textContent = value.toFixed(3);
        svg.appendChild(text);
      });

      let points = '';
      validRows.forEach((row, idx) => {
        points += `${xAt(idx)},${yAt(Number(row.nav))} `;
      });

      const path = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
      path.setAttribute('fill', 'none');
      path.setAttribute('stroke', '#c84c2f');
      path.setAttribute('stroke-width', '3');
      path.setAttribute('stroke-linecap', 'round');
      path.setAttribute('stroke-linejoin', 'round');
      path.setAttribute('points', points.trim());
      svg.appendChild(path);

      const first = validRows[0];
      const last = validRows[validRows.length - 1];
      [
        [first, 0, '#172121'],
        [last, validRows.length - 1, '#c84c2f']
      ].forEach(([row, idx, color]) => {
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', xAt(idx));
        circle.setAttribute('cy', yAt(Number(row.nav)));
        circle.setAttribute('r', '4');
        circle.setAttribute('fill', color);
        svg.appendChild(circle);
      });

      const tickIndexes = [0, Math.floor((validRows.length - 1) / 2), validRows.length - 1];
      [...new Set(tickIndexes)].forEach((idx) => {
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', xAt(idx));
        text.setAttribute('y', height - 10);
        text.setAttribute('font-size', '11');
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('fill', '#58636b');
        text.textContent = String(validRows[idx].scan_date);
        svg.appendChild(text);
      });

      caption.textContent = `${String(first.scan_date)} -> ${String(last.scan_date)}，最终净值 ${Number(last.nav).toFixed(4)}`;
    }

    function fillSelect(selectId, items, emptyLabel) {
      const select = document.getElementById(selectId);
      const currentValue = select.value;
      select.innerHTML = '';
      const base = document.createElement('option');
      base.value = '';
      base.textContent = emptyLabel;
      select.appendChild(base);
      (items || []).forEach((item) => {
        const option = document.createElement('option');
        option.value = item;
        option.textContent = item;
        select.appendChild(option);
      });
      if ([...select.options].some((node) => node.value === currentValue)) {
        select.value = currentValue;
      }
    }

    async function loadDates() {
      const resp = await fetch('/api/scan_dates');
      const payload = await resp.json();
      state.dates = payload.scan_dates || [];
      const select = document.getElementById('scan-date');
      select.innerHTML = '';
      state.dates.forEach((date) => {
        const option = document.createElement('option');
        option.value = date;
        option.textContent = date;
        select.appendChild(option);
      });
      state.scanDate = payload.latest_scan_date || state.dates[0] || '';
      select.value = state.scanDate;
    }

    async function loadFilterOptions() {
      const resp = await fetch(`/api/filter_options?scan_date=${encodeURIComponent(state.scanDate)}`);
      const payload = await resp.json();
      state.options = payload || { industries: [], markets: [], areas: [] };
      fillSelect('industry-filter', state.options.industries || [], '全部行业');
      fillSelect('market-filter', state.options.markets || [], '全部市场');
      fillSelect('area-filter', state.options.areas || [], '全部地域');
    }

    async function loadSummary() {
      const params = filtersToQuery();
      const resp = await fetch(`/api/backtest?${params.toString()}`);
      const payload = await resp.json();
      renderBacktestSummary(payload.summary || {});
      renderNavChart(payload.daily || []);
      renderTable('trade-head', 'trade-body', 'trade-empty', tableColumns.trades, payload.trades || []);
      document.getElementById('trade-count').textContent = `${(payload.trades || []).length} trades`;
    }

    async function loadIndustryStats() {
      const params = filtersToQuery();
      const resp = await fetch(`/api/industry_stats?${params.toString()}`);
      const payload = await resp.json();
      renderTable('industry-head', 'industry-body', 'industry-empty', tableColumns.industryStats, payload.rows || []);
      document.getElementById('industry-stamp').textContent = `${(payload.rows || []).length} industries`;
    }

    async function loadTable() {
      const params = filtersToQuery();
      setStatus('Loading');
      const resp = await fetch(`/api/scan?${params.toString()}`);
      const payload = await resp.json();
      renderTable('table-head', 'table-body', 'table-empty', tableColumns[state.view], payload.rows || []);
      const summary = payload.summary || {};
      document.getElementById('metric-scan-date').textContent = state.scanDate || '-';
      document.getElementById('metric-candidates').textContent = fmtValue(summary.n_resonance_candidates ?? 0);
      document.getElementById('metric-selected').textContent = fmtValue(summary.n_selected ?? 0);
      document.getElementById('table-stamp').textContent = `${state.view} / ${payload.row_count || 0}`;
      setStatus('Synced');
    }

    async function refreshAll() {
      state.scanDate = document.getElementById('scan-date').value;
      await loadFilterOptions();
      await Promise.all([loadTable(), loadSummary(), loadIndustryStats()]);
    }

    function bindEvents() {
      document.querySelectorAll('.tab').forEach((button) => {
        button.addEventListener('click', async () => {
          document.querySelectorAll('.tab').forEach((node) => node.classList.remove('active'));
          button.classList.add('active');
          state.view = button.dataset.view;
          await loadTable();
        });
      });

      document.getElementById('refresh-btn').addEventListener('click', refreshAll);
      document.getElementById('preset-btn').addEventListener('click', async () => {
        applyRecommendedPreset();
        await refreshAll();
      });
      document.getElementById('scan-date').addEventListener('change', refreshAll);
      document.getElementById('reset-btn').addEventListener('click', async () => {
        clearFilters();
        await refreshAll();
      });
    }

    (async function init() {
      await loadDates();
      bindEvents();
      await refreshAll();
    })();
  </script>
</body>
</html>
"""


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _scan_dates(output_dir: Path) -> list[str]:
    pattern = str(output_dir / "market_scan_snapshot_*.csv")
    dates: list[str] = []
    for path in glob.glob(pattern):
        stem = Path(path).stem
        dates.append(stem.rsplit("_", 1)[-1])
    return sorted(set(dates), reverse=True)


def _resolve_scan_date(output_dir: Path, requested_scan_date: str) -> str:
    if requested_scan_date:
        return str(requested_scan_date)
    dates = _scan_dates(output_dir)
    if not dates:
        raise FileNotFoundError(f"No scan outputs found under {output_dir}")
    return dates[0]


def _find_existing(output_dir: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(output_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _load_bundle(output_dir: Path, scan_date: str) -> dict[str, pd.DataFrame]:
    selected_path = _find_existing(output_dir, [f"selected_portfolio_{scan_date}_top*.csv"])
    return {
        "market": _read_csv(output_dir / f"market_scan_snapshot_{scan_date}.csv"),
        "candidates": _read_csv(output_dir / f"resonance_candidates_{scan_date}_all.csv"),
        "selected": _read_csv(selected_path) if selected_path else pd.DataFrame(),
        "summary": _read_csv(output_dir / f"resonance_summary_{scan_date}.csv"),
        "backtest_daily": _read_csv(output_dir / f"forward_backtest_daily_{scan_date}.csv"),
        "backtest_trades": _read_csv(output_dir / f"forward_backtest_trades_{scan_date}.csv"),
        "backtest_summary": _read_csv(output_dir / f"forward_backtest_summary_{scan_date}.csv"),
    }


def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _apply_filters(df: pd.DataFrame, args: dict[str, str]) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out = _coerce_numeric(
        out,
        [
            "resonance_score",
            "amount",
            "turnover_rate",
            "support_count",
            "selected_rank",
            "return_pct",
            "max_runup_pct",
            "nav",
            "strategy_daily_return",
        ],
    )

    min_resonance_score = float(args.get("min_resonance_score", 0.0) or 0.0)
    min_amount = float(args.get("min_amount", 0.0) or 0.0)
    min_turnover = float(args.get("min_turnover", 0.0) or 0.0)
    support_count_min = int(args.get("support_count_min", 0) or 0)
    include_st = str(args.get("include_st", "0")) == "1"
    name_query = str(args.get("name_query", "")).strip().lower()
    industry = str(args.get("industry", "")).strip()
    market = str(args.get("market", "")).strip()
    area = str(args.get("area", "")).strip()
    limit = max(1, int(args.get("limit", 100) or 100))

    if min_resonance_score > 0 and "resonance_score" in out.columns:
        out = out[out["resonance_score"].fillna(-999.0) >= min_resonance_score]
    if min_amount > 0 and "amount" in out.columns:
        out = out[out["amount"].fillna(0.0) >= min_amount]
    if min_turnover > 0 and "turnover_rate" in out.columns:
        out = out[out["turnover_rate"].fillna(0.0) >= min_turnover]
    if support_count_min > 0 and "support_count" in out.columns:
        out = out[out["support_count"].fillna(0) >= support_count_min]
    if not include_st and "is_st" in out.columns:
        out = out[~out["is_st"].fillna(False).astype(bool)]
    if industry and "industry" in out.columns:
        out = out[out["industry"].fillna("").astype(str) == industry]
    if market and "market" in out.columns:
        out = out[out["market"].fillna("").astype(str) == market]
    if area and "area" in out.columns:
        out = out[out["area"].fillna("").astype(str) == area]
    if name_query:
        name_series = out.get("name", pd.Series(index=out.index, dtype=str)).fillna("").astype(str).str.lower()
        symbol_series = out.get("symbol", pd.Series(index=out.index, dtype=str)).fillna("").astype(str).str.lower()
        ts_code_series = out.get("ts_code", pd.Series(index=out.index, dtype=str)).fillna("").astype(str).str.lower()
        industry_series = out.get("industry", pd.Series(index=out.index, dtype=str)).fillna("").astype(str).str.lower()
        mask = (
            name_series.str.contains(name_query, regex=False)
            | symbol_series.str.contains(name_query, regex=False)
            | ts_code_series.str.contains(name_query, regex=False)
            | industry_series.str.contains(name_query, regex=False)
        )
        out = out[mask]

    if "selected_rank" in out.columns:
        out = out.sort_values(["selected_rank"], ascending=[True])
    elif "nav" in out.columns and "scan_date" in out.columns:
        out = out.sort_values(["scan_date"], ascending=[True])
    elif "resonance_score" in out.columns:
        out = out.sort_values(["resonance_score", "support_count", "amount"], ascending=[False, False, False])

    return out.head(limit).reset_index(drop=True)


def _bundle_filter_options(bundle: dict[str, pd.DataFrame]) -> dict[str, list[str]]:
    source = bundle.get("market", pd.DataFrame())
    if source.empty:
        return {"industries": [], "markets": [], "areas": []}

    def _unique_values(column: str) -> list[str]:
        if column not in source.columns:
            return []
        return sorted({str(value) for value in source[column].dropna().astype(str).tolist() if str(value).strip()})

    return {
        "industries": _unique_values("industry"),
        "markets": _unique_values("market"),
        "areas": _unique_values("area"),
    }


def _build_industry_stats(bundle: dict[str, pd.DataFrame], args: dict[str, str]) -> list[dict[str, object]]:
    filter_args = {**args, "limit": "1000000"}
    market = _apply_filters(bundle.get("market", pd.DataFrame()), filter_args)
    candidates = _apply_filters(bundle.get("candidates", pd.DataFrame()), filter_args)
    selected = _apply_filters(bundle.get("selected", pd.DataFrame()), filter_args)
    trades = _apply_filters(bundle.get("backtest_trades", pd.DataFrame()), filter_args)

    frames: list[pd.DataFrame] = []

    if not market.empty and "industry" in market.columns:
        market_stats = (
            market.groupby("industry", dropna=False)
            .agg(universe_count=("symbol", "nunique"), avg_resonance_score=("resonance_score", "mean"))
            .reset_index()
        )
        frames.append(market_stats)

    if not candidates.empty and "industry" in candidates.columns:
        candidate_stats = (
            candidates.groupby("industry", dropna=False)
            .agg(candidate_count=("symbol", "nunique"), candidate_avg_resonance_score=("resonance_score", "mean"))
            .reset_index()
        )
        frames.append(candidate_stats)

    if not selected.empty and "industry" in selected.columns:
        selected_stats = (
            selected.groupby("industry", dropna=False)
            .agg(selected_count=("symbol", "nunique"))
            .reset_index()
        )
        frames.append(selected_stats)

    max_positions = 0
    backtest_summary = bundle.get("backtest_summary", pd.DataFrame())
    if not backtest_summary.empty and "max_positions" in backtest_summary.columns:
        max_positions = int(pd.to_numeric(backtest_summary.iloc[0]["max_positions"], errors="coerce") or 0)

    if not trades.empty and "industry" in trades.columns:
        trade_stats = (
            trades.groupby("industry", dropna=False)
            .agg(
                trade_count=("symbol", "count"),
                avg_trade_return_pct=("return_pct", "mean"),
                total_trade_return_pct=("return_pct", "sum"),
            )
            .reset_index()
        )
        if max_positions > 0:
            trade_stats["nav_contribution_pct"] = trade_stats["total_trade_return_pct"] / float(max_positions)
        else:
            trade_stats["nav_contribution_pct"] = pd.NA
        frames.append(trade_stats)

    if not frames:
        return []

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="industry", how="outer")

    merged["industry"] = merged["industry"].fillna("未分类")
    for column in [
        "candidate_count",
        "universe_count",
        "selected_count",
        "trade_count",
        "avg_resonance_score",
        "candidate_avg_resonance_score",
        "avg_trade_return_pct",
        "total_trade_return_pct",
        "nav_contribution_pct",
    ]:
        if column not in merged.columns:
            merged[column] = pd.NA

    merged["avg_resonance_score"] = merged["avg_resonance_score"].where(
      pd.notna(merged["avg_resonance_score"]),
      merged["candidate_avg_resonance_score"],
    )
    merged["avg_resonance_score"] = merged["candidate_avg_resonance_score"].where(
      pd.notna(merged["candidate_avg_resonance_score"]),
      merged["avg_resonance_score"],
    )
    merged["universe_count"] = pd.to_numeric(merged["universe_count"], errors="coerce").fillna(0).astype(int)
    merged["candidate_count"] = pd.to_numeric(merged["candidate_count"], errors="coerce").fillna(0).astype(int)
    merged["selected_count"] = pd.to_numeric(merged["selected_count"], errors="coerce").fillna(0).astype(int)
    merged["trade_count"] = pd.to_numeric(merged["trade_count"], errors="coerce").fillna(0).astype(int)
    merged = merged.drop(columns=["candidate_avg_resonance_score"], errors="ignore")
    merged = merged.sort_values(
        ["nav_contribution_pct", "selected_count", "candidate_count", "avg_resonance_score"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    return merged.where(pd.notna(merged), None).to_dict(orient="records")


def create_app(scan_output_dir: str) -> Flask:
    app = Flask(__name__)
    output_dir = Path(scan_output_dir).resolve()

    @app.get("/")
    def index() -> str:
        return DASHBOARD_HTML

    @app.get("/healthz")
    def healthz():
        return jsonify({"ok": True, "scan_output_dir": str(output_dir)})

    @app.get("/api/scan_dates")
    def api_scan_dates():
        dates = _scan_dates(output_dir)
        return jsonify({"scan_dates": dates, "latest_scan_date": dates[0] if dates else ""})

    @app.get("/api/filter_options")
    def api_filter_options():
        scan_date = _resolve_scan_date(output_dir, request.args.get("scan_date", ""))
        bundle = _load_bundle(output_dir, scan_date)
        return jsonify(_bundle_filter_options(bundle))

    @app.get("/api/industry_stats")
    def api_industry_stats():
        scan_date = _resolve_scan_date(output_dir, request.args.get("scan_date", ""))
        bundle = _load_bundle(output_dir, scan_date)
        rows = _build_industry_stats(bundle, dict(request.args))
        return jsonify({"scan_date": scan_date, "row_count": int(len(rows)), "rows": rows})

    @app.get("/api/scan")
    def api_scan():
        scan_date = _resolve_scan_date(output_dir, request.args.get("scan_date", ""))
        view = str(request.args.get("view", "selected")).strip().lower()
        bundle = _load_bundle(output_dir, scan_date)
        df = bundle.get(view, pd.DataFrame())
        filtered = _apply_filters(df, dict(request.args))
        summary = bundle["summary"].astype(object).where(pd.notna(bundle["summary"]), None) if not bundle["summary"].empty else pd.DataFrame()
        summary_rows = summary.to_dict(orient="records") if not summary.empty else []
        return jsonify(
            {
                "scan_date": scan_date,
                "view": view,
                "row_count": int(len(filtered)),
                "rows": filtered.where(pd.notna(filtered), None).to_dict(orient="records"),
                "summary": summary_rows[0] if summary_rows else {},
            }
        )

    @app.get("/api/backtest")
    def api_backtest():
        scan_date = _resolve_scan_date(output_dir, request.args.get("scan_date", ""))
        bundle = _load_bundle(output_dir, scan_date)
        summary = (
            bundle["backtest_summary"].astype(object).where(pd.notna(bundle["backtest_summary"]), None)
            if not bundle["backtest_summary"].empty
            else pd.DataFrame()
        )
        summary_rows = summary.to_dict(orient="records") if not summary.empty else []
        trades = _apply_filters(bundle["backtest_trades"], {**dict(request.args), "limit": request.args.get("limit", 200)})
        daily = _apply_filters(bundle["backtest_daily"], {**dict(request.args), "limit": request.args.get("limit", 2000)})
        return jsonify(
            {
                "scan_date": scan_date,
                "summary": summary_rows[0] if summary_rows else {},
                "daily": daily.where(pd.notna(daily), None).to_dict(orient="records"),
                "trades": trades.where(pd.notna(trades), None).to_dict(orient="records"),
            }
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-output-dir", type=str, default=str(DEFAULT_SCAN_OUTPUT_DIR))
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = create_app(args.scan_output_dir)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
