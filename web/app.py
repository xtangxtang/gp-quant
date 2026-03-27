import argparse
import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from flask import Flask, jsonify, request


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
STRATEGY_ROOT = REPO_ROOT / "src" / "strategy"
DEFAULT_DATA_DIR = WORKSPACE_ROOT / "gp-data" / "tushare-daily-full"
DEFAULT_BASIC_PATH = WORKSPACE_ROOT / "gp-data" / "tushare_stock_basic.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "strategy_runs"


APP_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>gp-quant Strategy Console</title>
  <style>
    :root {
      --paper: #f5efe6;
      --paper-strong: #efe2d1;
      --ink: #162024;
      --muted: #5c666d;
      --line: rgba(22, 32, 36, 0.12);
      --card: rgba(255,255,255,0.72);
      --accent: #0f766e;
      --accent-strong: #0b5c56;
      --accent-soft: rgba(15, 118, 110, 0.10);
      --warm: #b45309;
      --good: #1e7a48;
      --bad: #a63f3f;
      --shadow: 0 24px 70px rgba(48, 36, 24, 0.14);
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(240, 184, 122, 0.32), transparent 24%),
        radial-gradient(circle at right 10%, rgba(15, 118, 110, 0.12), transparent 22%),
        linear-gradient(180deg, #f8f2e8 0%, #efe5d7 48%, #f7f2ea 100%);
      font-family: Georgia, "Noto Serif SC", "Source Han Serif SC", serif;
    }

    .shell {
      max-width: 1480px;
      margin: 0 auto;
      padding: 24px 18px 44px;
    }

    .hero {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 28px;
      box-shadow: var(--shadow);
      padding: 26px 24px 22px;
      margin-bottom: 18px;
      position: relative;
      overflow: hidden;
    }

    .hero::after {
      content: "";
      position: absolute;
      right: -20px;
      top: -26px;
      width: 170px;
      height: 170px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(15, 118, 110, 0.16), transparent 68%);
      pointer-events: none;
    }

    .eyebrow {
      margin-bottom: 10px;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.18em;
      color: var(--accent-strong);
    }

    h1 {
      margin: 0 0 12px;
      font-size: clamp(34px, 5vw, 58px);
      line-height: 0.96;
    }

    .hero-copy {
      max-width: 760px;
      color: var(--muted);
      line-height: 1.7;
      font-size: 16px;
    }

    .status-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 18px;
    }

    .pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid rgba(15, 118, 110, 0.16);
      background: rgba(255,255,255,0.58);
      font-size: 13px;
      color: var(--muted);
    }

    .layout {
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }

    .panel {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(12px);
    }

    .sidebar {
      padding: 18px;
      position: sticky;
      top: 18px;
    }

    .sidebar h2,
    .content h2,
    .content h3 {
      margin: 0;
    }

    .sidebar-head {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      margin-bottom: 14px;
    }

    .sidebar-head span {
      color: var(--muted);
      font-size: 12px;
    }

    .strategy-list {
      display: grid;
      gap: 10px;
    }

    .strategy-card {
      padding: 14px 14px 12px;
      border-radius: 18px;
      border: 1px solid rgba(22, 32, 36, 0.08);
      background: rgba(255,255,255,0.56);
      cursor: pointer;
      transition: transform 180ms ease, border-color 180ms ease, background 180ms ease;
    }

    .strategy-card:hover {
      transform: translateY(-1px);
      border-color: rgba(15, 118, 110, 0.28);
      background: rgba(255,255,255,0.78);
    }

    .strategy-card.active {
      background: linear-gradient(135deg, rgba(15, 118, 110, 0.14), rgba(255,255,255,0.82));
      border-color: rgba(15, 118, 110, 0.38);
    }

    .strategy-name {
      font-size: 18px;
      margin-bottom: 6px;
    }

    .strategy-path {
      color: var(--warm);
      font-size: 12px;
      margin-bottom: 8px;
      word-break: break-all;
    }

    .strategy-summary {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.55;
    }

    .content {
      padding: 20px;
    }

    .content[hidden] { display: none !important; }

    .empty-state {
      padding: 32px;
      text-align: center;
      color: var(--muted);
      border: 1px dashed rgba(22, 32, 36, 0.18);
      border-radius: 20px;
      background: rgba(255,255,255,0.42);
    }

    .detail-top {
      display: grid;
      grid-template-columns: 1.08fr 0.92fr;
      gap: 18px;
      margin-bottom: 18px;
    }

    .detail-block,
    .result-block {
      padding: 18px;
      border-radius: 22px;
      border: 1px solid rgba(22, 32, 36, 0.08);
      background: rgba(255,255,255,0.55);
    }

    .detail-title {
      display: flex;
      justify-content: space-between;
      align-items: start;
      gap: 14px;
      margin-bottom: 14px;
    }

    .detail-title .meta {
      color: var(--muted);
      font-size: 13px;
    }

    .block-label {
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 12px;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--accent-strong);
      background: var(--accent-soft);
      margin-bottom: 10px;
    }

    .markdown p {
      margin: 0 0 12px;
      line-height: 1.72;
      color: var(--ink);
    }

    .markdown ul {
      margin: 0;
      padding-left: 18px;
      color: var(--muted);
      line-height: 1.7;
    }

    .form-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-top: 14px;
    }

    .field {
      padding: 12px;
      border-radius: 16px;
      border: 1px solid rgba(22, 32, 36, 0.08);
      background: rgba(255,255,255,0.48);
    }

    .field label {
      display: block;
      margin-bottom: 6px;
      font-size: 12px;
      color: var(--muted);
    }

    .field small {
      display: block;
      margin-top: 6px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.5;
    }

    input, select, textarea {
      width: 100%;
      font: inherit;
      color: var(--ink);
      background: rgba(255,255,255,0.82);
      border: 1px solid rgba(22, 32, 36, 0.14);
      border-radius: 12px;
      padding: 10px 12px;
    }

    textarea {
      min-height: 96px;
      resize: vertical;
    }

    .checkbox-wrap {
      display: flex;
      align-items: center;
      gap: 10px;
      min-height: 42px;
    }

    .checkbox-wrap input {
      width: auto;
      transform: scale(1.15);
    }

    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 16px;
      align-items: center;
    }

    button {
      font: inherit;
      cursor: pointer;
      border-radius: 999px;
      padding: 11px 16px;
      border: 0;
      transition: transform 180ms ease, opacity 180ms ease;
    }

    button:hover { transform: translateY(-1px); }
    button:disabled { cursor: not-allowed; opacity: 0.6; transform: none; }

    .btn-primary {
      background: var(--accent);
      color: #f3f8f7;
    }

    .btn-secondary {
      background: rgba(255,255,255,0.74);
      color: var(--ink);
      border: 1px solid rgba(22, 32, 36, 0.12);
    }

    .run-status {
      color: var(--muted);
      font-size: 13px;
    }

    .summary-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }

    .metric {
      padding: 14px;
      border-radius: 18px;
      background: rgba(255,255,255,0.5);
      border: 1px solid rgba(22, 32, 36, 0.08);
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

    .results-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
    }

    .table-wrap {
      overflow: auto;
      border-radius: 18px;
      border: 1px solid rgba(22, 32, 36, 0.08);
      background: rgba(255,255,255,0.44);
    }

    table {
      width: 100%;
      border-collapse: collapse;
      min-width: 780px;
      font-size: 13px;
    }

    th, td {
      padding: 10px 12px;
      text-align: left;
      border-bottom: 1px solid rgba(22, 32, 36, 0.08);
      white-space: nowrap;
    }

    thead th {
      position: sticky;
      top: 0;
      z-index: 1;
      background: #efe5d8;
    }

    pre {
      margin: 0;
      padding: 14px;
      border-radius: 18px;
      background: #1e2528;
      color: #e7efe9;
      overflow: auto;
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 12px;
      line-height: 1.55;
    }

    .muted {
      color: var(--muted);
    }

    @media (max-width: 1220px) {
      .layout { grid-template-columns: 1fr; }
      .sidebar { position: static; }
      .detail-top,
      .results-grid { grid-template-columns: 1fr; }
      .form-grid,
      .summary-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }

    @media (max-width: 760px) {
      .shell { padding: 16px 12px 28px; }
      .form-grid,
      .summary-grid { grid-template-columns: 1fr; }
      .hero { padding: 20px 18px; }
      .content { padding: 16px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="eyebrow">gp-quant Strategy Console</div>
      <h1>Strategy Index<br/>and Apply Studio</h1>
      <div class="hero-copy">首页自动发现当前 <code>src/strategy</code> 目录下的所有策略，展示每个策略 README 中的一句话概括。进入策略详情页后，可以查看 README 的“描述”和“主要参数”，按页面表单填写参数，再直接点击“应用”执行选股。</div>
      <div class="status-row">
        <div class="pill">策略目录驱动</div>
        <div class="pill">README 解析展示</div>
        <div class="pill">参数可编辑</div>
        <div class="pill">点击即可应用策略</div>
      </div>
    </section>

    <section class="layout">
      <aside class="panel sidebar">
        <div class="sidebar-head">
          <h2>策略目录</h2>
          <span id="strategy-count">0</span>
        </div>
        <div class="strategy-list" id="strategy-list"></div>
      </aside>

      <main class="panel content">
        <div id="strategy-empty" class="empty-state">正在加载策略目录。</div>

        <div id="strategy-detail" hidden>
          <div class="detail-top">
            <section class="detail-block">
              <div class="detail-title">
                <div>
                  <div class="block-label">README / 描述</div>
                  <h2 id="detail-name"></h2>
                </div>
                <div class="meta" id="detail-path"></div>
              </div>
              <div class="markdown" id="detail-description"></div>
            </section>

            <section class="detail-block">
              <div class="block-label">README / 主要参数</div>
              <div class="markdown" id="detail-params-doc"></div>
            </section>
          </div>

          <section class="detail-block" style="margin-bottom:18px;">
            <div class="detail-title">
              <div>
                <div class="block-label">Apply</div>
                <h3>参数配置</h3>
              </div>
              <div class="meta">修改参数后点击“应用”直接执行当前策略</div>
            </div>
            <form id="strategy-form">
              <div class="form-grid" id="param-form-grid"></div>
              <div class="actions">
                <button type="submit" class="btn-primary" id="apply-btn">应用</button>
                <button type="button" class="btn-secondary" id="reset-btn">恢复默认</button>
                <span class="run-status" id="run-status">Ready</span>
              </div>
            </form>
          </section>

          <section id="results-panel" hidden>
            <div class="summary-grid" id="result-metrics"></div>
            <div class="results-grid">
              <section class="result-block">
                <div class="detail-title">
                  <h3>入选股票</h3>
                  <div class="meta" id="selected-count">0 rows</div>
                </div>
                <div class="table-wrap">
                  <table>
                    <thead id="selected-head"></thead>
                    <tbody id="selected-body"></tbody>
                  </table>
                </div>
              </section>

              <section class="result-block">
                <div class="detail-title">
                  <h3>候选池</h3>
                  <div class="meta" id="candidate-count">0 rows</div>
                </div>
                <div class="table-wrap">
                  <table>
                    <thead id="candidate-head"></thead>
                    <tbody id="candidate-body"></tbody>
                  </table>
                </div>
              </section>
            </div>

            <section class="result-block" style="margin-top:18px;">
              <div class="detail-title">
                <h3>执行日志</h3>
                <div class="meta" id="run-command"></div>
              </div>
              <pre id="run-log"></pre>
            </section>
          </section>
        </div>
      </main>
    </section>
  </div>

  <script>
    const state = {
      strategies: [],
      current: null,
      defaults: {},
    };

    function fmt(value) {
      if (value === null || value === undefined || value === '') return '-';
      if (typeof value === 'number') {
        if (!Number.isFinite(value)) return '-';
        if (Math.abs(value) >= 1000) return value.toLocaleString('zh-CN', { maximumFractionDigits: 2 });
        return value.toLocaleString('zh-CN', { maximumFractionDigits: 4 });
      }
      return String(value);
    }

    function renderBlocks(hostId, blocks) {
      const host = document.getElementById(hostId);
      host.innerHTML = '';
      (blocks || []).forEach((block) => {
        if (block.type === 'ul') {
          const ul = document.createElement('ul');
          (block.items || []).forEach((item) => {
            const li = document.createElement('li');
            li.textContent = item;
            ul.appendChild(li);
          });
          host.appendChild(ul);
          return;
        }
        const p = document.createElement('p');
        p.textContent = block.text || '';
        host.appendChild(p);
      });
    }

    function renderStrategyList() {
      const host = document.getElementById('strategy-list');
      host.innerHTML = '';
      document.getElementById('strategy-count').textContent = `${state.strategies.length} strategies`;
      state.strategies.forEach((strategy) => {
        const card = document.createElement('article');
        card.className = 'strategy-card' + (state.current && state.current.id === strategy.id ? ' active' : '');
        card.innerHTML = `
          <div class="strategy-name">${strategy.display_name}</div>
          <div class="strategy-path">${strategy.relative_path}</div>
          <div class="strategy-summary">${strategy.tagline || '暂无一句话概括'}</div>
        `;
        card.addEventListener('click', () => loadStrategyDetail(strategy.id));
        host.appendChild(card);
      });
    }

    function inputTypeFor(param) {
      if (param.kind === 'boolean') return 'checkbox';
      if (param.kind === 'integer' || param.kind === 'float') return 'number';
      if (param.dest.endsWith('_date')) return 'text';
      return 'text';
    }

    function renderParamForm(detail) {
      const host = document.getElementById('param-form-grid');
      host.innerHTML = '';
      state.defaults = {};

      (detail.parameters || []).forEach((param) => {
        state.defaults[param.dest] = param.value;
        const wrapper = document.createElement('div');
        wrapper.className = 'field';

        const label = document.createElement('label');
        label.setAttribute('for', `param-${param.dest}`);
        label.textContent = param.label;
        wrapper.appendChild(label);

        if (param.kind === 'boolean') {
          const row = document.createElement('div');
          row.className = 'checkbox-wrap';
          const input = document.createElement('input');
          input.type = 'checkbox';
          input.id = `param-${param.dest}`;
          input.name = param.dest;
          input.checked = Boolean(param.value);
          row.appendChild(input);
          const hint = document.createElement('span');
          hint.textContent = Boolean(param.value) ? '当前启用' : '当前关闭';
          input.addEventListener('change', () => {
            hint.textContent = input.checked ? '当前启用' : '当前关闭';
          });
          row.appendChild(hint);
          wrapper.appendChild(row);
        } else if ((param.choices || []).length > 0) {
          const select = document.createElement('select');
          select.id = `param-${param.dest}`;
          select.name = param.dest;
          (param.choices || []).forEach((choice) => {
            const option = document.createElement('option');
            option.value = choice;
            option.textContent = choice;
            if (String(choice) === String(param.value)) option.selected = true;
            select.appendChild(option);
          });
          wrapper.appendChild(select);
        } else if ((param.dest === 'symbols' || param.dest === 'index_path') && String(param.value || '').length > 40) {
          const textarea = document.createElement('textarea');
          textarea.id = `param-${param.dest}`;
          textarea.name = param.dest;
          textarea.value = param.value == null ? '' : String(param.value);
          wrapper.appendChild(textarea);
        } else {
          const input = document.createElement('input');
          input.id = `param-${param.dest}`;
          input.name = param.dest;
          input.type = inputTypeFor(param);
          if (param.kind === 'integer') input.step = '1';
          if (param.kind === 'float') input.step = '0.01';
          input.value = param.value == null ? '' : String(param.value);
          wrapper.appendChild(input);
        }

        const meta = [];
        if (param.required) meta.push('必填');
        if (param.default_display) meta.push(`默认: ${param.default_display}`);
        if (param.option_strings && param.option_strings.length) meta.push(param.option_strings.join(' / '));
        if (param.help_text) meta.push(param.help_text);

        const help = document.createElement('small');
        help.textContent = meta.join(' | ');
        wrapper.appendChild(help);
        host.appendChild(wrapper);
      });
    }

    function collectFormValues(detail) {
      const values = {};
      (detail.parameters || []).forEach((param) => {
        const node = document.getElementById(`param-${param.dest}`);
        if (!node) return;
        if (param.kind === 'boolean') {
          values[param.dest] = Boolean(node.checked);
        } else if (param.kind === 'integer') {
          values[param.dest] = node.value === '' ? '' : Number.parseInt(node.value, 10);
        } else if (param.kind === 'float') {
          values[param.dest] = node.value === '' ? '' : Number.parseFloat(node.value);
        } else {
          values[param.dest] = node.value;
        }
      });
      return values;
    }

    function resetForm(detail) {
      (detail.parameters || []).forEach((param) => {
        const node = document.getElementById(`param-${param.dest}`);
        if (!node) return;
        const value = state.defaults[param.dest];
        if (param.kind === 'boolean') {
          node.checked = Boolean(value);
        } else {
          node.value = value == null ? '' : String(value);
        }
      });
    }

    function renderMetrics(result) {
      const host = document.getElementById('result-metrics');
      const summary = result.summary || {};
      const backtest = result.backtest_summary || {};
      const metrics = [
        ['扫描日期', result.scan_date || '-'],
        ['候选数', summary.n_resonance_candidates ?? result.candidate_count ?? 0],
        ['入选数', summary.n_selected ?? result.selected_count ?? 0],
        ['最终净值', backtest.final_nav ?? '-'],
      ];
      host.innerHTML = metrics.map(([label, value]) => `
        <div class="metric">
          <div class="metric-label">${label}</div>
          <div class="metric-value">${fmt(value)}</div>
        </div>
      `).join('');
    }

    function renderTable(headId, bodyId, rows, columns) {
      const thead = document.getElementById(headId);
      const tbody = document.getElementById(bodyId);
      thead.innerHTML = '';
      tbody.innerHTML = '';
      if (!rows || rows.length === 0) return;

      const headRow = document.createElement('tr');
      columns.forEach((column) => {
        const th = document.createElement('th');
        th.textContent = column;
        headRow.appendChild(th);
      });
      thead.appendChild(headRow);

      rows.forEach((row) => {
        const tr = document.createElement('tr');
        columns.forEach((column) => {
          const td = document.createElement('td');
          td.textContent = fmt(row[column]);
          tr.appendChild(td);
        });
        tbody.appendChild(tr);
      });
    }

    function renderApplyResult(result) {
      document.getElementById('results-panel').hidden = false;
      renderMetrics(result);
      const selectedRows = result.selected_rows || [];
      const candidateRows = result.candidate_rows || [];
      document.getElementById('selected-count').textContent = `${selectedRows.length} rows`;
      document.getElementById('candidate-count').textContent = `${candidateRows.length} rows`;
      renderTable('selected-head', 'selected-body', selectedRows, ['selected_rank', 'symbol', 'name', 'industry', 'market', 'resonance_score', 'support_count', 'amount', 'turnover_rate']);
      renderTable('candidate-head', 'candidate-body', candidateRows, ['symbol', 'name', 'industry', 'market', 'resonance_score', 'support_count', 'amount', 'turnover_rate']);
      document.getElementById('run-command').textContent = result.command_display || '';
      document.getElementById('run-log').textContent = result.log || 'No logs';
    }

    async function loadStrategies() {
      const resp = await fetch('/api/strategies');
      const payload = await resp.json();
      state.strategies = payload.strategies || [];
      renderStrategyList();
      if (state.strategies.length > 0) {
        await loadStrategyDetail(state.strategies[0].id);
      } else {
        document.getElementById('strategy-empty').textContent = '当前 src/strategy 下没有发现可展示的策略。';
      }
    }

    async function loadStrategyDetail(strategyId) {
      const resp = await fetch(`/api/strategies/${encodeURIComponent(strategyId)}`);
      const detail = await resp.json();
      state.current = detail;
      renderStrategyList();
      document.getElementById('strategy-empty').hidden = true;
      document.getElementById('strategy-detail').hidden = false;
      document.getElementById('detail-name').textContent = detail.display_name;
      document.getElementById('detail-path').textContent = detail.relative_path;
      renderBlocks('detail-description', detail.readme.description_blocks || []);
      renderBlocks('detail-params-doc', detail.readme.parameter_blocks || []);
      renderParamForm(detail);
      document.getElementById('results-panel').hidden = true;
      document.getElementById('run-status').textContent = 'Ready';
    }

    async function applyStrategy(event) {
      event.preventDefault();
      if (!state.current) return;
      const button = document.getElementById('apply-btn');
      const status = document.getElementById('run-status');
      button.disabled = true;
      status.textContent = 'Applying...';
      try {
        const values = collectFormValues(state.current);
        const resp = await fetch(`/api/strategies/${encodeURIComponent(state.current.id)}/apply`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ values }),
        });
        const payload = await resp.json();
        if (!resp.ok) {
          throw new Error(payload.error || 'Apply failed');
        }
        renderApplyResult(payload);
        status.textContent = 'Applied';
      } catch (error) {
        status.textContent = error.message || 'Apply failed';
      } finally {
        button.disabled = false;
      }
    }

    function bindEvents() {
      document.getElementById('strategy-form').addEventListener('submit', applyStrategy);
      document.getElementById('reset-btn').addEventListener('click', () => {
        if (state.current) resetForm(state.current);
      });
    }

    (async function init() {
      bindEvents();
      await loadStrategies();
    })();
  </script>
</body>
</html>
"""


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _section_map(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = []
            continue
        if current is not None:
            sections[current].append(line)
    return {key: "\n".join(value).strip() for key, value in sections.items()}


def _simple_markdown_blocks(text: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    current: list[str] = []
    mode: str | None = None

    def flush() -> None:
        nonlocal current, mode
        if not current:
            return
        if mode == "ul":
            items = [line[2:].strip() for line in current if line.startswith("- ")]
            if items:
                blocks.append({"type": "ul", "items": items})
        else:
            paragraph = " ".join(line.strip() for line in current if line.strip())
            if paragraph:
                blocks.append({"type": "p", "text": paragraph})
        current = []
        mode = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            flush()
            continue
        if line.startswith("- "):
            if mode not in {None, "ul"}:
                flush()
            mode = "ul"
            current.append(line)
        else:
            if mode not in {None, "p"}:
                flush()
            mode = "p"
            current.append(line)
    flush()
    return blocks


def _extract_tagline(description_text: str) -> str:
    match = re.search(r"一句话概括[^，。：:]*[：:，]?\s*(.+)", description_text)
    if match:
        return match.group(1).strip()
    blocks = _simple_markdown_blocks(description_text)
    for block in blocks:
        if block.get("type") == "p" and block.get("text"):
            return str(block["text"]).strip()
    return ""


def _extract_readme_metadata(readme_path: Path) -> dict[str, Any]:
    text = _read_text(readme_path)
    sections = _section_map(text)
    description_text = sections.get("描述", "")
    parameter_text = sections.get("主要参数", "")
    return {
        "description_text": description_text,
        "parameter_text": parameter_text,
        "description_blocks": _simple_markdown_blocks(description_text),
        "parameter_blocks": _simple_markdown_blocks(parameter_text),
        "tagline": _extract_tagline(description_text),
    }


def _parse_readme_parameter_help(parameter_text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in parameter_text.splitlines():
        match = re.match(r"-\s+`([^`]+)`：\s*(.+)", line.strip())
        if not match:
            continue
        raw_options = [item.strip() for item in match.group(1).split("/")]
        help_text = match.group(2).strip()
        for option in raw_options:
            result[option] = help_text
    return result


def _import_strategy_module(module_path: Path, strategy_id: str):
    module_name = f"gp_quant_strategy_{strategy_id.replace('-', '_').replace('/', '_')}"
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load strategy module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _infer_kind(action: argparse.Action) -> str:
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        return "boolean"
    if action.type is int:
        return "integer"
    if action.type is float:
        return "float"
    return "string"


def _default_value_for_dest(strategy_id: str, dest: str) -> Any:
    strategy_output_dir = DEFAULT_OUTPUT_ROOT / strategy_id / "latest"
    mapping: dict[str, Any] = {
        "data_dir": str(DEFAULT_DATA_DIR),
        "basic_path": str(DEFAULT_BASIC_PATH),
        "out_dir": str(strategy_output_dir),
        "scan_date": "",
        "index_path": "",
        "symbols": "",
    }
    return mapping.get(dest)


def _parameter_definitions(entrypoint_path: Path, strategy_id: str, readme_help: dict[str, str]) -> list[dict[str, Any]]:
    module = _import_strategy_module(entrypoint_path, strategy_id)
    if not hasattr(module, "_build_argument_parser"):
        return []
    parser = module._build_argument_parser()

    grouped: dict[str, list[argparse.Action]] = {}
    order: list[str] = []
    for action in parser._actions:
        if action.dest == "help":
            continue
        if action.dest not in grouped:
            grouped[action.dest] = []
            order.append(action.dest)
        grouped[action.dest].append(action)

    params: list[dict[str, Any]] = []
    for dest in order:
        actions = grouped[dest]
        primary = actions[0]
        option_strings = [item for action in actions for item in action.option_strings if item.startswith("--")]
        kind = "boolean" if any(isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)) for action in actions) else _infer_kind(primary)
        default_value = parser.get_default(dest)
        if default_value is None or (isinstance(default_value, str) and default_value == ""):
            inferred = _default_value_for_dest(strategy_id, dest)
            if inferred is not None:
                default_value = inferred
        required = bool(getattr(primary, "required", False))
        if required and (default_value is None or default_value == ""):
            inferred = _default_value_for_dest(strategy_id, dest)
            if inferred is not None:
                default_value = inferred
                required = False

        help_text = primary.help or ""
        for option in option_strings:
            if option in readme_help:
                help_text = readme_help[option]
                break

        bool_map = {
            "true_option": None,
            "false_option": None,
        }
        if kind == "boolean":
            for action in actions:
                if isinstance(action, argparse._StoreTrueAction) and action.option_strings:
                    bool_map["true_option"] = action.option_strings[0]
                if isinstance(action, argparse._StoreFalseAction) and action.option_strings:
                    bool_map["false_option"] = action.option_strings[0]
            if default_value is None:
                default_value = bool(primary.default)

        label = option_strings[0] if option_strings else dest
        params.append(
            {
                "dest": dest,
                "label": label,
                "kind": kind,
                "required": required,
                "value": default_value,
                "default": default_value,
                "default_display": "" if default_value in {None, ""} else str(default_value),
                "option_strings": option_strings,
                "choices": list(primary.choices) if primary.choices else [],
                "help_text": help_text,
                "bool_map": bool_map,
            }
        )
    return params


def _discover_strategies() -> list[dict[str, Any]]:
    strategies: list[dict[str, Any]] = []
    if not STRATEGY_ROOT.exists():
        return strategies

    for readme_path in sorted(STRATEGY_ROOT.rglob("README.md")):
        strategy_dir = readme_path.parent
        relative_path = strategy_dir.relative_to(REPO_ROOT)
        entrypoints = sorted(strategy_dir.glob("run_*.py"))
        if not entrypoints:
            continue
        strategy_id = str(strategy_dir.relative_to(STRATEGY_ROOT)).replace("/", "-")
        readme = _extract_readme_metadata(readme_path)
        parameter_help = _parse_readme_parameter_help(readme["parameter_text"])
        parameters = _parameter_definitions(entrypoints[0], strategy_id, parameter_help)
        strategies.append(
            {
                "id": strategy_id,
                "display_name": strategy_dir.name,
                "relative_path": str(relative_path),
                "directory": strategy_dir,
                "readme_path": readme_path,
                "entrypoint_path": entrypoints[0],
                "readme": readme,
                "parameters": parameters,
            }
        )
    return strategies


def _strategy_index() -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in _discover_strategies()}


def _scan_dates(output_dir: Path) -> list[str]:
    dates: list[str] = []
    for path in output_dir.glob("market_scan_snapshot_*.csv"):
        dates.append(path.stem.rsplit("_", 1)[-1])
    return sorted(set(dates), reverse=True)


def _find_existing(output_dir: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(output_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _load_bundle(output_dir: Path, scan_date: str) -> dict[str, pd.DataFrame]:
    selected_path = _find_existing(output_dir, [f"selected_portfolio_{scan_date}_top*.csv"])
    return {
        "summary": _read_csv(output_dir / f"resonance_summary_{scan_date}.csv"),
        "selected": _read_csv(selected_path) if selected_path else pd.DataFrame(),
        "candidates": _read_csv(output_dir / f"resonance_candidates_{scan_date}_all.csv"),
        "backtest_summary": _read_csv(output_dir / f"forward_backtest_summary_{scan_date}.csv"),
    }


def _normalize_rows(df: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    if df.empty:
        return []
    trimmed = df.head(limit).astype(object).where(pd.notna(df.head(limit)), None)
    return trimmed.to_dict(orient="records")


def _resolve_strategy(strategy_id: str) -> dict[str, Any]:
    strategies = _strategy_index()
    if strategy_id not in strategies:
        raise KeyError(strategy_id)
    return strategies[strategy_id]


def _serialize_strategy(strategy: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": strategy["id"],
        "display_name": strategy["display_name"],
        "relative_path": strategy["relative_path"],
        "tagline": strategy["readme"].get("tagline", ""),
    }


def _serialize_strategy_detail(strategy: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": strategy["id"],
        "display_name": strategy["display_name"],
        "relative_path": strategy["relative_path"],
        "readme": {
            "description_blocks": strategy["readme"].get("description_blocks", []),
            "parameter_blocks": strategy["readme"].get("parameter_blocks", []),
            "tagline": strategy["readme"].get("tagline", ""),
        },
        "parameters": strategy["parameters"],
    }


def _build_command(strategy: dict[str, Any], values: dict[str, Any]) -> tuple[list[str], str, Path]:
    command = [sys.executable, str(strategy["entrypoint_path"])]
    effective_values = dict(values)

    out_dir_value = effective_values.get("out_dir") or _default_value_for_dest(strategy["id"], "out_dir")
    effective_values["out_dir"] = out_dir_value
    output_dir = Path(str(out_dir_value)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for param in strategy["parameters"]:
        dest = param["dest"]
        value = effective_values.get(dest, param.get("default"))
        if param["kind"] == "boolean":
            desired = bool(value)
            default = bool(param.get("default"))
            if desired == default:
                continue
            option = param["bool_map"]["true_option"] if desired else param["bool_map"]["false_option"]
            if option:
                command.append(option)
            continue

        if value is None:
            value = ""
        value_str = str(value)
        if not value_str.strip() and not param.get("required"):
            continue
        option = param["option_strings"][0] if param["option_strings"] else f"--{dest}"
        command.extend([option, value_str])

    return command, " ".join(command), output_dir


def _run_strategy(strategy: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    command, command_display, output_dir = _build_command(strategy, values)
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        timeout=3600,
    )
    log = (completed.stdout or "") + ("\n" if completed.stdout and completed.stderr else "") + (completed.stderr or "")
    if completed.returncode != 0:
        raise RuntimeError(log.strip() or f"Strategy exited with code {completed.returncode}")

    scan_dates = _scan_dates(output_dir)
    scan_date = scan_dates[0] if scan_dates else str(values.get("scan_date") or "")
    bundle = _load_bundle(output_dir, scan_date) if scan_date else {"summary": pd.DataFrame(), "selected": pd.DataFrame(), "candidates": pd.DataFrame(), "backtest_summary": pd.DataFrame()}
    summary_rows = _normalize_rows(bundle.get("summary", pd.DataFrame()), 1)
    backtest_summary_rows = _normalize_rows(bundle.get("backtest_summary", pd.DataFrame()), 1)
    selected_rows = _normalize_rows(bundle.get("selected", pd.DataFrame()), 100)
    candidate_rows = _normalize_rows(bundle.get("candidates", pd.DataFrame()), 100)
    return {
        "ok": True,
        "scan_date": scan_date,
        "summary": summary_rows[0] if summary_rows else {},
        "backtest_summary": backtest_summary_rows[0] if backtest_summary_rows else {},
        "selected_rows": selected_rows,
        "candidate_rows": candidate_rows,
        "selected_count": len(selected_rows),
        "candidate_count": len(candidate_rows),
        "command_display": command_display,
        "log": log.strip() or "No logs captured",
        "output_dir": str(output_dir),
    }


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index() -> str:
        return APP_HTML

    @app.get("/healthz")
    def healthz():
        return jsonify({"ok": True, "strategy_root": str(STRATEGY_ROOT)})

    @app.get("/api/strategies")
    def api_strategies():
        strategies = [_serialize_strategy(item) for item in _discover_strategies()]
        return jsonify({"strategies": strategies})

    @app.get("/api/strategies/<strategy_id>")
    def api_strategy_detail(strategy_id: str):
        try:
            strategy = _resolve_strategy(strategy_id)
        except KeyError:
            return jsonify({"error": "Strategy not found"}), 404
        return jsonify(_serialize_strategy_detail(strategy))

    @app.post("/api/strategies/<strategy_id>/apply")
    def api_strategy_apply(strategy_id: str):
        try:
            strategy = _resolve_strategy(strategy_id)
        except KeyError:
            return jsonify({"error": "Strategy not found"}), 404

        payload = request.get_json(silent=True) or {}
        values = payload.get("values", {}) if isinstance(payload, dict) else {}
        try:
            result = _run_strategy(strategy, values)
            return jsonify(result)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
