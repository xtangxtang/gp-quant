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
STRATEGY_LABELS = {
  "entropy_bifurcation_setup": "熵分叉启动",
  "uptrend_hold_state_flow": "上升趋势持有状态图",
}
STRATEGY_TAGLINES = {
  "entropy_bifurcation_setup": "在低熵压缩态中，用临界慢化和突破触发去捕捉分叉启动时刻。",
  "uptrend_hold_state_flow": "把上升趋势里的熵秩序持有、快速扩张持有、快速扩张衰竭退出放进一张状态图里，从买点开始评估整段持有路径。",
}
STRATEGY_VARIANT_LABELS = {
  "compression_breakout": "低熵压缩突破",
  "self_organized_trend": "自组织趋势跟随",
  "fractal_pullback": "分形回踩续涨",
  "market_energy_flow": "市场能量流",
}
STRATEGY_VARIANT_TAGLINES = {
  "compression_breakout": "波动率压缩后跟踪放量突破与秩序切换。",
  "self_organized_trend": "资金持续注入后的顺势跟随与强者恒强筛选。",
  "fractal_pullback": "在大级别上升趋势里寻找缩量回踩后的再启动。",
  "market_energy_flow": "从横截面资金能量与行业共振中筛选领先方向。",
}


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

    .markdown .markdown-table-wrap {
      margin: 0 0 14px;
      overflow: auto;
      border-radius: 16px;
      border: 1px solid rgba(22, 32, 36, 0.08);
      background: rgba(255,255,255,0.46);
    }

    .markdown table {
      min-width: 680px;
      font-size: 13px;
    }

    .markdown th,
    .markdown td {
      white-space: normal;
      vertical-align: top;
      line-height: 1.6;
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

    .state-flow-block {
      margin-bottom: 18px;
      padding: 18px;
      border-radius: 22px;
      border: 1px solid rgba(22, 32, 36, 0.08);
      background: rgba(255,255,255,0.55);
    }

    .state-flow-summary {
      margin: 10px 0 16px;
      padding: 14px 16px;
      border-radius: 18px;
      background: rgba(15, 118, 110, 0.08);
      border: 1px solid rgba(15, 118, 110, 0.10);
      color: var(--ink);
      line-height: 1.7;
      font-size: 14px;
    }

    .state-flow-diagram {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 120px minmax(0, 1fr) 120px minmax(0, 1fr) 120px minmax(0, 1fr);
      gap: 12px;
      align-items: center;
      margin-bottom: 18px;
    }

    .state-node {
      position: relative;
      padding: 14px 14px 12px;
      border-radius: 18px;
      border: 1px solid rgba(22, 32, 36, 0.10);
      background: rgba(255,255,255,0.52);
      min-height: 132px;
    }

    .state-node.active {
      border-color: rgba(15, 118, 110, 0.45);
      background: linear-gradient(135deg, rgba(15, 118, 110, 0.16), rgba(255,255,255,0.86));
      box-shadow: inset 0 0 0 1px rgba(15, 118, 110, 0.16);
    }

    .state-node.path-active:not(.active) {
      border-color: rgba(15, 118, 110, 0.20);
      background: linear-gradient(180deg, rgba(15, 118, 110, 0.08), rgba(255,255,255,0.78));
    }

    .state-badge {
      display: inline-block;
      margin-bottom: 8px;
      padding: 4px 8px;
      border-radius: 999px;
      font-size: 11px;
      color: var(--accent-strong);
      background: rgba(15, 118, 110, 0.10);
    }

    .state-name {
      font-size: 16px;
      margin-bottom: 8px;
      font-weight: 700;
    }

    .state-score {
      font-size: 24px;
      line-height: 1;
      margin-bottom: 10px;
    }

    .state-reason {
      font-size: 12px;
      line-height: 1.6;
      color: var(--muted);
    }

    .flow-edge {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 8px;
      min-height: 132px;
    }

    .flow-edge-label {
      font-size: 12px;
      color: var(--muted);
      text-align: center;
      line-height: 1.4;
    }

    .flow-edge-rail {
      position: relative;
      width: 100%;
      height: 14px;
    }

    .flow-edge-rail::before {
      content: "";
      position: absolute;
      left: 0;
      right: 14px;
      top: 50%;
      height: 2px;
      transform: translateY(-50%);
      background: rgba(22, 32, 36, 0.18);
    }

    .flow-edge-rail::after {
      content: "";
      position: absolute;
      right: 2px;
      top: 50%;
      width: 9px;
      height: 9px;
      border-top: 2px solid rgba(22, 32, 36, 0.18);
      border-right: 2px solid rgba(22, 32, 36, 0.18);
      transform: translateY(-50%) rotate(45deg);
    }

    .flow-edge.active .flow-edge-label {
      color: var(--accent-strong);
    }

    .flow-edge.active .flow-edge-rail::before {
      background: linear-gradient(90deg, rgba(15, 118, 110, 0.55), rgba(15, 118, 110, 0.95));
    }

    .flow-edge.active .flow-edge-rail::after {
      border-top-color: rgba(15, 118, 110, 0.95);
      border-right-color: rgba(15, 118, 110, 0.95);
    }

    .state-compare-block {
      margin-top: 8px;
    }

    .state-compare-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-top: 12px;
    }

    .state-compare-card {
      padding: 16px;
      border-radius: 18px;
      border: 1px solid rgba(22, 32, 36, 0.08);
      background: rgba(255,255,255,0.58);
    }

    .state-compare-label {
      font-size: 16px;
      font-weight: 700;
      margin-bottom: 6px;
    }

    .state-compare-score {
      font-size: 12px;
      color: var(--accent-strong);
      margin-bottom: 8px;
    }

    .state-compare-reason {
      font-size: 13px;
      line-height: 1.7;
      color: var(--muted);
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
      .state-flow-diagram {
        grid-template-columns: 1fr;
      }
      .flow-edge {
        min-height: 58px;
      }
      .flow-edge-rail {
        width: 20px;
        height: 52px;
      }
      .flow-edge-rail::before {
        left: 50%;
        right: auto;
        top: 0;
        width: 2px;
        height: calc(100% - 14px);
        transform: translateX(-50%);
      }
      .flow-edge-rail::after {
        top: auto;
        bottom: 3px;
        left: 50%;
        right: auto;
        transform: translateX(-50%) rotate(135deg);
      }
      .state-compare-grid {
        grid-template-columns: 1fr;
      }
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
            <section id="state-flow-panel" class="state-flow-block" hidden>
              <div class="detail-title">
                <h3>状态图</h3>
                <div class="meta" id="state-flow-current"></div>
              </div>
              <div class="state-flow-summary" id="state-flow-summary"></div>
              <div class="state-flow-diagram" id="state-flow-diagram"></div>
              <section id="state-compare-block" class="state-compare-block" hidden>
                <div class="detail-title">
                  <h3>为什么不是另外两个状态</h3>
                  <div class="meta">看当前状态为什么压过其余两套持有/退出逻辑</div>
                </div>
                <div class="state-compare-grid" id="state-flow-compare"></div>
              </section>
            </section>
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
        if (block.type === 'table') {
          const wrap = document.createElement('div');
          wrap.className = 'markdown-table-wrap';
          const table = document.createElement('table');
          const thead = document.createElement('thead');
          const tbody = document.createElement('tbody');
          const headRow = document.createElement('tr');
          (block.headers || []).forEach((header) => {
            const th = document.createElement('th');
            th.textContent = header;
            headRow.appendChild(th);
          });
          thead.appendChild(headRow);
          (block.rows || []).forEach((row) => {
            const tr = document.createElement('tr');
            row.forEach((cell) => {
              const td = document.createElement('td');
              td.textContent = cell;
              tr.appendChild(td);
            });
            tbody.appendChild(tr);
          });
          table.appendChild(thead);
          table.appendChild(tbody);
          wrap.appendChild(table);
          host.appendChild(wrap);
          return;
        }
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
      const metrics = result.state_flow
        ? [
            ['买点日期', summary.start_date || '-'],
            ['扫描日期', result.scan_date || '-'],
            ['当前状态', summary.current_state_label || '-'],
            ['区间收益', summary.holding_return_pct === null || summary.holding_return_pct === undefined ? '-' : `${(Number(summary.holding_return_pct) * 100).toFixed(1)}%`],
          ]
        : [
            ['扫描日期', result.scan_date || '-'],
            ['候选数', summary.n_candidates ?? summary.n_resonance_candidates ?? result.candidate_count ?? 0],
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

    function renderStateFlow(result) {
      const panel = document.getElementById('state-flow-panel');
      const current = document.getElementById('state-flow-current');
      const summaryHost = document.getElementById('state-flow-summary');
      const diagram = document.getElementById('state-flow-diagram');
      const compareBlock = document.getElementById('state-compare-block');
      const compareHost = document.getElementById('state-flow-compare');
      const flow = result.state_flow;
      if (!flow || !flow.nodes || !flow.nodes.length) {
        panel.hidden = true;
        current.textContent = '';
        summaryHost.textContent = '';
        diagram.innerHTML = '';
        compareBlock.hidden = true;
        compareHost.innerHTML = '';
        return;
      }
      panel.hidden = false;
      current.textContent = `当前路径状态: ${flow.current_state_label || '-'}`;
      const summaryParts = [
        flow.start_date ? `买点: ${flow.start_date}` : '',
        flow.path_judgement_label ? `路径结论: ${flow.path_judgement_label}` : '',
        flow.current_reason || flow.current_advice || '',
        flow.path_transition_summary ? `路径切换: ${flow.path_transition_summary}` : '',
        flow.first_exit_date ? `首次退出确认: ${flow.first_exit_date}` : '',
      ].filter(Boolean);
      summaryHost.textContent = summaryParts.join(' ') || '当前状态图已更新。';
      diagram.innerHTML = '';
      const nodes = flow.nodes || [];
      const edges = flow.edges || [];
      nodes.forEach((node, index) => {
        const card = document.createElement('div');
        const pathActive = node.path_active ? ' path-active' : '';
        const badge = node.active ? '当前' : (node.path_active ? '路径经过' : '候选状态');
        card.className = 'state-node' + (node.active ? ' active' : '') + pathActive;
        card.innerHTML = `
          <div class="state-badge">${badge}</div>
          <div class="state-name">${node.state_label || '-'}</div>
          <div class="state-score">${fmt(node.state_score)}</div>
          <div class="state-reason">${node.reason || '-'}${node.first_entry_date ? ` 首次进入: ${node.first_entry_date}。` : ''}${node.days_in_state ? ` 路径内共 ${fmt(node.days_in_state)} 天。` : ''}</div>
        `;
        diagram.appendChild(card);
        if (index < edges.length) {
          const edge = edges[index];
          const arrow = document.createElement('div');
          arrow.className = 'flow-edge' + (edge.active ? ' active' : '');
          arrow.innerHTML = `
            <div class="flow-edge-label">${edge.label || ''}</div>
            <div class="flow-edge-rail"></div>
          `;
          diagram.appendChild(arrow);
        }
      });

      const comparisons = flow.comparisons || [];
      compareHost.innerHTML = '';
      compareBlock.hidden = comparisons.length === 0;
      comparisons.forEach((item) => {
        const card = document.createElement('div');
        card.className = 'state-compare-card';
        card.innerHTML = `
          <div class="state-compare-label">${item.state_label || '-'}</div>
          <div class="state-compare-score">候选分数: ${fmt(item.state_score)}${item.score_gap !== null && item.score_gap !== undefined ? ` · 与当前差值: ${fmt(item.score_gap)}` : ''}</div>
          <div class="state-compare-reason">${item.why_not || '-'}</div>
        `;
        compareHost.appendChild(card);
      });
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
      renderStateFlow(result);
      renderMetrics(result);
      const selectedRows = result.selected_rows || [];
      const candidateRows = result.candidate_rows || [];
      const summarySelectedColumns = ['股票名称', '股票代码', '选中原因', '操作建议'];
      const strategyId = (state.current && (state.current.base_id || state.current.id)) || '';
      const preferredSelectedColumns = strategyId.includes('entropy_bifurcation_setup')
        ? ['selected_rank', 'symbol', 'name', 'industry', 'market', 'strategy_score', 'entropy_quality', 'bifurcation_quality', 'trigger_quality', 'breakout_10', 'amount', 'turnover_rate']
        : ['selected_rank', 'symbol', 'name', 'industry', 'market', 'strategy_score', 'resonance_score', 'support_count', 'energy_term', 'amount', 'turnover_rate'];
      const preferredCandidateColumns = strategyId.includes('uptrend_hold_state_flow')
        ? ['state_label', 'active', 'activated_on_path', 'first_entry_date', 'last_entry_date', 'days_in_state', 'state_score', 'reason', 'entropy_reserve', 'disorder_pressure', 'expansion_thrust', 'directional_persistence', 'peak_extension_score', 'deceleration_score', 'fragility_score']
        : strategyId.includes('rapid_expansion_exhaustion_exit')
        ? ['symbol', 'name', 'start_date', 'scan_date', 'judgement', 'strategy_state', 'strategy_score', 'peak_extension_score', 'deceleration_score', 'fragility_score', 'first_exit_date', 'holding_return_pct']
        : strategyId.includes('rapid_expansion_hold')
        ? ['symbol', 'name', 'start_date', 'scan_date', 'judgement', 'strategy_state', 'strategy_score', 'expansion_thrust', 'acceptance_score', 'instability_risk', 'holding_return_pct']
        : strategyId.includes('entropy_hold_judgement')
        ? ['symbol', 'name', 'start_date', 'scan_date', 'judgement', 'strategy_state', 'strategy_score', 'disorder_pressure', 'first_exit_date', 'holding_return_pct']
        : strategyId.includes('entropy_bifurcation_setup')
        ? ['symbol', 'name', 'industry', 'market', 'strategy_score', 'entropy_quality', 'bifurcation_quality', 'trigger_quality', 'breakout_10', 'amount', 'turnover_rate']
        : ['symbol', 'name', 'industry', 'market', 'strategy_score', 'resonance_score', 'support_count', 'energy_term', 'amount', 'turnover_rate'];
      const selectedColumns = (selectedRows[0] ? summarySelectedColumns.filter((column) => column in selectedRows[0]) : []);
      const candidateColumns = (candidateRows[0] ? preferredCandidateColumns.filter((column) => column in candidateRows[0]) : []);
      document.getElementById('selected-count').textContent = `${selectedRows.length} rows`;
      document.getElementById('candidate-count').textContent = `${candidateRows.length} rows`;
      renderTable('selected-head', 'selected-body', selectedRows, selectedColumns.length ? selectedColumns : Object.keys(selectedRows[0] || {}).slice(0, 4));
      renderTable('candidate-head', 'candidate-body', candidateRows, candidateColumns.length ? candidateColumns : Object.keys(candidateRows[0] || {}).slice(0, 10));
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
      document.getElementById('state-flow-panel').hidden = true;
      document.getElementById('state-flow-summary').textContent = '';
      document.getElementById('state-flow-diagram').innerHTML = '';
      document.getElementById('state-compare-block').hidden = true;
      document.getElementById('state-flow-compare').innerHTML = '';
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

  def is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.count("|") >= 2 and not stripped.startswith("```")

  def split_table_line(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    return [cell.strip() for cell in stripped.split("|")]

  def is_separator_line(line: str) -> bool:
    cells = split_table_line(line)
    if not cells:
      return False
    return all(cell and set(cell) <= {":", "-", " "} for cell in cells)

  def flush() -> None:
    nonlocal current, mode
    if not current:
      return
    if mode == "ul":
      items = [line[2:].strip() for line in current if line.startswith("- ")]
      if items:
        blocks.append({"type": "ul", "items": items})
    elif mode == "table":
      lines = [line for line in current if line.strip()]
      if len(lines) >= 2 and is_separator_line(lines[1]):
        headers = split_table_line(lines[0])
        rows = [split_table_line(line) for line in lines[2:] if not is_separator_line(line)]
        if headers and rows:
          blocks.append({"type": "table", "headers": headers, "rows": rows})
      else:
        paragraph = " ".join(line.strip() for line in current if line.strip())
        if paragraph:
          blocks.append({"type": "p", "text": paragraph})
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
    if is_table_line(line):
      if mode not in {None, "table"}:
        flush()
      mode = "table"
      current.append(line)
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


def _variant_label(variant_name: str) -> str:
    return STRATEGY_VARIANT_LABELS.get(str(variant_name), str(variant_name))


def _variant_tagline(variant_name: str) -> str:
    return STRATEGY_VARIANT_TAGLINES.get(str(variant_name), str(variant_name))


def _strategy_label(strategy_id: str, fallback: str) -> str:
  return STRATEGY_LABELS.get(str(strategy_id), fallback)


def _strategy_tagline(strategy_id: str, fallback: str) -> str:
  return STRATEGY_TAGLINES.get(str(strategy_id), fallback)


def _with_variant_readme(readme: dict[str, Any], variant_name: str) -> dict[str, Any]:
    variant_label = _variant_label(variant_name)
    variant_tagline = _variant_tagline(variant_name)
    description_blocks = [{"type": "p", "text": variant_tagline}]
    description_blocks.extend(readme.get("description_blocks", []))
    return {
        **readme,
        "tagline": variant_tagline,
        "description_blocks": description_blocks,
        "variant_label": variant_label,
    }


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


def _strategy_variants(parameters: list[dict[str, Any]]) -> list[str]:
    for param in parameters:
        if param.get("dest") == "strategy_name":
            choices = [str(choice) for choice in param.get("choices") or [] if str(choice).strip()]
            if choices:
                return choices
    return []


def _discover_strategies() -> list[dict[str, Any]]:
    strategies: list[dict[str, Any]] = []
    if not STRATEGY_ROOT.exists():
        return strategies

    for strategy_dir in sorted(path for path in STRATEGY_ROOT.iterdir() if path.is_dir()):
        readme_path = strategy_dir / "README.md"
        if not readme_path.exists():
            continue
        relative_path = strategy_dir.relative_to(REPO_ROOT)
        entrypoints = sorted(strategy_dir.glob("run_*.py"))
        if not entrypoints:
            continue
        strategy_id = str(strategy_dir.relative_to(STRATEGY_ROOT)).replace("/", "-")
        readme = _extract_readme_metadata(readme_path)
        parameter_help = _parse_readme_parameter_help(readme["parameter_text"])
        parameters = _parameter_definitions(entrypoints[0], strategy_id, parameter_help)
        variants = _strategy_variants(parameters)
        if variants:
            visible_parameters = [param for param in parameters if param.get("dest") != "strategy_name"]
            for variant in variants:
                variant_id = f"{strategy_id}--{variant}"
                strategies.append(
                    {
                        "id": variant_id,
                        "base_id": strategy_id,
                        "display_name": f"{strategy_dir.name} / {_variant_label(variant)}",
                        "relative_path": f"{relative_path}#{variant}",
                        "directory": strategy_dir,
                        "readme_path": readme_path,
                        "entrypoint_path": entrypoints[0],
                        "readme": _with_variant_readme(readme, variant),
                        "parameters": visible_parameters,
                        "command_parameters": parameters,
                        "fixed_values": {"strategy_name": variant},
                    }
                )
            continue

        strategies.append(
          {
            "id": strategy_id,
            "base_id": strategy_id,
            "display_name": _strategy_label(strategy_id, strategy_dir.name),
            "relative_path": str(relative_path),
            "directory": strategy_dir,
            "readme_path": readme_path,
            "entrypoint_path": entrypoints[0],
            "readme": {**readme, "tagline": _strategy_tagline(strategy_id, readme.get("tagline", ""))},
            "parameters": parameters,
            "command_parameters": parameters,
            "fixed_values": {},
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
    summary_path = _find_existing(output_dir, [f"resonance_summary_{scan_date}.csv", f"strategy_summary_*_{scan_date}.csv"])
    selected_path = _find_existing(output_dir, [f"selected_portfolio_{scan_date}_top*.csv", f"selected_portfolio_*_{scan_date}_top*.csv"])
    candidates_path = _find_existing(output_dir, [f"resonance_candidates_{scan_date}_all.csv", f"*_candidates_{scan_date}_all.csv"])
    backtest_summary_path = _find_existing(output_dir, [f"forward_backtest_summary_{scan_date}.csv", f"forward_backtest_summary_*_{scan_date}.csv"])
    return {
        "summary": _read_csv(summary_path) if summary_path else pd.DataFrame(),
        "selected": _read_csv(selected_path) if selected_path else pd.DataFrame(),
        "candidates": _read_csv(candidates_path) if candidates_path else pd.DataFrame(),
        "backtest_summary": _read_csv(backtest_summary_path) if backtest_summary_path else pd.DataFrame(),
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
    "base_id": strategy.get("base_id", strategy["id"]),
        "display_name": strategy["display_name"],
        "relative_path": strategy["relative_path"],
        "tagline": strategy["readme"].get("tagline", ""),
    }


def _serialize_strategy_detail(strategy: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": strategy["id"],
    "base_id": strategy.get("base_id", strategy["id"]),
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
    effective_values = {**strategy.get("fixed_values", {}), **dict(values)}

    out_dir_value = effective_values.get("out_dir") or _default_value_for_dest(strategy["id"], "out_dir")
    effective_values["out_dir"] = out_dir_value
    output_dir = Path(str(out_dir_value)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for param in strategy.get("command_parameters", strategy["parameters"]):
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


def _fmt_pct(value: Any, digits: int = 1) -> str:
  try:
    number = float(value)
  except (TypeError, ValueError):
    return "-"
  if pd.isna(number):
    return "-"
  return f"{number * 100.0:.{digits}f}%"


def _fmt_num(value: Any, digits: int = 2) -> str:
  try:
    number = float(value)
  except (TypeError, ValueError):
    return "-"
  if pd.isna(number):
    return "-"
  return f"{number:.{digits}f}"


def _strategy_variant(strategy: dict[str, Any]) -> str:
  fixed_values = strategy.get("fixed_values", {})
  return str(fixed_values.get("strategy_name") or strategy.get("base_id") or strategy.get("id") or "")


STATE_FLOW_ORDER = [
  "observation",
  "entropy_hold_judgement",
  "rapid_expansion_hold",
  "rapid_expansion_exhaustion_exit",
]
STATE_FLOW_EDGES = [
  ("observation", "entropy_hold_judgement", "低熵重组"),
  ("entropy_hold_judgement", "rapid_expansion_hold", "扩张推力建立"),
  ("rapid_expansion_hold", "rapid_expansion_exhaustion_exit", "高位降速与脆弱化"),
]


def _float_or_none(value: Any) -> float | None:
  try:
    number = float(value)
  except (TypeError, ValueError):
    return None
  if pd.isna(number):
    return None
  return number


def _state_flow_why_not(target_state_id: str, row: dict[str, Any], current_state_id: str) -> str:
  score_text = _fmt_num(row.get("state_score"))
  if target_state_id == "entropy_hold_judgement":
    return (
      f"不是“熵秩序持有”，因为低熵储备 {_fmt_num(row.get('entropy_reserve'))} 偏低，"
      f"乱序压力 {_fmt_num(row.get('disorder_pressure'))} 偏高；这说明结构已经不再是稳定低熵段，"
      f"该状态分数只有 {score_text}。"
    )
  if target_state_id == "rapid_expansion_hold":
    if current_state_id == "rapid_expansion_exhaustion_exit":
      return (
        f"不是“快速扩张持有”，因为虽然扩张推力 {_fmt_num(row.get('expansion_thrust'))} 仍在，"
        f"但当前已经从纯扩张推进转向末端降速与脆弱化，单看扩张持有分数只有 {score_text}。"
      )
    return (
      f"不是“快速扩张持有”，因为扩张推力 {_fmt_num(row.get('expansion_thrust'))}、"
      f"方向持续性 {_fmt_num(row.get('directional_persistence'))} 或承接强度 {_fmt_num(row.get('acceptance_score'))} 还不足以接管当前结构；"
      f"该状态分数只有 {score_text}。"
    )
  if target_state_id == "rapid_expansion_exhaustion_exit":
    if current_state_id == "rapid_expansion_hold":
      return (
        f"不是“快速扩张衰竭退出”，因为虽然高位扩张分数 {_fmt_num(row.get('peak_extension_score'))} 不低，"
        f"但降速分数 {_fmt_num(row.get('deceleration_score'))} 和脆弱度 {_fmt_num(row.get('fragility_score'))} 还没达到衰竭确认。"
      )
    return (
      f"不是“快速扩张衰竭退出”，因为高位扩张分数 {_fmt_num(row.get('peak_extension_score'))}、"
      f"降速分数 {_fmt_num(row.get('deceleration_score'))}、脆弱度 {_fmt_num(row.get('fragility_score'))} 还没有同时形成退出共振；"
      f"该状态分数只有 {score_text}。"
    )
  return str(row.get("reason") or "当前证据还不足以支持这个状态。")


def _state_flow_comparisons(current_state_id: str, candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
  node_map = {str(row.get("state_id")): row for row in candidate_rows if row.get("state_id")}
  current_score = _float_or_none(node_map.get(current_state_id, {}).get("state_score"))
  compare_ids = [
    state_id
    for state_id in ["entropy_hold_judgement", "rapid_expansion_hold", "rapid_expansion_exhaustion_exit"]
    if state_id != current_state_id and state_id in node_map
  ]
  comparisons: list[dict[str, Any]] = []
  for state_id in compare_ids:
    row = node_map[state_id]
    score = _float_or_none(row.get("state_score"))
    comparisons.append(
      {
        "state_id": state_id,
        "state_label": row.get("state_label") or state_id,
        "state_score": row.get("state_score"),
        "score_gap": None if score is None or current_score is None else current_score - score,
        "why_not": _state_flow_why_not(state_id, row, current_state_id),
      }
    )
  return comparisons


def _state_flow_payload(summary: dict[str, Any], candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
  node_map = {str(row.get("state_id")): dict(row) for row in candidate_rows if row.get("state_id")}
  current_state_id = str(summary.get("current_state_id") or "")
  current_index = STATE_FLOW_ORDER.index(current_state_id) if current_state_id in STATE_FLOW_ORDER else 0
  has_path_activation = any("activated_on_path" in row for row in candidate_rows)

  def _row_path_active(row: dict[str, Any], index: int) -> bool:
    if has_path_activation:
      return bool(row.get("activated_on_path"))
    return index <= current_index

  ordered_nodes: list[dict[str, Any]] = []
  for index, state_id in enumerate(STATE_FLOW_ORDER):
    row = node_map.get(state_id, {"state_id": state_id, "state_label": state_id, "state_score": None, "reason": ""})
    row["path_active"] = _row_path_active(row, index)
    ordered_nodes.append(row)

  return {
    "start_date": summary.get("start_date"),
    "current_state_id": current_state_id,
    "current_state_label": summary.get("current_state_label"),
    "current_reason": summary.get("current_state_reason"),
    "current_advice": summary.get("state_advice"),
    "path_judgement": summary.get("path_judgement"),
    "path_judgement_label": summary.get("path_judgement_label"),
    "first_exit_date": summary.get("first_exit_date"),
    "path_transition_summary": summary.get("path_transition_summary"),
    "nodes": ordered_nodes,
    "edges": [
      {
        "from": source,
        "to": target,
        "label": label,
        "active": (
          bool(node_map.get(source, {}).get("activated_on_path"))
          and bool(node_map.get(target, {}).get("activated_on_path"))
          and (
            not node_map.get(source, {}).get("first_entry_date")
            or not node_map.get(target, {}).get("first_entry_date")
            or str(node_map.get(source, {}).get("first_entry_date")) <= str(node_map.get(target, {}).get("first_entry_date"))
          )
        ) if has_path_activation else (current_state_id in STATE_FLOW_ORDER and current_index >= STATE_FLOW_ORDER.index(target)),
      }
      for source, target, label in STATE_FLOW_EDGES
    ],
    "comparisons": _state_flow_comparisons(current_state_id, candidate_rows),
  }


def _selected_reason_text(row: dict[str, Any], strategy: dict[str, Any]) -> str:
  variant = _strategy_variant(strategy)
  if variant == "compression_breakout":
    return (
      f"波动压缩后开始突破，20日突破幅度 {_fmt_pct(row.get('breakout_20'))}，"
      f"压缩强度 {_fmt_num(row.get('compression_z_120'))}，策略分数 {_fmt_num(row.get('strategy_score'))}。"
    )
  if variant == "self_organized_trend":
    return (
      f"趋势结构已成型，20日收益 {_fmt_pct(row.get('ret_20'))}，60日收益 {_fmt_pct(row.get('ret_60'))}，"
      f"策略分数 {_fmt_num(row.get('strategy_score'))}。"
    )
  if variant == "fractal_pullback":
    return (
      f"母趋势中完成回踩并出现再启动，回踩深度 {_fmt_pct(row.get('pullback_depth_8'))}，"
      f"策略分数 {_fmt_num(row.get('strategy_score'))}。"
    )
  if variant == "market_energy_flow":
    return (
      f"资金与成交活跃度同步增强，20日收益 {_fmt_pct(row.get('ret_20'))}，"
      f"能量脉冲 {_fmt_num(row.get('energy_impulse'))}，策略分数 {_fmt_num(row.get('strategy_score'))}。"
    )
  if variant == "entropy_bifurcation_setup":
    return (
      f"当前处于低熵压缩后启动区，熵质量 {_fmt_num(row.get('entropy_quality'))}，"
      f"分叉质量 {_fmt_num(row.get('bifurcation_quality'))}，触发质量 {_fmt_num(row.get('trigger_quality'))}。"
    )
  if variant == "entropy_hold_judgement":
    if bool(row.get("strategy_state")):
      return (
        f"自 {row.get('start_date') or '-'} 以来，未出现持续性的高熵乱序与动力记忆坍缩，"
        f"当前持有分数 {_fmt_num(row.get('hold_score') or row.get('strategy_score'))}，"
        f"乱序压力 {_fmt_num(row.get('disorder_pressure'))}。"
      )
    return (
      f"自 {row.get('start_date') or '-'} 起在 {row.get('first_exit_date') or '未知日期'} 出现持续性的高熵乱序与记忆坍缩，"
      f"当前持有分数 {_fmt_num(row.get('hold_score') or row.get('strategy_score'))}，"
      f"乱序压力 {_fmt_num(row.get('disorder_pressure'))}。"
    )
  if variant == "rapid_expansion_hold":
    if bool(row.get("strategy_state")):
      return (
        f"自 {row.get('start_date') or '-'} 以来仍处于快速扩张上行段，"
        f"扩张推力 {_fmt_num(row.get('expansion_thrust'))}，方向持续性 {_fmt_num(row.get('directional_persistence'))}，"
        f"风险强度 {_fmt_num(row.get('instability_risk'))}。"
      )
    return (
      f"自 {row.get('start_date') or '-'} 起在 {row.get('first_exit_date') or '未知日期'} 出现扩张推力衰减与风险过载，"
      f"扩张推力 {_fmt_num(row.get('expansion_thrust'))}，风险强度 {_fmt_num(row.get('instability_risk'))}。"
    )
  if variant == "rapid_expansion_exhaustion_exit":
    if bool(row.get("strategy_state")):
      return (
        f"当前虽处于强扩张后段，但尚未形成衰竭退出确认，"
        f"高位扩张分数 {_fmt_num(row.get('peak_extension_score'))}，降速分数 {_fmt_num(row.get('deceleration_score'))}，"
        f"脆弱度 {_fmt_num(row.get('fragility_score'))}。"
      )
    return (
      f"自 {row.get('start_date') or '-'} 起在 {row.get('first_exit_date') or '未知日期'} 进入快速扩张末端衰竭区，"
      f"高位扩张分数 {_fmt_num(row.get('peak_extension_score'))}，降速分数 {_fmt_num(row.get('deceleration_score'))}，"
      f"脆弱度 {_fmt_num(row.get('fragility_score'))}。"
    )
  if variant == "uptrend_hold_state_flow":
    start_date = row.get("start_date") or "-"
    current_state = row.get("current_state_label") or "-"
    reason = row.get("current_state_reason") or current_state
    transitions = row.get("path_transition_summary") or ""
    transition_text = f" 状态切换: {transitions}。" if transitions else ""
    return f"自 {start_date} 买入后，路径当前处于 {current_state}；{reason}。{transition_text}"
  if row.get("resonance_score") is not None:
    return f"多周期共振分数 {_fmt_num(row.get('resonance_score'))}，结构与能量项同时满足入选条件。"
  return f"策略分数 {_fmt_num(row.get('strategy_score'))}，满足当前策略的核心筛选条件。"


def _selected_advice_text(row: dict[str, Any], strategy: dict[str, Any], values: dict[str, Any]) -> str:
  hold_days = values.get("hold_days")
  if hold_days in {None, ""}:
    hold_days = 5
  try:
    hold_days_int = int(hold_days)
  except (TypeError, ValueError):
    hold_days_int = 5

  variant = _strategy_variant(strategy)
  base = f"建议以 {hold_days_int} 个交易日作为基础持有周期。"
  if variant == "entropy_bifurcation_setup":
    return base + " 若突破动能回落，或重新跌回 20 日线下方，可提前止盈/止损。"
  if variant == "entropy_hold_judgement":
    if bool(row.get("strategy_state")):
      return base + " 当前可继续持有；只有当高熵乱序连续积累且动力记忆持续坍缩时，再考虑退出。"
    return "当前不建议按这套熵持有逻辑继续一致持有；更合理的是等待新的低熵重组后再评估。"
  if variant == "rapid_expansion_hold":
    if bool(row.get("strategy_state")):
      return base + " 当前仍属于快速扩张持有区；重点盯住扩张推力是否明显衰减，以及风险强度是否持续高于承接强度。"
    return "当前不建议继续按快速扩张逻辑持有；更合理的是等待推力重新建立，或回到新的整理重组后再评估。"
  if variant == "rapid_expansion_exhaustion_exit":
    if bool(row.get("strategy_state")):
      return base + " 当前尚未确认末端衰竭退出；重点观察推力回落是否继续扩大，并留意承接与方向连续性是否同步转弱。"
    return "当前已出现快速扩张末端的降速退出信号；更合理的是主动降仓或退出，而不是再按加速段逻辑继续持有。"
  if variant == "uptrend_hold_state_flow":
    prefix = str(row.get("path_judgement_label") or "路径结论已更新")
    return f"{prefix}。{str(row.get('state_advice') or '根据当前路径状态继续跟踪相邻状态的切换条件。')}"
  if variant == "compression_breakout":
    return base + " 若突破次日不能延续放量，或重新回到整理区间，可考虑提前退出。"
  if variant == "self_organized_trend":
    return base + " 若 20 日线转弱或趋势斜率明显放缓，可分批止盈。"
  if variant == "fractal_pullback":
    return base + " 若再启动失败并跌破短期整理低点，可优先离场。"
  if variant == "market_energy_flow":
    return base + " 若资金强度回落且相对强度掉队，可降低仓位或退出。"
  return base + " 若核心触发条件消失，可提前结束持有。"


def _selected_display_rows(selected_rows: list[dict[str, Any]], strategy: dict[str, Any], values: dict[str, Any]) -> list[dict[str, Any]]:
  display_rows: list[dict[str, Any]] = []
  for row in selected_rows:
    display_rows.append(
      {
        "股票名称": row.get("name") or row.get("ts_code") or row.get("symbol") or "-",
        "股票代码": row.get("symbol") or row.get("ts_code") or "-",
        "选中原因": _selected_reason_text(row, strategy),
        "操作建议": _selected_advice_text(row, strategy, values),
      }
    )
  return display_rows


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
    raw_selected_rows = _normalize_rows(bundle.get("selected", pd.DataFrame()), 100)
    selected_rows = _selected_display_rows(raw_selected_rows, strategy, values)
    candidate_rows = _normalize_rows(bundle.get("candidates", pd.DataFrame()), 100)
    return {
        "ok": True,
        "strategy_id": strategy.get("id"),
        "base_id": strategy.get("base_id", strategy.get("id")),
        "scan_date": scan_date,
        "summary": summary_rows[0] if summary_rows else {},
        "backtest_summary": backtest_summary_rows[0] if backtest_summary_rows else {},
        "selected_rows": selected_rows,
        "candidate_rows": candidate_rows,
        "state_flow": _state_flow_payload(summary_rows[0], candidate_rows) if strategy.get("base_id") == "uptrend_hold_state_flow" and summary_rows else None,
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
