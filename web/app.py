import argparse
import importlib
import logging
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
  "continuous_decline_recovery": "连续下跌修复买点",
  "entropy_accumulation_breakout": "熵惜售分岔突破",
}
STRATEGY_TAGLINES = {
  "entropy_bifurcation_setup": "用市场门控、低熵压缩、分叉触发和分段执行去筛选启动股。",
  "uptrend_hold_state_flow": "把上升趋势里的熵秩序持有、快速扩张持有、快速扩张衰竭退出放进一张状态图里，从买点开始评估整段持有路径。",
  "continuous_decline_recovery": "在连续下跌后的第一段修复里，先定市场买点，再排领先行业，最后选最先完成修复的股票。",
  "entropy_accumulation_breakout": "基于信息熵、路径不可逆性、临界减速与量子相干性的三阶段系统：惜售吸筹（叠加态）→ 分岔突破（坍缩）→ 结构崩塌（退相干）。纯理论驱动，无传统技术指标。",
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
MARKET_PHASE_LABELS = {
  "compression": "低熵压缩",
  "transition": "临界过渡",
  "expansion": "启动扩张",
  "distorted": "相位失真",
  "neutral": "中性观察",
  "abandon": "放弃交易",
}
ENTRY_MODE_LABELS = {
  "skip": "跳过",
  "probe": "试探建仓",
  "staged": "分段建仓",
  "full": "直接建仓",
}
EXIT_MODE_LABELS = {
  "abandon": "直接放弃",
  "reduce": "减仓观察",
  "trail": "跟踪退出",
}
EXECUTION_STATE_LABELS = {
  "normal": "执行正常",
  "cautious": "谨慎执行",
  "blocked": "执行阻断",
}
MARKET_BUY_STATE_LABELS = {
  "no_setup": "没有连续下跌前提",
  "selloff": "下跌仍在继续",
  "repair_watch": "修复观察期",
  "buy_window": "修复买入窗口",
  "rebound_crowded": "反弹过热期",
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

    .param-timeline-block {
      margin-top: 18px;
      padding: 16px;
      border-radius: 20px;
      border: 1px solid rgba(22, 32, 36, 0.08);
      background: rgba(255,255,255,0.42);
    }

    .param-timeline-summary {
      margin: 10px 0 16px;
      padding: 14px 16px;
      border-radius: 18px;
      background: rgba(15, 118, 110, 0.08);
      border: 1px solid rgba(15, 118, 110, 0.10);
      color: var(--ink);
      line-height: 1.7;
      font-size: 14px;
    }

    .param-timeline-lanes {
      display: grid;
      gap: 16px;
    }

    .param-timeline-lane {
      display: grid;
      grid-template-columns: 132px minmax(0, 1fr);
      gap: 14px;
      align-items: start;
    }

    .param-timeline-label {
      padding-top: 6px;
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }

    .param-timeline-track {
      position: relative;
      min-height: 88px;
      padding-top: 10px;
      padding-bottom: 26px;
    }

    .param-timeline-rail {
      position: absolute;
      left: 0;
      right: 0;
      top: 38px;
      height: 2px;
      border-radius: 999px;
      background: rgba(22, 32, 36, 0.18);
    }

    .param-timeline-window {
      position: absolute;
      top: 32px;
      height: 14px;
      border-radius: 999px;
      background: rgba(15, 118, 110, 0.12);
      border: 1px solid rgba(15, 118, 110, 0.18);
    }

    .param-timeline-hold {
      position: absolute;
      top: 32px;
      height: 14px;
      border-radius: 999px;
      background: linear-gradient(90deg, rgba(180, 83, 9, 0.16), rgba(15, 118, 110, 0.28));
    }

    .param-timeline-hold::after {
      content: "";
      position: absolute;
      right: 1px;
      top: 50%;
      width: 10px;
      height: 10px;
      border-top: 2px solid rgba(180, 83, 9, 0.84);
      border-right: 2px solid rgba(180, 83, 9, 0.84);
      transform: translateY(-50%) rotate(45deg);
    }

    .param-timeline-marker {
      position: absolute;
      top: 0;
      transform: translateX(-50%);
      width: max-content;
      max-width: 132px;
      text-align: center;
    }

    .param-timeline-dot {
      display: block;
      width: 11px;
      height: 11px;
      margin: 0 auto 8px;
      border-radius: 50%;
      border: 2px solid rgba(15, 118, 110, 0.72);
      background: rgba(255,255,255,0.88);
      box-shadow: 0 0 0 4px rgba(15, 118, 110, 0.10);
    }

    .param-timeline-marker.scan .param-timeline-dot {
      background: var(--accent);
      border-color: var(--accent);
      box-shadow: 0 0 0 4px rgba(15, 118, 110, 0.16);
    }

    .param-timeline-marker.hold .param-timeline-dot {
      background: var(--warm);
      border-color: var(--warm);
      box-shadow: 0 0 0 4px rgba(180, 83, 9, 0.10);
    }

    .param-timeline-marker.muted .param-timeline-dot {
      background: rgba(255,255,255,0.54);
      border-color: rgba(22, 32, 36, 0.24);
      box-shadow: none;
    }

    .param-timeline-marker-label {
      font-size: 11px;
      color: var(--muted);
      line-height: 1.4;
    }

    .param-timeline-marker-value {
      margin-top: 2px;
      font-size: 13px;
      font-weight: 700;
      color: var(--ink);
      line-height: 1.45;
    }

    .param-timeline-track-note {
      position: absolute;
      left: 0;
      right: 0;
      bottom: 0;
      font-size: 12px;
      line-height: 1.55;
      color: var(--muted);
    }

    .param-timeline-track-note.warning {
      color: var(--bad);
    }

    .param-timeline-notes {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-top: 16px;
    }

    .param-timeline-note {
      padding: 14px;
      border-radius: 16px;
      border: 1px solid rgba(22, 32, 36, 0.08);
      background: rgba(255,255,255,0.48);
    }

    .param-timeline-note-label {
      font-size: 11px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 6px;
    }

    .param-timeline-note-value {
      font-size: 18px;
      font-weight: 700;
      color: var(--ink);
      margin-bottom: 6px;
      line-height: 1.2;
    }

    .param-timeline-note-detail {
      font-size: 13px;
      line-height: 1.6;
      color: var(--muted);
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
      .param-timeline-notes {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
      .form-grid,
      .summary-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }

    @media (max-width: 760px) {
      .shell { padding: 16px 12px 28px; }
      .form-grid,
      .summary-grid { grid-template-columns: 1fr; }
      .param-timeline-lane,
      .param-timeline-notes {
        grid-template-columns: 1fr;
      }
      .param-timeline-label {
        padding-top: 0;
      }
      .param-timeline-track {
        min-height: 104px;
      }
      .hero { padding: 20px 18px; }
      .content { padding: 16px; }
    }
  </style>
  <style>
    :root {
      color-scheme: dark;
      --paper: #09131a;
      --paper-strong: #10212a;
      --ink: #e9f4ef;
      --muted: #93a9b2;
      --line: rgba(121, 164, 173, 0.18);
      --line-strong: rgba(121, 164, 173, 0.32);
      --card: linear-gradient(180deg, rgba(11, 19, 26, 0.96), rgba(6, 12, 17, 0.88));
      --panel-soft: rgba(11, 19, 26, 0.78);
      --accent: #2dd4bf;
      --accent-strong: #9af2dd;
      --accent-soft: rgba(45, 212, 191, 0.14);
      --warm: #f6b85f;
      --good: #5be38c;
      --bad: #ff7a7a;
      --shadow: 0 28px 90px rgba(0, 0, 0, 0.42);
      --code-bg: #081118;
      --code-ink: #d8f7ef;
      --table-head: #12202a;
    }

    body {
      color: var(--ink);
      background:
        radial-gradient(circle at 14% 18%, rgba(246, 184, 95, 0.13), transparent 18%),
        radial-gradient(circle at 86% 8%, rgba(45, 212, 191, 0.12), transparent 22%),
        radial-gradient(circle at 50% 120%, rgba(89, 132, 255, 0.10), transparent 32%),
        linear-gradient(180deg, #071017 0%, #09141b 42%, #061017 100%);
      font-family: "IBM Plex Serif", "Noto Serif SC", Georgia, serif;
    }

    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background:
        linear-gradient(rgba(154, 242, 221, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(154, 242, 221, 0.03) 1px, transparent 1px);
      background-size: 34px 34px;
      mask-image: linear-gradient(180deg, rgba(0, 0, 0, 0.42), transparent 92%);
      opacity: 0.45;
    }

    .shell {
      position: relative;
      z-index: 1;
    }

    .hero,
    .panel {
      background: var(--card);
      border-color: var(--line);
      box-shadow: var(--shadow);
    }

    .hero::after {
      background: radial-gradient(circle, rgba(45, 212, 191, 0.18), transparent 70%);
      filter: blur(2px);
      animation: driftGlow 14s ease-in-out infinite alternate;
    }

    .eyebrow {
      color: var(--warm);
    }

    .hero-copy,
    .strategy-summary,
    .detail-title .meta,
    .sidebar-head span,
    .field label,
    .field small,
    .run-status,
    .metric-label,
    .state-reason,
    .flow-edge-label,
    .state-compare-reason,
    .muted {
      color: var(--muted);
    }

    .strategy-path {
      color: var(--warm);
    }

    code {
      padding: 2px 7px;
      border-radius: 999px;
      background: rgba(45, 212, 191, 0.12);
      color: var(--accent-strong);
    }

    .pill {
      background: rgba(8, 16, 22, 0.72);
      border-color: rgba(45, 212, 191, 0.18);
      color: var(--ink);
    }

    .strategy-card,
    .detail-block,
    .result-block,
    .state-flow-block,
    .state-compare-card,
    .metric,
    .field,
    .param-timeline-block,
    .param-timeline-note,
    .state-node,
    .table-wrap,
    .markdown .markdown-table-wrap,
    .empty-state,
    .focus-metric,
    .focus-lead {
      background: var(--panel-soft);
      border-color: var(--line);
    }

    .strategy-card:hover {
      border-color: rgba(45, 212, 191, 0.34);
      background: rgba(15, 27, 35, 0.94);
    }

    .strategy-card.active {
      background: linear-gradient(135deg, rgba(45, 212, 191, 0.16), rgba(15, 27, 35, 0.96));
      border-color: rgba(45, 212, 191, 0.42);
    }

    .block-label,
    .state-badge {
      color: var(--accent-strong);
      background: rgba(45, 212, 191, 0.12);
    }

    .markdown p {
      color: var(--ink);
    }

    .markdown ul {
      color: var(--muted);
    }

    input,
    select,
    textarea {
      color: var(--ink);
      background: rgba(5, 12, 17, 0.88);
      border-color: var(--line-strong);
    }

    input::placeholder,
    textarea::placeholder {
      color: rgba(147, 169, 178, 0.72);
    }

    .checkbox-wrap input {
      accent-color: var(--accent);
    }

    .btn-primary {
      background: linear-gradient(135deg, #169f90, #2dd4bf);
      color: #041014;
      box-shadow: 0 10px 24px rgba(45, 212, 191, 0.22);
    }

    .btn-secondary {
      background: rgba(10, 18, 24, 0.92);
      color: var(--ink);
      border: 1px solid var(--line);
    }

    .state-flow-summary,
    .strategy-focus-summary,
    .param-timeline-summary {
      background: rgba(45, 212, 191, 0.08);
      border: 1px solid rgba(45, 212, 191, 0.12);
      color: var(--ink);
    }

    .param-timeline-rail {
      background: rgba(121, 164, 173, 0.24);
    }

    .param-timeline-window {
      background: rgba(45, 212, 191, 0.12);
      border-color: rgba(45, 212, 191, 0.24);
    }

    .param-timeline-hold {
      background: linear-gradient(90deg, rgba(246, 184, 95, 0.18), rgba(45, 212, 191, 0.34));
    }

    .param-timeline-hold::after {
      border-top-color: rgba(246, 184, 95, 0.92);
      border-right-color: rgba(246, 184, 95, 0.92);
    }

    .param-timeline-marker.boundary .param-timeline-dot {
      background: rgba(8, 16, 22, 0.88);
      border-color: rgba(154, 242, 221, 0.68);
      box-shadow: 0 0 0 4px rgba(154, 242, 221, 0.08);
    }

    .param-timeline-marker.muted .param-timeline-dot {
      background: rgba(8, 16, 22, 0.62);
      border-color: rgba(121, 164, 173, 0.34);
    }

    .state-node.active {
      border-color: rgba(45, 212, 191, 0.46);
      background: linear-gradient(135deg, rgba(45, 212, 191, 0.16), rgba(15, 27, 35, 0.96));
      box-shadow: inset 0 0 0 1px rgba(45, 212, 191, 0.16);
    }

    .state-node.path-active:not(.active) {
      border-color: rgba(45, 212, 191, 0.22);
      background: linear-gradient(180deg, rgba(45, 212, 191, 0.07), rgba(15, 27, 35, 0.9));
    }

    .flow-edge-rail::before {
      background: rgba(121, 164, 173, 0.24);
    }

    .flow-edge-rail::after {
      border-top-color: rgba(121, 164, 173, 0.24);
      border-right-color: rgba(121, 164, 173, 0.24);
    }

    .flow-edge.active .flow-edge-rail::before {
      background: linear-gradient(90deg, rgba(45, 212, 191, 0.45), rgba(45, 212, 191, 0.95));
    }

    .flow-edge.active .flow-edge-rail::after {
      border-top-color: rgba(45, 212, 191, 0.95);
      border-right-color: rgba(45, 212, 191, 0.95);
    }

    table {
      color: var(--ink);
    }

    th,
    td {
      border-bottom-color: rgba(121, 164, 173, 0.14);
    }

    td {
      white-space: normal;
    }

    thead th {
      background: var(--table-head);
    }

    pre {
      background: var(--code-bg);
      color: var(--code-ink);
      border: 1px solid rgba(45, 212, 191, 0.14);
    }

    .strategy-focus-block {
      margin-bottom: 18px;
    }

    .strategy-focus-summary {
      margin: 10px 0 16px;
      padding: 14px 16px;
      border-radius: 18px;
      line-height: 1.72;
      font-size: 14px;
    }

    .strategy-focus-grid,
    .focus-lead-kv {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }

    .focus-metric,
    .focus-lead {
      padding: 16px;
      border-radius: 18px;
      border: 1px solid var(--line);
    }

    .focus-metric-label,
    .focus-chip-label {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
    }

    .focus-metric-value,
    .focus-chip-value {
      font-size: 24px;
      font-weight: 700;
      line-height: 1.05;
      margin-bottom: 8px;
    }

    .focus-metric-detail {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.6;
    }

    .focus-lead {
      margin-top: 16px;
      background: linear-gradient(180deg, rgba(15, 27, 35, 0.96), rgba(8, 14, 20, 0.86));
    }

    .focus-lead-head {
      margin-bottom: 14px;
    }

    .focus-lead-head h3 {
      margin: 0 0 6px;
    }

    .focus-chip {
      padding: 12px 14px;
      border-radius: 16px;
      border: 1px solid rgba(121, 164, 173, 0.14);
      background: rgba(7, 13, 19, 0.76);
    }

    @keyframes driftGlow {
      0% {
        transform: translate3d(0, 0, 0) scale(1);
        opacity: 0.85;
      }
      100% {
        transform: translate3d(-10px, 12px, 0) scale(1.08);
        opacity: 1;
      }
    }

    @media (max-width: 1220px) {
      .strategy-focus-grid,
      .focus-lead-kv {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }

    @media (max-width: 760px) {
      .strategy-focus-grid,
      .focus-lead-kv {
        grid-template-columns: 1fr;
      }
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
        <a href="/quantum" class="pill" style="text-decoration:none;background:rgba(45,212,191,0.18);border-color:rgba(45,212,191,0.36);cursor:pointer;">⚛ 量子态扫描</a>
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
              <section id="parameter-timeline-panel" class="param-timeline-block" hidden>
                <div class="detail-title">
                  <div>
                    <div class="block-label">Timeline</div>
                    <h3>参数时间线</h3>
                  </div>
                  <div class="meta" id="parameter-timeline-meta"></div>
                </div>
                <div class="param-timeline-summary" id="parameter-timeline-summary"></div>
                <div class="param-timeline-lanes">
                  <div class="param-timeline-lane">
                    <div class="param-timeline-label">前瞻回测窗口</div>
                    <div class="param-timeline-track" id="parameter-window-track"></div>
                  </div>
                  <div class="param-timeline-lane">
                    <div class="param-timeline-label">单次信号持有</div>
                    <div class="param-timeline-track" id="parameter-hold-track"></div>
                  </div>
                </div>
                <div class="param-timeline-notes" id="parameter-timeline-notes"></div>
              </section>
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
            <section id="strategy-focus-panel" class="result-block strategy-focus-block" hidden>
              <div class="detail-title">
                <div>
                  <h3 id="strategy-focus-title">策略画像</h3>
                  <div class="meta" id="strategy-focus-meta"></div>
                </div>
              </div>
              <div class="strategy-focus-summary" id="strategy-focus-summary"></div>
              <div class="strategy-focus-grid" id="strategy-focus-grid"></div>
              <section id="strategy-focus-lead" class="focus-lead" hidden>
                <div class="focus-lead-head">
                  <h3 id="focus-lead-title"></h3>
                  <div class="meta" id="focus-lead-meta"></div>
                </div>
                <div class="focus-lead-kv" id="focus-lead-kv"></div>
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
      if (typeof value === 'boolean') return value ? '是' : '否';
      if (typeof value === 'number') {
        if (!Number.isFinite(value)) return '-';
        if (Math.abs(value) >= 1000) return value.toLocaleString('zh-CN', { maximumFractionDigits: 2 });
        return value.toLocaleString('zh-CN', { maximumFractionDigits: 4 });
      }
      return String(value);
    }

    function currentStrategyId() {
      return (state.current && (state.current.base_id || state.current.id)) || '';
    }

    function isEntropyStrategy(strategyId = currentStrategyId()) {
      return String(strategyId).includes('entropy_bifurcation_setup');
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
        if (param.hidden) return;
        const wrapper = document.createElement('div');
        wrapper.className = 'field';

        const label = document.createElement('label');
        label.setAttribute('for', `param-${param.dest}`);
        const labelParts = [];
        if (param.help_text) labelParts.push(param.help_text);
        else labelParts.push(param.label);
        if (param.required) labelParts.push(' *');
        label.textContent = labelParts.join('');
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
          textarea.placeholder = param.default_display ? `默认: ${param.default_display}` : '';
          wrapper.appendChild(textarea);
        } else {
          const input = document.createElement('input');
          input.id = `param-${param.dest}`;
          input.name = param.dest;
          input.type = inputTypeFor(param);
          if (param.kind === 'integer') input.step = '1';
          if (param.kind === 'float') input.step = '0.01';
          input.value = param.value == null ? '' : String(param.value);
          input.placeholder = param.default_display ? `默认: ${param.default_display}` : '';
          wrapper.appendChild(input);
        }

        const helpParts = [];
        if (param.default_display) helpParts.push(`默认值: ${param.default_display}`);
        helpParts.push(param.label);

        const help = document.createElement('small');
        help.textContent = helpParts.join(' · ');
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

    function hasParameterTimeline(detail) {
      if (!detail) return false;
      const strategyId = (detail.base_id || detail.id || '');
      if (!isEntropyStrategy(strategyId)) return false;
      const dests = new Set((detail.parameters || []).map((param) => param.dest));
      return dests.has('scan_date') && dests.has('hold_days');
    }

    function normalizeCompactDate(value) {
      const text = String(value == null ? '' : value).trim();
      if (!text) return '';
      const digits = text.replace(/-/g, '');
      return /^\\d{8}$/.test(digits) ? digits : '';
    }

    function parseCompactDate(value) {
      const digits = normalizeCompactDate(value);
      if (!digits) return null;
      const year = Number.parseInt(digits.slice(0, 4), 10);
      const month = Number.parseInt(digits.slice(4, 6), 10);
      const day = Number.parseInt(digits.slice(6, 8), 10);
      const date = new Date(Date.UTC(year, month - 1, day));
      if (
        Number.isNaN(date.getTime())
        || date.getUTCFullYear() !== year
        || date.getUTCMonth() !== month - 1
        || date.getUTCDate() !== day
      ) {
        return null;
      }
      return { digits, date };
    }

    function formatCompactDate(value) {
      const digits = typeof value === 'string' ? normalizeCompactDate(value) : ((value && value.digits) || '');
      if (!digits) return '-';
      return `${digits.slice(0, 4)}-${digits.slice(4, 6)}-${digits.slice(6, 8)}`;
    }

    function normalizeHoldDays(value) {
      const parsed = Number.parseInt(String(value == null ? '' : value).trim(), 10);
      if (!Number.isFinite(parsed) || parsed <= 0) return 5;
      return parsed;
    }

    function timelineMarkerPosition(percent) {
      const bounded = Math.max(5, Math.min(95, Number(percent) || 50));
      return Number(bounded.toFixed(2));
    }

    function timelinePositionWithinWindow(startDate, endDate, targetDate) {
      if (!startDate || !endDate || !targetDate) return 50;
      const start = startDate.getTime();
      const end = endDate.getTime();
      const target = targetDate.getTime();
      if (end <= start) return 50;
      const ratio = (target - start) / (end - start);
      return 14 + Math.max(-0.12, Math.min(1.12, ratio)) * 72;
    }

    function timelineHoldWidth(holdDays) {
      return Math.max(26, Math.min(56, 18 + holdDays * 4));
    }

    function createTimelineMarker(percent, label, value, tone = '') {
      const marker = document.createElement('div');
      marker.className = `param-timeline-marker ${tone}`.trim();
      marker.style.left = `${timelineMarkerPosition(percent)}%`;

      const dot = document.createElement('span');
      dot.className = 'param-timeline-dot';
      marker.appendChild(dot);

      const labelNode = document.createElement('div');
      labelNode.className = 'param-timeline-marker-label';
      labelNode.textContent = label;
      marker.appendChild(labelNode);

      const valueNode = document.createElement('div');
      valueNode.className = 'param-timeline-marker-value';
      valueNode.textContent = value;
      marker.appendChild(valueNode);

      return marker;
    }

    function createTimelineTrackNote(text, tone = '') {
      const note = document.createElement('div');
      note.className = `param-timeline-track-note ${tone}`.trim();
      note.textContent = text;
      return note;
    }

    function buildParameterTimelineSummary(values) {
      const scan = parseCompactDate(values.scan_date);
      const backtestStart = parseCompactDate(values.backtest_start_date);
      const backtestEnd = parseCompactDate(values.backtest_end_date);
      const holdDays = normalizeHoldDays(values.hold_days);

      if (!scan && !backtestStart && !backtestEnd) {
        return '当前还没有形成完整时间定义。scan_date 决定主扫描日，回测窗口由 backtest_start_date 和 backtest_end_date 共同给出，hold_days 决定每个信号的基础持有期。';
      }

      if (backtestStart && backtestEnd && backtestStart.date.getTime() <= backtestEnd.date.getTime()) {
        const inWindow = scan
          && scan.date.getTime() >= backtestStart.date.getTime()
          && scan.date.getTime() <= backtestEnd.date.getTime();
        return (
          `scan_date 是当前主扫描日，backtest_start_date 到 backtest_end_date 构成滚动前瞻回测窗口，` +
          `hold_days 表示每个信号向后持有 ${holdDays} 个交易日。` +
          (scan
            ? (inWindow
              ? ' 当前 scan_date 位于回测窗口内。'
              : ' 当前 scan_date 可以独立于回测窗口存在，主扫描结果和回测样本不必是同一天。')
            : ' 当前还没有填写 scan_date。')
        );
      }

      if (backtestStart || backtestEnd) {
        if (backtestStart && backtestEnd && backtestStart.date.getTime() > backtestEnd.date.getTime()) {
          return '当前回测窗口顺序不合法：backtest_start_date 不能晚于 backtest_end_date；在修正前，前瞻回测不会执行。';
        }
        return '当前只设置了一个回测边界；前瞻回测要求 backtest_start_date 和 backtest_end_date 同时存在。';
      }

      return (
        `当前只有主扫描日和持有期在生效：scan_date = ${formatCompactDate(scan)}，` +
        `hold_days = ${holdDays} 个交易日。若要做滚动前瞻回测，还需要同时设置 backtest_start_date 和 backtest_end_date。`
      );
    }

    function renderBacktestTimelineLane(host, values) {
      host.innerHTML = '';
      const backtestStart = parseCompactDate(values.backtest_start_date);
      const backtestEnd = parseCompactDate(values.backtest_end_date);
      const scan = parseCompactDate(values.scan_date);

      const rail = document.createElement('div');
      rail.className = 'param-timeline-rail';
      host.appendChild(rail);

      if (backtestStart && backtestEnd && backtestStart.date.getTime() <= backtestEnd.date.getTime()) {
        const window = document.createElement('div');
        window.className = 'param-timeline-window';
        window.style.left = '14%';
        window.style.width = '72%';
        host.appendChild(window);

        host.appendChild(createTimelineMarker(14, 'backtest_start_date', formatCompactDate(backtestStart), 'boundary'));
        host.appendChild(createTimelineMarker(86, 'backtest_end_date', formatCompactDate(backtestEnd), 'boundary'));
        if (scan) {
          host.appendChild(
            createTimelineMarker(
              timelinePositionWithinWindow(backtestStart.date, backtestEnd.date, scan.date),
              'scan_date',
              formatCompactDate(scan),
              'scan'
            )
          );
        }
        host.appendChild(createTimelineTrackNote('这段区间里的每个历史扫描日，都会各自进入一次前瞻回测。'));
        return;
      }

      if (backtestStart) {
        host.appendChild(createTimelineMarker(22, 'backtest_start_date', formatCompactDate(backtestStart), 'boundary'));
      }
      if (backtestEnd) {
        host.appendChild(createTimelineMarker(78, 'backtest_end_date', formatCompactDate(backtestEnd), 'boundary'));
      }
      if (!backtestStart && !backtestEnd && scan) {
        host.appendChild(createTimelineMarker(50, 'scan_date', formatCompactDate(scan), 'scan muted'));
      }
      const warning = backtestStart && backtestEnd && backtestStart.date.getTime() > backtestEnd.date.getTime();
      host.appendChild(
        createTimelineTrackNote(
          warning
            ? '起止顺序有误：backtest_start_date 不能晚于 backtest_end_date。'
            : '只有同时设置 backtest_start_date 和 backtest_end_date，前瞻回测窗口才会生效。',
          warning ? 'warning' : ''
        )
      );
    }

    function renderHoldTimelineLane(host, values) {
      host.innerHTML = '';
      const scan = parseCompactDate(values.scan_date);
      const holdDays = normalizeHoldDays(values.hold_days);
      const startPct = 22;
      const width = timelineHoldWidth(holdDays);
      const endPct = Math.min(86, startPct + width);

      const rail = document.createElement('div');
      rail.className = 'param-timeline-rail';
      host.appendChild(rail);

      host.appendChild(createTimelineMarker(startPct, 'scan_date', formatCompactDate(scan), scan ? 'scan' : 'scan muted'));

      const holdBar = document.createElement('div');
      holdBar.className = 'param-timeline-hold';
      holdBar.style.left = `${timelineMarkerPosition(startPct)}%`;
      holdBar.style.width = `${Math.max(12, endPct - startPct).toFixed(2)}%`;
      host.appendChild(holdBar);

      host.appendChild(createTimelineMarker(endPct, 'hold_days', `${holdDays} 个交易日`, 'hold'));
      host.appendChild(createTimelineTrackNote(`每个扫描信号都会从自己的 scan_date 起，向后持有 ${holdDays} 个交易日。`));
    }

    function renderParameterTimelineNotes(host, values) {
      host.innerHTML = '';
      const scan = parseCompactDate(values.scan_date);
      const backtestStart = parseCompactDate(values.backtest_start_date);
      const backtestEnd = parseCompactDate(values.backtest_end_date);
      const holdDays = normalizeHoldDays(values.hold_days);

      const notes = [
        {
          label: 'scan_date',
          value: formatCompactDate(scan),
          detail: '当前页面的主扫描输出、候选池和入选结果按这一天生成。',
        },
        {
          label: 'backtest window',
          value: backtestStart && backtestEnd ? `${formatCompactDate(backtestStart)} -> ${formatCompactDate(backtestEnd)}` : '-',
          detail: backtestStart && backtestEnd
            ? '这段区间里的可用扫描日会滚动进入前瞻回测。'
            : '只填一端不会执行前瞻回测。',
        },
        {
          label: 'hold_days',
          value: `${holdDays} 个交易日`,
          detail: `回测窗口内的每个信号，都会各自向后持有 ${holdDays} 个交易日。`,
        },
      ];

      notes.forEach((item) => {
        const card = document.createElement('article');
        card.className = 'param-timeline-note';

        const label = document.createElement('div');
        label.className = 'param-timeline-note-label';
        label.textContent = item.label;
        card.appendChild(label);

        const value = document.createElement('div');
        value.className = 'param-timeline-note-value';
        value.textContent = item.value;
        card.appendChild(value);

        const detail = document.createElement('div');
        detail.className = 'param-timeline-note-detail';
        detail.textContent = item.detail;
        card.appendChild(detail);

        host.appendChild(card);
      });
    }

    function renderParameterTimeline(detail) {
      const panel = document.getElementById('parameter-timeline-panel');
      const meta = document.getElementById('parameter-timeline-meta');
      const summary = document.getElementById('parameter-timeline-summary');
      const windowTrack = document.getElementById('parameter-window-track');
      const holdTrack = document.getElementById('parameter-hold-track');
      const notes = document.getElementById('parameter-timeline-notes');

      if (!hasParameterTimeline(detail)) {
        panel.hidden = true;
        meta.textContent = '';
        summary.textContent = '';
        windowTrack.innerHTML = '';
        holdTrack.innerHTML = '';
        notes.innerHTML = '';
        return;
      }

      const values = collectFormValues(detail);
      panel.hidden = false;
      meta.textContent = '按当前表单实时刷新';
      summary.textContent = buildParameterTimelineSummary(values);
      renderBacktestTimelineLane(windowTrack, values);
      renderHoldTimelineLane(holdTrack, values);
      renderParameterTimelineNotes(notes, values);
    }

    function renderMetrics(result) {
      const host = document.getElementById('result-metrics');
      const summary = result.summary || {};
      const backtest = result.backtest_summary || {};
      const strategyId = currentStrategyId();
      const metrics = result.state_flow
        ? [
            ['买点日期', summary.start_date || '-'],
            ['扫描日期', result.scan_date || '-'],
            ['当前状态', summary.current_state_label || '-'],
            ['区间收益', summary.holding_return_pct === null || summary.holding_return_pct === undefined ? '-' : `${(Number(summary.holding_return_pct) * 100).toFixed(1)}%`],
          ]
        : isEntropyStrategy(strategyId)
        ? [
            ['扫描日期', result.scan_date || '-'],
            ['市场相位', (result.strategy_focus && result.strategy_focus.phase_label) || summary.market_phase_state || '-'],
            ['候选数', summary.n_candidates ?? result.candidate_count ?? 0],
            ['入选数', summary.n_selected ?? result.selected_count ?? 0],
            ['放弃数', summary.n_abandoned ?? 0],
            ['最终净值', backtest.final_nav ?? '-'],
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

    function renderStrategyFocus(result) {
      const panel = document.getElementById('strategy-focus-panel');
      const title = document.getElementById('strategy-focus-title');
      const meta = document.getElementById('strategy-focus-meta');
      const summaryHost = document.getElementById('strategy-focus-summary');
      const gridHost = document.getElementById('strategy-focus-grid');
      const leadPanel = document.getElementById('strategy-focus-lead');
      const leadTitle = document.getElementById('focus-lead-title');
      const leadMeta = document.getElementById('focus-lead-meta');
      const leadKv = document.getElementById('focus-lead-kv');
      const focus = result.strategy_focus;

      if (!focus || !focus.cards || !focus.cards.length) {
        panel.hidden = true;
        title.textContent = '策略画像';
        meta.textContent = '';
        summaryHost.textContent = '';
        gridHost.innerHTML = '';
        leadPanel.hidden = true;
        leadTitle.textContent = '';
        leadMeta.textContent = '';
        leadKv.innerHTML = '';
        return;
      }

      panel.hidden = false;
      title.textContent = focus.title || '策略画像';
      meta.textContent = focus.subtitle || '';
      summaryHost.textContent = focus.summary || '';
      gridHost.innerHTML = '';

      (focus.cards || []).forEach((card) => {
        const node = document.createElement('article');
        node.className = 'focus-metric';
        node.innerHTML = `
          <div class="focus-metric-label">${card.label || '-'}</div>
          <div class="focus-metric-value">${fmt(card.value)}</div>
          <div class="focus-metric-detail">${card.detail || ''}</div>
        `;
        gridHost.appendChild(node);
      });

      const lead = focus.lead;
      if (!lead) {
        leadPanel.hidden = true;
        leadTitle.textContent = '';
        leadMeta.textContent = '';
        leadKv.innerHTML = '';
        return;
      }

      leadPanel.hidden = false;
      leadTitle.textContent = lead.title || '';
      leadMeta.textContent = lead.subtitle || '';
      leadKv.innerHTML = '';
      (lead.items || []).forEach((item) => {
        const node = document.createElement('div');
        node.className = 'focus-chip';
        node.innerHTML = `
          <div class="focus-chip-label">${item.label || '-'}</div>
          <div class="focus-chip-value">${fmt(item.value)}</div>
        `;
        leadKv.appendChild(node);
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
      renderStrategyFocus(result);
      renderMetrics(result);
      const selectedRows = result.selected_rows || [];
      const candidateRows = result.candidate_rows || [];
      const strategyId = currentStrategyId();
      const summarySelectedColumns = isEntropyStrategy(strategyId)
        ? ['股票名称', '股票代码', '建仓方式', '计划仓位', '退出模式', '执行状态', '选中原因', '操作建议']
        : ['股票名称', '股票代码', '选中原因', '操作建议'];
      const preferredCandidateColumns = strategyId.includes('uptrend_hold_state_flow')
        ? ['state_label', 'active', 'activated_on_path', 'first_entry_date', 'last_entry_date', 'days_in_state', 'state_score', 'reason', 'entropy_reserve', 'disorder_pressure', 'expansion_thrust', 'directional_persistence', 'peak_extension_score', 'deceleration_score', 'fragility_score']
        : strategyId.includes('rapid_expansion_exhaustion_exit')
        ? ['symbol', 'name', 'start_date', 'scan_date', 'judgement', 'strategy_state', 'strategy_score', 'peak_extension_score', 'deceleration_score', 'fragility_score', 'first_exit_date', 'holding_return_pct']
        : strategyId.includes('rapid_expansion_hold')
        ? ['symbol', 'name', 'start_date', 'scan_date', 'judgement', 'strategy_state', 'strategy_score', 'expansion_thrust', 'acceptance_score', 'instability_risk', 'holding_return_pct']
        : strategyId.includes('entropy_hold_judgement')
        ? ['symbol', 'name', 'start_date', 'scan_date', 'judgement', 'strategy_state', 'strategy_score', 'disorder_pressure', 'first_exit_date', 'holding_return_pct']
        : strategyId.includes('continuous_decline_recovery')
        ? ['symbol', 'name', 'industry', 'market_buy_state', 'sector_rank', 'sector_score', 'strategy_score', 'damage_score', 'repair_score', 'entry_window_score', 'flow_support_score', 'rebound_from_low_10', 'relative_strength_vs_sector_5', 'entry_mode']
        : strategyId.includes('entropy_accumulation_breakout')
        ? ['symbol', 'name', 'industry', 'phase', 'composite_score', 'accum_quality', 'bifurc_quality', 'perm_entropy_m', 'path_irrev_m', 'dom_eig_m', 'coherence_l1', 'purity_norm', 'coherence_decay_rate', 'von_neumann_entropy']
        : isEntropyStrategy(strategyId)
        ? ['symbol', 'name', 'industry', 'market', 'market_phase_state', 'strategy_state', 'context_score', 'stock_state_score', 'strategy_score', 'entropy_quality', 'bifurcation_quality', 'trigger_quality', 'execution_readiness_score', 'execution_penalty_score', 'abandonment_score', 'entry_mode', 'position_scale']
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
      renderParameterTimeline(detail);
      document.getElementById('results-panel').hidden = true;
      document.getElementById('state-flow-panel').hidden = true;
      document.getElementById('state-flow-summary').textContent = '';
      document.getElementById('state-flow-diagram').innerHTML = '';
      document.getElementById('state-compare-block').hidden = true;
      document.getElementById('state-flow-compare').innerHTML = '';
      document.getElementById('strategy-focus-panel').hidden = true;
      document.getElementById('strategy-focus-title').textContent = '策略画像';
      document.getElementById('strategy-focus-meta').textContent = '';
      document.getElementById('strategy-focus-summary').textContent = '';
      document.getElementById('strategy-focus-grid').innerHTML = '';
      document.getElementById('strategy-focus-lead').hidden = true;
      document.getElementById('focus-lead-title').textContent = '';
      document.getElementById('focus-lead-meta').textContent = '';
      document.getElementById('focus-lead-kv').innerHTML = '';
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
      const rerenderTimeline = () => {
        if (state.current) renderParameterTimeline(state.current);
      };
      document.getElementById('param-form-grid').addEventListener('input', rerenderTimeline);
      document.getElementById('param-form-grid').addEventListener('change', rerenderTimeline);
      document.getElementById('reset-btn').addEventListener('click', () => {
        if (state.current) {
          resetForm(state.current);
          renderParameterTimeline(state.current);
        }
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
    # 支持多种 section 标题写法
    description_text = ""
    for key in ("描述", "1. 策略总览", "策略总览", "概述", "Overview"):
        if key in sections:
            description_text = sections[key]
            break
    parameter_text = ""
    for key in ("主要参数", "9. 参数说明", "参数说明", "参数", "Parameters"):
        if key in sections:
            parameter_text = sections[key]
            break
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


def _strategy_module_name(module_path: Path) -> str:
    relative_path = module_path.resolve().relative_to(REPO_ROOT.resolve())
    return ".".join(relative_path.with_suffix("").parts)


def _import_strategy_module(module_path: Path, strategy_id: str):
  if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
  importlib.invalidate_caches()
  return importlib.import_module(_strategy_module_name(module_path))


def _infer_kind(action: argparse.Action) -> str:
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        return "boolean"
    if action.type is int:
        return "integer"
    if action.type is float:
        return "float"
    return "string"


# 在 web 表单中隐藏的参数 — 用默认值自动填充，不需要用户手动配置
HIDDEN_PARAM_DESTS = {
    "data_dir", "out_dir", "basic_path", "index_path",
    "lookback_days", "min_amount", "min_turnover",
    "exclude_st", "verbose",
}


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
                "hidden": dest in HIDDEN_PARAM_DESTS,
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


def _entrypoint_sort_key(path: Path) -> tuple[int, str]:
  name = path.name
  if name.endswith("_scan.py"):
    return (0, name)
  if "diagnostics" not in name:
    return (1, name)
  return (2, name)


def _discover_strategies() -> list[dict[str, Any]]:
  strategies: list[dict[str, Any]] = []
  if not STRATEGY_ROOT.exists():
    return strategies

  for strategy_dir in sorted(path for path in STRATEGY_ROOT.iterdir() if path.is_dir()):
    readme_path = strategy_dir / "README.md"
    if not readme_path.exists():
      continue

    relative_path = strategy_dir.relative_to(REPO_ROOT)
    entrypoints = sorted(strategy_dir.glob("run_*.py"), key=_entrypoint_sort_key)
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
    for pattern in ["market_scan_snapshot_*.csv", "market_snapshot_*.csv"]:
        for path in output_dir.glob(pattern):
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
    selected_path = _find_existing(output_dir, [
        f"selected_portfolio_{scan_date}_top*.csv", f"selected_portfolio_*_{scan_date}_top*.csv",
        f"breakout_candidates_{scan_date}.csv",
    ])
    candidates_path = _find_existing(output_dir, [
        f"resonance_candidates_{scan_date}_all.csv", f"*_candidates_{scan_date}_all.csv",
        f"market_snapshot_{scan_date}.csv",
    ])
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


def _normalize_date_value(value: str) -> str:
    """Normalize date strings like '2025/01/01' or '2025-01-01' to 'YYYYMMDD'."""
    stripped = value.strip()
    if not stripped:
        return stripped
    for sep in ("/", "-"):
        if sep in stripped:
            parts = stripped.split(sep)
            if len(parts) == 3 and all(p.isdigit() for p in parts):
                return f"{int(parts[0]):04d}{int(parts[1]):02d}{int(parts[2]):02d}"
    return stripped


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

        # 日期参数自动归一化为 YYYYMMDD
        if dest.endswith("_date"):
            value_str = _normalize_date_value(value_str)

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


def _text_or_dash(value: Any) -> str:
  if value is None:
    return "-"
  try:
    if pd.isna(value):
      return "-"
  except TypeError:
    pass
  text = str(value).strip()
  if not text or text.lower() == "nan":
    return "-"
  return text


def _lookup_label(labels: dict[str, str], value: Any) -> str:
  text = _text_or_dash(value)
  if text == "-":
    return text
  return labels.get(text, text)


def _phase_label(value: Any) -> str:
  return _lookup_label(MARKET_PHASE_LABELS, value)


def _market_buy_state_label(value: Any) -> str:
  return _lookup_label(MARKET_BUY_STATE_LABELS, value)


def _entry_mode_label(value: Any) -> str:
  return _lookup_label(ENTRY_MODE_LABELS, value)


def _exit_mode_label(value: Any) -> str:
  return _lookup_label(EXIT_MODE_LABELS, value)


def _execution_state_label(value: Any) -> str:
  return _lookup_label(EXECUTION_STATE_LABELS, value)


def _yes_no_label(value: Any) -> str:
  return "是" if bool(value) else "否"


def _int_or_zero(value: Any) -> int:
  number = _float_or_none(value)
  if number is None:
    return 0
  return int(number)


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
      f"当前市场处于 {_phase_label(row.get('market_phase_state'))}，个股位于低熵压缩后启动区，"
      f"上下文分数 {_fmt_num(row.get('context_score'))}，熵质量 {_fmt_num(row.get('entropy_quality'))}，"
      f"分叉质量 {_fmt_num(row.get('bifurcation_quality'))}，触发质量 {_fmt_num(row.get('trigger_quality'))}。"
    )
  if variant == "continuous_decline_recovery":
    return (
      f"当前市场处于 {_market_buy_state_label(row.get('market_buy_state'))}，所在行业排名 #{_int_or_zero(row.get('sector_rank'))}，"
      f"近10日跌幅 {_fmt_pct(row.get('ret_10'))}，当前自10日低点反弹 {_fmt_pct(row.get('rebound_from_low_10'))}，"
      f"修复分数 {_fmt_num(row.get('repair_score'))}，策略分数 {_fmt_num(row.get('strategy_score'))}。"
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
    entry_mode = _entry_mode_label(row.get("entry_mode"))
    position_scale = _fmt_pct(row.get("position_scale"))
    staged_entry_days = _text_or_dash(row.get("staged_entry_days"))
    staged_text = f"，预计分 {staged_entry_days} 天完成建仓" if staged_entry_days != "-" and str(row.get("entry_mode") or "") == "staged" else ""
    return base + f" 当前建议 {entry_mode}，计划仓位 {position_scale}{staged_text}；若突破动能回落，或重新跌回 20 日线下方，可提前止盈/止损。"
  if variant == "continuous_decline_recovery":
    state_label = _market_buy_state_label(row.get("market_buy_state"))
    entry_mode = _entry_mode_label(row.get("entry_mode"))
    position_scale = _fmt_pct(row.get("position_scale"))
    if str(row.get("entry_mode") or "") == "skip":
      return f"当前市场处于 {state_label}，更适合等待下一轮修复确认，而不是继续追这只股票。"
    return base + f" 当前市场处于 {state_label}，建议 {entry_mode}，计划仓位 {position_scale}；若 2 个交易日内不能继续放量走强，或重新跌回 5 日线下方，应优先减仓。"
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
        "建仓方式": _entry_mode_label(row.get("entry_mode")),
        "计划仓位": _fmt_pct(row.get("position_scale")),
        "退出模式": _exit_mode_label(row.get("exit_mode")),
        "执行状态": _execution_state_label(row.get("execution_cost_state")),
        "选中原因": _selected_reason_text(row, strategy),
        "操作建议": _selected_advice_text(row, strategy, values),
      }
    )
  return display_rows


def _entropy_bifurcation_focus_payload(
  summary: dict[str, Any],
  selected_rows: list[dict[str, Any]],
  candidate_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
  if not summary and not selected_rows and not candidate_rows:
    return None

  lead_row = dict(selected_rows[0]) if selected_rows else (dict(candidate_rows[0]) if candidate_rows else {})
  market_phase_label = _phase_label(summary.get("market_phase_state") or lead_row.get("market_phase_state"))
  candidate_count = _int_or_zero(summary.get("n_candidates")) or len(candidate_rows)
  selected_count = _int_or_zero(summary.get("n_selected")) or len(selected_rows)
  abandoned_count = _int_or_zero(summary.get("n_abandoned"))

  lead_name = _text_or_dash(lead_row.get("name") or lead_row.get("symbol"))
  if lead_name != "-":
    lead_message = (
      f"首选标的 {lead_name} 当前建议 {_entry_mode_label(lead_row.get('entry_mode'))}，"
      f"计划仓位 {_fmt_pct(lead_row.get('position_scale'))}，退出模式 {_exit_mode_label(lead_row.get('exit_mode'))}。"
    )
  else:
    lead_message = "当前没有形成最终入选，更接近观测盘而不是执行盘。"

  subtitle_parts = [
    _text_or_dash(lead_row.get("industry")),
    _text_or_dash(lead_row.get("market")),
    market_phase_label,
  ]
  subtitle = " / ".join(part for part in subtitle_parts if part != "-")

  lead_payload = None
  if lead_row:
    lead_title = lead_name
    lead_symbol = _text_or_dash(lead_row.get("symbol"))
    if lead_symbol != "-" and lead_symbol != lead_title:
      lead_title = f"{lead_title} ({lead_symbol})"
    lead_payload = {
      "title": lead_title,
      "subtitle": subtitle,
      "items": [
        {"label": "建仓方式", "value": _entry_mode_label(lead_row.get("entry_mode"))},
        {"label": "计划仓位", "value": _fmt_pct(lead_row.get("position_scale"))},
        {"label": "分段天数", "value": _text_or_dash(lead_row.get("staged_entry_days"))},
        {"label": "退出模式", "value": _exit_mode_label(lead_row.get("exit_mode"))},
        {"label": "执行状态", "value": _execution_state_label(lead_row.get("execution_cost_state"))},
        {"label": "状态门控", "value": "通过" if bool(lead_row.get("strategy_state")) else "观察"},
        {"label": "潜在标签", "value": _text_or_dash(lead_row.get("latent_state_label"))},
        {"label": "上下文分数", "value": _fmt_num(lead_row.get("context_score"))},
        {"label": "执行惩罚", "value": _fmt_num(lead_row.get("execution_penalty_score"))},
        {"label": "放弃交易", "value": _yes_no_label(lead_row.get("strategic_abandonment"))},
      ],
    }

  return {
    "strategy": "entropy_bifurcation_setup",
    "title": "熵分叉策略画像",
    "subtitle": "市场门控 + 个股状态 + 执行计划",
    "phase_label": market_phase_label,
    "summary": (
      f"当前市场处于 {market_phase_label}，门控分数 {_fmt_num(summary.get('market_regime_score'))}，"
      f"耦合熵 {_fmt_num(summary.get('market_coupling_entropy_20'))}，噪声成本 {_fmt_num(summary.get('market_noise_cost'))}。"
      f"本次共 {candidate_count} 只候选，最终 {selected_count} 只入选，显式放弃 {abandoned_count} 只。{lead_message}"
    ),
    "cards": [
      {"label": "市场相位", "value": market_phase_label, "detail": f"门控分数 {_fmt_num(summary.get('market_regime_score'))}"},
      {"label": "耦合熵", "value": _fmt_num(summary.get('market_coupling_entropy_20')), "detail": f"相位失真占比 {_fmt_pct(summary.get('market_phase_distortion_share'))}"},
      {"label": "噪声成本", "value": _fmt_num(summary.get('market_noise_cost')), "detail": f"显式放弃 {abandoned_count} 只"},
      {"label": "上下文强度", "value": _fmt_num(summary.get('avg_context_score')), "detail": f"平均个股状态 {_fmt_num(summary.get('avg_stock_state_score'))}"},
      {"label": "执行准备", "value": _fmt_num(summary.get('avg_execution_readiness_score')), "detail": f"执行惩罚 {_fmt_num(summary.get('avg_execution_penalty_score'))}"},
      {"label": "实验层观测", "value": _fmt_num(summary.get('avg_experimental_model_score')), "detail": f"平均放弃强度 {_fmt_num(summary.get('avg_abandonment_score'))}"},
    ],
    "lead": lead_payload,
  }


def _continuous_decline_recovery_focus_payload(
  summary: dict[str, Any],
  selected_rows: list[dict[str, Any]],
  candidate_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
  if not summary and not selected_rows and not candidate_rows:
    return None

  lead_row = dict(selected_rows[0]) if selected_rows else (dict(candidate_rows[0]) if candidate_rows else {})
  state_label = _market_buy_state_label(summary.get("market_buy_state") or lead_row.get("market_buy_state"))
  lead_name = _text_or_dash(lead_row.get("name") or lead_row.get("symbol"))
  lead_title = lead_name
  lead_symbol = _text_or_dash(lead_row.get("symbol"))
  if lead_symbol != "-" and lead_symbol != lead_title:
    lead_title = f"{lead_title} ({lead_symbol})"

  lead_payload = None
  if lead_row:
    lead_payload = {
      "title": lead_title,
      "subtitle": " / ".join(part for part in [_text_or_dash(lead_row.get("industry")), state_label] if part != "-"),
      "items": [
        {"label": "市场状态", "value": state_label},
        {"label": "板块排名", "value": _text_or_dash(lead_row.get("sector_rank"))},
        {"label": "策略分数", "value": _fmt_num(lead_row.get("strategy_score"))},
        {"label": "修复分数", "value": _fmt_num(lead_row.get("repair_score"))},
        {"label": "建仓方式", "value": _entry_mode_label(lead_row.get("entry_mode"))},
        {"label": "计划仓位", "value": _fmt_pct(lead_row.get("position_scale"))},
        {"label": "执行状态", "value": _execution_state_label(lead_row.get("execution_cost_state"))},
      ],
    }

  return {
    "strategy": "continuous_decline_recovery",
    "title": "连续下跌修复画像",
    "subtitle": "市场买点 + 板块领先 + 个股修复窗口",
    "phase_label": state_label,
    "summary": (
      f"当前市场处于 {state_label}，买点评分 {_fmt_num(summary.get('market_buy_score'))}，"
      f"最近洗盘峰值 {_fmt_num(summary.get('recent_washout_peak'))}，"
      f"候选 {len(candidate_rows)} 只，最终入选 {len(selected_rows)} 只，"
      f"领先行业 {summary.get('top_sector') or '-'}。"
    ),
    "cards": [
      {"label": "市场买点", "value": state_label, "detail": f"买点评分 {_fmt_num(summary.get('market_buy_score'))}"},
      {"label": "洗盘强度", "value": _fmt_num(summary.get('market_washout_score')), "detail": f"最近峰值 {_fmt_num(summary.get('recent_washout_peak'))}"},
      {"label": "修复强度", "value": _fmt_num(summary.get('market_repair_score')), "detail": f"候选 {len(candidate_rows)} 只"},
      {"label": "过热程度", "value": _fmt_num(summary.get('market_overheat_score')), "detail": f"入选 {len(selected_rows)} 只"},
      {"label": "领先行业", "value": _text_or_dash(summary.get('top_sector')), "detail": f"行业分数 {_fmt_num(summary.get('top_sector_score'))}"},
      {"label": "平均得分", "value": _fmt_num(summary.get('avg_selected_score')), "detail": f"候选均分 {_fmt_num(summary.get('avg_candidate_score'))}"},
    ],
    "lead": lead_payload,
  }


def _strategy_focus_payload(
  strategy: dict[str, Any],
  summary: dict[str, Any],
  selected_rows: list[dict[str, Any]],
  candidate_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
  if strategy.get("base_id") == "entropy_bifurcation_setup":
    return _entropy_bifurcation_focus_payload(summary, selected_rows, candidate_rows)
  if strategy.get("base_id") == "continuous_decline_recovery":
    return _continuous_decline_recovery_focus_payload(summary, selected_rows, candidate_rows)
  return None


# ═════════════════════════════════════════════════════════
# 量子态扫描 helpers
# ═════════════════════════════════════════════════════════

_quantum_logger = logging.getLogger("quantum_scan")

PHASE_LABELS_QUANTUM = {
    "breakout": "坍缩 (Collapse)",
    "accumulation": "叠加态 (Superposition)",
    "idle": "退相干 (Decoherence)",
}


def _quantum_scan_market(top_n: int = 50, scan_date: str = "") -> list[dict]:
    """全市场量子态扫描, 返回按 composite_score 降序排列的结果."""
    from src.strategy.entropy_accumulation_breakout.scan_service import (
        ScanConfig,
        _load_basic_info,
        _load_daily,
        _resolve_symbols,
        _should_skip,
        scan_single_symbol,
    )

    data_dir = str(DEFAULT_DATA_DIR)
    basic_path = str(DEFAULT_BASIC_PATH)
    cfg = ScanConfig(data_dir=data_dir, basic_path=basic_path, scan_date=scan_date, top_n=top_n)
    basic_info = _load_basic_info(basic_path)
    symbols = _resolve_symbols(data_dir, None)

    scan_dt = cfg.scan_date
    if not scan_dt:
        for sym in symbols[:10]:
            df = _load_daily(data_dir, sym)
            if df is not None and len(df) > 0:
                scan_dt = str(df["trade_date"].iloc[-1])
                break

    results = []
    for sym in symbols:
        sig = scan_single_symbol(data_dir, sym, cfg, scan_dt, basic_info)
        if sig is None:
            continue
        row = {
            "symbol": sig.symbol,
            "name": sig.details.get("name", ""),
            "industry": sig.details.get("industry", ""),
            "trade_date": sig.trade_date,
            "phase": sig.phase,
            "phase_label": PHASE_LABELS_QUANTUM.get(sig.phase, sig.phase),
            "accum_quality": sig.accum_quality,
            "bifurc_quality": sig.bifurc_quality,
            "composite_score": sig.composite_score,
            "entry_signal": sig.entry_signal,
            "perm_entropy": sig.details.get("perm_entropy_m"),
            "path_irrev": sig.details.get("path_irrev_m"),
            "dom_eig": sig.details.get("dom_eig_m"),
            "coherence_l1": sig.details.get("coherence_l1"),
            "purity": sig.details.get("purity_norm"),
            "coherence_decay_rate": sig.details.get("coherence_decay_rate"),
            "von_neumann_entropy": sig.details.get("von_neumann_entropy"),
        }
        results.append(row)

    results.sort(key=lambda r: r.get("composite_score") or 0, reverse=True)
    return results[:top_n], scan_dt


def _quantum_scan_single(symbol: str, scan_date: str = "") -> dict | None:
    """单只股票量子态查询."""
    from src.strategy.entropy_accumulation_breakout.scan_service import (
        ScanConfig,
        _load_basic_info,
        _load_daily,
        scan_single_symbol,
    )

    data_dir = str(DEFAULT_DATA_DIR)
    basic_path = str(DEFAULT_BASIC_PATH)
    cfg = ScanConfig(data_dir=data_dir, basic_path=basic_path, scan_date=scan_date)
    basic_info = _load_basic_info(basic_path)

    # 标准化 symbol
    sym = symbol.strip().lower()
    if not sym:
        return None

    scan_dt = scan_date
    if not scan_dt:
        df_tmp = _load_daily(data_dir, sym)
        if df_tmp is not None and len(df_tmp) > 0:
            scan_dt = str(df_tmp["trade_date"].iloc[-1])
        else:
            return None

    sig = scan_single_symbol(data_dir, sym, cfg, scan_dt, basic_info)
    if sig is None:
        return None

    return {
        "symbol": sig.symbol,
        "name": sig.details.get("name", ""),
        "industry": sig.details.get("industry", ""),
        "trade_date": sig.trade_date,
        "phase": sig.phase,
        "phase_label": PHASE_LABELS_QUANTUM.get(sig.phase, sig.phase),
        "accum_quality": sig.accum_quality,
        "bifurc_quality": sig.bifurc_quality,
        "composite_score": sig.composite_score,
        "entry_signal": sig.entry_signal,
        "perm_entropy": sig.details.get("perm_entropy_m"),
        "path_irrev": sig.details.get("path_irrev_m"),
        "dom_eig": sig.details.get("dom_eig_m"),
        "vol_impulse": sig.details.get("vol_impulse"),
        "entropy_accel": sig.details.get("entropy_accel"),
        "coherence_l1": sig.details.get("coherence_l1"),
        "purity": sig.details.get("purity_norm"),
        "coherence_decay_rate": sig.details.get("coherence_decay_rate"),
        "von_neumann_entropy": sig.details.get("von_neumann_entropy"),
    }


QUANTUM_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>量子态扫描 — gp-quant</title>
  <style>
    :root {
      color-scheme: dark;
      --paper: #09131a;
      --ink: #e9f4ef;
      --muted: #93a9b2;
      --line: rgba(121, 164, 173, 0.18);
      --card: rgba(11, 19, 26, 0.96);
      --accent: #2dd4bf;
      --accent-strong: #9af2dd;
      --warm: #f6b85f;
      --good: #5be38c;
      --bad: #ff7a7a;
      --shadow: 0 28px 90px rgba(0, 0, 0, 0.42);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; min-height: 100vh; color: var(--ink);
      background:
        radial-gradient(circle at 14% 18%, rgba(246, 184, 95, 0.13), transparent 18%),
        radial-gradient(circle at 86% 8%, rgba(45, 212, 191, 0.12), transparent 22%),
        linear-gradient(180deg, #071017 0%, #09141b 42%, #061017 100%);
      font-family: "IBM Plex Serif", "Noto Serif SC", Georgia, serif;
    }
    .shell { max-width: 1480px; margin: 0 auto; padding: 24px 18px 44px; }
    .hero {
      background: var(--card); border: 1px solid var(--line);
      border-radius: 28px; box-shadow: var(--shadow);
      padding: 26px 24px 22px; margin-bottom: 18px; position: relative; overflow: hidden;
    }
    .hero::after {
      content: ""; position: absolute; right: -20px; top: -26px;
      width: 170px; height: 170px; border-radius: 50%;
      background: radial-gradient(circle, rgba(45, 212, 191, 0.18), transparent 70%);
      pointer-events: none;
    }
    .eyebrow { margin-bottom: 10px; font-size: 12px; text-transform: uppercase; letter-spacing: 0.18em; color: var(--warm); }
    h1 { margin: 0 0 12px; font-size: clamp(34px, 5vw, 52px); line-height: 0.96; }
    .hero-copy { color: var(--muted); font-size: 15px; line-height: 1.7; max-width: 680px; }
    .status-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
    .pill {
      display: inline-flex; align-items: center; gap: 5px;
      padding: 6px 12px; border-radius: 999px; font-size: 12px;
      background: rgba(8, 16, 22, 0.72); border: 1px solid var(--line); color: var(--ink);
    }
    .pill-link { text-decoration: none; cursor: pointer; transition: border-color 180ms; }
    .pill-link:hover { border-color: var(--accent); }

    .controls {
      display: flex; flex-wrap: wrap; gap: 12px; align-items: center;
      margin-bottom: 18px;
    }
    .controls input, .controls select {
      font: inherit; color: var(--ink);
      background: rgba(5, 12, 17, 0.88); border: 1px solid rgba(121, 164, 173, 0.32);
      border-radius: 12px; padding: 10px 14px; font-size: 14px;
    }
    .controls input { width: 200px; }
    .controls input::placeholder { color: rgba(147, 169, 178, 0.72); }
    .controls select { width: 120px; }
    button {
      font: inherit; cursor: pointer; border-radius: 999px; padding: 10px 18px; border: 0;
      transition: transform 180ms ease, opacity 180ms ease;
    }
    button:hover { transform: translateY(-1px); }
    button:disabled { cursor: not-allowed; opacity: 0.6; transform: none; }
    .btn-primary {
      background: linear-gradient(135deg, #169f90, #2dd4bf); color: #041014;
      box-shadow: 0 10px 24px rgba(45, 212, 191, 0.22);
    }
    .btn-secondary {
      background: rgba(10, 18, 24, 0.92); color: var(--ink); border: 1px solid var(--line);
    }
    .status-text { color: var(--muted); font-size: 13px; }

    /* Single stock detail card */
    .stock-detail {
      background: var(--card); border: 1px solid var(--line);
      border-radius: 22px; padding: 22px; margin-bottom: 18px;
      box-shadow: var(--shadow);
    }
    .stock-detail h2 { margin: 0 0 6px; font-size: 24px; }
    .stock-detail .meta { color: var(--muted); font-size: 13px; margin-bottom: 16px; }
    .phase-badge {
      display: inline-block; padding: 6px 14px; border-radius: 999px;
      font-size: 14px; font-weight: 700; margin-bottom: 16px;
    }
    .phase-accumulation { background: rgba(45, 212, 191, 0.18); color: var(--accent-strong); border: 1px solid rgba(45, 212, 191, 0.36); }
    .phase-breakout { background: rgba(246, 184, 95, 0.18); color: var(--warm); border: 1px solid rgba(246, 184, 95, 0.36); }
    .phase-idle { background: rgba(147, 169, 178, 0.14); color: var(--muted); border: 1px solid rgba(147, 169, 178, 0.28); }
    .metrics-grid {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
    }
    .metric-card {
      padding: 14px; border-radius: 16px;
      background: rgba(7, 13, 19, 0.76); border: 1px solid rgba(121, 164, 173, 0.14);
    }
    .metric-card .label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; }
    .metric-card .value { font-size: 22px; font-weight: 700; line-height: 1; }

    /* State machine diagram */
    .state-machine {
      display: flex; align-items: center; justify-content: center;
      gap: 0; margin: 18px 0; flex-wrap: wrap;
    }
    .sm-node {
      padding: 16px 20px; border-radius: 18px; text-align: center;
      min-width: 160px; border: 1px solid var(--line);
      background: rgba(11, 19, 26, 0.78);
    }
    .sm-node.active { border-color: var(--accent); background: rgba(45, 212, 191, 0.10); box-shadow: 0 0 24px rgba(45, 212, 191, 0.12); }
    .sm-node .sm-label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px; }
    .sm-node .sm-name { font-size: 18px; font-weight: 700; }
    .sm-node .sm-score { font-size: 14px; color: var(--accent-strong); margin-top: 4px; }
    .sm-arrow { font-size: 24px; color: var(--muted); padding: 0 8px; }

    /* Scan result table */
    .panel {
      background: var(--card); border: 1px solid var(--line);
      border-radius: 22px; padding: 22px; box-shadow: var(--shadow);
    }
    .panel h3 { margin: 0 0 14px; }
    .table-wrap { overflow: auto; border-radius: 14px; border: 1px solid rgba(121, 164, 173, 0.10); }
    table { width: 100%; border-collapse: collapse; min-width: 900px; font-size: 13px; }
    th, td { padding: 9px 12px; text-align: left; border-bottom: 1px solid rgba(121, 164, 173, 0.10); white-space: nowrap; }
    thead th { position: sticky; top: 0; z-index: 1; background: #12202a; }
    .phase-cell { padding: 3px 8px; border-radius: 999px; font-size: 11px; font-weight: 700; }

    .empty-msg { text-align: center; padding: 40px; color: var(--muted); font-size: 15px; }

    @media (max-width: 760px) {
      .controls { flex-direction: column; }
      .controls input { width: 100%; }
      .metrics-grid { grid-template-columns: 1fr 1fr; }
      .state-machine { flex-direction: column; }
      .sm-arrow { transform: rotate(90deg); }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="eyebrow">gp-quant Quantum State Scanner</div>
      <h1>量子态扫描</h1>
      <div class="hero-copy">基于密度矩阵、相干性和退相干理论，将每只股票映射到量子态空间。叠加态 = 筹码压缩吸筹期，坍缩 = 分岔突破，退相干 = 结构衰散。</div>
      <div class="status-row">
        <a href="/" class="pill pill-link">← 策略控制台</a>
        <div class="pill">密度矩阵 ρ</div>
        <div class="pill">纯度 Tr(ρ²)</div>
        <div class="pill">von Neumann 熵</div>
        <div class="pill">相干衰减率</div>
      </div>
    </section>

    <div class="controls">
      <input type="text" id="symbol-input" placeholder="输入代码查询（如 sh600519）" />
      <button class="btn-primary" id="query-btn">查询个股</button>
      <span style="color:var(--muted);font-size:13px;">|</span>
      <select id="top-n-select">
        <option value="30">Top 30</option>
        <option value="50" selected>Top 50</option>
        <option value="100">Top 100</option>
        <option value="200">Top 200</option>
      </select>
      <button class="btn-secondary" id="scan-btn">全市场扫描</button>
      <span class="status-text" id="scan-status">Ready</span>
    </div>

    <section id="stock-detail-section" hidden></section>
    <section id="scan-results-section" hidden></section>
  </div>

  <script>
    function fmt(v) {
      if (v === null || v === undefined || v === '') return '-';
      if (typeof v === 'number') {
        if (!Number.isFinite(v)) return '-';
        return v.toLocaleString('zh-CN', { maximumFractionDigits: 4 });
      }
      return String(v);
    }

    function phaseClass(phase) {
      if (phase === 'accumulation') return 'phase-accumulation';
      if (phase === 'breakout') return 'phase-breakout';
      return 'phase-idle';
    }

    function phaseCellStyle(phase) {
      if (phase === 'accumulation') return 'background:rgba(45,212,191,0.16);color:#9af2dd;';
      if (phase === 'breakout') return 'background:rgba(246,184,95,0.16);color:#f6b85f;';
      return 'background:rgba(147,169,178,0.10);color:#93a9b2;';
    }

    function renderStockDetail(data) {
      const sec = document.getElementById('stock-detail-section');
      const phases = [
        { key: 'accumulation', label: '叠加态', name: 'Superposition', score: data.accum_quality },
        { key: 'breakout', label: '坍缩', name: 'Collapse', score: data.bifurc_quality },
        { key: 'idle', label: '退相干', name: 'Decoherence', score: null },
      ];
      const smHtml = phases.map((p, i) => {
        const active = data.phase === p.key ? ' active' : '';
        const arrow = i < phases.length - 1 ? '<div class="sm-arrow">→</div>' : '';
        return `<div class="sm-node${active}">
          <div class="sm-label">${p.label}</div>
          <div class="sm-name">${p.name}</div>
          ${p.score !== null ? `<div class="sm-score">${fmt(p.score)}</div>` : ''}
        </div>${arrow}`;
      }).join('');

      const metrics = [
        ['置换熵', data.perm_entropy],
        ['路径不可逆', data.path_irrev],
        ['主特征值', data.dom_eig],
        ['L1 相干性', data.coherence_l1],
        ['纯度', data.purity],
        ['相干衰减率', data.coherence_decay_rate],
        ['von Neumann 熵', data.von_neumann_entropy],
        ['量能脉冲', data.vol_impulse],
        ['熵加速度', data.entropy_accel],
        ['惜售质量', data.accum_quality],
        ['分岔质量', data.bifurc_quality],
        ['综合评分', data.composite_score],
      ];
      const metricsHtml = metrics.map(([label, value]) =>
        `<div class="metric-card"><div class="label">${label}</div><div class="value">${fmt(value)}</div></div>`
      ).join('');

      sec.innerHTML = `
        <div class="stock-detail">
          <h2>${data.name || data.symbol} <span style="font-size:14px;color:var(--muted);">${data.symbol}</span></h2>
          <div class="meta">${data.industry || '-'} · ${data.trade_date} · 入场信号: ${data.entry_signal ? '✓ 是' : '✗ 否'}</div>
          <div class="phase-badge ${phaseClass(data.phase)}">${data.phase_label}</div>
          <div class="state-machine">${smHtml}</div>
          <div class="metrics-grid">${metricsHtml}</div>
        </div>`;
      sec.hidden = false;
    }

    function renderScanResults(payload) {
      const sec = document.getElementById('scan-results-section');
      if (!payload.results || payload.results.length === 0) {
        sec.innerHTML = '<div class="panel"><div class="empty-msg">没有找到符合条件的股票。</div></div>';
        sec.hidden = false;
        return;
      }
      const cols = [
        { key: 'symbol', label: '代码' },
        { key: 'name', label: '名称' },
        { key: 'industry', label: '行业' },
        { key: 'phase_label', label: '量子态' },
        { key: 'composite_score', label: '综合评分' },
        { key: 'accum_quality', label: '惜售质量' },
        { key: 'bifurc_quality', label: '分岔质量' },
        { key: 'purity', label: '纯度' },
        { key: 'coherence_l1', label: 'L1相干' },
        { key: 'coherence_decay_rate', label: '衰减率' },
        { key: 'von_neumann_entropy', label: 'VN熵' },
        { key: 'perm_entropy', label: '置换熵' },
        { key: 'path_irrev', label: '不可逆' },
      ];
      const thead = '<tr>' + cols.map(c => `<th>${c.label}</th>`).join('') + '</tr>';
      const tbody = payload.results.map(row => {
        const cells = cols.map(c => {
          if (c.key === 'phase_label') {
            return `<td><span class="phase-cell" style="${phaseCellStyle(row.phase)}">${fmt(row.phase_label)}</span></td>`;
          }
          if (c.key === 'symbol') {
            return `<td><a href="#" onclick="querySingle('${row.symbol}');return false;" style="color:var(--accent);text-decoration:none;">${row.symbol}</a></td>`;
          }
          return `<td>${fmt(row[c.key])}</td>`;
        }).join('');
        return `<tr>${cells}</tr>`;
      }).join('');

      // Count by phase
      const accum = payload.results.filter(r => r.phase === 'accumulation').length;
      const brk = payload.results.filter(r => r.phase === 'breakout').length;
      const idle = payload.results.filter(r => r.phase === 'idle').length;

      sec.innerHTML = `
        <div class="panel">
          <h3>扫描结果 <span style="font-size:13px;color:var(--muted);">
            ${payload.scan_date} · ${payload.count} 只 ·
            叠加态 ${accum} · 坍缩 ${brk} · 退相干 ${idle}
          </span></h3>
          <div class="table-wrap"><table><thead>${thead}</thead><tbody>${tbody}</tbody></table></div>
        </div>`;
      sec.hidden = false;
    }

    async function querySingle(symbol) {
      if (!symbol) return;
      const status = document.getElementById('scan-status');
      status.textContent = '查询中...';
      document.getElementById('query-btn').disabled = true;
      try {
        const resp = await fetch('/api/quantum/stock/' + encodeURIComponent(symbol.trim()));
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || 'Query failed');
        renderStockDetail(data);
        status.textContent = '查询完成';
      } catch (e) {
        status.textContent = e.message;
      } finally {
        document.getElementById('query-btn').disabled = false;
      }
    }

    document.getElementById('query-btn').addEventListener('click', () => {
      const sym = document.getElementById('symbol-input').value.trim();
      if (sym) querySingle(sym);
    });

    document.getElementById('symbol-input').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        const sym = e.target.value.trim();
        if (sym) querySingle(sym);
      }
    });

    document.getElementById('scan-btn').addEventListener('click', async () => {
      const topN = parseInt(document.getElementById('top-n-select').value) || 50;
      const status = document.getElementById('scan-status');
      const btn = document.getElementById('scan-btn');
      status.textContent = '扫描中... 请耐心等待';
      btn.disabled = true;
      try {
        const resp = await fetch('/api/quantum/scan?top_n=' + topN);
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || 'Scan failed');
        renderScanResults(data);
        status.textContent = `扫描完成 · ${data.count} 只`;
      } catch (e) {
        status.textContent = e.message;
      } finally {
        btn.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


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
      "strategy_focus": _strategy_focus_payload(strategy, summary_rows[0] if summary_rows else {}, raw_selected_rows, candidate_rows),
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

    @app.get("/api/quantum/scan")
    def api_quantum_scan():
        top_n = request.args.get("top_n", 50, type=int)
        scan_date = request.args.get("scan_date", "", type=str)
        top_n = max(1, min(top_n, 500))
        try:
            results, actual_date = _quantum_scan_market(top_n=top_n, scan_date=scan_date)
            return jsonify({"ok": True, "scan_date": actual_date, "count": len(results), "results": results})
        except Exception as exc:
            _quantum_logger.exception("quantum scan failed")
            return jsonify({"error": str(exc)}), 500

    @app.get("/api/quantum/stock/<symbol>")
    def api_quantum_stock(symbol: str):
        scan_date = request.args.get("scan_date", "", type=str)
        try:
            result = _quantum_scan_single(symbol, scan_date=scan_date)
            if result is None:
                return jsonify({"error": f"Symbol {symbol} not found or insufficient data"}), 404
            return jsonify({"ok": True, **result})
        except Exception as exc:
            _quantum_logger.exception("quantum stock query failed")
            return jsonify({"error": str(exc)}), 500

    @app.get("/quantum")
    def quantum_page():
        return QUANTUM_HTML

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
