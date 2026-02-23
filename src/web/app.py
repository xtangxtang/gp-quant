import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from flask import Flask, jsonify, request


STOCK_PAGE_HTML = """<!doctype html>
<html lang=\"zh\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>gp-quant - 分钟轨迹</title>
  <style>
    :root { color-scheme: dark; }
    body { background: #0f1115; color: #eaeaea; font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 16px; }
    a { color: #ffffff; text-decoration: none; }
    a:hover { text-decoration: underline; }
    .bar { display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin-bottom:12px; }
    .muted { color: #a0a0a0; font-size: 12px; }
    .panel { border: 1px solid #2a2f3a; background: #121622; padding: 10px; }
    canvas { width: 100%; max-width: 1170px; height: auto; display:block; }
    .sep { height: 10px; }
    .err { color: #b00020; }
  </style>
</head>
<body>
  <div class=\"bar\">
    <div><a href=\"/\">返回</a></div>
    <div><strong id=\"title\">分钟轨迹</strong></div>
    <div class=\"muted\" id=\"subtitle\"></div>
  </div>

  <div class=\"panel\">
    <div id=\"status\" class=\"muted\">加载中...</div>
    <div class=\"sep\"></div>
    <canvas id=\"kline\" width=\"1170\" height=\"280\"></canvas>
    <div class=\"sep\"></div>
    <canvas id=\"turnbar\" width=\"1170\" height=\"160\"></canvas>
    <div class=\"sep\"></div>
    <canvas id=\"cv\" width=\"1170\" height=\"440\"></canvas>
  </div>

<script>
function qs() { return new URLSearchParams(window.location.search); }
function getSymbol() {
  const parts = window.location.pathname.split('/').filter(Boolean);
  return decodeURIComponent(parts[parts.length - 1] || '');
}
function parseTime(s) {
  if (!s) return null;
  // 'YYYY-MM-DD HH:MM'
  const iso = s.replace(' ', 'T') + ':00';
  const d = new Date(iso);
  return isNaN(d.getTime()) ? null : d;
}
function asNumber(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : NaN;
}

function drawCandles(canvas, points, meta) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const W = canvas.width;
  const H = canvas.height;
  const bg = '#121622';
  const fg = '#ffffff';
  const muted = '#a0a0a0';
  const border = '#2a2f3a';
  const up = '#ff4d4f';
  const down = '#7ee787';

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = border;
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, W - 1, H - 1);

  const padL = 64, padR = 18, padT = 32, padB = 30;
  const plotL = padL, plotR = W - padR, plotT = padT, plotB = H - padB;

  const rows = points
    .map(p => ({
      t: parseTime(p.t),
      o: asNumber(p.o),
      h: asNumber(p.h),
      l: asNumber(p.l),
      c: asNumber(p.c),
      ret: asNumber(p.y),
    }))
    .filter(p => p.t && !Number.isNaN(p.o) && !Number.isNaN(p.h) && !Number.isNaN(p.l) && !Number.isNaN(p.c) && !Number.isNaN(p.ret));
  if (rows.length === 0) return;

  const lows = rows.map(r => r.l);
  const highs = rows.map(r => r.h);
  let ymin = Math.min(...lows);
  let ymax = Math.max(...highs);
  if (ymin === ymax) { ymin -= 1; ymax += 1; }
  const pad = (ymax - ymin) * 0.03;
  ymin -= pad;
  ymax += pad;

  function x2px(i) {
    if (rows.length <= 1) return (plotL + plotR) / 2;
    return plotL + (i / (rows.length - 1)) * (plotR - plotL);
  }
  function y2py(y) {
    return plotB - (y - ymin) / (ymax - ymin) * (plotB - plotT);
  }

  // Axes
  ctx.strokeStyle = border;
  ctx.beginPath();
  ctx.moveTo(plotL + 0.5, plotT);
  ctx.lineTo(plotL + 0.5, plotB);
  ctx.lineTo(plotR, plotB + 0.5);
  ctx.stroke();

  // Title
  ctx.fillStyle = fg;
  ctx.font = '700 14px system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif';
  ctx.fillText(meta.title || '1分钟K线', 16, 20);
  ctx.fillStyle = muted;
  ctx.font = '12px system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif';
  ctx.fillText('OHLC', 16, 34);

  // Y ticks
  const fmtP = (v) => {
    if (!Number.isFinite(v)) return '';
    if (Math.abs(v) >= 100) return v.toFixed(2);
    return v.toFixed(3);
  };
  ctx.fillStyle = muted;
  ctx.font = '12px system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif';
  for (const yt of [ymin, (ymin + ymax) / 2, ymax]) {
    const py = y2py(yt);
    ctx.strokeStyle = border;
    ctx.beginPath();
    ctx.moveTo(plotL - 6, py + 0.5);
    ctx.lineTo(plotL, py + 0.5);
    ctx.stroke();
    ctx.fillText(fmtP(yt), 10, py + 4);
  }

  // Candles
  const span = (plotR - plotL);
  const candleW = Math.max(2, Math.min(8, Math.floor(span / Math.max(1, rows.length) * 0.6)));
  ctx.lineWidth = 1;
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    const x = x2px(i);
    const yO = y2py(r.o);
    const yC = y2py(r.c);
    const yH = y2py(r.h);
    const yL = y2py(r.l);
    const col = (r.ret > 0) ? up : (r.ret < 0 ? down : fg);

    // wick
    ctx.strokeStyle = col;
    ctx.beginPath();
    ctx.moveTo(x, yH);
    ctx.lineTo(x, yL);
    ctx.stroke();

    // body
    const top = Math.min(yO, yC);
    const bot = Math.max(yO, yC);
    const h = Math.max(1.2, bot - top);
    ctx.fillStyle = col;
    ctx.fillRect(Math.round(x - candleW / 2), top, candleW, h);
  }
}

function drawTurnoverBars(canvas, points, meta) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const W = canvas.width;
  const H = canvas.height;
  const bg = '#121622';
  const fg = '#ffffff';
  const muted = '#a0a0a0';
  const border = '#2a2f3a';
  const up = '#ff4d4f';
  const down = '#7ee787';

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = border;
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, W - 1, H - 1);

  const padL = 64, padR = 18, padT = 28, padB = 24;
  const plotL = padL, plotR = W - padR, plotT = padT, plotB = H - padB;

  const rows = points
    .map(p => ({
      t: parseTime(p.t),
      x: asNumber(p.x),
      y: asNumber(p.y),
    }))
    .filter(p => p.t && !Number.isNaN(p.x) && !Number.isNaN(p.y));
  if (rows.length === 0) return;

  const xs = rows.map(r => r.x);
  let ymax = Math.max(...xs);
  if (!Number.isFinite(ymax) || ymax <= 0) ymax = 1;

  function x2px(i) {
    if (rows.length <= 1) return (plotL + plotR) / 2;
    return plotL + (i / (rows.length - 1)) * (plotR - plotL);
  }
  function v2py(v) {
    return plotB - (v / ymax) * (plotB - plotT);
  }

  // Axes
  ctx.strokeStyle = border;
  ctx.beginPath();
  ctx.moveTo(plotL + 0.5, plotT);
  ctx.lineTo(plotL + 0.5, plotB);
  ctx.lineTo(plotR, plotB + 0.5);
  ctx.stroke();

  // Title
  ctx.fillStyle = fg;
  ctx.font = '700 14px system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif';
  ctx.fillText(meta.title || '换手率(%)', 16, 18);
  ctx.fillStyle = muted;
  ctx.font = '12px system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif';
  ctx.fillText('每分钟换手率柱状图', 16, 32);

  // Y tick labels (0 and max)
  const fmtV = (v) => {
    if (!Number.isFinite(v)) return '';
    if (v >= 1) return v.toFixed(3);
    return v.toFixed(4);
  };
  ctx.fillStyle = muted;
  ctx.font = '12px system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif';
  ctx.fillText(fmtV(ymax), 10, plotT + 10);
  ctx.fillText('0', 10, plotB + 4);

  // Bars
  const span = (plotR - plotL);
  const barW = Math.max(1, Math.min(6, Math.floor(span / Math.max(1, rows.length) * 0.7)));
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    const x = x2px(i);
    const y = v2py(r.x);
    const col = (r.y > 0) ? up : (r.y < 0 ? down : fg);
    ctx.fillStyle = col;
    ctx.fillRect(Math.round(x - barW / 2), y, barW, Math.max(1, plotB - y));
  }
}

function drawTrajectory(canvas, points, meta) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const W = canvas.width;
  const H = canvas.height;
  const bg = '#121622';
  const fg = '#ffffff';
  const muted = '#a0a0a0';
  const border = '#2a2f3a';
  const up = '#ff4d4f';
  const down = '#7ee787';

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = border;
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, W - 1, H - 1);

  const padL = 64, padR = 18, padT = 44, padB = 52;
  const plotL = padL, plotR = W - padR, plotT = padT, plotB = H - padB;

  // Interpret each minute as a vector v_t=(x_t,y_t), and place vectors head-to-tail.
  // Vertex path: P_0=(0,0), P_{k+1} = P_k + v_k.
  const raw = points
    .map(p => ({ t: parseTime(p.t), x: asNumber(p.x), y: asNumber(p.y) }))
    .filter(p => p.t && !Number.isNaN(p.x) && !Number.isNaN(p.y));
  if (raw.length === 0) return;

  let cx = 0;
  let cy = 0;
  const path = [];
  path.push({ t: raw[0].t, x: 0, y: 0, r: 0 });
  for (const p of raw) {
    cx += p.x;
    cy += p.y;
    path.push({ t: p.t, x: cx, y: cy, r: p.y });
  }

  const xs = path.map(p => p.x);
  const ys = path.map(p => p.y);
  let xmin = Math.min(...xs), xmax = Math.max(...xs);
  let ymin = Math.min(...ys), ymax = Math.max(...ys);
  if (xmin === xmax) { xmin -= 1; xmax += 1; }
  if (ymin === ymax) { ymin -= 1; ymax += 1; }

  // Center Y around 0 for readability
  const yAbs = Math.max(Math.abs(ymin), Math.abs(ymax));
  ymin = -yAbs;
  ymax = yAbs;

  function x2px(x) { return plotL + (x - xmin) / (xmax - xmin) * (plotR - plotL); }
  function y2py(y) { return plotB - (y - ymin) / (ymax - ymin) * (plotB - plotT); }

  // Axes
  ctx.strokeStyle = border;
  ctx.beginPath();
  ctx.moveTo(plotL + 0.5, plotT);
  ctx.lineTo(plotL + 0.5, plotB);
  ctx.lineTo(plotR, plotB + 0.5);
  ctx.stroke();

  // y=0 line
  if (ymin < 0 && ymax > 0) {
    const y0 = y2py(0);
    ctx.strokeStyle = '#2a2f3a';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(plotL, y0 + 0.5);
    ctx.lineTo(plotR, y0 + 0.5);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Labels
  ctx.fillStyle = fg;
  ctx.font = '700 16px system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif';
  ctx.fillText(meta.title || '2D 轨迹', 16, 24);
  ctx.fillStyle = muted;
  ctx.font = '12px system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif';
  ctx.fillText('向量首尾相接：P0=(0,0)，每分钟向量叠加', 16, 40);

  // Ticks (minimal)
  ctx.fillStyle = muted;
  const fmt = (v) => {
    if (!Number.isFinite(v)) return '';
    const av = Math.abs(v);
    if (av >= 10) return v.toFixed(2);
    if (av >= 1) return v.toFixed(3);
    return v.toFixed(4);
  };
  ctx.font = '12px system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif';
  const yTick = [ymin, 0, ymax].filter(v => v >= ymin && v <= ymax);
  for (const yt of yTick) {
    const py = y2py(yt);
    ctx.strokeStyle = border;
    ctx.beginPath();
    ctx.moveTo(plotL - 6, py + 0.5);
    ctx.lineTo(plotL, py + 0.5);
    ctx.stroke();
    ctx.fillText(fmt(yt), 10, py + 4);
  }

  // Path: split on large time gaps
  const GAP_MS = 15 * 60 * 1000;
  for (let i = 0; i < path.length - 1; i++) {
    const a = path[i];
    const b = path[i + 1];
    if ((b.t.getTime() - a.t.getTime()) > GAP_MS) {
      continue;
    }
    const ax = x2px(a.x), ay = y2py(a.y);
    const bx = x2px(b.x), by = y2py(b.y);
    const col = (b.r > 0) ? up : (b.r < 0 ? down : fg);
    ctx.strokeStyle = col;
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
  }

  // Start / End markers
  if (path.length > 0) {
    const s = path[0];
    const e = path[path.length - 1];
    const sx = x2px(s.x), sy = y2py(s.y);
    const ex = x2px(e.x), ey = y2py(e.y);

    ctx.fillStyle = fg;
    ctx.beginPath();
    ctx.arc(sx, sy, 4, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.rect(ex - 4, ey - 4, 8, 8);
    ctx.fill();
  }
}

async function main() {
  const symbol = getSymbol();
  const date = qs().get('date') || '';
  document.getElementById('title').textContent = `分钟轨迹 - ${symbol}`;
  document.getElementById('subtitle').textContent = date ? (`日期: ${date}`) : '';

  const status = document.getElementById('status');
  const kline = document.getElementById('kline');
  const turnbar = document.getElementById('turnbar');
  const canvas = document.getElementById('cv');

  if (!symbol || !date) {
    status.textContent = '缺少 symbol 或 date';
    status.className = 'err';
    return;
  }

  status.textContent = '加载中...';
  const url = `/api/minute_vectors?symbol=${encodeURIComponent(symbol)}&date=${encodeURIComponent(date)}`;
  const resp = await fetch(url);
  if (!resp.ok) {
    status.textContent = '加载失败: ' + resp.status;
    status.className = 'err';
    return;
  }
  const data = await resp.json();
  if (!data.ok) {
    status.textContent = '加载失败: ' + (data.error || 'unknown');
    status.className = 'err';
    return;
  }

  const points = data.points || [];
  status.textContent = `已加载 ${points.length} 分钟`;
  status.className = 'muted';
  drawCandles(kline, points, { title: `1分钟K线 - ${symbol} ${date}` });
  drawTurnoverBars(turnbar, points, { title: `换手率(%) - ${symbol} ${date}` });
  drawTrajectory(canvas, points, { title: `${symbol} ${date}` });
}

main();
</script>
</body>
</html>
"""


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from eastmoney_daily import default_data_dir, fetch_latest_daily_summary


def _ensure_dir(path: str) -> None:
  os.makedirs(path, exist_ok=True)


def _cache_dir(output_dir: str) -> str:
  return os.path.join(os.path.abspath(output_dir), "total-daily-view")


def _parse_cache_date_from_filename(name: str) -> str | None:
  # YYYY-MM-DD.csv
  if not name.endswith(".csv"):
    return None
  stem = name[:-4]
  if len(stem) != 10:
    return None
  if stem[4] != "-" or stem[7] != "-":
    return None
  y, m, d = stem[0:4], stem[5:7], stem[8:10]
  if not (y.isdigit() and m.isdigit() and d.isdigit()):
    return None
  return stem


def _find_latest_cache_file(output_dir: str) -> str | None:
  d = _cache_dir(output_dir)
  if not os.path.isdir(d):
    return None
  best_date = None
  best_path = None
  for fn in os.listdir(d):
    dt = _parse_cache_date_from_filename(fn)
    if not dt:
      continue
    if best_date is None or dt > best_date:
      best_date = dt
      best_path = os.path.join(d, fn)
  return best_path


def _load_rows_from_csv(path: str) -> list[dict]:
  if not path or not os.path.exists(path):
    return []
  with open(path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows: list[dict] = []
    for r in reader:
      if not r:
        continue
      rows.append(dict(r))
    return rows


def _save_rows_to_csv(path: str, rows: list[dict]) -> None:
  _ensure_dir(os.path.dirname(path))
  fieldnames = [
    "symbol",
    "symbol_name",
    "date",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "amount",
    "pctchg",
    "turnover",
    "chg",
    "amplitude",
  ]
  with open(path, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    for r in rows:
      w.writerow(r)


def _load_symbols_from_file(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return [str(x).strip() for x in obj if str(x).strip()]
    if isinstance(obj, dict) and isinstance(obj.get("symbols"), list):
        return [str(x).strip() for x in obj["symbols"] if str(x).strip()]
    return []


def _resolve_symbol_list(data_dir: str, mode: str) -> tuple[list[str], str]:
    data_dir = os.path.abspath(data_dir)
    total_path = os.path.join(data_dir, "total_gplist.json")
    self_path = os.path.join(data_dir, "self_gplist.json")

    m = (mode or "auto").strip().lower()
    if m == "total":
        return _load_symbols_from_file(total_path), total_path
    if m == "self":
        return _load_symbols_from_file(self_path), self_path

    # auto
    if os.path.exists(total_path):
        syms = _load_symbols_from_file(total_path)
        if syms:
            return syms, total_path
    if os.path.exists(self_path):
        return _load_symbols_from_file(self_path), self_path
    return [], ""


def create_app(data_dir: str, list_mode: str, threads: int, adj: str) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        # Single page: fetches JSON from /api/daily_summaries then renders a sortable table.
        return f"""<!doctype html>
<html lang=\"zh\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>gp-quant - 最近一天交易总结</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{ background: #0f1115; color: #eaeaea; font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 16px; }}
    .bar {{ display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin-bottom:12px; }}
    .muted {{ color: #a0a0a0; font-size: 12px; }}
    .panel {{ border: 1px solid #2a2f3a; background: #121622; padding: 10px; margin-bottom: 12px; }}
    .panel-title {{ font-weight: 700; margin-bottom: 8px; }}
    .stats-img {{ width: 100%; max-width: 1170px; height: auto; display:block; }}
    a.celllink {{ color: #ffffff; text-decoration: none; }}
    a.celllink:hover {{ text-decoration: underline; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #2a2f3a; padding: 8px; font-size: 13px; background: #121622; }}
    th {{ background: #1b1f2a; cursor: pointer; user-select:none; position: sticky; top: 0; }}
    .right {{ text-align: right; }}
    .loading {{ padding: 12px 0; }}
    .err {{ color: #b00020; }}
    td.fixedcol {{ background: #0f1115; color: #ffffff; font-weight: 600; white-space: nowrap; }}
    td.fixedwhite {{ color: #ffffff; white-space: nowrap; }}
    tr.up td:not(.fixedcol) {{ color: #ff4d4f; }}
    tr.down td:not(.fixedcol) {{ color: #66bb6a; }}
    tr.flat td:not(.fixedcol) {{ color: #ffffff; }}
    tr.up td.fixedwhite, tr.down td.fixedwhite, tr.flat td.fixedwhite {{ color: #ffffff; }}
  </style>
</head>
<body>
  <div class=\"bar\">
    <div><strong>最近一天交易总结</strong></div>
    <div class=\"muted\">数据源：东方财富（日线 klt=101）。默认按股票 id 排序；点击表头可排序。</div>
  </div>

  <div id=\"stats\" class=\"panel\" hidden>
    <div class=\"panel-title\">市场涨跌分布</div>
    <img id=\"statsImg\" class=\"stats-img\" alt=\"市场涨跌分布\" />
    <canvas id=\"statsCanvas\" width=\"1170\" height=\"210\" hidden></canvas>
  </div>

  <div id=\"status\" class=\"loading\">加载中...</div>
  <div style=\"overflow:auto; max-height: calc(100vh - 140px);\">
    <table id=\"tbl\" hidden>
      <thead>
        <tr>
          <th data-key=\"symbol\">股票id</th>
          <th data-key=\"symbol_name\">股票名</th>
          <th data-key=\"date\">日期</th>
          <th class=\"right\" data-key=\"open\">开盘</th>
          <th class=\"right\" data-key=\"close\">收盘</th>
          <th class=\"right\" data-key=\"high\">最高</th>
          <th class=\"right\" data-key=\"low\">最低</th>
          <th class=\"right\" data-key=\"volume\">成交量(手)</th>
          <th class=\"right\" data-key=\"amount\">成交额(元)</th>
          <th class=\"right\" data-key=\"pctchg\">涨跌幅(%)</th>
          <th class=\"right\" data-key=\"turnover\">换手率(%)</th>
        </tr>
      </thead>
      <tbody></tbody>
    </table>
  </div>

<script>
const state = {{
  rows: [],
  sortKey: 'symbol',
  sortAsc: true,
}};

function fmt(v) {{
  if (v === null || v === undefined) return '';
  if (typeof v === 'number') {{
    if (!Number.isFinite(v)) return '';
    return v.toString();
  }}
  return String(v);
}}

function asNumber(v) {{
  if (v === null || v === undefined || v === '') return NaN;
  const n = Number(v);
  return Number.isFinite(n) ? n : NaN;
}}

function limitPctForSymbol(symbol) {{
  const s = (symbol || '').toString().toLowerCase();
  if (s.startsWith('sh688') || s.startsWith('sz300') || s.startsWith('sz301')) return 20;
  if (s.startsWith('bj')) return 30;
  return 10;
}}

function isSuspendedRow(r) {{
  const v = asNumber(r.volume);
  if (!Number.isNaN(v) && v === 0) return true;
  const amt = asNumber(r.amount);
  if (!Number.isNaN(amt) && amt === 0) return true;
  return false;
}}

function computeStats(rows) {{
  const stats = {{
    total: 0,
    nonSuspended: 0,
    suspended: 0,
    up: 0,
    flat: 0,
    down: 0,
    limitUp: 0,
    limitDown: 0,
    buckets: [
      {{ key: 'limitUp', label: '涨停', side: 'up', count: 0 }},
      {{ key: 'gt7', label: '>7', side: 'up', count: 0 }},
      {{ key: '7_5', label: '7~5', side: 'up', count: 0 }},
      {{ key: '5_2', label: '5~2', side: 'up', count: 0 }},
      {{ key: '2_0', label: '2~0', side: 'up', count: 0 }},
      {{ key: '0', label: '0', side: 'flat', count: 0 }},
      {{ key: '0_-2', label: '0~-2', side: 'down', count: 0 }},
      {{ key: '-2_-5', label: '-2~-5', side: 'down', count: 0 }},
      {{ key: '-5_-7', label: '-5~-7', side: 'down', count: 0 }},
      {{ key: 'lt-7', label: '-7<', side: 'down', count: 0 }},
      {{ key: 'limitDown', label: '跌停', side: 'down', count: 0 }},
    ],
  }};

  stats.total = rows.length;
  const bucketIndex = Object.create(null);
  for (let i = 0; i < stats.buckets.length; i++) bucketIndex[stats.buckets[i].key] = i;

  for (const r of rows) {{
    if (isSuspendedRow(r)) {{
      stats.suspended += 1;
      continue;
    }}

    const pct = asNumber(r.pctchg);
    if (Number.isNaN(pct)) continue;
    stats.nonSuspended += 1;

    const lim = limitPctForSymbol(r.symbol);
    const tol = 0.05;
    const isLimitUp = pct >= (lim - tol);
    const isLimitDown = pct <= (-lim + tol);

    if (isLimitUp) {{
      stats.limitUp += 1;
      stats.buckets[bucketIndex['limitUp']].count += 1;
      stats.up += 1;
      continue;
    }}
    if (isLimitDown) {{
      stats.limitDown += 1;
      stats.buckets[bucketIndex['limitDown']].count += 1;
      stats.down += 1;
      continue;
    }}

    if (Math.abs(pct) < 1e-9) {{
      stats.flat += 1;
      stats.buckets[bucketIndex['0']].count += 1;
      continue;
    }}
    if (pct > 0) {{
      stats.up += 1;
      if (pct > 7) stats.buckets[bucketIndex['gt7']].count += 1;
      else if (pct > 5) stats.buckets[bucketIndex['7_5']].count += 1;
      else if (pct > 2) stats.buckets[bucketIndex['5_2']].count += 1;
      else stats.buckets[bucketIndex['2_0']].count += 1;
      continue;
    }}

    // pct < 0
    stats.down += 1;
    if (pct < -7) stats.buckets[bucketIndex['lt-7']].count += 1;
    else if (pct < -5) stats.buckets[bucketIndex['-5_-7']].count += 1;
    else if (pct < -2) stats.buckets[bucketIndex['-2_-5']].count += 1;
    else stats.buckets[bucketIndex['0_-2']].count += 1;
  }}

  return stats;
}}

function renderStatsImage(stats) {{
  const root = document.getElementById('stats');
  const img = document.getElementById('statsImg');
  const canvas = document.getElementById('statsCanvas');
  if (!root || !img || !canvas) return;
  root.hidden = false;

  const W = 1170;
  const H = 210;
  const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
  canvas.width = Math.round(W * dpr);
  canvas.height = Math.round(H * dpr);
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  // Colors (match existing page theme)
  const bg = '#121622';
  const fg = '#ffffff';
  const muted = '#a0a0a0';
  const border = '#2a2f3a';
  const up = '#ff4d4f';
  const down = '#66bb6a';

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = border;
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, W - 1, H - 1);

  ctx.fillStyle = fg;
  ctx.font = '700 18px system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif';
  ctx.fillText('市场涨跌分布', 18, 28);
  ctx.fillStyle = muted;
  ctx.font = '12px system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif';
  ctx.fillText('基于当前列表样本统计', 18, 48);

  let maxCount = 0;
  for (const b of stats.buckets) {{
    if (typeof b.count === 'number' && b.count > maxCount) maxCount = b.count;
  }}

  const chartLeft = 18;
  const chartRight = W - 18;
  const chartTop = 68;
  const chartBottom = 140;
  const labelY = 160;
  const step = (chartRight - chartLeft) / stats.buckets.length;
  const barW = Math.min(30, Math.max(12, Math.floor(step * 0.58)));

  // Baseline
  ctx.strokeStyle = border;
  ctx.beginPath();
  ctx.moveTo(chartLeft, chartBottom + 0.5);
  ctx.lineTo(chartRight, chartBottom + 0.5);
  ctx.stroke();

  // Bars + counts + labels
  for (let i = 0; i < stats.buckets.length; i++) {{
    const b = stats.buckets[i];
    const cx = chartLeft + step * (i + 0.5);
    const x = Math.round(cx - barW / 2);
    const ratio = maxCount > 0 ? (b.count / maxCount) : 0;
    const h = Math.round((chartBottom - chartTop) * ratio);
    const y = chartBottom - h;

    const color = (b.side === 'up') ? up : (b.side === 'down' ? down : fg);
    ctx.fillStyle = color;
    ctx.fillRect(x, y, barW, Math.max(2, h));

    // count text
    ctx.font = '700 13px system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif';
    ctx.fillStyle = color;
    const cnt = String(b.count);
    const tw = ctx.measureText(cnt).width;
    ctx.fillText(cnt, Math.round(cx - tw / 2), y - 6);

    // label
    ctx.font = '12px system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif';
    ctx.fillStyle = muted;
    const lab = String(b.label);
    const lw = ctx.measureText(lab).width;
    ctx.fillText(lab, Math.round(cx - lw / 2), labelY);
  }}

  // Summary line (colored segments)
  const segs = [
    {{ t: `上涨 ${{stats.up}}家`, c: up }},
    {{ t: `平盘 ${{stats.flat}}家`, c: fg }},
    {{ t: `下跌 ${{stats.down}}家`, c: down }},
    {{ t: `涨停 ${{stats.limitUp}}家`, c: up }},
    {{ t: `停牌 ${{stats.suspended}}家`, c: muted }},
    {{ t: `跌停 ${{stats.limitDown}}家`, c: down }},
  ];
  ctx.font = '13px system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif';
  let x = 18;
  const y = 192;
  for (const s of segs) {{
    ctx.fillStyle = s.c;
    ctx.fillText(s.t, x, y);
    x += ctx.measureText(s.t).width + 18;
  }}

  img.src = canvas.toDataURL('image/png');
}}

function renderStats(stats) {{
  renderStatsImage(stats);
}}

function compare(a, b, key, asc) {{
  const av = a[key];
  const bv = b[key];

  if (key === 'symbol' || key === 'symbol_name' || key === 'date') {{
    const sa = (av ?? '').toString();
    const sb = (bv ?? '').toString();
    if (sa < sb) return asc ? -1 : 1;
    if (sa > sb) return asc ? 1 : -1;
    return 0;
  }}

  const na = asNumber(av);
  const nb = asNumber(bv);
  const aNan = Number.isNaN(na);
  const bNan = Number.isNaN(nb);
  if (aNan && bNan) return 0;
  if (aNan) return asc ? 1 : -1;
  if (bNan) return asc ? -1 : 1;
  if (na < nb) return asc ? -1 : 1;
  if (na > nb) return asc ? 1 : -1;
  return 0;
}}

function render() {{
  const tbody = document.querySelector('#tbl tbody');
  tbody.innerHTML = '';

  const rows = [...state.rows].sort((a,b) => compare(a,b,state.sortKey,state.sortAsc));
  for (const r of rows) {{
    const tr = document.createElement('tr');

    const chg = asNumber(r.chg);
    const pct = asNumber(r.pctchg);
    if (!Number.isNaN(chg)) {{
      tr.className = chg > 0 ? 'up' : (chg < 0 ? 'down' : 'flat');
    }} else if (!Number.isNaN(pct)) {{
      tr.className = pct > 0 ? 'up' : (pct < 0 ? 'down' : 'flat');
    }} else {{
      tr.className = 'flat';
    }}

    const symHref = `/stock/${{encodeURIComponent(fmt(r.symbol))}}?date=${{encodeURIComponent(fmt(r.date))}}`;
    tr.innerHTML = `
      <td class=\"fixedcol\"><a class=\"celllink\" href=\"${{symHref}}\">${{fmt(r.symbol)}}</a></td>
      <td class=\"fixedcol\"><a class=\"celllink\" href=\"${{symHref}}\">${{fmt(r.symbol_name)}}</a></td>
      <td class=\"fixedwhite\">${{fmt(r.date)}}</td>
      <td class=\"right\">${{fmt(r.open)}}</td>
      <td class=\"right\">${{fmt(r.close)}}</td>
      <td class=\"right\">${{fmt(r.high)}}</td>
      <td class=\"right\">${{fmt(r.low)}}</td>
      <td class=\"right\">${{fmt(r.volume)}}</td>
      <td class=\"right\">${{fmt(r.amount)}}</td>
      <td class=\"right\">${{fmt(r.pctchg)}}</td>
      <td class=\"right\">${{fmt(r.turnover)}}</td>
    `;
    tbody.appendChild(tr);
  }}
}}

function setStatus(msg, isErr=false) {{
  const el = document.getElementById('status');
  el.textContent = msg;
  el.className = isErr ? 'loading err' : 'loading';
}}

async function load() {{
  setStatus('加载中...');
  const resp = await fetch('/api/daily_summaries');
  if (!resp.ok) {{
    setStatus('加载失败: ' + resp.status, true);
    return;
  }}
  const data = await resp.json();
  if (!data.ok) {{
    setStatus('加载失败: ' + (data.error || 'unknown'), true);
    return;
  }}
  state.rows = data.rows || [];
  renderStats(computeStats(state.rows));
  document.getElementById('tbl').hidden = false;
  setStatus(`已加载 ${{state.rows.length}} 条`);
  render();
}}

function initSort() {{
  document.querySelectorAll('th[data-key]').forEach(th => {{
    th.addEventListener('click', () => {{
      const key = th.getAttribute('data-key');
      if (state.sortKey === key) {{
        state.sortAsc = !state.sortAsc;
      }} else {{
        state.sortKey = key;
        state.sortAsc = true;
      }}
      render();
    }});
  }});
}}

initSort();
load();
</script>
</body>
</html>"""

    @app.get("/api/daily_summaries")
    def api_daily_summaries():
        nonlocal data_dir, list_mode, threads, adj

        # Default behavior: use total list (no --list flag needed).
        # If total list is missing/empty, fall back to self list.
        effective_mode = (list_mode or "total").strip().lower()
        if effective_mode == "auto":
            effective_mode = "total"

        syms, from_file = _resolve_symbol_list(data_dir, effective_mode)
        if not syms and effective_mode != "self":
            syms, from_file = _resolve_symbol_list(data_dir, "self")
            if syms:
                effective_mode = "self"

        if not syms:
            return jsonify(
                {
                    "ok": False,
                    "error": f"No symbols found. Checked data_dir={data_dir} file={from_file or '(none)'}",
                    "rows": [],
                }
            )

        # CSV cache: <output_dir>/total-daily-view/YYYY-MM-DD.csv
        # Use benchmark symbol to determine the latest trading day; if cache exists, return it.
        cache_used = False
        cache_file = None
        benchmark_date = None
        t0 = time.time()
        try:
          bench = fetch_latest_daily_summary(syms[0], adj=adj)
          if bench and bench.get("date"):
            benchmark_date = str(bench.get("date"))
        except Exception:
          benchmark_date = None

        cdir = _cache_dir(data_dir)
        if benchmark_date:
          candidate = os.path.join(cdir, f"{benchmark_date}.csv")
          if os.path.exists(candidate):
            rows = _load_rows_from_csv(candidate)
            if rows:
              cache_used = True
              cache_file = candidate
              rows.sort(key=lambda x: str(x.get("symbol") or ""))
              return jsonify(
                {
                  "ok": True,
                  "meta": {
                    "data_dir": os.path.abspath(data_dir),
                    "symbols": len(syms),
                    "threads": 0,
                    "adj": adj,
                    "list_mode": effective_mode,
                    "list_file": from_file,
                    "errors": 0,
                    "cache_used": True,
                    "cache_file": cache_file,
                    "benchmark_date": benchmark_date,
                    "elapsed_ms": int((time.time() - t0) * 1000),
                  },
                  "rows": rows,
                  "errors": [],
                }
              )

        # Fallback: if any cache exists, use the latest one.
        latest_cache = _find_latest_cache_file(data_dir)
        if latest_cache and os.path.exists(latest_cache):
          rows = _load_rows_from_csv(latest_cache)
          if rows:
            cache_used = True
            cache_file = latest_cache
            rows.sort(key=lambda x: str(x.get("symbol") or ""))
            return jsonify(
              {
                "ok": True,
                "meta": {
                  "data_dir": os.path.abspath(data_dir),
                  "symbols": len(syms),
                  "threads": 0,
                  "adj": adj,
                  "list_mode": effective_mode,
                  "list_file": from_file,
                  "errors": 0,
                  "cache_used": True,
                  "cache_file": cache_file,
                  "benchmark_date": benchmark_date,
                  "elapsed_ms": int((time.time() - t0) * 1000),
                },
                "rows": rows,
                "errors": [],
              }
            )

        max_workers = max(1, int(threads))
        rows: list[dict] = []
        errors: list[dict] = []

        def _work(sym: str):
            s = sym.strip()
            if not s:
                return None
            try:
                r = fetch_latest_daily_summary(s, adj=adj)
                if not r:
                    return {"symbol": s, "_error": "No data"}
                return r
            except Exception as e:
                return {"symbol": s, "_error": f"{type(e).__name__}: {str(e)[:180]}"}

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_work, s) for s in syms]
            for fut in as_completed(futures):
                r = fut.result()
                if not r:
                    continue
                if r.get("_error"):
                    errors.append(r)
                    continue
                rows.append(r)

        rows.sort(key=lambda x: str(x.get("symbol") or ""))

        # Persist cache using the most common date (usually the same for all symbols).
        date_for_file = benchmark_date
        if not date_for_file and rows:
          counts: dict[str, int] = {}
          for r in rows:
            d = str(r.get("date") or "")
            if not d:
              continue
            counts[d] = counts.get(d, 0) + 1
          if counts:
            date_for_file = max(counts.items(), key=lambda kv: kv[1])[0]
        if date_for_file:
          cache_file = os.path.join(_cache_dir(data_dir), f"{date_for_file}.csv")
          try:
            _save_rows_to_csv(cache_file, rows)
          except Exception:
            cache_file = None

        return jsonify(
            {
                "ok": True,
                "meta": {
                    "data_dir": os.path.abspath(data_dir),
                    "symbols": len(syms),
                    "threads": max_workers,
                    "adj": adj,
                    "list_mode": effective_mode,
                    "list_file": from_file,
                    "errors": len(errors),
              "cache_used": cache_used,
              "cache_file": cache_file,
              "benchmark_date": benchmark_date,
              "elapsed_ms": int((time.time() - t0) * 1000),
                },
                "rows": rows,
                "errors": errors,
            }
        )

    @app.get("/stock/<symbol>")
    def stock_page(symbol: str):
        # static HTML; symbol/date are parsed on client from URL
        return STOCK_PAGE_HTML

    @app.get("/api/minute_vectors")
    def api_minute_vectors():
        nonlocal data_dir
        symbol = (request.args.get("symbol") or "").strip()
        date = (request.args.get("date") or "").strip()
        if not symbol or not date:
            return jsonify({"ok": False, "error": "symbol and date are required", "points": []})

        # Expected location: <output_dir>/trade/<symbol>/<YYYY-MM-DD>.csv
        path = os.path.join(os.path.abspath(data_dir), "trade", symbol, f"{date}.csv")
        if not os.path.exists(path):
            return jsonify({"ok": False, "error": f"minute csv not found: {path}", "points": []})

        def sf(v):
            try:
                return float(v)
            except Exception:
                return None

        points: list[dict] = []
        prev_close = None
        with open(path, "r", encoding="utf-8") as f:
          reader = csv.DictReader(f)
          for row in reader:
            if not row:
              continue

            t = (row.get("时间") or "").strip()
            turnover = sf(row.get("换手率(%)"))
            open_ = sf(row.get("开盘"))
            high_ = sf(row.get("最高"))
            low_ = sf(row.get("最低"))
            close_ = sf(row.get("收盘"))
            if turnover is None or open_ is None or high_ is None or low_ is None or close_ is None or not t:
              continue

            if prev_close is None or prev_close == 0:
              ret = 0.0
            else:
              ret = (close_ / prev_close - 1.0) * 100.0
            prev_close = close_

            points.append({"t": t, "x": turnover, "y": ret, "o": open_, "h": high_, "l": low_, "c": close_})

        return jsonify(
            {
                "ok": True,
                "meta": {"symbol": symbol, "date": date, "file": path, "points": len(points)},
                "points": points,
            }
        )

    @app.get("/healthz")
    def healthz():
        return jsonify({"ok": True})

    return app


def main() -> None:
  p = argparse.ArgumentParser(description="gp-quant web UI")
  p.add_argument("--host", default="0.0.0.0")
  p.add_argument("--port", type=int, default=30200)

  # output_dir is used to locate total_gplist.json/self_gplist.json and to store cache in total-daily-view/
  p.add_argument("--data_dir", default=None, help="Alias of --output-dir")
  p.add_argument(
    "--output-dir",
    dest="output_dir",
    default=None,
    help="Directory that contains total_gplist.json (and is used to store total-daily-view cache)",
  )
  # Backward compatible flag
  p.add_argument("--output_dir", dest="output_dir")

  # Kept for compatibility; default is total so you generally don't need to pass it.
  p.add_argument("--list", dest="list_mode", default="total", choices=["auto", "total", "self"])
  p.add_argument("--threads", type=int, default=20)
  p.add_argument("--adj", default="none", choices=["none", "qfq", "hfq"])
  args = p.parse_args()

  out_dir = args.output_dir or args.data_dir or default_data_dir()
  app = create_app(out_dir, args.list_mode, args.threads, args.adj)
  app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
