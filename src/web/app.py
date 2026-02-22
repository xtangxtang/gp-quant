import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from flask import Flask, jsonify


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

    tr.innerHTML = `
      <td class=\"fixedcol\">${{fmt(r.symbol)}}</td>
      <td class=\"fixedcol\">${{fmt(r.symbol_name)}}</td>
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
