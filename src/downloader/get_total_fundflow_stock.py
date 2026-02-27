import argparse
import csv
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from eastmoney_universe import symbol_to_em_secid

_WEB_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "web"))
if _WEB_DIR not in sys.path:
    sys.path.insert(0, _WEB_DIR)

from eastmoney_daily import default_data_dir, fetch_latest_daily_summary
from csv_utils import write_rows_csv


_HTTP = requests.Session()
if os.getenv("GP_NO_PROXY", "").strip().lower() in {"1", "true", "yes"}:
    _HTTP.trust_env = False


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _out_dir(output_dir: str) -> str:
    return os.path.join(os.path.abspath(output_dir), "total-fundflow-stock")


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
        syms = _load_symbols_from_file(self_path)
        if syms:
            return syms, self_path
    return [], ""


def _em_headers() -> dict:
    return {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Referer": "https://quote.eastmoney.com/",
        "Accept": "application/json,text/plain,*/*",
    }


def _get_json_with_retry(url: str, params: dict, timeout: int, attempts: int = 3) -> dict:
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            r = _HTTP.get(url, params=params, headers=_em_headers(), timeout=timeout)
            r.raise_for_status()
            return r.json() or {}
        except Exception as e:
            last_exc = e
            time.sleep(0.6 * (i + 1) + random.uniform(0.0, 0.5))
    raise last_exc or RuntimeError("request failed")


def fetch_latest_stock_fundflow(symbol: str) -> dict | None:
    """Fetch latest daily fundflow for a single stock.

    Uses Eastmoney endpoint:
      https://push2.eastmoney.com/api/qt/stock/fflow/kline/get

    Returns fields (all amounts in Yuan):
      date, main_net_in, super_net_in, large_net_in, medium_net_in, small_net_in

    Note: verified with sample where main ~= super + large.
    """

    url = "https://push2.eastmoney.com/api/qt/stock/fflow/kline/get"
    params = {
        "lmt": 1,
        "klt": 101,  # daily
        "fields1": "f1,f2,f3,f7",
        "fields2": "f51,f52,f53,f54,f55,f56",
        "secid": symbol_to_em_secid(symbol),
    }
    j = _get_json_with_retry(url, params, timeout=20, attempts=3)
    data = j.get("data") or {}
    klines = data.get("klines") or []
    if not klines:
        return None

    # line format: date, main, small, medium, large, super
    parts = str(klines[-1]).split(",")
    if len(parts) < 6:
        return None

    def sf(x):
        try:
            return float(x)
        except Exception:
            return None

    return {
        "symbol": symbol,
        "symbol_name": data.get("name"),
        "date": parts[0],
        "main_net_in": sf(parts[1]),
        "small_net_in": sf(parts[2]),
        "medium_net_in": sf(parts[3]),
        "large_net_in": sf(parts[4]),
        "super_net_in": sf(parts[5]),
    }


def _save_rows(path: str, rows: list[dict]) -> None:
    fieldnames = [
        "symbol",
        "symbol_name",
        "date",
        "main_net_in",
        "super_net_in",
        "large_net_in",
        "medium_net_in",
        "small_net_in",
    ]
    write_rows_csv(path, fieldnames=fieldnames, rows=rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Download latest daily fund flow for stocks and cache as CSV")
    p.add_argument("--data_dir", default=None, help="Alias of --output-dir")
    p.add_argument("--output-dir", dest="output_dir", default=None)
    p.add_argument("--output_dir", dest="output_dir", default=None)
    p.add_argument("--list", dest="list_mode", default="total", choices=["auto", "total", "self"])
    p.add_argument("--threads", type=int, default=24)
    p.add_argument("--force", action="store_true", help="Force refresh even if cache exists")
    args = p.parse_args()

    out_dir = args.output_dir or args.data_dir or default_data_dir()
    out_dir = os.path.abspath(out_dir)

    syms, from_file = _resolve_symbol_list(out_dir, args.list_mode)
    effective_mode = (args.list_mode or "auto").strip().lower()
    if not syms and effective_mode != "self":
        syms, from_file = _resolve_symbol_list(out_dir, "self")
        if syms:
            effective_mode = "self"

    if not syms:
        raise SystemExit(
            f"No symbols found. Checked output_dir={out_dir} file={from_file or '(none)'}; "
            "please run total download first (to generate total_gplist.json) or create self_gplist.json."
        )

    # Determine latest trading date.
    benchmark_date = None
    try:
        bench = fetch_latest_daily_summary(syms[0])
        if bench and bench.get("date"):
            benchmark_date = str(bench.get("date"))
    except Exception:
        benchmark_date = None
    if not benchmark_date:
        benchmark_date = datetime.today().strftime("%Y-%m-%d")

    out_path = os.path.join(_out_dir(out_dir), f"{benchmark_date}.csv")
    if os.path.exists(out_path) and not args.force:
        print(f"Cache exists, skip: {out_path}")
        raise SystemExit(0)

    t0 = time.time()
    max_workers = max(1, int(args.threads))
    rows: list[dict] = []
    errors: list[dict] = []

    def _work(sym: str) -> dict:
        s = sym.strip()
        if not s:
            return {"symbol": sym, "_error": "empty"}
        time.sleep(random.uniform(0.01, 0.08))
        try:
            r = fetch_latest_stock_fundflow(s)
            if not r:
                return {"symbol": s, "date": benchmark_date, "_error": "No data"}
            # normalize date if API returns different date
            if r.get("date"):
                r["date"] = str(r.get("date"))
            else:
                r["date"] = benchmark_date
            return r
        except Exception as e:
            return {"symbol": s, "date": benchmark_date, "_error": f"{type(e).__name__}: {str(e)[:160]}"}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_work, s) for s in syms]
        try:
            for fut in as_completed(futures):
                r = fut.result()
                if r.get("_error"):
                    errors.append(r)
                rows.append(r)
        except KeyboardInterrupt:
            try:
                ex.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            raise

    rows.sort(key=lambda x: str(x.get("symbol") or ""))
    _save_rows(out_path, rows)

    meta = {
        "ok": True,
        "output_dir": out_dir,
        "symbols": len(syms),
        "threads": max_workers,
        "list_mode": effective_mode,
        "list_file": from_file,
        "date": benchmark_date,
        "cache_file": out_path,
        "errors": len(errors),
        "elapsed_ms": int((time.time() - t0) * 1000),
    }
    print(json.dumps(meta, ensure_ascii=False))

    if errors:
        print(f"Errors (showing up to 10/{len(errors)}):")
        for e in errors[:10]:
            print({k: e.get(k) for k in ["symbol", "symbol_name", "date", "_error"]})


if __name__ == "__main__":
    main()
