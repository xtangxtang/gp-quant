import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import datetime

import requests


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

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
    return os.path.join(os.path.abspath(output_dir), "total-fundflow-industry")


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
            time.sleep(0.7 * (i + 1) + random.uniform(0.0, 0.5))
    raise last_exc or RuntimeError("request failed")


def fetch_all_industry_fundflows() -> list[dict]:
    """Fetch latest fund flow snapshot for all industry boards (行业板块)."""

    url = "https://push2.eastmoney.com/api/qt/clist/get"
    fields = "f12,f14,f2,f3,f62,f66,f69,f72,f75,f78,f81,f84,f87,f184,f204,f205,f206"

    # Industry board list
    fs = "m:90+t:2"

    # page size capped at 100
    pz = 100
    pn = 1
    out: list[dict] = []

    while True:
        params = {
            "pn": pn,
            "pz": pz,
            "po": 1,
            "np": 1,
            "ut": "bd1d9ddb04089700cf9c27f6f7426281",
            "fltt": 2,
            "invt": 2,
            "fid": "f62",
            "fs": fs,
            "fields": fields,
        }
        j = _get_json_with_retry(url, params, timeout=20, attempts=3)
        data = j.get("data") or {}
        diff = data.get("diff") or []
        if not isinstance(diff, list) or not diff:
            break

        out.extend(diff)
        total = int(data.get("total") or 0)
        if pn * pz >= total:
            break
        pn += 1

        # jitter
        time.sleep(random.uniform(0.05, 0.15))

    return out


def _save_rows(path: str, rows: list[dict]) -> None:
    # field meanings (common in Eastmoney):
    # f62: 主力净流入, f66: 超大单净流入, f72: 大单净流入, f78: 中单净流入, f84: 小单净流入
    fieldnames = [
        "date",
        "industry_code",
        "industry_name",
        "price",
        "pctchg",
        "main_net_in",
        "super_net_in",
        "large_net_in",
        "medium_net_in",
        "small_net_in",
        "main_net_in_pct",
        "super_net_in_pct",
        "large_net_in_pct",
        "medium_net_in_pct",
        "small_net_in_pct",
        "strength",
        "top_stock",
        "top_stock_code",
    ]

    def g(d: dict, k: str):
        return d.get(k)

    mapped_rows: list[dict] = []
    for r in rows:
        mapped_rows.append(
            {
                "date": r.get("date"),
                "industry_code": g(r, "f12"),
                "industry_name": g(r, "f14"),
                "price": g(r, "f2"),
                "pctchg": g(r, "f3"),
                "main_net_in": g(r, "f62"),
                "super_net_in": g(r, "f66"),
                "large_net_in": g(r, "f72"),
                "medium_net_in": g(r, "f78"),
                "small_net_in": g(r, "f84"),
                "main_net_in_pct": g(r, "f69"),
                "super_net_in_pct": g(r, "f75"),
                "large_net_in_pct": g(r, "f81"),
                "medium_net_in_pct": g(r, "f87"),
                "small_net_in_pct": g(r, "f184"),
                "strength": g(r, "f206"),
                "top_stock": g(r, "f204"),
                "top_stock_code": g(r, "f205"),
            }
        )

    write_rows_csv(path, fieldnames=fieldnames, rows=mapped_rows)


def _load_symbols_for_date(output_dir: str) -> list[str]:
    # Use total list first.
    total_path = os.path.join(os.path.abspath(output_dir), "total_gplist.json")
    if os.path.exists(total_path):
        try:
            with open(total_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                return [str(x).strip() for x in obj if str(x).strip()]
            if isinstance(obj, dict) and isinstance(obj.get("symbols"), list):
                return [str(x).strip() for x in obj["symbols"] if str(x).strip()]
        except Exception:
            return []
    self_path = os.path.join(os.path.abspath(output_dir), "self_gplist.json")
    if os.path.exists(self_path):
        try:
            with open(self_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                return [str(x).strip() for x in obj if str(x).strip()]
        except Exception:
            return []
    return []


def main() -> None:
    p = argparse.ArgumentParser(description="Download latest industry fund flow snapshot and cache as CSV")
    p.add_argument("--data_dir", default=None, help="Alias of --output-dir")
    p.add_argument("--output-dir", dest="output_dir", default=None)
    p.add_argument("--output_dir", dest="output_dir", default=None)
    p.add_argument("--force", action="store_true", help="Force refresh even if cache exists")
    args = p.parse_args()

    out_dir = args.output_dir or args.data_dir or default_data_dir()
    out_dir = os.path.abspath(out_dir)

    # Determine latest trading date using a benchmark symbol if possible.
    syms = _load_symbols_for_date(out_dir)
    benchmark_date = None
    if syms:
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
    rows = fetch_all_industry_fundflows()
    for r in rows:
        r["date"] = benchmark_date

    # sort by main net inflow desc
    def _sf(x):
        try:
            return float(x)
        except Exception:
            return 0.0

    rows.sort(key=lambda r: _sf(r.get("f62")), reverse=True)
    _save_rows(out_path, rows)

    meta = {
        "ok": True,
        "output_dir": out_dir,
        "industries": len(rows),
        "date": benchmark_date,
        "cache_file": out_path,
        "elapsed_ms": int((time.time() - t0) * 1000),
    }
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
