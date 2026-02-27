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


def _fund_dir(output_dir: str) -> str:
    return os.path.join(os.path.abspath(output_dir), "total-fundamentals")


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


def _em_headers() -> dict:
    return {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Referer": "https://quote.eastmoney.com/",
        "Accept": "application/json,text/plain,*/*",
    }


def _symbol_to_f10_code(symbol: str) -> str:
    s = (symbol or "").strip().lower()
    if s.startswith("sh"):
        return "SH" + s[2:]
    if s.startswith("sz"):
        return "SZ" + s[2:]
    if s.startswith("bj"):
        return "BJ" + s[2:]
    # fallback by first digit
    if s.startswith("6") or s.startswith("9"):
        return "SH" + s
    return "SZ" + s


def _safe_float(v):
    try:
        if v in (None, "-", "", "None"):
            return None
        return float(v)
    except Exception:
        return None


def _safe_int(v):
    try:
        if v in (None, "-", "", "None"):
            return None
        return int(float(v))
    except Exception:
        return None


def _scale_price_x100(v):
    x = _safe_float(v)
    if x is None:
        return None
    # Eastmoney quote price fields are usually *100
    return x / 100.0


def _scale_ratio_x100(v):
    x = _safe_float(v)
    if x is None:
        return None
    return x / 100.0


def _get_json_with_retry(url: str, params: dict, timeout: int, attempts: int = 3) -> dict:
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            r = _HTTP.get(url, params=params, headers=_em_headers(), timeout=timeout)
            r.raise_for_status()
            return r.json() or {}
        except Exception as e:
            last_exc = e
            # backoff + jitter
            time.sleep(0.6 * (i + 1) + random.uniform(0.0, 0.4))
    raise last_exc or RuntimeError("request failed")


def fetch_valuation_snapshot(symbol: str) -> dict | None:
    url = "https://push2.eastmoney.com/api/qt/stock/get"
    fields = ",".join(
        [
            "f57",  # code
            "f58",  # name
            "f43",  # last price *100
            "f60",  # prev close *100
            "f44",  # high *100
            "f45",  # low *100
            "f46",  # open *100
            "f116",  # total mkt cap
            "f117",  # float mkt cap
            "f162",  # PE TTM (?) *100
            "f167",  # PB (?) *100
            "f127",  # industry
            "f128",  # area
            "f129",  # concepts tags
            "f189",  # IPO date YYYYMMDD
        ]
    )
    params = {"secid": symbol_to_em_secid(symbol), "fields": fields}
    j = _get_json_with_retry(url, params, timeout=20, attempts=3)
    data = (j.get("data") or {})
    if not data:
        return None

    ipo = data.get("f189")
    ipo_date = None
    try:
        if ipo not in (None, "-", ""):
            v = int(ipo)
            s = f"{v:08d}"
            ipo_date = f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    except Exception:
        ipo_date = None

    return {
        "symbol": symbol,
        "symbol_name": data.get("f58"),
        "price": _scale_price_x100(data.get("f43")),
        "prev_close": _scale_price_x100(data.get("f60")),
        "open": _scale_price_x100(data.get("f46")),
        "high": _scale_price_x100(data.get("f44")),
        "low": _scale_price_x100(data.get("f45")),
        "market_cap_total": _safe_float(data.get("f116")),
        "market_cap_float": _safe_float(data.get("f117")),
        "pe_ttm": _scale_ratio_x100(data.get("f162")),
        "pb": _scale_ratio_x100(data.get("f167")),
        "industry": data.get("f127"),
        "area": data.get("f128"),
        "concepts": data.get("f129"),
        "ipo_date": ipo_date,
    }


def fetch_finance_key_indicators(symbol: str) -> dict | None:
    # Eastmoney F10: 主要指标
    url = "https://emweb.securities.eastmoney.com/PC_HSF10/NewFinanceAnalysis/ZYZBAjaxNew"
    params = {"type": "0", "code": _symbol_to_f10_code(symbol)}
    j = _get_json_with_retry(url, params, timeout=25, attempts=3)
    rows = j.get("data") or []
    if not isinstance(rows, list) or not rows:
        return None

    # pick latest by REPORT_DATE
    def _key(x: dict):
        return str(x.get("REPORT_DATE") or "")

    latest = max(rows, key=_key)

    report_date = str(latest.get("REPORT_DATE") or "")
    if report_date:
        report_date = report_date.split(" ")[0]

    return {
        "fin_report_date": report_date,
        "fin_report_type": latest.get("REPORT_TYPE"),
        "eps_basic": _safe_float(latest.get("EPSJB")),
        "bps": _safe_float(latest.get("BPS")),
        "revenue": _safe_float(latest.get("TOTALOPERATEREVE")),
        "revenue_yoy": _safe_float(latest.get("TOTALOPERATEREVETZ")),
        "gross_profit": _safe_float(latest.get("MLR")),
        "net_profit_parent": _safe_float(latest.get("PARENTNETPROFIT")),
        "net_profit_parent_yoy": _safe_float(latest.get("PARENTNETPROFITTZ")),
        "net_profit_kcfj": _safe_float(latest.get("KCFJCXSYJLR")),
        "net_profit_kcfj_yoy": _safe_float(latest.get("KCFJCXSYJLRTZ")),
        "roe": _safe_float(latest.get("ROEJQ")),
        "rev_qoq": _safe_float(latest.get("YYZSRGDHBZC")),
        "profit_qoq": _safe_float(latest.get("NETPROFITRPHBZC")),
        "kcfj_qoq": _safe_float(latest.get("KFJLRGDHBZC")),
        "op_cashflow_per_share": _safe_float(latest.get("MGJYXJJE")),
    }


def _save_rows(path: str, rows: list[dict]) -> None:
    fieldnames = [
        "symbol",
        "symbol_name",
        "date",
        "price",
        "prev_close",
        "open",
        "high",
        "low",
        "market_cap_total",
        "market_cap_float",
        "pe_ttm",
        "pb",
        "industry",
        "area",
        "concepts",
        "ipo_date",
        "fin_report_date",
        "fin_report_type",
        "eps_basic",
        "bps",
        "revenue",
        "revenue_yoy",
        "gross_profit",
        "net_profit_parent",
        "net_profit_parent_yoy",
        "net_profit_kcfj",
        "net_profit_kcfj_yoy",
        "roe",
        "rev_qoq",
        "profit_qoq",
        "kcfj_qoq",
        "op_cashflow_per_share",
    ]
    write_rows_csv(path, fieldnames=fieldnames, rows=rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Download valuation snapshot + key finance indicators and cache as CSV")
    p.add_argument("--data_dir", default=None, help="Alias of --output-dir")
    p.add_argument("--output-dir", dest="output_dir", default=None)
    p.add_argument("--output_dir", dest="output_dir", default=None)
    p.add_argument("--list", dest="list_mode", default="total", choices=["auto", "total", "self"])
    p.add_argument("--threads", type=int, default=20)
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

    # Determine latest trading date using a benchmark symbol daily summary.
    benchmark_date = None
    try:
        bench = fetch_latest_daily_summary(syms[0])
        if bench and bench.get("date"):
            benchmark_date = str(bench.get("date"))
    except Exception:
        benchmark_date = None

    if not benchmark_date:
        benchmark_date = datetime.today().strftime("%Y-%m-%d")

    out_path = os.path.join(_fund_dir(out_dir), f"{benchmark_date}.csv")
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

        # jitter to reduce burst
        time.sleep(random.uniform(0.01, 0.08))

        out: dict = {"symbol": s, "date": benchmark_date}
        try:
            v = fetch_valuation_snapshot(s)
            if v:
                out.update(v)
        except Exception as e:
            out["_valuation_error"] = f"{type(e).__name__}: {str(e)[:160]}"

        try:
            fin = fetch_finance_key_indicators(s)
            if fin:
                out.update(fin)
        except Exception as e:
            out["_finance_error"] = f"{type(e).__name__}: {str(e)[:160]}"

        if out.get("_valuation_error") or out.get("_finance_error"):
            out["_error"] = "partial"
        return out

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_work, s) for s in syms]
        try:
            for fut in as_completed(futures):
                r = fut.result()
                if r.get("_error") == "partial":
                    errors.append(r)
                rows.append(r)
        except KeyboardInterrupt:
            # Best-effort fast exit: stop waiting and keep whatever is completed.
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
        "partial_errors": len(errors),
        "elapsed_ms": int((time.time() - t0) * 1000),
    }
    print(json.dumps(meta, ensure_ascii=False))

    if errors:
        print(f"Partial errors (showing up to 10/{len(errors)}):")
        for e in errors[:10]:
            print({k: e.get(k) for k in ["symbol", "symbol_name", "_valuation_error", "_finance_error"]})


if __name__ == "__main__":
    main()
