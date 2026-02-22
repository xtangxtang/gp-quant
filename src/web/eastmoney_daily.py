import os
from datetime import datetime, timedelta
from functools import lru_cache

import requests


def _em_headers() -> dict:
    return {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Referer": "https://quote.eastmoney.com/",
        "Accept": "application/json,text/plain,*/*",
    }


def symbol_to_em_secid(symbol: str) -> str:
    s = str(symbol).strip().lower()
    if s.startswith("sh"):
        return f"1.{s[2:]}"
    if s.startswith("sz"):
        return f"0.{s[2:]}"
    if s.startswith("bj"):
        return f"0.{s[2:]}"

    if s.startswith("92"):
        market_code = 0
    else:
        market_code = 1 if s.startswith(("6", "9")) else 0
    return f"{market_code}.{s}"


def _normalize_adj(adj: str) -> str:
    a = (adj or "none").strip().lower()
    if a in {"none", "0", "raw", "nfq"}:
        return "0"
    if a in {"qfq", "1", "forward"}:
        return "1"
    if a in {"hfq", "2", "backward"}:
        return "2"
    raise ValueError(f"Unknown adj: {adj}")


def fetch_daily_kline_data(symbol: str, beg_yyyymmdd: str, end_yyyymmdd: str, adj: str = "none") -> dict:
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "klt": "101",
        "fqt": _normalize_adj(adj),
        "secid": symbol_to_em_secid(symbol),
        "beg": beg_yyyymmdd,
        "end": end_yyyymmdd,
    }
    r = requests.get(url, params=params, headers=_em_headers(), timeout=20)
    r.raise_for_status()
    j = r.json()
    return j.get("data") or {}


def fetch_daily_klines(symbol: str, beg_yyyymmdd: str, end_yyyymmdd: str, adj: str = "none") -> list[str]:
    data = fetch_daily_kline_data(symbol, beg_yyyymmdd, end_yyyymmdd, adj=adj)
    return data.get("klines") or []


@lru_cache(maxsize=4096)
def fetch_stock_name(symbol: str) -> str | None:
    url = "https://push2.eastmoney.com/api/qt/stock/get"
    params = {
        "secid": symbol_to_em_secid(symbol),
        "fields": "f58",
    }
    r = requests.get(url, params=params, headers=_em_headers(), timeout=20)
    r.raise_for_status()
    j = r.json()
    data = j.get("data") or {}
    name = data.get("f58")
    if not name or name in {"-", "None"}:
        return None
    return str(name)


def _safe_float(v):
    try:
        return float(v)
    except Exception:
        return None


def parse_daily_kline_line(line: str) -> dict | None:
    # expected: YYYY-MM-DD,open,close,high,low,volume,amount,amplitude,pctchg,chg,turnover
    if not line or not isinstance(line, str):
        return None
    parts = line.split(",")
    if len(parts) < 7:
        return None

    d = parts[0]
    open_ = _safe_float(parts[1])
    close_ = _safe_float(parts[2])
    high_ = _safe_float(parts[3])
    low_ = _safe_float(parts[4])
    vol = _safe_float(parts[5])
    amt = _safe_float(parts[6])

    amplitude = _safe_float(parts[7]) if len(parts) > 7 else None
    pctchg = _safe_float(parts[8]) if len(parts) > 8 else None
    chg = _safe_float(parts[9]) if len(parts) > 9 else None
    turnover = _safe_float(parts[10]) if len(parts) > 10 else None

    return {
        "date": d,
        "open": open_,
        "close": close_,
        "high": high_,
        "low": low_,
        "volume": vol,
        "amount": amt,
        "amplitude": amplitude,
        "pctchg": pctchg,
        "chg": chg,
        "turnover": turnover,
    }


def fetch_latest_daily_summary(symbol: str, adj: str = "none") -> dict | None:
    # Keep window small so response is fast.
    today = datetime.today().date()
    beg = (today - timedelta(days=120)).strftime("%Y%m%d")
    end = today.strftime("%Y%m%d")
    data = fetch_daily_kline_data(symbol, beg, end, adj=adj)
    klines = data.get("klines") or []
    if not klines:
        return None

    last = parse_daily_kline_line(klines[-1])
    if not last:
        return None

    last["symbol"] = symbol
    name = data.get("name") or data.get("stockName")
    if name and name not in {"-", "None"}:
        last["symbol_name"] = str(name)
    else:
        last["symbol_name"] = fetch_stock_name(symbol)
    return last


def default_data_dir() -> str:
    env = os.environ.get("GP_DATA_DIR") or os.environ.get("GP_OUTPUT_DIR")
    if env:
        return env

    # common layout in your workspace: ../gp-data relative to gp-quant
    guess = os.path.abspath(os.path.join(os.getcwd(), "..", "gp-data"))
    if os.path.isdir(guess):
        return guess

    return os.path.abspath(os.path.join(os.getcwd(), "gp_daily"))
