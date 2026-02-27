import os
from datetime import datetime, timedelta
from functools import lru_cache

import requests


_HTTP = requests.Session()
if os.getenv("GP_NO_PROXY", "").strip().lower() in {"1", "true", "yes"}:
    _HTTP.trust_env = False


def _daily_kline_source() -> str:
    s = (os.getenv("GP_DAILY_KLINE_SOURCE", "") or "").strip().lower()
    if not s or s in {"em", "eastmoney"}:
        return "em"
    if s in {"tx", "tencent"}:
        return "tx"
    return "em"


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
    if _daily_kline_source() == "tx":
        return _fetch_daily_kline_data_tencent(symbol, beg_yyyymmdd, end_yyyymmdd, adj=adj)

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
    r = _HTTP.get(url, params=params, headers=_em_headers(), timeout=20)
    r.raise_for_status()
    j = r.json()
    return j.get("data") or {}


def _fetch_daily_kline_data_tencent(symbol: str, beg_yyyymmdd: str, end_yyyymmdd: str, adj: str = "none") -> dict:
    """Fetch daily kline data from Tencent and adapt to Eastmoney-like structure.

    Returns dict with key `klines` being a list[str] formatted as:
    YYYY-MM-DD,open,close,high,low,volume,amount,amplitude,pctchg,chg,turnover
    (Tencent provides OHLCV; the rest are empty)
    """

    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"

    def _yyyymmdd_to_yyyy_mm_dd(s: str) -> str:
        s = (s or "").strip()
        if len(s) == 8 and s.isdigit():
            return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
        return s

    def _symbol_to_tx_code(sym: str) -> str:
        s = (sym or "").strip().lower()
        if s.startswith(("sh", "sz", "bj")):
            return s
        if s.startswith("92"):
            return "bj" + s
        if s.startswith(("6", "9")):
            return "sh" + s
        return "sz" + s

    a = (adj or "none").strip().lower()
    adj_param = "none" if a in {"none", "0", "raw", "nfq"} else ("qfq" if a in {"qfq", "1", "forward"} else "hfq")
    key = "day" if adj_param == "none" else ("qfqday" if adj_param == "qfq" else "hfqday")

    beg = _yyyymmdd_to_yyyy_mm_dd(beg_yyyymmdd)
    end = _yyyymmdd_to_yyyy_mm_dd(end_yyyymmdd)
    code = _symbol_to_tx_code(symbol)
    params = {"param": f"{code},day,{beg},{end},640,{adj_param}"}
    headers = {
        "User-Agent": _em_headers()["User-Agent"],
        "Referer": "https://gu.qq.com/",
        "Accept": "application/json,text/plain,*/*",
    }

    r = _HTTP.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    j = r.json()
    data = (j.get("data") or {}).get(code) or {}
    rows = data.get(key) or data.get("day") or []

    float_shares = fetch_float_shares(symbol)

    def _sf(v):
        try:
            return float(v)
        except Exception:
            return None

    def _fmt(v) -> str:
        return "" if v is None else str(v)

    klines: list[str] = []
    prev_close: float | None = None
    for row in rows:
        if not isinstance(row, list) or len(row) < 6:
            continue

        d = str(row[0])
        open_ = _sf(row[1])
        close_ = _sf(row[2])
        high_ = _sf(row[3])
        low_ = _sf(row[4])
        volume = _sf(row[5])

        amount: float | None = None
        chg: float | None = None
        pctchg: float | None = None
        amplitude: float | None = None
        turnover: float | None = None

        if prev_close is not None and prev_close > 0 and close_ is not None:
            chg = close_ - prev_close
            pctchg = chg / prev_close * 100.0
            if high_ is not None and low_ is not None:
                amplitude = (high_ - low_) / prev_close * 100.0

        if volume is not None and open_ is not None and close_ is not None and high_ is not None and low_ is not None:
            avg_price = (open_ + close_ + high_ + low_) / 4.0
            # Eastmoney daily kline uses volume in hands (手). Approximate amount in CNY.
            amount = avg_price * volume * 100.0

        if float_shares is not None and float_shares > 0 and volume is not None:
            shares_traded = volume * 100.0
            turnover = shares_traded / float_shares * 100.0

        klines.append(
            ",".join(
                [
                    d,
                    _fmt(open_),
                    _fmt(close_),
                    _fmt(high_),
                    _fmt(low_),
                    _fmt(volume),
                    _fmt(amount),
                    _fmt(amplitude),
                    _fmt(pctchg),
                    _fmt(chg),
                    _fmt(turnover),
                ]
            )
        )

        if close_ is not None:
            prev_close = close_

    return {"klines": klines}


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
    r = _HTTP.get(url, params=params, headers=_em_headers(), timeout=20)
    r.raise_for_status()
    j = r.json()
    data = j.get("data") or {}
    name = data.get("f58")
    if not name or name in {"-", "None"}:
        return None
    return str(name)


@lru_cache(maxsize=4096)
def fetch_float_shares(symbol: str) -> float | None:
    """Fetch float shares (流通股本) as number of shares (股)."""

    url = "https://push2.eastmoney.com/api/qt/stock/get"
    params = {
        "secid": symbol_to_em_secid(symbol),
        "fields": "f84,f85",
    }
    r = _HTTP.get(url, params=params, headers=_em_headers(), timeout=20)
    r.raise_for_status()
    j = r.json()
    data = j.get("data") or {}
    float_shares = data.get("f85")
    total_shares = data.get("f84")

    def _as_positive_number(v):
        try:
            x = float(v)
        except Exception:
            return None
        if x <= 0:
            return None
        return x

    return _as_positive_number(float_shares) or _as_positive_number(total_shares)


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
