import math
from datetime import datetime, timedelta

import chinese_calendar as calendar
import requests


def last_trading_day(today: datetime | None = None) -> str:
    """Return last trading day as YYYY-MM-DD using chinese_calendar workday rules."""
    base = (today or datetime.today()).date()
    d = base - timedelta(days=1)
    while True:
        dt = datetime(d.year, d.month, d.day)
        if dt.weekday() < 5 and calendar.is_workday(dt):
            return d.isoformat()
        d -= timedelta(days=1)


def _code_to_symbol(code: str) -> str | None:
    code = str(code).strip()
    if len(code) != 6 or not code.isdigit():
        return None

    # Shanghai main board / STAR
    if code.startswith(("60", "68", "90")):
        return "sh" + code

    # Beijing Stock Exchange (Eastmoney uses 8xxx and 92xxxx a lot)
    if code.startswith("8") or code.startswith("92"):
        return "bj" + code

    # Shenzhen main board / ChiNext
    if code.startswith(("00", "30", "20")):
        return "sz" + code

    # Fallback: infer from first digit
    if code.startswith("6"):
        return "sh" + code
    return "sz" + code


def fetch_all_symbols_eastmoney(fs: str | None = None) -> dict:
    """Fetch all A-share-like symbols from Eastmoney clist.

    Returns a dict with keys: asof, source, fs, symbols.
    """

    url = "https://push2.eastmoney.com/api/qt/clist/get"

    # Base A-share filters (SZ main+ChiNext, SH main+STAR)
    # Plus an extra bucket that includes many BJ-listed codes (820/83/92/81xx).
    default_fs = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048"
    fs = fs or default_fs

    # This endpoint caps page size at 100.
    base_params = {
        "pn": 1,
        "pz": 100,
        "po": 1,
        "np": 1,
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": 2,
        "invt": 2,
        "fid": "f12",
        "fs": fs,
        "fields": "f12,f14,f13",
    }

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Referer": "https://quote.eastmoney.com/",
        "Accept": "application/json,text/plain,*/*",
    }

    first = requests.get(url, params=base_params, headers=headers, timeout=20)
    first.raise_for_status()
    first_json = first.json()
    total = int((first_json.get("data") or {}).get("total") or 0)
    pages = int(math.ceil(total / 100)) if total else 0

    symbols: list[str] = []
    seen: set[str] = set()

    def _consume(diff_list: list[dict]):
        for item in diff_list:
            code = str(item.get("f12") or "").strip()
            sym = _code_to_symbol(code)
            if not sym:
                continue
            if sym in seen:
                continue
            seen.add(sym)
            symbols.append(sym)

    _consume((first_json.get("data") or {}).get("diff") or [])

    for pn in range(2, pages + 1):
        params = dict(base_params)
        params["pn"] = pn
        r = requests.get(url, params=params, headers=headers, timeout=20)
        r.raise_for_status()
        j = r.json()
        _consume((j.get("data") or {}).get("diff") or [])

    return {
        "asof": last_trading_day(),
        "source": "eastmoney_clist",
        "fs": fs,
        "symbols": symbols,
    }


def symbol_to_em_secid(symbol: str) -> str:
    s = symbol.strip().lower()
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


def fetch_listing_date_eastmoney(symbol: str) -> str | None:
    """Fetch listing/IPO date for a symbol as YYYY-MM-DD.

    Eastmoney quote API `qt/stock/get` returns `f189` which is typically YYYYMMDD.
    """

    url = "https://push2.eastmoney.com/api/qt/stock/get"
    params = {
        "secid": symbol_to_em_secid(symbol),
        "fields": "f189",
    }
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Referer": "https://quote.eastmoney.com/",
        "Accept": "application/json,text/plain,*/*",
    }
    r = requests.get(url, params=params, headers=headers, timeout=20)
    r.raise_for_status()
    j = r.json()
    data = j.get("data") or {}
    raw = data.get("f189")
    if raw in (None, "-", ""):
        return None
    try:
        v = int(raw)
    except Exception:
        return None
    if v <= 0:
        return None
    s = f"{v:08d}"
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"


def fetch_float_shares_eastmoney(symbol: str) -> float | None:
    """Fetch float shares (流通股本) for a symbol as number of shares (股).

    Uses Eastmoney quote API `qt/stock/get` fields:
    - f85: often corresponds to float shares (流通股本)
    - f84: often corresponds to total shares (总股本)

    If f85 is unavailable, falls back to f84.
    """

    url = "https://push2.eastmoney.com/api/qt/stock/get"
    params = {
        "secid": symbol_to_em_secid(symbol),
        "fields": "f84,f85",
    }
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Referer": "https://quote.eastmoney.com/",
        "Accept": "application/json,text/plain,*/*",
    }
    r = requests.get(url, params=params, headers=headers, timeout=20)
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
