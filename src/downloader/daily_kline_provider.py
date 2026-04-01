import os
import threading
import time

import requests


_NO_PROXY = os.getenv("GP_NO_PROXY", "").strip().lower() in {"1", "true", "yes"}
_HTTP_LOCAL = threading.local()


def _get_http_session() -> requests.Session:
    session = getattr(_HTTP_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        if _NO_PROXY:
            session.trust_env = False
        _HTTP_LOCAL.session = session
    return session


def _em_headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Referer": "https://quote.eastmoney.com/",
        "Accept": "application/json,text/plain,*/*",
    }


def _symbol_to_em_secid(symbol: str) -> str:
    value = (symbol or "").strip().lower()
    if value.startswith("sh"):
        return f"1.{value[2:]}"
    if value.startswith(("sz", "bj")):
        return f"0.{value[2:]}"
    if value.startswith("92"):
        return f"0.{value}"
    market_code = 1 if value.startswith(("6", "9")) else 0
    return f"{market_code}.{value}"


def _normalize_yyyymmdd(value: str) -> str:
    text = (value or "").strip()
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        return text.replace("-", "")
    return text


def _normalize_fqt(value: int | str) -> str:
    if isinstance(value, int):
        return str(value)
    text = str(value).strip().lower()
    if text in {"0", "raw", "none", "nfq"}:
        return "0"
    if text in {"1", "qfq", "forward"}:
        return "1"
    if text in {"2", "hfq", "backward"}:
        return "2"
    raise ValueError(f"Unsupported Eastmoney fqt value: {value}")


def _http_get_json(url: str, params: dict[str, str], timeout: int = 20, retries: int = 3) -> dict:
    last_error: Exception | None = None
    for attempt in range(max(1, int(retries))):
        try:
            response = _get_http_session().get(url, params=params, headers=_em_headers(), timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(min(6.0, 0.5 * (2**attempt)))
    if last_error is None:
        raise RuntimeError("Eastmoney daily kline request failed")
    raise last_error


def fetch_daily_kline_lines(symbol: str, beg_yyyymmdd: str, end_yyyymmdd: str, fqt: int | str = 1) -> list[str]:
    """Fetch daily K-line rows from Eastmoney.

    Returns the raw `klines` string list, where each row looks like:
    `YYYY-MM-DD,open,close,high,low,volume,amount,...`
    """

    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": _symbol_to_em_secid(symbol),
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101",
        "fqt": _normalize_fqt(fqt),
        "beg": _normalize_yyyymmdd(beg_yyyymmdd),
        "end": _normalize_yyyymmdd(end_yyyymmdd),
    }
    payload = _http_get_json(url, params=params, timeout=20, retries=4)
    data = payload.get("data") or {}
    rows = data.get("klines") or []
    if not isinstance(rows, list):
        return []
    return [str(item) for item in rows if isinstance(item, str)]