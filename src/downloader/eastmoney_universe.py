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
        raise RuntimeError("Eastmoney stock snapshot request failed")
    raise last_error


def _to_positive_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    if numeric <= 0:
        return None
    return numeric


def fetch_float_shares_eastmoney(symbol: str) -> float | None:
    """Return float shares in share units.

    Eastmoney quote fields commonly expose:
    - f84: total shares
    - f85: float shares
    """

    url = "https://push2.eastmoney.com/api/qt/stock/get"
    params = {
        "secid": _symbol_to_em_secid(symbol),
        "fields": "f57,f58,f84,f85",
    }
    payload = _http_get_json(url, params=params, timeout=20, retries=4)
    data = payload.get("data") or {}
    float_shares = _to_positive_float(data.get("f85"))
    if float_shares is not None:
        return float_shares
    return _to_positive_float(data.get("f84"))