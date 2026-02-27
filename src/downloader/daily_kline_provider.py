import os
import random
import threading
import time
from dataclasses import dataclass

import requests

from eastmoney_universe import fetch_float_shares_eastmoney, symbol_to_em_secid


_NO_PROXY = os.getenv("GP_NO_PROXY", "").strip().lower() in {"1", "true", "yes"}


_HTTP_LOCAL = threading.local()


def _get_http_session() -> requests.Session:
    s = getattr(_HTTP_LOCAL, "session", None)
    if s is None:
        s = requests.Session()
        if _NO_PROXY:
            s.trust_env = False
        _HTTP_LOCAL.session = s
    return s


_EM_MAX_INFLIGHT = max(1, int(os.getenv("GP_EM_MAX_INFLIGHT", "4") or "4"))
_EM_HTTP_SEM = threading.BoundedSemaphore(_EM_MAX_INFLIGHT)


def _em_headers() -> dict:
    return {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Referer": "https://quote.eastmoney.com/",
        "Accept": "application/json,text/plain,*/*",
    }


def _http_get_json(url: str, params: dict, timeout: int = 30, retries: int = 3, headers: dict | None = None) -> dict:
    last_err: Exception | None = None
    for i in range(max(1, int(retries))):
        try:
            with _EM_HTTP_SEM:
                r = _get_http_session().get(url, params=params, headers=headers or _em_headers(), timeout=timeout)
                r.raise_for_status()
                return r.json() or {}
        except requests.exceptions.ProxyError:
            # Fail fast: proxy issues in this environment are usually persistent.
            raise
        except requests.exceptions.RequestException as e:
            last_err = e
            if i < retries - 1:
                time.sleep(min(8.0, 0.6 * (2**i)) + random.uniform(0.2, 1.0))

    if last_err is None:
        raise RuntimeError("HTTP request failed for unknown reason")
    raise last_err


def daily_kline_source() -> str:
    """Return daily kline source selector.

    Controlled by env var GP_DAILY_KLINE_SOURCE:
      - ts/tushare
      - em/eastmoney
      - tx/tencent
    """

    s = (os.getenv("GP_DAILY_KLINE_SOURCE", "") or "").strip().lower()
    if not s or s in {"ts", "tushare"}:
        return "ts"
    if s in {"em", "eastmoney"}:
        return "em"
    if s in {"tx", "tencent"}:
        return "tx"
    return "ts"


def _normalize_em_fqt(fqt: int | str) -> str:
    if isinstance(fqt, int):
        return str(fqt)
    s = str(fqt).strip().lower()
    if s in {"0", "none", "raw", "nfq"}:
        return "0"
    if s in {"1", "qfq", "forward"}:
        return "1"
    if s in {"2", "hfq", "backward"}:
        return "2"
    raise ValueError(f"Unknown fqt/adj mode: {fqt}")


def _symbol_to_tx_code(symbol: str) -> str:
    s = (symbol or "").strip().lower()
    if s.startswith(("sh", "sz", "bj")):
        return s
    if s.startswith("92"):
        return "bj" + s
    if s.startswith(("6", "9")):
        return "sh" + s
    return "sz" + s


def _yyyymmdd_to_yyyy_mm_dd(s: str) -> str:
    s = (s or "").strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    return s


_FLOAT_SHARES_LOCK = threading.Lock()
_FLOAT_SHARES_CACHE: dict[str, float | None] = {}


def _get_float_shares_cached(symbol: str) -> float | None:
    with _FLOAT_SHARES_LOCK:
        if symbol in _FLOAT_SHARES_CACHE:
            return _FLOAT_SHARES_CACHE[symbol]
    try:
        v = fetch_float_shares_eastmoney(symbol)
    except Exception:
        v = None
    with _FLOAT_SHARES_LOCK:
        _FLOAT_SHARES_CACHE[symbol] = v
    return v


@dataclass(frozen=True)
class DailyKlineRow:
    date: str
    open: float | None
    close: float | None
    high: float | None
    low: float | None
    volume: float | None
    amount: float | None
    amplitude: float | None
    pctchg: float | None
    chg: float | None
    turnover: float | None

    def to_em_line(self) -> str:
        def f(v):
            return "" if v is None else v

        return (
            f"{self.date},{f(self.open)},{f(self.close)},{f(self.high)},{f(self.low)},{f(self.volume)},"
            f"{f(self.amount)},{f(self.amplitude)},{f(self.pctchg)},{f(self.chg)},{f(self.turnover)}"
        )


def _safe_float(v) -> float | None:
    try:
        if v in (None, "", "-", "None"):
            return None
        return float(v)
    except Exception:
        return None


def _compute_derived_fields(rows: list[DailyKlineRow], float_shares: float | None) -> list[DailyKlineRow]:
    out: list[DailyKlineRow] = []
    prev_close: float | None = None

    for r in rows:
        close = r.close
        high = r.high
        low = r.low
        open_ = r.open
        volume = r.volume

        chg = None
        pctchg = None
        amplitude = None
        amount = r.amount
        turnover = None

        if prev_close and prev_close > 0 and close is not None:
            chg = close - prev_close
            pctchg = (chg / prev_close) * 100.0
            if high is not None and low is not None:
                amplitude = ((high - low) / prev_close) * 100.0

        # Approximate amount (成交额) when missing and we have OHLCV.
        if amount is None and volume is not None and open_ is not None and close is not None and high is not None and low is not None:
            avg_price = (open_ + close + high + low) / 4.0
            # Heuristic: volume is usually in "hand" (手). 1 手 = 100 shares.
            amount = float(volume) * 100.0 * float(avg_price)

        # Compute turnover (%) when we know float shares.
        if float_shares and float_shares > 0 and volume is not None:
            # turnover(%) = volume_shares / float_shares * 100
            # volume_shares = volume_hands * 100
            turnover = float(volume) * 10000.0 / float_shares

        out.append(
            DailyKlineRow(
                date=r.date,
                open=open_,
                close=close,
                high=high,
                low=low,
                volume=volume,
                amount=amount,
                amplitude=r.amplitude if r.amplitude is not None else amplitude,
                pctchg=r.pctchg if r.pctchg is not None else pctchg,
                chg=r.chg if r.chg is not None else chg,
                turnover=r.turnover if r.turnover is not None else turnover,
            )
        )

        if close is not None:
            prev_close = close

    return out


def fetch_daily_kline_rows(symbol: str, beg_yyyymmdd: str, end_yyyymmdd: str, fqt: int | str) -> list[DailyKlineRow]:
    """Fetch daily kline rows and normalize into a single canonical schema.

    Returned rows are sorted by date ascending when possible.
    """

    src = daily_kline_source()
    fqt_norm = _normalize_em_fqt(fqt)

    if src == "ts":
        try:
            from tushare_provider import fetch_ts_daily_df

            df = fetch_ts_daily_df(symbol, beg_yyyymmdd, end_yyyymmdd, fqt=fqt_norm)
            if df.empty:
                return []

            rows: list[DailyKlineRow] = []
            for _idx, r in df.iterrows():
                d = str(r.get("date", "") or "").strip()
                if not d:
                    continue
                rows.append(
                    DailyKlineRow(
                        date=d,
                        open=_safe_float(r.get("open")),
                        close=_safe_float(r.get("close")),
                        high=_safe_float(r.get("high")),
                        low=_safe_float(r.get("low")),
                        volume=_safe_float(r.get("vol") if "vol" in df.columns else r.get("volume")),
                        amount=_safe_float(r.get("amount")),
                        amplitude=None,
                        pctchg=_safe_float(r.get("pct_chg")),
                        chg=_safe_float(r.get("change")),
                        turnover=_safe_float(r.get("turnover_rate")),
                    )
                )

            rows.sort(key=lambda x: x.date)
            float_shares = _get_float_shares_cached(symbol)
            return _compute_derived_fields(rows, float_shares=float_shares)
        except Exception as e:
            # Fallback: tushare might not have sufficient points/permission.
            msg = str(e)
            if len(msg) > 200:
                msg = msg[:200] + "..."
            print(f"[WARN] Tushare daily fetch failed for {symbol} ({beg_yyyymmdd}-{end_yyyymmdd}): {type(e).__name__}: {msg}. Falling back to Tencent/Eastmoney.")
            # Prefer Tencent, then Eastmoney (Tencent branch is below).
            src = "tx"

    if src == "em":
        url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        params = {
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "ut": "7eea3edcaed734bea9cbfc24409ed989",
            "klt": "101",
            "fqt": fqt_norm,
            "secid": symbol_to_em_secid(symbol),
            "beg": beg_yyyymmdd,
            "end": end_yyyymmdd,
        }
        j = _http_get_json(url, params=params, timeout=40, retries=6, headers=_em_headers())
        data = j.get("data") or {}
        klines = data.get("klines") or []
        rows: list[DailyKlineRow] = []
        for line in klines:
            if not isinstance(line, str):
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            d = str(parts[0]).strip()
            if not d:
                continue
            open_ = _safe_float(parts[1])
            close = _safe_float(parts[2])
            high = _safe_float(parts[3])
            low = _safe_float(parts[4])
            vol = _safe_float(parts[5])
            amt = _safe_float(parts[6]) if len(parts) > 6 else None
            amp = _safe_float(parts[7]) if len(parts) > 7 else None
            pct = _safe_float(parts[8]) if len(parts) > 8 else None
            chg = _safe_float(parts[9]) if len(parts) > 9 else None
            to = _safe_float(parts[10]) if len(parts) > 10 else None
            rows.append(
                DailyKlineRow(
                    date=d,
                    open=open_,
                    close=close,
                    high=high,
                    low=low,
                    volume=vol,
                    amount=amt,
                    amplitude=amp,
                    pctchg=pct,
                    chg=chg,
                    turnover=to,
                )
            )
        return rows

    # Tencent
    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    adj_param = "none" if fqt_norm == "0" else ("qfq" if fqt_norm == "1" else "hfq")
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

    j = _http_get_json(url, params=params, timeout=40, retries=6, headers=headers)
    data = (j.get("data") or {}).get(code) or {}
    raw_rows = data.get(key) or data.get("day") or []

    rows: list[DailyKlineRow] = []
    for item in raw_rows:
        if not isinstance(item, list) or len(item) < 6:
            continue
        d, o, c, h, l, v = item[0:6]
        rows.append(
            DailyKlineRow(
                date=str(d),
                open=_safe_float(o),
                close=_safe_float(c),
                high=_safe_float(h),
                low=_safe_float(l),
                volume=_safe_float(v),
                amount=None,
                amplitude=None,
                pctchg=None,
                chg=None,
                turnover=None,
            )
        )

    # Ensure sorted by date ascending for derived calculations.
    rows.sort(key=lambda r: r.date)
    float_shares = _get_float_shares_cached(symbol)
    return _compute_derived_fields(rows, float_shares=float_shares)


def fetch_daily_kline_lines(symbol: str, beg_yyyymmdd: str, end_yyyymmdd: str, fqt: int | str) -> list[str]:
    return [r.to_em_line() for r in fetch_daily_kline_rows(symbol, beg_yyyymmdd, end_yyyymmdd, fqt)]
