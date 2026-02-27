import os
import threading
from datetime import datetime

import pandas as pd


def _require_token() -> str:
    token = (os.getenv("TUSHARE_TOKEN", "") or "").strip()
    if not token:
        token = (os.getenv("GP_TUSHARE_TOKEN", "") or "").strip()
    if not token:
        raise RuntimeError(
            "Tushare token not found. Please set env var TUSHARE_TOKEN (or GP_TUSHARE_TOKEN)."
        )
    return token


def _import_tushare():
    try:
        import tushare as ts  # type: ignore

        return ts
    except Exception as e:
        raise RuntimeError(
            "Python package 'tushare' is required. Install with: pip install tushare (or pip install -r requirements.txt)"
        ) from e


_LOCAL = threading.local()


def _get_pro_api():
    pro = getattr(_LOCAL, "pro", None)
    if pro is None:
        ts = _import_tushare()
        token = _require_token()
        pro = ts.pro_api(token)
        _LOCAL.pro = pro
    return pro


def symbol_to_ts_code(symbol: str) -> str:
    s = (symbol or "").strip().lower()
    if not s:
        raise ValueError("Empty symbol")

    if s.startswith("sh"):
        return f"{s[2:]}.SH"
    if s.startswith("sz"):
        return f"{s[2:]}.SZ"
    if s.startswith("bj"):
        return f"{s[2:]}.BJ"

    # numeric fallback
    if s.startswith("92"):
        return f"{s}.BJ"
    if s.startswith(("6", "9")):
        return f"{s}.SH"
    return f"{s}.SZ"


def _normalize_adj_mode(fqt: int | str) -> str:
    if isinstance(fqt, int):
        fqt = str(fqt)
    s = str(fqt).strip().lower()
    if s in {"0", "none", "raw", "nfq"}:
        return "none"
    if s in {"1", "qfq", "forward"}:
        return "qfq"
    if s in {"2", "hfq", "backward"}:
        return "hfq"
    raise ValueError(f"Unknown adj/fqt mode: {fqt}")


def _yyyymmdd(date_yyyy_mm_dd: str) -> str:
    d = (date_yyyy_mm_dd or "").strip()
    if len(d) == 10 and d[4] == "-" and d[7] == "-":
        return d.replace("-", "")
    return d


def _yyyymmdd_to_yyyy_mm_dd(s: str) -> str:
    s = (s or "").strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    return s


def _to_datetime_series(df: pd.DataFrame) -> pd.Series:
    # Try common columns produced by tushare minute endpoints/pro_bar.
    for col in ("trade_time", "datetime", "time"):
        if col in df.columns:
            return pd.to_datetime(df[col], errors="coerce")

    # Some variants expose date + time separately.
    if "trade_date" in df.columns and "trade_time" in df.columns:
        return pd.to_datetime(df["trade_date"].astype(str) + " " + df["trade_time"].astype(str), errors="coerce")

    # Fall back to index
    try:
        return pd.to_datetime(df.index, errors="coerce")
    except Exception:
        return pd.to_datetime(pd.Series([pd.NaT] * len(df)))


def fetch_pro_bar(
    symbol: str,
    start_yyyymmdd: str,
    end_yyyymmdd: str,
    freq: str,
    adj: str,
) -> pd.DataFrame:
    """Fetch bars via tushare.pro_bar.

    This uses the tushare Python SDK and requires TUSHARE_TOKEN.
    """

    ts = _import_tushare()
    pro = _get_pro_api()

    ts_code = symbol_to_ts_code(symbol)

    # Note: pro_bar returns newest-first; we'll normalize sort later.
    df = ts.pro_bar(
        ts_code=ts_code,
        api=pro,
        start_date=start_yyyymmdd,
        end_date=end_yyyymmdd,
        freq=freq,
        adj=None if adj == "none" else adj,
    )

    if df is None:
        return pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    return df.copy()


def fetch_ts_daily_df(symbol: str, beg_yyyymmdd: str, end_yyyymmdd: str, fqt: int | str) -> pd.DataFrame:
    adj = _normalize_adj_mode(fqt)
    df = fetch_pro_bar(symbol, beg_yyyymmdd, end_yyyymmdd, freq="D", adj=adj)
    if df.empty:
        return df

    # Normalize date column
    if "trade_date" in df.columns:
        df["date"] = df["trade_date"].astype(str).map(_yyyymmdd_to_yyyy_mm_dd)
    elif "date" in df.columns:
        df["date"] = df["date"].astype(str)
    else:
        # If no explicit date, try datetime-derived date
        dt = _to_datetime_series(df)
        df["date"] = dt.dt.strftime("%Y-%m-%d")

    df = df.dropna(subset=["date"])
    df = df.sort_values("date")
    return df


def fetch_ts_minute_df(symbol: str, date_yyyy_mm_dd: str, fqt: int | str) -> pd.DataFrame:
    adj = _normalize_adj_mode(fqt)
    ymd = _yyyymmdd(date_yyyy_mm_dd)
    df = fetch_pro_bar(symbol, ymd, ymd, freq="1min", adj=adj)
    if df.empty:
        return df

    dt = _to_datetime_series(df)
    df = df.assign(_dt=dt)
    df = df.dropna(subset=["_dt"])

    # Restrict to the requested calendar day.
    target_date = datetime.strptime(date_yyyy_mm_dd, "%Y-%m-%d").date()
    df = df[df["_dt"].dt.date == target_date]

    # Normalize to YYYY-MM-DD HH:MM
    df["时间"] = df["_dt"].dt.strftime("%Y-%m-%d %H:%M")

    def _pick(colnames: tuple[str, ...]) -> str | None:
        for c in colnames:
            if c in df.columns:
                return c
        return None

    c_open = _pick(("open",))
    c_close = _pick(("close",))
    c_high = _pick(("high",))
    c_low = _pick(("low",))
    c_vol = _pick(("vol", "volume"))
    c_amt = _pick(("amount", "amt"))

    if c_open is None or c_close is None or c_high is None or c_low is None:
        raise RuntimeError(f"Tushare minute df missing OHLC columns: {list(df.columns)}")

    out = pd.DataFrame(
        {
            "时间": df["时间"].astype(str),
            "开盘": pd.to_numeric(df[c_open], errors="coerce"),
            "收盘": pd.to_numeric(df[c_close], errors="coerce"),
            "最高": pd.to_numeric(df[c_high], errors="coerce"),
            "最低": pd.to_numeric(df[c_low], errors="coerce"),
            "成交量(手)": pd.to_numeric(df[c_vol], errors="coerce") if c_vol else pd.NA,
            "成交额(元)": pd.to_numeric(df[c_amt], errors="coerce") if c_amt else pd.NA,
        }
    )

    # Avg price if possible.
    if c_amt and c_vol:
        denom = out["成交量(手)"].astype(float) * 100.0
        out["均价"] = out["成交额(元)"].astype(float) / denom.replace({0.0: pd.NA})
    else:
        out["均价"] = (out["开盘"] + out["收盘"] + out["最高"] + out["最低"]) / 4.0

    out = out.dropna(subset=["时间"])
    out = out.sort_values("时间").reset_index(drop=True)
    return out
