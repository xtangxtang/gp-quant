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


def ts_code_to_symbol(ts_code: str) -> str:
    value = (ts_code or "").strip()
    if not value or "." not in value:
        raise ValueError(f"Invalid ts_code: {ts_code}")
    code, market = value.split(".", 1)
    return f"{market.lower()}{code}"


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


def fetch_stk_mins(
    symbol: str,
    start_datetime: str,
    end_datetime: str,
    freq: str,
) -> pd.DataFrame:
    """Fetch historical minute bars via the documented Tushare stk_mins API."""

    pro = _get_pro_api()
    ts_code = symbol_to_ts_code(symbol)

    df = pro.stk_mins(
        ts_code=ts_code,
        freq=freq,
        start_date=start_datetime,
        end_date=end_datetime,
    )

    if df is None:
        return pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    return df.copy()


def fetch_ts_daily_basic_df(
    trade_date: str,
    fields: str = "ts_code,trade_date,float_share,free_share,turnover_rate,turnover_rate_f",
) -> pd.DataFrame:
    pro = _get_pro_api()
    ymd = _yyyymmdd(trade_date)
    df = pro.daily_basic(trade_date=ymd, fields=fields)
    if df is None:
        return pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    return df.copy()


def fetch_ts_float_share_map(trade_date: str, use_free_share: bool = False) -> dict[str, float]:
    share_col = "free_share" if use_free_share else "float_share"
    df = fetch_ts_daily_basic_df(trade_date, fields=f"ts_code,trade_date,float_share,free_share")
    if df.empty or share_col not in df.columns or "ts_code" not in df.columns:
        return {}

    share_series = pd.to_numeric(df[share_col], errors="coerce")
    result: dict[str, float] = {}
    for ts_code, shares_wan in zip(df["ts_code"].astype(str), share_series):
        if pd.isna(shares_wan):
            continue
        shares_wan_value = float(shares_wan)
        if shares_wan_value <= 0:
            continue
        result[ts_code_to_symbol(ts_code)] = shares_wan_value * 10000.0
    return result


def _daily_close_map(df: pd.DataFrame) -> dict[str, float]:
    if df.empty or "close" not in df.columns:
        return {}

    if "trade_date" in df.columns:
        trade_dates = df["trade_date"].astype(str).map(_yyyymmdd_to_yyyy_mm_dd)
    else:
        trade_dates = _to_datetime_series(df).dt.strftime("%Y-%m-%d")

    closes = pd.to_numeric(df["close"], errors="coerce")
    result: dict[str, float] = {}
    for trade_date, close in zip(trade_dates, closes):
        if not trade_date or pd.isna(close):
            continue
        result[str(trade_date)] = float(close)
    return result


def _daily_adjustment_factor_map(symbol: str, trade_dates: list[str], fqt: int | str) -> dict[str, float]:
    normalized_dates = sorted({str(item).strip() for item in trade_dates if str(item).strip()})
    if not normalized_dates:
        return {}

    adj = _normalize_adj_mode(fqt)
    if adj == "none":
        return {trade_date: 1.0 for trade_date in normalized_dates}

    start_yyyymmdd = _yyyymmdd(normalized_dates[0])
    end_yyyymmdd = _yyyymmdd(normalized_dates[-1])
    raw_df = fetch_pro_bar(symbol, start_yyyymmdd, end_yyyymmdd, freq="D", adj="none")
    adj_df = fetch_pro_bar(symbol, start_yyyymmdd, end_yyyymmdd, freq="D", adj=adj)
    raw_map = _daily_close_map(raw_df)
    adj_map = _daily_close_map(adj_df)

    result: dict[str, float] = {}
    for trade_date in normalized_dates:
        raw_close = raw_map.get(trade_date)
        adj_close = adj_map.get(trade_date)
        if raw_close is None or adj_close is None or raw_close <= 0:
            result[trade_date] = 1.0
            continue
        result[trade_date] = float(adj_close) / float(raw_close)
    return result


def _normalize_ts_minute_frame(
    raw_df: pd.DataFrame,
    symbol: str,
    fqt: int | str,
    start_datetime: str,
    end_datetime: str,
) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df

    dt = _to_datetime_series(raw_df)
    df = raw_df.assign(_dt=dt)
    df = df.dropna(subset=["_dt"])
    start_ts = pd.Timestamp(start_datetime)
    end_ts = pd.Timestamp(end_datetime)
    df = df[(df["_dt"] >= start_ts) & (df["_dt"] <= end_ts)]
    if df.empty:
        return df

    df = df.sort_values("_dt").reset_index(drop=True)
    df["时间"] = df["_dt"].dt.strftime("%Y-%m-%d %H:%M")

    def _pick(colnames: tuple[str, ...]) -> str | None:
        for colname in colnames:
            if colname in df.columns:
                return colname
        return None

    c_open = _pick(("open",))
    c_close = _pick(("close",))
    c_high = _pick(("high",))
    c_low = _pick(("low",))
    c_vol = _pick(("vol", "volume"))
    c_amt = _pick(("amount", "amt"))

    if c_open is None or c_close is None or c_high is None or c_low is None:
        raise RuntimeError(f"Tushare minute df missing OHLC columns: {list(df.columns)}")

    trade_dates = df["_dt"].dt.strftime("%Y-%m-%d")
    factor_map = _daily_adjustment_factor_map(symbol, trade_dates.tolist(), fqt)
    factor_series = trade_dates.map(factor_map).fillna(1.0).astype(float)
    volume_shares = pd.to_numeric(df[c_vol], errors="coerce") if c_vol else pd.Series([float("nan")] * len(df))
    volume_hands = volume_shares / 100.0
    amount_yuan = pd.to_numeric(df[c_amt], errors="coerce") if c_amt else pd.Series([float("nan")] * len(df))
    if c_amt:
        amount_yuan = amount_yuan.astype(float) * factor_series

    out = pd.DataFrame(
        {
            "时间": df["时间"].astype(str),
            "开盘": pd.to_numeric(df[c_open], errors="coerce") * factor_series,
            "收盘": pd.to_numeric(df[c_close], errors="coerce") * factor_series,
            "最高": pd.to_numeric(df[c_high], errors="coerce") * factor_series,
            "最低": pd.to_numeric(df[c_low], errors="coerce") * factor_series,
            "成交量(手)": volume_hands,
            "成交额(元)": amount_yuan,
        }
    )

    if c_amt and c_vol:
        out["均价"] = out["成交额(元)"].astype(float) / volume_shares.replace({0.0: pd.NA})
    else:
        out["均价"] = (out["开盘"] + out["收盘"] + out["最高"] + out["最低"]) / 4.0

    out = out.dropna(subset=["时间"])
    out = out.sort_values("时间").reset_index(drop=True)
    return out


def fetch_ts_minute_range_df(symbol: str, start_datetime: str, end_datetime: str, fqt: int | str) -> pd.DataFrame:
    raw_df = fetch_stk_mins(symbol, start_datetime, end_datetime, freq="1min")
    if raw_df.empty:
        return raw_df
    return _normalize_ts_minute_frame(raw_df, symbol, fqt, start_datetime, end_datetime)


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
    target_dt = datetime.strptime(date_yyyy_mm_dd, "%Y-%m-%d")
    start_datetime = target_dt.strftime("%Y-%m-%d 09:00:00")
    end_datetime = target_dt.strftime("%Y-%m-%d 19:00:00")
    df = fetch_ts_minute_range_df(symbol, start_datetime, end_datetime, fqt)
    if df.empty:
        return df
    target_date = target_dt.strftime("%Y-%m-%d")
    return df[df["时间"].astype(str).str.startswith(target_date + " ")].reset_index(drop=True)
