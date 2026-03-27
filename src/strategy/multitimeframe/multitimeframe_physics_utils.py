import numpy as np
import pandas as pd


def calc_entropy(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    if len(x) < 5:
        return 999.0
    hist, _ = np.histogram(x, bins=10)
    total = float(np.sum(hist))
    if total <= 0:
        return 999.0
    p = hist / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def fast_hurst(ts: np.ndarray) -> float:
    ts = np.asarray(ts, dtype=np.float64)
    if len(ts) < 10:
        return 0.5
    lags = [2, 4, 8, 16]
    tau: list[int] = []
    msd: list[float] = []
    for lag in lags:
        if lag >= len(ts):
            break
        diffs = ts[lag:] - ts[:-lag]
        val = float(np.mean(diffs**2))
        if val > 0:
            msd.append(val)
            tau.append(lag)
    if len(tau) < 2:
        return 0.5
    m = np.polyfit(np.log(tau), np.log(msd), 1)
    return float(m[0] / 2.0)


def rolling_apply_1d(arr: np.ndarray, window: int, func) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    out = np.full(len(arr), np.nan, dtype=np.float64)
    if window <= 0:
        return out
    for i in range(window - 1, len(arr)):
        out[i] = float(func(arr[i - window + 1 : i + 1]))
    return out


def rolling_zscore(arr: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(arr, dtype="float64")
    mu = s.rolling(window=window, min_periods=max(3, window // 3)).mean()
    sd = s.rolling(window=window, min_periods=max(3, window // 3)).std(ddof=0)
    z = (s - mu) / sd.replace(0.0, np.nan)
    return z.to_numpy(dtype=np.float64)


def smooth_regime_with_hysteresis(raw: list[str], persist: int = 2) -> list[str]:
    if not raw:
        return raw
    persist = max(1, int(persist))
    smoothed: list[str] = [raw[0]]
    pending: str | None = None
    pending_count = 0

    for r in raw[1:]:
        cur = smoothed[-1]
        if r == cur:
            pending = None
            pending_count = 0
            smoothed.append(cur)
            continue

        if pending == r:
            pending_count += 1
        else:
            pending = r
            pending_count = 1

        if pending_count >= persist:
            smoothed.append(r)
            pending = None
            pending_count = 0
        else:
            smoothed.append(cur)

    return smoothed


def build_index_monthly_regime_by_date(
    df_index_daily: pd.DataFrame,
    window_s: int = 12,
    window_l: int = 36,
    hysteresis_months: int = 2,
) -> tuple[pd.DataFrame, dict[str, str]]:
    if df_index_daily is None or not isinstance(df_index_daily, pd.DataFrame) or df_index_daily.empty:
        return pd.DataFrame(), {}

    df = df_index_daily.copy()
    if "trade_date_str" not in df.columns:
        if "trade_date" not in df.columns:
            return pd.DataFrame(), {}
        df["trade_date_str"] = df["trade_date"].astype(str)

    df["dt"] = pd.to_datetime(df["trade_date_str"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)
    if df.empty or "close" not in df.columns:
        return pd.DataFrame(), {}

    close = pd.to_numeric(df["close"], errors="coerce")
    if "net_mf_amount" in df.columns:
        net_mf = pd.to_numeric(df["net_mf_amount"], errors="coerce").fillna(0.0)
    else:
        net_mf = pd.Series(np.zeros(len(df), dtype=np.float64))

    dfi = pd.DataFrame(
        {
            "close": close.to_numpy(dtype=np.float64),
            "net_mf_amount": net_mf.to_numpy(dtype=np.float64),
        },
        index=pd.DatetimeIndex(df["dt"].to_numpy(), name="dt"),
    )
    dfi = dfi.dropna(subset=["close"]).sort_index()
    if dfi.empty:
        return pd.DataFrame(), {}

    monthly_close = dfi["close"].resample("ME").last()
    monthly_ret = monthly_close.pct_change()
    monthly_mf_sum = dfi["net_mf_amount"].resample("ME").sum()

    m = pd.DataFrame({"close": monthly_close, "ret": monthly_ret, "mf_sum": monthly_mf_sum}).dropna(
        subset=["close"]
    )

    w_s = int(window_s)
    w_l = int(window_l)

    m["ma_s"] = m["close"].rolling(window=w_s, min_periods=max(3, w_s // 3)).mean()
    m["close_over_ma_s"] = m["close"] / m["ma_s"]

    m["entropy_s"] = rolling_apply_1d(m["ret"].to_numpy(dtype=np.float64), w_s, calc_entropy)
    m["entropy_l"] = rolling_apply_1d(m["ret"].to_numpy(dtype=np.float64), w_l, calc_entropy)
    m["entropy_s_over_l"] = (m["entropy_s"] - m["entropy_l"]) / m["entropy_l"].replace(0.0, np.nan)

    m["hurst_s"] = rolling_apply_1d(m["close"].to_numpy(dtype=np.float64), w_s, fast_hurst)
    m["hurst_l"] = rolling_apply_1d(m["close"].to_numpy(dtype=np.float64), w_l, fast_hurst)
    m["hurst_s"] = m["hurst_s"].clip(0.0, 1.0)
    m["hurst_l"] = m["hurst_l"].clip(0.0, 1.0)
    m["hurst_s_over_l"] = m["hurst_s"] - m["hurst_l"]

    m["mf_z_12p"] = rolling_zscore(m["mf_sum"].to_numpy(dtype=np.float64), 12)

    raw: list[str] = []
    for _, row in m.iterrows():
        c_ma = float(row["close_over_ma_s"]) if np.isfinite(row["close_over_ma_s"]) else np.nan
        h_s = float(row["hurst_s"]) if np.isfinite(row["hurst_s"]) else np.nan
        mfz = float(row["mf_z_12p"]) if np.isfinite(row["mf_z_12p"]) else 0.0
        ent_rel = float(row["entropy_s_over_l"]) if np.isfinite(row["entropy_s_over_l"]) else 0.0

        if np.isfinite(c_ma) and np.isfinite(h_s) and (c_ma >= 1.02) and (h_s >= 0.55) and (mfz >= -0.25):
            raw.append("BULL")
        elif np.isfinite(c_ma) and ((c_ma <= 0.98) or (mfz <= -0.75)) and (ent_rel > 0.0):
            raw.append("RISK")
        else:
            raw.append("BASE")

    m["index_regime_raw"] = raw
    m["index_regime"] = smooth_regime_with_hysteresis(raw, persist=hysteresis_months)
    m.index.name = "month_end"

    month_reg = m[["index_regime"]].reset_index()
    daily = df[["dt", "trade_date_str"]].sort_values("dt").reset_index(drop=True)
    merged = pd.merge_asof(
        daily,
        month_reg.sort_values("month_end"),
        left_on="dt",
        right_on="month_end",
        direction="backward",
    )
    merged["index_regime"] = merged["index_regime"].bfill().fillna("BASE")
    by_date = dict(zip(merged["trade_date_str"].astype(str), merged["index_regime"].astype(str)))

    m = m.reset_index()
    return m, by_date