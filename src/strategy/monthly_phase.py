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
    """Build a monthly regime state machine for an index from daily bars.

    Returns:
    - monthly_df: month-end rows with features + regime
    - by_daily_trade_date: mapping trade_date_str(YYYYMMDD) -> regime for that day

    Regimes are coarse and stable: BULL / BASE / RISK.
    """

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


def daily_to_monthly_stock_bars(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Convert daily stock bars to monthly bars using trading-day data.

    Output columns:
    - month_end_dt (Timestamp, month-end trading date)
    - month_end (YYYYMMDD string)
    - open (first open of month)
    - close (last close of month)
    - ret (monthly close-to-close return)
    - mf_sum (sum of net_mf_amount)
    """

    if df_daily is None or not isinstance(df_daily, pd.DataFrame) or df_daily.empty:
        return pd.DataFrame()

    df = df_daily.copy()
    if "trade_date_str" not in df.columns:
        if "trade_date" not in df.columns:
            return pd.DataFrame()
        df["trade_date_str"] = df["trade_date"].astype(str)

    for col in ["open", "close"]:
        if col not in df.columns:
            return pd.DataFrame()

    df["dt"] = pd.to_datetime(df["trade_date_str"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()

    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    if "net_mf_amount" in df.columns:
        df["net_mf_amount"] = pd.to_numeric(df["net_mf_amount"], errors="coerce").fillna(0.0)
    else:
        df["net_mf_amount"] = 0.0

    df["ym"] = df["dt"].dt.to_period("M")

    g = df.groupby("ym", sort=True)
    out = pd.DataFrame(
        {
            "month_start_dt": g["dt"].min(),
            "month_end_dt": g["dt"].max(),
            "month_start": g["trade_date_str"].first().astype(str),
            "month_end": g["trade_date_str"].last().astype(str),
            "open": g["open"].first(),
            "close": g["close"].last(),
            "mf_sum": g["net_mf_amount"].sum(),
        }
    ).reset_index(drop=True)

    out = out.sort_values("month_end_dt").reset_index(drop=True)
    out["month"] = out["month_end"].astype(str).str.slice(0, 6)
    out["ret"] = out["close"].pct_change()
    return out


def compute_monthly_stock_score(
    m: pd.DataFrame,
    window_s: int = 12,
    window_l: int = 36,
    score_mode: str = "full",
) -> pd.DataFrame:
    """Compute monthly features + a bounded score in [-1,1].

    score_mode:
    - "full": entropy + hurst + MA deviation + moneyflow zscore (default, backward compatible)
    - "thermo": entropy + hurst only (thermodynamics / nonlinear-system proxies)
    """

    if m is None or not isinstance(m, pd.DataFrame) or m.empty:
        return pd.DataFrame()

    m = m.copy().reset_index(drop=True)
    if "close" not in m.columns or "ret" not in m.columns or "mf_sum" not in m.columns:
        return pd.DataFrame()

    close = pd.to_numeric(m["close"], errors="coerce").to_numpy(dtype=np.float64)
    ret = pd.to_numeric(m["ret"], errors="coerce").to_numpy(dtype=np.float64)
    mf_sum = pd.to_numeric(m["mf_sum"], errors="coerce").to_numpy(dtype=np.float64)

    w_s = int(window_s)
    w_l = int(window_l)

    ma_s = pd.Series(close).rolling(window=w_s, min_periods=max(3, w_s // 3)).mean().to_numpy(dtype=np.float64)
    close_over_ma = close / np.where(ma_s == 0, np.nan, ma_s)

    entropy_s = rolling_apply_1d(ret, w_s, calc_entropy)
    entropy_l = rolling_apply_1d(ret, w_l, calc_entropy)
    entropy_rel = (entropy_l - entropy_s) / np.where(entropy_l == 0, np.nan, entropy_l)

    hurst_s = rolling_apply_1d(close, w_s, fast_hurst)
    hurst_l = rolling_apply_1d(close, w_l, fast_hurst)
    hurst_s = np.clip(hurst_s, 0.0, 1.0)
    hurst_l = np.clip(hurst_l, 0.0, 1.0)
    hurst_rel = hurst_s - hurst_l

    mf_z = rolling_zscore(mf_sum, 12)

    # Terms roughly in [-1, 1]
    hurst_term = np.clip((hurst_s - 0.45) / 0.2, -1.0, 1.0)
    entropy_term = np.clip(entropy_rel / 0.25, -1.0, 1.0)
    ma_term = np.clip((close_over_ma - 1.0) / 0.10, -1.0, 1.0)
    mf_term = np.clip(mf_z / 2.0, -1.0, 1.0)

    mode = str(score_mode or "full").strip().lower()
    if mode == "thermo":
        score = 0.60 * hurst_term + 0.40 * entropy_term
    else:
        score = 0.35 * hurst_term + 0.25 * entropy_term + 0.25 * ma_term + 0.15 * mf_term
    score = np.clip(score, -1.0, 1.0)

    m["ma_s"] = ma_s
    m["close_over_ma_s"] = close_over_ma
    m["entropy_s"] = entropy_s
    m["entropy_l"] = entropy_l
    m["entropy_rel"] = entropy_rel
    m["hurst_s"] = hurst_s
    m["hurst_l"] = hurst_l
    m["hurst_rel"] = hurst_rel
    m["mf_z_12p"] = mf_z
    m["score"] = score
    return m


def compute_intramonth_monthly_score_series(
    df_daily: pd.DataFrame,
    start_date: str,
    end_date: str,
    window_s: int = 12,
    window_l: int = 36,
    score_mode: str = "full",
) -> pd.DataFrame:
    """Compute a *daily-updated* monthly score using the current month-to-date bar.

    Conceptually, for each trading day t, we treat the current month as a partial monthly bar
    (close=close(t), mf_sum=sum(net_mf_amount within month up to t), ret=close(t)/prev_month_close-1)
    appended to the completed monthly history, and compute the score for this partial month.

    Returns a DataFrame with columns:
    - trade_date_str (YYYYMMDD)
    - month (YYYYMM)
    - score (float)

    Notes:
    - The score formula matches `compute_monthly_stock_score(..., score_mode=...)` for the last element.
    - This is intended for signal generation at day t close; execution should be at t+1 open.
    """

    if df_daily is None or not isinstance(df_daily, pd.DataFrame) or df_daily.empty:
        return pd.DataFrame()

    df = df_daily.copy()
    if "trade_date_str" not in df.columns:
        if "trade_date" not in df.columns:
            return pd.DataFrame()
        df["trade_date_str"] = df["trade_date"].astype(str)

    need_cols = {"open", "close"}
    if not need_cols.issubset(set(df.columns)):
        return pd.DataFrame()

    df["trade_date_str"] = df["trade_date_str"].astype(str)
    df = df[(df["trade_date_str"] >= "20200101") & (df["trade_date_str"] <= str(end_date))].copy()
    if df.empty:
        return pd.DataFrame()

    df["dt"] = pd.to_datetime(df["trade_date_str"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()

    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    if "net_mf_amount" in df.columns:
        df["net_mf_amount"] = pd.to_numeric(df["net_mf_amount"], errors="coerce").fillna(0.0)
    else:
        df["net_mf_amount"] = 0.0

    # Completed monthly history
    m_all = daily_to_monthly_stock_bars(df)
    if m_all is None or not isinstance(m_all, pd.DataFrame) or m_all.empty:
        return pd.DataFrame()

    w_s = int(window_s)
    w_l = int(window_l)
    minp_s = max(3, w_s // 3)
    minp_mf = max(3, 12 // 3)

    # Month-to-date partial aggregates (vectorized within each month)
    df["month"] = df["trade_date_str"].str.slice(0, 6)
    df["mf_mtd"] = df.groupby("month", sort=False)["net_mf_amount"].cumsum()

    # Only compute for requested date range (signals at day close)
    df_sig = df[(df["trade_date_str"] >= str(start_date)) & (df["trade_date_str"] <= str(end_date))].copy()
    if df_sig.empty:
        return pd.DataFrame()

    out_rows: list[dict[str, object]] = []

    # Process month by month to reuse the completed-month base arrays.
    for month, g in df_sig.groupby("month", sort=True):
        g = g.sort_values("dt").reset_index(drop=True)

        base = m_all[m_all["month"].astype(str) < str(month)].copy()
        if base.empty:
            continue
        # Keep only the tail needed for window_l
        base = base.tail(max(w_l - 1, 40)).reset_index(drop=True)

        base_close = pd.to_numeric(base["close"], errors="coerce").to_numpy(dtype=np.float64)
        base_ret = pd.to_numeric(base["ret"], errors="coerce").to_numpy(dtype=np.float64)
        base_mf = pd.to_numeric(base["mf_sum"], errors="coerce").to_numpy(dtype=np.float64)
        base_close = base_close[np.isfinite(base_close)]
        # ret and mf arrays should remain aligned with close length; if not, skip.
        if len(base_close) < (w_l - 1):
            continue

        prev_month_close = float(base_close[-1])
        if not np.isfinite(prev_month_close) or prev_month_close <= 0:
            continue

        # Ensure aligned lengths by trimming to the minimum.
        n_base = min(len(base_close), len(base_ret), len(base_mf))
        base_close = base_close[-n_base:]
        base_ret = base_ret[-n_base:]
        base_mf = base_mf[-n_base:]

        close_arr = np.empty(n_base + 1, dtype=np.float64)
        ret_arr = np.empty(n_base + 1, dtype=np.float64)
        mf_arr = np.empty(n_base + 1, dtype=np.float64)
        close_arr[:-1] = base_close
        ret_arr[:-1] = base_ret
        mf_arr[:-1] = base_mf

        for _, row in g.iterrows():
            trade_date = str(row["trade_date_str"])
            c = float(row["close"])
            mf_mtd = float(row["mf_mtd"])
            if not np.isfinite(c) or c <= 0:
                continue

            close_arr[-1] = c
            mf_arr[-1] = mf_mtd
            ret_arr[-1] = (c / prev_month_close) - 1.0

            # --- Rolling features (last element only) ---
            # MA term
            ma_s = np.nan
            close_over_ma = np.nan
            if len(close_arr) >= minp_s:
                w = min(w_s, len(close_arr))
                ma_s = float(np.nanmean(close_arr[-w:]))
                if np.isfinite(ma_s) and ma_s != 0:
                    close_over_ma = float(close_arr[-1] / ma_s)

            # Entropy term
            entropy_s = np.nan
            entropy_l = np.nan
            entropy_rel = np.nan
            if len(ret_arr) >= w_s:
                entropy_s = float(calc_entropy(ret_arr[-w_s:]))
            if len(ret_arr) >= w_l:
                entropy_l = float(calc_entropy(ret_arr[-w_l:]))
            if np.isfinite(entropy_s) and np.isfinite(entropy_l) and entropy_l != 0:
                entropy_rel = float((entropy_l - entropy_s) / entropy_l)

            # Hurst term
            hurst_s = np.nan
            hurst_l = np.nan
            if len(close_arr) >= w_s:
                hurst_s = float(fast_hurst(close_arr[-w_s:]))
            if len(close_arr) >= w_l:
                hurst_l = float(fast_hurst(close_arr[-w_l:]))
            if np.isfinite(hurst_s):
                hurst_s = float(np.clip(hurst_s, 0.0, 1.0))
            if np.isfinite(hurst_l):
                hurst_l = float(np.clip(hurst_l, 0.0, 1.0))

            # Moneyflow zscore (12-month window)
            mf_z = np.nan
            if len(mf_arr) >= minp_mf:
                w = min(12, len(mf_arr))
                x = mf_arr[-w:]
                mu = float(np.mean(x))
                sd = float(np.std(x, ddof=0))
                if sd > 0:
                    mf_z = float((mf_arr[-1] - mu) / sd)

            # --- Score formula (same as compute_monthly_stock_score) ---
            hurst_term = np.nan
            entropy_term = np.nan
            ma_term = np.nan
            mf_term = np.nan

            if np.isfinite(hurst_s):
                hurst_term = float(np.clip((hurst_s - 0.45) / 0.2, -1.0, 1.0))
            if np.isfinite(entropy_rel):
                entropy_term = float(np.clip(entropy_rel / 0.25, -1.0, 1.0))
            if np.isfinite(close_over_ma):
                ma_term = float(np.clip((close_over_ma - 1.0) / 0.10, -1.0, 1.0))
            if np.isfinite(mf_z):
                mf_term = float(np.clip(mf_z / 2.0, -1.0, 1.0))

            # Missing terms default to 0.0 (matches earlier defensive handling patterns)
            mode = str(score_mode or "full").strip().lower()
            if mode == "thermo":
                score = 0.0
                score += 0.60 * (hurst_term if np.isfinite(hurst_term) else 0.0)
                score += 0.40 * (entropy_term if np.isfinite(entropy_term) else 0.0)
            else:
                score = 0.0
                score += 0.35 * (hurst_term if np.isfinite(hurst_term) else 0.0)
                score += 0.25 * (entropy_term if np.isfinite(entropy_term) else 0.0)
                score += 0.25 * (ma_term if np.isfinite(ma_term) else 0.0)
                score += 0.15 * (mf_term if np.isfinite(mf_term) else 0.0)
            score = float(np.clip(score, -1.0, 1.0))

            out_rows.append(
                {
                    "trade_date_str": trade_date,
                    "month": str(month),
                    "score": float(score),
                }
            )

    out = pd.DataFrame(out_rows)
    if out.empty:
        return out
    out = out.drop_duplicates(subset=["trade_date_str"], keep="last")
    out = out.sort_values("trade_date_str").reset_index(drop=True)
    return out
