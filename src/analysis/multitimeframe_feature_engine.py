import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .multitimeframe_physics_utils import calc_entropy, fast_hurst, rolling_zscore
else:
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from src.analysis.multitimeframe_physics_utils import calc_entropy, fast_hurst, rolling_zscore


def to_trade_date_str(df: pd.DataFrame) -> pd.DataFrame:
    if "trade_date_str" not in df.columns:
        if "trade_date" not in df.columns:
            return df
        df["trade_date_str"] = df["trade_date"].astype(str)
    df["trade_date_str"] = df["trade_date_str"].astype(str)
    df["dt"] = pd.to_datetime(df["trade_date_str"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)
    return df


def rolling_apply_1d(arr: np.ndarray, window: int, func) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    out = np.full(len(arr), np.nan, dtype=np.float64)
    if window <= 0:
        return out
    for i in range(window - 1, len(arr)):
        out[i] = float(func(arr[i - window + 1 : i + 1]))
    return out


def rolling_min(arr: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(arr, dtype="float64").rolling(window=int(window), min_periods=int(window)).min().to_numpy(dtype=np.float64)


def clip_term(values: np.ndarray | pd.Series, scale: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if scale <= 0:
        return np.full(len(arr), np.nan, dtype=np.float64)
    return np.clip(arr / float(scale), -1.0, 1.0)


def _numeric_series_or_default(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


def build_daily_bars(df_daily: pd.DataFrame) -> pd.DataFrame:
    df = to_trade_date_str(df_daily.copy())
    if df.empty:
        return pd.DataFrame()

    numeric_cols = ["open", "close", "amount", "turnover_rate", "net_mf_amount"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "open" not in df.columns or "close" not in df.columns:
        return pd.DataFrame()

    df["amount"] = _numeric_series_or_default(df, "amount", 0.0)
    df["turnover_rate"] = _numeric_series_or_default(df, "turnover_rate", 0.0)
    df["net_mf_amount"] = _numeric_series_or_default(df, "net_mf_amount", 0.0)

    bars = pd.DataFrame(
        {
            "bar_start_dt": df["dt"],
            "bar_end_dt": df["dt"],
            "bar_start": df["trade_date_str"].astype(str),
            "bar_end": df["trade_date_str"].astype(str),
            "open": df["open"],
            "close": df["close"],
            "amount_sum": df["amount"],
            "turnover_sum": df["turnover_rate"],
            "turnover_mean": df["turnover_rate"],
            "mf_sum": df["net_mf_amount"],
        }
    )
    bars = bars.dropna(subset=["open", "close"]).sort_values("bar_end_dt").reset_index(drop=True)
    bars["ret"] = pd.to_numeric(bars["close"], errors="coerce").pct_change()
    return bars


def aggregate_stock_bars(df_daily: pd.DataFrame, freq: str) -> pd.DataFrame:
    freq = str(freq).upper().strip()
    if freq == "D":
        return build_daily_bars(df_daily)

    df = to_trade_date_str(df_daily.copy())
    if df.empty:
        return pd.DataFrame()

    numeric_cols = ["open", "close", "amount", "turnover_rate", "net_mf_amount"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "open" not in df.columns or "close" not in df.columns:
        return pd.DataFrame()

    df["amount"] = _numeric_series_or_default(df, "amount", 0.0)
    df["turnover_rate"] = _numeric_series_or_default(df, "turnover_rate", 0.0)
    df["net_mf_amount"] = _numeric_series_or_default(df, "net_mf_amount", 0.0)

    if freq == "W":
        df["period"] = df["dt"].dt.to_period("W-FRI")
    elif freq == "M":
        df["period"] = df["dt"].dt.to_period("M")
    else:
        raise ValueError(f"Unsupported frequency: {freq}")

    grouped = df.groupby("period", sort=True)
    bars = pd.DataFrame(
        {
            "bar_start_dt": grouped["dt"].min(),
            "bar_end_dt": grouped["dt"].max(),
            "bar_start": grouped["trade_date_str"].first().astype(str),
            "bar_end": grouped["trade_date_str"].last().astype(str),
            "open": grouped["open"].first(),
            "close": grouped["close"].last(),
            "amount_sum": grouped["amount"].sum(),
            "turnover_sum": grouped["turnover_rate"].sum(),
            "turnover_mean": grouped["turnover_rate"].mean(),
            "mf_sum": grouped["net_mf_amount"].sum(),
        }
    ).reset_index(drop=True)
    bars = bars.dropna(subset=["open", "close"]).sort_values("bar_end_dt").reset_index(drop=True)
    bars["ret"] = pd.to_numeric(bars["close"], errors="coerce").pct_change()
    return bars


def compute_physics_state_features(
    bars: pd.DataFrame,
    window_s: int,
    window_l: int,
    entry_threshold: float,
    persist_bars: int,
    energy_min: float,
    order_min: float,
    phase_min: float,
) -> pd.DataFrame:
    if bars is None or not isinstance(bars, pd.DataFrame) or bars.empty:
        return pd.DataFrame()

    df = bars.copy().reset_index(drop=True)
    for col in ["open", "close", "ret", "mf_sum", "amount_sum", "turnover_sum", "turnover_mean"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "close"]).reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()

    close = df["close"].to_numpy(dtype=np.float64)
    ret = df["ret"].to_numpy(dtype=np.float64)
    mf_sum = df["mf_sum"].fillna(0.0).to_numpy(dtype=np.float64)
    amount_sum = df["amount_sum"].fillna(0.0).to_numpy(dtype=np.float64)
    turnover_sum = df["turnover_sum"].fillna(0.0).to_numpy(dtype=np.float64)

    w_s = int(window_s)
    w_l = int(window_l)
    minp_s = max(3, w_s // 3)
    minp_l = max(5, w_l // 3)

    close_s = pd.Series(close, dtype="float64")
    ret_s = pd.Series(ret, dtype="float64")

    ma_s = close_s.rolling(window=w_s, min_periods=minp_s).mean().to_numpy(dtype=np.float64)
    ma_l = close_s.rolling(window=w_l, min_periods=minp_l).mean().to_numpy(dtype=np.float64)
    prev_high_l = close_s.shift(1).rolling(window=w_l, min_periods=minp_l).max().to_numpy(dtype=np.float64)

    close_over_ma_s = close / np.where(ma_s == 0.0, np.nan, ma_s)
    ma_spread = ma_s / np.where(ma_l == 0.0, np.nan, ma_l) - 1.0
    breakout = close / np.where(prev_high_l == 0.0, np.nan, prev_high_l) - 1.0

    vol_s = ret_s.rolling(window=w_s, min_periods=minp_s).std(ddof=0).to_numpy(dtype=np.float64)
    vol_l = ret_s.rolling(window=w_l, min_periods=minp_l).std(ddof=0).to_numpy(dtype=np.float64)
    vol_ratio = vol_s / np.where(vol_l == 0.0, np.nan, vol_l) - 1.0
    vol_z = rolling_zscore(np.nan_to_num(vol_s, nan=np.nan), max(w_l, 12))
    turnover_z = rolling_zscore(turnover_sum, max(w_l, 12))
    amount_z = rolling_zscore(np.log1p(np.clip(amount_sum, 0.0, None)), max(w_l, 12))
    mf_z = rolling_zscore(mf_sum, max(w_l, 12))

    entropy_s = rolling_apply_1d(ret, w_s, calc_entropy)
    entropy_l = rolling_apply_1d(ret, w_l, calc_entropy)
    entropy_rel = (entropy_l - entropy_s) / np.where(entropy_l == 0.0, np.nan, np.abs(entropy_l))

    hurst_s = np.clip(rolling_apply_1d(close, w_s, fast_hurst), 0.0, 1.0)
    hurst_l = np.clip(rolling_apply_1d(close, w_l, fast_hurst), 0.0, 1.0)
    hurst_rel = hurst_s - hurst_l

    energy_term = np.clip((0.60 * mf_z + 0.40 * amount_z) / 2.5, -1.0, 1.0)
    temperature_term = np.clip((0.65 * turnover_z - 0.85 * vol_z) / 2.0, -1.0, 1.0)
    ma_term = clip_term(close_over_ma_s - 1.0, 0.10)
    spread_term = clip_term(ma_spread, 0.12)
    breakout_term = clip_term(breakout, 0.08)
    fractal_term = clip_term(hurst_rel, 0.12)
    entropy_term = clip_term(entropy_rel, 0.25)

    order_term = np.clip(0.45 * ma_term + 0.35 * spread_term + 0.20 * breakout_term, -1.0, 1.0)
    switch_term = np.clip(0.40 * breakout_term + 0.30 * entropy_term + 0.30 * fractal_term, -1.0, 1.0)
    phase_term = np.clip(0.55 * entropy_term + 0.45 * fractal_term, -1.0, 1.0)

    score = np.clip(
        0.25 * energy_term
        + 0.15 * temperature_term
        + 0.30 * order_term
        + 0.20 * phase_term
        + 0.10 * switch_term,
        -1.0,
        1.0,
    )
    score_min_persist = rolling_min(score, max(1, int(persist_bars)))

    state = (
        np.isfinite(score_min_persist)
        & (score_min_persist >= float(entry_threshold))
        & np.isfinite(energy_term)
        & (energy_term >= float(energy_min))
        & np.isfinite(order_term)
        & (order_term >= float(order_min))
        & np.isfinite(phase_term)
        & (phase_term >= float(phase_min))
    )

    df["ma_s"] = ma_s
    df["ma_l"] = ma_l
    df["close_over_ma_s"] = close_over_ma_s
    df["ma_spread"] = ma_spread
    df["breakout"] = breakout
    df["vol_s"] = vol_s
    df["vol_l"] = vol_l
    df["vol_ratio"] = vol_ratio
    df["entropy_s"] = entropy_s
    df["entropy_l"] = entropy_l
    df["entropy_rel"] = entropy_rel
    df["hurst_s"] = hurst_s
    df["hurst_l"] = hurst_l
    df["hurst_rel"] = hurst_rel
    df["energy_term"] = energy_term
    df["temperature_term"] = temperature_term
    df["order_term"] = order_term
    df["switch_term"] = switch_term
    df["phase_term"] = phase_term
    df["score"] = score
    df["score_min_persist"] = score_min_persist
    df["state"] = state
    return df