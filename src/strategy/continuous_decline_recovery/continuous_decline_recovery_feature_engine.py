import numpy as np
import pandas as pd


def _numeric_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def _safe_div(num: pd.Series | np.ndarray, den: pd.Series | np.ndarray) -> np.ndarray:
    numerator = np.asarray(num, dtype=np.float64)
    denominator = np.asarray(den, dtype=np.float64)
    out = np.full(len(numerator), np.nan, dtype=np.float64)
    mask = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > 1e-12)
    out[mask] = numerator[mask] / denominator[mask]
    return out


def _scaled_pos(values: pd.Series | np.ndarray, low: float, high: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if high <= low:
        return np.zeros(len(array), dtype=np.float64)
    return np.clip((array - float(low)) / float(high - low), 0.0, 1.0)


def _triangular_score(values: pd.Series | np.ndarray, low: float, mid: float, high: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    out = np.zeros(len(array), dtype=np.float64)
    if not (low < mid < high):
        return out

    left_mask = np.isfinite(array) & (array >= low) & (array <= mid)
    right_mask = np.isfinite(array) & (array > mid) & (array <= high)
    out[left_mask] = (array[left_mask] - float(low)) / float(mid - low)
    out[right_mask] = (float(high) - array[right_mask]) / float(high - mid)
    return np.clip(out, 0.0, 1.0)


def to_trade_date_str(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    if "trade_date_str" not in out.columns:
        out["trade_date_str"] = out["trade_date"].astype(str)
    out["trade_date_str"] = out["trade_date_str"].astype(str)
    out["dt"] = pd.to_datetime(out["trade_date_str"], format="%Y%m%d", errors="coerce")
    out = out.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)
    return out


def build_continuous_decline_recovery_feature_frame(df_daily: pd.DataFrame) -> pd.DataFrame:
    df = to_trade_date_str(df_daily.copy())
    if df.empty:
        return pd.DataFrame()

    required_columns = ["open", "high", "low", "close"]
    if any(column not in df.columns for column in required_columns):
        return pd.DataFrame()

    numeric_columns = ["open", "high", "low", "close", "amount", "turnover_rate", "net_mf_amount"]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    close = pd.to_numeric(df["close"], errors="coerce")
    open_ = pd.to_numeric(df["open"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    amount = _numeric_series(df, "amount", 0.0).fillna(0.0)
    turnover_rate = _numeric_series(df, "turnover_rate", 0.0).fillna(0.0)
    net_mf_amount = _numeric_series(df, "net_mf_amount", 0.0).fillna(0.0)

    out = pd.DataFrame(
        {
            "trade_date": df["trade_date_str"].astype(str),
            "trade_date_str": df["trade_date_str"].astype(str),
            "dt": df["dt"],
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "amount": amount,
            "turnover_rate": turnover_rate,
            "net_mf_amount": net_mf_amount,
        }
    )

    out["ret_1"] = close.pct_change(fill_method=None)
    out["ret_3"] = close.pct_change(3, fill_method=None)
    out["ret_5"] = close.pct_change(5, fill_method=None)
    out["ret_10"] = close.pct_change(10, fill_method=None)
    out["ret_20"] = close.pct_change(20, fill_method=None)
    out["ret_60"] = close.pct_change(60, fill_method=None)

    out["ma_5"] = close.rolling(5, min_periods=5).mean()
    out["ma_10"] = close.rolling(10, min_periods=10).mean()
    out["ma_20"] = close.rolling(20, min_periods=20).mean()
    out["ma_60"] = close.rolling(60, min_periods=40).mean()

    out["amount_ma_20"] = amount.rolling(20, min_periods=10).mean()
    out["turnover_ma_20"] = turnover_rate.rolling(20, min_periods=10).mean()
    out["amount_ratio_20"] = _safe_div(amount, out["amount_ma_20"])
    out["turnover_ratio_20"] = _safe_div(turnover_rate, out["turnover_ma_20"])

    flow_ratio = _safe_div(net_mf_amount, amount)
    out["flow_ratio"] = flow_ratio
    out["flow_ratio_5"] = pd.Series(flow_ratio, index=out.index, dtype="float64").rolling(5, min_periods=3).mean()

    out["rolling_high_20"] = high.rolling(20, min_periods=10).max()
    out["rolling_high_60"] = high.rolling(60, min_periods=30).max()
    out["rolling_low_10"] = low.rolling(10, min_periods=5).min()
    out["rolling_low_20"] = low.rolling(20, min_periods=10).min()

    out["drawdown_20"] = _safe_div(close, out["rolling_high_20"]) - 1.0
    out["drawdown_60"] = _safe_div(close, out["rolling_high_60"]) - 1.0
    out["rebound_from_low_10"] = _safe_div(close, out["rolling_low_10"]) - 1.0
    out["rebound_from_low_20"] = _safe_div(close, out["rolling_low_20"]) - 1.0
    out["close_to_ma20"] = _safe_div(close, out["ma_20"]) - 1.0

    true_range = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr_ratio_14"] = _safe_div(true_range.rolling(14, min_periods=7).mean(), close)
    out["volatility_10"] = out["ret_1"].rolling(10, min_periods=6).std() * np.sqrt(10.0)
    out["negative_days_7"] = out["ret_1"].lt(0.0).rolling(7, min_periods=4).sum()
    out["ma_5_slope_3"] = out["ma_5"].pct_change(3, fill_method=None)

    out["close_above_ma5"] = close >= out["ma_5"]
    out["close_above_ma10"] = close >= out["ma_10"]
    out["close_above_ma20"] = close >= out["ma_20"]
    out["close_below_ma20"] = close < out["ma_20"]

    damage_score = (
        0.35 * _scaled_pos(-out["ret_10"], 0.01, 0.10)
        + 0.25 * _scaled_pos(-out["drawdown_20"], 0.04, 0.18)
        + 0.25 * _scaled_pos(-out["drawdown_60"], 0.08, 0.30)
        + 0.15 * _scaled_pos(out["negative_days_7"], 3.0, 7.0)
    )
    repair_score = (
        0.35 * _scaled_pos(out["ret_3"], -0.01, 0.08)
        + 0.20 * _scaled_pos(out["amount_ratio_20"], 0.80, 1.80)
        + 0.20 * _scaled_pos(out["flow_ratio_5"], -0.02, 0.06)
        + 0.15 * out["close_above_ma5"].astype(np.float64)
        + 0.10 * _scaled_pos(out["ma_5_slope_3"], -0.01, 0.03)
    )
    entry_window_score = (
        0.65 * _triangular_score(out["rebound_from_low_10"], 0.02, 0.07, 0.16)
        + 0.35 * _triangular_score(out["close_to_ma20"], -0.08, -0.01, 0.08)
    )
    flow_support_score = (
        0.50 * _scaled_pos(out["flow_ratio_5"], -0.01, 0.05)
        + 0.30 * _scaled_pos(out["amount_ratio_20"], 0.90, 1.80)
        + 0.20 * _scaled_pos(out["turnover_ratio_20"], 0.85, 1.50)
    )
    stability_score = (
        0.45 * _scaled_pos(out["close_to_ma20"], -0.08, 0.04)
        + 0.25 * (1.0 - _scaled_pos(out["volatility_10"], 0.02, 0.09))
        + 0.15 * (1.0 - _scaled_pos(out["atr_ratio_14"], 0.01, 0.06))
        + 0.15 * out["close_above_ma10"].astype(np.float64)
    )
    overheat_score = (
        0.55 * _scaled_pos(out["rebound_from_low_10"], 0.12, 0.25)
        + 0.45 * _scaled_pos(out["ret_5"], 0.06, 0.16)
    )

    out["damage_score"] = np.clip(damage_score, 0.0, 1.0)
    out["repair_score"] = np.clip(repair_score, 0.0, 1.0)
    out["entry_window_score"] = np.clip(entry_window_score, 0.0, 1.0)
    out["flow_support_score"] = np.clip(flow_support_score, 0.0, 1.0)
    out["stability_score"] = np.clip(stability_score, 0.0, 1.0)
    out["overheat_score"] = np.clip(overheat_score, 0.0, 1.0)

    out["repair_flag"] = (
        out["close_above_ma5"]
        & (out["ret_3"] >= 0.01)
        & (out["amount_ratio_20"] >= 0.90)
    )
    out["early_entry_flag"] = (
        out["close_above_ma5"]
        & (out["rebound_from_low_10"] >= 0.03)
        & (out["rebound_from_low_10"] <= 0.15)
        & (out["ret_3"] >= 0.01)
        & (out["flow_ratio_5"] >= -0.01)
    )
    out["overheat_flag"] = out["overheat_score"] >= 0.60
    out["base_candidate_flag"] = (
        (out["damage_score"] >= 0.32)
        & (out["repair_score"] >= 0.45)
        & (out["entry_window_score"] >= 0.40)
        & (out["flow_support_score"] >= 0.35)
        & (~out["overheat_flag"])
    )

    return out.reset_index(drop=True)