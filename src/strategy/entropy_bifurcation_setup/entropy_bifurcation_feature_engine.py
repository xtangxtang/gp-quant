import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from ..complexity.complexity_feature_engine import build_complexity_feature_frame
else:
    _PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from src.strategy.complexity.complexity_feature_engine import build_complexity_feature_frame


def _rolling_apply_1d(arr: np.ndarray, window: int, func) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float64)
    out = np.full(len(values), np.nan, dtype=np.float64)
    if window <= 0:
        return out
    for idx in range(window - 1, len(values)):
        out[idx] = float(func(values[idx - window + 1 : idx + 1]))
    return out


def _permutation_entropy(window_values: np.ndarray, order: int = 3, delay: int = 1) -> float:
    values = np.asarray(window_values, dtype=np.float64)
    values = values[np.isfinite(values)]
    n_required = (order - 1) * delay + 1
    if len(values) < max(n_required + 2, 8):
        return np.nan

    pattern_counts: dict[tuple[int, ...], int] = {}
    total = 0
    for start in range(len(values) - (order - 1) * delay):
        segment = values[start : start + order * delay : delay]
        if len(segment) != order or not np.all(np.isfinite(segment)):
            continue
        pattern = tuple(np.argsort(segment, kind="mergesort"))
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        total += 1

    if total <= 1:
        return np.nan
    probabilities = np.asarray(list(pattern_counts.values()), dtype=np.float64) / float(total)
    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(entropy / np.log(float(math.factorial(order))))


def _rolling_percentile_of_last(arr: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float64)
    out = np.full(len(values), np.nan, dtype=np.float64)
    if window <= 0:
        return out
    for idx in range(window - 1, len(values)):
        sample = values[idx - window + 1 : idx + 1]
        sample = sample[np.isfinite(sample)]
        if len(sample) < max(10, window // 3):
            continue
        current = sample[-1]
        less = float(np.sum(sample < current))
        equal = float(np.sum(sample == current))
        out[idx] = (less + 0.5 * equal) / float(len(sample))
    return out


def _rolling_ar1(arr: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float64)
    out = np.full(len(values), np.nan, dtype=np.float64)
    if window <= 1:
        return out
    for idx in range(window - 1, len(values)):
        sample = values[idx - window + 1 : idx + 1]
        sample = sample[np.isfinite(sample)]
        if len(sample) < max(8, window // 2):
            continue
        prev = sample[:-1]
        curr = sample[1:]
        if len(prev) < 3:
            continue
        prev_centered = prev - float(np.mean(prev))
        curr_centered = curr - float(np.mean(curr))
        denom = float(np.dot(prev_centered, prev_centered))
        if denom <= 1e-12:
            continue
        phi = float(np.dot(prev_centered, curr_centered) / denom)
        out[idx] = phi
    return out


def _safe_div(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray) -> np.ndarray:
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    out = np.full(len(num), np.nan, dtype=np.float64)
    mask = np.isfinite(num) & np.isfinite(den) & (den != 0.0)
    out[mask] = num[mask] / den[mask]
    return out


def build_entropy_bifurcation_feature_frame(df_daily: pd.DataFrame) -> pd.DataFrame:
    base = build_complexity_feature_frame(df_daily)
    if base.empty:
        return pd.DataFrame()

    out = base.copy().sort_values("bar_end_dt").reset_index(drop=True)
    close = pd.to_numeric(out["close"], errors="coerce")
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    amount = pd.to_numeric(out["amount"], errors="coerce").fillna(0.0)
    net_mf = pd.to_numeric(out["net_mf_amount"], errors="coerce").fillna(0.0)

    safe_close = close.where(close > 0.0)
    log_close = np.log(safe_close)
    log_ret_1 = log_close.diff()

    log_close_ma_20 = log_close.rolling(window=20, min_periods=10).mean()
    log_close_std_20 = log_close.rolling(window=20, min_periods=10).std(ddof=0)
    detrended_close_20 = (log_close - log_close_ma_20) / log_close_std_20.replace(0.0, np.nan)

    range_low_20 = low.rolling(window=20, min_periods=10).min()
    range_high_20 = high.rolling(window=20, min_periods=10).max()
    range_state_20 = _safe_div(close - range_low_20, range_high_20 - range_low_20)

    perm_entropy_20_norm = _rolling_apply_1d(detrended_close_20.to_numpy(dtype=np.float64), 20, _permutation_entropy)
    perm_entropy_60_norm = _rolling_apply_1d(detrended_close_20.to_numpy(dtype=np.float64), 60, _permutation_entropy)
    entropy_gap = perm_entropy_60_norm - perm_entropy_20_norm
    entropy_percentile_120 = _rolling_percentile_of_last(perm_entropy_20_norm, 120)

    ar1_20 = _rolling_ar1(detrended_close_20.to_numpy(dtype=np.float64), 20)
    recovery_rate_20 = 1.0 - ar1_20
    state_var_10 = detrended_close_20.rolling(window=10, min_periods=5).var(ddof=0)
    state_var_20 = detrended_close_20.rolling(window=20, min_periods=10).var(ddof=0)
    state_skew_20 = detrended_close_20.rolling(window=20, min_periods=10).skew()
    var_lift_10_20 = state_var_10 / state_var_20.replace(0.0, np.nan) - 1.0

    prev_high_10 = high.shift(1).rolling(window=10, min_periods=5).max()
    breakout_10 = close / prev_high_10 - 1.0

    amount_ma_5 = amount.rolling(window=5, min_periods=3).mean()
    amount_ma_20 = amount.rolling(window=20, min_periods=10).mean()
    volume_impulse_5_20 = amount_ma_5 / amount_ma_20.replace(0.0, np.nan) - 1.0

    flow_ma_5 = net_mf.rolling(window=5, min_periods=3).mean()
    flow_scale_20 = net_mf.abs().rolling(window=20, min_periods=10).mean()
    flow_impulse_5_20 = flow_ma_5 / flow_scale_20.replace(0.0, np.nan)

    entropy_floor_flag = (pd.Series(entropy_percentile_120, dtype="float64") <= 0.25).to_numpy(dtype=np.float64)
    entropy_expand_flag = (pd.Series(entropy_gap, dtype="float64") >= 0.03).to_numpy(dtype=np.float64)
    out["log_ret_1"] = log_ret_1
    out["detrended_close_20"] = detrended_close_20
    out["range_state_20"] = range_state_20
    out["perm_entropy_20_norm"] = perm_entropy_20_norm
    out["perm_entropy_60_norm"] = perm_entropy_60_norm
    out["entropy_gap"] = entropy_gap
    out["entropy_percentile_120"] = entropy_percentile_120
    out["ar1_20"] = ar1_20
    out["recovery_rate_20"] = recovery_rate_20
    out["state_var_20"] = state_var_20
    out["state_skew_20"] = state_skew_20
    out["var_lift_10_20"] = var_lift_10_20
    out["breakout_10"] = breakout_10
    out["volume_impulse_5_20"] = volume_impulse_5_20
    out["flow_impulse_5_20"] = flow_impulse_5_20
    out["entropy_floor_flag"] = entropy_floor_flag
    out["entropy_expand_flag"] = entropy_expand_flag
    return out
