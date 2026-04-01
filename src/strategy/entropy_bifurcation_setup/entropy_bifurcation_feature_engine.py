import math

import numpy as np
import pandas as pd


_RESERVOIR_DIM = 12
_RESERVOIR_SEED = 7
_RESERVOIR_SPECTRAL_RADIUS = 0.85
_reservoir_rng = np.random.default_rng(_RESERVOIR_SEED)
_reservoir_weights = _reservoir_rng.normal(0.0, 0.35, size=(_RESERVOIR_DIM, _RESERVOIR_DIM))
_reservoir_eig_scale = float(np.max(np.abs(np.linalg.eigvals(_reservoir_weights))))
if _reservoir_eig_scale > 1e-12:
    _reservoir_weights *= _RESERVOIR_SPECTRAL_RADIUS / _reservoir_eig_scale
_reservoir_input = _reservoir_rng.normal(0.0, 0.50, size=_RESERVOIR_DIM)


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


def _rolling_apply_1d(values: np.ndarray, window: int, func) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    out = np.full(len(array), np.nan, dtype=np.float64)
    if window <= 0:
        return out
    for idx in range(window - 1, len(array)):
        out[idx] = float(func(array[idx - window + 1 : idx + 1]))
    return out


def _rolling_percentile_of_last(values: np.ndarray, window: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    out = np.full(len(array), np.nan, dtype=np.float64)
    for idx in range(window - 1, len(array)):
        current = array[idx - window + 1 : idx + 1]
        current = current[np.isfinite(current)]
        if len(current) < max(8, window // 3):
            continue
        last = current[-1]
        out[idx] = float(np.mean(current <= last))
    return out


def _rolling_ar1(window_values: np.ndarray) -> float:
    values = np.asarray(window_values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 8:
        return np.nan
    left = values[:-1]
    right = values[1:]
    if len(left) < 5:
        return np.nan
    left_centered = left - float(np.mean(left))
    right_centered = right - float(np.mean(right))
    denominator = float(np.dot(left_centered, left_centered))
    if denominator <= 1e-12:
        return np.nan
    return float(np.dot(left_centered, right_centered) / denominator)


def _permutation_entropy(window_values: np.ndarray, order: int = 3) -> float:
    values = np.asarray(window_values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < max(order + 2, 6):
        return np.nan

    counts: dict[tuple[int, ...], int] = {}
    for idx in range(len(values) - order + 1):
        pattern = tuple(np.argsort(values[idx : idx + order], kind="mergesort"))
        counts[pattern] = counts.get(pattern, 0) + 1

    if not counts:
        return np.nan

    freq = np.asarray(list(counts.values()), dtype=np.float64)
    prob = freq / float(freq.sum())
    entropy = float(-(prob * np.log(prob)).sum())
    normalizer = float(np.log(math.factorial(order)))
    if normalizer <= 0.0:
        return np.nan
    return entropy / normalizer


def _phase_adjust_series(values: pd.Series, phase_ids: pd.Series, lookback: int = 12) -> pd.Series:
    frame = pd.DataFrame(
        {
            "value": pd.to_numeric(values, errors="coerce"),
            "phase": pd.Series(phase_ids, index=values.index),
        }
    )
    baseline = frame.groupby("phase")["value"].transform(
        lambda series: series.shift(1).rolling(window=lookback, min_periods=max(3, lookback // 4)).mean()
    )
    return frame["value"] - baseline.fillna(0.0)


def _month_segment_phase(dt_series: pd.Series) -> pd.Series:
    days = pd.Series(dt_series).dt.day.fillna(15).astype(int)
    return pd.Series(np.where(days <= 7, 0, np.where(days >= 23, 2, 1)), index=dt_series.index)


def _quarter_phase(dt_series: pd.Series) -> pd.Series:
    months = pd.Series(dt_series).dt.month.fillna(1).astype(int)
    return pd.Series((months - 1) % 3, index=dt_series.index)


def _dominant_autocorr_eig(window_values: np.ndarray, order: int = 2, return_abs: bool = False) -> float:
    values = np.asarray(window_values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < max(12, order + 6):
        return np.nan

    centered = values - float(np.mean(values))
    if float(np.std(centered)) <= 1e-12:
        return np.nan

    acov: list[float] = []
    for lag in range(order + 1):
        left = centered[: len(centered) - lag]
        right = centered[lag:]
        if len(right) == 0:
            return np.nan
        acov.append(float(np.dot(left, right)) / float(len(right)))

    system = np.asarray([[acov[abs(i - j)] for j in range(order)] for i in range(order)], dtype=np.float64)
    rhs = np.asarray(acov[1 : order + 1], dtype=np.float64)

    try:
        phi = np.linalg.solve(system + np.eye(order, dtype=np.float64) * 1e-8, rhs)
    except np.linalg.LinAlgError:
        return np.nan

    companion = np.zeros((order, order), dtype=np.float64)
    companion[0, :] = phi
    if order > 1:
        companion[1:, :-1] = np.eye(order - 1, dtype=np.float64)

    eigvals = np.linalg.eigvals(companion)
    dominant = eigvals[np.argmax(np.abs(eigvals))]
    if return_abs:
        return float(np.abs(dominant))
    return float(np.real(dominant))


def _dominant_eig_from_autocorr(window_values: np.ndarray) -> float:
    return _dominant_autocorr_eig(window_values, order=2, return_abs=False)


def _dominant_eig_abs_from_autocorr(window_values: np.ndarray) -> float:
    return _dominant_autocorr_eig(window_values, order=2, return_abs=True)


def _path_irreversibility(window_values: np.ndarray) -> float:
    values = np.asarray(window_values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 10:
        return np.nan

    sigma = float(np.std(values))
    if sigma <= 1e-12:
        return 0.0

    threshold = 0.5 * sigma
    states = np.full(len(values), 1, dtype=np.int64)
    states[values < -threshold] = 0
    states[values > threshold] = 2

    counts = np.zeros((3, 3), dtype=np.float64)
    for left, right in zip(states[:-1], states[1:], strict=False):
        counts[left, right] += 1.0

    total = float(counts.sum())
    if total <= 0.0:
        return np.nan

    forward = counts / total
    backward = counts.T / total
    mask = (forward > 0.0) & (backward > 0.0)
    if not np.any(mask):
        return 0.0
    divergence = float(np.sum(forward[mask] * np.log(forward[mask] / backward[mask])))
    return max(0.0, divergence)


def _delay_embedding_loop_score(window_values: np.ndarray, lag: int = 2) -> float:
    values = np.asarray(window_values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < max(12, lag + 8):
        return np.nan

    left = values[:-lag]
    right = values[lag:]
    points = np.column_stack([left, right])
    points = points - points.mean(axis=0, keepdims=True)
    covariance = np.cov(points, rowvar=False)
    eigvals = np.sort(np.maximum(np.linalg.eigvalsh(covariance), 0.0))
    if eigvals[-1] <= 1e-12:
        return 0.0
    loop_roundness = float(np.sqrt(eigvals[-2] / eigvals[-1])) if len(eigvals) >= 2 else 0.0

    first_diff = np.diff(values)
    if len(first_diff) < 4:
        turning_density = 0.0
    else:
        signs = np.sign(first_diff)
        turning_density = float(np.mean(signs[1:] * signs[:-1] < 0.0))
    return float(np.clip(0.70 * loop_roundness + 0.30 * turning_density, 0.0, 1.0))


def _reservoir_tipping_score(window_values: np.ndarray) -> float:
    values = np.asarray(window_values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 20:
        return np.nan

    centered = values - float(np.mean(values))
    scale = float(np.std(centered))
    if scale <= 1e-12:
        return 0.0
    normalized = centered / scale

    state = np.zeros(_RESERVOIR_DIM, dtype=np.float64)
    norms: list[float] = []
    activation_deltas: list[float] = []
    for value in normalized:
        next_state = np.tanh(_reservoir_weights @ state + _reservoir_input * float(value))
        activation_deltas.append(float(np.linalg.norm(next_state - state)))
        state = next_state
        norms.append(float(np.linalg.norm(state)))

    tail_norm = np.asarray(norms[-6:], dtype=np.float64)
    tail_delta = np.asarray(activation_deltas[-6:], dtype=np.float64)
    if len(tail_norm) < 3:
        return np.nan

    norm_trend = float(np.clip((tail_norm[-1] - float(np.mean(tail_norm[:-1]))) / 2.0, 0.0, 1.0))
    delta_intensity = float(np.clip(float(np.mean(tail_delta)) / 1.5, 0.0, 1.0))
    tail_vol = float(np.clip(float(np.std(tail_norm)) / 0.5, 0.0, 1.0))
    return float(np.clip(0.45 * norm_trend + 0.35 * delta_intensity + 0.20 * tail_vol, 0.0, 1.0))


def _prepare_daily_frame(df_daily: pd.DataFrame) -> pd.DataFrame:
    if df_daily is None or not isinstance(df_daily, pd.DataFrame) or df_daily.empty:
        return pd.DataFrame()
    if "trade_date" not in df_daily.columns:
        return pd.DataFrame()

    out = df_daily.copy()
    out["trade_date_str"] = out["trade_date"].astype(str)
    out["bar_end_dt"] = pd.to_datetime(out["trade_date_str"], format="%Y%m%d", errors="coerce")
    out = out.dropna(subset=["bar_end_dt"]).sort_values("bar_end_dt").reset_index(drop=True)
    if out.empty:
        return pd.DataFrame()

    out["open"] = _numeric_series(out, "open")
    out["close"] = _numeric_series(out, "close")
    out["high"] = _numeric_series(out, "high")
    out["low"] = _numeric_series(out, "low")
    out["amount"] = _numeric_series(out, "amount", 0.0).fillna(0.0)
    out["turnover_rate"] = _numeric_series(out, "turnover_rate", 0.0).fillna(0.0)
    out["net_mf_amount"] = _numeric_series(out, "net_mf_amount", 0.0).fillna(0.0)

    high_fallback = pd.concat([out["open"], out["close"]], axis=1).max(axis=1)
    low_fallback = pd.concat([out["open"], out["close"]], axis=1).min(axis=1)
    out["high"] = out["high"].fillna(high_fallback)
    out["low"] = out["low"].fillna(low_fallback)
    out["bar_start"] = out["trade_date_str"]
    out["bar_end"] = out["trade_date_str"]
    return out.dropna(subset=["open", "close"]).reset_index(drop=True)


def build_entropy_bifurcation_feature_frame(df_daily: pd.DataFrame) -> pd.DataFrame:
    out = _prepare_daily_frame(df_daily)
    if out.empty:
        return pd.DataFrame()

    close = pd.to_numeric(out["close"], errors="coerce")
    open_ = pd.to_numeric(out["open"], errors="coerce")
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    amount = pd.to_numeric(out["amount"], errors="coerce").fillna(0.0)
    turnover_rate = pd.to_numeric(out["turnover_rate"], errors="coerce").fillna(0.0)
    net_mf = pd.to_numeric(out["net_mf_amount"], errors="coerce").fillna(0.0)

    safe_close = close.where(close > 0.0)
    log_close = np.log(safe_close)
    log_ret_1 = log_close.diff()

    out["ret_1"] = close.pct_change()
    out["ma_20"] = close.rolling(window=20, min_periods=10).mean()
    out["ma_60"] = close.rolling(window=60, min_periods=30).mean()
    out["vol_20"] = log_ret_1.rolling(window=20, min_periods=10).std(ddof=0)

    log_close_ma_20 = log_close.rolling(window=20, min_periods=10).mean()
    log_close_std_20 = log_close.rolling(window=20, min_periods=10).std(ddof=0)
    detrended_close_20 = (log_close - log_close_ma_20) / log_close_std_20.replace(0.0, np.nan)

    perm_entropy_20_norm = _rolling_apply_1d(detrended_close_20.to_numpy(dtype=np.float64), 20, _permutation_entropy)
    perm_entropy_60_norm = _rolling_apply_1d(detrended_close_20.to_numpy(dtype=np.float64), 60, _permutation_entropy)
    entropy_gap = perm_entropy_60_norm - perm_entropy_20_norm
    entropy_percentile_120 = _rolling_percentile_of_last(perm_entropy_20_norm, 120)

    weekday_phase = out["bar_end_dt"].dt.weekday.fillna(-1).astype(int)
    month_segment_phase = _month_segment_phase(out["bar_end_dt"])
    quarter_phase = _quarter_phase(out["bar_end_dt"])

    phase_adjusted_state_20 = _phase_adjust_series(detrended_close_20, weekday_phase, lookback=12)
    phase_adjusted_state_20 = _phase_adjust_series(phase_adjusted_state_20, month_segment_phase, lookback=10)
    phase_adjusted_state_20 = _phase_adjust_series(phase_adjusted_state_20, quarter_phase, lookback=18)

    ar1_20 = _rolling_apply_1d(detrended_close_20.to_numpy(dtype=np.float64), 20, _rolling_ar1)
    phase_adjusted_ar1_20 = _rolling_apply_1d(phase_adjusted_state_20.to_numpy(dtype=np.float64), 20, _rolling_ar1)
    dominant_eig_20 = _rolling_apply_1d(phase_adjusted_state_20.to_numpy(dtype=np.float64), 20, _dominant_eig_from_autocorr)
    dominant_eig_abs_20 = _rolling_apply_1d(
        phase_adjusted_state_20.to_numpy(dtype=np.float64),
        20,
        _dominant_eig_abs_from_autocorr,
    )
    phase_distortion_20 = np.abs(
        pd.Series(ar1_20, dtype="float64").to_numpy(dtype=np.float64)
        - pd.Series(phase_adjusted_ar1_20, dtype="float64").to_numpy(dtype=np.float64)
    )

    recovery_rate_20 = 1.0 - pd.Series(phase_adjusted_ar1_20, dtype="float64")
    state_var_10 = detrended_close_20.rolling(window=10, min_periods=5).var(ddof=0)
    state_var_20 = detrended_close_20.rolling(window=20, min_periods=10).var(ddof=0)
    state_skew_20 = detrended_close_20.rolling(window=20, min_periods=10).skew()
    var_lift_10_20 = state_var_10 / state_var_20.replace(0.0, np.nan) - 1.0

    path_irreversibility_20 = _rolling_apply_1d(log_ret_1.to_numpy(dtype=np.float64), 20, _path_irreversibility)
    coarse_entropy_lb_20 = np.asarray(path_irreversibility_20, dtype=np.float64)
    entropy_slope_5 = pd.Series(perm_entropy_20_norm, dtype="float64").diff(5) / 5.0
    entropy_accel_5 = entropy_slope_5.diff()

    prev_high_10 = high.shift(1).rolling(window=10, min_periods=5).max()
    prev_high_20 = high.shift(1).rolling(window=20, min_periods=10).max()
    breakout_10 = close / prev_high_10.replace(0.0, np.nan) - 1.0
    breakout_20 = close / prev_high_20.replace(0.0, np.nan) - 1.0

    amount_ma_5 = amount.rolling(window=5, min_periods=3).mean()
    amount_ma_20 = amount.rolling(window=20, min_periods=10).mean()
    turnover_ma_20 = turnover_rate.rolling(window=20, min_periods=10).mean()
    turnover_std_20 = turnover_rate.rolling(window=20, min_periods=10).std(ddof=0)
    volume_impulse_5_20 = amount_ma_5 / amount_ma_20.replace(0.0, np.nan) - 1.0

    flow_ma_5 = net_mf.rolling(window=5, min_periods=3).mean()
    flow_scale_20 = net_mf.abs().rolling(window=20, min_periods=10).mean()
    flow_impulse_5_20 = flow_ma_5 / flow_scale_20.replace(0.0, np.nan)

    energy_impulse = np.clip(
        0.55 * pd.Series(volume_impulse_5_20, dtype="float64").fillna(0.0).to_numpy(dtype=np.float64)
        + 0.45 * pd.Series(flow_impulse_5_20, dtype="float64").fillna(0.0).to_numpy(dtype=np.float64),
        -1.0,
        1.0,
    )
    order_alignment = np.clip(
        0.35 * np.nan_to_num(_safe_div(close - out["ma_20"], out["ma_20"]), nan=0.0)
        + 0.35 * np.nan_to_num(_safe_div(out["ma_20"] - out["ma_60"], out["ma_60"]), nan=0.0)
        + 0.30 * pd.Series(breakout_10, dtype="float64").fillna(0.0).to_numpy(dtype=np.float64),
        -1.0,
        1.0,
    )

    mf_z_60_mean = net_mf.rolling(window=60, min_periods=20).mean()
    mf_z_60_std = net_mf.rolling(window=60, min_periods=20).std(ddof=0)
    mf_z_60 = (net_mf - mf_z_60_mean) / mf_z_60_std.replace(0.0, np.nan)

    amount_support = np.log1p(amount_ma_20.clip(lower=0.0)) / 14.0
    execution_cost_proxy_20 = np.clip(
        0.45 * (out["vol_20"].fillna(0.0).to_numpy(dtype=np.float64) / 0.04)
        + 0.30 * turnover_std_20.fillna(0.0).to_numpy(dtype=np.float64)
        - 0.15 * amount_support.fillna(0.0).to_numpy(dtype=np.float64)
        - 0.10 * (turnover_ma_20.fillna(0.0).to_numpy(dtype=np.float64) / 6.0),
        0.0,
        1.0,
    )
    experimental_tda_score = _rolling_apply_1d(phase_adjusted_state_20.to_numpy(dtype=np.float64), 30, _delay_embedding_loop_score)
    experimental_reservoir_tipping_score = _rolling_apply_1d(
        phase_adjusted_state_20.to_numpy(dtype=np.float64),
        30,
        _reservoir_tipping_score,
    )

    out["log_ret_1"] = log_ret_1
    out["detrended_close_20"] = detrended_close_20
    out["perm_entropy_20_norm"] = perm_entropy_20_norm
    out["perm_entropy_60_norm"] = perm_entropy_60_norm
    out["entropy_gap"] = entropy_gap
    out["entropy_percentile_120"] = entropy_percentile_120
    out["ar1_20"] = ar1_20
    out["phase_adjusted_ar1_20"] = phase_adjusted_ar1_20
    out["phase_distortion_20"] = phase_distortion_20
    out["dominant_eig_20"] = dominant_eig_20
    out["dominant_eig_abs_20"] = dominant_eig_abs_20
    out["recovery_rate_20"] = recovery_rate_20
    out["state_skew_20"] = state_skew_20
    out["var_lift_10_20"] = var_lift_10_20
    out["path_irreversibility_20"] = path_irreversibility_20
    out["coarse_entropy_lb_20"] = coarse_entropy_lb_20
    out["entropy_accel_5"] = entropy_accel_5
    out["breakout_10"] = breakout_10
    out["breakout_20"] = breakout_20
    out["volume_impulse_5_20"] = volume_impulse_5_20
    out["flow_impulse_5_20"] = flow_impulse_5_20
    out["energy_impulse"] = energy_impulse
    out["order_alignment"] = order_alignment
    out["mf_z_60"] = mf_z_60
    out["execution_cost_proxy_20"] = execution_cost_proxy_20
    out["experimental_tda_score"] = experimental_tda_score
    out["experimental_reservoir_tipping_score"] = experimental_reservoir_tipping_score
    out["experimental_structure_latent_score"] = np.nan
    return out