import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from ..entropy_hold_judgement.entropy_hold_feature_engine import build_entropy_hold_feature_frame
else:
    _PROJECT_ROOT = Path(__file__).resolve().parents[4]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from src.strategy.uptrend_hold_state_flow.entropy_hold_judgement.entropy_hold_feature_engine import build_entropy_hold_feature_frame


def _scaled_range(values: pd.Series | np.ndarray, low: float, high: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if high <= low:
        return np.zeros(len(arr), dtype=np.float64)
    return np.clip((arr - float(low)) / float(high - low), 0.0, 1.0)


def _numeric_series(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(default, index=frame.index, dtype="float64")


def _coalesce_series(primary: pd.Series, fallback: pd.Series | np.ndarray) -> pd.Series:
    primary_series = pd.Series(primary, dtype="float64")
    fallback_series = pd.Series(fallback, index=primary_series.index, dtype="float64")
    return primary_series.combine_first(fallback_series)


def build_rapid_expansion_feature_frame(df_daily: pd.DataFrame) -> pd.DataFrame:
    base = build_entropy_hold_feature_frame(df_daily)
    if base.empty:
        return pd.DataFrame()

    out = base.copy().sort_values("bar_end_dt").reset_index(drop=True)
    log_ret_1 = _numeric_series(out, "log_ret_1").fillna(0.0)
    close = _numeric_series(out, "close")
    high = _numeric_series(out, "high").fillna(close)
    low = _numeric_series(out, "low").fillna(close)
    ma_20 = _numeric_series(out, "ma_20")
    ma_60 = _numeric_series(out, "ma_60")
    breakout_10 = _numeric_series(out, "breakout_10")
    breakout_20 = _numeric_series(out, "breakout_20")
    volume_impulse_5_20 = _numeric_series(out, "volume_impulse_5_20")
    flow_impulse_5_20 = _numeric_series(out, "flow_impulse_5_20")
    energy_impulse = _numeric_series(out, "energy_impulse")
    order_alignment = _numeric_series(out, "order_alignment").fillna(0.0)
    state_skew_20 = _numeric_series(out, "state_skew_20")
    phase_stability = _numeric_series(out, "phase_stability")
    disorder_pressure = _numeric_series(out, "disorder_pressure")
    entropy_gap_mean_5 = _numeric_series(out, "entropy_gap_mean_5")

    rolling_high_20 = high.rolling(window=20, min_periods=10).max()
    rolling_low_20 = low.rolling(window=20, min_periods=10).min()
    range_width_20 = (rolling_high_20 - rolling_low_20).replace(0.0, np.nan)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_20 = true_range.rolling(window=20, min_periods=10).mean()

    ret_5 = _coalesce_series(
        _numeric_series(out, "ret_5"),
        np.expm1(log_ret_1.rolling(window=5, min_periods=3).sum()),
    )
    range_state_20 = _coalesce_series(
        _numeric_series(out, "range_state_20"),
        ((close - rolling_low_20) / range_width_20).clip(0.0, 1.0),
    )
    trend_strength = _coalesce_series(
        _numeric_series(out, "trend_strength"),
        np.clip(
            0.40 * np.nan_to_num(_scaled_range(close.div(ma_20.replace(0.0, np.nan)) - 1.0, 0.0, 0.12), nan=0.0)
            + 0.30 * np.nan_to_num(_scaled_range(ma_20.div(ma_60.replace(0.0, np.nan)) - 1.0, 0.0, 0.18), nan=0.0)
            + 0.20 * np.nan_to_num(_scaled_range(breakout_20, 0.0, 0.15), nan=0.0)
            + 0.10 * np.nan_to_num(_scaled_range(order_alignment, 0.0, 1.0), nan=0.0),
            0.0,
            1.0,
        ),
    )
    atr_ratio_20 = _coalesce_series(_numeric_series(out, "atr_ratio_20"), atr_20 / range_width_20)

    net_ret_5 = log_ret_1.rolling(window=5, min_periods=3).sum()
    abs_path_5 = log_ret_1.abs().rolling(window=5, min_periods=3).sum()
    directional_efficiency_5 = np.clip(
        np.divide(
            net_ret_5.to_numpy(dtype=np.float64),
            abs_path_5.replace(0.0, np.nan).to_numpy(dtype=np.float64),
        ),
        0.0,
        1.0,
    )
    up_day_ratio_5 = log_ret_1.gt(0.0).rolling(window=5, min_periods=3).mean().fillna(0.0)

    expansion_thrust = np.clip(
        0.35 * _scaled_range(ret_5, 0.03, 0.22)
        + 0.25 * _scaled_range(breakout_10, 0.00, 0.15)
        + 0.20 * _scaled_range(volume_impulse_5_20, -0.10, 1.00)
        + 0.20 * _scaled_range(flow_impulse_5_20, -0.20, 1.00),
        0.0,
        1.0,
    )

    directional_persistence = np.clip(
        0.55 * _scaled_range(directional_efficiency_5, 0.25, 0.95)
        + 0.25 * up_day_ratio_5.to_numpy(dtype=np.float64)
        + 0.20 * _scaled_range(range_state_20, 0.55, 1.00),
        0.0,
        1.0,
    )

    acceptance_score = np.clip(
        0.35 * _scaled_range(energy_impulse, -0.05, 0.90)
        + 0.35 * _scaled_range(trend_strength, 0.05, 0.90)
        + 0.15 * _scaled_range(state_skew_20, 0.00, 1.50)
        + 0.15 * _scaled_range(range_state_20, 0.60, 1.00),
        0.0,
        1.0,
    )

    expansion_thrust_series = pd.Series(expansion_thrust, index=out.index, dtype="float64")
    thrust_drift_3 = expansion_thrust_series - expansion_thrust_series.shift(3)

    instability_risk = np.clip(
        0.35 * _scaled_range(disorder_pressure, 0.60, 1.00)
        + 0.20 * _scaled_range(atr_ratio_20, 0.10, 1.20)
        + 0.20 * _scaled_range(-entropy_gap_mean_5, 0.00, 0.05)
        + 0.15 * _scaled_range(-thrust_drift_3, 0.00, 0.35)
        + 0.10 * _scaled_range(1.0 - directional_efficiency_5, 0.00, 0.80),
        0.0,
        1.0,
    )

    expansion_hold_score = np.clip(
        0.35 * expansion_thrust
        + 0.25 * directional_persistence
        + 0.20 * acceptance_score
        + 0.10 * phase_stability.fillna(0.0).to_numpy(dtype=np.float64)
        + 0.10 * (1.0 - instability_risk),
        0.0,
        1.0,
    )

    constructive_expansion = (
        (expansion_thrust_series >= 0.55)
        & (pd.Series(directional_persistence, index=out.index, dtype="float64") >= 0.50)
        & (pd.Series(acceptance_score, index=out.index, dtype="float64") >= 0.45)
    )
    expansion_exhaustion = (expansion_thrust_series <= 0.38) | (thrust_drift_3 <= -0.18)
    support_loss = (
        (pd.Series(directional_persistence, index=out.index, dtype="float64") <= 0.45)
        & (pd.Series(acceptance_score, index=out.index, dtype="float64") <= 0.40)
    )
    phase_break_fast = pd.Series(out["phase_break"], dtype="boolean").fillna(False).astype(bool) | (phase_stability <= 0.25)
    exit_seed = phase_break_fast | (
        (pd.Series(instability_risk, index=out.index, dtype="float64") >= 0.70)
        & (pd.Series(disorder_pressure, index=out.index, dtype="float64") >= 0.80)
        & expansion_exhaustion
        & support_loss
    )

    out["directional_efficiency_5"] = directional_efficiency_5
    out["up_day_ratio_5"] = up_day_ratio_5
    out["ret_5"] = ret_5
    out["range_state_20"] = range_state_20
    out["trend_strength"] = trend_strength
    out["atr_ratio_20"] = atr_ratio_20
    out["expansion_thrust"] = expansion_thrust
    out["directional_persistence"] = directional_persistence
    out["acceptance_score"] = acceptance_score
    out["thrust_drift_3"] = thrust_drift_3
    out["instability_risk"] = instability_risk
    out["expansion_hold_score"] = expansion_hold_score
    out["constructive_expansion"] = constructive_expansion.to_numpy(dtype=bool)
    out["expansion_exhaustion"] = expansion_exhaustion.to_numpy(dtype=bool)
    out["support_loss"] = support_loss.to_numpy(dtype=bool)
    out["phase_break_fast"] = phase_break_fast.to_numpy(dtype=bool)
    out["exit_seed"] = exit_seed.to_numpy(dtype=bool)
    return out