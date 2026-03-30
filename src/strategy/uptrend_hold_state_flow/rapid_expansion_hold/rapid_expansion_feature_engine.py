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


def build_rapid_expansion_feature_frame(df_daily: pd.DataFrame) -> pd.DataFrame:
    base = build_entropy_hold_feature_frame(df_daily)
    if base.empty:
        return pd.DataFrame()

    out = base.copy().sort_values("bar_end_dt").reset_index(drop=True)
    log_ret_1 = pd.Series(out["log_ret_1"], dtype="float64").fillna(0.0)
    ret_5 = pd.Series(out["ret_5"], dtype="float64")
    breakout_10 = pd.Series(out["breakout_10"], dtype="float64")
    volume_impulse_5_20 = pd.Series(out["volume_impulse_5_20"], dtype="float64")
    flow_impulse_5_20 = pd.Series(out["flow_impulse_5_20"], dtype="float64")
    range_state_20 = pd.Series(out["range_state_20"], dtype="float64")
    energy_impulse = pd.Series(out["energy_impulse"], dtype="float64")
    trend_strength = pd.Series(out["trend_strength"], dtype="float64")
    state_skew_20 = pd.Series(out["state_skew_20"], dtype="float64")
    atr_ratio_20 = pd.Series(out["atr_ratio_20"], dtype="float64")
    phase_stability = pd.Series(out["phase_stability"], dtype="float64")
    disorder_pressure = pd.Series(out["disorder_pressure"], dtype="float64")
    entropy_gap_mean_5 = pd.Series(out["entropy_gap_mean_5"], dtype="float64")

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