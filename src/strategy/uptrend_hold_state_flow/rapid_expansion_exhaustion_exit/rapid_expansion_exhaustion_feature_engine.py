import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from ..rapid_expansion_hold.rapid_expansion_feature_engine import build_rapid_expansion_feature_frame
else:
    _PROJECT_ROOT = Path(__file__).resolve().parents[4]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from src.strategy.uptrend_hold_state_flow.rapid_expansion_hold.rapid_expansion_feature_engine import build_rapid_expansion_feature_frame


def _scaled_range(values: pd.Series | np.ndarray, low: float, high: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if high <= low:
        return np.zeros(len(arr), dtype=np.float64)
    return np.clip((arr - float(low)) / float(high - low), 0.0, 1.0)


def build_rapid_expansion_exhaustion_feature_frame(df_daily: pd.DataFrame) -> pd.DataFrame:
    base = build_rapid_expansion_feature_frame(df_daily)
    if base.empty:
        return pd.DataFrame()

    out = base.copy().sort_values("bar_end_dt").reset_index(drop=True)
    expansion_thrust = pd.Series(out["expansion_thrust"], dtype="float64")
    directional_persistence = pd.Series(out["directional_persistence"], dtype="float64")
    acceptance_score = pd.Series(out["acceptance_score"], dtype="float64")
    instability_risk = pd.Series(out["instability_risk"], dtype="float64")
    phase_stability = pd.Series(out["phase_stability"], dtype="float64")
    disorder_pressure = pd.Series(out["disorder_pressure"], dtype="float64")
    entropy_percentile_120 = pd.Series(out["entropy_percentile_120"], dtype="float64")
    entropy_gap_mean_5 = pd.Series(out["entropy_gap_mean_5"], dtype="float64")
    thrust_drift_3 = pd.Series(out["thrust_drift_3"], dtype="float64")
    atr_ratio_20 = pd.Series(out["atr_ratio_20"], dtype="float64")
    range_state_20 = pd.Series(out["range_state_20"], dtype="float64")
    ret_5 = pd.Series(out["ret_5"], dtype="float64")
    breakout_10 = pd.Series(out["breakout_10"], dtype="float64")

    rolling_peak_thrust_5 = expansion_thrust.rolling(window=5, min_periods=2).max()
    rolling_peak_persistence_5 = directional_persistence.rolling(window=5, min_periods=2).max()
    rolling_peak_acceptance_5 = acceptance_score.rolling(window=5, min_periods=2).max()
    acceptance_drift_3 = acceptance_score - acceptance_score.shift(3)
    persistence_fade = rolling_peak_persistence_5 - directional_persistence
    thrust_rollover = rolling_peak_thrust_5 - expansion_thrust
    acceptance_rollover = rolling_peak_acceptance_5 - acceptance_score

    recent_constructive_expansion = (
        pd.Series(out["constructive_expansion"], dtype="boolean").fillna(False).astype(bool)
        .rolling(window=5, min_periods=1)
        .max()
        .astype(bool)
    )

    peak_extension_score = np.clip(
        0.35 * _scaled_range(range_state_20, 0.72, 1.00)
        + 0.25 * _scaled_range(ret_5, 0.08, 0.28)
        + 0.20 * _scaled_range(entropy_percentile_120, 0.85, 1.00)
        + 0.20 * _scaled_range(atr_ratio_20, 0.20, 1.00),
        0.0,
        1.0,
    )

    deceleration_score = np.clip(
        0.35 * _scaled_range(-thrust_drift_3, 0.00, 0.30)
        + 0.25 * _scaled_range(thrust_rollover, 0.05, 0.35)
        + 0.20 * _scaled_range(persistence_fade, 0.05, 0.35)
        + 0.20 * _scaled_range(-acceptance_drift_3, 0.00, 0.30),
        0.0,
        1.0,
    )

    fragility_score = np.clip(
        0.35 * instability_risk.to_numpy(dtype=np.float64)
        + 0.25 * _scaled_range(disorder_pressure, 0.55, 1.00)
        + 0.20 * _scaled_range(1.0 - phase_stability, 0.00, 0.60)
        + 0.20 * _scaled_range(-entropy_gap_mean_5, 0.00, 0.05),
        0.0,
        1.0,
    )

    exhaustion_exit_score = np.clip(
        0.40 * peak_extension_score
        + 0.35 * deceleration_score
        + 0.25 * fragility_score,
        0.0,
        1.0,
    )

    terminal_zone = recent_constructive_expansion & (pd.Series(peak_extension_score, index=out.index, dtype="float64") >= 0.60)
    deceleration_state = (pd.Series(deceleration_score, index=out.index, dtype="float64") >= 0.55) & (thrust_drift_3 <= -0.05)
    support_fracture = (
        (directional_persistence <= 0.55)
        | (acceptance_score <= 0.48)
        | (breakout_10 <= 0.01)
    )
    fragility_state = (
        (pd.Series(fragility_score, index=out.index, dtype="float64") >= 0.48)
        | (instability_risk >= 0.55)
    )
    exit_seed = terminal_zone & deceleration_state & fragility_state & support_fracture

    out["rolling_peak_thrust_5"] = rolling_peak_thrust_5
    out["rolling_peak_persistence_5"] = rolling_peak_persistence_5
    out["rolling_peak_acceptance_5"] = rolling_peak_acceptance_5
    out["acceptance_drift_3"] = acceptance_drift_3
    out["persistence_fade"] = persistence_fade
    out["thrust_rollover"] = thrust_rollover
    out["acceptance_rollover"] = acceptance_rollover
    out["recent_constructive_expansion"] = recent_constructive_expansion.to_numpy(dtype=bool)
    out["peak_extension_score"] = peak_extension_score
    out["deceleration_score"] = deceleration_score
    out["fragility_score"] = fragility_score
    out["exhaustion_exit_score"] = exhaustion_exit_score
    out["terminal_zone"] = terminal_zone.to_numpy(dtype=bool)
    out["deceleration_state"] = deceleration_state.to_numpy(dtype=bool)
    out["support_fracture"] = support_fracture.to_numpy(dtype=bool)
    out["fragility_state"] = fragility_state.to_numpy(dtype=bool)
    out["exit_seed"] = exit_seed.to_numpy(dtype=bool)
    return out