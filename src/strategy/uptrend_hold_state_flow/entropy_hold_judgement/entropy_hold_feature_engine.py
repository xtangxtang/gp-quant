import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from ...entropy_bifurcation_setup.entropy_bifurcation_feature_engine import build_entropy_bifurcation_feature_frame
else:
    _PROJECT_ROOT = Path(__file__).resolve().parents[4]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from src.strategy.entropy_bifurcation_setup.entropy_bifurcation_feature_engine import build_entropy_bifurcation_feature_frame


def _scaled_range(values: pd.Series | np.ndarray, low: float, high: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if high <= low:
        return np.zeros(len(arr), dtype=np.float64)
    return np.clip((arr - float(low)) / float(high - low), 0.0, 1.0)


def build_entropy_hold_feature_frame(df_daily: pd.DataFrame) -> pd.DataFrame:
    base = build_entropy_bifurcation_feature_frame(df_daily)
    if base.empty:
        return pd.DataFrame()

    out = base.copy().sort_values("bar_end_dt").reset_index(drop=True)
    entropy_gap = pd.Series(out["entropy_gap"], dtype="float64")
    entropy_percentile_120 = pd.Series(out["entropy_percentile_120"], dtype="float64")
    ar1_20 = pd.Series(out["ar1_20"], dtype="float64")
    recovery_rate_20 = pd.Series(out["recovery_rate_20"], dtype="float64")
    var_lift_10_20 = pd.Series(out["var_lift_10_20"], dtype="float64")
    perm_entropy_20_norm = pd.Series(out["perm_entropy_20_norm"], dtype="float64")

    entropy_gap_mean_5 = entropy_gap.rolling(window=5, min_periods=3).mean()
    entropy_drift_10 = perm_entropy_20_norm - perm_entropy_20_norm.shift(10)

    entropy_reserve = np.clip(
        0.55 * (1.0 - entropy_percentile_120.fillna(1.0).to_numpy(dtype=np.float64))
        + 0.45 * _scaled_range(entropy_gap_mean_5, -0.02, 0.08),
        0.0,
        1.0,
    )

    memory_reserve = np.clip(
        0.60 * _scaled_range(ar1_20, 0.65, 0.95)
        + 0.40 * _scaled_range(0.35 - recovery_rate_20, 0.0, 0.35),
        0.0,
        1.0,
    )

    phase_stability = np.clip(1.0 - np.abs(var_lift_10_20.to_numpy(dtype=np.float64)) / 1.0, 0.0, 1.0)

    positive_entropy_drift = np.clip(entropy_drift_10.fillna(0.0).to_numpy(dtype=np.float64), 0.0, None)
    disorder_pressure = np.clip(
        0.50 * _scaled_range(entropy_percentile_120, 0.70, 1.00)
        + 0.25 * _scaled_range(-entropy_gap_mean_5, 0.00, 0.08)
        + 0.15 * _scaled_range(recovery_rate_20, 0.20, 0.50)
        + 0.10 * _scaled_range(positive_entropy_drift, 0.00, 0.10),
        0.0,
        1.0,
    )

    hold_score = np.clip(
        0.45 * entropy_reserve + 0.35 * memory_reserve + 0.20 * phase_stability,
        0.0,
        1.0,
    )

    disorder_state = (entropy_percentile_120 >= 0.85) & (entropy_gap_mean_5 <= -0.01)
    memory_collapse = (ar1_20 <= 0.68) | (recovery_rate_20 >= 0.32)
    phase_break = np.abs(var_lift_10_20) >= 0.90
    exit_seed = disorder_state & (memory_collapse | phase_break)

    out["entropy_gap_mean_5"] = entropy_gap_mean_5
    out["entropy_drift_10"] = entropy_drift_10
    out["entropy_reserve"] = entropy_reserve
    out["memory_reserve"] = memory_reserve
    out["phase_stability"] = phase_stability
    out["disorder_pressure"] = disorder_pressure
    out["hold_score"] = hold_score
    out["disorder_state"] = disorder_state.to_numpy(dtype=bool)
    out["memory_collapse"] = memory_collapse.to_numpy(dtype=bool)
    out["phase_break"] = phase_break.to_numpy(dtype=bool)
    out["exit_seed"] = exit_seed.to_numpy(dtype=bool)
    return out