import numpy as np
import pandas as pd


VALID_STRATEGIES = {"entropy_bifurcation_setup"}


def _scaled_pos(values: pd.Series | np.ndarray, low: float, high: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if high <= low:
        return np.zeros(len(arr), dtype=np.float64)
    return np.clip((arr - float(low)) / float(high - low), 0.0, 1.0)


def _center_pos(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return np.clip((arr + 1.0) / 2.0, 0.0, 1.0)


def _bool_series(values: pd.Series | np.ndarray) -> np.ndarray:
    return pd.Series(values, dtype="boolean").fillna(False).to_numpy(dtype=bool)


def _apply_entropy_bifurcation_setup(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    entropy_quality = np.clip(
        0.60 * (1.0 - pd.Series(out["entropy_percentile_120"], dtype="float64").fillna(1.0).to_numpy(dtype=np.float64))
        + 0.40 * _scaled_pos(out["entropy_gap"], 0.0, 0.08),
        0.0,
        1.0,
    )

    bifurcation_quality = np.clip(
        0.50 * _scaled_pos(out["ar1_20"], 0.85, 0.97)
        + 0.30 * _scaled_pos(0.15 - pd.Series(out["recovery_rate_20"], dtype="float64"), 0.0, 0.15)
        + 0.20 * _scaled_pos(out["var_lift_10_20"], -0.05, 0.20),
        0.0,
        1.0,
    )

    trigger_quality = np.clip(
        0.45 * _scaled_pos(out["breakout_10"], 0.0, 0.04)
        + 0.25 * _scaled_pos(out["breakout_20"], 0.0, 0.06)
        + 0.15 * _scaled_pos(out["volume_impulse_5_20"], 0.0, 0.30)
        + 0.15 * _center_pos(out["energy_impulse"]),
        0.0,
        1.0,
    )

    order_quality = _center_pos(out["order_alignment"])
    score = np.clip(
        0.35 * entropy_quality
        + 0.35 * bifurcation_quality
        + 0.20 * trigger_quality
        + 0.10 * order_quality,
        0.0,
        1.0,
    )

    state = (
        (pd.Series(out["entropy_percentile_120"], dtype="float64") <= 0.25)
        & (pd.Series(out["entropy_gap"], dtype="float64") >= 0.03)
        & (pd.Series(out["ar1_20"], dtype="float64") >= 0.88)
        & (pd.Series(out["breakout_10"], dtype="float64") >= 0.0)
        & (pd.Series(out["breakout_20"], dtype="float64") >= -0.01)
        & (pd.Series(out["close"], dtype="float64") > pd.Series(out["ma_20"], dtype="float64"))
        & (pd.Series(out["ma_20"], dtype="float64") >= pd.Series(out["ma_60"], dtype="float64"))
        & (pd.Series(out["mf_z_60"], dtype="float64") >= -0.20)
        & (pd.Series(out["volume_impulse_5_20"], dtype="float64") >= -0.10)
        & (score >= 0.58)
    )

    out["entropy_quality"] = entropy_quality
    out["bifurcation_quality"] = bifurcation_quality
    out["trigger_quality"] = trigger_quality
    out["strategy_component_a"] = entropy_quality
    out["strategy_component_b"] = bifurcation_quality
    out["strategy_component_c"] = trigger_quality
    out["strategy_score"] = score
    out["strategy_state"] = _bool_series(state)
    out["strategy_name"] = "entropy_bifurcation_setup"
    return out


def apply_strategy_model(df: pd.DataFrame, strategy_name: str = "entropy_bifurcation_setup") -> pd.DataFrame:
    strategy_name = str(strategy_name or "").strip()
    if strategy_name not in VALID_STRATEGIES:
        raise ValueError(f"Unsupported strategy_name={strategy_name}")
    return _apply_entropy_bifurcation_setup(df)
