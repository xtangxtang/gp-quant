import numpy as np
import pandas as pd


VALID_STRATEGIES = {
    "compression_breakout",
    "self_organized_trend",
    "fractal_pullback",
    "market_energy_flow",
}


def _pos(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return np.clip((arr + 1.0) / 2.0, 0.0, 1.0)


def _scaled_pos(values: pd.Series | np.ndarray, scale: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if scale <= 0:
        return np.zeros(len(arr), dtype=np.float64)
    return np.clip(arr / float(scale), 0.0, 1.0)


def _bool_series(values: pd.Series | np.ndarray) -> np.ndarray:
    return pd.Series(values, dtype="boolean").fillna(False).to_numpy(dtype=bool)


def _apply_compression_breakout(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    recent_compression = (
        pd.Series(out["compression_z_120"], dtype="float64").shift(1).rolling(window=5, min_periods=1).min() <= -0.50
    )
    breakout_strength = _scaled_pos(out["breakout_20"], 0.06)
    compression_quality = np.clip((-pd.Series(out["compression_z_120"], dtype="float64") - 0.30) / 1.70, 0.0, 1.0)
    energy_quality = _pos(out["energy_term"])
    switch_quality = _pos(out["switch_term"])
    order_quality = _pos(out["order_term"])

    score = np.clip(
        0.30 * compression_quality
        + 0.30 * breakout_strength
        + 0.20 * energy_quality
        + 0.10 * order_quality
        + 0.10 * switch_quality,
        0.0,
        1.0,
    )

    state = (
        _bool_series(recent_compression)
        & (pd.Series(out["breakout_20"], dtype="float64") >= 0.01)
        & (pd.Series(out["amount_z_60"], dtype="float64") >= 0.0)
        & (pd.Series(out["mf_z_60"], dtype="float64") >= -0.20)
        & (pd.Series(out["close"], dtype="float64") > pd.Series(out["ma_20"], dtype="float64"))
        & (pd.Series(out["ma_20"], dtype="float64") > pd.Series(out["ma_60"], dtype="float64"))
        & (score >= 0.55)
    )

    out["strategy_component_a"] = compression_quality
    out["strategy_component_b"] = breakout_strength
    out["strategy_component_c"] = energy_quality
    out["strategy_score"] = score
    out["strategy_state"] = state
    return out


def _apply_self_organized_trend(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    trend_strength = _pos(out["trend_strength"])
    order_alignment = _pos(out["order_alignment"])
    energy_quality = _pos(out["energy_term"])
    phase_quality = _pos(out["phase_term"])
    breakout_proximity = np.clip(1.0 + pd.Series(out["distance_from_20d_high"], dtype="float64") / 0.10, 0.0, 1.0)

    score = np.clip(
        0.30 * trend_strength
        + 0.25 * order_alignment
        + 0.20 * energy_quality
        + 0.15 * phase_quality
        + 0.10 * breakout_proximity,
        0.0,
        1.0,
    )

    state = (
        (pd.Series(out["close"], dtype="float64") > pd.Series(out["ma_20"], dtype="float64"))
        & (pd.Series(out["ma_20"], dtype="float64") > pd.Series(out["ma_60"], dtype="float64"))
        & (pd.Series(out["ma_60"], dtype="float64") > pd.Series(out["ma_120"], dtype="float64"))
        & (pd.Series(out["ma_20_slope_10"], dtype="float64") >= 0.02)
        & (pd.Series(out["ret_20"], dtype="float64") >= 0.05)
        & (pd.Series(out["ret_60"], dtype="float64") >= 0.10)
        & (pd.Series(out["energy_term"], dtype="float64") >= -0.10)
        & (pd.Series(out["phase_term"], dtype="float64") >= -0.10)
        & (score >= 0.58)
    )

    out["strategy_component_a"] = trend_strength
    out["strategy_component_b"] = order_alignment
    out["strategy_component_c"] = breakout_proximity
    out["strategy_score"] = score
    out["strategy_state"] = state
    return out


def _apply_fractal_pullback(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parent_trend = (
        (pd.Series(out["ma_20"], dtype="float64") > pd.Series(out["ma_60"], dtype="float64"))
        & (pd.Series(out["ma_60"], dtype="float64") > pd.Series(out["ma_120"], dtype="float64"))
        & (pd.Series(out["ret_60"], dtype="float64") >= 0.10)
    )
    pullback_depth = pd.Series(out["pullback_depth_8"], dtype="float64")
    pullback_ok = (pullback_depth <= -0.03) & (pullback_depth >= -0.12)
    contraction_ok = (pd.Series(out["amount_contraction"], dtype="float64") <= -0.10) & (
        pd.Series(out["atr_ratio_20"], dtype="float64") <= 0.10
    )
    restart_quality = _scaled_pos(out["restart_breakout_3"], 0.04)
    energy_quality = _pos(out["energy_term"])
    trend_quality = _pos(out["trend_strength"])
    pullback_quality = pd.Series(out["pullback_quality"], dtype="float64").fillna(0.0).clip(0.0, 1.0)

    score = np.clip(
        0.30 * trend_quality
        + 0.25 * pullback_quality
        + 0.20 * restart_quality
        + 0.15 * np.clip(-(pd.Series(out["amount_contraction"], dtype="float64")) / 0.40, 0.0, 1.0)
        + 0.10 * energy_quality,
        0.0,
        1.0,
    )

    state = (
        _bool_series(parent_trend)
        & _bool_series(pullback_ok)
        & _bool_series(contraction_ok)
        & (pd.Series(out["restart_breakout_3"], dtype="float64") >= 0.0)
        & (pd.Series(out["close"], dtype="float64") > pd.Series(out["ma_20"], dtype="float64"))
        & (score >= 0.55)
    )

    out["strategy_component_a"] = trend_quality
    out["strategy_component_b"] = pullback_quality
    out["strategy_component_c"] = restart_quality
    out["strategy_score"] = score
    out["strategy_state"] = state
    return out


def _apply_market_energy_flow(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    energy_impulse = _pos(out["energy_impulse"])
    leadership = np.clip(
        0.45 * _scaled_pos(out["ret_20"], 0.20)
        + 0.30 * _scaled_pos(out["breakout_20"], 0.08)
        + 0.25 * _pos(out["order_term"]),
        0.0,
        1.0,
    )
    turnover_quality = _scaled_pos(out["turnover_z_60"], 2.0)
    switch_quality = _pos(out["switch_term"])

    score = np.clip(
        0.45 * energy_impulse
        + 0.30 * leadership
        + 0.15 * turnover_quality
        + 0.10 * switch_quality,
        0.0,
        1.0,
    )

    state = (
        (pd.Series(out["energy_impulse"], dtype="float64") >= 0.15)
        & (pd.Series(out["mf_z_60"], dtype="float64") >= 0.0)
        & (pd.Series(out["amount_z_60"], dtype="float64") >= 0.0)
        & (pd.Series(out["ret_20"], dtype="float64") >= 0.03)
        & (pd.Series(out["close"], dtype="float64") > pd.Series(out["ma_20"], dtype="float64"))
        & (score >= 0.58)
    )

    out["strategy_component_a"] = energy_impulse
    out["strategy_component_b"] = leadership
    out["strategy_component_c"] = turnover_quality
    out["strategy_score"] = score
    out["strategy_state"] = state
    return out


def apply_strategy_model(df: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
    strategy_name = str(strategy_name or "").strip()
    if strategy_name not in VALID_STRATEGIES:
        raise ValueError(f"Unsupported strategy_name={strategy_name}")

    if strategy_name == "compression_breakout":
        out = _apply_compression_breakout(df)
    elif strategy_name == "self_organized_trend":
        out = _apply_self_organized_trend(df)
    elif strategy_name == "fractal_pullback":
        out = _apply_fractal_pullback(df)
    else:
        out = _apply_market_energy_flow(df)

    out["strategy_name"] = strategy_name
    return out