import numpy as np
import pandas as pd


VALID_STRATEGIES = {"entropy_bifurcation_setup"}

STOCK_LAYER_WEIGHTS = {
    "entropy_quality": 0.32,
    "bifurcation_quality": 0.48,
    "trigger_quality": 0.20,
}

ENTROPY_QUALITY_WEIGHTS = {
    "entropy_percentile": 0.50,
    "entropy_gap": 0.30,
    "short_entropy_compression": 0.20,
}

BIFURCATION_QUALITY_WEIGHTS = {
    "ar1_20": 0.06,
    "phase_adjusted_ar1_20": 0.18,
    "dominant_eig_abs_20": 0.30,
    "path_irreversibility_20": 0.24,
    "recovery_rate_20": 0.12,
    "var_lift_10_20": 0.10,
}

TRIGGER_QUALITY_WEIGHTS = {
    "breakout_10": 0.30,
    "breakout_20": 0.20,
    "volume_impulse_5_20": 0.18,
    "flow_impulse_5_20": 0.14,
    "energy_impulse": 0.10,
    "entropy_accel_5": 0.08,
}

LATENT_STATE_NAMES = np.asarray(["compression", "instability", "launch", "diffusion"], dtype=object)


def _scaled_pos(values: pd.Series | np.ndarray, low: float, high: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if high <= low:
        return np.zeros(len(array), dtype=np.float64)
    return np.clip((array - float(low)) / float(high - low), 0.0, 1.0)


def _center_pos(values: pd.Series | np.ndarray, scale: float = 1.0) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if scale <= 0.0:
        return np.zeros(len(array), dtype=np.float64)
    return np.clip(0.5 + 0.5 * array / float(scale), 0.0, 1.0)


def _latent_state_model(out: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    compression_score = np.clip(
        0.45 * (1.0 - pd.Series(out["entropy_percentile_120"], dtype="float64").fillna(1.0).to_numpy(dtype=np.float64))
        + 0.30 * _scaled_pos(out["entropy_gap"], 0.0, 0.10)
        + 0.15 * _scaled_pos(-pd.Series(out["breakout_10"], dtype="float64"), -0.03, 0.03)
        + 0.10 * _scaled_pos(0.20 - pd.Series(out["phase_distortion_20"], dtype="float64"), 0.0, 0.20),
        0.0,
        1.0,
    )
    instability_score = np.clip(
        0.32 * _scaled_pos(out["dominant_eig_abs_20"], 0.68, 0.99)
        + 0.28 * _scaled_pos(out["path_irreversibility_20"], 0.01, 0.25)
        + 0.20 * _scaled_pos(out["phase_adjusted_ar1_20"], 0.80, 0.97)
        + 0.12 * _scaled_pos(out["phase_distortion_20"], 0.02, 0.15)
        + 0.08 * _scaled_pos(out["var_lift_10_20"], -0.05, 0.20),
        0.0,
        1.0,
    )
    launch_score = np.clip(
        0.30 * _scaled_pos(out["breakout_10"], 0.0, 0.04)
        + 0.20 * _scaled_pos(out["volume_impulse_5_20"], 0.0, 0.30)
        + 0.18 * _scaled_pos(out["flow_impulse_5_20"], -0.10, 0.80)
        + 0.17 * _center_pos(out["order_alignment"])
        + 0.15 * _scaled_pos(out["entropy_accel_5"], -0.01, 0.02),
        0.0,
        1.0,
    )
    diffusion_score = np.clip(
        0.28 * _scaled_pos(out["breakout_20"], 0.0, 0.07)
        + 0.22 * _center_pos(out["energy_impulse"])
        + 0.22 * _scaled_pos(out["mf_z_60"], -0.20, 1.50)
        + 0.18 * _scaled_pos(pd.Series(out["close"], dtype="float64") / pd.Series(out["ma_20"], dtype="float64") - 1.0, -0.02, 0.08)
        + 0.10 * _scaled_pos(0.25 - pd.Series(out["execution_cost_proxy_20"], dtype="float64"), 0.0, 0.25),
        0.0,
        1.0,
    )

    latent_matrix = np.column_stack([compression_score, instability_score, launch_score, diffusion_score])
    raw_idx = np.argmax(latent_matrix, axis=1)
    raw_score = np.max(latent_matrix, axis=1)

    stabilized_idx = raw_idx.copy()
    for idx in range(1, len(stabilized_idx)):
        prev_idx = stabilized_idx[idx - 1]
        current_idx = raw_idx[idx]
        if current_idx == prev_idx:
            continue
        if latent_matrix[idx, current_idx] - latent_matrix[idx, prev_idx] < 0.05:
            stabilized_idx[idx] = prev_idx

    switch_flag = np.zeros(len(stabilized_idx), dtype=np.float64)
    if len(stabilized_idx) > 1:
        switch_flag[1:] = (stabilized_idx[1:] != stabilized_idx[:-1]).astype(float)
    switch_rate_5 = pd.Series(switch_flag, dtype="float64").rolling(window=5, min_periods=1).mean().to_numpy(dtype=np.float64)
    structure_score = np.clip(
        raw_score - 0.25 * switch_rate_5,
        0.0,
        1.0,
    )
    latent_label = LATENT_STATE_NAMES[stabilized_idx]
    return compression_score, instability_score, launch_score, diffusion_score, latent_label, structure_score


def _apply_entropy_bifurcation_setup(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("bar_end_dt").reset_index(drop=True)

    entropy_quality = np.clip(
        ENTROPY_QUALITY_WEIGHTS["entropy_percentile"]
        * (1.0 - pd.Series(out["entropy_percentile_120"], dtype="float64").fillna(1.0).to_numpy(dtype=np.float64))
        + ENTROPY_QUALITY_WEIGHTS["entropy_gap"] * _scaled_pos(out["entropy_gap"], 0.0, 0.10)
        + ENTROPY_QUALITY_WEIGHTS["short_entropy_compression"]
        * _scaled_pos(0.95 - pd.Series(out["perm_entropy_20_norm"], dtype="float64"), 0.0, 0.40),
        0.0,
        1.0,
    )

    bifurcation_quality = np.clip(
        BIFURCATION_QUALITY_WEIGHTS["ar1_20"] * _scaled_pos(out["ar1_20"], 0.82, 0.97)
        + BIFURCATION_QUALITY_WEIGHTS["phase_adjusted_ar1_20"] * _scaled_pos(out["phase_adjusted_ar1_20"], 0.80, 0.97)
        + BIFURCATION_QUALITY_WEIGHTS["dominant_eig_abs_20"] * _scaled_pos(out["dominant_eig_abs_20"], 0.68, 0.99)
        + BIFURCATION_QUALITY_WEIGHTS["path_irreversibility_20"] * _scaled_pos(out["path_irreversibility_20"], 0.01, 0.25)
        + BIFURCATION_QUALITY_WEIGHTS["recovery_rate_20"]
        * _scaled_pos(0.22 - pd.Series(out["recovery_rate_20"], dtype="float64"), 0.0, 0.22)
        + BIFURCATION_QUALITY_WEIGHTS["var_lift_10_20"] * _scaled_pos(out["var_lift_10_20"], -0.05, 0.20),
        0.0,
        1.0,
    )

    trigger_quality = np.clip(
        TRIGGER_QUALITY_WEIGHTS["breakout_10"] * _scaled_pos(out["breakout_10"], 0.0, 0.04)
        + TRIGGER_QUALITY_WEIGHTS["breakout_20"] * _scaled_pos(out["breakout_20"], 0.0, 0.06)
        + TRIGGER_QUALITY_WEIGHTS["volume_impulse_5_20"] * _scaled_pos(out["volume_impulse_5_20"], 0.0, 0.30)
        + TRIGGER_QUALITY_WEIGHTS["flow_impulse_5_20"] * _scaled_pos(out["flow_impulse_5_20"], -0.10, 0.80)
        + TRIGGER_QUALITY_WEIGHTS["energy_impulse"] * _center_pos(out["energy_impulse"])
        + TRIGGER_QUALITY_WEIGHTS["entropy_accel_5"] * _scaled_pos(out["entropy_accel_5"], -0.01, 0.02),
        0.0,
        1.0,
    )

    execution_quality = np.clip(1.0 - pd.Series(out["execution_cost_proxy_20"], dtype="float64").fillna(1.0), 0.0, 1.0)
    latent_compression_score, latent_instability_score, latent_launch_score, latent_diffusion_score, latent_state_label, experimental_structure_latent_score = _latent_state_model(out)
    stock_state_score = np.clip(
        STOCK_LAYER_WEIGHTS["entropy_quality"] * entropy_quality
        + STOCK_LAYER_WEIGHTS["bifurcation_quality"] * bifurcation_quality
        + STOCK_LAYER_WEIGHTS["trigger_quality"] * trigger_quality,
        0.0,
        1.0,
    )

    strategic_abandonment_seed = (
        (pd.Series(out["execution_cost_proxy_20"], dtype="float64") >= 0.82)
        & (pd.Series(stock_state_score, dtype="float64") < 0.65)
    )

    entropy_percentile = pd.Series(out["entropy_percentile_120"], dtype="float64")
    entropy_gap = pd.Series(out["entropy_gap"], dtype="float64")
    phase_adjusted_ar1 = pd.Series(out["phase_adjusted_ar1_20"], dtype="float64")
    phase_distortion = pd.Series(out.get("phase_distortion_20"), dtype="float64")
    dominant_eig_abs = pd.Series(out["dominant_eig_abs_20"], dtype="float64")
    path_irreversibility = pd.Series(out["path_irreversibility_20"], dtype="float64")
    breakout_10 = pd.Series(out["breakout_10"], dtype="float64")
    close = pd.Series(out["close"], dtype="float64")
    ma_20 = pd.Series(out["ma_20"], dtype="float64")
    ma_60 = pd.Series(out["ma_60"], dtype="float64")
    mf_z_60 = pd.Series(out["mf_z_60"], dtype="float64")
    stock_state_series = pd.Series(stock_state_score, dtype="float64")

    compression_state = (entropy_percentile <= 0.40) & (entropy_gap >= 0.015)
    instability_state = ((dominant_eig_abs >= 0.66) & (path_irreversibility >= 0.008)) | (bifurcation_quality >= 0.45)
    phase_state = (phase_adjusted_ar1 >= 0.60) | ((phase_distortion <= 0.12) & (pd.Series(out["ar1_20"], dtype="float64") >= 0.55))
    trigger_state = (
        (breakout_10 >= -0.02)
        & (close >= ma_20 * 0.985)
        & (ma_20 >= ma_60 * 0.98)
        & (mf_z_60 >= -0.80)
    )
    quality_state = (
        (stock_state_series >= 0.44)
        & (entropy_quality >= 0.62)
        & (bifurcation_quality >= 0.20)
        & (execution_quality >= 0.35)
    )
    state = compression_state & instability_state & phase_state & trigger_state & quality_state

    out["entropy_quality"] = entropy_quality
    out["bifurcation_quality"] = bifurcation_quality
    out["trigger_quality"] = trigger_quality
    out["execution_quality"] = execution_quality
    out["stock_state_score"] = stock_state_score
    out["latent_compression_score"] = latent_compression_score
    out["latent_instability_score"] = latent_instability_score
    out["latent_launch_score"] = latent_launch_score
    out["latent_diffusion_score"] = latent_diffusion_score
    out["latent_state_label"] = latent_state_label
    out["experimental_structure_latent_score"] = experimental_structure_latent_score
    out["strategy_component_a"] = entropy_quality
    out["strategy_component_b"] = bifurcation_quality
    out["strategy_component_c"] = trigger_quality
    out["strategic_abandonment_seed"] = strategic_abandonment_seed.to_numpy(dtype=bool)
    out["strategy_score"] = stock_state_score
    out["strategy_state"] = state.to_numpy(dtype=bool)
    out["strategy_name"] = "entropy_bifurcation_setup"
    return out


def apply_strategy_model(df: pd.DataFrame, strategy_name: str = "entropy_bifurcation_setup") -> pd.DataFrame:
    if str(strategy_name) not in VALID_STRATEGIES:
        supported = ", ".join(sorted(VALID_STRATEGIES))
        raise ValueError(f"Unsupported strategy_name={strategy_name}. Supported: {supported}")
    return _apply_entropy_bifurcation_setup(df)