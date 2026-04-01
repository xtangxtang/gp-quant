import glob
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .entropy_bifurcation_feature_engine import build_entropy_bifurcation_feature_frame
from .entropy_bifurcation_report_writer import write_entropy_bifurcation_outputs
from .entropy_bifurcation_signal_models import VALID_STRATEGIES, apply_strategy_model


MARKET_REGIME_WEIGHTS = {
    "low_entropy_share": 0.22,
    "bifurcation_share": 0.26,
    "energy_share": 0.10,
    "breakout_share": 0.10,
    "market_coupling_entropy_20": 0.20,
    "low_noise_support": 0.12,
}

DECISION_LAYER_WEIGHTS = {
    "stock_state": 0.70,
    "market_gate": 0.20,
    "execution_readiness": 0.10,
    "experimental_model": 0.00,
}

EXECUTION_LAYER_WEIGHTS = {
    "asset_execution_cost": 0.65,
    "market_noise_cost": 0.35,
}


@dataclass(frozen=True)
class EntropyBifurcationScanConfig:
    strategy_name: str = "entropy_bifurcation_setup"
    data_dir: str = ""
    out_dir: str = ""
    scan_date: str = ""
    top_n: int = 30
    symbols: str = ""
    basic_path: str = ""
    lookback_years: int = 5
    min_amount: float = 500000.0
    min_turnover: float = 1.0
    exclude_st: bool = True
    backtest_start_date: str = ""
    backtest_end_date: str = ""
    hold_days: int = 5
    max_positions: int = 10
    max_positions_per_industry: int = 2


@dataclass
class PreparedSymbol:
    symbol: str
    ts_code: str
    name: str
    area: str
    industry: str
    market: str
    is_st: bool
    signal_daily: pd.DataFrame
    date_to_index: dict[str, int]
    close_by_date: dict[str, float]


def _industry_bucket(row: dict[str, Any]) -> str:
    return str(row.get("industry") or "UNKNOWN")


def _normalized_shannon_entropy(weights: pd.Series) -> float:
    values = pd.to_numeric(weights, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    values = values[values > 0.0]
    if len(values) <= 1:
        return 0.0
    probabilities = values / float(values.sum())
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    return entropy / float(np.log(len(probabilities)))


def _market_phase_state(
    regime_score: float,
    low_entropy_share: float,
    breakout_share: float,
    noise_cost: float,
    coupling_entropy: float,
    phase_distortion_share: float,
) -> str:
    if noise_cost >= 0.82 or (phase_distortion_share >= 0.55 and regime_score < 0.30):
        return "abandon"
    if low_entropy_share >= 0.22 and breakout_share < 0.12 and coupling_entropy < 0.55 and phase_distortion_share < 0.45:
        return "compression"
    if regime_score >= 0.42 and breakout_share >= 0.16 and coupling_entropy >= 0.55 and phase_distortion_share < 0.40:
        return "expansion"
    if phase_distortion_share >= 0.45:
        return "distorted"
    if regime_score >= 0.26:
        return "transition"
    return "neutral"


def _load_basic_info_map(basic_path: str) -> dict[str, dict[str, str]]:
    if not basic_path or not os.path.exists(basic_path):
        return {}
    try:
        df = pd.read_csv(basic_path, usecols=["ts_code", "name", "area", "industry", "market"])
        if df.empty:
            return {}
        df = df.fillna("")
        return {
            str(row["ts_code"]): {
                "name": str(row.get("name", "") or ""),
                "area": str(row.get("area", "") or ""),
                "industry": str(row.get("industry", "") or ""),
                "market": str(row.get("market", "") or ""),
            }
            for _, row in df.iterrows()
        }
    except Exception:
        return {}


def _is_st_name(name: str) -> bool:
    normalized = str(name or "").strip().upper()
    return normalized.startswith("ST") or normalized.startswith("*ST")


def _resolve_files(data_dir: str, symbols_arg: str) -> list[str]:
    symbols = [symbol.strip() for symbol in str(symbols_arg).split(",") if symbol.strip()]
    if symbols:
        return sorted(
            [os.path.join(data_dir, f"{symbol}.csv") for symbol in symbols if os.path.exists(os.path.join(data_dir, f"{symbol}.csv"))]
        )
    return sorted(glob.glob(os.path.join(data_dir, "*.csv")))


def _read_last_non_empty_line(file_path: str) -> str:
    with open(file_path, "rb") as file_obj:
        file_obj.seek(0, os.SEEK_END)
        position = file_obj.tell()
        buffer = bytearray()
        while position > 0:
            position -= 1
            file_obj.seek(position)
            char = file_obj.read(1)
            if char == b"\n" and buffer:
                break
            if char != b"\n":
                buffer.extend(char)
        return bytes(reversed(buffer)).decode("utf-8", errors="ignore").strip()


def _latest_trade_date_for_file(file_path: str) -> str | None:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file_obj:
            header_line = file_obj.readline().strip()
        last_line = _read_last_non_empty_line(file_path)
    except OSError:
        return None
    if not header_line or not last_line:
        return None
    header_parts = [part.strip() for part in header_line.split(",")]
    if "trade_date" not in header_parts:
        return None
    trade_date_idx = header_parts.index("trade_date")
    parts = last_line.split(",")
    if trade_date_idx >= len(parts):
        return None
    trade_date = str(parts[trade_date_idx]).strip()
    return trade_date if trade_date.isdigit() else None


def _infer_scan_date(files: list[str], requested_scan_date: str) -> str:
    if requested_scan_date:
        return str(requested_scan_date)
    latest_dates = [_latest_trade_date_for_file(file_path) for file_path in files]
    latest_dates = [trade_date for trade_date in latest_dates if trade_date]
    if not latest_dates:
        raise SystemExit("Unable to infer scan_date from CSV files.")
    return max(latest_dates)


def _build_symbol_from_ts_code(ts_code: str) -> str:
    if not ts_code or "." not in ts_code:
        return str(ts_code).lower()
    code, exch = ts_code.split(".", 1)
    return f"{exch.lower()}{code}"


def _prepare_symbol_state(file_path: str, config: EntropyBifurcationScanConfig, basic_info_map: dict[str, dict[str, str]]) -> PreparedSymbol | None:
    try:
        df = pd.read_csv(file_path)
    except Exception:
        return None
    if df.empty or "trade_date" not in df.columns:
        return None

    max_needed_date = str(config.backtest_end_date or config.scan_date)
    if len(max_needed_date) == 8:
        start_keep = f"{int(max_needed_date[:4]) - int(config.lookback_years)}0101"
        df = df[(df["trade_date"].astype(str) >= start_keep) & (df["trade_date"].astype(str) <= max_needed_date)].copy()
    if df.empty:
        return None

    ts_code = str(df["ts_code"].iloc[0]) if "ts_code" in df.columns else ""
    symbol = _build_symbol_from_ts_code(ts_code or os.path.splitext(os.path.basename(file_path))[0])
    basic_info = basic_info_map.get(ts_code, {})
    name = str(basic_info.get("name", "") or "")
    area = str(basic_info.get("area", "") or "")
    industry = str(basic_info.get("industry", "") or "")
    market = str(basic_info.get("market", "") or "")
    is_st = _is_st_name(name)

    features = build_entropy_bifurcation_feature_frame(df)
    if features.empty:
        return None
    signal_daily = apply_strategy_model(features, config.strategy_name)
    if signal_daily.empty:
        return None

    signal_daily = signal_daily.reset_index(drop=True)
    signal_daily["bar_end"] = signal_daily["bar_end"].astype(str)
    date_to_index = {trade_date: idx for idx, trade_date in enumerate(signal_daily["bar_end"].tolist())}
    close_by_date = {
        str(trade_date): float(close)
        for trade_date, close in zip(signal_daily["bar_end"].tolist(), signal_daily["close"].tolist(), strict=False)
        if pd.notna(close)
    }
    return PreparedSymbol(
        symbol=symbol,
        ts_code=ts_code,
        name=name,
        area=area,
        industry=industry,
        market=market,
        is_st=is_st,
        signal_daily=signal_daily,
        date_to_index=date_to_index,
        close_by_date=close_by_date,
    )


def _prepare_all_symbols(files: list[str], config: EntropyBifurcationScanConfig, basic_info_map: dict[str, dict[str, str]]) -> list[PreparedSymbol]:
    prepared: list[PreparedSymbol] = []
    for file_path in files:
        item = _prepare_symbol_state(file_path, config, basic_info_map)
        if item is not None:
            prepared.append(item)
    return prepared


def _build_snapshot_row(prepared: PreparedSymbol, scan_date: str) -> dict[str, Any] | None:
    idx = prepared.date_to_index.get(str(scan_date))
    if idx is None:
        return None
    row = prepared.signal_daily.iloc[idx]
    return {
        "symbol": prepared.symbol,
        "ts_code": prepared.ts_code,
        "name": prepared.name,
        "area": prepared.area,
        "industry": prepared.industry,
        "market": prepared.market,
        "is_st": prepared.is_st,
        "scan_date": str(scan_date),
        "signal_date": str(row["bar_end"]),
        "close": float(row["close"]) if pd.notna(row.get("close")) else None,
        "amount": float(row["amount"]) if pd.notna(row.get("amount")) else None,
        "turnover_rate": float(row["turnover_rate"]) if pd.notna(row.get("turnover_rate")) else None,
        "strategy_name": str(row.get("strategy_name") or ""),
        "strategy_state": bool(row["strategy_state"]) if pd.notna(row.get("strategy_state")) else False,
        "strategy_score": float(row["strategy_score"]) if pd.notna(row.get("strategy_score")) else None,
        "strategy_component_a": float(row["strategy_component_a"]) if pd.notna(row.get("strategy_component_a")) else None,
        "strategy_component_b": float(row["strategy_component_b"]) if pd.notna(row.get("strategy_component_b")) else None,
        "strategy_component_c": float(row["strategy_component_c"]) if pd.notna(row.get("strategy_component_c")) else None,
        "entropy_quality": float(row["entropy_quality"]) if pd.notna(row.get("entropy_quality")) else None,
        "bifurcation_quality": float(row["bifurcation_quality"]) if pd.notna(row.get("bifurcation_quality")) else None,
        "trigger_quality": float(row["trigger_quality"]) if pd.notna(row.get("trigger_quality")) else None,
        "execution_quality": float(row["execution_quality"]) if pd.notna(row.get("execution_quality")) else None,
        "entropy_percentile_120": float(row["entropy_percentile_120"]) if pd.notna(row.get("entropy_percentile_120")) else None,
        "entropy_gap": float(row["entropy_gap"]) if pd.notna(row.get("entropy_gap")) else None,
        "perm_entropy_20_norm": float(row["perm_entropy_20_norm"]) if pd.notna(row.get("perm_entropy_20_norm")) else None,
        "perm_entropy_60_norm": float(row["perm_entropy_60_norm"]) if pd.notna(row.get("perm_entropy_60_norm")) else None,
        "ar1_20": float(row["ar1_20"]) if pd.notna(row.get("ar1_20")) else None,
        "phase_adjusted_ar1_20": float(row["phase_adjusted_ar1_20"]) if pd.notna(row.get("phase_adjusted_ar1_20")) else None,
        "phase_distortion_20": float(row["phase_distortion_20"]) if pd.notna(row.get("phase_distortion_20")) else None,
        "dominant_eig_20": float(row["dominant_eig_20"]) if pd.notna(row.get("dominant_eig_20")) else None,
        "dominant_eig_abs_20": float(row["dominant_eig_abs_20"]) if pd.notna(row.get("dominant_eig_abs_20")) else None,
        "recovery_rate_20": float(row["recovery_rate_20"]) if pd.notna(row.get("recovery_rate_20")) else None,
        "state_skew_20": float(row["state_skew_20"]) if pd.notna(row.get("state_skew_20")) else None,
        "var_lift_10_20": float(row["var_lift_10_20"]) if pd.notna(row.get("var_lift_10_20")) else None,
        "path_irreversibility_20": float(row["path_irreversibility_20"]) if pd.notna(row.get("path_irreversibility_20")) else None,
        "coarse_entropy_lb_20": float(row["coarse_entropy_lb_20"]) if pd.notna(row.get("coarse_entropy_lb_20")) else None,
        "entropy_accel_5": float(row["entropy_accel_5"]) if pd.notna(row.get("entropy_accel_5")) else None,
        "breakout_10": float(row["breakout_10"]) if pd.notna(row.get("breakout_10")) else None,
        "breakout_20": float(row["breakout_20"]) if pd.notna(row.get("breakout_20")) else None,
        "volume_impulse_5_20": float(row["volume_impulse_5_20"]) if pd.notna(row.get("volume_impulse_5_20")) else None,
        "flow_impulse_5_20": float(row["flow_impulse_5_20"]) if pd.notna(row.get("flow_impulse_5_20")) else None,
        "energy_impulse": float(row["energy_impulse"]) if pd.notna(row.get("energy_impulse")) else None,
        "order_alignment": float(row["order_alignment"]) if pd.notna(row.get("order_alignment")) else None,
        "mf_z_60": float(row["mf_z_60"]) if pd.notna(row.get("mf_z_60")) else None,
        "execution_cost_proxy_20": float(row["execution_cost_proxy_20"]) if pd.notna(row.get("execution_cost_proxy_20")) else None,
        "strategic_abandonment_seed": bool(row.get("strategic_abandonment_seed")) if pd.notna(row.get("strategic_abandonment_seed")) else False,
        "experimental_tda_score": row.get("experimental_tda_score"),
        "experimental_reservoir_tipping_score": row.get("experimental_reservoir_tipping_score"),
        "experimental_structure_latent_score": row.get("experimental_structure_latent_score"),
        "latent_compression_score": float(row["latent_compression_score"]) if pd.notna(row.get("latent_compression_score")) else None,
        "latent_instability_score": float(row["latent_instability_score"]) if pd.notna(row.get("latent_instability_score")) else None,
        "latent_launch_score": float(row["latent_launch_score"]) if pd.notna(row.get("latent_launch_score")) else None,
        "latent_diffusion_score": float(row["latent_diffusion_score"]) if pd.notna(row.get("latent_diffusion_score")) else None,
        "latent_state_label": str(row.get("latent_state_label") or ""),
        "stock_state_score": float(row["stock_state_score"]) if pd.notna(row.get("stock_state_score")) else None,
    }


def _execution_cost_state(execution_penalty_score: float) -> str:
    if execution_penalty_score >= 0.78:
        return "blocked"
    if execution_penalty_score >= 0.60:
        return "cautious"
    return "normal"


def _experimental_model_score(df: pd.DataFrame) -> pd.Series:
    columns = [
        "experimental_tda_score",
        "experimental_reservoir_tipping_score",
        "experimental_structure_latent_score",
    ]
    present = [column for column in columns if column in df.columns]
    if not present:
        return pd.Series(0.0, index=df.index, dtype="float64")
    frame = df[present].apply(pd.to_numeric, errors="coerce")
    return frame.mean(axis=1).fillna(0.0)


def _series_window_values(prepared: PreparedSymbol, scan_date: str, column: str, lookback: int) -> np.ndarray | None:
    idx = prepared.date_to_index.get(str(scan_date))
    if idx is None:
        return None
    start_idx = max(0, idx - int(lookback) + 1)
    window = pd.to_numeric(prepared.signal_daily.iloc[start_idx : idx + 1][column], errors="coerce").to_numpy(dtype=np.float64)
    if np.isfinite(window).sum() < max(8, int(lookback) // 2):
        return None
    return window


def _market_coupling_entropy_20(prepared_symbols: list[PreparedSymbol], scan_date: str, lookback: int = 20) -> float:
    industry_windows: dict[str, list[np.ndarray]] = {}
    for prepared in prepared_symbols:
        industry = str(prepared.industry or "UNKNOWN")
        window = _series_window_values(prepared, scan_date, "log_ret_1", lookback)
        if window is None:
            continue
        industry_windows.setdefault(industry, []).append(window)

    aggregated: dict[str, np.ndarray] = {}
    for industry, windows in industry_windows.items():
        min_len = min(len(window) for window in windows)
        if min_len < 8:
            continue
        stack = np.vstack([window[-min_len:] for window in windows])
        aggregated[industry] = np.nanmean(stack, axis=0)

    if len(aggregated) <= 1:
        return 0.0

    industries = sorted(aggregated)
    adjacency = np.zeros((len(industries), len(industries)), dtype=np.float64)
    for left_idx, left_industry in enumerate(industries):
        left = aggregated[left_industry]
        for right_idx, right_industry in enumerate(industries):
            if left_idx == right_idx:
                continue
            right = aggregated[right_industry]
            common_len = min(len(left), len(right))
            if common_len < 8:
                continue
            left_now = left[-common_len + 1 :]
            right_prev = right[-common_len:-1]
            mask = np.isfinite(left_now) & np.isfinite(right_prev)
            if int(mask.sum()) < 6:
                continue
            corr = float(np.corrcoef(left_now[mask], right_prev[mask])[0, 1])
            if np.isfinite(corr) and corr > 0.0:
                adjacency[left_idx, right_idx] = corr

    strengths = adjacency.sum(axis=1)
    return _normalized_shannon_entropy(pd.Series(strengths, dtype="float64"))


def _entry_plan(row: dict[str, Any]) -> dict[str, Any]:
    context_score = float(row.get("context_score") or 0.0)
    execution_state = str(row.get("execution_cost_state") or "normal")
    market_phase_state = str(row.get("market_phase_state") or "neutral")
    if bool(row.get("strategic_abandonment")):
        return {"position_scale": 0.0, "entry_mode": "skip", "staged_entry_days": 0, "exit_mode": "abandon"}

    if execution_state == "cautious" or market_phase_state in {"compression", "distorted"}:
        position_scale = float(np.clip(0.25 + 0.40 * context_score, 0.25, 0.60))
        return {
            "position_scale": position_scale,
            "entry_mode": "probe",
            "staged_entry_days": 3,
            "exit_mode": "reduce",
        }

    if market_phase_state == "transition":
        position_scale = float(np.clip(0.35 + 0.45 * context_score, 0.35, 0.75))
        return {
            "position_scale": position_scale,
            "entry_mode": "staged",
            "staged_entry_days": 2,
            "exit_mode": "trail",
        }

    position_scale = float(np.clip(0.45 + 0.55 * context_score, 0.45, 1.00))
    return {
        "position_scale": position_scale,
        "entry_mode": "full",
        "staged_entry_days": 1,
        "exit_mode": "trail",
    }


def _augment_cross_section(
    rows: list[dict[str, Any]], prepared_symbols: list[PreparedSymbol] | None = None, scan_date: str = ""
) -> list[dict[str, Any]]:
    if not rows:
        return rows
    df = pd.DataFrame(rows)
    for column in [
        "strategy_score",
        "stock_state_score",
        "entropy_quality",
        "bifurcation_quality",
        "trigger_quality",
        "execution_quality",
        "breakout_10",
        "entropy_percentile_120",
        "dominant_eig_abs_20",
        "energy_impulse",
        "turnover_rate",
        "state_skew_20",
        "path_irreversibility_20",
        "execution_cost_proxy_20",
        "phase_distortion_20",
        "experimental_tda_score",
        "experimental_reservoir_tipping_score",
        "experimental_structure_latent_score",
    ]:
        df[column] = pd.to_numeric(df.get(column), errors="coerce")
    df["strategy_state"] = pd.Series(df.get("strategy_state"), dtype="boolean").fillna(False).astype(bool)

    low_entropy_share = float((df["entropy_percentile_120"] <= 0.25).mean()) if not df.empty else 0.0
    breakout_share = float((df["breakout_10"] > 0.0).mean()) if not df.empty else 0.0
    energy_share = float((df["energy_impulse"] > 0.0).mean()) if not df.empty else 0.0
    bifurcation_share = float(((df["dominant_eig_abs_20"] >= 0.72) & (df["path_irreversibility_20"] >= 0.01)).mean()) if not df.empty else 0.0
    phase_distortion_share = float((df["phase_distortion_20"] >= 0.08).mean()) if df["phase_distortion_20"].notna().any() else 0.0

    industry_series = df["industry"].fillna("UNKNOWN") if "industry" in df.columns else pd.Series("UNKNOWN", index=df.index)
    positive_industry_energy = (
        df.assign(industry_bucket=industry_series)
        .groupby("industry_bucket")["energy_impulse"]
        .apply(lambda series: float(np.maximum(series.to_numpy(dtype=float), 0.0).sum()))
    )
    if prepared_symbols and scan_date:
        market_coupling_entropy_20 = _market_coupling_entropy_20(prepared_symbols, scan_date, lookback=20)
    else:
        market_coupling_entropy_20 = _normalized_shannon_entropy(positive_industry_energy)

    breakout_noise = float(df["breakout_10"].abs().median()) if df["breakout_10"].notna().any() else 0.0
    turnover_noise = float(df["turnover_rate"].median()) if df["turnover_rate"].notna().any() else 0.0
    skew_noise = float(df["state_skew_20"].abs().median()) if df["state_skew_20"].notna().any() else 0.0
    execution_noise = float(df["execution_cost_proxy_20"].median()) if df["execution_cost_proxy_20"].notna().any() else 0.0
    market_noise_cost = float(
        np.clip(
            0.28 * (breakout_noise / 0.04)
            + 0.18 * (turnover_noise / 6.0)
            + 0.16 * (skew_noise / 1.0)
            + 0.18 * execution_noise
            + 0.20 * phase_distortion_share,
            0.0,
            1.0,
        )
    )
    market_regime_score = float(
        np.clip(
            MARKET_REGIME_WEIGHTS["low_entropy_share"] * low_entropy_share
            + MARKET_REGIME_WEIGHTS["bifurcation_share"] * bifurcation_share
            + MARKET_REGIME_WEIGHTS["energy_share"] * energy_share
            + MARKET_REGIME_WEIGHTS["breakout_share"] * breakout_share
            + MARKET_REGIME_WEIGHTS["market_coupling_entropy_20"] * market_coupling_entropy_20
            + MARKET_REGIME_WEIGHTS["low_noise_support"] * (1.0 - market_noise_cost),
            0.0,
            1.0,
        )
    )
    phase_state = _market_phase_state(
        market_regime_score,
        low_entropy_share,
        breakout_share,
        market_noise_cost,
        market_coupling_entropy_20,
        phase_distortion_share,
    )
    execution_penalty_score = np.clip(
        EXECUTION_LAYER_WEIGHTS["asset_execution_cost"] * df["execution_cost_proxy_20"].fillna(1.0)
        + EXECUTION_LAYER_WEIGHTS["market_noise_cost"] * market_noise_cost,
        0.0,
        1.0,
    )
    execution_readiness_score = np.clip(1.0 - execution_penalty_score, 0.0, 1.0)
    experimental_model_score = _experimental_model_score(df)
    stock_state_series = df["stock_state_score"].fillna(df["strategy_score"].fillna(0.0))
    abandonment_score = np.clip(
        0.30 * execution_penalty_score
        + 0.25 * market_noise_cost
        + 0.20 * phase_distortion_share
        + 0.15 * (1.0 - stock_state_series)
        + 0.10 * (1.0 - market_coupling_entropy_20),
        0.0,
        1.0,
    )

    df["market_low_entropy_share"] = low_entropy_share
    df["market_breakout_share"] = breakout_share
    df["market_energy_share"] = energy_share
    df["market_bifurcation_share"] = bifurcation_share
    df["market_phase_distortion_share"] = phase_distortion_share
    df["market_coupling_entropy_20"] = market_coupling_entropy_20
    df["market_noise_cost"] = market_noise_cost
    df["market_regime_score"] = market_regime_score
    df["market_phase_state"] = phase_state
    df["execution_penalty_score"] = execution_penalty_score
    df["execution_readiness_score"] = execution_readiness_score
    df["experimental_model_score"] = experimental_model_score
    df["execution_cost_state"] = [_execution_cost_state(float(score or 0.0)) for score in execution_penalty_score.tolist()]
    df["abandonment_score"] = abandonment_score
    df["strategic_abandonment"] = (
        df["strategic_abandonment_seed"].fillna(False).astype(bool)
        | (df["execution_cost_state"] == "blocked")
        | (df["market_phase_state"] == "abandon")
        | ((df["execution_penalty_score"] >= 0.70) & (stock_state_series < 0.62))
        | ((market_noise_cost >= 0.78) & (market_coupling_entropy_20 < 0.30))
        | (abandonment_score >= 0.74)
    )
    df["context_score"] = np.clip(
        DECISION_LAYER_WEIGHTS["stock_state"] * stock_state_series
        + DECISION_LAYER_WEIGHTS["market_gate"] * market_regime_score
        + DECISION_LAYER_WEIGHTS["execution_readiness"] * df["execution_readiness_score"].fillna(0.0)
        + DECISION_LAYER_WEIGHTS["experimental_model"] * df["experimental_model_score"].fillna(0.0)
        - 0.30 * df["strategic_abandonment"].astype(float),
        0.0,
        1.0,
    )
    plans = [_entry_plan(row) for row in df.to_dict(orient="records")]
    df["position_scale"] = [float(plan["position_scale"]) for plan in plans]
    df["entry_mode"] = [str(plan["entry_mode"]) for plan in plans]
    df["staged_entry_days"] = [int(plan["staged_entry_days"]) for plan in plans]
    df["exit_mode"] = [str(plan["exit_mode"]) for plan in plans]

    df["score_pct_rank"] = df["strategy_score"].rank(pct=True, method="average")
    df["entropy_pct_rank"] = df["entropy_quality"].rank(pct=True, method="average")
    df["bifurcation_pct_rank"] = df["bifurcation_quality"].rank(pct=True, method="average")
    df["trigger_pct_rank"] = df["trigger_quality"].rank(pct=True, method="average")
    df["context_score_pct_rank"] = df["context_score"].rank(pct=True, method="average")
    return df.to_dict(orient="records")


def _passes_candidate_filters(row: dict[str, Any], config: EntropyBifurcationScanConfig) -> bool:
    if not row or not bool(row.get("strategy_state")):
        return False
    if bool(config.exclude_st) and bool(row.get("is_st")):
        return False
    amount = float(row.get("amount") or 0.0)
    turnover_rate = float(row.get("turnover_rate") or 0.0)
    if amount < float(config.min_amount):
        return False
    if turnover_rate < float(config.min_turnover):
        return False
    if bool(row.get("strategic_abandonment")):
        return False
    if str(row.get("execution_cost_state") or "") == "blocked":
        return False
    if str(row.get("market_phase_state") or "") == "abandon":
        return False
    if str(row.get("execution_cost_state") or "") == "cautious" and float(row.get("stock_state_score") or row.get("strategy_score") or 0.0) < 0.62:
        return False
    if str(row.get("market_phase_state") or "") == "distorted" and float(row.get("stock_state_score") or row.get("strategy_score") or 0.0) < 0.45:
        return False
    if float(row.get("market_regime_score") or 0.0) < 0.10:
        return False
    if float(row.get("context_score") or 0.0) < 0.45:
        return False
    return True


def _sort_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            float(row.get("context_score") or -999.0),
            float(row.get("stock_state_score") or row.get("strategy_score") or -999.0),
            float(row.get("market_regime_score") or -999.0),
            float(row.get("execution_readiness_score") or -999.0),
            float(row.get("position_scale") or -999.0),
            float(row.get("bifurcation_quality") or -999.0),
            float(row.get("path_irreversibility_20") or -999.0),
            float(row.get("dominant_eig_abs_20") or -999.0),
            float(row.get("amount") or -999.0),
        ),
        reverse=True,
    )


def _select_candidate_rows(rows: list[dict[str, Any]], top_n: int, max_positions_per_industry: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    industry_counts: dict[str, int] = {}
    for row in rows:
        if len(selected) >= int(top_n):
            break
        industry = _industry_bucket(row)
        if int(max_positions_per_industry) > 0 and industry_counts.get(industry, 0) >= int(max_positions_per_industry):
            skipped.append(dict(row))
            continue
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
        selected.append(dict(row))
    return selected, skipped


def _build_scan_rows(prepared_symbols: list[PreparedSymbol], config: EntropyBifurcationScanConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    market_rows = [row for row in (_build_snapshot_row(prepared, str(config.scan_date)) for prepared in prepared_symbols) if row]
    market_rows = _augment_cross_section(market_rows, prepared_symbols, str(config.scan_date))
    candidate_rows = [dict(row) for row in market_rows if _passes_candidate_filters(row, config)]
    candidate_rows = _sort_candidate_rows(candidate_rows)
    selected_candidates, _ = _select_candidate_rows(candidate_rows, int(config.top_n), int(config.max_positions_per_industry))
    selected_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(selected_candidates, start=1):
        selected_row = dict(row)
        selected_row["selected_rank"] = rank
        selected_rows.append(selected_row)
    return market_rows, candidate_rows, selected_rows


def _trade_plan_from_snapshot(prepared: PreparedSymbol, scan_date: str, hold_days: int) -> dict[str, Any] | None:
    idx = prepared.date_to_index.get(str(scan_date))
    if idx is None:
        return None
    entry_idx = idx + 1
    if entry_idx >= len(prepared.signal_daily):
        return None
    exit_idx = min(entry_idx + max(1, int(hold_days)) - 1, len(prepared.signal_daily) - 1)
    entry_row = prepared.signal_daily.iloc[entry_idx]
    exit_row = prepared.signal_daily.iloc[exit_idx]
    entry_price = float(entry_row["open"])
    exit_price = float(exit_row["close"])
    if entry_price <= 0:
        return None
    max_close = float(prepared.signal_daily.iloc[entry_idx : exit_idx + 1]["close"].max())
    return {
        "entry_date": str(entry_row["bar_start"]),
        "exit_date": str(exit_row["bar_end"]),
        "entry_scan_date": str(scan_date),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "hold_days_realized": int(exit_idx - entry_idx + 1),
        "max_runup_pct": (max_close / entry_price - 1.0) * 100.0,
        "return_pct": (exit_price / entry_price - 1.0) * 100.0,
    }


def _daily_mark_to_market_return(prepared: PreparedSymbol, trade: dict[str, Any], current_date: str) -> float | None:
    if str(current_date) < str(trade["entry_date"]) or str(current_date) > str(trade["exit_date"]):
        return None
    current_close = prepared.close_by_date.get(str(current_date))
    if current_close is None or current_close <= 0:
        return None
    if str(current_date) == str(trade["entry_date"]):
        entry_price = float(trade.get("entry_price") or 0.0)
        if entry_price <= 0:
            return None
        return current_close / entry_price - 1.0
    current_idx = prepared.date_to_index.get(str(current_date))
    if current_idx is None or current_idx <= 0:
        return None
    prev_date = str(prepared.signal_daily.iloc[current_idx - 1]["bar_end"])
    prev_close = prepared.close_by_date.get(prev_date)
    if prev_close is None or prev_close <= 0:
        return None
    return current_close / prev_close - 1.0


def _position_live_scale(prepared: PreparedSymbol, trade: dict[str, Any], current_date: str) -> float:
    if str(current_date) < str(trade.get("entry_date") or "") or str(current_date) > str(trade.get("exit_date") or ""):
        return 0.0
    target_scale = float(trade.get("position_scale") or 0.0)
    if target_scale <= 0.0:
        return 0.0
    current_idx = prepared.date_to_index.get(str(current_date))
    entry_idx = prepared.date_to_index.get(str(trade.get("entry_date") or ""))
    if current_idx is None or entry_idx is None:
        return target_scale
    held_days = max(0, current_idx - entry_idx)
    mode = str(trade.get("entry_mode") or "full")
    if mode == "probe":
        schedule = [0.35, 0.65, 1.00]
        fraction = schedule[min(held_days, len(schedule) - 1)]
        return target_scale * fraction
    if mode == "staged":
        schedule = [0.50, 1.00]
        fraction = schedule[min(held_days, len(schedule) - 1)]
        return target_scale * fraction
    return target_scale


def _run_forward_backtest(prepared_symbols: list[PreparedSymbol], config: EntropyBifurcationScanConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    if not config.backtest_start_date or not config.backtest_end_date:
        return [], [], None

    scan_dates: set[str] = set()
    for prepared in prepared_symbols:
        for trade_date in prepared.date_to_index:
            if str(config.backtest_start_date) <= trade_date <= str(config.backtest_end_date):
                scan_dates.add(trade_date)
    ordered_scan_dates = sorted(scan_dates)
    if not ordered_scan_dates:
        return [], [], None

    symbol_lookup = {prepared.symbol: prepared for prepared in prepared_symbols}
    active_positions: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    nav = 1.0

    for scan_date in ordered_scan_dates:
        active_positions = [pos for pos in active_positions if str(pos["exit_date"]) >= scan_date]
        market_rows_day = [row for row in (_build_snapshot_row(prepared, scan_date) for prepared in prepared_symbols) if row is not None]
        market_rows_day = _augment_cross_section(market_rows_day, prepared_symbols, scan_date)
        candidate_rows_day = [dict(row) for row in market_rows_day if _passes_candidate_filters(row, config)]
        candidate_rows_day = _sort_candidate_rows(candidate_rows_day)[: int(config.top_n)]

        accepted_today = 0
        skipped_full = 0
        skipped_industry = 0

        for candidate in candidate_rows_day:
            prepared = symbol_lookup.get(str(candidate["symbol"]))
            if prepared is None:
                continue
            plan = _trade_plan_from_snapshot(prepared, scan_date, int(config.hold_days))
            if plan is None:
                continue

            entry_date = str(plan["entry_date"])
            active_positions = [pos for pos in active_positions if str(pos["exit_date"]) >= entry_date]
            if any(str(pos["symbol"]) == str(candidate["symbol"]) for pos in active_positions):
                continue
            if len(active_positions) >= int(config.max_positions):
                skipped_full += 1
                continue
            industry = _industry_bucket(candidate)
            industry_open_count = sum(1 for pos in active_positions if _industry_bucket(pos) == industry)
            if int(config.max_positions_per_industry) > 0 and industry_open_count >= int(config.max_positions_per_industry):
                skipped_industry += 1
                continue

            trade = {**candidate, **plan, "scan_date": scan_date}
            trades.append(trade)
            active_positions.append(
                {
                    "symbol": trade["symbol"],
                    "ts_code": trade["ts_code"],
                    "industry": trade.get("industry"),
                    "entry_date": entry_date,
                    "exit_date": str(trade["exit_date"]),
                    "entry_price": float(trade["entry_price"]),
                    "position_scale": float(trade.get("position_scale") or 0.0),
                    "entry_mode": str(trade.get("entry_mode") or "full"),
                    "staged_entry_days": int(trade.get("staged_entry_days") or 1),
                }
            )
            accepted_today += 1

        weighted_return_sum = 0.0
        gross_exposure = 0.0
        for position in active_positions:
            prepared = symbol_lookup.get(str(position["symbol"]))
            if prepared is None:
                continue
            trade_return = _daily_mark_to_market_return(prepared, position, scan_date)
            live_scale = _position_live_scale(prepared, position, scan_date)
            if trade_return is not None and live_scale > 0.0:
                weighted_return_sum += float(live_scale) * float(trade_return)
                gross_exposure += float(live_scale)

        strategy_daily_return = weighted_return_sum / max(float(config.max_positions), 1.0)
        nav *= 1.0 + strategy_daily_return
        realized_trades = [trade for trade in trades if str(trade["exit_date"]) == scan_date]

        daily_rows.append(
            {
                "strategy_name": str(config.strategy_name),
                "scan_date": scan_date,
                "n_candidates": len(candidate_rows_day),
                "n_selected": accepted_today,
                "n_skipped_full": skipped_full,
                "n_skipped_industry": skipped_industry,
                "active_positions": int(sum(_position_live_scale(symbol_lookup[str(pos["symbol"])], pos, scan_date) > 0.0 for pos in active_positions if str(pos["symbol"]) in symbol_lookup)),
                "gross_exposure": float(gross_exposure),
                "realized_trades": int(len(realized_trades)),
                "strategy_daily_return": float(strategy_daily_return * 100.0),
                "nav": float(nav),
            }
        )

    if not trades:
        return daily_rows, [], {
            "strategy_name": str(config.strategy_name),
            "backtest_start_date": str(config.backtest_start_date),
            "backtest_end_date": str(config.backtest_end_date),
            "hold_days": int(config.hold_days),
            "top_n": int(config.top_n),
            "max_positions": int(config.max_positions),
            "max_positions_per_industry": int(config.max_positions_per_industry),
            "n_trades": 0,
            "final_nav": float(nav),
            "total_return_pct": float((nav - 1.0) * 100.0),
        }

    df_trades = pd.DataFrame(trades)
    df_daily = pd.DataFrame(daily_rows)
    max_drawdown_pct = None
    if not df_daily.empty and "nav" in df_daily.columns:
        nav_series = df_daily["nav"].astype(float)
        running_peak = nav_series.cummax()
        drawdown = nav_series / running_peak - 1.0
        max_drawdown_pct = float(drawdown.min() * 100.0)
    return daily_rows, trades, {
        "strategy_name": str(config.strategy_name),
        "backtest_start_date": str(config.backtest_start_date),
        "backtest_end_date": str(config.backtest_end_date),
        "hold_days": int(config.hold_days),
        "top_n": int(config.top_n),
        "max_positions": int(config.max_positions),
        "max_positions_per_industry": int(config.max_positions_per_industry),
        "n_trades": int(len(df_trades)),
        "win_rate": float((df_trades["return_pct"] > 0).mean()),
        "avg_return_pct": float(df_trades["return_pct"].mean()),
        "median_return_pct": float(df_trades["return_pct"].median()),
        "avg_max_runup_pct": float(df_trades["max_runup_pct"].mean()),
        "avg_position_scale": float(df_trades["position_scale"].mean()) if "position_scale" in df_trades.columns else np.nan,
        "final_nav": float(nav),
        "total_return_pct": float((nav - 1.0) * 100.0),
        "max_drawdown_pct": max_drawdown_pct,
        "total_skipped_industry": int(df_daily["n_skipped_industry"].sum()) if not df_daily.empty else 0,
    }


def run_entropy_bifurcation_scan(config: EntropyBifurcationScanConfig) -> list[str]:
    if str(config.strategy_name) not in VALID_STRATEGIES:
        supported = ", ".join(sorted(VALID_STRATEGIES))
        raise SystemExit(f"Unsupported strategy_name={config.strategy_name}. Supported: {supported}")

    files = _resolve_files(config.data_dir, config.symbols)
    if not files:
        raise SystemExit(f"No CSVs found in data_dir={config.data_dir}")

    effective_scan_date = _infer_scan_date(files, config.scan_date)
    if config.backtest_end_date and str(config.backtest_end_date) < effective_scan_date:
        effective_scan_date = str(config.backtest_end_date)
    config = EntropyBifurcationScanConfig(**{**config.__dict__, "scan_date": effective_scan_date})

    basic_info_map = _load_basic_info_map(config.basic_path)
    prepared_symbols = _prepare_all_symbols(files, config, basic_info_map)
    market_rows, candidate_rows, selected_rows = _build_scan_rows(prepared_symbols, config)
    backtest_daily_rows, backtest_trade_rows, backtest_summary = _run_forward_backtest(prepared_symbols, config)

    return write_entropy_bifurcation_outputs(
        out_dir=config.out_dir,
        strategy_name=str(config.strategy_name),
        scan_date=str(config.scan_date),
        top_n=int(config.top_n),
        market_rows=market_rows,
        candidate_rows=candidate_rows,
        selected_rows=selected_rows,
        backtest_daily_rows=backtest_daily_rows,
        backtest_trade_rows=backtest_trade_rows,
        backtest_summary=backtest_summary,
    )