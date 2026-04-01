from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..target_resolver import resolve_target
from ..io_utils import normalize_trade_date
from .entropy_hold_feature_engine import build_entropy_hold_feature_frame
from .entropy_hold_report_writer import write_entropy_hold_outputs
from .entropy_hold_signal_model import evaluate_hold_path


@dataclass(frozen=True)
class EntropyHoldConfig:
    strategy_name: str = "entropy_hold_judgement"
    data_dir: str = ""
    out_dir: str = ""
    symbol_or_name: str = ""
    start_date: str = ""
    scan_date: str = ""
    basic_path: str = ""
    lookback_years: int = 5
    exit_persist_days: int = 3


def _resolve_target(symbol_or_name: str, data_dir: str, basic_path: str) -> dict[str, str]:
    return resolve_target(symbol_or_name, data_dir, basic_path)


def _infer_scan_date(df: pd.DataFrame, requested_scan_date: str) -> str:
    if requested_scan_date:
        return normalize_trade_date(requested_scan_date, allow_empty=False)
    if df.empty or "trade_date" not in df.columns:
        raise ValueError("Unable to infer scan_date")
    return normalize_trade_date(df["trade_date"].astype(str).max(), allow_empty=False)


def _prepare_evaluation_frame(config: EntropyHoldConfig) -> tuple[dict[str, str], pd.DataFrame, str]:
    target = _resolve_target(config.symbol_or_name, config.data_dir, config.basic_path)
    df = pd.read_csv(target["file_path"])
    if df.empty or "trade_date" not in df.columns:
        raise ValueError("Daily file is empty or missing trade_date")

    normalized_start_date = normalize_trade_date(config.start_date, allow_empty=False)
    effective_scan_date = _infer_scan_date(df, config.scan_date)
    start_keep = f"{int(effective_scan_date[:4]) - int(config.lookback_years)}0101"
    df = df[(df["trade_date"].astype(str) >= start_keep) & (df["trade_date"].astype(str) <= effective_scan_date)].copy()
    if df.empty:
        raise ValueError("No rows available after applying lookback window")

    features = build_entropy_hold_feature_frame(df)
    if features.empty:
        raise ValueError("Unable to build entropy hold features")
    evaluated = evaluate_hold_path(features, normalized_start_date, effective_scan_date, int(config.exit_persist_days))
    if evaluated.empty:
        raise ValueError("No rows available for hold evaluation")
    return target, evaluated, effective_scan_date


def _build_result_row(target: dict[str, str], evaluated: pd.DataFrame, config: EntropyHoldConfig, scan_date: str) -> dict[str, Any]:
    latest = evaluated.iloc[-1]
    entry = evaluated.iloc[0]
    exit_rows = evaluated[evaluated["exit_persist"].astype(bool)]
    first_exit_date = str(exit_rows.iloc[0]["bar_end"]) if not exit_rows.empty else ""
    can_hold = not bool(evaluated["exit_persist"].astype(bool).any())
    entry_close = float(entry["close"]) if pd.notna(entry.get("close")) else None
    latest_close = float(latest["close"]) if pd.notna(latest.get("close")) else None
    holding_return_pct = None
    if entry_close and latest_close and entry_close > 0:
        holding_return_pct = latest_close / entry_close - 1.0

    return {
        "symbol": target["symbol"],
        "ts_code": target["ts_code"],
        "name": target["name"],
        "area": target["area"],
        "industry": target["industry"],
        "market": target["market"],
        "start_date": str(entry["bar_end"]),
        "scan_date": str(scan_date),
        "first_exit_date": first_exit_date,
        "judgement": "continue_hold" if can_hold else "exit_hold",
        "strategy_name": str(config.strategy_name),
        "strategy_state": bool(can_hold),
        "strategy_score": float(latest["strategy_score"]) if pd.notna(latest.get("strategy_score")) else None,
        "hold_score": float(latest["hold_score"]) if pd.notna(latest.get("hold_score")) else None,
        "entropy_reserve": float(latest["entropy_reserve"]) if pd.notna(latest.get("entropy_reserve")) else None,
        "memory_reserve": float(latest["memory_reserve"]) if pd.notna(latest.get("memory_reserve")) else None,
        "phase_stability": float(latest["phase_stability"]) if pd.notna(latest.get("phase_stability")) else None,
        "disorder_pressure": float(latest["disorder_pressure"]) if pd.notna(latest.get("disorder_pressure")) else None,
        "max_disorder_pressure": float(pd.to_numeric(evaluated["disorder_pressure"], errors="coerce").max()),
        "entropy_percentile_120": float(latest["entropy_percentile_120"]) if pd.notna(latest.get("entropy_percentile_120")) else None,
        "entropy_gap_mean_5": float(latest["entropy_gap_mean_5"]) if pd.notna(latest.get("entropy_gap_mean_5")) else None,
        "ar1_20": float(latest["ar1_20"]) if pd.notna(latest.get("ar1_20")) else None,
        "recovery_rate_20": float(latest["recovery_rate_20"]) if pd.notna(latest.get("recovery_rate_20")) else None,
        "entropy_drift_10": float(latest["entropy_drift_10"]) if pd.notna(latest.get("entropy_drift_10")) else None,
        "holding_days": int(len(evaluated)),
        "entry_close": entry_close,
        "latest_close": latest_close,
        "holding_return_pct": holding_return_pct,
    }


def run_entropy_hold_scan(config: EntropyHoldConfig) -> list[str]:
    target, evaluated, scan_date = _prepare_evaluation_frame(config)
    result_row = _build_result_row(target, evaluated, config, scan_date)

    summary_row = {
        "strategy_name": str(config.strategy_name),
        "scan_date": str(scan_date),
        "symbol": result_row["symbol"],
        "name": result_row["name"],
        "start_date": result_row["start_date"],
        "can_hold": bool(result_row["strategy_state"]),
        "first_exit_date": result_row["first_exit_date"],
        "holding_days": int(result_row["holding_days"]),
        "holding_return_pct": result_row["holding_return_pct"],
        "avg_strategy_score": result_row["strategy_score"],
        "avg_disorder_pressure": result_row["disorder_pressure"],
        "n_scanned": 1,
        "n_candidates": 1,
        "n_selected": 1,
    }

    diagnostics_rows = evaluated[
        [
            "bar_end",
            "close",
            "strategy_score",
            "strategy_state",
            "hold_score",
            "entropy_reserve",
            "memory_reserve",
            "phase_stability",
            "disorder_pressure",
            "entropy_percentile_120",
            "entropy_gap_mean_5",
            "ar1_20",
            "recovery_rate_20",
            "entropy_drift_10",
            "disorder_state",
            "memory_collapse",
            "phase_break",
            "exit_seed",
            "exit_persist",
        ]
    ].astype(object).where(pd.notna(evaluated), None).to_dict(orient="records")

    return write_entropy_hold_outputs(
        out_dir=config.out_dir,
        strategy_name=str(config.strategy_name),
        scan_date=str(scan_date),
        market_rows=[result_row],
        selected_rows=[result_row],
        candidate_rows=[result_row],
        summary_row=summary_row,
        diagnostics_rows=diagnostics_rows,
    )