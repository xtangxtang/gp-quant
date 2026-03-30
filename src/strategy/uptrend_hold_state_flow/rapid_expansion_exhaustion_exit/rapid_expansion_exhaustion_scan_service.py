import os
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..io_utils import normalize_trade_date
from .rapid_expansion_exhaustion_feature_engine import build_rapid_expansion_exhaustion_feature_frame
from .rapid_expansion_exhaustion_report_writer import write_rapid_expansion_exhaustion_outputs
from .rapid_expansion_exhaustion_signal_model import evaluate_rapid_expansion_exhaustion_path


@dataclass(frozen=True)
class RapidExpansionExhaustionConfig:
    strategy_name: str = "rapid_expansion_exhaustion_exit"
    data_dir: str = ""
    out_dir: str = ""
    symbol_or_name: str = ""
    start_date: str = ""
    scan_date: str = ""
    basic_path: str = ""
    lookback_years: int = 5
    exit_persist_days: int = 2


def _build_symbol_from_ts_code(ts_code: str) -> str:
    if not ts_code or "." not in ts_code:
        return str(ts_code).lower()
    code, exch = ts_code.split(".", 1)
    return f"{exch.lower()}{code}"


def _infer_ts_code_from_numeric(code: str) -> str:
    c = str(code).strip()
    if c.startswith("92"):
        return f"{c}.BJ"
    if c.startswith(("6", "9")):
        return f"{c}.SH"
    return f"{c}.SZ"


def _load_basic_frame(basic_path: str) -> pd.DataFrame:
    if not basic_path or not os.path.exists(basic_path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(basic_path, usecols=["ts_code", "symbol", "name", "area", "industry", "market"])
        return df.fillna("") if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _resolve_target(symbol_or_name: str, data_dir: str, basic_path: str) -> dict[str, str]:
    raw = str(symbol_or_name or "").strip()
    if not raw:
        raise ValueError("symbol_or_name is required")

    basic_df = _load_basic_frame(basic_path)
    lower = raw.lower()
    ts_code = ""
    symbol = ""
    name = ""
    area = ""
    industry = ""
    market = ""

    if lower.startswith(("sh", "sz", "bj")):
        symbol = lower
        ts_code = _infer_ts_code_from_numeric(lower[2:]) if "." not in lower else lower.upper()
    elif raw.endswith((".SH", ".SZ", ".BJ", ".sh", ".sz", ".bj")):
        ts_code = raw.upper()
        symbol = _build_symbol_from_ts_code(ts_code)
    elif raw.isdigit():
        ts_code = _infer_ts_code_from_numeric(raw)
        symbol = _build_symbol_from_ts_code(ts_code)
    else:
        if basic_df.empty:
            raise ValueError("basic_path is required when resolving by stock name")
        exact = basic_df[basic_df["name"].astype(str) == raw]
        if exact.empty:
            partial = basic_df[basic_df["name"].astype(str).str.contains(raw, regex=False)]
            if len(partial) != 1:
                raise ValueError(f"Unable to uniquely resolve stock name: {raw}")
            exact = partial
        row = exact.iloc[0]
        ts_code = str(row.get("ts_code") or "")
        symbol = _build_symbol_from_ts_code(ts_code)
        name = str(row.get("name") or "")
        area = str(row.get("area") or "")
        industry = str(row.get("industry") or "")
        market = str(row.get("market") or "")

    if basic_df is not None and not basic_df.empty and ts_code:
        matched = basic_df[basic_df["ts_code"].astype(str) == ts_code]
        if not matched.empty:
            row = matched.iloc[0]
            name = str(row.get("name") or name)
            area = str(row.get("area") or area)
            industry = str(row.get("industry") or industry)
            market = str(row.get("market") or market)

    file_path = os.path.join(data_dir, f"{symbol}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV not found for {raw}: {file_path}")

    return {
        "symbol": symbol,
        "ts_code": ts_code,
        "name": name,
        "area": area,
        "industry": industry,
        "market": market,
        "file_path": file_path,
    }


def _infer_scan_date(df: pd.DataFrame, requested_scan_date: str) -> str:
    if requested_scan_date:
        return normalize_trade_date(requested_scan_date, allow_empty=False)
    if df.empty or "trade_date" not in df.columns:
        raise ValueError("Unable to infer scan_date")
    return normalize_trade_date(df["trade_date"].astype(str).max(), allow_empty=False)


def _prepare_evaluation_frame(config: RapidExpansionExhaustionConfig) -> tuple[dict[str, str], pd.DataFrame, str]:
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

    features = build_rapid_expansion_exhaustion_feature_frame(df)
    if features.empty:
        raise ValueError("Unable to build rapid expansion exhaustion features")
    evaluated = evaluate_rapid_expansion_exhaustion_path(features, normalized_start_date, effective_scan_date, int(config.exit_persist_days))
    if evaluated.empty:
        raise ValueError("No rows available for rapid expansion exhaustion evaluation")
    return target, evaluated, effective_scan_date


def _build_result_row(target: dict[str, str], evaluated: pd.DataFrame, config: RapidExpansionExhaustionConfig, scan_date: str) -> dict[str, Any]:
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
        "exit_score": float(latest["exit_score"]) if pd.notna(latest.get("exit_score")) else None,
        "peak_extension_score": float(latest["peak_extension_score"]) if pd.notna(latest.get("peak_extension_score")) else None,
        "deceleration_score": float(latest["deceleration_score"]) if pd.notna(latest.get("deceleration_score")) else None,
        "fragility_score": float(latest["fragility_score"]) if pd.notna(latest.get("fragility_score")) else None,
        "exhaustion_exit_score": float(latest["exhaustion_exit_score"]) if pd.notna(latest.get("exhaustion_exit_score")) else None,
        "expansion_thrust": float(latest["expansion_thrust"]) if pd.notna(latest.get("expansion_thrust")) else None,
        "directional_persistence": float(latest["directional_persistence"]) if pd.notna(latest.get("directional_persistence")) else None,
        "acceptance_score": float(latest["acceptance_score"]) if pd.notna(latest.get("acceptance_score")) else None,
        "instability_risk": float(latest["instability_risk"]) if pd.notna(latest.get("instability_risk")) else None,
        "thrust_drift_3": float(latest["thrust_drift_3"]) if pd.notna(latest.get("thrust_drift_3")) else None,
        "holding_days": int(len(evaluated)),
        "entry_close": entry_close,
        "latest_close": latest_close,
        "holding_return_pct": holding_return_pct,
    }


def run_rapid_expansion_exhaustion_scan(config: RapidExpansionExhaustionConfig) -> list[str]:
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
        "avg_disorder_pressure": float(pd.to_numeric(evaluated["fragility_score"], errors="coerce").mean()),
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
            "exit_score",
            "peak_extension_score",
            "deceleration_score",
            "fragility_score",
            "exhaustion_exit_score",
            "expansion_thrust",
            "directional_persistence",
            "acceptance_score",
            "instability_risk",
            "disorder_pressure",
            "thrust_drift_3",
            "terminal_zone",
            "deceleration_state",
            "support_fracture",
            "fragility_state",
            "exit_seed",
            "exit_persist",
        ]
    ].astype(object).where(pd.notna(evaluated), None).to_dict(orient="records")

    return write_rapid_expansion_exhaustion_outputs(
        out_dir=config.out_dir,
        strategy_name=str(config.strategy_name),
        scan_date=str(scan_date),
        market_rows=[result_row],
        selected_rows=[result_row],
        candidate_rows=[result_row],
        summary_row=summary_row,
        diagnostics_rows=diagnostics_rows,
    )