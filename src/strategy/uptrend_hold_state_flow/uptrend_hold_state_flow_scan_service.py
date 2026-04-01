from dataclasses import dataclass
from typing import Any

import pandas as pd

from .target_resolver import resolve_target
from .entropy_hold_judgement.entropy_hold_signal_model import evaluate_hold_path
from .entropy_hold_judgement.entropy_hold_feature_engine import build_entropy_hold_feature_frame
from .rapid_expansion_hold.rapid_expansion_signal_model import evaluate_rapid_expansion_path
from .rapid_expansion_hold.rapid_expansion_feature_engine import build_rapid_expansion_feature_frame
from .rapid_expansion_exhaustion_exit.rapid_expansion_exhaustion_signal_model import (
    evaluate_rapid_expansion_exhaustion_path,
)
from .rapid_expansion_exhaustion_exit.rapid_expansion_exhaustion_feature_engine import (
    build_rapid_expansion_exhaustion_feature_frame,
)
from .io_utils import normalize_trade_date
from .uptrend_hold_state_flow_report_writer import write_uptrend_hold_state_flow_outputs


STATE_LABELS = {
    "observation": "观察区",
    "entropy_hold_judgement": "熵秩序持有",
    "rapid_expansion_hold": "快速扩张持有",
    "rapid_expansion_exhaustion_exit": "快速扩张衰竭退出",
}


@dataclass(frozen=True)
class UptrendHoldStateFlowConfig:
    strategy_name: str = "uptrend_hold_state_flow"
    data_dir: str = ""
    out_dir: str = ""
    symbol_or_name: str = ""
    start_date: str = ""
    scan_date: str = ""
    basic_path: str = ""
    lookback_years: int = 5


def _resolve_target(symbol_or_name: str, data_dir: str, basic_path: str) -> dict[str, str]:
    return resolve_target(symbol_or_name, data_dir, basic_path)


def _infer_scan_date(df: pd.DataFrame, requested_scan_date: str) -> str:
    if requested_scan_date:
        return normalize_trade_date(requested_scan_date, allow_empty=False)
    if df.empty or "trade_date" not in df.columns:
        raise ValueError("Unable to infer scan_date")
    return normalize_trade_date(df["trade_date"].astype(str).max(), allow_empty=False)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if pd.isna(number):
        return float(default)
    return number


def _as_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _state_reason(state_id: str, latest_row: pd.Series) -> str:
    if state_id == "entropy_hold_judgement":
        return (
            f"低熵秩序仍占优，持有分数 {_as_float(latest_row.get('entropy_hold_score')):.2f}，"
            f"乱序压力 {_as_float(latest_row.get('entropy_disorder_pressure')):.2f}。"
        )
    if state_id == "rapid_expansion_hold":
        return (
            f"当前仍在快速扩张持有段，扩张推力 {_as_float(latest_row.get('rapid_expansion_thrust')):.2f}，"
            f"方向持续性 {_as_float(latest_row.get('rapid_directional_persistence')):.2f}。"
        )
    if state_id == "rapid_expansion_exhaustion_exit":
        return (
            f"强扩张末端开始衰竭，高位扩张分数 {_as_float(latest_row.get('exhaustion_peak_extension_score')):.2f}，"
            f"降速分数 {_as_float(latest_row.get('exhaustion_deceleration_score')):.2f}。"
        )
    return "当前三类状态都未充分激活，暂时更接近观察区。"


def _state_advice(state_id: str, first_exit_date: str) -> str:
    if state_id == "entropy_hold_judgement":
        return "买点以来仍可按熵秩序持有逻辑继续跟踪，重点防范高熵乱序和记忆坍缩。"
    if state_id == "rapid_expansion_hold":
        return "买点以来仍可按快速扩张持有逻辑跟踪，重点盯住推力是否衰减以及承接是否转弱。"
    if state_id == "rapid_expansion_exhaustion_exit":
        if first_exit_date:
            return f"持有路径已在 {first_exit_date} 触发退出确认，更合理的是减仓或退出，而不是继续按加速段逻辑持有。"
        return "持有路径已经进入强扩张末端风险区，更合理的是减仓或退出，而不是继续按加速段逻辑持有。"
    return "买点以来尚未形成稳定持有优势，更合理的是重新评估，而不是只按当前路径继续持有。"


def _path_judgement_label(state_id: str) -> tuple[str, str]:
    if state_id in {"entropy_hold_judgement", "rapid_expansion_hold"}:
        return "continue_hold", "继续持有"
    if state_id == "rapid_expansion_exhaustion_exit":
        return "exit_hold", "退出持有"
    return "reassess_hold", "重新评估"


def _prepare_path_frame(
    entropy_eval: pd.DataFrame,
    rapid_eval: pd.DataFrame,
    exhaustion_eval: pd.DataFrame,
) -> pd.DataFrame:
    entropy_frame = entropy_eval[
        [
            "bar_end",
            "close",
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
            "hold_path_state",
        ]
    ].rename(
        columns={
            "close": "path_close",
            "hold_score": "entropy_hold_score",
            "entropy_reserve": "entropy_entropy_reserve",
            "memory_reserve": "entropy_memory_reserve",
            "phase_stability": "entropy_phase_stability",
            "disorder_pressure": "entropy_disorder_pressure",
            "entropy_percentile_120": "entropy_percentile_120",
            "entropy_gap_mean_5": "entropy_gap_mean_5",
            "ar1_20": "entropy_ar1_20",
            "recovery_rate_20": "entropy_recovery_rate_20",
            "entropy_drift_10": "entropy_drift_10",
            "disorder_state": "entropy_disorder_state",
            "memory_collapse": "entropy_memory_collapse",
            "phase_break": "entropy_phase_break",
            "exit_seed": "entropy_exit_seed",
            "exit_persist": "entropy_exit_persist",
            "hold_path_state": "entropy_hold_path_state",
        }
    )
    rapid_frame = rapid_eval[
        [
            "bar_end",
            "expansion_hold_score",
            "expansion_thrust",
            "directional_persistence",
            "acceptance_score",
            "instability_risk",
            "constructive_expansion",
            "expansion_exhaustion",
            "support_loss",
            "phase_break_fast",
            "thrust_drift_3",
            "exit_seed",
            "exit_persist",
            "hold_path_state",
        ]
    ].rename(
        columns={
            "expansion_hold_score": "rapid_expansion_hold_score",
            "expansion_thrust": "rapid_expansion_thrust",
            "directional_persistence": "rapid_directional_persistence",
            "acceptance_score": "rapid_acceptance_score",
            "instability_risk": "rapid_instability_risk",
            "constructive_expansion": "rapid_constructive_expansion",
            "expansion_exhaustion": "rapid_expansion_exhaustion",
            "support_loss": "rapid_support_loss",
            "phase_break_fast": "rapid_phase_break_fast",
            "thrust_drift_3": "rapid_thrust_drift_3",
            "exit_seed": "rapid_exit_seed",
            "exit_persist": "rapid_exit_persist",
            "hold_path_state": "rapid_hold_path_state",
        }
    )
    exhaustion_frame = exhaustion_eval[
        [
            "bar_end",
            "exhaustion_exit_score",
            "peak_extension_score",
            "deceleration_score",
            "fragility_score",
            "terminal_zone",
            "deceleration_state",
            "support_fracture",
            "fragility_state",
            "exit_seed",
            "exit_persist",
            "hold_path_state",
        ]
    ].rename(
        columns={
            "exhaustion_exit_score": "exhaustion_exit_score",
            "peak_extension_score": "exhaustion_peak_extension_score",
            "deceleration_score": "exhaustion_deceleration_score",
            "fragility_score": "exhaustion_fragility_score",
            "terminal_zone": "exhaustion_terminal_zone",
            "deceleration_state": "exhaustion_deceleration_state",
            "support_fracture": "exhaustion_support_fracture",
            "fragility_state": "exhaustion_fragility_state",
            "exit_seed": "exhaustion_exit_seed",
            "exit_persist": "exhaustion_exit_persist",
            "hold_path_state": "exhaustion_hold_path_state",
        }
    )

    path_frame = entropy_frame.merge(rapid_frame, on="bar_end", how="inner").merge(exhaustion_frame, on="bar_end", how="inner")
    if path_frame.empty:
        raise ValueError("Unable to align path frames across the three hold-state models")
    return path_frame.sort_values("bar_end").reset_index(drop=True)


def _classify_path_state(row: pd.Series, exit_confirmed_before: bool) -> str:
    if exit_confirmed_before:
        return "rapid_expansion_exhaustion_exit"

    exhaustion_active = _as_bool(row.get("exhaustion_exit_persist")) or _as_bool(row.get("exhaustion_exit_seed")) or (
        _as_bool(row.get("exhaustion_terminal_zone"))
        and _as_float(row.get("exhaustion_deceleration_score")) >= 0.55
        and _as_float(row.get("exhaustion_fragility_score")) >= 0.48
    )
    rapid_active = (
        _as_bool(row.get("rapid_hold_path_state"))
        and _as_bool(row.get("rapid_constructive_expansion"))
        and _as_float(row.get("rapid_expansion_hold_score")) >= 0.55
        and _as_float(row.get("rapid_instability_risk"), 1.0) < 0.65
    )
    entropy_active = (
        _as_bool(row.get("entropy_hold_path_state"))
        and not _as_bool(row.get("entropy_exit_persist"))
        and _as_float(row.get("entropy_hold_score")) >= 0.45
        and _as_float(row.get("entropy_entropy_reserve")) >= 0.20
    )

    if exhaustion_active:
        return "rapid_expansion_exhaustion_exit"
    if rapid_active:
        return "rapid_expansion_hold"
    if entropy_active:
        return "entropy_hold_judgement"
    return "observation"


def _build_path_trace(path_frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    trace_rows: list[dict[str, Any]] = []
    transitions: list[dict[str, str]] = []
    previous_state_id = ""
    exit_confirmed = False

    for _, row in path_frame.iterrows():
        state_id = _classify_path_state(row, exit_confirmed)
        if state_id == "rapid_expansion_exhaustion_exit":
            exit_confirmed = True
        judgement, judgement_label = _path_judgement_label(state_id)
        trace_rows.append(
            {
                "bar_end": str(row.get("bar_end") or ""),
                "close": _as_float(row.get("path_close"), default=float("nan")),
                "path_state_id": state_id,
                "path_state_label": STATE_LABELS[state_id],
                "path_judgement": judgement,
                "path_judgement_label": judgement_label,
                "entropy_hold_score": _as_float(row.get("entropy_hold_score")),
                "entropy_reserve": _as_float(row.get("entropy_entropy_reserve")),
                "disorder_pressure": _as_float(row.get("entropy_disorder_pressure")),
                "rapid_hold_score": _as_float(row.get("rapid_expansion_hold_score")),
                "expansion_thrust": _as_float(row.get("rapid_expansion_thrust")),
                "directional_persistence": _as_float(row.get("rapid_directional_persistence")),
                "acceptance_score": _as_float(row.get("rapid_acceptance_score")),
                "instability_risk": _as_float(row.get("rapid_instability_risk")),
                "exhaustion_exit_score": _as_float(row.get("exhaustion_exit_score")),
                "peak_extension_score": _as_float(row.get("exhaustion_peak_extension_score")),
                "deceleration_score": _as_float(row.get("exhaustion_deceleration_score")),
                "fragility_score": _as_float(row.get("exhaustion_fragility_score")),
                "entropy_hold_path_state": _as_bool(row.get("entropy_hold_path_state")),
                "rapid_hold_path_state": _as_bool(row.get("rapid_hold_path_state")),
                "exhaustion_confirmed": _as_bool(row.get("exhaustion_exit_persist")) or state_id == "rapid_expansion_exhaustion_exit",
            }
        )
        if state_id != previous_state_id:
            transitions.append(
                {
                    "trade_date": str(row.get("bar_end") or ""),
                    "state_id": state_id,
                    "state_label": STATE_LABELS[state_id],
                }
            )
            previous_state_id = state_id

    return pd.DataFrame(trace_rows), transitions


def _transition_summary(transitions: list[dict[str, str]]) -> str:
    if not transitions:
        return ""
    return " -> ".join(f"{item['trade_date']} {item['state_label']}" for item in transitions)


def _prepare_frames(
    config: UptrendHoldStateFlowConfig,
) -> tuple[dict[str, str], pd.DataFrame, pd.DataFrame, list[dict[str, str]], str, str, float | None]:
    target = _resolve_target(config.symbol_or_name, config.data_dir, config.basic_path)
    df = pd.read_csv(target["file_path"])
    if df.empty or "trade_date" not in df.columns:
        raise ValueError("Daily file is empty or missing trade_date")

    start_date = normalize_trade_date(config.start_date, allow_empty=False)
    scan_date = _infer_scan_date(df, config.scan_date)
    if start_date > scan_date:
        raise ValueError("start_date cannot be later than scan_date")

    start_keep = f"{int(scan_date[:4]) - int(config.lookback_years)}0101"
    df = df[(df["trade_date"].astype(str) >= start_keep) & (df["trade_date"].astype(str) <= scan_date)].copy()
    if df.empty:
        raise ValueError("No rows available after applying lookback window")

    entropy_df = build_entropy_hold_feature_frame(df)
    rapid_df = build_rapid_expansion_feature_frame(df)
    exhaustion_df = build_rapid_expansion_exhaustion_feature_frame(df)
    if entropy_df.empty or rapid_df.empty or exhaustion_df.empty:
        raise ValueError("Unable to build grouped state features")

    entropy_eval = evaluate_hold_path(entropy_df, start_date, scan_date, 3)
    rapid_eval = evaluate_rapid_expansion_path(rapid_df, start_date, scan_date, 2)
    exhaustion_eval = evaluate_rapid_expansion_exhaustion_path(exhaustion_df, start_date, scan_date, 2)
    path_frame = _prepare_path_frame(entropy_eval, rapid_eval, exhaustion_eval)
    trace_df, transitions = _build_path_trace(path_frame)
    latest_close = _as_float(path_frame.iloc[-1].get("path_close"), default=float("nan")) if not path_frame.empty else None
    if latest_close is not None and pd.isna(latest_close):
        latest_close = None
    return target, path_frame, trace_df, transitions, start_date, scan_date, latest_close


def _build_state_rows(latest_row: pd.Series, trace_df: pd.DataFrame, current_state_id: str) -> list[dict[str, Any]]:
    activation_map: dict[str, dict[str, Any]] = {}
    if not trace_df.empty:
        for state_id, group in trace_df.groupby("path_state_id"):
            activation_map[str(state_id)] = {
                "activated_on_path": True,
                "first_entry_date": str(group.iloc[0]["bar_end"]),
                "last_entry_date": str(group.iloc[-1]["bar_end"]),
                "days_in_state": int(len(group)),
            }

    def activation_fields(state_id: str) -> dict[str, Any]:
        info = activation_map.get(state_id, {})
        return {
            "activated_on_path": bool(info.get("activated_on_path", False)),
            "first_entry_date": str(info.get("first_entry_date") or ""),
            "last_entry_date": str(info.get("last_entry_date") or ""),
            "days_in_state": int(info.get("days_in_state") or 0),
        }

    rows = [
        {
            "state_id": "observation",
            "state_label": STATE_LABELS["observation"],
            "state_score": 0.0,
            "active": current_state_id == "observation",
            "reason": "买点以来三套状态都没有形成持续优势时，路径暂时停留在观察区。",
            **activation_fields("observation"),
        },
        {
            "state_id": "entropy_hold_judgement",
            "state_label": STATE_LABELS["entropy_hold_judgement"],
            "state_score": _as_float(latest_row.get("entropy_hold_score")),
            "active": current_state_id == "entropy_hold_judgement",
            "reason": _state_reason("entropy_hold_judgement", latest_row),
            "entropy_reserve": _as_float(latest_row.get("entropy_entropy_reserve")),
            "disorder_pressure": _as_float(latest_row.get("entropy_disorder_pressure")),
            **activation_fields("entropy_hold_judgement"),
        },
        {
            "state_id": "rapid_expansion_hold",
            "state_label": STATE_LABELS["rapid_expansion_hold"],
            "state_score": _as_float(latest_row.get("rapid_expansion_hold_score")),
            "active": current_state_id == "rapid_expansion_hold",
            "reason": _state_reason("rapid_expansion_hold", latest_row),
            "expansion_thrust": _as_float(latest_row.get("rapid_expansion_thrust")),
            "directional_persistence": _as_float(latest_row.get("rapid_directional_persistence")),
            "acceptance_score": _as_float(latest_row.get("rapid_acceptance_score")),
            **activation_fields("rapid_expansion_hold"),
        },
        {
            "state_id": "rapid_expansion_exhaustion_exit",
            "state_label": STATE_LABELS["rapid_expansion_exhaustion_exit"],
            "state_score": _as_float(latest_row.get("exhaustion_exit_score")),
            "active": current_state_id == "rapid_expansion_exhaustion_exit",
            "reason": _state_reason("rapid_expansion_exhaustion_exit", latest_row),
            "peak_extension_score": _as_float(latest_row.get("exhaustion_peak_extension_score")),
            "deceleration_score": _as_float(latest_row.get("exhaustion_deceleration_score")),
            "fragility_score": _as_float(latest_row.get("exhaustion_fragility_score")),
            **activation_fields("rapid_expansion_exhaustion_exit"),
        },
    ]
    return rows


def run_uptrend_hold_state_flow_scan(config: UptrendHoldStateFlowConfig) -> list[str]:
    target, path_frame, trace_df, transitions, start_date, scan_date, latest_close = _prepare_frames(config)
    latest_row = path_frame.iloc[-1]
    current_state_id = str(trace_df.iloc[-1]["path_state_id"]) if not trace_df.empty else "observation"
    current_state_label = STATE_LABELS[current_state_id]
    first_exit_rows = trace_df[trace_df["path_state_id"].astype(str) == "rapid_expansion_exhaustion_exit"] if not trace_df.empty else pd.DataFrame()
    first_exit_date = str(first_exit_rows.iloc[0]["bar_end"]) if not first_exit_rows.empty else ""
    current_reason = _state_reason(current_state_id, latest_row)
    current_advice = _state_advice(current_state_id, first_exit_date)
    path_judgement, path_judgement_label = _path_judgement_label(current_state_id)
    state_rows = _build_state_rows(latest_row, trace_df, current_state_id)
    transition_summary = _transition_summary(transitions)

    strategy_score = next((row["state_score"] for row in state_rows if row["state_id"] == current_state_id), 0.0)
    entry_close = _as_float(path_frame.iloc[0].get("path_close"), default=float("nan")) if not path_frame.empty else None
    if entry_close is not None and pd.isna(entry_close):
        entry_close = None
    holding_return_pct = None
    if entry_close and latest_close and entry_close > 0:
        holding_return_pct = latest_close / entry_close - 1.0

    selected_row = {
        "symbol": target["symbol"],
        "ts_code": target["ts_code"],
        "name": target["name"],
        "area": target["area"],
        "industry": target["industry"],
        "market": target["market"],
        "start_date": str(start_date),
        "scan_date": str(scan_date),
        "strategy_name": str(config.strategy_name),
        "strategy_state": path_judgement == "continue_hold",
        "strategy_score": strategy_score,
        "path_judgement": path_judgement,
        "path_judgement_label": path_judgement_label,
        "current_state_id": current_state_id,
        "current_state_label": current_state_label,
        "current_state_reason": current_reason,
        "state_advice": current_advice,
        "first_exit_date": first_exit_date,
        "holding_days": int(len(path_frame)),
        "entry_close": entry_close,
        "latest_close": latest_close,
        "holding_return_pct": holding_return_pct,
        "path_transition_summary": transition_summary,
        "entropy_hold_score": _as_float(latest_row.get("entropy_hold_score")),
        "rapid_hold_score": _as_float(latest_row.get("rapid_expansion_hold_score")),
        "exhaustion_exit_score": _as_float(latest_row.get("exhaustion_exit_score")),
    }

    summary_row = {
        "strategy_name": str(config.strategy_name),
        "start_date": str(start_date),
        "scan_date": str(scan_date),
        "symbol": target["symbol"],
        "name": target["name"],
        "path_judgement": path_judgement,
        "path_judgement_label": path_judgement_label,
        "current_state_id": current_state_id,
        "current_state_label": current_state_label,
        "current_state_reason": current_reason,
        "state_advice": current_advice,
        "first_exit_date": first_exit_date,
        "holding_days": int(len(path_frame)),
        "holding_return_pct": holding_return_pct,
        "latest_close": latest_close,
        "state_score": strategy_score,
        "entropy_hold_score": selected_row["entropy_hold_score"],
        "rapid_hold_score": selected_row["rapid_hold_score"],
        "exhaustion_exit_score": selected_row["exhaustion_exit_score"],
        "path_transition_summary": transition_summary,
        "n_scanned": 1,
        "n_candidates": len(state_rows),
        "n_selected": 1,
    }

    diagnostics_rows = trace_df.astype(object).where(pd.notna(trace_df), None).to_dict(orient="records")

    return write_uptrend_hold_state_flow_outputs(
        out_dir=config.out_dir,
        strategy_name=str(config.strategy_name),
        scan_date=str(scan_date),
        market_rows=[selected_row],
        selected_rows=[selected_row],
        candidate_rows=state_rows,
        summary_row=summary_row,
        diagnostics_rows=diagnostics_rows,
    )