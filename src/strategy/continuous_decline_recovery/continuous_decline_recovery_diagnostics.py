import os
from typing import Any

import numpy as np
import pandas as pd

from .continuous_decline_recovery_scan_service import (
    ContinuousDeclineRecoveryConfig,
    _as_float,
    _build_market_aggregate_cache,
    _evaluate_scan_date,
    _infer_scan_date,
    _load_basic_info_map,
    _master_trade_dates,
    _prepare_symbol_state,
    _resolve_files,
    _resolve_scan_date_from_master,
    _trade_plan_from_snapshot,
)

def _setup_failures(row: dict[str, Any], config: ContinuousDeclineRecoveryConfig) -> list[str]:
    failures: list[str] = []
    if not bool(row.get("tradable_ok")):
        failures.append("tradability_gate")
    damage_threshold = _as_float(row.get("damage_threshold"), 0.32)
    sector_score_threshold = _as_float(row.get("sector_score_threshold"), 0.48)
    if _as_float(row.get("damage_score"), 0.0) < damage_threshold:
        failures.append("damage_gate")
    if _as_float(row.get("repair_score"), 0.0) < 0.48:
        failures.append("repair_gate")
    if _as_float(row.get("entry_window_score"), 0.0) < 0.40:
        failures.append("entry_window_gate")
    if _as_float(row.get("flow_support_score"), 0.0) < 0.35:
        failures.append("flow_support_gate")

    rebound_from_low = _as_float(row.get("rebound_from_low_10"), 0.0)
    if rebound_from_low < float(config.min_rebound_from_low):
        failures.append("rebound_too_early_gate")
    if rebound_from_low > float(config.max_rebound_from_low):
        failures.append("rebound_too_late_gate")

    if _as_float(row.get("sector_score"), 0.0) < sector_score_threshold:
        failures.append("sector_score_gate")
    if int(row.get("sector_rank", 999) or 999) > int(config.top_sectors):
        failures.append("sector_rank_gate")
    return failures


def _primary_near_miss_blocker(row: dict[str, Any], config: ContinuousDeclineRecoveryConfig) -> str:
    failures = _setup_failures(row, config)
    if not failures:
        return "unknown_setup_gate"
    if len(failures) == 1:
        return failures[0]
    return "multi_setup_gates"


def _prepare_runtime_context(
    config: ContinuousDeclineRecoveryConfig,
) -> tuple[list[Any], dict[str, Any], np.ndarray, pd.DataFrame, ContinuousDeclineRecoveryConfig, str]:
    files = _resolve_files(config.data_dir, config.symbols)
    if not files:
        raise SystemExit(f"No CSV files found under {config.data_dir}")
    if bool(config.backtest_start_date) != bool(config.backtest_end_date):
        raise SystemExit("backtest_start_date and backtest_end_date must be provided together.")

    requested_scan_date = _infer_scan_date(files, config.scan_date)
    latest_available_date = _infer_scan_date(files, "")
    history_anchor_date = str(config.backtest_start_date or requested_scan_date)
    min_keep_date = f"{int(history_anchor_date[:4]) - int(config.lookback_years)}0101"
    max_needed_date = latest_available_date if config.backtest_end_date else requested_scan_date

    basic_info_map = _load_basic_info_map(config.basic_path)
    prepared_symbols = []
    for file_path in files:
        prepared = _prepare_symbol_state(file_path, min_keep_date, max_needed_date, basic_info_map)
        if prepared is not None:
            prepared_symbols.append(prepared)
    if not prepared_symbols:
        raise SystemExit("Unable to prepare any symbols for continuous_decline_recovery diagnostics.")

    master_dates = _master_trade_dates(prepared_symbols)
    market_cache = _build_market_aggregate_cache(prepared_symbols, master_dates, config)
    scan_date = _resolve_scan_date_from_master(master_dates, requested_scan_date)
    resolved_backtest_start = _resolve_scan_date_from_master(master_dates, config.backtest_start_date) if config.backtest_start_date else ""
    resolved_backtest_end = _resolve_scan_date_from_master(master_dates, config.backtest_end_date) if config.backtest_end_date else ""
    runtime_config = ContinuousDeclineRecoveryConfig(
        **{
            **vars(config),
            "scan_date": scan_date,
            "backtest_start_date": resolved_backtest_start,
            "backtest_end_date": resolved_backtest_end,
        }
    )
    prepared_lookup = {prepared.symbol: prepared for prepared in prepared_symbols}
    return prepared_symbols, prepared_lookup, master_dates, market_cache, runtime_config, scan_date


def _aggregate_trade_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                group_col,
                "n_trades",
                "win_rate",
                "avg_return_pct",
                "median_return_pct",
                "avg_max_runup_pct",
                "avg_max_drawdown_pct",
                "avg_strategy_score",
                "avg_market_buy_score",
                "avg_repair_score",
                "avg_entry_window_score",
                "avg_flow_support_score",
                "avg_position_scale",
            ]
        )
    grouped = (
        df.groupby(group_col, dropna=False)
        .agg(
            n_trades=("symbol", "count"),
            win_rate=("return_pct", lambda series: float((series > 0.0).mean())),
            avg_return_pct=("return_pct", "mean"),
            median_return_pct=("return_pct", "median"),
            avg_max_runup_pct=("max_runup_pct", "mean"),
            avg_max_drawdown_pct=("max_drawdown_pct", "mean"),
            avg_strategy_score=("strategy_score", "mean"),
            avg_market_buy_score=("market_buy_score", "mean"),
            avg_repair_score=("repair_score", "mean"),
            avg_entry_window_score=("entry_window_score", "mean"),
            avg_flow_support_score=("flow_support_score", "mean"),
            avg_position_scale=("position_scale", "mean"),
        )
        .reset_index()
        .sort_values(["n_trades", "avg_return_pct"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return grouped


def _build_trade_feature_profile(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "outcome_bucket",
        "n_trades",
        "avg_return_pct",
        "median_return_pct",
        "avg_market_buy_score",
        "avg_strategy_score",
        "avg_sector_score",
        "avg_damage_score",
        "avg_repair_score",
        "avg_entry_window_score",
        "avg_flow_support_score",
        "avg_stability_score",
        "avg_rebound_from_low_10",
        "avg_amount_ratio_20",
        "avg_flow_ratio_5",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    frames = []
    buckets = {
        "all": df,
        "winner": df[df["return_pct"] > 0.0],
        "loser": df[df["return_pct"] <= 0.0],
    }
    for bucket, frame in buckets.items():
        if frame.empty:
            continue
        frames.append(
            {
                "outcome_bucket": bucket,
                "n_trades": int(len(frame)),
                "avg_return_pct": float(frame["return_pct"].mean()),
                "median_return_pct": float(frame["return_pct"].median()),
                "avg_market_buy_score": float(frame["market_buy_score"].mean()),
                "avg_strategy_score": float(frame["strategy_score"].mean()),
                "avg_sector_score": float(frame["sector_score"].mean()),
                "avg_damage_score": float(frame["damage_score"].mean()),
                "avg_repair_score": float(frame["repair_score"].mean()),
                "avg_entry_window_score": float(frame["entry_window_score"].mean()),
                "avg_flow_support_score": float(frame["flow_support_score"].mean()),
                "avg_stability_score": float(frame["stability_score"].mean()),
                "avg_rebound_from_low_10": float(frame["rebound_from_low_10"].mean()),
                "avg_amount_ratio_20": float(frame["amount_ratio_20"].mean()),
                "avg_flow_ratio_5": float(frame["flow_ratio_5"].mean()),
            }
        )
    return pd.DataFrame(frames, columns=columns)


def _build_false_negative_threshold_summary(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "blocker",
        "n_rows",
        "n_positive_return",
        "n_strong_positive_return",
        "positive_hit_rate",
        "strong_positive_hit_rate",
        "avg_return_pct",
        "median_return_pct",
        "avg_strategy_score",
        "avg_market_buy_score",
        "avg_repair_score",
        "avg_entry_window_score",
        "avg_flow_support_score",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    grouped = (
        df.groupby("blocker", dropna=False)
        .agg(
            n_rows=("symbol", "count"),
            n_positive_return=("positive_return_flag", "sum"),
            n_strong_positive_return=("strong_positive_return_flag", "sum"),
            avg_return_pct=("return_pct", "mean"),
            median_return_pct=("return_pct", "median"),
            avg_strategy_score=("strategy_score", "mean"),
            avg_market_buy_score=("market_buy_score", "mean"),
            avg_repair_score=("repair_score", "mean"),
            avg_entry_window_score=("entry_window_score", "mean"),
            avg_flow_support_score=("flow_support_score", "mean"),
        )
        .reset_index()
    )
    grouped["positive_hit_rate"] = grouped["n_positive_return"] / grouped["n_rows"].clip(lower=1)
    grouped["strong_positive_hit_rate"] = grouped["n_strong_positive_return"] / grouped["n_rows"].clip(lower=1)
    return grouped[columns].sort_values(["n_positive_return", "avg_return_pct"], ascending=[False, False]).reset_index(drop=True)


def _build_fail_flag_summary(df: pd.DataFrame) -> pd.DataFrame:
    fail_columns = [
        "fail_market_gate",
        "fail_score_gate",
        "fail_tradability_gate",
        "fail_damage_gate",
        "fail_repair_gate",
        "fail_entry_window_gate",
        "fail_flow_support_gate",
        "fail_rebound_too_early_gate",
        "fail_rebound_too_late_gate",
        "fail_sector_score_gate",
        "fail_sector_rank_gate",
        "fail_portfolio_capacity",
        "fail_industry_capacity",
    ]
    rows: list[dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame(
            columns=[
                "fail_flag",
                "n_rows",
                "n_positive_return",
                "n_strong_positive_return",
                "positive_hit_rate",
                "strong_positive_hit_rate",
                "avg_return_pct",
                "median_return_pct",
                "avg_strategy_score",
                "avg_market_buy_score",
            ]
        )

    for column in fail_columns:
        subset = df[df[column] == True].copy()
        if subset.empty:
            continue
        rows.append(
            {
                "fail_flag": str(column).replace("fail_", ""),
                "n_rows": int(len(subset)),
                "n_positive_return": int(subset["positive_return_flag"].sum()),
                "n_strong_positive_return": int(subset["strong_positive_return_flag"].sum()),
                "positive_hit_rate": float(subset["positive_return_flag"].mean()),
                "strong_positive_hit_rate": float(subset["strong_positive_return_flag"].mean()),
                "avg_return_pct": float(subset["return_pct"].mean()),
                "median_return_pct": float(subset["return_pct"].median()),
                "avg_strategy_score": float(subset["strategy_score"].mean()),
                "avg_market_buy_score": float(subset["market_buy_score"].mean()),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "fail_flag",
                "n_rows",
                "n_positive_return",
                "n_strong_positive_return",
                "positive_hit_rate",
                "strong_positive_hit_rate",
                "avg_return_pct",
                "median_return_pct",
                "avg_strategy_score",
                "avg_market_buy_score",
            ]
        )
    return pd.DataFrame(rows).sort_values(["n_positive_return", "avg_return_pct"], ascending=[False, False]).reset_index(drop=True)


def run_continuous_decline_recovery_diagnostics(
    config: ContinuousDeclineRecoveryConfig,
    out_dir: str,
    min_near_miss_strategy_score: float = 0.45,
    top_sample_count: int = 100,
) -> list[str]:
    prepared_symbols, prepared_lookup, master_dates, market_cache, runtime_config, scan_date = _prepare_runtime_context(config)
    start_pos = int(np.searchsorted(master_dates, str(runtime_config.backtest_start_date), side="left"))
    end_pos = int(np.searchsorted(master_dates, str(runtime_config.backtest_end_date), side="right") - 1)
    if start_pos >= len(master_dates) or end_pos < 0 or start_pos > end_pos:
        raise SystemExit("Invalid backtest range for diagnostics.")

    scan_dates = [str(value) for value in master_dates[start_pos : end_pos + 1]]
    executed_rows: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []
    active_positions: list[dict[str, Any]] = []

    for current_date in scan_dates:
        active_positions = [position for position in active_positions if str(position.get("exit_date") or "") >= str(current_date)]
        market_rows, _, candidate_rows, selected_rows, summary = _evaluate_scan_date(
            prepared_symbols,
            master_dates,
            market_cache,
            current_date,
            runtime_config,
        )
        accepted_selected_map: dict[str, dict[str, Any]] = {}
        blocked_selected_map: dict[str, str] = {}

        for selected_row in selected_rows:
            prepared = prepared_lookup.get(str(selected_row.get("symbol") or ""))
            if prepared is None:
                continue
            trade_plan = _trade_plan_from_snapshot(prepared, selected_row, current_date, int(runtime_config.hold_days))
            if trade_plan is None:
                continue
            entry_date = str(trade_plan.get("entry_date") or "")
            active_positions = [position for position in active_positions if str(position.get("exit_date") or "") >= entry_date]
            symbol = str(selected_row.get("symbol") or "")
            industry = str(selected_row.get("industry") or "UNKNOWN")

            if any(str(position.get("symbol") or "") == symbol for position in active_positions):
                blocked_selected_map[symbol] = "duplicate_open_position"
                continue
            if len(active_positions) >= int(runtime_config.max_positions):
                blocked_selected_map[symbol] = "portfolio_capacity"
                continue
            industry_open_count = sum(1 for position in active_positions if str(position.get("industry") or "UNKNOWN") == industry)
            if int(runtime_config.max_positions_per_industry) > 0 and industry_open_count >= int(runtime_config.max_positions_per_industry):
                blocked_selected_map[symbol] = "industry_capacity"
                continue

            accepted_selected_map[symbol] = {**selected_row, **trade_plan}
            executed_rows.append(
                {
                    **selected_row,
                    **trade_plan,
                    "outcome_bucket": "winner" if _as_float(trade_plan.get("return_pct"), 0.0) > 0.0 else "loser",
                }
            )
            active_positions.append(
                {
                    "symbol": symbol,
                    "industry": industry,
                    "entry_date": entry_date,
                    "exit_date": str(trade_plan.get("exit_date") or ""),
                }
            )

        for row in market_rows:
            symbol = str(row.get("symbol") or "")
            if symbol in accepted_selected_map:
                continue
            if not bool(row.get("tradable_ok")):
                continue
            strategy_score = _as_float(row.get("strategy_score"), 0.0)
            if not bool(row.get("candidate_flag")) and strategy_score < float(min_near_miss_strategy_score):
                continue

            prepared = prepared_lookup.get(symbol)
            if prepared is None:
                continue
            trade_plan = _trade_plan_from_snapshot(prepared, row, current_date, int(runtime_config.hold_days))
            if trade_plan is None:
                continue

            market_state = str(row.get("market_buy_state") or "")
            market_allows_trade = bool(row.get("market_allows_trade"))
            threshold = _as_float(row.get("strategy_score_threshold"), 0.50)
            blocker = ""
            blocker_scope = ""
            setup_failures = set(_setup_failures(row, runtime_config))
            fail_market_gate = False
            fail_score_gate = False
            fail_portfolio_capacity = False
            fail_industry_capacity = False

            if bool(row.get("candidate_flag")):
                if not market_allows_trade:
                    blocker = f"market_gate::{market_state}"
                    blocker_scope = "market_gate"
                    fail_market_gate = True
                elif strategy_score < threshold:
                    blocker = f"score_gate::{market_state}"
                    blocker_scope = "score_gate"
                    fail_score_gate = True
                elif blocked_selected_map.get(symbol) == "portfolio_capacity":
                    blocker = "portfolio_capacity"
                    blocker_scope = "portfolio_capacity"
                    fail_portfolio_capacity = True
                elif blocked_selected_map.get(symbol) == "industry_capacity":
                    blocker = "industry_capacity"
                    blocker_scope = "industry_capacity"
                    fail_industry_capacity = True
                elif blocked_selected_map.get(symbol) == "duplicate_open_position":
                    blocker = "duplicate_open_position"
                    blocker_scope = "portfolio_capacity"
                else:
                    blocker = "unclassified_candidate_block"
                    blocker_scope = "other"
            else:
                blocker = _primary_near_miss_blocker(row, runtime_config)
                blocker_scope = "setup_gate"

            blocked_rows.append(
                {
                    **row,
                    **trade_plan,
                    "blocker": blocker,
                    "blocker_scope": blocker_scope,
                    "positive_return_flag": bool(_as_float(trade_plan.get("return_pct"), 0.0) > 0.0),
                    "strong_positive_return_flag": bool(_as_float(trade_plan.get("return_pct"), 0.0) >= 5.0),
                    "strategy_threshold": threshold,
                    "score_gap_to_threshold": float(strategy_score - threshold),
                    "fail_market_gate": bool(fail_market_gate),
                    "fail_score_gate": bool(fail_score_gate),
                    "fail_tradability_gate": bool("tradability_gate" in setup_failures),
                    "fail_damage_gate": bool("damage_gate" in setup_failures),
                    "fail_repair_gate": bool("repair_gate" in setup_failures),
                    "fail_entry_window_gate": bool("entry_window_gate" in setup_failures),
                    "fail_flow_support_gate": bool("flow_support_gate" in setup_failures),
                    "fail_rebound_too_early_gate": bool("rebound_too_early_gate" in setup_failures),
                    "fail_rebound_too_late_gate": bool("rebound_too_late_gate" in setup_failures),
                    "fail_sector_score_gate": bool("sector_score_gate" in setup_failures),
                    "fail_sector_rank_gate": bool("sector_rank_gate" in setup_failures),
                    "fail_portfolio_capacity": bool(fail_portfolio_capacity),
                    "fail_industry_capacity": bool(fail_industry_capacity),
                }
            )

    os.makedirs(out_dir, exist_ok=True)
    trade_df = pd.DataFrame(executed_rows)
    blocked_df = pd.DataFrame(blocked_rows)

    by_market_state_path = os.path.join(out_dir, f"signal_diagnostic_trades_by_market_state_{runtime_config.strategy_name}_{scan_date}.csv")
    by_industry_path = os.path.join(out_dir, f"signal_diagnostic_trades_by_industry_{runtime_config.strategy_name}_{scan_date}.csv")
    feature_profile_path = os.path.join(out_dir, f"signal_diagnostic_trade_feature_profile_{runtime_config.strategy_name}_{scan_date}.csv")
    threshold_summary_path = os.path.join(out_dir, f"signal_diagnostic_false_negative_thresholds_{runtime_config.strategy_name}_{scan_date}.csv")
    fail_flag_summary_path = os.path.join(out_dir, f"signal_diagnostic_fail_flag_summary_{runtime_config.strategy_name}_{scan_date}.csv")
    false_negative_samples_path = os.path.join(out_dir, f"signal_diagnostic_false_negative_samples_{runtime_config.strategy_name}_{scan_date}.csv")

    _aggregate_trade_group(trade_df, "market_buy_state").to_csv(by_market_state_path, index=False)
    _aggregate_trade_group(trade_df, "industry").to_csv(by_industry_path, index=False)
    _build_trade_feature_profile(trade_df).to_csv(feature_profile_path, index=False)
    _build_false_negative_threshold_summary(blocked_df).to_csv(threshold_summary_path, index=False)
    _build_fail_flag_summary(blocked_df).to_csv(fail_flag_summary_path, index=False)

    if not blocked_df.empty:
        blocked_df = blocked_df.sort_values(["return_pct", "strategy_score"], ascending=[False, False]).reset_index(drop=True)
        blocked_df.head(int(top_sample_count)).to_csv(false_negative_samples_path, index=False)
    else:
        pd.DataFrame(
            columns=[
                "scan_date",
                "symbol",
                "industry",
                "market_buy_state",
                "blocker",
                "return_pct",
                "strategy_score",
                "repair_score",
                "entry_window_score",
                "flow_support_score",
            ]
        ).to_csv(false_negative_samples_path, index=False)

    return [
        by_market_state_path,
        by_industry_path,
        feature_profile_path,
        threshold_summary_path,
        fail_flag_summary_path,
        false_negative_samples_path,
    ]