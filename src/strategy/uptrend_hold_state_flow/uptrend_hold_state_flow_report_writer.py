import os

import pandas as pd

from .io_utils import safe_file_component


def write_uptrend_hold_state_flow_outputs(
    out_dir: str,
    strategy_name: str,
    scan_date: str,
    market_rows: list[dict],
    selected_rows: list[dict],
    candidate_rows: list[dict],
    summary_row: dict,
    diagnostics_rows: list[dict] | None = None,
) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)

    safe_strategy_name = safe_file_component(strategy_name)
    safe_scan_date = safe_file_component(scan_date)

    market_snapshot_path = os.path.join(out_dir, f"market_scan_snapshot_{safe_strategy_name}_{safe_scan_date}.csv")
    candidates_all_path = os.path.join(out_dir, f"{safe_strategy_name}_candidates_{safe_scan_date}_all.csv")
    selected_path = os.path.join(out_dir, f"selected_portfolio_{safe_strategy_name}_{safe_scan_date}_top1.csv")
    summary_path = os.path.join(out_dir, f"strategy_summary_{safe_strategy_name}_{safe_scan_date}.csv")
    diagnostics_path = os.path.join(out_dir, f"state_path_diagnostics_{safe_strategy_name}_{safe_scan_date}.csv")
    backtest_daily_path = os.path.join(out_dir, f"forward_backtest_daily_{safe_strategy_name}_{safe_scan_date}.csv")
    backtest_trades_path = os.path.join(out_dir, f"forward_backtest_trades_{safe_strategy_name}_{safe_scan_date}.csv")
    backtest_summary_path = os.path.join(out_dir, f"forward_backtest_summary_{safe_strategy_name}_{safe_scan_date}.csv")

    pd.DataFrame(market_rows).to_csv(market_snapshot_path, index=False)
    pd.DataFrame(candidate_rows).to_csv(candidates_all_path, index=False)
    pd.DataFrame(selected_rows).to_csv(selected_path, index=False)
    pd.DataFrame([summary_row]).to_csv(summary_path, index=False)
    pd.DataFrame(diagnostics_rows or []).to_csv(diagnostics_path, index=False)
    pd.DataFrame().to_csv(backtest_daily_path, index=False)
    pd.DataFrame().to_csv(backtest_trades_path, index=False)
    pd.DataFrame([summary_row]).to_csv(backtest_summary_path, index=False)

    return [
        market_snapshot_path,
        candidates_all_path,
        selected_path,
        summary_path,
        diagnostics_path,
        backtest_daily_path,
        backtest_trades_path,
        backtest_summary_path,
    ]