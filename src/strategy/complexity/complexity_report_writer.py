import os

import numpy as np
import pandas as pd


def write_complexity_outputs(
    out_dir: str,
    strategy_name: str,
    scan_date: str,
    top_n: int,
    market_rows: list[dict],
    candidate_rows: list[dict],
    selected_rows: list[dict],
    backtest_daily_rows: list[dict],
    backtest_trade_rows: list[dict],
    backtest_summary: dict | None,
) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)

    market_snapshot_path = os.path.join(out_dir, f"market_scan_snapshot_{strategy_name}_{scan_date}.csv")
    candidates_all_path = os.path.join(out_dir, f"{strategy_name}_candidates_{scan_date}_all.csv")
    candidates_top_path = os.path.join(out_dir, f"{strategy_name}_candidates_{scan_date}_top{top_n}.csv")
    selected_path = os.path.join(out_dir, f"selected_portfolio_{strategy_name}_{scan_date}_top{top_n}.csv")
    summary_path = os.path.join(out_dir, f"strategy_summary_{strategy_name}_{scan_date}.csv")
    backtest_daily_path = os.path.join(out_dir, f"forward_backtest_daily_{strategy_name}_{scan_date}.csv")
    backtest_trades_path = os.path.join(out_dir, f"forward_backtest_trades_{strategy_name}_{scan_date}.csv")
    backtest_summary_path = os.path.join(out_dir, f"forward_backtest_summary_{strategy_name}_{scan_date}.csv")

    df_market = pd.DataFrame(market_rows)
    if not df_market.empty:
        df_market = df_market.sort_values(
            ["strategy_state", "strategy_score", "energy_term", "amount"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)
    df_market.to_csv(market_snapshot_path, index=False)

    df_candidates = pd.DataFrame(candidate_rows)
    if not df_candidates.empty:
        df_candidates = df_candidates.sort_values(
            ["strategy_score", "score_pct_rank", "amount"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    df_candidates.to_csv(candidates_all_path, index=False)
    df_candidates.head(int(top_n)).to_csv(candidates_top_path, index=False)

    df_selected = pd.DataFrame(selected_rows)
    if not df_selected.empty:
        df_selected = df_selected.sort_values(["selected_rank"], ascending=[True]).reset_index(drop=True)
    df_selected.to_csv(selected_path, index=False)

    summary = pd.DataFrame(
        [
            {
                "strategy_name": str(strategy_name),
                "scan_date": str(scan_date),
                "n_scanned": int(df_market["symbol"].nunique()) if not df_market.empty else 0,
                "n_state_true": int(df_market["strategy_state"].sum()) if not df_market.empty else 0,
                "n_candidates": int(df_candidates["symbol"].nunique()) if not df_candidates.empty else 0,
                "n_selected": int(df_selected["symbol"].nunique()) if not df_selected.empty else 0,
                "n_selected_industries": int(df_selected["industry"].nunique()) if not df_selected.empty and "industry" in df_selected.columns else 0,
                "avg_strategy_score": float(df_candidates["strategy_score"].mean()) if not df_candidates.empty else np.nan,
                "avg_energy_term": float(df_candidates["energy_term"].mean()) if not df_candidates.empty else np.nan,
            }
        ]
    )
    summary.to_csv(summary_path, index=False)

    pd.DataFrame(backtest_daily_rows).to_csv(backtest_daily_path, index=False)

    df_backtest_trades = pd.DataFrame(backtest_trade_rows)
    if not df_backtest_trades.empty:
        df_backtest_trades = df_backtest_trades.sort_values(["scan_date", "symbol"], ascending=[True, True]).reset_index(drop=True)
    df_backtest_trades.to_csv(backtest_trades_path, index=False)
    pd.DataFrame([backtest_summary or {}]).to_csv(backtest_summary_path, index=False)

    return [
        market_snapshot_path,
        candidates_all_path,
        candidates_top_path,
        selected_path,
        summary_path,
        backtest_daily_path,
        backtest_trades_path,
        backtest_summary_path,
    ]