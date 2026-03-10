import os

import numpy as np
import pandas as pd


def write_scan_outputs(
    out_dir: str,
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

    market_snapshot_path = os.path.join(out_dir, f"market_scan_snapshot_{scan_date}.csv")
    resonance_candidates_all_path = os.path.join(out_dir, f"resonance_candidates_{scan_date}_all.csv")
    resonance_candidates_top_path = os.path.join(out_dir, f"resonance_candidates_{scan_date}_top{top_n}.csv")
    selected_portfolio_path = os.path.join(out_dir, f"selected_portfolio_{scan_date}_top{top_n}.csv")
    resonance_summary_path = os.path.join(out_dir, f"resonance_summary_{scan_date}.csv")
    backtest_daily_path = os.path.join(out_dir, f"forward_backtest_daily_{scan_date}.csv")
    backtest_trades_path = os.path.join(out_dir, f"forward_backtest_trades_{scan_date}.csv")
    backtest_summary_path = os.path.join(out_dir, f"forward_backtest_summary_{scan_date}.csv")

    df_market = pd.DataFrame(market_rows)
    if not df_market.empty:
        df_market = df_market.sort_values(
            ["resonance_state", "resonance_score", "support_count", "daily_score"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)
    df_market.to_csv(market_snapshot_path, index=False)

    df_candidates = pd.DataFrame(candidate_rows)
    if not df_candidates.empty:
        df_candidates = df_candidates.sort_values(
            ["resonance_score", "support_count", "daily_score"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    df_candidates.to_csv(resonance_candidates_all_path, index=False)
    df_candidates.head(int(top_n)).to_csv(resonance_candidates_top_path, index=False)

    df_selected = pd.DataFrame(selected_rows)
    if not df_selected.empty:
        df_selected = df_selected.sort_values(["selected_rank"], ascending=[True]).reset_index(drop=True)
    df_selected.to_csv(selected_portfolio_path, index=False)

    summary = pd.DataFrame(
        [
            {
                "scan_date": str(scan_date),
                "n_scanned": int(df_market["symbol"].nunique()) if not df_market.empty else 0,
                "n_daily_state": int(df_market["daily_state"].sum()) if not df_market.empty else 0,
                "n_weekly_state": int(df_market["weekly_state"].sum()) if not df_market.empty else 0,
                "n_monthly_state": int(df_market["monthly_state"].sum()) if not df_market.empty else 0,
                "n_resonance_candidates": int(df_candidates["symbol"].nunique()) if not df_candidates.empty else 0,
                "n_selected": int(df_selected["symbol"].nunique()) if not df_selected.empty else 0,
                "n_selected_industries": int(df_selected["industry"].nunique()) if not df_selected.empty and "industry" in df_selected.columns else 0,
                "avg_resonance_score": float(df_candidates["resonance_score"].mean()) if not df_candidates.empty else np.nan,
                "avg_support_count": float(df_candidates["support_count"].mean()) if not df_candidates.empty else np.nan,
                "n_industries": int(df_candidates["industry"].nunique()) if not df_candidates.empty and "industry" in df_candidates.columns else 0,
            }
        ]
    )
    summary.to_csv(resonance_summary_path, index=False)

    df_backtest_daily = pd.DataFrame(backtest_daily_rows)
    df_backtest_daily.to_csv(backtest_daily_path, index=False)

    df_backtest_trades = pd.DataFrame(backtest_trade_rows)
    if not df_backtest_trades.empty:
        df_backtest_trades = df_backtest_trades.sort_values(["scan_date", "symbol"], ascending=[True, True]).reset_index(drop=True)
    df_backtest_trades.to_csv(backtest_trades_path, index=False)

    pd.DataFrame([backtest_summary or {}]).to_csv(backtest_summary_path, index=False)

    return [
        market_snapshot_path,
        resonance_candidates_all_path,
        resonance_candidates_top_path,
        selected_portfolio_path,
        resonance_summary_path,
        backtest_daily_path,
        backtest_trades_path,
        backtest_summary_path,
    ]