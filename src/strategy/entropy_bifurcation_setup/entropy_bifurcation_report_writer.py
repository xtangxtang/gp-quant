import os

import numpy as np
import pandas as pd


def write_entropy_bifurcation_outputs(
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
            ["strategy_state", "context_score", "strategy_score", "bifurcation_quality", "amount"],
            ascending=[False, False, False, False, False],
        ).reset_index(drop=True)
    df_market.to_csv(market_snapshot_path, index=False)

    df_candidates = pd.DataFrame(candidate_rows)
    if not df_candidates.empty:
        df_candidates = df_candidates.sort_values(
            ["context_score", "strategy_score", "bifurcation_quality", "path_irreversibility_20", "amount"],
            ascending=[False, False, False, False, False],
        ).reset_index(drop=True)
    df_candidates.to_csv(candidates_all_path, index=False)
    df_candidates.head(int(top_n)).to_csv(candidates_top_path, index=False)

    df_selected = pd.DataFrame(selected_rows)
    if not df_selected.empty:
        df_selected = df_selected.sort_values(["selected_rank"], ascending=[True]).reset_index(drop=True)
    df_selected.to_csv(selected_path, index=False)

    context_source = df_candidates if not df_candidates.empty else df_market
    summary = pd.DataFrame(
        [
            {
                "strategy_name": str(strategy_name),
                "scan_date": str(scan_date),
                "n_scanned": int(df_market["symbol"].nunique()) if not df_market.empty else 0,
                "n_state_true": int(df_market["strategy_state"].sum()) if not df_market.empty else 0,
                "n_candidates": int(df_candidates["symbol"].nunique()) if not df_candidates.empty else 0,
                "n_selected": int(df_selected["symbol"].nunique()) if not df_selected.empty else 0,
                "n_abandoned": int(df_market["strategic_abandonment"].sum()) if not df_market.empty and "strategic_abandonment" in df_market.columns else 0,
                "market_phase_state": str(context_source["market_phase_state"].iloc[0]) if not context_source.empty and "market_phase_state" in context_source.columns else "",
                "market_regime_score": float(context_source["market_regime_score"].iloc[0]) if not context_source.empty and "market_regime_score" in context_source.columns else np.nan,
                "market_coupling_entropy_20": float(context_source["market_coupling_entropy_20"].iloc[0]) if not context_source.empty and "market_coupling_entropy_20" in context_source.columns else np.nan,
                "market_phase_distortion_share": float(context_source["market_phase_distortion_share"].iloc[0]) if not context_source.empty and "market_phase_distortion_share" in context_source.columns else np.nan,
                "market_noise_cost": float(context_source["market_noise_cost"].iloc[0]) if not context_source.empty and "market_noise_cost" in context_source.columns else np.nan,
                "avg_context_score": float(context_source["context_score"].mean()) if not context_source.empty and "context_score" in context_source.columns else np.nan,
                "avg_stock_state_score": float(df_candidates["stock_state_score"].mean()) if not df_candidates.empty and "stock_state_score" in df_candidates.columns else np.nan,
                "avg_execution_readiness_score": float(df_candidates["execution_readiness_score"].mean()) if not df_candidates.empty and "execution_readiness_score" in df_candidates.columns else np.nan,
                "avg_execution_penalty_score": float(df_candidates["execution_penalty_score"].mean()) if not df_candidates.empty and "execution_penalty_score" in df_candidates.columns else np.nan,
                "avg_abandonment_score": float(context_source["abandonment_score"].mean()) if not context_source.empty and "abandonment_score" in context_source.columns else np.nan,
                "avg_position_scale": float(df_selected["position_scale"].mean()) if not df_selected.empty and "position_scale" in df_selected.columns else np.nan,
                "avg_experimental_model_score": float(context_source["experimental_model_score"].mean()) if not context_source.empty and "experimental_model_score" in context_source.columns else np.nan,
                "avg_strategy_score": float(df_candidates["strategy_score"].mean()) if not df_candidates.empty else np.nan,
                "avg_entropy_quality": float(df_candidates["entropy_quality"].mean()) if not df_candidates.empty else np.nan,
                "avg_bifurcation_quality": float(df_candidates["bifurcation_quality"].mean()) if not df_candidates.empty else np.nan,
                "avg_path_irreversibility_20": float(df_candidates["path_irreversibility_20"].mean()) if not df_candidates.empty and "path_irreversibility_20" in df_candidates.columns else np.nan,
                "avg_execution_cost_proxy_20": float(df_candidates["execution_cost_proxy_20"].mean()) if not df_candidates.empty and "execution_cost_proxy_20" in df_candidates.columns else np.nan,
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