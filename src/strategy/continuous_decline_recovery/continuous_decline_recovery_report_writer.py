import os

import numpy as np
import pandas as pd


def _empty_backtest_performance_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "period_label",
            "start_date",
            "end_date",
            "calendar_days",
            "n_eval_days",
            "n_scan_days",
            "n_days_with_signal",
            "n_trades_opened",
            "n_trades_closed",
            "win_rate",
            "avg_trade_return_pct",
            "median_trade_return_pct",
            "avg_max_runup_pct",
            "avg_max_drawdown_pct",
            "avg_gross_exposure",
            "end_nav",
            "interval_nav",
            "total_return_pct",
            "annualized_return_pct",
            "max_drawdown_pct",
            "buy_window_days",
            "repair_watch_days",
            "selloff_days",
            "rebound_crowded_days",
        ]
    )


def _prepare_backtest_frames(
    backtest_daily_rows: list[dict],
    backtest_trade_rows: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_daily = pd.DataFrame(backtest_daily_rows)
    if not df_daily.empty:
        df_daily = df_daily.copy()
        df_daily["scan_date"] = df_daily["scan_date"].astype(str)
        df_daily = df_daily.sort_values(["scan_date"], ascending=[True]).reset_index(drop=True)
        df_daily["nav"] = pd.to_numeric(df_daily["nav"], errors="coerce")
        df_daily["gross_exposure"] = pd.to_numeric(df_daily.get("gross_exposure"), errors="coerce")
        df_daily["n_selected"] = pd.to_numeric(df_daily.get("n_selected"), errors="coerce").fillna(0)
        df_daily["is_scan_day"] = df_daily.get("is_scan_day", False).fillna(False).astype(bool)
        df_daily["dt"] = pd.to_datetime(df_daily["scan_date"], format="%Y%m%d", errors="coerce")

    df_trades = pd.DataFrame(backtest_trade_rows)
    if not df_trades.empty:
        df_trades = df_trades.copy()
        for column in ["scan_date", "entry_date", "exit_date"]:
            df_trades[column] = df_trades[column].astype(str)
        for column in ["return_pct", "max_runup_pct", "max_drawdown_pct"]:
            df_trades[column] = pd.to_numeric(df_trades.get(column), errors="coerce")
        df_trades = df_trades.sort_values(["scan_date", "symbol"], ascending=[True, True]).reset_index(drop=True)
    return df_daily, df_trades


def _summarize_backtest_window(
    df_daily: pd.DataFrame,
    df_trades: pd.DataFrame,
    period_label: str,
    start_date: str,
    end_date: str,
) -> dict[str, object] | None:
    if df_daily.empty:
        return None

    window_mask = df_daily["scan_date"].between(str(start_date), str(end_date))
    window = df_daily.loc[window_mask].copy()
    if window.empty:
        return None

    first_index = int(window.index[0])
    base_nav = float(df_daily.loc[first_index - 1, "nav"]) if first_index > 0 else 1.0
    end_nav = float(window["nav"].iloc[-1]) if np.isfinite(window["nav"].iloc[-1]) else np.nan
    interval_nav = float(end_nav / base_nav) if base_nav > 0.0 and np.isfinite(end_nav) else np.nan

    normalized_nav = window["nav"].astype(float) / base_nav if base_nav > 0.0 else pd.Series(dtype=float)
    if not normalized_nav.empty:
        running_peak = normalized_nav.cummax()
        drawdown = normalized_nav / running_peak - 1.0
        max_drawdown_pct = float(drawdown.min() * 100.0)
    else:
        max_drawdown_pct = np.nan

    actual_start = str(window["scan_date"].iloc[0])
    actual_end = str(window["scan_date"].iloc[-1])
    start_dt = pd.to_datetime(actual_start, format="%Y%m%d", errors="coerce")
    end_dt = pd.to_datetime(actual_end, format="%Y%m%d", errors="coerce")
    calendar_days = int((end_dt - start_dt).days + 1) if pd.notna(start_dt) and pd.notna(end_dt) else int(len(window))
    annualized_return_pct = np.nan
    if np.isfinite(interval_nav) and interval_nav > 0.0 and calendar_days > 0:
        annualized_return_pct = float((interval_nav ** (365.0 / float(calendar_days)) - 1.0) * 100.0)

    opened_trades = df_trades[df_trades["scan_date"].between(actual_start, actual_end)].copy() if not df_trades.empty else pd.DataFrame()
    closed_trades = df_trades[df_trades["exit_date"].between(actual_start, actual_end)].copy() if not df_trades.empty else pd.DataFrame()

    if not closed_trades.empty:
        positive_sum = float(closed_trades.loc[closed_trades["return_pct"] > 0.0, "return_pct"].sum())
        negative_sum = float(closed_trades.loc[closed_trades["return_pct"] < 0.0, "return_pct"].sum())
        win_rate = float((closed_trades["return_pct"] > 0.0).mean())
        avg_trade_return_pct = float(closed_trades["return_pct"].mean())
        median_trade_return_pct = float(closed_trades["return_pct"].median())
        avg_max_runup_pct = float(closed_trades["max_runup_pct"].mean())
        avg_max_drawdown_pct = float(closed_trades["max_drawdown_pct"].mean())
        _ = positive_sum, negative_sum
    else:
        win_rate = np.nan
        avg_trade_return_pct = np.nan
        median_trade_return_pct = np.nan
        avg_max_runup_pct = np.nan
        avg_max_drawdown_pct = np.nan

    state_counts = window.loc[window["is_scan_day"] == True, "market_buy_state"].value_counts().to_dict()
    return {
        "period_label": str(period_label),
        "start_date": actual_start,
        "end_date": actual_end,
        "calendar_days": int(calendar_days),
        "n_eval_days": int(len(window)),
        "n_scan_days": int(window["is_scan_day"].sum()),
        "n_days_with_signal": int((window["n_selected"] > 0).sum()),
        "n_trades_opened": int(len(opened_trades)),
        "n_trades_closed": int(len(closed_trades)),
        "win_rate": win_rate,
        "avg_trade_return_pct": avg_trade_return_pct,
        "median_trade_return_pct": median_trade_return_pct,
        "avg_max_runup_pct": avg_max_runup_pct,
        "avg_max_drawdown_pct": avg_max_drawdown_pct,
        "avg_gross_exposure": float(window["gross_exposure"].mean()) if not window.empty else np.nan,
        "end_nav": end_nav,
        "interval_nav": interval_nav,
        "total_return_pct": float((interval_nav - 1.0) * 100.0) if np.isfinite(interval_nav) else np.nan,
        "annualized_return_pct": annualized_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "buy_window_days": int(state_counts.get("buy_window", 0)),
        "repair_watch_days": int(state_counts.get("repair_watch", 0)),
        "selloff_days": int(state_counts.get("selloff", 0)),
        "rebound_crowded_days": int(state_counts.get("rebound_crowded", 0)),
    }


def _build_backtest_yearly_performance(df_daily: pd.DataFrame, df_trades: pd.DataFrame) -> pd.DataFrame:
    if df_daily.empty:
        return _empty_backtest_performance_frame()
    years = sorted(df_daily["scan_date"].str[:4].dropna().unique())
    rows = [
        _summarize_backtest_window(df_daily, df_trades, f"{year}", f"{year}0101", f"{year}1231")
        for year in years
    ]
    rows = [row for row in rows if row is not None]
    if not rows:
        return _empty_backtest_performance_frame()
    return pd.DataFrame(rows)


def _build_backtest_interval_performance(df_daily: pd.DataFrame, df_trades: pd.DataFrame) -> pd.DataFrame:
    if df_daily.empty:
        return _empty_backtest_performance_frame()

    first_dt = df_daily["dt"].iloc[0]
    last_dt = df_daily["dt"].iloc[-1]
    if pd.isna(first_dt) or pd.isna(last_dt):
        return _empty_backtest_performance_frame()

    interval_specs = [
        ("since_inception", first_dt),
        ("last_3y", last_dt - pd.DateOffset(years=3) + pd.Timedelta(days=1)),
        ("last_2y", last_dt - pd.DateOffset(years=2) + pd.Timedelta(days=1)),
        ("last_1y", last_dt - pd.DateOffset(years=1) + pd.Timedelta(days=1)),
        ("ytd", pd.Timestamp(year=last_dt.year, month=1, day=1)),
        ("last_6m", last_dt - pd.DateOffset(months=6) + pd.Timedelta(days=1)),
        ("last_3m", last_dt - pd.DateOffset(months=3) + pd.Timedelta(days=1)),
    ]
    rows: list[dict[str, object]] = []
    for label, start_dt in interval_specs:
        actual_start_dt = max(first_dt, start_dt)
        if actual_start_dt > last_dt:
            continue
        row = _summarize_backtest_window(
            df_daily,
            df_trades,
            label,
            actual_start_dt.strftime("%Y%m%d"),
            last_dt.strftime("%Y%m%d"),
        )
        if row is not None:
            rows.append(row)

    if not rows:
        return _empty_backtest_performance_frame()
    return pd.DataFrame(rows)


def write_continuous_decline_recovery_outputs(
    out_dir: str,
    strategy_name: str,
    scan_date: str,
    top_n: int,
    market_rows: list[dict],
    sector_rows: list[dict],
    candidate_rows: list[dict],
    selected_rows: list[dict],
    summary: dict,
    backtest_daily_rows: list[dict],
    backtest_trade_rows: list[dict],
    backtest_summary: dict | None,
) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)

    market_snapshot_path = os.path.join(out_dir, f"market_scan_snapshot_{strategy_name}_{scan_date}.csv")
    sector_ranking_path = os.path.join(out_dir, f"sector_ranking_{strategy_name}_{scan_date}.csv")
    candidates_all_path = os.path.join(out_dir, f"{strategy_name}_candidates_{scan_date}_all.csv")
    candidates_top_path = os.path.join(out_dir, f"{strategy_name}_candidates_{scan_date}_top{top_n}.csv")
    selected_path = os.path.join(out_dir, f"selected_portfolio_{strategy_name}_{scan_date}_top{top_n}.csv")
    summary_path = os.path.join(out_dir, f"strategy_summary_{strategy_name}_{scan_date}.csv")
    backtest_daily_path = os.path.join(out_dir, f"forward_backtest_daily_{strategy_name}_{scan_date}.csv")
    backtest_trades_path = os.path.join(out_dir, f"forward_backtest_trades_{strategy_name}_{scan_date}.csv")
    backtest_summary_path = os.path.join(out_dir, f"forward_backtest_summary_{strategy_name}_{scan_date}.csv")
    backtest_yearly_path = os.path.join(out_dir, f"forward_backtest_yearly_performance_{strategy_name}_{scan_date}.csv")
    backtest_interval_path = os.path.join(out_dir, f"forward_backtest_interval_performance_{strategy_name}_{scan_date}.csv")

    df_market = pd.DataFrame(market_rows)
    if not df_market.empty:
        df_market = df_market.sort_values(
            ["strategy_state", "sector_rank", "strategy_score", "repair_score", "amount"],
            ascending=[False, True, False, False, False],
        ).reset_index(drop=True)
    elif market_rows == []:
        df_market = pd.DataFrame(columns=["symbol", "name", "industry", "market_buy_state", "strategy_score"])
    df_market.to_csv(market_snapshot_path, index=False)

    df_sectors = pd.DataFrame(sector_rows)
    if not df_sectors.empty:
        df_sectors = df_sectors.sort_values(["sector_rank", "sector_score"], ascending=[True, False]).reset_index(drop=True)
    elif sector_rows == []:
        df_sectors = pd.DataFrame(columns=["industry", "sector_rank", "sector_score", "sector_state"])
    df_sectors.to_csv(sector_ranking_path, index=False)

    df_candidates = pd.DataFrame(candidate_rows)
    if not df_candidates.empty:
        df_candidates = df_candidates.sort_values(
            ["sector_rank", "strategy_score", "repair_score", "flow_support_score", "amount"],
            ascending=[True, False, False, False, False],
        ).reset_index(drop=True)
    elif candidate_rows == []:
        df_candidates = pd.DataFrame(columns=["symbol", "name", "industry", "sector_rank", "strategy_score"])
    df_candidates.to_csv(candidates_all_path, index=False)
    df_candidates.head(int(top_n)).to_csv(candidates_top_path, index=False)

    df_selected = pd.DataFrame(selected_rows)
    if not df_selected.empty:
        df_selected = df_selected.sort_values(["selected_rank"], ascending=[True]).reset_index(drop=True)
    elif selected_rows == []:
        df_selected = pd.DataFrame(columns=["selected_rank", "symbol", "name", "industry", "strategy_score", "entry_mode"])
    df_selected.to_csv(selected_path, index=False)

    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    df_backtest_daily = pd.DataFrame(backtest_daily_rows)
    if not df_backtest_daily.empty:
        df_backtest_daily = df_backtest_daily.sort_values(["scan_date"], ascending=[True]).reset_index(drop=True)
    elif backtest_daily_rows == []:
        df_backtest_daily = pd.DataFrame(
            columns=[
                "strategy_name",
                "scan_date",
                "is_scan_day",
                "market_buy_state",
                "market_buy_score",
                "top_sector",
                "n_sector_ranked",
                "n_candidates",
                "n_selected",
                "n_skipped_full",
                "n_skipped_industry",
                "n_skipped_duplicate",
                "active_positions",
                "gross_exposure",
                "realized_trades",
                "strategy_daily_return",
                "nav",
            ]
        )
    df_backtest_daily.to_csv(backtest_daily_path, index=False)

    df_backtest_trades = pd.DataFrame(backtest_trade_rows)
    if not df_backtest_trades.empty:
        df_backtest_trades = df_backtest_trades.sort_values(["scan_date", "symbol"], ascending=[True, True]).reset_index(drop=True)
    elif backtest_trade_rows == []:
        df_backtest_trades = pd.DataFrame(
            columns=[
                "scan_date",
                "entry_scan_date",
                "entry_date",
                "exit_date",
                "symbol",
                "ts_code",
                "name",
                "industry",
                "market",
                "selected_rank",
                "entry_mode",
                "position_scale",
                "staged_entry_days",
                "entry_price",
                "exit_price",
                "hold_days_realized",
                "full_hold_reached",
                "return_pct",
                "max_runup_pct",
                "max_drawdown_pct",
            ]
        )
    df_backtest_trades.to_csv(backtest_trades_path, index=False)

    pd.DataFrame([backtest_summary or {}]).to_csv(backtest_summary_path, index=False)

    perf_daily, perf_trades = _prepare_backtest_frames(backtest_daily_rows, backtest_trade_rows)
    _build_backtest_yearly_performance(perf_daily, perf_trades).to_csv(backtest_yearly_path, index=False)
    _build_backtest_interval_performance(perf_daily, perf_trades).to_csv(backtest_interval_path, index=False)

    return [
        market_snapshot_path,
        sector_ranking_path,
        candidates_all_path,
        candidates_top_path,
        selected_path,
        summary_path,
        backtest_daily_path,
        backtest_trades_path,
        backtest_summary_path,
        backtest_yearly_path,
        backtest_interval_path,
    ]