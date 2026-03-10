import os

import numpy as np
import pandas as pd


def write_scan_outputs(
    out_dir: str,
    test_year: int,
    top_n: int,
    df_ret: pd.DataFrame,
    df_bull: pd.DataFrame,
    out_rows: list[dict],
    resonance_rows: list[dict],
    resonance_signal_rows: list[dict],
) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)

    bull_all_path = os.path.join(out_dir, f"bull_stocks_{test_year}_all.csv")
    bull_top_path = os.path.join(out_dir, f"bull_stocks_{test_year}_top{top_n}.csv")
    entry_eval_path = os.path.join(out_dir, f"multitimeframe_entry_eval_{test_year}.csv")
    entry_agg_path = os.path.join(out_dir, f"multitimeframe_entry_eval_{test_year}_agg.csv")
    resonance_eval_path = os.path.join(out_dir, f"multitimeframe_resonance_eval_{test_year}.csv")
    resonance_agg_path = os.path.join(out_dir, f"multitimeframe_resonance_eval_{test_year}_agg.csv")
    resonance_signals_path = os.path.join(out_dir, f"multitimeframe_resonance_signals_{test_year}.csv")

    df_ret.to_csv(bull_all_path, index=False)
    df_bull.to_csv(bull_top_path, index=False)

    df_eval = pd.DataFrame(out_rows)
    df_eval.to_csv(entry_eval_path, index=False)

    df_res = pd.DataFrame(resonance_rows)
    df_res.to_csv(resonance_eval_path, index=False)

    df_res_signals = pd.DataFrame(resonance_signal_rows)
    df_res_signals.to_csv(resonance_signals_path, index=False)

    if not df_eval.empty:
        agg = (
            df_eval.assign(hit=df_eval["entry_date"].notna())
            .groupby("freq", as_index=False)
            .agg(
                n_stocks=("symbol", "nunique"),
                hit_rate=("hit", "mean"),
                avg_ret_to_year_end=("ret_to_year_end", "mean"),
                median_ret_to_year_end=("ret_to_year_end", "median"),
                avg_max_runup=("max_runup_to_year_end", "mean"),
                median_max_runup=("max_runup_to_year_end", "median"),
                avg_score=("score_min_persist", "mean"),
                avg_energy=("energy_term", "mean"),
                avg_order=("order_term", "mean"),
                avg_phase=("phase_term", "mean"),
            )
        )
        agg.to_csv(entry_agg_path, index=False)

    if not df_res.empty:
        res_agg = pd.DataFrame(
            [
                {
                    "n_stocks": int(df_res["symbol"].nunique()),
                    "hit_rate": float(df_res["entry_date"].notna().mean()),
                    "avg_ret_to_year_end": float(df_res["ret_to_year_end"].mean()) if df_res["ret_to_year_end"].notna().any() else np.nan,
                    "median_ret_to_year_end": float(df_res["ret_to_year_end"].median()) if df_res["ret_to_year_end"].notna().any() else np.nan,
                    "avg_max_runup": float(df_res["max_runup_to_year_end"].mean()) if df_res["max_runup_to_year_end"].notna().any() else np.nan,
                    "median_max_runup": float(df_res["max_runup_to_year_end"].median()) if df_res["max_runup_to_year_end"].notna().any() else np.nan,
                    "avg_resonance_score": float(df_res["resonance_score"].mean()) if df_res["resonance_score"].notna().any() else np.nan,
                    "avg_support_count": float(df_res["support_count"].mean()) if df_res["support_count"].notna().any() else np.nan,
                }
            ]
        )
        res_agg.to_csv(resonance_agg_path, index=False)

    return [
        bull_all_path,
        bull_top_path,
        entry_eval_path,
        entry_agg_path,
        resonance_eval_path,
        resonance_agg_path,
        resonance_signals_path,
    ]