import glob
import os
from multiprocessing import Pool, cpu_count

import pandas as pd


LEADERS = ["300502.SZ", "300394.SZ"]  # 新易盛 / 天孚通信
WEIGHT_INDEX = 1.5


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    strategy_dir = os.path.join(here, "src", "strategy")
    if not os.path.isdir(strategy_dir):
        raise FileNotFoundError(strategy_dir)

    # Make the strategy module importable by name so multiprocessing can pickle worker functions.
    import sys

    if strategy_dir not in sys.path:
        sys.path.insert(0, strategy_dir)

    import phase_transition_backtest_index_prob_weighted as m

    # Build index calibration + daily index scores
    df_index = m.load_index_df()
    _, index_s_by_date = m.build_index_calibration_and_scores(df_index)

    # Stock calibration (trained on 2023-2024 triggered signals)
    files = glob.glob(os.path.join(m.STOCK_DATA_DIR, "*.csv"))
    print(f"Found {len(files)} stock CSV files. Calibrating stock probabilities...")
    stock_calib = m.calibrate_stock_probability(files)

    # Set globals for workers (Linux fork will inherit these)
    m.INDEX_S_BY_DATE = index_s_by_date
    m.STOCK_CALIB = stock_calib
    m.WEIGHT_INDEX = float(WEIGHT_INDEX)
    m.WEIGHT_STOCK = float(m.WEIGHT_STOCK)

    print("Generating full 2025 candidate pool (this may take a bit)...")
    with Pool(max(1, cpu_count() - 1)) as pool:
        results = pool.map(m.process_single_stock_prob, files)

    all_rows: list[dict] = []
    for r in results:
        all_rows.extend(r)

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("No candidates produced.")
        return

    df["entry_date"] = df["entry_date"].astype(str)
    df["ts_code"] = df["ts_code"].astype(str)
    df["entry_score"] = pd.to_numeric(df["entry_score"], errors="coerce")
    df = df.dropna(subset=["entry_score"]).copy()

    # Daily top-by-score among candidates (ignores portfolio constraints)
    daily_top = (
        df.sort_values(["entry_date", "entry_score"], ascending=[True, False])
        .groupby("entry_date", as_index=False)
        .head(1)
        .rename(columns={"ts_code": "top_ts_code", "entry_score": "top_entry_score"})
        [["entry_date", "top_ts_code", "top_entry_score"]]
    )

    print("\n=== Leader rank diagnostics (candidate pool; ignores portfolio) ===")
    for leader in LEADERS:
        df_l = df[df["ts_code"] == leader].copy()
        if df_l.empty:
            print(f"{leader}: no candidate signals in 2025 under current filters.")
            continue

        # In case a stock emits multiple candidates for the same entry_date (rare, but safe)
        df_l = df_l.sort_values(["entry_date", "entry_score"], ascending=[True, False]).groupby("entry_date", as_index=False).head(1)

        merged = df_l.merge(daily_top, on="entry_date", how="left")
        # compute rank via per-row comparison (fast enough given small leader rows)
        ranks = []
        for _, row in merged.iterrows():
            d = str(row["entry_date"])
            s0 = float(row["entry_score"])
            s_all = df.loc[df["entry_date"] == d, "entry_score"]
            rank = int((s_all > s0).sum() + 1)
            ranks.append(rank)
        merged["rank_in_day"] = ranks

        n_days = len(merged)
        n_top1 = int((merged["rank_in_day"] == 1).sum())
        best_rank = int(merged["rank_in_day"].min())
        worst_rank = int(merged["rank_in_day"].max())

        print(
            f"\n{leader}: days_with_candidate={n_days}, top1_days={n_top1}, best_rank={best_rank}, worst_rank={worst_rank}"
        )
        # show up to 15 worst days by rank
        show = merged.sort_values(["rank_in_day", "entry_score"], ascending=[False, True]).head(15)
        print(show[["entry_date", "entry_score", "rank_in_day", "top_ts_code", "top_entry_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
