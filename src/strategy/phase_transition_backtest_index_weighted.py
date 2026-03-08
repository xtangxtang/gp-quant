import os
import glob
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

START_DATE = "20250101"
END_DATE = "20251230"

INDEX_SYMBOL = "sh000001"  # 上证指数
INDEX_TS_CODE = "000001.SH"
INDEX_DATA_PATH = "/nvme5/xtang/gp-workspace/gp-data/tushare-index-daily/sh000001.csv"
STOCK_DATA_DIR = "/nvme5/xtang/gp-workspace/gp-data/tushare-daily-full/"
BASIC_INFO_PATH = "/nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv"

REPORT_PATH = "/nvme5/xtang/gp-workspace/gp-quant/backtest_report_phase_transition_2025_index_weighted.md"
GRID_CSV_PATH = "/nvme5/xtang/gp-workspace/gp-quant/backtest_grid_2025_index_weighted.csv"
BEST_TRADES_PATH = "/nvme5/xtang/gp-workspace/gp-quant/backtest_trades_2025_index_weighted.csv"

# Globals used by multiprocessing workers
INDEX_STATE_BY_DATE: dict[str, int] = {}
WEIGHT_INDEX: float = 0.0
WEIGHT_STOCK: float = 1.0


def calc_entropy(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if len(x) < 5:
        return 999.0
    hist, _ = np.histogram(x, bins=10)
    total = np.sum(hist)
    if total <= 0:
        return 999.0
    p = hist / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def fast_hurst(ts: np.ndarray) -> float:
    ts = np.asarray(ts)
    if len(ts) < 10:
        return 0.5
    lags = [2, 4, 8, 16]
    tau: list[int] = []
    msd: list[float] = []
    for lag in lags:
        if lag >= len(ts):
            break
        diffs = ts[lag:] - ts[:-lag]
        val = float(np.mean(diffs**2))
        if val > 0:
            msd.append(val)
            tau.append(lag)
    if len(tau) < 2:
        return 0.5
    m = np.polyfit(np.log(tau), np.log(msd), 1)
    return float(m[0] / 2.0)


def _import_tushare():
    try:
        import tushare as ts  # type: ignore

        return ts
    except Exception as e:
        raise RuntimeError("Python package 'tushare' is required.") from e


def _get_pro_api():
    """Get tushare pro client.

    Uses env var token to avoid relying on (possibly stale) ~/.tushare_token,
    and does NOT call ts.set_token (which would overwrite local token files).
    """
    ts = _import_tushare()
    token = (os.getenv("TUSHARE_TOKEN", "") or "").strip() or (os.getenv("GP_TUSHARE_TOKEN", "") or "").strip()
    if not token:
        raise RuntimeError("Tushare token not found. Set env var TUSHARE_TOKEN (or GP_TUSHARE_TOKEN).")
    return ts.pro_api(token)


def fetch_index_daily_to_csv(ts_code: str, start_yyyymmdd: str, end_yyyymmdd: str, out_path: str) -> None:
    pro = _get_pro_api()

    df = pro.index_daily(ts_code=ts_code, start_date=start_yyyymmdd, end_date=end_yyyymmdd)
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise RuntimeError(f"Failed to fetch index_daily for {ts_code}")

    df = df.copy()
    df["trade_date_str"] = df["trade_date"].astype(str)
    df = df.sort_values("trade_date_str").reset_index(drop=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)


def load_index_df() -> pd.DataFrame:
    if not os.path.exists(INDEX_DATA_PATH):
        # Grab some buffer before START_DATE for rolling windows.
        fetch_index_daily_to_csv(INDEX_TS_CODE, "20240901", END_DATE, INDEX_DATA_PATH)

    df = pd.read_csv(INDEX_DATA_PATH)
    if df.empty or "trade_date" not in df.columns:
        raise RuntimeError(f"Index CSV invalid: {INDEX_DATA_PATH}")

    df["trade_date_str"] = df["trade_date"].astype(str)
    df = df.sort_values("trade_date_str").reset_index(drop=True)

    # Normalize to the columns we need
    for col in ("close", "pct_chg"):
        if col not in df.columns:
            raise RuntimeError(f"Index CSV missing required column: {col}")

    return df


def build_index_state(df_index: pd.DataFrame) -> dict[str, int]:
    closes = pd.to_numeric(df_index["close"], errors="coerce").values
    pct_chg = pd.to_numeric(df_index["pct_chg"], errors="coerce").values
    dates = df_index["trade_date_str"].values

    ma60 = pd.Series(closes).rolling(window=60).mean().values
    ma20 = pd.Series(closes).rolling(window=20).mean().values

    state: dict[str, int] = {}

    for i in range(60, len(df_index)):
        d = str(dates[i])
        if d < START_DATE or d > END_DATE:
            continue

        w60 = closes[i - 60 : i + 1]
        w20 = closes[i - 20 : i + 1]
        r60 = pct_chg[i - 60 : i + 1]
        r20 = pct_chg[i - 20 : i + 1]

        h60 = fast_hurst(w60)
        h20 = fast_hurst(w20)
        e60 = calc_entropy(r60)
        e20 = calc_entropy(r20)

        # Index buy regime (we use a softer daily change threshold vs individual stocks)
        idx_buy = (
            pct_chg[i] >= 0.5
            and closes[i] >= ma60[i]
            and (0.20 <= h60 <= 0.60)
            and (h20 >= 0.45)
            and (e20 <= e60 + 0.1)
        )

        # Index sell regime: lose the medium-term trend or disorder spikes on weakness
        idx_sell = (
            (not np.isnan(ma20[i]) and closes[i] < ma20[i] * 0.98)
            or (pct_chg[i] <= -1.0 and (e20 > e60 + 0.2))
        )

        if idx_sell:
            state[d] = -1
        elif idx_buy:
            state[d] = 1
        else:
            state[d] = 0

    return state


def process_single_stock(file_path: str):
    global INDEX_STATE_BY_DATE, WEIGHT_INDEX, WEIGHT_STOCK

    try:
        df = pd.read_csv(file_path)
    except Exception:
        return []

    if df.empty or "trade_date" not in df.columns:
        return []

    df["trade_date_str"] = df["trade_date"].astype(str)
    # Need buffer for 60-day MA
    df = df[df["trade_date_str"] >= "20240901"].copy()
    if df.empty:
        return []

    df = df.sort_values("trade_date").reset_index(drop=True)

    required_cols = ["close", "open", "pct_chg", "net_mf_amount", "ts_code"]
    for col in required_cols:
        if col not in df.columns:
            return []

    # prereqs
    df["ma60"] = pd.to_numeric(df["close"], errors="coerce").rolling(window=60).mean()
    df["ma20"] = pd.to_numeric(df["close"], errors="coerce").rolling(window=20).mean()
    df["mf_5d"] = pd.to_numeric(df["net_mf_amount"], errors="coerce").fillna(0).rolling(window=5).sum()

    closes = pd.to_numeric(df["close"], errors="coerce").values
    pct_chgs = pd.to_numeric(df["pct_chg"], errors="coerce").values
    dates = df["trade_date_str"].values
    ma20 = df["ma20"].values

    end_cut_idx = int(np.searchsorted(dates, END_DATE, side="right") - 1)
    if end_cut_idx < 0:
        return []

    out_trades = []

    for i in range(60, len(df)):
        current_date = str(dates[i])
        if current_date < START_DATE or current_date > END_DATE:
            continue

        # Fast filters
        if pct_chgs[i] < 3.0:
            continue
        if closes[i] < df["ma60"].iloc[i]:
            continue
        if df["mf_5d"].iloc[i] <= 0:
            continue

        # Physics metrics
        w60 = closes[i - 60 : i + 1]
        w20 = closes[i - 20 : i + 1]
        r60 = pct_chgs[i - 60 : i + 1]
        r20 = pct_chgs[i - 20 : i + 1]

        h60 = fast_hurst(w60)
        h20 = fast_hurst(w20)
        e60 = calc_entropy(r60)
        e20 = calc_entropy(r20)

        if not (0.20 <= h60 <= 0.60):
            continue
        if h20 < 0.45:
            continue
        if e20 > e60 + 0.1:
            continue

        # Weighted entry gate (index state votes for/against)
        idx_state = int(INDEX_STATE_BY_DATE.get(current_date, 0))
        entry_score = WEIGHT_STOCK * 1.0 + WEIGHT_INDEX * idx_state
        if entry_score <= 0:
            continue

        if i + 1 >= len(df) or (i + 1) > end_cut_idx:
            continue

        ts_code = str(df["ts_code"].iloc[0])
        entry_price = float(df["open"].iloc[i + 1])
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue

        max_forward_idx = end_cut_idx + 1
        f_closes = closes[i + 1 : max_forward_idx]
        f_ma20 = ma20[i + 1 : max_forward_idx]
        f_pct = pct_chgs[i + 1 : max_forward_idx]
        f_dates = dates[i + 1 : max_forward_idx]

        if len(f_closes) == 0:
            continue

        exit_idx = len(f_closes) - 1
        exit_reason = "End Date"

        for j in range(len(f_closes)):
            d = str(f_dates[j])
            p_close = float(f_closes[j])
            current_return = (p_close - entry_price) / entry_price

            stock_state = 1

            # Stock stop-loss
            if current_return <= -0.08 or (np.isfinite(f_ma20[j]) and p_close < float(f_ma20[j]) * 0.95):
                stock_state = -1
                exit_idx = j
                exit_reason = "Stop Loss"

            # Stock entropy spike take-profit
            if exit_reason == "End Date" and j >= 5 and current_return > 0.15:
                recent_ret = f_pct[max(0, j - 10) : j + 1]
                if len(recent_ret) >= 5:
                    cur_e = calc_entropy(recent_ret)
                    if cur_e > e60 * 1.2:
                        stock_state = -1
                        exit_idx = j
                        exit_reason = "Take Profit (Entropy Spike)"

            # Index weighted exit
            idx_s = int(INDEX_STATE_BY_DATE.get(d, 0))
            combined_state = WEIGHT_STOCK * stock_state + WEIGHT_INDEX * idx_s
            if combined_state < 0:
                if exit_reason == "End Date":
                    exit_idx = j
                    exit_reason = "Weighted Exit (Index)"
                break

            if exit_reason != "End Date":
                break

        max_close = float(np.max(f_closes[: exit_idx + 1]))
        exit_close = float(f_closes[exit_idx])
        max_ret = (max_close - entry_price) / entry_price * 100.0
        hold_ret = (exit_close - entry_price) / entry_price * 100.0

        out_trades.append(
            {
                "ts_code": ts_code,
                "signal_date": current_date,
                "entry_date": str(dates[i + 1]),
                "exit_reason": exit_reason,
                "hold_days": int(exit_idx + 1),
                "hurst_60": round(float(h60), 3),
                "hurst_20": round(float(h20), 3),
                "entropy_drop_pct": round(float((e60 - e20) / e60 * 100.0), 2) if np.isfinite(e60) and e60 != 0 else np.nan,
                "energy_mf": round(float(df["mf_5d"].iloc[i]), 2),
                "max_hold_return(%)": round(float(max_ret), 2),
                "actual_return(%)": round(float(hold_ret), 2),
            }
        )

    return out_trades


def run_backtest_for_weight(weight_index: float, files: list[str]) -> pd.DataFrame:
    global WEIGHT_INDEX
    WEIGHT_INDEX = float(weight_index)

    pool = Pool(max(1, cpu_count() - 1))
    results = pool.map(process_single_stock, files)
    pool.close()
    pool.join()

    all_trades: list[dict] = []
    for r in results:
        all_trades.extend(r)

    return pd.DataFrame(all_trades)


def main():
    global INDEX_STATE_BY_DATE

    # 1) Build index state map
    df_index = load_index_df()
    INDEX_STATE_BY_DATE = build_index_state(df_index)

    # 2) Load stock files
    files = glob.glob(os.path.join(STOCK_DATA_DIR, "*.csv"))
    print(f"Found {len(files)} stock CSV files. Starting 2025 index-weighted backtest...")

    # 3) Weight grid search (index weight relative to stock weight=1)
    weight_grid = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]

    grid_rows = []
    best = None

    start = time.time()
    for w_idx in weight_grid:
        df_trades = run_backtest_for_weight(w_idx, files)
        if df_trades.empty:
            grid_rows.append({"w_index": w_idx, "trades": 0, "win_rate(%)": np.nan, "avg_return(%)": np.nan})
            continue

        total = len(df_trades)
        win_rate = (df_trades["actual_return(%)"] > 0).mean() * 100.0
        avg_ret = df_trades["actual_return(%)"].mean()
        med_ret = df_trades["actual_return(%)"].median()
        avg_hold_days = df_trades["hold_days"].mean()
        avg_max = df_trades["max_hold_return(%)"].mean()

        exit_counts = df_trades["exit_reason"].value_counts().to_dict()
        stop_loss_pct = exit_counts.get("Stop Loss", 0) / total * 100.0
        take_profit_pct = exit_counts.get("Take Profit (Entropy Spike)", 0) / total * 100.0
        weighted_exit_pct = exit_counts.get("Weighted Exit (Index)", 0) / total * 100.0
        end_date_pct = exit_counts.get("End Date", 0) / total * 100.0

        row = {
            "w_index": w_idx,
            "trades": int(total),
            "win_rate(%)": round(float(win_rate), 2),
            "avg_return(%)": round(float(avg_ret), 2),
            "median_return(%)": round(float(med_ret), 2),
            "avg_max_return(%)": round(float(avg_max), 2),
            "avg_hold_days": round(float(avg_hold_days), 1),
            "stop_loss_exit(%)": round(float(stop_loss_pct), 1),
            "take_profit_exit(%)": round(float(take_profit_pct), 1),
            "weighted_exit(%)": round(float(weighted_exit_pct), 1),
            "end_date_exit(%)": round(float(end_date_pct), 1),
        }
        grid_rows.append(row)

        if best is None or win_rate > best["win_rate_raw"]:
            best = {
                "w_index": w_idx,
                "win_rate_raw": win_rate,
                "row": row,
                "df_trades": df_trades,
            }

    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(GRID_CSV_PATH, index=False)

    if best is None:
        print("No trades in any weight setting.")
        return

    best_w = best["w_index"]
    best_trades = best["df_trades"].copy()

    # Attach names if available
    if os.path.exists(BASIC_INFO_PATH):
        df_basic = pd.read_csv(BASIC_INFO_PATH)
        name_map = dict(zip(df_basic["ts_code"], df_basic["name"]))
        best_trades.insert(1, "name", best_trades["ts_code"].map(name_map))

    best_trades.to_csv(BEST_TRADES_PATH, index=False)

    # Bull list per symbol: best single trade per ts_code
    bull = (
        best_trades.sort_values("actual_return(%)", ascending=False)
        .groupby("ts_code", as_index=False)
        .head(1)
        .head(30)
    )

    report = f"""
# 2025 相变策略：上证指数加权回测报告

* **回测区间**: {START_DATE} 到 {END_DATE}
* **指数标的**: 上证指数 ({INDEX_TS_CODE})
* **组合信号**: $Score = 1.0 * Signal_stock + w_index * Signal_index$，其中 Signal 取值 {{+1, 0, -1}}
* **本次网格**: w_index ∈ {weight_grid}

## 权重回归结果（按胜率选最优）
{grid_df.sort_values('w_index').to_markdown(index=False)}

## 最优权重
* **w_index**: {best_w}
* **胜率**: {best['row']['win_rate(%)']}%
* **平均平仓收益率**: {best['row']['avg_return(%)']}%
* **中位数收益率**: {best['row']['median_return(%)']}%
* **平均持仓天数**: {best['row']['avg_hold_days']}

## 2025 牛股榜（最优权重下，按标的最佳一笔收益 Top 30）
{bull.to_markdown(index=False)}

* 输出：
  - 网格结果 CSV: {GRID_CSV_PATH}
  - 最优权重交易明细 CSV: {BEST_TRADES_PATH}
  - 指数数据 CSV: {INDEX_DATA_PATH}

运行耗时：{time.time() - start:.2f} 秒
"""

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nDone. Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
