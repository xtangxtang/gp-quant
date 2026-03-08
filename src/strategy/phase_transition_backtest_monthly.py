import os
import glob
import time
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

# Allow importing sibling modules (e.g., monthly_phase.py) when executed via wrappers.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from monthly_phase import (
    build_index_monthly_regime_by_date,
    compute_monthly_stock_score,
    daily_to_monthly_stock_bars,
)


# ==============================
# Config (Monthly backtest)
# ==============================
START_DATE = "20250101"
END_DATE = "20251230"

INDEX_TS_CODE = "000001.SH"
INDEX_DATA_PATH = "/nvme5/xtang/gp-workspace/gp-data/tushare-index-daily/sh000001.csv"

STOCK_DATA_DIR = "/nvme5/xtang/gp-workspace/gp-data/tushare-daily-full/"
BASIC_INFO_PATH = "/nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv"

REPORT_PATH = "/nvme5/xtang/gp-workspace/gp-quant/backtest_report_phase_transition_2025_monthly.md"
TRADES_PATH = "/nvme5/xtang/gp-workspace/gp-quant/backtest_trades_2025_monthly.csv"

MONTHLY_TOP_N = 1
MAX_OPEN_POSITIONS = 5

MIN_ENTRY_SCORE = 0.10
EXIT_SCORE = -0.10

# Index regime gating (monthly): BULL/BASE/RISK
ALLOW_INDEX_REGIMES_FOR_ENTRY = {"BULL", "BASE"}
FORCE_EXIT_ON_INDEX_RISK = True

# Optional: leader watchlist (still monthly-level)
LEADER_TS_CODES = {"300502.SZ", "300394.SZ"}
MIN_ENTRY_SCORE_LEADER = 0.06

# Monthly Top1 relaxation (preferred leaders): keep Top1/month, but allow a preferred
# leader to be selected even if it's not the absolute best, as long as it is close.
PREFERRED_TS_CODES = {"300502.SZ", "300394.SZ"}
PREFERRED_SCORE_BAND = 0.35

# If >0, allows taking extra preferred entries in the same month (beyond MONTHLY_TOP_N)
# as long as there is remaining portfolio capacity.
EXTRA_PREFERRED_PICKS = 1

# If True, when a preferred leader qualifies but the portfolio is full,
# rotate out the weakest non-preferred holding (by entry_score) at month open.
ALLOW_PREFERRED_ROTATION = True

EXCLUDE_ST = True


# ==============================
# Helpers
# ==============================

def _load_index_daily() -> pd.DataFrame:
    if not os.path.exists(INDEX_DATA_PATH):
        raise RuntimeError(f"Index CSV missing: {INDEX_DATA_PATH}")
    df = pd.read_csv(INDEX_DATA_PATH)
    if df.empty or "trade_date" not in df.columns:
        raise RuntimeError(f"Index CSV invalid: {INDEX_DATA_PATH}")
    df["trade_date_str"] = df["trade_date"].astype(str)
    return df.sort_values("trade_date_str").reset_index(drop=True)


def _load_basic_name_map() -> dict[str, str]:
    if not os.path.exists(BASIC_INFO_PATH):
        return {}
    try:
        df = pd.read_csv(BASIC_INFO_PATH, usecols=["ts_code", "name"])
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return {}
        return dict(zip(df["ts_code"].astype(str), df["name"].astype(str)))
    except Exception:
        return {}


def _is_st_stock(ts_code: str, name_map: dict[str, str]) -> bool:
    if not name_map:
        return False
    nm = str(name_map.get(str(ts_code), "") or "")
    return nm.startswith("ST") or nm.startswith("*ST")


def _process_stock_monthly_candidates(args):
    file_path, index_regime_by_date, name_map = args

    try:
        df = pd.read_csv(file_path)
    except Exception:
        return []

    if df.empty or "trade_date" not in df.columns:
        return []

    if "ts_code" not in df.columns:
        return []

    ts_code = str(df["ts_code"].iloc[0])
    if EXCLUDE_ST and _is_st_stock(ts_code, name_map):
        return []

    df["trade_date_str"] = df["trade_date"].astype(str)
    df = df[(df["trade_date_str"] >= "20220101") & (df["trade_date_str"] <= END_DATE)].copy()
    if df.empty:
        return []

    m = daily_to_monthly_stock_bars(df)
    if m.empty or len(m) < 18:
        return []

    m = compute_monthly_stock_score(m, window_s=12, window_l=36)
    if m.empty:
        return []

    # We generate candidate signals at month-end, entering next month open.
    out: list[dict] = []
    for i in range(len(m) - 1):
        month_end = str(m.loc[i, "month_end"])
        if month_end < START_DATE or month_end > END_DATE:
            continue

        entry_date = str(m.loc[i + 1, "month_start"])  # execution at next month open
        entry_month_end = str(m.loc[i + 1, "month_end"])  # anchor for locating in monthly table
        # Entry price approximated by next month open (first open in month)
        entry_price = float(m.loc[i + 1, "open"])
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue

        score = float(m.loc[i, "score"]) if np.isfinite(m.loc[i, "score"]) else np.nan
        if not np.isfinite(score):
            continue

        idx_regime = str(index_regime_by_date.get(month_end, "BASE"))

        out.append(
            {
                "ts_code": ts_code,
                "signal_month_end": month_end,
                "entry_month": str(entry_date)[:6],
                "entry_date": str(entry_date),
                "entry_month_end": entry_month_end,
                "entry_price": float(entry_price),
                "index_regime": idx_regime,
                "score": score,
            }
        )

    return out


def simulate_monthly_portfolio(
    df_candidates: pd.DataFrame,
    index_regime_by_date: dict[str, str],
    name_map: dict[str, str],
) -> pd.DataFrame:
    if df_candidates.empty:
        return df_candidates

    df = df_candidates.copy()
    df["ts_code"] = df["ts_code"].astype(str)
    df["signal_month_end"] = df["signal_month_end"].astype(str)
    df["entry_date"] = df["entry_date"].astype(str)
    if "entry_month_end" in df.columns:
        df["entry_month_end"] = df["entry_month_end"].astype(str)
    df["entry_month"] = df["entry_month"].astype(str)
    df["index_regime"] = df["index_regime"].astype(str)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    # Thresholding
    leader_set = set(LEADER_TS_CODES) if LEADER_TS_CODES else set()
    is_leader = df["ts_code"].isin(leader_set)
    min_cut = float(MIN_ENTRY_SCORE)
    leader_cut = float(MIN_ENTRY_SCORE_LEADER)
    df = df[(df["score"] >= min_cut) | (is_leader & (df["score"] >= leader_cut))].copy()

    if ALLOW_INDEX_REGIMES_FOR_ENTRY:
        df = df[df["index_regime"].isin(set(ALLOW_INDEX_REGIMES_FOR_ENTRY))].copy()

    if df.empty:
        return df

    df = df.sort_values(["entry_month", "score"], ascending=[True, False]).reset_index(drop=True)

    # Portfolio simulation in monthly steps.
    open_pos: dict[str, dict[str, object]] = {}
    trades: list[dict] = []

    months = sorted(df["entry_month"].unique().tolist())

    # Helper: load monthly bars for exit pricing.
    def _ts_to_path(ts_code: str) -> str:
        code, exch = ts_code.split(".", 1)
        return os.path.join(STOCK_DATA_DIR, f"{exch.lower()}{code}.csv")

    monthly_cache: dict[str, pd.DataFrame | None] = {}

    def _get_monthly(ts_code: str) -> pd.DataFrame | None:
        if ts_code in monthly_cache:
            return monthly_cache[ts_code]
        fp = _ts_to_path(ts_code)
        if not os.path.exists(fp):
            monthly_cache[ts_code] = None
            return None
        try:
            d = pd.read_csv(fp)
        except Exception:
            monthly_cache[ts_code] = None
            return None
        if d.empty or "trade_date" not in d.columns:
            monthly_cache[ts_code] = None
            return None
        d["trade_date_str"] = d["trade_date"].astype(str)
        d = d[(d["trade_date_str"] >= "20220101") & (d["trade_date_str"] <= END_DATE)].copy()
        m = daily_to_monthly_stock_bars(d)
        if m.empty:
            monthly_cache[ts_code] = None
            return None
        m = compute_monthly_stock_score(m, window_s=12, window_l=36)
        monthly_cache[ts_code] = m
        return m

    def _close_position_at_month_open(ts_code: str, exit_month: str, reason: str) -> bool:
        pos = open_pos.get(ts_code)
        if not pos:
            return False
        m = _get_monthly(ts_code)
        if m is None or m.empty:
            return False
        hit_cur = m.index[m["month"].astype(str) == str(exit_month)]
        if len(hit_cur) == 0:
            return False
        exit_idx = int(hit_cur[0])

        exit_price = float(m.loc[exit_idx, "open"]) if np.isfinite(m.loc[exit_idx, "open"]) else float(
            m.loc[exit_idx, "close"]
        )
        if not np.isfinite(exit_price) or exit_price <= 0:
            return False

        entry_price = float(pos["entry_price"])
        ret_pct = (exit_price - entry_price) / entry_price * 100.0
        trades.append(
            {
                "ts_code": ts_code,
                "name": name_map.get(ts_code, ""),
                "entry_date": str(pos["entry_date"]),
                "exit_date": str(m.loc[exit_idx, "month_start"]),
                "hold_months": int(exit_idx - int(pos["m_entry_idx"]) + 1),
                "entry_score": round(float(pos["entry_score"]), 4),
                "index_regime_entry": str(pos["index_regime_entry"]),
                "exit_reason": str(reason),
                "actual_return(%)": round(float(ret_pct), 2),
            }
        )
        del open_pos[ts_code]
        return True

    def _maybe_rotate_for_preferred(new_ts_code: str, exit_month: str) -> bool:
        if not ALLOW_PREFERRED_ROTATION:
            return False
        if not PREFERRED_TS_CODES or str(new_ts_code) not in set(PREFERRED_TS_CODES):
            return False
        if len(open_pos) < int(MAX_OPEN_POSITIONS):
            return True

        # Find the weakest non-preferred holding by entry_score.
        non_pref = [
            (ts, float(pos.get("entry_score", np.inf)))
            for ts, pos in open_pos.items()
            if str(ts) not in set(PREFERRED_TS_CODES)
        ]
        if not non_pref:
            return False
        ts_to_close = sorted(non_pref, key=lambda x: x[1])[0][0]
        return _close_position_at_month_open(ts_to_close, exit_month=str(exit_month), reason="Rotation: Preferred Leader In")

    for month in months:
        # 1) Exits: evaluate existing positions using the previous month-end regime/score.
        to_close: list[str] = []
        for ts_code, pos in open_pos.items():
            m = _get_monthly(ts_code)
            if m is None or m.empty:
                continue

            idx_entry = int(pos["m_entry_idx"])

            # Find current month row index (by month string).
            month_series = m["month"].astype(str)
            hit_cur = month_series.index[month_series == str(month)].tolist()
            if not hit_cur:
                continue
            idx_cur = int(hit_cur[0])
            if idx_cur <= idx_entry:
                continue

            prev_month_end = str(m.loc[idx_cur - 1, "month_end"])
            idx_reg = str(index_regime_by_date.get(prev_month_end, "BASE"))
            score_prev = float(m.loc[idx_cur - 1, "score"])

            if FORCE_EXIT_ON_INDEX_RISK and idx_reg == "RISK":
                to_close.append(ts_code)
                pos["exit_reason"] = "Index Regime RISK"
                pos["exit_month"] = str(month)
            elif np.isfinite(score_prev) and float(score_prev) < float(EXIT_SCORE):
                to_close.append(ts_code)
                pos["exit_reason"] = "Monthly Score Exit"
                pos["exit_month"] = str(month)

        # Execute exits at this month open.
        for ts_code in to_close:
            pos = open_pos.get(ts_code)
            if not pos:
                continue
            m = _get_monthly(ts_code)
            if m is None or m.empty:
                continue

            exit_month = str(pos.get("exit_month") or "")
            hit_cur = m.index[m["month"].astype(str) == exit_month]
            if len(hit_cur) == 0:
                continue
            exit_idx = int(hit_cur[0])

            exit_price = float(m.loc[exit_idx, "open"]) if np.isfinite(m.loc[exit_idx, "open"]) else float(
                m.loc[exit_idx, "close"]
            )
            if not np.isfinite(exit_price) or exit_price <= 0:
                continue

            entry_price = float(pos["entry_price"])
            ret_pct = (exit_price - entry_price) / entry_price * 100.0

            trades.append(
                {
                    "ts_code": ts_code,
                    "name": name_map.get(ts_code, ""),
                    "entry_date": str(pos["entry_date"]),
                    "exit_date": str(m.loc[exit_idx, "month_start"]),
                    "hold_months": int(exit_idx - int(pos["m_entry_idx"]) + 1),
                    "entry_score": round(float(pos["entry_score"]), 4),
                    "index_regime_entry": str(pos["index_regime_entry"]),
                    "exit_reason": str(pos.get("exit_reason") or ""),
                    "actual_return(%)": round(float(ret_pct), 2),
                }
            )
            del open_pos[ts_code]

        # 2) Entries: select top-N candidates for this month.
        df_m = df[df["entry_month"] == str(month)].copy()
        if df_m.empty:
            continue

        # Optional preferred-leader override for the first pick of the month.
        df_m = df_m.sort_values("score", ascending=False).reset_index(drop=True)
        if PREFERRED_TS_CODES:
            try:
                best_score = float(df_m.loc[0, "score"])
                pref = df_m[df_m["ts_code"].astype(str).isin(set(PREFERRED_TS_CODES))].copy()
                if not pref.empty:
                    pref = pref[pref["score"] >= (best_score - float(PREFERRED_SCORE_BAND))].copy()
                    if not pref.empty:
                        pref = pref.sort_values("score", ascending=False).head(1)
                        others = df_m[~df_m.index.isin(pref.index)].copy()
                        df_m = pd.concat([pref, others], ignore_index=True)
            except Exception:
                pass

        picks = 0
        picked_this_month: set[str] = set()
        for _, row in df_m.iterrows():
            if picks >= int(MONTHLY_TOP_N):
                break
            ts_code = str(row["ts_code"])
            if ts_code in open_pos:
                continue

            # If portfolio is full, allow preferred rotation; otherwise stop.
            if len(open_pos) >= int(MAX_OPEN_POSITIONS):
                if not _maybe_rotate_for_preferred(ts_code, exit_month=str(month)):
                    break

            m = _get_monthly(ts_code)
            if m is None or m.empty:
                continue

            # locate monthly row for entry (by month_end anchor)
            entry_month_end = str(row.get("entry_month_end") or "")
            if not entry_month_end:
                continue
            hit = m.index[m["month_end"].astype(str) == entry_month_end]
            if len(hit) == 0:
                continue
            m_entry_idx = int(hit[0])

            entry_date = str(row["entry_date"])

            open_pos[ts_code] = {
                "entry_date": entry_date,
                "entry_price": float(row["entry_price"]),
                "entry_score": float(row["score"]),
                "index_regime_entry": str(row["index_regime"]),
                "m_entry_idx": int(m_entry_idx),
            }
            picks += 1
            picked_this_month.add(ts_code)

        # Optional extra preferred-leader entries (beyond MONTHLY_TOP_N).
        if int(EXTRA_PREFERRED_PICKS) > 0 and PREFERRED_TS_CODES:
            extra = df_m[df_m["ts_code"].astype(str).isin(set(PREFERRED_TS_CODES))].copy()
            if not extra.empty:
                extra = extra[~extra["ts_code"].astype(str).isin(set(open_pos.keys()) | picked_this_month)].copy()
                extra = extra.sort_values("score", ascending=False)
                extra_added = 0
                for _, row in extra.iterrows():
                    if extra_added >= int(EXTRA_PREFERRED_PICKS):
                        break

                    ts_code = str(row["ts_code"])
                    if ts_code in open_pos:
                        continue

                    if len(open_pos) >= int(MAX_OPEN_POSITIONS):
                        if not _maybe_rotate_for_preferred(ts_code, exit_month=str(month)):
                            break

                    m = _get_monthly(ts_code)
                    if m is None or m.empty:
                        continue

                    entry_month_end = str(row.get("entry_month_end") or "")
                    if not entry_month_end:
                        continue
                    hit = m.index[m["month_end"].astype(str) == entry_month_end]
                    if len(hit) == 0:
                        continue
                    m_entry_idx = int(hit[0])

                    entry_date = str(row["entry_date"])

                    open_pos[ts_code] = {
                        "entry_date": entry_date,
                        "entry_price": float(row["entry_price"]),
                        "entry_score": float(row["score"]),
                        "index_regime_entry": str(row["index_regime"]),
                        "m_entry_idx": int(m_entry_idx),
                    }
                    extra_added += 1
                    picked_this_month.add(ts_code)

    # Force close remaining positions at END_DATE (approx: last available month_end close)
    for ts_code, pos in list(open_pos.items()):
        m = _get_monthly(ts_code)
        if m is None or m.empty:
            continue
        exit_idx = len(m) - 1
        exit_price = float(m.loc[exit_idx, "close"])
        entry_price = float(pos["entry_price"])
        ret_pct = (exit_price - entry_price) / entry_price * 100.0
        trades.append(
            {
                "ts_code": ts_code,
                "name": name_map.get(ts_code, ""),
                "entry_date": str(pos["entry_date"]),
                "exit_date": str(m.loc[exit_idx, "month_end"]),
                "hold_months": int(exit_idx - int(pos["m_entry_idx"]) + 1),
                "entry_score": round(float(pos["entry_score"]), 4),
                "index_regime_entry": str(pos["index_regime_entry"]),
                "exit_reason": "End Date",
                "actual_return(%)": round(float(ret_pct), 2),
            }
        )

    out = pd.DataFrame(trades)
    if out.empty:
        return out

    out = out.sort_values(["entry_date", "actual_return(%)"], ascending=[True, False]).reset_index(drop=True)
    return out


def main():
    start = time.time()

    # 1) Build index monthly regime mapping
    df_index = _load_index_daily()
    _, index_regime_by_date = build_index_monthly_regime_by_date(df_index)

    name_map = _load_basic_name_map()

    # 2) Build monthly candidate table (parallel)
    files = glob.glob(os.path.join(STOCK_DATA_DIR, "*.csv"))
    print(f"Found {len(files)} stock CSV files. Building monthly candidates...")

    with Pool(max(1, cpu_count() - 1)) as pool:
        rows = pool.map(
            _process_stock_monthly_candidates,
            [(fp, index_regime_by_date, name_map) for fp in files],
        )

    all_rows: list[dict] = []
    for r in rows:
        all_rows.extend(r)

    df_candidates = pd.DataFrame(all_rows)
    if df_candidates.empty:
        print("No monthly candidates.")
        return

    # 3) Simulate monthly portfolio
    trades = simulate_monthly_portfolio(df_candidates, index_regime_by_date, name_map)
    if trades.empty:
        print("No trades generated.")
        return

    trades.to_csv(TRADES_PATH, index=False)

    win_rate = float((trades["actual_return(%)"] > 0).mean() * 100.0)
    avg_ret = float(trades["actual_return(%)"].mean())
    med_ret = float(trades["actual_return(%)"].median())
    avg_hold = float(trades["hold_months"].mean())

    report = "\n".join(
        [
            "# 2025 相变策略：月线筛选与月线交易回测报告",
            "",
            f"* 回测区间: {START_DATE}..{END_DATE}",
            f"* 指数: {INDEX_TS_CODE}",
            f"* 月线入场: 每月 Top {MONTHLY_TOP_N}，最多持仓 {MAX_OPEN_POSITIONS}",
            f"* 入场阈值: score >= {MIN_ENTRY_SCORE}（龙头 {sorted(list(LEADER_TS_CODES))} 放宽到 {MIN_ENTRY_SCORE_LEADER}）",
            f"* 指数相位门控: 仅允许 {sorted(list(ALLOW_INDEX_REGIMES_FOR_ENTRY))} 入场；{'强制' if FORCE_EXIT_ON_INDEX_RISK else '不'}在 RISK 下退出",
            "",
            "## 汇总",
            f"* 交易数: {len(trades)}",
            f"* 胜率: {win_rate:.2f}%",
            f"* 平均收益: {avg_ret:.2f}%",
            f"* 中位数收益: {med_ret:.2f}%",
            f"* 平均持有(月): {avg_hold:.2f}",
            "",
            "## 交易明细（按收益排序 Top 50）",
            trades.sort_values("actual_return(%)", ascending=False).head(50).to_markdown(index=False),
            "",
            f"输出 CSV: {TRADES_PATH}",
            f"运行耗时: {time.time() - start:.2f} 秒",
        ]
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Done. Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
