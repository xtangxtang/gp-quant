import os
import glob
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from monthly_phase import (
    build_index_monthly_regime_by_date,
    calc_entropy,
    compute_intramonth_monthly_score_series,
    fast_hurst,
    rolling_apply_1d,
)


# ==============================
# Config (Monthly-view, daily-updated signals)
# ==============================
START_DATE = "20250101"  # signal date range (inclusive)
END_DATE = "20251230"    # last possible execution date

INDEX_DATA_PATH = "/nvme5/xtang/gp-workspace/gp-data/tushare-index-daily/sh000001.csv"
STOCK_DATA_DIR = "/nvme5/xtang/gp-workspace/gp-data/tushare-daily-full/"
BASIC_INFO_PATH = "/nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv"

REPORT_PATH = "/nvme5/xtang/gp-workspace/gp-quant/backtest_report_phase_transition_2025_monthly_intramonth_daily.md"
TRADES_PATH = "/nvme5/xtang/gp-workspace/gp-quant/backtest_trades_2025_monthly_intramonth_daily.csv"

MAX_OPEN_POSITIONS = 5
ENTRIES_PER_DAY = 1

# Require conditions to persist to avoid churn
ENTRY_PERSIST_DAYS = 3
EXIT_PERSIST_DAYS = 3

MIN_ENTRY_SCORE = 0.10
EXIT_SCORE = -0.10

# Score computation mode (see monthly_phase.compute_*): "full" or "thermo"
SCORE_MODE = "thermo"

# Strict thermo-only: Leader Early must not use price-action/MA conditions.
STRICT_THERMO_ONLY = True

# Index regime gating (daily mapping but monthly-regime semantics): BULL/BASE/RISK
ALLOW_INDEX_REGIMES_FOR_ENTRY = {"BULL", "BASE"}
FORCE_EXIT_ON_INDEX_RISK = True

# Preferred leaders
LEADER_TS_CODES = {"300502.SZ", "300394.SZ"}
MIN_ENTRY_SCORE_LEADER = 0.06

PREFERRED_TS_CODES = {"300502.SZ", "300394.SZ"}
PREFERRED_SCORE_BAND = 0.35
EXTRA_PREFERRED_PICKS = 1

# Reserve capacity for preferred leaders to reduce the need for forced rotations.
# If you want to be able to hold both 300502 and 300394 concurrently, set this to 2.
RESERVE_PREFERRED_TARGET = 2

# Rotation is powerful but can create churn in a daily-execution setup.
# Keep it off by default; enable only if you explicitly want forced leader-in.
ALLOW_PREFERRED_ROTATION = False
ROTATE_ONLY_IF_SCORE_AT_LEAST = 0.20

EXCLUDE_ST = True

# Leader Early (strict thermo-only): entropy + hurst on daily windows
LEADER_EARLY_ENABLE = True
LEADER_EARLY_ENTROPY_S_WINDOW = 12
LEADER_EARLY_ENTROPY_L_WINDOW = 36
LEADER_EARLY_HURST_S_WINDOW = 40
LEADER_EARLY_HURST_L_WINDOW = 80
LEADER_EARLY_ENT_REL_MIN = 0.02
LEADER_EARLY_HURST_REL_MIN = 0.08
LEADER_EARLY_HURST_S_MIN = 0.36
LEADER_EARLY_CONFIRM_WINDOW_DAYS = 3
LEADER_EARLY_CONFIRM_HITS = 2
LEADER_EARLY_MIN_HOLD_DAYS = 15
LEADER_EARLY_EXIT_ENT_REL_MAX = -0.01
LEADER_EARLY_EXIT_HURST_REL_MAX = 0.03
LEADER_EARLY_EXIT_HURST_S_MAX = 0.32
LEADER_EARLY_EXIT_PERSIST_DAYS = 18


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


def _process_stock_intramonth_signals(args):
    file_path, index_regime_by_date, name_map = args

    try:
        df = pd.read_csv(file_path)
    except Exception:
        return []

    if df.empty or "trade_date" not in df.columns or "ts_code" not in df.columns:
        return []

    ts_code = str(df["ts_code"].iloc[0])
    if EXCLUDE_ST and _is_st_stock(ts_code, name_map):
        return []

    df["trade_date_str"] = df["trade_date"].astype(str)
    df = df[(df["trade_date_str"] >= "20200101") & (df["trade_date_str"] <= END_DATE)].copy()
    if df.empty:
        return []

    # Ensure required cols for execution pricing
    if "open" not in df.columns or "close" not in df.columns:
        return []

    df = df.sort_values("trade_date_str").reset_index(drop=True)
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    # Exec is next trading day for this stock
    df_exec = df[["trade_date_str", "open", "close"]].copy()
    df_exec["exec_date"] = df_exec["trade_date_str"].shift(-1)
    df_exec["exec_open"] = df_exec["open"].shift(-1)

    df_exec = df_exec[
        (df_exec["trade_date_str"] >= START_DATE)
        & (df_exec["trade_date_str"] <= END_DATE)
        & (df_exec["exec_date"].notna())
    ].copy()

    if df_exec.empty:
        return []

    # Compute intramonth (daily-updated) monthly score as-of each signal day close.
    s = compute_intramonth_monthly_score_series(
        df,
        start_date=START_DATE,
        end_date=END_DATE,
        window_s=12,
        window_l=36,
        score_mode=str(SCORE_MODE),
    )
    if s is None or not isinstance(s, pd.DataFrame) or s.empty:
        return []

    merged = pd.merge(df_exec, s[["trade_date_str", "score"]], on="trade_date_str", how="inner")
    if merged.empty:
        return []

    merged = merged.sort_values("trade_date_str").reset_index(drop=True)
    merged["score"] = pd.to_numeric(merged["score"], errors="coerce")
    # Entry/exit persistence (rolling over this stock's trading days)
    if int(ENTRY_PERSIST_DAYS) > 1:
        merged["score_min_entry_persist"] = merged["score"].rolling(
            window=int(ENTRY_PERSIST_DAYS), min_periods=int(ENTRY_PERSIST_DAYS)
        ).min()
    else:
        merged["score_min_entry_persist"] = merged["score"]

    if int(EXIT_PERSIST_DAYS) > 1:
        merged["score_max_exit_persist"] = merged["score"].rolling(
            window=int(EXIT_PERSIST_DAYS), min_periods=int(EXIT_PERSIST_DAYS)
        ).max()
    else:
        merged["score_max_exit_persist"] = merged["score"]

    merged["leader_early_ok"] = False
    merged["leader_early_exit_persist"] = False

    # Leader Early (thermo-only): only compute for leaders to keep runtime low.
    if LEADER_EARLY_ENABLE and (ts_code in set(LEADER_TS_CODES)):
        close_arr = df["close"].astype("float64").to_numpy(dtype=np.float64)
        ret_d1 = pd.Series(close_arr).pct_change(1).to_numpy(dtype=np.float64)

        ent_s = rolling_apply_1d(ret_d1, int(LEADER_EARLY_ENTROPY_S_WINDOW), calc_entropy)
        ent_l = rolling_apply_1d(ret_d1, int(LEADER_EARLY_ENTROPY_L_WINDOW), calc_entropy)
        ent_rel = (ent_l - ent_s) / np.where(ent_l == 0, np.nan, ent_l)

        h_s = rolling_apply_1d(close_arr, int(LEADER_EARLY_HURST_S_WINDOW), fast_hurst)
        h_l = rolling_apply_1d(close_arr, int(LEADER_EARLY_HURST_L_WINDOW), fast_hurst)
        h_s = np.clip(h_s, 0.0, 1.0)
        h_l = np.clip(h_l, 0.0, 1.0)
        h_rel = h_s - h_l

        df_leader = pd.DataFrame(
            {
                "trade_date_str": df["trade_date_str"].astype(str),
                "leader_ent_rel": ent_rel,
                "leader_hurst_s": h_s,
                "leader_hurst_rel": h_rel,
            }
        )
        merged = pd.merge(merged, df_leader, on="trade_date_str", how="left")

        early_raw = (
            (pd.to_numeric(merged["leader_ent_rel"], errors="coerce") >= float(LEADER_EARLY_ENT_REL_MIN))
            & (pd.to_numeric(merged["leader_hurst_rel"], errors="coerce") >= float(LEADER_EARLY_HURST_REL_MIN))
            & (pd.to_numeric(merged["leader_hurst_s"], errors="coerce") >= float(LEADER_EARLY_HURST_S_MIN))
        ).astype(int)

        w = int(LEADER_EARLY_CONFIRM_WINDOW_DAYS)
        k = int(LEADER_EARLY_CONFIRM_HITS)
        if w <= 1:
            merged["leader_early_ok"] = early_raw >= k
        else:
            merged["leader_early_ok"] = early_raw.rolling(window=w, min_periods=w).sum().fillna(0) >= k

        # Exit only when BOTH entropy and hurst trend deteriorate, and short-window hurst collapses.
        break_raw = (
            (pd.to_numeric(merged["leader_ent_rel"], errors="coerce") < float(LEADER_EARLY_EXIT_ENT_REL_MAX))
            & (pd.to_numeric(merged["leader_hurst_rel"], errors="coerce") < float(LEADER_EARLY_EXIT_HURST_REL_MAX))
            & (pd.to_numeric(merged["leader_hurst_s"], errors="coerce") < float(LEADER_EARLY_EXIT_HURST_S_MAX))
        ).astype(int)
        merged["leader_early_exit_persist"] = (
            break_raw.rolling(
                window=int(LEADER_EARLY_EXIT_PERSIST_DAYS),
                min_periods=int(LEADER_EARLY_EXIT_PERSIST_DAYS),
            ).min().fillna(0)
            >= 1
        )

    merged["index_regime"] = merged["trade_date_str"].map(index_regime_by_date).fillna("BASE")
    merged["ts_code"] = ts_code
    merged["name"] = name_map.get(ts_code, "")

    # Guard: execution date must be within END_DATE
    merged["exec_date"] = merged["exec_date"].astype(str)
    merged = merged[merged["exec_date"] <= END_DATE].copy()
    if merged.empty:
        return []

    out = []
    for _, r in merged.iterrows():
        exec_open = float(r["exec_open"]) if np.isfinite(r["exec_open"]) else np.nan
        if not np.isfinite(exec_open) or exec_open <= 0:
            continue
        score = float(r["score"]) if np.isfinite(r["score"]) else np.nan
        if not np.isfinite(score):
            continue
        out.append(
            {
                "ts_code": ts_code,
                "name": str(r.get("name") or ""),
                "signal_date": str(r["trade_date_str"]),
                "exec_date": str(r["exec_date"]),
                "exec_open": float(exec_open),
                "signal_close": float(r["close"]) if np.isfinite(r["close"]) else np.nan,
                "score": float(score),
                "score_min_entry_persist": float(r.get("score_min_entry_persist"))
                if np.isfinite(r.get("score_min_entry_persist"))
                else np.nan,
                "score_max_exit_persist": float(r.get("score_max_exit_persist"))
                if np.isfinite(r.get("score_max_exit_persist"))
                else np.nan,
                "leader_early_ok": bool(r.get("leader_early_ok")) if pd.notna(r.get("leader_early_ok")) else False,
                "leader_early_exit_persist": bool(r.get("leader_early_exit_persist"))
                if pd.notna(r.get("leader_early_exit_persist"))
                else False,
                "index_regime": str(r["index_regime"]),
            }
        )

    return out


def simulate_portfolio(df_signals: pd.DataFrame) -> pd.DataFrame:
    if df_signals is None or not isinstance(df_signals, pd.DataFrame) or df_signals.empty:
        return pd.DataFrame()

    df = df_signals.copy()
    for c in ["ts_code", "signal_date", "exec_date", "index_regime"]:
        df[c] = df[c].astype(str)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["exec_open"] = pd.to_numeric(df["exec_open"], errors="coerce")
    if "score_min_entry_persist" in df.columns:
        df["score_min_entry_persist"] = pd.to_numeric(df["score_min_entry_persist"], errors="coerce")
    else:
        df["score_min_entry_persist"] = df["score"]
    if "score_max_exit_persist" in df.columns:
        df["score_max_exit_persist"] = pd.to_numeric(df["score_max_exit_persist"], errors="coerce")
    else:
        df["score_max_exit_persist"] = df["score"]
    if "leader_early_ok" in df.columns:
        df["leader_early_ok"] = df["leader_early_ok"].astype(bool)
    else:
        df["leader_early_ok"] = False
    if "leader_early_exit_persist" in df.columns:
        df["leader_early_exit_persist"] = df["leader_early_exit_persist"].astype(bool)
    else:
        df["leader_early_exit_persist"] = False

    # MultiIndex for quick lookups: (ts_code, exec_date) -> signal row executed today
    df_exec_idx = df.set_index(["ts_code", "exec_date"]).sort_index()

    # Group rows by execution day (the day we can actually trade at open)
    exec_days = sorted(df["exec_date"].unique().tolist())

    open_pos: dict[str, dict[str, object]] = {}
    trades: list[dict] = []

    leader_set = set(LEADER_TS_CODES) if LEADER_TS_CODES else set()
    preferred_set = set(PREFERRED_TS_CODES) if PREFERRED_TS_CODES else set()

    def _close_position(ts_code: str, exec_day: str, exit_open: float, reason: str):
        pos = open_pos.get(ts_code)
        if not pos:
            return
        entry_price = float(pos["entry_price"])
        ret_pct = (float(exit_open) - entry_price) / entry_price * 100.0
        trades.append(
            {
                "ts_code": ts_code,
                "name": str(pos.get("name") or ""),
                "entry_date": str(pos["entry_date"]),
                "exit_date": str(exec_day),
                "hold_days": int(pos.get("hold_days", 0)),
                "entry_score": round(float(pos.get("entry_score") or np.nan), 4),
                "index_regime_entry": str(pos.get("index_regime_entry") or ""),
                "entry_mode": str(pos.get("entry_mode") or ""),
                "exit_reason": str(reason),
                "actual_return(%)": round(float(ret_pct), 2),
            }
        )
        del open_pos[ts_code]

    def _maybe_rotate_for_preferred(new_ts_code: str, exec_day: str) -> bool:
        if not ALLOW_PREFERRED_ROTATION:
            return False
        if str(new_ts_code) not in preferred_set:
            return False
        if float(df_exec_idx.loc[(str(new_ts_code), str(exec_day))].get("score", np.nan)) < float(ROTATE_ONLY_IF_SCORE_AT_LEAST):
            return False
        if len(open_pos) < int(MAX_OPEN_POSITIONS):
            return True

        non_pref = [
            (ts, float(pos.get("entry_score", np.inf)))
            for ts, pos in open_pos.items()
            if str(ts) not in preferred_set
        ]
        if not non_pref:
            return False
        ts_to_close = sorted(non_pref, key=lambda x: x[1])[0][0]

        # Use today's open for that stock if available; otherwise can't rotate.
        try:
            row = df_exec_idx.loc[(ts_to_close, str(exec_day))]
        except Exception:
            return False
        px = float(row["exec_open"]) if np.isfinite(row["exec_open"]) else np.nan
        if not np.isfinite(px) or px <= 0:
            return False
        _close_position(ts_to_close, exec_day=str(exec_day), exit_open=px, reason="Rotation: Preferred Leader In")
        return True

    for exec_day in exec_days:
        # Update hold days
        for ts in list(open_pos.keys()):
            open_pos[ts]["hold_days"] = int(open_pos[ts].get("hold_days", 0)) + 1

        # Today's actionable signal rows (their decisions were made at each stock's previous close)
        today = df[df["exec_date"] == str(exec_day)].copy()
        if today.empty:
            continue

        # 1) Exits
        to_close: list[tuple[str, str]] = []
        for ts_code in list(open_pos.keys()):
            try:
                r = df_exec_idx.loc[(ts_code, str(exec_day))]
            except Exception:
                continue

            idx_reg = str(r.get("index_regime") or "BASE")
            sc_exit = float(r.get("score_max_exit_persist")) if np.isfinite(r.get("score_max_exit_persist")) else np.nan
            entry_mode = str(open_pos.get(ts_code, {}).get("entry_mode") or "")
            hold_days = int(open_pos.get(ts_code, {}).get("hold_days") or 0)

            if FORCE_EXIT_ON_INDEX_RISK and idx_reg == "RISK":
                to_close.append((ts_code, "Index Regime RISK"))
            elif entry_mode == "LEADER_EARLY":
                # For early entries (strict thermo-only): after a minimum hold, exit on thermo trend break persistence.
                if hold_days < int(LEADER_EARLY_MIN_HOLD_DAYS):
                    continue
                break_persist = bool(r.get("leader_early_exit_persist"))
                if break_persist:
                    to_close.append((ts_code, "Thermo Trend Exit"))
            else:
                if np.isfinite(sc_exit) and float(sc_exit) < float(EXIT_SCORE):
                    to_close.append((ts_code, "Intramonth Monthly Score Exit"))

        for ts_code, reason in to_close:
            try:
                r = df_exec_idx.loc[(ts_code, str(exec_day))]
            except Exception:
                continue
            px = float(r["exec_open"]) if np.isfinite(r["exec_open"]) else np.nan
            if not np.isfinite(px) or px <= 0:
                continue
            _close_position(ts_code, exec_day=str(exec_day), exit_open=px, reason=reason)

        # 2) Entries
        # Thresholding (leader-specific cut)
        is_leader = today["ts_code"].isin(leader_set)
        sc_entry = today["score_min_entry_persist"]
        leader_early_ok = today["leader_early_ok"] if (LEADER_EARLY_ENABLE and "leader_early_ok" in today.columns) else False

        base_ok = sc_entry >= float(MIN_ENTRY_SCORE)
        leader_ok = is_leader & (sc_entry >= float(MIN_ENTRY_SCORE_LEADER))
        early_ok = is_leader & leader_early_ok

        today = today[base_ok | leader_ok | early_ok].copy()
        if ALLOW_INDEX_REGIMES_FOR_ENTRY:
            today = today[today["index_regime"].isin(set(ALLOW_INDEX_REGIMES_FOR_ENTRY))].copy()
        if today.empty:
            continue

        today = today.sort_values("score", ascending=False).reset_index(drop=True)

        # Preferred leader override for the first entry of the day
        if preferred_set:
            try:
                best_score = float(today.loc[0, "score"])
                pref = today[today["ts_code"].astype(str).isin(preferred_set)].copy()
                if not pref.empty:
                    pref = pref[pref["score"] >= (best_score - float(PREFERRED_SCORE_BAND))].copy()
                    if not pref.empty:
                        pref = pref.sort_values("score", ascending=False).head(1)
                        others = today[~today.index.isin(pref.index)].copy()
                        today = pd.concat([pref, others], ignore_index=True)
            except Exception:
                pass

        entries = 0
        picked: set[str] = set()
        for _, r in today.iterrows():
            if entries >= int(ENTRIES_PER_DAY):
                break
            ts_code = str(r["ts_code"])

            # Reserve capacity for preferred leaders (avoid filling slots needed for preferred).
            if int(RESERVE_PREFERRED_TARGET) > 0 and (ts_code not in preferred_set):
                preferred_held = sum(1 for x in open_pos.keys() if str(x) in preferred_set)
                needed = max(0, int(RESERVE_PREFERRED_TARGET) - int(preferred_held))
                remaining = int(MAX_OPEN_POSITIONS) - int(len(open_pos))
                if remaining <= needed:
                    continue

            if len(open_pos) >= int(MAX_OPEN_POSITIONS):
                # allow preferred rotation; otherwise stop
                if not _maybe_rotate_for_preferred(ts_code, exec_day=str(exec_day)):
                    break
            if ts_code in open_pos or ts_code in picked:
                continue

            px = float(r["exec_open"]) if np.isfinite(r["exec_open"]) else np.nan
            if not np.isfinite(px) or px <= 0:
                continue

            open_pos[ts_code] = {
                "name": str(r.get("name") or ""),
                "entry_date": str(exec_day),
                "entry_price": float(px),
                "entry_score": float(r.get("score")) if np.isfinite(r.get("score")) else np.nan,
                "index_regime_entry": str(r.get("index_regime") or ""),
                "entry_mode": "LEADER_EARLY"
                if (LEADER_EARLY_ENABLE and bool(r.get("leader_early_ok")) and (float(r.get("score_min_entry_persist") or 0.0) < float(MIN_ENTRY_SCORE_LEADER)))
                else "NORMAL",
                "hold_days": 0,
            }
            picked.add(ts_code)
            entries += 1

        # Optional extra preferred picks (beyond ENTRIES_PER_DAY)
        if int(EXTRA_PREFERRED_PICKS) > 0 and preferred_set:
            extra = today[today["ts_code"].astype(str).isin(preferred_set)].copy()
            extra = extra[~extra["ts_code"].astype(str).isin(set(open_pos.keys()) | picked)].copy()
            extra = extra.sort_values("score", ascending=False)
            extra_added = 0
            for _, r in extra.iterrows():
                if extra_added >= int(EXTRA_PREFERRED_PICKS):
                    break
                if len(open_pos) >= int(MAX_OPEN_POSITIONS):
                    if not _maybe_rotate_for_preferred(str(r["ts_code"]), exec_day=str(exec_day)):
                        break

                ts_code = str(r["ts_code"])
                if ts_code in open_pos or ts_code in picked:
                    continue

                px = float(r["exec_open"]) if np.isfinite(r["exec_open"]) else np.nan
                if not np.isfinite(px) or px <= 0:
                    continue

                open_pos[ts_code] = {
                    "name": str(r.get("name") or ""),
                    "entry_date": str(exec_day),
                    "entry_price": float(px),
                    "entry_score": float(r.get("score")) if np.isfinite(r.get("score")) else np.nan,
                    "index_regime_entry": str(r.get("index_regime") or ""),
                    "entry_mode": "LEADER_EARLY"
                    if (LEADER_EARLY_ENABLE and bool(r.get("leader_early_ok")) and (float(r.get("score_min_entry_persist") or 0.0) < float(MIN_ENTRY_SCORE_LEADER)))
                    else "NORMAL",
                    "hold_days": 0,
                }
                picked.add(ts_code)
                extra_added += 1

    # Force close at each stock's last available exec day (open)
    last_exec = df.sort_values(["ts_code", "exec_date"]).groupby("ts_code", as_index=False).tail(1)
    last_by_code = {
        str(r["ts_code"]): (str(r["exec_date"]), float(r["exec_open"]))
        for _, r in last_exec.iterrows()
        if np.isfinite(r["exec_open"]) and float(r["exec_open"]) > 0
    }

    for ts_code in list(open_pos.keys()):
        if ts_code not in last_by_code:
            continue
        d, px = last_by_code[ts_code]
        _close_position(ts_code, exec_day=str(d), exit_open=float(px), reason="End Date")

    out = pd.DataFrame(trades)
    if out.empty:
        return out
    out = out.sort_values(["entry_date", "actual_return(%)"], ascending=[True, False]).reset_index(drop=True)
    return out


def main():
    start = time.time()

    df_index = _load_index_daily()
    _, index_regime_by_date = build_index_monthly_regime_by_date(df_index)
    name_map = _load_basic_name_map()

    files = glob.glob(os.path.join(STOCK_DATA_DIR, "*.csv"))
    print(f"Found {len(files)} stock CSV files. Building intramonth monthly signals...")

    with Pool(max(1, cpu_count() - 1)) as pool:
        rows = pool.map(
            _process_stock_intramonth_signals,
            [(fp, index_regime_by_date, name_map) for fp in files],
        )

    all_rows: list[dict] = []
    for r in rows:
        all_rows.extend(r)

    df_signals = pd.DataFrame(all_rows)
    if df_signals.empty:
        print("No signals generated.")
        return

    trades = simulate_portfolio(df_signals)
    if trades.empty:
        print("No trades generated.")
        return

    trades.to_csv(TRADES_PATH, index=False)

    # Minimal report
    leaders = sorted(list(LEADER_TS_CODES))
    hits = trades[trades["ts_code"].astype(str).isin(set(leaders))]

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Monthly-View Intramonth Daily Backtest (2025)\n\n")
        f.write(f"- Trades: {len(trades)}\n")
        f.write(f"- Max positions: {MAX_OPEN_POSITIONS}\n")
        f.write(f"- Entries/day: {ENTRIES_PER_DAY} (+ preferred extra {EXTRA_PREFERRED_PICKS})\n")
        f.write(f"- Entry threshold: {MIN_ENTRY_SCORE} (leaders {MIN_ENTRY_SCORE_LEADER})\n")
        f.write(f"- Exit threshold: {EXIT_SCORE}\n")
        f.write(f"- Allow entry regimes: {sorted(list(ALLOW_INDEX_REGIMES_FOR_ENTRY))}\n")
        f.write(f"- Force exit on RISK: {FORCE_EXIT_ON_INDEX_RISK}\n\n")

        f.write("## Leader Hits\n\n")
        if hits.empty:
            f.write("- NO_HITS\n")
        else:
            f.write(hits.to_markdown(index=False))
            f.write("\n")

    elapsed = time.time() - start
    print(f"Done. Trades: {TRADES_PATH}")
    print(f"Report: {REPORT_PATH}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
