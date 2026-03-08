import glob
import os
import time
import sys
from pathlib import Path
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

# Allow importing sibling modules (e.g., monthly_phase.py) when executed via wrappers.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from monthly_phase import build_index_monthly_regime_by_date

# ==============================
# Config
# ==============================
TRAIN_START = "20230101"
TRAIN_END = "20241231"

START_DATE = "20250101"
END_DATE = "20251230"

# Prob horizons (trading days after entry)
HORIZONS = (5, 20)

INDEX_TS_CODE = "000001.SH"  # 上证指数
INDEX_DATA_PATH = "/nvme5/xtang/gp-workspace/gp-data/tushare-index-daily/sh000001.csv"

STOCK_DATA_DIR = "/nvme5/xtang/gp-workspace/gp-data/tushare-daily-full/"
BASIC_INFO_PATH = "/nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv"

REPORT_PATH = "/nvme5/xtang/gp-workspace/gp-quant/backtest_report_phase_transition_2025_index_prob_weighted.md"
GRID_CSV_PATH = "/nvme5/xtang/gp-workspace/gp-quant/backtest_grid_2025_index_prob_weighted.csv"
BEST_TRADES_PATH = "/nvme5/xtang/gp-workspace/gp-quant/backtest_trades_2025_index_prob_weighted.csv"

# Score binning
N_BINS = 15
SCORE_MIN = -1.0
SCORE_MAX = 1.0

# Weighting between horizons: equal mix
HORIZON_MIX = {5: 0.5, 20: 0.5}

# Weighted exit threshold: exit only if combined score is sufficiently negative.
# (A small negative threshold avoids over-frequent exits caused by tiny score flips.)
WEIGHTED_EXIT_THRESHOLD = -0.10

# Monthly index regime gating (state machine): optionally use the coarse index
# regime (BULL/BASE/RISK) to modulate entry strictness.
INDEX_REGIME_GATING_ENABLED = True
# Additive adjustment to the entry-score threshold by index regime.
# Example: in RISK, require higher entry_score; in BULL, allow slightly lower.
INDEX_REGIME_MIN_ENTRY_ADJ = {"BULL": -0.02, "BASE": 0.0, "RISK": 0.05}
# Optional: avoid ultra-early leader entries when index regime is risk-off.
INDEX_REGIME_BLOCK_LEADER_EARLY_IN_RISK = True

# Portfolio-style constraints to reduce over-trading and focus on a few big cycles.
# - Only take Top-N signals per day (by entry_score).
# - Limit max concurrent open positions.
# - Require a minimum entry_score to trade at all.
DAILY_TOP_N = 1
MAX_OPEN_POSITIONS = 5
MIN_ENTRY_SCORE = 0.10

# Leader-specific entry threshold: allows leader candidates slightly below global cutoff
# without increasing overall trade frequency too much.
MIN_ENTRY_SCORE_LEADER = 0.08

# Entry filters (candidate generation)
# User request: "明显放宽：允许更多候选，再靠概率分数筛"
# These are applied consistently in:
# - training calibration rows (2023-2024 triggered signals)
# - evaluation candidate generation (2025)
ENTRY_PCT_CHG_MIN = 1.5
ENTRY_CLOSE_MA60_RATIO = 0.98
ENTRY_MF5D_MIN = 0.0

# Leader early-phase entry (post-crash reversal / near-critical transition)
# Goal: allow earlier entry for a small leader watchlist even when price is still
# far below MA60, as long as short-term order improves and capital flow stabilizes.
EARLY_LEADER_ENTRY_ENABLED = True
EARLY_LEADER_PCT_CHG_MIN = 0.5
EARLY_LEADER_CRASH_LOOKBACK = 12
EARLY_LEADER_CRASH_MIN_PCT_CHG = -15.0
EARLY_LEADER_CLOSE_MA60_MIN = 0.70
EARLY_LEADER_CLOSE_MA60_MAX = 0.98
EARLY_LEADER_MF5D_Z20_MIN = 0.0
EARLY_LEADER_HURST20_MIN = 0.40
EARLY_LEADER_HURST20_OVER60_MIN = 0.10
EARLY_LEADER_ENTROPY20_OVER60_MAX = 0.20

# Score gate for leader early-phase entries. Keep this low because the probabilistic
# score is often pessimistic right after a crash, even if the regime is turning.
MIN_ENTRY_SCORE_LEADER_EARLY = -0.20

# Phase-transition constraints (relaxed)
ENTRY_HURST60_MIN = 0.15
ENTRY_HURST60_MAX = 0.70
ENTRY_HURST20_MIN = 0.40
ENTRY_ENTROPY20_OVER60_MAX = 0.25

# Relaxation of strict Top1/day (Mode B):
# Keep 1 buy/day, but allow picking a preferred "leader" candidate even if it's not the
# absolute Top1, as long as it is close to the day's best score.
PREFERRED_TS_CODES = {"300502.SZ", "300394.SZ"}  # 新易盛 / 天孚通信; set to empty set() to disable
PREFERRED_SCORE_BAND = 0.15  # allow preferred if leader_score >= day_best_score - band

# Leader-friendly policies (for a small watchlist)
LEADER_TS_CODES = {"300502.SZ", "300394.SZ"}

# Portfolio behavior
LEADER_PROTECT_FROM_ROTATION = True
LEADER_LOCK_BARS = 20  # if protecting, do not rotate leaders out before this many bars held
LEADER_FORCE_ENTRY_ON_FULL = True  # if leader appears while full, allow replacing a non-leader holding

# Selection preference behavior
LEADER_IGNORE_PREFERRED_BAND = True  # if True, leaders can be preferred regardless of day_best score

# Exit behavior (trend exit driven by entropy/Hurst/"energy")
LEADER_TREND_EXIT = True
LEADER_TREND_MIN_HOLD_BARS = 10
LEADER_TREND_MA20_BREAK_RATIO = 0.98
LEADER_TREND_HURST20_MIN = 0.40
LEADER_TREND_ENTROPY20_OVER60 = 0.25
LEADER_TREND_MF5D_MIN = 0.0

# Disable early profit-taking/exits for leaders to better capture big cycles
LEADER_DISABLE_ENTROPY_SPIKE_TAKE_PROFIT = True
LEADER_DISABLE_WEIGHTED_EXIT = True

# Leader early-phase (post-crash) position management
# Rationale: right after a crash, price often stays far below MA20; using MA20-based
# stop loss will instantly exit. Use a wider return stop and a longer minimum hold
# before applying the multi-confirmation trend-exit.
LEADER_EARLY_STOP_LOSS_RET = -0.12
LEADER_EARLY_TREND_MIN_HOLD_BARS = 90

# Optional universe hygiene: exclude ST/*ST names from selection (reduces noisy picks).
EXCLUDE_ST = True

# Rotation (replacement) when portfolio is full.
# Goal: keep trade count low but avoid missing strong leaders that appear while fully invested.
ROTATE_ON_FULL = True
# More aggressive by default: allow replacing the weakest holding even if the candidate is
# slightly worse on entry_score, as long as it is within a margin.
ROTATE_SCORE_MARGIN = -0.20    # allow candidate_score - worst_score >= this (negative => candidate can be worse)
ROTATE_MIN_HOLD_BARS = 5       # avoid churning positions too quickly (approx. trading bars)

# Globals for multiprocessing workers
INDEX_S_BY_DATE: dict[str, float] = {}
INDEX_REGIME_BY_DATE: dict[str, str] = {}
STOCK_CALIB = None  # set after calibration
WEIGHT_INDEX: float = 0.0
WEIGHT_STOCK: float = 1.0


# ==============================
# Utilities
# ==============================

def _import_tushare():
    try:
        import tushare as ts  # type: ignore

        return ts
    except Exception as e:
        raise RuntimeError("Python package 'tushare' is required.") from e


def _get_pro_api():
    """Get tushare pro client.

    Reads token from env var to avoid relying on local token files.
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
    need_start = "20220101"  # buffer for rolling + training

    if os.path.exists(INDEX_DATA_PATH):
        df = pd.read_csv(INDEX_DATA_PATH)
        if not df.empty and "trade_date" in df.columns:
            min_d = str(df["trade_date"].astype(str).min())
            max_d = str(df["trade_date"].astype(str).max())
            if min_d <= need_start and max_d >= END_DATE:
                df["trade_date_str"] = df["trade_date"].astype(str)
                return df.sort_values("trade_date_str").reset_index(drop=True)

    # Fetch full range if missing or insufficient
    fetch_index_daily_to_csv(INDEX_TS_CODE, need_start, END_DATE, INDEX_DATA_PATH)
    df = pd.read_csv(INDEX_DATA_PATH)
    if df.empty or "trade_date" not in df.columns:
        raise RuntimeError(f"Index CSV invalid: {INDEX_DATA_PATH}")
    df["trade_date_str"] = df["trade_date"].astype(str)
    return df.sort_values("trade_date_str").reset_index(drop=True)


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


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def score_to_bin(score: float) -> int:
    score = _clip(score, SCORE_MIN, SCORE_MAX)
    # map [-1,1] -> [0, N_BINS)
    pos = (score - SCORE_MIN) / (SCORE_MAX - SCORE_MIN)
    b = int(pos * N_BINS)
    if b == N_BINS:
        b = N_BINS - 1
    return b


def bin_to_center(bin_id: int) -> float:
    width = (SCORE_MAX - SCORE_MIN) / N_BINS
    return SCORE_MIN + (bin_id + 0.5) * width


def signed_from_p(p_up: float) -> float:
    """Map probability to a signed direction score in [-1, 1].

    NOTE: This is a mathematically valid mapping, but if a strategy's baseline
    win-rate is < 50%, then most values will be negative.
    """

    p_up = _clip(float(p_up), 0.0, 1.0)
    return float(2.0 * p_up - 1.0)


def rel_score_from_p(p_up: float, p_base: float) -> float:
    """Relative (excess) probability score in [-1, 1].

    Centers around training global base rate p_base so that 0 means
    "better than baseline" thresholding is meaningful even if p_base < 0.5.
    """

    p_up = _clip(float(p_up), 0.0, 1.0)
    p_base = _clip(float(p_base), 1e-6, 1.0 - 1e-6)
    denom = max(p_base, 1.0 - p_base)
    return float(_clip((p_up - p_base) / denom, -1.0, 1.0))


@dataclass(frozen=True)
class CalibTable:
    # per horizon: arrays shape (N_BINS,)
    p_up_by_bin: dict[int, np.ndarray]
    global_p_up: dict[int, float]

    # Optional diagnostics
    pos_by_bin: dict[int, np.ndarray] | None = None
    tot_by_bin: dict[int, np.ndarray] | None = None
    global_pos: dict[int, float] | None = None
    global_tot: dict[int, float] | None = None

    def p_up(self, horizon: int, score: float) -> float:
        b = score_to_bin(score)
        arr = self.p_up_by_bin[horizon]
        val = float(arr[b])
        if not np.isfinite(val):
            return float(self.global_p_up[horizon])
        return val


def build_calibration_from_bins(rows: list[dict], horizons: tuple[int, ...]) -> CalibTable:
    # Laplace smoothing
    p_up_by_bin: dict[int, np.ndarray] = {}
    global_p_up: dict[int, float] = {}

    pos_by_bin: dict[int, np.ndarray] = {}
    tot_by_bin: dict[int, np.ndarray] = {}
    global_pos: dict[int, float] = {}
    global_tot: dict[int, float] = {}

    for h in horizons:
        pos = np.zeros(N_BINS, dtype=np.float64)
        tot = np.zeros(N_BINS, dtype=np.float64)
        gpos = 0.0
        gtot = 0.0

        for r in rows:
            b = int(r["bin"])
            y = int(r[f"y_up_{h}"])
            tot[b] += 1.0
            pos[b] += float(y)
            gtot += 1.0
            gpos += float(y)

        # Laplace: (pos+1)/(tot+2)
        p = (pos + 1.0) / (tot + 2.0)
        p_up_by_bin[h] = p
        global_p_up[h] = float((gpos + 1.0) / (gtot + 2.0)) if gtot > 0 else 0.5

        pos_by_bin[h] = pos
        tot_by_bin[h] = tot
        global_pos[h] = float(gpos)
        global_tot[h] = float(gtot)

    return CalibTable(
        p_up_by_bin=p_up_by_bin,
        global_p_up=global_p_up,
        pos_by_bin=pos_by_bin,
        tot_by_bin=tot_by_bin,
        global_pos=global_pos,
        global_tot=global_tot,
    )


def _calib_diag_tables_md(calib: CalibTable, name: str) -> str:
    lines: list[str] = []
    lines.append(f"### {name}")

    # Global base rates
    for h in HORIZONS:
        p_base = float(calib.global_p_up[h])
        n = None
        if calib.global_tot is not None:
            n = int(calib.global_tot.get(h, 0.0))
        if n is None:
            lines.append(f"* p_base_{h}: {p_base:.4f}")
        else:
            lines.append(f"* p_base_{h}: {p_base:.4f} (n={n})")

    # Per-bin diagnostics tables
    for h in HORIZONS:
        rows: list[dict] = []
        for b in range(N_BINS):
            n = np.nan
            win_emp = np.nan
            if calib.tot_by_bin is not None and calib.pos_by_bin is not None:
                tot = float(calib.tot_by_bin[h][b])
                pos = float(calib.pos_by_bin[h][b])
                n = tot
                win_emp = (pos / tot) if tot > 0 else np.nan

            rows.append(
                {
                    "horizon": h,
                    "bin": b,
                    "score_center": round(float(bin_to_center(b)), 4),
                    "n": int(n) if np.isfinite(n) else 0,
                    "win_rate_emp": round(float(win_emp), 4) if np.isfinite(win_emp) else np.nan,
                    "p_up_smoothed": round(float(calib.p_up_by_bin[h][b]), 4),
                }
            )

        df = pd.DataFrame(rows)
        lines.append(f"\n#### {name} - Horizon {h}\n")
        lines.append(df.to_markdown(index=False))

    return "\n".join(lines)


# ==============================
# Index probability score
# ==============================

def index_feature_score(
    close: float,
    pct_chg: float,
    ma60: float,
    hurst_60: float,
    hurst_20: float,
    entropy_60: float,
    entropy_20: float,
) -> float:
    # bounded terms in [-1,1]
    hurst_term = _clip((hurst_20 - 0.45) / 0.2, -1.0, 1.0)
    chaos_term = 1.0 - _clip(abs(hurst_60 - 0.4) / 0.2, 0.0, 1.0)

    e_base = entropy_60 if np.isfinite(entropy_60) and entropy_60 > 0 else 1.0
    entropy_term = _clip((entropy_60 - entropy_20) / e_base, -1.0, 1.0)

    ma_term = 0.0
    if np.isfinite(ma60) and ma60 > 0:
        ma_term = _clip((close / ma60 - 1.0) / 0.1, -1.0, 1.0)

    pct_term = _clip((pct_chg - 0.5) / 3.0, -1.0, 1.0)

    score = 0.30 * hurst_term + 0.25 * entropy_term + 0.20 * ma_term + 0.15 * pct_term + 0.10 * chaos_term
    return float(_clip(score, SCORE_MIN, SCORE_MAX))


def build_index_calibration_and_scores(df_index: pd.DataFrame) -> tuple[CalibTable, dict[str, float]]:
    closes = pd.to_numeric(df_index["close"], errors="coerce").values
    pct_chg = pd.to_numeric(df_index["pct_chg"], errors="coerce").values
    dates = df_index["trade_date_str"].astype(str).values

    ma60 = pd.Series(closes).rolling(window=60).mean().values

    # Build per-day feature score (need at least 60 bars)
    scores = np.full(len(df_index), np.nan, dtype=np.float64)
    for i in range(60, len(df_index)):
        d = str(dates[i])
        if d < "20220101" or d > END_DATE:
            continue

        w60 = closes[i - 60 : i + 1]
        w20 = closes[i - 20 : i + 1]
        r60 = pct_chg[i - 60 : i + 1]
        r20 = pct_chg[i - 20 : i + 1]

        h60 = fast_hurst(w60)
        h20 = fast_hurst(w20)
        e60 = calc_entropy(r60)
        e20 = calc_entropy(r20)

        scores[i] = index_feature_score(float(closes[i]), float(pct_chg[i]), float(ma60[i]), h60, h20, e60, e20)

    # Build training rows for calibration
    rows: list[dict] = []

    # last idx within TRAIN_END
    end_train_idx = int(np.searchsorted(dates, TRAIN_END, side="right") - 1)
    for i in range(60, min(len(df_index), end_train_idx + 1)):
        d = str(dates[i])
        if d < TRAIN_START or d > TRAIN_END:
            continue
        if not np.isfinite(scores[i]):
            continue

        ok = True
        y_labels = {}
        for h in HORIZONS:
            j = i + h
            if j > end_train_idx:
                ok = False
                break
            # label: close[t+h] vs close[t]
            y_labels[h] = int(float(closes[j]) > float(closes[i]))
        if not ok:
            continue

        r = {"bin": score_to_bin(float(scores[i]))}
        for h in HORIZONS:
            r[f"y_up_{h}"] = y_labels[h]
        rows.append(r)

    calib = build_calibration_from_bins(rows, HORIZONS)

    # Build eval-date relative score s = mix_h rel_score(p, p_base)
    s_by_date: dict[str, float] = {}
    for i in range(60, len(df_index)):
        d = str(dates[i])
        if d < START_DATE or d > END_DATE:
            continue
        if not np.isfinite(scores[i]):
            continue

        s_mix = 0.0
        for h, w in HORIZON_MIX.items():
            p = calib.p_up(h, float(scores[i]))
            s_mix += float(w) * rel_score_from_p(p, calib.global_p_up[h])

        s_by_date[d] = float(_clip(s_mix, -1.0, 1.0))

    return calib, s_by_date


# ==============================
# Stock probability score
# ==============================

def stock_feature_score(
    close: float,
    pct_chg: float,
    ma60: float,
    hurst_60: float,
    hurst_20: float,
    entropy_60: float,
    entropy_20: float,
) -> float:
    # bounded terms in [-1,1]
    hurst_term = _clip((hurst_20 - 0.45) / 0.2, -1.0, 1.0)
    chaos_term = 1.0 - _clip(abs(hurst_60 - 0.4) / 0.2, 0.0, 1.0)

    e_base = entropy_60 if np.isfinite(entropy_60) and entropy_60 > 0 else 1.0
    entropy_term = _clip((entropy_60 - entropy_20) / e_base, -1.0, 1.0)

    ma_term = 0.0
    if np.isfinite(ma60) and ma60 > 0:
        ma_term = _clip((close / ma60 - 1.0) / 0.1, -1.0, 1.0)

    pct_term = _clip((pct_chg - 3.0) / 7.0, -1.0, 1.0)

    score = 0.30 * hurst_term + 0.25 * entropy_term + 0.20 * ma_term + 0.15 * pct_term + 0.10 * chaos_term
    return float(_clip(score, SCORE_MIN, SCORE_MAX))


def _collect_stock_calib_rows(file_path: str) -> list[dict]:
    try:
        df = pd.read_csv(file_path)
    except Exception:
        return []

    if df.empty or "trade_date" not in df.columns:
        return []

    df["trade_date_str"] = df["trade_date"].astype(str)
    # Need long buffers; and include training
    df = df[df["trade_date_str"] >= "20220901"].copy()
    if df.empty:
        return []

    df = df.sort_values("trade_date").reset_index(drop=True)

    required_cols = ["close", "open", "pct_chg", "net_mf_amount", "ts_code"]
    for col in required_cols:
        if col not in df.columns:
            return []

    ts_code = str(df["ts_code"].iloc[0])
    is_leader = ts_code in set(LEADER_TS_CODES) if LEADER_TS_CODES else False

    closes = pd.to_numeric(df["close"], errors="coerce").values
    opens = pd.to_numeric(df["open"], errors="coerce").values
    pct_chg = pd.to_numeric(df["pct_chg"], errors="coerce").values
    dates = df["trade_date_str"].astype(str).values

    ma60 = pd.Series(closes).rolling(window=60).mean().values
    mf_5d = pd.to_numeric(df["net_mf_amount"], errors="coerce").fillna(0).rolling(window=5).sum().values
    mf5 = pd.Series(mf_5d)
    mf_5d_z20 = ((mf5 - mf5.rolling(window=20).mean()) / (mf5.rolling(window=20).std() + 1e-9)).values

    end_train_idx = int(np.searchsorted(dates, TRAIN_END, side="right") - 1)
    if end_train_idx < 60:
        return []

    out: list[dict] = []

    for i in range(60, end_train_idx + 1):
        d = str(dates[i])
        if d < TRAIN_START or d > TRAIN_END:
            continue

        # Fast filters (relaxed; kept consistent with eval candidate generation)
        strict_ok = True
        if not np.isfinite(pct_chg[i]) or float(pct_chg[i]) < float(ENTRY_PCT_CHG_MIN):
            strict_ok = False
        if (
            not np.isfinite(ma60[i])
            or not np.isfinite(closes[i])
            or float(closes[i]) < float(ma60[i]) * float(ENTRY_CLOSE_MA60_RATIO)
        ):
            strict_ok = False
        if not np.isfinite(mf_5d[i]) or float(mf_5d[i]) < float(ENTRY_MF5D_MIN):
            strict_ok = False

        leader_early_ok = False
        if is_leader and EARLY_LEADER_ENTRY_ENABLED:
            if np.isfinite(pct_chg[i]) and float(pct_chg[i]) >= float(EARLY_LEADER_PCT_CHG_MIN):
                if np.isfinite(ma60[i]) and np.isfinite(closes[i]) and float(ma60[i]) > 0:
                    ratio = float(closes[i]) / float(ma60[i])
                    if float(EARLY_LEADER_CLOSE_MA60_MIN) <= ratio <= float(EARLY_LEADER_CLOSE_MA60_MAX):
                        if ratio < float(ENTRY_CLOSE_MA60_RATIO):
                            lb = int(EARLY_LEADER_CRASH_LOOKBACK)
                            j0 = max(0, i - lb)
                            recent = pct_chg[j0 : i + 1]
                            if len(recent) > 0 and float(np.nanmin(recent)) <= float(EARLY_LEADER_CRASH_MIN_PCT_CHG):
                                z = float(mf_5d_z20[i]) if np.isfinite(mf_5d_z20[i]) else float("-inf")
                                if z >= float(EARLY_LEADER_MF5D_Z20_MIN):
                                    leader_early_ok = True

        if not (strict_ok or leader_early_ok):
            continue

        # Metrics windows
        w60 = closes[i - 60 : i + 1]
        w20 = closes[i - 20 : i + 1]
        r60 = pct_chg[i - 60 : i + 1]
        r20 = pct_chg[i - 20 : i + 1]

        h60 = fast_hurst(w60)
        h20 = fast_hurst(w20)
        e60 = calc_entropy(r60)
        e20 = calc_entropy(r20)

        if not (float(ENTRY_HURST60_MIN) <= h60 <= float(ENTRY_HURST60_MAX)):
            continue
        if h20 < float(ENTRY_HURST20_MIN):
            continue
        if e20 > e60 + float(ENTRY_ENTROPY20_OVER60_MAX):
            continue

        if leader_early_ok:
            if h20 < float(EARLY_LEADER_HURST20_MIN):
                continue
            if (h20 - h60) < float(EARLY_LEADER_HURST20_OVER60_MIN):
                continue
            if e20 > e60 + float(EARLY_LEADER_ENTROPY20_OVER60_MAX):
                continue

        # Entry next day open
        entry_i = i + 1
        if entry_i > end_train_idx:
            continue
        entry_price = float(opens[entry_i])
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue

        # labels: close after entry+h days vs entry_price
        ok = True
        y_labels = {}
        for h in HORIZONS:
            j = entry_i + h
            if j > end_train_idx:
                ok = False
                break
            y_labels[h] = int(float(closes[j]) > entry_price)
        if not ok:
            continue

        score = stock_feature_score(float(closes[i]), float(pct_chg[i]), float(ma60[i]), h60, h20, e60, e20)
        r = {"bin": score_to_bin(score)}
        for h in HORIZONS:
            r[f"y_up_{h}"] = y_labels[h]
        out.append(r)

    return out


def calibrate_stock_probability(files: list[str]) -> CalibTable:
    pool = Pool(max(1, cpu_count() - 1))
    results = pool.map(_collect_stock_calib_rows, files)
    pool.close()
    pool.join()

    rows: list[dict] = []
    for r in results:
        rows.extend(r)

    if not rows:
        raise RuntimeError("No training rows collected for stock calibration. Check TRAIN_START/TRAIN_END and data coverage.")

    return build_calibration_from_bins(rows, HORIZONS)


def compute_stock_signed_score_at_signal(
    close: float,
    pct_chg: float,
    ma60: float,
    hurst_60: float,
    hurst_20: float,
    entropy_60: float,
    entropy_20: float,
    calib: CalibTable,
) -> tuple[float, dict[int, float]]:
    score = stock_feature_score(close, pct_chg, ma60, hurst_60, hurst_20, entropy_60, entropy_20)

    p_by_h: dict[int, float] = {}
    s_mix = 0.0
    for h, w in HORIZON_MIX.items():
        p = float(calib.p_up(h, score))
        p_by_h[h] = p
        s_mix += float(w) * rel_score_from_p(p, calib.global_p_up[h])

    return float(_clip(s_mix, -1.0, 1.0)), p_by_h


# ==============================
# Backtest (prob-weighted)
# ==============================

def process_single_stock_prob(file_path: str):
    global INDEX_S_BY_DATE, INDEX_REGIME_BY_DATE, STOCK_CALIB, WEIGHT_INDEX, WEIGHT_STOCK

    try:
        df = pd.read_csv(file_path)
    except Exception:
        return []

    if df.empty or "trade_date" not in df.columns:
        return []

    df["trade_date_str"] = df["trade_date"].astype(str)
    df = df[df["trade_date_str"] >= "20240901"].copy()  # buffer for 60-day
    if df.empty:
        return []

    df = df.sort_values("trade_date").reset_index(drop=True)

    required_cols = ["close", "open", "pct_chg", "net_mf_amount", "ts_code"]
    for col in required_cols:
        if col not in df.columns:
            return []

    ts_code = str(df["ts_code"].iloc[0])
    is_leader = ts_code in set(LEADER_TS_CODES) if LEADER_TS_CODES else False

    closes = pd.to_numeric(df["close"], errors="coerce").values
    opens = pd.to_numeric(df["open"], errors="coerce").values
    pct_chg = pd.to_numeric(df["pct_chg"], errors="coerce").values
    dates = df["trade_date_str"].astype(str).values

    ma60 = pd.Series(closes).rolling(window=60).mean().values
    ma20 = pd.Series(closes).rolling(window=20).mean().values
    mf_5d = pd.to_numeric(df["net_mf_amount"], errors="coerce").fillna(0).rolling(window=5).sum().values
    mf5 = pd.Series(mf_5d)
    mf_5d_z20 = ((mf5 - mf5.rolling(window=20).mean()) / (mf5.rolling(window=20).std() + 1e-9)).values

    end_eval_idx = int(np.searchsorted(dates, END_DATE, side="right") - 1)
    if end_eval_idx < 60:
        return []

    out: list[dict] = []

    i = 60
    i_end = min(len(df) - 1, end_eval_idx)
    while i <= i_end:
        d = str(dates[i])
        if d < START_DATE or d > END_DATE:
            i += 1
            continue

        # fast filters (relaxed)
        strict_ok = True
        if not np.isfinite(pct_chg[i]) or float(pct_chg[i]) < float(ENTRY_PCT_CHG_MIN):
            strict_ok = False
        if (
            not np.isfinite(ma60[i])
            or not np.isfinite(closes[i])
            or float(closes[i]) < float(ma60[i]) * float(ENTRY_CLOSE_MA60_RATIO)
        ):
            strict_ok = False
        if not np.isfinite(mf_5d[i]) or float(mf_5d[i]) < float(ENTRY_MF5D_MIN):
            strict_ok = False

        leader_early_ok = False
        if is_leader and EARLY_LEADER_ENTRY_ENABLED:
            if np.isfinite(pct_chg[i]) and float(pct_chg[i]) >= float(EARLY_LEADER_PCT_CHG_MIN):
                if np.isfinite(ma60[i]) and np.isfinite(closes[i]) and float(ma60[i]) > 0:
                    ratio = float(closes[i]) / float(ma60[i])
                    if float(EARLY_LEADER_CLOSE_MA60_MIN) <= ratio <= float(EARLY_LEADER_CLOSE_MA60_MAX):
                        if ratio < float(ENTRY_CLOSE_MA60_RATIO):
                            lb = int(EARLY_LEADER_CRASH_LOOKBACK)
                            j0 = max(0, i - lb)
                            recent = pct_chg[j0 : i + 1]
                            if len(recent) > 0 and float(np.nanmin(recent)) <= float(EARLY_LEADER_CRASH_MIN_PCT_CHG):
                                z = float(mf_5d_z20[i]) if np.isfinite(mf_5d_z20[i]) else float("-inf")
                                if z >= float(EARLY_LEADER_MF5D_Z20_MIN):
                                    leader_early_ok = True

        if not (strict_ok or leader_early_ok):
            i += 1
            continue

        entry_mode = "Leader Early Phase" if (leader_early_ok and not strict_ok) else "Normal"

        # metrics
        w60 = closes[i - 60 : i + 1]
        w20 = closes[i - 20 : i + 1]
        r60 = pct_chg[i - 60 : i + 1]
        r20 = pct_chg[i - 20 : i + 1]

        h60 = fast_hurst(w60)
        h20 = fast_hurst(w20)
        e60 = calc_entropy(r60)
        e20 = calc_entropy(r20)

        if not (float(ENTRY_HURST60_MIN) <= h60 <= float(ENTRY_HURST60_MAX)):
            i += 1
            continue
        if h20 < float(ENTRY_HURST20_MIN):
            i += 1
            continue
        if e20 > e60 + float(ENTRY_ENTROPY20_OVER60_MAX):
            i += 1
            continue

        if leader_early_ok:
            if h20 < float(EARLY_LEADER_HURST20_MIN):
                i += 1
                continue
            if (h20 - h60) < float(EARLY_LEADER_HURST20_OVER60_MIN):
                i += 1
                continue
            if e20 > e60 + float(EARLY_LEADER_ENTROPY20_OVER60_MAX):
                i += 1
                continue

        # entry next day
        entry_i = i + 1
        if entry_i > end_eval_idx:
            i += 1
            continue
        entry_price = float(opens[entry_i])
        if not np.isfinite(entry_price) or entry_price <= 0:
            i += 1
            continue

        # compute signed-prob score for stock at this signal
        s_stock, p_by_h = compute_stock_signed_score_at_signal(
            close=float(closes[i]),
            pct_chg=float(pct_chg[i]),
            ma60=float(ma60[i]),
            hurst_60=float(h60),
            hurst_20=float(h20),
            entropy_60=float(e60),
            entropy_20=float(e20),
            calib=STOCK_CALIB,
        )

        s_index = float(INDEX_S_BY_DATE.get(d, 0.0))
        index_regime_entry = str(INDEX_REGIME_BY_DATE.get(str(dates[entry_i]), "")) or str(
            INDEX_REGIME_BY_DATE.get(d, "")
        )
        if not index_regime_entry:
            index_regime_entry = "BASE"
        entry_score = WEIGHT_STOCK * s_stock + WEIGHT_INDEX * s_index

        # Optional monthly regime gating: modulate entry strictness by coarse index phase.
        regime_adj = 0.0
        if INDEX_REGIME_GATING_ENABLED:
            regime_adj = float(INDEX_REGIME_MIN_ENTRY_ADJ.get(index_regime_entry, 0.0))
            if (
                INDEX_REGIME_BLOCK_LEADER_EARLY_IN_RISK
                and is_leader
                and entry_mode == "Leader Early Phase"
                and index_regime_entry == "RISK"
            ):
                i += 1
                continue

        if is_leader and entry_mode == "Leader Early Phase":
            min_entry = float(MIN_ENTRY_SCORE_LEADER_EARLY) + regime_adj
        else:
            min_entry = (float(MIN_ENTRY_SCORE_LEADER) if is_leader else float(MIN_ENTRY_SCORE)) + regime_adj
        if entry_score < min_entry:
            i += 1
            continue

        # holding window until END_DATE
        f_end = end_eval_idx
        f_closes = closes[entry_i : f_end + 1]
        f_ma20 = ma20[entry_i : f_end + 1]
        f_pct = pct_chg[entry_i : f_end + 1]
        f_dates = dates[entry_i : f_end + 1]

        if len(f_closes) == 0:
            i += 1
            continue

        exit_idx = len(f_closes) - 1
        exit_reason = "End Date"

        # For leader trend-following: track running max for drawdown-style logic (optional future extension)
        for j in range(len(f_closes)):
            dj = str(f_dates[j])
            p_close = float(f_closes[j])
            cur_ret = (p_close - entry_price) / entry_price

            # stop loss
            if is_leader and entry_mode == "Leader Early Phase":
                if cur_ret <= float(LEADER_EARLY_STOP_LOSS_RET):
                    exit_idx = j
                    exit_reason = "Stop Loss"
                    break
            else:
                if cur_ret <= -0.08 or (
                    np.isfinite(f_ma20[j]) and float(p_close) < float(f_ma20[j]) * 0.95
                ):
                    exit_idx = j
                    exit_reason = "Stop Loss"
                    break

            # entropy spike take profit (same rule as before)
            if not (is_leader and LEADER_DISABLE_ENTROPY_SPIKE_TAKE_PROFIT):
                if j >= 5 and cur_ret > 0.15:
                    recent_ret = f_pct[max(0, j - 10) : j + 1]
                    if len(recent_ret) >= 5:
                        cur_e = calc_entropy(recent_ret)
                        if np.isfinite(cur_e) and cur_e > e60 * 1.2:
                            exit_idx = j
                            exit_reason = "Take Profit (Entropy Spike)"
                            break

            # Leader trend exit: entropy/Hurst/energy/MA20 jointly indicate disorder & trend breakdown
            min_hold_bars = (
                int(LEADER_EARLY_TREND_MIN_HOLD_BARS)
                if (is_leader and entry_mode == "Leader Early Phase")
                else int(LEADER_TREND_MIN_HOLD_BARS)
            )
            if is_leader and LEADER_TREND_EXIT and j + 1 >= int(min_hold_bars):
                k = entry_i + j
                if k >= 60:
                    w60_j = closes[k - 60 : k + 1]
                    w20_j = closes[k - 20 : k + 1]
                    r60_j = pct_chg[k - 60 : k + 1]
                    r20_j = pct_chg[k - 20 : k + 1]
                    h20_j = fast_hurst(w20_j)
                    e60_j = calc_entropy(r60_j)
                    e20_j = calc_entropy(r20_j)
                    energy_j = float(mf_5d[k]) if np.isfinite(mf_5d[k]) else 0.0

                    ma20_break = (
                        np.isfinite(f_ma20[j])
                        and float(p_close) < float(f_ma20[j]) * float(LEADER_TREND_MA20_BREAK_RATIO)
                    )
                    hurst_break = bool(h20_j < float(LEADER_TREND_HURST20_MIN))
                    entropy_break = bool(e20_j > e60_j + float(LEADER_TREND_ENTROPY20_OVER60))
                    energy_break = bool(energy_j < float(LEADER_TREND_MF5D_MIN))

                    # Require at least two confirmations to avoid over-sensitive exits.
                    n_break = int(ma20_break) + int(hurst_break) + int(entropy_break) + int(energy_break)
                    if n_break >= 2:
                        exit_idx = j
                        exit_reason = "Leader Trend Exit"
                        break

            # weighted exit: if combined signed score turns negative
            if not (is_leader and LEADER_DISABLE_WEIGHTED_EXIT):
                s_idx_j = float(INDEX_S_BY_DATE.get(dj, 0.0))
                combined = WEIGHT_STOCK * s_stock + WEIGHT_INDEX * s_idx_j
                if combined < WEIGHTED_EXIT_THRESHOLD:
                    exit_idx = j
                    exit_reason = "Weighted Exit (Index Prob)"
                    break

        max_close = float(np.max(f_closes[: exit_idx + 1]))
        exit_close = float(f_closes[exit_idx])
        max_ret = (max_close - entry_price) / entry_price * 100.0
        hold_ret = (exit_close - entry_price) / entry_price * 100.0

        exit_date = str(f_dates[exit_idx])
        index_regime_exit = str(INDEX_REGIME_BY_DATE.get(exit_date, ""))
        if not index_regime_exit:
            index_regime_exit = "BASE"

        out.append(
            {
                "ts_code": ts_code,
                "entry_mode": entry_mode,
                "signal_date": d,
                "entry_date": str(dates[entry_i]),
                "exit_date": exit_date,
                "index_regime_entry": index_regime_entry,
                "index_regime_exit": index_regime_exit,
                "exit_reason": exit_reason,
                "hold_days": int(exit_idx + 1),
                "hurst_60": round(float(h60), 3),
                "hurst_20": round(float(h20), 3),
                "entropy_drop_pct": round(float((e60 - e20) / e60 * 100.0), 2) if np.isfinite(e60) and e60 != 0 else np.nan,
                "p_up_5": round(float(p_by_h[5]), 4),
                "p_up_20": round(float(p_by_h[20]), 4),
                "s_stock_rel": round(float(s_stock), 4),
                "s_index_rel_entry": round(float(s_index), 4),
                "entry_score": round(float(entry_score), 4),
                "max_hold_return(%)": round(float(max_ret), 2),
                "actual_return(%)": round(float(hold_ret), 2),
            }
        )

        # Prevent overlapping trades on the same stock:
        # jump i to the exit day index (in this stock's dataframe).
        i = entry_i + exit_idx
        continue

    return out


def select_trades_daily_topn(
    df_candidates: pd.DataFrame,
    top_n: int,
    max_open_positions: int,
    min_entry_score: float,
) -> pd.DataFrame:
    if df_candidates.empty:
        return df_candidates

    df = df_candidates.copy()
    df["entry_date"] = df["entry_date"].astype(str)
    df["exit_date"] = df["exit_date"].astype(str)
    df["ts_code"] = df["ts_code"].astype(str)

    df = df[np.isfinite(pd.to_numeric(df["entry_score"], errors="coerce"))].copy()
    df["entry_score"] = pd.to_numeric(df["entry_score"], errors="coerce")
    # Apply base cutoff, but keep leader candidates with their own (lower) cutoff.
    base_cut = float(min_entry_score)
    leader_set = set(LEADER_TS_CODES) if LEADER_TS_CODES else set()
    leader_cut = float(MIN_ENTRY_SCORE_LEADER)
    is_leader = df["ts_code"].isin(leader_set) if leader_set else False
    allow = (df["entry_score"] >= base_cut) | (is_leader & (df["entry_score"] >= leader_cut))
    if "entry_mode" in df.columns:
        mode = df["entry_mode"].astype(str)
        leader_early = is_leader & (mode == "Leader Early Phase")
        allow = allow | (leader_early & (df["entry_score"] >= float(MIN_ENTRY_SCORE_LEADER_EARLY)))
    df = df[allow].copy()
    if df.empty:
        return df

    df = df.sort_values(["entry_date", "entry_score"], ascending=[True, False]).reset_index(drop=True)

    # Optional filtering: exclude ST/*ST names.
    if EXCLUDE_ST and os.path.exists(BASIC_INFO_PATH):
        try:
            df_basic = pd.read_csv(BASIC_INFO_PATH, usecols=["ts_code", "name"])
            if isinstance(df_basic, pd.DataFrame) and not df_basic.empty:
                name_map = dict(zip(df_basic["ts_code"].astype(str), df_basic["name"].astype(str)))
                nm = df["ts_code"].map(name_map).fillna("").astype(str)
                is_st = nm.str.startswith("ST") | nm.str.startswith("*ST")
                df = df[~is_st].copy()
        except Exception:
            # If basic info is unavailable/corrupt, fall back to no filtering.
            pass

    def _ts_code_to_daily_path(ts_code: str) -> str:
        ts_code = (ts_code or "").strip()
        if "." not in ts_code:
            return os.path.join(STOCK_DATA_DIR, "")
        code, exch = ts_code.split(".", 1)
        return os.path.join(STOCK_DATA_DIR, f"{exch.lower()}{code}.csv")

    price_cache: dict[str, dict[str, object] | None] = {}

    def _get_price_cache(ts_code: str) -> dict[str, object] | None:
        ts_code = str(ts_code)
        if ts_code in price_cache:
            return price_cache[ts_code]

        fp = _ts_code_to_daily_path(ts_code)
        if not fp or not os.path.exists(fp):
            price_cache[ts_code] = None
            return None

        try:
            p = pd.read_csv(fp, usecols=["trade_date", "open", "close"])
        except Exception:
            price_cache[ts_code] = None
            return None

        if p is None or not isinstance(p, pd.DataFrame) or p.empty:
            price_cache[ts_code] = None
            return None

        p = p.copy()
        p["trade_date_str"] = p["trade_date"].astype(str)
        p = p.sort_values("trade_date_str").reset_index(drop=True)

        dates = p["trade_date_str"].astype(str).tolist()
        date_to_i = {d: i for i, d in enumerate(dates)}
        opens = pd.to_numeric(p["open"], errors="coerce").values
        closes = pd.to_numeric(p["close"], errors="coerce").values

        price_cache[ts_code] = {
            "dates": dates,
            "date_to_i": date_to_i,
            "opens": opens,
            "closes": closes,
        }
        return price_cache[ts_code]

    def _force_exit_for_rotation(df_mut: pd.DataFrame, df_idx: int, pos_ts_code: str, entry_date: str) -> str | None:
        """Force-exit a selected position on the trading day immediately before `entry_date`.

        Updates exit fields in-place on `df_mut` row `df_idx`. Returns the forced exit_date, or None if unavailable.
        """

        cache = _get_price_cache(pos_ts_code)
        if cache is None:
            return None

        date_to_i = cache["date_to_i"]  # type: ignore[assignment]
        dates = cache["dates"]          # type: ignore[assignment]
        opens = cache["opens"]          # type: ignore[assignment]
        closes = cache["closes"]        # type: ignore[assignment]

        pos_entry_date = str(df_mut.at[df_idx, "entry_date"])
        i_entry = date_to_i.get(pos_entry_date)
        i_new = date_to_i.get(str(entry_date))
        if i_entry is None or i_new is None:
            return None
        i_exit = i_new - 1
        if i_exit < i_entry:
            return None

        entry_open = float(opens[i_entry])
        exit_close = float(closes[i_exit])
        if not np.isfinite(entry_open) or entry_open <= 0 or not np.isfinite(exit_close):
            return None

        forced_exit_date = str(dates[i_exit])
        hold_bars = int(i_exit - i_entry + 1)
        win_max_close = float(np.nanmax(closes[i_entry : i_exit + 1]))

        hold_ret = (exit_close - entry_open) / entry_open * 100.0
        max_ret = (win_max_close - entry_open) / entry_open * 100.0

        df_mut.at[df_idx, "exit_date"] = forced_exit_date
        df_mut.at[df_idx, "exit_reason"] = "Rotation (Replaced)"
        if "hold_days" in df_mut.columns:
            df_mut.at[df_idx, "hold_days"] = hold_bars
        if "actual_return(%)" in df_mut.columns:
            df_mut.at[df_idx, "actual_return(%)"] = round(float(hold_ret), 2)
        if "max_hold_return(%)" in df_mut.columns:
            df_mut.at[df_idx, "max_hold_return(%)"] = round(float(max_ret), 2)

        return forced_exit_date

    # Track open positions by ts_code -> position state
    open_pos: dict[str, dict[str, object]] = {}
    selected_rows: list[int] = []

    for entry_date, g in df.groupby("entry_date", sort=True):
        # Close positions that have already exited before today
        to_close = [k for k, v in open_pos.items() if str(v.get("exit_date", "")) < entry_date]
        for k in to_close:
            open_pos.pop(k, None)

        # Prefer a configured watchlist leader if it is close to day's best score.
        # Still enforces: 1 buy/day, no duplicate holdings, full-portfolio rotation rules.
        day_best = float(g["entry_score"].max()) if not g.empty else float("-inf")
        preferred_codes = set(PREFERRED_TS_CODES) if isinstance(PREFERRED_TS_CODES, set) else set(PREFERRED_TS_CODES)
        preferred_idx: list[int] = []
        other_idx: list[int] = []
        if preferred_codes:
            for idx, row in g.iterrows():
                ts_code = str(row["ts_code"])
                s = float(row["entry_score"])
                if ts_code in preferred_codes:
                    if (LEADER_IGNORE_PREFERRED_BAND and (ts_code in leader_set)):
                        preferred_idx.append(int(idx))
                    elif s >= (day_best - float(PREFERRED_SCORE_BAND)):
                        preferred_idx.append(int(idx))
                    else:
                        other_idx.append(int(idx))
                else:
                    other_idx.append(int(idx))
            # keep preferred in descending score order
            preferred_idx = (
                g.loc[preferred_idx].sort_values("entry_score", ascending=False).index.astype(int).tolist()
                if preferred_idx
                else []
            )
            other_idx = (
                g.loc[other_idx].sort_values("entry_score", ascending=False).index.astype(int).tolist()
                if other_idx
                else []
            )
            day_iter_idx = preferred_idx + other_idx
        else:
            day_iter_idx = g.sort_values("entry_score", ascending=False).index.astype(int).tolist()

        picks = 0
        for idx in day_iter_idx:
            if picks >= int(top_n):
                break

            row = df.loc[idx]
            ts_code = str(row["ts_code"])
            if ts_code in open_pos:
                continue

            # If full, optionally rotate out the weakest holding.
            if len(open_pos) >= int(max_open_positions):
                if not ROTATE_ON_FULL:
                    break

                cand_score = float(row["entry_score"])
                cand_is_leader = ts_code in set(LEADER_TS_CODES) if LEADER_TS_CODES else False
                # pick the weakest *eligible* holding by entry_score (held long enough)
                eligible: list[tuple[float, str, int]] = []  # (entry_score, ts_code, df_idx)
                for k, v in open_pos.items():
                    # Leader protection: do not rotate leaders out (or lock for N bars)
                    if LEADER_PROTECT_FROM_ROTATION and (k in set(LEADER_TS_CODES)):
                        cache0 = _get_price_cache(k)
                        if cache0 is None:
                            continue
                        date_to_i0 = cache0["date_to_i"]  # type: ignore[assignment]
                        pos_entry0 = str(v.get("entry_date", ""))
                        i_entry0 = date_to_i0.get(pos_entry0)
                        i_new0 = date_to_i0.get(str(entry_date))
                        if i_entry0 is None or i_new0 is None:
                            continue
                        held_bars0 = int(max(0, (i_new0 - 1) - i_entry0 + 1))
                        if held_bars0 < int(LEADER_LOCK_BARS):
                            continue
                        # Even after lock, we still keep leaders protected from rotation.
                        continue

                    pos_entry = str(v.get("entry_date", ""))
                    cache = _get_price_cache(k)
                    if cache is None:
                        continue
                    date_to_i = cache["date_to_i"]  # type: ignore[assignment]
                    i_entry = date_to_i.get(pos_entry)
                    i_new = date_to_i.get(str(entry_date))
                    if i_entry is None or i_new is None:
                        continue
                    held_bars_so_far = int(max(0, (i_new - 1) - i_entry + 1))
                    if held_bars_so_far < int(ROTATE_MIN_HOLD_BARS):
                        continue

                    eligible.append((float(v.get("entry_score", 0.0)), str(k), int(v.get("df_idx", -1))))

                if not eligible:
                    break

                worst_score, worst_code, worst_df_idx = min(eligible, key=lambda x: x[0])

                # Margin rule: leaders can optionally force-entry even when not much better,
                # but still only by replacing a non-leader holding.
                if cand_is_leader and LEADER_FORCE_ENTRY_ON_FULL:
                    pass
                else:
                    if (cand_score - float(worst_score)) < float(ROTATE_SCORE_MARGIN):
                        break

                # Force exit the weakest eligible holding and free a slot.
                if worst_df_idx >= 0:
                    forced_exit = _force_exit_for_rotation(df, worst_df_idx, worst_code, str(entry_date))
                    if forced_exit is None:
                        break
                    open_pos.pop(worst_code, None)
                else:
                    break

            selected_rows.append(idx)
            open_pos[ts_code] = {
                "exit_date": str(row["exit_date"]),
                "entry_date": str(row["entry_date"]),
                "entry_score": float(row["entry_score"]),
                "df_idx": int(idx),
            }
            picks += 1

    if not selected_rows:
        return df.head(0)

    selected = df.loc[selected_rows].copy().reset_index(drop=True)
    return selected


def run_backtest_for_weight(weight_index: float, files: list[str]) -> pd.DataFrame:
    global WEIGHT_INDEX
    WEIGHT_INDEX = float(weight_index)

    pool = Pool(max(1, cpu_count() - 1))
    results = pool.map(process_single_stock_prob, files)
    pool.close()
    pool.join()

    all_trades: list[dict] = []
    for r in results:
        all_trades.extend(r)

    df_candidates = pd.DataFrame(all_trades)
    if df_candidates.empty:
        return df_candidates

    df_selected = select_trades_daily_topn(
        df_candidates,
        top_n=DAILY_TOP_N,
        max_open_positions=MAX_OPEN_POSITIONS,
        min_entry_score=MIN_ENTRY_SCORE,
    )

    return df_selected


def main():
    global INDEX_S_BY_DATE, INDEX_REGIME_BY_DATE, STOCK_CALIB

    start = time.time()

    # 1) Index signed-probability score by date (trained on 2023-2024)
    df_index = load_index_df()
    index_calib, INDEX_S_BY_DATE = build_index_calibration_and_scores(df_index)

    # 1b) Index monthly regime state machine (mapped back to daily dates)
    _, INDEX_REGIME_BY_DATE = build_index_monthly_regime_by_date(df_index)

    # 2) Stock calibration (trained on 2023-2024 triggered signals)
    files = glob.glob(os.path.join(STOCK_DATA_DIR, "*.csv"))
    print(f"Found {len(files)} stock CSV files. Calibrating stock probabilities on {TRAIN_START}..{TRAIN_END}...")
    STOCK_CALIB = calibrate_stock_probability(files)

    # 3) Weight grid backtest for 2025
    print(f"Starting 2025 prob-weighted backtest on {START_DATE}..{END_DATE}...")
    weight_grid = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

    grid_rows = []
    best = None

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
        weighted_exit_pct = exit_counts.get("Weighted Exit (Index Prob)", 0) / total * 100.0
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

        # Select best by avg_return(%) (user preference)
        if best is None or avg_ret > best["avg_ret_raw"]:
            best = {"w_index": w_idx, "avg_ret_raw": float(avg_ret), "row": row, "df_trades": df_trades}

    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(GRID_CSV_PATH, index=False)

    if best is None:
        print("No trades in any weight setting.")
        return

    best_w = best["w_index"]
    best_trades = best["df_trades"].copy()

    # Attach names
    if os.path.exists(BASIC_INFO_PATH):
        df_basic = pd.read_csv(BASIC_INFO_PATH)
        if not df_basic.empty and "ts_code" in df_basic.columns and "name" in df_basic.columns:
            name_map = dict(zip(df_basic["ts_code"], df_basic["name"]))
            best_trades.insert(1, "name", best_trades["ts_code"].map(name_map))

    best_trades.to_csv(BEST_TRADES_PATH, index=False)

    # Bull list: best single trade per symbol
    bull = (
        best_trades.sort_values("actual_return(%)", ascending=False)
        .groupby("ts_code", as_index=False)
        .head(1)
        .head(30)
    )

    regime_stats_md = ""
    if "index_regime_entry" in best_trades.columns:
        try:
            t = best_trades.copy()
            t["index_regime_entry"] = t["index_regime_entry"].astype(str).replace({"": "BASE"}).fillna("BASE")
            g = (
                t.groupby("index_regime_entry", as_index=False)
                .agg(
                    trades=("ts_code", "count"),
                    win_rate_pct=("actual_return(%)", lambda s: float((s > 0).mean() * 100.0)),
                    avg_return_pct=("actual_return(%)", "mean"),
                    median_return_pct=("actual_return(%)", "median"),
                    avg_hold_days=("hold_days", "mean"),
                )
                .sort_values("trades", ascending=False)
            )
            g["win_rate_pct"] = g["win_rate_pct"].round(2)
            g["avg_return_pct"] = g["avg_return_pct"].round(2)
            g["median_return_pct"] = g["median_return_pct"].round(2)
            g["avg_hold_days"] = g["avg_hold_days"].round(1)

            regime_stats_md = "\n".join(
                [
                    "## 指数月线相位（state machine）与交易表现（按入场相位分组）",
                    "* 相位是从上证指数日线重采样月线后计算（熵/赫斯特/资金能量/均线偏离），并用 2 个月滞回做平滑，输出为 `BULL/BASE/RISK`。",
                    g.to_markdown(index=False),
                    "",
                ]
            )
        except Exception:
            regime_stats_md = ""

    calib_diag_md = "\n".join(
        [
            "## 校准诊断（训练期 2023-2024）",
            _calib_diag_tables_md(STOCK_CALIB, "Stock (triggered signals)"),
            "",
            _calib_diag_tables_md(index_calib, "Index (daily)"),
        ]
    )

    report = rf"""
# 2025 相变策略：上证指数概率加权回测报告

* **训练校准区间**: {TRAIN_START} 到 {TRAIN_END}
* **回测区间**: {START_DATE} 到 {END_DATE}
* **指数标的**: 上证指数 ({INDEX_TS_CODE})
* **概率信号**: 预测未来 5 天/20 天 “收盘价相对入场价收益为正”的概率 $p_{{up}}$
* **相对分数**: 将 $p_{{up}}$ 与训练期全局基准胜率 $p_{{base}}$ 做中心化，得到 $s = \frac{{p_{{up}}-p_{{base}}}}{{\max(p_{{base}}, 1-p_{{base}})}} \in [-1,1]$（0 表示与基准一致，>0 表示更优，<0 表示更差）
* **组合分数**: $Score = w_s\cdot s_{{stock}} + w_i\cdot s_{{index}}$，其中 $w_s=1.0$，$w_i = w_{{index}}$
* **5/20 天融合**: $s = 0.5\cdot s_5 + 0.5\cdot s_{{20}}$
* **指数概率退出阈值**: $Score < {WEIGHTED_EXIT_THRESHOLD}$ 才触发 `Weighted Exit (Index Prob)`
* **指数月线相位**: 额外计算上证指数 `BULL/BASE/RISK`（月线 + 2 个月滞回），写入交易明细的 `index_regime_entry/index_regime_exit` 字段
* **相位门控**: {'启用' if INDEX_REGIME_GATING_ENABLED else '关闭'}（启用时对入场阈值做加性调整：{INDEX_REGIME_MIN_ENTRY_ADJ}；且{'禁止' if INDEX_REGIME_BLOCK_LEADER_EARLY_IN_RISK else '允许'}在 `RISK` 下的 `Leader Early Phase` 早期入场）
* **选股约束**: 每天只选 Top {DAILY_TOP_N}（按 entry_score），最多同时持仓 {MAX_OPEN_POSITIONS}，且 entry_score >= {MIN_ENTRY_SCORE}
* **Top1 放宽（B）**: 若当日出现 `PREFERRED_TS_CODES` 且其分数满足 $Score \ge Score_{{day\_best}} - {PREFERRED_SCORE_BAND}$，则优先选入（仍保持 1 买/日与 5 仓上限）
* **入场放宽**: pct_chg >= {ENTRY_PCT_CHG_MIN}，close >= MA60*{ENTRY_CLOSE_MA60_RATIO}，mf_5d >= {ENTRY_MF5D_MIN}；相变约束放宽到 hurst60∈[{ENTRY_HURST60_MIN},{ENTRY_HURST60_MAX}]、hurst20>={ENTRY_HURST20_MIN}、entropy20<=entropy60+{ENTRY_ENTROPY20_OVER60_MAX}
* **龙头早期入场（临界态反转）**: {'启用' if EARLY_LEADER_ENTRY_ENABLED else '关闭'}（仅作用于 {sorted(list(LEADER_TS_CODES))}）。当价格仍低于 MA60（close/MA60 ∈ [{EARLY_LEADER_CLOSE_MA60_MIN},{EARLY_LEADER_CLOSE_MA60_MAX}] 且 < {ENTRY_CLOSE_MA60_RATIO}）且最近 {EARLY_LEADER_CRASH_LOOKBACK} 日出现单日暴跌（min(pct_chg) <= {EARLY_LEADER_CRASH_MIN_PCT_CHG}）后，若当日涨幅 >= {EARLY_LEADER_PCT_CHG_MIN} 且资金能量 z 分数 mf_5d_z20 >= {EARLY_LEADER_MF5D_Z20_MIN}，并满足更严格的“有序化”条件（hurst20 >= {EARLY_LEADER_HURST20_MIN}、hurst20-hurst60 >= {EARLY_LEADER_HURST20_OVER60_MIN}、entropy20 <= entropy60+{EARLY_LEADER_ENTROPY20_OVER60_MAX}），则允许提前触发信号
* **龙头趋势退出**: {'启用' if LEADER_TREND_EXIT else '关闭'}（{sorted(list(LEADER_TS_CODES))}），并{'禁用' if LEADER_DISABLE_WEIGHTED_EXIT else '保留'}指数概率退出、并{'禁用' if LEADER_DISABLE_ENTROPY_SPIKE_TAKE_PROFIT else '保留'}熵尖峰止盈
* **ST 过滤**: {'启用' if EXCLUDE_ST else '关闭'}（启用时剔除名称前缀为 ST 或 *ST 的标的）
* **本次网格**: w_index ∈ {weight_grid}

## 权重回归结果（按平均收益选最优）
{grid_df.sort_values('w_index').to_markdown(index=False)}

## 最优权重
* **w_index**: {best_w}
* **胜率**: {best['row']['win_rate(%)']}%
* **平均平仓收益率**: {best['row']['avg_return(%)']}%
* **中位数收益率**: {best['row']['median_return(%)']}%
* **平均持仓天数**: {best['row']['avg_hold_days']}

{calib_diag_md}

## 2025 牛股榜（最优权重下，按标的最佳一笔收益 Top 30）
{bull.to_markdown(index=False)}

{regime_stats_md}

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
