import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from ..multitimeframe.multitimeframe_feature_engine import (
        aggregate_stock_bars,
        compute_physics_state_features,
        to_trade_date_str,
    )
    from ..multitimeframe.multitimeframe_physics_utils import rolling_zscore
else:
    _PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from src.strategy.multitimeframe.multitimeframe_feature_engine import (
        aggregate_stock_bars,
        compute_physics_state_features,
        to_trade_date_str,
    )
    from src.strategy.multitimeframe.multitimeframe_physics_utils import rolling_zscore


def _numeric_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


def _safe_div(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray) -> np.ndarray:
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    out = np.full(len(num), np.nan, dtype=np.float64)
    mask = np.isfinite(num) & np.isfinite(den) & (den != 0.0)
    out[mask] = num[mask] / den[mask]
    return out


def _clip_ratio(values: pd.Series | np.ndarray, scale: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if scale <= 0:
        return np.full(len(arr), np.nan, dtype=np.float64)
    return np.clip(arr / float(scale), -1.0, 1.0)


def build_complexity_feature_frame(df_daily: pd.DataFrame) -> pd.DataFrame:
    df = to_trade_date_str(df_daily.copy())
    if df.empty:
        return pd.DataFrame()

    for column in [
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "amount",
        "turnover_rate",
        "net_mf_amount",
    ]:
        df[column] = _numeric_series(df, column, 0.0)

    df = df.dropna(subset=["open", "close"]).reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()

    bars = aggregate_stock_bars(df, "D")
    if bars.empty:
        return pd.DataFrame()

    physics = compute_physics_state_features(
        bars,
        window_s=20,
        window_l=60,
        entry_threshold=-1.0,
        persist_bars=1,
        energy_min=-1.0,
        order_min=-1.0,
        phase_min=-1.0,
    )
    if physics.empty:
        return pd.DataFrame()

    merged = physics.merge(
        df[
            [
                "trade_date_str",
                "dt",
                "high",
                "low",
                "pre_close",
                "amount",
                "turnover_rate",
                "net_mf_amount",
            ]
        ],
        left_on="bar_end",
        right_on="trade_date_str",
        how="left",
    )

    merged = merged.sort_values("bar_end_dt").reset_index(drop=True)
    close = pd.to_numeric(merged["close"], errors="coerce")
    open_ = pd.to_numeric(merged["open"], errors="coerce")
    high = pd.to_numeric(merged["high"], errors="coerce").replace(0.0, np.nan).fillna(pd.to_numeric(merged["close"], errors="coerce"))
    low = pd.to_numeric(merged["low"], errors="coerce").replace(0.0, np.nan).fillna(pd.to_numeric(merged["close"], errors="coerce"))
    pre_close = pd.to_numeric(merged["pre_close"], errors="coerce")
    pre_close = pre_close.replace(0.0, np.nan).fillna(close.shift(1))

    amount = pd.to_numeric(merged["amount"], errors="coerce").fillna(0.0)
    turnover = pd.to_numeric(merged["turnover_rate"], errors="coerce").fillna(0.0)
    net_mf = pd.to_numeric(merged["net_mf_amount"], errors="coerce").fillna(0.0)

    ma_20 = close.rolling(window=20, min_periods=10).mean()
    ma_60 = close.rolling(window=60, min_periods=20).mean()
    ma_120 = close.rolling(window=120, min_periods=40).mean()
    ma_20_slope_10 = ma_20 / ma_20.shift(10) - 1.0
    ma_60_slope_20 = ma_60 / ma_60.shift(20) - 1.0

    prev_high_20 = high.shift(1).rolling(window=20, min_periods=10).max()
    prev_low_20 = low.shift(1).rolling(window=20, min_periods=10).min()
    prev_high_5 = high.shift(1).rolling(window=5, min_periods=3).max()
    prev_high_3 = high.shift(1).rolling(window=3, min_periods=2).max()
    prev_close_high_8 = close.shift(1).rolling(window=8, min_periods=4).max()

    range_width_20 = high.rolling(window=20, min_periods=10).max() / low.rolling(window=20, min_periods=10).min() - 1.0
    compression_z_120 = rolling_zscore(np.log1p(np.clip(range_width_20.to_numpy(dtype=np.float64), 0.0, None)), 120)

    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - pre_close).abs(),
            (low - pre_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_14 = true_range.rolling(window=14, min_periods=7).mean()
    atr_ratio_20 = atr_14 / atr_14.rolling(window=20, min_periods=10).mean() - 1.0

    amount_ma_3 = amount.rolling(window=3, min_periods=2).mean()
    amount_ma_20 = amount.rolling(window=20, min_periods=10).mean()
    amount_contraction = amount_ma_3 / amount_ma_20 - 1.0

    merged["ret_5"] = close.pct_change(5)
    merged["ret_20"] = close.pct_change(20)
    merged["ret_60"] = close.pct_change(60)
    merged["ret_120"] = close.pct_change(120)
    merged["ma_20"] = ma_20
    merged["ma_60"] = ma_60
    merged["ma_120"] = ma_120
    merged["ma_20_slope_10"] = ma_20_slope_10
    merged["ma_60_slope_20"] = ma_60_slope_20
    merged["trend_spread_20_60"] = ma_20 / ma_60 - 1.0
    merged["trend_spread_60_120"] = ma_60 / ma_120 - 1.0
    merged["breakout_20"] = close / prev_high_20 - 1.0
    merged["breakout_5"] = close / prev_high_5 - 1.0
    merged["restart_breakout_3"] = close / prev_high_3 - 1.0
    merged["compression_z_120"] = compression_z_120
    merged["range_width_20"] = range_width_20
    merged["pullback_depth_8"] = close / prev_close_high_8 - 1.0
    merged["distance_from_20d_high"] = close / high.rolling(window=20, min_periods=10).max() - 1.0
    merged["close_range_pos_20"] = _safe_div(close - prev_low_20, prev_high_20 - prev_low_20)
    merged["atr_14"] = atr_14
    merged["atr_ratio_20"] = atr_ratio_20
    merged["amount_z_60"] = rolling_zscore(np.log1p(np.clip(amount.to_numpy(dtype=np.float64), 0.0, None)), 60)
    merged["turnover_z_60"] = rolling_zscore(turnover.to_numpy(dtype=np.float64), 60)
    merged["mf_z_60"] = rolling_zscore(net_mf.to_numpy(dtype=np.float64), 60)
    merged["amount_contraction"] = amount_contraction
    merged["energy_impulse"] = np.clip(
        0.45 * _clip_ratio(merged["mf_z_60"], 2.5)
        + 0.35 * _clip_ratio(merged["amount_z_60"], 2.5)
        + 0.20 * _clip_ratio(merged["turnover_z_60"], 2.5),
        -1.0,
        1.0,
    )
    merged["order_alignment"] = np.clip(
        0.50 * _clip_ratio(merged["trend_spread_20_60"], 0.12)
        + 0.30 * _clip_ratio(merged["trend_spread_60_120"], 0.15)
        + 0.20 * _clip_ratio(merged["close_range_pos_20"] - 0.5, 0.5),
        -1.0,
        1.0,
    )
    merged["trend_strength"] = np.clip(
        0.40 * _clip_ratio(merged["ret_20"], 0.20)
        + 0.35 * _clip_ratio(merged["ret_60"], 0.35)
        + 0.25 * _clip_ratio(merged["ma_20_slope_10"], 0.10),
        -1.0,
        1.0,
    )
    merged["pullback_quality"] = np.clip(
        1.0 - np.abs(np.abs(merged["pullback_depth_8"].to_numpy(dtype=np.float64)) - 0.07) / 0.07,
        0.0,
        1.0,
    )
    merged["trade_date_str"] = merged["bar_end"].astype(str)
    return merged