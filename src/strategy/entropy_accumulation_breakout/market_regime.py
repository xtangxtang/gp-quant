"""
市场状态判定器 — 基于指数数据的熵/趋势/动量多维分析

判定 5 种市场状态:
  DECLINING     — 下跌趋势中, 禁止开仓
  DECLINE_ENDED — 下跌结束/企稳, 轻仓试探
  CONSOLIDATION — 横盘整理, 惜售吸筹的理想环境, 正常开仓
  RISING        — 上涨趋势, 正常开仓
  RISE_ENDING   — 上涨末端/即将结束, 禁止新开仓

理论依据:
  - Dmitriev (2025): 熵作为控制参数, 低熵 = 接近相变
  - Fan (2025): 不可逆性上升 = 市场偏离均衡
  - 均线 + 动量 + 波动率组合: 传统趋势判定的可靠方法
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from src.core.tick_entropy import (
    rolling_permutation_entropy,
    rolling_dominant_eigenvalue,
)

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    DECLINING = "declining"
    DECLINE_ENDED = "decline_ended"
    CONSOLIDATION = "consolidation"
    RISING = "rising"
    RISE_ENDING = "rise_ending"


# 不同状态对应的仓位系数
REGIME_POSITION_SCALE = {
    MarketRegime.DECLINING: 0.0,       # 禁止开仓
    MarketRegime.DECLINE_ENDED: 0.3,   # 轻仓试探
    MarketRegime.CONSOLIDATION: 1.0,   # 正常开仓 — 惜售吸筹的理想环境
    MarketRegime.RISING: 0.8,          # 正常开仓
    MarketRegime.RISE_ENDING: 0.0,     # 禁止新开仓
}


@dataclass
class MarketState:
    """单日市场状态快照"""
    date: str
    regime: MarketRegime
    position_scale: float
    details: dict


# ═════════════════════════════════════════════════════════
# 指数数据加载
# ═════════════════════════════════════════════════════════

INDEX_DIR = "/nvme5/xtang/gp-workspace/gp-data/tushare-index-daily"

def load_index(
    code: str = "000001_sh",
    index_dir: str = INDEX_DIR,
) -> pd.DataFrame:
    """加载指数日线数据"""
    fpath = os.path.join(index_dir, f"{code}.csv")
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Index file not found: {fpath}")
    df = pd.read_csv(fpath)
    df = df.sort_values("trade_date").reset_index(drop=True)
    df["trade_date_str"] = df["trade_date"].astype(str)
    return df


# ═════════════════════════════════════════════════════════
# 指数特征计算
# ═════════════════════════════════════════════════════════

def compute_index_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    为指数日线数据计算趋势 + 熵 + 动量特征.
    """
    out = df.copy()
    close = out["close"].astype(np.float64)
    ret = close.pct_change().fillna(0.0)

    # ── 均线系统 ──
    out["ma5"] = close.rolling(5).mean()
    out["ma10"] = close.rolling(10).mean()
    out["ma20"] = close.rolling(20).mean()
    out["ma60"] = close.rolling(60).mean()
    out["ma120"] = close.rolling(120).mean()

    # 均线多头/空头排列
    out["ma_bull"] = (
        (out["ma5"] > out["ma10"]).astype(int)
        + (out["ma10"] > out["ma20"]).astype(int)
        + (out["ma20"] > out["ma60"]).astype(int)
    )  # 0-3, 3=完美多头

    out["ma_bear"] = (
        (out["ma5"] < out["ma10"]).astype(int)
        + (out["ma10"] < out["ma20"]).astype(int)
        + (out["ma20"] < out["ma60"]).astype(int)
    )  # 0-3, 3=完美空头

    # 价格相对均线位置
    out["price_vs_ma20"] = (close - out["ma20"]) / out["ma20"].replace(0, np.nan)
    out["price_vs_ma60"] = (close - out["ma60"]) / out["ma60"].replace(0, np.nan)

    # ── 动量指标 ──
    out["ret_5d"] = close.pct_change(5)
    out["ret_10d"] = close.pct_change(10)
    out["ret_20d"] = close.pct_change(20)
    out["ret_60d"] = close.pct_change(60)

    # 动量斜率: 20日收益的变化率
    out["momentum_accel"] = out["ret_20d"].diff(5)

    # ── 波动率 ──
    out["volatility_10"] = ret.rolling(10).std()
    out["volatility_20"] = ret.rolling(20).std()
    out["volatility_60"] = ret.rolling(60).std()
    out["vol_ratio"] = out["volatility_10"] / out["volatility_60"].replace(0, np.nan)

    # ── 熵指标 ──
    out["perm_entropy_10"] = rolling_permutation_entropy(close, window=10, order=3)
    out["perm_entropy_20"] = rolling_permutation_entropy(close, window=20, order=3)
    out["perm_entropy_60"] = rolling_permutation_entropy(close, window=60, order=3)

    # 熵斜率
    out["entropy_slope"] = out["perm_entropy_10"] - out["perm_entropy_60"]

    # ── 主特征值 (临界减速) ──
    out["dom_eig_20"] = rolling_dominant_eigenvalue(close, window=20, order=2)
    out["dom_eig_60"] = rolling_dominant_eigenvalue(close, window=60, order=2)

    # ── 高低点判断 ──
    out["high_20"] = close.rolling(20).max()
    out["low_20"] = close.rolling(20).min()
    out["high_60"] = close.rolling(60).max()
    out["low_60"] = close.rolling(60).min()

    # 距离20日/60日高点的回撤幅度
    out["drawdown_20"] = (close - out["high_20"]) / out["high_20"].replace(0, np.nan)
    out["drawdown_60"] = (close - out["high_60"]) / out["high_60"].replace(0, np.nan)

    # 距离20日/60日低点的反弹幅度
    out["bounce_20"] = (close - out["low_20"]) / out["low_20"].replace(0, np.nan)
    out["bounce_60"] = (close - out["low_60"]) / out["low_60"].replace(0, np.nan)

    # ── 成交量特征 ──
    if "vol" in out.columns:
        vol = out["vol"].astype(np.float64)
        out["vol_ma20"] = vol.rolling(20).mean()
        out["vol_ratio_vs_ma"] = vol / out["vol_ma20"].replace(0, np.nan)
        out["vol_shrink_5"] = vol.rolling(5).mean() / out["vol_ma20"].replace(0, np.nan)

    return out


# ═════════════════════════════════════════════════════════
# 市场状态判定
# ═════════════════════════════════════════════════════════

def classify_regime(row: pd.Series) -> tuple[MarketRegime, dict]:
    """
    基于单行特征判定市场状态.

    判定逻辑优先级:
      1. DECLINING:    空头排列 + 动量为负 + 价格在均线下方
      2. RISE_ENDING:  上涨后熵飙升 + 高位放量 + 动量减速
      3. DECLINE_ENDED: 前期下跌后企稳 + 低位缩量 + 跌幅收窄
      4. RISING:       多头排列 + 动量为正
      5. CONSOLIDATION: 其他 (横盘整理)
    """
    details = {}

    ma_bull = row.get("ma_bull", 0)
    ma_bear = row.get("ma_bear", 0)
    price_ma20 = row.get("price_vs_ma20", 0)
    price_ma60 = row.get("price_vs_ma60", 0)
    ret_5d = row.get("ret_5d", 0)
    ret_20d = row.get("ret_20d", 0)
    ret_60d = row.get("ret_60d", 0)
    mom_accel = row.get("momentum_accel", 0)
    entropy_20 = row.get("perm_entropy_20", 0.5)
    entropy_60 = row.get("perm_entropy_60", 0.5)
    entropy_slope = row.get("entropy_slope", 0)
    dom_eig_20 = row.get("dom_eig_20", 0.5)
    vol_ratio = row.get("vol_ratio", 1.0)
    dd_20 = row.get("drawdown_20", 0)
    dd_60 = row.get("drawdown_60", 0)
    bounce_20 = row.get("bounce_20", 0)
    vol_shrink = row.get("vol_shrink_5", 1.0)

    # 用 nan-safe 取值
    def safe(v, default=0.0):
        return default if pd.isna(v) else float(v)

    ma_bull = safe(ma_bull, 0)
    ma_bear = safe(ma_bear, 0)
    price_ma20 = safe(price_ma20, 0)
    price_ma60 = safe(price_ma60, 0)
    ret_5d = safe(ret_5d, 0)
    ret_20d = safe(ret_20d, 0)
    ret_60d = safe(ret_60d, 0)
    mom_accel = safe(mom_accel, 0)
    entropy_20 = safe(entropy_20, 0.5)
    entropy_60 = safe(entropy_60, 0.5)
    entropy_slope = safe(entropy_slope, 0)
    dom_eig_20 = safe(dom_eig_20, 0.5)
    vol_ratio = safe(vol_ratio, 1.0)
    dd_20 = safe(dd_20, 0)
    dd_60 = safe(dd_60, 0)
    bounce_20 = safe(bounce_20, 0)
    vol_shrink = safe(vol_shrink, 1.0)

    details = {
        "ma_bull": ma_bull, "ma_bear": ma_bear,
        "ret_20d": round(ret_20d * 100, 2),
        "entropy_20": round(entropy_20, 4),
        "dom_eig_20": round(dom_eig_20, 4),
        "dd_60": round(dd_60 * 100, 2),
    }

    # ── 1. DECLINING (下跌趋势) ──
    # 空头排列 + 价格在均线下方 + 近期收益为负 + 持续下跌
    declining_score = 0
    if ma_bear >= 2:
        declining_score += 1
    if price_ma20 < -0.02:  # 低于20日均线2%
        declining_score += 1
    if price_ma60 < -0.03:  # 低于60日均线3%
        declining_score += 1
    if ret_20d < -0.05:     # 20日跌幅>5%
        declining_score += 1
    if ret_5d < -0.02:      # 近5日仍在下跌
        declining_score += 1

    if declining_score >= 3:
        return MarketRegime.DECLINING, details

    # ── 2. RISE_ENDING (上涨即将结束) ──
    # 前期有涨幅 + 熵飙升 + 动量减速 + 高位波动率放大
    rise_ending_score = 0
    if ret_60d > 0.10:      # 过去60天涨了10%+
        rise_ending_score += 1
    if entropy_20 > 0.85:   # 短期熵很高(无序)
        rise_ending_score += 1
    if entropy_slope > 0.05:  # 短期熵 > 长期熵(熵扩散)
        rise_ending_score += 1
    if mom_accel < -0.02:   # 动量在减速
        rise_ending_score += 1
    if vol_ratio > 1.3:     # 短期波动率明显高于长期
        rise_ending_score += 1
    if dd_20 < -0.03:       # 已经从20日高点回撤3%+
        rise_ending_score += 1

    if rise_ending_score >= 4:
        return MarketRegime.RISE_ENDING, details

    # ── 3. DECLINE_ENDED (下跌结束/企稳) ──
    # 前期有跌幅 + 跌幅在收窄 + 低位缩量 + 开始企稳
    decline_ended_score = 0
    if ret_60d < -0.05:     # 过去60天跌了
        decline_ended_score += 1
    if ret_20d > ret_60d:   # 近期跌幅收窄
        decline_ended_score += 1
    if ret_5d > -0.01:      # 近5日不再大跌
        decline_ended_score += 1
    if bounce_20 > 0.02:    # 从20日低点反弹了2%+
        decline_ended_score += 1
    if vol_shrink < 0.8:    # 成交缩量
        decline_ended_score += 1
    if entropy_20 < 0.65:   # 短期有序化(可能在筑底)
        decline_ended_score += 1

    if decline_ended_score >= 4:
        return MarketRegime.DECLINE_ENDED, details

    # ── 4. RISING (上涨趋势) ──
    rising_score = 0
    if ma_bull >= 2:
        rising_score += 1
    if price_ma20 > 0.01:
        rising_score += 1
    if price_ma60 > 0.0:
        rising_score += 1
    if ret_20d > 0.02:
        rising_score += 1
    if ret_5d > 0.0:
        rising_score += 1

    if rising_score >= 3:
        return MarketRegime.RISING, details

    # ── 5. CONSOLIDATION (横盘整理) ──
    return MarketRegime.CONSOLIDATION, details


# ═════════════════════════════════════════════════════════
# 批量计算全时间序列的市场状态
# ═════════════════════════════════════════════════════════

def build_market_regime_series(
    index_code: str = "000001_sh",
    index_dir: str = INDEX_DIR,
) -> pd.DataFrame:
    """
    为指数全时间序列计算市场状态.

    Returns
    -------
    DataFrame with columns: trade_date_str, regime, position_scale, + details
    """
    df = load_index(index_code, index_dir)
    df = compute_index_features(df)

    regimes = []
    for i in range(len(df)):
        row = df.iloc[i]
        regime, details = classify_regime(row)
        regimes.append({
            "trade_date_str": str(row["trade_date"]),
            "regime": regime.value,
            "position_scale": REGIME_POSITION_SCALE[regime],
            **details,
        })

    df_regimes = pd.DataFrame(regimes)
    return df_regimes


def get_regime_on_date(
    regime_df: pd.DataFrame,
    date: str,
) -> MarketState:
    """获取指定日期的市场状态"""
    row = regime_df[regime_df["trade_date_str"] == date]
    if len(row) == 0:
        # 找最近的交易日
        all_dates = regime_df["trade_date_str"].values
        prior = [d for d in all_dates if d <= date]
        if not prior:
            return MarketState(
                date=date,
                regime=MarketRegime.CONSOLIDATION,
                position_scale=1.0,
                details={},
            )
        nearest = prior[-1]
        row = regime_df[regime_df["trade_date_str"] == nearest]

    r = row.iloc[0]
    regime = MarketRegime(r["regime"])
    return MarketState(
        date=date,
        regime=regime,
        position_scale=REGIME_POSITION_SCALE[regime],
        details={k: r[k] for k in r.index if k not in ("trade_date_str", "regime", "position_scale")},
    )
