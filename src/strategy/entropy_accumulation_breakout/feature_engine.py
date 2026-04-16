"""
熵惜售分岔突破策略 — 特征引擎

理论基础:
──────────────────────────────────────────────────────────────────
1. Seifert (2025): 粗粒化轨迹的熵产生下界
   → path_irreversibility 是保守估计，检测到的不可逆性至少这么强

2. Fan et al. (2025): "Instability of Financial Time Series Revealed
   by Irreversibility Analysis" (Entropy 27(4):402)
   → KLD + DHVG 滑动窗口检测金融时序不稳定性; KLD 优于矩统计量

3. Bielinskyi et al. (2025): "Early Warning Signs: Evaluating
   Permutation Entropy Metrics for Stock Market Crashes"
   → 加权置换熵 (WPE) 捕捉不同市场复杂度特征

4. Dmitriev et al. (2025): "Self-organization of the stock exchange
   to the edge of a phase transition" (Front. Phys. 12:1508465)
   → 熵作为控制参数, 成交量为序参量;
     自组织到相变边缘; 临界减速测度 (方差, AR(1), 峰度, 偏度)

5. Yan et al. (2023): "Thermodynamic and dynamical predictions for
   bifurcations" (Commun. Phys. 6:16)
   → 熵产生率在分岔点达峰, 可作为分岔预测指标

6. Ardakani (2025): "Detecting Financial Bubbles with Tail-Weighted
   Entropy" → 尾部加权熵检测泡沫 / 结构崩塌

核心逻辑 — 三阶段状态机:
──────────────────────────────────────────────────────────────────
Phase 1  惜售吸筹 (Accumulation)
  - 置换熵持续低位 (有序化, 筹码集中)
  - 路径不可逆性上升 (定向力量在运作)
  - 换手率萎缩 + 换手率熵下降 (流动性收缩 = 惜售)
  - 波动率压缩 (布林带收窄)

Phase 2  分岔突破 (Bifurcation Breakout)
  - 主特征值 → 1 (临界减速 = 即将分岔)
  - 量能脉冲 (成交量突然放大 = 能量注入打破对称性)
  - 价格突破压力位
  - 熵仍然保持低位 (有序突破, 非噪声)

Phase 3  结构崩塌退出 (Structural Collapse Exit)
  - 置换熵快速扩张 (有序结构瓦解)
  - 路径不可逆性骤降 (主力撤离)
  - 换手率熵飙升 (混乱交易)
  - 多尺度熵发散 (日线 vs 周线信号矛盾)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.tick_entropy import (
    dominant_eigenvalue_from_autocorr,
    path_irreversibility_entropy,
    permutation_entropy,
    rolling_dominant_eigenvalue,
    rolling_path_irreversibility,
    rolling_permutation_entropy,
    rolling_turnover_entropy,
    _discretize_trinary,
    _rolling_apply_1d,
)
from src.core.quantum_coherence import compute_quantum_coherence_features


# ─────────────────────────────────────────────────────────
# 辅助: 从日线 DataFrame 计算收益率序列
# ─────────────────────────────────────────────────────────

def _returns(close: pd.Series) -> pd.Series:
    return close.pct_change().fillna(0.0)


def _order_flow_proxy(df: pd.DataFrame) -> pd.Series:
    """用 net_mf_amount 或 (close-open)/range 作为 order flow 代理"""
    if "net_mf_amount" in df.columns:
        return df["net_mf_amount"].fillna(0.0)
    c, o, h, l = df["close"], df["open"], df["high"], df["low"]
    rng = h - l
    return ((c - o) / rng.replace(0, np.nan)).fillna(0.0)


# ─────────────────────────────────────────────────────────
# 单时间框架特征计算
# ─────────────────────────────────────────────────────────

def compute_single_timeframe_features(
    df: pd.DataFrame,
    windows: dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    从日线 / 周线 DataFrame 计算全部熵-分岔特征.

    Parameters
    ----------
    df : DataFrame
        必须包含 trade_date, open, high, low, close, vol, amount 列.
        可选: turnover_rate, net_mf_amount, buy_*_amount, sell_*_amount.
    windows : dict
        窗口参数, 默认适用日线级别.

    Returns
    -------
    DataFrame  原始 df 追加多列熵/分岔特征.
    """
    w = windows or {
        "short": 10,
        "medium": 20,
        "long": 60,
    }
    ws, wm, wl = w["short"], w["medium"], w["long"]

    out = df.copy()
    close = out["close"].astype(np.float64)
    vol = out["vol"].astype(np.float64)
    ret = _returns(close)
    oflow = _order_flow_proxy(out)

    # ── 1) 置换熵 (Permutation Entropy) ──
    out["perm_entropy_s"] = rolling_permutation_entropy(close, window=ws, order=3)
    out["perm_entropy_m"] = rolling_permutation_entropy(close, window=wm, order=3)
    out["perm_entropy_l"] = rolling_permutation_entropy(close, window=wl, order=3)

    # 多尺度熵斜率: 短窗口 vs 长窗口的变化率
    out["entropy_slope"] = out["perm_entropy_s"] - out["perm_entropy_l"]

    # 熵加速度: 熵短期变化率
    out["entropy_accel"] = out["perm_entropy_s"].diff(w.get("accel", 5))

    # ── 2) 路径不可逆性 (Path Irreversibility) ──
    out["path_irrev_m"] = rolling_path_irreversibility(ret, oflow, window=wm)
    out["path_irrev_l"] = rolling_path_irreversibility(ret, oflow, window=wl)

    # ── 3) 主特征值 (Dominant Eigenvalue) — 临界减速指标 ──
    out["dom_eig_m"] = rolling_dominant_eigenvalue(close, window=wm, order=2)
    out["dom_eig_l"] = rolling_dominant_eigenvalue(close, window=wl, order=2)

    # ── 4) 换手率熵 (Turnover Entropy) ──
    if "turnover_rate" in out.columns:
        tr = out["turnover_rate"].astype(np.float64)
        out["turnover_entropy_m"] = rolling_turnover_entropy(tr, window=wm)
        out["turnover_entropy_l"] = rolling_turnover_entropy(tr, window=wl)
    else:
        out["turnover_entropy_m"] = np.nan
        out["turnover_entropy_l"] = np.nan

    # ── 5) 波动率特征 ──
    out["volatility_m"] = ret.rolling(wm).std()
    out["volatility_l"] = ret.rolling(wl).std()
    out["vol_compression"] = out["volatility_m"] / out["volatility_l"].replace(0, np.nan)

    # 布林带宽度 (BBW) — 压缩度
    ma = close.rolling(wm).mean()
    std = close.rolling(wm).std()
    out["bbw"] = (2 * std / ma.replace(0, np.nan))

    # BBW 分位数 (过去 120 天)
    out["bbw_pctl"] = out["bbw"].rolling(120, min_periods=60).apply(
        lambda x: (x.iloc[-1] <= x).mean() if len(x) > 0 else np.nan,
        raw=False,
    )

    # ── 6) 成交量特征 ──
    vol_ma = vol.rolling(wm).mean()
    out["vol_ratio_s"] = vol.rolling(ws).mean() / vol_ma.replace(0, np.nan)
    out["vol_impulse"] = vol / vol_ma.replace(0, np.nan)

    # 量缩判断: 近期成交量相对长期均值的萎缩程度
    vol_ma_l = vol.rolling(wl).mean()
    out["vol_shrink"] = vol.rolling(ws).mean() / vol_ma_l.replace(0, np.nan)

    # ── 7) 价格位置特征 (保留计算供其他模块使用) ──
    high_m = close.rolling(wm).max()
    low_m = close.rolling(wm).min()
    out["breakout_range"] = (close - low_m) / (high_m - low_m).replace(0, np.nan)

    # ── 8) 资金流特征 (如果有) ──
    if "net_mf_amount" in out.columns:
        nmf = out["net_mf_amount"].astype(np.float64)
        out["mf_cumsum_s"] = nmf.rolling(ws).sum()
        out["mf_cumsum_m"] = nmf.rolling(wm).sum()
        out["mf_impulse"] = nmf / nmf.rolling(wm).std().replace(0, np.nan)
    else:
        out["mf_cumsum_s"] = np.nan
        out["mf_cumsum_m"] = np.nan
        out["mf_impulse"] = np.nan

    # ── 9) 大单净额占比 (如果有) ──
    if "buy_elg_amount" in out.columns and "sell_elg_amount" in out.columns:
        buy_big = out["buy_elg_amount"].fillna(0) + out.get("buy_lg_amount", pd.Series(0, index=out.index)).fillna(0)
        sell_big = out["sell_elg_amount"].fillna(0) + out.get("sell_lg_amount", pd.Series(0, index=out.index)).fillna(0)
        total = out["amount"].replace(0, np.nan)
        out["big_net_ratio"] = (buy_big - sell_big) / total
        out["big_net_ratio_ma"] = out["big_net_ratio"].rolling(ws).mean()
    else:
        out["big_net_ratio"] = np.nan
        out["big_net_ratio_ma"] = np.nan

    # ── 10) 量子相干性特征 (Quantum Coherence) ──
    # 密度矩阵 + 退相干速率: 度量市场从"叠加"到"方向确认"的速度
    if (
        "perm_entropy_m" in out.columns
        and "path_irrev_m" in out.columns
        and "dom_eig_m" in out.columns
    ):
        qc = compute_quantum_coherence_features(
            out["perm_entropy_m"],
            out["path_irrev_m"],
            out["dom_eig_m"],
            rho_window=wm,
            decay_window=w.get("accel", 5),
        )
        for col in qc.columns:
            out[col] = qc[col].values

    return out


# ─────────────────────────────────────────────────────────
# 聚合周线
# ─────────────────────────────────────────────────────────

def aggregate_to_weekly(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    将日线 DataFrame 聚合为周线.
    要求 trade_date 列为 YYYYMMDD 字符串.
    """
    tmp = df_daily.copy()
    tmp["_dt"] = pd.to_datetime(tmp["trade_date"].astype(str), format="%Y%m%d")
    tmp = tmp.set_index("_dt").sort_index()

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "vol": "sum",
        "amount": "sum",
        "trade_date": "last",
    }
    if "turnover_rate" in tmp.columns:
        agg["turnover_rate"] = "sum"
    if "net_mf_amount" in tmp.columns:
        agg["net_mf_amount"] = "sum"

    weekly = tmp.resample("W-FRI").agg(agg).dropna(subset=["close"])
    weekly = weekly.reset_index(drop=True)
    return weekly


# ─────────────────────────────────────────────────────────
# 完整特征构建入口
# ─────────────────────────────────────────────────────────

def build_features(
    df_daily: pd.DataFrame,
    daily_windows: dict[str, int] | None = None,
    weekly_windows: dict[str, int] | None = None,
    skip_weekly: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    构建日线 + 周线双时间框架特征.

    Returns
    -------
    dict with keys 'daily', 'weekly', each containing a featured DataFrame.
    """
    daily_w = daily_windows or {"short": 10, "medium": 20, "long": 60, "accel": 5}
    weekly_w = weekly_windows or {"short": 4, "medium": 8, "long": 24, "accel": 2}

    df_d = compute_single_timeframe_features(df_daily, daily_w)

    # 周线特征 (回测模式可跳过以加速)
    df_w = None
    if not skip_weekly:
        df_w_raw = aggregate_to_weekly(df_daily)
        df_w = compute_single_timeframe_features(df_w_raw, weekly_w) if len(df_w_raw) >= 12 else df_w_raw

    return {"daily": df_d, "weekly": df_w}
