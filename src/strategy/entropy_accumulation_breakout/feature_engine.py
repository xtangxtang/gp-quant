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

import os
import glob
import logging

logger = logging.getLogger(__name__)


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
# 周线数据: 从 tushare-weekly-5d 预计算文件加载
# ─────────────────────────────────────────────────────────

def load_weekly_precomputed(data_root: str, symbol: str) -> pd.DataFrame | None:
    """
    从 tushare-weekly-5d/{symbol}.csv 加载预计算周线数据.
    比日线 resample 更丰富: 包含 pe, pb, turnover_rate, dv_ratio, 资金流等.
    """
    fpath = os.path.join(data_root, "tushare-weekly-5d", f"{symbol}.csv")
    if not os.path.exists(fpath):
        return None
    try:
        df = pd.read_csv(fpath)
    except Exception:
        return None
    if "trade_date" not in df.columns or "close" not in df.columns:
        return None
    df["trade_date"] = df["trade_date"].astype(str)
    return df.sort_values("trade_date").reset_index(drop=True)


# ─────────────────────────────────────────────────────────
# 分钟数据: 从 trade/{symbol}/YYYY-MM-DD.csv 加载
# ─────────────────────────────────────────────────────────

_MINUTE_COL_MAP = {
    "时间": "datetime",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量(手)": "vol",
    "成交额(元)": "amount",
    "均价": "avg_price",
    "换手率(%)": "turnover_pct",
}


def _load_minute_day(data_root: str, symbol: str, date_str: str) -> pd.DataFrame | None:
    """加载单日分钟数据. date_str 格式 YYYYMMDD → 文件名 YYYY-MM-DD.csv"""
    date_fmt = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    fpath = os.path.join(data_root, "trade", symbol, f"{date_fmt}.csv")
    if not os.path.exists(fpath):
        return None
    try:
        df = pd.read_csv(fpath)
    except Exception:
        return None
    df = df.rename(columns=_MINUTE_COL_MAP)
    return df


def compute_minute_features(
    data_root: str,
    symbol: str,
    trade_dates: list[str],
    n_days: int = 5,
) -> dict[str, float]:
    """
    从最近 n_days 天的分钟线数据计算日内微观结构特征.

    返回 dict, 可直接合并到日线末行:
      - intraday_perm_entropy: 日内置换熵均值 (价格微观有序度)
      - intraday_path_irrev:   日内路径不可逆性均值 (日内定向力量)
      - intraday_vol_concentration: 成交量集中度 (HHI, 高=大单/集中时段主导)
      - intraday_range_ratio: 日内振幅均值 (high-low)/close
    """
    recent_dates = trade_dates[-n_days:]
    pe_vals, irrev_vals, hhi_vals, range_vals = [], [], [], []

    for dt in recent_dates:
        df_min = _load_minute_day(data_root, symbol, dt)
        if df_min is None or len(df_min) < 30:
            continue

        close = df_min["close"].astype(np.float64).values
        vol = df_min["vol"].astype(np.float64).values

        # 日内置换熵
        pe_val = permutation_entropy(close, order=3)
        if pe_val is not None and not np.isnan(pe_val):
            pe_vals.append(pe_val)

        # 日内路径不可逆性
        ret = np.diff(close) / np.maximum(close[:-1], 1e-8)
        of = np.diff(vol) / np.maximum(vol[:-1], 1e-8)
        if len(ret) >= 20:
            irrev = path_irreversibility_entropy(ret[:len(of)], of)
            if irrev is not None and not np.isnan(irrev):
                irrev_vals.append(irrev)

        # 成交量集中度 (HHI): 每分钟成交量占比的平方和
        total_vol = vol.sum()
        if total_vol > 0:
            shares = vol / total_vol
            hhi = float(np.sum(shares ** 2))
            hhi_vals.append(hhi)

        # 日内振幅
        if "high" in df_min.columns and "low" in df_min.columns:
            day_high = df_min["high"].max()
            day_low = df_min["low"].min()
            day_close = close[-1]
            if day_close > 0:
                range_vals.append((day_high - day_low) / day_close)

    return {
        "intraday_perm_entropy": float(np.mean(pe_vals)) if pe_vals else np.nan,
        "intraday_path_irrev": float(np.mean(irrev_vals)) if irrev_vals else np.nan,
        "intraday_vol_concentration": float(np.mean(hhi_vals)) if hhi_vals else np.nan,
        "intraday_range_ratio": float(np.mean(range_vals)) if range_vals else np.nan,
    }


# ─────────────────────────────────────────────────────────
# 资金流向数据: 从 tushare-moneyflow 加载
# ─────────────────────────────────────────────────────────

def load_moneyflow(data_root: str, symbol: str) -> pd.DataFrame | None:
    """从 tushare-moneyflow/{symbol}.csv 加载资金流向数据."""
    fpath = os.path.join(data_root, "tushare-moneyflow", f"{symbol}.csv")
    if not os.path.exists(fpath):
        return None
    try:
        df = pd.read_csv(fpath)
    except Exception:
        return None
    if "trade_date" not in df.columns:
        return None
    df["trade_date"] = df["trade_date"].astype(str)
    return df.sort_values("trade_date").reset_index(drop=True)


def compute_moneyflow_features(
    df_mf: pd.DataFrame,
    windows: dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    从资金流向数据计算多维度资金特征.

    资金分层:
      - 超大单 (elg): 机构主力
      - 大单 (lg):     大户/游资
      - 中单 (md):     中等资金
      - 小单 (sm):     散户

    返回与 df_mf 同 index 的 DataFrame, 列:
      - mf_big_net:         大单+超大单净额 (buy - sell)
      - mf_big_net_ratio:   大单净额占总净额比
      - mf_big_cumsum_s:    大单净额短期累计
      - mf_big_cumsum_m:    大单净额中期累计
      - mf_sm_proportion:   散户成交占比 (高=散户主导)
      - mf_flow_imbalance:  资金流不平衡度 (大单-散户方向差异)
      - mf_big_momentum:    大单动量 (净额变化率)
    """
    w = windows or {"short": 5, "medium": 10, "long": 20}
    ws, wm, wl = w["short"], w["medium"], w["long"]
    out = pd.DataFrame(index=df_mf.index)

    # 计算各档净额
    buy_big = df_mf.get("buy_elg_amount", pd.Series(0, index=df_mf.index)).fillna(0) + \
              df_mf.get("buy_lg_amount", pd.Series(0, index=df_mf.index)).fillna(0)
    sell_big = df_mf.get("sell_elg_amount", pd.Series(0, index=df_mf.index)).fillna(0) + \
               df_mf.get("sell_lg_amount", pd.Series(0, index=df_mf.index)).fillna(0)
    buy_sm = df_mf.get("buy_sm_amount", pd.Series(0, index=df_mf.index)).fillna(0)
    sell_sm = df_mf.get("sell_sm_amount", pd.Series(0, index=df_mf.index)).fillna(0)

    total_buy = buy_big + df_mf.get("buy_md_amount", pd.Series(0, index=df_mf.index)).fillna(0) + buy_sm
    total_sell = sell_big + df_mf.get("sell_md_amount", pd.Series(0, index=df_mf.index)).fillna(0) + sell_sm
    total_amount = (total_buy + total_sell).replace(0, np.nan)

    big_net = buy_big - sell_big
    sm_net = buy_sm - sell_sm

    # 大单净额
    out["mf_big_net"] = big_net
    out["mf_big_net_ratio"] = big_net / total_amount

    # 累计净额
    out["mf_big_cumsum_s"] = big_net.rolling(ws, min_periods=1).sum()
    out["mf_big_cumsum_m"] = big_net.rolling(wm, min_periods=1).sum()
    out["mf_big_cumsum_l"] = big_net.rolling(wl, min_periods=1).sum()

    # 散户占比: 散户成交额 / 总成交额
    out["mf_sm_proportion"] = (buy_sm + sell_sm) / total_amount

    # 资金流不平衡: 大单方向 vs 散户方向 (正=大单买散户卖=吸筹)
    big_dir = big_net / big_net.abs().replace(0, np.nan)
    sm_dir = sm_net / sm_net.abs().replace(0, np.nan)
    out["mf_flow_imbalance"] = (big_dir.fillna(0) - sm_dir.fillna(0)).rolling(ws, min_periods=1).mean()

    # 大单动量: 净额的变化率
    out["mf_big_momentum"] = big_net.rolling(ws, min_periods=1).mean() - \
                              big_net.rolling(wm, min_periods=ws).mean()

    # 大单持续性: 连续为正的天数
    is_positive = (big_net > 0).astype(int)
    # cumcount trick: 连续正天数
    groups = (is_positive != is_positive.shift()).cumsum()
    out["mf_big_streak"] = is_positive.groupby(groups).cumsum()

    return out


# ─────────────────────────────────────────────────────────
# 周线特征增强 (估值 + 资金流)
# ─────────────────────────────────────────────────────────

def compute_weekly_extra_features(df_weekly: pd.DataFrame) -> pd.DataFrame:
    """
    从预计算周线中提取额外特征 (PE/PB估值 + 周线资金流).
    这些列只有 tushare-weekly-5d 才有, resample 无法获得.
    """
    out = df_weekly.copy()

    # PE / PB 分位数 (用于估值过滤)
    for col in ("pe_ttm", "pb"):
        if col in out.columns:
            vals = out[col].astype(float)
            pctl_col = f"{col}_pctl"
            out[pctl_col] = vals.rolling(52, min_periods=12).apply(
                lambda x: (x.iloc[-1] <= x).mean() if len(x) > 0 else np.nan,
                raw=False,
            )

    # 周线大单净额 (如果周线数据有)
    if "buy_elg_amount" in out.columns and "sell_elg_amount" in out.columns:
        buy_big_w = out["buy_elg_amount"].fillna(0) + out.get("buy_lg_amount", pd.Series(0, index=out.index)).fillna(0)
        sell_big_w = out["sell_elg_amount"].fillna(0) + out.get("sell_lg_amount", pd.Series(0, index=out.index)).fillna(0)
        out["weekly_big_net"] = buy_big_w - sell_big_w
        out["weekly_big_net_cumsum"] = out["weekly_big_net"].rolling(4, min_periods=1).sum()
    else:
        out["weekly_big_net"] = np.nan
        out["weekly_big_net_cumsum"] = np.nan

    # 周线换手率趋势
    if "turnover_rate" in out.columns:
        tr = out["turnover_rate"].astype(float)
        out["weekly_turnover_ma4"] = tr.rolling(4, min_periods=1).mean()
        out["weekly_turnover_shrink"] = tr / tr.rolling(12, min_periods=4).mean().replace(0, np.nan)

    return out


# ─────────────────────────────────────────────────────────
# 完整特征构建入口
# ─────────────────────────────────────────────────────────

def _merge_moneyflow(df_featured: pd.DataFrame, data_root: str, symbol: str) -> pd.DataFrame:
    """加载并 merge 资金流特征到日线 DataFrame."""
    df_mf = load_moneyflow(data_root, symbol)
    if df_mf is None or len(df_mf) == 0:
        return df_featured
    mf_feats = compute_moneyflow_features(df_mf, {
        "short": 10, "medium": 20, "long": 60,
    })
    mf_feats["trade_date"] = df_mf["trade_date"].astype(str).values
    df_featured["trade_date"] = df_featured["trade_date"].astype(str)
    df_featured = df_featured.merge(mf_feats, on="trade_date", how="left")
    return df_featured


def build_features(
    df_daily: pd.DataFrame,
    daily_windows: dict[str, int] | None = None,
    weekly_windows: dict[str, int] | None = None,
    skip_weekly: bool = False,
    data_root: str = "",
    symbol: str = "",
    cache_dir: str = "",
) -> dict[str, pd.DataFrame]:
    """
    构建日线 + 周线 + 分钟 + 资金流 多源特征.

    Parameters
    ----------
    data_root : str
        数据根目录 (包含 tushare-weekly-5d/, trade/, tushare-moneyflow/).
        为空时退化为仅使用日线数据 (向后兼容).
    symbol : str
        股票代码, 用于加载周线/分钟/资金流文件.
    cache_dir : str
        特征缓存目录. 为空时不使用缓存(全量计算).
        非空时启用增量模式: 只计算新增行, 历史特征从缓存读取.

    Returns
    -------
    dict with keys 'daily', 'weekly', each containing a featured DataFrame.
    """
    daily_w = daily_windows or {"short": 10, "medium": 20, "long": 60, "accel": 5}
    weekly_w = weekly_windows or {"short": 4, "medium": 8, "long": 24, "accel": 2}

    compute_daily = lambda df: compute_single_timeframe_features(df, daily_w)
    compute_weekly = lambda df: compute_single_timeframe_features(df, weekly_w)

    # ── 日线特征 (支持缓存) ──
    if cache_dir and symbol:
        from .feature_cache import get_cached_daily_features, get_cached_weekly_features

        moneyflow_fn = (lambda df, dr, sym: _merge_moneyflow(df, dr, sym)) if data_root else None

        df_d = get_cached_daily_features(
            cache_dir=cache_dir,
            symbol=symbol,
            df_daily_raw=df_daily,
            compute_fn=compute_daily,
            moneyflow_merge_fn=moneyflow_fn,
            data_root=data_root,
        )
    else:
        df_d = compute_daily(df_daily)

        # 资金流特征 (merge 到日线)
        if data_root and symbol:
            df_d = _merge_moneyflow(df_d, data_root, symbol)

    # ── 分钟特征 (最近 N 天, 追加到最后一行, 不缓存) ──
    if data_root and symbol and not skip_weekly:
        trade_dates = df_d["trade_date"].astype(str).tolist()
        minute_feats = compute_minute_features(data_root, symbol, trade_dates, n_days=5)
        for k, v in minute_feats.items():
            df_d[k] = np.nan
            if pd.notna(v):
                df_d.loc[df_d.index[-1], k] = v

    # ── 周线特征 (支持缓存) ──
    df_w = None
    if not skip_weekly:
        # 优先使用预计算周线 (数据更丰富: PE/PB/资金流)
        df_w_raw = None
        if data_root and symbol:
            df_w_raw = load_weekly_precomputed(data_root, symbol)

        if df_w_raw is not None and len(df_w_raw) >= 12:
            if cache_dir and symbol:
                df_w = get_cached_weekly_features(
                    cache_dir=cache_dir,
                    symbol=symbol,
                    df_weekly_raw=df_w_raw,
                    compute_fn=compute_weekly,
                    extra_fn=compute_weekly_extra_features,
                )
            else:
                df_w = compute_weekly(df_w_raw)
                df_w = compute_weekly_extra_features(df_w)
        else:
            # fallback: 从日线 resample
            df_w_raw = aggregate_to_weekly(df_daily)
            if len(df_w_raw) >= 12:
                if cache_dir and symbol:
                    df_w = get_cached_weekly_features(
                        cache_dir=cache_dir,
                        symbol=symbol,
                        df_weekly_raw=df_w_raw,
                        compute_fn=compute_weekly,
                    )
                else:
                    df_w = compute_weekly(df_w_raw)
            else:
                df_w = df_w_raw

    return {"daily": df_d, "weekly": df_w}
