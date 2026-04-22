"""
Bull Hunter v3 — Agent 1: 增量因子生成

职责: 检查 feature-cache 中每只股票的最新日期,
      如果 < scan_date 则跳过 (已是最新), 否则标记需要更新。
      实际特征计算复用 feature_engine + feature_cache 流水线。
      本 Agent 主要做 "增量检查 + 快照提取"。

输入: cache_dir, scan_date
输出: factor_snapshot DataFrame (index=symbol, 全市场因子截面)
"""

from __future__ import annotations

import glob
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 复用已有的因子列表
DAILY_FACTORS = [
    "perm_entropy_s", "perm_entropy_m", "perm_entropy_l",
    "entropy_slope", "entropy_accel",
    "path_irrev_m", "path_irrev_l",
    "dom_eig_m", "dom_eig_l",
    "turnover_entropy_m", "turnover_entropy_l",
    "volatility_m", "volatility_l",
    "vol_compression", "bbw_pctl",
    "vol_ratio_s", "vol_impulse", "vol_shrink", "breakout_range",
    "mf_big_net", "mf_big_net_ratio",
    "mf_big_cumsum_s", "mf_big_cumsum_m", "mf_big_cumsum_l",
    "mf_sm_proportion", "mf_flow_imbalance",
    "mf_big_momentum", "mf_big_streak",
    "coherence_l1", "purity_norm", "von_neumann_entropy", "coherence_decay_rate",
    # ── v2: factor_advisor 建议新增的 8 个衍生因子 ──
    "momentum_5d", "momentum_20d", "momentum_60d", "price_vs_ma20",
    "vol_price_synergy", "volatility_ratio", "mf_reversal_zscore",
    "atr_20d", "close_vs_high_60d",
]

# ── v10: 行业动量/共振因子 (截面聚合, 非缓存) ──
INDUSTRY_FACTORS = [
    "industry_mom_5d",         # 行业中位数 5 日涨幅
    "industry_mom_20d",        # 行业中位数 20 日涨幅
    "industry_breadth_5d",     # 行业中 5 日正收益比例
    "industry_rs_20d",         # 行业 20 日相对全市场超额
    "industry_vol_surge",      # 行业量比中位数
]


def compute_derived_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    从原始 OHLCV + 资金流列计算 9 个新衍生因子。

    需要列: close (必须), high/low/amount/net_mf_amount (可选)。
    在时间序列上计算 rolling 指标, 适用于 agent1 快照提取和 agent2 训练 panel 构建。
    """
    out = df.copy()
    close = out["close"].astype(np.float64)

    # 0. 5日动量 (v10: 供行业因子聚合)
    out["momentum_5d"] = close.pct_change(5)

    # 1. 20日动量
    out["momentum_20d"] = close.pct_change(20)

    # 2. 60日动量
    out["momentum_60d"] = close.pct_change(60)

    # 3. 价格相对20日均线偏离度
    ma20 = close.rolling(20).mean()
    out["price_vs_ma20"] = close / ma20.replace(0, np.nan) - 1

    # 4. 量价协同度
    if "amount" in out.columns:
        amt = out["amount"].astype(np.float64)
        amt_ratio = amt.rolling(5).mean() / amt.rolling(20).mean().replace(0, np.nan)
        ret_5 = close.pct_change(5).clip(lower=0)
        out["vol_price_synergy"] = amt_ratio * ret_5
    else:
        out["vol_price_synergy"] = np.nan

    # 5. 短长期波动率之比
    ret = close.pct_change()
    vol_10 = ret.rolling(10).std()
    vol_60 = ret.rolling(60).std()
    out["volatility_ratio"] = vol_10 / (vol_60 + 1e-8)

    # 6. 资金流反转Z值
    if "net_mf_amount" in out.columns:
        nmf = out["net_mf_amount"].astype(np.float64)
        mf_ma5 = nmf.rolling(5).mean()
        mf_ma20 = nmf.rolling(20).mean()
        mf_std20 = nmf.rolling(20).std()
        out["mf_reversal_zscore"] = (mf_ma5 - mf_ma20) / (mf_std20 + 1e-8)
    else:
        out["mf_reversal_zscore"] = np.nan

    # 7. 20日ATR比率 (振幅/股价)
    if "high" in out.columns and "low" in out.columns:
        atr = (out["high"].astype(np.float64) - out["low"].astype(np.float64)).rolling(20).mean()
        out["atr_20d"] = atr / close.replace(0, np.nan)
    else:
        out["atr_20d"] = np.nan

    # 8. 当前价格相对60日最高价
    if "high" in out.columns:
        high_60 = out["high"].astype(np.float64).rolling(60).max()
        out["close_vs_high_60d"] = close / high_60.replace(0, np.nan)
    else:
        out["close_vs_high_60d"] = np.nan

    return out


def compute_industry_factors(snap: pd.DataFrame) -> pd.DataFrame:
    """
    从全市场截面快照计算行业动量/共振因子 (v10)。

    输入 snap 必须含: _industry, momentum_5d, momentum_20d, vol_ratio_s。
    返回原 snap 附加 INDUSTRY_FACTORS 列。
    """
    if "_industry" not in snap.columns:
        for col in INDUSTRY_FACTORS:
            snap[col] = np.nan
        return snap

    ind = snap["_industry"]

    # 行业 5 日中位数涨幅
    if "momentum_5d" in snap.columns:
        grp5 = snap.groupby(ind)["momentum_5d"]
        snap["industry_mom_5d"] = ind.map(grp5.median())
        # 行业中 5 日正收益比例
        snap["industry_breadth_5d"] = ind.map(
            snap.groupby(ind)["momentum_5d"].apply(lambda x: (x > 0).mean())
        )
    else:
        snap["industry_mom_5d"] = np.nan
        snap["industry_breadth_5d"] = np.nan

    # 行业 20 日中位数涨幅 + 相对全市场
    if "momentum_20d" in snap.columns:
        grp20 = snap.groupby(ind)["momentum_20d"]
        ind_mom20 = grp20.median()
        snap["industry_mom_20d"] = ind.map(ind_mom20)
        market_mom20 = snap["momentum_20d"].median()
        snap["industry_rs_20d"] = snap["industry_mom_20d"] - market_mom20
    else:
        snap["industry_mom_20d"] = np.nan
        snap["industry_rs_20d"] = np.nan

    # 行业量比中位数
    if "vol_ratio_s" in snap.columns:
        snap["industry_vol_surge"] = ind.map(
            snap.groupby(ind)["vol_ratio_s"].median()
        )
    else:
        snap["industry_vol_surge"] = np.nan

    return snap


def preload_factor_cache(
    cache_dir: str,
    min_rows: int = 60,
) -> dict:
    """
    一次性预加载全部因子缓存到内存 (daily + weekly)。
    回测用: 避免每天重复读 ~10000 个 CSV 文件。

    Returns:
        {"daily": {symbol: DataFrame}, "weekly": {symbol: DataFrame}}
    """
    result = {}
    for sub in ("daily", "weekly"):
        sub_dir = os.path.join(cache_dir, sub)
        csv_files = sorted(glob.glob(os.path.join(sub_dir, "*.csv")))
        cache = {}
        for fpath in csv_files:
            symbol = os.path.basename(fpath).replace(".csv", "")
            try:
                df = pd.read_csv(fpath)
            except Exception:
                continue
            if len(df) < min_rows:
                continue
            df = df.sort_values("trade_date").reset_index(drop=True)
            df["trade_date"] = df["trade_date"].astype(str)
            df = compute_derived_factors(df)
            cache[symbol] = df
        logger.info(f"预加载 {sub}: {len(cache)} 只股票")
        result[sub] = cache
    return result


def _snapshot_from_preloaded(
    cache: dict[str, pd.DataFrame],
    scan_date: str,
    min_rows: int = 60,
) -> pd.DataFrame:
    """从预加载的内存缓存中提取指定日期的快照。"""
    rows = []
    for symbol, df in cache.items():
        filtered = df[df["trade_date"] <= scan_date]
        if len(filtered) < min_rows:
            continue
        last = filtered.iloc[-1].to_dict()
        last["symbol"] = symbol
        last["_trade_date"] = str(filtered["trade_date"].iloc[-1])
        last["_avg_amount_20"] = (
            float(filtered.tail(20)["amount"].mean()) if "amount" in filtered.columns else 0.0
        )
        rows.append(last)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("symbol")


def run_factor_generation(
    cache_dir: str,
    data_dir: str,
    scan_date: str,
    basic_path: str = "",
    min_rows: int = 60,
    min_amount: float = 5000.0,
    preloaded: dict | None = None,
) -> pd.DataFrame:
    """
    增量因子生成 + 快照提取。

    1. 检查 cache 是否已包含 scan_date 的数据 (增量检查)
    2. 提取 scan_date 的全市场因子截面

    Returns:
        DataFrame, index=symbol, 含因子列 + 元信息列
    """
    # ── 提取快照 (preloaded 模式 vs 磁盘模式) ──
    basic_info = _load_basic_info(basic_path)

    if preloaded:
        daily_snap = _snapshot_from_preloaded(
            preloaded.get("daily", {}), scan_date, min_rows)
        weekly_snap = _snapshot_from_preloaded(
            preloaded.get("weekly", {}), scan_date, min_rows=30)
    else:
        daily_dir = os.path.join(cache_dir, "daily")

        # ── 增量检查 ──
        csv_files = sorted(glob.glob(os.path.join(daily_dir, "*.csv")))
        if not csv_files:
            logger.error("No daily cache files found")
            return pd.DataFrame()

        sample = csv_files[:10]
        outdated = 0
        for fp in sample:
            try:
                df = pd.read_csv(fp, usecols=["trade_date"])
                latest = str(df["trade_date"].max())
                if latest < scan_date:
                    outdated += 1
            except Exception:
                continue

        if outdated > len(sample) // 2:
            logger.warning(f"Cache may be outdated: {outdated}/{len(sample)} "
                           f"stocks have latest date < {scan_date}. "
                           f"Consider running feature_cache update first.")

        daily_snap = _load_snapshot(daily_dir, scan_date, min_rows)

        weekly_dir = os.path.join(cache_dir, "weekly")
        weekly_snap = _load_snapshot(weekly_dir, scan_date, min_rows=30)

    if daily_snap.empty:
        logger.error("No daily features loaded")
        return pd.DataFrame()

    # 合并周线因子
    if not weekly_snap.empty:
        weekly_renamed = weekly_snap.add_prefix("w_")
        drop_cols = [c for c in weekly_renamed.columns if c.startswith("w__")]
        weekly_renamed.drop(columns=drop_cols, inplace=True, errors="ignore")
        daily_snap = daily_snap.join(weekly_renamed, how="left")

    # 过滤 ST + 低流动性 + 北交所 (因子覆盖差)
    mask = pd.Series(True, index=daily_snap.index)
    for sym in daily_snap.index:
        name = basic_info.get(sym, {}).get("name", "")
        if "ST" in name or "退" in name:
            mask[sym] = False
        # 北交所股票 (bj 开头) 因子覆盖差, 排除
        if sym.startswith("bj"):
            mask[sym] = False
    if "_avg_amount_20" in daily_snap.columns:
        mask = mask & (daily_snap["_avg_amount_20"] >= min_amount)

    daily_snap = daily_snap[mask]

    # 附加元信息
    daily_snap["_name"] = daily_snap.index.map(
        lambda s: basic_info.get(s, {}).get("name", ""))
    daily_snap["_industry"] = daily_snap.index.map(
        lambda s: basic_info.get(s, {}).get("industry", ""))

    # ── v10: 行业动量/共振因子 ──
    daily_snap = compute_industry_factors(daily_snap)

    logger.info(f"Agent 1 完成: {len(daily_snap)} 只股票, scan_date={scan_date}")
    return daily_snap


def _load_snapshot(
    cache_dir: str,
    scan_date: str,
    min_rows: int = 60,
) -> pd.DataFrame:
    """从缓存目录加载截止 scan_date 的最新特征行。"""
    csv_files = sorted(glob.glob(os.path.join(cache_dir, "*.csv")))
    if not csv_files:
        return pd.DataFrame()

    rows = []
    for fpath in csv_files:
        symbol = os.path.basename(fpath).replace(".csv", "")
        try:
            df = pd.read_csv(fpath)
        except Exception:
            continue

        if len(df) < min_rows:
            continue

        df = df.sort_values("trade_date").reset_index(drop=True)
        df["trade_date"] = df["trade_date"].astype(str)

        if scan_date:
            df = df[df["trade_date"] <= scan_date]
            if len(df) < min_rows:
                continue

        # 计算衍生因子 (rolling)
        df = compute_derived_factors(df)

        last = df.iloc[-1].to_dict()
        last["symbol"] = symbol
        last["_trade_date"] = str(df["trade_date"].iloc[-1])
        last["_avg_amount_20"] = (
            float(df.tail(20)["amount"].mean()) if "amount" in df.columns else 0.0
        )
        rows.append(last)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("symbol")


def _load_basic_info(basic_path: str) -> dict[str, dict]:
    if not basic_path or not os.path.exists(basic_path):
        return {}
    df = pd.read_csv(basic_path, dtype=str)
    info = {}
    for _, row in df.iterrows():
        ts = str(row.get("ts_code", ""))
        parts = ts.split(".")
        if len(parts) == 2:
            sym = parts[1].lower() + parts[0]
            info[sym] = {
                "name": str(row.get("name", "")),
                "industry": str(row.get("industry", "")),
            }
    return info
