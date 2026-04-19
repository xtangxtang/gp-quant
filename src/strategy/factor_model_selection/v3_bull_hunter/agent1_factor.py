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
]


def run_factor_generation(
    cache_dir: str,
    data_dir: str,
    scan_date: str,
    basic_path: str = "",
    min_rows: int = 60,
    min_amount: float = 5000.0,
) -> pd.DataFrame:
    """
    增量因子生成 + 快照提取。

    1. 检查 cache 是否已包含 scan_date 的数据 (增量检查)
    2. 提取 scan_date 的全市场因子截面

    Returns:
        DataFrame, index=symbol, 含因子列 + 元信息列
    """
    daily_dir = os.path.join(cache_dir, "daily")

    # ── 增量检查 ──
    csv_files = sorted(glob.glob(os.path.join(daily_dir, "*.csv")))
    if not csv_files:
        logger.error("No daily cache files found")
        return pd.DataFrame()

    # 抽样检查 10 只股票的最新日期
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

    # ── 提取快照 ──
    basic_info = _load_basic_info(basic_path)
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
