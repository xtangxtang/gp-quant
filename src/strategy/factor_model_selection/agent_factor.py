"""
因子模型选股 — Agent 1: 因子计算

职责: 从特征缓存中提取截止 scan_date 的全市场因子截面。
输出: PipelineState.factor_snapshot (DataFrame, index=symbol)

不做选股、不做回测，只负责 "获取当前所有股票的参数值"。
"""

from __future__ import annotations

import glob
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_factor_snapshot(
    cache_dir: str,
    scan_date: str,
    basic_path: str = "",
    min_rows: int = 60,
    min_amount: float = 5000.0,
    exclude_st: bool = True,
) -> pd.DataFrame:
    """
    从特征缓存加载截止 scan_date 的全市场因子截面 (每只股票一行)。

    返回 DataFrame, index=symbol, 列=因子名 + 元信息列:
      _trade_date: 该股票特征对应的最后交易日
      _avg_amount_20: 近 20 日均成交额
      _name: 股票名称 (如果有 basic_path)
      _industry: 行业 (如果有 basic_path)
    """
    # ── 加载基本信息 ──
    basic_info = _load_basic_info(basic_path)

    # ── 日线因子 ──
    daily_dir = os.path.join(cache_dir, "daily")
    daily_snap = _load_snapshot(daily_dir, scan_date, min_rows)

    # ── 周线因子 ──
    weekly_dir = os.path.join(cache_dir, "weekly")
    weekly_snap = _load_snapshot(weekly_dir, scan_date, min_rows=30)

    if daily_snap.empty:
        logger.error("No daily features loaded")
        return pd.DataFrame()

    # ── 合并周线因子 (加前缀 w_) ──
    if not weekly_snap.empty:
        weekly_renamed = weekly_snap.add_prefix("w_")
        weekly_renamed.rename(columns={"w__trade_date": "_weekly_trade_date"}, inplace=True)
        # 只保留因子列 (去掉 w__avg_amount_20 等重复元信息)
        drop_cols = [c for c in weekly_renamed.columns if c.startswith("w__") and c != "_weekly_trade_date"]
        weekly_renamed.drop(columns=drop_cols, inplace=True, errors="ignore")
        daily_snap = daily_snap.join(weekly_renamed, how="left")

    # ── 过滤 ──
    mask = pd.Series(True, index=daily_snap.index)

    # ST 过滤
    if exclude_st:
        for sym in daily_snap.index:
            name = basic_info.get(sym, {}).get("name", "")
            if "ST" in name or "退" in name:
                mask[sym] = False

    # 流动性过滤
    if "_avg_amount_20" in daily_snap.columns:
        mask = mask & (daily_snap["_avg_amount_20"] >= min_amount)

    before = len(daily_snap)
    daily_snap = daily_snap[mask]
    logger.info(f"Factor snapshot: {before} → {len(daily_snap)} stocks after filtering")

    # ── 附加基本信息 ──
    daily_snap["_name"] = daily_snap.index.map(lambda s: basic_info.get(s, {}).get("name", ""))
    daily_snap["_industry"] = daily_snap.index.map(lambda s: basic_info.get(s, {}).get("industry", ""))

    logger.info(f"Agent 1 完成: {len(daily_snap)} 只股票, "
                f"scan_date={scan_date}, "
                f"日线因子 {sum(1 for c in daily_snap.columns if not c.startswith('_'))} 个, "
                f"周线因子 {sum(1 for c in daily_snap.columns if c.startswith('w_'))} 个")

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
        last["_avg_amount_20"] = float(df.tail(20)["amount"].mean()) if "amount" in df.columns else 0.0

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
            info[sym] = {"name": str(row.get("name", "")), "industry": str(row.get("industry", ""))}
    return info
