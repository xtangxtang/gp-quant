"""
特征缓存 — 增量计算模式

思路:
─────────────────────────────────────────────────────────
1. 首次运行: 全量计算 500 天特征 → 存 CSV
2. 后续运行: 读缓存 → 只对新增行做增量计算 → 追加到缓存
3. 增量计算: 取缓存末尾 max_window 行 + 新数据 → rolling → 只取新行

缓存文件结构:
  {cache_dir}/daily/{symbol}.csv     — 日线特征缓存
  {cache_dir}/weekly/{symbol}.csv    — 周线特征缓存
  {cache_dir}/moneyflow/{symbol}.csv — 资金流特征缓存

缓存有效性:
  - 通过 trade_date 判断: 缓存最后日期 vs 原始数据最后日期
  - 如果缓存最后日期 == 原始数据最后日期 → 直接返回 (命中)
  - 如果缓存最后日期 <  原始数据最后日期 → 增量计算新行
  - 如果缓存最后日期 >  原始数据最后日期 → 不应发生, 重建
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 增量计算需要的最大回看窗口 (BBW 分位数需要 120 天)
_MAX_LOOKBACK = 130


def _cache_path(cache_dir: str, sub: str, symbol: str) -> str:
    return os.path.join(cache_dir, sub, f"{symbol}.csv")


def _read_cache(fpath: str) -> pd.DataFrame | None:
    if not os.path.exists(fpath):
        return None
    try:
        df = pd.read_csv(fpath)
        if "trade_date" not in df.columns or len(df) == 0:
            return None
        df["trade_date"] = df["trade_date"].astype(str)
        return df
    except Exception:
        return None


def _write_cache(fpath: str, df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    df.to_csv(fpath, index=False)


def _last_date(df: pd.DataFrame) -> str:
    return str(df["trade_date"].iloc[-1])


def get_cached_daily_features(
    cache_dir: str,
    symbol: str,
    df_daily_raw: pd.DataFrame,
    compute_fn,
    moneyflow_merge_fn=None,
    data_root: str = "",
) -> pd.DataFrame:
    """
    获取日线特征 (缓存优先 + 增量计算).

    Parameters
    ----------
    cache_dir : str
        缓存根目录
    symbol : str
        股票代码
    df_daily_raw : pd.DataFrame
        原始日线数据 (含 trade_date, OHLCV 等), 已排序
    compute_fn : callable
        (df_raw) -> df_featured 的计算函数
    moneyflow_merge_fn : callable or None
        (df_featured, data_root, symbol) -> df_featured 的资金流 merge 函数
    data_root : str
        数据根目录

    Returns
    -------
    pd.DataFrame — 带特征的日线 DataFrame
    """
    fpath = _cache_path(cache_dir, "daily", symbol)
    df_raw = df_daily_raw.copy()
    df_raw["trade_date"] = df_raw["trade_date"].astype(str)
    raw_last = _last_date(df_raw)

    cached = _read_cache(fpath)

    # 命中: 缓存已包含最新数据
    if cached is not None and _last_date(cached) >= raw_last:
        # 截取到 raw_last (防止缓存比原始数据多)
        cached = cached[cached["trade_date"] <= raw_last].copy()
        return cached

    # 增量: 找出新增的行
    if cached is not None:
        cached_last = _last_date(cached)
        new_dates = df_raw[df_raw["trade_date"] > cached_last]
        n_new = len(new_dates)

        if n_new == 0:
            return cached

        if n_new > 50:
            # 新增太多 (>50天), 不如全量重算
            logger.info("[cache] %s: %d new rows, full recompute", symbol, n_new)
            cached = None
        else:
            logger.debug("[cache] %s: %d new rows, incremental", symbol, n_new)

    if cached is not None:
        # 增量计算: 取缓存末尾 + 新数据, 一起算 rolling, 只保留新行
        cached_last = _last_date(cached)
        # 需要足够的原始数据做 rolling 窗口
        overlap_start_idx = max(0, len(df_raw[df_raw["trade_date"] <= cached_last]) - _MAX_LOOKBACK)
        df_for_compute = df_raw.iloc[overlap_start_idx:].reset_index(drop=True)

        df_featured = compute_fn(df_for_compute)

        # 资金流 merge
        if moneyflow_merge_fn is not None:
            df_featured = moneyflow_merge_fn(df_featured, data_root, symbol)

        # 只取新增的行 (trade_date > cached_last)
        df_featured["trade_date"] = df_featured["trade_date"].astype(str)
        new_featured = df_featured[df_featured["trade_date"] > cached_last].copy()

        # 对齐列 (新版本可能多了列)
        all_cols = list(dict.fromkeys(list(cached.columns) + list(new_featured.columns)))
        for c in all_cols:
            if c not in cached.columns:
                cached[c] = np.nan
            if c not in new_featured.columns:
                new_featured[c] = np.nan

        result = pd.concat([cached, new_featured[all_cols]], ignore_index=True)
    else:
        # 全量计算
        result = compute_fn(df_raw)

        # 资金流 merge
        if moneyflow_merge_fn is not None:
            result = moneyflow_merge_fn(result, data_root, symbol)

        result["trade_date"] = result["trade_date"].astype(str)

    # 写入缓存
    _write_cache(fpath, result)
    return result


def get_cached_weekly_features(
    cache_dir: str,
    symbol: str,
    df_weekly_raw: pd.DataFrame | None,
    compute_fn,
    extra_fn=None,
) -> pd.DataFrame | None:
    """
    获取周线特征 (缓存优先 + 增量计算).
    """
    if df_weekly_raw is None or len(df_weekly_raw) < 12:
        return df_weekly_raw

    fpath = _cache_path(cache_dir, "weekly", symbol)
    df_raw = df_weekly_raw.copy()
    df_raw["trade_date"] = df_raw["trade_date"].astype(str)
    raw_last = _last_date(df_raw)

    cached = _read_cache(fpath)

    if cached is not None and _last_date(cached) >= raw_last:
        cached = cached[cached["trade_date"] <= raw_last].copy()
        return cached

    if cached is not None:
        cached_last = _last_date(cached)
        n_new = len(df_raw[df_raw["trade_date"] > cached_last])

        if n_new == 0:
            return cached

        if n_new > 20:
            cached = None

    if cached is not None:
        cached_last = _last_date(cached)
        lookback = 30  # 周线窗口较小
        overlap_start_idx = max(0, len(df_raw[df_raw["trade_date"] <= cached_last]) - lookback)
        df_for_compute = df_raw.iloc[overlap_start_idx:].reset_index(drop=True)

        df_featured = compute_fn(df_for_compute)
        if extra_fn:
            df_featured = extra_fn(df_featured)

        df_featured["trade_date"] = df_featured["trade_date"].astype(str)
        new_featured = df_featured[df_featured["trade_date"] > cached_last].copy()

        all_cols = list(dict.fromkeys(list(cached.columns) + list(new_featured.columns)))
        for c in all_cols:
            if c not in cached.columns:
                cached[c] = np.nan
            if c not in new_featured.columns:
                new_featured[c] = np.nan

        result = pd.concat([cached, new_featured[all_cols]], ignore_index=True)
    else:
        result = compute_fn(df_raw)
        if extra_fn:
            result = extra_fn(result)
        result["trade_date"] = result["trade_date"].astype(str)

    _write_cache(fpath, result)
    return result


def invalidate_cache(cache_dir: str, symbol: str | None = None) -> int:
    """
    清除缓存. symbol=None 时清除全部.
    返回删除的文件数.
    """
    count = 0
    for sub in ("daily", "weekly"):
        d = os.path.join(cache_dir, sub)
        if not os.path.isdir(d):
            continue
        if symbol:
            fpath = os.path.join(d, f"{symbol}.csv")
            if os.path.exists(fpath):
                os.remove(fpath)
                count += 1
        else:
            for f in os.listdir(d):
                if f.endswith(".csv"):
                    os.remove(os.path.join(d, f))
                    count += 1
    return count


def cache_stats(cache_dir: str) -> dict:
    """返回缓存统计信息."""
    stats = {}
    for sub in ("daily", "weekly"):
        d = os.path.join(cache_dir, sub)
        if os.path.isdir(d):
            files = [f for f in os.listdir(d) if f.endswith(".csv")]
            total_size = sum(os.path.getsize(os.path.join(d, f)) for f in files)
            stats[sub] = {"count": len(files), "size_mb": round(total_size / 1e6, 1)}
        else:
            stats[sub] = {"count": 0, "size_mb": 0}
    return stats
