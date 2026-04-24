"""
Agent 1: Factor Calculator

职责: 从 OHLCV + 资金流数据计算全市场 32 个因子
      复用 entropy_accumulation_breakout 的 feature_engine.py

输入: tushare-daily-full/*.csv + tushare-moneyflow/*.csv
输出: {symbol: DataFrame_with_features} 截面快照

频率: 每日
"""

from __future__ import annotations

import glob
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _compute_one_symbol(
    daily_dir: str,
    data_root: str,
    symbol: str,
    scan_date: str = "",
    cache_dir: str = "",
    min_rows: int = 80,
) -> tuple[str, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    计算单只股票的日线 + 周线特征。

    Returns:
        (symbol, df_daily_featured, df_weekly_featured)
        如果数据不足或出错, 返回 (symbol, None, None)
    """
    try:
        from src.strategy.entropy_accumulation_breakout.feature_engine import (
            build_features,
        )

        fpath = os.path.join(daily_dir, f"{symbol}.csv")
        if not os.path.exists(fpath):
            return symbol, None, None

        df_daily = pd.read_csv(fpath)
        if "trade_date" not in df_daily.columns or "close" not in df_daily.columns:
            return symbol, None, None

        df_daily["trade_date"] = df_daily["trade_date"].astype(str)
        if scan_date:
            df_daily = df_daily[df_daily["trade_date"] <= scan_date]

        if len(df_daily) < min_rows:
            return symbol, None, None

        result = build_features(
            df_daily=df_daily,
            data_root=data_root,
            symbol=symbol,
            cache_dir=cache_dir,
        )

        df_d = result.get("daily")
        df_w = result.get("weekly")

        if df_d is None or len(df_d) < min_rows:
            return symbol, None, None

        return symbol, df_d, df_w

    except Exception as e:
        logger.debug(f"Error computing {symbol}: {e}")
        return symbol, None, None


class FactorCalculator:
    """Agent 1: 全市场因子计算器"""

    def __init__(
        self,
        daily_dir: str,
        data_root: str = "",
        cache_dir: str = "",
        max_workers: int = 4,
    ):
        """
        Args:
            daily_dir: tushare-daily-full 目录
            data_root: 数据根目录 (含 moneyflow, weekly 等)
            cache_dir: 特征缓存目录 (增量计算加速)
            max_workers: 并行进程数
        """
        self.daily_dir = daily_dir
        self.data_root = data_root
        self.cache_dir = cache_dir
        self.max_workers = max_workers

    def compute_all(
        self,
        scan_date: str = "",
        symbols: Optional[list[str]] = None,
    ) -> dict[str, tuple[pd.DataFrame, Optional[pd.DataFrame]]]:
        """
        计算全市场因子。

        Args:
            scan_date: 扫描日期 (YYYYMMDD), 为空则用最新数据
            symbols: 指定股票列表, 为空则扫描全部

        Returns:
            {symbol: (df_daily_featured, df_weekly_featured)}
        """
        if symbols is None:
            csv_files = glob.glob(os.path.join(self.daily_dir, "*.csv"))
            symbols = [os.path.basename(f).replace(".csv", "") for f in csv_files]

        logger.info(f"Agent 1: Computing factors for {len(symbols)} stocks ...")

        results = {}
        with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(
                    _compute_one_symbol,
                    self.daily_dir,
                    self.data_root,
                    sym,
                    scan_date,
                    self.cache_dir,
                ): sym
                for sym in symbols
            }

            for future in as_completed(futures):
                sym, df_d, df_w = future.result()
                if df_d is not None:
                    results[sym] = (df_d, df_w)

        logger.info(f"Agent 1: Done. {len(results)}/{len(symbols)} stocks computed.")
        return results

    def build_cross_section(
        self,
        results: dict[str, tuple[pd.DataFrame, Optional[pd.DataFrame]]],
    ) -> pd.DataFrame:
        """
        构建截面快照: 每只股票一行, 列为最新因子值。

        Returns:
            DataFrame, index=symbol, columns=因子名 + "_trade_date"
        """
        rows = []
        for sym, (df_d, df_w) in results.items():
            if df_d is None or len(df_d) == 0:
                continue
            last = df_d.iloc[-1]
            row = {"symbol": sym, "_trade_date": str(last.get("trade_date", ""))}
            for col in df_d.columns:
                if col in ("trade_date", "open", "high", "low", "close",
                           "vol", "amount", "turnover_rate", "net_mf_amount"):
                    continue
                v = last[col]
                if pd.notna(v):
                    try:
                        row[col] = float(v)
                    except (ValueError, TypeError):
                        row[col] = None
                else:
                    row[col] = None
            rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("symbol")
        logger.info(f"Agent 1: Cross-section built: {len(df)} stocks × {len(df.columns)} cols")
        return df
