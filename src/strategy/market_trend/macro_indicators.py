"""
大盘趋势判断策略 - 宏观指标计算模块

两个维度:
  6. 杠杆资金 Leverage   — 两融余额变化
  7. 流动性 Liquidity    — SHIBOR 利率
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .config import MarketTrendConfig


@dataclass
class MacroSnapshot:
    """单日宏观指标快照"""
    date: int

    # 杠杆
    margin_balance: float        # 两融余额 (亿元)
    margin_net_buy: float        # 融资净买入 = rzmre - rzche (亿元)
    margin_balance_ma5: float    # 余额 5 日均
    margin_balance_ma20: float   # 余额 20 日均
    margin_balance_chg_pct: float  # 余额日变化率

    # 流动性
    shibor_on: float             # 隔夜 SHIBOR
    shibor_on_ma5: float         # 隔夜 SHIBOR 5 日均
    shibor_on_ma20: float        # 隔夜 SHIBOR 20 日均
    shibor_on_change: float      # 隔夜 SHIBOR 日变化 (bp)


class MacroIndicatorEngine:
    """宏观指标引擎

    预处理两融和 SHIBOR 数据，按交易日查询。
    """

    def __init__(self, cfg: MarketTrendConfig,
                 margin_df: pd.DataFrame, shibor_df: pd.DataFrame):
        self.cfg = cfg
        self._margin = self._prepare_margin(margin_df)
        self._shibor = self._prepare_shibor(shibor_df)

    def _prepare_margin(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        # 转成亿元
        for col in ["rzye", "rzmre", "rzche", "rqye", "rzrqye"]:
            if col in df.columns:
                df[col] = df[col] / 1e8
        # 融资净买入
        df["margin_net_buy"] = df["rzmre"] - df["rzche"]
        # 余额变化率
        df["margin_chg_pct"] = df["rzrqye"].pct_change()
        # 滚动均线
        df["margin_ma5"] = df["rzrqye"].rolling(
            window=self.cfg.margin_ma_short, min_periods=1).mean()
        df["margin_ma20"] = df["rzrqye"].rolling(
            window=self.cfg.margin_ma_long, min_periods=1).mean()
        df = df.sort_values("trade_date").reset_index(drop=True)
        return df

    def _prepare_shibor(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        # 隔夜利率日变化
        df["on_change"] = df["on"].diff()
        # 滚动均线
        df["on_ma5"] = df["on"].rolling(
            window=self.cfg.shibor_ma_short, min_periods=1).mean()
        df["on_ma20"] = df["on"].rolling(
            window=self.cfg.shibor_ma_long, min_periods=1).mean()
        df = df.sort_values("trade_date").reset_index(drop=True)
        return df

    def _lookup_row(self, df: pd.DataFrame, date: int) -> Optional[pd.Series]:
        """查找 <= date 的最近一行 (两融或 SHIBOR 可能比交易日少)。"""
        if df.empty:
            return None
        td = df["trade_date"].values
        idx = np.searchsorted(td, date, side="right") - 1
        if idx < 0:
            return None
        return df.iloc[idx]

    def snapshot(self, date: int) -> MacroSnapshot:
        """获取单日宏观指标快照。"""
        # 两融
        m_row = self._lookup_row(self._margin, date)
        if m_row is not None:
            margin_balance = float(m_row.get("rzrqye", 0) or 0)
            margin_net_buy = float(m_row.get("margin_net_buy", 0) or 0)
            margin_ma5 = float(m_row.get("margin_ma5", 0) or 0)
            margin_ma20 = float(m_row.get("margin_ma20", 0) or 0)
            margin_chg = float(m_row.get("margin_chg_pct", 0) or 0)
        else:
            margin_balance = 0.0
            margin_net_buy = 0.0
            margin_ma5 = 0.0
            margin_ma20 = 0.0
            margin_chg = 0.0

        # SHIBOR
        s_row = self._lookup_row(self._shibor, date)
        if s_row is not None:
            shibor_on = float(s_row.get("on", 0) or 0)
            shibor_ma5 = float(s_row.get("on_ma5", 0) or 0)
            shibor_ma20 = float(s_row.get("on_ma20", 0) or 0)
            shibor_change = float(s_row.get("on_change", 0) or 0)
        else:
            shibor_on = 0.0
            shibor_ma5 = 0.0
            shibor_ma20 = 0.0
            shibor_change = 0.0

        return MacroSnapshot(
            date=date,
            margin_balance=margin_balance,
            margin_net_buy=margin_net_buy,
            margin_balance_ma5=margin_ma5,
            margin_balance_ma20=margin_ma20,
            margin_balance_chg_pct=margin_chg,
            shibor_on=shibor_on,
            shibor_on_ma5=shibor_ma5,
            shibor_on_ma20=shibor_ma20,
            shibor_on_change=shibor_change,
        )
