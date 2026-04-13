"""
大盘趋势判断策略 - 微观指标计算模块

从个股数据聚合出 5 个维度的市场宽度指标:
  1. 广度 Breadth        — 涨跌比/均线比/涨跌停
  2. 资金流 Money Flow   — 大单分布
  3. 波动结构 Volatility — 波动率分布 + 恐慌度
  4. 熵/有序度 Entropy   — 排列熵分布
  5. 动量扩散 Momentum   — 行业动量离散 + 方向一致性
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import MarketTrendConfig


@dataclass
class MicroSnapshot:
    """单日微观指标快照"""
    date: int
    total_stocks: int

    # 广度
    advance_ratio: float        # 上涨占比
    above_ma20_ratio: float     # 站上 MA20 占比
    above_ma60_ratio: float     # 站上 MA60 占比
    new_high_ratio: float       # 创 N 日新高占比
    limit_up_count: int         # 涨停家数
    limit_down_count: int       # 跌停家数
    breadth_thrust: bool        # 广度推力

    # 资金流
    net_inflow_ratio: float     # 净流入为正的占比
    big_order_net_sum: float    # 全市场大单净额 (万元)

    # 波动
    vol_median: float           # 5 日实现波动率中位数
    panic_ratio: float          # 日跌幅 > 5% 的股票占比

    # 熵
    entropy_median: float       # 排列熵中位数
    ordering_ratio: float       # 有序股票占比

    # 动量
    sector_momentum_std: float  # 行业平均涨幅的标准差
    trend_alignment: float      # 涨跌方向与前 5 日一致的占比


def _permutation_entropy(x: np.ndarray, order: int = 3) -> float:
    """计算排列熵 (Bandt & Pompe 2002)。

    order=3 时有 3!=6 种排列模式。
    返回值范围 [0, ln(order!)]，越低越有序。
    """
    n = len(x)
    if n < order + 1:
        return np.nan
    # 构造嵌入
    patterns: Dict[tuple, int] = {}
    total = 0
    for i in range(n - order):
        window = x[i:i + order]
        # 排名模式
        pattern = tuple(np.argsort(window))
        patterns[pattern] = patterns.get(pattern, 0) + 1
        total += 1
    if total == 0:
        return np.nan
    # 计算香农熵
    entropy = 0.0
    for count in patterns.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log(p)
    return entropy


class MicroIndicatorEngine:
    """微观指标引擎

    预计算每只股票的滚动指标，然后按日期聚合成市场宽度指标。
    """

    def __init__(self, cfg: MarketTrendConfig):
        self.cfg = cfg

    def compute_stock_features(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """为单只股票计算所有需要的特征列。

        输入: 含 trade_date, open, high, low, close, pct_chg, vol, amount,
              turnover_rate, buy_elg_amount, buy_lg_amount,
              sell_elg_amount, sell_lg_amount, net_mf_amount 的 DataFrame。

        输出: 新增列:
          - ma20, ma60: 均线
          - above_ma20, above_ma60: 是否在均线上方
          - new_high: 是否创 N 日新高
          - realized_vol: 5 日实现波动率
          - big_order_net: 大单净额 (万元)
          - perm_entropy: 排列熵
          - ret_20d: 20 日涨幅
          - ret_5d_sign: 前 5 日涨跌方向
          - today_sign: 今日涨跌方向
        """
        c = df["close"].values.astype(np.float64)
        n = len(c)
        if n < self.cfg.min_bars:
            return df

        s = pd.Series(c)

        # 均线
        df = df.copy()
        df["ma20"] = s.rolling(window=self.cfg.ma_short, min_periods=self.cfg.ma_short).mean().values
        df["ma60"] = s.rolling(window=self.cfg.ma_long, min_periods=self.cfg.ma_long).mean().values
        df["above_ma20"] = (c > df["ma20"].values).astype(np.int8)
        df["above_ma60"] = (c > df["ma60"].values).astype(np.int8)

        # 创新高
        rolling_max = s.rolling(
            window=self.cfg.new_high_lookback,
            min_periods=self.cfg.new_high_lookback,
        ).max().values
        # 今日收盘 >= 前 N 日最高 (不含今日)
        prev_max = np.empty(n, dtype=np.float64)
        prev_max[:] = np.nan
        prev_max[self.cfg.new_high_lookback:] = rolling_max[self.cfg.new_high_lookback - 1:-1]
        df["new_high"] = (c >= prev_max).astype(np.int8)

        # 实现波动率 (5 日日收益率标准差)
        log_ret = np.log(c[1:] / c[:-1])
        log_ret = np.concatenate([[np.nan], log_ret])
        vol_s = pd.Series(log_ret)
        df["realized_vol"] = vol_s.rolling(
            window=self.cfg.vol_window, min_periods=self.cfg.vol_window
        ).std().values

        # 大单净额
        buy_big = df["buy_elg_amount"].fillna(0).values + df["buy_lg_amount"].fillna(0).values
        sell_big = df["sell_elg_amount"].fillna(0).values + df["sell_lg_amount"].fillna(0).values
        df["big_order_net"] = buy_big - sell_big

        # 排列熵 (滚动)
        perm_ent = np.full(n, np.nan)
        w = self.cfg.entropy_window
        order = self.cfg.entropy_order
        for i in range(w - 1, n):
            perm_ent[i] = _permutation_entropy(c[i - w + 1: i + 1], order=order)
        df["perm_entropy"] = perm_ent

        # 20 日收益率
        mw = self.cfg.momentum_window
        ret_20d = np.full(n, np.nan)
        if n > mw:
            ret_20d[mw:] = c[mw:] / c[:-mw] - 1.0
        df["ret_20d"] = ret_20d

        # 方向一致性
        pct = df["pct_chg"].values.astype(np.float64)
        today_sign = np.sign(pct)
        ret_5d = np.full(n, np.nan)
        if n > 5:
            ret_5d[5:] = c[5:] / c[:-5] - 1.0
        prev_5d_sign = np.sign(ret_5d)
        df["today_sign"] = today_sign
        df["ret_5d_sign"] = prev_5d_sign

        return df

    def aggregate_daily(
        self,
        date: int,
        stocks: Dict[str, pd.DataFrame],
        limit_data: Dict[str, pd.DataFrame],
        industry_map: Dict[str, str],
        names_map: Dict[str, str],
        ema_advance: Optional[float] = None,
    ) -> Optional[MicroSnapshot]:
        """聚合单日微观指标。

        stocks: 已经过 compute_stock_features 添加特征列的 DataFrame 字典。
        limit_data: 涨跌停数据字典。
        industry_map: symbol -> 行业名称。
        names_map: symbol -> 股票名称 (ST 过滤)。
        ema_advance: 由外部维护的 advance_ratio EMA (用于广度推力)。
        """
        advances = 0
        above_ma20 = 0
        above_ma60 = 0
        new_highs = 0
        net_inflow_positive = 0
        big_order_net_total = 0.0
        panic_count = 0
        vol_list: List[float] = []
        entropy_list: List[float] = []
        sector_rets: Dict[str, List[float]] = {}
        alignment_count = 0
        alignment_total = 0
        total = 0

        for symbol, df in stocks.items():
            # ST 过滤
            name = names_map.get(symbol, "")
            if name.upper().startswith("ST") or name.upper().startswith("*ST"):
                continue

            idx = df["trade_date"].searchsorted(date, side="right") - 1
            if idx < 0 or df["trade_date"].values[idx] != date:
                continue

            total += 1
            pct = df["pct_chg"].values[idx]

            # 广度
            if not np.isnan(pct) and pct > 0:
                advances += 1
            if not np.isnan(df["above_ma20"].values[idx]) and df["above_ma20"].values[idx] > 0:
                above_ma20 += 1
            if not np.isnan(df["above_ma60"].values[idx]) and df["above_ma60"].values[idx] > 0:
                above_ma60 += 1
            if not np.isnan(df["new_high"].values[idx]) and df["new_high"].values[idx] > 0:
                new_highs += 1

            # 资金流
            mf = df["net_mf_amount"].values[idx]
            if not np.isnan(mf) and mf > 0:
                net_inflow_positive += 1
            bon = df["big_order_net"].values[idx]
            if not np.isnan(bon):
                big_order_net_total += bon

            # 波动
            rv = df["realized_vol"].values[idx]
            if not np.isnan(rv):
                vol_list.append(rv)
            if not np.isnan(pct) and pct < self.cfg.panic_threshold * 100:
                # pct_chg 是百分比, panic_threshold 是小数
                panic_count += 1

            # 熵
            pe = df["perm_entropy"].values[idx]
            if not np.isnan(pe):
                entropy_list.append(pe)

            # 动量
            industry = industry_map.get(symbol, "UNKNOWN")
            r20 = df["ret_20d"].values[idx]
            if not np.isnan(r20):
                if industry not in sector_rets:
                    sector_rets[industry] = []
                sector_rets[industry].append(r20)

            ts = df["today_sign"].values[idx]
            r5s = df["ret_5d_sign"].values[idx]
            if not np.isnan(ts) and not np.isnan(r5s):
                alignment_total += 1
                if ts == r5s:
                    alignment_count += 1

        if total == 0:
            return None

        # --- 涨跌停 ---
        limit_up = 0
        limit_down = 0
        for symbol, ldf in limit_data.items():
            lidx = ldf["trade_date"].searchsorted(date, side="right") - 1
            if lidx < 0 or ldf["trade_date"].values[lidx] != date:
                continue
            # 需要对应股票的收盘价
            if symbol not in stocks:
                continue
            sdf = stocks[symbol]
            sidx = sdf["trade_date"].searchsorted(date, side="right") - 1
            if sidx < 0 or sdf["trade_date"].values[sidx] != date:
                continue
            close_val = sdf["close"].values[sidx]
            up_lim = ldf["up_limit"].values[lidx]
            down_lim = ldf["down_limit"].values[lidx]
            if not np.isnan(close_val) and not np.isnan(up_lim) and close_val >= up_lim:
                limit_up += 1
            if not np.isnan(close_val) and not np.isnan(down_lim) and close_val <= down_lim:
                limit_down += 1

        # --- 汇总 ---
        advance_ratio = advances / total
        above_ma20_ratio = above_ma20 / total
        above_ma60_ratio = above_ma60 / total
        new_high_ratio = new_highs / total
        net_inflow_ratio = net_inflow_positive / total
        panic_ratio = panic_count / total

        vol_median = float(np.median(vol_list)) if vol_list else 0.0
        entropy_median = float(np.median(entropy_list)) if entropy_list else 0.0
        ordering_ratio = (
            sum(1 for e in entropy_list if e < self.cfg.entropy_ordering_threshold) / len(entropy_list)
            if entropy_list else 0.0
        )

        # 行业动量标准差
        sector_means = []
        for rets in sector_rets.values():
            if len(rets) >= 3:
                sector_means.append(float(np.mean(rets)))
        sector_momentum_std = float(np.std(sector_means)) if len(sector_means) >= 2 else 0.0

        trend_alignment = alignment_count / alignment_total if alignment_total > 0 else 0.5

        # 广度推力
        breadth_thrust = False
        if ema_advance is not None:
            breadth_thrust = ema_advance > self.cfg.breadth_thrust_threshold

        return MicroSnapshot(
            date=date,
            total_stocks=total,
            advance_ratio=advance_ratio,
            above_ma20_ratio=above_ma20_ratio,
            above_ma60_ratio=above_ma60_ratio,
            new_high_ratio=new_high_ratio,
            limit_up_count=limit_up,
            limit_down_count=limit_down,
            breadth_thrust=breadth_thrust,
            net_inflow_ratio=net_inflow_ratio,
            big_order_net_sum=big_order_net_total,
            vol_median=vol_median,
            panic_ratio=panic_ratio,
            entropy_median=entropy_median,
            ordering_ratio=ordering_ratio,
            sector_momentum_std=sector_momentum_std,
            trend_alignment=trend_alignment,
        )
