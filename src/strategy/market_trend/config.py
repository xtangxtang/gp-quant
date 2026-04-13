"""
大盘趋势判断策略 - 配置模块

从小见大：通过全市场个股微观交易行为 + 宏观资金/流动性数据，
聚合判断大盘趋势状态。

7 维度:
  微观 (个股聚合):
    1. 广度 Breadth        — 涨跌/均线/涨跌停
    2. 资金流 Money Flow   — 大单净流入分布
    3. 波动结构 Volatility — 波动率分布 + 涨跌停
    4. 熵/有序度 Entropy   — 排列熵分布
    5. 动量扩散 Momentum   — 行业扩散 + 个股动量一致性
  宏观 (外部数据):
    6. 杠杆资金 Leverage   — 两融余额变化
    7. 流动性 Liquidity    — SHIBOR 利率
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class MarketTrendConfig:
    """大盘趋势判断主配置"""

    # ---- 数据路径 ----
    data_dir: str = ""                   # tushare-daily-full/
    index_dir: str = ""                  # tushare-index-daily/
    stk_limit_dir: str = ""              # tushare-stk_limit/
    margin_path: str = ""                # tushare-margin/margin.csv
    shibor_path: str = ""                # tushare-shibor/shibor.csv
    index_member_path: str = ""          # tushare-index_member_all/index_member_all.csv
    basic_path: str = ""                 # tushare_stock_basic.csv
    out_dir: str = ""

    # ---- 扫描范围 ----
    start_date: str = ""
    end_date: str = ""
    # 指数对照 (上证综指)
    index_code: str = "000001_sh"

    # ---- 微观指标参数 ----
    # 广度
    ma_short: int = 20
    ma_long: int = 60
    new_high_lookback: int = 20
    breadth_thrust_window: int = 10
    breadth_thrust_threshold: float = 0.615

    # 波动
    vol_window: int = 5
    panic_threshold: float = -0.05       # 跌幅 < 此值 算恐慌

    # 熵
    entropy_order: int = 3               # 排列熵阶数
    entropy_window: int = 20             # 排列熵滚动窗口
    entropy_ordering_threshold: float = 1.0  # 低于此值视为有序

    # 动量
    momentum_window: int = 20

    # ---- 宏观指标参数 ----
    # 两融
    margin_ma_short: int = 5
    margin_ma_long: int = 20

    # SHIBOR
    shibor_ma_short: int = 5
    shibor_ma_long: int = 20
    shibor_spike_threshold: float = 0.5  # 隔夜利率日变化 > 0.5% 视为飙升

    # ---- 评分权重 ----
    weights: Dict[str, float] = field(default_factory=lambda: {
        'breadth': 0.25,
        'money_flow': 0.20,
        'volatility': 0.10,
        'entropy': 0.15,
        'momentum': 0.10,
        'leverage': 0.10,
        'liquidity': 0.10,
    })

    # ---- 趋势判定门槛 ----
    strong_up_threshold: float = 0.5
    up_threshold: float = 0.2
    down_threshold: float = -0.2
    strong_down_threshold: float = -0.5

    # ---- 性能 ----
    workers: int = 8
    min_bars: int = 60                   # 个股最少数据天数


@dataclass(frozen=True)
class TrendState:
    """单日趋势状态"""
    date: str
    trend: str                    # STRONG_UP / UP / NEUTRAL / DOWN / STRONG_DOWN

    composite_score: float        # 综合得分 [-1, 1]

    # 7 维子分 (各自 [-1, 1])
    breadth_score: float
    money_flow_score: float
    volatility_score: float
    entropy_score: float
    momentum_score: float
    leverage_score: float
    liquidity_score: float

    # 广度原始指标
    advance_ratio: float          # 上涨股占比
    above_ma20_ratio: float       # 站上 MA20 占比
    above_ma60_ratio: float       # 站上 MA60 占比
    new_high_ratio: float         # 创新高占比
    limit_up_count: int           # 涨停家数
    limit_down_count: int         # 跌停家数
    breadth_thrust: bool          # 广度推力触发

    # 资金流原始指标
    net_inflow_ratio: float       # 净流入为正的股票占比
    big_order_net_sum: float      # 全市场大单净额(亿)

    # 波动原始指标
    vol_median: float             # 波动率中位数
    panic_ratio: float            # 跌幅>5%占比

    # 熵原始指标
    entropy_median: float         # 排列熵中位数
    ordering_ratio: float         # 低熵(有序)占比

    # 动量原始指标
    sector_momentum_std: float    # 行业动量标准差
    trend_alignment: float        # 动量方向一致性

    # 宏观原始指标
    margin_balance: float         # 两融余额(亿)
    margin_net_buy: float         # 融资净买入(亿)
    shibor_on: float              # 隔夜 SHIBOR
    shibor_on_change: float       # 隔夜 SHIBOR 日变化

    # 指数对照
    index_close: float
    index_pct_chg: float

    # 统计
    total_stocks: int
