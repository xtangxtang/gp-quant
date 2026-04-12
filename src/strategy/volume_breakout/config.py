"""
量价突破策略 - 配置模块
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class BreakoutDetectorConfig:
    """突破检测配置"""

    # ---- 波动率压缩门槛 ----
    # 20日实现波动率 < 此值 → 底部横盘
    max_realized_vol_20: float = 0.035
    # 20日价格区间 (max-min)/mean < 此值 → 窄幅整理
    max_price_range_20: float = 0.25
    # 布林带宽度 < 此值 → 极度压缩
    max_bb_width: float = 0.14

    # ---- 放量突破门槛 ----
    # 5日均量 / 20日均量 > 此值 → 量能放大
    min_vol_surge: float = 1.2
    # 当日成交量 / 20日均量 > 此值 → 单日放量
    min_vol_spike: float = 1.5
    # 突破 N 日新高
    breakout_lookback: int = 60

    # ---- 流动性门槛 ----
    # 日均成交额 > 此值（元）
    min_liquidity: float = 3_000_000
    # 股价范围
    min_close: float = 5.0
    max_close: float = 200.0

    # ---- 趋势确认 ----
    # 5日收益率 > 此值 → 上行动能
    min_ret_5d: float = 0.02
    # 20日收益率 > 此值 → 中期转强
    min_ret_20d: float = -0.05
    # 20日收益率 < 此值 → 防止追高 (过度延伸过滤)
    max_ret_20d: float = 0.10

    # ---- 评分权重 ----
    weights: Dict[str, float] = field(default_factory=lambda: {
        'compression': 0.35,    # 压缩质量 (最重要)
        'volume': 0.30,         # 放量新鲜度
        'breakout': 0.20,       # 位置质量 (接近前高)
        'momentum': 0.15,       # 动量确认 (最低权重)
    })

    # ---- 信号门槛 ----
    buy_score_min: float = 0.50
    watch_score_min: float = 0.35


@dataclass
class BreakoutExitConfig:
    """突破策略退出配置"""

    # 硬止损
    stop_loss: float = -0.08

    # 追踪止盈: 峰值回撤 > 此值 → 止盈
    trailing_stop_pct: float = 0.18
    # 追踪止盈最低盈利要求
    trailing_min_profit: float = 0.10

    # 量能衰竭: 5日均量 / 入场时5日均量 < 此值
    vol_exhaustion_ratio: float = 0.3
    # 量能衰竭需要盈利 > 此值才触发
    vol_exhaustion_min_profit: float = 0.15

    # 最大持仓天数
    max_hold_days: int = 90


@dataclass
class BreakoutBacktestConfig:
    """回测配置"""

    start_date: str = '2025-04-01'
    end_date: str = '2025-09-30'

    initial_capital: float = 1_000_000
    max_positions: int = 30
    max_buys_per_day: int = 5

    commission_rate: float = 0.0015
    slippage_rate: float = 0.001
    stamp_tax_rate: float = 0.0005

    scan_interval: int = 1
    workers: int = 8
