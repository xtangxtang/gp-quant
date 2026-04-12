"""
双熵共振策略 - 配置模块
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class DailyEntropyConfig:
    """日线熵计算配置"""

    # 排列熵窗口
    perm_entropy_window: int = 20
    perm_entropy_order: int = 3

    # 路径不可逆性
    path_irrev_window: int = 20
    irrev_threshold_sigma: float = 0.5

    # 主导特征值
    dominant_eig_window: int = 20
    ar_order: int = 2

    # 多尺度熵
    perm_entropy_long_window: int = 60

    # 百分位排名
    percentile_window: int = 120

    # 方差抬升
    var_lift_short: int = 10
    var_lift_long: int = 20


@dataclass
class IntradayEntropyConfig:
    """日内熵计算配置"""

    # 滚动窗口（分钟）
    rolling_window: int = 60

    # 排列熵嵌入维度
    perm_order: int = 3

    # 路径不可逆性阈值
    irrev_threshold_sigma: float = 0.5

    # AR 阶数
    ar_order: int = 2

    # 日内熵分析的最近天数
    lookback_days: int = 10


@dataclass
class FusionSignalConfig:
    """双熵融合信号配置"""

    # ---- 日线压缩态门槛 ----
    # 排列熵百分位 < 此值 → 日线处于压缩
    daily_entropy_percentile_max: float = 0.40
    # 排列熵绝对值 < 此值 → 日线绝对低熵
    daily_perm_entropy_max: float = 0.85
    # 多尺度熵差 > 此值 → 短期比长期更有序
    daily_entropy_gap_min: float = 0.01

    # ---- 日内集中化门槛 ----
    # 日内成交量熵 < 此值 → 成交集中（绝对阈值）
    intraday_turnover_entropy_max: float = 0.88
    # 日内成交量熵降幅 > 此值 → 成交突然集中化（相对近N天均值）
    intraday_turnover_entropy_drop_min: float = 0.02
    # 日内路径不可逆性 > 此值 → 有方向性
    intraday_path_irrev_min: float = 0.01

    # ---- 辅助确认 ----
    # 最低有效分钟占比（过滤极低流动性股）
    min_active_bar_ratio: float = 0.70
    # 当日振幅上限（悄然吸筹 = 价格波动小）
    max_price_range_ratio: float = 0.06
    # 成交量集中度 > 此值 → 量能集中
    min_volume_concentration: float = 0.008

    # ---- 得分权重 ----
    weights: Dict[str, float] = field(default_factory=lambda: {
        'daily_compression': 0.30,       # 日线压缩态质量
        'intraday_concentration': 0.30,  # 日内集中化程度
        'direction_signal': 0.20,        # 方向性信号
        'volume_pattern': 0.20,          # 量能形态
    })

    # ---- 最终信号门槛 ----
    buy_score_min: float = 0.55
    watch_score_min: float = 0.40


@dataclass
class SellSignalConfig:
    """卖出信号配置"""

    # ---- 权重 ----
    weights: Dict[str, float] = field(default_factory=lambda: {
        'entropy_diffusion': 0.30,       # 熵扩散（结构瓦解）
        'stealth_distribution': 0.35,    # 暗中派发
        'exhaustion': 0.20,              # 熵衰竭
        'volume_anomaly': 0.15,          # 量能异常
    })

    # ---- 熵扩散门槛 ----
    # 日线 PE20 近 5 日上升速度 > 此值 → 熵膨胀
    daily_entropy_velocity_min: float = 0.02
    # 日线百分位快速穿越此值 → 脱离压缩
    daily_percentile_exit_threshold: float = 0.70

    # ---- 暗中派发门槛 ----
    # 日内成交量熵上升幅度（相对近期均值）→ 成交分散化
    intraday_te_rise_min: float = 0.02
    # 上午/下午熵偏移 > 此值 → 尾盘混乱（下午比上午更无序）
    entropy_shift_reversal_min: float = 0.02
    # 日内路径不可逆性下降幅度 → 方向性丧失
    intraday_pi_drop_min: float = 0.005

    # ---- 衰竭门槛 ----
    # 日线排列熵 > 此值 → 回到无序
    daily_perm_entropy_exhaustion: float = 0.92
    # 日内排列熵 > 此值 → 日内彻底无序
    intraday_perm_entropy_exhaustion: float = 0.96

    # ---- 信号门槛 ----
    sell_score_min: float = 0.50
    warning_score_min: float = 0.35

    # ---- 辅助 ----
    min_active_bar_ratio: float = 0.60


@dataclass
class ScannerConfig:
    """扫描器配置"""

    # 数据路径
    daily_data_dir: str = '../../gp-data/tushare-daily-full'
    minute_data_dir: str = '../../gp-data/trade'
    basic_path: str = '../../gp-data/tushare_stock_basic.csv'

    # 输出
    output_dir: str = '../../results/dual_entropy'

    # 扫描参数
    scan_date: str = ''  # 空 = 最新
    max_stocks: int = 200
    min_data_days: int = 60  # 日线最少天数

    # 日线预筛
    min_close: float = 3.0   # 最低股价
    max_close: float = 100.0  # 最高股价

    # 并行
    workers: int = 8


@dataclass
class Config:
    """双熵共振策略总配置"""

    daily: DailyEntropyConfig = field(default_factory=DailyEntropyConfig)
    intraday: IntradayEntropyConfig = field(default_factory=IntradayEntropyConfig)
    fusion: FusionSignalConfig = field(default_factory=FusionSignalConfig)
    sell: SellSignalConfig = field(default_factory=SellSignalConfig)
    scanner: ScannerConfig = field(default_factory=ScannerConfig)
