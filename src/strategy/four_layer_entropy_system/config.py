"""
四层熵交易系统配置参数

基于 12 篇论文的阈值和权重配置。
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class MarketGateConfig:
    """市场门控层配置"""

    # 耦合熵阈值（communication_induced_bifurcation 论文）
    coupling_entropy_low: float = 0.4       # 低耦合（开仓窗口）
    coupling_entropy_high: float = 0.6      # 中耦合（谨慎）
    coupling_entropy_critical: float = 0.8  # 高耦合（观望）

    # 噪声成本阈值
    noise_cost_low: float = 0.3             # 低噪声
    noise_cost_high: float = 0.6            # 高噪声
    noise_cost_critical: float = 0.8        # 临界噪声（触发战略放弃）

    # 市场相位状态权重
    market_phase_weights: Dict[str, float] = field(default_factory=lambda: {
        'compression': 1.0,      # 压缩态（最佳开仓）
        'transition': 0.7,       # 转换态（谨慎开仓）
        'expansion': 0.3,        # 扩张态（减仓）
        'distorted': 0.0,        # 失真态（观望）
        'neutral': 0.5,          # 中性态（正常）
        'abandon': 0.0,          # 放弃态（清仓）
    })

    # 行业网络熵计算参数
    industry_window: int = 20   # 20 日 lead-lag 窗口
    industry_min_stocks: int = 5  # 最小行业股票数

    # A 股日历效应周期去偏（论文 12: Statistical warning indicators）
    calendar_debiasing: bool = True
    # 季报窗口：1/4/7/10 月下旬
    earnings_months: List[int] = field(default_factory=lambda: [1, 4, 7, 10])
    earnings_day_start: int = 15
    # 重大节假日前后窗口天数
    holiday_buffer_days: int = 5


@dataclass
class StockStateConfig:
    """个股状态层配置"""

    # 路径不可逆性阈值（Seifert 2025）
    path_irrev_low: float = 0.15     # 低熵（接近可逆）- 原 0.05
    path_irrev_high: float = 0.35    # 高熵（强非平衡）- 原 0.30

    # 主导特征值阈值（period_doubling 论文）
    dominant_eig_threshold: float = 0.85    # 临界减速预警 - 原 0.9
    dominant_eig_critical: float = 0.95     # 强临界

    # 排列熵阈值
    perm_entropy_low: float = 0.80   # 有序 - 原 0.50
    perm_entropy_high: float = 0.95  # 混沌 - 原 0.90

    # 状态流转参数
    compression_days: int = 5        # 压缩态持续天数
    trigger_confirmation: int = 2    # 触发确认天数

    # 特征权重（bifurcation_quality）
    feature_weights: Dict[str, float] = field(default_factory=lambda: {
        'dominant_eig_abs': 0.25,    # 主导特征值绝对值
        'path_irreversibility': 0.20, # 路径不可逆性
        'phase_adjusted_ar1': 0.15,  # 相位校正 AR(1)
        'entropy_accel': 0.12,       # 熵加速度（二阶变化率）
        'var_lift': 0.12,            # 方差抬升
        'recovery_rate': 0.10,       # 恢复率
        'ar1_raw': 0.06,             # 原始 AR(1)
    })

    # 熵质量权重（entropy_quality）
    entropy_quality_weights: Dict[str, float] = field(default_factory=lambda: {
        'entropy_percentile': 0.40,  # 熵百分位排名
        'entropy_gap': 0.30,         # 多尺度熵差
        'perm_entropy_compression': 0.15,  # 短期熵压缩
        'path_irrev_compression': 0.15,    # 路径不可逆性压缩
    })

    # 硬性门槛（对齐基线五门条件）
    hard_gate_entropy_quality_min: float = 0.62   # 熵质量最低门槛
    hard_gate_bifurcation_quality_min: float = 0.20  # 分叉质量最低门槛
    hard_gate_entropy_percentile_max: float = 0.40   # 熵百分位上限
    hard_gate_entropy_gap_min: float = 0.015         # 熵差下限

    # 特征计算窗口
    window_short: int = 10
    window_medium: int = 20
    window_long: int = 60


@dataclass
class ExecutionCostConfig:
    """执行成本层配置"""

    # 保守驱动配置（conservative_driving 论文）
    # 核心发现：保守驱动耗散≤2 倍最优
    staged_entry_days: List[int] = field(default_factory=lambda: [1, 3, 5])
    staged_entry_ratios: List[float] = field(default_factory=lambda: [0.25, 0.50, 1.0])

    # 信号强度分档
    signal_weak: float = 0.5    # 弱信号阈值（低于此值直接 skip，不再 probe）
    signal_medium: float = 0.6  # 中等信号
    signal_strong: float = 0.8  # 强信号

    # 仓位配置
    initial_capital: float = 1_000_000.0  # 初始资金
    base_position: float = 0.1    # 基础仓位 10%
    max_position: float = 0.25    # 单票最大 25%
    max_total_position: float = 0.8  # 总仓位最大 80%

    # 止盈止损
    take_profit: float = 0.08     # 8% 止盈（全部清仓）
    take_profit_partial: float = 0.04  # 4% 分段止盈（减半仓）
    stop_loss: float = 0.05       # 5% 止损
    trailing_stop: float = 0.03   # 3% 移动止损

    # 战略放弃阈值（论文 9: 高噪声环境少做 > 多做）
    abandonment_cost_threshold: float = 0.5  # 成本过高触发放弃
    abandonment_noise_threshold: float = 0.6  # 噪声过高触发放弃（原 0.7，降低以更早放弃）

    # 交易费用
    commission: float = 0.0003    # 万三
    stamp_tax: float = 0.001      # 千一
    slippage_base: float = 0.001  # 千一基础滑点


@dataclass
class ExperimentalConfig:
    """实验模型层配置"""

    # 权重明确为 0%，仅辅助观察
    weight_tda: float = 0.0
    weight_reservoir: float = 0.0
    weight_structure: float = 0.0

    # TDA 参数（hopf_bifurcation_persistent_homology 论文）
    tda_embedding_dim: int = 3
    tda_delay: int = 5
    tda_persistence_threshold: float = 0.3

    # Reservoir 参数（tipping_points_reservoir_computing 论文）
    reservoir_size: int = 100
    reservoir_spectral_radius: float = 1.2
    reservoir_sparsity: float = 0.1

    # 结构信息参数（pinn_vs_neural_ode 论文）
    latent_dim: int = 4  # compression/instability/launch/diffusion


@dataclass
class LayerConfig:
    """各层权重配置"""

    # 最终决策权重
    stock_state_weight: float = 0.70      # 个股状态 70%
    market_gate_weight: float = 0.20      # 市场门控 20%
    execution_weight: float = 0.10        # 执行成本 10%
    experimental_weight: float = 0.00     # 实验层 0%

    # 市场门控内部权重
    market_coupling_weight: float = 0.30
    market_noise_weight: float = 0.30
    market_phase_weight: float = 0.40

    # 个股状态内部权重
    entropy_quality_weight: float = 0.32
    bifurcation_quality_weight: float = 0.48
    trigger_quality_weight: float = 0.20


@dataclass
class Config:
    """总配置"""

    market_gate: MarketGateConfig = field(default_factory=MarketGateConfig)
    stock_state: StockStateConfig = field(default_factory=StockStateConfig)
    execution: ExecutionCostConfig = field(default_factory=ExecutionCostConfig)
    experimental: ExperimentalConfig = field(default_factory=ExperimentalConfig)
    layer: LayerConfig = field(default_factory=LayerConfig)

    # 数据配置
    data_dir: str = "/nvme5/xtang/gp-workspace/gp-data/tushare-daily-full"
    output_dir: str = "/nvme5/xtang/gp-workspace/gp-quant/results/four_layer_system"

    # 股票池配置
    max_stocks: int = 100
    min_market_cap: float = 50  # 最小市值 50 亿
    min_liquidity: float = 10   # 最小日均成交 10 亿

    # 时间配置
    lookback_days: int = 120    # 回看天数
    rebalance_freq: str = "daily"  # 调仓频率

    # 并发配置
    num_workers: int = -1  # -1 表示使用 CPU 核心数
    use_parallel: bool = True  # 是否启用并行计算
