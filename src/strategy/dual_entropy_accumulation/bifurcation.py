"""
分岔预警模块 v2 (Bifurcation Onset Detection)

基于论文思路 + 经验数据挖掘, 双层检测大牛股的相变起始:

第一层: 硬性预筛 (prescreen gates)
  1. 主导特征值 |λ| 处于全市场 top-10% → 系统接近临界 (period-doubling)
  2. 价格加速度为负 (bottom-15%) → 首轮拉升后回踩, "蓄势"形态
  3. 流动性代理 (close × vol20) 处于全市场 top-20% → 大资金参与

  经验验证: 三条件交集在 2025-05-19 全 A 筛出 37 只 (0.76%),
  包含中际旭创(#7)、天孚通信(#4)、新易盛(#29) — 3/3 命中。

第二层: 熵综合打分 (entropy scoring)
  4. 多尺度熵剪刀差 (PE60 - PE20)
  5. 路径不可逆性 (时间方向性突现)
  6. 熵百分位 (是否处于历史低位)

用途:
  - 作为常规"压缩态扫描"的补充, 捕捉爆发型大牛股
  - 不依赖长期低熵压缩, 而是检测临界态 + 大资金 + 回踩蓄势
  - 买入后使用 "趋势持有" 模式, 抑制过早卖出

理论基础:
  - period-doubling bifurcation: |λ| → ±1 时系统即将翻倍分叉
  - 价格加速度为负 = "保守驱动"阶段 (near-optimality of conservative driving)
  - 大市值=高耦合度系统, 相变信号更可靠 (entropy in time-evolving networks)
  - statistical warning indicators: 方差/AR(1) 不够, 需要特征值+结构指标

参考论文:
  - Predicting the onset of period-doubling bifurcations via dominant eigenvalue
  - Statistical warning indicators for abrupt transitions in dynamical systems
  - Near-optimality of conservative driving in discrete systems
  - Entropy Production Rate in Stochastically Time-evolving Asymmetric Networks
  - Beyond the Largest Lyapunov Exponent: Entropy-Based Diagnostics of Chaos
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Literal, Dict


@dataclass
class BifurcationConfig:
    """分岔预警配置 v2"""

    # ---- 第一层: 硬性门槛 (prescreen gates) ----
    # 主导特征值: |λ| 必须 > 此绝对阈值
    eigenvalue_gate: float = 0.55           # 放宽: 0.65→0.55, 捕捉更多临界态

    # 价格加速度: 5日收益 - 前5日收益 < 此阈值 (负数=回踩)
    price_accel_gate: float = -0.05         # 放宽: -0.10→-0.05, 允许温和回踩

    # 流动性代理: close × avg_vol_20 > 此阈值
    liquidity_gate: float = 3_000_000.0     # ~全市场 top-20%

    # 20日动量下限: 排除深度下跌的票 (避免"死猫反弹")
    momentum_floor: float = -0.08           # 放宽: -0.05→-0.08, 容忍轻度回调

    # ---- 第二层: 熵综合打分 ----
    # 主导特征值权重更高 (经验: 最强区分力)
    eigenvalue_critical: float = 0.60       # 放宽: 0.70→0.60

    # 多尺度熵差
    entropy_gap_min: float = 0.04           # 放宽: 0.06→0.04

    # 路径不可逆性
    path_irrev_spike_min: float = 0.02

    # ---- 综合阈值 ----
    bifurcation_score_min: float = 0.45     # 放宽: 0.50→0.45 (预筛已缩窄候选池)
    bifurcation_watch_min: float = 0.30

    # ---- 权重 (v2: 更重特征值, 减弱方差项) ----
    weights: Dict[str, float] = field(default_factory=lambda: {
        'eigenvalue': 0.40,         # 主导特征值 (最强区分因子)
        'entropy_divergence': 0.25, # 多尺度熵剪刀差
        'directionality': 0.20,     # 方向性突现
        'momentum_quality': 0.15,   # 动量质量 (替代 variance_state)
    })


BifurcationSignal = Literal['buy', 'watch', 'skip']


@dataclass
class BifurcationResult:
    """分岔预警结果"""

    stock_code: str
    trade_date: str

    # 子得分
    eigenvalue_score: float
    entropy_divergence_score: float
    directionality_score: float
    variance_state_score: float

    # 综合
    total_score: float
    signal: BifurcationSignal
    reason: str

    # 关键特征值
    dominant_eigenvalue: float
    eigenvalue_velocity: float
    perm_entropy_20: float
    perm_entropy_60: float
    entropy_gap: float
    entropy_percentile: float
    path_irreversibility: float
    var_lift: float

    # 分岔类型标记
    is_bifurcation: bool = False

    def to_dict(self) -> Dict:
        return {
            'stock_code': self.stock_code,
            'trade_date': self.trade_date,
            'eigenvalue_score': round(self.eigenvalue_score, 4),
            'entropy_divergence_score': round(self.entropy_divergence_score, 4),
            'directionality_score': round(self.directionality_score, 4),
            'variance_state_score': round(self.variance_state_score, 4),
            'total_score': round(self.total_score, 4),
            'signal': self.signal,
            'reason': self.reason,
            'dominant_eigenvalue': round(self.dominant_eigenvalue, 4),
            'eigenvalue_velocity': round(self.eigenvalue_velocity, 4),
            'perm_entropy_20': round(self.perm_entropy_20, 4),
            'entropy_gap': round(self.entropy_gap, 4),
            'entropy_percentile': round(self.entropy_percentile, 4),
            'path_irreversibility': round(self.path_irreversibility, 4),
            'var_lift': round(self.var_lift, 4),
            'is_bifurcation': self.is_bifurcation,
        }


class BifurcationDetector:
    """
    分岔预警探测器 v2。

    双层设计:
      第一层 (prescreen): 硬性门槛过滤, 在 backtest worker 中执行
        - |λ| > eigenvalue_gate
        - price_accel < price_accel_gate
        - liquidity > liquidity_gate
        - mom20 > momentum_floor
      第二层 (scoring): 熵综合打分, 在 evaluate() 中执行

    调用者需要先通过 passes_prescreen() 检查硬性门槛,
    满足后再调用 evaluate() 计算综合得分。
    """

    def __init__(self, config: Optional[BifurcationConfig] = None):
        self.config = config or BifurcationConfig()

    def passes_prescreen(
        self,
        dominant_eigenvalue: float,
        price_accel: float,
        liquidity: float,
        mom20: float,
    ) -> bool:
        """
        第一层硬性门槛检查。

        参数
        ----
        dominant_eigenvalue : 主导特征值
        price_accel : 价格加速度 (5日收益 - 前5日收益)
        liquidity : 流动性代理 (close × avg_vol_20)
        mom20 : 20日对数收益率
        """
        cfg = self.config
        return (
            abs(dominant_eigenvalue) >= cfg.eigenvalue_gate
            and price_accel <= cfg.price_accel_gate
            and liquidity >= cfg.liquidity_gate
            and mom20 >= cfg.momentum_floor
        )

    def evaluate(
        self,
        stock_code: str,
        trade_date: str,
        # 日线熵特征 (当前值)
        perm_entropy_20: float,
        perm_entropy_60: float,
        entropy_gap: float,
        entropy_percentile: float,
        path_irreversibility: float,
        dominant_eigenvalue: float,
        var_lift: float,
        # 新增: 价格结构特征
        price_accel: float = 0.0,
        mom20: float = 0.0,
        liquidity: float = 0.0,
        # 日线熵特征 (5天前值, 用于计算速度)
        dominant_eigenvalue_prev5: float = 0.0,
        path_irrev_prev5: float = 0.0,
    ) -> BifurcationResult:
        """
        评估分岔预警信号 (第二层打分)。

        调用前应已通过 passes_prescreen()。
        """
        cfg = self.config

        # ---- 1. 主导特征值得分 (权重 0.40) ----
        eig_velocity = (abs(dominant_eigenvalue) - abs(dominant_eigenvalue_prev5)) / 5.0
        eig_magnitude = abs(dominant_eigenvalue)

        # 接近临界值: 0.65 → 0, 1.0 → 1.0
        proximity_score = max(0, min(1, (eig_magnitude - cfg.eigenvalue_gate) /
                                     (1.0 - cfg.eigenvalue_gate)))
        # 加速逼近: 有加速额外加分
        velocity_bonus = max(0, min(0.3, eig_velocity / 0.03)) if eig_velocity > 0 else 0

        eigenvalue_score = min(1.0, proximity_score + velocity_bonus)

        # ---- 2. 多尺度熵剪刀差得分 (权重 0.25) ----
        gap_score = max(0, min(1, entropy_gap / 0.12))

        # 百分位低 → 短期更有序
        if np.isfinite(entropy_percentile) and entropy_percentile < 0.40:
            pctile_score = max(0, min(1, (0.40 - entropy_percentile) / 0.40))
        else:
            pctile_score = 0

        entropy_divergence_score = 0.6 * gap_score + 0.4 * pctile_score

        # ---- 3. 方向性得分 (权重 0.20) ----
        pi_score = max(0, min(1, path_irreversibility / 0.05))

        pi_velocity = path_irreversibility - path_irrev_prev5
        pi_spike_score = max(0, min(1, pi_velocity / 0.03)) if pi_velocity > 0 else 0

        directionality_score = 0.6 * pi_score + 0.4 * pi_spike_score

        # ---- 4. 动量质量得分 (权重 0.15, 替代 variance_state) ----
        # 正动量 + 负加速度 = "回踩蓄势" → 最好的入场形态
        # mom20 正且 price_accel 负 → 高分
        if mom20 > 0 and price_accel < 0:
            # 动量越强、回踩越深 → 分越高
            mom_quality = min(1.0, mom20 / 0.30)       # 30% mom → 满分
            pullback_quality = min(1.0, abs(price_accel) / 0.20)  # -20% accel → 满分
            momentum_quality_score = 0.5 * mom_quality + 0.5 * pullback_quality
        elif mom20 > 0:
            # 正动量但无回踩, 中等分
            momentum_quality_score = min(0.5, mom20 / 0.30)
        else:
            momentum_quality_score = 0.0

        # ---- 综合得分 ----
        w = cfg.weights
        total_score = (
            w['eigenvalue'] * eigenvalue_score +
            w['entropy_divergence'] * entropy_divergence_score +
            w['directionality'] * directionality_score +
            w['momentum_quality'] * momentum_quality_score
        )

        # ---- 信号判定 ----
        is_bifurcation = False
        if total_score >= cfg.bifurcation_score_min:
            signal: BifurcationSignal = 'buy'
            is_bifurcation = True
            reason = (f'分岔v2(|λ|={eig_magnitude:.3f} Δ={entropy_gap:.3f} '
                      f'PI={path_irreversibility:.3f} pa={price_accel:.3f})')
        elif total_score >= cfg.bifurcation_watch_min:
            signal = 'watch'
            reason = f'分岔观察v2(|λ|={eig_magnitude:.3f})'
        else:
            signal = 'skip'
            reason = ''

        return BifurcationResult(
            stock_code=stock_code,
            trade_date=trade_date,
            eigenvalue_score=eigenvalue_score,
            entropy_divergence_score=entropy_divergence_score,
            directionality_score=directionality_score,
            variance_state_score=momentum_quality_score,
            total_score=total_score,
            signal=signal,
            reason=reason,
            dominant_eigenvalue=dominant_eigenvalue,
            eigenvalue_velocity=eig_velocity,
            perm_entropy_20=perm_entropy_20,
            perm_entropy_60=perm_entropy_60,
            entropy_gap=entropy_gap,
            entropy_percentile=entropy_percentile,
            path_irreversibility=path_irreversibility,
            var_lift=var_lift,
            is_bifurcation=is_bifurcation,
        )


@dataclass
class TrendHoldConfig:
    """趋势持有配置 - 用于分岔买入的持仓"""

    # 卖出信号抑制: 当系统仍处于趋势态时, 提高卖出门槛
    sell_score_boost: float = 0.25          # 增强: 0.15→0.25, 趋势中更不容易被震出

    # 趋势强度指标
    eigenvalue_trend_min: float = 0.35      # 放宽: 0.50→0.35, 趋势判定更宽松
    path_irrev_trend_min: float = 0.015     # 放宽: 0.02→0.015
    entropy_pctile_exit: float = 0.92       # 提高: 0.90→0.92, 更晚才判定衰竭

    # 止损仍然生效 (分岔买入不豁免止损)
    stop_loss_override: bool = False

    # 最大持仓天数可以延长
    max_hold_days_extension: int = 60       # 增强: 30→60天, 覆盖大牛行情

    # 追踪止盈: 从最高点回撤此比例触发卖出
    trailing_stop_pct: float = 0.20         # 从峰值回撤20%止盈


class TrendHoldEvaluator:
    """
    趋势持有评估器。

    对通过 "分岔预警" 买入的持仓, 评估是否仍处于趋势态:
    - 若趋势尚在, 抑制卖出信号 (提高卖出门槛)
    - 若趋势已竭, 恢复正常卖出门槛
    """

    def __init__(self, config: Optional[TrendHoldConfig] = None):
        self.config = config or TrendHoldConfig()

    def is_trend_alive(
        self,
        dominant_eigenvalue: float,
        path_irreversibility: float,
        entropy_percentile: float,
        pnl_pct: float,
    ) -> bool:
        """
        判断趋势是否仍然存活。

        当以下条件满足时, 趋势被视为存活:
        - 主导特征值仍显著 (|λ| > 0.50)
        - 或路径不可逆性仍显著 (PI > 0.02)
        - 且熵百分位未到衰竭区 (EP < 0.90)

        额外: 如果盈利 > 20%, 只要熵未完全衰竭就继续持有
        """
        cfg = self.config

        if entropy_percentile > cfg.entropy_pctile_exit:
            return False

        eig_ok = abs(dominant_eigenvalue) > cfg.eigenvalue_trend_min
        pi_ok = path_irreversibility > cfg.path_irrev_trend_min
        profit_buffer = pnl_pct > 0.20  # 大幅盈利给予更多容忍

        if eig_ok or pi_ok or profit_buffer:
            return True

        return False

    def adjusted_sell_threshold(
        self,
        base_sell_score_min: float,
        dominant_eigenvalue: float,
        path_irreversibility: float,
        entropy_percentile: float,
        pnl_pct: float,
    ) -> float:
        """
        返回调整后的卖出门槛。

        趋势存活时提高门槛, 使策略更不容易过早卖出。
        """
        if self.is_trend_alive(dominant_eigenvalue, path_irreversibility,
                               entropy_percentile, pnl_pct):
            return base_sell_score_min + self.config.sell_score_boost
        return base_sell_score_min
