"""
Layer 3: 执行成本层 (Execution Cost)

基于以下论文实现：
1. conservative_driving_discrete.pdf - 保守驱动近优性
2. communication_induced_bifurcation.pdf - 战略放弃机制

核心思想：
- 保守驱动方案的耗散最多是最优值的 2 倍
- 高噪声环境下应主动放弃交易，而非等待止损

输出：
- 仓位规模
- 建仓模式（skip/probe/staged/full）
- 退出模式（abandon/reduce/trail）
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Literal
from dataclasses import dataclass

from .config import ExecutionCostConfig


ExecutionMode = Literal[
    'skip',     # 跳过
    'probe',    # 试探（25% 仓位）
    'staged',   # 分段（50% 仓位，分 3 天）
    'full',     # 全额（100% 仓位）
]

ExitMode = Literal[
    'abandon',  # 放弃（立即清仓）
    'reduce',   # 减仓
    'trail',    # 移动止盈
]


@dataclass
class ExecutionCostOutput:
    """执行成本层输出"""

    # 执行模式
    entry_mode: ExecutionMode
    exit_mode: ExitMode

    # 仓位建议
    position_scale: float       # 仓位比例（0-1）
    position_size: float        # 建议持仓金额

    # 成本评估
    estimated_cost: float       # 预估交易成本
    cost_adjusted_return: float # 成本调整后收益

    # 战略放弃标记
    abandonment_flag: bool      # 是否触发战略放弃
    abandonment_reason: str     # 放弃原因

    # 分段建仓计划
    staged_entry_days: int      # 分几天建仓
    staged_entry_ratios: list   # 每天建仓比例

    # 止盈止损
    take_profit_price: float    # 止盈价（全部清仓）
    stop_loss_price: float      # 止损价
    trailing_stop_pct: float    # 移动止损比例
    take_profit_partial_price: float = 0.0  # 分段止盈价（减半仓）

    def to_dict(self) -> Dict:
        return {
            'entry_mode': self.entry_mode,
            'exit_mode': self.exit_mode,
            'position_scale': self.position_scale,
            'position_size': self.position_size,
            'estimated_cost': self.estimated_cost,
            'cost_adjusted_return': self.cost_adjusted_return,
            'abandonment_flag': self.abandonment_flag,
            'abandonment_reason': self.abandonment_reason,
            'staged_entry_days': self.staged_entry_days,
            'staged_entry_ratios': self.staged_entry_ratios,
            'take_profit_price': self.take_profit_price,
            'take_profit_partial_price': self.take_profit_partial_price,
            'stop_loss_price': self.stop_loss_price,
            'trailing_stop_pct': self.trailing_stop_pct,
        }


class ExecutionCost:
    """
    执行成本评估器

    基于 12 篇论文中的 conservative_driving 和 communication_induced_bifurcation 实现。
    """

    def __init__(self, config: ExecutionCostConfig, initial_capital: float = 1_000_000.0):
        self.config = config
        self.initial_capital = initial_capital

    def estimate_transaction_cost(
        self,
        trade_value: float,
        price: float,
        volume: float,
        avg_volume: float,
    ) -> float:
        """
        估算交易成本

        包括：
        1. 佣金
        2. 印花税
        3. 滑点（基于成交量冲击）

        参数
        ----
        trade_value : float
            交易金额
        price : float
            当前价格
        volume : float
            当前成交量
        avg_volume : float
            平均成交量（20 日）
        """
        # 1. 佣金（万三，最低 5 元）
        commission = max(5, trade_value * self.config.commission)

        # 2. 印花税（千一，卖出收取，这里假设双向预估）
        stamp_tax = trade_value * self.config.stamp_tax * 0.5  # 预计一半概率卖出

        # 3. 滑点（基于成交量冲击）
        volume_ratio = trade_value / (avg_volume * price * 100) if avg_volume > 0 else 0
        # 冲击成本：交易量占日均成交量比例越大，滑点越高
        market_impact = min(0.02, volume_ratio * 0.01)  # 最大 2%
        slippage = trade_value * market_impact

        total_cost = commission + stamp_tax + slippage

        return total_cost

    def compute_abandonment_score(
        self,
        noise_cost: float,
        market_gate_score: float,
        volatility: float,
    ) -> float:
        """
        计算战略放弃分数

        基于 communication_induced_bifurcation 论文：
        当噪声超过临界阈值时，最优策略是放弃精细控制。
        """
        # 1. 噪声成本过高
        noise_penalty = max(0, (noise_cost - self.config.abandonment_noise_threshold) / 0.3)

        # 2. 市场门控关闭
        gate_penalty = (1 - market_gate_score) * 0.5

        # 3. 波动率过高
        vol_penalty = max(0, (volatility - 0.05) / 0.05)  # 年化波动>50% 时惩罚

        # 综合放弃分数
        abandonment_score = (
            noise_penalty * 0.5 +
            gate_penalty * 0.3 +
            vol_penalty * 0.2
        )

        return min(1, max(0, abandonment_score))

    def determine_entry_mode(
        self,
        signal_strength: float,
        abandonment_score: float,
        gate_open: bool,
    ) -> Tuple[ExecutionMode, float, int, list]:
        """
        确定建仓模式

        基于 conservative_driving 论文：
        分段建仓的保守策略接近最优。
        """
        if not gate_open or abandonment_score > 0.6:
            return 'skip', 0.0, 0, []

        # 信号强度分档
        # 弱信号直接跳过（P1：回测显示低信号交易的手续费+滑点是主要亏损来源）
        if signal_strength < self.config.signal_weak:
            return 'skip', 0.0, 0, []

        elif signal_strength < self.config.signal_medium:
            # 中等信号：试探性建仓
            return 'probe', 0.25, 1, [1.0]

        elif signal_strength < self.config.signal_strong:
            # 中等信号：分段建仓
            return 'staged', 0.50, 3, [0.25, 0.25, 0.50]

        else:
            # 强信号：全额建仓（但仍分 2 天）
            return 'full', 0.80, 2, [0.50, 0.50]

    def determine_exit_mode(
        self,
        current_return: float,
        state: str,
        abandonment_score: float,
    ) -> ExitMode:
        """
        确定退出模式
        """
        # 战略放弃触发
        if abandonment_score > 0.7:
            return 'abandon'

        # 状态恶化
        if state in ['diffusion', 'exhaustion']:
            if current_return < -self.config.stop_loss:
                return 'abandon'
            else:
                return 'reduce'

        # 止盈状态
        if current_return >= self.config.take_profit:
            return 'trail'  # 移动止盈

        return 'trail'  # 默认移动止盈

    def compute_position_size(
        self,
        position_scale: float,
        signal: str,
        market_state: str,
    ) -> float:
        """
        计算建议持仓金额
        """
        # 基础仓位
        base = self.initial_capital * self.config.base_position

        # 根据信号调整
        if signal == 'buy':
            base *= 1.5
        elif signal == 'hold':
            base *= 1.0
        else:
            base *= 0.5

        # 根据市场状态调整
        state_adjust = {
            'compression': 1.2,
            'transition': 1.0,
            'expansion': 0.8,
            'distorted': 0.5,
            'neutral': 1.0,
            'abandon': 0.0,
        }
        base *= state_adjust.get(market_state, 1.0)

        # 应用仓位比例
        final_size = base * position_scale

        # 限制在最大仓位内
        max_size = self.initial_capital * self.config.max_position
        final_size = min(final_size, max_size)

        return final_size

    def evaluate(
        self,
        signal_strength: float,
        market_gate_score: float,
        market_state: str,
        stock_state: str,
        current_price: float,
        entry_price: Optional[float] = None,
        current_return: float = 0.0,
        volatility: float = 0.02,
        noise_cost: float = 0.3,
        avg_volume: float = 1000000,
        target_volume: float = 10000,
    ) -> ExecutionCostOutput:
        """
        执行成本层评估

        参数
        ----
        signal_strength : float
            信号强度（0-1）
        market_gate_score : float
            市场门控得分
        market_state : str
            市场状态
        stock_state : str
            个股状态
        current_price : float
            当前价格
        entry_price : float, optional
            建仓价格（已有持仓时）
        current_return : float
            当前收益
        volatility : float
            波动率
        noise_cost : float
            噪声成本
        avg_volume : float
            平均成交量
        target_volume : float
            目标交易股数
        """
        gate_open = market_state not in ['abandon', 'distorted']

        # 1. 计算战略放弃分数
        abandonment_score = self.compute_abandonment_score(
            noise_cost, market_gate_score, volatility
        )

        # 2. 确定建仓模式
        entry_mode, position_scale, staged_days, staged_ratios = self.determine_entry_mode(
            signal_strength, abandonment_score, gate_open
        )

        # 3. 确定退出模式
        exit_mode = self.determine_exit_mode(
            current_return, stock_state, abandonment_score
        )

        # 4. 计算交易成本
        trade_value = target_volume * current_price
        est_cost = self.estimate_transaction_cost(
            trade_value, current_price, target_volume, avg_volume
        )

        # 5. 成本调整后收益（预估）
        cost_ratio = est_cost / trade_value if trade_value > 0 else 0
        # 假设预期收益 10%，减去成本
        cost_adjusted_return = 0.10 - cost_ratio

        # 6. 计算仓位大小
        position_size = self.compute_position_size(
            position_scale, 'buy' if entry_mode != 'skip' else 'wait', market_state
        )

        # 7. 止盈止损价格（分段止盈：+4% 减半仓，+8% 清仓）
        if entry_price is None:
            entry_price = current_price

        take_profit_partial = getattr(self.config, 'take_profit_partial', 0.04)
        take_profit_partial_price = entry_price * (1 + take_profit_partial)
        take_profit_price = entry_price * (1 + self.config.take_profit)
        stop_loss_price = entry_price * (1 - self.config.stop_loss)

        # 8. 放弃原因（使用降低后的阈值 0.6）
        abandonment_reason = ""
        if abandonment_flag := (abandonment_score > 0.5 or not gate_open):
            if not gate_open:
                abandonment_reason = "market_gate_closed"
            elif noise_cost > self.config.abandonment_noise_threshold:
                abandonment_reason = "high_noise_cost"
            elif volatility > 0.05:
                abandonment_reason = "high_volatility"
            else:
                abandonment_reason = "combined_risk"

        return ExecutionCostOutput(
            entry_mode=entry_mode,
            exit_mode=exit_mode,
            position_scale=position_scale,
            position_size=position_size,
            estimated_cost=est_cost,
            cost_adjusted_return=cost_adjusted_return,
            abandonment_flag=abandonment_flag,
            abandonment_reason=abandonment_reason,
            staged_entry_days=staged_days,
            staged_entry_ratios=staged_ratios,
            take_profit_price=take_profit_price,
            take_profit_partial_price=take_profit_partial_price,
            stop_loss_price=stop_loss_price,
            trailing_stop_pct=self.config.trailing_stop,
        )
