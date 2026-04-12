"""
Layer 1: 市场门控层 (Market Gate)

基于 communication_induced_bifurcation_power_packet.pdf 论文实现。

核心思想：
- 当噪声超过临界阈值时，系统应采用"战略性放弃"策略
- 市场耦合熵衡量行业间联动程度
- 噪声成本衡量交易环境恶劣程度

输出状态：
- compression: 低熵压缩态（最佳开仓窗口）
- transition: 转换态（谨慎开仓）
- expansion: 扩张态（减仓）
- distorted: 失真态（观望）
- neutral: 中性态（正常）
- abandon: 放弃态（清仓）
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Literal
from dataclasses import dataclass
from scipy import stats

from .config import MarketGateConfig


MarketState = Literal[
    'compression',   # 压缩态（低熵，最佳开仓）
    'transition',    # 转换态
    'expansion',     # 扩张态（减仓）
    'distorted',     # 失真态（观望）
    'neutral',       # 中性态
    'abandon',       # 放弃态（清仓）
]


@dataclass
class MarketGateOutput:
    """市场门控输出"""

    state: MarketState
    coupling_entropy: float           # 行业耦合熵
    noise_cost: float                 # 噪声成本
    phase_state: MarketState          # 相位状态
    abandonment_flag: bool            # 战略放弃标记

    # 详细指标
    industry_correlation: float       # 行业相关性
    volume_imbalance: float           # 成交量不平衡度
    volatility_cluster: float         # 波动率聚集度
    liquidity_stress: float           # 流动性压力

    # 门控得分（0-1，越高越适合交易）
    gate_score: float

    def to_dict(self) -> Dict:
        return {
            'state': self.state,
            'coupling_entropy': self.coupling_entropy,
            'noise_cost': self.noise_cost,
            'phase_state': self.phase_state,
            'abandonment_flag': self.abandonment_flag,
            'gate_score': self.gate_score,
            'industry_correlation': self.industry_correlation,
            'volume_imbalance': self.volume_imbalance,
            'volatility_cluster': self.volatility_cluster,
            'liquidity_stress': self.liquidity_stress,
        }


class MarketGate:
    """
    市场门控器

    基于 12 篇论文中的 communication_induced_bifurcation 论文实现。
    核心发现：当环境噪声超过临界阈值 Dc 时，最优策略是"战略性放弃"。
    """

    def __init__(self, config: MarketGateConfig):
        self.config = config

    def compute_coupling_entropy(
        self,
        returns: pd.DataFrame,
        industry_map: Dict[str, str],
    ) -> float:
        """
        计算行业耦合熵

        基于 industry lead-lag returns 网络的 Shannon 熵。

        参数
        ----
        returns : pd.DataFrame
            股票收益率矩阵 (日期×股票)
        industry_map : Dict[str, str]
            股票代码→行业映射

        返回
        ----
        float
            归一化耦合熵 [0, 1]
            - 接近 0：行业独立（低耦合）
            - 接近 1：行业高度联动（高耦合）
        """
        if returns.empty or len(returns) < self.config.industry_window:
            return 0.5

        # 按行业分组计算行业收益率
        industry_returns = {}
        for stock, industry in industry_map.items():
            if stock in returns.columns:
                if industry not in industry_returns:
                    industry_returns[industry] = []
                industry_returns[industry].append(returns[stock])

        # 计算行业等权收益率
        industry_indices = {}
        for industry, stocks in industry_returns.items():
            if len(stocks) >= self.config.industry_min_stocks:
                industry_indices[industry] = pd.concat(stocks, axis=1).mean(axis=1)

        if len(industry_indices) < 2:
            return 0.5

        industry_df = pd.DataFrame(industry_indices)

        # 计算行业间相关性矩阵
        corr_matrix = industry_df.tail(self.config.industry_window).corr()

        # 提取上三角相关系数
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        ).stack()

        if len(upper_tri) == 0:
            return 0.5

        # 将相关性映射到 [0, 1] 并计算熵
        # 高相关 → 高耦合 → 高熵
        correlations = (upper_tri.values + 1) / 2  # [-1,1] → [0,1]

        # Shannon 熵
        n_bins = 10
        hist, _ = np.histogram(correlations, bins=n_bins)
        prob = hist / hist.sum()
        prob = prob[prob > 0]

        entropy = -np.sum(prob * np.log(prob))
        max_entropy = np.log(n_bins)

        return entropy / max_entropy

    def compute_noise_cost(
        self,
        returns: pd.DataFrame,
        volumes: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> float:
        """
        计算噪声成本

        综合指标：
        1. 波动率聚集度（volatility clustering）
        2. 成交量不平衡度
        3. 流动性压力
        4. 极端收益频率

        返回
        ----
        float
            噪声成本 [0, 1]，越高表示环境越恶劣
        """
        if returns.empty or len(returns) < 20:
            return 0.5

        window = 20

        # 1. 波动率聚集度
        # 使用 GARCH(1,1) 的简化代理：波动率自相关
        vol = returns.rolling(window).std()
        vol_ac = vol.apply(lambda x: x.autocorr() if len(x) > 10 else 0).fillna(0).mean().mean()
        volatility_cluster = max(0, min(1, (vol_ac + 1) / 2))  # [-1,1] → [0,1]

        # 2. 成交量不平衡度
        vol_change = volumes.pct_change(fill_method=None)
        vol_imbalance = vol_change.abs().mean().mean()  # 双重 mean 处理 DataFrame
        volume_imbalance = min(1, vol_imbalance * 10)  # 归一化

        # 3. 流动性压力（用价格跳跃代理）
        log_returns = np.log(prices / prices.shift(1))
        jumps = log_returns.abs() > (log_returns.rolling(window).std() * 3)
        liquidity_stress = jumps.mean().mean()

        # 4. 极端收益频率
        extreme_returns = (returns.abs() > returns.rolling(window).std().mean() * 2).mean().mean()

        # 综合噪声成本
        noise_cost = (
            volatility_cluster * 0.30 +
            volume_imbalance * 0.25 +
            liquidity_stress * 0.25 +
            extreme_returns * 0.20
        )

        return min(1, max(0, noise_cost))

    def compute_phase_state(
        self,
        market_returns: pd.Series,
        coupling_entropy: float,
        noise_cost: float,
    ) -> Tuple[MarketState, Dict]:
        """
        计算市场相位状态

        基于耦合熵和噪声成本的联合判断。
        包含 A 股日历效应去偏（论文 12: Statistical warning indicators）。
        """
        details = {
            'coupling_level': 'low',
            'noise_level': 'low',
            'combined_signal': 'neutral',
            'calendar_adjustment': 'none',
        }

        # 0. 日历效应去偏
        calendar_penalty = 0.0
        if self.config.calendar_debiasing and hasattr(market_returns.index, 'month'):
            last_date = market_returns.index[-1] if len(market_returns) > 0 else None
            if last_date is not None and hasattr(last_date, 'month'):
                month = last_date.month
                day = last_date.day if hasattr(last_date, 'day') else 15

                # 季报窗口期：1/4/7/10 月下旬，耦合熵偏高是正常现象
                if month in self.config.earnings_months and day >= self.config.earnings_day_start:
                    calendar_penalty = -0.10  # 降低耦合熵，避免误判
                    details['calendar_adjustment'] = 'earnings_window'

                # 春节/国庆前后窗口：噪声偏高是正常现象
                # 春节大致在 1-2 月，国庆在 10 月初
                if (month == 1 and day >= 20) or (month == 2 and day <= 15):
                    calendar_penalty += -0.05
                    details['calendar_adjustment'] = 'spring_festival'
                elif month == 10 and day <= 10:
                    calendar_penalty += -0.05
                    details['calendar_adjustment'] = 'national_day'

        # 对耦合熵做日历校正
        adjusted_coupling = max(0.0, coupling_entropy + calendar_penalty)

        # 1. 耦合熵判断（使用日历校正后的值）
        if adjusted_coupling < self.config.coupling_entropy_low:
            coupling_state = 'low'
        elif adjusted_coupling < self.config.coupling_entropy_high:
            coupling_state = 'medium'
        elif adjusted_coupling < self.config.coupling_entropy_critical:
            coupling_state = 'high'
        else:
            coupling_state = 'critical'

        details['coupling_level'] = coupling_state

        # 2. 噪声成本判断
        if noise_cost < self.config.noise_cost_low:
            noise_state = 'low'
        elif noise_cost < self.config.noise_cost_high:
            noise_state = 'medium'
        elif noise_cost < self.config.noise_cost_critical:
            noise_state = 'high'
        else:
            noise_state = 'critical'

        details['noise_level'] = noise_state

        # 3. 联合判断
        if noise_state == 'critical':
            state = 'abandon'
            details['combined_signal'] = 'critical_noise'
        elif coupling_state == 'critical':
            state = 'distorted'
            details['combined_signal'] = 'critical_coupling'
        elif coupling_state == 'high' and noise_state == 'high':
            state = 'expansion'
            details['combined_signal'] = 'high_activity'
        elif coupling_state == 'low' and noise_state == 'low':
            state = 'compression'
            details['combined_signal'] = 'optimal_entry'
        elif coupling_state == 'medium' and noise_state == 'low':
            state = 'transition'
            details['combined_signal'] = 'cautious_entry'
        else:
            state = 'neutral'
            details['combined_signal'] = 'normal'

        return state, details

    def compute_gate_score(
        self,
        state: MarketState,
        coupling_entropy: float,
        noise_cost: float,
    ) -> float:
        """
        计算门控得分（0-1，越高越适合交易）
        """
        # 基础得分
        base_scores = {
            'compression': 1.0,
            'transition': 0.7,
            'expansion': 0.3,
            'distorted': 0.0,
            'neutral': 0.5,
            'abandon': 0.0,
        }

        base = base_scores.get(state, 0.5)

        # 调整：低耦合熵加分
        coupling_adjust = (1 - coupling_entropy) * 0.1

        # 调整：低噪声成本加分
        noise_adjust = (1 - noise_cost) * 0.1

        return max(0, min(1, base + coupling_adjust + noise_adjust))

    def evaluate(
        self,
        returns: pd.DataFrame,
        volumes: pd.DataFrame,
        prices: pd.DataFrame,
        industry_map: Optional[Dict[str, str]] = None,
    ) -> MarketGateOutput:
        """
        执行市场门控评估

        参数
        ----
        returns : pd.DataFrame
            股票收益率矩阵
        volumes : pd.DataFrame
            成交量矩阵
        prices : pd.DataFrame
            价格矩阵
        industry_map : Dict[str, str], optional
            股票代码→行业映射

        返回
        ----
        MarketGateOutput
            门控评估结果
        """
        # 默认行业映射（单行业）
        if industry_map is None:
            industry_map = {col: 'default' for col in returns.columns}

        # 1. 计算耦合熵
        coupling_entropy = self.compute_coupling_entropy(returns, industry_map)

        # 2. 计算噪声成本
        noise_cost = self.compute_noise_cost(returns, volumes, prices)

        # 3. 计算市场指数收益率（用于相位判断）
        market_returns = returns.mean(axis=1)

        # 4. 计算相位状态
        phase_state, phase_details = self.compute_phase_state(
            market_returns, coupling_entropy, noise_cost
        )

        # 5. 计算门控得分
        gate_score = self.compute_gate_score(phase_state, coupling_entropy, noise_cost)

        # 6. 判断是否需要战略放弃
        abandonment_flag = (
            noise_cost >= self.config.noise_cost_critical or
            phase_state == 'abandon'
        )

        # 计算辅助指标
        industry_returns = {}
        for stock, industry in industry_map.items():
            if stock in returns.columns:
                if industry not in industry_returns:
                    industry_returns[industry] = []
                industry_returns[industry].append(returns[stock])

        if len(industry_returns) >= 2:
            industry_indices = {
                ind: pd.concat(stocks, axis=1).mean(axis=1)
                for ind, stocks in industry_returns.items()
            }
            industry_df = pd.DataFrame(industry_indices)
            industry_correlation = industry_df.tail(20).corr().values[np.triu_indices(len(industry_df), k=1)].mean()
        else:
            industry_correlation = 0.0

        vol_change = volumes.pct_change(fill_method=None)
        volume_imbalance = vol_change.abs().mean().mean()

        vol = returns.rolling(20).std()
        vol_ac = vol.apply(lambda x: x.autocorr() if len(x) > 10 else 0).fillna(0).mean().mean()
        volatility_cluster = max(0, min(1, (vol_ac + 1) / 2))

        log_returns = np.log(prices / prices.shift(1))
        jumps = log_returns.abs() > (log_returns.rolling(20).std().mean() * 3)
        liquidity_stress = jumps.mean().mean()

        return MarketGateOutput(
            state=phase_state,
            coupling_entropy=coupling_entropy,
            noise_cost=noise_cost,
            phase_state=phase_state,
            abandonment_flag=abandonment_flag,
            gate_score=gate_score,
            industry_correlation=industry_correlation if not pd.isna(industry_correlation) else 0.0,
            volume_imbalance=volume_imbalance if not pd.isna(volume_imbalance) else 0.0,
            volatility_cluster=volatility_cluster,
            liquidity_stress=liquidity_stress if not pd.isna(liquidity_stress) else 0.0,
        )
