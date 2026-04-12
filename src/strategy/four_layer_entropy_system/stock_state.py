"""
Layer 2: 个股状态层 (Stock State)

基于以下论文实现：
1. Seifert (2025): 粗粒化熵产生理论 - 路径不可逆性
2. Ma et al. (2026): 主导特征值与倍周期分岔预测

状态流转：
低熵压缩 → 临界减速 → 分叉启动 → 扩散/衰竭
  ↓          ↓          ↓          ↓
path_irrev  dominant_eig  突破确认  退出信号
< 0.05      → 0.9        → 放量     → 熵增

输出：
- 状态流转路径
- 继续持有/退出/重新评估结论
- 三套子策略得分
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Literal
from dataclasses import dataclass, field

from .config import LayerConfig, StockStateConfig


StateFlow = Literal[
    'observation',      # 观察态（未激活）
    'compression',      # 低熵压缩态
    'critical_slowing', # 临界减速态
    'bifurcation',      # 分叉启动态
    'diffusion',        # 扩散态
    'exhaustion',       # 衰竭态
]


@dataclass
class StockStateOutput:
    """个股状态输出"""

    # 当前状态
    current_state: StateFlow

    # 核心特征
    path_irreversibility: float       # 路径不可逆性
    dominant_eigenvalue: float        # 主导特征值
    permutation_entropy: float        # 排列熵
    phase_adjusted_ar1: float         # 相位校正 AR(1)

    # 质量得分
    entropy_quality: float            # 熵质量得分
    bifurcation_quality: float        # 分叉质量得分
    trigger_quality: float            # 触发质量得分

    # 综合得分
    total_score: float

    # 新增特征（P1）— 带默认值的字段放在后面
    entropy_accel: float = 0.0        # 熵加速度（二阶变化率）
    entropy_gap: float = 0.0          # 多尺度熵差（60-20）
    entropy_percentile: float = 0.5   # 熵百分位排名（120窗口）
    var_lift: float = 0.0             # 方差抬升（10/20）

    # 硬性门槛是否通过
    hard_gate_passed: bool = True

    # 状态历史
    state_history: list = field(default_factory=list)

    # 交易信号
    signal: Literal['buy', 'hold', 'sell', 'wait'] = 'wait'

    def to_dict(self) -> Dict:
        return {
            'current_state': self.current_state,
            'path_irreversibility': self.path_irreversibility,
            'dominant_eigenvalue': self.dominant_eigenvalue,
            'permutation_entropy': self.permutation_entropy,
            'phase_adjusted_ar1': self.phase_adjusted_ar1,
            'entropy_accel': self.entropy_accel,
            'entropy_gap': self.entropy_gap,
            'entropy_percentile': self.entropy_percentile,
            'var_lift': self.var_lift,
            'entropy_quality': self.entropy_quality,
            'bifurcation_quality': self.bifurcation_quality,
            'trigger_quality': self.trigger_quality,
            'total_score': self.total_score,
            'hard_gate_passed': self.hard_gate_passed,
            'signal': self.signal,
        }


class StockState:
    """
    个股状态评估器

    基于 12 篇论文中的熵产生理论和分岔预测理论实现。
    """

    def __init__(self, config: StockStateConfig, layer_config: Optional[LayerConfig] = None):
        self.config = config
        self.layer_config = layer_config

    # ========== 核心特征计算 ==========

    def compute_path_irreversibility(
        self,
        returns: pd.Series,
        window: int = 20,
    ) -> pd.Series:
        """
        计算路径不可逆性熵

        基于 Seifert (2025) 的粗粒化轨迹方法。
        衡量系统时间反演对称性的破缺程度。
        """
        if len(returns) < window + 10:
            return pd.Series(np.nan, index=returns.index)

        def calc_irreversibility(ret_window):
            """计算单窗口的路径不可逆性"""
            if len(ret_window) < 10:
                return np.nan

            # 三态离散化
            sigma = np.std(ret_window)
            if sigma < 1e-10:
                return 0.0

            threshold = 0.5 * sigma
            states = np.zeros(len(ret_window), dtype=np.int64)
            states[ret_window < -threshold] = -1
            states[ret_window > threshold] = 1

            # 状态转移计数
            n_states = 3
            counts = np.zeros((n_states, n_states), dtype=np.float64)

            for t in range(len(states) - 1):
                i, j = int(states[t] + 1), int(states[t + 1] + 1)
                if 0 <= i < n_states and 0 <= j < n_states:
                    counts[i, j] += 1.0

            # KL 散度
            total = counts.sum()
            if total < 10:
                return np.nan

            forward = counts / total
            backward = counts.T / total

            mask = (forward > 1e-10) & (backward > 1e-10)
            if not np.any(mask):
                return 0.0

            kl_div = np.sum(forward[mask] * np.log(forward[mask] / backward[mask]))
            return max(0.0, kl_div)

        # 滚动计算
        result = returns.rolling(window=window, min_periods=10).apply(
            lambda x: calc_irreversibility(x if isinstance(x, np.ndarray) else x.values), raw=False
        )

        return result

    def compute_dominant_eigenvalue(
        self,
        returns: pd.Series,
        window: int = 20,
        order: int = 2,
    ) -> pd.Series:
        """
        计算主导特征值

        基于 Ma et al. (2026) 的方法。
        从自相关结构提取特征值，用于检测临界减速。

        |λ| → 1 表示系统接近失稳
        """
        if len(returns) < window + 10:
            return pd.Series(np.nan, index=returns.index)

        def calc_eigenvalue(ret_window):
            """计算单窗口的主导特征值"""
            if isinstance(ret_window, np.ndarray):
                values = ret_window
            else:
                values = ret_window.values
            values = values[np.isfinite(values)]

            if len(values) < max(12, order + 6):
                return np.nan

            # 中心化
            centered = values - np.mean(values)
            if np.std(centered) <= 1e-12:
                return np.nan

            # 计算自协方差
            acov = []
            for lag in range(order + 1):
                left = centered[:len(centered) - lag]
                right = centered[lag:]
                if len(right) == 0:
                    return np.nan
                acov.append(np.dot(left, right) / len(right))

            # Yule-Walker 方程
            system = np.array([
                [acov[abs(i - j)] for j in range(order)]
                for i in range(order)
            ])
            rhs = np.array(acov[1:order + 1])

            try:
                phi = np.linalg.solve(system + np.eye(order) * 1e-8, rhs)
            except np.linalg.LinAlgError:
                return np.nan

            # 伴随矩阵
            companion = np.zeros((order, order))
            companion[0, :] = phi
            for i in range(1, order):
                companion[i, i - 1] = 1.0

            # 特征值
            eigs = np.linalg.eigvals(companion)
            dominant = eigs[np.argmax(np.abs(eigs))]

            return float(np.real(dominant))

        result = returns.rolling(window=window, min_periods=10).apply(
            lambda x: calc_eigenvalue(x), raw=True
        )

        return result

    def compute_permutation_entropy(
        self,
        returns: pd.Series,
        window: int = 20,
        order: int = 3,
    ) -> pd.Series:
        """
        计算排列熵

        基于 Bandt & Pompe (2002) 的标准方法。
        """
        import math

        if len(returns) < window + 10:
            return pd.Series(np.nan, index=returns.index)

        def calc_perm_entropy(values):
            """计算单窗口的排列熵"""
            if isinstance(values, np.ndarray):
                values = values
            else:
                values = values.values
            values = values[np.isfinite(values)]

            if len(values) < order + 2:
                return np.nan

            # 序列表计数
            counts = {}
            for idx in range(len(values) - order + 1):
                pattern = tuple(np.argsort(values[idx:idx + order], kind='mergesort'))
                counts[pattern] = counts.get(pattern, 0) + 1

            if not counts:
                return np.nan

            # Shannon 熵
            freq = np.array(list(counts.values()), dtype=np.float64)
            prob = freq / freq.sum()
            entropy = -np.sum(prob * np.log(prob))

            # 归一化
            normalizer = np.log(math.factorial(order))
            if normalizer <= 0:
                return np.nan

            return entropy / normalizer

        result = returns.rolling(window=window, min_periods=10).apply(
            lambda x: calc_perm_entropy(x if not isinstance(x, np.ndarray) else x), raw=False
        )

        return result

    def compute_phase_adjusted_ar1(
        self,
        returns: pd.Series,
        dates: pd.Series,
        window: int = 20,
    ) -> pd.Series:
        """
        计算相位校正的 AR(1)

        基于 statistical_warning_indicators 论文的启示：
        周期强迫会扭曲传统预警信号，需要做相位校正。

        校正因素：
        1. 星期效应
        2. 月度效应
        3. 季度效应
        """
        if len(returns) < window + 10:
            return pd.Series(np.nan, index=returns.index)

        def calc_ar1(values):
            """计算 AR(1)"""
            if len(values) < 10:
                return np.nan

            x = values[:-1]
            y = values[1:]

            if len(x) < 10:
                return np.nan

            # 简单线性回归
            corr = np.corrcoef(x, y)[0, 1]
            if np.isnan(corr):
                return np.nan

            std_x = np.std(x)
            std_y = np.std(y)

            if std_x < 1e-10 or std_y < 1e-10:
                return np.nan

            return corr * (std_y / std_x)

        # 基础 AR(1)
        raw_ar1 = returns.rolling(window=window).apply(
            lambda x: calc_ar1(x if not isinstance(x, np.ndarray) else x), raw=False
        )

        # 相位校正（简化版本：减去星期效应）
        dates = pd.to_datetime(dates)
        day_of_week = dates.dt.dayofweek

        # 计算星期效应
        dow_effect = returns.groupby(day_of_week).mean()
        dow_adjustment = returns.map(lambda d: dow_effect.get(dates[returns == d].dt.dayofweek.iloc[0] if len(dates[returns == d]) > 0 else 0, 0))

        # 校正后收益率
        adjusted_returns = returns - dow_adjustment

        # 校正后 AR(1)
        adjusted_ar1 = adjusted_returns.rolling(window=window).apply(
            lambda x: calc_ar1(x if not isinstance(x, np.ndarray) else x), raw=False
        )

        return adjusted_ar1

    # ========== 新增特征计算（P1） ==========

    def compute_entropy_accel(
        self,
        perm_entropy_series: pd.Series,
        step: int = 5,
    ) -> pd.Series:
        """
        计算熵加速度（二阶变化率）

        entropy_slope_5 = diff(perm_entropy, 5) / 5
        entropy_accel_5 = diff(entropy_slope_5)

        负值表示熵在加速收缩（压缩加深），正值表示熵在加速扩张。
        """
        slope = perm_entropy_series.diff(step) / step
        accel = slope.diff()
        return accel

    def compute_entropy_gap(
        self,
        returns: pd.Series,
    ) -> pd.Series:
        """
        计算多尺度熵差（60 日排列熵 - 20 日排列熵）

        正值表示长期比短期更无序，短期压缩有效。
        """
        perm_20 = self.compute_permutation_entropy(returns, window=20)
        perm_60 = self.compute_permutation_entropy(returns, window=60)
        return perm_60 - perm_20

    def compute_entropy_percentile(
        self,
        perm_entropy_series: pd.Series,
        window: int = 120,
    ) -> pd.Series:
        """
        计算排列熵的滚动百分位排名（120 日窗口）

        低百分位（< 0.40）表示当前熵处于历史低位 = 压缩态。
        """
        def percentile_of_last(values):
            if len(values) < max(8, window // 3):
                return np.nan
            finite = values[np.isfinite(values)]
            if len(finite) < 8:
                return np.nan
            last = finite[-1]
            return float(np.sum(finite < last)) / len(finite)

        result = perm_entropy_series.rolling(window=window, min_periods=max(8, window // 3)).apply(
            percentile_of_last, raw=True
        )
        return result

    def compute_var_lift(
        self,
        returns: pd.Series,
        short_window: int = 10,
        long_window: int = 20,
    ) -> pd.Series:
        """
        计算方差抬升（临界减速经典指标）

        var_lift = var(short) / var(long) - 1
        正值表示短期方差大于长期 = 失稳前兆。
        """
        var_short = returns.rolling(window=short_window, min_periods=5).var(ddof=0)
        var_long = returns.rolling(window=long_window, min_periods=10).var(ddof=0)
        return var_short / var_long.replace(0.0, np.nan) - 1.0

    # ========== 质量得分计算 ==========

    def compute_entropy_quality(
        self,
        path_irrev: float,
        perm_entropy: float,
        entropy_percentile: float = 0.5,
        entropy_gap: float = 0.0,
    ) -> float:
        """
        熵质量得分

        使用配置中的 entropy_quality_weights，综合评估：
        1. entropy_percentile: 低百分位 = 高质量
        2. entropy_gap: 正值（长期 > 短期） = 短期压缩有效
        3. perm_entropy_compression: 低排列熵 = 有序
        4. path_irrev_compression: 低路径不可逆 = 接近可逆
        """
        if np.isnan(path_irrev) or np.isnan(perm_entropy):
            return 0.5

        weights = self.config.entropy_quality_weights

        # 熵百分位：低值好（处于历史低位）
        pctile = entropy_percentile if np.isfinite(entropy_percentile) else 0.5
        pctile_score = 1.0 - min(1.0, pctile)

        # 熵差：正值好（长期 > 短期 = 短期压缩）
        gap = entropy_gap if np.isfinite(entropy_gap) else 0.0
        gap_score = min(1.0, max(0, gap * 20 + 0.5))  # 0.025 → 1.0

        # 排列熵压缩：低值好
        entropy_score = 1.0 - min(1.0, perm_entropy / 1.0)

        # 路径不可逆性压缩：低值好
        irrev_score = 1.0 - min(1.0, path_irrev / 0.3)

        quality = (
            weights.get('entropy_percentile', 0.40) * pctile_score +
            weights.get('entropy_gap', 0.30) * gap_score +
            weights.get('perm_entropy_compression', 0.15) * entropy_score +
            weights.get('path_irrev_compression', 0.15) * irrev_score
        )

        return max(0, min(1, quality))

    def compute_bifurcation_quality(
        self,
        dominant_eig: float,
        path_irrev: float,
        phase_ar1: float,
        entropy_accel: float = 0.0,
        var_lift: float = 0.0,
    ) -> float:
        """
        分叉质量得分

        基于 dominant eigenvalue 的临界减速检测，
        新增 entropy_accel（结构切换速度）和 var_lift（方差抬升）。
        """
        weights = self.config.feature_weights

        # 主导特征值绝对值（接近 1 表示临界）
        eig_score = min(1.0, abs(dominant_eig) / 0.9) if np.isfinite(dominant_eig) else 0.5

        # 路径不可逆性（低值表示压缩）
        irrev_score = 1.0 - min(1.0, path_irrev / 0.3) if np.isfinite(path_irrev) else 0.5

        # 相位校正 AR(1)（高值表示减速）
        ar1_score = min(1.0, abs(phase_ar1) / 0.9) if np.isfinite(phase_ar1) else 0.5

        # 熵加速度（负值=加速压缩=好信号，映射到 [0,1]）
        ea = entropy_accel if np.isfinite(entropy_accel) else 0.0
        accel_score = min(1.0, max(0, -ea * 50 + 0.5))  # 负值加分

        # 方差抬升（正值=失稳前兆=临界减速信号）
        vl = var_lift if np.isfinite(var_lift) else 0.0
        var_lift_score = min(1.0, max(0, vl * 2 + 0.3))

        # 综合得分
        quality = (
            weights.get('dominant_eig_abs', 0.25) * eig_score +
            weights.get('path_irreversibility', 0.20) * irrev_score +
            weights.get('phase_adjusted_ar1', 0.15) * ar1_score +
            weights.get('entropy_accel', 0.12) * accel_score +
            weights.get('var_lift', 0.12) * var_lift_score
        )

        return max(0, min(1, quality))

    def compute_trigger_quality(
        self,
        returns: pd.Series,
        volumes: pd.Series,
    ) -> float:
        """
        触发质量得分

        检测是否有突破确认信号。
        """
        if len(returns) < 10:
            return 0.5

        # 1. 近期收益动量
        momentum = returns.tail(5).mean()
        momentum_score = min(1.0, max(0, momentum * 100 + 0.5))

        # 2. 成交量放大
        vol_ratio = volumes.tail(5).mean() / volumes.tail(20).mean() if len(volumes) >= 20 else 1.0
        vol_score = min(1.0, max(0, (vol_ratio - 1) * 2 + 0.5))

        # 3. 波动率扩张
        vol_recent = returns.tail(5).std()
        vol_past = returns.tail(20).std()
        vol_expansion = vol_recent / vol_past if vol_past > 0 else 1.0
        expansion_score = min(1.0, max(0, (vol_expansion - 1) * 2 + 0.5))

        # 综合得分
        quality = momentum_score * 0.4 + vol_score * 0.3 + expansion_score * 0.3

        return max(0, min(1, quality))

    # ========== 状态判断 ==========

    def determine_state(
        self,
        path_irrev: float,
        dominant_eig: float,
        perm_entropy: float,
    ) -> StateFlow:
        """
        确定当前状态
        """
        # 检查临界减速
        if np.isfinite(dominant_eig) and abs(dominant_eig) > self.config.dominant_eig_threshold:
            # 临界减速状态
            if np.isfinite(path_irrev) and path_irrev < self.config.path_irrev_low:
                return 'critical_slowing'
            else:
                return 'bifurcation'

        # 检查低熵压缩
        if np.isfinite(path_irrev) and path_irrev < self.config.path_irrev_low:
            if np.isfinite(perm_entropy) and perm_entropy < self.config.perm_entropy_low:
                return 'compression'

        # 检查高熵扩散
        if np.isfinite(path_irrev) and path_irrev > self.config.path_irrev_high:
            return 'diffusion'

        # 检查衰竭
        if np.isfinite(perm_entropy) and perm_entropy > self.config.perm_entropy_high:
            return 'exhaustion'

        # 默认观察态
        return 'observation'

    def check_hard_gates(
        self,
        entropy_quality: float,
        bifurcation_quality: float,
        entropy_percentile: float,
        entropy_gap: float,
    ) -> bool:
        """
        检查硬性门槛条件（对齐基线五门过滤）

        所有条件必须同时满足才能产生 buy 信号。
        """
        if entropy_quality < self.config.hard_gate_entropy_quality_min:
            return False
        if bifurcation_quality < self.config.hard_gate_bifurcation_quality_min:
            return False
        if np.isfinite(entropy_percentile) and entropy_percentile > self.config.hard_gate_entropy_percentile_max:
            return False
        if np.isfinite(entropy_gap) and entropy_gap < self.config.hard_gate_entropy_gap_min:
            return False
        return True

    def determine_signal(
        self,
        state: StateFlow,
        total_score: float,
        gate_open: bool,
        hard_gate_passed: bool = True,
    ) -> Literal['buy', 'hold', 'sell', 'wait']:
        """
        确定交易信号

        除了状态和得分外，还检查硬性门槛。
        """
        if not gate_open:
            return 'wait'

        if not hard_gate_passed:
            # 硬性门槛未通过，不允许 buy，只允许 hold/sell/wait
            if state in ['diffusion', 'exhaustion']:
                return 'sell'
            return 'wait'

        if state == 'compression' and total_score > 0.45:
            return 'buy'

        if state == 'critical_slowing' and total_score > 0.5:
            return 'buy'

        if state == 'bifurcation' and total_score > 0.4:
            return 'hold'

        if state in ['diffusion', 'exhaustion']:
            return 'sell'

        return 'wait'

    # ========== 主评估函数 ==========

    def evaluate(
        self,
        prices: pd.Series,
        volumes: pd.Series,
        dates: pd.Series,
        gate_open: bool = True,
    ) -> StockStateOutput:
        """
        执行个股状态评估

        参数
        ----
        prices : pd.Series
            价格序列
        volumes : pd.Series
            成交量序列
        dates : pd.Series
            日期序列
        gate_open : bool
            市场门控是否打开

        返回
        ----
        StockStateOutput
            状态评估结果
        """
        # 计算收益率
        returns = np.log(prices / prices.shift(1))

        # 计算核心特征
        path_irrev = self.compute_path_irreversibility(returns, self.config.window_medium)
        dominant_eig = self.compute_dominant_eigenvalue(returns, self.config.window_medium)
        perm_entropy = self.compute_permutation_entropy(returns, self.config.window_medium)
        phase_ar1 = self.compute_phase_adjusted_ar1(returns, dates, self.config.window_medium)

        # 计算新增特征（P1）
        entropy_accel = self.compute_entropy_accel(perm_entropy, step=5)
        entropy_gap = self.compute_entropy_gap(returns)
        entropy_percentile = self.compute_entropy_percentile(perm_entropy, window=120)
        var_lift = self.compute_var_lift(returns, self.config.window_short, self.config.window_medium)

        # 获取最新值
        latest_idx = len(prices) - 1

        def _safe_latest(series):
            v = series.iloc[latest_idx]
            return v if np.isfinite(v) else np.nan

        path_irrev_val = _safe_latest(path_irrev)
        dominant_eig_val = _safe_latest(dominant_eig)
        perm_entropy_val = _safe_latest(perm_entropy)
        phase_ar1_val = _safe_latest(phase_ar1)
        entropy_accel_val = _safe_latest(entropy_accel)
        entropy_gap_val = _safe_latest(entropy_gap)
        entropy_percentile_val = _safe_latest(entropy_percentile)
        var_lift_val = _safe_latest(var_lift)

        # 计算质量得分（使用新增特征）
        entropy_quality = self.compute_entropy_quality(
            path_irrev_val, perm_entropy_val, entropy_percentile_val, entropy_gap_val
        )
        bifurcation_quality = self.compute_bifurcation_quality(
            dominant_eig_val, path_irrev_val, phase_ar1_val, entropy_accel_val, var_lift_val
        )
        trigger_quality = self.compute_trigger_quality(returns, volumes)

        entropy_weight = (
            self.layer_config.entropy_quality_weight
            if self.layer_config is not None
            else 0.32
        )
        bifurcation_weight = (
            self.layer_config.bifurcation_quality_weight
            if self.layer_config is not None
            else 0.48
        )
        trigger_weight = (
            self.layer_config.trigger_quality_weight
            if self.layer_config is not None
            else 0.20
        )

        # 综合得分
        total_score = (
            entropy_quality * entropy_weight +
            bifurcation_quality * bifurcation_weight +
            trigger_quality * trigger_weight
        )

        # 归一化权重
        weight_sum = entropy_weight + bifurcation_weight + trigger_weight
        total_score = total_score / weight_sum if weight_sum > 0 else 0.0

        # 确定状态
        state = self.determine_state(path_irrev_val, dominant_eig_val, perm_entropy_val)

        # 检查硬性门槛（P2）
        hard_gate_passed = self.check_hard_gates(
            entropy_quality, bifurcation_quality, entropy_percentile_val, entropy_gap_val
        )

        # 确定信号（传入硬性门槛结果）
        signal = self.determine_signal(state, total_score, gate_open, hard_gate_passed)

        # 状态历史
        state_history = list(path_irrev.tail(10).index)

        return StockStateOutput(
            current_state=state,
            path_irreversibility=path_irrev_val if np.isfinite(path_irrev_val) else 0.0,
            dominant_eigenvalue=dominant_eig_val if np.isfinite(dominant_eig_val) else 0.0,
            permutation_entropy=perm_entropy_val if np.isfinite(perm_entropy_val) else 0.5,
            phase_adjusted_ar1=phase_ar1_val if np.isfinite(phase_ar1_val) else 0.0,
            entropy_accel=entropy_accel_val if np.isfinite(entropy_accel_val) else 0.0,
            entropy_gap=entropy_gap_val if np.isfinite(entropy_gap_val) else 0.0,
            entropy_percentile=entropy_percentile_val if np.isfinite(entropy_percentile_val) else 0.5,
            var_lift=var_lift_val if np.isfinite(var_lift_val) else 0.0,
            entropy_quality=entropy_quality,
            bifurcation_quality=bifurcation_quality,
            trigger_quality=trigger_quality,
            total_score=total_score,
            hard_gate_passed=hard_gate_passed,
            state_history=state_history,
            signal=signal,
        )
