"""
日线熵计算模块 (Daily Entropy)

从日线数据计算跨天维度的熵特征：
- 排列熵（20日/60日）  → 价格有序化程度
- 路径不可逆性          → 时间方向性
- 主导特征值            → 临界减速
- 多尺度熵差            → 短期压缩有效性
- 熵百分位              → 历史排名

数据来源：gp-data/tushare-daily-full/{stock_code}.csv
"""

import math
import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass

from .config import DailyEntropyConfig


@dataclass
class DailyEntropyResult:
    """日线熵计算结果（最新一天的快照 + 序列）"""

    # 最新值快照
    perm_entropy_20: float         # 20 日排列熵
    perm_entropy_60: float         # 60 日排列熵
    entropy_gap: float             # 多尺度熵差（60 - 20）
    entropy_percentile: float      # 排列熵 120 日百分位
    path_irreversibility: float    # 路径不可逆性
    dominant_eigenvalue: float     # 主导特征值
    var_lift: float                # 方差抬升（10/20）

    # 日线状态诊断
    is_compressed: bool            # 是否处于压缩态
    compression_days: int          # 连续压缩天数

    # ---- 卖出相关特征 ----
    entropy_velocity_5: float = 0.0     # PE20 近 5 日变化率（正=熵膨胀）
    path_irrev_velocity_5: float = 0.0  # 路径不可逆性近 5 日变化率
    entropy_percentile_prev5: float = 0.5  # 5 天前的百分位（用于加速度判断）

    # 序列（供融合模块做趋势分析）
    perm_entropy_series: Optional[pd.Series] = None
    path_irrev_series: Optional[pd.Series] = None
    entropy_percentile_series: Optional[pd.Series] = None


class DailyEntropy:
    """
    日线熵计算器。

    从日线 close/vol 序列计算跨天维度的熵特征。
    算法与 StockState / IntradayEntropy 保持一致。
    """

    def __init__(self, config: Optional[DailyEntropyConfig] = None):
        self.config = config or DailyEntropyConfig()

    # ================================================================
    #  核心计算（与 StockState 算法一致）
    # ================================================================

    def _permutation_entropy(self, values: np.ndarray, order: int = 3) -> float:
        if len(values) < order + 2:
            return np.nan
        counts = {}
        for i in range(len(values) - order + 1):
            pattern = tuple(np.argsort(values[i:i + order], kind='mergesort'))
            counts[pattern] = counts.get(pattern, 0) + 1
        if not counts:
            return np.nan
        freq = np.array(list(counts.values()), dtype=np.float64)
        prob = freq / freq.sum()
        entropy = -np.sum(prob * np.log(prob))
        normalizer = np.log(math.factorial(order))
        if normalizer <= 0:
            return np.nan
        return float(entropy / normalizer)

    def _path_irreversibility(self, returns: np.ndarray, threshold_sigma: float = 0.5) -> float:
        if len(returns) < 15:
            return np.nan
        sigma = np.std(returns)
        if sigma < 1e-10:
            return 0.0
        threshold = threshold_sigma * sigma
        states = np.zeros(len(returns), dtype=np.int64)
        states[returns < -threshold] = -1
        states[returns > threshold] = 1
        n_states = 3
        counts = np.zeros((n_states, n_states), dtype=np.float64)
        for t in range(len(states) - 1):
            i, j = int(states[t] + 1), int(states[t + 1] + 1)
            if 0 <= i < n_states and 0 <= j < n_states:
                counts[i, j] += 1.0
        total = counts.sum()
        if total < 10:
            return np.nan
        forward = counts / total
        backward = counts.T / total
        mask = (forward > 1e-10) & (backward > 1e-10)
        if not np.any(mask):
            return 0.0
        kl_div = np.sum(forward[mask] * np.log(forward[mask] / backward[mask]))
        return max(0.0, float(kl_div))

    def _dominant_eigenvalue(self, returns: np.ndarray, order: int = 2) -> float:
        values = returns[np.isfinite(returns)]
        if len(values) < max(12, order + 6):
            return np.nan
        centered = values - np.mean(values)
        if np.std(centered) <= 1e-12:
            return np.nan
        acov = []
        for lag in range(order + 1):
            left = centered[:len(centered) - lag]
            right = centered[lag:]
            if len(right) == 0:
                return np.nan
            acov.append(np.dot(left, right) / len(right))
        system = np.array([
            [acov[abs(i - j)] for j in range(order)]
            for i in range(order)
        ])
        rhs = np.array(acov[1:order + 1])
        try:
            phi = np.linalg.solve(system + np.eye(order) * 1e-8, rhs)
        except np.linalg.LinAlgError:
            return np.nan
        companion = np.zeros((order, order))
        companion[0, :] = phi
        for i in range(1, order):
            companion[i, i - 1] = 1.0
        eigs = np.linalg.eigvals(companion)
        dominant = eigs[np.argmax(np.abs(eigs))]
        return float(np.real(dominant))

    # ================================================================
    #  滚动计算
    # ================================================================

    def _rolling_perm_entropy(self, returns: pd.Series, window: int) -> pd.Series:
        order = self.config.perm_entropy_order

        def _calc(vals):
            return self._permutation_entropy(vals, order)

        return returns.rolling(window=window, min_periods=max(10, window // 2)).apply(
            _calc, raw=True
        )

    def _rolling_path_irrev(self, returns: pd.Series, window: int) -> pd.Series:
        sigma_mult = self.config.irrev_threshold_sigma

        def _calc(vals):
            return self._path_irreversibility(vals, sigma_mult)

        return returns.rolling(window=window, min_periods=max(10, window // 2)).apply(
            _calc, raw=True
        )

    def _rolling_dominant_eig(self, returns: pd.Series, window: int) -> pd.Series:
        order = self.config.ar_order

        def _calc(vals):
            return self._dominant_eigenvalue(vals, order)

        return returns.rolling(window=window, min_periods=max(10, window // 2)).apply(
            _calc, raw=True
        )

    # ================================================================
    #  主计算入口
    # ================================================================

    def compute(self, prices: pd.Series, volumes: Optional[pd.Series] = None) -> Optional[DailyEntropyResult]:
        """
        从日线价格序列计算熵特征。

        参数
        ----
        prices : pd.Series
            日线收盘价（至少 60 天）
        volumes : pd.Series, optional
            日成交量（当前未使用，预留接口）

        返回
        ----
        DailyEntropyResult 或 None
        """
        if len(prices) < self.config.perm_entropy_long_window + 10:
            return None

        returns = np.log(prices / prices.shift(1)).dropna()
        if len(returns) < self.config.perm_entropy_long_window:
            return None

        cfg = self.config

        # 排列熵
        pe_20 = self._rolling_perm_entropy(returns, cfg.perm_entropy_window)
        pe_60 = self._rolling_perm_entropy(returns, cfg.perm_entropy_long_window)

        # 路径不可逆性
        pi_20 = self._rolling_path_irrev(returns, cfg.path_irrev_window)

        # 主导特征值
        de_20 = self._rolling_dominant_eig(returns, cfg.dominant_eig_window)

        # 方差抬升
        var_short = returns.rolling(window=cfg.var_lift_short, min_periods=5).var(ddof=0)
        var_long = returns.rolling(window=cfg.var_lift_long, min_periods=10).var(ddof=0)
        vl = var_short / var_long.replace(0.0, np.nan) - 1.0

        # 百分位排名
        pctile_window = cfg.percentile_window

        def _percentile_of_last(values):
            finite = values[np.isfinite(values)]
            if len(finite) < 8:
                return np.nan
            last = finite[-1]
            return float(np.sum(finite < last)) / len(finite)

        ep = pe_20.rolling(window=pctile_window, min_periods=max(8, pctile_window // 3)).apply(
            _percentile_of_last, raw=True
        )

        # 提取最新值
        def _last_finite(series):
            v = series.iloc[-1] if len(series) > 0 else np.nan
            return v if np.isfinite(v) else np.nan

        pe_20_val = _last_finite(pe_20)
        pe_60_val = _last_finite(pe_60)
        entropy_gap = (pe_60_val - pe_20_val) if np.isfinite(pe_60_val) and np.isfinite(pe_20_val) else np.nan
        ep_val = _last_finite(ep)
        pi_val = _last_finite(pi_20)
        de_val = _last_finite(de_20)
        vl_val = _last_finite(vl)

        # 压缩态判断
        is_compressed = (
            np.isfinite(ep_val) and ep_val < 0.40
            and np.isfinite(pe_20_val) and pe_20_val < 0.85
        )

        # 连续压缩天数
        compression_days = 0
        if is_compressed and len(ep) > 1:
            for i in range(len(ep) - 1, -1, -1):
                v = ep.iloc[i]
                if np.isfinite(v) and v < 0.40:
                    compression_days += 1
                else:
                    break

        # ---- 卖出相关：变化率 ----
        # PE20 近 5 日变化率
        if len(pe_20) >= 6:
            pe_5ago = pe_20.iloc[-6]
            entropy_velocity_5 = (pe_20_val - pe_5ago) if np.isfinite(pe_5ago) and np.isfinite(pe_20_val) else 0.0
        else:
            entropy_velocity_5 = 0.0

        # 路径不可逆性近 5 日变化率
        if len(pi_20) >= 6:
            pi_5ago = pi_20.iloc[-6]
            pi_velocity_5 = (pi_val - pi_5ago) if np.isfinite(pi_5ago) and np.isfinite(pi_val) else 0.0
        else:
            pi_velocity_5 = 0.0

        # 5 天前百分位
        ep_prev5 = ep.iloc[-6] if len(ep) >= 6 and np.isfinite(ep.iloc[-6]) else 0.5

        return DailyEntropyResult(
            perm_entropy_20=pe_20_val,
            perm_entropy_60=pe_60_val,
            entropy_gap=entropy_gap if np.isfinite(entropy_gap) else 0.0,
            entropy_percentile=ep_val if np.isfinite(ep_val) else 0.5,
            path_irreversibility=pi_val if np.isfinite(pi_val) else 0.0,
            dominant_eigenvalue=de_val if np.isfinite(de_val) else 0.0,
            var_lift=vl_val if np.isfinite(vl_val) else 0.0,
            is_compressed=is_compressed,
            compression_days=compression_days,
            entropy_velocity_5=entropy_velocity_5 if np.isfinite(entropy_velocity_5) else 0.0,
            path_irrev_velocity_5=pi_velocity_5 if np.isfinite(pi_velocity_5) else 0.0,
            entropy_percentile_prev5=ep_prev5 if np.isfinite(ep_prev5) else 0.5,
            perm_entropy_series=pe_20,
            path_irrev_series=pi_20,
            entropy_percentile_series=ep,
        )
