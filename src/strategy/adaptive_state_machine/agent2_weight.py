"""
Agent 2: Weight Learner

职责:
  1. 截面 IC 计算: IC(factor, future_return) → 因子权重
  2. 阈值搜索: 在回溯窗口内搜索最优阈值组合
  3. 权重更新: 结合 Agent 4 的奖励/惩罚信号调整权重

输入: 过去 N 天的因子矩阵 + 后续实际收益 + Agent 4 奖励信号
输出: 因子权重 + 各状态判定最优阈值

频率: 每日
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import (
    AdaptiveConfig,
    ALL_FACTORS,
    AQ_FACTORS,
    BQ_FACTORS,
    THRESHOLD_SEARCH_RANGES,
    DEFAULT_THRESHOLDS,
)

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════
# IC 计算
# ═════════════════════════════════════════════════════════

def compute_factor_ic(
    feature_matrix: pd.DataFrame,
    forward_returns: pd.Series,
    factors: Optional[list[str]] = None,
) -> dict[str, float]:
    """
    计算每个因子与未来收益率的 Spearman 秩相关系数 (IC)。

    Args:
        feature_matrix: rows=symbol×date, columns=factors
        forward_returns: 对应的未来 N 天收益率
        factors: 要计算的因子列表, 默认 ALL_FACTORS

    Returns:
        {factor: ic_value}
    """
    factors = factors or ALL_FACTORS
    ics = {}

    for factor in factors:
        if factor not in feature_matrix.columns:
            ics[factor] = 0.0
            continue

        vals = feature_matrix[factor]
        valid = vals.notna() & forward_returns.notna() & np.isfinite(forward_returns)

        if valid.sum() < 30:
            ics[factor] = 0.0
            continue

        corr, _ = spearmanr(vals[valid], forward_returns[valid])
        ics[factor] = corr if np.isfinite(corr) else 0.0

    return ics


# ═════════════════════════════════════════════════════════
# 阈值搜索
# ═════════════════════════════════════════════════════════

def search_optimal_thresholds(
    feature_matrix: pd.DataFrame,
    forward_returns: pd.Series,
    current_thresholds: dict,
    n_grid: int = 5,
) -> dict:
    """
    在回溯窗口内搜索最优阈值组合。

    对每个阈值参数, 在搜索范围内做网格搜索, 目标函数:
      最大化 "信号方向正确率 × 平均收益率"

    Args:
        feature_matrix: rows=symbol×date, columns=factors
        forward_returns: 对应的未来 N 天收益率
        current_thresholds: 当前阈值
        n_grid: 每个参数的网格点数

    Returns:
        最优阈值 dict
    """
    best_thresholds = dict(current_thresholds)
    best_score = -np.inf

    # 逐个优化每个阈值 (坐标下降, 避免组合爆炸)
    for key, (lo, hi) in THRESHOLD_SEARCH_RANGES.items():
        if key not in current_thresholds:
            continue

        current_val = current_thresholds[key]
        # 在当前值附近搜索
        search_lo = max(lo, current_val * 0.85)
        search_hi = min(hi, current_val * 1.15)

        if search_hi <= search_lo:
            continue

        if isinstance(current_val, int):
            candidates = list(range(int(search_lo), int(search_hi) + 1))
        else:
            candidates = np.linspace(search_lo, search_hi, n_grid)

        best_local_score = -np.inf
        best_local_val = current_val

        for val in candidates:
            score = _evaluate_threshold(
                feature_matrix, forward_returns, key, val,
                {**current_thresholds, key: val},
            )
            if score > best_local_score:
                best_local_score = score
                best_local_val = val

        if best_local_score > best_score:
            best_score = best_local_score
            best_thresholds[key] = best_local_val

    # 整数阈值取整
    for key in ("accum_min_days", "collapse_need_n"):
        if key in best_thresholds:
            best_thresholds[key] = int(round(best_thresholds[key]))

    return best_thresholds


def _evaluate_threshold(
    feature_matrix: pd.DataFrame,
    forward_returns: pd.Series,
    key: str,
    value: float,
    thresholds: dict,
) -> float:
    """
    评估单个阈值的得分。

    根据阈值类型选择对应的评估逻辑:
      - accumulation 类: 信号 vs 未来正收益
      - breakout 类: 信号 vs 未来正收益
      - collapse 类: 信号 vs 未来负收益
    """
    if "accum" in key or key == "perm_entropy_acc":
        # accumulation 阈值: 越低越容易触发
        return _eval_accumulation_threshold(feature_matrix, forward_returns, key, value)
    elif "breakout" in key or "dom_eig" in key or "vol_impulse" in key:
        return _eval_breakout_threshold(feature_matrix, forward_returns, key, value)
    elif "collapse" in key:
        return _eval_collapse_threshold(feature_matrix, forward_returns, key, value)
    else:
        return 0.0


def _eval_accumulation_threshold(
    feature_matrix: pd.DataFrame,
    forward_returns: pd.Series,
    key: str,
    value: float,
) -> float:
    """评估 accumulation 阈值的得分"""
    if "perm_entropy" in key:
        factor = "perm_entropy_m"
        direction = "below"
    elif "path_irrev" in key:
        factor = "path_irrev_m"
        direction = "above"
    elif "accum_min_days" in key:
        # 简化: 用 perm_entropy_m 持续低于 0.65 的天数代理
        return 0.0  # 需要时序数据, 简化跳过
    else:
        return 0.0

    if factor not in feature_matrix.columns:
        return 0.0

    vals = feature_matrix[factor]
    if direction == "below":
        signals = vals < value
    else:
        signals = vals > value

    return _score_signals(signals, forward_returns, direction="positive")


def _eval_breakout_threshold(
    feature_matrix: pd.DataFrame,
    forward_returns: pd.Series,
    key: str,
    value: float,
) -> float:
    """评估 breakout 阈值的得分"""
    if "dom_eig" in key:
        factor = "dom_eig_m"
        direction = "above"
    elif "vol_impulse" in key:
        factor = "vol_impulse"
        direction = "above"
    elif "perm_entropy" in key:
        factor = "perm_entropy_m"
        direction = "below"
    else:
        return 0.0

    if factor not in feature_matrix.columns:
        return 0.0

    vals = feature_matrix[factor]
    signals = vals > value if direction == "above" else vals < value
    return _score_signals(signals, forward_returns, direction="positive")


def _eval_collapse_threshold(
    feature_matrix: pd.DataFrame,
    forward_returns: pd.Series,
    key: str,
    value: float,
) -> float:
    """评估 collapse 阈值的得分 — 信号正确时应预测负收益"""
    if "perm_entropy" in key:
        factor = "perm_entropy_m"
        direction = "above"
    elif "path_irrev" in key:
        factor = "path_irrev_m"
        direction = "below"
    elif "entropy_accel" in key:
        factor = "entropy_accel"
        direction = "above"
    elif "purity" in key:
        factor = "purity_norm"
        direction = "below"
    elif "collapse_need_n" in key:
        return 0.0  # 需要多信号组合, 简化跳过
    else:
        return 0.0

    if factor not in feature_matrix.columns:
        return 0.0

    vals = feature_matrix[factor]
    signals = vals > value if direction == "above" else vals < value
    return _score_signals(signals, forward_returns, direction="negative")


def _score_signals(
    signals: pd.Series,
    forward_returns: pd.Series,
    direction: str = "positive",
) -> float:
    """
    信号评分: 正确率 × 平均收益率幅度

    direction="positive": 信号出现后应涨
    direction="negative": 信号出现后应跌
    """
    # 对齐 index
    common = signals.index.intersection(forward_returns.index)
    if len(common) < 5:
        return -0.5

    signals = signals.reindex(common)
    forward_returns = forward_returns.reindex(common)

    if signals.sum() == 0:
        return -1.0  # 无信号, 惩罚

    matched_returns = forward_returns[signals]
    if len(matched_returns) < 5:
        return -0.5

    if direction == "positive":
        win_rate = (matched_returns > 0).mean()
        avg_return = matched_returns.mean()
    else:
        win_rate = (matched_returns < 0).mean()
        avg_return = -matched_returns.mean()  # 翻转

    return win_rate * max(avg_return, 0.01)  # 最低 1% 基准


# ═════════════════════════════════════════════════════════
# Weight Learner Agent
# ═════════════════════════════════════════════════════════

class WeightLearner:
    """
    Agent 2: 权重学习器。

    核心逻辑:
      1. 计算每个因子的 IC (截面 Spearman 秩相关)
      2. 结合 Agent 4 的 reward/penalty 更新因子权重
      3. 搜索最优阈值
      4. 更新 AQ/BQ 内部权重
    """

    def __init__(
        self,
        lookback_days: int = 60,
        forward_window: int = 10,
    ):
        self.lookback_days = lookback_days
        self.forward_window = forward_window  # 未来 N 天收益率作为目标

    def update(
        self,
        feature_matrix: pd.DataFrame,
        price_series: pd.DataFrame,  # index=trade_date, columns=symbol
        config: AdaptiveConfig,
        scan_date: str = "",
        agent4_rewards: Optional[dict[str, float]] = None,
    ) -> AdaptiveConfig:
        """
        执行一次权重更新。

        Args:
            feature_matrix: rows=股票 (截面快照), columns=因子
            price_series: index=trade_date (字符串), columns=股票收盘价
            config: 当前配置
            scan_date: 当前扫描日期 (YYYYMMDD)
            agent4_rewards: Agent 4 的奖励信号 {factor: reward}

        Returns:
            更新后的 AdaptiveConfig
        """
        # 1. 计算 forward returns (用价格矩阵直接查)
        forward_returns = self._compute_forward_returns_simple(
            feature_matrix, price_series, scan_date, self.forward_window,
        )

        if forward_returns.empty or forward_returns.dropna().empty:
            logger.warning("Agent 2: No valid forward returns, skipping update")
            return config

        logger.info(
            f"Agent 2: {forward_returns.dropna().shape[0]} valid forward returns, "
            f"mean={forward_returns.mean():.4f}"
        )

        # 2. 计算 IC
        ics = compute_factor_ic(feature_matrix, forward_returns)

        # 3. 更新因子权重 (带 Agent 4 奖励)
        self._update_factor_weights(config, ics, agent4_rewards)

        # 4. 搜索最优阈值
        new_thresholds = search_optimal_thresholds(
            feature_matrix, forward_returns, config.thresholds,
        )

        # 5. 平滑过渡
        old_thresholds = dict(config.thresholds)
        config.thresholds = new_thresholds
        config.smooth_update(
            AdaptiveConfig(thresholds=old_thresholds),
            alpha=0.2,
        )
        config.clamp_thresholds()

        # 6. 更新 AQ/BQ 内部权重
        self._update_aq_bq_weights(config, ics)

        # 7. 更新元数据
        config.version += 1
        config.update_count += 1

        logger.info(
            f"Agent 2: Config updated (v{config.version}, "
            f"lr={config.learning_rate:.3f}, "
            f"top_ic_factors={self._top_ic_factors(ics)}"
        )

        return config

    def _compute_forward_returns_simple(
        self,
        feature_matrix: pd.DataFrame,
        price_series: pd.DataFrame,
        scan_date: str,
        n_days: int,
    ) -> pd.Series:
        """
        计算未来 N 天收益率。

        直接用价格矩阵: ret = price(t+n_days) / price(t) - 1

        feature_matrix 必须有 'symbol' 列。
        """
        trade_dates = sorted(price_series.index.astype(str).tolist())
        try:
            idx = trade_dates.index(str(scan_date))
        except ValueError:
            return pd.Series(dtype=float)

        future_idx = idx + n_days
        if future_idx >= len(trade_dates):
            return pd.Series(dtype=float)

        future_dt = trade_dates[future_idx]

        returns = []
        for _, row in feature_matrix.iterrows():
            sym = row.get("symbol")
            if sym is None or sym not in price_series.columns:
                continue
            try:
                p_now = price_series.at[str(scan_date), sym]
                p_future = price_series.at[str(future_dt), sym]
                if pd.notna(p_now) and pd.notna(p_future) and float(p_now) > 0:
                    ret = (float(p_future) - float(p_now)) / float(p_now)
                    returns.append((sym, ret))
                else:
                    returns.append((sym, np.nan))
            except (KeyError, TypeError):
                returns.append((sym, np.nan))

        syms, vals = zip(*returns) if returns else ([], [])
        return pd.Series(vals, index=syms)

    def _update_factor_weights(
        self,
        config: AdaptiveConfig,
        ics: dict[str, float],
        agent4_rewards: Optional[dict[str, float]],
    ):
        """
        更新因子权重: w += lr * IC * (1 + reward)
        """
        lr = config.learning_rate
        agent4_rewards = agent4_rewards or {}

        for factor in ALL_FACTORS:
            ic = ics.get(factor, 0.0)
            reward = agent4_rewards.get(factor, 0.0)

            # 权重更新: IC 驱动 + Agent 4 奖励/惩罚
            delta = lr * ic * (1.0 + reward)
            current = config.factor_weights.get(factor, 1.0)
            config.factor_weights[factor] = current + delta

        # 归一化: 保持权重总和不变
        total = sum(config.factor_weights.values())
        if total > 0:
            target_sum = len(config.factor_weights)
            for factor in config.factor_weights:
                config.factor_weights[factor] *= target_sum / total

    def _update_aq_bq_weights(self, config: AdaptiveConfig, ics: dict[str, float]):
        """
        根据 IC 动态更新 AQ/BQ 内部权重。

        IC 高的因子获得更多权重。
        """
        # 更新 AQ 权重
        self._update_sub_weights(config, config.aq_weights, AQ_FACTORS, ics)
        # 更新 BQ 权重
        self._update_sub_weights(config, config.bq_weights, BQ_FACTORS, ics)

    @staticmethod
    def _update_sub_weights(
        config: AdaptiveConfig,
        weights: dict,
        factors: list[str],
        ics: dict[str, float],
    ):
        """更新一组子权重 (AQ 或 BQ)"""
        ic_values = {f: abs(ics.get(f, 0.0)) + 0.01 for f in factors if f in weights}
        if not ic_values:
            return

        total_ic = sum(ic_values.values())
        for f in weights:
            if f in ic_values:
                weights[f] = ic_values[f] / total_ic

    @staticmethod
    def _top_ic_factors(ics: dict[str, float], n: int = 5) -> str:
        """返回 IC 最高的 N 个因子"""
        sorted_ics = sorted(ics.items(), key=lambda x: abs(x[1]), reverse=True)[:n]
        return ", ".join(f"{f}={ic:+.4f}" for f, ic in sorted_ics)
