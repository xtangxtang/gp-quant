"""
Agent 4: Validator

职责:
  1. 回顾 Agent 3 的历史状态判定, 对比后续实际走势
  2. 判断预测是否正确 (accumulation→涨, breakout→突破, collapse→跌)
  3. 生成奖励/惩罚信号 → Agent 2

输入: Agent 3 的历史预测 + 实际价格数据
输出: 每因子的奖励/惩罚信号 dict[str, float]

频率: 每日 (滚动验证)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .config import (
    AdaptiveConfig,
    StockState,
    ALL_FACTORS,
)
from .agent3_state import StateResult

logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """单次状态判定的记录"""
    symbol: str
    trade_date: str
    state: StockState
    confidence: float
    factors: dict  # 当时的因子值快照
    aq_score: float
    bq_score: float


# 各状态的验证周期 (交易日)
VERIFY_HORIZONS = {
    StockState.ACCUMULATION: 10,  # accumulation: 等 10 天看是否涨
    StockState.BREAKOUT: 5,       # breakout: 等 5 天看是否突破
    StockState.HOLD: 10,          # hold: 等 10 天看是否继续涨
    StockState.COLLAPSE: 5,       # collapse: 等 5 天看是否跌
}


class Validator:
    """
    Agent 4: 验证器。

    维护一个预测队列, 到期后验证结果, 生成奖励/惩罚信号。
    """

    def __init__(self, decay_lambda: float = 0.05):
        # 预测队列: {symbol: [PredictionRecord, ...]}
        self.prediction_queue: dict[str, list[PredictionRecord]] = defaultdict(list)
        # 已验证记录
        self.verified_records: list[tuple[PredictionRecord, bool, float]] = []
        # 每因子累计奖励
        self.factor_rewards: dict[str, float] = defaultdict(float)
        # 每因子验证次数
        self.factor_hit_counts: dict[str, int] = defaultdict(int)
        # 奖励衰减系数
        self.decay_lambda = decay_lambda

    def ingest_predictions(self, results: list[StateResult]):
        """
        接收 Agent 3 的最新状态判定, 加入预测队列。

        记录所有有意义状态: accumulation, breakout, hold, collapse
        """
        for r in results:
            if r.state in (StockState.ACCUMULATION, StockState.BREAKOUT,
                           StockState.HOLD, StockState.COLLAPSE):
                record = PredictionRecord(
                    symbol=r.symbol,
                    trade_date=r.trade_date,
                    state=r.state,
                    confidence=r.confidence,
                    factors=r.details,
                    aq_score=r.aq_score,
                    bq_score=r.bq_score,
                )
                self.prediction_queue[r.symbol].append(record)

    def verify(
        self,
        price_series: pd.DataFrame,  # index=trade_date, columns=symbol
        current_date: str,
        available_dates: Optional[list[str]] = None,
    ) -> dict[str, float]:
        """
        执行滚动验证。

        检查队列中到期的预测, 对比实际价格变化,
        生成奖励/惩罚信号。

        Args:
            price_series: 价格矩阵, index=trade_date, columns=symbol
            current_date: 当前日期 (YYYYMMDD)
            available_dates: 交易日期列表 (用于查找未来日期)

        Returns:
            {factor: reward_signal} 传给 Agent 2
        """
        if available_dates is None:
            available_dates = sorted(price_series.index.astype(str).tolist())

        expired = []  # (record, symbol)
        still_pending = []

        for sym, records in self.prediction_queue.items():
            for rec in records:
                horizon = VERIFY_HORIZONS.get(rec.state, 10)
                verify_date = self._find_future_date(
                    rec.trade_date, horizon, available_dates,
                )

                if verify_date and verify_date <= current_date:
                    # 到期, 验证
                    expired.append((rec, sym, verify_date))
                else:
                    still_pending.append(rec)

            self.prediction_queue[sym] = still_pending
            still_pending = []

        logger.info(f"Agent 4: Verifying {len(expired)} expired predictions")

        # 逐条验证
        for rec, sym, verify_date in expired:
            self._verify_single(rec, sym, verify_date, price_series, available_dates)

        # 生成因子奖励信号
        rewards = self._generate_factor_rewards()
        return rewards

    def _find_future_date(
        self,
        base_date: str,
        n_days: int,
        available_dates: list[str],
    ) -> Optional[str]:
        """找到 base_date 之后第 n_days 个交易日的日期"""
        base_str = str(base_date)
        try:
            idx = available_dates.index(base_str)
        except ValueError:
            return None

        target_idx = idx + n_days
        if target_idx >= len(available_dates):
            return None

        return available_dates[target_idx]

    def _verify_single(
        self,
        record: PredictionRecord,
        symbol: str,
        verify_date: str,
        price_series: pd.DataFrame,
        available_dates: list[str],
    ):
        """验证单条预测"""
        try:
            entry_date = record.trade_date
            entry_price = self._get_price(price_series, entry_date, symbol)
            exit_price = self._get_price(price_series, verify_date, symbol)

            if entry_price is None or exit_price is None or entry_price <= 0:
                return

            actual_return = (exit_price - entry_price) / entry_price
            correct = self._is_correct(record.state, actual_return)

            # 计算奖励 (按幅度加权)
            reward_magnitude = abs(actual_return)
            if correct:
                reward = +1.0 * reward_magnitude
            else:
                reward = -1.0 * reward_magnitude

            # 衰减: 越远的预测权重越低
            days_ago = self._days_between(entry_date, record.trade_date, available_dates)
            decay_factor = np.exp(-self.decay_lambda * max(days_ago, 0))
            effective_reward = reward * decay_factor

            # 记录
            self.verified_records.append((record, correct, effective_reward))

            # 更新因子奖励 (按贡献度分配)
            self._update_factor_rewards(record, effective_reward)

            logger.debug(
                f"  {symbol} [{record.state.value}]: "
                f"predicted {entry_date} → verify {verify_date}, "
                f"return={actual_return:+.2%}, correct={correct}, "
                f"reward={effective_reward:+.4f}"
            )

        except Exception as e:
            logger.debug(f"Error verifying {symbol} @ {record.trade_date}: {e}")

    def _get_price(
        self,
        price_series: pd.DataFrame,
        date: str,
        symbol: str,
    ) -> Optional[float]:
        """获取指定日期的收盘价"""
        try:
            price = price_series.loc[str(date), symbol]
            return float(price) if pd.notna(price) else None
        except (KeyError, TypeError):
            return None

    def _is_correct(self, state: StockState, actual_return: float) -> bool:
        """
        判断预测是否正确。

        accumulation → 未来应涨
        breakout → 未来应涨 (突破后继续)
        hold → 未来应涨 (持仓期间盈利)
        collapse → 未来应跌
        """
        if state in (StockState.ACCUMULATION, StockState.BREAKOUT, StockState.HOLD):
            return actual_return > 0
        elif state == StockState.COLLAPSE:
            return actual_return < 0
        return False

    def _days_between(
        self,
        date1: str,
        date2: str,
        available_dates: list[str],
    ) -> int:
        """计算两个日期之间的交易日数"""
        try:
            idx1 = available_dates.index(str(date1))
            idx2 = available_dates.index(str(date2))
            return abs(idx2 - idx1)
        except (ValueError, IndexError):
            return 0

    def _update_factor_rewards(self, record: PredictionRecord, effective_reward: float):
        """
        更新因子累计奖励。

        对判定起关键作用的因子 (贡献度高的) 获得更多奖励/惩罚。
        用因子绝对值作为贡献度代理。
        """
        # 根据状态选择相关因子
        if record.state == StockState.ACCUMULATION:
            relevant_factors = {
                "perm_entropy_m", "path_irrev_m",
                "mf_flow_imbalance", "mf_big_streak",
                "purity_norm", "big_net_ratio_ma",
            }
        elif record.state == StockState.BREAKOUT:
            relevant_factors = {
                "dom_eig_m", "vol_impulse", "perm_entropy_m",
                "path_irrev_m", "coherence_decay_rate",
                "mf_big_momentum", "mf_big_net_ratio",
            }
        elif record.state == StockState.COLLAPSE:
            relevant_factors = {
                "perm_entropy_m", "path_irrev_m",
                "entropy_accel", "purity_norm",
            }
        else:
            relevant_factors = set(record.factors.keys())

        # 计算每个因子的贡献度权重 (用因子绝对值代理)
        contributions = {}
        total = 0.0
        for f in relevant_factors:
            if f in record.factors and record.factors[f] is not None:
                val = abs(float(record.factors[f]))
                contributions[f] = val
                total += val

        if total == 0:
            # 均匀分配
            n = len(relevant_factors)
            if n > 0:
                for f in relevant_factors:
                    self.factor_rewards[f] += effective_reward / n
                    self.factor_hit_counts[f] += 1
            return

        for f, contrib in contributions.items():
            weight = contrib / total
            self.factor_rewards[f] += effective_reward * weight
            self.factor_hit_counts[f] += 1

    def _generate_factor_rewards(self) -> dict[str, float]:
        """
        生成传给 Agent 2 的因子奖励信号。

        归一化: 将累计奖励除以验证次数, 得到平均每因子奖励。
        """
        rewards = {}
        for factor in ALL_FACTORS:
            count = self.factor_hit_counts.get(factor, 0)
            if count > 0:
                rewards[factor] = self.factor_rewards[factor] / count
            else:
                rewards[factor] = 0.0

        # 统计摘要
        positive = sum(1 for v in rewards.values() if v > 0)
        negative = sum(1 for v in rewards.values() if v < 0)
        total_verified = len(self.verified_records)

        logger.info(
            f"Agent 4: Factor rewards generated. "
            f"Verified: {total_verified}, "
            f"Positive factors: {positive}, Negative factors: {negative}"
        )

        return rewards

    def get_performance_summary(self) -> dict:
        """获取验证性能摘要"""
        if not self.verified_records:
            return {"total_verified": 0}

        total = len(self.verified_records)
        correct = sum(1 for _, c, _ in self.verified_records if c)
        wrong = total - correct

        by_state = defaultdict(lambda: {"total": 0, "correct": 0})
        for rec, correct_flag, _ in self.verified_records:
            by_state[rec.state.value]["total"] += 1
            if correct_flag:
                by_state[rec.state.value]["correct"] += 1

        return {
            "total_verified": total,
            "correct": correct,
            "wrong": wrong,
            "accuracy": correct / total if total > 0 else 0.0,
            "by_state": dict(by_state),
        }
