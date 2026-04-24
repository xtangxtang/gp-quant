"""
Agent 3: State Evaluator

职责: 用动态权重和阈值判定每只股票的状态
      替代原 signal_detector.py 的硬编码逻辑

输入: 今日因子 + Agent 2 的权重 + 阈值
输出: 每只股票的状态 (idle/accumulation/breakout/hold/collapse) + 置信度

频率: 每日
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .config import (
    AdaptiveConfig,
    StockState,
    AQ_FACTORS,
    BQ_FACTORS,
)

logger = logging.getLogger(__name__)


@dataclass
class StateResult:
    """单只股票的状态判定结果"""
    symbol: str
    trade_date: str
    state: StockState
    confidence: float       # [0, 1] 置信度
    aq_score: float         # [0, 1] 积累质量
    bq_score: float         # [0, 1] 突破质量
    composite_score: float  # 0.4*AQ + 0.6*BQ
    details: dict = field(default_factory=dict)


class StateEvaluator:
    """
    Agent 3: 状态评估器。

    用 AdaptiveConfig 中的动态权重和阈值，
    对全市场股票进行状态判定。
    """

    def __init__(self, config: Optional[AdaptiveConfig] = None):
        self.config = config or AdaptiveConfig()
        # 跟踪处于 hold 状态的股票
        self._hold_state: set[str] = set()

    def evaluate_all(
        self,
        cross_section: pd.DataFrame,
        daily_data: dict[str, tuple[pd.DataFrame, Optional[pd.DataFrame]]],
        config: Optional[AdaptiveConfig] = None,
    ) -> list[StateResult]:
        """
        对全市场股票进行状态判定。

        Args:
            cross_section: 截面快照 (index=symbol, columns=因子)
            daily_data: Agent 1 的输出 {symbol: (df_d, df_w)}
            config: 动态配置 (可选, 默认用 self.config)

        Returns:
            所有股票的 StateResult 列表
        """
        cfg = config or self.config
        results = []

        for sym in cross_section.index:
            if sym not in daily_data:
                continue

            df_d, df_w = daily_data[sym]
            result = self._evaluate_single(sym, df_d, df_w, cfg)
            results.append(result)

        # 统计
        state_counts = {}
        for r in results:
            state_counts[r.state] = state_counts.get(r.state, 0) + 1

        logger.info(
            f"Agent 3: Evaluated {len(results)} stocks. "
            f"State distribution: {state_counts}"
        )
        return results

    def _evaluate_single(
        self,
        symbol: str,
        df_d: pd.DataFrame,
        df_w: Optional[pd.DataFrame],
        cfg: AdaptiveConfig,
    ) -> StateResult:
        """评估单只股票的状态"""
        last = df_d.iloc[-1]
        trade_date = str(last.get("trade_date", ""))

        # 1. 检测 accumulation
        is_accum, aq = self._check_accumulation(last, cfg)

        # 2. 检测 breakout (需要近期有 accumulation)
        is_breakout = False
        if is_accum:
            is_breakout, _ = self._check_breakout(last, cfg)

        # 3. 检测 collapse (对 hold 状态的股票也检测)
        is_collapse = self._check_collapse(last, cfg)

        # 4. 确定最终状态
        state = self._resolve_state(
            symbol, is_accum, is_breakout, is_collapse, cfg,
        )

        # 5. 计算置信度
        bq = 0.0
        if is_breakout or state == StockState.BREAKOUT:
            bq = self._calc_bq(last, cfg)

        composite = cfg.aq_bq_weight * aq + (1 - cfg.aq_bq_weight) * bq
        confidence = min(1.0, max(0.0, composite))

        # 6. 收集详情
        details = {}
        for col in ["perm_entropy_m", "path_irrev_m", "dom_eig_m", "vol_impulse",
                     "entropy_accel", "coherence_l1", "purity_norm", "coherence_decay_rate",
                     "mf_big_momentum", "mf_big_net_ratio", "mf_flow_imbalance",
                     "mf_big_streak"]:
            if col in last.index and pd.notna(last[col]):
                details[col] = round(float(last[col]), 4)

        return StateResult(
            symbol=symbol,
            trade_date=trade_date,
            state=state,
            confidence=round(confidence, 4),
            aq_score=round(aq, 4),
            bq_score=round(bq, 4),
            composite_score=round(composite, 4),
            details=details,
        )

    def _check_accumulation(
        self,
        last_row: pd.Series,
        cfg: AdaptiveConfig,
    ) -> tuple[bool, float]:
        """
        判定是否处于 accumulation 状态。

        Returns:
            (is_accumulation, aq_score)
        """
        t = cfg.thresholds

        # 核心条件
        conditions = []

        # perm_entropy_m < threshold
        if "perm_entropy_m" in last_row.index and pd.notna(last_row["perm_entropy_m"]):
            conditions.append(last_row["perm_entropy_m"] < t.get("perm_entropy_acc", 0.65))

        # path_irrev_m > threshold
        if "path_irrev_m" in last_row.index and pd.notna(last_row["path_irrev_m"]):
            conditions.append(last_row["path_irrev_m"] > t.get("path_irrev_acc", 0.05))

        # mf_flow_imbalance > threshold (可选)
        if "mf_flow_imbalance" in last_row.index and pd.notna(last_row["mf_flow_imbalance"]):
            conditions.append(last_row["mf_flow_imbalance"] > t.get("mf_flow_imbalance_min", 0.3))

        # mf_big_streak > threshold (可选)
        if "mf_big_streak" in last_row.index and pd.notna(last_row["mf_big_streak"]):
            conditions.append(last_row["mf_big_streak"] > t.get("mf_big_streak_min", 3))

        if len(conditions) < 2:
            return False, 0.0

        # 至少满足 N-1 个条件
        is_accum = sum(conditions) >= max(2, len(conditions) - 1)

        # 计算 AQ 分数
        aq = self._calc_aq(last_row, cfg)

        return is_accum, aq

    def _check_breakout(
        self,
        last_row: pd.Series,
        cfg: AdaptiveConfig,
    ) -> tuple[bool, float]:
        """
        判定是否处于 breakout 状态。

        Returns:
            (is_breakout, bq_score)
        """
        t = cfg.thresholds
        conditions = []

        # dom_eig_m > threshold
        if "dom_eig_m" in last_row.index and pd.notna(last_row["dom_eig_m"]):
            conditions.append(last_row["dom_eig_m"] > t.get("dom_eig_breakout", 0.85))

        # vol_impulse > threshold
        if "vol_impulse" in last_row.index and pd.notna(last_row["vol_impulse"]):
            conditions.append(last_row["vol_impulse"] > t.get("vol_impulse_breakout", 1.8))

        # perm_entropy_m < threshold (有序突破)
        if "perm_entropy_m" in last_row.index and pd.notna(last_row["perm_entropy_m"]):
            conditions.append(last_row["perm_entropy_m"] < t.get("perm_entropy_breakout_max", 0.75))

        # mf_big_momentum > 0 (可选)
        if "mf_big_momentum" in last_row.index and pd.notna(last_row["mf_big_momentum"]):
            conditions.append(last_row["mf_big_momentum"] > 0)

        if len(conditions) < 3:
            return False, 0.0

        is_breakout = sum(conditions) >= len(conditions) - 1
        bq = self._calc_bq(last_row, cfg)

        return is_breakout, bq

    def _check_collapse(
        self,
        last_row: pd.Series,
        cfg: AdaptiveConfig,
    ) -> bool:
        """
        判定是否出现 collapse 信号。

        5 个信号中需要 N 个同时触发。
        """
        t = cfg.thresholds
        signals = 0

        # Signal 1: perm_entropy_m > threshold
        if "perm_entropy_m" in last_row.index and pd.notna(last_row["perm_entropy_m"]):
            if last_row["perm_entropy_m"] > t.get("perm_entropy_collapse", 0.90):
                signals += 1

        # Signal 2: path_irrev_m < threshold
        if "path_irrev_m" in last_row.index and pd.notna(last_row["path_irrev_m"]):
            if last_row["path_irrev_m"] < t.get("path_irrev_collapse", 0.01):
                signals += 1

        # Signal 3: entropy_accel > threshold
        if "entropy_accel" in last_row.index and pd.notna(last_row["entropy_accel"]):
            if last_row["entropy_accel"] > t.get("entropy_accel_collapse", 0.05):
                signals += 1

        # Signal 4: purity_norm < threshold
        if "purity_norm" in last_row.index and pd.notna(last_row["purity_norm"]):
            if last_row["purity_norm"] < t.get("purity_collapse_max", 0.3):
                signals += 1

        need_n = int(t.get("collapse_need_n", 3))
        return signals >= need_n

    def _resolve_state(
        self,
        symbol: str,
        is_accum: bool,
        is_breakout: bool,
        is_collapse: bool,
        cfg: AdaptiveConfig,
    ) -> StockState:
        """解析最终状态

        状态优先级:
          collapse (最高) → breakout → hold → accumulation → idle
        """
        # collapse 优先级最高 — 触发则退出
        if is_collapse:
            self._hold_state.discard(symbol)
            return StockState.COLLAPSE

        # breakout → 新入场信号
        if is_breakout:
            self._hold_state.add(symbol)
            return StockState.BREAKOUT

        # 已在持仓中 → 维持 hold，不要求 accumulation 持续
        if symbol in self._hold_state:
            return StockState.HOLD

        # accumulation — 未入场的蓄力状态
        if is_accum:
            return StockState.ACCUMULATION

        return StockState.IDLE

    def _calc_aq(self, last_row: pd.Series, cfg: AdaptiveConfig) -> float:
        """
        计算 Accumulation Quality 分数 [0, 1]。
        使用动态权重。
        """
        weights = cfg.aq_weights
        score = 0.0

        for factor, w in weights.items():
            if factor not in last_row.index or pd.isna(last_row[factor]):
                continue
            val = last_row[factor]
            score += w * self._normalize_factor(factor, val)

        return min(1.0, max(0.0, score))

    def _calc_bq(self, last_row: pd.Series, cfg: AdaptiveConfig) -> float:
        """
        计算 Breakout Quality 分数 [0, 1]。
        使用动态权重。
        """
        weights = cfg.bq_weights
        score = 0.0

        for factor, w in weights.items():
            if factor not in last_row.index or pd.isna(last_row[factor]):
                continue
            val = last_row[factor]
            score += w * self._normalize_factor(factor, val)

        return min(1.0, max(0.0, score))

    @staticmethod
    def _normalize_factor(factor: str, value: float) -> float:
        """
        将因子值归一化到 [0, 1] 范围。
        正向因子: 越大越好 → 直接归一化
        逆向因子: 越小越好 → 1 - 归一化
        """
        # 逆向因子 (越低越好)
        reverse_factors = {
            "perm_entropy_m": (0.3, 1.0),
            "turnover_entropy_m": (0.0, 1.0),
            "von_neumann_entropy": (0.0, 1.0),
        }

        # 正向因子 (越高越好)
        positive_factors = {
            "path_irrev_m": (0.0, 0.5),
            "dom_eig_m": (0.5, 1.0),
            "vol_impulse": (1.0, 5.0),
            "purity_norm": (0.0, 1.0),
            "coherence_l1": (0.0, 1.0),
            "mf_big_momentum": (-2.0, 2.0),
            "mf_big_net_ratio": (-0.3, 0.3),
            "mf_flow_imbalance": (-1.0, 2.0),
            "mf_big_streak": (0.0, 10.0),
            "coherence_decay_rate": (-0.05, 0.05),
            "big_net_ratio_ma": (-0.1, 0.1),
        }

        if factor in reverse_factors:
            lo, hi = reverse_factors[factor]
            normalized = (value - lo) / (hi - lo) if hi > lo else 0.5
            return 1.0 - normalized.clip(0, 1)
        elif factor in positive_factors:
            lo, hi = positive_factors[factor]
            normalized = (value - lo) / (hi - lo) if hi > lo else 0.5
            # coherence_decay_rate: 负值越大(越负)越好
            if factor == "coherence_decay_rate":
                return (-value / 0.05).clip(0, 1)
            return normalized.clip(0, 1)

        return 0.5  # 未知因子, 中性

    def get_signals_by_state(
        self,
        results: list[StateResult],
        target_state: Optional[StockState] = None,
        min_confidence: float = 0.0,
    ) -> list[StateResult]:
        """按状态筛选信号"""
        filtered = [r for r in results if r.confidence >= min_confidence]
        if target_state:
            filtered = [r for r in filtered if r.state == target_state]
        return filtered
