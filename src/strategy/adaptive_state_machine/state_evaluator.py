"""
State Evaluator — 状态判定

用动态权重和阈值判定每只股票的状态, 替代原 signal_detector.py 的硬编码逻辑。
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
    pred_return: float = 0.0      # 模型预测收益率
    pred_up_prob: float = 0.5     # 模型预测上涨概率
    details: dict = field(default_factory=dict)


class StateEvaluator:
    """状态评估器。

    用 AdaptiveConfig 中的动态权重和阈值，
    对全市场股票进行状态判定。
    """

    def __init__(self, config: Optional[AdaptiveConfig] = None):
        self.config = config or AdaptiveConfig()
        # 跟踪处于 hold 状态的股票及其进入时间 {symbol: scan_index}
        self._hold_state: dict[str, int] = {}
        # 当前扫描日期索引 (由 evaluate_all 设置)
        self._current_scan_index: int = 0
        # hold 超时: 超过 N 个扫描日自动释放
        self.hold_max_scans: int = 3

    def evaluate_all(
        self,
        cross_section: pd.DataFrame,
        daily_data: dict[str, tuple[pd.DataFrame, Optional[pd.DataFrame]]],
        config: Optional[AdaptiveConfig] = None,
        model_predictions: Optional[dict[str, dict]] = None,
    ) -> list[StateResult]:
        """
        对全市场股票进行状态判定。

        Args:
            cross_section: 截面快照 (index=symbol, columns=因子)
            daily_data: Factor calculation 输出 {symbol: (df_d, df_w)}
            config: 动态配置 (可选, 默认用 self.config)
            model_predictions: Transformer 模型预测 {symbol: {"pred_return": float, "pred_up_prob": float}}

        Returns:
            所有股票的 StateResult 列表
        """
        cfg = config or self.config
        self._current_scan_index += 1  # 每次调用递增扫描索引
        results = []

        for sym in cross_section.index:
            if sym not in daily_data:
                continue

            df_d, df_w = daily_data[sym]
            pred = model_predictions.get(sym) if model_predictions else None
            result = self._evaluate_single(sym, df_d, df_w, cfg, pred)
            results.append(result)

        # 统计
        state_counts = {}
        for r in results:
            state_counts[r.state] = state_counts.get(r.state, 0) + 1

        logger.info(
            f"State evaluator: evaluated {len(results)} stocks. "
            f"State distribution: {state_counts}"
        )
        return results

    def _evaluate_single(
        self,
        symbol: str,
        df_d: pd.DataFrame,
        df_w: Optional[pd.DataFrame],
        cfg: AdaptiveConfig,
        model_pred: Optional[dict] = None,
    ) -> StateResult:
        """评估单只股票的状态"""
        last = df_d.iloc[-1]
        trade_date = str(last.get("trade_date", ""))

        pred_return = model_pred["pred_return"] if model_pred else 0.0
        pred_up_prob = model_pred["pred_up_prob"] if model_pred else 0.5

        # 1. 计算各状态得分（0~1）
        accum_score, aq = self._check_accumulation(last, cfg)
        breakout_score = self._check_breakout(last, cfg)  # 独立检测，不依赖 accumulation
        collapse_score = self._check_collapse(last, cfg)

        # 2. 分数竞争决定最终状态
        state = self._resolve_state(
            symbol, accum_score, breakout_score, collapse_score, pred_up_prob, cfg,
        )

        # 5. 计算置信度
        bq = 0.0
        if state == StockState.BREAKOUT:
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
            pred_return=round(pred_return, 6),
            pred_up_prob=round(pred_up_prob, 4),
            details=details,
        )

    def _check_accumulation(
        self,
        last_row: pd.Series,
        cfg: AdaptiveConfig,
    ) -> tuple[float, float]:
        """
        判定是否处于 accumulation 状态。

        Returns:
            (accumulation_score, aq_score)
            score = 满足条件数 / 总条件数
        """
        t = cfg.thresholds

        conditions_passed = 0
        total_conditions = 0

        # perm_entropy_m < threshold
        if "perm_entropy_m" in last_row.index and pd.notna(last_row["perm_entropy_m"]):
            total_conditions += 1
            if last_row["perm_entropy_m"] < t.get("perm_entropy_acc", 0.65):
                conditions_passed += 1

        # path_irrev_m > threshold
        if "path_irrev_m" in last_row.index and pd.notna(last_row["path_irrev_m"]):
            total_conditions += 1
            if last_row["path_irrev_m"] > t.get("path_irrev_acc", 0.05):
                conditions_passed += 1

        # mf_flow_imbalance > threshold (可选)
        if "mf_flow_imbalance" in last_row.index and pd.notna(last_row["mf_flow_imbalance"]):
            total_conditions += 1
            if last_row["mf_flow_imbalance"] > t.get("mf_flow_imbalance_min", 0.3):
                conditions_passed += 1

        # mf_big_streak > threshold (可选)
        if "mf_big_streak" in last_row.index and pd.notna(last_row["mf_big_streak"]):
            total_conditions += 1
            if last_row["mf_big_streak"] > t.get("mf_big_streak_min", 3):
                conditions_passed += 1

        if total_conditions < 2:
            return 0.0, 0.0

        score = conditions_passed / total_conditions

        # 计算 AQ 分数
        aq = self._calc_aq(last_row, cfg)

        return score, aq

    def _check_breakout(
        self,
        last_row: pd.Series,
        cfg: AdaptiveConfig,
    ) -> float:
        """
        判定是否处于 breakout 状态。
        独立检测，不依赖 accumulation。

        Returns:
            breakout_score = 满足条件数 / 总条件数
        """
        t = cfg.thresholds

        conditions_passed = 0
        total_conditions = 0

        # dom_eig_m > threshold
        if "dom_eig_m" in last_row.index and pd.notna(last_row["dom_eig_m"]):
            total_conditions += 1
            if last_row["dom_eig_m"] > t.get("dom_eig_breakout", 0.85):
                conditions_passed += 1

        # vol_impulse > threshold
        if "vol_impulse" in last_row.index and pd.notna(last_row["vol_impulse"]):
            total_conditions += 1
            if last_row["vol_impulse"] > t.get("vol_impulse_breakout", 1.8):
                conditions_passed += 1

        # perm_entropy_m < threshold (有序突破)
        if "perm_entropy_m" in last_row.index and pd.notna(last_row["perm_entropy_m"]):
            total_conditions += 1
            if last_row["perm_entropy_m"] < t.get("perm_entropy_breakout_max", 0.75):
                conditions_passed += 1

        # mf_big_momentum > 0 (可选)
        if "mf_big_momentum" in last_row.index and pd.notna(last_row["mf_big_momentum"]):
            total_conditions += 1
            if last_row["mf_big_momentum"] > 0:
                conditions_passed += 1

        if total_conditions < 2:
            return 0.0

        return conditions_passed / total_conditions

    def _check_collapse(
        self,
        last_row: pd.Series,
        cfg: AdaptiveConfig,
    ) -> float:
        """
        判定是否出现 collapse 信号。

        Returns:
            collapse_score = 触发信号数 / 需要数 (capped at 1.0)
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
        return min(1.0, signals / need_n)

    def _resolve_state(
        self,
        symbol: str,
        accum_score: float,
        breakout_score: float,
        collapse_score: float,
        model_up_prob: float,
        cfg: AdaptiveConfig,
    ) -> StockState:
        """解析最终状态 — 分数竞争制。

        规则:
          1. collapse_score >= 1.0 (全部信号触发) → 直接 COLLAPSE
          2. breakout_score 最高且 > 0.5 → BREAKOUT (入场)
          3. 已在持仓中 → 检查超时 → HOLD
          4. accum_score 最高且 > 0.5 → ACCUMULATION
          5. 其他 → IDLE

        模型预测校准:
          - model_up_prob < 0.4: 下调 accum/breakout (模型不看好)
          - model_up_prob < 0.3 + collapse > 0.5: 放宽 collapse 判定
          - model_up_prob > 0.6: 上调 accum/breakout (模型看好)
        """
        scan_idx = self._current_scan_index

        # 模型预测校准
        adj_accum = accum_score
        adj_breakout = breakout_score
        adj_collapse = collapse_score

        if model_up_prob < 0.3 and collapse_score > 0.5:
            adj_collapse = min(1.0, collapse_score * 1.3)
        elif model_up_prob < 0.4:
            adj_accum *= 0.7
            adj_breakout *= 0.7
        elif model_up_prob > 0.6:
            adj_accum = min(1.0, accum_score * 1.1)
            adj_breakout = min(1.0, breakout_score * 1.1)

        # collapse: 全部信号触发时优先级最高，直接退出
        if adj_collapse >= 1.0:
            self._hold_state.pop(symbol, None)
            return StockState.COLLAPSE

        # breakout: 得分最高且超过阈值时入场
        if adj_breakout > 0.5 and adj_breakout >= adj_accum:
            self._hold_state[symbol] = scan_idx
            return StockState.BREAKOUT

        # 已在持仓中 → 检查超时
        if symbol in self._hold_state:
            entry_scan = self._hold_state[symbol]
            if scan_idx - entry_scan >= self.hold_max_scans:
                self._hold_state.pop(symbol)
                return StockState.IDLE
            return StockState.HOLD

        # accumulation: 得分最高且超过阈值时蓄力
        if adj_accum > 0.5:
            return StockState.ACCUMULATION

        return StockState.IDLE

    def _calc_aq(self, last_row: pd.Series, cfg: AdaptiveConfig) -> float:
        """
        计算 Accumulation Quality 分数 [0, 1]。
        当 attention_weights 存在时，用 attention 权重替代固定权重。
        """
        if cfg.attention_weights:
            weights = self._derive_sub_weights(
                cfg.attention_weights, AQ_FACTORS,
            )
        else:
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
        当 attention_weights 存在时，用 attention 权重替代固定权重。
        """
        if cfg.attention_weights:
            weights = self._derive_sub_weights(
                cfg.attention_weights, BQ_FACTORS,
            )
        else:
            weights = cfg.bq_weights
        score = 0.0

        for factor, w in weights.items():
            if factor not in last_row.index or pd.isna(last_row[factor]):
                continue
            val = last_row[factor]
            score += w * self._normalize_factor(factor, val)

        return min(1.0, max(0.0, score))

    @staticmethod
    def _derive_sub_weights(
        attention_weights: dict[str, float],
        sub_factors: list[str],
    ) -> dict[str, float]:
        """从 attention 权重中提取指定因子的权重并归一化。"""
        extracted = {}
        for f in sub_factors:
            if f in attention_weights:
                extracted[f] = attention_weights[f]
            else:
                extracted[f] = 1.0  # fallback: 未知因子给中性权重
        total = sum(extracted.values())
        if total > 0:
            for f in extracted:
                extracted[f] /= total
        return extracted

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
