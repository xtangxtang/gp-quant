"""
双熵融合信号模块 (Fusion Signal)

核心逻辑：
  日线低熵（压缩态） + 日内熵突然下降（成交集中化）
  = 主力在压缩态中悄然建仓

信号生成流程：
  1. 日线压缩态评分 → 价格是否被"压缩"
  2. 日内集中化评分 → 微观结构是否有异常集中
  3. 方向性信号评分 → 资金是否有方向性驱动
  4. 量能形态评分   → 量能是否配合
  5. 综合打分 → buy / watch / skip
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass, field

from .config import FusionSignalConfig
from .daily_entropy import DailyEntropyResult
from .intraday_entropy import IntradayEntropyResult


Signal = Literal['buy', 'watch', 'skip']


@dataclass
class FusionResult:
    """双熵融合信号结果"""

    stock_code: str
    trade_date: str

    # 四个子得分
    daily_compression_score: float     # 日线压缩态质量
    intraday_concentration_score: float  # 日内集中化程度
    direction_score: float              # 方向性信号
    volume_pattern_score: float         # 量能形态

    # 综合
    total_score: float
    signal: Signal
    reason: str

    # 关键特征值（供调试/展示）
    daily_perm_entropy_20: float
    daily_entropy_percentile: float
    daily_entropy_gap: float
    daily_compression_days: int
    intraday_turnover_entropy: float
    intraday_turnover_entropy_drop: float  # 相对近期均值的降幅
    intraday_path_irrev: float
    intraday_volume_concentration: float
    intraday_state: str

    def to_dict(self) -> Dict:
        return {
            'stock_code': self.stock_code,
            'trade_date': self.trade_date,
            'daily_compression_score': round(self.daily_compression_score, 4),
            'intraday_concentration_score': round(self.intraday_concentration_score, 4),
            'direction_score': round(self.direction_score, 4),
            'volume_pattern_score': round(self.volume_pattern_score, 4),
            'total_score': round(self.total_score, 4),
            'signal': self.signal,
            'reason': self.reason,
            'daily_perm_entropy_20': round(self.daily_perm_entropy_20, 4),
            'daily_entropy_percentile': round(self.daily_entropy_percentile, 4),
            'daily_entropy_gap': round(self.daily_entropy_gap, 4),
            'daily_compression_days': self.daily_compression_days,
            'intraday_turnover_entropy': round(self.intraday_turnover_entropy, 4),
            'intraday_turnover_entropy_drop': round(self.intraday_turnover_entropy_drop, 4),
            'intraday_path_irrev': round(self.intraday_path_irrev, 4),
            'intraday_volume_concentration': round(self.intraday_volume_concentration, 4),
            'intraday_state': self.intraday_state,
        }


class FusionSignal:
    """
    双熵融合信号生成器。

    输入日线熵特征（DailyEntropyResult）和日内熵历史（IntradayEntropyResult 列表），
    输出融合信号。
    """

    def __init__(self, config: Optional[FusionSignalConfig] = None):
        self.config = config or FusionSignalConfig()

    # ================================================================
    #  子得分计算
    # ================================================================

    def _score_daily_compression(self, daily: DailyEntropyResult) -> float:
        """
        日线压缩态评分 [0, 1]

        高分条件：
        - 排列熵百分位低（处于历史低位）
        - 排列熵绝对值低（价格确实有序）
        - 多尺度熵差为正（短期比长期更有序）
        - 连续压缩天数多
        """
        scores = []

        # 百分位得分：越低越好 → 线性映射 [0, 0.5] → [1, 0]
        ep = daily.entropy_percentile
        if np.isfinite(ep):
            pctile_score = max(0, 1.0 - ep / 0.5)
            scores.append(('pctile', pctile_score, 0.35))
        else:
            scores.append(('pctile', 0.0, 0.35))

        # 绝对排列熵得分：< 0.70 = 1.0, > 0.95 = 0.0
        pe = daily.perm_entropy_20
        if np.isfinite(pe):
            pe_score = max(0, min(1, (0.95 - pe) / 0.25))
            scores.append(('pe', pe_score, 0.25))
        else:
            scores.append(('pe', 0.0, 0.25))

        # 熵差得分：正值好，> 0.05 = 满分
        gap = daily.entropy_gap
        if np.isfinite(gap):
            gap_score = max(0, min(1, gap / 0.05))
            scores.append(('gap', gap_score, 0.20))
        else:
            scores.append(('gap', 0.0, 0.20))

        # 连续压缩天数：> 10 天 = 满分
        days_score = min(1.0, daily.compression_days / 10.0)
        scores.append(('days', days_score, 0.20))

        # 加权
        total_weight = sum(w for _, _, w in scores)
        if total_weight <= 0:
            return 0.0
        return sum(s * w for _, s, w in scores) / total_weight

    def _score_intraday_concentration(
        self,
        today: IntradayEntropyResult,
        recent_days: List[IntradayEntropyResult],
    ) -> float:
        """
        日内集中化评分 [0, 1]

        高分条件：
        - 今日成交量熵低（成交集中）
        - 成交量熵相比近期明显下降
        - 成交量 Herfindahl 指数高（集中在少数分钟）
        """
        scores = []

        # 成交量熵绝对值：< 0.80 = 1.0, > 0.95 = 0.0
        te = today.turnover_entropy
        if np.isfinite(te):
            te_score = max(0, min(1, (0.95 - te) / 0.15))
            scores.append(('te_abs', te_score, 0.30))
        else:
            scores.append(('te_abs', 0.0, 0.30))

        # 成交量熵降幅（相比近期均值）
        drop = self._compute_turnover_entropy_drop(today, recent_days)
        if np.isfinite(drop):
            # drop > 0 表示今日比近期均值低 → 集中化
            drop_score = max(0, min(1, drop / 0.08))
            scores.append(('te_drop', drop_score, 0.35))
        else:
            scores.append(('te_drop', 0.0, 0.35))

        # 体积集中度
        vc = today.volume_concentration
        if np.isfinite(vc):
            # vc 越大 = 越集中；典型范围 0.004 ~ 0.05
            vc_score = max(0, min(1, (vc - 0.004) / 0.030))
            scores.append(('vc', vc_score, 0.35))
        else:
            scores.append(('vc', 0.0, 0.35))

        total_weight = sum(w for _, _, w in scores)
        if total_weight <= 0:
            return 0.0
        return sum(s * w for _, s, w in scores) / total_weight

    def _score_direction(
        self,
        today: IntradayEntropyResult,
        daily: DailyEntropyResult,
    ) -> float:
        """
        方向性信号评分 [0, 1]

        高分条件：
        - 日内路径不可逆性高（有因果方向）
        - 主导特征值接近临界（日线）
        - 日内排列熵下午 < 上午（尾盘有序化）
        """
        scores = []

        # 日内路径不可逆性
        pi = today.path_irreversibility
        if np.isfinite(pi):
            # > 0.05 = 满分
            pi_score = max(0, min(1, pi / 0.05))
            scores.append(('pi', pi_score, 0.40))
        else:
            scores.append(('pi', 0.0, 0.40))

        # 日线主导特征值（接近 ±1 = 临界减速）
        de = daily.dominant_eigenvalue
        if np.isfinite(de):
            de_score = min(1.0, abs(de) / 0.8)
            scores.append(('de', de_score, 0.30))
        else:
            scores.append(('de', 0.0, 0.30))

        # 日内熵偏移（下午 < 上午 = 尾盘有序化 → 好信号）
        shift = today.entropy_shift
        if np.isfinite(shift):
            # shift < 0 → 下午更有序 → 加分
            shift_score = max(0, min(1, -shift / 0.05 + 0.5))
            scores.append(('shift', shift_score, 0.30))
        else:
            scores.append(('shift', 0.5, 0.30))

        total_weight = sum(w for _, _, w in scores)
        if total_weight <= 0:
            return 0.0
        return sum(s * w for _, s, w in scores) / total_weight

    def _score_volume_pattern(
        self,
        today: IntradayEntropyResult,
        daily: DailyEntropyResult,
    ) -> float:
        """
        量能形态评分 [0, 1]

        高分条件：
        - 活跃分钟占比高（不是冷门股）
        - 振幅小（悄然吸筹 ≠ 大幅拉升）
        - 日线方差抬升（临界前兆）
        """
        scores = []

        # 活跃度
        ar = today.active_bar_ratio
        ar_score = max(0, min(1, (ar - 0.5) / 0.4))  # 0.5→0, 0.9→1
        scores.append(('active', ar_score, 0.25))

        # 振幅小 → 好（悄然吸筹）
        pr = today.price_range_ratio
        if np.isfinite(pr):
            # < 0.02 = 满分, > 0.06 = 0
            pr_score = max(0, min(1, (0.06 - pr) / 0.04))
            scores.append(('range', pr_score, 0.35))
        else:
            scores.append(('range', 0.5, 0.35))

        # 日线方差抬升（正值 = 失稳前兆 = 有利）
        vl = daily.var_lift
        if np.isfinite(vl):
            vl_score = max(0, min(1, vl * 2 + 0.3))
            scores.append(('var_lift', vl_score, 0.40))
        else:
            scores.append(('var_lift', 0.3, 0.40))

        total_weight = sum(w for _, _, w in scores)
        if total_weight <= 0:
            return 0.0
        return sum(s * w for _, s, w in scores) / total_weight

    # ================================================================
    #  辅助计算
    # ================================================================

    def _compute_turnover_entropy_drop(
        self,
        today: IntradayEntropyResult,
        recent_days: List[IntradayEntropyResult],
    ) -> float:
        """计算今日成交量熵相对近期均值的降幅。"""
        if not recent_days:
            return 0.0

        recent_te = [d.turnover_entropy for d in recent_days
                     if np.isfinite(d.turnover_entropy)]
        if not recent_te:
            return 0.0

        mean_te = np.mean(recent_te)
        today_te = today.turnover_entropy

        if not np.isfinite(today_te) or not np.isfinite(mean_te):
            return 0.0

        return float(mean_te - today_te)  # 正值 = 今日更集中

    # ================================================================
    #  硬性门槛检查
    # ================================================================

    def _check_hard_gates(
        self,
        daily: DailyEntropyResult,
        today: IntradayEntropyResult,
        turnover_entropy_drop: float,
    ) -> tuple:
        """
        检查硬性门槛，返回 (passed, reason)。

        任一不满足即返回 False。
        """
        cfg = self.config

        # 日线必须处于低熵区域
        if daily.entropy_percentile > cfg.daily_entropy_percentile_max:
            return False, f'daily_percentile={daily.entropy_percentile:.2f} > {cfg.daily_entropy_percentile_max}'

        if daily.perm_entropy_20 > cfg.daily_perm_entropy_max:
            return False, f'daily_pe20={daily.perm_entropy_20:.3f} > {cfg.daily_perm_entropy_max}'

        # 日内活跃度
        if today.active_bar_ratio < cfg.min_active_bar_ratio:
            return False, f'active_bar_ratio={today.active_bar_ratio:.2f} < {cfg.min_active_bar_ratio}'

        # 日内振幅不能过大（悄然吸筹 ≠ 拉升）
        if today.price_range_ratio > cfg.max_price_range_ratio:
            return False, f'price_range={today.price_range_ratio:.4f} > {cfg.max_price_range_ratio}'

        return True, 'passed'

    # ================================================================
    #  主评估入口
    # ================================================================

    def evaluate(
        self,
        stock_code: str,
        daily: DailyEntropyResult,
        today_intraday: IntradayEntropyResult,
        recent_intraday: List[IntradayEntropyResult],
    ) -> FusionResult:
        """
        评估单只股票的双熵融合信号。

        参数
        ----
        stock_code : str
            股票代码
        daily : DailyEntropyResult
            日线熵计算结果
        today_intraday : IntradayEntropyResult
            当日日内熵分析结果
        recent_intraday : list[IntradayEntropyResult]
            最近 N 天的日内熵结果（不含今日，用于计算drop）
        """
        cfg = self.config
        trade_date = today_intraday.trade_date

        # 成交量熵降幅
        te_drop = self._compute_turnover_entropy_drop(today_intraday, recent_intraday)

        # 硬性门槛
        gate_passed, gate_reason = self._check_hard_gates(daily, today_intraday, te_drop)

        # 四个子得分
        s1 = self._score_daily_compression(daily)
        s2 = self._score_intraday_concentration(today_intraday, recent_intraday)
        s3 = self._score_direction(today_intraday, daily)
        s4 = self._score_volume_pattern(today_intraday, daily)

        # 加权总分
        w = cfg.weights
        total = (
            s1 * w['daily_compression']
            + s2 * w['intraday_concentration']
            + s3 * w['direction_signal']
            + s4 * w['volume_pattern']
        )

        # 信号决策
        if not gate_passed:
            signal: Signal = 'skip'
            reason = f'gate_failed: {gate_reason}'
        elif total >= cfg.buy_score_min:
            signal = 'buy'
            reason = f'score={total:.3f} >= {cfg.buy_score_min}'
        elif total >= cfg.watch_score_min:
            signal = 'watch'
            reason = f'score={total:.3f} >= {cfg.watch_score_min}'
        else:
            signal = 'skip'
            reason = f'score={total:.3f} < {cfg.watch_score_min}'

        return FusionResult(
            stock_code=stock_code,
            trade_date=trade_date,
            daily_compression_score=s1,
            intraday_concentration_score=s2,
            direction_score=s3,
            volume_pattern_score=s4,
            total_score=total,
            signal=signal,
            reason=reason,
            daily_perm_entropy_20=daily.perm_entropy_20 if np.isfinite(daily.perm_entropy_20) else 0.0,
            daily_entropy_percentile=daily.entropy_percentile,
            daily_entropy_gap=daily.entropy_gap,
            daily_compression_days=daily.compression_days,
            intraday_turnover_entropy=today_intraday.turnover_entropy if np.isfinite(today_intraday.turnover_entropy) else 0.0,
            intraday_turnover_entropy_drop=te_drop,
            intraday_path_irrev=today_intraday.path_irreversibility if np.isfinite(today_intraday.path_irreversibility) else 0.0,
            intraday_volume_concentration=today_intraday.volume_concentration if np.isfinite(today_intraday.volume_concentration) else 0.0,
            intraday_state=today_intraday.intraday_state,
        )
