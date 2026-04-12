"""
双熵卖出信号模块 (Sell Signal)

三类卖出场景：
  1. 熵扩散卖出（结构瓦解）：日线熵快速膨胀 + 日内成交分散
  2. 暗中派发卖出（日内领先日线）：日线还没反映但日内微观结构已恶化
  3. 熵衰竭卖出（趋势终结）：双高熵 + 路径不可逆性衰退

信号生成流程：
  1. 熵扩散得分 → 日线 PE 上升速度 + 百分位穿越
  2. 派发检测得分 → 日内 TE 升高 + AM/PM 反转 + 方向性丧失
  3. 衰竭得分 → 日内/日线双高熵 + 不可逆性衰退
  4. 量能异常得分 → 高振幅 + 集中度下降
  5. 综合打分 → sell / warning / hold
"""

import numpy as np
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass

from .config import SellSignalConfig
from .daily_entropy import DailyEntropyResult
from .intraday_entropy import IntradayEntropyResult


SellSignal = Literal['sell', 'warning', 'hold']


@dataclass
class SellResult:
    """卖出信号结果"""

    stock_code: str
    trade_date: str

    # 四个子得分
    entropy_diffusion_score: float      # 熵扩散（结构瓦解）
    stealth_distribution_score: float   # 暗中派发
    exhaustion_score: float             # 熵衰竭
    volume_anomaly_score: float         # 量能异常

    # 综合
    total_score: float
    signal: SellSignal
    reason: str
    sell_type: str                      # 主要卖出类型

    # 关键特征值（供调试/展示）
    daily_perm_entropy_20: float
    daily_entropy_percentile: float
    daily_entropy_velocity_5: float     # PE20 近 5 日变化率
    daily_path_irrev_velocity_5: float  # 路径不可逆近 5 日变化率
    intraday_turnover_entropy: float
    intraday_te_rise: float             # 成交量熵上升幅度
    intraday_path_irrev: float
    intraday_entropy_shift: float       # 下午 - 上午
    intraday_perm_entropy: float
    intraday_volume_concentration: float
    intraday_price_range: float

    def to_dict(self) -> Dict:
        return {
            'stock_code': self.stock_code,
            'trade_date': self.trade_date,
            'entropy_diffusion_score': round(self.entropy_diffusion_score, 4),
            'stealth_distribution_score': round(self.stealth_distribution_score, 4),
            'exhaustion_score': round(self.exhaustion_score, 4),
            'volume_anomaly_score': round(self.volume_anomaly_score, 4),
            'total_score': round(self.total_score, 4),
            'signal': self.signal,
            'reason': self.reason,
            'sell_type': self.sell_type,
            'daily_perm_entropy_20': round(self.daily_perm_entropy_20, 4),
            'daily_entropy_percentile': round(self.daily_entropy_percentile, 4),
            'daily_entropy_velocity_5': round(self.daily_entropy_velocity_5, 4),
            'daily_path_irrev_velocity_5': round(self.daily_path_irrev_velocity_5, 4),
            'intraday_turnover_entropy': round(self.intraday_turnover_entropy, 4),
            'intraday_te_rise': round(self.intraday_te_rise, 4),
            'intraday_path_irrev': round(self.intraday_path_irrev, 4),
            'intraday_entropy_shift': round(self.intraday_entropy_shift, 4),
            'intraday_perm_entropy': round(self.intraday_perm_entropy, 4),
            'intraday_volume_concentration': round(self.intraday_volume_concentration, 4),
            'intraday_price_range': round(self.intraday_price_range, 4),
        }


class SellSignalEngine:
    """
    双熵卖出信号生成器。

    与 FusionSignal（买入）互为镜像：
    - 买入捕捉 "压缩态中的秘密建仓"
    - 卖出捕捉 "有序瓦解和暗中派发"
    """

    def __init__(self, config: Optional[SellSignalConfig] = None):
        self.config = config or SellSignalConfig()

    # ================================================================
    #  辅助计算
    # ================================================================

    def _compute_te_rise(
        self,
        today: IntradayEntropyResult,
        recent_days: List[IntradayEntropyResult],
    ) -> float:
        """计算今日成交量熵相对近期均值的上升幅度（正值 = 今日更分散）。"""
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
        return float(today_te - mean_te)  # 正值 = 今日更分散

    def _compute_pi_drop(
        self,
        today: IntradayEntropyResult,
        recent_days: List[IntradayEntropyResult],
    ) -> float:
        """计算日内路径不可逆性相对近期的下降幅度（正值 = 今日方向性更弱）。"""
        if not recent_days:
            return 0.0
        recent_pi = [d.path_irreversibility for d in recent_days
                     if np.isfinite(d.path_irreversibility)]
        if not recent_pi:
            return 0.0
        mean_pi = np.mean(recent_pi)
        today_pi = today.path_irreversibility
        if not np.isfinite(today_pi) or not np.isfinite(mean_pi):
            return 0.0
        return float(mean_pi - today_pi)  # 正值 = 今日方向性降低

    def _compute_concentration_drop(
        self,
        today: IntradayEntropyResult,
        recent_days: List[IntradayEntropyResult],
    ) -> float:
        """计算成交量集中度下降幅度（正值 = 今日更分散）。"""
        if not recent_days:
            return 0.0
        recent_vc = [d.volume_concentration for d in recent_days
                     if np.isfinite(d.volume_concentration)]
        if not recent_vc:
            return 0.0
        mean_vc = np.mean(recent_vc)
        today_vc = today.volume_concentration
        if not np.isfinite(today_vc) or not np.isfinite(mean_vc):
            return 0.0
        return float(mean_vc - today_vc)  # 正值 = 集中度下降

    # ================================================================
    #  1. 熵扩散得分（结构瓦解）
    # ================================================================

    def _score_entropy_diffusion(self, daily: DailyEntropyResult) -> float:
        """
        熵扩散得分 [0, 1]

        高分条件：
        - PE20 近 5 日快速上升（熵膨胀）
        - 百分位跃升脱离压缩区
        - 多尺度熵差变负或收窄（短期不再比长期有序）
        """
        scores = []

        # PE20 变化速度：正值 = 熵膨胀
        ev = daily.entropy_velocity_5
        if np.isfinite(ev):
            # > 0.05 满分, < 0 零分
            ev_score = max(0, min(1, ev / 0.05))
            scores.append(('ev', ev_score, 0.40))
        else:
            scores.append(('ev', 0.0, 0.40))

        # 百分位穿越：从低位快速升至 0.70+
        ep = daily.entropy_percentile
        ep_prev = daily.entropy_percentile_prev5
        if np.isfinite(ep) and np.isfinite(ep_prev):
            # 百分位加速度
            ep_accel = ep - ep_prev
            # 当前处于高位 且 加速上升
            high_score = max(0, min(1, (ep - 0.50) / 0.30))  # 0.50→0, 0.80→1
            accel_score = max(0, min(1, ep_accel / 0.20))
            combined = high_score * 0.6 + accel_score * 0.4
            scores.append(('ep_cross', combined, 0.35))
        else:
            scores.append(('ep_cross', 0.0, 0.35))

        # 熵差收窄/变负
        gap = daily.entropy_gap
        if np.isfinite(gap):
            # 负值 = 短期比长期更无序 → 卖出信号
            gap_score = max(0, min(1, (0.02 - gap) / 0.04))  # 0.02→0, -0.02→1
            scores.append(('gap', gap_score, 0.25))
        else:
            scores.append(('gap', 0.0, 0.25))

        total_weight = sum(w for _, _, w in scores)
        return sum(s * w for _, s, w in scores) / total_weight if total_weight > 0 else 0.0

    # ================================================================
    #  2. 暗中派发得分（日内领先日线）
    # ================================================================

    def _score_stealth_distribution(
        self,
        today: IntradayEntropyResult,
        recent_days: List[IntradayEntropyResult],
        daily: DailyEntropyResult,
    ) -> float:
        """
        暗中派发得分 [0, 1]

        日线还没反映，但日内微观结构已经恶化：
        - 成交量熵上升（从集中 → 分散，对倒/分批出货）
        - AM/PM 反转（上午拉升+下午混乱）
        - 路径不可逆性下降（来回震荡无方向）
        - 集中度下降（不再有少数分钟集中成交）
        """
        scores = []

        # 成交量熵上升
        te_rise = self._compute_te_rise(today, recent_days)
        if np.isfinite(te_rise):
            # > 0.05 满分
            te_score = max(0, min(1, te_rise / 0.05))
            scores.append(('te_rise', te_score, 0.30))
        else:
            scores.append(('te_rise', 0.0, 0.30))

        # AM/PM 偏移反转：下午比上午更无序 → 尾盘出货
        shift = today.entropy_shift
        if np.isfinite(shift):
            # shift > 0 = 下午更无序 → 卖出信号
            shift_score = max(0, min(1, shift / 0.05))
            scores.append(('shift', shift_score, 0.25))
        else:
            scores.append(('shift', 0.0, 0.25))

        # 日内路径不可逆性下降
        pi_drop = self._compute_pi_drop(today, recent_days)
        if np.isfinite(pi_drop):
            # 正值 = 方向性丧失 → > 0.02 满分
            pi_score = max(0, min(1, pi_drop / 0.02))
            scores.append(('pi_drop', pi_score, 0.25))
        else:
            scores.append(('pi_drop', 0.0, 0.25))

        # 集中度下降
        vc_drop = self._compute_concentration_drop(today, recent_days)
        if np.isfinite(vc_drop):
            # 正值 = 集中度降低 → > 0.01 满分
            vc_score = max(0, min(1, vc_drop / 0.01))
            scores.append(('vc_drop', vc_score, 0.20))
        else:
            scores.append(('vc_drop', 0.0, 0.20))

        total_weight = sum(w for _, _, w in scores)
        return sum(s * w for _, s, w in scores) / total_weight if total_weight > 0 else 0.0

    # ================================================================
    #  3. 衰竭得分（趋势终结）
    # ================================================================

    def _score_exhaustion(
        self,
        daily: DailyEntropyResult,
        today: IntradayEntropyResult,
    ) -> float:
        """
        熵衰竭得分 [0, 1]

        趋势走完后的自然耗竭：
        - 日线排列熵回到高位（> 0.92）
        - 日内排列熵高位（> 0.96）
        - 日线路径不可逆性已从高位回落
        - 日线主导特征值回归低绝对值
        """
        scores = []

        # 日线排列熵高位
        pe = daily.perm_entropy_20
        if np.isfinite(pe):
            pe_score = max(0, min(1, (pe - 0.85) / 0.10))  # 0.85→0, 0.95→1
            scores.append(('daily_pe', pe_score, 0.30))
        else:
            scores.append(('daily_pe', 0.0, 0.30))

        # 日内排列熵高位
        ipe = today.perm_entropy
        if np.isfinite(ipe):
            ipe_score = max(0, min(1, (ipe - 0.90) / 0.08))  # 0.90→0, 0.98→1
            scores.append(('intra_pe', ipe_score, 0.25))
        else:
            scores.append(('intra_pe', 0.0, 0.25))

        # 路径不可逆性衰退（曾高现低）
        pi_vel = daily.path_irrev_velocity_5
        if np.isfinite(pi_vel):
            # 负值 = 不可逆性正在下降 → 卖出信号
            pi_score = max(0, min(1, -pi_vel / 0.03))
            scores.append(('pi_decay', pi_score, 0.25))
        else:
            scores.append(('pi_decay', 0.0, 0.25))

        # 主导特征值回归低绝对值
        de = daily.dominant_eigenvalue
        if np.isfinite(de):
            # 低绝对值 = 系统不再临界 → 趋势结束
            de_score = max(0, 1.0 - abs(de) / 0.5)  # |de|=0→1, |de|=0.5→0
            scores.append(('de_low', de_score, 0.20))
        else:
            scores.append(('de_low', 0.5, 0.20))

        total_weight = sum(w for _, _, w in scores)
        return sum(s * w for _, s, w in scores) / total_weight if total_weight > 0 else 0.0

    # ================================================================
    #  4. 量能异常得分
    # ================================================================

    def _score_volume_anomaly(
        self,
        today: IntradayEntropyResult,
        recent_days: List[IntradayEntropyResult],
    ) -> float:
        """
        量能异常得分 [0, 1]

        高分条件：
        - 高振幅（出货需要空间）
        - 集中度下降（不再有主力集中操作）
        - 活跃分钟保持但质量下降
        """
        scores = []

        # 高振幅
        pr = today.price_range_ratio
        if np.isfinite(pr):
            # > 0.05 满分, < 0.02 零分
            pr_score = max(0, min(1, (pr - 0.02) / 0.03))
            scores.append(('range', pr_score, 0.40))
        else:
            scores.append(('range', 0.0, 0.40))

        # 集中度下降
        vc_drop = self._compute_concentration_drop(today, recent_days)
        if np.isfinite(vc_drop):
            vc_score = max(0, min(1, vc_drop / 0.008))
            scores.append(('vc_drop', vc_score, 0.35))
        else:
            scores.append(('vc_drop', 0.0, 0.35))

        # 日内熵趋势（日内从有序走向无序）
        et = today.entropy_trend
        if np.isfinite(et):
            # 正值 = 日内越来越无序 → 卖出
            et_score = max(0, min(1, et / 0.001))
            scores.append(('entropy_trend', et_score, 0.25))
        else:
            scores.append(('entropy_trend', 0.0, 0.25))

        total_weight = sum(w for _, _, w in scores)
        return sum(s * w for _, s, w in scores) / total_weight if total_weight > 0 else 0.0

    # ================================================================
    #  卖出类型判定
    # ================================================================

    def _determine_sell_type(
        self,
        s1: float,
        s2: float,
        s3: float,
        s4: float,
    ) -> str:
        """根据子得分判断主要卖出类型。"""
        scores = {
            'entropy_diffusion': s1,
            'stealth_distribution': s2,
            'exhaustion': s3,
            'volume_anomaly': s4,
        }
        return max(scores, key=scores.get)

    # ================================================================
    #  主评估入口
    # ================================================================

    def evaluate(
        self,
        stock_code: str,
        daily: DailyEntropyResult,
        today_intraday: IntradayEntropyResult,
        recent_intraday: List[IntradayEntropyResult],
    ) -> SellResult:
        """
        评估单只股票的卖出信号。

        参数
        ----
        stock_code : str
            股票代码
        daily : DailyEntropyResult
            日线熵计算结果
        today_intraday : IntradayEntropyResult
            当日日内熵结果
        recent_intraday : list[IntradayEntropyResult]
            最近 N 天日内熵结果（不含今日）
        """
        cfg = self.config
        trade_date = today_intraday.trade_date

        # 辅助指标
        te_rise = self._compute_te_rise(today_intraday, recent_intraday)

        # 四个子得分
        s1 = self._score_entropy_diffusion(daily)
        s2 = self._score_stealth_distribution(today_intraday, recent_intraday, daily)
        s3 = self._score_exhaustion(daily, today_intraday)
        s4 = self._score_volume_anomaly(today_intraday, recent_intraday)

        # 加权总分
        w = cfg.weights
        total = (
            s1 * w['entropy_diffusion']
            + s2 * w['stealth_distribution']
            + s3 * w['exhaustion']
            + s4 * w['volume_anomaly']
        )

        # 卖出类型
        sell_type = self._determine_sell_type(s1, s2, s3, s4)

        # 活跃度门槛
        if today_intraday.active_bar_ratio < cfg.min_active_bar_ratio:
            signal: SellSignal = 'hold'
            reason = f'low_activity={today_intraday.active_bar_ratio:.2f}'
        elif total >= cfg.sell_score_min:
            signal = 'sell'
            reason = f'score={total:.3f} >= {cfg.sell_score_min} ({sell_type})'
        elif total >= cfg.warning_score_min:
            signal = 'warning'
            reason = f'score={total:.3f} >= {cfg.warning_score_min} ({sell_type})'
        else:
            signal = 'hold'
            reason = f'score={total:.3f} < {cfg.warning_score_min}'

        return SellResult(
            stock_code=stock_code,
            trade_date=trade_date,
            entropy_diffusion_score=s1,
            stealth_distribution_score=s2,
            exhaustion_score=s3,
            volume_anomaly_score=s4,
            total_score=total,
            signal=signal,
            reason=reason,
            sell_type=sell_type,
            daily_perm_entropy_20=daily.perm_entropy_20 if np.isfinite(daily.perm_entropy_20) else 0.0,
            daily_entropy_percentile=daily.entropy_percentile,
            daily_entropy_velocity_5=daily.entropy_velocity_5,
            daily_path_irrev_velocity_5=daily.path_irrev_velocity_5,
            intraday_turnover_entropy=today_intraday.turnover_entropy if np.isfinite(today_intraday.turnover_entropy) else 0.0,
            intraday_te_rise=te_rise,
            intraday_path_irrev=today_intraday.path_irreversibility if np.isfinite(today_intraday.path_irreversibility) else 0.0,
            intraday_entropy_shift=today_intraday.entropy_shift if np.isfinite(today_intraday.entropy_shift) else 0.0,
            intraday_perm_entropy=today_intraday.perm_entropy if np.isfinite(today_intraday.perm_entropy) else 0.0,
            intraday_volume_concentration=today_intraday.volume_concentration if np.isfinite(today_intraday.volume_concentration) else 0.0,
            intraday_price_range=today_intraday.price_range_ratio if np.isfinite(today_intraday.price_range_ratio) else 0.0,
        )
