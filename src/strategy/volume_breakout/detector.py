"""
量价突破策略 - 突破检测模块

检测模式: 波动率压缩 → 放量突破前高
与分岔策略互补: 不依赖主导特征值，捕获跳变式启动
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .config import BreakoutDetectorConfig


@dataclass
class BreakoutResult:
    """突破检测结果"""
    stock_code: str
    signal: str               # buy / watch / skip
    total_score: float

    # 子分
    compression_score: float = 0.0
    volume_score: float = 0.0
    breakout_score: float = 0.0
    momentum_score: float = 0.0

    # 原始指标
    realized_vol_20: float = 0.0
    price_range_20: float = 0.0
    bb_width: float = 0.0
    vol_surge: float = 0.0
    vol_spike: float = 0.0
    breakout_new_high: bool = False
    ret_5d: float = 0.0
    ret_20d: float = 0.0
    close: float = 0.0


class BreakoutDetector:
    """量价突破检测器"""

    def __init__(self, config: Optional[BreakoutDetectorConfig] = None):
        self.cfg = config or BreakoutDetectorConfig()

    def passes_prescreen(
        self,
        close: np.ndarray,
        vol: np.ndarray,
    ) -> bool:
        """快速预筛: 流动性 + 股价 + 最低数据量"""
        if len(close) < 70:
            return False
        last_c = close[-1]
        if last_c < self.cfg.min_close or last_c > self.cfg.max_close:
            return False
        avg_vol = np.mean(vol[-20:])
        liq = last_c * avg_vol
        if liq < self.cfg.min_liquidity:
            return False
        return True

    def evaluate(
        self,
        stock_code: str,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        vol: np.ndarray,
    ) -> Optional[BreakoutResult]:
        """
        计算突破评分。

        需要 >= 70 天的 close/high/low/vol 数据。
        """
        n = len(close)
        if n < 70:
            return None

        # ---- 基础指标 ----
        c20 = close[-20:]
        h20 = high[-20:]
        l20 = low[-20:]
        v20 = vol[-20:]

        # 实现波动率 (20日日收益率标准差)
        rets20 = np.diff(np.log(c20))
        realized_vol = float(np.std(rets20))

        # 价格区间
        price_range = float((np.max(h20) - np.min(l20)) / np.mean(c20))

        # 布林带宽度 (近似: 4 * 20日波动率)
        bb_width = 4.0 * realized_vol

        # 放量指标
        v5 = float(np.mean(vol[-5:]))
        v20_avg = float(np.mean(v20))
        vol_surge = v5 / max(v20_avg, 1.0)
        vol_spike = float(vol[-1]) / max(v20_avg, 1.0)

        # 突破前高
        lookback = min(self.cfg.breakout_lookback, n - 1)
        prev_high = float(np.max(close[-lookback - 1:-1]))
        breakout_new_high = close[-1] > prev_high

        # 收益率
        ret_5d = float(np.log(close[-1] / close[-6])) if n > 6 else 0.0
        ret_20d = float(np.log(close[-1] / close[-21])) if n > 21 else 0.0

        # ---- 硬门槛 ----
        # 至少满足压缩条件之一 (当前或前期)
        is_compressed = (
            realized_vol < self.cfg.max_realized_vol_20
            or price_range < self.cfg.max_price_range_20
            or bb_width < self.cfg.max_bb_width
        )
        # 前期压缩也算
        if not is_compressed and n >= 40:
            c_prev_gate = close[-40:-20]
            h_prev_gate = high[-40:-20]
            l_prev_gate = low[-40:-20]
            rets_prev_gate = np.diff(np.log(c_prev_gate))
            rv_prev_gate = float(np.std(rets_prev_gate))
            pr_prev_gate = float((np.max(h_prev_gate) - np.min(l_prev_gate)) / np.mean(c_prev_gate))
            bb_prev_gate = 4.0 * rv_prev_gate
            is_compressed = (
                rv_prev_gate < self.cfg.max_realized_vol_20
                or pr_prev_gate < self.cfg.max_price_range_20
                or bb_prev_gate < self.cfg.max_bb_width
            )
        if not is_compressed:
            return BreakoutResult(
                stock_code=stock_code, signal='skip', total_score=0.0,
                realized_vol_20=realized_vol, price_range_20=price_range,
                bb_width=bb_width, vol_surge=vol_surge, vol_spike=vol_spike,
                breakout_new_high=breakout_new_high, ret_5d=ret_5d,
                ret_20d=ret_20d, close=float(close[-1]),
            )

        # 动量最低要求
        if ret_20d < self.cfg.min_ret_20d:
            return BreakoutResult(
                stock_code=stock_code, signal='skip', total_score=0.0,
                realized_vol_20=realized_vol, price_range_20=price_range,
                bb_width=bb_width, vol_surge=vol_surge, vol_spike=vol_spike,
                breakout_new_high=breakout_new_high, ret_5d=ret_5d,
                ret_20d=ret_20d, close=float(close[-1]),
            )

        # 过度延伸过滤: 已大涨的股票不再追
        if ret_20d > self.cfg.max_ret_20d:
            return BreakoutResult(
                stock_code=stock_code, signal='skip', total_score=0.0,
                realized_vol_20=realized_vol, price_range_20=price_range,
                bb_width=bb_width, vol_surge=vol_surge, vol_spike=vol_spike,
                breakout_new_high=breakout_new_high, ret_5d=ret_5d,
                ret_20d=ret_20d, close=float(close[-1]),
            )

        # 放量硬门槛: 量价突破策略的核心 — 必须有放量信号
        has_volume = (vol_surge >= self.cfg.min_vol_surge or vol_spike >= self.cfg.min_vol_spike)
        if not has_volume:
            return BreakoutResult(
                stock_code=stock_code, signal='skip', total_score=0.0,
                realized_vol_20=realized_vol, price_range_20=price_range,
                bb_width=bb_width, vol_surge=vol_surge, vol_spike=vol_spike,
                breakout_new_high=breakout_new_high, ret_5d=ret_5d,
                ret_20d=ret_20d, close=float(close[-1]),
            )

        # ---- 子分计算 ----

        # 1. 压缩质量 (0 ~ 1) — 最重要信号
        #    使用 max(当前, 前期) 来避免突破日当天压缩分下降
        rv_score = np.clip(1.0 - realized_vol / 0.035, 0, 1)
        pr_score = np.clip(1.0 - price_range / 0.25, 0, 1)
        bb_score = np.clip(1.0 - bb_width / 0.14, 0, 1)

        # 前期压缩 (20-40日前的窗口) — 捕获"曾经压缩、现在启动"模式
        if n >= 40:
            c_prev = close[-40:-20]
            h_prev = high[-40:-20]
            l_prev = low[-40:-20]
            rets_prev = np.diff(np.log(c_prev))
            rv_prev = float(np.std(rets_prev))
            pr_prev = float((np.max(h_prev) - np.min(l_prev)) / np.mean(c_prev))
            bb_prev = 4.0 * rv_prev
            rv_prev_score = np.clip(1.0 - rv_prev / 0.035, 0, 1)
            pr_prev_score = np.clip(1.0 - pr_prev / 0.25, 0, 1)
            bb_prev_score = np.clip(1.0 - bb_prev / 0.14, 0, 1)
            rv_score = max(rv_score, rv_prev_score)
            pr_score = max(pr_score, pr_prev_score)
            bb_score = max(bb_score, bb_prev_score)

        # 压缩持续天数
        compress_days = 0
        for i in range(min(20, n - 20)):
            sub = close[-(20 + i):-(i) if i > 0 else n]
            if len(sub) < 20:
                break
            rv_i = np.std(np.diff(np.log(sub)))
            if rv_i < self.cfg.max_realized_vol_20:
                compress_days += 1
            else:
                break
        days_score = np.clip(compress_days / 15.0, 0, 1)
        compression_score = float(
            0.30 * rv_score + 0.25 * pr_score + 0.25 * bb_score + 0.20 * days_score
        )

        # 2. 放量新鲜度 (0 ~ 1)
        #    核心洞察: 真正的起爆信号是"今天突然放量"而不是"持续放量"
        #    vol_spike 高 + vol_surge 低 = 量能刚刚启动 (理想)
        #    vol_spike 高 + vol_surge 高 = 量能已持续数日 (较晚)
        spike_score = np.clip((vol_spike - 1.0) / 3.0, 0, 1)
        surge_score = np.clip((vol_surge - 1.0) / 2.0, 0, 1)
        # 新鲜度: spike/surge 越高说明今天的放量相对于近5天更突出
        if vol_surge > 1.0:
            freshness = vol_spike / vol_surge
            fresh_score = np.clip((freshness - 1.0) / 2.0, 0, 1)
        else:
            fresh_score = spike_score
        # 量能加速
        if n >= 8:
            v3_recent = np.mean(vol[-3:])
            v3_prev = np.mean(vol[-6:-3])
            vol_accel = (v3_recent - v3_prev) / max(v3_prev, 1.0)
            accel_score = np.clip(vol_accel / 1.0, 0, 1)
        else:
            accel_score = 0.0
        volume_score = float(0.30 * spike_score + 0.35 * fresh_score + 0.15 * surge_score + 0.20 * accel_score)

        # 3. 位置质量 (0 ~ 1) — 接近前高比已突破更有价值
        #    这是"预突破"策略: 买在突破前/初期, 不是追高
        proximity = close[-1] / max(prev_high, 0.01)
        if breakout_new_high:
            # 已突破: 小幅突破给分, 大幅突破惩罚(已离开最佳买点)
            overshoot = (close[-1] - prev_high) / prev_high
            if overshoot <= 0.03:
                breakout_s = 0.90  # 刚好突破 — 确认信号
            elif overshoot <= 0.08:
                breakout_s = 0.70  # 小幅突破
            else:
                breakout_s = max(0.40, 0.70 - (overshoot - 0.08) * 5)  # 大幅突破递减
        else:
            # 未突破: 越接近前高越好
            if proximity >= 0.95:
                breakout_s = 0.80  # 非常接近 — 即将突破
            elif proximity >= 0.90:
                breakout_s = float(0.30 + 0.50 * (proximity - 0.90) / 0.05)
            else:
                breakout_s = float(max(0, 0.30 * (proximity - 0.80) / 0.10))

        # 4. 动量确认 (0 ~ 1) — 仅做最低确认
        r5_score = np.clip(ret_5d / 0.10, 0, 1)
        r20_score = np.clip((ret_20d + 0.05) / 0.15, 0, 1)
        if n > 11:
            r5_prev = float(np.log(close[-6] / close[-11]))
            accel_mom = ret_5d - r5_prev
            mom_accel_score = np.clip(accel_mom / 0.10, 0, 1)
        else:
            mom_accel_score = 0.0
        momentum_score = float(0.40 * r5_score + 0.30 * r20_score + 0.30 * mom_accel_score)

        # ---- 加权总分 ----
        w = self.cfg.weights
        total = (
            w['compression'] * compression_score
            + w['volume'] * volume_score
            + w['breakout'] * breakout_s
            + w['momentum'] * momentum_score
        )
        total = float(total)

        # ---- 信号判定 ----
        if total >= self.cfg.buy_score_min:
            signal = 'buy'
        elif total >= self.cfg.watch_score_min:
            signal = 'watch'
        else:
            signal = 'skip'

        return BreakoutResult(
            stock_code=stock_code,
            signal=signal,
            total_score=total,
            compression_score=compression_score,
            volume_score=volume_score,
            breakout_score=breakout_s,
            momentum_score=momentum_score,
            realized_vol_20=realized_vol,
            price_range_20=price_range,
            bb_width=bb_width,
            vol_surge=vol_surge,
            vol_spike=vol_spike,
            breakout_new_high=breakout_new_high,
            ret_5d=ret_5d,
            ret_20d=ret_20d,
            close=float(close[-1]),
        )
