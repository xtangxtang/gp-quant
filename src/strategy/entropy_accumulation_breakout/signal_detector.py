"""
三阶段状态检测器 — 惜售吸筹 / 分岔突破 / 结构崩塌

状态机:
  idle → accumulation → breakout_ready → breakout → hold → collapse → exit

理论对应:
  accumulation  : 耗散结构局部熵降, 资本流入创造有序 (Prigogine)
  breakout      : 分岔临界点, 对称性破缺 (Dmitriev 2025, 主特征值 → 1)
  collapse      : 耗散结构维持失败, 熵扩散, 能量耗尽 (Seifert 2025)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ═════════════════════════════════════════════════════════
# 阈值配置
# ═════════════════════════════════════════════════════════

@dataclass
class DetectorConfig:
    """可调阈值, 默认值基于经验和论文建议"""

    # ── Phase 1: 惜售吸筹 ──
    # 置换熵低于此值视为有序状态
    perm_entropy_low: float = 0.65
    # 路径不可逆性高于此值视为有定向力量
    path_irrev_high: float = 0.05
    # 换手率熵低于此值视为流动性收缩
    turnover_entropy_low: float = 0.6
    # 惜售状态最少持续天数
    accum_min_days: int = 5
    # 大单净额累计为正 (如果有数据)
    big_net_positive: bool = True

    # ── Phase 2: 分岔突破 ──
    # 主特征值 > 此值 = 临界减速
    dom_eig_threshold: float = 0.85
    # 量能脉冲 > 此值 = 放量
    vol_impulse_threshold: float = 1.8
    # 突破时熵仍需低于此值 (有序突破)
    perm_entropy_breakout_max: float = 0.75

    # ── Phase 3: 结构崩塌退出 ──
    # 置换熵超过此值 = 无序扩散
    perm_entropy_collapse: float = 0.90
    # 路径不可逆性跌破此值 = 主力撤离
    path_irrev_collapse: float = 0.01
    # 熵加速度 > 此值 = 熵快速扩张
    entropy_accel_collapse: float = 0.05
    # 成交量衰竭 (相对峰值)
    vol_exhaustion_ratio: float = 0.3
    # 最长持有天数 (安全网)
    max_hold_days: int = 20

    # ── 量子相干性 (Quantum Coherence) ──
    # 惜售阶段纯度下限 (状态越纯 = 筹码越集中)
    purity_accum_min: float = 0.6
    # 突破阶段退相干速率阈值 (负值 = 方向正在确认)
    coherence_decay_breakout: float = -0.005
    # 崩塌阶段纯度上限 (极低纯度 = 共识瓦解)
    purity_collapse_max: float = 0.3

    # ── 周线确认 ──
    weekly_perm_entropy_max: float = 0.75
    weekly_trend_confirm: bool = True

    # ── 分钟线微观结构 ──
    # 日内置换熵低于此值 = 日内价格有序 (吸筹时日内也应有序)
    intraday_entropy_low: float = 0.70
    # 日内成交量集中度高于此值 = 大单/集中时段主导
    intraday_vol_concentration_high: float = 0.01
    # 日内路径不可逆性高于此值 = 日内有定向力量
    intraday_irrev_high: float = 0.03

    # ── 资金流向 ──
    # 大单净额累计(短期)为正 = 主力在买入
    mf_big_cumsum_positive: bool = True
    # 资金流不平衡度 > 此值 = 大单买散户卖 (吸筹特征)
    mf_flow_imbalance_min: float = 0.3
    # 大单连续流入天数 > 此值 = 持续吸筹
    mf_big_streak_min: int = 3
    # 周线大单净额累计为正 = 周级别主力流入
    weekly_big_net_positive: bool = True


# ═════════════════════════════════════════════════════════
# Phase 1: 惜售吸筹检测
# ═════════════════════════════════════════════════════════

def detect_accumulation(
    df: pd.DataFrame,
    cfg: DetectorConfig,
) -> pd.Series:
    """
    逐行检测是否处于惜售吸筹状态.

    惜售的物理含义:
      持有者不愿以当前价格卖出 → 换手率下降 → 流动性收缩 →
      置换熵降低 (价格序列变得更有序) → 路径不可逆性上升
      (少量交易被同一方向力量主导)

    Returns: bool Series, True = 当前处于惜售吸筹状态
    """
    conds = []

    # 1. 置换熵处于低位 (有序)
    if "perm_entropy_m" in df.columns:
        conds.append(df["perm_entropy_m"] < cfg.perm_entropy_low)

    # 2. 路径不可逆性 > 阈值 (定向力量)
    if "path_irrev_m" in df.columns:
        conds.append(df["path_irrev_m"] > cfg.path_irrev_high)

    # 3. 资金流: 大单净额累计为正 (主力在买入)
    if "mf_big_cumsum_s" in df.columns and cfg.mf_big_cumsum_positive:
        mf_ok = df["mf_big_cumsum_s"] > 0
        conds.append(mf_ok)

    # 4. 资金流不平衡: 大单买 + 散户卖 = 典型吸筹
    if "mf_flow_imbalance" in df.columns:
        conds.append(df["mf_flow_imbalance"] > cfg.mf_flow_imbalance_min)

    if len(conds) == 0:
        return pd.Series(False, index=df.index)

    # 至少满足 N-1 个条件
    score = sum(c.astype(int) for c in conds)
    min_required = max(2, len(conds) - 1)
    raw_signal = score >= min_required

    # 持续性要求: 连续至少 accum_min_days 天
    sustained = raw_signal.rolling(cfg.accum_min_days, min_periods=cfg.accum_min_days).sum()
    return sustained >= cfg.accum_min_days


def accumulation_quality(df: pd.DataFrame, cfg: DetectorConfig) -> pd.Series:
    """
    惜售质量分数 [0, 1].
    分数越高 = 惜售信号越强.
    """
    scores = []

    # 置换熵: 越低越好, 归一化到 [0, 1]
    if "perm_entropy_m" in df.columns:
        s = 1.0 - df["perm_entropy_m"].clip(0.3, 1.0) / 1.0
        scores.append(s.fillna(0) * 0.25)

    # 路径不可逆性: 越高越好
    if "path_irrev_m" in df.columns:
        s = df["path_irrev_m"].clip(0, 0.5) / 0.5
        scores.append(s.fillna(0) * 0.20)

    # 大单净额: 正向流入
    if "big_net_ratio_ma" in df.columns:
        s = df["big_net_ratio_ma"].clip(-0.1, 0.1) / 0.1  # [-1, 1]
        s = (s + 1) / 2  # [0, 1]
        scores.append(s.fillna(0.5) * 0.10)

    # 密度矩阵纯度: 越高 = 状态越集中 = 惜售越明确
    if "purity_norm" in df.columns:
        s = df["purity_norm"].clip(0, 1)
        scores.append(s.fillna(0) * 0.15)

    # 资金流: 大单连续流入天数 (持续性越强越好)
    if "mf_big_streak" in df.columns:
        s = df["mf_big_streak"].clip(0, 10) / 10
        scores.append(s.fillna(0) * 0.15)

    # 资金流不平衡: 大单买散户卖越明显越好
    if "mf_flow_imbalance" in df.columns:
        s = df["mf_flow_imbalance"].clip(-1, 2) / 2
        s = (s + 0.5).clip(0, 1)  # normalize
        scores.append(s.fillna(0.5) * 0.15)

    if len(scores) == 0:
        return pd.Series(0.0, index=df.index)

    return sum(scores).clip(0, 1)


# ═════════════════════════════════════════════════════════
# Phase 2: 分岔突破检测
# ═════════════════════════════════════════════════════════

def detect_bifurcation_breakout(
    df: pd.DataFrame,
    is_accumulating: pd.Series,
    cfg: DetectorConfig,
) -> pd.Series:
    """
    在惜售吸筹状态之后, 检测分岔突破信号.

    分岔的物理含义:
      系统参数(资金压力)达到临界阈值 → 主特征值 → 1 →
      原有均衡不稳定 → 放量突破 → 进入新的吸引子(上升趋势)

    条件:
      1. 前一阶段处于惜售状态 (有足够的能量积累)
      2. 主特征值接近 1 (临界减速 → 即将分岔)
      3. 量能脉冲 (能量注入打破对称性)
      4. 熵仍然保持较低 (有序突破, 非噪声驱动)
    """
    conds = []

    # 前提: 近期处于惜售状态 (过去 N 天有过惜售)
    recent_accum = is_accumulating.rolling(10, min_periods=1).max() > 0
    conds.append(recent_accum)

    # 主特征值接近 1
    if "dom_eig_m" in df.columns:
        conds.append(df["dom_eig_m"] > cfg.dom_eig_threshold)

    # 量能脉冲
    if "vol_impulse" in df.columns:
        conds.append(df["vol_impulse"] > cfg.vol_impulse_threshold)

    # 熵仍低 (有序突破)
    if "perm_entropy_m" in df.columns:
        conds.append(df["perm_entropy_m"] < cfg.perm_entropy_breakout_max)

    # 5. 资金流: 大单动量为正 (突破时大单加速流入)
    if "mf_big_momentum" in df.columns:
        conds.append(df["mf_big_momentum"] > 0)

    if len(conds) < 3:
        return pd.Series(False, index=df.index)

    score = sum(c.astype(int) for c in conds)
    return score >= len(conds) - 1  # 允许一个条件不满足


def bifurcation_quality(df: pd.DataFrame, cfg: DetectorConfig) -> pd.Series:
    """分岔突破质量分数 [0, 1].

    权重分配:
      - dom_eig:                0.20  (临界减速)
      - vol_impulse:            0.20  (能量注入)
      - perm_entropy:           0.10  (有序度)
      - path_irrev:             0.10  (方向性)
      - coherence_decay_rate:   0.15  (退相干速率)
      - mf_big_momentum:        0.15  (大单资金动量)
      - mf_big_net_ratio:       0.10  (大单净额占比)
    """
    scores = []

    if "dom_eig_m" in df.columns:
        s = df["dom_eig_m"].clip(0.5, 1.0)
        s = (s - 0.5) / 0.5  # [0, 1]
        scores.append(s.fillna(0) * 0.20)

    if "vol_impulse" in df.columns:
        s = (df["vol_impulse"].clip(1, 5) - 1) / 4  # [0, 1]
        scores.append(s.fillna(0) * 0.20)

    if "perm_entropy_m" in df.columns:
        s = 1.0 - df["perm_entropy_m"].clip(0.3, 1.0) / 1.0
        scores.append(s.fillna(0) * 0.10)

    if "path_irrev_m" in df.columns:
        s = df["path_irrev_m"].clip(0, 0.5) / 0.5
        scores.append(s.fillna(0) * 0.10)

    # 退相干速率: 负值越大 = 方向确认越快 = 突破质量越高
    if "coherence_decay_rate" in df.columns:
        cdr = df["coherence_decay_rate"].fillna(0.0)
        s = (-cdr).clip(0, 0.05) / 0.05  # [0, 1]
        scores.append(s * 0.15)

    # 大单资金动量: 正值越大 = 资金加速流入
    if "mf_big_momentum" in df.columns:
        s = df["mf_big_momentum"].fillna(0)
        s_std = s.rolling(20, min_periods=5).std().replace(0, np.nan)
        s_norm = (s / s_std).clip(-2, 2)
        s_score = (s_norm + 2) / 4  # [-2,2] → [0,1]
        scores.append(s_score.fillna(0.5) * 0.15)

    # 大单净额占比: 正值越大 = 机构控盘越强
    if "mf_big_net_ratio" in df.columns:
        s = df["mf_big_net_ratio"].clip(-0.3, 0.3)
        s = (s + 0.3) / 0.6  # [0, 1]
        scores.append(s.fillna(0.5) * 0.10)

    if len(scores) == 0:
        return pd.Series(0.0, index=df.index)

    return sum(scores).clip(0, 1)


# ═════════════════════════════════════════════════════════
# Phase 3: 结构崩塌退出检测
# ═════════════════════════════════════════════════════════

def detect_structural_collapse(
    df: pd.DataFrame,
    entry_idx: int | None,
    cfg: DetectorConfig,
    min_collapse_score: int = 3,
) -> pd.Series:
    """
    检测耗散结构是否即将崩塌, 触发卖出.

    耗散结构崩塌的物理含义:
      趋势(有序结构)需要持续的能量输入(成交量 + 资金流)来维持.
      当能量输入停止 → 有序结构无法对抗熵增 → 局部熵上升 →
      耗散结构解体 → 价格趋势结束.

    信号:
      1. 置换熵快速扩张 (无序化)
      2. 路径不可逆性骤降 (主力撤离)
      3. 成交量相对高点大幅萎缩 (能量耗竭)
      4. 熵加速度为正且大 (熵在加速膨胀)
      5. 持仓时间超过安全上限
    """
    collapse_signals = pd.Series(False, index=df.index)

    # Signal 1: 置换熵飙升
    sig_entropy_high = pd.Series(False, index=df.index)
    if "perm_entropy_m" in df.columns:
        sig_entropy_high = df["perm_entropy_m"] > cfg.perm_entropy_collapse

    # Signal 2: 路径不可逆性骤降
    sig_irrev_low = pd.Series(False, index=df.index)
    if "path_irrev_m" in df.columns:
        sig_irrev_low = df["path_irrev_m"] < cfg.path_irrev_collapse

    # Signal 3: 熵加速扩张
    sig_entropy_accel = pd.Series(False, index=df.index)
    if "entropy_accel" in df.columns:
        sig_entropy_accel = df["entropy_accel"] > cfg.entropy_accel_collapse

    # Signal 4: 成交量衰竭 (相对入场后最大成交量)
    sig_vol_exhaustion = pd.Series(False, index=df.index)
    if entry_idx is not None and "vol_impulse" in df.columns:
        post_entry = df.index >= entry_idx
        peak_vol = df.loc[post_entry, "vol_impulse"].expanding().max()
        current_vs_peak = df.loc[post_entry, "vol_impulse"] / peak_vol.replace(0, np.nan)
        sig_vol_exhaustion.loc[post_entry] = current_vs_peak < cfg.vol_exhaustion_ratio

    # Signal 5: 密度矩阵纯度骤降 (共识瓦解 = 退相干完成后状态混合)
    sig_purity_collapse = pd.Series(False, index=df.index)
    if "purity_norm" in df.columns:
        sig_purity_collapse = df["purity_norm"] < cfg.purity_collapse_max

    # 综合: 至少三个信号同时触发 → 崩塌
    score = (
        sig_entropy_high.astype(int)
        + sig_irrev_low.astype(int)
        + sig_entropy_accel.astype(int)
        + sig_vol_exhaustion.astype(int)
        + sig_purity_collapse.astype(int)
    )
    collapse_signals = score >= min_collapse_score

    # 安全网: 超过最大持有天数
    if entry_idx is not None:
        days_held = pd.Series(range(len(df)), index=df.index) - entry_idx
        too_long = days_held > cfg.max_hold_days
        collapse_signals = collapse_signals | (too_long & (entry_idx <= df.index))

    return collapse_signals


# ═════════════════════════════════════════════════════════
# 综合评分
# ═════════════════════════════════════════════════════════

@dataclass
class SymbolSignal:
    """单只股票的信号输出"""
    symbol: str
    trade_date: str
    phase: str  # idle | accumulation | breakout_ready | breakout | collapse
    accum_quality: float
    bifurc_quality: float
    composite_score: float
    entry_signal: bool
    exit_signal: bool
    details: dict = field(default_factory=dict)


def evaluate_symbol(
    df_daily_featured: pd.DataFrame,
    df_weekly_featured: pd.DataFrame | None,
    symbol: str,
    cfg: DetectorConfig | None = None,
) -> SymbolSignal:
    """
    对单只股票做完整的三阶段评估, 返回最新一天的信号.
    """
    cfg = cfg or DetectorConfig()
    df = df_daily_featured

    if len(df) < 60:
        return SymbolSignal(
            symbol=symbol,
            trade_date=str(df["trade_date"].iloc[-1]) if len(df) > 0 else "",
            phase="idle",
            accum_quality=0.0,
            bifurc_quality=0.0,
            composite_score=0.0,
            entry_signal=False,
            exit_signal=False,
        )

    # Phase 1
    is_accum = detect_accumulation(df, cfg)
    aq = accumulation_quality(df, cfg)

    # Phase 2
    is_breakout = detect_bifurcation_breakout(df, is_accum, cfg)
    bq = bifurcation_quality(df, cfg)

    # 周线确认
    weekly_ok = True
    if df_weekly_featured is not None and cfg.weekly_trend_confirm:
        if "perm_entropy_m" in df_weekly_featured.columns and len(df_weekly_featured) > 0:
            last_w_entropy = df_weekly_featured["perm_entropy_m"].iloc[-1]
            if pd.notna(last_w_entropy):
                weekly_ok = last_w_entropy < cfg.weekly_perm_entropy_max

        # 周线大单净额确认 (如果有预计算周线数据)
        if cfg.weekly_big_net_positive and "weekly_big_net_cumsum" in df_weekly_featured.columns:
            last_w_mf = df_weekly_featured["weekly_big_net_cumsum"].iloc[-1]
            if pd.notna(last_w_mf) and last_w_mf < 0:
                weekly_ok = False  # 周线大单持续流出, 不确认

    # 分钟线微观确认
    last = len(df) - 1
    minute_ok = True
    if "intraday_perm_entropy" in df.columns:
        intra_pe = df["intraday_perm_entropy"].iloc[last]
        if pd.notna(intra_pe) and intra_pe > cfg.intraday_entropy_low:
            minute_ok = False  # 日内价格无序, 不确认

    # 最新状态
    today_accum = bool(is_accum.iloc[last]) if last < len(is_accum) else False
    today_breakout = bool(is_breakout.iloc[last]) if last < len(is_breakout) else False

    # 判断阶段
    if today_breakout and weekly_ok and minute_ok:
        phase = "breakout"
    elif today_accum:
        phase = "accumulation"
    else:
        phase = "idle"

    # 入场信号: 处于 breakout 阶段 + 周线确认
    entry = phase == "breakout"

    # 综合评分
    aq_last = float(aq.iloc[last]) if last < len(aq) else 0.0
    bq_last = float(bq.iloc[last]) if last < len(bq) else 0.0
    composite = 0.4 * aq_last + 0.6 * bq_last

    # 收集详情
    details = {}
    for col in ["perm_entropy_m", "path_irrev_m", "dom_eig_m", "vol_impulse",
                 "entropy_accel",
                 "coherence_l1", "purity_norm", "coherence_decay_rate",
                 "von_neumann_entropy",
                 "intraday_perm_entropy", "intraday_path_irrev",
                 "intraday_vol_concentration", "intraday_range_ratio",
                 "mf_big_net_ratio", "mf_big_cumsum_s", "mf_big_cumsum_m",
                 "mf_flow_imbalance", "mf_big_streak", "mf_big_momentum"]:
        if col in df.columns:
            v = df[col].iloc[last]
            details[col] = round(float(v), 4) if pd.notna(v) else None

    return SymbolSignal(
        symbol=symbol,
        trade_date=str(df["trade_date"].iloc[last]),
        phase=phase,
        accum_quality=round(aq_last, 4),
        bifurc_quality=round(bq_last, 4),
        composite_score=round(composite, 4),
        entry_signal=entry,
        exit_signal=False,  # 退出在持仓管理中判断
        details=details,
    )
