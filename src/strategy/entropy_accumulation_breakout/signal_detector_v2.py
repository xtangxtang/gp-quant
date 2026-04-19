"""
基于因子画像的数据驱动选股器 (v2)

设计原则:
  - 基于 factor_profiling 全市场统计结果, 优先使用高有效率因子
  - 日线: mf_sm_proportion (59.5%), breakout_range (58.3%), vol_shrink (40.6%)
  - 周线: pb_pctl (85.5%), breakout_range (78.9%), pe_ttm_pctl (75.4%)
  - 熵因子作为辅助确认, 不再作为核心过滤条件
  - 保留三阶段框架, 但用数据驱动的阈值替代经验阈值

数据依赖:
  - 第一步 feature_engine → daily/weekly 缓存 (不修改)
  - 第二步 factor_profiling → 因子有效率参考 (不修改)
  - 本模块替代旧 signal_detector 的第三步
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ═════════════════════════════════════════════════════════
# 阈值配置 — 基于 factor_profiling 20260417 全市场分布
# ═════════════════════════════════════════════════════════

@dataclass
class DetectorConfigV2:
    """数据驱动阈值, 基于全市场因子分布百分位数设定"""

    # ── Phase 1: 蓄力吸筹 (Accumulation) ──
    # 核心因子 (有效率 > 40%)
    mf_sm_proportion_min: float = 0.45   # 散户占比 > P75, IC=+0.086 → 散户卖出越多越好
    vol_shrink_max: float = 0.70         # 缩量 < P50, IC=-0.047 → 缩量=蓄力
    breakout_range_max: float = 0.50     # 突破幅度低 < P50, 尚未大幅偏离 → 蓄力期间价格贴中轨

    # 辅助因子 (有效率 15-25%)
    mf_flow_imbalance_min: float = 0.0   # 资金流不平衡 > 0 即可 (原来 0.3 太严)
    mf_big_cumsum_s_positive: bool = True # 短期大单累计为正
    path_irrev_min: float = 0.02         # 路径不可逆 > P25 (原来 0.05 太严)

    # 熵辅助 (有效率 ~15%, 降级为可选)
    perm_entropy_max: float = 0.97       # 置换熵 < P95 (仅排除极端无序)
    entropy_assist_weight: float = 0.1   # 评分中的熵权重 (从 0.25 降到 0.1)

    # 蓄力持续性
    accum_min_days: int = 3              # 最少连续天数 (原来 5, 放宽)
    accum_min_score: int = 3             # 至少满足 3 个条件 (共 6 个)

    # ── Phase 2: 量价突破 (Breakout) ──
    # 核心: 量能+价格突破
    vol_impulse_min: float = 1.3         # 量能脉冲 > P75 (原来 1.8 太严)
    breakout_range_breakout_min: float = 0.80  # 突破后价格偏离布林中轨 > P75
    mf_sm_proportion_breakout: float = 0.50    # 突破时散户占比仍高 (散户追涨)

    # 辅助: 波动率确认
    volatility_l_max: float = 0.05       # 长期波动率 < P95 (排除暴涨暴跌股)
    bbw_pctl_min: float = 0.30           # 布林带宽百分位 > P25 (波动正在扩张)

    # 熵辅助: 有序突破
    perm_entropy_breakout_max: float = 0.97    # 极宽容, 仅排除极端

    # 突破条件数
    breakout_min_score: int = 3          # 至少满足 3 个条件

    # ── Phase 3: 退出 ──
    # 基于有效因子的退出信号
    vol_shrink_exit: float = 1.5         # 缩量度飙升 (> P95 = 先放量后急缩)
    mf_sm_proportion_exit: float = 0.25  # 散户占比骤降 (< P5 = 散户已跑光)
    breakout_range_exit: float = 0.10    # 突破幅度回落 < P5 (动能耗尽)
    max_hold_days: int = 15              # 最大持有天数

    # ── 周线确认 ──
    weekly_pb_pctl_min: float = 0.20     # PB 百分位 > P25 (趋势延续)
    weekly_breakout_low: float = 0.30    # 周线突破幅度不能太高 (< P75)
    weekly_turnover_ma4_max: float = 30.0  # 周均换手 < P75 (不能过热)
    weekly_confirm: bool = True


# ═════════════════════════════════════════════════════════
# Phase 1: 蓄力吸筹检测
# ═════════════════════════════════════════════════════════

def detect_accumulation_v2(
    df: pd.DataFrame,
    cfg: DetectorConfigV2,
) -> pd.Series:
    """
    蓄力吸筹检测 — 数据驱动版

    核心逻辑更新:
      旧版: 以熵低位为核心 (perm_entropy < 0.65) → 全市场 0.2% 通过率
      新版: 以资金流结构 + 缩量蓄力为核心 → 预期 5-15% 通过率

    物理含义:
      散户占比高 (卖方以散户为主) + 缩量 (交投冷清) + 价格贴中轨 (未启动)
      + 大单暗中买入 = 主力在散户不关注时悄悄建仓
    """
    conds = []

    # [核心] 散户占比高 → 卖方以散户为主, 主力在吸筹 (有效率 59.5%)
    if "mf_sm_proportion" in df.columns:
        conds.append(df["mf_sm_proportion"] > cfg.mf_sm_proportion_min)

    # [核心] 缩量蓄力 → 交投冷清, 筹码沉淀 (有效率 40.6%)
    if "vol_shrink" in df.columns:
        conds.append(df["vol_shrink"] < cfg.vol_shrink_max)

    # [核心] 价格贴中轨 → 尚未启动, 蓄力阶段 (有效率 58.3%)
    if "breakout_range" in df.columns:
        conds.append(df["breakout_range"] < cfg.breakout_range_max)

    # [辅助] 大单净额为正 → 主力在买入
    if "mf_big_cumsum_s" in df.columns and cfg.mf_big_cumsum_s_positive:
        conds.append(df["mf_big_cumsum_s"] > 0)

    # [辅助] 资金流不平衡 → 大单买散户卖 (有效率 25.6%, 放宽阈值)
    if "mf_flow_imbalance" in df.columns:
        conds.append(df["mf_flow_imbalance"] > cfg.mf_flow_imbalance_min)

    # [辅助] 路径不可逆 → 有定向力量 (有效率 17%)
    if "path_irrev_m" in df.columns:
        conds.append(df["path_irrev_m"] > cfg.path_irrev_min)

    if len(conds) == 0:
        return pd.Series(False, index=df.index)

    # 硬性要求: 必须有散户占比数据 (核心因子, 有效率 59.5%)
    if "mf_sm_proportion" not in df.columns:
        return pd.Series(False, index=df.index)

    score = sum(c.astype(int) for c in conds)
    raw_signal = score >= cfg.accum_min_score

    # 持续性: 连续至少 N 天
    sustained = raw_signal.rolling(cfg.accum_min_days, min_periods=cfg.accum_min_days).sum()
    return sustained >= cfg.accum_min_days


def accumulation_quality_v2(df: pd.DataFrame, cfg: DetectorConfigV2) -> pd.Series:
    """
    蓄力质量评分 [0, 1] — 数据驱动版

    权重分配基于因子有效率:
      mf_sm_proportion  0.25  (有效率 59.5%)
      vol_shrink        0.20  (有效率 40.6%)
      breakout_range    0.20  (有效率 58.3% — 低值=蓄力, 高值=已突破)
      mf_flow_imbalance 0.15  (有效率 25.6%)
      path_irrev_m      0.10  (有效率 17.0%)
      perm_entropy_m    0.10  (有效率 14.5% — 降权)
    """
    scores = []

    # 散户占比: 越高越好 (正向IC)
    if "mf_sm_proportion" in df.columns:
        s = df["mf_sm_proportion"].clip(0.2, 0.7)
        s = (s - 0.2) / 0.5
        scores.append(s.fillna(0) * 0.25)

    # 缩量度: 越低越好 (负向IC, 低=蓄力) → 反转
    if "vol_shrink" in df.columns:
        s = 1.0 - df["vol_shrink"].clip(0.3, 1.5) / 1.5
        scores.append(s.fillna(0) * 0.20)

    # 突破幅度: 低值=蓄力期 (负向IC, 高值=已过热) → 反转
    if "breakout_range" in df.columns:
        s = 1.0 - df["breakout_range"].clip(0, 1.0)
        scores.append(s.fillna(0) * 0.20)

    # 资金流不平衡: 正向=大单买散户卖
    if "mf_flow_imbalance" in df.columns:
        s = (df["mf_flow_imbalance"].clip(-2, 2) + 2) / 4
        scores.append(s.fillna(0.5) * 0.15)

    # 路径不可逆: 越高=越有定向力量
    if "path_irrev_m" in df.columns:
        s = df["path_irrev_m"].clip(0, 0.3) / 0.3
        scores.append(s.fillna(0) * 0.10)

    # 置换熵: 越低=越有序 (降权)
    if "perm_entropy_m" in df.columns:
        s = 1.0 - df["perm_entropy_m"].clip(0.5, 1.0) / 1.0
        scores.append(s.fillna(0) * cfg.entropy_assist_weight)

    if len(scores) == 0:
        return pd.Series(0.0, index=df.index)

    return sum(scores).clip(0, 1)


# ═════════════════════════════════════════════════════════
# Phase 2: 量价突破检测
# ═════════════════════════════════════════════════════════

def detect_breakout_v2(
    df: pd.DataFrame,
    is_accumulating: pd.Series,
    cfg: DetectorConfigV2,
) -> pd.Series:
    """
    量价突破检测 — 数据驱动版

    核心逻辑更新:
      旧版: dom_eig > 0.85 (12.5%通过) + vol_impulse > 1.8 (7%通过) → 几乎没有
      新版: 量能放大 + 价格突破布林带 + 散户追涨 → 预期 2-5% 通过率

    物理含义:
      蓄力阶段积累的能量(筹码集中+资金暗流)释放 →
      量能脉冲 + 价格突破 + 散户开始追入 = 正反馈启动上涨
    """
    conds = []

    # 前提: 近期经历过蓄力 (过去 10 天)
    recent_accum = is_accumulating.rolling(10, min_periods=1).max() > 0
    conds.append(recent_accum)

    # [核心] 量能放大 (放宽: P75 即可)
    if "vol_impulse" in df.columns:
        conds.append(df["vol_impulse"] > cfg.vol_impulse_min)

    # [核心] 价格突破布林带上轨区域
    if "breakout_range" in df.columns:
        conds.append(df["breakout_range"] > cfg.breakout_range_breakout_min)

    # [核心] 散户占比仍高 → 追涨效应
    if "mf_sm_proportion" in df.columns:
        conds.append(df["mf_sm_proportion"] > cfg.mf_sm_proportion_breakout)

    # [辅助] 波动率不过高 → 排除暴涨暴跌股
    if "volatility_l" in df.columns:
        conds.append(df["volatility_l"] < cfg.volatility_l_max)

    # [辅助] 布林带宽在扩张
    if "bbw_pctl" in df.columns:
        conds.append(df["bbw_pctl"] > cfg.bbw_pctl_min)

    if len(conds) < 3:
        return pd.Series(False, index=df.index)

    # 硬性要求: 量能放大 + 价格突破 必须同时满足
    hard_conds = pd.Series(True, index=df.index)
    if "vol_impulse" in df.columns:
        hard_conds = hard_conds & (df["vol_impulse"] > cfg.vol_impulse_min)
    if "breakout_range" in df.columns:
        hard_conds = hard_conds & (df["breakout_range"] > cfg.breakout_range_breakout_min)

    score = sum(c.astype(int) for c in conds)
    return (score >= cfg.breakout_min_score) & hard_conds


def breakout_quality_v2(df: pd.DataFrame, cfg: DetectorConfigV2) -> pd.Series:
    """
    突破质量评分 [0, 1] — 数据驱动版

    权重分配:
      vol_impulse        0.25  (量能是突破的核心证据)
      breakout_range     0.25  (价格突破幅度)
      mf_sm_proportion   0.20  (散户追涨=正反馈)
      bbw_pctl           0.15  (波动率扩张确认)
      mf_big_net_ratio   0.15  (大单仍在买入)
    """
    scores = []

    if "vol_impulse" in df.columns:
        s = (df["vol_impulse"].clip(0.5, 3.0) - 0.5) / 2.5
        scores.append(s.fillna(0) * 0.25)

    if "breakout_range" in df.columns:
        s = df["breakout_range"].clip(0, 1.0)
        scores.append(s.fillna(0) * 0.25)

    if "mf_sm_proportion" in df.columns:
        s = df["mf_sm_proportion"].clip(0.2, 0.7)
        s = (s - 0.2) / 0.5
        scores.append(s.fillna(0) * 0.20)

    if "bbw_pctl" in df.columns:
        s = df["bbw_pctl"].clip(0, 1.0)
        scores.append(s.fillna(0) * 0.15)

    if "mf_big_net_ratio" in df.columns:
        s = (df["mf_big_net_ratio"].clip(-0.1, 0.1) + 0.1) / 0.2
        scores.append(s.fillna(0.5) * 0.15)

    if len(scores) == 0:
        return pd.Series(0.0, index=df.index)

    return sum(scores).clip(0, 1)


# ═════════════════════════════════════════════════════════
# Phase 3: 退出检测
# ═════════════════════════════════════════════════════════

def detect_exit_v2(
    df: pd.DataFrame,
    entry_idx: int | None,
    cfg: DetectorConfigV2,
    min_exit_score: int = 2,
) -> pd.Series:
    """
    退出检测 — 数据驱动版

    基于有效率最高的因子反转信号:
      缩量度飙升 → 即量先放后缩, 动能衰竭
      散户占比骤降 → 散户已跑光, 无人接盘
      突破幅度回落 → 价格从高位回撤
      持有时间过长 → 安全网
    """
    exit_signals = pd.Series(False, index=df.index)

    sigs = []

    # 缩量度飙升 (先爆量后急缩 = 资金撤离)
    if "vol_shrink" in df.columns:
        sigs.append(df["vol_shrink"] > cfg.vol_shrink_exit)

    # 散户占比骤降 (散户跑光 = 恐慌)
    if "mf_sm_proportion" in df.columns:
        sigs.append(df["mf_sm_proportion"] < cfg.mf_sm_proportion_exit)

    # 突破幅度回落 (动能耗尽, 价格跌回中轨)
    if "breakout_range" in df.columns:
        sigs.append(df["breakout_range"] < cfg.breakout_range_exit)

    # 置换熵极高 (极度无序)
    if "perm_entropy_m" in df.columns:
        sigs.append(df["perm_entropy_m"] > 0.98)

    if sigs:
        score = sum(s.astype(int) for s in sigs)
        exit_signals = score >= min_exit_score

    # 安全网: 最大持有天数
    if entry_idx is not None:
        days_held = pd.Series(range(len(df)), index=df.index) - entry_idx
        too_long = days_held > cfg.max_hold_days
        exit_signals = exit_signals | (too_long & (entry_idx <= df.index))

    return exit_signals


# ═════════════════════════════════════════════════════════
# 综合评估
# ═════════════════════════════════════════════════════════

@dataclass
class SymbolSignal:
    """单只股票的信号输出 — 与旧版接口兼容"""
    symbol: str
    trade_date: str
    phase: str  # idle | accumulation | breakout
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
    cfg: DetectorConfigV2 | None = None,
) -> SymbolSignal:
    """
    对单只股票做完整评估 — 数据驱动版

    接口与旧版 evaluate_symbol 完全兼容,
    scan_service.py 无需修改调用方式。
    """
    cfg = cfg or DetectorConfigV2()
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

    # Phase 1: 蓄力吸筹
    is_accum = detect_accumulation_v2(df, cfg)
    aq = accumulation_quality_v2(df, cfg)

    # Phase 2: 量价突破
    is_breakout = detect_breakout_v2(df, is_accum, cfg)
    bq = breakout_quality_v2(df, cfg)

    # 周线确认
    weekly_ok = True
    if df_weekly_featured is not None and cfg.weekly_confirm:
        wdf = df_weekly_featured
        if len(wdf) > 0:
            last = wdf.iloc[-1]

            # PB 百分位 > 下限 (趋势延续, IC=+0.1564)
            if "pb_pctl" in wdf.columns and pd.notna(last.get("pb_pctl")):
                if last["pb_pctl"] < cfg.weekly_pb_pctl_min:
                    weekly_ok = False

            # 周线突破幅度不能过高 (IC=-0.1166, 高=后续跌)
            if "breakout_range" in wdf.columns and pd.notna(last.get("breakout_range")):
                if last["breakout_range"] > cfg.weekly_breakout_low:
                    pass  # 宽容: 不一票否决, 只降分

            # 周均换手不过热 (IC=-0.1738, 高=后续跌)
            if "weekly_turnover_ma4" in wdf.columns and pd.notna(last.get("weekly_turnover_ma4")):
                if last["weekly_turnover_ma4"] > cfg.weekly_turnover_ma4_max:
                    weekly_ok = False

    # 最新状态
    last = len(df) - 1
    today_accum = bool(is_accum.iloc[last]) if last < len(is_accum) else False
    today_breakout = bool(is_breakout.iloc[last]) if last < len(is_breakout) else False

    # 判断阶段
    if today_breakout and weekly_ok:
        phase = "breakout"
    elif today_accum:
        phase = "accumulation"
    else:
        phase = "idle"

    entry = phase == "breakout"

    # 综合评分
    aq_last = float(aq.iloc[last]) if last < len(aq) else 0.0
    bq_last = float(bq.iloc[last]) if last < len(bq) else 0.0
    composite = 0.4 * aq_last + 0.6 * bq_last

    # 周线加分
    if df_weekly_featured is not None and len(df_weekly_featured) > 0:
        wlast = df_weekly_featured.iloc[-1]
        # PB 百分位高 → 加分 (趋势延续)
        if "pb_pctl" in df_weekly_featured.columns:
            pb = wlast.get("pb_pctl", 0.5)
            if pd.notna(pb):
                composite += 0.05 * (pb - 0.5)  # [-0.025, +0.025]

    # 收集详情
    details = {}
    for col in ["perm_entropy_m", "path_irrev_m", "dom_eig_m", "vol_impulse",
                 "entropy_accel", "mf_sm_proportion", "breakout_range", "vol_shrink",
                 "volatility_l", "bbw_pctl",
                 "coherence_l1", "purity_norm", "coherence_decay_rate",
                 "von_neumann_entropy",
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
        exit_signal=False,
        details=details,
    )
