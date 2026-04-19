"""
因子模型选股 — Agent 2: 因子状态机选股 (v2)

三阶段状态机替代 LightGBM 回归:
  Phase 1: 蓄力吸筹 (Accumulation) — 6 条件取 >=3, 连续 >=3 天
  Phase 2: 量价突破 (Breakout) — 蓄力后量能释放 + 价格突破
  (Phase 3: 退出在 Agent 3 中执行)

选股逻辑:
  1. 用 factor_snapshot 快速预筛 (~5100 -> ~1000-2000)
  2. 对预筛股票加载近 60 天因子时序, 判断当前状态
  3. Breakout 优先, Accumulation 备选, 按复合评分排序
  4. 所有 horizon 共用同一选股结果 (状态机不区分持有周期)

条件体系来自 signal_detector_v2 (数据驱动阈值):
  蓄力: mf_sm_proportion, vol_shrink, breakout_range,
         mf_big_cumsum_s, mf_flow_imbalance, path_irrev_m
  突破: vol_impulse, breakout_range, mf_sm_proportion,
         volatility_l, bbw_pctl
  周线: pb_pctl, weekly_turnover_ma4
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
# 状态机阈值配置
# ═══════════════════════════════════════════════════════

@dataclass
class StateConfig:
    """状态机阈值 — 基于 signal_detector_v2 数据驱动阈值。"""

    # ── Phase 1: 蓄力吸筹 ──
    mf_sm_proportion_min: float = 0.45    # 散户占比高 (IC=+0.086)
    vol_shrink_max: float = 0.70          # 缩量 (IC=-0.047)
    breakout_range_max: float = 0.50      # 价格贴中轨
    mf_big_cumsum_s_positive: bool = True # 短期大单累计为正
    mf_flow_imbalance_min: float = 0.0    # 大单买散户卖
    path_irrev_min: float = 0.02          # 有定向力量
    accum_min_days: int = 3               # 连续天数
    accum_min_conditions: int = 3         # 满足条件数 (共 6)

    # ── Phase 2: 量价突破 ──
    vol_impulse_min: float = 1.3          # 量能脉冲 > P75
    breakout_range_breakout_min: float = 0.80  # 价格突破布林上轨
    mf_sm_proportion_breakout: float = 0.50    # 散户追涨
    volatility_l_max: float = 0.05        # 排除暴涨暴跌
    bbw_pctl_min: float = 0.30            # 波动正在扩张
    breakout_min_conditions: int = 3      # 满足条件数
    recent_accum_lookback: int = 10       # 近 N 天有过蓄力

    # ── 周线确认 ──
    weekly_confirm: bool = True
    weekly_pb_pctl_min: float = 0.20
    weekly_turnover_ma4_max: float = 30.0

    # ── 预筛宽松阈值 ──
    prefilter_mf_sm_min: float = 0.30     # 宽于正式阈值
    prefilter_vol_shrink_max: float = 1.2

    # ── 通用 ──
    lookback_days: int = 60               # 加载因子时序长度


# ═══════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════

def run_selection(
    factor_snapshot: pd.DataFrame,
    cache_dir: str,
    scan_date: str,
    top_n: int = 5,
    horizons: list[str] | None = None,
    cfg: StateConfig | None = None,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """
    因子状态机选股。

    Args:
        factor_snapshot: Agent 1 输出的因子截面 (index=symbol)
        cache_dir: 特征缓存根目录
        scan_date: 选股日期
        top_n: 选几只
        horizons: 活跃 horizon 列表
        cfg: 状态机配置
        **kwargs: 兼容旧调用 (train_cutoff 等)

    Returns:
        {horizon: DataFrame with [symbol, name, industry, composite_score,
                                   rank, phase, accum_quality, breakout_quality]}
    """
    cfg = cfg or StateConfig()
    horizons = horizons or ["5d"]

    # ── Step 1: 预筛 ──
    universe_size = len(factor_snapshot)
    pre_filtered = _prefilter(factor_snapshot, cfg)
    logger.info(f"预筛: {universe_size} -> {len(pre_filtered)} 只")

    # ── Step 2: 逐股状态检测 ──
    candidates = []
    for sym in pre_filtered:
        result = _evaluate_stock(sym, cache_dir, scan_date, cfg, factor_snapshot)
        if result is not None:
            candidates.append(result)

    breakouts = [c for c in candidates if c["phase"] == "breakout"]
    accums = [c for c in candidates if c["phase"] == "accumulation"]
    logger.info(f"状态检测: {len(breakouts)} 只突破, {len(accums)} 只蓄力 "
                f"(共 {len(candidates)} 只有状态)")

    # ── Step 3: 排序选取 ──
    breakouts.sort(key=lambda x: x["composite_score"], reverse=True)
    accums.sort(key=lambda x: x["accum_quality"], reverse=True)

    selected = breakouts[:top_n]
    if len(selected) < top_n:
        remaining = top_n - len(selected)
        selected.extend(accums[:remaining])

    # ── Step 4: 构建输出 ──
    if not selected:
        logger.warning("Agent 2: 无候选股票")
        return {h: pd.DataFrame() for h in horizons}

    for i, s in enumerate(selected, 1):
        s["rank"] = i

    result_df = pd.DataFrame(selected)

    # 所有 horizon 共用同一选股结果
    results = {}
    for h in horizons:
        df = result_df.copy()
        df["horizon"] = h
        results[h] = df

    phases = [s["phase"] for s in selected]
    n_bo = sum(1 for p in phases if p == "breakout")
    n_ac = sum(1 for p in phases if p == "accumulation")
    logger.info(f"Agent 2 完成: {len(selected)} 只 ({n_bo} 突破 + {n_ac} 蓄力), "
                f"scan_date={scan_date}")

    return results


# ═══════════════════════════════════════════════════════
# 预筛
# ═══════════════════════════════════════════════════════

def _prefilter(snapshot: pd.DataFrame, cfg: StateConfig) -> list[str]:
    """快速预筛: 用 snapshot 单日值做宽松过滤。"""
    accum_mask = pd.Series(True, index=snapshot.index)

    if "mf_sm_proportion" in snapshot.columns:
        accum_mask = accum_mask & (snapshot["mf_sm_proportion"] > cfg.prefilter_mf_sm_min)
    if "vol_shrink" in snapshot.columns:
        accum_mask = accum_mask & (snapshot["vol_shrink"] < cfg.prefilter_vol_shrink_max)

    # 也包括可能处于突破状态的
    breakout_mask = pd.Series(False, index=snapshot.index)
    if "breakout_range" in snapshot.columns and "vol_impulse" in snapshot.columns:
        br_cond = snapshot["breakout_range"] > cfg.breakout_range_breakout_min * 0.6
        vi_cond = snapshot["vol_impulse"] > cfg.vol_impulse_min * 0.8
        breakout_mask = br_cond & vi_cond

    combined = accum_mask | breakout_mask
    return snapshot[combined].index.tolist()


# ═══════════════════════════════════════════════════════
# 逐股评估
# ═══════════════════════════════════════════════════════

def _evaluate_stock(
    symbol: str,
    cache_dir: str,
    scan_date: str,
    cfg: StateConfig,
    snapshot: pd.DataFrame,
) -> dict | None:
    """对单只股票做完整状态评估。"""
    daily_path = os.path.join(cache_dir, "daily", f"{symbol}.csv")
    if not os.path.exists(daily_path):
        return None

    try:
        df = pd.read_csv(daily_path)
        df["trade_date"] = df["trade_date"].astype(str)
    except Exception:
        return None

    df = df[df["trade_date"] <= scan_date].tail(cfg.lookback_days).reset_index(drop=True)
    if len(df) < 20:
        return None

    # Phase 1
    is_accum = _detect_accumulation(df, cfg)
    accum_quality = _accumulation_quality(df, cfg)

    # Phase 2
    is_breakout = _detect_breakout(df, is_accum, cfg)
    breakout_quality = _breakout_quality(df, cfg)

    last = len(df) - 1
    today_accum = bool(is_accum.iloc[last])
    today_breakout = bool(is_breakout.iloc[last])

    if not today_breakout and not today_accum:
        return None

    # 周线确认 (仅突破需要)
    weekly_ok = True
    if cfg.weekly_confirm and today_breakout:
        weekly_ok = _weekly_confirm(symbol, cache_dir, scan_date, cfg)

    if today_breakout and weekly_ok:
        phase = "breakout"
    elif today_accum:
        phase = "accumulation"
    else:
        return None

    aq = float(accum_quality.iloc[last])
    bq = float(breakout_quality.iloc[last])
    composite = 0.4 * aq + 0.6 * bq

    # name / industry
    name = ""
    industry = ""
    if symbol in snapshot.index:
        row = snapshot.loc[symbol]
        name = row.get("_name", "") if "_name" in snapshot.columns else ""
        industry = row.get("_industry", "") if "_industry" in snapshot.columns else ""

    # 当日因子详情
    details = {}
    for col in ["perm_entropy_m", "path_irrev_m", "dom_eig_m", "vol_impulse",
                 "mf_sm_proportion", "breakout_range", "vol_shrink",
                 "volatility_l", "bbw_pctl", "mf_flow_imbalance",
                 "mf_big_cumsum_s", "mf_big_momentum"]:
        if col in df.columns:
            v = df[col].iloc[last]
            details[col] = round(float(v), 4) if pd.notna(v) else None

    return {
        "symbol": symbol,
        "name": name,
        "industry": industry,
        "phase": phase,
        "accum_quality": round(aq, 4),
        "breakout_quality": round(bq, 4),
        "composite_score": round(composite, 4),
        "details": details,
    }


# ═══════════════════════════════════════════════════════
# Phase 1: 蓄力吸筹检测
# ═══════════════════════════════════════════════════════

def _detect_accumulation(df: pd.DataFrame, cfg: StateConfig) -> pd.Series:
    """蓄力吸筹: 6 条件取 >=3, 连续 >=3 天。"""
    conds = []

    if "mf_sm_proportion" in df.columns:
        conds.append(df["mf_sm_proportion"] > cfg.mf_sm_proportion_min)
    else:
        return pd.Series(False, index=df.index)

    if "vol_shrink" in df.columns:
        conds.append(df["vol_shrink"] < cfg.vol_shrink_max)

    if "breakout_range" in df.columns:
        conds.append(df["breakout_range"] < cfg.breakout_range_max)

    if "mf_big_cumsum_s" in df.columns and cfg.mf_big_cumsum_s_positive:
        conds.append(df["mf_big_cumsum_s"] > 0)

    if "mf_flow_imbalance" in df.columns:
        conds.append(df["mf_flow_imbalance"] > cfg.mf_flow_imbalance_min)

    if "path_irrev_m" in df.columns:
        conds.append(df["path_irrev_m"] > cfg.path_irrev_min)

    score = sum(c.astype(int) for c in conds)
    raw_signal = score >= cfg.accum_min_conditions

    sustained = raw_signal.rolling(
        cfg.accum_min_days, min_periods=cfg.accum_min_days
    ).sum()
    return sustained >= cfg.accum_min_days


def _accumulation_quality(df: pd.DataFrame, cfg: StateConfig) -> pd.Series:
    """蓄力质量评分 [0, 1]。"""
    scores = []

    if "mf_sm_proportion" in df.columns:
        s = df["mf_sm_proportion"].clip(0.2, 0.7)
        s = (s - 0.2) / 0.5
        scores.append(s.fillna(0) * 0.25)

    if "vol_shrink" in df.columns:
        s = 1.0 - df["vol_shrink"].clip(0.3, 1.5) / 1.5
        scores.append(s.fillna(0) * 0.20)

    if "breakout_range" in df.columns:
        s = 1.0 - df["breakout_range"].clip(0, 1.0)
        scores.append(s.fillna(0) * 0.20)

    if "mf_flow_imbalance" in df.columns:
        s = (df["mf_flow_imbalance"].clip(-2, 2) + 2) / 4
        scores.append(s.fillna(0.5) * 0.15)

    if "path_irrev_m" in df.columns:
        s = df["path_irrev_m"].clip(0, 0.3) / 0.3
        scores.append(s.fillna(0) * 0.10)

    if "perm_entropy_m" in df.columns:
        s = 1.0 - df["perm_entropy_m"].clip(0.5, 1.0) / 1.0
        scores.append(s.fillna(0) * 0.10)

    if len(scores) == 0:
        return pd.Series(0.0, index=df.index)

    return sum(scores).clip(0, 1)


# ═══════════════════════════════════════════════════════
# Phase 2: 量价突破检测
# ═══════════════════════════════════════════════════════

def _detect_breakout(
    df: pd.DataFrame,
    is_accumulating: pd.Series,
    cfg: StateConfig,
) -> pd.Series:
    """量价突破: 前置蓄力 + 量能放大 + 价格突破。"""
    conds = []

    recent_accum = is_accumulating.rolling(
        cfg.recent_accum_lookback, min_periods=1
    ).max() > 0
    conds.append(recent_accum)

    if "vol_impulse" in df.columns:
        conds.append(df["vol_impulse"] > cfg.vol_impulse_min)

    if "breakout_range" in df.columns:
        conds.append(df["breakout_range"] > cfg.breakout_range_breakout_min)

    if "mf_sm_proportion" in df.columns:
        conds.append(df["mf_sm_proportion"] > cfg.mf_sm_proportion_breakout)

    if "volatility_l" in df.columns:
        conds.append(df["volatility_l"] < cfg.volatility_l_max)

    if "bbw_pctl" in df.columns:
        conds.append(df["bbw_pctl"] > cfg.bbw_pctl_min)

    if len(conds) < 3:
        return pd.Series(False, index=df.index)

    # 硬性: 量能 + 价格突破 必须同时满足
    hard = pd.Series(True, index=df.index)
    if "vol_impulse" in df.columns:
        hard = hard & (df["vol_impulse"] > cfg.vol_impulse_min)
    if "breakout_range" in df.columns:
        hard = hard & (df["breakout_range"] > cfg.breakout_range_breakout_min)

    score = sum(c.astype(int) for c in conds)
    return (score >= cfg.breakout_min_conditions) & hard


def _breakout_quality(df: pd.DataFrame, cfg: StateConfig) -> pd.Series:
    """突破质量评分 [0, 1]。"""
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


# ═══════════════════════════════════════════════════════
# 周线确认
# ═══════════════════════════════════════════════════════

def _weekly_confirm(
    symbol: str,
    cache_dir: str,
    scan_date: str,
    cfg: StateConfig,
) -> bool:
    """周线确认: PB百分位 + 周均换手。无数据不否决。"""
    weekly_path = os.path.join(cache_dir, "weekly", f"{symbol}.csv")
    if not os.path.exists(weekly_path):
        return True

    try:
        wdf = pd.read_csv(weekly_path)
        wdf["trade_date"] = wdf["trade_date"].astype(str)
        wdf = wdf[wdf["trade_date"] <= scan_date]
        if wdf.empty:
            return True

        last = wdf.iloc[-1]

        if "pb_pctl" in wdf.columns and pd.notna(last.get("pb_pctl")):
            if last["pb_pctl"] < cfg.weekly_pb_pctl_min:
                return False

        if "weekly_turnover_ma4" in wdf.columns and pd.notna(last.get("weekly_turnover_ma4")):
            if last["weekly_turnover_ma4"] > cfg.weekly_turnover_ma4_max:
                return False

        return True
    except Exception:
        return True
