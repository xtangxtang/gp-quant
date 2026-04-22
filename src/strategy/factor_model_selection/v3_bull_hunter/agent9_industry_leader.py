"""
Bull Hunter v11-A — Agent 9: Industry Leader Signal (行业趋势 + 龙头独立通道)

动机:
  V10 的 200% 模型架构天然排斥已涨过的高位龙头, 导致 CPO/固态/存储 三大行情
  的真正主线 (中际旭创/新易盛/利元亨/佰维存储 等) 始终未进入 Agent 3 候选.

策略:
  纯规则引擎, 不依赖 prob_200 模型, 平行于 Agent 3 输出, 在 Agent 5 决策
  前与模型候选合并.

  两阶段筛选:
    阶段 1 ── 识别热门行业 (利用 V10 行业因子):
      - industry_rs_20d ≥ +4%        (相对市场超额)
      - industry_breadth_5d ≥ 0.55   (行业普涨)
      - industry_mom_20d ≥ +5%       (行业中期向上)
      - industry_vol_surge ≥ 1.1     (行业放量)
      - 行业内股票数 ≥ 4             (避免小池子噪音)
      → 取 RS 排名前 N 的热门行业池

    阶段 2 ── 在热门行业内挑龙头:
      - 个股 momentum_20d ≥ +10%     (强势股)
      - 个股 momentum_5d ≥ 0         (近期不弱)
      - 个股 close_vs_high_60d ≥ 0.85 且 ≤ 1.05  (突破区间, 不追疯狂)
      - 个股 vol_ratio_s ≥ 1.2       (放量)
      - 个股 momentum_20d - industry_mom_20d ≥ +5%  (跑赢行业中位)
      - path_irrev_m ≥ 0             (主力意图非负)
      → 加权评分排序 → 取 Top N (每行业最多 max_per_industry 只)

  绕开 Agent 8 prob_200 过滤; 仍走 Agent 5 仓位/止损管理.

输出:
  与 Agent 3 同列结构, 额外字段:
    - source: 'industry_leader' / 'model'
    - il_score: 龙头评分 (0~1)
    - il_reason: 触发条件文本
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── 默认行业因子列名 (与 agent1_factor.INDUSTRY_FACTORS 保持一致) ──
INDUSTRY_FACTOR_COLS = [
    "industry_mom_5d",
    "industry_mom_20d",
    "industry_breadth_5d",
    "industry_rs_20d",
    "industry_vol_surge",
]


@dataclass
class IndustryLeaderConfig:
    # ── 阶段 1: 行业筛选 ──
    min_industry_rs_20d: float = 0.04
    min_industry_breadth_5d: float = 0.55
    min_industry_mom_20d: float = 0.05
    min_industry_vol_surge: float = 1.10
    min_stocks_per_industry: int = 4
    top_industries: int = 6  # 仅看行业 RS 排名前 N

    # ── 阶段 2: 龙头确认 ──
    min_momentum_20d: float = 0.10
    min_momentum_5d: float = 0.0
    min_close_vs_high_60d: float = 0.85
    max_close_vs_high_60d: float = 1.05
    min_vol_ratio_s: float = 1.20
    min_rs_vs_industry_20d: float = 0.05
    min_path_irrev_m: float = 0.0

    # ── 排序权重 ──
    rank_weight_rs_industry: float = 0.40
    rank_weight_breakout: float = 0.30
    rank_weight_volume: float = 0.20
    rank_weight_irrev: float = 0.10

    # ── 输出控制 ──
    max_per_industry: int = 2
    top_n: int = 5
    dedup_days: int = 5


def _safe_col(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return df[col].fillna(default)
    return pd.Series(default, index=df.index)


def _select_hot_industries(
    snapshot: pd.DataFrame,
    cfg: IndustryLeaderConfig,
) -> list[str]:
    """阶段 1: 识别热门行业."""
    if "_industry" not in snapshot.columns:
        logger.warning("Agent 9: 缺少 _industry 列, 跳过")
        return []

    # 每行业取代表性指标 (任一行的行业因子都相同, 取 first)
    industry_stats = (
        snapshot.dropna(subset=["_industry"])
        .groupby("_industry")
        .agg(
            n_stocks=("_industry", "size"),
            industry_rs_20d=("industry_rs_20d", "first"),
            industry_breadth_5d=("industry_breadth_5d", "first"),
            industry_mom_20d=("industry_mom_20d", "first"),
            industry_vol_surge=("industry_vol_surge", "first"),
        )
    )

    mask = (
        (industry_stats["n_stocks"] >= cfg.min_stocks_per_industry)
        & (industry_stats["industry_rs_20d"] >= cfg.min_industry_rs_20d)
        & (industry_stats["industry_breadth_5d"] >= cfg.min_industry_breadth_5d)
        & (industry_stats["industry_mom_20d"] >= cfg.min_industry_mom_20d)
        & (industry_stats["industry_vol_surge"] >= cfg.min_industry_vol_surge)
    )
    hot = industry_stats[mask].sort_values("industry_rs_20d", ascending=False)
    hot = hot.head(cfg.top_industries)
    if not hot.empty:
        logger.info(f"  Agent 9 阶段1: {len(hot)} 个热门行业")
        for ind, row in hot.iterrows():
            logger.info(
                f"    {ind}: RS={row['industry_rs_20d']:+.1%} "
                f"BR={row['industry_breadth_5d']:.0%} "
                f"M20={row['industry_mom_20d']:+.1%} "
                f"V={row['industry_vol_surge']:.2f} "
                f"N={int(row['n_stocks'])}"
            )
    return hot.index.tolist()


def _score_leaders(
    candidates: pd.DataFrame,
    cfg: IndustryLeaderConfig,
) -> pd.DataFrame:
    """阶段 2: 在候选池内评分."""
    if candidates.empty:
        return candidates

    # 各分量 min-max 截断到 [0, 1]
    rs_industry = (candidates["_rs_vs_industry_20d"] - cfg.min_rs_vs_industry_20d) / 0.30
    breakout = (candidates["close_vs_high_60d"] - cfg.min_close_vs_high_60d) / (
        cfg.max_close_vs_high_60d - cfg.min_close_vs_high_60d
    )
    volume = (candidates["vol_ratio_s"] - cfg.min_vol_ratio_s) / 1.5
    irrev = candidates["path_irrev_m"].clip(lower=0) / 0.20

    rs_industry = rs_industry.clip(0, 1)
    breakout = breakout.clip(0, 1)
    volume = volume.clip(0, 1)
    irrev = irrev.clip(0, 1)

    score = (
        cfg.rank_weight_rs_industry * rs_industry
        + cfg.rank_weight_breakout * breakout
        + cfg.rank_weight_volume * volume
        + cfg.rank_weight_irrev * irrev
    )

    out = candidates.copy()
    out["il_score"] = score.round(4)
    return out


def _build_reason(row: pd.Series) -> str:
    parts = [
        f"行业RS={row.get('industry_rs_20d', 0):+.1%}",
        f"个股M20={row.get('momentum_20d', 0):+.1%}",
        f"超额={row.get('_rs_vs_industry_20d', 0):+.1%}",
        f"突破={row.get('close_vs_high_60d', 0):.2f}",
        f"量={row.get('vol_ratio_s', 1):.1f}x",
    ]
    return " | ".join(parts)


def run_industry_leader_signal(
    factor_snapshot: pd.DataFrame,
    scan_date: str,
    cfg: IndustryLeaderConfig | None = None,
    recent_predictions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    生成行业龙头候选 (与 Agent 3 平行的独立通道).

    Args:
        factor_snapshot: Agent 1 全市场截面 (含 V10 行业因子)
        scan_date: 当日日期 (YYYYMMDD)
        cfg: 配置
        recent_predictions: 近期预测记录 (含 symbol+scan_date 列), 用于去重

    Returns:
        DataFrame columns:
            symbol, name, industry,
            prob_200, prob_100, prob_30 (全部填 NaN, 标识非模型来源),
            il_score, il_reason, source='industry_leader',
            grade='IL', rank, scan_date
    """
    cfg = cfg or IndustryLeaderConfig()

    if factor_snapshot.empty:
        logger.warning("Agent 9: 空快照")
        return pd.DataFrame()

    # 检查必备列
    required = ["_industry", "industry_rs_20d", "industry_mom_20d",
                "industry_breadth_5d", "industry_vol_surge",
                "momentum_20d", "momentum_5d", "close_vs_high_60d",
                "vol_ratio_s"]
    missing = [c for c in required if c not in factor_snapshot.columns]
    if missing:
        logger.warning(f"Agent 9: 缺少必备列 {missing}, 跳过")
        return pd.DataFrame()

    snapshot = factor_snapshot.copy()
    if snapshot.index.name != "symbol":
        if "symbol" in snapshot.columns:
            snapshot = snapshot.set_index("symbol")

    # ── 阶段 1: 选热门行业 ──
    hot_industries = _select_hot_industries(snapshot, cfg)
    if not hot_industries:
        logger.info("Agent 9: 无热门行业, 输出空")
        return pd.DataFrame()

    # ── 阶段 2: 在热门行业内挑龙头 ──
    pool = snapshot[snapshot["_industry"].isin(hot_industries)].copy()
    pool["_rs_vs_industry_20d"] = pool["momentum_20d"] - pool["industry_mom_20d"]

    mask = (
        (pool["momentum_20d"] >= cfg.min_momentum_20d)
        & (pool["momentum_5d"] >= cfg.min_momentum_5d)
        & (pool["close_vs_high_60d"] >= cfg.min_close_vs_high_60d)
        & (pool["close_vs_high_60d"] <= cfg.max_close_vs_high_60d)
        & (pool["vol_ratio_s"] >= cfg.min_vol_ratio_s)
        & (pool["_rs_vs_industry_20d"] >= cfg.min_rs_vs_industry_20d)
        & (_safe_col(pool, "path_irrev_m") >= cfg.min_path_irrev_m)
    )
    leaders = pool[mask].copy()
    logger.info(f"  Agent 9 阶段2: {len(leaders)} 只龙头候选")

    if leaders.empty:
        return pd.DataFrame()

    # ── 评分排序 ──
    leaders = _score_leaders(leaders, cfg)
    leaders = leaders.sort_values("il_score", ascending=False)

    # ── 行业内 max_per_industry 限制 ──
    leaders = (
        leaders.groupby("_industry", group_keys=False)
        .head(cfg.max_per_industry)
        .sort_values("il_score", ascending=False)
    )

    # ── 去重 ──
    if (
        recent_predictions is not None
        and not recent_predictions.empty
        and cfg.dedup_days > 0
    ):
        recent_syms = set(recent_predictions["symbol"].unique())
        before = len(leaders)
        leaders = leaders[~leaders.index.isin(recent_syms)]
        if before > len(leaders):
            logger.info(f"  Agent 9 去重: {before} → {len(leaders)} 只 (近 {cfg.dedup_days} 天已推荐)")

    # ── 取 Top N ──
    final = leaders.head(cfg.top_n).copy()
    if final.empty:
        return pd.DataFrame()

    # ── 构造输出 ──
    out = pd.DataFrame({
        "symbol": final.index,
        "name": final.get("_name", pd.Series("", index=final.index)).values,
        "industry": final["_industry"].values,
        "prob_200": np.nan,
        "prob_100": np.nan,
        "prob_30": np.nan,
        "il_score": final["il_score"].values,
        "il_reason": final.apply(_build_reason, axis=1).values,
        "source": "industry_leader",
        "grade": "IL",
        "scan_date": scan_date,
    })
    out["rank"] = range(1, len(out) + 1)

    logger.info(f"Agent 9 完成: {len(out)} 只龙头候选")
    for _, r in out.iterrows():
        logger.info(f"    {r['symbol']} {r['name']} ({r['industry']}) "
                    f"score={r['il_score']:.3f} | {r['il_reason']}")
    return out


def merge_with_model_predictions(
    model_preds: pd.DataFrame,
    leader_preds: pd.DataFrame,
    max_total: int = 8,
) -> pd.DataFrame:
    """合并模型通道 + 龙头通道.

    去重策略: 同 symbol 优先保留模型通道 (信号更稳定).

    Args:
        model_preds: Agent 3 输出 (含 source 列时按 source='model' 处理)
        leader_preds: Agent 9 输出
        max_total: 合并后最多多少只

    Returns:
        合并后 DataFrame, 含统一列 source/grade.
    """
    if leader_preds is None or leader_preds.empty:
        if model_preds is not None and not model_preds.empty:
            out = model_preds.copy()
            if "source" not in out.columns:
                out["source"] = "model"
            return out
        return pd.DataFrame()

    if model_preds is None or model_preds.empty:
        return leader_preds.head(max_total).reset_index(drop=True)

    model_out = model_preds.copy()
    if "source" not in model_out.columns:
        model_out["source"] = "model"

    # 去除龙头通道里已经在模型通道的 symbol
    model_syms = set(model_out["symbol"].astype(str))
    leader_out = leader_preds[~leader_preds["symbol"].astype(str).isin(model_syms)].copy()

    merged = pd.concat([model_out, leader_out], ignore_index=True, sort=False)
    merged = merged.head(max_total).reset_index(drop=True)
    merged["rank"] = range(1, len(merged) + 1)
    return merged
