"""
IC加权多horizon选股模型 (v3)

对齐关系:
  factor_profiling 计算 IC(factor, fwd_ret_Nd) → 本模块用该 IC 作为权重
  → 选出的股票 = "因子值组合预测未来 Nd 涨幅最高" 的股票

流程:
  1. 从 factor_profile/{date}/*.json 加载各因子在 6 个 horizon 的 IC
  2. 取全市场中位 IC 作为因子权重 (跨股票鲁棒估计)
  3. 从特征缓存加载各股票最新因子值
  4. 截面 z-score 标准化 → IC 加权评分
  5. 每个 horizon 取 Top 5, 共输出 30 只

数据依赖:
  - 第一步 feature_engine → daily/weekly 缓存 (不修改)
  - 第二步 factor_profiling → factor_profile JSONs (不修改)
  - 本模块替代 signal_detector_v2 的选股逻辑
"""

from __future__ import annotations

import json
import logging
import os
import glob
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DAILY_HORIZONS = ["1d", "3d", "5d"]
WEEKLY_HORIZONS = ["1w", "3w", "5w"]
ALL_HORIZONS = DAILY_HORIZONS + WEEKLY_HORIZONS


# ═════════════════════════════════════════════════════════
# 配置
# ═════════════════════════════════════════════════════════

@dataclass
class ScoringConfig:
    """IC加权选股配置"""
    top_n_per_horizon: int = 5       # 每个 horizon 选 N 只
    min_ic_abs: float = 0.02         # IC 绝对值门槛 (低于此的因子不参与评分)
    min_stocks_with_ic: int = 50     # 至少 N 只股票有该因子 IC 才使用
    min_data_rows: int = 60          # 特征缓存最少行数
    min_amount: float = 5000.0       # 最低日均成交额 (千元)
    exclude_st: bool = True          # 排除 ST


# ═════════════════════════════════════════════════════════
# Step 1: 加载 IC 权重
# ═════════════════════════════════════════════════════════

def load_market_ic_weights(
    profile_dir: str,
    min_ic_abs: float = 0.02,
    min_stocks: int = 50,
) -> dict[str, dict[str, float]]:
    """
    从全市场 factor_profile 加载每个 (factor, horizon) 的中位 IC。

    优先读取 ic_weights.json (由 factor_profiling 自动生成),
    不存在时回退到逐个读取 *.json (兼容旧数据)。

    Returns:
        {horizon: {factor: median_ic, ...}, ...}
    """
    # ── 快速路径: 读取预生成的配置文件 ──
    config_path = os.path.join(profile_dir, "ic_weights.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            weights = config.get("horizons", {})
            generated = config.get("generated_at", "unknown")
            n_daily = config.get("n_stocks_daily", "?")
            n_weekly = config.get("n_stocks_weekly", "?")
            logger.info(
                f"Loaded ic_weights.json (generated {generated}, "
                f"{n_daily} daily / {n_weekly} weekly stocks)"
            )
            for h in ALL_HORIZONS:
                hw = weights.get(h, {})
                logger.info(f"  {h}: {len(hw)} effective factors")
            return weights
        except Exception as e:
            logger.warning(f"Failed to load ic_weights.json: {e}, falling back to JSONs")

    # ── 慢速路径: 从个股 JSON 逐个汇总 ──
    return _load_ic_weights_from_jsons(profile_dir, min_ic_abs, min_stocks)


def _load_ic_weights_from_jsons(
    profile_dir: str,
    min_ic_abs: float = 0.02,
    min_stocks: int = 50,
) -> dict[str, dict[str, float]]:
    """从个股 profile JSONs 逐个读取并计算中位 IC (兼容旧数据)。"""
    ic_collector: dict[str, dict[str, list[float]]] = {h: {} for h in ALL_HORIZONS}

    json_files = sorted(glob.glob(os.path.join(profile_dir, "*.json")))
    if not json_files:
        logger.error(f"No profile JSONs found in {profile_dir}")
        return {h: {} for h in ALL_HORIZONS}

    logger.info(f"Loading IC from {len(json_files)} profile JSONs ...")

    for fpath in json_files:
        try:
            with open(fpath, encoding="utf-8") as f:
                profile = json.load(f)
        except Exception:
            continue

        # 日线 horizons
        mfi = profile.get("multi_forward_ic", {})
        for h in DAILY_HORIZONS:
            ics = mfi.get(h, {})
            for factor, ic_val in ics.items():
                if not np.isfinite(ic_val):
                    continue
                ic_collector[h].setdefault(factor, []).append(ic_val)

        # 周线 horizons
        weekly = profile.get("weekly") or {}
        wmfi = weekly.get("multi_forward_ic", {})
        for h in WEEKLY_HORIZONS:
            ics = wmfi.get(h, {})
            for factor, ic_val in ics.items():
                if not np.isfinite(ic_val):
                    continue
                ic_collector[h].setdefault(factor, []).append(ic_val)

    # 计算中位 IC
    weights: dict[str, dict[str, float]] = {}
    for h in ALL_HORIZONS:
        weights[h] = {}
        for factor, ic_list in ic_collector[h].items():
            if len(ic_list) < min_stocks:
                continue
            median_ic = float(np.median(ic_list))
            if abs(median_ic) < min_ic_abs:
                continue
            weights[h][factor] = median_ic
        logger.info(f"  {h}: {len(weights[h])} effective factors (|IC| >= {min_ic_abs})")

    return weights


# ═════════════════════════════════════════════════════════
# Step 2: 加载最新特征
# ═════════════════════════════════════════════════════════

def load_latest_features(
    cache_dir: str,
    scan_date: str = "",
    min_rows: int = 60,
) -> pd.DataFrame:
    """
    从特征缓存加载所有股票的最新特征向量 (一行一只股票)。

    Returns:
        DataFrame, index=symbol, columns=因子名 + "_trade_date"
    """
    csv_files = sorted(glob.glob(os.path.join(cache_dir, "*.csv")))
    if not csv_files:
        logger.error(f"No CSV files in {cache_dir}")
        return pd.DataFrame()

    rows = []
    for fpath in csv_files:
        symbol = os.path.basename(fpath).replace(".csv", "")
        try:
            df = pd.read_csv(fpath)
        except Exception:
            continue

        if len(df) < min_rows:
            continue

        df = df.sort_values("trade_date").reset_index(drop=True)

        # 截取到 scan_date
        if scan_date:
            df = df[df["trade_date"].astype(str) <= scan_date]
            if len(df) < min_rows:
                continue

        last = df.iloc[-1].to_dict()
        last["symbol"] = symbol
        last["_trade_date"] = str(df["trade_date"].iloc[-1])

        # 近 20 日均成交额 (用于流动性过滤)
        last["_avg_amount_20"] = float(df.tail(20)["amount"].mean()) if "amount" in df.columns else 0.0

        rows.append(last)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).set_index("symbol")
    logger.info(f"Loaded {len(result)} stocks' latest features from {cache_dir}")
    return result


# ═════════════════════════════════════════════════════════
# Step 3: 截面评分
# ═════════════════════════════════════════════════════════

def score_stocks(
    features: pd.DataFrame,
    ic_weights: dict[str, float],
) -> tuple[pd.Series, dict[str, pd.Series]]:
    """
    对一组股票在某个 horizon 计算 IC 加权评分。

    公式:
      score_i = (1/N) * sum_f [ IC(f) * z_score_cross_sectional(factor_f, stock_i) ]

    其中 z_score 做了 ±3σ winsorize, 缺失值填 0 (中性)。

    Args:
        features: DataFrame, rows=stocks, columns=因子
        ic_weights: {factor: median_ic}

    Returns:
        (composite_score_series, {factor: contribution_series})
    """
    score = pd.Series(0.0, index=features.index)
    contributions: dict[str, pd.Series] = {}
    n_used = 0

    for factor, ic_w in ic_weights.items():
        if factor not in features.columns:
            continue

        vals = features[factor].astype(float)
        valid = vals.notna() & np.isfinite(vals)

        if valid.sum() < 30:
            continue

        # 截面 z-score + winsorize
        mean_v = vals[valid].mean()
        std_v = vals[valid].std()
        if std_v < 1e-12:
            continue

        z = ((vals - mean_v) / std_v).clip(-3, 3).fillna(0)
        contrib = ic_w * z
        score += contrib
        contributions[factor] = contrib
        n_used += 1

    if n_used > 0:
        score /= n_used

    return score, contributions


# ═════════════════════════════════════════════════════════
# Step 4: 选股主流程
# ═════════════════════════════════════════════════════════

def run_ic_scoring(
    profile_dir: str,
    daily_cache_dir: str,
    weekly_cache_dir: str,
    scan_date: str = "",
    basic_info: Optional[dict[str, dict]] = None,
    cfg: Optional[ScoringConfig] = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    全市场 IC 加权选股。

    Returns:
        (horizon_tops, combined_df)
        - horizon_tops: {horizon: DataFrame with top_n stocks}
        - combined_df: 合并的 30 只候选 (含 horizon 标签)
    """
    cfg = cfg or ScoringConfig()
    basic_info = basic_info or {}

    # 1. IC 权重
    ic_weights = load_market_ic_weights(
        profile_dir,
        min_ic_abs=cfg.min_ic_abs,
        min_stocks=cfg.min_stocks_with_ic,
    )

    # 2. 加载特征
    daily_features = load_latest_features(daily_cache_dir, scan_date, cfg.min_data_rows)
    weekly_features = load_latest_features(weekly_cache_dir, scan_date, cfg.min_data_rows)

    if daily_features.empty:
        logger.error("No daily features loaded")
        return {}, pd.DataFrame()

    # 3. 过滤
    daily_features = _apply_filters(daily_features, basic_info, cfg)
    if not weekly_features.empty:
        weekly_features = _apply_filters(weekly_features, basic_info, cfg)

    logger.info(f"After filtering: {len(daily_features)} daily, {len(weekly_features)} weekly stocks")

    # 4. 计算所有 horizon 的原始评分
    raw_scores: dict[str, pd.Series] = {}
    raw_contribs: dict[str, dict[str, pd.Series]] = {}

    for h in DAILY_HORIZONS:
        hw = ic_weights.get(h, {})
        if hw:
            s, c = score_stocks(daily_features, hw)
            raw_scores[h] = s
            raw_contribs[h] = c

    for h in WEEKLY_HORIZONS:
        hw = ic_weights.get(h, {})
        if hw and not weekly_features.empty:
            s, c = score_stocks(weekly_features, hw)
            raw_scores[h] = s
            raw_contribs[h] = c

    # 5. 残差法选股: 每个 horizon 去除前序 horizon 已解释的部分
    #
    #   residual_h = score_h - proj(score_h, score_{h-1})
    #   proj(y, x) = (x'y / x'x) * x   (OLS 投影)
    #
    #   物理含义: "3d 评分中, 已被 1d 评分解释的部分" 被扣除,
    #   剩下的是 "3d 相对于 1d 的增量 alpha"
    #
    #   日线序列: 1d → 3d → 5d  (逐级残差)
    #   周线序列: 1w → 3w → 5w  (逐级残差)
    #   日线和周线之间独立 (数据源不同, 不做跨时间框架残差)

    horizon_tops: dict[str, pd.DataFrame] = {}

    def _ols_residual(y: pd.Series, x: pd.Series) -> pd.Series:
        """y 对 x 做 OLS 回归, 返回残差"""
        # 对齐 index
        common = y.index.intersection(x.index)
        if len(common) < 30:
            return y
        yc, xc = y.reindex(common).fillna(0), x.reindex(common).fillna(0)
        dot_xx = (xc * xc).sum()
        if dot_xx < 1e-12:
            return y
        beta = (xc * yc).sum() / dot_xx
        residual = yc - beta * xc
        # 保留 y 中不在 common 里的部分 (不减)
        result = y.copy()
        result.loc[common] = residual
        return result

    # 日线: 1d → 3d → 5d
    prev_score = None
    for h in DAILY_HORIZONS:
        if h not in raw_scores:
            horizon_tops[h] = pd.DataFrame()
            continue

        selection_score = raw_scores[h]
        if prev_score is not None:
            selection_score = _ols_residual(selection_score, prev_score)

        corr_with_raw = np.corrcoef(
            raw_scores[h].reindex(selection_score.index).fillna(0),
            selection_score.fillna(0),
        )[0, 1] if prev_score is not None else 1.0
        logger.info(f"  {h}: residual corr with raw = {corr_with_raw:.3f}")

        top_idx = selection_score.nlargest(cfg.top_n_per_horizon).index
        horizon_tops[h] = _build_horizon_df(
            h, top_idx, selection_score, raw_scores[h],
            raw_contribs.get(h, {}), ic_weights.get(h, {}),
            daily_features, basic_info,
        )

        prev_score = raw_scores[h]  # 下一轮用原始分做投影基

    # 周线: 1w → 3w → 5w
    prev_score = None
    for h in WEEKLY_HORIZONS:
        if h not in raw_scores:
            horizon_tops[h] = pd.DataFrame()
            continue

        selection_score = raw_scores[h]
        if prev_score is not None:
            selection_score = _ols_residual(selection_score, prev_score)

        corr_with_raw = np.corrcoef(
            raw_scores[h].reindex(selection_score.index).fillna(0),
            selection_score.fillna(0),
        )[0, 1] if prev_score is not None else 1.0
        logger.info(f"  {h}: residual corr with raw = {corr_with_raw:.3f}")

        top_idx = selection_score.nlargest(cfg.top_n_per_horizon).index
        horizon_tops[h] = _build_horizon_df(
            h, top_idx, selection_score, raw_scores[h],
            raw_contribs.get(h, {}), ic_weights.get(h, {}),
            weekly_features, basic_info,
        )

        prev_score = raw_scores[h]

    # 6. 合并
    combined = pd.concat(
        [df for df in horizon_tops.values() if len(df) > 0],
        ignore_index=True,
    )

    n_unique = combined["symbol"].nunique() if len(combined) > 0 else 0
    logger.info(f"Selected {len(combined)} candidates ({n_unique} unique) across {len(ALL_HORIZONS)} horizons")

    return horizon_tops, combined


def _build_horizon_df(
    horizon: str,
    top_idx: pd.Index,
    selection_score: pd.Series,
    raw_score: pd.Series,
    contribs: dict[str, pd.Series],
    hw: dict[str, float],
    features: pd.DataFrame,
    basic_info: dict[str, dict],
) -> pd.DataFrame:
    """构建单个 horizon 的输出 DataFrame"""
    rows = []
    for rank, sym in enumerate(top_idx, 1):
        row = {
            "rank": rank,
            "symbol": sym,
            "name": basic_info.get(sym, {}).get("name", ""),
            "industry": basic_info.get(sym, {}).get("industry", ""),
            "horizon": horizon,
            "score": round(float(raw_score[sym]), 6),
            "residual_score": round(float(selection_score[sym]), 6),
            "trade_date": features.loc[sym, "_trade_date"] if sym in features.index else "",
        }
        # Top contributing factors
        factor_contribs = {f: float(contribs[f][sym]) for f in contribs if sym in contribs[f].index}
        top_factors = sorted(factor_contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        row["top_factors"] = "; ".join(f"{f}={c:+.4f}" for f, c in top_factors)

        # 附加关键因子原始值
        for factor in list(hw.keys())[:8]:
            if factor in features.columns and sym in features.index:
                v = features.loc[sym, factor]
                row[factor] = round(float(v), 4) if pd.notna(v) else None
        rows.append(row)

    return pd.DataFrame(rows)


def _apply_filters(
    features: pd.DataFrame,
    basic_info: dict[str, dict],
    cfg: ScoringConfig,
) -> pd.DataFrame:
    """应用预过滤: ST, 流动性, 北交所微盘"""
    mask = pd.Series(True, index=features.index)

    # ST 过滤
    if cfg.exclude_st:
        for sym in features.index:
            name = basic_info.get(sym, {}).get("name", "")
            if "ST" in name or "退" in name:
                mask[sym] = False

    # 流动性过滤
    if "_avg_amount_20" in features.columns:
        mask = mask & (features["_avg_amount_20"] >= cfg.min_amount)

    before = mask.sum()
    result = features[mask]
    logger.info(f"Filter: {len(features)} → {len(result)} stocks")
    return result


# ═════════════════════════════════════════════════════════
# IC 权重诊断 (调试用)
# ═════════════════════════════════════════════════════════

def print_ic_weights(ic_weights: dict[str, dict[str, float]]):
    """打印 IC 权重摘要"""
    print("\n" + "=" * 70)
    print("IC Weights Summary (market median IC per factor per horizon)")
    print("=" * 70)

    for h in ALL_HORIZONS:
        hw = ic_weights.get(h, {})
        if not hw:
            print(f"\n  {h}: (no effective factors)")
            continue

        sorted_factors = sorted(hw.items(), key=lambda x: abs(x[1]), reverse=True)
        print(f"\n  {h} ({len(hw)} factors):")
        for f, ic in sorted_factors[:10]:
            direction = "↑" if ic > 0 else "↓"
            print(f"    {direction} {f:<30} IC = {ic:+.4f}")
