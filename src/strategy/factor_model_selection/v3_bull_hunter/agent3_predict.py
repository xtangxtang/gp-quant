"""
Bull Hunter v3 — Agent 3: 每日 A 类大牛股预测

v4 重构: 聚焦 A 类大牛股 (200% 目标), 每日输出 Top 5 候选。

输入:
  - 最新模型 (feature-cache/bull_models/latest/)
  - 当日因子快照 (Agent 1)

输出:
  results/bull_hunter/daily/{scan_date}.csv
  (symbol, name, industry, prob_200, prob_100, rank)

规则:
  - 只输出 A 级: prob_200 > 阈值
  - 最多 5 只, 不足 5 只按实际数量输出
  - 0 只也可以 (无候选日)
  - 3 天内已推荐过的不重复输出 (去重)
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from .agent2_train import DAILY_FACTORS, TARGETS

# 每日最多输出候选数
TOP_N = 5
# 去重窗口 (交易日)
DEDUP_WINDOW_DAYS = 3


@dataclass
class PredictConfig:
    """预测阈值配置。"""
    threshold_200pct: float = 0.15
    threshold_100pct: float = 0.15
    use_model_threshold: bool = True  # 优先用 meta 中的 best_threshold
    top_n: int = TOP_N
    dedup_days: int = DEDUP_WINDOW_DAYS


def run_prediction(
    factor_snapshot: pd.DataFrame,
    model_dir: str,
    scan_date: str,
    cfg: PredictConfig | None = None,
    recent_predictions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    每日 A 类大牛股预测: 输出 Top 5 候选。

    Args:
        factor_snapshot: Agent 1 的因子截面 (index=symbol)
        model_dir: bull_models/latest/ 或 bull_models/{date}/ 目录
        scan_date: 预测日期
        cfg: 预测阈值配置
        recent_predictions: 近期预测记录 (用于去重), 含 symbol+scan_date 列

    Returns:
        DataFrame: 含 symbol, prob_200, prob_100, rank 等列 (最多 TOP_N 行)
    """
    cfg = cfg or PredictConfig()

    # ── 加载模型 ──
    models = {}
    for tname in TARGETS:
        model_path = os.path.join(model_dir, f"model_{tname}.pkl")
        if not os.path.exists(model_path):
            logger.warning(f"模型不存在: {model_path}")
            continue
        with open(model_path, "rb") as f:
            models[tname] = pickle.load(f)

    if "200pct" not in models:
        logger.error("Agent 3: 无 200pct 主模型, 无法预测")
        return pd.DataFrame()

    # 加载 meta 获取特征列和阈值
    meta_path = os.path.join(model_dir, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    # ── 构建特征矩阵 ──
    symbols = factor_snapshot.index.tolist()
    results = {"symbol": symbols}

    # 元信息
    for col in ["_name", "_industry"]:
        if col in factor_snapshot.columns:
            results[col[1:]] = factor_snapshot[col].tolist()

    # ── 逐模型预测 ──
    for tname, model in models.items():
        if tname in meta and "feature_cols" in meta[tname]:
            feature_cols = meta[tname]["feature_cols"]
        else:
            feature_cols = [f for f in DAILY_FACTORS if f in factor_snapshot.columns]

        X_rows = []
        for sym in symbols:
            row_vals = []
            for f in feature_cols:
                v = factor_snapshot.loc[sym].get(f, np.nan) if f in factor_snapshot.columns else np.nan
                row_vals.append(float(v) if pd.notna(v) else 0.0)
            X_rows.append(row_vals)

        X = np.array(X_rows)
        probas = model.predict_proba(X)[:, 1]
        prob_col = f"prob_{tname.replace('pct', '')}"
        results[prob_col] = probas.tolist()

    df = pd.DataFrame(results)

    # ── 阈值 (优先用模型训练时的最优阈值) ──
    thresh_200 = cfg.threshold_200pct
    if cfg.use_model_threshold and "200pct" in meta:
        if "best_threshold" in meta["200pct"]:
            thresh_200 = meta["200pct"]["best_threshold"]
            logger.info(f"  200pct: 使用模型最优阈值 {thresh_200:.3f}")

    # ── 筛选 A 级: prob_200 > 阈值, 按 prob_200 降序 ──
    if "prob_200" not in df.columns:
        logger.error("Agent 3: prob_200 列缺失")
        return pd.DataFrame()

    selected = df[df["prob_200"] > thresh_200].copy()
    selected = selected.sort_values("prob_200", ascending=False)

    # ── 去重: 排除近 dedup_days 天内已推荐的 ──
    if recent_predictions is not None and not recent_predictions.empty and cfg.dedup_days > 0:
        recent_syms = set(recent_predictions["symbol"].unique())
        before = len(selected)
        selected = selected[~selected["symbol"].isin(recent_syms)]
        deduped = before - len(selected)
        if deduped > 0:
            logger.info(f"  去重: 排除 {deduped} 只近 {cfg.dedup_days} 天已推荐")

    # ── 取 Top N ──
    final = selected.head(cfg.top_n).copy()
    final["rank"] = range(1, len(final) + 1)
    final["grade"] = "A"
    final["scan_date"] = scan_date

    n_total = len(df)
    n_above = len(selected)
    n_out = len(final)
    logger.info(f"Agent 3 完成: {n_out} 只 A 级候选 "
                f"(全市场 {n_total} 只, 过阈值 {n_above} 只, "
                f"Top {cfg.top_n}, scan_date={scan_date})")

    return final
