"""
Bull Hunter v3 — Agent 3: 大牛股预测

职责: 用 Agent 2 训练的模型, 对 scan_date 的因子截面预测,
      输出三个维度的概率排名 + 分层推荐。

分层:
  A 级: prob_200 > 阈值 (极少, 高置信度超级牛股)
  B 级: prob_100 > 阈值 (少, 翻倍候选)
  C 级: prob_30  > 阈值 (多, 短期爆发候选)

输出:
  results/{scan_date}/predictions.csv
  (symbol, name, industry, prob_30, prob_100, prob_200, grade)
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


@dataclass
class PredictConfig:
    """预测阈值配置。"""
    threshold_30pct: float = 0.15    # 正样本稀少, 用训练时的 best_threshold
    threshold_100pct: float = 0.15   # 同上
    threshold_200pct: float = 0.15   # 同上
    use_model_threshold: bool = True # 优先用 meta 中的 best_threshold
    top_n_per_grade: int = 20        # 每个等级最多输出 N 只


def run_prediction(
    factor_snapshot: pd.DataFrame,
    model_dir: str,
    scan_date: str,
    cfg: PredictConfig | None = None,
) -> pd.DataFrame:
    """
    用训练好的模型预测全市场大牛股概率。

    Args:
        factor_snapshot: Agent 1 的因子截面 (index=symbol)
        model_dir: bull_models/{scan_date}/ 目录
        scan_date: 预测日期
        cfg: 预测阈值配置

    Returns:
        DataFrame: 含 symbol, prob_30, prob_100, prob_200, grade 等列
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

    if not models:
        logger.error("Agent 3: 无可用模型")
        return pd.DataFrame()

    # 加载 meta 获取特征列
    meta_path = os.path.join(model_dir, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    # ── 构建特征矩阵 ──
    symbols = factor_snapshot.index.tolist()
    results = {"symbol": symbols}

    # 从 snapshot 中获取元信息
    for col in ["_name", "_industry"]:
        if col in factor_snapshot.columns:
            results[col[1:]] = factor_snapshot[col].tolist()

    # ── 逐模型预测 ──
    for tname, model in models.items():
        # 确定特征列 (优先用 meta 中记录的)
        if tname in meta and "feature_cols" in meta[tname]:
            feature_cols = meta[tname]["feature_cols"]
        else:
            feature_cols = [f for f in DAILY_FACTORS if f in factor_snapshot.columns]

        # 提取特征
        X_rows = []
        for sym in symbols:
            row_vals = []
            for f in feature_cols:
                v = factor_snapshot.loc[sym].get(f, np.nan) if f in factor_snapshot.columns else np.nan
                row_vals.append(float(v) if pd.notna(v) else 0.0)
            X_rows.append(row_vals)

        X = np.array(X_rows)

        # 预测概率
        probas = model.predict_proba(X)[:, 1]
        prob_col = f"prob_{tname.replace('pct', '')}"
        results[prob_col] = probas.tolist()

    df = pd.DataFrame(results)

    # ── 分层评级 (优先用模型训练时的最优阈值) ──
    thresholds = {
        "200pct": cfg.threshold_200pct,
        "100pct": cfg.threshold_100pct,
        "30pct": cfg.threshold_30pct,
    }
    if cfg.use_model_threshold:
        for tname in ["200pct", "100pct", "30pct"]:
            if tname in meta and "best_threshold" in meta[tname]:
                thresholds[tname] = meta[tname]["best_threshold"]
                logger.info(f"  {tname}: 使用模型最优阈值 {thresholds[tname]:.3f}")

    df["grade"] = "D"

    if "prob_200" in df.columns:
        df.loc[df["prob_200"] > thresholds["200pct"], "grade"] = "A"
    if "prob_100" in df.columns:
        mask_b = (df["prob_100"] > thresholds["100pct"]) & (df["grade"] == "D")
        df.loc[mask_b, "grade"] = "B"
    if "prob_30" in df.columns:
        mask_c = (df["prob_30"] > thresholds["30pct"]) & (df["grade"] == "D")
        df.loc[mask_c, "grade"] = "C"

    # ── 过滤 + 排序 ──
    selected = df[df["grade"] != "D"].copy()

    if selected.empty:
        logger.warning("Agent 3: 无候选股票 (全部低于阈值)")
        # 回退: 取 prob_30 最高的 top_n
        if "prob_30" in df.columns:
            selected = df.nlargest(cfg.top_n_per_grade, "prob_30").copy()
            selected["grade"] = "C"

    # 按等级排序 (A > B > C), 同等级按对应概率排序
    grade_order = {"A": 0, "B": 1, "C": 2, "D": 3}
    selected["_grade_order"] = selected["grade"].map(grade_order)

    sort_col = "prob_30"
    for c in ["prob_200", "prob_100", "prob_30"]:
        if c in selected.columns:
            sort_col = c
            break

    selected = selected.sort_values(
        ["_grade_order", sort_col], ascending=[True, False]
    ).drop(columns=["_grade_order"])

    # 限制每个等级的数量
    final_parts = []
    for g in ["A", "B", "C"]:
        part = selected[selected["grade"] == g].head(cfg.top_n_per_grade)
        final_parts.append(part)
    final = pd.concat(final_parts, ignore_index=True)

    n_a = (final["grade"] == "A").sum()
    n_b = (final["grade"] == "B").sum()
    n_c = (final["grade"] == "C").sum()
    logger.info(f"Agent 3 完成: {len(final)} 只候选, "
                f"A={n_a} B={n_b} C={n_c}, scan_date={scan_date}")

    return final
