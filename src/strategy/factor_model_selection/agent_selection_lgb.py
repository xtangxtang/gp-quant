"""
因子模型选股 — Agent 2: 模型选股

职责: 接收因子截面, 用 LightGBM 模型或 IC 加权评分选出 Top N。
输入: PipelineState.factor_snapshot (来自 Agent 1)
输出: PipelineState.selections (dict[horizon, DataFrame])

在 walk-forward 模式下, 用截止 train_cutoff 的数据训练模型,
然后对 scan_date 的因子截面评分。
"""

from __future__ import annotations

import glob
import logging
import os
import pickle
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from .factor_model import (
    DAILY_FACTORS, WEEKLY_FACTORS,
    DAILY_HORIZONS, WEEKLY_HORIZONS, ALL_HORIZONS,
    ModelConfig, build_panel, compute_sample_weights,
)


def run_selection(
    factor_snapshot: pd.DataFrame,
    cache_dir: str,
    scan_date: str,
    train_cutoff: str,
    top_n: int = 5,
    cfg: ModelConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """
    用 walk-forward 方式选股:
      1. 用 <= train_cutoff 的数据训练 LightGBM
      2. 对 scan_date 的因子截面评分
      3. 每个 horizon 取 Top N

    Args:
        factor_snapshot: Agent 1 输出的因子截面 (index=symbol)
        cache_dir: 特征缓存根目录
        scan_date: 选股日期
        train_cutoff: 训练截止日期 (必须 < scan_date, 且前瞻收益已实现)
        top_n: 每个 horizon 选几只
        cfg: LightGBM 配置

    Returns:
        {horizon: DataFrame with columns [symbol, name, industry, pred_score, rank]}
    """
    cfg = cfg or ModelConfig(cache_dir=cache_dir)
    results: dict[str, pd.DataFrame] = {}

    # ── 训练并评分: 日线 ──
    daily_panel = build_panel(cache_dir, "daily", cfg)
    if not daily_panel.empty:
        daily_panel["trade_date"] = daily_panel["trade_date"].astype(str)
        for fd in [1, 3, 5]:
            h = f"{fd}d"
            target_col = f"fwd_ret_{h}"
            if target_col not in daily_panel.columns:
                continue

            model, feature_cols, info = _train_for_date(
                daily_panel, target_col, DAILY_FACTORS, train_cutoff,
                cfg.decay_lambda, cfg,
            )
            if model is None:
                logger.warning(f"  {h}: 训练失败, 跳过")
                continue

            top_df = _predict_and_rank(
                model, feature_cols, factor_snapshot, h, top_n, info
            )
            results[h] = top_df

    # ── 训练并评分: 周线 ──
    weekly_panel = build_panel(cache_dir, "weekly", cfg)
    if not weekly_panel.empty:
        weekly_panel["trade_date"] = weekly_panel["trade_date"].astype(str)

        # 周线因子在 snapshot 中有 w_ 前缀, 需要映射
        weekly_snapshot = _extract_weekly_factors(factor_snapshot)

        for fw in [1, 3, 5]:
            h = f"{fw}w"
            target_col = f"fwd_ret_{h}"
            if target_col not in weekly_panel.columns:
                continue

            model, feature_cols, info = _train_for_date(
                weekly_panel, target_col, WEEKLY_FACTORS, train_cutoff,
                cfg.weekly_decay_lambda, cfg,
            )
            if model is None:
                logger.warning(f"  {h}: 训练失败, 跳过")
                continue

            top_df = _predict_and_rank(
                model, feature_cols, weekly_snapshot, h, top_n, info,
            )
            # 附加 name/industry
            if "_name" in factor_snapshot.columns:
                top_df["name"] = top_df["symbol"].map(
                    factor_snapshot["_name"].to_dict()
                ).fillna("")
                top_df["industry"] = top_df["symbol"].map(
                    factor_snapshot["_industry"].to_dict()
                ).fillna("")
            results[h] = top_df

    n_total = sum(len(df) for df in results.values())
    n_unique = len(set(s for df in results.values() for s in df["symbol"]))
    logger.info(f"Agent 2 完成: {n_total} 条推荐 ({n_unique} 只去重), "
                f"train_cutoff={train_cutoff}, scan_date={scan_date}")

    return results


def _train_for_date(
    panel: pd.DataFrame,
    target_col: str,
    factors: list[str],
    train_cutoff: str,
    decay_lambda: float,
    cfg: ModelConfig,
) -> tuple[Optional[lgb.LGBMRegressor], list[str], dict]:
    """用截止 train_cutoff 的数据训练 LightGBM, 返回 (model, feature_cols, info)."""
    feature_cols = [f for f in factors if f in panel.columns and panel[f].notna().sum() > 100]
    if not feature_cols:
        return None, [], {}

    mask = (panel["trade_date"] <= train_cutoff) & panel[target_col].notna()
    data = panel[mask]

    if len(data) < 5000:
        return None, [], {}

    all_dates = sorted(data["trade_date"].unique())
    if len(all_dates) <= cfg.val_days + 20:
        return None, [], {}

    val_start = all_dates[-cfg.val_days]
    t_mask = data["trade_date"] < val_start
    v_mask = data["trade_date"] >= val_start

    X_train = data[t_mask][feature_cols].fillna(0).values
    y_train = data[t_mask][target_col].values
    X_val = data[v_mask][feature_cols].fillna(0).values
    y_val = data[v_mask][target_col].values

    if len(X_train) < 1000 or len(X_val) < 100:
        return None, [], {}

    weights = compute_sample_weights(data[t_mask]["trade_date"], decay_lambda)

    model = lgb.LGBMRegressor(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        num_leaves=cfg.num_leaves,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        min_child_samples=cfg.min_child_samples,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        random_state=42,
        n_jobs=cfg.workers,
        verbose=-1,
    )

    model.fit(
        X_train, y_train,
        sample_weight=weights,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    from scipy.stats import spearmanr
    pred_val = model.predict(X_val)
    val_ic, _ = spearmanr(pred_val, y_val)

    info = {
        "train_rows": int(t_mask.sum()),
        "val_rows": int(v_mask.sum()),
        "train_cutoff": train_cutoff,
        "val_ic": round(float(val_ic), 4),
        "best_iter": model.best_iteration_,
    }

    logger.info(f"  {target_col}: cutoff={train_cutoff}, "
                f"train={info['train_rows']}, val={info['val_rows']}, "
                f"val_IC={info['val_ic']}, iter={info['best_iter']}")

    return model, feature_cols, info


def _predict_and_rank(
    model: lgb.LGBMRegressor,
    feature_cols: list[str],
    snapshot: pd.DataFrame,
    horizon: str,
    top_n: int,
    train_info: dict,
) -> pd.DataFrame:
    """对因子截面预测并排序, 返回 Top N."""
    rows = []
    syms = []
    for sym in snapshot.index:
        feat_vals = []
        for f in feature_cols:
            v = snapshot.loc[sym].get(f, np.nan)
            feat_vals.append(float(v) if pd.notna(v) else 0.0)
        rows.append(feat_vals)
        syms.append(sym)

    if len(rows) < 50:
        return pd.DataFrame()

    X = np.array(rows)
    preds = model.predict(X)
    pred_series = pd.Series(preds, index=syms)
    top_syms = pred_series.nlargest(top_n).index.tolist()

    result_rows = []
    for rank, sym in enumerate(top_syms, 1):
        row = {
            "rank": rank,
            "symbol": sym,
            "name": snapshot.loc[sym].get("_name", "") if "_name" in snapshot.columns else "",
            "industry": snapshot.loc[sym].get("_industry", "") if "_industry" in snapshot.columns else "",
            "horizon": horizon,
            "pred_score": round(float(pred_series[sym]), 6),
            "train_cutoff": train_info.get("train_cutoff", ""),
            "val_ic": train_info.get("val_ic", 0),
        }
        result_rows.append(row)

    return pd.DataFrame(result_rows)


def _extract_weekly_factors(snapshot: pd.DataFrame) -> pd.DataFrame:
    """从带 w_ 前缀的 snapshot 中提取周线因子, 去掉前缀。"""
    weekly_cols = {c: c[2:] for c in snapshot.columns if c.startswith("w_") and not c.startswith("w__")}
    if not weekly_cols:
        return pd.DataFrame(index=snapshot.index)
    result = snapshot[list(weekly_cols.keys())].rename(columns=weekly_cols)
    # 保留元信息
    for c in ("_name", "_industry", "_avg_amount_20", "_trade_date"):
        if c in snapshot.columns:
            result[c] = snapshot[c]
    return result
