"""
LightGBM 因子模型 — 时间衰减加权 Walk-Forward 训练

替代 IC 线性加权, 用梯度提升树捕捉因子间非线性交互。

核心设计:
  1. 每只股票的历史数据拼成全市场截面 panel (date × stock × factors)
  2. 样本权重 = exp(-λ(T-t)), 近期权重指数级更大
  3. Walk-forward: 用 train_end 之前的数据训练, train_end~train_end+gap 为验证
  4. 日线/周线各训练独立模型
  5. 输出与 ic_scoring 兼容的评分

用法:
  # 训练 (默认 walk-forward)
  python -m src.strategy.factor_model_selection.factor_model \
    --cache_dir /path/to/feature-cache \
    --mode train

  # 预测 (加载已训练模型, 输出评分)
  python -m src.strategy.factor_model_selection.factor_model \
    --cache_dir /path/to/feature-cache \
    --mode predict --scan_date 20260417
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── 因子列表 (与 factor_profiling 一致) ──

DAILY_FACTORS = [
    "perm_entropy_s", "perm_entropy_m", "perm_entropy_l",
    "entropy_slope", "entropy_accel",
    "path_irrev_m", "path_irrev_l",
    "dom_eig_m", "dom_eig_l",
    "turnover_entropy_m", "turnover_entropy_l",
    "volatility_m", "volatility_l",
    "vol_compression", "bbw_pctl",
    "vol_ratio_s", "vol_impulse", "vol_shrink", "breakout_range",
    "mf_big_net", "mf_big_net_ratio",
    "mf_big_cumsum_s", "mf_big_cumsum_m", "mf_big_cumsum_l",
    "mf_sm_proportion", "mf_flow_imbalance",
    "mf_big_momentum", "mf_big_streak",
    "coherence_l1", "purity_norm", "von_neumann_entropy", "coherence_decay_rate",
]

WEEKLY_FACTORS = [
    "perm_entropy_s", "perm_entropy_m", "perm_entropy_l",
    "entropy_slope", "entropy_accel",
    "path_irrev_m", "path_irrev_l",
    "dom_eig_m", "dom_eig_l",
    "turnover_entropy_m", "turnover_entropy_l",
    "volatility_m", "volatility_l",
    "vol_compression", "bbw_pctl",
    "vol_ratio_s", "vol_impulse", "vol_shrink", "breakout_range",
    "coherence_l1", "purity_norm", "von_neumann_entropy", "coherence_decay_rate",
    "pe_ttm_pctl", "pb_pctl",
    "weekly_big_net", "weekly_big_net_cumsum",
    "weekly_turnover_ma4", "weekly_turnover_shrink",
]

DAILY_HORIZONS = ["1d", "3d", "5d"]
WEEKLY_HORIZONS = ["1w", "3w", "5w"]
ALL_HORIZONS = DAILY_HORIZONS + WEEKLY_HORIZONS


# ═════════════════════════════════════════════════════════
# 配置
# ═════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    cache_dir: str = ""
    model_dir: str = ""              # 模型存储目录, 默认 cache_dir/lgb_models/
    scan_date: str = ""              # 预测日期
    # 训练参数
    decay_lambda: float = 0.007      # 日线时间衰减 (~100天半衰期)
    weekly_decay_lambda: float = 0.035  # 周线时间衰减 (~20周半衰期)
    min_rows: int = 120              # 单股最少行数
    min_weekly_rows: int = 30
    train_end: str = ""              # walk-forward 训练截止日 (YYYYMMDD), 默认=最新日期-gap
    val_days: int = 60               # 验证集天数
    # LightGBM 参数
    n_estimators: int = 300
    max_depth: int = 4               # 浅树, 防过拟合
    num_leaves: int = 15
    learning_rate: float = 0.05
    subsample: float = 0.7
    colsample_bytree: float = 0.7
    min_child_samples: int = 50
    reg_alpha: float = 0.1           # L1 正则
    reg_lambda: float = 1.0          # L2 正则
    # 样本
    max_stocks: int = 0              # >0: 限制股数 (调试用)
    workers: int = 8
    top_n_per_horizon: int = 5


# ═════════════════════════════════════════════════════════
# Step 1: 构建训练 panel
# ═════════════════════════════════════════════════════════

def _load_one_stock_panel(
    csv_path: str,
    factors: list[str],
    forward_periods: list[int],
    period_suffix: str,
    min_rows: int,
) -> Optional[pd.DataFrame]:
    """
    加载一只股票的时序数据, 计算前瞻收益, 返回 panel 片段。

    Returns:
        DataFrame with columns: [trade_date, symbol] + factors + [fwd_ret_1d, fwd_ret_3d, ...]
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    if len(df) < min_rows:
        return None

    if "trade_date" not in df.columns or "close" not in df.columns:
        return None

    df = df.sort_values("trade_date").reset_index(drop=True)
    symbol = os.path.basename(csv_path).replace(".csv", "")

    # 计算前瞻收益
    for fp in forward_periods:
        df[f"fwd_ret_{fp}{period_suffix}"] = df["close"].shift(-fp) / df["close"] - 1

    # 只保留需要的列
    keep_cols = ["trade_date"] + [f for f in factors if f in df.columns] + \
                [f"fwd_ret_{fp}{period_suffix}" for fp in forward_periods]
    result = df[keep_cols].copy()
    result["symbol"] = symbol

    # 去掉前瞻收益为 NaN 的最后几行 (最大 forward period)
    max_fp = max(forward_periods)
    result = result.iloc[:len(result) - max_fp]

    return result


def build_panel(
    cache_dir: str,
    timeframe: str,
    cfg: ModelConfig,
) -> pd.DataFrame:
    """
    构建全市场截面 panel。

    Args:
        cache_dir: daily/ 或 weekly/ 目录
        timeframe: "daily" 或 "weekly"

    Returns:
        DataFrame: (date×stock) 行, 因子+前瞻收益 列
    """
    if timeframe == "daily":
        factors = DAILY_FACTORS
        forward_periods = [1, 3, 5]
        period_suffix = "d"
        min_rows = cfg.min_rows
    else:
        factors = WEEKLY_FACTORS
        forward_periods = [1, 3, 5]
        period_suffix = "w"
        min_rows = cfg.min_weekly_rows

    csv_dir = os.path.join(cache_dir, timeframe)
    all_csvs = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if cfg.max_stocks > 0:
        all_csvs = all_csvs[:cfg.max_stocks]

    logger.info(f"Building {timeframe} panel from {len(all_csvs)} CSVs ...")

    panels = []
    for csv_path in all_csvs:
        p = _load_one_stock_panel(csv_path, factors, forward_periods, period_suffix, min_rows)
        if p is not None:
            panels.append(p)

    if not panels:
        return pd.DataFrame()

    panel = pd.concat(panels, ignore_index=True)
    logger.info(f"{timeframe} panel: {len(panel)} rows, {panel['symbol'].nunique()} stocks, "
                f"date range {panel['trade_date'].min()} ~ {panel['trade_date'].max()}")
    return panel


# ═════════════════════════════════════════════════════════
# Step 2: 时间衰减权重
# ═════════════════════════════════════════════════════════

def compute_sample_weights(
    dates: pd.Series,
    decay_lambda: float,
) -> np.ndarray:
    """
    指数时间衰减权重: w_t = exp(-λ * (T - rank_t))
    T = 最大秩 (最新日期), rank_t = 日期排序后的秩

    越近的日期权重越大。
    """
    # 将日期映射为时间序号 (最新=T, 最旧=0)
    unique_dates = sorted(dates.unique())
    date_rank = {d: i for i, d in enumerate(unique_dates)}
    T = len(unique_dates) - 1

    ranks = dates.map(date_rank).values
    weights = np.exp(-decay_lambda * (T - ranks))

    return weights


# ═════════════════════════════════════════════════════════
# Step 3: Walk-Forward 训练
# ═════════════════════════════════════════════════════════

def train_one_horizon(
    panel: pd.DataFrame,
    horizon: str,
    factors: list[str],
    cfg: ModelConfig,
    decay_lambda: float,
) -> dict:
    """
    训练单个 horizon 的 LightGBM 模型。

    Walk-forward 分割:
      train: trade_date <= train_end
      val:   train_end < trade_date <= val_end

    Returns:
        dict with model, metrics, feature_importance
    """
    target_col = f"fwd_ret_{horizon}"
    if target_col not in panel.columns:
        logger.warning(f"Target {target_col} not in panel")
        return {}

    # ── 特征列 (只保留存在且非全NaN的因子) ──
    feature_cols = [f for f in factors if f in panel.columns and panel[f].notna().sum() > 100]
    if not feature_cols:
        logger.warning(f"No valid features for {horizon}")
        return {}

    # ── 过滤有效行 ──
    valid_mask = panel[target_col].notna()
    df = panel[valid_mask].copy()

    # ── Walk-forward 分割 ──
    all_dates = sorted(df["trade_date"].unique())
    if cfg.train_end:
        train_end = cfg.train_end
    else:
        # 默认: 最新日期往前推 val_days
        train_end = all_dates[-cfg.val_days] if len(all_dates) > cfg.val_days else all_dates[-1]

    # 验证集
    val_dates = [d for d in all_dates if d > train_end]
    val_end = val_dates[min(cfg.val_days - 1, len(val_dates) - 1)] if val_dates else train_end

    train_mask = df["trade_date"] <= train_end
    val_mask = (df["trade_date"] > train_end) & (df["trade_date"] <= val_end)

    train_df = df[train_mask]
    val_df = df[val_mask]

    logger.info(f"  {horizon}: train {len(train_df)} rows (≤{train_end}), "
                f"val {len(val_df)} rows ({train_end}~{val_end})")

    if len(train_df) < 1000 or len(val_df) < 100:
        logger.warning(f"  {horizon}: insufficient data, skipping")
        return {}

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_val = val_df[feature_cols].values
    y_val = val_df[target_col].values

    # ── 时间衰减样本权重 ──
    train_weights = compute_sample_weights(train_df["trade_date"], decay_lambda)
    val_weights = compute_sample_weights(val_df["trade_date"], decay_lambda)

    # ── LightGBM 训练 ──
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
        sample_weight=train_weights,
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[val_weights],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    # ── 评估 ──
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)

    # Rank IC (Spearman)
    from scipy.stats import spearmanr
    train_ic, _ = spearmanr(pred_train, y_train)
    val_ic, _ = spearmanr(pred_val, y_val)

    # 加权 Rank IC (验证集)
    weighted_val_ic = _weighted_rank_ic(pred_val, y_val, val_weights)

    # Top/Bottom 分组收益
    val_quantile = _quantile_analysis(pred_val, y_val, n_q=5)

    # 特征重要性
    importance = dict(zip(feature_cols, model.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    metrics = {
        "horizon": horizon,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "train_date_range": [str(train_df["trade_date"].min()), str(train_end)],
        "val_date_range": [str(val_dates[0]) if val_dates else "", str(val_end)],
        "n_features": len(feature_cols),
        "best_iteration": model.best_iteration_,
        "train_rank_ic": round(train_ic, 4),
        "val_rank_ic": round(val_ic, 4),
        "val_weighted_rank_ic": round(weighted_val_ic, 4),
        "val_quantile_returns": val_quantile,
        "top_features": {k: int(v) for k, v in list(importance.items())[:10]},
    }

    logger.info(f"  {horizon}: train IC={train_ic:.4f}, val IC={val_ic:.4f}, "
                f"val weighted IC={weighted_val_ic:.4f}, "
                f"best_iter={model.best_iteration_}")

    return {
        "model": model,
        "feature_cols": feature_cols,
        "metrics": metrics,
    }


def _weighted_rank_ic(pred: np.ndarray, actual: np.ndarray, weights: np.ndarray) -> float:
    """加权 Spearman Rank IC"""
    mask = np.isfinite(pred) & np.isfinite(actual)
    if mask.sum() < 30:
        return 0.0
    p, a, w = pred[mask], actual[mask], weights[mask]
    rp = pd.Series(p).rank().values
    ra = pd.Series(a).rank().values
    w_sum = w.sum()
    mp = np.average(rp, weights=w)
    ma = np.average(ra, weights=w)
    cov = np.sum(w * (rp - mp) * (ra - ma)) / w_sum
    sp = np.sqrt(np.sum(w * (rp - mp) ** 2) / w_sum)
    sa = np.sqrt(np.sum(w * (ra - ma) ** 2) / w_sum)
    if sp < 1e-12 or sa < 1e-12:
        return 0.0
    return float(cov / (sp * sa))


def _quantile_analysis(pred: np.ndarray, actual: np.ndarray, n_q: int = 5) -> dict:
    """分组分析"""
    mask = np.isfinite(pred) & np.isfinite(actual)
    if mask.sum() < n_q * 20:
        return {}
    p, a = pred[mask], actual[mask]
    try:
        bins = pd.qcut(pd.Series(p), n_q, labels=False, duplicates="drop")
    except ValueError:
        return {}
    result = {}
    for q in sorted(bins.unique()):
        idx = bins.values == q
        result[f"Q{q+1}"] = round(float(a[idx].mean()), 6)
    return result


# ═════════════════════════════════════════════════════════
# Step 4: 训练主流程
# ═════════════════════════════════════════════════════════

def train_all(cfg: ModelConfig) -> dict:
    """
    训练全部 6 个 horizon 的模型。

    Returns:
        {horizon: {model, feature_cols, metrics}}
    """
    model_dir = cfg.model_dir or os.path.join(cfg.cache_dir, "lgb_models")
    os.makedirs(model_dir, exist_ok=True)

    results = {}

    # ── 日线 ──
    daily_panel = build_panel(cfg.cache_dir, "daily", cfg)
    if not daily_panel.empty:
        for fd in [1, 3, 5]:
            h = f"{fd}d"
            res = train_one_horizon(daily_panel, h, DAILY_FACTORS, cfg, cfg.decay_lambda)
            if res:
                results[h] = res

    # ── 周线 ──
    weekly_panel = build_panel(cfg.cache_dir, "weekly", cfg)
    if not weekly_panel.empty:
        for fw in [1, 3, 5]:
            h = f"{fw}w"
            res = train_one_horizon(weekly_panel, h, WEEKLY_FACTORS, cfg, cfg.weekly_decay_lambda)
            if res:
                results[h] = res

    # ── 保存模型 ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {
        "trained_at": timestamp,
        "horizons": {},
    }

    for h, res in results.items():
        model_path = os.path.join(model_dir, f"lgb_{h}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": res["model"],
                "feature_cols": res["feature_cols"],
            }, f)
        meta["horizons"][h] = res["metrics"]
        logger.info(f"Saved {model_path}")

    meta_path = os.path.join(model_dir, "model_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {meta_path}")

    # ── 打印总结 ──
    _print_summary(results)

    return results


def _print_summary(results: dict):
    """打印训练总结"""
    print("\n" + "=" * 70)
    print("LightGBM 因子模型训练总结")
    print("=" * 70)
    print(f"{'Horizon':<10} {'Train IC':>10} {'Val IC':>10} {'Val WtdIC':>10} "
          f"{'BestIter':>10} {'Q1→Q5':>15}")
    print("-" * 70)

    for h in ALL_HORIZONS:
        if h not in results:
            print(f"{h:<10} {'(skipped)':>10}")
            continue
        m = results[h]["metrics"]
        qr = m.get("val_quantile_returns", {})
        q1 = qr.get("Q1", 0)
        q5 = qr.get("Q5", 0) if len(qr) >= 5 else qr.get(f"Q{len(qr)}", 0)
        spread = f"{q1:.4f}→{q5:.4f}" if qr else "N/A"
        print(f"{h:<10} {m['train_rank_ic']:>10.4f} {m['val_rank_ic']:>10.4f} "
              f"{m['val_weighted_rank_ic']:>10.4f} {m['best_iteration']:>10} "
              f"{spread:>15}")

    print()

    # 特征重要性汇总
    print("Top 特征 (按模型重要性):")
    for h in ALL_HORIZONS:
        if h not in results:
            continue
        m = results[h]["metrics"]
        top_f = list(m.get("top_features", {}).items())[:5]
        if top_f:
            feats = ", ".join(f"{f}({v})" for f, v in top_f)
            print(f"  {h}: {feats}")


# ═════════════════════════════════════════════════════════
# Step 5: 预测 (供 signal_detector 调用)
# ═════════════════════════════════════════════════════════

def load_models(model_dir: str) -> dict:
    """
    加载已训练的模型。

    Returns:
        {horizon: {"model": LGBMRegressor, "feature_cols": [...]}}
    """
    models = {}
    for h in ALL_HORIZONS:
        pkl_path = os.path.join(model_dir, f"lgb_{h}.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                models[h] = pickle.load(f)
            logger.info(f"Loaded model for {h} from {pkl_path}")
    return models


def predict_scores(
    models: dict,
    daily_features: pd.DataFrame,
    weekly_features: pd.DataFrame,
) -> dict[str, pd.Series]:
    """
    用模型对最新截面做预测。

    Args:
        models: {horizon: {"model", "feature_cols"}}
        daily_features: 日线最新因子值 (index=symbol, columns=因子)
        weekly_features: 周线最新因子值

    Returns:
        {horizon: pd.Series(index=symbol, values=predicted_return)}
    """
    scores = {}

    for h in DAILY_HORIZONS:
        if h not in models or daily_features.empty:
            continue
        m = models[h]
        feature_cols = m["feature_cols"]
        available = [c for c in feature_cols if c in daily_features.columns]
        if len(available) < len(feature_cols) * 0.5:
            logger.warning(f"{h}: only {len(available)}/{len(feature_cols)} features available")
            continue

        X = daily_features[available].copy()
        # 补齐缺失列
        for c in feature_cols:
            if c not in X.columns:
                X[c] = 0.0
        X = X[feature_cols]  # 确保列顺序
        X = X.fillna(0.0)

        pred = m["model"].predict(X.values)
        scores[h] = pd.Series(pred, index=daily_features.index, name=h)

    for h in WEEKLY_HORIZONS:
        if h not in models or weekly_features.empty:
            continue
        m = models[h]
        feature_cols = m["feature_cols"]
        available = [c for c in feature_cols if c in weekly_features.columns]
        if len(available) < len(feature_cols) * 0.5:
            logger.warning(f"{h}: only {len(available)}/{len(feature_cols)} features available")
            continue

        X = weekly_features[available].copy()
        for c in feature_cols:
            if c not in X.columns:
                X[c] = 0.0
        X = X[feature_cols]
        X = X.fillna(0.0)

        pred = m["model"].predict(X.values)
        scores[h] = pd.Series(pred, index=weekly_features.index, name=h)

    return scores


# ═════════════════════════════════════════════════════════
# Step 6: 选股 (整合残差法)
# ═════════════════════════════════════════════════════════

def run_lgb_scoring(
    model_dir: str,
    daily_cache_dir: str,
    weekly_cache_dir: str,
    scan_date: str = "",
    basic_info: Optional[dict[str, dict]] = None,
    cfg: Optional[ModelConfig] = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    LightGBM 模型选股: 加载模型 → 预测 → 残差法去重 → Top 5/horizon

    接口与 ic_scoring.run_ic_scoring 兼容。
    """
    from .ic_scoring import (
        load_latest_features, ScoringConfig, _apply_filters,
    )

    cfg = cfg or ModelConfig()
    basic_info = basic_info or {}

    # 1. 加载模型
    models = load_models(model_dir)
    if not models:
        logger.error(f"No models found in {model_dir}")
        return {}, pd.DataFrame()

    # 2. 加载特征
    daily_features = load_latest_features(daily_cache_dir, scan_date, cfg.min_rows)
    weekly_features = load_latest_features(weekly_cache_dir, scan_date, cfg.min_rows)

    # 3. 过滤
    scfg = ScoringConfig(min_data_rows=cfg.min_rows)
    daily_features = _apply_filters(daily_features, basic_info, scfg)
    if not weekly_features.empty:
        weekly_features = _apply_filters(weekly_features, basic_info, scfg)

    logger.info(f"After filtering: {len(daily_features)} daily, {len(weekly_features)} weekly stocks")

    # 4. 预测
    raw_scores = predict_scores(models, daily_features, weekly_features)

    # 5. 残差法选股
    horizon_tops: dict[str, pd.DataFrame] = {}
    top_n = cfg.top_n_per_horizon

    def _ols_residual(y: pd.Series, x: pd.Series) -> pd.Series:
        common = y.index.intersection(x.index)
        if len(common) < 30:
            return y
        yc, xc = y.reindex(common).fillna(0), x.reindex(common).fillna(0)
        dot_xx = (xc * xc).sum()
        if dot_xx < 1e-12:
            return y
        beta = (xc * yc).sum() / dot_xx
        residual = yc - beta * xc
        result = y.copy()
        result.loc[common] = residual
        return result

    prev = None
    for h in DAILY_HORIZONS:
        if h not in raw_scores:
            horizon_tops[h] = pd.DataFrame()
            continue
        sel = raw_scores[h]
        if prev is not None:
            sel = _ols_residual(sel, prev)
        top_idx = sel.nlargest(top_n).index
        features_src = daily_features
        rows = []
        for rank, sym in enumerate(top_idx, 1):
            rows.append({
                "rank": rank, "symbol": sym,
                "name": basic_info.get(sym, {}).get("name", ""),
                "industry": basic_info.get(sym, {}).get("industry", ""),
                "horizon": h,
                "score": round(float(raw_scores[h][sym]), 6),
                "residual_score": round(float(sel[sym]), 6),
            })
        horizon_tops[h] = pd.DataFrame(rows)
        prev = raw_scores[h]

    prev = None
    for h in WEEKLY_HORIZONS:
        if h not in raw_scores:
            horizon_tops[h] = pd.DataFrame()
            continue
        sel = raw_scores[h]
        if prev is not None:
            sel = _ols_residual(sel, prev)
        top_idx = sel.nlargest(top_n).index
        rows = []
        for rank, sym in enumerate(top_idx, 1):
            rows.append({
                "rank": rank, "symbol": sym,
                "name": basic_info.get(sym, {}).get("name", ""),
                "industry": basic_info.get(sym, {}).get("industry", ""),
                "horizon": h,
                "score": round(float(raw_scores[h][sym]), 6),
                "residual_score": round(float(sel[sym]), 6),
            })
        horizon_tops[h] = pd.DataFrame(rows)
        prev = raw_scores[h]

    # 6. 合并
    combined = pd.concat(
        [df for df in horizon_tops.values() if len(df) > 0],
        ignore_index=True,
    )
    n_unique = combined["symbol"].nunique() if len(combined) > 0 else 0
    logger.info(f"Selected {len(combined)} candidates ({n_unique} unique) across {len(ALL_HORIZONS)} horizons")

    return horizon_tops, combined


# ═════════════════════════════════════════════════════════
# Step 7: Walk-Forward 回测
# ═════════════════════════════════════════════════════════

def run_backtest(
    cache_dir: str,
    data_dir: str,
    model_dir: str,
    start_date: str,
    end_date: str,
    horizons: list[str] | None = None,
    top_n: int = 5,
    scan_interval: int = 5,
    basic_path: str = "",
    out_dir: str = "",
) -> dict:
    """
    Walk-forward 回测:
      每隔 scan_interval 天扫描 → 模型评分 Top N → 次日开盘买 → 持有 horizon 天 → 收盘卖

    每个 horizon 独立回测 (买入组合不互相干扰)。

    Returns:
        {horizon: {"trades": [...], "metrics": {...}}}
    """
    horizons = horizons or ["3d", "5d", "1w"]

    # 映射 horizon → 持有天数
    hold_map = {"1d": 1, "3d": 3, "5d": 5, "1w": 5, "3w": 15, "5w": 25}

    # 加载模型
    models = load_models(model_dir)
    if not models:
        logger.error("No models found")
        return {}

    # 加载 basic_info
    basic_info = {}
    if basic_path and os.path.exists(basic_path):
        bdf = pd.read_csv(basic_path, dtype=str)
        for _, row in bdf.iterrows():
            ts = str(row.get("ts_code", ""))
            parts = ts.split(".")
            if len(parts) == 2:
                sym = parts[1].lower() + parts[0]
                basic_info[sym] = {
                    "name": str(row.get("name", "")),
                    "industry": str(row.get("industry", "")),
                }

    # 加载全部日线原始数据 (需要 OHLC 做买卖价格模拟)
    logger.info("Loading raw daily data for backtest ...")
    daily_dir = os.path.join(data_dir)
    raw_data: dict[str, pd.DataFrame] = {}
    for csv_file in sorted(glob.glob(os.path.join(daily_dir, "*.csv"))):
        sym = os.path.basename(csv_file).replace(".csv", "")
        try:
            df = pd.read_csv(csv_file, usecols=["trade_date", "open", "close", "amount"])
            df["trade_date"] = df["trade_date"].astype(str)
            df = df.sort_values("trade_date").reset_index(drop=True)
            raw_data[sym] = df
        except Exception:
            continue
    logger.info(f"Loaded raw data for {len(raw_data)} stocks")

    # 构建交易日历
    all_dates = set()
    for df in list(raw_data.values())[:50]:
        all_dates.update(df["trade_date"].tolist())
    calendar = sorted(all_dates)
    bt_dates = [d for d in calendar if start_date <= d <= end_date]
    logger.info(f"Backtest period: {len(bt_dates)} trading days [{start_date} ~ {end_date}]")

    # 加载 feature panel (需要全量数据按日期切片)
    logger.info("Loading feature panels ...")
    daily_cache = os.path.join(cache_dir, "daily")
    weekly_cache = os.path.join(cache_dir, "weekly")

    # 日线 panel: {symbol: DataFrame}
    daily_panels: dict[str, pd.DataFrame] = {}
    for csv_file in sorted(glob.glob(os.path.join(daily_cache, "*.csv"))):
        sym = os.path.basename(csv_file).replace(".csv", "")
        try:
            df = pd.read_csv(csv_file)
            df["trade_date"] = df["trade_date"].astype(str)
            df = df.sort_values("trade_date").reset_index(drop=True)
            daily_panels[sym] = df
        except Exception:
            continue

    weekly_panels: dict[str, pd.DataFrame] = {}
    has_weekly = any(h in horizons for h in WEEKLY_HORIZONS)
    if has_weekly:
        for csv_file in sorted(glob.glob(os.path.join(weekly_cache, "*.csv"))):
            sym = os.path.basename(csv_file).replace(".csv", "")
            try:
                df = pd.read_csv(csv_file)
                df["trade_date"] = df["trade_date"].astype(str)
                df = df.sort_values("trade_date").reset_index(drop=True)
                weekly_panels[sym] = df
            except Exception:
                continue

    logger.info(f"Feature panels: {len(daily_panels)} daily, {len(weekly_panels)} weekly")

    # ── 逐 horizon 回测 ──
    all_results = {}

    for h in horizons:
        if h not in models:
            logger.warning(f"No model for {h}, skipping")
            continue

        hold_days = hold_map.get(h, 5)
        is_weekly = h in WEEKLY_HORIZONS
        model_info = models[h]
        feature_cols = model_info["feature_cols"]
        model = model_info["model"]

        logger.info(f"\n{'='*50}")
        logger.info(f"Backtesting {h} (hold={hold_days}d, top_n={top_n}, interval={scan_interval}d)")
        logger.info(f"{'='*50}")

        trades = []
        scan_days = bt_dates[::scan_interval]

        for si, scan_date in enumerate(scan_days):
            # 找到 scan_date 在日历中的位置
            try:
                cal_idx = calendar.index(scan_date)
            except ValueError:
                continue

            # 下一个交易日作为买入日
            if cal_idx + 1 >= len(calendar):
                continue
            entry_date = calendar[cal_idx + 1]

            # 卖出日 = 买入日后 hold_days 个交易日
            if cal_idx + 1 + hold_days >= len(calendar):
                continue
            exit_date = calendar[cal_idx + 1 + hold_days]

            # 为当天截面构建特征矩阵
            panels = weekly_panels if is_weekly else daily_panels
            rows = []
            syms = []
            for sym, pdf in panels.items():
                # 找到 <= scan_date 的最后一行
                mask = pdf["trade_date"] <= scan_date
                if mask.sum() < 30:
                    continue
                last_row = pdf[mask].iloc[-1]
                # 过滤: 检查成交额
                if sym in raw_data:
                    rdf = raw_data[sym]
                    rmask = rdf["trade_date"] <= scan_date
                    if rmask.sum() > 20:
                        recent = rdf[rmask].tail(20)
                        if recent["amount"].mean() < 5000:
                            continue
                # 过滤: ST
                info = basic_info.get(sym, {})
                name = info.get("name", "")
                if "ST" in name or "退" in name:
                    continue

                feat_vals = []
                for f in feature_cols:
                    v = last_row.get(f, np.nan)
                    feat_vals.append(float(v) if pd.notna(v) else 0.0)
                rows.append(feat_vals)
                syms.append(sym)

            if len(rows) < 50:
                continue

            X = np.array(rows)
            preds = model.predict(X)
            pred_series = pd.Series(preds, index=syms)

            # 取 top N
            top_syms = pred_series.nlargest(top_n).index.tolist()

            # 记录交易
            for sym in top_syms:
                if sym not in raw_data:
                    continue
                rdf = raw_data[sym]

                # 买入价: entry_date 的开盘价
                entry_rows = rdf[rdf["trade_date"] == entry_date]
                if entry_rows.empty:
                    continue
                entry_price = float(entry_rows.iloc[0]["open"])
                if entry_price <= 0:
                    continue

                # 卖出价: exit_date 的收盘价
                exit_rows = rdf[rdf["trade_date"] == exit_date]
                if exit_rows.empty:
                    continue
                exit_price = float(exit_rows.iloc[0]["close"])

                pnl_pct = (exit_price - entry_price) / entry_price * 100

                trades.append({
                    "signal_date": scan_date,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "symbol": sym,
                    "name": basic_info.get(sym, {}).get("name", ""),
                    "industry": basic_info.get(sym, {}).get("industry", ""),
                    "pred_score": round(float(pred_series[sym]), 6),
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(exit_price, 2),
                    "pnl_pct": round(pnl_pct, 2),
                    "hold_days": hold_days,
                })

            if (si + 1) % 10 == 0:
                logger.info(f"  {h}: {si+1}/{len(scan_days)} scan days processed, {len(trades)} trades")

        # 计算绩效
        metrics = _compute_backtest_metrics(trades, h, hold_days)
        all_results[h] = {"trades": trades, "metrics": metrics}
        _print_backtest_result(h, trades, metrics)

    # 保存结果
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for h, res in all_results.items():
            if res["trades"]:
                df_trades = pd.DataFrame(res["trades"])
                df_trades.to_csv(os.path.join(out_dir, f"lgb_backtest_{h}.csv"), index=False)
        # 汇总
        summary_rows = []
        for h, res in all_results.items():
            m = res["metrics"]
            m["horizon"] = h
            summary_rows.append(m)
        if summary_rows:
            pd.DataFrame(summary_rows).to_csv(
                os.path.join(out_dir, "lgb_backtest_summary.csv"), index=False)
        logger.info(f"Backtest results saved to {out_dir}")

    return all_results


def _compute_backtest_metrics(trades: list[dict], horizon: str, hold_days: int) -> dict:
    """计算单个 horizon 的回测绩效"""
    if not trades:
        return {"n_trades": 0}

    pnls = [t["pnl_pct"] for t in trades]
    pnl_arr = np.array(pnls)

    n = len(pnl_arr)
    wins = int((pnl_arr > 0).sum())
    win_rate = wins / n
    avg_pnl = float(pnl_arr.mean())
    med_pnl = float(np.median(pnl_arr))

    avg_win = float(pnl_arr[pnl_arr > 0].mean()) if wins > 0 else 0
    avg_loss = float(pnl_arr[pnl_arr < 0].mean()) if (n - wins) > 0 else 0
    profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    # 按信号日聚合的等权组合收益曲线
    df = pd.DataFrame(trades)
    group_ret = df.groupby("signal_date")["pnl_pct"].mean()
    cum_ret = (1 + group_ret / 100).cumprod()
    total_return = float((cum_ret.iloc[-1] - 1) * 100) if len(cum_ret) > 0 else 0

    # 最大回撤
    peak = cum_ret.expanding().max()
    dd = (cum_ret - peak) / peak * 100
    max_drawdown = float(dd.min())

    # 年化 (假设每次持有 hold_days 天)
    n_groups = len(group_ret)
    if n_groups > 1:
        first_date = group_ret.index.min()
        last_date = group_ret.index.max()
        # 粗略年化
        days_span = (pd.Timestamp(last_date) - pd.Timestamp(first_date)).days
        if days_span > 30:
            annual_return = total_return / days_span * 365
        else:
            annual_return = total_return
    else:
        annual_return = total_return

    # Sharpe (基于每组收益)
    if group_ret.std() > 0:
        # 年化 Sharpe: mean/std * sqrt(交易周期数/年)
        trades_per_year = 252 / max(hold_days, 1)
        sharpe = float(group_ret.mean() / group_ret.std() * np.sqrt(trades_per_year))
    else:
        sharpe = 0.0

    return {
        "n_trades": n,
        "n_groups": n_groups,
        "win_rate": round(win_rate, 4),
        "avg_pnl": round(avg_pnl, 4),
        "median_pnl": round(med_pnl, 4),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "profit_loss_ratio": round(profit_loss_ratio, 2),
        "total_return": round(total_return, 2),
        "annual_return": round(annual_return, 2),
        "max_drawdown": round(max_drawdown, 2),
        "sharpe": round(sharpe, 2),
        "best_trade": round(float(pnl_arr.max()), 2),
        "worst_trade": round(float(pnl_arr.min()), 2),
    }


def _print_backtest_result(horizon: str, trades: list[dict], metrics: dict):
    """打印单个 horizon 的回测结果"""
    if not trades:
        print(f"\n  {horizon}: 无交易")
        return

    m = metrics
    print(f"\n{'─'*60}")
    print(f"  {horizon} 回测结果 (持有{trades[0]['hold_days']}天)")
    print(f"{'─'*60}")
    print(f"  交易次数:    {m['n_trades']} ({m['n_groups']} 组)")
    print(f"  胜率:        {m['win_rate']:.1%}")
    print(f"  平均收益:    {m['avg_pnl']:+.2f}%")
    print(f"  中位收益:    {m['median_pnl']:+.2f}%")
    print(f"  盈亏比:      {m['profit_loss_ratio']:.2f}")
    print(f"  总收益:      {m['total_return']:+.2f}%")
    print(f"  年化收益:    {m['annual_return']:+.2f}%")
    print(f"  最大回撤:    {m['max_drawdown']:.2f}%")
    print(f"  Sharpe:      {m['sharpe']:.2f}")
    print(f"  最好单笔:    {m['best_trade']:+.2f}%")
    print(f"  最差单笔:    {m['worst_trade']:+.2f}%")

    # 按月分组收益
    df = pd.DataFrame(trades)
    df["month"] = df["signal_date"].str[:6]
    monthly = df.groupby("month")["pnl_pct"].agg(["mean", "count", lambda x: (x > 0).mean()])
    monthly.columns = ["avg_pnl", "n_trades", "win_rate"]
    print(f"\n  月度表现:")
    print(f"  {'月份':<10} {'平均收益':>8} {'交易数':>6} {'胜率':>6}")
    for month, row in monthly.iterrows():
        print(f"  {month:<10} {row['avg_pnl']:>+8.2f}% {int(row['n_trades']):>6} {row['win_rate']:>6.1%}")


# ═════════════════════════════════════════════════════════
# Step 8: Walk-Forward 回测 (带滚动再训练, 无前瞻偏差)
# ═════════════════════════════════════════════════════════

def _train_inline(
    panel: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    train_cutoff: str,
    decay_lambda: float,
    cfg: ModelConfig,
) -> tuple[Optional[lgb.LGBMRegressor], dict]:
    """
    为 walk-forward 回测内联训练单个 LightGBM 模型。

    只使用 trade_date <= train_cutoff 的数据, 消除前瞻偏差。
    最后 val_days 天做验证集, 其余做训练集。
    """
    mask = (panel["trade_date"] <= train_cutoff) & panel[target_col].notna()
    data = panel[mask]

    if len(data) < 5000:
        return None, {}

    all_dates = sorted(data["trade_date"].unique())
    if len(all_dates) <= cfg.val_days + 20:
        return None, {}

    val_start = all_dates[-cfg.val_days]
    t_mask = data["trade_date"] < val_start
    v_mask = data["trade_date"] >= val_start

    X_train = data[t_mask][feature_cols].fillna(0).values
    y_train = data[t_mask][target_col].values
    X_val = data[v_mask][feature_cols].fillna(0).values
    y_val = data[v_mask][target_col].values

    if len(X_train) < 1000 or len(X_val) < 100:
        return None, {}

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

    return model, info


def run_backtest_wf(
    cache_dir: str,
    data_dir: str,
    start_date: str,
    end_date: str,
    horizons: list[str] | None = None,
    top_n: int = 5,
    scan_interval: int = 5,
    retrain_every: int = 1,
    basic_path: str = "",
    out_dir: str = "",
    cfg: ModelConfig | None = None,
) -> dict:
    """
    Walk-forward 回测 (带滚动再训练, 无前瞻偏差):

    每隔 scan_interval 天:
      1. 用截止当日之前已实现收益的数据重新训练 LightGBM
      2. 模型评分 Top N
      3. 次日开盘买 → 持有 hold 天 → 收盘卖

    与 run_backtest 的区别: 不加载预训练模型, 而是在每个扫描日
    (或每 retrain_every 个扫描日) 用历史数据重新训练, 消除前瞻偏差。

    Args:
        retrain_every: 每隔 N 个 scan day 重新训练 (1=每次, 4=每20天)
    """
    cfg = cfg or ModelConfig()
    horizons = horizons or ["3d", "5d", "1w"]
    hold_map = {"1d": 1, "3d": 3, "5d": 5, "1w": 5, "3w": 15, "5w": 25}

    # ── 1. 构建训练 panel (一次加载, 含前瞻收益) ──
    logger.info("Building training panels (one-time) ...")
    daily_panel = pd.DataFrame()
    weekly_panel = pd.DataFrame()

    need_daily = any(h in DAILY_HORIZONS for h in horizons)
    need_weekly = any(h in WEEKLY_HORIZONS for h in horizons)

    if need_daily:
        daily_panel = build_panel(cache_dir, "daily", cfg)
        if not daily_panel.empty:
            daily_panel["trade_date"] = daily_panel["trade_date"].astype(str)
    if need_weekly:
        weekly_panel = build_panel(cache_dir, "weekly", cfg)
        if not weekly_panel.empty:
            weekly_panel["trade_date"] = weekly_panel["trade_date"].astype(str)

    # ── 2. 加载预测用特征 (per-stock, 不截断尾部) ──
    logger.info("Loading feature panels for prediction ...")
    daily_feat_panels: dict[str, pd.DataFrame] = {}
    for csv_file in sorted(glob.glob(os.path.join(cache_dir, "daily", "*.csv"))):
        sym = os.path.basename(csv_file).replace(".csv", "")
        try:
            df = pd.read_csv(csv_file)
            df["trade_date"] = df["trade_date"].astype(str)
            df = df.sort_values("trade_date").reset_index(drop=True)
            daily_feat_panels[sym] = df
        except Exception:
            continue

    weekly_feat_panels: dict[str, pd.DataFrame] = {}
    if need_weekly:
        for csv_file in sorted(glob.glob(os.path.join(cache_dir, "weekly", "*.csv"))):
            sym = os.path.basename(csv_file).replace(".csv", "")
            try:
                df = pd.read_csv(csv_file)
                df["trade_date"] = df["trade_date"].astype(str)
                df = df.sort_values("trade_date").reset_index(drop=True)
                weekly_feat_panels[sym] = df
            except Exception:
                continue

    logger.info(f"Feature panels: {len(daily_feat_panels)} daily, {len(weekly_feat_panels)} weekly")

    # ── 3. 加载原始日线数据 (OHLC 做交易模拟) ──
    logger.info("Loading raw daily data ...")
    raw_data: dict[str, pd.DataFrame] = {}
    for csv_file in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        sym = os.path.basename(csv_file).replace(".csv", "")
        try:
            df = pd.read_csv(csv_file, usecols=["trade_date", "open", "close", "amount"])
            df["trade_date"] = df["trade_date"].astype(str)
            df = df.sort_values("trade_date").reset_index(drop=True)
            raw_data[sym] = df
        except Exception:
            continue
    logger.info(f"Loaded raw data for {len(raw_data)} stocks")

    # ── 4. 交易日历 ──
    all_dates_set: set[str] = set()
    for df in list(raw_data.values())[:50]:
        all_dates_set.update(df["trade_date"].tolist())
    calendar = sorted(all_dates_set)
    bt_dates = [d for d in calendar if start_date <= d <= end_date]
    logger.info(f"Backtest: {len(bt_dates)} trading days [{start_date} ~ {end_date}]")

    # ── 5. basic_info ──
    basic_info: dict[str, dict] = {}
    if basic_path and os.path.exists(basic_path):
        bdf = pd.read_csv(basic_path, dtype=str)
        for _, row in bdf.iterrows():
            ts = str(row.get("ts_code", ""))
            parts = ts.split(".")
            if len(parts) == 2:
                sym = parts[1].lower() + parts[0]
                basic_info[sym] = {
                    "name": str(row.get("name", "")),
                    "industry": str(row.get("industry", "")),
                }

    # ── 6. 逐 horizon Walk-forward 回测 ──
    all_results = {}

    for h in horizons:
        hold_days = hold_map.get(h, 5)
        is_weekly = h in WEEKLY_HORIZONS
        panel = weekly_panel if is_weekly else daily_panel
        factors = WEEKLY_FACTORS if is_weekly else DAILY_FACTORS
        decay = cfg.weekly_decay_lambda if is_weekly else cfg.decay_lambda
        target_col = f"fwd_ret_{h}"
        feat_panels = weekly_feat_panels if is_weekly else daily_feat_panels

        if panel.empty or target_col not in panel.columns:
            logger.warning(f"No panel/target for {h}, skipping")
            continue

        feature_cols = [f for f in factors if f in panel.columns
                        and panel[f].notna().sum() > 100]
        if not feature_cols:
            logger.warning(f"No valid features for {h}")
            continue

        logger.info(f"\n{'='*50}")
        logger.info(f"Walk-forward {h} (hold={hold_days}d, top_n={top_n}, "
                    f"interval={scan_interval}d, retrain_every={retrain_every})")
        logger.info(f"{'='*50}")

        trades: list[dict] = []
        current_model: Optional[lgb.LGBMRegressor] = None
        scan_days = bt_dates[::scan_interval]
        train_count = 0

        for si, scan_date in enumerate(scan_days):
            try:
                cal_idx = calendar.index(scan_date)
            except ValueError:
                continue

            if cal_idx + 1 >= len(calendar):
                continue
            entry_date = calendar[cal_idx + 1]

            if cal_idx + 1 + hold_days >= len(calendar):
                continue
            exit_date = calendar[cal_idx + 1 + hold_days]

            # 训练截止: 前瞻收益必须已实现
            # 对于 hold_days=5 的模型, trade_date + 5 天的 close 必须已知
            # 所以训练数据的 trade_date <= calendar[cal_idx - hold_days]
            train_cutoff_idx = cal_idx - hold_days
            if train_cutoff_idx < 60:
                continue
            train_cutoff = calendar[train_cutoff_idx]

            # 是否重新训练
            should_retrain = (current_model is None) or (si % retrain_every == 0)

            if should_retrain:
                model, tinfo = _train_inline(
                    panel, target_col, feature_cols, train_cutoff, decay, cfg
                )
                if model is not None:
                    current_model = model
                    train_count += 1
                    if train_count <= 3 or train_count % 5 == 0:
                        logger.info(
                            f"  [{scan_date}] Retrained #{train_count}: "
                            f"cutoff={train_cutoff}, "
                            f"val_IC={tinfo.get('val_ic','?')}, "
                            f"iter={tinfo.get('best_iter','?')}"
                        )

            if current_model is None:
                continue

            # 构建当日截面
            rows: list[list[float]] = []
            syms: list[str] = []
            for sym, pdf in feat_panels.items():
                fmask = pdf["trade_date"] <= scan_date
                if fmask.sum() < 30:
                    continue
                last_row = pdf[fmask].iloc[-1]

                if sym in raw_data:
                    rdf = raw_data[sym]
                    rmask = rdf["trade_date"] <= scan_date
                    if rmask.sum() > 20:
                        if rdf[rmask].tail(20)["amount"].mean() < 5000:
                            continue

                sym_info = basic_info.get(sym, {})
                if "ST" in sym_info.get("name", "") or "退" in sym_info.get("name", ""):
                    continue

                feat_vals = []
                for f in feature_cols:
                    v = last_row.get(f, np.nan)
                    feat_vals.append(float(v) if pd.notna(v) else 0.0)
                rows.append(feat_vals)
                syms.append(sym)

            if len(rows) < 50:
                continue

            X = np.array(rows)
            preds = current_model.predict(X)
            pred_series = pd.Series(preds, index=syms)
            top_syms = pred_series.nlargest(top_n).index.tolist()

            for sym in top_syms:
                if sym not in raw_data:
                    continue
                rdf = raw_data[sym]

                entry_rows = rdf[rdf["trade_date"] == entry_date]
                if entry_rows.empty:
                    continue
                entry_price = float(entry_rows.iloc[0]["open"])
                if entry_price <= 0:
                    continue

                exit_rows = rdf[rdf["trade_date"] == exit_date]
                if exit_rows.empty:
                    continue
                exit_price = float(exit_rows.iloc[0]["close"])

                pnl_pct = (exit_price - entry_price) / entry_price * 100

                trades.append({
                    "signal_date": scan_date,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "symbol": sym,
                    "name": basic_info.get(sym, {}).get("name", ""),
                    "industry": basic_info.get(sym, {}).get("industry", ""),
                    "pred_score": round(float(pred_series[sym]), 6),
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(exit_price, 2),
                    "pnl_pct": round(pnl_pct, 2),
                    "hold_days": hold_days,
                })

            if (si + 1) % 10 == 0:
                logger.info(f"  {h}: {si+1}/{len(scan_days)} scan days, "
                            f"{len(trades)} trades, {train_count} retrains")

        logger.info(f"  {h}: completed — {train_count} retrains, {len(trades)} trades")

        metrics = _compute_backtest_metrics(trades, h, hold_days)
        all_results[h] = {"trades": trades, "metrics": metrics}
        _print_backtest_result(h, trades, metrics)

    # ── 保存 ──
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for h, res in all_results.items():
            if res["trades"]:
                df_trades = pd.DataFrame(res["trades"])
                df_trades.to_csv(os.path.join(out_dir, f"lgb_wf_{h}.csv"), index=False)
        summary_rows = []
        for h, res in all_results.items():
            m = res["metrics"].copy()
            m["horizon"] = h
            summary_rows.append(m)
        if summary_rows:
            pd.DataFrame(summary_rows).to_csv(
                os.path.join(out_dir, "lgb_wf_summary.csv"), index=False)
        logger.info(f"Walk-forward results saved to {out_dir}")

    return all_results


# ═════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LightGBM 因子模型")
    p.add_argument("--cache_dir", type=str, required=True, help="特征缓存根目录")
    p.add_argument("--model_dir", type=str, default="", help="模型目录 (默认 cache_dir/lgb_models/)")
    p.add_argument("--data_dir", type=str, default="", help="日线原始数据目录 (回测用)")
    p.add_argument("--basic_path", type=str, default="", help="股票基本信息 CSV")
    p.add_argument("--out_dir", type=str, default="", help="结果输出目录")
    p.add_argument("--mode", type=str, default="train",
                   choices=["train", "predict", "backtest", "backtest_wf"],
                   help="train=训练, predict=预测, backtest=回测(用预训练模型), "
                        "backtest_wf=Walk-Forward回测(滚动再训练,无前瞻偏差)")
    p.add_argument("--scan_date", type=str, default="", help="预测日期 (predict 模式)")
    p.add_argument("--backtest_start_date", type=str, default="", help="回测起始日期")
    p.add_argument("--backtest_end_date", type=str, default="", help="回测结束日期")
    p.add_argument("--horizons", type=str, default="3d,5d,1w",
                   help="回测的 horizon 列表, 逗号分隔 (默认 3d,5d,1w)")
    p.add_argument("--top_n", type=int, default=5, help="每次选 N 只")
    p.add_argument("--scan_interval", type=int, default=5, help="扫描间隔天数")
    p.add_argument("--retrain_every", type=int, default=1,
                   help="Walk-forward: 每隔N个scan day重新训练 (1=每次, 4=每20天)")
    p.add_argument("--train_end", type=str, default="", help="训练集截止日期 (YYYYMMDD)")
    p.add_argument("--val_days", type=int, default=60)
    p.add_argument("--decay_lambda", type=float, default=0.007)
    p.add_argument("--n_estimators", type=int, default=300)
    p.add_argument("--max_depth", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--max_stocks", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--verbose", action="store_true")
    return p


def main():
    args = _build_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = ModelConfig(
        cache_dir=args.cache_dir,
        model_dir=args.model_dir,
        scan_date=args.scan_date,
        train_end=args.train_end,
        val_days=args.val_days,
        decay_lambda=args.decay_lambda,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        max_stocks=args.max_stocks,
        workers=args.workers,
    )

    if args.mode == "train":
        train_all(cfg)
    elif args.mode == "predict":
        model_dir = cfg.model_dir or os.path.join(cfg.cache_dir, "lgb_models")
        horizon_tops, combined = run_lgb_scoring(
            model_dir=model_dir,
            daily_cache_dir=os.path.join(cfg.cache_dir, "daily"),
            weekly_cache_dir=os.path.join(cfg.cache_dir, "weekly"),
            scan_date=cfg.scan_date,
            cfg=cfg,
        )
        if len(combined) > 0:
            print(f"\n候选总数: {len(combined)} 只 ({combined['symbol'].nunique()} 去重)")
            for h in ALL_HORIZONS:
                df_h = horizon_tops.get(h, pd.DataFrame())
                if len(df_h) > 0:
                    print(f"\n  ── {h} Top {len(df_h)} ──")
                    print(df_h.to_string(index=False))
    elif args.mode == "backtest":
        model_dir = cfg.model_dir or os.path.join(cfg.cache_dir, "lgb_models")
        data_dir = args.data_dir
        if not data_dir:
            logger.error("--data_dir required for backtest mode")
            return
        horizons = [h.strip() for h in args.horizons.split(",")]
        run_backtest(
            cache_dir=cfg.cache_dir,
            data_dir=data_dir,
            model_dir=model_dir,
            start_date=args.backtest_start_date,
            end_date=args.backtest_end_date,
            horizons=horizons,
            top_n=args.top_n,
            scan_interval=args.scan_interval,
            basic_path=args.basic_path,
            out_dir=args.out_dir,
        )
    elif args.mode == "backtest_wf":
        data_dir = args.data_dir
        if not data_dir:
            logger.error("--data_dir required for backtest_wf mode")
            return
        horizons = [h.strip() for h in args.horizons.split(",")]
        run_backtest_wf(
            cache_dir=cfg.cache_dir,
            data_dir=data_dir,
            start_date=args.backtest_start_date,
            end_date=args.backtest_end_date,
            horizons=horizons,
            top_n=args.top_n,
            scan_interval=args.scan_interval,
            retrain_every=args.retrain_every,
            basic_path=args.basic_path,
            out_dir=args.out_dir,
            cfg=cfg,
        )


if __name__ == "__main__":
    main()
