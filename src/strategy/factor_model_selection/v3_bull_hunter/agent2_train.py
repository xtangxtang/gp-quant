"""
Bull Hunter v3 — Agent 2: 大牛股分类器训练

职责: 回看历史, 找到涨幅 ≥30%/100%/200% 的股票, 分析其因子画像,
      训练三个独立的 LightGBM 二分类模型, 存储到 models/ 目录。

目标窗口:
  30%  → 2 周 (10 个交易日)
  100% → 2 个月 (40 个交易日)
  200% → 6 个月 (120 个交易日)

训练数据:
  回看 1 年的滚动样本 — 每个月初取因子快照, 计算对应窗口后的实际涨幅,
  标记 label=1 (达到目标) / label=0 (未达到)。

输出:
  feature-cache/bull_models/{scan_date}/
    model_30pct.pkl
    model_100pct.pkl
    model_200pct.pkl
    meta.json  (训练样本数, 正样本比, 特征重要性 Top 20)
"""

from __future__ import annotations

import glob
import json
import logging
import os
import pickle
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── 目标配置 ──
TARGETS = {
    "30pct": {"threshold": 0.30, "forward_days": 10, "label": "2周涨30%"},
    "100pct": {"threshold": 1.00, "forward_days": 40, "label": "2月涨100%"},
    "200pct": {"threshold": 2.00, "forward_days": 120, "label": "6月涨200%"},
}

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


@dataclass
class TrainConfig:
    """训练配置。"""
    lookback_months: int = 12       # 回看多少个月构造训练样本
    sample_interval_days: int = 5   # 采样间隔 (每周一次, 增加样本密度)
    val_ratio: float = 0.2          # 验证集比例 (时间末尾)
    max_scale_pos_weight: float = 5.0   # 上限, 避免过高导致loss震荡 (20太大会在第1棵树就过拟合)
    # LightGBM
    n_estimators: int = 800
    max_depth: int = 5
    num_leaves: int = 31
    learning_rate: float = 0.03
    subsample: float = 0.7
    colsample_bytree: float = 0.7
    min_child_samples: int = 50
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    workers: int = 8


def run_training(
    cache_dir: str,
    scan_date: str,
    cfg: TrainConfig | None = None,
) -> dict[str, dict]:
    """
    训练三个大牛股分类器。

    Args:
        cache_dir: 特征缓存根目录 (含 daily/ 子目录)
        scan_date: 训练日期 (YYYYMMDD)
        cfg: 训练配置

    Returns:
        {target_name: {"model_path": str, "meta": dict}}
    """
    cfg = cfg or TrainConfig()

    # 检查是否已有模型
    model_dir = os.path.join(cache_dir, "bull_models", scan_date)
    meta_path = os.path.join(model_dir, "meta.json")
    if os.path.exists(meta_path):
        logger.info(f"Agent 2: 模型已存在 {model_dir}, 跳过训练")
        with open(meta_path) as f:
            existing_meta = json.load(f)
        results = {}
        for tname in TARGETS:
            mp = os.path.join(model_dir, f"model_{tname}.pkl")
            if os.path.exists(mp):
                results[tname] = {
                    "model_path": mp,
                    "meta": existing_meta.get(tname, {}),
                }
        return results

    # ── 构建训练日历 ──
    calendar = _build_calendar(cache_dir)
    if scan_date not in calendar:
        earlier = [d for d in calendar if d <= scan_date]
        if not earlier:
            logger.error(f"scan_date {scan_date} 不在日历中")
            return {}
        scan_date = earlier[-1]

    scan_idx = calendar.index(scan_date)

    # ── 确定采样日期 ──
    # 回看 lookback_months 个月, 每 sample_interval_days 采样一次
    lookback_days = cfg.lookback_months * 20  # 粗略换算
    max_forward = max(t["forward_days"] for t in TARGETS.values())

    # 采样起点: 至少需要 max_forward 天的前瞻空间
    sample_end_idx = scan_idx - max_forward
    sample_start_idx = max(0, sample_end_idx - lookback_days)

    if sample_end_idx <= sample_start_idx:
        logger.error(f"数据不足: scan_idx={scan_idx}, 需要 {lookback_days + max_forward} 天历史")
        return {}

    sample_dates = []
    for i in range(sample_start_idx, sample_end_idx, cfg.sample_interval_days):
        sample_dates.append(calendar[i])

    logger.info(f"训练采样: {len(sample_dates)} 个日期, "
                f"{calendar[sample_start_idx]} ~ {calendar[sample_end_idx - 1]}")

    # ── 构建训练 panel ──
    panel = _build_training_panel(cache_dir, calendar, sample_dates, scan_idx)
    if panel.empty:
        logger.error("训练 panel 为空")
        return {}

    logger.info(f"训练 panel: {len(panel)} 行, {panel['symbol'].nunique()} 只股票")

    # ── 训练每个目标 ──
    os.makedirs(model_dir, exist_ok=True)
    all_meta = {}
    results = {}

    for tname, tspec in TARGETS.items():
        threshold = tspec["threshold"]
        fwd_days = tspec["forward_days"]

        label_col = f"gain_{fwd_days}d"
        if label_col not in panel.columns:
            logger.warning(f"  {tname}: 缺少 {label_col} 列, 跳过")
            continue

        # 构造二分类标签
        labels = (panel[label_col] >= threshold).astype(int)
        n_pos = labels.sum()
        n_total = len(labels)
        pos_rate = n_pos / n_total if n_total > 0 else 0

        if n_pos < 10:
            logger.warning(f"  {tname}: 正样本仅 {n_pos} 个, 跳过")
            continue

        logger.info(f"  {tname}: {n_total} 样本, {n_pos} 正样本 ({pos_rate:.1%})")

        # 训练
        model, meta = _train_one_target(
            panel, labels, tname, threshold, fwd_days, cfg
        )

        if model is not None:
            model_path = os.path.join(model_dir, f"model_{tname}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            all_meta[tname] = meta
            results[tname] = {"model_path": model_path, "meta": meta}
            logger.info(f"  {tname}: 训练完成, val_auc={meta.get('val_auc', 0):.4f}, "
                        f"saved → {model_path}")

    # 保存 meta
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=2)

    logger.info(f"Agent 2 完成: {len(results)} 个模型, 存储于 {model_dir}")
    return results


def _build_training_panel(
    cache_dir: str,
    calendar: list[str],
    sample_dates: list[str],
    scan_idx: int,
) -> pd.DataFrame:
    """
    构建训练 panel: 对每个采样日期, 取全市场因子快照 + 计算前瞻涨幅。

    Returns:
        DataFrame: columns = [symbol, sample_date] + factors + [gain_10d, gain_40d, gain_120d]
    """
    daily_dir = os.path.join(cache_dir, "daily")
    csv_files = sorted(glob.glob(os.path.join(daily_dir, "*.csv")))

    # 预计算日期→日历索引映射 (O(1) 查找)
    date_to_idx = {d: i for i, d in enumerate(calendar)}
    sample_set = set(sample_dates)

    # 预计算每个 forward_days 的目标日期映射
    fwd_target_dates = {}
    for fwd_days in [10, 40, 120]:
        mapping = {}
        for sd in sample_dates:
            sd_idx = date_to_idx.get(sd)
            if sd_idx is None:
                continue
            target_idx = sd_idx + fwd_days
            if target_idx < len(calendar):
                mapping[sd] = calendar[target_idx]
        fwd_target_dates[fwd_days] = mapping

    # 所有需要的日期 (采样日 + 目标日)
    needed_dates = set(sample_dates)
    for mapping in fwd_target_dates.values():
        needed_dates.update(mapping.values())

    # 预加载所有股票 (一次性读取, 按 trade_date 索引)
    logger.info(f"加载 {len(csv_files)} 只股票的因子时序...")
    all_panels = []
    n_loaded = 0
    cols_needed = ["trade_date", "close"] + DAILY_FACTORS

    for fp in csv_files:
        sym = os.path.basename(fp).replace(".csv", "")
        if sym.startswith("bj"):
            continue
        try:
            df = pd.read_csv(fp)
            if len(df) < 60:
                continue
            df["trade_date"] = df["trade_date"].astype(str)
            # 只保留需要的日期行 (大幅减少内存)
            df = df[df["trade_date"].isin(needed_dates)]
            if df.empty:
                continue
            # 只保留需要的列
            avail_cols = [c for c in cols_needed if c in df.columns]
            df = df[avail_cols].copy()
            df["symbol"] = sym
            df = df.set_index("trade_date")
            all_panels.append(df)
            n_loaded += 1
        except Exception:
            continue

    logger.info(f"加载完成: {n_loaded} 只股票")

    if not all_panels:
        return pd.DataFrame()

    # 合并为大宽表 (trade_date 为 index, symbol 列)
    big = pd.concat(all_panels, axis=0)
    big = big.reset_index()

    # 提取采样日快照
    snapshots = big[big["trade_date"].isin(sample_set)].copy()
    snapshots = snapshots.rename(columns={"trade_date": "sample_date"})

    # 计算前瞻涨幅 — 向量化 merge
    for fwd_days in [10, 40, 120]:
        mapping = fwd_target_dates[fwd_days]
        snapshots[f"_target_date_{fwd_days}"] = snapshots["sample_date"].map(mapping)

        target_closes = big[["trade_date", "symbol", "close"]].rename(
            columns={"trade_date": f"_target_date_{fwd_days}", "close": f"_target_close_{fwd_days}"}
        )
        snapshots = snapshots.merge(
            target_closes,
            on=["symbol", f"_target_date_{fwd_days}"],
            how="left",
        )

        sd_close = snapshots["close"]
        tgt_close = snapshots[f"_target_close_{fwd_days}"]
        valid = (sd_close > 0) & sd_close.notna() & tgt_close.notna() & (tgt_close > 0)
        snapshots[f"gain_{fwd_days}d"] = np.where(
            valid, (tgt_close - sd_close) / sd_close, np.nan
        )
        snapshots.drop(columns=[f"_target_date_{fwd_days}", f"_target_close_{fwd_days}"], inplace=True)

    # 过滤无效行 (close <= 0)
    snapshots = snapshots[snapshots["close"].gt(0) & snapshots["close"].notna()]

    # 整理列
    factor_cols = [f for f in DAILY_FACTORS if f in snapshots.columns]
    keep_cols = ["symbol", "sample_date"] + factor_cols + ["gain_10d", "gain_40d", "gain_120d"]
    keep_cols = [c for c in keep_cols if c in snapshots.columns]
    result = snapshots[keep_cols].reset_index(drop=True)

    logger.info(f"训练 panel 构建完成: {len(result)} 行")
    return result


def _train_one_target(
    panel: pd.DataFrame,
    labels: pd.Series,
    target_name: str,
    threshold: float,
    fwd_days: int,
    cfg: TrainConfig,
) -> tuple:
    """训练单个目标的 LightGBM 二分类模型。"""
    # 特征列
    feature_cols = [f for f in DAILY_FACTORS if f in panel.columns
                    and panel[f].notna().sum() > 100]
    if not feature_cols:
        return None, {}

    # 去除 label 为 NaN 的行
    gain_col = f"gain_{fwd_days}d"
    valid_mask = panel[gain_col].notna()
    X = panel.loc[valid_mask, feature_cols].fillna(0).values
    y = labels[valid_mask].values

    if len(X) < 200:
        return None, {}

    # 时间分割: 最后 val_ratio 的样本做验证
    n_val = max(int(len(X) * cfg.val_ratio), 50)
    X_train, X_val = X[:-n_val], X[-n_val:]
    y_train, y_val = y[:-n_val], y[-n_val:]

    n_pos_train = int(y_train.sum())
    n_neg_train = len(y_train) - n_pos_train
    if n_pos_train < 5:
        return None, {}

    raw_ratio = n_neg_train / max(n_pos_train, 1)
    scale_pos = min(raw_ratio, cfg.max_scale_pos_weight)
    logger.info(f"    scale_pos_weight: {scale_pos:.1f} (raw={raw_ratio:.1f}, cap={cfg.max_scale_pos_weight})")

    model = lgb.LGBMClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        num_leaves=cfg.num_leaves,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        min_child_samples=cfg.min_child_samples,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        scale_pos_weight=scale_pos,
        random_state=42,
        n_jobs=cfg.workers,
        verbose=-1,
    )

    # 不用 early stopping: 极端类别不平衡下, validation logloss
    # 从第1棵树就单调递增, early stopping 会在 iteration=1 就停止.
    # 依赖 regularization (reg_alpha/lambda, subsample, colsample) 防止过拟合.
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[
            lgb.log_evaluation(period=200),
        ],
    )

    # 评估 — 用最优 F1 阈值而非固定 0.5
    from sklearn.metrics import roc_auc_score, precision_score, recall_score
    val_proba = model.predict_proba(X_val)[:, 1]

    try:
        val_auc = roc_auc_score(y_val, val_proba)
    except ValueError:
        val_auc = 0.5

    # 搜索最优阈值 (正样本稀少, 固定 0.5 不合理)
    best_f1, best_th = 0, 0.5
    for th in np.arange(0.01, 0.50, 0.01):
        pred_th = (val_proba > th).astype(int)
        tp = ((pred_th == 1) & (y_val == 1)).sum()
        fp = ((pred_th == 1) & (y_val == 0)).sum()
        fn = ((pred_th == 0) & (y_val == 1)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best_f1:
            best_f1, best_th = f1, th

    val_pred = (val_proba > best_th).astype(int)
    val_precision = precision_score(y_val, val_pred, zero_division=0)
    val_recall = recall_score(y_val, val_pred, zero_division=0)

    # 特征重要性
    importance = dict(zip(feature_cols, model.feature_importances_.tolist()))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]

    meta = {
        "target": target_name,
        "threshold": threshold,
        "forward_days": fwd_days,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_pos_train": n_pos_train,
        "pos_rate_train": round(n_pos_train / len(y_train), 4),
        "n_pos_val": int(y_val.sum()),
        "pos_rate_val": round(int(y_val.sum()) / len(y_val), 4),
        "scale_pos_weight": round(scale_pos, 2),
        "val_auc": round(val_auc, 4),
        "val_precision": round(val_precision, 4),
        "val_recall": round(val_recall, 4),
        "best_threshold": round(best_th, 3),
        "best_f1": round(best_f1, 4),
        "best_iteration": model.best_iteration_,
        "feature_cols": feature_cols,
        "top_features": [{"name": n, "importance": v} for n, v in top_features],
    }

    return model, meta


def _build_calendar(cache_dir: str) -> list[str]:
    """从 daily cache 构建交易日历。"""
    daily_dir = os.path.join(cache_dir, "daily")
    csvs = sorted(glob.glob(os.path.join(daily_dir, "*.csv")))[:50]
    all_dates: set[str] = set()
    for fp in csvs:
        try:
            df = pd.read_csv(fp, usecols=["trade_date"])
            all_dates.update(df["trade_date"].astype(str).tolist())
        except Exception:
            continue
    return sorted(all_dates)
