"""
Bull Hunter v3 — Pipeline 编排器

Agent 1 (因子快照) → Agent 2 (训练分类器) → Agent 3 (预测) → Agent 4 (监控)

支持两种模式:
  1. scan  — 对 scan_date 训练+预测, 输出候选
  2. rolling — 滚动回测, 每月初训练+预测, 评估历史表现
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import pandas as pd

from .agent1_factor import run_factor_generation
from .agent2_train import TrainConfig, run_training
from .agent3_predict import PredictConfig, run_prediction
from .agent4_monitor import run_monitor

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Pipeline 全局配置。"""
    cache_dir: str = "/nvme5/xtang/gp-workspace/gp-data/feature-cache"
    data_dir: str = "/nvme5/xtang/gp-workspace/gp-data/tushare-daily-full"
    basic_path: str = "/nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv"
    results_dir: str = "results/bull_hunter"
    train_cfg: TrainConfig | None = None
    predict_cfg: PredictConfig | None = None


def run_scan(
    scan_date: str,
    cfg: PipelineConfig | None = None,
) -> pd.DataFrame:
    """
    单日扫描: 训练 + 预测。

    Returns:
        预测结果 DataFrame
    """
    cfg = cfg or PipelineConfig()
    train_cfg = cfg.train_cfg or TrainConfig()
    predict_cfg = cfg.predict_cfg or PredictConfig()

    logger.info(f"=== Bull Hunter v3 SCAN: {scan_date} ===")

    # ── Agent 1: 因子快照 ──
    logger.info("── Agent 1: 因子快照 ──")
    snapshot = run_factor_generation(
        cache_dir=cfg.cache_dir,
        data_dir=cfg.data_dir,
        scan_date=scan_date,
        basic_path=cfg.basic_path,
    )
    if snapshot.empty:
        logger.error("Agent 1 返回空快照, 终止")
        return pd.DataFrame()

    logger.info(f"  → {len(snapshot)} 只股票")

    # ── Agent 2: 训练 ──
    logger.info("── Agent 2: 训练分类器 ──")
    train_results = run_training(
        cache_dir=cfg.cache_dir,
        scan_date=scan_date,
        cfg=train_cfg,
    )
    if not train_results:
        logger.error("Agent 2 训练失败, 终止")
        return pd.DataFrame()

    model_dir = os.path.join(cfg.cache_dir, "bull_models", scan_date)
    logger.info(f"  → {len(train_results)} 个模型")

    # ── Agent 3: 预测 ──
    logger.info("── Agent 3: 预测 ──")
    predictions = run_prediction(
        factor_snapshot=snapshot,
        model_dir=model_dir,
        scan_date=scan_date,
        cfg=predict_cfg,
    )

    if predictions.empty:
        logger.warning("Agent 3 无候选")
    else:
        # 保存结果
        out_dir = os.path.join(cfg.results_dir, scan_date)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "predictions.csv")
        predictions.to_csv(out_path, index=False, encoding="utf-8-sig")
        logger.info(f"  → 已保存 {len(predictions)} 只候选 → {out_path}")

    # ── Agent 4: 监控 ──
    logger.info("── Agent 4: 监控 ──")
    health = run_monitor(
        cache_dir=cfg.cache_dir,
        data_dir=cfg.data_dir,
        scan_date=scan_date,
    )

    if health.get("suggestions"):
        for s in health["suggestions"]:
            logger.info(f"  💡 {s}")

    # 保存健康报告
    out_dir = os.path.join(cfg.results_dir, scan_date)
    os.makedirs(out_dir, exist_ok=True)
    health_path = os.path.join(out_dir, "health_report.json")
    with open(health_path, "w", encoding="utf-8") as f:
        json.dump(health, f, ensure_ascii=False, indent=2)

    logger.info(f"=== 完成: status={health.get('status', 'unknown')} ===")
    return predictions


def run_rolling(
    start_date: str,
    end_date: str,
    cfg: PipelineConfig | None = None,
    interval_days: int = 20,
) -> pd.DataFrame:
    """
    滚动回测: 每 interval_days 天训练+预测, 汇总结果。

    Returns:
        所有预测日的合并结果
    """
    cfg = cfg or PipelineConfig()

    # 构建日历
    from .agent2_train import _build_calendar
    calendar = _build_calendar(cfg.cache_dir)

    valid_dates = [d for d in calendar if start_date <= d <= end_date]
    if not valid_dates:
        logger.error(f"日期范围 {start_date}~{end_date} 内无交易日")
        return pd.DataFrame()

    # 每 interval_days 采样一次
    scan_dates = valid_dates[::interval_days]
    logger.info(f"滚动模式: {len(scan_dates)} 次扫描, "
                f"{scan_dates[0]} ~ {scan_dates[-1]}")

    all_predictions = []
    for sd in scan_dates:
        try:
            preds = run_scan(sd, cfg)
            if not preds.empty:
                preds["scan_date"] = sd
                all_predictions.append(preds)
        except Exception as e:
            logger.error(f"  {sd} 失败: {e}")
            continue

    if not all_predictions:
        return pd.DataFrame()

    merged = pd.concat(all_predictions, ignore_index=True)

    # 保存合并结果
    out_path = os.path.join(cfg.results_dir, f"rolling_{start_date}_{end_date}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"滚动结果: {len(merged)} 行 → {out_path}")

    return merged


def run_backtest(
    start_date: str,
    end_date: str,
    cfg: PipelineConfig | None = None,
    interval_days: int = 20,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    滚动回测 + 实际收益验证。

    每 interval_days 天:
      1. 训练模型 + 预测候选
      2. 取 A/B/C 各等级 top_n 只
      3. 查看实际 10d/40d/120d 涨幅
      4. 汇总统计

    Returns:
        带实际收益的预测结果 DataFrame
    """
    import numpy as np
    cfg = cfg or PipelineConfig()

    # 构建日历
    from .agent2_train import _build_calendar
    calendar = _build_calendar(cfg.cache_dir)

    valid_dates = [d for d in calendar if start_date <= d <= end_date]
    if not valid_dates:
        logger.error(f"日期范围 {start_date}~{end_date} 内无交易日")
        return pd.DataFrame()

    scan_dates = valid_dates[::interval_days]
    logger.info(f"回测模式: {len(scan_dates)} 次扫描, "
                f"{scan_dates[0]} ~ {scan_dates[-1]}")

    # ── 1. 滚动扫描 ──
    all_predictions = []
    for i, sd in enumerate(scan_dates):
        logger.info(f"[{i+1}/{len(scan_dates)}] 扫描 {sd} ...")
        try:
            preds = run_scan(sd, cfg)
            if not preds.empty:
                preds["scan_date"] = sd
                all_predictions.append(preds.head(top_n * 3))  # A+B+C 各 top_n
        except Exception as e:
            logger.error(f"  {sd} 失败: {e}")
            continue

    if not all_predictions:
        logger.error("回测无任何预测结果")
        return pd.DataFrame()

    merged = pd.concat(all_predictions, ignore_index=True)
    logger.info(f"共 {len(merged)} 条预测, {merged['scan_date'].nunique()} 个扫描日")

    # ── 2. 计算实际收益 ──
    date_to_idx = {d: i for i, d in enumerate(calendar)}

    # 预加载需要的股票价格
    symbols = merged["symbol"].unique().tolist()
    logger.info(f"加载 {len(symbols)} 只股票的价格数据...")
    price_data = {}
    for sym in symbols:
        fpath = os.path.join(cfg.data_dir, f"{sym}.csv")
        if not os.path.exists(fpath):
            continue
        try:
            df = pd.read_csv(fpath, usecols=["trade_date", "open", "close"])
            df["trade_date"] = df["trade_date"].astype(str)
            df = df.set_index("trade_date")
            price_data[sym] = df
        except Exception:
            continue

    # 对每条预测计算实际涨幅
    actual_returns = {f"actual_{d}d": [] for d in [10, 40, 120]}
    actual_returns["actual_max_40d"] = []

    for _, row in merged.iterrows():
        sym = row["symbol"]
        sd = row["scan_date"]
        sd_idx = date_to_idx.get(sd)

        if sym not in price_data or sd_idx is None:
            for k in actual_returns:
                actual_returns[k].append(np.nan)
            continue

        pdf = price_data[sym]

        # 次日开盘买入
        buy_date_idx = sd_idx + 1
        if buy_date_idx >= len(calendar):
            for k in actual_returns:
                actual_returns[k].append(np.nan)
            continue

        buy_date = calendar[buy_date_idx]
        if buy_date not in pdf.index:
            for k in actual_returns:
                actual_returns[k].append(np.nan)
            continue

        buy_price = float(pdf.loc[buy_date, "open"])
        if buy_price <= 0 or pd.isna(buy_price):
            for k in actual_returns:
                actual_returns[k].append(np.nan)
            continue

        # N 天后收盘价
        for fwd_days in [10, 40, 120]:
            target_idx = buy_date_idx + fwd_days
            if target_idx < len(calendar):
                target_date = calendar[target_idx]
                if target_date in pdf.index:
                    sell_price = float(pdf.loc[target_date, "close"])
                    actual_returns[f"actual_{fwd_days}d"].append(
                        (sell_price - buy_price) / buy_price
                    )
                else:
                    actual_returns[f"actual_{fwd_days}d"].append(np.nan)
            else:
                actual_returns[f"actual_{fwd_days}d"].append(np.nan)

        # 40 天内最大涨幅
        max_gain = np.nan
        for j in range(1, 41):
            t_idx = buy_date_idx + j
            if t_idx >= len(calendar):
                break
            t_date = calendar[t_idx]
            if t_date in pdf.index:
                c = float(pdf.loc[t_date, "close"])
                g = (c - buy_price) / buy_price
                if pd.isna(max_gain) or g > max_gain:
                    max_gain = g
        actual_returns["actual_max_40d"].append(max_gain)

    for k, v in actual_returns.items():
        merged[k] = v

    # ── 3. 统计汇总 ──
    out_dir = os.path.join(cfg.results_dir, f"backtest_{start_date}_{end_date}")
    os.makedirs(out_dir, exist_ok=True)

    # 保存明细
    detail_path = os.path.join(out_dir, "backtest_detail.csv")
    merged.to_csv(detail_path, index=False, encoding="utf-8-sig")

    # 按等级统计
    summary_rows = []
    for grade in ["A", "B", "C", "ALL"]:
        subset = merged if grade == "ALL" else merged[merged["grade"] == grade]
        if subset.empty:
            continue

        n = len(subset)
        row = {"grade": grade, "n_predictions": n}

        for fwd_days, target_gain in [(10, 0.30), (40, 1.00), (120, 2.00)]:
            col = f"actual_{fwd_days}d"
            valid = subset[col].dropna()
            if len(valid) > 0:
                row[f"avg_{fwd_days}d"] = round(float(valid.mean()), 4)
                row[f"median_{fwd_days}d"] = round(float(valid.median()), 4)
                row[f"hit_{fwd_days}d"] = round(float((valid >= target_gain).mean()), 4)
                row[f"win_{fwd_days}d"] = round(float((valid > 0).mean()), 4)
                row[f"n_valid_{fwd_days}d"] = len(valid)
            else:
                row[f"avg_{fwd_days}d"] = np.nan

        # 40d 最大涨幅
        max40 = subset["actual_max_40d"].dropna()
        if len(max40) > 0:
            row["avg_max_40d"] = round(float(max40.mean()), 4)
            row["hit_30pct_max_40d"] = round(float((max40 >= 0.30).mean()), 4)

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, "backtest_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # 按扫描日统计
    date_stats = []
    for sd in sorted(merged["scan_date"].unique()):
        sub = merged[merged["scan_date"] == sd]
        row = {"scan_date": sd, "n": len(sub)}
        for fwd_days in [10, 40, 120]:
            col = f"actual_{fwd_days}d"
            valid = sub[col].dropna()
            if len(valid) > 0:
                row[f"avg_{fwd_days}d"] = round(float(valid.mean()), 4)
                row[f"win_{fwd_days}d"] = round(float((valid > 0).mean()), 4)
        date_stats.append(row)

    date_df = pd.DataFrame(date_stats)
    date_path = os.path.join(out_dir, "backtest_by_date.csv")
    date_df.to_csv(date_path, index=False, encoding="utf-8-sig")

    # 打印摘要
    logger.info(f"\n{'='*60}")
    logger.info(f"回测结果: {start_date} ~ {end_date}")
    logger.info(f"{'='*60}")
    for _, r in summary.iterrows():
        g = r["grade"]
        n = r["n_predictions"]
        parts = [f"等级={g}, 预测数={n}"]
        for fwd_days, label in [(10, "10d"), (40, "40d"), (120, "120d")]:
            avg = r.get(f"avg_{fwd_days}d")
            win = r.get(f"win_{fwd_days}d")
            hit = r.get(f"hit_{fwd_days}d")
            if pd.notna(avg):
                parts.append(f"{label}: 均值={avg:.1%} 胜率={win:.0%}")
        logger.info("  " + " | ".join(parts))

    logger.info(f"\n明细: {detail_path}")
    logger.info(f"汇总: {summary_path}")
    logger.info(f"按日: {date_path}")

    return merged
