"""
Bull Hunter v3 — Agent 4: 总监控 (Monitor / Supervisor)

职责:
  1. 追踪历史预测 vs 实际涨幅, 评估模型质量
  2. 月度健康检查: AUC, 精确率, 召回率, FP率
  3. 预定义 SOP: 自动诊断问题并建议调参

SOP 规则:
  - 精确率下降 → 提高预测阈值
  - 召回率下降 → 降低阈值 / 增加训练样本
  - AUC < 0.55  → 缩短训练窗口 / 重新选特征
  - 正样本过多  → 可能市场普涨, 减少仓位
  - 正样本为零  → 市场恶劣, 暂停买入

输出:
  health_report.json (最近一轮模型的健康状态)
  tuning_suggestions (list[str])
"""

from __future__ import annotations

import glob
import json
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from .agent2_train import TARGETS


# ── 健康阈值 ──
HEALTH_THRESHOLDS = {
    "30pct": {"min_auc": 0.58, "min_precision": 0.10, "min_recall": 0.15},
    "100pct": {"min_auc": 0.55, "min_precision": 0.05, "min_recall": 0.10},
    "200pct": {"min_auc": 0.55, "min_precision": 0.03, "min_recall": 0.05},
}


def run_monitor(
    cache_dir: str,
    data_dir: str,
    scan_date: str,
    predictions_history: list[dict] | None = None,
) -> dict:
    """
    总监控: 评估模型质量, 生成健康报告 + 调参建议。

    Args:
        cache_dir: 特征缓存根目录
        data_dir: 日线数据目录 (用于计算实际涨幅)
        scan_date: 当前日期
        predictions_history: 历史预测记录 [{scan_date, symbol, prob_30, prob_100, prob_200, grade}]

    Returns:
        {"health": {...}, "suggestions": [...], "status": "healthy|warning|critical"}
    """
    report = {
        "scan_date": scan_date,
        "health": {},
        "suggestions": [],
        "status": "healthy",
    }

    # ── 1. 检查最近模型的训练指标 ──
    model_dir = os.path.join(cache_dir, "bull_models", scan_date)
    meta_path = os.path.join(model_dir, "meta.json")

    if not os.path.exists(meta_path):
        # 尝试找最近的模型
        model_root = os.path.join(cache_dir, "bull_models")
        if os.path.exists(model_root):
            all_dates = sorted(os.listdir(model_root))
            recent = [d for d in all_dates if d <= scan_date]
            if recent:
                model_dir = os.path.join(model_root, recent[-1])
                meta_path = os.path.join(model_dir, "meta.json")

    training_meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            training_meta = json.load(f)

    for tname, thresholds in HEALTH_THRESHOLDS.items():
        tmeta = training_meta.get(tname, {})
        if not tmeta:
            report["health"][tname] = {"status": "no_model"}
            report["suggestions"].append(f"{tname}: 无训练模型, 需要先训练")
            continue

        val_auc = tmeta.get("val_auc", 0)
        val_precision = tmeta.get("val_precision", 0)
        val_recall = tmeta.get("val_recall", 0)
        pos_rate = tmeta.get("pos_rate_train", 0)
        n_train = tmeta.get("n_train", 0)

        health = {
            "val_auc": val_auc,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "pos_rate": pos_rate,
            "n_train": n_train,
            "issues": [],
        }

        # ── 诊断 ──
        if val_auc < thresholds["min_auc"]:
            health["issues"].append("auc_low")
            report["suggestions"].append(
                f"{tname}: AUC={val_auc:.3f} < {thresholds['min_auc']}, "
                f"建议缩短训练窗口或重新选特征"
            )

        if val_precision < thresholds["min_precision"]:
            health["issues"].append("precision_low")
            report["suggestions"].append(
                f"{tname}: Precision={val_precision:.3f}, "
                f"建议提高预测阈值 (当前模型假阳性过多)"
            )

        if val_recall < thresholds["min_recall"]:
            health["issues"].append("recall_low")
            report["suggestions"].append(
                f"{tname}: Recall={val_recall:.3f}, "
                f"建议降低预测阈值或增加训练样本"
            )

        if n_train < 500:
            health["issues"].append("sample_small")
            report["suggestions"].append(
                f"{tname}: 训练样本仅 {n_train}, 建议扩大回看窗口"
            )

        # 正样本比例异常
        if pos_rate > 0.3 and tname == "30pct":
            health["issues"].append("pos_rate_high")
            report["suggestions"].append(
                f"{tname}: 正样本比 {pos_rate:.1%} 偏高, "
                f"可能处于普涨市, 建议降低仓位"
            )
        elif pos_rate < 0.005:
            health["issues"].append("pos_rate_zero")
            report["suggestions"].append(
                f"{tname}: 正样本几乎为零, 市场极弱, 建议暂停买入"
            )

        health["status"] = "healthy" if not health["issues"] else "warning"
        report["health"][tname] = health

    # ── 2. 回测验证 (如果有历史预测) ──
    if predictions_history:
        backtest = _verify_past_predictions(predictions_history, data_dir)
        report["backtest_verification"] = backtest

        if backtest.get("hit_rate_30", 0) < 0.1:
            report["suggestions"].append(
                "过去预测的 30% 命中率过低, 模型可能失效"
            )
            report["status"] = "critical"

    # ── 3. 综合状态 ──
    n_issues = sum(
        len(h.get("issues", []))
        for h in report["health"].values()
        if isinstance(h, dict)
    )
    if n_issues >= 3:
        report["status"] = "critical"
    elif n_issues >= 1:
        report["status"] = "warning"

    logger.info(f"Agent 4 完成: status={report['status']}, "
                f"{len(report['suggestions'])} 条建议")

    return report


def _verify_past_predictions(
    predictions: list[dict],
    data_dir: str,
) -> dict:
    """回溯验证过去的预测结果。"""
    if not predictions or not data_dir:
        return {}

    # 按预测日期分组
    from collections import defaultdict
    by_date = defaultdict(list)
    for p in predictions:
        by_date[p["scan_date"]].append(p)

    hits_30 = 0
    hits_100 = 0
    total = 0

    for sd, preds in sorted(by_date.items()):
        for p in preds:
            sym = p["symbol"]
            fpath = os.path.join(data_dir, f"{sym}.csv")
            if not os.path.exists(fpath):
                continue

            try:
                df = pd.read_csv(fpath)
                df["trade_date"] = df["trade_date"].astype(str)
                df = df.sort_values("trade_date").reset_index(drop=True)
            except Exception:
                continue

            sd_rows = df[df["trade_date"] == sd]
            if sd_rows.empty:
                continue

            entry_close = float(sd_rows.iloc[0]["close"])
            if entry_close <= 0:
                continue

            # 计算实际涨幅 (取后续最高 close)
            future = df[df["trade_date"] > sd].head(40)
            if future.empty:
                continue

            max_close = float(future["close"].max())
            actual_gain = (max_close - entry_close) / entry_close

            total += 1
            if actual_gain >= 0.30:
                hits_30 += 1
            if actual_gain >= 1.00:
                hits_100 += 1

    return {
        "total_predictions": total,
        "hit_rate_30": round(hits_30 / max(total, 1), 4),
        "hit_rate_100": round(hits_100 / max(total, 1), 4),
        "n_hit_30": hits_30,
        "n_hit_100": hits_100,
    }
