"""
Bull Hunter v3 — Agent 4: 双回路反馈监控

v4 重构: 聚焦 A 类大牛股, 双回路闭环优化。

回路 A — 漏网之鱼 (Missed Bull Analysis):
  每次 Agent 2 重训后, 回看上一周期:
  1. 扫描全市场找出实际 120d max_gain >= 100% 的大牛股
  2. 对比 Agent 3 在那段时间的预测: 是否出现在 Top 5?
  3. 对漏选股做因子 profiling, 生成调参建议给 Agent 2

回路 B — Top 5 跟踪 (Prediction Tracking):
  基于 tracker.py 的到期评估:
  1. 分析成功/失败的因子模式
  2. rank=1 vs rank=5 的表现差异
  3. 行业/市值分布
  4. 反馈给 Agent 2 (因子调整) 和 Agent 3 (阈值调整)

输出:
  tuning_directives (dict) — Agent 2/3 可消费的结构化调参指令
  health_report.json — 模型健康状态

安全机制:
  - 两次重训间至少间隔 2 个交易日 (MIN_RETRAIN_INTERVAL)
  - 单次最多改 3 个参数 (MAX_ACTIONS_PER_DIRECTIVE)
"""

from __future__ import annotations

import glob
import json
import logging
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from .agent2_train import DAILY_FACTORS, TARGETS


# ── 健康阈值 ──
HEALTH_THRESHOLDS = {
    "200pct": {"min_auc": 0.55, "min_precision": 0.03, "min_recall": 0.05},
    "100pct": {"min_auc": 0.55, "min_precision": 0.05, "min_recall": 0.10},
}

# 安全限制
MIN_RETRAIN_INTERVAL = 2    # 两次重训最小间隔 (交易日)
MAX_ACTIONS_PER_DIRECTIVE = 3  # 单次最多调参数

# Miss 诊断参数
_MISS_LOOKBACK_DAYS = 120
_MISS_TOP_N = 30
_MISS_THRESHOLD = 1.0


def run_monitor(
    cache_dir: str,
    data_dir: str,
    scan_date: str,
    predictions_history: list[dict] | None = None,
    factor_snapshot: pd.DataFrame | None = None,
    current_predictions: pd.DataFrame | None = None,
    basic_path: str = "",
    use_llm: bool = False,
    tracking_summary: dict | None = None,
    expired_evals: pd.DataFrame | None = None,
) -> dict:
    """
    双回路反馈监控: 评估模型质量 + 跟踪反馈 + 生成调参指令。

    Args:
        cache_dir: 特征缓存根目录
        data_dir: 日线数据目录
        scan_date: 当前日期
        predictions_history: 历史预测记录
        factor_snapshot: Agent 1 全市场因子截面
        current_predictions: Agent 3 当期预测
        basic_path: tushare_stock_basic.csv 路径
        use_llm: 是否启用 LLM 因子顾问
        tracking_summary: tracker 状态摘要 (回路 B)
        expired_evals: tracker 到期评估结果 (回路 B)

    Returns:
        {"health": {...}, "suggestions": [...], "status": str,
         "tuning_directives": {...}, "miss_diagnosis": {...},
         "tracking_feedback": {...}}
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

    # ── 3. Miss Rate 诊断 + 行业分析 + 因子审计 + 漏选复盘 ──
    miss_report = _diagnose_misses(
        data_dir=data_dir,
        cache_dir=cache_dir,
        scan_date=scan_date,
        current_predictions=current_predictions,
        factor_snapshot=factor_snapshot,
        basic_path=basic_path,
    )
    if miss_report:
        report["miss_diagnosis"] = miss_report

        miss_rate = miss_report.get("miss_rate")
        if miss_rate is not None and miss_rate > 0.9:
            report["suggestions"].append(
                f"Miss Rate={miss_rate:.0%}: 市场 Top {miss_report.get('top_n_market', '?')} "
                f"大牛股中模型仅捕获 {miss_report.get('n_caught', 0)} 只, "
                f"因子集可能存在系统性盲区"
            )

        # 行业集中漏选
        blind_industries = miss_report.get("blind_industries", [])
        if blind_industries:
            top3 = blind_industries[:3]
            ind_str = ", ".join(
                f"{item['industry']}({item['n_missed']}只)" for item in top3
            )
            report["suggestions"].append(
                f"行业盲区: {ind_str} — 漏选集中在这些行业, "
                f"建议增加行业动量/板块共振因子"
            )

        # 因子盲区
        factor_blind = miss_report.get("factor_blind_spots", [])
        if factor_blind:
            top3 = factor_blind[:3]
            fb_str = ", ".join(
                f"{item['factor']}(漏选均值={item['missed_mean']:.3f} "
                f"vs 入选={item['selected_mean']:.3f})"
                for item in top3
            )
            report["suggestions"].append(
                f"因子盲区: {fb_str} — 漏选大牛股在这些因子上与入选股差异显著"
            )

    # ── 4. 漏选复盘: 用模型给漏选大牛股打分 ──
    if miss_report and miss_report.get("missed_bulls") and factor_snapshot is not None:
        replay = _replay_missed_bulls(
            missed_bulls=miss_report["missed_bulls"],
            factor_snapshot=factor_snapshot,
            cache_dir=cache_dir,
            scan_date=scan_date,
            training_meta=training_meta,
        )
        if replay:
            report["missed_bull_replay"] = replay
            # 根据复盘结果生成建议
            diag = replay.get("diagnosis", "")
            if diag == "threshold_issue":
                report["suggestions"].append(
                    f"漏选复盘: {replay.get('n_above_half_threshold', 0)}/{replay.get('n_replayed', 0)} "
                    f"只漏选大牛股概率 > 阈值的50% — 可能降低阈值即可捕获"
                )
            elif diag == "factor_blind":
                report["suggestions"].append(
                    f"漏选复盘: 漏选大牛股平均概率仅 {replay.get('avg_prob_200', 0):.4f} "
                    f"(阈值={replay.get('threshold_200', 0):.3f}), "
                    f"当前因子无法刻画大牛股特征, 需要扩充因子集"
                )
            elif diag == "label_gap":
                report["suggestions"].append(
                    f"漏选复盘: {replay.get('n_was_positive', 0)}/{replay.get('n_replayed', 0)} "
                    f"只漏选大牛股在训练集中曾是正样本但模型仍给低分, "
                    f"建议增加训练深度或调整正样本权重"
                )

    # ── 5. 回路 B: 跟踪反馈 (基于 tracker 到期评估) ──
    tracking_feedback = _analyze_tracking_feedback(
        tracking_summary=tracking_summary,
        expired_evals=expired_evals,
        factor_snapshot=factor_snapshot,
    )
    if tracking_feedback:
        report["tracking_feedback"] = tracking_feedback
        for s in tracking_feedback.get("suggestions", []):
            report["suggestions"].append(f"[跟踪] {s}")

    # ── 6. 生成结构化调参指令 ──
    tuning = _generate_tuning_directives(report, training_meta, use_llm=use_llm)
    if tuning:
        report["tuning_directives"] = tuning
        if tuning.get("retrain_required"):
            logger.info(f"  🔄 Agent 4 建议重训: {tuning.get('reason', '')}")

    # ── 7. 综合状态 ──
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


def _analyze_tracking_feedback(
    tracking_summary: dict | None,
    expired_evals: pd.DataFrame | None,
    factor_snapshot: pd.DataFrame | None,
) -> dict:
    """
    回路 B: 分析 tracker 到期评估结果, 找出成功/失败模式。

    分析维度:
      1. 总体胜率/亏损率
      2. rank=1 vs rank=5 的表现差异
      3. 成功 vs 失败的因子特征差异
      4. 行业分布

    Returns:
        {
            "win_rate": float,
            "loss_rate": float,
            "rank_analysis": {...},
            "factor_patterns": [...],
            "suggestions": [...],
            "retrain_needed": bool,
        }
    """
    result = {"suggestions": [], "retrain_needed": False}

    if tracking_summary:
        result["summary"] = tracking_summary

    if expired_evals is None or expired_evals.empty:
        return result

    n_total = len(expired_evals)
    n_success = int((expired_evals["eval_result"] == "success").sum())
    n_fail = int((expired_evals["eval_result"] == "fail").sum())
    n_loss = int((expired_evals["eval_result"] == "loss").sum())

    win_rate = n_success / max(n_total, 1)
    loss_rate = n_loss / max(n_total, 1)
    result["win_rate"] = round(win_rate, 4)
    result["loss_rate"] = round(loss_rate, 4)
    result["n_evaluated"] = n_total

    # ── rank 分析 ──
    if "rank" in expired_evals.columns:
        rank_stats = {}
        for r in sorted(expired_evals["rank"].unique()):
            sub = expired_evals[expired_evals["rank"] == r]
            if len(sub) < 2:
                continue
            rank_stats[int(r)] = {
                "n": len(sub),
                "avg_max_gain": round(float(sub["max_gain"].mean()), 4),
                "win_rate": round(float((sub["eval_result"] == "success").mean()), 4),
            }
        result["rank_analysis"] = rank_stats

        # rank=1 显著优于 rank>=4?
        top1 = expired_evals[expired_evals["rank"] == 1]
        bottom = expired_evals[expired_evals["rank"] >= 4]
        if len(top1) >= 3 and len(bottom) >= 3:
            top1_win = float((top1["eval_result"] == "success").mean())
            bot_win = float((bottom["eval_result"] == "success").mean())
            if top1_win > bot_win + 0.2:
                result["suggestions"].append(
                    f"rank=1 胜率 {top1_win:.0%} 远高于 rank>=4 的 {bot_win:.0%}, "
                    f"建议缩减 top_n 到 3"
                )

    # ── 因子模式分析 (成功 vs 失败) ──
    if factor_snapshot is not None and not factor_snapshot.empty:
        success_syms = expired_evals[expired_evals["eval_result"] == "success"]["symbol"].tolist()
        fail_syms = expired_evals[expired_evals["eval_result"].isin(["fail", "loss"])]["symbol"].tolist()

        success_in_snap = [s for s in success_syms if s in factor_snapshot.index]
        fail_in_snap = [s for s in fail_syms if s in factor_snapshot.index]

        if len(success_in_snap) >= 3 and len(fail_in_snap) >= 3:
            factor_cols = [f for f in DAILY_FACTORS if f in factor_snapshot.columns]
            patterns = []
            for f in factor_cols:
                s_vals = factor_snapshot.loc[success_in_snap, f].dropna()
                f_vals = factor_snapshot.loc[fail_in_snap, f].dropna()
                if len(s_vals) < 2 or len(f_vals) < 2:
                    continue
                diff = float(s_vals.mean() - f_vals.mean())
                std = float(factor_snapshot[f].dropna().std())
                if std > 0:
                    norm_diff = abs(diff) / std
                    if norm_diff > 0.5:
                        patterns.append({
                            "factor": f,
                            "success_mean": round(float(s_vals.mean()), 4),
                            "fail_mean": round(float(f_vals.mean()), 4),
                            "norm_diff": round(norm_diff, 4),
                        })
            patterns.sort(key=lambda x: x["norm_diff"], reverse=True)
            result["factor_patterns"] = patterns[:10]

    # ── 触发重训建议 ──
    if win_rate < 0.30 and n_total >= 10:
        result["suggestions"].append(
            f"跟踪胜率仅 {win_rate:.0%} ({n_success}/{n_total}), 建议触发重训"
        )
        result["retrain_needed"] = True

    if loss_rate > 0.30 and n_total >= 10:
        result["suggestions"].append(
            f"跟踪亏损率 {loss_rate:.0%} ({n_loss}/{n_total}) 过高, 建议调高阈值"
        )
        result["retrain_needed"] = True

    # 整体跟踪: 历史累计
    if tracking_summary:
        hist_loss = tracking_summary.get("history_loss_rate", 0)
        hist_success = tracking_summary.get("history_success_rate", 0)
        if hist_loss > 0.3:
            result["suggestions"].append(
                f"历史累计亏损率 {hist_loss:.0%}, 模型可能系统性偏差"
            )
        if hist_success > 0.5:
            result["suggestions"].append(
                f"历史累计成功率 {hist_success:.0%}, 模型表现良好"
            )

    logger.info(
        f"  回路B: {n_total} 条到期评估, "
        f"成功={n_success} 失败={n_fail} 亏损={n_loss} "
        f"(胜率={win_rate:.0%})"
    )

    return result


def _verify_past_predictions(
    predictions: list[dict],
    data_dir: str,
) -> dict:
    """回溯验证过去的预测结果。

    每个目标使用对应的前瞻窗口:
      30pct  → 10 个交易日内最大涨幅 ≥ 30%
      100pct → 40 个交易日内最大涨幅 ≥ 100%
      200pct → 120 个交易日内最大涨幅 ≥ 200%
    """
    if not predictions or not data_dir:
        return {}

    # 按预测日期分组
    from collections import defaultdict
    by_date = defaultdict(list)
    for p in predictions:
        by_date[p["scan_date"]].append(p)

    # 各目标对应的前瞻窗口和涨幅阈值
    TARGET_WINDOWS = {
        "30": (10, 0.30),
        "100": (40, 1.00),
        "200": (120, 2.00),
    }

    hits = {"30": 0, "100": 0, "200": 0}
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

            future = df[df["trade_date"] > sd]
            if future.empty:
                continue

            total += 1

            # 按各自窗口验证
            for key, (window, threshold) in TARGET_WINDOWS.items():
                window_df = future.head(window)
                if window_df.empty:
                    continue
                max_close = float(window_df["close"].max())
                actual_gain = (max_close - entry_close) / entry_close
                if actual_gain >= threshold:
                    hits[key] += 1

    return {
        "total_predictions": total,
        "hit_rate_30": round(hits["30"] / max(total, 1), 4),
        "hit_rate_100": round(hits["100"] / max(total, 1), 4),
        "hit_rate_200": round(hits["200"] / max(total, 1), 4),
        "n_hit_30": hits["30"],
        "n_hit_100": hits["100"],
        "n_hit_200": hits["200"],
    }


# ──────────────────────────────────────────────────────────
#  新增诊断: Miss Rate + 行业分析 + 因子审计
# ──────────────────────────────────────────────────────────

# 回看窗口: 用上一个扫描周期的数据来计算 "真实大牛股"
# 默认回看 120 交易日, 对应 200% 目标窗口
_MISS_LOOKBACK_DAYS = 120
_MISS_TOP_N = 30  # 取实际涨幅 Top N 作为 "真实大牛股"
_MISS_THRESHOLD = 1.0  # 120 日 max gain >= 100% 视为 "大牛股"


def _diagnose_misses(
    data_dir: str,
    cache_dir: str,
    scan_date: str,
    current_predictions: pd.DataFrame | None,
    factor_snapshot: pd.DataFrame | None,
    basic_path: str = "",
) -> dict:
    """
    Miss Rate 综合诊断: 找出市场上的真实大牛股, 与模型预测做对比。

    回看上一个窗口 (scan_date 前 120 交易日) 的实际涨幅,
    找出 top-N 大涨股, 检查当期模型是否能"看到"它们。

    Returns:
        {
            "miss_rate": float,
            "top_n_market": int,
            "n_caught": int,
            "actual_bulls": [{symbol, name, industry, gain_120d}],
            "missed_bulls": [{symbol, name, industry, gain_120d}],
            "blind_industries": [{industry, n_missed, n_total}],
            "factor_blind_spots": [{factor, missed_mean, selected_mean, diff}],
        }
    """
    # 需要日历来定位回看窗口
    daily_dir = os.path.join(cache_dir, "daily")
    calendar = _build_calendar_from_cache(daily_dir)
    if not calendar or scan_date not in set(calendar):
        # scan_date 不在日历中, 找最近的
        earlier = [d for d in calendar if d <= scan_date]
        if not earlier:
            return {}
        effective_date = earlier[-1]
    else:
        effective_date = scan_date

    # 回看起点: scan_date 前 120 日 → 那个日期之后 120 日的涨幅
    # 我们要找的是: 从 "lookback_start" 买入, 持有到 scan_date 附近, 涨了多少
    # 更好的做法: 检查过去一段时间内哪些股票是大牛 (从 past_date 到 scan_date)
    cal_idx = calendar.index(effective_date)
    lookback_start_idx = max(0, cal_idx - _MISS_LOOKBACK_DAYS)
    lookback_start = calendar[lookback_start_idx]

    # ── 扫描全市场: 从 lookback_start 到 scan_date 的涨幅 ──
    basic_info = _load_basic_info(basic_path) if basic_path else {}
    selected_symbols = set()
    if current_predictions is not None and not current_predictions.empty:
        selected_symbols = set(current_predictions["symbol"].tolist())

    market_bulls = []
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

    for fpath in csv_files:
        sym = os.path.basename(fpath).replace(".csv", "")
        if sym.startswith("bj"):
            continue
        try:
            df = pd.read_csv(fpath, usecols=["trade_date", "close"],
                             dtype={"trade_date": str})
            df = df.sort_values("trade_date").reset_index(drop=True)
        except Exception:
            continue

        start_rows = df[df["trade_date"] == lookback_start]
        end_rows = df[df["trade_date"] == effective_date]
        if start_rows.empty or end_rows.empty:
            continue

        start_close = float(start_rows.iloc[0]["close"])
        end_close = float(end_rows.iloc[0]["close"])
        if start_close <= 0:
            continue

        # 区间最大涨幅 (而非终点涨幅)
        window = df[(df["trade_date"] >= lookback_start) &
                     (df["trade_date"] <= effective_date)]
        if window.empty:
            continue
        max_close = float(window["close"].max())
        max_gain = (max_close - start_close) / start_close

        if max_gain < _MISS_THRESHOLD:
            continue

        info = basic_info.get(sym, {})
        market_bulls.append({
            "symbol": sym,
            "name": info.get("name", ""),
            "industry": info.get("industry", ""),
            "gain_120d": round(max_gain, 4),
            "selected": sym in selected_symbols,
        })

    if not market_bulls:
        return {}

    # 按涨幅排序, 取 top N
    market_bulls.sort(key=lambda x: x["gain_120d"], reverse=True)
    top_bulls = market_bulls[:_MISS_TOP_N]
    n_total = len(top_bulls)
    n_caught = sum(1 for b in top_bulls if b["selected"])
    n_missed = n_total - n_caught
    miss_rate = n_missed / max(n_total, 1)

    missed_bulls = [b for b in top_bulls if not b["selected"]]
    caught_bulls = [b for b in top_bulls if b["selected"]]

    result = {
        "lookback_start": lookback_start,
        "scan_date": effective_date,
        "miss_threshold": _MISS_THRESHOLD,
        "top_n_market": n_total,
        "n_caught": n_caught,
        "n_missed": n_missed,
        "miss_rate": round(miss_rate, 4),
        "actual_bulls": top_bulls,
        "missed_bulls": missed_bulls,
        "caught_bulls": caught_bulls,
    }

    # ── 行业集中度分析 ──
    result["blind_industries"] = _analyze_industry_blindness(
        missed_bulls, caught_bulls, selected_symbols, basic_info
    )

    # ── 因子覆盖度审计 ──
    if factor_snapshot is not None and not factor_snapshot.empty:
        result["factor_blind_spots"] = _audit_factor_coverage(
            missed_bulls, selected_symbols, factor_snapshot
        )

    # Log summary
    if n_total > 0:
        logger.info(
            f"  Miss Rate: {miss_rate:.0%} "
            f"(市场 Top{n_total} 大牛股中仅捕获 {n_caught} 只, "
            f"漏选 {n_missed} 只)"
        )
        if missed_bulls:
            top3_missed = missed_bulls[:3]
            for b in top3_missed:
                logger.info(
                    f"    漏选: {b['symbol']} {b['name']} "
                    f"{b['industry']} +{b['gain_120d']:.0%}"
                )

    return result


def _analyze_industry_blindness(
    missed_bulls: list[dict],
    caught_bulls: list[dict],
    selected_symbols: set,
    basic_info: dict,
) -> list[dict]:
    """
    行业盲区分析: 漏选的大牛股集中在哪些行业?

    Returns:
        [{industry, n_missed, n_caught, miss_rate}] 按漏选数降序
    """
    if not missed_bulls:
        return []

    # 漏选大牛股的行业分布
    missed_by_ind = defaultdict(int)
    for b in missed_bulls:
        ind = b.get("industry") or "未知"
        missed_by_ind[ind] += 1

    # 捕获大牛股的行业分布
    caught_by_ind = defaultdict(int)
    for b in caught_bulls:
        ind = b.get("industry") or "未知"
        caught_by_ind[ind] += 1

    # 合并
    all_inds = set(missed_by_ind.keys()) | set(caught_by_ind.keys())
    results = []
    for ind in all_inds:
        n_missed = missed_by_ind.get(ind, 0)
        n_caught = caught_by_ind.get(ind, 0)
        total = n_missed + n_caught
        results.append({
            "industry": ind,
            "n_missed": n_missed,
            "n_caught": n_caught,
            "n_total": total,
            "miss_rate": round(n_missed / max(total, 1), 4),
        })

    results.sort(key=lambda x: x["n_missed"], reverse=True)
    return results


def _audit_factor_coverage(
    missed_bulls: list[dict],
    selected_symbols: set,
    factor_snapshot: pd.DataFrame,
) -> list[dict]:
    """
    因子盲区审计: 漏选大牛股 vs 入选股在各因子上的均值差异。

    找出哪些因子上漏选股和入选股差异最大 — 即模型的 "盲区"。

    Returns:
        [{factor, missed_mean, selected_mean, diff, abs_diff}] 按 abs_diff 降序
    """
    if not missed_bulls or factor_snapshot.empty:
        return []

    missed_syms = [b["symbol"] for b in missed_bulls]

    # 取交集: 只分析在 factor_snapshot 中存在的股票
    missed_in_snap = [s for s in missed_syms if s in factor_snapshot.index]
    selected_in_snap = [s for s in selected_symbols if s in factor_snapshot.index]

    if not missed_in_snap or not selected_in_snap:
        return []

    # 提取因子列 (只用 DAILY_FACTORS)
    factor_cols = [f for f in DAILY_FACTORS if f in factor_snapshot.columns]
    if not factor_cols:
        return []

    missed_df = factor_snapshot.loc[missed_in_snap, factor_cols]
    selected_df = factor_snapshot.loc[selected_in_snap, factor_cols]

    results = []
    for f in factor_cols:
        m_vals = missed_df[f].dropna()
        s_vals = selected_df[f].dropna()
        if len(m_vals) < 3 or len(s_vals) < 3:
            continue

        m_mean = float(m_vals.mean())
        s_mean = float(s_vals.mean())
        diff = m_mean - s_mean

        # 标准化差异 (用全市场标准差)
        all_vals = factor_snapshot[f].dropna()
        std = float(all_vals.std()) if len(all_vals) > 10 else 1.0
        norm_diff = abs(diff) / max(std, 1e-8)

        results.append({
            "factor": f,
            "missed_mean": round(m_mean, 4),
            "selected_mean": round(s_mean, 4),
            "diff": round(diff, 4),
            "norm_diff": round(norm_diff, 4),
        })

    results.sort(key=lambda x: x["norm_diff"], reverse=True)
    return results


def _build_calendar_from_cache(daily_dir: str) -> list[str]:
    """从 daily cache 目录推断交易日历。"""
    csv_files = sorted(glob.glob(os.path.join(daily_dir, "*.csv")))
    if not csv_files:
        return []

    # 抽样几只大盘股获取日历
    all_dates = set()
    for fp in csv_files[:5]:
        try:
            df = pd.read_csv(fp, usecols=["trade_date"])
            df["trade_date"] = df["trade_date"].astype(str)
            all_dates.update(df["trade_date"].tolist())
        except Exception:
            continue

    return sorted(all_dates)


def _load_basic_info(basic_path: str) -> dict[str, dict]:
    """加载股票基本信息 (名称、行业)。"""
    if not basic_path or not os.path.exists(basic_path):
        return {}
    try:
        df = pd.read_csv(basic_path, dtype=str)
    except Exception:
        return {}
    info = {}
    for _, row in df.iterrows():
        ts = str(row.get("ts_code", ""))
        parts = ts.split(".")
        if len(parts) == 2:
            sym = parts[1].lower() + parts[0]
            info[sym] = {
                "name": str(row.get("name", "")),
                "industry": str(row.get("industry", "")),
            }
    return info


# ──────────────────────────────────────────────────────────
#  漏选复盘 + 结构化调参指令
# ──────────────────────────────────────────────────────────

def _replay_missed_bulls(
    missed_bulls: list[dict],
    factor_snapshot: pd.DataFrame,
    cache_dir: str,
    scan_date: str,
    training_meta: dict,
) -> dict:
    """
    漏选复盘: 用当前模型给漏选大牛股打分, 诊断失败原因。

    三种诊断结论:
      1. threshold_issue — 概率不低(>阈值50%), 降阈值可能捕获
      2. factor_blind — 概率极低, 因子无法刻画这类股票
      3. label_gap — 训练中有正样本但模型仍给低分, 模型欠学习

    Returns:
        {
            "n_replayed": int,
            "scores": [{symbol, name, prob_30, prob_100, prob_200}],
            "avg_prob_200": float,
            "threshold_200": float,
            "n_above_half_threshold": int,
            "n_was_positive": int,  # 训练集中曾是正样本的数量
            "diagnosis": "threshold_issue|factor_blind|label_gap",
        }
    """
    model_dir = os.path.join(cache_dir, "bull_models", scan_date)
    if not os.path.exists(model_dir):
        return {}

    # 加载模型
    models = {}
    for tname in TARGETS:
        mp = os.path.join(model_dir, f"model_{tname}.pkl")
        if os.path.exists(mp):
            try:
                with open(mp, "rb") as f:
                    models[tname] = pickle.load(f)
            except Exception:
                continue

    if not models:
        return {}

    # 从 meta 获取阈值和特征列
    thresholds = {}
    feature_cols_map = {}
    for tname in TARGETS:
        tmeta = training_meta.get(tname, {})
        thresholds[tname] = tmeta.get("best_threshold", 0.15)
        feature_cols_map[tname] = tmeta.get("feature_cols", DAILY_FACTORS)

    # 给漏选大牛股打分
    scores = []
    for b in missed_bulls:
        sym = b["symbol"]
        if sym not in factor_snapshot.index:
            continue

        row_scores = {"symbol": sym, "name": b.get("name", ""), "gain_120d": b["gain_120d"]}
        for tname, model in models.items():
            feat_cols = feature_cols_map.get(tname, DAILY_FACTORS)
            X = []
            for f in feat_cols:
                v = factor_snapshot.loc[sym].get(f, np.nan) if f in factor_snapshot.columns else np.nan
                X.append(float(v) if pd.notna(v) else 0.0)
            X = np.array([X])
            try:
                prob = float(model.predict_proba(X)[0, 1])
            except Exception:
                prob = 0.0
            key = tname.replace("pct", "")
            row_scores[f"prob_{key}"] = round(prob, 6)
        scores.append(row_scores)

    if not scores:
        return {}

    # 统计
    n_replayed = len(scores)
    avg_prob_200 = np.mean([s.get("prob_200", 0) for s in scores])
    avg_prob_100 = np.mean([s.get("prob_100", 0) for s in scores])
    avg_prob_30 = np.mean([s.get("prob_30", 0) for s in scores])
    thresh_200 = thresholds.get("200pct", 0.15)
    thresh_100 = thresholds.get("100pct", 0.15)
    thresh_30 = thresholds.get("30pct", 0.15)

    # 有多少只漏选大牛股的概率 > 阈值的 50%?
    n_above_half_200 = sum(1 for s in scores if s.get("prob_200", 0) > thresh_200 * 0.5)
    n_above_half_100 = sum(1 for s in scores if s.get("prob_100", 0) > thresh_100 * 0.5)

    # 检查训练集中的正样本 (从 meta 的 top_features 推断)
    # 更直接: 检查这些股票在过去训练窗口内是否有 label=1
    n_was_positive = _count_training_positives(
        [s["symbol"] for s in scores], cache_dir, scan_date
    )

    # 诊断
    if n_above_half_200 > n_replayed * 0.3:
        diagnosis = "threshold_issue"
    elif n_was_positive > n_replayed * 0.3:
        diagnosis = "label_gap"
    else:
        diagnosis = "factor_blind"

    result = {
        "n_replayed": n_replayed,
        "scores": scores[:10],  # 只保存 top 10 避免 JSON 过大
        "avg_prob_30": round(avg_prob_30, 6),
        "avg_prob_100": round(avg_prob_100, 6),
        "avg_prob_200": round(avg_prob_200, 6),
        "threshold_30": thresh_30,
        "threshold_100": thresh_100,
        "threshold_200": thresh_200,
        "n_above_half_threshold": n_above_half_200,
        "n_was_positive": n_was_positive,
        "diagnosis": diagnosis,
    }

    # Log
    logger.info(f"  漏选复盘: {n_replayed} 只, "
                f"avg_prob_200={avg_prob_200:.4f} (阈值={thresh_200:.3f}), "
                f"诊断={diagnosis}")
    for s in scores[:3]:
        logger.info(f"    {s['symbol']} {s.get('name','')} "
                    f"涨幅={s['gain_120d']:.0%} → "
                    f"p30={s.get('prob_30',0):.4f} "
                    f"p100={s.get('prob_100',0):.4f} "
                    f"p200={s.get('prob_200',0):.4f}")

    return result


def _count_training_positives(
    symbols: list[str],
    cache_dir: str,
    scan_date: str,
) -> int:
    """
    检查漏选大牛股在训练数据中是否曾有 label=1。

    回看 scan_date 前 12 个月的训练窗口, 每只股票是否在
    某个采样日有 120d 涨幅 >= 200%。
    """
    daily_dir = os.path.join(cache_dir, "daily")
    symbol_set = set(symbols)
    n_positive = 0

    for sym in symbols:
        fpath = os.path.join(daily_dir, f"{sym}.csv")
        if not os.path.exists(fpath):
            continue
        try:
            df = pd.read_csv(fpath, usecols=["trade_date", "close"],
                             dtype={"trade_date": str})
            df = df.sort_values("trade_date").reset_index(drop=True)
        except Exception:
            continue

        # 训练窗口: scan_date - 12月 ~ scan_date - 120日
        # 粗略: 前 240 日到前 120 日
        all_dates = df["trade_date"].tolist()
        sd_rows = df[df["trade_date"] <= scan_date]
        if len(sd_rows) < 240:
            continue

        # 检查训练窗口内是否有任何一天 +120d 涨幅 >= 200%
        start_idx = max(0, len(sd_rows) - 240)
        end_idx = max(0, len(sd_rows) - 120)
        was_positive = False
        for i in range(start_idx, end_idx, 5):  # 每 5 天采样
            if i + 120 >= len(df):
                continue
            entry = float(df.iloc[i]["close"])
            if entry <= 0:
                continue
            future_close = float(df.iloc[i + 120]["close"])
            if (future_close - entry) / entry >= 2.0:
                was_positive = True
                break

        if was_positive:
            n_positive += 1

    return n_positive


def _generate_tuning_directives(
    report: dict,
    training_meta: dict,
    use_llm: bool = False,
) -> dict:
    """
    从 Agent 4 双回路诊断结果生成结构化调参指令。

    输入来源:
      - 回路 A: miss_diagnosis, missed_bull_replay (漏网之鱼)
      - 回路 B: tracking_feedback (Top 5 跟踪到期评估)
      - 模型健康: health (AUC, precision, recall)

    安全限制:
      - 单次最多 MAX_ACTIONS_PER_DIRECTIVE 个修改动作
      - 不含信息类 (flag_*) 和阈值调整 (Agent 3 层面)

    返回格式:
    {
        "retrain_required": bool,
        "reason": str,
        "diagnosis": str,
        "escalation_level": int,    # 1=调参 2=换因子 3=换模型
        "actions": [...]
    }
    """
    actions = []
    reasons = []

    miss_diag = report.get("miss_diagnosis", {})
    replay = report.get("missed_bull_replay", {})
    health = report.get("health", {})
    tracking = report.get("tracking_feedback", {})

    miss_rate = miss_diag.get("miss_rate", 0)
    diagnosis = replay.get("diagnosis", "")

    # 回路 B 信号
    tracking_retrain = tracking.get("retrain_needed", False)
    tracking_win_rate = tracking.get("win_rate", 1.0)
    tracking_loss_rate = tracking.get("loss_rate", 0.0)

    # ── 确定当前迭代轮次 (从 meta 中读 model_type 推断) ──
    current_model = "lgbm"
    current_drop_factors = []
    for tname, tmeta in training_meta.items():
        if isinstance(tmeta, dict):
            current_model = tmeta.get("model_type", "lgbm")
            current_drop_factors = tmeta.get("drop_factors", [])
            break

    # ── 方案 A: LLM 因子顾问 ──
    llm_advice = {}
    if use_llm and miss_rate > 0.5:
        try:
            from .llm_advisor import run_factor_advisor
            llm_advice = run_factor_advisor(
                training_meta=training_meta,
                miss_diagnosis=miss_diag,
                missed_bull_replay=replay,
                factor_blind_spots=miss_diag.get("factor_blind_spots", []),
                blind_industries=miss_diag.get("blind_industries", []),
                current_model=current_model,
                current_drop_factors=current_drop_factors,
            )
        except Exception as e:
            logger.warning(f"  LLM 因子顾问调用失败: {e}")

    # ── 分析因子有效性 ──
    factor_blind_spots = miss_diag.get("factor_blind_spots", [])
    blind_industries = miss_diag.get("blind_industries", [])

    # 低重要性因子: 在所有 target 模型中排名倒数的因子
    low_importance_factors = _identify_low_importance_factors(training_meta)
    # 方向错误因子: 漏选大牛股均值方向与模型偏好相反 (norm_diff > 1.5)
    wrong_direction_factors = [
        s["factor"] for s in factor_blind_spots
        if s.get("norm_diff", 0) > 1.5
    ]

    # ── Level 1: 调参 ──
    if diagnosis == "threshold_issue":
        for tname in ["200pct", "100pct", "30pct"]:
            tmeta = training_meta.get(tname, {})
            old_thresh = tmeta.get("best_threshold", 0.15)
            new_thresh = max(old_thresh * 0.5, 0.01)
            actions.append({
                "type": "adjust_threshold",
                "target": tname,
                "old_value": old_thresh,
                "new_value": round(new_thresh, 3),
            })
        reasons.append("漏选大牛股概率 > 阈值的50%, 降低阈值可捕获")

    elif diagnosis == "label_gap" and not current_drop_factors:
        # 第一次 label_gap: 先调参
        actions.append({
            "type": "adjust_pos_weight",
            "new_max": 10.0,
        })
        actions.append({
            "type": "adjust_estimators",
            "new_n_estimators": 1200,
        })
        # 同时剔除方向错误因子 (Level 2 前置)
        if wrong_direction_factors:
            actions.append({
                "type": "drop_factors",
                "factors": wrong_direction_factors,
                "reason": "这些因子在漏选大牛股上方向相反 (norm_diff>1.5)",
            })
            reasons.append(f"剔除 {len(wrong_direction_factors)} 个方向错误因子: "
                          f"{', '.join(wrong_direction_factors[:5])}")
        reasons.append("调参 + 因子剔除: 提升正样本权重, 移除误导因子")

    elif diagnosis == "label_gap" and current_drop_factors and current_model == "lgbm":
        # 第二次 label_gap (已调参+剔因子仍失败): 换模型
        actions.append({
            "type": "switch_model",
            "new_model": "xgboost",
            "reason": "LightGBM 多轮调参后仍无法学习正样本, 尝试 XGBoost",
        })
        # 恢复默认因子 (新模型可能有不同的因子利用方式)
        if current_drop_factors:
            actions.append({
                "type": "restore_factors",
                "reason": "换模型后恢复全量因子, 让新模型自行筛选",
            })
        reasons.append("LightGBM 多轮调参失败 → 换 XGBoost")

    elif diagnosis == "label_gap" and current_model == "xgboost":
        # 第三次 label_gap (XGBoost 也失败): 换 RF
        actions.append({
            "type": "switch_model",
            "new_model": "random_forest",
            "reason": "XGBoost 也无法学习, 尝试 RandomForest (不同的特征选择机制)",
        })
        reasons.append("XGBoost 也失败 → 换 RandomForest")

    elif diagnosis == "factor_blind":
        # 因子盲区: 剔除低效因子 + 标记缺失因子
        factors_to_drop = list(set(low_importance_factors) & set(wrong_direction_factors))
        if not factors_to_drop and wrong_direction_factors:
            factors_to_drop = wrong_direction_factors[:5]

        if factors_to_drop:
            actions.append({
                "type": "drop_factors",
                "factors": factors_to_drop,
                "reason": "低重要性且方向错误",
            })
            reasons.append(f"剔除 {len(factors_to_drop)} 个无效因子")

        missing_factor_hints = []
        if blind_industries:
            top_ind = [item["industry"] for item in blind_industries[:5]]
            missing_factor_hints.append(f"行业动量因子 (漏选行业: {', '.join(top_ind)})")
        for spot in factor_blind_spots[:3]:
            if spot["factor"].startswith("volatility") and spot["diff"] > 0:
                missing_factor_hints.append("趋势强度因子 (大牛股高波动)")
            if spot["factor"].startswith("mf_big_cumsum") and spot["diff"] < 0:
                missing_factor_hints.append("主力洗盘识别因子 (大单流出≠利空)")

        actions.append({
            "type": "flag_factor_gap",
            "missing_factor_hints": missing_factor_hints,
        })
        reasons.append("因子集无法刻画大牛股特征, 需扩充因子")

    # ── 通用: AUC 过低 → 增加回看窗口 ──
    for tname, h in health.items():
        if isinstance(h, dict) and "auc_low" in h.get("issues", []):
            actions.append({
                "type": "increase_lookback",
                "new_months": 18,
            })
            reasons.append(f"{tname} AUC 过低, 扩大训练窗口")
            break

    # ── 通用: 正样本极少 ──
    for tname in ["200pct", "100pct"]:
        tmeta = training_meta.get(tname, {})
        pos_rate = tmeta.get("pos_rate_train", 0)
        if 0 < pos_rate < 0.003:
            actions.append({
                "type": "flag_sparse_positives",
                "target": tname,
                "pos_rate": pos_rate,
                "suggestion": "考虑用区间最大涨幅替代终点涨幅作为标签",
            })
            reasons.append(f"{tname} 正样本仅 {pos_rate:.1%}, 考虑放宽标签定义")

    # ── 计算 escalation level ──
    has_factor_change = any(a["type"] in ("drop_factors", "restore_factors") for a in actions)
    has_model_switch = any(a["type"] == "switch_model" for a in actions)
    if has_model_switch:
        escalation = 3
    elif has_factor_change:
        escalation = 2
    else:
        escalation = 1

    retrain_required = (
        diagnosis in ("threshold_issue", "label_gap")
        and miss_rate > 0.8
    ) or has_factor_change or has_model_switch or tracking_retrain

    # 回路 B 信号追加原因
    if tracking_retrain:
        reasons.append(f"跟踪反馈: 胜率={tracking_win_rate:.0%} 亏损率={tracking_loss_rate:.0%}")

    reason = "; ".join(reasons) if reasons else ""

    # ── 安全限制: 修改类动作不超过 MAX_ACTIONS_PER_DIRECTIVE ──
    modify_types = {"adjust_pos_weight", "adjust_estimators", "increase_lookback",
                    "drop_factors", "restore_factors", "switch_model"}
    modify_actions = [a for a in actions if a["type"] in modify_types]
    info_actions = [a for a in actions if a["type"] not in modify_types]
    if len(modify_actions) > MAX_ACTIONS_PER_DIRECTIVE:
        modify_actions = modify_actions[:MAX_ACTIONS_PER_DIRECTIVE]
        reasons.append(f"安全限制: 截断为 {MAX_ACTIONS_PER_DIRECTIVE} 个修改动作")
    actions = modify_actions + info_actions

    # ── 合并 LLM 建议 ──
    if llm_advice:
        # LLM 建议剔除的因子 (与规则引擎取并集)
        llm_drops = llm_advice.get("drop_factors", [])
        if llm_drops:
            existing_drop_actions = [a for a in actions if a["type"] == "drop_factors"]
            if existing_drop_actions:
                # 合并到已有的 drop_factors action
                existing_factors = existing_drop_actions[0].get("factors", [])
                merged = list(set(existing_factors + llm_drops))
                existing_drop_actions[0]["factors"] = merged
                existing_drop_actions[0]["reason"] = (
                    existing_drop_actions[0].get("reason", "") +
                    "; LLM 建议: " +
                    ", ".join(f"{f}: {llm_advice.get('drop_reasons', {}).get(f, '')}"
                             for f in llm_drops[:3])
                )
            else:
                actions.append({
                    "type": "drop_factors",
                    "factors": llm_drops,
                    "reason": "LLM 因子顾问建议剔除",
                })
            if not has_factor_change:
                escalation = max(escalation, 2)
                has_factor_change = True
                retrain_required = True

        # LLM 建议换模型
        llm_model = llm_advice.get("model_suggestion", "keep")
        if llm_model.startswith("switch_") and not has_model_switch:
            new_model = llm_model.replace("switch_", "").replace("rf", "random_forest")
            if new_model != current_model:
                actions.append({
                    "type": "switch_model",
                    "new_model": new_model,
                    "reason": f"LLM 建议: {llm_advice.get('analysis', '')}",
                })
                escalation = 3
                retrain_required = True

        # LLM 新因子建议 (记录到 actions, 由方案 C 执行)
        new_factors = llm_advice.get("new_factor_ideas", [])
        if new_factors:
            actions.append({
                "type": "llm_new_factors",
                "factors": new_factors,
                "reason": "LLM 因子工程师建议",
            })

        if llm_advice.get("analysis"):
            reason = reason + ("; " if reason else "") + f"LLM: {llm_advice['analysis']}"

    return {
        "retrain_required": retrain_required,
        "reason": reason,
        "diagnosis": diagnosis,
        "miss_rate": miss_rate,
        "escalation_level": escalation,
        "current_model": current_model,
        "actions": actions,
        "llm_advice": llm_advice if llm_advice else None,
    }


def _identify_low_importance_factors(training_meta: dict) -> list[str]:
    """
    从训练 meta 中找出低重要性因子: 在所有 target 模型中平均 importance 排名倒数 30%。
    """
    factor_scores = {}
    n_targets = 0

    for tname, tmeta in training_meta.items():
        if not isinstance(tmeta, dict):
            continue
        top_features = tmeta.get("top_features", [])
        if not top_features:
            continue
        n_targets += 1
        # top_features 是按重要性降序的
        for rank, feat_info in enumerate(top_features):
            fname = feat_info.get("name", "")
            imp = feat_info.get("importance", 0)
            if fname not in factor_scores:
                factor_scores[fname] = []
            factor_scores[fname].append(imp)

    if not factor_scores or n_targets == 0:
        return []

    # 计算平均重要性
    avg_imp = {f: sum(vs) / len(vs) for f, vs in factor_scores.items()}
    sorted_factors = sorted(avg_imp.items(), key=lambda x: x[1])

    # 底部 30%
    n_low = max(1, len(sorted_factors) * 3 // 10)
    return [f for f, _ in sorted_factors[:n_low]]
