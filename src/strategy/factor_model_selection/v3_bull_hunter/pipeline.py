"""
Bull Hunter v3 — Pipeline 编排器

v4 重构: 三种运行模式 + tracker 跟踪 + 双回路反馈。

模式:
  1. daily  — 每日轻量预测: Agent 1 + Agent 3 (复用 latest 模型), 记录到 tracker
  2. train  — 周训/事件驱动训练: Agent 1 + Agent 2, 更新 latest 模型
  3. review — Agent 4 双回路评估: 漏网之鱼 + 跟踪到期, 可能触发 Agent 2 重训
  4. scan   — 完整单日: train + daily (兼容旧接口)
  5. backtest — 滚动回测 + 实际收益验证
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, replace

import pandas as pd

from .agent1_factor import run_factor_generation
from .agent2_train import TrainConfig, run_training, needs_training, get_latest_model_dir, _build_calendar
from .agent3_predict import PredictConfig, run_prediction
from .agent4_monitor import run_monitor
from .tracker import PredictionTracker

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
    use_llm: bool = False


# ──────────────────────────────────────────────────────────
#  Mode 1: 每日轻量预测
# ──────────────────────────────────────────────────────────

def run_daily(
    scan_date: str,
    cfg: PipelineConfig | None = None,
) -> pd.DataFrame:
    """
    每日预测: 复用 latest 模型, 输出 Top 5 A 级候选, 记录到 tracker。

    流程:
      1. Agent 1: 因子快照
      2. Agent 3: Top 5 预测 (用 latest 模型)
      3. Tracker: 记录预测 + 更新活跃项价格 + 评估到期项
    """
    cfg = cfg or PipelineConfig()
    predict_cfg = cfg.predict_cfg or PredictConfig()

    logger.info(f"=== Bull Hunter v3 DAILY: {scan_date} ===")

    # 获取 latest 模型
    model_dir = get_latest_model_dir(cfg.cache_dir)
    if model_dir is None:
        logger.error("无可用模型 (latest 不存在), 请先运行 --train")
        return pd.DataFrame()
    logger.info(f"  使用模型: {os.path.basename(model_dir)}")

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

    # ── Tracker: 获取近期预测 (去重用) ──
    calendar = _build_calendar(cfg.cache_dir)
    tracker = PredictionTracker(os.path.join(cfg.results_dir, "tracking"))
    recent = tracker.get_recent_predictions(scan_date, calendar, predict_cfg.dedup_days)

    # ── Agent 3: 每日 Top 5 预测 ──
    logger.info("── Agent 3: Top 5 预测 ──")
    predictions = run_prediction(
        factor_snapshot=snapshot,
        model_dir=model_dir,
        scan_date=scan_date,
        cfg=predict_cfg,
        recent_predictions=recent,
    )

    # 保存结果
    daily_dir = os.path.join(cfg.results_dir, "daily")
    os.makedirs(daily_dir, exist_ok=True)
    if not predictions.empty:
        out_path = os.path.join(daily_dir, f"{scan_date}.csv")
        predictions.to_csv(out_path, index=False, encoding="utf-8-sig")
        logger.info(f"  → 已保存 {len(predictions)} 只候选 → {out_path}")

        # 记录到 tracker
        model_date = os.path.basename(os.path.realpath(model_dir))
        tracker.record_predictions(predictions, scan_date, cfg.data_dir, calendar, model_date)
    else:
        logger.info("  → 今日无 A 级候选")

    # ── Tracker: 更新活跃项 + 评估到期项 ──
    tracker.update_prices(scan_date, cfg.data_dir, calendar)
    expired = tracker.evaluate_expired(scan_date, calendar)

    n_active = tracker.get_active_count()
    logger.info(f"  Tracker: {n_active} 条活跃跟踪, {len(expired)} 条到期")

    logger.info(f"=== DAILY 完成 ===")
    return predictions


# ──────────────────────────────────────────────────────────
#  Mode 2: 训练 (周训 / 事件驱动)
# ──────────────────────────────────────────────────────────

def run_train(
    scan_date: str,
    cfg: PipelineConfig | None = None,
    force: bool = False,
    trigger: str = "weekly",
    trigger_reason: str = "",
) -> dict:
    """
    训练模型: Agent 1 + Agent 2。

    Args:
        force: 强制训练 (忽略间隔和缓存)
        trigger: 训练触发方式 (weekly / agent4_feedback / manual)
        trigger_reason: 触发原因描述

    Returns:
        Agent 2 训练结果 dict
    """
    cfg = cfg or PipelineConfig()
    train_cfg = cfg.train_cfg or TrainConfig()

    logger.info(f"=== Bull Hunter v3 TRAIN: {scan_date} (trigger={trigger}) ===")

    # 检查是否需要训练
    if not force and trigger == "weekly" and not needs_training(cfg.cache_dir, scan_date):
        logger.info("  距上次训练未满一周, 跳过 (用 --force 强制训练)")
        return {}

    # 注入触发元数据
    train_cfg = replace(train_cfg, trigger=trigger, trigger_reason=trigger_reason)
    if force:
        train_cfg = replace(train_cfg, force_retrain=True)

    # ── Agent 1: 因子快照 (训练需要) ──
    logger.info("── Agent 1: 因子快照 ──")
    snapshot = run_factor_generation(
        cache_dir=cfg.cache_dir,
        data_dir=cfg.data_dir,
        scan_date=scan_date,
        basic_path=cfg.basic_path,
    )
    if snapshot.empty:
        logger.error("Agent 1 返回空快照, 终止")
        return {}
    logger.info(f"  → {len(snapshot)} 只股票")

    # ── Agent 2: 训练 ──
    logger.info("── Agent 2: 训练分类器 ──")
    train_results = run_training(
        cache_dir=cfg.cache_dir,
        scan_date=scan_date,
        cfg=train_cfg,
    )
    if not train_results:
        logger.error("Agent 2 训练失败")
        return {}

    logger.info(f"=== TRAIN 完成: {len(train_results)} 个模型 ===")
    return train_results


# ──────────────────────────────────────────────────────────
#  Mode 3: 复盘 (Agent 4 双回路评估)
# ──────────────────────────────────────────────────────────

def run_review(
    scan_date: str,
    cfg: PipelineConfig | None = None,
) -> dict:
    """
    Agent 4 双回路评估, 可能触发 Agent 2 重训。

    流程:
      1. Agent 1: 因子快照
      2. Agent 4: 双回路诊断 (漏网之鱼 + 跟踪反馈)
      3. 如果 tuning_directives.retrain_required:
         → Agent 2: 带调参指令重训
         → Agent 3: 新模型重跑当日
         → Agent 4: 验证新模型是否改善
    """
    cfg = cfg or PipelineConfig()
    train_cfg = cfg.train_cfg or TrainConfig()
    predict_cfg = cfg.predict_cfg or PredictConfig()

    logger.info(f"=== Bull Hunter v3 REVIEW: {scan_date} ===")

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
        return {}

    # ── 获取当前预测 (用 latest 模型) ──
    model_dir = get_latest_model_dir(cfg.cache_dir)
    current_predictions = pd.DataFrame()
    if model_dir:
        current_predictions = run_prediction(
            factor_snapshot=snapshot,
            model_dir=model_dir,
            scan_date=scan_date,
            cfg=predict_cfg,
        )

    # ── Tracker 数据 ──
    calendar = _build_calendar(cfg.cache_dir)
    tracker = PredictionTracker(os.path.join(cfg.results_dir, "tracking"))
    tracking_summary = tracker.get_tracking_summary()
    expired_evals = tracker.get_expired_for_review()

    # ── Agent 4: 双回路诊断 ──
    logger.info("── Agent 4: 双回路评估 ──")
    health = run_monitor(
        cache_dir=cfg.cache_dir,
        data_dir=cfg.data_dir,
        scan_date=scan_date,
        factor_snapshot=snapshot,
        current_predictions=current_predictions,
        basic_path=cfg.basic_path,
        use_llm=cfg.use_llm,
        tracking_summary=tracking_summary,
        expired_evals=expired_evals,
    )

    if health.get("suggestions"):
        for s in health["suggestions"]:
            logger.info(f"  💡 {s}")

    # ── 检查是否需要重训 ──
    tuning = health.get("tuning_directives", {})
    if tuning.get("retrain_required"):
        logger.info(f"  🔄 触发重训: {tuning.get('reason', '')}")

        # 应用调参指令
        new_train_cfg = _apply_tuning_directives(train_cfg, tuning)
        new_train_cfg = replace(
            new_train_cfg,
            trigger="agent4_feedback",
            trigger_reason=tuning.get("reason", ""),
            force_retrain=True,
        )

        # Agent 2: 重训
        logger.info("── Agent 2: 重训 (Agent 4 反馈) ──")
        retrain_results = run_training(
            cache_dir=cfg.cache_dir,
            scan_date=scan_date,
            cfg=new_train_cfg,
        )

        if retrain_results:
            # Agent 3: 用新模型重跑
            new_model_dir = get_latest_model_dir(cfg.cache_dir)
            if new_model_dir:
                logger.info("── Agent 3: 新模型验证 ──")
                new_predictions = run_prediction(
                    factor_snapshot=snapshot,
                    model_dir=new_model_dir,
                    scan_date=scan_date,
                    cfg=predict_cfg,
                )
                n_old = len(current_predictions) if not current_predictions.empty else 0
                n_new = len(new_predictions) if not new_predictions.empty else 0
                logger.info(f"  对比: 旧模型 {n_old} 只 → 新模型 {n_new} 只")
        else:
            logger.warning("  重训失败, 保持旧模型")
    else:
        logger.info(f"  Agent 4: 无需重训 (status={health.get('status', 'unknown')})")

    # 保存健康报告
    out_dir = os.path.join(cfg.results_dir, scan_date)
    os.makedirs(out_dir, exist_ok=True)
    health_path = os.path.join(out_dir, "health_report.json")
    with open(health_path, "w", encoding="utf-8") as f:
        json.dump(health, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"=== REVIEW 完成: status={health.get('status', 'unknown')} ===")
    return health


# ──────────────────────────────────────────────────────────
#  Mode 4: 完整扫描 (train + daily, 兼容旧接口)
# ──────────────────────────────────────────────────────────

def run_scan(
    scan_date: str,
    cfg: PipelineConfig | None = None,
    skip_monitor: bool = False,
) -> pd.DataFrame:
    """
    完整单日扫描: 训练 + 预测 (兼容旧接口, 回测模式用)。

    Args:
        skip_monitor: 跳过 Agent 4 (回测模式)
    """
    cfg = cfg or PipelineConfig()
    train_cfg = cfg.train_cfg or TrainConfig()
    predict_cfg = cfg.predict_cfg or PredictConfig()

    logger.info(f"=== Bull Hunter v3 SCAN: {scan_date} ===")

    # Agent 1
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

    # Agent 2
    logger.info("── Agent 2: 训练分类器 ──")
    train_results = run_training(
        cache_dir=cfg.cache_dir,
        scan_date=scan_date,
        cfg=train_cfg,
    )
    if not train_results:
        logger.error("Agent 2 训练失败, 终止")
        return pd.DataFrame()

    model_dir = get_latest_model_dir(cfg.cache_dir)
    if model_dir is None:
        model_dir = os.path.join(cfg.cache_dir, "bull_models", scan_date)
    logger.info(f"  → {len(train_results)} 个模型")

    # Agent 3
    logger.info("── Agent 3: 预测 ──")
    predictions = run_prediction(
        factor_snapshot=snapshot,
        model_dir=model_dir,
        scan_date=scan_date,
        cfg=predict_cfg,
    )

    if not predictions.empty:
        out_dir = os.path.join(cfg.results_dir, scan_date)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "predictions.csv")
        predictions.to_csv(out_path, index=False, encoding="utf-8-sig")
        logger.info(f"  → 已保存 {len(predictions)} 只候选 → {out_path}")
    else:
        logger.warning("Agent 3 无候选")

    # Agent 4
    if skip_monitor:
        logger.info("── Agent 4: 跳过 (回测模式) ──")
    else:
        logger.info("── Agent 4: 监控 ──")
        health = run_monitor(
            cache_dir=cfg.cache_dir,
            data_dir=cfg.data_dir,
            scan_date=scan_date,
            factor_snapshot=snapshot,
            current_predictions=predictions,
            basic_path=cfg.basic_path,
            use_llm=cfg.use_llm,
        )
        if health.get("suggestions"):
            for s in health["suggestions"]:
                logger.info(f"  💡 {s}")

    logger.info(f"=== 完成 ===")
    return predictions


# ──────────────────────────────────────────────────────────
#  调参指令应用
# ──────────────────────────────────────────────────────────

def _apply_tuning_directives(
    cfg: TrainConfig,
    tuning: dict,
) -> TrainConfig:
    """
    将 Agent 4 的调参指令应用到 TrainConfig, 返回新配置。

    支持的指令类型:
      Level 1 — 调参:
        - adjust_threshold: 调整预测阈值 (Agent 3 层面, 此处仅 log)
        - adjust_pos_weight: 调整 max_scale_pos_weight
        - adjust_estimators: 调整 n_estimators
        - increase_lookback: 增加 lookback_months
      Level 2 — 因子增删:
        - drop_factors: 剔除指定因子
        - restore_factors: 恢复全量因子 (换模型时)
      Level 3 — 模型切换:
        - switch_model: 切换 model_type (lgbm → xgboost → random_forest)
    """
    actions = tuning.get("actions", [])
    if not actions:
        return cfg

    escalation = tuning.get("escalation_level", 1)
    logger.info(f"    调参等级: Level {escalation} "
                f"({'调参' if escalation == 1 else '换因子' if escalation == 2 else '换模型'})")

    updates = {}
    for action in actions:
        atype = action.get("type", "")

        # ── Level 1: 超参调整 ──
        if atype == "adjust_pos_weight":
            new_weight = action.get("new_max", cfg.max_scale_pos_weight)
            updates["max_scale_pos_weight"] = new_weight
            logger.info(f"    调参: max_scale_pos_weight {cfg.max_scale_pos_weight} → {new_weight}")

        elif atype == "adjust_estimators":
            new_n = action.get("new_n_estimators", cfg.n_estimators)
            updates["n_estimators"] = new_n
            logger.info(f"    调参: n_estimators {cfg.n_estimators} → {new_n}")

        elif atype == "increase_lookback":
            new_months = action.get("new_months", cfg.lookback_months)
            updates["lookback_months"] = new_months
            logger.info(f"    调参: lookback_months {cfg.lookback_months} → {new_months}")

        elif atype == "adjust_threshold":
            target = action.get("target", "")
            old_v = action.get("old_value", "?")
            new_v = action.get("new_value", "?")
            logger.info(f"    调参 (Agent 3): {target} 阈值 {old_v} → {new_v}")

        # ── Level 2: 因子增删 ──
        elif atype == "drop_factors":
            factors = action.get("factors", [])
            if factors:
                # 累积: 之前剔除的 + 新剔除的
                existing_drops = list(cfg.drop_factors)
                new_drops = list(set(existing_drops + factors))
                updates["drop_factors"] = new_drops
                added = [f for f in factors if f not in existing_drops]
                logger.info(f"    因子剔除: +{len(added)} 个 → 总剔除 {len(new_drops)} 个")
                for f in added[:5]:
                    logger.info(f"      - {f} ({action.get('reason', '')})")

        elif atype == "restore_factors":
            updates["drop_factors"] = []
            updates["add_factors"] = []
            logger.info(f"    因子恢复: 清空 drop/add, 使用全量 DAILY_FACTORS")

        # ── Level 3: 模型切换 ──
        elif atype == "switch_model":
            new_model = action.get("new_model", "lgbm")
            updates["model_type"] = new_model
            logger.info(f"    🔄 模型切换: {cfg.model_type} → {new_model}")
            logger.info(f"       原因: {action.get('reason', '')}")

        # ── 信息类 (不修改配置) ──
        elif atype == "flag_factor_gap":
            hints = action.get("missing_factor_hints", [])
            for h in hints:
                logger.info(f"    ⚠️ 因子缺口: {h}")

        elif atype == "flag_sparse_positives":
            tgt = action.get("target", "")
            pr = action.get("pos_rate", 0)
            logger.info(f"    ⚠️ {tgt} 正样本稀少 ({pr:.3%}): {action.get('suggestion', '')}")

        # ── LLM 新因子建议 (记录, 不直接修改 TrainConfig) ──
        elif atype == "llm_new_factors":
            factors = action.get("factors", [])
            logger.info(f"    🧠 LLM 新因子建议: {len(factors)} 个")
            for f in factors[:3]:
                logger.info(f"      - {f.get('name', '?')}: {f.get('description', '')[:60]}")

    if updates:
        return replace(cfg, **updates)
    return cfg


def _apply_llm_final_filter(
    predictions: pd.DataFrame,
    snapshot: pd.DataFrame,
    scan_date: str,
) -> pd.DataFrame:
    """方案 B: LLM 终筛 — 对 Agent 3 候选做定性过滤。"""
    try:
        from .llm_advisor import run_final_filter
    except ImportError:
        return predictions

    logger.info("── 🧠 LLM 终筛 ──")
    result = run_final_filter(predictions, snapshot, scan_date)

    approved = set(result.get("approved", []))
    if not approved:
        logger.warning("  LLM 终筛无结果, 保留全部候选")
        return predictions

    n_before = len(predictions)
    filtered = predictions[predictions["symbol"].isin(approved)].copy()

    # 添加 LLM 风险标注
    reasons = result.get("reasons", {})
    if reasons:
        filtered["llm_note"] = filtered["symbol"].map(
            lambda s: reasons.get(s, "")
        )

    n_after = len(filtered)
    logger.info(f"  终筛结果: {n_before} → {n_after} 只 (剔除 {n_before - n_after} 只)")

    # 输出风险提示
    for w in result.get("risk_warnings", [])[:3]:
        logger.info(f"  ⚠️ {w}")

    # 输出被剔除的
    rejected = result.get("rejected", [])
    for sym in rejected[:5]:
        reason = reasons.get(sym, "")
        logger.info(f"  ❌ {sym}: {reason[:60]}")

    return filtered


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
            preds = run_scan(sd, cfg, skip_monitor=True)
            if not preds.empty:
                preds["scan_date"] = sd
                all_predictions.append(preds.head(top_n))  # Top N A 级
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
