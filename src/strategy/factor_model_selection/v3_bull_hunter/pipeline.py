"""
Bull Hunter v3 — Pipeline 编排器

v4 重构: 三种运行模式 + tracker 跟踪 + 双回路反馈。
v5 扩展: 新增 live 模式 (Agent 5/6/7 持仓管理闭环)。
v6 简化: 3 层架构 — 选股层(1+2+3) → 交易层(5+6) → 监控层(统一复盘)。
  - 每日循环仅包含选股+交易, 不含监控评估
  - 监控评估统一为周期性复盘 (每 N 天), 同时评选股和交易质量

模式:
  1. daily  — 每日轻量预测: Agent 1 + Agent 3 (复用 latest 模型), 记录到 tracker
  2. train  — 周训/事件驱动训练: Agent 1 + Agent 2, 更新 latest 模型
  3. review — 统一复盘: 选股质量 + 交易质量评估, 可能触发重训
  4. scan   — 完整单日: train + daily (兼容旧接口)
  5. backtest — 滚动回测 + 实际收益验证
  6. live   — 每日 live 循环: 选股(1→3) → 交易(6→5), 无日内监控
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, replace

import pandas as pd

from .agent1_factor import run_factor_generation
from .agent2_train import TrainConfig, run_training, needs_training, get_latest_model_dir, get_model_for_date, _build_calendar
from .agent3_predict import PredictConfig, run_prediction
from .agent4_monitor import run_monitor
from .agent5_portfolio import PortfolioConfig, run_portfolio_decisions
from .agent6_exit_signal import ExitSignalConfig, run_exit_signal, train_exit_model, auto_adjust_exit_weights
from .agent8_buy_signal import BuySignalConfig, run_buy_signal, train_buy_model, auto_adjust_buy_weights
from .agent9_industry_leader import (
    IndustryLeaderConfig,
    run_industry_leader_signal,
    merge_with_model_predictions,
)
from .portfolio import Portfolio
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
    portfolio_cfg: PortfolioConfig | None = None
    exit_cfg: ExitSignalConfig | None = None
    buy_signal_cfg: BuySignalConfig | None = None
    # ── 市场过滤 (建议1) ──
    # date(YYYYMMDD) → trend ("STRONG_UP"/"UP"/"NEUTRAL"/"DOWN"/"STRONG_DOWN")
    market_state_lookup: dict | None = None
    # 在以下市场状态下暂停买入
    market_no_buy_states: tuple = ("STRONG_DOWN",)
    # 在以下市场状态下收紧止损
    market_tighten_stop_states: tuple = ("DOWN", "STRONG_DOWN")
    market_tightened_stop_pct: float = -0.08  # 收紧后的止损线
    # ── 弱模型门槛 (建议2) ──
    min_model_auc: float = 0.55   # 200pct 模型 AUC 低于此值, 跳过买入
    # ── V11-A: 行业趋势+龙头独立信号通道 ──
    enable_industry_leader_channel: bool = True
    industry_leader_cfg: "IndustryLeaderConfig | None" = None
    industry_leader_max_total: int = 8   # 模型 + 龙头 合并后最多候选数


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
        basic_path=cfg.basic_path,
    )
    if not train_results:
        logger.error("Agent 2 训练失败")
        return {}

    logger.info(f"=== TRAIN 完成: {len(train_results)} 个模型 ===")
    return train_results


# ──────────────────────────────────────────────────────────
#  Mode 3: 复盘 (统一复盘, 可触发重训)
# ──────────────────────────────────────────────────────────

def run_review(
    scan_date: str,
    cfg: PipelineConfig | None = None,
) -> dict:
    """
    单日统一复盘 — 选股质量 + 交易质量统一评分, 可能触发 Agent 2 重训。

    v6 简化: 直接调用 run_unified_review (不再独立维护双回路逻辑)。

    流程:
      1. Agent 1: 因子快照
      2. Agent 3: 当日预测 (latest 模型)
      3. run_unified_review: 选股 + 交易统一评分
      4. 若 retrain_required: 应用 tuning_directives → Agent 2 重训 → Agent 3 验证
    """
    cfg = cfg or PipelineConfig()
    train_cfg = cfg.train_cfg or TrainConfig()
    predict_cfg = cfg.predict_cfg or PredictConfig()

    logger.info(f"=== Bull Hunter v6 REVIEW: {scan_date} ===")

    # ── Agent 1: 因子快照 ──
    snapshot = run_factor_generation(
        cache_dir=cfg.cache_dir,
        data_dir=cfg.data_dir,
        scan_date=scan_date,
        basic_path=cfg.basic_path,
    )
    if snapshot.empty:
        logger.error("Agent 1 返回空快照, 终止")
        return {}

    # ── Agent 3: 当前预测 ──
    model_dir = get_latest_model_dir(cfg.cache_dir)
    current_predictions = pd.DataFrame()
    if model_dir:
        current_predictions = run_prediction(
            factor_snapshot=snapshot,
            model_dir=model_dir,
            scan_date=scan_date,
            cfg=predict_cfg,
        )

    # ── 统一复盘 ──
    review = run_unified_review(
        scan_date=scan_date,
        cfg=cfg,
        factor_snapshot=snapshot,
        predictions=current_predictions,
        bt_results_dir=cfg.results_dir,
    )

    for s in review.get("suggestions", []):
        logger.info(f"  💡 {s}")

    # ── 应用重训决策 ──
    if review.get("retrain_required"):
        tuning = review.get("tuning_directives", {})
        logger.info(f"  🔄 触发重训: {tuning.get('reason', '')}")

        new_train_cfg = _apply_tuning_directives(train_cfg, tuning)
        new_train_cfg = replace(
            new_train_cfg,
            trigger="unified_review",
            trigger_reason=tuning.get("reason", ""),
            force_retrain=True,
        )

        logger.info("── Agent 2: 重训 (统一复盘反馈) ──")
        retrain_results = run_training(
            cache_dir=cfg.cache_dir,
            scan_date=scan_date,
            cfg=new_train_cfg,
            basic_path=cfg.basic_path,
        )

        if retrain_results:
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

    # 保存复盘报告
    out_dir = os.path.join(cfg.results_dir, scan_date)
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "review_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(review, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"=== REVIEW 完成: unified_score={review.get('unified_score', 0):.2f} ===")
    return review


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
        basic_path=cfg.basic_path,
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
    将调参指令应用到 TrainConfig (v6 简化: 仅 L1 调参)。

    支持的指令类型:
      - adjust_threshold: 调整预测阈值 (Agent 3 层面, 此处仅 log)
      - adjust_pos_weight: 调整 max_scale_pos_weight
      - adjust_estimators: 调整 n_estimators
      - increase_lookback: 增加 lookback_months
    """
    actions = tuning.get("actions", [])
    if not actions:
        return cfg

    logger.info(f"    调参: {len(actions)} 个 L1 动作")

    updates = {}
    for action in actions:
        atype = action.get("type", "")

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

    if updates:
        return replace(cfg, **updates)
    return cfg



# ──────────────────────────────────────────────────────────
#  Mode 6: Live 模式 (持仓管理闭环)
# ──────────────────────────────────────────────────────────

def run_live(
    scan_date: str,
    cfg: PipelineConfig | None = None,
    preloaded: dict | None = None,
    skip_train: bool = False,
    override_model_dir: str | None = None,
) -> dict:
    """
    每日 Live 循环 (v6 简化): 选股层(1→3) → 交易层(6→5)。

    流程:
      1. Agent 1: 因子快照
      2. Agent 2 (按需): 检查是否需要周训
      3. Agent 3: Top 5 候选
      4. Agent 8: 买入时机评估 (过滤/排序候选)
      5. Agent 6: 对已持仓计算卖出权重
      6. Agent 5: 先卖后买 (消费 Agent 8 过滤后候选 + Agent 6 权重)

    监控评估不在日内执行, 统一由 run_unified_review() 周期性处理。

    Args:
        override_model_dir: 强制指定模型目录 (回测时用于模型轮换, 避免前瞻偏差)

    Returns:
        {
            "predictions": DataFrame,
            "factor_snapshot": DataFrame,
            "portfolio_result": {...},
        }
    """
    cfg = cfg or PipelineConfig()
    predict_cfg = cfg.predict_cfg or PredictConfig()
    portfolio_cfg = cfg.portfolio_cfg or PortfolioConfig()
    exit_cfg = cfg.exit_cfg or ExitSignalConfig()
    buy_signal_cfg = cfg.buy_signal_cfg or BuySignalConfig()

    logger.info(f"=== Bull Hunter v4 LIVE: {scan_date} ===")

    # 构建日历 + 基础组件
    calendar = _build_calendar(cfg.cache_dir)
    portfolio_dir = os.path.join(cfg.results_dir, "portfolio")
    portfolio = Portfolio(portfolio_dir)

    # ── Agent 2 (按需): 周训 ──
    if not skip_train and needs_training(cfg.cache_dir, scan_date):
        logger.info("── Agent 2: 触发周训 ──")
        run_train(scan_date, cfg, trigger="weekly")

    # ── 获取模型 ──
    if override_model_dir:
        model_dir = override_model_dir
    else:
        model_dir = get_latest_model_dir(cfg.cache_dir)
    if model_dir is None:
        logger.error("无可用模型 (latest 不存在), 请先运行 --train")
        return {}
    logger.info(f"  使用模型: {os.path.basename(model_dir)}")

    # ── 弱模型闸门 (建议2): AUC 太低则不允许新买入 ──
    weak_model = False
    try:
        meta_path = os.path.join(model_dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                _meta = json.load(f)
            _auc = _meta.get("200pct", {}).get("val_auc", 0.0)
            if _auc < cfg.min_model_auc:
                weak_model = True
                logger.info(f"  ⚠️ 弱模型闸门: AUC={_auc:.3f} < {cfg.min_model_auc:.2f}, 暂停新买入")
    except Exception as _e:
        logger.warning(f"  读取 meta.json 失败: {_e}")

    # ── 大盘过滤 (建议1): 查询当日市场状态 ──
    market_state = None
    if cfg.market_state_lookup is not None:
        market_state = cfg.market_state_lookup.get(scan_date)
        if market_state:
            logger.info(f"  📊 大盘状态: {market_state}")

    no_buy_today = weak_model
    if market_state and market_state in cfg.market_no_buy_states:
        no_buy_today = True
        logger.info(f"  🛑 大盘过滤: {market_state} → 暂停新买入")

    # 收紧止损 (DOWN/STRONG_DOWN 时)
    if market_state and market_state in cfg.market_tighten_stop_states:
        portfolio_cfg = replace(portfolio_cfg, stop_loss_pct=cfg.market_tightened_stop_pct)
        logger.info(f"  🛡️ 大盘过滤: {market_state} → 收紧止损 {cfg.market_tightened_stop_pct:+.0%}")

    # ── Agent 1: 因子快照 ──
    logger.info("── Agent 1: 因子快照 ──")
    snapshot = run_factor_generation(
        cache_dir=cfg.cache_dir,
        data_dir=cfg.data_dir,
        scan_date=scan_date,
        basic_path=cfg.basic_path,
        preloaded=preloaded,
    )
    if snapshot.empty:
        logger.error("Agent 1 返回空快照, 终止")
        return {}
    logger.info(f"  → {len(snapshot)} 只股票")

    # ── Agent 3: Top 5 候选 ──
    logger.info("── Agent 3: Top 5 预测 ──")
    tracker = PredictionTracker(os.path.join(cfg.results_dir, "tracking"))
    recent = tracker.get_recent_predictions(scan_date, calendar, predict_cfg.dedup_days)

    predictions = run_prediction(
        factor_snapshot=snapshot,
        model_dir=model_dir,
        scan_date=scan_date,
        cfg=predict_cfg,
        recent_predictions=recent,
    )

    if not predictions.empty:
        predictions = predictions.copy()
        predictions["source"] = "model"
        model_date = os.path.basename(os.path.realpath(model_dir))
        tracker.record_predictions(predictions, scan_date, cfg.data_dir, calendar, model_date)
    logger.info(f"  → {len(predictions)} 只 A 级候选")

    # ── Agent 9: 行业趋势 + 龙头独立通道 (V11-A) ──
    if cfg.enable_industry_leader_channel:
        logger.info("── Agent 9: 行业龙头独立通道 ──")
        il_cfg = cfg.industry_leader_cfg or IndustryLeaderConfig()
        leader_preds = run_industry_leader_signal(
            factor_snapshot=snapshot,
            scan_date=scan_date,
            cfg=il_cfg,
            recent_predictions=recent,
        )
        if not leader_preds.empty:
            # 合并: 模型 + 龙头, symbol 去重 (优先模型通道)
            n_model = len(predictions)
            predictions = merge_with_model_predictions(
                model_preds=predictions,
                leader_preds=leader_preds,
                max_total=cfg.industry_leader_max_total,
            )
            n_leaders_added = len(predictions) - n_model
            logger.info(f"  合并: 模型 {n_model} + 龙头 {n_leaders_added} = {len(predictions)} 只")

    # ── Agent 8: 买入时机评估 ──
    logger.info("── Agent 8: 买入时机评估 ──")
    if not predictions.empty:
        predictions = run_buy_signal(
            candidates=predictions,
            factor_snapshot=snapshot,
            current_date=scan_date,
            data_dir=cfg.data_dir,
            calendar=calendar,
            portfolio_dir=portfolio_dir,
            cfg=buy_signal_cfg,
        )
        # 过滤低质量买点
        n_before = len(predictions)
        predictions = predictions[predictions["buy_quality"] >= buy_signal_cfg.min_buy_quality].reset_index(drop=True)
        if n_before > len(predictions):
            logger.info(f"  Agent 8 过滤: {n_before} → {len(predictions)} 只 (min_quality={buy_signal_cfg.min_buy_quality})")
        # P1: prob_200 硬门槛过滤 — 仅对模型通道生效, 行业龙头通道豁免
        if "prob_200" in predictions.columns:
            is_model = predictions.get("source", pd.Series("model", index=predictions.index)) == "model"
            prob_ok = predictions["prob_200"].fillna(0) >= buy_signal_cfg.min_prob_200
            keep = (~is_model) | prob_ok
            n_before_p200 = len(predictions)
            predictions = predictions[keep].reset_index(drop=True)
            if n_before_p200 > len(predictions):
                logger.info(f"  prob_200 过滤 (仅模型通道): {n_before_p200} → {len(predictions)} 只 "
                            f"(min_prob_200={buy_signal_cfg.min_prob_200})")

    # ── Agent 6: 卖出因子评估 ──
    logger.info("── Agent 6: 卖出因子评估 ──")
    positions = portfolio.get_positions()
    sell_weights = run_exit_signal(
        positions=positions,
        factor_snapshot=snapshot,
        current_date=scan_date,
        data_dir=cfg.data_dir,
        calendar=calendar,
        portfolio_dir=portfolio_dir,
        cfg=exit_cfg,
    )

    # ── Agent 5: 买卖执行 ──
    logger.info("── Agent 5: 组合决策 ──")
    # 弱模型 / 大盘过滤: 仅清空模型通道候选, 行业龙头通道豁免
    if no_buy_today and not predictions.empty:
        if "source" in predictions.columns:
            n_before_gate = len(predictions)
            predictions = predictions[predictions["source"] == "industry_leader"].reset_index(drop=True)
            n_cleared = n_before_gate - len(predictions)
            if n_cleared > 0:
                logger.info(f"  🚫 清空 {n_cleared} 只模型通道候选 (弱模型或大盘过滤), 保留 {len(predictions)} 只龙头通道")
        else:
            logger.info(f"  🚫 清空 {len(predictions)} 只买入候选 (弱模型或大盘过滤)")
            predictions = predictions.iloc[0:0]
    portfolio_result = run_portfolio_decisions(
        candidates=predictions,
        sell_weights=sell_weights,
        portfolio=portfolio,
        current_date=scan_date,
        data_dir=cfg.data_dir,
        calendar=calendar,
        cfg=portfolio_cfg,
    )

    # Tracker 维护
    tracker.update_prices(scan_date, cfg.data_dir, calendar)
    tracker.evaluate_expired(scan_date, calendar)

    logger.info(f"=== LIVE 完成: 买入 {len(portfolio_result.get('buys', []))} 只, "
                f"卖出 {len(portfolio_result.get('sells', []))} 只, "
                f"持有 {len(portfolio_result.get('hold', []))} 只 ===")

    return {
        "predictions": predictions,
        "factor_snapshot": snapshot,
        "portfolio_result": portfolio_result,
    }


# ──────────────────────────────────────────────────────────
#  统一周期性复盘 (合并原 Agent 4 + Agent 7)
# ──────────────────────────────────────────────────────────

def run_unified_review(
    scan_date: str,
    cfg: PipelineConfig,
    factor_snapshot: pd.DataFrame,
    predictions: pd.DataFrame | None,
    bt_results_dir: str,
) -> dict:
    """
    统一周期性复盘: 同时评估选股质量和交易质量, 生成统一健康评分。

    合并逻辑:
      1. 选股评估 (原 Agent 4): miss_rate + 跟踪胜率
      2. 交易评估 (原 Agent 7): 买入质量 + 卖出时机 + 信号有效性
      3. 统一健康评分 = 加权(选股健康, 交易健康)
      4. 是否触发重训: 基于统一评分决定
      5. 交易参数调整: Agent 5/6 参数微调
      6. 仅 L1 调参 (不换因子/不换模型)

    Returns:
        {
            "selection_health": {...},   # 选股评估
            "trading_health": {...},     # 交易评估
            "unified_score": float,      # 0~1, 越高越健康
            "retrain_required": bool,
            "tuning_directives": {...},  # 给 Agent 2 的调参指令
            "trading_directives": {...}, # 给 Agent 5/6 的参数调整
        }
    """
    from .agent7_supervisor import (
        _evaluate_performance, _evaluate_exit_signals,
        _generate_agent5_directives, _generate_agent6_directives,
        _check_intervention_needed,
    )

    portfolio_dir = os.path.join(bt_results_dir, "portfolio")
    calendar = _build_calendar(cfg.cache_dir)

    report = {
        "scan_date": scan_date,
        "selection_health": {},
        "trading_health": {},
        "unified_score": 1.0,
        "retrain_required": False,
        "tuning_directives": {},
        "trading_directives": {},
        "suggestions": [],
    }

    # ── 1. 选股评估 (原 Agent 4) ──
    tracker = PredictionTracker(os.path.join(bt_results_dir, "tracking"))
    tracking_summary = tracker.get_tracking_summary()
    expired_evals = tracker.get_expired_for_review()

    selection_result = run_monitor(
        cache_dir=cfg.cache_dir,
        data_dir=cfg.data_dir,
        scan_date=scan_date,
        factor_snapshot=factor_snapshot,
        current_predictions=predictions if predictions is not None else pd.DataFrame(),
        basic_path=cfg.basic_path,
        tracking_summary=tracking_summary,
        expired_evals=expired_evals,
    )
    report["selection_health"] = selection_result

    # 选股评分: miss_rate 越低越好, 跟踪胜率越高越好
    miss_rate = selection_result.get("miss_diagnosis", {}).get("miss_rate", 1.0)
    tracking_fb = selection_result.get("tracking_feedback", {})
    tracking_win = tracking_fb.get("win_rate", 0.5)
    selection_score = (1.0 - miss_rate) * 0.4 + tracking_win * 0.6

    # ── 2. 交易评估 (原 Agent 7) ──
    trading_eval = _evaluate_performance(portfolio_dir, cfg.data_dir, calendar, scan_date)
    signal_eval = _evaluate_exit_signals(portfolio_dir, cfg.data_dir, calendar, scan_date)
    trading_eval["signal_quality"] = signal_eval
    report["trading_health"] = trading_eval

    # 交易评分: 胜率 + 买入质量
    trade_win_rate = trading_eval.get("win_rate", 0.5)
    buy_gain_10d = trading_eval.get("buy_quality", {}).get("avg_gain_10d", 0)
    trading_score = trade_win_rate * 0.6 + max(0, min(1, buy_gain_10d + 0.5)) * 0.4

    # ── 3. 统一健康评分 ──
    unified_score = selection_score * 0.5 + trading_score * 0.5
    report["unified_score"] = round(unified_score, 4)

    # ── 4. 是否触发重训 (基于统一评分) ──
    tuning = selection_result.get("tuning_directives", {})
    needs_retrain = tuning.get("retrain_required", False)

    # 额外条件: 交易胜率持续低
    n_sells = trading_eval.get("n_sells", 0)
    if n_sells >= 5 and trade_win_rate < 0.30:
        needs_retrain = True
        report["suggestions"].append(f"交易胜率仅 {trade_win_rate:.0%}, 触发重训")

    report["retrain_required"] = needs_retrain
    report["tuning_directives"] = tuning

    # ── 5. 交易参数调整 (Agent 5/6 指令; 因子权重交由 auto_adjust_*_weights) ──
    needs_intervention, issues = _check_intervention_needed(trading_eval)
    if needs_intervention:
        portfolio_cfg = cfg.portfolio_cfg or PortfolioConfig()
        exit_cfg = cfg.exit_cfg or ExitSignalConfig()
        a5_dir = _generate_agent5_directives(trading_eval, issues, portfolio_cfg)
        a6_dir = _generate_agent6_directives(trading_eval, signal_eval, issues, exit_cfg, portfolio_dir)
        report["trading_directives"] = {
            "agent5": a5_dir,
            "agent6": a6_dir,
            "issues": issues,
        }
        for issue in issues:
            report["suggestions"].append(f"[交易] {issue}")

    # 合并选股建议
    for s in selection_result.get("suggestions", [])[:3]:
        report["suggestions"].append(f"[选股] {s}")

    logger.info(f"  统一复盘: 选股={selection_score:.2f} 交易={trading_score:.2f} "
                f"综合={unified_score:.2f} 重训={'是' if needs_retrain else '否'}")

    return report


def run_live_backtest(
    start_date: str,
    end_date: str,
    cfg: PipelineConfig | None = None,
    retrain_interval: int = 60,
    enable_exit_train: bool = True,
    monitor_interval: int = 20,
) -> dict:
    """
    完整整合回测 (v6 简化): 选股 + 交易 + 统一周期性复盘。

    3 层架构:
      - 选股层 (每日): Agent 1→3 因子快照+模型预测
      - 交易层 (每日): Agent 6→5 卖出信号+组合管理
      - 监控层 (每 N 天): 统一复盘 (选股质量+交易质量), 可能触发重训/调参

    Args:
        retrain_interval: 每隔多少交易日触发 Agent 2 周期重训 (0=不重训, 只用已有模型)
        enable_exit_train: 是否在交易样本充足时训练 Agent 6 退出模型
        monitor_interval: 统一复盘间隔 (交易日, 0=禁用, 默认 20)

    Returns:
        {"trades": DataFrame, "daily_pnl": DataFrame, "stats": dict}
    """
    cfg = cfg or PipelineConfig()
    calendar = _build_calendar(cfg.cache_dir)

    valid_dates = [d for d in calendar if start_date <= d <= end_date]
    if not valid_dates:
        logger.error(f"日期范围 {start_date}~{end_date} 内无交易日")
        return {}

    logger.info(f"=== 完整整合回测: {start_date} ~ {end_date}, {len(valid_dates)} 个交易日 ===")
    logger.info(f"  模型轮换: ON (自动选择 scan_date 前可用的最新模型)")
    logger.info(f"  Agent 2 重训间隔: {retrain_interval}天 ({'ON' if retrain_interval > 0 else 'OFF'})")
    logger.info(f"  统一复盘间隔: {monitor_interval}天 ({'ON' if monitor_interval > 0 else 'OFF'})")
    logger.info(f"  Agent 6 退出模型训练: {'ON' if enable_exit_train else 'OFF'}")

    # ── 预加载因子缓存到内存 ──
    logger.info("预加载因子缓存到内存...")
    from .agent1_factor import preload_factor_cache
    preloaded = preload_factor_cache(cfg.cache_dir)
    logger.info("预加载完成")

    # 创建独立的回测目录
    bt_results_dir = os.path.join(
        cfg.results_dir, f"live_backtest_{start_date}_{end_date}"
    )
    # 清理旧回测 (如果存在)
    portfolio_dir = os.path.join(bt_results_dir, "portfolio")
    if os.path.exists(portfolio_dir):
        shutil.rmtree(portfolio_dir)

    bt_cfg = PipelineConfig(
        cache_dir=cfg.cache_dir,
        data_dir=cfg.data_dir,
        basic_path=cfg.basic_path,
        results_dir=bt_results_dir,
        train_cfg=cfg.train_cfg,
        predict_cfg=cfg.predict_cfg,
        portfolio_cfg=cfg.portfolio_cfg or PortfolioConfig(),
        exit_cfg=cfg.exit_cfg or ExitSignalConfig(),
        buy_signal_cfg=cfg.buy_signal_cfg or BuySignalConfig(),
        market_state_lookup=cfg.market_state_lookup,
        market_no_buy_states=cfg.market_no_buy_states,
        market_tighten_stop_states=cfg.market_tighten_stop_states,
        market_tightened_stop_pct=cfg.market_tightened_stop_pct,
        min_model_auc=cfg.min_model_auc,
    )

    # ── 跟踪模型轮换和重训 ──
    last_model_date = ""
    last_retrain_idx = -retrain_interval  # 确保第一次检查时可以触发
    last_monitor_idx = -monitor_interval  # Agent 4 监控计时
    monitor_retrains = 0  # Agent 4 触发的重训次数
    exit_model_trained = False
    latest_live_result = {}  # 保存最近一次 run_live 结果 (供 Agent 4 使用)
    cold_started = False  # P0: cold-start flag, 训练成功后不再尝试
    cold_start_last_attempt = -20  # 上次尝试的 index, 初始值确保首日可触发

    for i, date in enumerate(valid_dates):
        if (i + 1) % 20 == 0 or i == 0:
            logger.info(f"[{i+1}/{len(valid_dates)}] {date}")

        try:
            # ── 模型轮换: 选择当前日期可用的最新模型 ──
            model_dir = get_model_for_date(cfg.cache_dir, date)
            if model_dir is None:
                # ── P0: cold-start 训练 (无模型时周期性尝试训练) ──
                if not cold_started and (i - cold_start_last_attempt) >= 20:
                    cold_start_last_attempt = i
                    logger.info(f"  🚀 Cold-start: {date} 无可用模型, 自动触发初始训练")
                    try:
                        run_train(date, cfg, force=True, trigger="cold_start_backtest")
                        model_dir = get_model_for_date(cfg.cache_dir, date)
                        if model_dir is None:
                            logger.warning(f"  Cold-start 训练完成但仍无可用模型, 跳过 {date}")
                            continue
                        cold_started = True  # 成功才标记, 失败则下次 20 天后重试
                    except Exception as e:
                        logger.warning(f"  Cold-start 训练失败: {e}, 跳过")
                        continue
                else:
                    logger.warning(f"  {date}: 无可用模型, 跳过")
                    continue
            cur_model_date = os.path.basename(model_dir)

            # 日志: 模型切换
            if cur_model_date != last_model_date:
                logger.info(f"  📊 模型轮换: {last_model_date or '无'} → {cur_model_date}")
                last_model_date = cur_model_date

            # ── Agent 2 周期重训 (按需) ──
            if retrain_interval > 0 and (i - last_retrain_idx) >= retrain_interval:
                logger.info(f"  🔄 触发 Agent 2 重训 (间隔 {i - last_retrain_idx}天)")
                try:
                    run_train(date, cfg, force=True, trigger="periodic_backtest")
                    # 重训后刷新模型
                    new_model = get_model_for_date(cfg.cache_dir, date)
                    if new_model:
                        model_dir = new_model
                        last_model_date = os.path.basename(model_dir)
                        logger.info(f"  📊 重训完成, 模型更新: {last_model_date}")
                except Exception as e:
                    logger.warning(f"  Agent 2 重训失败: {e}")
                last_retrain_idx = i

            # ── Agent 6 退出模型训练 (交易样本充足后, 每 30 天检查一次) ──
            if enable_exit_train and not exit_model_trained and i % 30 == 0 and i > 0:
                trades_path = os.path.join(portfolio_dir, "trades.csv")
                if os.path.exists(trades_path):
                    trades_df = pd.read_csv(trades_path, dtype={"trade_date": str})
                    n_sells = (trades_df["direction"] == "sell").sum()
                    if n_sells >= 8:  # 回测中降低门槛
                        logger.info(f"  🧠 Agent 6 退出模型训练 ({n_sells} 条卖出记录)")
                        try:
                            result = train_exit_model(
                                portfolio_dir, cfg.data_dir, calendar, bt_cfg.exit_cfg,
                                min_samples=8,
                            )
                            if result.get("trained"):
                                exit_model_trained = True
                                bt_cfg.exit_cfg.use_model = True
                                logger.info(f"  ✅ Agent 6 模型训练成功: AUC={result.get('auc', 0):.3f}")
                        except Exception as e:
                            logger.warning(f"  Agent 6 训练失败: {e}")

            # ── Agent 8 买入模型训练 (交易样本充足后, 每 30 天检查一次) ──
            if enable_exit_train and i % 30 == 0 and i > 0:
                trades_path = os.path.join(portfolio_dir, "trades.csv")
                if os.path.exists(trades_path):
                    trades_df = pd.read_csv(trades_path, dtype={"trade_date": str})
                    n_buys = (trades_df["direction"] == "buy").sum()
                    if n_buys >= 8:
                        try:
                            buy_cfg = bt_cfg.buy_signal_cfg or BuySignalConfig()
                            result = train_buy_model(
                                portfolio_dir, cfg.data_dir, calendar, buy_cfg,
                                min_samples=8,
                            )
                            if result.get("trained"):
                                buy_cfg.use_model = True
                                bt_cfg = replace(bt_cfg, buy_signal_cfg=buy_cfg)
                                logger.info(f"  ✅ Agent 8 买入模型训练成功: AUC={result.get('auc', 0):.3f}")
                        except Exception as e:
                            logger.warning(f"  Agent 8 训练失败: {e}")

            # ── P2: 首日批量建仓 — 空仓时增大 top_n 以填充仓位 ──
            _portfolio_tmp = Portfolio(portfolio_dir)
            n_positions = _portfolio_tmp.get_position_count()
            max_pos = (bt_cfg.portfolio_cfg or PortfolioConfig()).max_positions
            if n_positions == 0 and max_pos > (bt_cfg.predict_cfg or PredictConfig()).top_n:
                batch_top_n = max_pos * 2  # 多取候选, 允许 Agent 8 过滤后仍有足够建仓
                orig_predict_cfg = bt_cfg.predict_cfg or PredictConfig()
                bt_cfg = replace(bt_cfg, predict_cfg=replace(orig_predict_cfg, top_n=batch_top_n))
                logger.info(f"  📦 空仓批量建仓: top_n 临时提升 {orig_predict_cfg.top_n} → {batch_top_n}")
                batch_mode = True
            else:
                batch_mode = False

            # ── 每日 live 循环 (用轮换后的模型) ──
            latest_live_result = run_live(
                date, bt_cfg,
                preloaded=preloaded,
                skip_train=True,  # Agent 2 重训在上面单独处理
                override_model_dir=model_dir,
            )

            # 恢复 top_n
            if batch_mode:
                bt_cfg = replace(bt_cfg, predict_cfg=orig_predict_cfg)

            # ── 统一周期性复盘 (选股+交易质量评估) ──
            if monitor_interval > 0 and (i - last_monitor_idx) >= monitor_interval and i > 0:
                last_monitor_idx = i

                # ── 强制重训: 模型长期无预测 (空转保护) ──
                trades_path = os.path.join(portfolio_dir, "trades.csv")
                n_total_buys = 0
                if os.path.exists(trades_path):
                    _tdf = pd.read_csv(trades_path, dtype={"trade_date": str})
                    n_total_buys = (_tdf["direction"] == "buy").sum()
                days_with_model = i - cold_start_last_attempt  # 有模型后已过天数
                if cold_started and n_total_buys == 0 and days_with_model >= 40:
                    logger.info(f"  🔄 空转保护: 有模型 {days_with_model} 天仍无买入, 强制重训 (date={date})")
                    try:
                        run_train(date, cfg, force=True,
                                  trigger="stale_model_retrain",
                                  trigger_reason=f"有模型{days_with_model}天无买入")
                        new_model = get_model_for_date(cfg.cache_dir, date)
                        if new_model:
                            model_dir = new_model
                            last_model_date = os.path.basename(model_dir)
                            monitor_retrains += 1
                            cold_start_last_attempt = i  # 重置计时, 避免连续触发
                            logger.info(f"    ✅ 空转重训完成: {last_model_date}")
                    except Exception as e:
                        logger.warning(f"    空转重训失败: {e}")
                    continue  # 重训后跳过本次复盘

                try:
                    snapshot = latest_live_result.get("factor_snapshot")
                    predictions = latest_live_result.get("predictions")
                    if snapshot is not None and not snapshot.empty:
                        logger.info(f"  🔍 统一复盘 (第 {i} 天)")
                        review = run_unified_review(
                            scan_date=date,
                            cfg=cfg,
                            factor_snapshot=snapshot,
                            predictions=predictions,
                            bt_results_dir=bt_results_dir,
                        )

                        # 输出建议
                        for s in review.get("suggestions", [])[:5]:
                            logger.info(f"    💡 {s}")

                        # 应用交易参数调整 (Agent 5/6)
                        td = review.get("trading_directives", {})
                        a5_dir = td.get("agent5", {})
                        a6_dir = td.get("agent6", {})
                        if a5_dir.get("actions") or a6_dir.get("actions"):
                            from .agent7_supervisor import apply_directives
                            new_p_cfg, new_e_cfg = apply_directives(
                                bt_cfg.portfolio_cfg or PortfolioConfig(),
                                bt_cfg.exit_cfg or ExitSignalConfig(),
                                a5_dir, a6_dir, portfolio_dir,
                            )
                            bt_cfg = replace(bt_cfg, portfolio_cfg=new_p_cfg, exit_cfg=new_e_cfg)

                        # 触发 Agent 6 退出模型重训 (如果信号质量差)
                        sig_q = review.get("trading_health", {}).get("signal_quality", {})
                        if sig_q.get("correlation", 0) > 0.1 and sig_q.get("n_samples", 0) >= 20:
                            try:
                                result = train_exit_model(
                                    portfolio_dir, cfg.data_dir, calendar, bt_cfg.exit_cfg,
                                    min_samples=8,
                                )
                                if result.get("trained"):
                                    exit_model_trained = True
                                    logger.info(f"    ✅ Agent 6 重训: AUC={result.get('auc', 0):.3f}")
                            except Exception:
                                pass

                        # Agent 6 卖出因子权重自动调整
                        try:
                            adj_result = auto_adjust_exit_weights(
                                portfolio_dir, cfg.data_dir, calendar, bt_cfg.exit_cfg,
                            )
                            if adj_result.get("adjusted"):
                                changes = adj_result.get("changes", {})
                                if changes:
                                    top3 = sorted(changes.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                                    chg_str = ", ".join(f"{k}:{v:+.3f}" for k, v in top3)
                                    logger.info(f"    ⚖️ Agent 6 权重调整: {adj_result['n_samples']}样本, {chg_str}")
                        except Exception as e:
                            logger.warning(f"    Agent 6 权重调整失败: {e}")

                        # Agent 8 买入因子权重自动调整
                        try:
                            buy_cfg = bt_cfg.buy_signal_cfg or BuySignalConfig()
                            adj_result = auto_adjust_buy_weights(
                                portfolio_dir, cfg.data_dir, calendar, buy_cfg,
                            )
                            if adj_result.get("adjusted"):
                                changes = adj_result.get("changes", {})
                                if changes:
                                    top3 = sorted(changes.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                                    chg_str = ", ".join(f"{k}:{v:+.3f}" for k, v in top3)
                                    logger.info(f"    ⚖️ Agent 8 权重调整: {adj_result['n_samples']}样本, {chg_str}")
                        except Exception as e:
                            logger.warning(f"    Agent 8 权重调整失败: {e}")

                        # 检查是否需要选股模型重训
                        if review.get("retrain_required"):
                            tuning = review.get("tuning_directives", {})
                            reason = tuning.get("reason", "") or "统一复盘触发"
                            logger.info(f"    🔄 触发 Agent 2 重训: {reason[:80]}")
                            try:
                                train_cfg = cfg.train_cfg or TrainConfig()
                                new_train_cfg = _apply_tuning_directives(train_cfg, tuning)
                                new_train_cfg = replace(
                                    new_train_cfg,
                                    trigger="unified_review",
                                    trigger_reason=reason,
                                    force_retrain=True,
                                )
                                retrain_results = run_training(
                                    cache_dir=cfg.cache_dir,
                                    scan_date=date,
                                    cfg=new_train_cfg,
                                    basic_path=cfg.basic_path,
                                )
                                if retrain_results:
                                    new_model = get_model_for_date(cfg.cache_dir, date)
                                    if new_model:
                                        model_dir = new_model
                                        last_model_date = os.path.basename(model_dir)
                                        monitor_retrains += 1
                                        logger.info(f"    ✅ 重训完成: {last_model_date} "
                                                    f"(累计 {monitor_retrains} 次)")
                            except Exception as e:
                                logger.warning(f"    重训失败: {e}")

                        # 保存复盘报告
                        review_out = os.path.join(bt_results_dir, "review_reports")
                        os.makedirs(review_out, exist_ok=True)
                        report_path = os.path.join(review_out, f"review_{date}.json")
                        with open(report_path, "w", encoding="utf-8") as f:
                            json.dump(review, f, ensure_ascii=False, indent=2, default=str)
                except Exception as e:
                    logger.warning(f"  统一复盘失败: {e}")
        except Exception as e:
            logger.error(f"  {date} 失败: {e}")
            continue

    # ── 汇总结果 ──
    portfolio = Portfolio(portfolio_dir)

    stats = portfolio.get_trade_stats()
    trades = portfolio.get_trades()

    # 保存汇总
    summary = {
        "start_date": start_date,
        "end_date": end_date,
        "n_trading_days": len(valid_dates),
        "retrain_interval": retrain_interval,
        "monitor_interval": monitor_interval,
        "enable_exit_train": enable_exit_train,
        "exit_model_trained": exit_model_trained,
        "monitor_retrains": monitor_retrains,
        "stats": stats,
    }
    summary_path = os.path.join(bt_results_dir, "live_backtest_summary.json")
    os.makedirs(bt_results_dir, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"\n{'='*60}")
    logger.info(f"完整整合回测完成: {start_date} ~ {end_date}")
    logger.info(f"  交易笔数: {stats.get('n_trades', 0)}")
    logger.info(f"  胜率: {stats.get('win_rate', 0):.1%}")
    logger.info(f"  平均盈亏: {stats.get('avg_pnl_pct', 0):+.1%}")
    logger.info(f"  总盈亏: {stats.get('total_pnl', 0):+.0f}")
    logger.info(f"  统一复盘: {monitor_retrains} 次反馈驱动重训 (每 {monitor_interval} 天)")
    logger.info(f"  Agent 6 模型: {'已训练' if exit_model_trained else '未训练 (样本不足)'}")
    logger.info(f"  结果目录: {bt_results_dir}")
    logger.info(f"{'='*60}")

    return {"trades": trades, "stats": stats, "results_dir": bt_results_dir}


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
