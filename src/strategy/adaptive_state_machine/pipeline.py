"""
Adaptive State Machine — Pipeline 编排

4 Agent 每日循环:
  Agent 1 (Factor Calculator) → 因子截面
  Agent 2 (Weight Learner) → 动态权重 + 阈值
  Agent 3 (State Evaluator) → 状态判定
  Agent 4 (Validator) → 验证历史 → 奖励信号 → Agent 2

支持模式:
  --scan:  单日扫描
  --live:  每日循环 (从 start_date 到 end_date)
  --backtest: 历史回测 (带滚动训练)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import AdaptiveConfig, StockState
from .agent1_factor import FactorCalculator
from .agent2_weight import WeightLearner
from .agent3_state import StateEvaluator
from .agent4_validator import Validator

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════
# 工具: 加载交易日历
# ═════════════════════════════════════════════════════════

def load_trade_cal(data_root: str) -> list[str]:
    """加载交易日历, 返回 YYYYMMDD 列表"""
    cal_path = os.path.join(data_root, "tushare-trade_cal", "trade_cal.csv")
    if not os.path.exists(cal_path):
        logger.warning(f"Trade calendar not found at {cal_path}")
        return []

    df = pd.read_csv(cal_path)
    # trade_cal 格式: exchange,cal_date,is_open,pretrade_date
    if "cal_date" in df.columns and "is_open" in df.columns:
        df = df[df["is_open"] == 1]
        return sorted(df["cal_date"].astype(str).tolist())

    return []


def load_price_matrix(
    daily_dir: str,
    trade_dates: list[str],
) -> pd.DataFrame:
    """
    构建价格矩阵: index=trade_date, columns=symbol, values=close

    用于 Agent 2 (计算 forward returns) 和 Agent 4 (验证)。
    """
    import glob
    price_data = {}

    for fpath in glob.glob(os.path.join(daily_dir, "*.csv")):
        symbol = os.path.basename(fpath).replace(".csv", "")
        try:
            df = pd.read_csv(fpath, usecols=["trade_date", "close"])
            df["trade_date"] = df["trade_date"].astype(str)
            df = df.set_index("trade_date")["close"]
            price_data[symbol] = df
        except Exception:
            continue

    if not price_data:
        return pd.DataFrame()

    price_df = pd.DataFrame(price_data)
    price_df.index.name = "trade_date"
    logger.info(f"Price matrix: {len(price_df)} dates × {len(price_df.columns)} symbols")
    return price_df


# ═════════════════════════════════════════════════════════
# 单日扫描
# ═════════════════════════════════════════════════════════

def run_scan(
    daily_dir: str,
    data_root: str,
    scan_date: str,
    cache_dir: str = "",
    config_dir: str = "",
    output_dir: str = "",
) -> dict:
    """
    执行单次扫描。

    流程:
      1. 加载/初始化 AdaptiveConfig
      2. Agent 1: 计算全市场因子
      3. Agent 2: 更新权重 (如果有历史数据)
      4. Agent 3: 状态判定
      5. Agent 4: 验证历史预测 (如果有价格数据)
      6. 保存结果 + 更新 config
    """
    config_path = os.path.join(config_dir, "adaptive_config.json") if config_dir else ""
    os.makedirs(output_dir, exist_ok=True) if output_dir else None

    # Step 0: 加载配置
    if config_path and os.path.exists(config_path):
        config = AdaptiveConfig.load(config_path)
        logger.info(f"Loaded config v{config.version} from {config_path}")
    else:
        config = AdaptiveConfig()
        config.last_updated = scan_date
        logger.info("Using default config (first run)")

    # Step 1: Agent 1 — 因子计算
    calculator = FactorCalculator(
        daily_dir=daily_dir,
        data_root=data_root,
        cache_dir=cache_dir,
    )
    daily_results = calculator.compute_all(scan_date=scan_date)
    cross_section = calculator.build_cross_section(daily_results)

    if cross_section.empty:
        logger.error("No data from Agent 1, aborting")
        return {}

    # Step 2: Agent 2 — 权重更新 (需要价格数据)
    price_df = load_price_matrix(daily_dir, []) if data_root else pd.DataFrame()
    if not price_df.empty:
        learner = WeightLearner(lookback_days=60, forward_window=10)

        feature_matrix = cross_section.reset_index()
        feature_matrix.columns = [str(c) for c in feature_matrix.columns]
        if "symbol" not in feature_matrix.columns:
            feature_matrix["symbol"] = cross_section.index

        config = learner.update(
            feature_matrix=feature_matrix,
            price_series=price_df,
            config=config,
            scan_date=scan_date,
        )
    else:
        logger.info("Agent 2: No price data available, skipping weight update")

    # Step 3: Agent 3 — 状态判定
    evaluator = StateEvaluator(config=config)
    results = evaluator.evaluate_all(cross_section, daily_results, config)

    # 筛选有意义的信号
    accumulation_signals = evaluator.get_signals_by_state(
        results, StockState.ACCUMULATION, min_confidence=0.3,
    )
    breakout_signals = evaluator.get_signals_by_state(
        results, StockState.BREAKOUT, min_confidence=0.3,
    )
    collapse_signals = evaluator.get_signals_by_state(
        results, StockState.COLLAPSE, min_confidence=0.3,
    )

    logger.info(
        f"Signals: {len(accumulation_signals)} accumulation, "
        f"{len(breakout_signals)} breakout, "
        f"{len(collapse_signals)} collapse"
    )

    # Step 4: Agent 4 — 验证历史 (并入新的预测)
    validator = Validator()
    validator.ingest_predictions(results)

    if not price_df.empty:
        trade_dates = load_trade_cal(data_root)
        if trade_dates:
            rewards = validator.verify(price_df, scan_date, trade_dates)
            # 用验证结果重新更新权重
            if any(v != 0 for v in rewards.values()):
                learner = WeightLearner(lookback_days=60, forward_window=10)
                feature_matrix = cross_section.reset_index()
                config = learner.update(
                    feature_matrix=feature_matrix,
                    price_series=price_df,
                    config=config,
                    scan_date=scan_date,
                    agent4_rewards=rewards,
                )

    # Step 5: 保存结果
    if output_dir:
        # 信号输出
        signal_rows = []
        for r in results:
            if r.state in (StockState.ACCUMULATION, StockState.BREAKOUT,
                           StockState.HOLD, StockState.COLLAPSE):
                signal_rows.append({
                    "symbol": r.symbol,
                    "trade_date": r.trade_date,
                    "state": r.state.value,
                    "confidence": r.confidence,
                    "aq_score": r.aq_score,
                    "bq_score": r.bq_score,
                    "composite_score": r.composite_score,
                    "details": json.dumps(r.details, ensure_ascii=False),
                })

        if signal_rows:
            signal_df = pd.DataFrame(signal_rows)
            output_path = os.path.join(output_dir, f"signals_{scan_date}.csv")
            signal_df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(signal_df)} signals to {output_path}")

        # 保存配置
        if config_path:
            config.last_updated = scan_date
            config.save(config_path)
            logger.info(f"Saved config v{config.version} to {config_path}")

    # Step 6: 打印摘要
    _print_scan_summary(results, scan_date)

    return {
        "scan_date": scan_date,
        "config_version": config.version,
        "total_stocks": len(results),
        "accumulation": len(accumulation_signals),
        "breakout": len(breakout_signals),
        "collapse": len(collapse_signals),
    }


# ═════════════════════════════════════════════════════════
# 回测 (滚动日期)
# ═════════════════════════════════════════════════════════

def run_backtest(
    daily_dir: str,
    data_root: str,
    start_date: str,
    end_date: str,
    interval_days: int = 5,
    cache_dir: str = "",
    config_dir: str = "",
    output_dir: str = "",
) -> pd.DataFrame:
    """
    历史回测: 从 start_date 到 end_date, 每隔 interval_days 天执行一次扫描。

    Returns:
        所有日期的信号汇总 DataFrame
    """
    trade_dates = load_trade_cal(data_root)
    if not trade_dates:
        logger.error("No trade calendar found")
        return pd.DataFrame()

    # 过滤日期范围
    trade_dates = [d for d in trade_dates if start_date <= d <= end_date]
    if not trade_dates:
        logger.error(f"No trading dates in range {start_date} to {end_date}")
        return pd.DataFrame()

    # 每隔 interval_days 取一个日期
    scan_dates = trade_dates[::interval_days]
    logger.info(f"Backtest: {len(scan_dates)} scan dates from {start_date} to {end_date}")

    os.makedirs(output_dir, exist_ok=True)

    # 加载价格矩阵 (一次性)
    price_df = load_price_matrix(daily_dir, trade_dates)

    # 初始化
    config_path = os.path.join(config_dir, "adaptive_config.json") if config_dir else ""
    config = AdaptiveConfig.load(config_path) if config_path and os.path.exists(config_path) else AdaptiveConfig()

    # 验证器 (持久化, 跨日期积累)
    validator = Validator()
    evaluator = StateEvaluator(config=config)

    all_signals = []
    summary_rows = []

    for i, scan_date in enumerate(scan_dates):
        logger.info(f"\n{'='*60}")
        logger.info(f"Backtest: Date {i+1}/{len(scan_dates)} — {scan_date}")

        # Agent 1
        calculator = FactorCalculator(
            daily_dir=daily_dir,
            data_root=data_root,
            cache_dir=cache_dir,
        )
        daily_results = calculator.compute_all(scan_date=scan_date)
        cross_section = calculator.build_cross_section(daily_results)

        if cross_section.empty:
            continue

        # Agent 2 (带 Agent 4 的奖励)
        if not price_df.empty:
            feature_matrix = cross_section.reset_index()
            if "symbol" not in feature_matrix.columns:
                feature_matrix["symbol"] = cross_section.index

            # 先验证历史预测
            rewards = validator.verify(price_df, scan_date, trade_dates)

            # 再更新权重
            learner = WeightLearner(lookback_days=60, forward_window=10)
            config = learner.update(
                feature_matrix=feature_matrix,
                price_series=price_df,
                config=config,
                scan_date=scan_date,
                agent4_rewards=rewards,
            )

        # Agent 3
        evaluator = StateEvaluator(config=config)
        results = evaluator.evaluate_all(cross_section, daily_results, config)

        # 收集信号 (包括 hold)
        for r in results:
            if r.state in (StockState.ACCUMULATION, StockState.BREAKOUT,
                           StockState.HOLD, StockState.COLLAPSE):
                all_signals.append({
                    "scan_date": scan_date,
                    "symbol": r.symbol,
                    "state": r.state.value,
                    "confidence": r.confidence,
                    "aq_score": r.aq_score,
                    "bq_score": r.bq_score,
                    "composite_score": r.composite_score,
                })

        # Agent 4: 验证并更新
        validator.ingest_predictions(results)

        summary_rows.append({
            "scan_date": scan_date,
            "total": len(results),
            "accumulation": sum(1 for r in results if r.state == StockState.ACCUMULATION),
            "breakout": sum(1 for r in results if r.state == StockState.BREAKOUT),
            "hold": sum(1 for r in results if r.state == StockState.HOLD),
            "collapse": sum(1 for r in results if r.state == StockState.COLLAPSE),
            "config_version": config.version,
        })

    # 保存结果
    if all_signals:
        signal_df = pd.DataFrame(all_signals)
        output_path = os.path.join(output_dir, "backtest_signals.csv")
        signal_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(signal_df)} backtest signals to {output_path}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(output_dir, "backtest_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Saved backtest summary to {summary_path}")

    # 验证性能
    perf = validator.get_performance_summary()
    logger.info(f"\nValidation summary: {json.dumps(perf, indent=2, ensure_ascii=False)}")

    return pd.DataFrame(all_signals) if all_signals else pd.DataFrame()


# ═════════════════════════════════════════════════════════
# 摘要打印
# ═════════════════════════════════════════════════════════

def _print_scan_summary(results: list, scan_date: str):
    """打印扫描摘要"""
    state_counts = {}
    for r in results:
        state_counts[r.state.value] = state_counts.get(r.state.value, 0) + 1

    print(f"\n{'='*60}")
    print(f"  Adaptive State Machine Scan — {scan_date}")
    print(f"{'='*60}")
    print(f"  Total stocks evaluated: {len(results)}")
    for state in ["idle", "accumulation", "breakout", "hold", "collapse"]:
        count = state_counts.get(state, 0)
        if count > 0:
            print(f"  {state:>15}: {count}")
    print(f"{'='*60}")

    # Top signals by confidence
    accumulation = [r for r in results if r.state == StockState.ACCUMULATION]
    breakout = [r for r in results if r.state == StockState.BREAKOUT]

    if accumulation:
        top_acc = sorted(accumulation, key=lambda x: x.confidence, reverse=True)[:5]
        print(f"\n  Top Accumulation Signals:")
        for r in top_acc:
            print(f"    {r.symbol:>10}  confidence={r.confidence:.3f}  aq={r.aq_score:.3f}")

    if breakout:
        top_bo = sorted(breakout, key=lambda x: x.confidence, reverse=True)[:5]
        print(f"\n  Top Breakout Signals:")
        for r in top_bo:
            print(f"    {r.symbol:>10}  confidence={r.confidence:.3f}  bq={r.bq_score:.3f}")
