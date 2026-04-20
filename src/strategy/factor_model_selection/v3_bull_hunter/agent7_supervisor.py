"""
Bull Hunter v4 — Agent 7: Execution Supervisor (交互式监控优化 Agent)

职责:
  1. 监控 Agent 5 (Portfolio) 的买卖决策质量
  2. 监控 Agent 6 (Exit Signal) 的卖出信号准确性
  3. 与 Agent 5 交互: 调整买入条件/仓位/止损参数
  4. 与 Agent 6 交互: 调整退出因子权重/阈值
  5. 周期性评估: 交易绩效 vs 基准, 信号有效性

反馈回路:
  ┌─────────────┐      directives       ┌─────────────┐
  │   Agent 5   │ ◄──────────────────── │   Agent 7   │
  │ (Portfolio) │ ────────────────────► │(Supervisor) │
  └─────────────┘   buy/sell results    └──────┬──────┘
                                               │ directives
  ┌─────────────┐      directives       ┌──────▼──────┐
  │   Agent 6   │ ◄──────────────────── │   Agent 7   │
  │(Exit Signal)│ ────────────────────► │(Supervisor) │
  └─────────────┘   signal accuracy     └─────────────┘

评估维度:
  A. 买入质量: 买入后 N 天的涨幅分布, 是否跑赢大盘
  B. 卖出时机: 卖出后继续涨还是继续跌
  C. 持有效率: 持有期间的盈亏比, 最大回撤
  D. 信号有效性: Agent 6 sell_weight vs 实际表现的相关性

持久化:
  results/bull_hunter/portfolio/supervisor/
    supervisor_log.csv       # 每日评估日志
    directives_history.json  # 历史调优指令
    performance_report.json  # 最新绩效报告
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .agent5_portfolio import PortfolioConfig
from .agent6_exit_signal import ExitSignalConfig, save_rule_weights, MIN_TRAIN_SAMPLES

logger = logging.getLogger(__name__)

# ── 评估阈值 ──
# 买入质量: 买入后 10 天平均涨幅低于此值 → 需要调整
MIN_BUY_QUALITY_10D = -0.03
# 卖出准确率: 卖出后 10 天涨幅 > 10% 的比例 (卖早了) 超此值 → 需调整
MAX_PREMATURE_SELL_RATE = 0.40
# 最小胜率
MIN_WIN_RATE = 0.40
# 最大平均亏损
MAX_AVG_LOSS = -0.15
# 评估间隔 (至少 N 笔新交易后才评估)
MIN_TRADES_FOR_EVAL = 5

# 调参步长限制
MAX_WEIGHT_ADJUSTMENT = 0.05   # 单个权重最大调整幅度
MAX_THRESHOLD_ADJUSTMENT = 0.10  # 阈值最大调整幅度


@dataclass
class SupervisorState:
    """Supervisor 状态 (跨日保持)。"""
    last_eval_date: str = ""
    last_eval_n_trades: int = 0
    total_directives_issued: int = 0
    consecutive_underperform: int = 0


def run_supervisor(
    portfolio_dir: str,
    data_dir: str,
    current_date: str,
    calendar: list[str],
    portfolio_cfg: PortfolioConfig,
    exit_cfg: ExitSignalConfig,
    today_result: dict | None = None,
) -> dict:
    """
    Agent 7 每日运行: 评估 + 生成调优指令。

    Args:
        portfolio_dir: 持仓管理目录
        data_dir: 日线数据目录
        current_date: 当日日期
        calendar: 交易日历
        portfolio_cfg: Agent 5 当前配置
        exit_cfg: Agent 6 当前配置
        today_result: Agent 5 当日的买卖结果

    Returns:
        {
            "evaluation": {...},           # 绩效评估
            "agent5_directives": {...},    # 给 Agent 5 的调参指令
            "agent6_directives": {...},    # 给 Agent 6 的调参指令
            "should_retrain_exit": bool,   # 是否触发 Agent 6 重训
            "status": str,                 # healthy / warning / critical
        }
    """
    supervisor_dir = os.path.join(portfolio_dir, "supervisor")
    os.makedirs(supervisor_dir, exist_ok=True)

    result = {
        "evaluation": {},
        "agent5_directives": {},
        "agent6_directives": {},
        "should_retrain_exit": False,
        "status": "healthy",
    }

    # 加载状态
    state = _load_state(supervisor_dir)

    # ── 1. 评估交易绩效 ──
    evaluation = _evaluate_performance(portfolio_dir, data_dir, calendar, current_date)
    result["evaluation"] = evaluation

    # ── 2. 评估 Agent 6 信号质量 ──
    signal_eval = _evaluate_exit_signals(portfolio_dir, data_dir, calendar, current_date)
    evaluation["signal_quality"] = signal_eval

    # ── 3. 判断是否需要干预 ──
    needs_intervention, issues = _check_intervention_needed(evaluation, state)

    if not needs_intervention:
        logger.info(f"Agent 7: 绩效正常, 无需干预 (status=healthy)")
        result["status"] = "healthy"
        state.consecutive_underperform = 0
    else:
        state.consecutive_underperform += 1
        severity = "warning" if state.consecutive_underperform < 3 else "critical"
        result["status"] = severity
        logger.info(f"Agent 7: 检测到 {len(issues)} 个问题, status={severity}")
        for issue in issues:
            logger.info(f"  ⚠️ {issue}")

        # ── 4. 生成调优指令 ──
        a5_directives = _generate_agent5_directives(evaluation, issues, portfolio_cfg)
        a6_directives = _generate_agent6_directives(evaluation, signal_eval, issues, exit_cfg, portfolio_dir)
        result["agent5_directives"] = a5_directives
        result["agent6_directives"] = a6_directives

        # ── 5. 判断是否需要重训 Agent 6 模型 ──
        if signal_eval.get("correlation", 0) < 0.1 and evaluation.get("n_sells", 0) >= MIN_TRAIN_SAMPLES:
            result["should_retrain_exit"] = True
            logger.info("  🔄 触发 Agent 6 卖出模型重训")

    # ── 6. 记录日志 ──
    _log_evaluation(supervisor_dir, current_date, evaluation, result)

    # 更新状态
    state.last_eval_date = current_date
    state.last_eval_n_trades = evaluation.get("n_total_trades", 0)
    _save_state(supervisor_dir, state)

    return result


def apply_directives(
    portfolio_cfg: PortfolioConfig,
    exit_cfg: ExitSignalConfig,
    agent5_directives: dict,
    agent6_directives: dict,
    portfolio_dir: str,
) -> tuple[PortfolioConfig, ExitSignalConfig]:
    """
    应用 Agent 7 的调参指令, 返回更新后的配置。

    Agent 5 指令类型:
      - adjust_sell_threshold: 调整卖出权重阈值
      - adjust_stop_loss: 调整止损线
      - adjust_trailing_stop: 调整 trailing stop 参数
      - adjust_max_positions: 调整最大持仓数

    Agent 6 指令类型:
      - adjust_factor_weight: 调整退出因子权重
      - reset_weights: 重置为默认权重
    """
    new_p_cfg = portfolio_cfg
    new_e_cfg = exit_cfg

    # ── Agent 5 指令 ──
    for action in agent5_directives.get("actions", []):
        atype = action.get("type", "")

        if atype == "adjust_sell_threshold":
            old = new_p_cfg.sell_weight_threshold
            delta = _clamp(action.get("delta", 0), -MAX_THRESHOLD_ADJUSTMENT, MAX_THRESHOLD_ADJUSTMENT)
            new_val = max(0.3, min(0.9, old + delta))
            new_p_cfg = PortfolioConfig(
                sell_weight_threshold=new_val,
                stop_loss_pct=new_p_cfg.stop_loss_pct,
                trailing_stop_gain=new_p_cfg.trailing_stop_gain,
                trailing_stop_pct=new_p_cfg.trailing_stop_pct,
                max_positions=new_p_cfg.max_positions,
                max_single_pct=new_p_cfg.max_single_pct,
                min_hold_days=new_p_cfg.min_hold_days,
                min_prob_200=new_p_cfg.min_prob_200,
            )
            logger.info(f"  Agent 7 → Agent 5: sell_threshold {old:.2f} → {new_val:.2f}")

        elif atype == "adjust_stop_loss":
            old = new_p_cfg.stop_loss_pct
            delta = _clamp(action.get("delta", 0), -MAX_THRESHOLD_ADJUSTMENT, MAX_THRESHOLD_ADJUSTMENT)
            new_val = max(-0.30, min(-0.05, old + delta))
            new_p_cfg = PortfolioConfig(
                sell_weight_threshold=new_p_cfg.sell_weight_threshold,
                stop_loss_pct=new_val,
                trailing_stop_gain=new_p_cfg.trailing_stop_gain,
                trailing_stop_pct=new_p_cfg.trailing_stop_pct,
                max_positions=new_p_cfg.max_positions,
                max_single_pct=new_p_cfg.max_single_pct,
                min_hold_days=new_p_cfg.min_hold_days,
                min_prob_200=new_p_cfg.min_prob_200,
            )
            logger.info(f"  Agent 7 → Agent 5: stop_loss {old:.2f} → {new_val:.2f}")

    # ── Agent 6 指令 ──
    for action in agent6_directives.get("actions", []):
        atype = action.get("type", "")

        if atype == "adjust_factor_weight":
            factor = action.get("factor", "")
            delta = _clamp(action.get("delta", 0), -MAX_WEIGHT_ADJUSTMENT, MAX_WEIGHT_ADJUSTMENT)
            new_weights = dict(new_e_cfg.rule_weights)
            if factor in new_weights:
                old_w = new_weights[factor]
                new_w = max(0, min(0.30, old_w + delta))
                new_weights[factor] = round(new_w, 4)
                logger.info(f"  Agent 7 → Agent 6: {factor} 权重 {old_w:.3f} → {new_w:.3f}")

            # 重新归一化
            total = sum(new_weights.values())
            if total > 0:
                new_weights = {k: round(v / total, 4) for k, v in new_weights.items()}

            new_e_cfg = ExitSignalConfig(
                rule_weights=new_weights,
                use_model=new_e_cfg.use_model,
                retrain_interval=new_e_cfg.retrain_interval,
                target_gain=new_e_cfg.target_gain,
                max_hold_days=new_e_cfg.max_hold_days,
            )

            # 持久化新权重
            model_dir = os.path.join(portfolio_dir, "exit_models")
            save_rule_weights(model_dir, new_weights)

        elif atype == "reset_weights":
            from .agent6_exit_signal import DEFAULT_RULE_WEIGHTS
            new_e_cfg = ExitSignalConfig(
                rule_weights=dict(DEFAULT_RULE_WEIGHTS),
                use_model=new_e_cfg.use_model,
                retrain_interval=new_e_cfg.retrain_interval,
                target_gain=new_e_cfg.target_gain,
                max_hold_days=new_e_cfg.max_hold_days,
            )
            logger.info("  Agent 7 → Agent 6: 重置为默认权重")

    return new_p_cfg, new_e_cfg


# ─── 评估逻辑 ───

def _evaluate_performance(
    portfolio_dir: str,
    data_dir: str,
    calendar: list[str],
    current_date: str,
) -> dict:
    """评估整体交易绩效。"""
    trades_path = os.path.join(portfolio_dir, "trades.csv")
    if not os.path.exists(trades_path):
        return {"n_total_trades": 0}

    trades = pd.read_csv(trades_path, dtype={"trade_date": str})
    buys = trades[trades["direction"] == "buy"]
    sells = trades[trades["direction"] == "sell"]

    eval_result = {
        "n_total_trades": len(trades),
        "n_buys": len(buys),
        "n_sells": len(sells),
    }

    if sells.empty:
        return eval_result

    # 胜率
    wins = sells[sells["pnl"] > 0]
    eval_result["win_rate"] = round(len(wins) / len(sells), 4)
    eval_result["avg_pnl_pct"] = round(float(sells["pnl_pct"].mean()), 4)
    eval_result["total_pnl"] = round(float(sells["pnl"].sum()), 2)
    eval_result["avg_holding_days"] = round(float(sells["holding_days"].mean()), 1)
    eval_result["best_trade"] = round(float(sells["pnl_pct"].max()), 4)
    eval_result["worst_trade"] = round(float(sells["pnl_pct"].min()), 4)

    # 按原因分类
    reason_stats = {}
    for reason in sells["reason"].unique():
        subset = sells[sells["reason"] == reason]
        reason_stats[reason] = {
            "count": len(subset),
            "avg_pnl_pct": round(float(subset["pnl_pct"].mean()), 4),
            "win_rate": round(len(subset[subset["pnl"] > 0]) / len(subset), 4) if len(subset) > 0 else 0,
        }
    eval_result["by_reason"] = reason_stats

    # 买入质量: 最近 10 笔买入后 10 天的表现
    buy_quality = _eval_buy_quality(buys, data_dir, calendar, current_date, lookback_trades=10)
    eval_result["buy_quality"] = buy_quality

    # 卖出时机: 卖出后是否继续涨
    sell_timing = _eval_sell_timing(sells, data_dir, calendar, current_date, lookback_trades=10)
    eval_result["sell_timing"] = sell_timing

    return eval_result


def _evaluate_exit_signals(
    portfolio_dir: str,
    data_dir: str,
    calendar: list[str],
    current_date: str,
) -> dict:
    """评估 Agent 6 卖出信号质量: sell_weight vs 实际后续表现的相关性。"""
    weights_dir = os.path.join(portfolio_dir, "sell_weights")
    if not os.path.exists(weights_dir):
        return {"correlation": 0, "n_samples": 0}

    # 读取历史卖出权重
    weight_files = sorted([f for f in os.listdir(weights_dir) if f.endswith(".csv")])
    if not weight_files:
        return {"correlation": 0, "n_samples": 0}

    # 只看最近 30 个文件
    weight_files = weight_files[-30:]

    samples = []
    date_set = set(calendar)

    for wf in weight_files:
        date_str = wf.replace(".csv", "")
        try:
            df = pd.read_csv(os.path.join(weights_dir, wf))
        except Exception:
            continue

        if "symbol" not in df.columns or "sell_weight" not in df.columns:
            continue

        # 计算该日期后 10 天的实际涨幅
        if date_str not in date_set:
            continue
        d_idx = calendar.index(date_str)
        fwd_idx = min(d_idx + 10, len(calendar) - 1)
        fwd_date = calendar[fwd_idx]

        for _, row in df.iterrows():
            sym = row["symbol"]
            sw = float(row["sell_weight"])
            # 获取 forward return
            close_now = _get_price_cached(data_dir, sym, date_str)
            close_fwd = _get_price_cached(data_dir, sym, fwd_date)
            if close_now and close_fwd and close_now > 0:
                fwd_ret = (close_fwd - close_now) / close_now
                samples.append({"sell_weight": sw, "fwd_return": fwd_ret})

    if len(samples) < 10:
        return {"correlation": 0, "n_samples": len(samples)}

    df_samples = pd.DataFrame(samples)
    # sell_weight 高 → fwd_return 应该低 (负相关 = 好)
    corr = float(df_samples["sell_weight"].corr(df_samples["fwd_return"]))

    # 按 sell_weight 分组看效果
    high_sw = df_samples[df_samples["sell_weight"] >= 0.6]
    low_sw = df_samples[df_samples["sell_weight"] < 0.3]

    return {
        "correlation": round(corr, 4),
        "n_samples": len(samples),
        "high_weight_avg_fwd": round(float(high_sw["fwd_return"].mean()), 4) if not high_sw.empty else 0,
        "low_weight_avg_fwd": round(float(low_sw["fwd_return"].mean()), 4) if not low_sw.empty else 0,
    }


def _eval_buy_quality(
    buys: pd.DataFrame,
    data_dir: str,
    calendar: list[str],
    current_date: str,
    lookback_trades: int = 10,
) -> dict:
    """评估最近 N 笔买入的质量。"""
    if buys.empty:
        return {}

    recent_buys = buys.tail(lookback_trades)
    date_set = set(calendar)
    gains_10d = []

    for _, row in recent_buys.iterrows():
        sym = row["symbol"]
        buy_date = str(row.get("trade_date", ""))
        buy_price = float(row.get("price", 0))
        if buy_price <= 0 or buy_date not in date_set:
            continue

        b_idx = calendar.index(buy_date)
        fwd_idx = min(b_idx + 10, len(calendar) - 1)
        fwd_date = calendar[fwd_idx]

        fwd_close = _get_price_cached(data_dir, sym, fwd_date)
        if fwd_close and fwd_close > 0:
            gain = (fwd_close - buy_price) / buy_price
            gains_10d.append(gain)

    if not gains_10d:
        return {}

    return {
        "avg_gain_10d": round(float(np.mean(gains_10d)), 4),
        "win_rate_10d": round(sum(1 for g in gains_10d if g > 0) / len(gains_10d), 4),
        "n_evaluated": len(gains_10d),
    }


def _eval_sell_timing(
    sells: pd.DataFrame,
    data_dir: str,
    calendar: list[str],
    current_date: str,
    lookback_trades: int = 10,
) -> dict:
    """评估最近 N 笔卖出的时机: 卖出后是否继续涨。"""
    if sells.empty:
        return {}

    recent_sells = sells.tail(lookback_trades)
    date_set = set(calendar)
    post_sell_gains = []

    for _, row in recent_sells.iterrows():
        sym = row["symbol"]
        sell_date = str(row.get("trade_date", ""))
        sell_price = float(row.get("price", 0))
        if sell_price <= 0 or sell_date not in date_set:
            continue

        s_idx = calendar.index(sell_date)
        fwd_idx = min(s_idx + 10, len(calendar) - 1)
        fwd_date = calendar[fwd_idx]

        fwd_close = _get_price_cached(data_dir, sym, fwd_date)
        if fwd_close and fwd_close > 0:
            post_gain = (fwd_close - sell_price) / sell_price
            post_sell_gains.append(post_gain)

    if not post_sell_gains:
        return {}

    premature_rate = sum(1 for g in post_sell_gains if g > 0.10) / len(post_sell_gains)

    return {
        "avg_post_sell_10d": round(float(np.mean(post_sell_gains)), 4),
        "premature_sell_rate": round(premature_rate, 4),
        "n_evaluated": len(post_sell_gains),
    }


# ─── 干预判断 ───

def _check_intervention_needed(evaluation: dict, state: SupervisorState) -> tuple[bool, list[str]]:
    """判断是否需要干预。"""
    issues = []

    n_sells = evaluation.get("n_sells", 0)
    if n_sells < MIN_TRADES_FOR_EVAL:
        return False, []

    # 胜率太低
    win_rate = evaluation.get("win_rate", 1.0)
    if win_rate < MIN_WIN_RATE:
        issues.append(f"胜率过低: {win_rate:.1%} (阈值 {MIN_WIN_RATE:.0%})")

    # 平均亏损太大
    avg_pnl = evaluation.get("avg_pnl_pct", 0)
    if avg_pnl < MAX_AVG_LOSS:
        issues.append(f"平均亏损过大: {avg_pnl:+.1%} (阈值 {MAX_AVG_LOSS:+.0%})")

    # 买入质量差
    buy_q = evaluation.get("buy_quality", {})
    if buy_q.get("avg_gain_10d", 0) < MIN_BUY_QUALITY_10D:
        issues.append(f"买入质量差: 10日平均 {buy_q['avg_gain_10d']:+.1%}")

    # 卖出过早
    sell_t = evaluation.get("sell_timing", {})
    if sell_t.get("premature_sell_rate", 0) > MAX_PREMATURE_SELL_RATE:
        issues.append(f"卖出过早: {sell_t['premature_sell_rate']:.0%} 卖出后 10 天涨超 10%")

    # 信号质量差
    sig_q = evaluation.get("signal_quality", {})
    if sig_q.get("n_samples", 0) >= 20:
        corr = sig_q.get("correlation", 0)
        if corr > 0.05:  # 正相关 = sell_weight 高但后续反而涨, 信号反了
            issues.append(f"退出信号反向: 相关性 {corr:+.3f} (应为负)")

    return len(issues) > 0, issues


# ─── 生成调优指令 ───

def _generate_agent5_directives(
    evaluation: dict,
    issues: list[str],
    cfg: PortfolioConfig,
) -> dict:
    """生成给 Agent 5 的调参指令。"""
    actions = []

    buy_q = evaluation.get("buy_quality", {})
    sell_t = evaluation.get("sell_timing", {})
    win_rate = evaluation.get("win_rate", 1.0)

    # 胜率低 → 提高卖出阈值 (更不容易卖, 多持有)
    if win_rate < MIN_WIN_RATE:
        actions.append({
            "type": "adjust_sell_threshold",
            "delta": 0.05,
            "reason": f"胜率低({win_rate:.0%}), 提高卖出门槛",
        })

    # 卖出过早 → 提高卖出阈值
    if sell_t.get("premature_sell_rate", 0) > MAX_PREMATURE_SELL_RATE:
        actions.append({
            "type": "adjust_sell_threshold",
            "delta": 0.05,
            "reason": f"卖出过早({sell_t['premature_sell_rate']:.0%}), 提高门槛",
        })

    # 亏损大 → 收紧止损
    worst = evaluation.get("worst_trade", 0)
    if worst < -0.25:
        actions.append({
            "type": "adjust_stop_loss",
            "delta": 0.03,  # 收紧 (更接近 0)
            "reason": f"最差交易 {worst:+.0%}, 收紧止损",
        })

    return {"actions": actions[:3]}  # 限制最多 3 个指令


def _generate_agent6_directives(
    evaluation: dict,
    signal_eval: dict,
    issues: list[str],
    cfg: ExitSignalConfig,
    portfolio_dir: str,
) -> dict:
    """生成给 Agent 6 的调参指令。"""
    actions = []

    sell_t = evaluation.get("sell_timing", {})
    by_reason = evaluation.get("by_reason", {})

    # 分析哪类卖出原因效果最差
    worst_reason = None
    worst_pnl = 0
    for reason, stats in by_reason.items():
        if stats.get("count", 0) >= 3 and stats.get("avg_pnl_pct", 0) < worst_pnl:
            worst_pnl = stats["avg_pnl_pct"]
            worst_reason = reason

    # 如果某类卖出原因亏损严重, 降低对应因子权重
    if worst_reason and worst_pnl < -0.10:
        # 解析 reason 中的因子 (格式: "momentum_decay(0.15)|vol_expansion(0.08)")
        factor_to_adjust = _extract_factor_from_reason(worst_reason)
        if factor_to_adjust:
            actions.append({
                "type": "adjust_factor_weight",
                "factor": factor_to_adjust,
                "delta": -0.02,
                "reason": f"'{worst_reason}' 类卖出平均亏损 {worst_pnl:+.1%}",
            })

    # 信号反向 → 重置权重
    corr = signal_eval.get("correlation", 0)
    if corr > 0.10 and signal_eval.get("n_samples", 0) >= 20:
        actions.append({
            "type": "reset_weights",
            "reason": f"退出信号反向 (corr={corr:+.3f}), 重置为默认",
        })

    # 高权重信号的效果
    high_fwd = signal_eval.get("high_weight_avg_fwd", 0)
    low_fwd = signal_eval.get("low_weight_avg_fwd", 0)

    # 如果高权重组 forward return 比低权重组还高 → 信号无效
    if high_fwd > low_fwd + 0.02 and signal_eval.get("n_samples", 0) >= 20:
        actions.append({
            "type": "reset_weights",
            "reason": f"高权重组反而涨更多 ({high_fwd:+.1%} vs {low_fwd:+.1%})",
        })

    return {"actions": actions[:3]}


def _extract_factor_from_reason(reason: str) -> str | None:
    """从卖出原因字符串中提取因子名。"""
    # 格式: "exit_signal(w=0.75)" 或 "momentum_decay(0.15)|vol_expansion(0.08)"
    if "(" in reason:
        factor = reason.split("(")[0].strip()
        # 移除 exit_ 前缀
        if factor.startswith("exit_"):
            factor = factor[5:]
        if factor in DEFAULT_RULE_WEIGHTS_KEYS:
            return factor
    return None


# 用于 _extract_factor_from_reason
DEFAULT_RULE_WEIGHTS_KEYS = {
    "momentum_decay", "momentum_accel", "vol_expansion", "vol_spike",
    "entropy_disorder", "irrev_collapse", "mf_outflow_accel", "mf_streak_negative",
    "below_ma20", "below_ma60", "atr_anomaly", "drawdown_from_peak",
    "gain_vs_target", "holding_days_norm",
}


# ─── 状态持久化 ───

def _load_state(supervisor_dir: str) -> SupervisorState:
    state_path = os.path.join(supervisor_dir, "state.json")
    if os.path.exists(state_path):
        try:
            with open(state_path) as f:
                data = json.load(f)
            return SupervisorState(**{k: data[k] for k in SupervisorState.__dataclass_fields__ if k in data})
        except Exception:
            pass
    return SupervisorState()


def _save_state(supervisor_dir: str, state: SupervisorState):
    state_path = os.path.join(supervisor_dir, "state.json")
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump({
            "last_eval_date": state.last_eval_date,
            "last_eval_n_trades": state.last_eval_n_trades,
            "total_directives_issued": state.total_directives_issued,
            "consecutive_underperform": state.consecutive_underperform,
        }, f, ensure_ascii=False, indent=2)


def _log_evaluation(supervisor_dir: str, date: str, evaluation: dict, result: dict):
    """记录每日评估日志。"""
    log_path = os.path.join(supervisor_dir, "supervisor_log.csv")
    row = {
        "date": date,
        "status": result.get("status", ""),
        "n_sells": evaluation.get("n_sells", 0),
        "win_rate": evaluation.get("win_rate", 0),
        "avg_pnl_pct": evaluation.get("avg_pnl_pct", 0),
        "n_a5_directives": len(result.get("agent5_directives", {}).get("actions", [])),
        "n_a6_directives": len(result.get("agent6_directives", {}).get("actions", [])),
        "retrain_exit": result.get("should_retrain_exit", False),
    }

    if os.path.exists(log_path):
        try:
            df = pd.read_csv(log_path, dtype={"date": str})
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        except Exception:
            df = pd.DataFrame([row])
    else:
        df = pd.DataFrame([row])

    df.to_csv(log_path, index=False, encoding="utf-8-sig")

    # 保存详细报告
    report_path = os.path.join(supervisor_dir, "performance_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"date": date, "evaluation": evaluation, "result": result},
                  f, ensure_ascii=False, indent=2, default=str)


# ─── 辅助 ───

_price_cache: dict[str, float | None] = {}


def _get_price_cached(data_dir: str, symbol: str, date: str) -> float | None:
    key = f"{symbol}_{date}"
    if key not in _price_cache:
        fpath = os.path.join(data_dir, f"{symbol}.csv")
        if not os.path.exists(fpath):
            _price_cache[key] = None
        else:
            try:
                df = pd.read_csv(fpath, usecols=["trade_date", "close"], dtype={"trade_date": str})
                row = df[df["trade_date"] == date]
                _price_cache[key] = float(row.iloc[0]["close"]) if not row.empty else None
            except Exception:
                _price_cache[key] = None
    return _price_cache[key]


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))
