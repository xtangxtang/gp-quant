"""
Bull Hunter v6 — Review Helpers (formerly Agent 7 Supervisor)

v6 简化:
  原 Agent 7 (run_supervisor 每日干预) 已被 pipeline.run_unified_review 替代,
  本模块仅保留统一复盘所需的纯函数工具:
    - _evaluate_performance / _evaluate_exit_signals  : 交易+信号质量评估
    - _eval_buy_quality / _eval_sell_timing           : 买卖时机评估
    - _check_intervention_needed                      : 是否需要参数干预
    - _generate_agent5_directives                     : Agent 5 阈值/止损调整
    - _generate_agent6_directives                     : Agent 6 通用调整 (因子权重交由 auto_adjust_exit_weights)
    - apply_directives                                : 把指令落到 PortfolioConfig / ExitSignalConfig

注意: Agent 6 因子权重的相关性自适应已统一由 agent6_exit_signal.auto_adjust_exit_weights()
处理, 本模块不再生成 adjust_factor_weight / reset_weights 类型指令。
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from .agent5_portfolio import PortfolioConfig
from .agent6_exit_signal import ExitSignalConfig

logger = logging.getLogger(__name__)

# ── 评估阈值 ──
MIN_BUY_QUALITY_10D = -0.03
MAX_PREMATURE_SELL_RATE = 0.40
MIN_WIN_RATE = 0.40
MAX_AVG_LOSS = -0.15
MIN_TRADES_FOR_EVAL = 5

# 调参步长限制
MAX_THRESHOLD_ADJUSTMENT = 0.10


def apply_directives(
    portfolio_cfg: PortfolioConfig,
    exit_cfg: ExitSignalConfig,
    agent5_directives: dict,
    agent6_directives: dict,
    portfolio_dir: str,
) -> tuple[PortfolioConfig, ExitSignalConfig]:
    """
    应用统一复盘的调参指令, 返回更新后的配置。

    Agent 5 指令类型:
      - adjust_sell_threshold
      - adjust_stop_loss

    Agent 6 指令类型:
      - (因子权重由 auto_adjust_exit_weights 处理, 此处仅保留扩展位)

    portfolio_dir 参数保留用于将来可能的持久化, 当前未使用。
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
            logger.info(f"  调参 → Agent 5: sell_threshold {old:.2f} → {new_val:.2f}")

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
            logger.info(f"  调参 → Agent 5: stop_loss {old:.2f} → {new_val:.2f}")

    # ── Agent 6 指令 (预留, 当前因子权重由 auto_adjust 处理) ──
    # 此处保留扩展点, 暂不消费任何 action

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

    wins = sells[sells["pnl"] > 0]
    eval_result["win_rate"] = round(len(wins) / len(sells), 4)
    eval_result["avg_pnl_pct"] = round(float(sells["pnl_pct"].mean()), 4)
    eval_result["total_pnl"] = round(float(sells["pnl"].sum()), 2)
    eval_result["avg_holding_days"] = round(float(sells["holding_days"].mean()), 1)
    eval_result["best_trade"] = round(float(sells["pnl_pct"].max()), 4)
    eval_result["worst_trade"] = round(float(sells["pnl_pct"].min()), 4)

    reason_stats = {}
    for reason in sells["reason"].unique():
        subset = sells[sells["reason"] == reason]
        reason_stats[reason] = {
            "count": len(subset),
            "avg_pnl_pct": round(float(subset["pnl_pct"].mean()), 4),
            "win_rate": round(len(subset[subset["pnl"] > 0]) / len(subset), 4) if len(subset) > 0 else 0,
        }
    eval_result["by_reason"] = reason_stats

    eval_result["buy_quality"] = _eval_buy_quality(buys, data_dir, calendar, current_date, lookback_trades=10)
    eval_result["sell_timing"] = _eval_sell_timing(sells, data_dir, calendar, current_date, lookback_trades=10)
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

    weight_files = sorted([f for f in os.listdir(weights_dir) if f.endswith(".csv")])
    if not weight_files:
        return {"correlation": 0, "n_samples": 0}

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
        if date_str not in date_set:
            continue

        d_idx = calendar.index(date_str)
        fwd_idx = min(d_idx + 10, len(calendar) - 1)
        fwd_date = calendar[fwd_idx]

        for _, row in df.iterrows():
            sym = row["symbol"]
            sw = float(row["sell_weight"])
            close_now = _get_price_cached(data_dir, sym, date_str)
            close_fwd = _get_price_cached(data_dir, sym, fwd_date)
            if close_now and close_fwd and close_now > 0:
                fwd_ret = (close_fwd - close_now) / close_now
                samples.append({"sell_weight": sw, "fwd_return": fwd_ret})

    if len(samples) < 10:
        return {"correlation": 0, "n_samples": len(samples)}

    df_samples = pd.DataFrame(samples)
    corr = float(df_samples["sell_weight"].corr(df_samples["fwd_return"]))
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

def _check_intervention_needed(evaluation: dict) -> tuple[bool, list[str]]:
    """判断是否需要参数干预 (v6: 无状态, 纯基于当期评估)。"""
    issues = []

    n_sells = evaluation.get("n_sells", 0)
    if n_sells < MIN_TRADES_FOR_EVAL:
        return False, []

    win_rate = evaluation.get("win_rate", 1.0)
    if win_rate < MIN_WIN_RATE:
        issues.append(f"胜率过低: {win_rate:.1%} (阈值 {MIN_WIN_RATE:.0%})")

    avg_pnl = evaluation.get("avg_pnl_pct", 0)
    if avg_pnl < MAX_AVG_LOSS:
        issues.append(f"平均亏损过大: {avg_pnl:+.1%} (阈值 {MAX_AVG_LOSS:+.0%})")

    buy_q = evaluation.get("buy_quality", {})
    if buy_q.get("avg_gain_10d", 0) < MIN_BUY_QUALITY_10D:
        issues.append(f"买入质量差: 10日平均 {buy_q['avg_gain_10d']:+.1%}")

    sell_t = evaluation.get("sell_timing", {})
    if sell_t.get("premature_sell_rate", 0) > MAX_PREMATURE_SELL_RATE:
        issues.append(f"卖出过早: {sell_t['premature_sell_rate']:.0%} 卖出后 10 天涨超 10%")

    sig_q = evaluation.get("signal_quality", {})
    if sig_q.get("n_samples", 0) >= 20:
        corr = sig_q.get("correlation", 0)
        if corr > 0.05:
            issues.append(f"退出信号反向: 相关性 {corr:+.3f} (应为负)")

    return len(issues) > 0, issues


# ─── 生成调优指令 ───

def _generate_agent5_directives(
    evaluation: dict,
    issues: list[str],
    cfg: PortfolioConfig,
) -> dict:
    """生成给 Agent 5 的调参指令 (阈值/止损)。"""
    actions = []

    sell_t = evaluation.get("sell_timing", {})
    win_rate = evaluation.get("win_rate", 1.0)

    if win_rate < MIN_WIN_RATE:
        actions.append({
            "type": "adjust_sell_threshold",
            "delta": 0.05,
            "reason": f"胜率低({win_rate:.0%}), 提高卖出门槛",
        })

    if sell_t.get("premature_sell_rate", 0) > MAX_PREMATURE_SELL_RATE:
        actions.append({
            "type": "adjust_sell_threshold",
            "delta": 0.05,
            "reason": f"卖出过早({sell_t['premature_sell_rate']:.0%}), 提高门槛",
        })

    worst = evaluation.get("worst_trade", 0)
    if worst < -0.25:
        actions.append({
            "type": "adjust_stop_loss",
            "delta": 0.03,
            "reason": f"最差交易 {worst:+.0%}, 收紧止损",
        })

    return {"actions": actions[:3]}


def _generate_agent6_directives(
    evaluation: dict,
    signal_eval: dict,
    issues: list[str],
    cfg: ExitSignalConfig,
    portfolio_dir: str,
) -> dict:
    """
    生成给 Agent 6 的调参指令。

    v6 简化: 因子权重的相关性自适应已交由 auto_adjust_exit_weights 统一处理,
    本函数仅保留 should_retrain_model 提示, 不再生成权重调整指令。
    """
    suggestions = []

    corr = signal_eval.get("correlation", 0)
    n_samples = signal_eval.get("n_samples", 0)
    if corr > 0.10 and n_samples >= 20:
        suggestions.append(f"退出信号反向 (corr={corr:+.3f}), 建议触发 Agent 6 模型重训")

    high_fwd = signal_eval.get("high_weight_avg_fwd", 0)
    low_fwd = signal_eval.get("low_weight_avg_fwd", 0)
    if high_fwd > low_fwd + 0.02 and n_samples >= 20:
        suggestions.append(f"高权重组反而涨更多 ({high_fwd:+.1%} vs {low_fwd:+.1%}), 建议重训")

    return {"actions": [], "suggestions": suggestions}


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
