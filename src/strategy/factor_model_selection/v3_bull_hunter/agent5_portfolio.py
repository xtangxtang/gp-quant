"""
Bull Hunter v6 — Agent 5: Portfolio Manager (买卖执行 Agent)

职责:
  1. 接收 Agent 3 的 Top N 候选 (经 Agent 8 买入质量过滤) → 决策买入
  2. 接收 Agent 6 的卖出权重 → 执行卖出 (sell_weight > 阈值)
  3. 管理组合约束: 最大持仓数、单仓上限、资金分配

交互:
  Agent 3 → (Top N candidates) → Agent 8 → Agent 5
  Agent 6 → (sell_weights)      → Agent 5
  统一复盘 → (directives)        → Agent 5 (调参: 卖出阈值, 止损)

持久化:
  results/bull_hunter/portfolio/  (由 Portfolio 类管理)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .portfolio import Portfolio, _get_price, MAX_POSITIONS

logger = logging.getLogger(__name__)


@dataclass
class PortfolioConfig:
    """Agent 5 配置 (可被 Agent 7 调整)。"""
    # 卖出阈值: Agent 6 sell_weight 超过此值触发卖出
    sell_weight_threshold: float = 0.6
    # 止损线: 持仓跌幅超此值强制卖出
    stop_loss_pct: float = -0.15
    # 止盈线: 最大涨幅超此值后回撤一定比例卖出
    trailing_stop_gain: float = 0.30   # 涨 30% 后开始 trailing stop
    trailing_stop_pct: float = 0.15    # 从最高点回撤 15% 触发
    # 最大持仓数
    max_positions: int = MAX_POSITIONS
    # 单仓最大资金占比
    max_single_pct: float = 0.20
    # 最小持有天数 (买入后至少持有, 避免频繁交易)
    min_hold_days: int = 5
    # Agent 3 最低入选概率 (低于此不买)
    min_prob_200: float = 0.10
    # P1: 行业亏损过滤 — 近 N 天内同行业亏损 >= M 次则跳过
    industry_loss_lookback_days: int = 60
    industry_loss_max_count: int = 2


def run_portfolio_decisions(
    candidates: pd.DataFrame,
    sell_weights: pd.DataFrame,
    portfolio: Portfolio,
    current_date: str,
    data_dir: str,
    calendar: list[str],
    cfg: PortfolioConfig | None = None,
) -> dict:
    """
    Agent 5 每日决策: 先卖后买。

    Args:
        candidates: Agent 3 Top 5 候选 (symbol, prob_200, prob_100, rank, ...)
        sell_weights: Agent 6 卖出权重 (symbol, sell_weight, sell_reason, ...)
        portfolio: 持仓管理器
        current_date: 当日日期
        data_dir: 日线数据目录
        calendar: 交易日历
        cfg: Agent 5 配置

    Returns:
        {
            "buys": [{"symbol": ..., "price": ..., "shares": ...}, ...],
            "sells": [{"symbol": ..., "price": ..., "reason": ..., "pnl_pct": ...}, ...],
            "hold": [{"symbol": ..., "sell_weight": ..., "current_gain": ...}, ...],
            "skipped": [{"symbol": ..., "reason": ...}, ...],
            "portfolio_summary": {...},
        }
    """
    cfg = cfg or PortfolioConfig()
    result = {
        "buys": [],
        "sells": [],
        "hold": [],
        "skipped": [],
        "portfolio_summary": {},
    }

    # ── 0. 更新持仓价格 ──
    portfolio.update_prices(current_date, data_dir, calendar)

    # ── 1. 更新 Agent 6 卖出权重 ──
    if not sell_weights.empty:
        portfolio.update_sell_weights(sell_weights)

    positions = portfolio.get_positions()
    logger.info(f"Agent 5: 当前持仓 {len(positions)} 只, 现金 {portfolio.get_available_cash():.0f}")

    # ── 2. 卖出决策 (先卖后买, 释放资金) ──
    sells = _decide_sells(positions, cfg, current_date, data_dir, calendar)
    for sell_info in sells:
        sym = sell_info["symbol"]
        sell_price = _get_next_open_price(data_dir, sym, current_date, calendar)
        if sell_price is None or sell_price <= 0:
            sell_price = _get_price(data_dir, sym, current_date) or 0
        if sell_price <= 0:
            continue

        trade = portfolio.sell(
            symbol=sym,
            sell_date=current_date,
            sell_price=sell_price,
            reason=sell_info["reason"],
            sell_weight=sell_info.get("sell_weight", 0),
        )
        if trade:
            result["sells"].append(trade)

    # ── 3. 买入决策 ──
    buys = _decide_buys(candidates, portfolio, cfg, current_date, data_dir, calendar)
    for buy_info in buys:
        sym = buy_info["symbol"]
        buy_price = _get_next_open_price(data_dir, sym, current_date, calendar)
        if buy_price is None or buy_price <= 0:
            buy_price = _get_price(data_dir, sym, current_date) or 0
        if buy_price <= 0:
            result["skipped"].append({"symbol": sym, "reason": "无法获取价格"})
            continue

        shares = _calc_shares(buy_price, portfolio, cfg)
        if shares <= 0:
            result["skipped"].append({"symbol": sym, "reason": "资金不足"})
            continue

        success = portfolio.buy(
            symbol=sym,
            name=buy_info.get("name", ""),
            industry=buy_info.get("industry", ""),
            buy_date=current_date,
            buy_price=buy_price,
            shares=shares,
            prob_200=buy_info.get("prob_200", 0),
            prob_100=buy_info.get("prob_100", 0),
            rank=buy_info.get("rank", 0),
            model_date=buy_info.get("model_date", ""),
        )
        if success:
            result["buys"].append({
                "symbol": sym,
                "name": buy_info.get("name", ""),
                "price": buy_price,
                "shares": shares,
            })
        else:
            result["skipped"].append({"symbol": sym, "reason": "买入失败"})

    # ── 4. 持有列表 ──
    remaining = portfolio.get_positions()
    for _, row in remaining.iterrows():
        result["hold"].append({
            "symbol": row["symbol"],
            "name": row.get("name", ""),
            "sell_weight": float(row.get("sell_weight", 0)),
            "current_gain": float(row.get("current_gain", 0)),
            "days_held": int(row.get("days_held", 0)),
        })

    # ── 5. 每日盈亏快照 ──
    portfolio.snapshot_daily_pnl(current_date)

    result["portfolio_summary"] = {
        "n_positions": portfolio.get_position_count(),
        "cash": round(portfolio.get_available_cash(), 0),
        "n_buys": len(result["buys"]),
        "n_sells": len(result["sells"]),
        "trade_stats": portfolio.get_trade_stats(),
    }

    logger.info(f"Agent 5 完成: 买入 {len(result['buys'])} 只, "
                f"卖出 {len(result['sells'])} 只, "
                f"持有 {len(result['hold'])} 只")

    return result


# ─── 内部决策逻辑 ───

def _decide_sells(
    positions: pd.DataFrame,
    cfg: PortfolioConfig,
    current_date: str,
    data_dir: str,
    calendar: list[str],
) -> list[dict]:
    """决定哪些持仓应该卖出。"""
    if positions.empty:
        return []

    sells = []
    for _, row in positions.iterrows():
        sym = row["symbol"]
        days_held = int(row.get("days_held", 0))
        current_gain = float(row.get("current_gain", 0))
        max_gain = float(row.get("max_gain", 0))
        sell_weight = float(row.get("sell_weight", 0))
        sell_reason = str(row.get("sell_reason", ""))

        # 最小持有天数保护
        if days_held < cfg.min_hold_days:
            continue

        # ── 规则 1: Agent 6 卖出信号 (sell_weight 超阈值) ──
        if sell_weight >= cfg.sell_weight_threshold:
            reason = sell_reason if sell_reason else f"exit_signal(w={sell_weight:.2f})"
            sells.append({"symbol": sym, "reason": reason, "sell_weight": sell_weight})
            continue

        # ── 规则 2: 强制止损 ──
        if current_gain <= cfg.stop_loss_pct:
            sells.append({
                "symbol": sym,
                "reason": f"stop_loss({current_gain:+.1%})",
                "sell_weight": sell_weight,
            })
            continue

        # ── 规则 3: Trailing stop (涨幅达到阈值后回撤) ──
        if max_gain >= cfg.trailing_stop_gain:
            drawdown_from_peak = max_gain - current_gain
            if drawdown_from_peak >= cfg.trailing_stop_pct:
                sells.append({
                    "symbol": sym,
                    "reason": f"trailing_stop(max={max_gain:+.1%},dd={drawdown_from_peak:.1%})",
                    "sell_weight": sell_weight,
                })
                continue

    return sells


def _decide_buys(
    candidates: pd.DataFrame,
    portfolio: Portfolio,
    cfg: PortfolioConfig,
    current_date: str,
    data_dir: str,
    calendar: list[str],
) -> list[dict]:
    """决定买入哪些候选。"""
    if candidates.empty:
        return []

    held_symbols = portfolio.get_position_symbols()
    n_current = portfolio.get_position_count()
    available_slots = cfg.max_positions - n_current

    if available_slots <= 0:
        logger.info(f"  仓位已满 ({n_current}/{cfg.max_positions}), 不买入")
        return []

    # P1: 构建行业亏损黑名单 (近 N 天同行业亏损 >= M 次)
    blocked_industries = set()
    if cfg.industry_loss_max_count > 0:
        trades = portfolio.get_trades()
        if not trades.empty and "direction" in trades.columns:
            sells = trades[trades["direction"] == "sell"].copy()
            if not sells.empty and "trade_date" in sells.columns:
                cutoff_idx = max(0, len(calendar) - 1)
                for ci, cd in enumerate(calendar):
                    if cd >= current_date:
                        cutoff_idx = ci
                        break
                lookback_start_idx = max(0, cutoff_idx - cfg.industry_loss_lookback_days)
                lookback_start_date = calendar[lookback_start_idx] if lookback_start_idx < len(calendar) else current_date
                recent_sells = sells[sells["trade_date"] >= lookback_start_date]
                if not recent_sells.empty and "pnl_pct" in recent_sells.columns:
                    losses = recent_sells[recent_sells["pnl_pct"] < 0]
                    if not losses.empty and "industry" in losses.columns:
                        industry_loss_counts = losses.groupby("industry").size()
                        blocked_industries = set(
                            industry_loss_counts[industry_loss_counts >= cfg.industry_loss_max_count].index
                        )
                        if blocked_industries:
                            logger.info(f"  行业亏损黑名单: {blocked_industries}")

    buys = []
    for _, row in candidates.iterrows():
        if len(buys) >= available_slots:
            break

        sym = row["symbol"]
        prob_200 = float(row.get("prob_200", 0))

        # 跳过已持有
        if sym in held_symbols:
            continue

        # 概率门槛
        if prob_200 < cfg.min_prob_200:
            continue

        # P1: 行业亏损过滤
        industry = row.get("industry", "")
        if industry and industry in blocked_industries:
            logger.info(f"  跳过 {sym} ({industry}): 行业近期亏损过多")
            continue

        buys.append({
            "symbol": sym,
            "name": row.get("name", ""),
            "industry": industry,
            "prob_200": prob_200,
            "prob_100": float(row.get("prob_100", 0)),
            "rank": int(row.get("rank", 0)),
        })

    return buys


def _calc_shares(buy_price: float, portfolio: Portfolio, cfg: PortfolioConfig) -> int:
    """计算买入股数 (100 的整数倍)。"""
    cash = portfolio.get_available_cash()
    max_amount = min(cash, portfolio.initial_capital * cfg.max_single_pct)
    if max_amount <= 0:
        return 0
    shares = int(max_amount / buy_price / 100) * 100
    return max(shares, 0)


def _get_next_open_price(
    data_dir: str,
    symbol: str,
    current_date: str,
    calendar: list[str],
) -> float | None:
    """获取次日开盘价 (模拟实际交易: 信号日收盘决策, 次日开盘执行)。"""
    if current_date not in set(calendar):
        return _get_price(data_dir, symbol, current_date)
    idx = calendar.index(current_date)
    next_idx = idx + 1
    if next_idx >= len(calendar):
        return _get_price(data_dir, symbol, current_date)
    next_date = calendar[next_idx]
    price = _get_price(data_dir, symbol, next_date, "open")
    if price is None or price <= 0:
        price = _get_price(data_dir, symbol, current_date)
    return price


def _get_price(data_dir: str, symbol: str, date: str, col: str = "close") -> float | None:
    """获取价格。"""
    fpath = os.path.join(data_dir, f"{symbol}.csv")
    if not os.path.exists(fpath):
        return None
    try:
        df = pd.read_csv(fpath, usecols=["trade_date", col], dtype={"trade_date": str})
        row = df[df["trade_date"] == date]
        if row.empty:
            return None
        return float(row.iloc[0][col])
    except Exception:
        return None
