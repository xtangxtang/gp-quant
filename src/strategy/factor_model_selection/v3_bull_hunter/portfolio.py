"""
Bull Hunter v4 — Portfolio 持仓状态管理

持久化管理买入持仓、交易记录和每日损益。

存储结构:
  results/bull_hunter/portfolio/
    positions.csv     # 当前持仓
    trades.csv        # 交易记录 (买入/卖出)
    daily_pnl.csv     # 每日持仓盈亏快照
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── 列定义 ──

POSITION_COLS = [
    "symbol", "name", "industry",
    "buy_date", "buy_price", "shares",
    "prob_200", "prob_100", "rank", "model_date",
    "days_held", "current_close", "current_gain", "max_gain", "max_drawdown",
    "sell_weight",       # Agent 6 输出的卖出权重 (0~1, 越大越该卖)
    "sell_reason",       # Agent 6 最新卖出原因 (空=继续持有)
]

TRADE_COLS = [
    "symbol", "name", "industry",
    "trade_date", "direction",   # buy / sell
    "price", "shares",
    "reason",                    # buy_signal / exit_signal / stop_loss / profit_target / supervisor_force
    "pnl",                       # 卖出时的盈亏金额
    "pnl_pct",                   # 卖出时的盈亏百分比
    "holding_days",              # 持有天数
    "prob_200", "prob_100",
    "sell_weight",               # 卖出时的 Agent 6 权重
]

DAILY_PNL_COLS = [
    "date", "n_positions", "total_value", "total_cost",
    "daily_pnl", "cumulative_pnl", "cumulative_pnl_pct",
]

# 默认初始资金
DEFAULT_INITIAL_CAPITAL = 1_000_000.0
# 单只最大仓位比例
MAX_POSITION_PCT = 0.20
# 最大持仓数
MAX_POSITIONS = 10


class Portfolio:
    """管理持仓、交易记录和每日损益。"""

    def __init__(self, portfolio_dir: str, initial_capital: float = DEFAULT_INITIAL_CAPITAL):
        self.portfolio_dir = portfolio_dir
        self.initial_capital = initial_capital
        self.positions_path = os.path.join(portfolio_dir, "positions.csv")
        self.trades_path = os.path.join(portfolio_dir, "trades.csv")
        self.daily_pnl_path = os.path.join(portfolio_dir, "daily_pnl.csv")
        os.makedirs(portfolio_dir, exist_ok=True)

    # ─── 持仓操作 ───

    def get_positions(self) -> pd.DataFrame:
        """获取当前持仓。"""
        if os.path.exists(self.positions_path):
            try:
                df = pd.read_csv(
                    self.positions_path,
                    dtype={"buy_date": str, "model_date": str, "sell_reason": str},
                )
                # sell_reason 空值填为空字符串
                df["sell_reason"] = df["sell_reason"].fillna("")
                return df
            except Exception:
                pass
        return pd.DataFrame(columns=POSITION_COLS)

    def get_position_symbols(self) -> set[str]:
        """当前持仓代码集合。"""
        pos = self.get_positions()
        if pos.empty:
            return set()
        return set(pos["symbol"])

    def get_position_count(self) -> int:
        pos = self.get_positions()
        return len(pos)

    def get_available_cash(self) -> float:
        """可用资金 = 初始资金 - 持仓成本。"""
        pos = self.get_positions()
        if pos.empty:
            return self.initial_capital
        total_cost = (pos["buy_price"] * pos["shares"]).sum()
        return self.initial_capital - total_cost

    def buy(
        self,
        symbol: str,
        name: str,
        industry: str,
        buy_date: str,
        buy_price: float,
        shares: int,
        prob_200: float = 0.0,
        prob_100: float = 0.0,
        rank: int = 0,
        model_date: str = "",
    ) -> bool:
        """
        买入: 添加一笔新持仓。

        Returns:
            是否成功买入
        """
        if buy_price <= 0 or shares <= 0:
            logger.warning(f"买入失败: {symbol} 价格或数量无效")
            return False

        positions = self.get_positions()

        # 检查是否已持有
        if not positions.empty and symbol in positions["symbol"].values:
            logger.info(f"已持有 {symbol}, 跳过买入")
            return False

        # 检查仓位上限
        if len(positions) >= MAX_POSITIONS:
            logger.warning(f"持仓已满 ({MAX_POSITIONS}), 无法买入 {symbol}")
            return False

        # 检查资金
        cost = buy_price * shares
        cash = self.get_available_cash()
        if cost > cash:
            logger.warning(f"资金不足: 需要 {cost:.0f}, 可用 {cash:.0f}")
            return False

        new_row = {
            "symbol": symbol,
            "name": name,
            "industry": industry,
            "buy_date": buy_date,
            "buy_price": round(buy_price, 2),
            "shares": shares,
            "prob_200": round(prob_200, 6),
            "prob_100": round(prob_100, 6),
            "rank": rank,
            "model_date": model_date,
            "days_held": 0,
            "current_close": round(buy_price, 2),
            "current_gain": 0.0,
            "max_gain": 0.0,
            "max_drawdown": 0.0,
            "sell_weight": 0.0,
            "sell_reason": "",
        }

        positions = pd.concat([positions, pd.DataFrame([new_row])], ignore_index=True)
        self._save_positions(positions)

        # 记录买入交易
        self._record_trade(
            symbol=symbol, name=name, industry=industry,
            trade_date=buy_date, direction="buy",
            price=buy_price, shares=shares,
            reason="buy_signal",
            prob_200=prob_200, prob_100=prob_100,
        )

        logger.info(f"✅ 买入 {symbol} ({name}) {shares}股 @ {buy_price:.2f}, "
                     f"成本 {cost:.0f}")
        return True

    def sell(
        self,
        symbol: str,
        sell_date: str,
        sell_price: float,
        reason: str = "exit_signal",
        sell_weight: float = 0.0,
    ) -> dict | None:
        """
        卖出: 移除持仓, 记录交易。

        Returns:
            卖出交易详情 dict, 或 None (持仓不存在)
        """
        positions = self.get_positions()
        if positions.empty or symbol not in positions["symbol"].values:
            logger.warning(f"卖出失败: {symbol} 不在持仓中")
            return None

        row = positions[positions["symbol"] == symbol].iloc[0]
        buy_price = float(row["buy_price"])
        shares = int(row["shares"])
        pnl = (sell_price - buy_price) * shares
        pnl_pct = (sell_price - buy_price) / buy_price if buy_price > 0 else 0.0

        trade_info = {
            "symbol": symbol,
            "name": row.get("name", ""),
            "buy_price": buy_price,
            "sell_price": sell_price,
            "shares": shares,
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 4),
            "holding_days": int(row.get("days_held", 0)),
            "reason": reason,
            "max_gain": float(row.get("max_gain", 0)),
        }

        # 移除持仓
        positions = positions[positions["symbol"] != symbol].reset_index(drop=True)
        self._save_positions(positions)

        # 记录卖出交易
        self._record_trade(
            symbol=symbol, name=row.get("name", ""),
            industry=row.get("industry", ""),
            trade_date=sell_date, direction="sell",
            price=sell_price, shares=shares,
            reason=reason,
            pnl=pnl, pnl_pct=pnl_pct,
            holding_days=int(row.get("days_held", 0)),
            prob_200=float(row.get("prob_200", 0)),
            prob_100=float(row.get("prob_100", 0)),
            sell_weight=sell_weight,
        )

        emoji = "📈" if pnl >= 0 else "📉"
        logger.info(f"{emoji} 卖出 {symbol} ({row.get('name', '')}) {shares}股 @ {sell_price:.2f}, "
                     f"盈亏 {pnl:+.0f} ({pnl_pct:+.1%}), 持有 {trade_info['holding_days']}天, "
                     f"原因: {reason}")

        return trade_info

    def update_prices(
        self,
        current_date: str,
        data_dir: str,
        calendar: list[str],
    ) -> int:
        """更新所有持仓的当前价格和涨幅。"""
        positions = self.get_positions()
        if positions.empty:
            return 0

        date_set = set(calendar)
        n_updated = 0

        for idx, row in positions.iterrows():
            sym = row["symbol"]
            buy_price = float(row["buy_price"])
            buy_date = str(row["buy_date"])

            # 持有天数
            if buy_date in date_set and current_date in date_set:
                b_idx = calendar.index(buy_date)
                c_idx = calendar.index(current_date)
                days = c_idx - b_idx
            else:
                days = int(row.get("days_held", 0))

            # 当前价格
            current_close = _get_price(data_dir, sym, current_date)
            if current_close is None or current_close <= 0:
                continue

            current_gain = (current_close - buy_price) / buy_price
            old_max = float(row.get("max_gain", 0))
            max_gain = max(old_max, current_gain)
            # 最大回撤 = 从最高点的回落
            if max_gain > 0:
                max_dd = (max_gain - current_gain)
            else:
                max_dd = abs(min(0.0, current_gain))

            positions.at[idx, "days_held"] = days
            positions.at[idx, "current_close"] = round(current_close, 2)
            positions.at[idx, "current_gain"] = round(current_gain, 4)
            positions.at[idx, "max_gain"] = round(max_gain, 4)
            positions.at[idx, "max_drawdown"] = round(max_dd, 4)
            n_updated += 1

        self._save_positions(positions)
        return n_updated

    def update_sell_weights(self, sell_weights: pd.DataFrame):
        """
        更新 Agent 6 生成的卖出权重。

        Args:
            sell_weights: DataFrame (symbol, sell_weight, sell_reason)
        """
        positions = self.get_positions()
        if positions.empty or sell_weights.empty:
            return

        weight_map = dict(zip(sell_weights["symbol"], sell_weights["sell_weight"]))
        reason_map = dict(zip(sell_weights["symbol"], sell_weights.get("sell_reason", "")))

        for idx, row in positions.iterrows():
            sym = row["symbol"]
            if sym in weight_map:
                positions.at[idx, "sell_weight"] = round(float(weight_map[sym]), 4)
                positions.at[idx, "sell_reason"] = str(reason_map.get(sym, ""))

        self._save_positions(positions)

    def snapshot_daily_pnl(self, date: str):
        """记录当日持仓盈亏快照。"""
        positions = self.get_positions()
        if positions.empty:
            return

        n_positions = len(positions)
        total_value = (positions["current_close"] * positions["shares"]).sum()
        total_cost = (positions["buy_price"] * positions["shares"]).sum()
        daily_pnl = total_value - total_cost

        # 读取历史
        pnl_df = self._load_daily_pnl()
        cumulative_pnl = daily_pnl  # 简化: 只看持仓部分
        cumulative_pnl_pct = daily_pnl / total_cost if total_cost > 0 else 0.0

        new_row = {
            "date": date,
            "n_positions": n_positions,
            "total_value": round(total_value, 2),
            "total_cost": round(total_cost, 2),
            "daily_pnl": round(daily_pnl, 2),
            "cumulative_pnl": round(cumulative_pnl, 2),
            "cumulative_pnl_pct": round(cumulative_pnl_pct, 4),
        }

        pnl_df = pd.concat([pnl_df, pd.DataFrame([new_row])], ignore_index=True)
        # 去重 (同一天只保留最新)
        pnl_df = pnl_df.drop_duplicates(subset=["date"], keep="last")
        pnl_df.to_csv(self.daily_pnl_path, index=False, encoding="utf-8-sig")

    def get_trades(self) -> pd.DataFrame:
        """获取交易记录。"""
        if os.path.exists(self.trades_path):
            try:
                return pd.read_csv(self.trades_path, dtype={"trade_date": str, "buy_date": str})
            except Exception:
                pass
        return pd.DataFrame(columns=TRADE_COLS)

    def get_trade_stats(self) -> dict:
        """统计交易绩效。"""
        trades = self.get_trades()
        sells = trades[trades["direction"] == "sell"]
        if sells.empty:
            return {"n_trades": 0}

        n = len(sells)
        wins = sells[sells["pnl"] > 0]
        losses = sells[sells["pnl"] < 0]

        return {
            "n_trades": n,
            "win_rate": round(len(wins) / n, 4) if n > 0 else 0.0,
            "avg_pnl_pct": round(float(sells["pnl_pct"].mean()), 4),
            "total_pnl": round(float(sells["pnl"].sum()), 2),
            "avg_holding_days": round(float(sells["holding_days"].mean()), 1),
            "best_trade": round(float(sells["pnl_pct"].max()), 4),
            "worst_trade": round(float(sells["pnl_pct"].min()), 4),
        }

    # ─── 内部方法 ───

    def _save_positions(self, df: pd.DataFrame):
        df.to_csv(self.positions_path, index=False, encoding="utf-8-sig")

    def _record_trade(self, **kwargs):
        trades = self.get_trades()
        # 填充缺失列
        for col in TRADE_COLS:
            if col not in kwargs:
                kwargs[col] = "" if col in ("symbol", "name", "industry", "direction", "reason") else 0
        new_row = {k: kwargs.get(k, "") for k in TRADE_COLS}
        trades = pd.concat([trades, pd.DataFrame([new_row])], ignore_index=True)
        trades.to_csv(self.trades_path, index=False, encoding="utf-8-sig")

    def _load_daily_pnl(self) -> pd.DataFrame:
        if os.path.exists(self.daily_pnl_path):
            try:
                return pd.read_csv(self.daily_pnl_path, dtype={"date": str})
            except Exception:
                pass
        return pd.DataFrame(columns=DAILY_PNL_COLS)


def _get_price(data_dir: str, symbol: str, date: str, col: str = "close") -> float | None:
    """获取某只股票在某日的价格。"""
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
