#!/usr/bin/env python3
"""
Adaptive State Machine — P&L 验证脚本

从已有的 Phase 1 v3 信号文件中提取 BREAKOUT + ACCUMULATION 信号，
用 tushare 日线数据计算持有期真实收益，统计 Sharpe / 胜率 / 最大回撤。

用法:
  .venv/bin/python -m src.strategy.adaptive_state_machine.verify_pnl \
    --signal-dir results/adaptive_state_machine_phase1_backtest_v3 \
    --data-dir /mnt/nvme2/xtang/gp-workspace/gp-data/tushare-daily-full \
    --top-n 10 \
    --hold-days 10,20
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ── 路径 ──
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@dataclass
class Trade:
    symbol: str
    state: str
    signal_date: str
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    hold_days: int
    pnl_pct: float
    composite_score: float
    confidence: float


def load_trade_calendar() -> list[str]:
    """加载交易日历，返回排序后的交易日列表 (YYYYMMDD str)"""
    cal_path = "/mnt/nvme2/xtang/gp-workspace/gp-data/tushare-trade_cal/trade_cal.csv"
    if not os.path.exists(cal_path):
        return []
    df = pd.read_csv(cal_path)
    df = df[df["is_open"] == 1].sort_values("cal_date")
    return df["cal_date"].astype(str).str.zfill(8).tolist()


def find_forward_date(trade_dates: list[str], start_date: str, n_days: int) -> str | None:
    """在交易日历中找到 start_date 之后第 n_days 个交易日的日期"""
    try:
        idx = trade_dates.index(start_date)
    except ValueError:
        return None
    target_idx = idx + n_days
    if target_idx < len(trade_dates):
        return trade_dates[target_idx]
    return None


def load_daily_price(data_dir: str, symbol: str) -> pd.DataFrame | None:
    """加载单只股票的日线数据，返回按 trade_date 排序的 DataFrame"""
    fpath = os.path.join(data_dir, f"{symbol}.csv")
    if not os.path.exists(fpath):
        return None
    try:
        df = pd.read_csv(fpath)
    except Exception:
        return None
    if "trade_date" not in df.columns or "close" not in df.columns:
        return None
    df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "")
    df = df.sort_values("trade_date").reset_index(drop=True)
    return df[["trade_date", "close", "open"]]


def load_signals(signal_dir: str) -> dict[str, pd.DataFrame]:
    """
    加载所有信号文件，返回 {signal_date_str: DataFrame}。
    DataFrame 列: symbol, trade_date, state, confidence, aq_score, bq_score, composite_score
    """
    signals_by_date = {}
    signal_files = sorted(Path(signal_dir).glob("signals_*.csv"))
    for fpath in signal_files:
        # 从文件名提取日期: signals_20250102.csv
        date_str = fpath.stem.split("_", 1)[1]
        df = pd.read_csv(fpath)
        if df.empty:
            continue
        signals_by_date[date_str] = df
    return signals_by_date


def run_pnl_verify(
    signal_dir: str,
    data_dir: str,
    top_n: int = 10,
    hold_days: int = 10,
) -> list[Trade]:
    """
    执行 P&L 验证。

    对每个扫描日期:
    1. 筛选 BREAKOUT + ACCUMULATION 信号
    2. 按 composite_score 取 top N
    3. 用次日开盘价入场（避免当日成交假设）
    4. 持有 N 个交易日后以收盘价出场
    5. 记录逐笔交易
    """
    # 加载交易日历
    trade_dates = load_trade_calendar()
    if not trade_dates:
        print("警告: 无法加载交易日历，将用日线数据推断交易日")
        trade_dates = None

    # 加载信号
    signals_by_date = load_signals(signal_dir)
    if not signals_by_date:
        print(f"错误: {signal_dir} 中未找到信号文件")
        return []

    print(f"加载了 {len(signals_by_date)} 个扫描日期的信号")

    trades = []

    # 缓存日线数据（避免重复加载）
    daily_cache: dict[str, pd.DataFrame] = {}

    for signal_date in sorted(signals_by_date.keys()):
        sig_df = signals_by_date[signal_date]

        # 筛选 BREAKOUT + ACCUMULATION
        pos_states = {"breakout", "accumulation"}
        pos_df = sig_df[sig_df["state"].str.lower().isin(pos_states)]

        if pos_df.empty:
            continue

        # 按 composite_score 降序取 top N
        top_df = pos_df.nlargest(top_n, "composite_score")

        # 找到入场日期：信号日期的下一个交易日
        if trade_dates:
            entry_date = find_forward_date(trade_dates, signal_date, 1)
            exit_date = find_forward_date(trade_dates, signal_date, hold_days + 1)
        else:
            # 用第一个有数据的股票的日期推断
            entry_date = None
            exit_date = None

        if entry_date is None:
            print(f"  {signal_date}: 无法确定入/出场日期，跳过")
            continue

        # 逐只处理
        for _, row in top_df.iterrows():
            symbol = row["symbol"]

            # 加载日线
            if symbol not in daily_cache:
                daily_cache[symbol] = load_daily_price(data_dir, symbol)
            daily = daily_cache[symbol]

            if daily is None:
                continue

            # 找入场价（次日开盘价）
            entry_row = daily[daily["trade_date"] == entry_date]
            if entry_row.empty:
                continue
            entry_price = float(entry_row.iloc[0]["open"])
            if entry_price <= 0:
                continue

            # 找出场价（第 N 个交易日收盘价）
            exit_row = daily[daily["trade_date"] == exit_date]
            if exit_row.empty:
                # 如果超过了数据范围，用最后一天收盘价
                last_row = daily.iloc[-1]
                if last_row["trade_date"] >= entry_date:
                    exit_date_actual = str(last_row["trade_date"])
                    exit_price = float(last_row["close"])
                else:
                    continue
            else:
                exit_date_actual = exit_date
                exit_price = float(exit_row.iloc[0]["close"])
                if exit_price <= 0:
                    continue

            pnl_pct = (exit_price - entry_price) / entry_price * 100

            # 计算实际持仓天数
            if trade_dates:
                try:
                    e_idx = trade_dates.index(entry_date)
                    x_idx = trade_dates.index(exit_date_actual)
                    actual_hold = x_idx - e_idx
                except ValueError:
                    actual_hold = hold_days
            else:
                actual_hold = hold_days

            trades.append(Trade(
                symbol=symbol,
                state=row["state"],
                signal_date=signal_date,
                entry_date=entry_date,
                entry_price=round(entry_price, 2),
                exit_date=exit_date_actual,
                exit_price=round(exit_price, 2),
                hold_days=actual_hold,
                pnl_pct=round(pnl_pct, 2),
                composite_score=round(float(row["composite_score"]), 4),
                confidence=round(float(row["confidence"]), 4),
            ))

    return trades


def analyze_and_print(trades: list[Trade], top_n: int, hold_days: int):
    """分析 P&L 结果并打印摘要"""
    if not trades:
        print("\n没有产生任何交易，信号可能未匹配到日线数据。")
        return

    pnl_values = [t.pnl_pct for t in trades]
    pnl_arr = np.array(pnl_values)

    n = len(trades)
    wins = int(np.sum(pnl_arr > 0))
    losses = int(np.sum(pnl_arr < 0))
    win_rate = wins / n * 100

    avg_pnl = float(np.mean(pnl_arr))
    med_pnl = float(np.median(pnl_arr))
    std_pnl = float(np.std(pnl_arr))

    avg_win = float(np.mean(pnl_arr[pnl_arr > 0])) if wins > 0 else 0
    avg_loss = float(np.mean(pnl_arr[pnl_arr < 0])) if losses > 0 else 0

    best = float(np.max(pnl_arr))
    worst = float(np.min(pnl_arr))

    # 年化夏普比率（用权益曲线的日收益率计算）
    # 先构建权益曲线和最大回撤，再算 Sharpe
    trade_dates = load_trade_calendar()

    # 收集所有需要的日期范围
    if trade_dates:
        all_dates_min = min(t.entry_date for t in trades)
        all_dates_max = max(t.exit_date for t in trades)
        active_dates = [d for d in trade_dates if all_dates_min <= d <= all_dates_max]
    else:
        active_dates = sorted(set(t.entry_date for t in trades) | set(t.exit_date for t in trades))

    # 逐日计算组合收益（等权持仓平均日收益率）
    daily_cache: dict[str, pd.DataFrame] = {}
    data_dir = "/mnt/nvme2/xtang/gp-workspace/gp-data/tushare-daily-full"

    equity_curve = [(active_dates[0], 1.0)] if active_dates else []
    equity = 1.0

    for i in range(1, len(active_dates)):
        today = active_dates[i]
        prev = active_dates[i - 1]

        # 找出当天持仓的所有交易
        day_returns = []
        for t in trades:
            if t.entry_date <= today <= t.exit_date:
                sym = t.symbol
                if sym not in daily_cache:
                    daily_cache[sym] = load_daily_price(data_dir, sym)
                daily = daily_cache[sym]
                if daily is None:
                    continue

                prev_row = daily[daily["trade_date"] == prev]
                today_row = daily[daily["trade_date"] == today]
                if not prev_row.empty and not today_row.empty:
                    prev_close = float(prev_row.iloc[0]["close"])
                    today_close = float(today_row.iloc[0]["close"])
                    if prev_close > 0:
                        day_returns.append((today_close - prev_close) / prev_close)

        if day_returns:
            day_ret = np.mean(day_returns)
            equity *= (1 + day_ret)
        equity_curve.append((today, equity))

    # 年化夏普比率（用权益曲线的日收益率计算）
    if equity_curve and len(equity_curve) > 2:
        daily_rets = []
        for i in range(1, len(equity_curve)):
            prev_eq = equity_curve[i - 1][1]
            curr_eq = equity_curve[i][1]
            if prev_eq > 0:
                daily_rets.append((curr_eq - prev_eq) / prev_eq)
        daily_rets = np.array(daily_rets)
        sharpe_daily = float(np.mean(daily_rets) / np.std(daily_rets)) if np.std(daily_rets) > 0 else 0
        sharpe_annual = sharpe_daily * np.sqrt(252)
    else:
        sharpe_daily = 0
        sharpe_annual = 0

    # 最大回撤
    if equity_curve:
        peak = equity_curve[0][1]
        max_dd = 0.0
        max_dd_date = ""
        for d, v in equity_curve:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
                max_dd_date = d
    else:
        max_dd = 0
        max_dd_date = "N/A"

    # 按状态分组
    state_pnl: dict[str, list[float]] = {}
    for t in trades:
        state_pnl.setdefault(t.state, []).append(t.pnl_pct)

    # 月度收益
    df_trades = pd.DataFrame([
        {
            "symbol": t.symbol,
            "state": t.state,
            "signal_date": t.signal_date,
            "entry_date": t.entry_date,
            "exit_date": t.exit_date,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "hold_days": t.hold_days,
            "pnl_pct": t.pnl_pct,
            "composite_score": t.composite_score,
            "confidence": t.confidence,
        }
        for t in trades
    ])
    df_trades["exit_month"] = df_trades["exit_date"].str[:6]
    monthly = df_trades.groupby("exit_month").agg(
        trades=("pnl_pct", "count"),
        avg_pnl=("pnl_pct", "mean"),
        total_pnl=("pnl_pct", "sum"),
        win_rate=("pnl_pct", lambda x: (x > 0).mean() * 100),
    ).round(2)

    first_date = min(t.signal_date for t in trades)
    last_date = max(t.exit_date for t in trades)

    print(f"\n{'='*65}")
    print(f"  Adaptive State Machine — P&L 验证报告")
    print(f"  Top-{top_n} | 持有 {hold_days} 天")
    print(f"  区间: {first_date} → {last_date}")
    print(f"{'='*65}")

    print(f"\n总体绩效")
    print(f"  {'总交易数':12s}: {n}")
    print(f"  {'胜率':12s}: {wins}/{n} = {win_rate:.1f}%")
    print(f"  {'平均收益':12s}: {avg_pnl:+.2f}%")
    print(f"  {'中位数收益':12s}: {med_pnl:+.2f}%")
    print(f"  {'收益标准差':10s}: {std_pnl:.2f}%")
    print(f"  {'平均盈利':12s}: +{avg_win:.2f}%")
    print(f"  {'平均亏损':12s}: {avg_loss:.2f}%")
    if avg_loss != 0:
        print(f"  {'盈亏比':12s}: {abs(avg_win / avg_loss):.2f}")
    print(f"  {'最大单笔盈利':10s}: +{best:.2f}%")
    print(f"  {'最大单笔亏损':10s}: {worst:.2f}%")

    print(f"\n风险指标")
    print(f"  {'日 Sharpe':12s}: {sharpe_daily:.4f}")
    print(f"  {'年化 Sharpe':10s}: {sharpe_annual:.4f}")
    print(f"  {'最大回撤':12s}: {max_dd * 100:.2f}% (截至 {max_dd_date})")

    # 关键判断标准
    print(f"\n{'='*65}")
    if sharpe_annual > 1.0:
        print(f"  结论: Sharpe {sharpe_annual:.2f} > 1.0 → 纳入策略池，继续优化")
    elif sharpe_annual > 0.5:
        print(f"  结论: Sharpe {sharpe_annual:.2f} 在 0.5~1.0 之间 → 作为辅助信号，继续优化")
    else:
        print(f"  结论: Sharpe {sharpe_annual:.2f} < 0.5 → 模型无实际交易价值，建议停止投入")
    print(f"{'='*65}")

    print(f"\n按状态分组")
    for state in sorted(state_pnl.keys()):
        pnl_list = np.array(state_pnl[state])
        s_n = len(pnl_list)
        s_win = int(np.sum(pnl_list > 0))
        s_avg = float(np.mean(pnl_list))
        print(f"  {state:15s}: {s_n:4d} 笔, 胜率 {s_win/s_n*100:5.1f}%, 平均收益 {s_avg:+.2f}%")

    print(f"\n月度收益")
    print(monthly.to_string())

    # Top 10 最佳
    print(f"\nTop 10 最佳交易")
    top10 = df_trades.nlargest(10, "pnl_pct")[["symbol", "state", "signal_date", "entry_date", "exit_date", "pnl_pct", "composite_score"]]
    print(top10.to_string(index=False))

    # Bottom 10 最差
    print(f"\nTop 10 最差交易")
    bot10 = df_trades.nsmallest(10, "pnl_pct")[["symbol", "state", "signal_date", "entry_date", "exit_date", "pnl_pct", "composite_score"]]
    print(bot10.to_string(index=False))

    print(f"\n{'='*65}\n")


def save_results(trades: list[Trade], out_dir: str, top_n: int, hold_days: int):
    """保存逐笔交易记录"""
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame([
        {
            "symbol": t.symbol,
            "state": t.state,
            "signal_date": t.signal_date,
            "entry_date": t.entry_date,
            "entry_price": t.entry_price,
            "exit_date": t.exit_date,
            "exit_price": t.exit_price,
            "hold_days": t.hold_days,
            "pnl_pct": t.pnl_pct,
            "composite_score": t.composite_score,
            "confidence": t.confidence,
        }
        for t in trades
    ])

    tag = f"top{top_n}_hold{hold_days}d"
    df.to_csv(os.path.join(out_dir, f"pnl_verify_{tag}.csv"), index=False)
    print(f"\n结果已保存至 {out_dir}/pnl_verify_{tag}.csv")


def main():
    parser = argparse.ArgumentParser(description="Adaptive State Machine P&L 验证")
    parser.add_argument(
        "--signal-dir",
        default="results/adaptive_state_machine_phase1_backtest_v3",
        help="信号文件目录",
    )
    parser.add_argument(
        "--data-dir",
        default="/mnt/nvme2/xtang/gp-workspace/gp-data/tushare-daily-full",
        help="tushare 日线数据目录",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="每个扫描日选取的信号数量",
    )
    parser.add_argument(
        "--hold-days",
        type=int,
        default=10,
        help="持有天数",
    )
    parser.add_argument(
        "--out-dir",
        default="results/adaptive_state_machine_pnl_verify",
        help="输出目录",
    )
    args = parser.parse_args()

    print(f"信号目录: {args.signal_dir}")
    print(f"数据目录: {args.data_dir}")
    print(f"Top N: {args.top_n}, 持有: {args.hold_days} 天")

    trades = run_pnl_verify(
        signal_dir=args.signal_dir,
        data_dir=args.data_dir,
        top_n=args.top_n,
        hold_days=args.hold_days,
    )

    analyze_and_print(trades, args.top_n, args.hold_days)
    save_results(trades, args.out_dir, args.top_n, args.hold_days)


if __name__ == "__main__":
    main()
