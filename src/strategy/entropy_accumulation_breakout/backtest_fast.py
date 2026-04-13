#!/usr/bin/env python3
"""
熵惜售分岔突破策略 — 高速回测脚本

优化点:
  1. 预加载全部股票数据到内存
  2. 预计算全部特征 (只算一次)
  3. 按日期切片评估信号，而非逐周期重新计算
  4. 多进程并行特征计算
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ── 路径修正 ──
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.strategy.entropy_accumulation_breakout.feature_engine import (
    build_features,
    compute_single_timeframe_features,
)
from src.strategy.entropy_accumulation_breakout.signal_detector import (
    DetectorConfig,
    accumulation_quality,
    bifurcation_quality,
    detect_accumulation,
    detect_bifurcation_breakout,
    detect_structural_collapse,
)
from src.strategy.entropy_accumulation_breakout.market_regime import (
    MarketRegime,
    build_market_regime_series,
    get_regime_on_date,
    REGIME_POSITION_SCALE,
)

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════
# 数据加载
# ═════════════════════════════════════════════════════════

def load_basic_info(basic_path: str) -> dict[str, dict]:
    if not basic_path or not os.path.exists(basic_path):
        return {}
    df = pd.read_csv(basic_path, dtype=str)
    info = {}
    for _, row in df.iterrows():
        ts = str(row.get("ts_code", ""))
        if ts.endswith(".SH"):
            sym = "sh" + ts[:6]
        elif ts.endswith(".SZ"):
            sym = "sz" + ts[:6]
        elif ts.endswith(".BJ"):
            sym = "bj" + ts[:6]
        else:
            continue
        info[sym] = {
            "name": str(row.get("name", "")),
            "industry": str(row.get("industry", "")),
        }
    return info


def resolve_symbols(data_dir: str, symbols: list[str] | None) -> list[str]:
    if symbols:
        return symbols
    csvs = sorted(Path(data_dir).glob("*.csv"))
    # 只取主板: sh60*, sz00*
    return [f.stem for f in csvs if f.stem[:4] in ("sh60", "sz00")]


def load_daily(data_dir: str, symbol: str) -> pd.DataFrame | None:
    fpath = os.path.join(data_dir, f"{symbol}.csv")
    if not os.path.exists(fpath):
        return None
    try:
        df = pd.read_csv(fpath)
    except Exception:
        return None
    if "trade_date" not in df.columns or "close" not in df.columns:
        return None
    df = df.sort_values("trade_date").reset_index(drop=True)
    return df


def should_skip(df: pd.DataFrame, info: dict, min_amount: float) -> bool:
    name = info.get("name", "")
    if "ST" in name or "退" in name:
        return True
    if len(df) < 250:
        return True
    recent = df.tail(20)
    avg_amount = recent["amount"].mean() if "amount" in recent.columns else 0
    return avg_amount < min_amount


# ═════════════════════════════════════════════════════════
# 单股票预计算
# ═════════════════════════════════════════════════════════

def _compute_one_symbol(args: tuple) -> tuple[str, pd.DataFrame | None]:
    """为单个股票计算特征 (用于多进程)"""
    data_dir, symbol, basic_info, min_amount = args
    df = load_daily(data_dir, symbol)
    if df is None:
        return symbol, None

    info = basic_info.get(symbol, {})
    if should_skip(df, info, min_amount):
        return symbol, None

    try:
        feats = build_features(df)
        df_feat = feats["daily"]
        df_feat["trade_date_str"] = df_feat["trade_date"].astype(str)

        # Pre-compute entry signals (causally safe — rolling backward-only)
        cfg = DetectorConfig()
        is_accum = detect_accumulation(df_feat, cfg)
        is_breakout = detect_bifurcation_breakout(df_feat, is_accum, cfg)
        aq = accumulation_quality(df_feat, cfg)
        bq = bifurcation_quality(df_feat, cfg)
        df_feat["is_breakout"] = is_breakout
        df_feat["entry_score"] = 0.4 * aq + 0.6 * bq

        # Precompute trend indicators for adaptive filtering
        close_f = df_feat["close"].astype(float)
        df_feat["ma20"] = close_f.rolling(20).mean()
        df_feat["ma60"] = close_f.rolling(60).mean()

        return symbol, df_feat
    except Exception:
        return symbol, None


# ═════════════════════════════════════════════════════════
# 高速回测核心
# ═════════════════════════════════════════════════════════

@dataclass
class BacktestResult:
    trades: list[dict] = field(default_factory=list)
    equity_curve: list[tuple[str, float]] = field(default_factory=list)


def _calc_equity(
    portfolio: dict, all_features: dict, all_date_idx: dict,
    today: str, cash: float,
) -> float:
    """根据当日收盘价计算组合总权益 (cash + 持仓市值)"""
    total = cash
    for sym, pos in portfolio.items():
        idx = all_date_idx[sym].get(today)
        if idx is not None:
            price = float(all_features[sym].loc[idx, "close"])
            total += pos["allocation"] * (price / pos["entry_price"])
        else:
            total += pos["allocation"]
    return total


def run_fast_backtest(
    data_dir: str,
    basic_path: str,
    start_date: str,
    end_date: str,
    max_positions: int = 5,
    min_amount: float = 500_000,
    n_workers: int = 8,
    symbols: list[str] | None = None,
    use_market_gate: bool = True,
    stop_loss_pct: float = -10.0,
    index_code: str = "000001_sh",
    scan_interval: int = 1,
    min_entry_score: float = 0.0,
    adaptive_regime: bool = True,
) -> BacktestResult:
    """
    组合跟踪式回测 — 纯理论驱动持有 / 退出.

    - 入场: 惜售吸筹 + 分岔突破信号
    - 退出: 结构崩塌 / 止损 (无时间限制)
    - 每日检查退出, 按 scan_interval 扫描新入场机会
    """
    basic_info = load_basic_info(basic_path)
    sym_list = resolve_symbols(data_dir, symbols)
    logger.info("Universe: %d symbols", len(sym_list))

    # ── Step 1: 并行预计算特征 + 信号 ──
    t0 = time.time()
    all_features: dict[str, pd.DataFrame] = {}
    all_date_idx: dict[str, dict[str, int]] = {}  # sym -> {date_str: df_index}
    tasks = [(data_dir, sym, basic_info, min_amount) for sym in sym_list]

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_compute_one_symbol, t): t[1] for t in tasks}
        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            if done_count % 500 == 0:
                logger.info("  Feature computed: %d / %d", done_count, len(tasks))
            sym, df_feat = future.result()
            if df_feat is not None:
                all_features[sym] = df_feat
                all_date_idx[sym] = dict(zip(df_feat["trade_date_str"], df_feat.index))

    t1 = time.time()
    logger.info("Feature + signal precompute done: %d stocks in %.1fs", len(all_features), t1 - t0)

    if not all_features:
        return BacktestResult()

    # ── Step 2: 构建交易日历 ──
    all_dates = set()
    for df in list(all_features.values())[:20]:
        all_dates.update(df["trade_date_str"].tolist())
    cal = sorted(all_dates)
    bt_dates = [d for d in cal if start_date <= d <= end_date]
    if not bt_dates:
        logger.warning("No trading dates in [%s, %s]", start_date, end_date)
        return BacktestResult()

    logger.info("Backtest: %d trading days in [%s, %s]", len(bt_dates), start_date, end_date)

    # ── Step 2.5: 预计算市场状态 ──
    regime_df = None
    regime_cache: dict[str, str] = {}
    if use_market_gate:
        logger.info("Building market regime series from index %s ...", index_code)
        regime_df = build_market_regime_series(index_code)
        logger.info("Market regime series built: %d rows", len(regime_df))
        regime_cache = dict(zip(regime_df["trade_date_str"], regime_df["regime"]))

    # ── Step 3: 逐日仿真 ──
    cfg = DetectorConfig()
    cfg.max_hold_days = 9999  # 纯理论驱动, 不设时间安全网

    result = BacktestResult()
    portfolio: dict[str, dict] = {}   # sym -> position
    cash = 1.0
    pending_entries: list[dict] = []  # 昨日信号, 今日入场
    skipped_by_regime = 0

    result.equity_curve.append((start_date, 1.0))

    for day_i, today in enumerate(bt_dates):

        # ── Adaptive regime parameters for today ──
        if adaptive_regime and regime_cache:
            cur_regime = regime_cache.get(today, "consolidation")
            is_weak_env = cur_regime in ("declining", "decline_ended", "rise_ending")
        else:
            cur_regime = "consolidation"
            is_weak_env = False

        # ── 3a. 执行待入场 (昨日信号 → 今日收盘价入场) ──
        for entry in pending_entries:
            sym = entry["symbol"]
            if sym in portfolio:
                continue
            idx = all_date_idx[sym].get(today)
            if idx is None:
                continue

            entry_price = float(all_features[sym].loc[idx, "close"])
            total_eq = _calc_equity(portfolio, all_features, all_date_idx, today, cash)
            allocation = total_eq / max_positions
            if allocation > cash:
                allocation = cash
            if allocation <= 0:
                continue

            cash -= allocation
            portfolio[sym] = {
                "entry_date": today,
                "entry_price": entry_price,
                "entry_idx": idx,
                "allocation": allocation,
                "hold_days": 0,
                "signal_date": entry["signal_date"],
                "score": entry["score"],
                "name": entry.get("name", ""),
                "industry": entry.get("industry", ""),
                "market_regime": entry.get("market_regime", "N/A"),
            }
        pending_entries = []

        # ── 3b. 每日检查退出 ──
        # Adaptive: 弱环境崩塌检测更敏感 (score>=2), 止损不变
        cur_collapse_score = 2 if is_weak_env else 3

        to_close: list[tuple] = []
        for sym in list(portfolio.keys()):
            pos = portfolio[sym]
            pos["hold_days"] += 1

            idx = all_date_idx[sym].get(today)
            if idx is None:
                continue

            today_close = float(all_features[sym].loc[idx, "close"])

            # 入场当天不退出
            if pos["hold_days"] <= 1:
                continue

            pnl_pct = (today_close - pos["entry_price"]) / pos["entry_price"] * 100

            # 止损
            if pnl_pct <= stop_loss_pct:
                to_close.append((sym, today, today_close, "stop_loss"))
                continue

            # 结构崩塌 (adaptive collapse score)
            df = all_features[sym]
            collapse = detect_structural_collapse(df, pos["entry_idx"], cfg,
                                                  min_collapse_score=cur_collapse_score)
            if bool(collapse.loc[idx]):
                to_close.append((sym, today, today_close, "collapse"))
                continue

        # 处理退出
        for sym, exit_date, exit_price, exit_reason in to_close:
            pos = portfolio.pop(sym)
            pnl_ratio = (exit_price - pos["entry_price"]) / pos["entry_price"]
            cash += pos["allocation"] * (1 + pnl_ratio)

            result.trades.append({
                "symbol": sym,
                "name": pos["name"],
                "industry": pos["industry"],
                "signal_date": pos["signal_date"],
                "entry_date": pos["entry_date"],
                "entry_price": round(pos["entry_price"], 2),
                "exit_date": exit_date,
                "exit_price": round(exit_price, 2),
                "exit_reason": exit_reason,
                "pnl_pct": round(pnl_ratio * 100, 2),
                "hold_days": pos["hold_days"],
                "score": round(pos["score"], 4),
                "market_regime": pos["market_regime"],
            })

        # ── 3c. 记录净值 ──
        total_equity = _calc_equity(portfolio, all_features, all_date_idx, today, cash)
        if day_i % 5 == 0 or to_close:
            result.equity_curve.append((today, total_equity))

        # ── 3d. 扫描新入场机会 ──
        open_slots = max_positions - len(portfolio)
        if open_slots <= 0:
            continue

        # 市场门控
        regime_str = "N/A"
        pos_scale = 1.0
        if regime_df is not None:
            mstate = get_regime_on_date(regime_df, today)
            regime_str = mstate.regime.value
            pos_scale = mstate.position_scale
            if pos_scale <= 0:
                skipped_by_regime += 1
                continue

        # 按 scan_interval 扫描
        if day_i % scan_interval != 0:
            continue

        # 查找突破候选
        candidates = []
        for sym, df in all_features.items():
            if sym in portfolio:
                continue
            idx = all_date_idx[sym].get(today)
            if idx is None:
                continue
            row = df.loc[idx]
            if not bool(row.get("is_breakout", False)):
                continue
            score = float(row.get("entry_score", 0))
            if score < min_entry_score:
                continue

            # 个股趋势过滤 (adaptive)
            if adaptive_regime:
                close_val = float(row.get("close", 0))
                ma20_val = row.get("ma20", np.nan)
                ma60_val = row.get("ma60", np.nan)
                # 始终要求股价 > MA20
                if pd.notna(ma20_val) and close_val < float(ma20_val):
                    continue
                # 弱环境额外要求 > MA60
                if is_weak_env and pd.notna(ma60_val) and close_val < float(ma60_val):
                    continue

            candidates.append({
                "symbol": sym,
                "score": score,
                "name": basic_info.get(sym, {}).get("name", ""),
                "industry": basic_info.get(sym, {}).get("industry", ""),
                "signal_date": today,
                "market_regime": regime_str,
            })

        if not candidates:
            continue

        candidates.sort(key=lambda x: x["score"], reverse=True)
        effective_slots = max(1, int(open_slots * pos_scale))
        pending_entries = candidates[:effective_slots]

    # ── 回测结束: 强制平仓剩余持仓 ──
    last_date = bt_dates[-1]
    for sym in list(portfolio.keys()):
        pos = portfolio[sym]
        idx = all_date_idx[sym].get(last_date)
        if idx is not None:
            exit_price = float(all_features[sym].loc[idx, "close"])
        else:
            exit_price = pos["entry_price"]

        pnl_ratio = (exit_price - pos["entry_price"]) / pos["entry_price"]
        cash += pos["allocation"] * (1 + pnl_ratio)

        result.trades.append({
            "symbol": sym,
            "name": pos["name"],
            "industry": pos["industry"],
            "signal_date": pos["signal_date"],
            "entry_date": pos["entry_date"],
            "entry_price": round(pos["entry_price"], 2),
            "exit_date": last_date,
            "exit_price": round(exit_price, 2),
            "exit_reason": "end_of_backtest",
            "pnl_pct": round(pnl_ratio * 100, 2),
            "hold_days": pos["hold_days"],
            "score": round(pos["score"], 4),
            "market_regime": pos["market_regime"],
        })
    portfolio.clear()
    result.equity_curve.append((last_date, cash))

    if skipped_by_regime > 0:
        logger.info("Days skipped by market regime gate: %d", skipped_by_regime)
    logger.info("Total trades: %d", len(result.trades))

    return result


# ═════════════════════════════════════════════════════════
# 结果分析与输出
# ═════════════════════════════════════════════════════════

def analyze_and_print(result: BacktestResult, start_date: str, end_date: str):
    """分析回测结果并打印摘要"""
    trades = result.trades
    if not trades:
        print("\n没有产生任何交易。策略在此期间未找到满足条件的突破信号。")
        return

    df_trades = pd.DataFrame(trades)
    n = len(df_trades)
    wins = (df_trades["pnl_pct"] > 0).sum()
    losses = (df_trades["pnl_pct"] < 0).sum()
    avg_pnl = df_trades["pnl_pct"].mean()
    med_pnl = df_trades["pnl_pct"].median()
    total_pnl = df_trades["pnl_pct"].sum()

    avg_win = df_trades.loc[df_trades["pnl_pct"] > 0, "pnl_pct"].mean() if wins > 0 else 0
    avg_loss = df_trades.loc[df_trades["pnl_pct"] < 0, "pnl_pct"].mean() if losses > 0 else 0
    best = df_trades["pnl_pct"].max()
    worst = df_trades["pnl_pct"].min()

    # 净值曲线
    eq_curve = result.equity_curve
    if len(eq_curve) > 1:
        final_eq = eq_curve[-1][1]
        total_return = (final_eq - 1.0) * 100

        # 最大回撤
        eq_vals = [e[1] for e in eq_curve]
        peak = eq_vals[0]
        max_dd = 0
        for v in eq_vals:
            peak = max(peak, v)
            dd = (peak - v) / peak
            max_dd = max(max_dd, dd)

        # 年化
        n_periods = len(eq_curve) - 1
    else:
        final_eq = 1.0
        total_return = 0
        max_dd = 0

    # 月度收益
    df_trades["entry_month"] = df_trades["entry_date"].astype(str).str[:6]
    monthly = df_trades.groupby("entry_month").agg(
        trades=("pnl_pct", "count"),
        avg_pnl=("pnl_pct", "mean"),
        total_pnl=("pnl_pct", "sum"),
        win_rate=("pnl_pct", lambda x: (x > 0).mean() * 100),
    ).round(2)

    # 退出原因分布
    exit_dist = df_trades["exit_reason"].value_counts()

    # 行业分布
    industry_pnl = df_trades.groupby("industry")["pnl_pct"].agg(["count", "mean"]).sort_values("mean", ascending=False).head(15)

    print(f"\n{'='*65}")
    print(f"  熵惜售分岔突破策略 — 2025 年回测报告")
    print(f"  回测区间: {start_date} → {end_date}")
    print(f"{'='*65}")

    avg_hold = df_trades["hold_days"].mean()
    med_hold = df_trades["hold_days"].median()
    max_hold = df_trades["hold_days"].max()

    print(f"\n📊 总体绩效")
    print(f"  {'总交易数':12s}: {n}")
    print(f"  {'胜率':12s}: {wins}/{n} = {wins/n*100:.1f}%")
    print(f"  {'平均收益':12s}: {avg_pnl:.2f}%")
    print(f"  {'中位数收益':12s}: {med_pnl:.2f}%")
    print(f"  {'平均盈利':12s}: +{avg_win:.2f}%")
    print(f"  {'平均亏损':12s}: {avg_loss:.2f}%")
    print(f"  {'盈亏比':12s}: {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "")
    print(f"  {'最大单笔盈利':12s}: +{best:.2f}%")
    print(f"  {'最大单笔亏损':12s}: {worst:.2f}%")
    print(f"  {'平均持仓天数':12s}: {avg_hold:.1f}")
    print(f"  {'中位持仓天数':12s}: {med_hold:.0f}")
    print(f"  {'最长持仓天数':12s}: {max_hold:.0f}")

    print(f"\n📈 净值曲线")
    print(f"  {'最终净值':12s}: {final_eq:.4f}")
    print(f"  {'总回报率':12s}: {total_return:+.2f}%")
    print(f"  {'最大回撤':12s}: {max_dd*100:.2f}%")

    print(f"\n📅 月度收益")
    print(monthly.to_string())

    print(f"\n🚪 退出原因分布")
    for reason, count in exit_dist.items():
        print(f"  {reason}: {count} ({count/n*100:.1f}%)")

    print(f"\n🏭 行业表现 (Top 15)")
    print(industry_pnl.to_string())

    # 市场状态分布 (如果有)
    if "market_regime" in df_trades.columns:
        print(f"\n🌐 按市场状态统计")
        regime_stats = df_trades.groupby("market_regime").agg(
            trades=("pnl_pct", "count"),
            avg_pnl=("pnl_pct", "mean"),
            win_rate=("pnl_pct", lambda x: (x > 0).mean() * 100),
        ).round(2)
        print(regime_stats.to_string())

    # Top 10 最佳交易
    print(f"\n🏆 Top 10 最佳交易")
    top10 = df_trades.nlargest(10, "pnl_pct")[["symbol", "name", "entry_date", "exit_date", "pnl_pct", "exit_reason", "score"]]
    print(top10.to_string(index=False))

    # Bottom 10 最差交易
    print(f"\n💔 Top 10 最差交易")
    bot10 = df_trades.nsmallest(10, "pnl_pct")[["symbol", "name", "entry_date", "exit_date", "pnl_pct", "exit_reason", "score"]]
    print(bot10.to_string(index=False))

    print(f"\n{'='*65}\n")


def save_results(result: BacktestResult, out_dir: str, start_date: str, end_date: str):
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{start_date}_{end_date}"

    if result.trades:
        pd.DataFrame(result.trades).to_csv(
            os.path.join(out_dir, f"backtest_trades_{tag}.csv"), index=False
        )

    if result.equity_curve:
        pd.DataFrame(result.equity_curve, columns=["date", "equity"]).to_csv(
            os.path.join(out_dir, f"backtest_equity_{tag}.csv"), index=False
        )

    logger.info("Results saved to %s", out_dir)


# ═════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="高速回测 — 熵惜售分岔突破策略")
    parser.add_argument("--data_dir", default="/nvme5/xtang/gp-workspace/gp-data/tushare-daily-full")
    parser.add_argument("--basic_path", default="/nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv")
    parser.add_argument("--out_dir", default="results/entropy_accumulation_breakout/backtest_2025")
    parser.add_argument("--start_date", default="20250101")
    parser.add_argument("--end_date", default="20251231")
    parser.add_argument("--max_positions", type=int, default=5)
    parser.add_argument("--min_amount", type=float, default=500_000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument("--scan_interval", type=int, default=1, help="入场扫描间隔天数 (默认1=每日)")
    parser.add_argument("--min_score", type=float, default=0.30, help="最低入场分数门槛 (默认0.30)")
    parser.add_argument("--no_market_gate", action="store_true", help="禁用市场状态门控")
    parser.add_argument("--no_adaptive", action="store_true", help="禁用市场环境自适应 (趋势过滤/动态止损/自适应崩塌)")
    parser.add_argument("--stop_loss", type=float, default=-10.0, help="止损百分比 (默认 -10%%)")
    parser.add_argument("--index_code", type=str, default="000001_sh", help="指数代码")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] if args.symbols else None

    t_start = time.time()
    result = run_fast_backtest(
        data_dir=args.data_dir,
        basic_path=args.basic_path,
        start_date=args.start_date,
        end_date=args.end_date,
        max_positions=args.max_positions,
        min_amount=args.min_amount,
        n_workers=args.workers,
        symbols=symbols,
        use_market_gate=not args.no_market_gate,
        stop_loss_pct=args.stop_loss,
        index_code=args.index_code,
        scan_interval=args.scan_interval,
        min_entry_score=args.min_score,
        adaptive_regime=not args.no_adaptive,
    )
    t_end = time.time()
    logger.info("Total backtest time: %.1fs", t_end - t_start)

    analyze_and_print(result, args.start_date, args.end_date)
    save_results(result, args.out_dir, args.start_date, args.end_date)


if __name__ == "__main__":
    main()
