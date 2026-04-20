"""
Bull Hunter v3 — CLI 入口

v4 模式:
  # 每日预测 (轻量, 复用 latest 模型)
  python -m src.strategy.factor_model_selection.v3_bull_hunter.run_bull_hunter \
      --daily --scan_date 20260420

  # 周训 (训练新模型, 更新 latest)
  python -m src.strategy.factor_model_selection.v3_bull_hunter.run_bull_hunter \
      --train --scan_date 20260420

  # 复盘 (Agent 4 双回路评估 + 可能触发重训)
  python -m src.strategy.factor_model_selection.v3_bull_hunter.run_bull_hunter \
      --review --scan_date 20260420

  # Live 模式 (Agent 5/6/7 持仓管理闭环)
  python -m src.strategy.factor_model_selection.v3_bull_hunter.run_bull_hunter \
      --live --scan_date 20260420

  # Live 回测 (逐日模拟买卖持仓)
  python -m src.strategy.factor_model_selection.v3_bull_hunter.run_bull_hunter \
      --live-backtest --start_date 20250101 --end_date 20251230

  # 回测
  python -m src.strategy.factor_model_selection.v3_bull_hunter.run_bull_hunter \
      --backtest --start_date 20250101 --end_date 20251230

  # 完整扫描 (train + predict, 兼容旧接口)
  python -m src.strategy.factor_model_selection.v3_bull_hunter.run_bull_hunter \
      --scan_date 20260420
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from .pipeline import (
    PipelineConfig, run_backtest, run_daily, run_live, run_live_backtest,
    run_review, run_scan, run_train,
)
from .agent2_train import TrainConfig
from .agent3_predict import PredictConfig
from .agent5_portfolio import PortfolioConfig
from .agent6_exit_signal import ExitSignalConfig


def main():
    parser = argparse.ArgumentParser(description="Bull Hunter v3 — 大牛股预测系统 (v4)")

    # 模式
    parser.add_argument("--daily", action="store_true",
                        help="每日预测模式: 复用 latest 模型, 输出 Top 5 A 级候选")
    parser.add_argument("--train", action="store_true",
                        help="训练模式: Agent 1 + Agent 2, 更新 latest 模型")
    parser.add_argument("--review", action="store_true",
                        help="复盘模式: Agent 4 双回路评估 + 可能触发重训")
    parser.add_argument("--live", action="store_true",
                        help="Live 模式: Agent 1→3→6→5→7 持仓管理闭环")
    parser.add_argument("--live-backtest", action="store_true",
                        help="Live 回测: 逐日模拟 live 循环, 验证持仓管理效果")
    parser.add_argument("--backtest", action="store_true",
                        help="回测模式: 滚动扫描 + 实际收益验证")
    parser.add_argument("--scan_date", type=str, default="",
                        help="扫描/训练/复盘日期 (YYYYMMDD)")
    parser.add_argument("--start_date", type=str, default="",
                        help="回测起始日期")
    parser.add_argument("--end_date", type=str, default="",
                        help="回测结束日期")
    parser.add_argument("--force", action="store_true",
                        help="强制训练 (忽略间隔和缓存)")

    # 路径
    parser.add_argument("--cache_dir", type=str,
                        default="/nvme5/xtang/gp-workspace/gp-data/feature-cache")
    parser.add_argument("--data_dir", type=str,
                        default="/nvme5/xtang/gp-workspace/gp-data/tushare-daily-full")
    parser.add_argument("--basic_path", type=str,
                        default="/nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv")
    parser.add_argument("--results_dir", type=str,
                        default="results/bull_hunter")

    # 训练参数
    parser.add_argument("--lookback_months", type=int, default=12)
    parser.add_argument("--n_estimators", type=int, default=800)
    parser.add_argument("--learning_rate", type=float, default=0.03)

    # 预测参数
    parser.add_argument("--threshold_200", type=float, default=0.15)
    parser.add_argument("--top_n", type=int, default=5,
                        help="每日最多输出候选数 (默认 5)")

    # 回测参数
    parser.add_argument("--interval_days", type=int, default=20)

    # Agent 5 (Portfolio) 参数
    parser.add_argument("--sell_threshold", type=float, default=0.6,
                        help="Agent 6 卖出权重阈值 (默认 0.6)")
    parser.add_argument("--stop_loss", type=float, default=-0.15,
                        help="止损线 (默认 -0.15)")
    parser.add_argument("--max_positions", type=int, default=10,
                        help="最大持仓数 (默认 10)")
    parser.add_argument("--initial_capital", type=float, default=1000000,
                        help="初始资金 (默认 1000000)")

    # Live 回测整合参数
    parser.add_argument("--retrain_interval", type=int, default=60,
                        help="Agent 2 重训间隔天数 (0=不重训, 默认 60)")
    parser.add_argument("--monitor_interval", type=int, default=20,
                        help="Agent 4 选股纠正监控间隔天数 (0=禁用, 默认 20)")
    parser.add_argument("--no_exit_train", action="store_true",
                        help="禁用 Agent 6 退出模型训练")

    # LLM
    parser.add_argument("--use-llm", action="store_true",
                        help="启用 LLM 因子顾问")

    args = parser.parse_args()

    # 日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # 构建配置
    train_cfg = TrainConfig(
        lookback_months=args.lookback_months,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
    )
    predict_cfg = PredictConfig(
        threshold_200pct=args.threshold_200,
        top_n=args.top_n,
    )
    portfolio_cfg = PortfolioConfig(
        sell_weight_threshold=args.sell_threshold,
        stop_loss_pct=args.stop_loss,
        max_positions=args.max_positions,
    )
    exit_cfg = ExitSignalConfig()
    pipe_cfg = PipelineConfig(
        cache_dir=args.cache_dir,
        data_dir=args.data_dir,
        basic_path=args.basic_path,
        results_dir=args.results_dir,
        train_cfg=train_cfg,
        predict_cfg=predict_cfg,
        portfolio_cfg=portfolio_cfg,
        exit_cfg=exit_cfg,
        use_llm=args.use_llm,
    )

    if args.daily:
        if not args.scan_date:
            print("ERROR: --daily 需要 --scan_date")
            sys.exit(1)
        result = run_daily(args.scan_date, pipe_cfg)
        _print_daily_result(result, args)

    elif args.train:
        if not args.scan_date:
            print("ERROR: --train 需要 --scan_date")
            sys.exit(1)
        run_train(args.scan_date, pipe_cfg, force=args.force, trigger="manual")

    elif args.review:
        if not args.scan_date:
            print("ERROR: --review 需要 --scan_date")
            sys.exit(1)
        health = run_review(args.scan_date, pipe_cfg)
        _print_review_result(health, args)

    elif args.live:
        if not args.scan_date:
            print("ERROR: --live 需要 --scan_date")
            sys.exit(1)
        result = run_live(args.scan_date, pipe_cfg)
        _print_live_result(result, args)

    elif args.live_backtest:
        if not args.start_date or not args.end_date:
            print("ERROR: --live-backtest 需要 --start_date 和 --end_date")
            sys.exit(1)
        result = run_live_backtest(
            args.start_date, args.end_date, pipe_cfg,
            retrain_interval=args.retrain_interval,
            enable_exit_train=not args.no_exit_train,
            monitor_interval=args.monitor_interval,
        )
        _print_live_backtest_summary(result, args)

    elif args.backtest:
        if not args.start_date or not args.end_date:
            print("ERROR: --backtest 需要 --start_date 和 --end_date")
            sys.exit(1)
        result = run_backtest(
            args.start_date, args.end_date, pipe_cfg,
            args.interval_days, args.top_n,
        )
        if not result.empty:
            _print_backtest_summary(result, args)

    elif args.scan_date:
        # 兼容旧模式: 完整扫描
        result = run_scan(args.scan_date, pipe_cfg)
        _print_scan_result(result, args)

    else:
        print("ERROR: 需要 --daily, --train, --review, --live, --live-backtest, --backtest 或 --scan_date")
        sys.exit(1)


def _print_live_result(result, args):
    """打印 Live 模式结果。"""
    if not result:
        print("Live 运行无结果")
        return

    print(f"\n{'='*60}")
    print(f"  Bull Hunter v4 — Live 持仓管理")
    print(f"  日期: {args.scan_date}")
    print(f"{'='*60}")

    pr = result.get("portfolio_result", {})

    # 买入
    buys = pr.get("buys", [])
    if buys:
        print(f"\n  📈 买入 ({len(buys)} 只):")
        for b in buys:
            print(f"    {b['symbol']} {b.get('name', ''):<8s} | "
                  f"{b['shares']}股 @ {b['price']:.2f}")

    # 卖出
    sells = pr.get("sells", [])
    if sells:
        print(f"\n  📉 卖出 ({len(sells)} 只):")
        for s in sells:
            emoji = "✅" if s.get("pnl", 0) >= 0 else "❌"
            print(f"    {emoji} {s['symbol']} {s.get('name', ''):<8s} | "
                  f"盈亏 {s.get('pnl_pct', 0):+.1%} | "
                  f"持有 {s.get('holding_days', 0)}天 | "
                  f"原因: {s.get('reason', '')}")

    # 持有
    holds = pr.get("hold", [])
    if holds:
        print(f"\n  💼 持有 ({len(holds)} 只):")
        for h in holds:
            sw = h.get("sell_weight", 0)
            cg = h.get("current_gain", 0)
            sw_bar = "🟢" if sw < 0.3 else "🟡" if sw < 0.6 else "🔴"
            print(f"    {h['symbol']} {h.get('name', ''):<8s} | "
                  f"涨幅 {cg:+.1%} | 持有 {h.get('days_held', 0)}天 | "
                  f"卖出权重 {sw_bar} {sw:.2f}")

    # Supervisor 状态
    sr = result.get("supervisor_result", {})
    status = sr.get("status", "healthy")
    status_cn = {"healthy": "✅ 健康", "warning": "⚠️ 警告", "critical": "🔴 严重"}.get(status, status)
    print(f"\n  Agent 7 状态: {status_cn}")

    # 汇总
    summary = pr.get("portfolio_summary", {})
    stats = summary.get("trade_stats", {})
    if stats.get("n_trades", 0) > 0:
        print(f"  累计: {stats['n_trades']} 笔交易, "
              f"胜率 {stats.get('win_rate', 0):.0%}, "
              f"平均 {stats.get('avg_pnl_pct', 0):+.1%}")

    print(f"\n  仓位: {summary.get('n_positions', 0)} 只, "
          f"现金: {summary.get('cash', 0):,.0f}")
    print(f"{'='*60}")


def _print_live_backtest_summary(result, args):
    """打印 Live 回测摘要。"""
    if not result:
        print("Live 回测无结果")
        return

    stats = result.get("stats", {})

    print(f"\n{'='*60}")
    print(f"  Bull Hunter v4 — Live 回测报告")
    print(f"  回测区间: {args.start_date} ~ {args.end_date}")
    print(f"{'='*60}")

    print(f"\n  交易统计:")
    print(f"    总交易笔数: {stats.get('n_trades', 0)}")
    print(f"    胜率:       {stats.get('win_rate', 0):.1%}")
    print(f"    平均盈亏:   {stats.get('avg_pnl_pct', 0):+.1%}")
    print(f"    总盈亏:     {stats.get('total_pnl', 0):+,.0f}")
    print(f"    平均持有:   {stats.get('avg_holding_days', 0):.1f} 天")
    print(f"    最佳交易:   {stats.get('best_trade', 0):+.1%}")
    print(f"    最差交易:   {stats.get('worst_trade', 0):+.1%}")

    results_dir = result.get("results_dir", "")
    if results_dir:
        print(f"\n  📁 结果目录: {results_dir}/")

    print(f"{'='*60}")


def _print_daily_result(result, args):
    """打印每日预测结果。"""
    print(f"\n{'='*60}")
    print(f"  Bull Hunter v3 — 每日 A 级候选")
    print(f"  日期: {args.scan_date}")
    print(f"{'='*60}")

    if result.empty:
        print("  今日无 A 级候选")
    else:
        for _, row in result.iterrows():
            sym = row["symbol"]
            name = row.get("name", "")
            rank = row.get("rank", "?")
            p200 = row.get("prob_200", 0)
            p100 = row.get("prob_100", 0)
            print(f"  #{rank} {sym} {name:<8s} | "
                  f"6月涨200%={p200:.1%} | 2月翻倍={p100:.1%}")

    print(f"{'='*60}")


def _print_review_result(health, args):
    """打印复盘结果。"""
    print(f"\n{'='*60}")
    print(f"  Bull Hunter v3 — 复盘报告")
    print(f"  日期: {args.scan_date}")
    print(f"{'='*60}")

    status = health.get("status", "unknown")
    status_cn = {"healthy": "✅ 健康", "warning": "⚠️ 警告", "critical": "🔴 严重"}.get(status, status)
    print(f"\n  系统状态: {status_cn}")

    suggestions = health.get("suggestions", [])
    if suggestions:
        print(f"\n  建议 ({len(suggestions)} 条):")
        for s in suggestions:
            print(f"    💡 {s}")

    tuning = health.get("tuning_directives", {})
    if tuning.get("retrain_required"):
        print(f"\n  🔄 已触发重训: {tuning.get('reason', '')}")
    else:
        print(f"\n  无需重训")

    tracking = health.get("tracking_feedback", {})
    if tracking:
        n_eval = tracking.get("n_evaluated", 0)
        win = tracking.get("win_rate", 0)
        loss = tracking.get("loss_rate", 0)
        if n_eval > 0:
            print(f"\n  跟踪评估: {n_eval} 条到期, 胜率={win:.0%}, 亏损率={loss:.0%}")

    print(f"{'='*60}")


def _print_scan_result(result, args):
    """打印完整扫描结果。"""
    if result.empty:
        print("无候选")
        return

    print(f"\n{'='*60}")
    print(f"  Bull Hunter v3 — 扫描结果")
    print(f"  日期: {args.scan_date}")
    print(f"  候选: {len(result)} 只 A 级")
    print(f"{'='*60}")

    for _, row in result.iterrows():
        sym = row["symbol"]
        name = row.get("name", "")
        p200 = row.get("prob_200", 0)
        p100 = row.get("prob_100", 0)
        print(f"  {sym} {name:<8s} | 200%={p200:.1%} | 100%={p100:.1%}")

    print(f"{'='*60}")


def _print_backtest_summary(result, args):
    """打印回测结果中文摘要。"""
    import numpy as np

    n_dates = result["scan_date"].nunique()
    n_total = len(result)

    print(f"\n{'='*70}")
    print(f"  Bull Hunter v3 — 回测报告")
    print(f"  回测区间: {args.start_date} ~ {args.end_date}")
    print(f"  扫描间隔: 每 {args.interval_days} 个交易日")
    print(f"  扫描次数: {n_dates} 次, 总预测: {n_total} 条")
    print(f"{'='*70}")

    # 总体统计 (只有 A 级)
    for grade, label in [("A", "A级(200%目标)"), ("ALL", "全部")]:
        subset = result if grade == "ALL" else (
            result[result["grade"] == grade] if "grade" in result.columns else result
        )
        if subset.empty:
            continue

        n = len(subset)
        print(f"\n  【{label}】 {n} 条预测")
        print(f"  {'-'*66}")

        for fwd_days, target, fwd_label in [
            (10, 0.30, "10日(短线)"),
            (40, 1.00, "40日(中线)"),
            (120, 2.00, "120日(长线)"),
        ]:
            col = f"actual_{fwd_days}d"
            if col not in subset.columns:
                continue
            valid = subset[col].dropna()
            if len(valid) == 0:
                print(f"    {fwd_label}: 无数据")
                continue

            avg = valid.mean()
            med = valid.median()
            win_rate = (valid > 0).mean()
            hit_rate = (valid >= target).mean()
            max_gain = valid.max()
            max_loss = valid.min()

            print(f"    {fwd_label}: 均值={avg:+.1%}  中位数={med:+.1%}  "
                  f"胜率={win_rate:.0%}  命中率(≥{target:.0%})={hit_rate:.0%}  "
                  f"最大涨={max_gain:+.1%}  最大亏={max_loss:+.1%}  "
                  f"(有效={len(valid)}/{n})")

        # 40d 最大涨幅
        if "actual_max_40d" in subset.columns:
            max40 = subset["actual_max_40d"].dropna()
            if len(max40) > 0:
                print(f"    40日最大涨幅: 均值={max40.mean():+.1%}  "
                      f">=30%比例={float((max40 >= 0.30).mean()):.0%}  "
                      f">=100%比例={float((max40 >= 1.00).mean()):.0%}")

    # 按扫描日期的时间序列
    print(f"\n  📅 按扫描日期:")
    print(f"  {'扫描日':>10s}  {'数量':>4s}  {'10d均值':>8s}  {'10d胜率':>7s}  "
          f"{'40d均值':>8s}  {'40d胜率':>7s}")
    print(f"  {'-'*58}")
    for sd in sorted(result["scan_date"].unique()):
        sub = result[result["scan_date"] == sd]
        n = len(sub)
        parts = [f"  {sd:>10s}  {n:>4d}"]
        for fwd in [10, 40]:
            col = f"actual_{fwd}d"
            if col in sub.columns:
                valid = sub[col].dropna()
                if len(valid) > 0:
                    parts.append(f"  {valid.mean():>+7.1%}")
                    parts.append(f"  {(valid > 0).mean():>6.0%}")
                else:
                    parts.append(f"  {'N/A':>7s}")
                    parts.append(f"  {'N/A':>6s}")
        print("".join(parts))

    out_dir = os.path.join(args.results_dir, f"backtest_{args.start_date}_{args.end_date}")
    print(f"\n  📁 结果目录: {out_dir}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
