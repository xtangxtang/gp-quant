"""
熵惜售分岔突破策略 - 主入口

用法:
  python src/strategy/entropy_accumulation_breakout/run_entropy_accumulation_breakout.py \
    --data_dir /path/to/tushare-daily-full \
    --out_dir  results/entropy_accumulation_breakout \
    [--scan_date 20250411] [--symbols sh600519,sz000001] [--backtest_start_date 20240101]
"""

import argparse
import logging
import sys
from pathlib import Path

if __package__:
    from .scan_service import ScanConfig, run_scan, run_backtest, write_results
else:
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.strategy.entropy_accumulation_breakout.scan_service import (
        ScanConfig,
        run_scan,
        run_backtest,
        write_results,
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entropy-Accumulation-Breakout Strategy Scanner"
    )
    parser.add_argument("--data_dir", required=True, help="日线 CSV 目录")
    parser.add_argument("--data_root", type=str, default="", help="数据根目录 (自动推断自 data_dir 父目录)")
    parser.add_argument("--out_dir", required=True, help="输出目录")
    parser.add_argument("--scan_date", type=str, default="", help="扫描日期 YYYYMMDD")
    parser.add_argument("--top_n", type=int, default=30, help="输出前 N 只候选")
    parser.add_argument("--symbols", type=str, default="", help="逗号分隔股票列表")
    parser.add_argument("--basic_path", type=str, default="", help="tushare_stock_basic.csv")
    parser.add_argument("--lookback_days", type=int, default=500, help="回看天数")
    parser.add_argument("--min_amount", type=float, default=500000.0, help="最低日均成交额")
    parser.add_argument("--min_turnover", type=float, default=0.5, help="最低换手率")
    parser.add_argument("--exclude_st", action="store_true", default=True)
    parser.add_argument("--include_st", action="store_false", dest="exclude_st")
    parser.add_argument("--backtest_start_date", type=str, default="", help="回测起始日期")
    parser.add_argument("--backtest_end_date", type=str, default="", help="回测结束日期")
    parser.add_argument("--hold_days", type=int, default=5, help="持有天数")
    parser.add_argument("--max_positions", type=int, default=10, help="最大持仓数")
    parser.add_argument("--max_positions_per_industry", type=int, default=2, help="每行业最大持仓")
    parser.add_argument("--feature_cache_dir", type=str, default="", help="特征缓存目录 (启用增量计算)")
    parser.add_argument("--verbose", action="store_true", help="打印详细日志")
    return parser


def main() -> None:
    args = _build_argument_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] if args.symbols else []

    cfg = ScanConfig(
        data_dir=args.data_dir,
        data_root=args.data_root,
        basic_path=args.basic_path,
        out_dir=args.out_dir,
        scan_date=args.scan_date,
        symbols=symbols,
        top_n=args.top_n,
        lookback_days=args.lookback_days,
        min_amount=args.min_amount,
        min_turnover=args.min_turnover,
        exclude_st=args.exclude_st,
        hold_days=args.hold_days,
        max_positions=args.max_positions,
        max_per_industry=args.max_positions_per_industry,
        backtest_start_date=args.backtest_start_date,
        backtest_end_date=args.backtest_end_date,
        feature_cache_dir=args.feature_cache_dir,
    )

    # ── 扫描 ──
    all_result, top_picks = run_scan(cfg)
    print(f"\n{'='*60}")
    print(f"  扫描日期: {cfg.scan_date or 'auto'}")
    print(f"  全部结果: {len(all_result)} 只")
    print(f"  突破候选: {len(top_picks)} 只")
    print(f"{'='*60}\n")

    if len(top_picks) > 0:
        cols = ["symbol", "name", "industry", "phase", "accum_quality", "bifurc_quality", "composite_score"]
        show_cols = [c for c in cols if c in top_picks.columns]
        print(top_picks[show_cols].to_string(index=False))
        print()

    # ── 回测 ──
    df_equity = None
    trades = None
    if args.backtest_start_date:
        print("Running forward backtest ...")
        df_equity, trades = run_backtest(cfg)
        if trades:
            wins = sum(1 for t in trades if t.pnl_pct > 0)
            total = len(trades)
            avg_pnl = sum(t.pnl_pct for t in trades) / total if total else 0
            print(f"\n  交易数: {total}")
            print(f"  胜率: {wins}/{total} = {wins/total*100:.1f}%")
            print(f"  平均收益: {avg_pnl:.2f}%")
            if df_equity is not None and len(df_equity) > 1:
                final_eq = df_equity["equity"].iloc[-1]
                print(f"  净值: {final_eq:.4f}")
            print()

    # ── 写出 ──
    scan_date_str = cfg.scan_date or "latest"
    write_results(cfg.out_dir, scan_date_str, all_result, top_picks, df_equity, trades)
    print(f"Results → {cfg.out_dir}")


if __name__ == "__main__":
    main()
