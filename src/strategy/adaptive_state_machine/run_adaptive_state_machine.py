#!/usr/bin/env python3
"""
Adaptive State Machine — CLI 入口

用法:
  # 单日扫描
  python -m src.strategy.adaptive_state_machine.run_adaptive_state_machine \
      --scan_date 20260423

  # 历史回测
  python -m src.strategy.adaptive_state_machine.run_adaptive_state_machine \
      --backtest \
      --start_date 20250101 \
      --end_date 20251231 \
      --interval_days 5

  # 指定股票
  python -m src.strategy.adaptive_state_machine.run_adaptive_state_machine \
      --scan_date 20260423 \
      --symbols sh600519,sz000001
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from .pipeline import run_scan, run_backtest

# ═════════════════════════════════════════════════════════
# 默认路径
# ═════════════════════════════════════════════════════════

DEFAULT_DATA_ROOT = os.environ.get("GP_DATA_DIR", "/home/xtang/gp-workspace/gp-data")
DEFAULT_DAILY_DIR = os.path.join(DEFAULT_DATA_ROOT, "tushare-daily-full")
DEFAULT_CACHE_DIR = os.path.join(DEFAULT_DATA_ROOT, "feature-cache")
DEFAULT_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "results", "adaptive_state_machine")


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive State Machine — 4 Agent 自适应状态机策略",
    )

    # 模式
    parser.add_argument("--scan_date", type=str, help="扫描日期 (YYYYMMDD)")
    parser.add_argument("--backtest", action="store_true", help="历史回测模式")
    parser.add_argument("--start_date", type=str, default="", help="回测起始日期")
    parser.add_argument("--end_date", type=str, default="", help="回测结束日期")
    parser.add_argument("--interval_days", type=int, default=5, help="回测扫描间隔 (交易日)")

    # 数据路径
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT, help="数据根目录")
    parser.add_argument("--daily_dir", type=str, default="", help="日线数据目录")
    parser.add_argument("--cache_dir", type=str, default="", help="特征缓存目录")
    parser.add_argument("--config_dir", type=str, default="", help="配置文件目录")
    parser.add_argument("--output_dir", type=str, default="", help="输出目录")

    # 其他
    parser.add_argument("--symbols", type=str, help="指定股票列表 (逗号分隔)")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细日志")

    args = parser.parse_args()
    setup_logging(args.verbose)

    # 路径处理
    daily_dir = args.daily_dir or DEFAULT_DAILY_DIR
    cache_dir = args.cache_dir or DEFAULT_CACHE_DIR
    config_dir = args.config_dir or DEFAULT_CONFIG_DIR
    output_dir = args.output_dir or os.path.abspath(DEFAULT_OUTPUT_DIR)

    symbols = args.symbols.split(",") if args.symbols else None

    if not args.backtest and not args.scan_date:
        parser.error("请指定 --scan_date 或 --backtest")

    if args.backtest:
        if not args.start_date or not args.end_date:
            parser.error("回测模式需要 --start_date 和 --end_date")

        logger = logging.getLogger("adaptive_state_machine")
        logger.info(f"Backtest mode: {args.start_date} → {args.end_date}, interval={args.interval_days}d")

        result = run_backtest(
            daily_dir=daily_dir,
            data_root=args.data_root,
            start_date=args.start_date,
            end_date=args.end_date,
            interval_days=args.interval_days,
            cache_dir=cache_dir,
            config_dir=config_dir,
            output_dir=output_dir,
        )

        if result.empty:
            print("No signals found.")
        else:
            print(f"\nBacktest complete: {len(result)} signals generated")
            print(f"Output: {output_dir}/")

    else:
        logger = logging.getLogger("adaptive_state_machine")
        logger.info(f"Scan mode: {args.scan_date}")

        result = run_scan(
            daily_dir=daily_dir,
            data_root=args.data_root,
            scan_date=args.scan_date,
            cache_dir=cache_dir,
            config_dir=config_dir,
            output_dir=output_dir,
        )

        if not result:
            print("No data available.")
        else:
            print(f"\nScan complete: {result.get('total_stocks', 0)} stocks evaluated")


if __name__ == "__main__":
    main()
