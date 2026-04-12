#!/usr/bin/env python3
"""
双熵共振策略 - 命令行扫描入口

用法::

    # 买入信号扫描（默认模式）
    python -m src.strategy.dual_entropy_accumulation.run_scan

    # 卖出信号扫描（持仓列表）
    python -m src.strategy.dual_entropy_accumulation.run_scan \
        --mode sell \
        --watchlist sh600000,sz000001,sh600036

    # 指定数据路径和日期
    python -m src.strategy.dual_entropy_accumulation.run_scan \
        --daily_data_dir ../gp-data/tushare-daily-full \
        --minute_data_dir ../gp-data/trade \
        --scan_date 2026-04-03 \
        --max_stocks 100 \
        --workers 8

    # 只看 buy 信号
    python -m src.strategy.dual_entropy_accumulation.run_scan --show buy
"""

import argparse
import os
import sys

from .config import Config
from .scanner import Scanner


def parse_args():
    parser = argparse.ArgumentParser(
        description='双熵共振策略 - 全市场扫描（买入 / 卖出）',
    )

    parser.add_argument('--mode', type=str, default='buy',
                        choices=['buy', 'sell'],
                        help='扫描模式: buy=建仓信号, sell=卖出信号')
    parser.add_argument('--watchlist', type=str, default='',
                        help='卖出模式的持仓列表（逗号分隔，如 sh600000,sz000001）')
    parser.add_argument('--daily_data_dir', type=str, default='',
                        help='日线数据目录 (tushare-daily-full)')
    parser.add_argument('--minute_data_dir', type=str, default='',
                        help='分钟数据目录 (trade/)')
    parser.add_argument('--output_dir', type=str, default='',
                        help='结果输出目录')
    parser.add_argument('--scan_date', type=str, default='',
                        help='扫描日期 (YYYY-MM-DD)')
    parser.add_argument('--max_stocks', type=int, default=200,
                        help='日内分析的最大股票数')
    parser.add_argument('--workers', type=int, default=8,
                        help='并行进程数')
    parser.add_argument('--min_close', type=float, default=3.0,
                        help='最低股价')
    parser.add_argument('--max_close', type=float, default=100.0,
                        help='最高股价')
    parser.add_argument('--show', type=str, default='all',
                        choices=['all', 'buy', 'watch', 'sell', 'warning'],
                        help='显示哪些信号')
    parser.add_argument('--top', type=int, default=20,
                        help='显示前 N 只')

    return parser.parse_args()


def print_results(results, show='all', top=20):
    """打印扫描结果到控制台。"""
    if show == 'buy':
        filtered = [r for r in results if r.signal == 'buy']
    elif show == 'watch':
        filtered = [r for r in results if r.signal == 'watch']
    else:
        filtered = results

    if not filtered:
        print('\n无匹配信号。')
        return

    print(f'\n{"=" * 120}')
    print(f'  双熵共振信号  (显示: {show}, 共 {len(filtered)} 只)')
    print(f'{"=" * 120}')

    header = (
        f'{"排名":>4s}  '
        f'{"代码":<10s}  '
        f'{"信号":<6s}  '
        f'{"总分":>6s}  '
        f'{"日线压缩":>8s}  '
        f'{"日内集中":>8s}  '
        f'{"方向性":>6s}  '
        f'{"量能":>6s}  '
        f'{"日PE20":>7s}  '
        f'{"日百分位":>8s}  '
        f'{"日熵差":>7s}  '
        f'{"压缩天":>6s}  '
        f'{"分钟TE":>7s}  '
        f'{"TE降幅":>7s}  '
        f'{"分钟PI":>7s}  '
        f'{"日内态":<13s}'
    )
    print(header)
    print('-' * 120)

    for i, r in enumerate(filtered[:top]):
        line = (
            f'{i+1:>4d}  '
            f'{r.stock_code:<10s}  '
            f'{r.signal:<6s}  '
            f'{r.total_score:>6.3f}  '
            f'{r.daily_compression_score:>8.3f}  '
            f'{r.intraday_concentration_score:>8.3f}  '
            f'{r.direction_score:>6.3f}  '
            f'{r.volume_pattern_score:>6.3f}  '
            f'{r.daily_perm_entropy_20:>7.3f}  '
            f'{r.daily_entropy_percentile:>8.3f}  '
            f'{r.daily_entropy_gap:>7.3f}  '
            f'{r.daily_compression_days:>6d}  '
            f'{r.intraday_turnover_entropy:>7.3f}  '
            f'{r.intraday_turnover_entropy_drop:>7.3f}  '
            f'{r.intraday_path_irrev:>7.4f}  '
            f'{r.intraday_state:<13s}'
        )
        print(line)

    if len(filtered) > top:
        print(f'  ... 还有 {len(filtered) - top} 只未显示')


def print_sell_results(results, show='all', top=20):
    """打印卖出扫描结果到控制台。"""
    if show == 'sell':
        filtered = [r for r in results if r.signal == 'sell']
    elif show == 'warning':
        filtered = [r for r in results if r.signal == 'warning']
    else:
        filtered = results

    if not filtered:
        print('\n无匹配卖出信号。')
        return

    print(f'\n{"=" * 130}')
    print(f'  双熵卖出信号  (显示: {show}, 共 {len(filtered)} 只)')
    print(f'{"=" * 130}')

    header = (
        f'{"排名":>4s}  '
        f'{"代码":<10s}  '
        f'{"信号":<8s}  '
        f'{"总分":>6s}  '
        f'{"卖出类型":<12s}  '
        f'{"熵扩散":>6s}  '
        f'{"暗中派发":>8s}  '
        f'{"熵衰竭":>6s}  '
        f'{"量异常":>6s}  '
        f'{"日PE20":>7s}  '
        f'{"日百分位":>8s}  '
        f'{"PE速度":>7s}  '
        f'{"PI速度":>7s}  '
        f'{"分钟TE":>7s}  '
        f'{"TE升幅":>7s}  '
    )
    print(header)
    print('-' * 130)

    for i, r in enumerate(filtered[:top]):
        line = (
            f'{i+1:>4d}  '
            f'{r.stock_code:<10s}  '
            f'{r.signal:<8s}  '
            f'{r.total_score:>6.3f}  '
            f'{r.sell_type:<12s}  '
            f'{r.entropy_diffusion_score:>6.3f}  '
            f'{r.stealth_distribution_score:>8.3f}  '
            f'{r.exhaustion_score:>6.3f}  '
            f'{r.volume_anomaly_score:>6.3f}  '
            f'{r.daily_perm_entropy_20:>7.3f}  '
            f'{r.daily_entropy_percentile:>8.3f}  '
            f'{r.daily_entropy_velocity_5:>7.4f}  '
            f'{r.daily_path_irrev_velocity_5:>7.4f}  '
            f'{r.intraday_turnover_entropy:>7.3f}  '
            f'{r.intraday_te_rise:>7.3f}  '
        )
        print(line)

    if len(filtered) > top:
        print(f'  ... 还有 {len(filtered) - top} 只未显示')


def main():
    args = parse_args()

    config = Config()

    # 路径覆盖
    if args.daily_data_dir:
        config.scanner.daily_data_dir = args.daily_data_dir
    if args.minute_data_dir:
        config.scanner.minute_data_dir = args.minute_data_dir
    if args.output_dir:
        config.scanner.output_dir = args.output_dir
    if args.scan_date:
        config.scanner.scan_date = args.scan_date

    config.scanner.max_stocks = args.max_stocks
    config.scanner.workers = args.workers
    config.scanner.min_close = args.min_close
    config.scanner.max_close = args.max_close

    # 扫描
    scanner = Scanner(config)

    if args.mode == 'sell':
        # 卖出模式
        watchlist = None
        if args.watchlist:
            watchlist = [s.strip() for s in args.watchlist.split(',') if s.strip()]
        results = scanner.scan_sell(watchlist=watchlist, scan_date=args.scan_date)
        print_sell_results(results, show=args.show, top=args.top)
        scanner.save_sell_results(results)
    else:
        # 买入模式
        results = scanner.scan(args.scan_date)
        print_results(results, show=args.show, top=args.top)
        scanner.save_results(results)

    return results


if __name__ == '__main__':
    main()
