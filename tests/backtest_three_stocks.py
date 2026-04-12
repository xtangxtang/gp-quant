#!/usr/bin/env python3
"""
目标回测: 验证改进后的分岔策略对中际旭创/天孚通信/新易盛的效果。

分两步:
1. 先对这3只做精确诊断 (什么时候产生信号、什么时候卖出)
2. 再做全市场回测 (看整体表现, 避免过拟合)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from datetime import datetime

from src.strategy.dual_entropy_accumulation.config import (
    Config, DailyEntropyConfig, IntradayEntropyConfig,
)
from src.strategy.dual_entropy_accumulation.daily_entropy import DailyEntropy
from src.strategy.dual_entropy_accumulation.intraday_entropy import IntradayEntropyAnalyzer
from src.strategy.dual_entropy_accumulation.bifurcation import (
    BifurcationDetector, BifurcationConfig, TrendHoldEvaluator, TrendHoldConfig,
)
from src.strategy.dual_entropy_accumulation.sell_signal import SellSignalEngine
from src.strategy.dual_entropy_accumulation.backtest import (
    Backtester, BacktestConfig,
)


DAILY_DIR = '/nvme5/xtang/gp-workspace/gp-data/tushare-daily-full'
MINUTE_DIR = '/nvme5/xtang/gp-workspace/gp-data/trade'

TARGETS = {
    'sz300308': '中际旭创',
    'sz300394': '天孚通信',
    'sz300502': '新易盛',
}


def step1_diagnose_improved():
    """诊断改进后的分岔预警对三只目标股票的信号。"""
    print('=' * 90)
    print('  Step 1: 改进后的分岔预警诊断')
    print('=' * 90)

    detector = BifurcationDetector()
    cfg = detector.config
    print(f'\n  当前参数:')
    print(f'    eigenvalue_gate = {cfg.eigenvalue_gate}')
    print(f'    price_accel_gate = {cfg.price_accel_gate}')
    print(f'    momentum_floor = {cfg.momentum_floor}')
    print(f'    bifurcation_score_min = {cfg.bifurcation_score_min}')
    print(f'    bifurcation_watch_min = {cfg.bifurcation_watch_min}')

    trend_hold = TrendHoldEvaluator()
    th_cfg = trend_hold.config
    print(f'\n  趋势持有参数:')
    print(f'    sell_score_boost = {th_cfg.sell_score_boost}')
    print(f'    trailing_stop_pct = {th_cfg.trailing_stop_pct}')
    print(f'    max_hold_days_extension = {th_cfg.max_hold_days_extension}')

    calculator = DailyEntropy(DailyEntropyConfig())

    for code, name in TARGETS.items():
        print(f'\n{"─" * 90}')
        print(f'  {name} ({code})')
        print(f'{"─" * 90}')

        df = pd.read_csv(os.path.join(DAILY_DIR, f'{code}.csv'))
        df['trade_date'] = pd.to_numeric(df['trade_date'], errors='coerce')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['vol'] = pd.to_numeric(df['vol'], errors='coerce')
        df['open'] = pd.to_numeric(df['open'], errors='coerce')
        df = df.dropna(subset=['trade_date', 'close', 'vol', 'open'])
        df = df.sort_values('trade_date').reset_index(drop=True)

        # 分析 2025-04 ~ 2025-09 (完整的入场+持有+退出周期)
        start_int, end_int = 20250401, 20250930
        mask = (df['trade_date'] >= start_int) & (df['trade_date'] <= end_int)
        target_dates = df.loc[mask, 'trade_date'].astype(int).tolist()

        buy_signals = []
        all_results = []

        for d in target_dates:
            idx = df.index[df['trade_date'] == d][0] + 1
            if idx < 80:
                continue

            close_arr = df['close'].values[:idx]
            vol_arr = df['vol'].values[:idx]

            # 价格结构
            returns = np.diff(np.log(close_arr[-25:]))
            if len(returns) < 20:
                continue
            ret20 = returns[-20:]
            mom20 = float(np.sum(ret20))
            if len(ret20) >= 10:
                price_accel = np.sum(ret20[-5:]) - np.sum(ret20[-10:-5])
            else:
                price_accel = 0

            avg_vol_20 = np.mean(vol_arr[-20:])
            liquidity = close_arr[-1] * avg_vol_20

            # 日线熵
            start = max(0, idx - 250)
            close_sub = close_arr[start:]
            vol_sub = vol_arr[start:]
            if len(close_sub) < 70:
                continue

            prices = pd.Series(close_sub, index=np.arange(len(close_sub)))
            volumes = pd.Series(vol_sub, index=np.arange(len(vol_sub)))
            result = calculator.compute(prices, volumes)
            if result is None:
                continue

            # 5天前的日线熵
            idx5 = max(70, idx - 5)
            start5 = max(0, idx5 - 250)
            close_sub5 = df['close'].values[start5:idx5]
            vol_sub5 = df['vol'].values[start5:idx5]
            result5 = result
            if len(close_sub5) >= 70:
                p5 = pd.Series(close_sub5, index=np.arange(len(close_sub5)))
                v5 = pd.Series(vol_sub5, index=np.arange(len(vol_sub5)))
                r5 = calculator.compute(p5, v5)
                if r5 is not None:
                    result5 = r5

            prescreen = detector.passes_prescreen(
                dominant_eigenvalue=result.dominant_eigenvalue,
                price_accel=price_accel,
                liquidity=liquidity,
                mom20=mom20,
            )

            d_str = f'{d // 10000}-{(d % 10000) // 100:02d}-{d % 100:02d}'
            br = detector.evaluate(
                stock_code=code, trade_date=d_str,
                perm_entropy_20=result.perm_entropy_20,
                perm_entropy_60=result.perm_entropy_60,
                entropy_gap=result.entropy_gap,
                entropy_percentile=result.entropy_percentile,
                path_irreversibility=result.path_irreversibility,
                dominant_eigenvalue=result.dominant_eigenvalue,
                var_lift=result.var_lift,
                price_accel=price_accel, mom20=mom20, liquidity=liquidity,
                dominant_eigenvalue_prev5=result5.dominant_eigenvalue,
                path_irrev_prev5=result5.path_irreversibility,
            )

            all_results.append({
                'date': d_str, 'close': close_arr[-1],
                'prescreen': prescreen, 'signal': br.signal,
                'score': br.total_score, '|DE|': abs(result.dominant_eigenvalue),
                'accel': price_accel, 'mom20': mom20,
            })

            if prescreen and br.signal == 'buy':
                buy_signals.append({
                    'date': d_str, 'close': close_arr[-1],
                    'score': br.total_score,
                })

        # 打印买入信号
        if buy_signals:
            print(f'\n  ✅ 分岔买入信号 ({len(buy_signals)} 天):')
            for s in buy_signals:
                print(f'    {s["date"]}: close={s["close"]:.2f} score={s["score"]:.3f}')

            # 模拟最优入场: 取第一个 buy 信号
            first_buy = buy_signals[0]
            entry_date = first_buy['date']
            entry_price = first_buy['close']

            # 计算后续走势
            entry_int = int(entry_date.replace('-', ''))
            future = df[df['trade_date'] > entry_int].head(120)
            if len(future) > 0:
                peak_price = future['close'].max()
                peak_idx = future['close'].idxmax()
                peak_date_int = int(df.loc[peak_idx, 'trade_date'])
                peak_date = f'{peak_date_int // 10000}-{(peak_date_int % 10000) // 100:02d}-{peak_date_int % 100:02d}'

                # 模拟追踪止盈
                trailing_pct = TrendHoldConfig().trailing_stop_pct
                exit_date = None
                exit_price = None
                running_peak = entry_price
                for _, row in future.iterrows():
                    c = row['close']
                    if c > running_peak:
                        running_peak = c
                    pnl_pct = (c - entry_price) / entry_price
                    dd_from_peak = (c - running_peak) / running_peak
                    if pnl_pct > 0.15 and dd_from_peak <= -trailing_pct:
                        d_int = int(row['trade_date'])
                        exit_date = f'{d_int // 10000}-{(d_int % 10000) // 100:02d}-{d_int % 100:02d}'
                        exit_price = c
                        break

                print(f'\n  📈 模拟交易:')
                print(f'    入场: {entry_date} @ {entry_price:.2f}')
                print(f'    峰值: {peak_date} @ {peak_price:.2f} (涨{(peak_price/entry_price-1)*100:.1f}%)')
                if exit_date:
                    pnl = (exit_price - entry_price) / entry_price
                    print(f'    追踪止盈退出: {exit_date} @ {exit_price:.2f} (盈{pnl*100:.1f}%)')
                else:
                    last = future.iloc[-1]
                    d_int = int(last['trade_date'])
                    last_date = f'{d_int // 10000}-{(d_int % 10000) // 100:02d}-{d_int % 100:02d}'
                    pnl = (last['close'] - entry_price) / entry_price
                    print(f'    仍持有至: {last_date} @ {last["close"]:.2f} (盈{pnl*100:.1f}%)')
        else:
            print(f'\n  ❌ 无买入信号!')
            # 找得分最高的日期
            if all_results:
                sorted_r = sorted(all_results, key=lambda x: x['score'], reverse=True)[:5]
                print(f'  得分最高的5天:')
                for r in sorted_r:
                    print(f'    {r["date"]}: score={r["score"]:.3f} '
                          f'prescreen={r["prescreen"]} |DE|={r["|DE|"]:.3f} '
                          f'accel={r["accel"]:.4f} mom20={r["mom20"]:.3f}')


def step2_full_backtest():
    """运行全市场回测。"""
    print('\n' + '=' * 90)
    print('  Step 2: 全市场回测 (2025-04 ~ 2025-09)')
    print('=' * 90)

    config = Config()
    config.scanner.daily_data_dir = DAILY_DIR
    config.scanner.minute_data_dir = MINUTE_DIR
    config.scanner.output_dir = os.path.join(
        os.path.dirname(__file__), '..', 'results', 'bifurcation_v2_backtest')
    config.scanner.max_stocks = 300

    bt_config = BacktestConfig(
        start_date='2025-04-01',
        end_date='2025-09-30',
        initial_capital=1_000_000,
        max_positions=10,
        stop_loss=-0.08,
        take_profit=0.30,
        max_hold_days=90,
        enable_bifurcation=True,
        bifurcation_max_close=500.0,
        bifurcation_extra_positions=5,
        bifurcation_scan_interval=3,
        workers=8,
        sell_entropy_interval=3,
    )

    backtester = Backtester(config, bt_config)
    equity_df = backtester.run()
    metrics = backtester.report(equity_df, config.scanner.output_dir)

    # 检查目标股票是否被交易
    print(f'\n--- 目标股票命中情况 ---')
    for t in backtester.trades:
        if t.stock_code in TARGETS:
            print(f'  ✅ {TARGETS[t.stock_code]} ({t.stock_code}): '
                  f'{t.entry_date} → {t.exit_date} '
                  f'入{t.entry_price:.2f} 出{t.exit_price:.2f} '
                  f'盈亏{t.pnl_pct:.1%} 持{t.hold_days}天 '
                  f'原因: {t.exit_reason}'
                  f' {"[分岔]" if t.is_bifurcation else ""}')

    for code, name in TARGETS.items():
        found = any(t.stock_code == code for t in backtester.trades)
        if not found:
            print(f'  ❌ {name} ({code}): 未交易')

    return metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=0,
                        help='1=仅诊断, 2=仅回测, 0=全部')
    args = parser.parse_args()

    if args.step in (0, 1):
        step1_diagnose_improved()

    if args.step in (0, 2):
        step2_full_backtest()
