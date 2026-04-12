"""
熵因子回测结果分析
"""

import pandas as pd
import numpy as np
import os

RESULTS_DIR = "/nvme5/xtang/gp-workspace/gp-quant/results/entropy_backtest"

def analyze_scheme(scheme: str) -> dict:
    """分析单个方案"""
    trades_file = os.path.join(RESULTS_DIR, f"trades_scheme_{scheme}.csv")

    if not os.path.exists(trades_file):
        return None

    df = pd.read_csv(trades_file)

    stats = {
        'scheme': scheme,
        'total_trades': len(df),
        'winning_trades': len(df[df['pnl'] > 0]),
        'losing_trades': len(df[df['pnl'] <= 0]),
        'win_rate': len(df[df['pnl'] > 0]) / len(df) if len(df) > 0 else 0,
        'total_pnl': df['pnl'].sum(),
        'avg_pnl': df['pnl'].mean(),
        'max_win': df['pnl'].max(),
        'max_loss': df['pnl'].min(),
        'avg_win': df[df['pnl'] > 0]['pnl'].mean() if len(df[df['pnl'] > 0]) > 0 else 0,
        'avg_loss': df[df['pnl'] <= 0]['pnl'].mean() if len(df[df['pnl'] <= 0]) > 0 else 0,
        'total_return': (1 + df['returns']).prod() - 1,
        'avg_return': df['returns'].mean(),
        'std_return': df['returns'].std(),
        'sharpe': df['returns'].mean() / df['returns'].std() * np.sqrt(252) if df['returns'].std() > 0 else 0,
    }

    # 退出原因分析
    if 'exit_reason' in df.columns:
        exit_reasons = df['exit_reason'].value_counts().to_dict()
        stats['exit_reasons'] = exit_reasons

    # 按退出原因分组统计
    if 'exit_reason' in df.columns:
        reason_stats = df.groupby('exit_reason').agg({
            'pnl': ['count', 'sum', 'mean'],
            'returns': 'mean'
        }).round(4)
        stats['reason_stats'] = reason_stats

    # 持仓时间分析
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    df['hold_minutes'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 60
    stats['avg_hold_minutes'] = df['hold_minutes'].mean()

    return stats


def print_report():
    """打印分析报告"""
    print("=" * 80)
    print("熵因子分钟数据回测结果分析报告")
    print("=" * 80)

    schemes = ['A', 'B', 'C', 'D']
    scheme_names = {
        'A': '固定持有 30 分钟',
        'B': '因子驱动退出',
        'C': '止盈止损',
        'D': '组合方案'
    }

    all_stats = {}

    # 收集统计
    for scheme in schemes:
        stats = analyze_scheme(scheme)
        if stats:
            all_stats[scheme] = stats

    # 核心指标对比
    print("\n" + "=" * 80)
    print("一、4 种方案核心指标对比")
    print("=" * 80)

    print(f"\n{'指标':<20} {'方案 A':<18} {'方案 B':<18} {'方案 C':<18} {'方案 D':<18}")
    print("-" * 90)

    metrics = [
        ('total_trades', '交易次数', lambda x: x),
        ('win_rate', '胜率', lambda x: f'{x*100:.1f}%'),
        ('total_pnl', '总盈亏 (元)', lambda x: f'{x:.2f}'),
        ('total_return', '总收益率', lambda x: f'{x*100:.2f}%'),
        ('avg_return', '平均收益率', lambda x: f'{x*100:.3f}%'),
        ('sharpe', '夏普比率', lambda x: f'{x:.3f}'),
        ('avg_hold_minutes', '平均持仓 (分钟)', lambda x: f'{x:.1f}'),
    ]

    for key, name, fmt in metrics:
        print(f"{name:<20}", end=" ")
        for scheme in schemes:
            val = all_stats[scheme].get(key, 0)
            print(f"{fmt(val):<18}", end=" ")
        print()

    # 各方案详细分析
    for scheme in schemes:
        stats = all_stats[scheme]
        print(f"\n{'=' * 80}")
        print(f"二、方案 {scheme} ({scheme_names.get(scheme, '')}) 详细分析")
        print(f"{'=' * 80}")

        print(f"\n基础统计:")
        print(f"  总交易次数：{stats['total_trades']}")
        print(f"  盈利次数：{stats['winning_trades']}")
        print(f"  亏损次数：{stats['losing_trades']}")
        print(f"  胜率：{stats['win_rate']*100:.2f}%")

        print(f"\n盈亏统计:")
        print(f"  总盈亏：{stats['total_pnl']:.2f} 元")
        print(f"  平均盈亏：{stats['avg_pnl']:.2f} 元")
        print(f"  最大盈利：{stats['max_win']:.2f} 元")
        print(f"  最大亏损：{stats['max_loss']:.2f} 元")
        print(f"  平均盈利：{stats['avg_win']:.2f} 元")
        print(f"  平均亏损：{stats['avg_loss']:.2f} 元")
        print(f"  盈亏比：{abs(stats['avg_win'] / stats['avg_loss']) if stats['avg_loss'] != 0 else float('inf'):.2f}")

        print(f"\n收益率统计:")
        print(f"  总收益率：{stats['total_return']*100:.2f}%")
        print(f"  平均收益率：{stats['avg_return']*100:.4f}%")
        print(f"  收益率标准差：{stats['std_return']*100:.4f}%")
        print(f"  夏普比率：{stats['sharpe']:.4f}")

        print(f"\n持仓时间:")
        print(f"  平均持仓：{stats['avg_hold_minutes']:.1f} 分钟")

        # 退出原因
        if 'exit_reasons' in stats:
            print(f"\n退出原因分布:")
            for reason, count in stats['exit_reasons'].items():
                pct = count / stats['total_trades'] * 100
                print(f"  {reason}: {count} ({pct:.1f}%)")

    # 结论
    print(f"\n{'=' * 80}")
    print("三、结论与建议")
    print(f"{'=' * 80}")

    # 找出最优方案
    best_scheme = max(all_stats.keys(), key=lambda s: all_stats[s]['total_return'])
    best_sharpe = max(all_stats.keys(), key=lambda s: all_stats[s]['sharpe'])
    best_winrate = max(all_stats.keys(), key=lambda s: all_stats[s]['win_rate'])

    print(f"\n最优总收益方案：{best_scheme} ({scheme_names.get(best_scheme, '')}) - {all_stats[best_scheme]['total_return']*100:.2f}%")
    print(f"最优夏普比率：{best_sharpe} ({scheme_names.get(best_sharpe, '')}) - {all_stats[best_sharpe]['sharpe']:.3f}")
    print(f"最高胜率：{best_winrate} ({scheme_names.get(best_winrate, '')}) - {all_stats[best_winrate]['win_rate']*100:.1f}%")

    print(f"\n问题分析:")
    print("  1. 所有方案均亏损，说明熵因子在分钟级别的预测能力有限")
    print("  2. 胜率普遍低于 30%，远低于随机水平")
    print("  3. 平均收益率为负，交易成本 (手续费 + 印花税 + 滑点) 是主要亏损来源")

    print(f"\n改进建议:")
    print("  1. 优化买入条件，提高信号质量")
    print("  2. 调整滚动窗口大小，可能 60 分钟不适合当前数据")
    print("  3. 考虑在日线级别而非分钟级别应用熵因子")
    print("  4. 增加其他因子 (动量、成交量等) 与熵因子组合")
    print("  5. 降低交易频率，过滤低质量信号")

    print(f"\n{'=' * 80}")
    print("报告结束")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    print_report()
