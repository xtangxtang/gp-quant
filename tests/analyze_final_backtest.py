"""
熵因子回测结果详细分析
"""

import pandas as pd
import numpy as np

RESULTS_FILE = "/nvme5/xtang/gp-workspace/gp-quant/results/entropy_backtest_v2/trades_all.csv"

def main():
    df = pd.read_csv(RESULTS_FILE)

    print("=" * 80)
    print("熵因子回测结果详细分析（2025 年 4 月 -12 月，240 分钟窗口）")
    print("=" * 80)

    # 基础统计
    print(f"\n基础统计:")
    print(f"  总交易数：{len(df)}")
    print(f"  盈利交易：{len(df[df['pnl'] > 0])} ({len(df[df['pnl'] > 0])/len(df)*100:.1f}%)")
    print(f"  亏损交易：{len(df[df['pnl'] <= 0])} ({len(df[df['pnl'] <= 0])/len(df)*100:.1f}%)")

    # 盈亏统计
    print(f"\n盈亏统计:")
    winning = df[df['pnl'] > 0]['pnl']
    losing = df[df['pnl'] <= 0]['pnl']
    print(f"  总盈亏：{df['pnl'].sum():.2f} 元")
    print(f"  平均盈利：{winning.mean():.2f} 元")
    print(f"  平均亏损：{losing.mean():.2f} 元")
    print(f"  最大盈利：{df['pnl'].max():.2f} 元")
    print(f"  最大亏损：{df['pnl'].min():.2f} 元")
    print(f"  盈亏比：{abs(winning.mean() / losing.mean()) if losing.mean() != 0 else float('inf'):.2f}")

    # 收益率统计
    print(f"\n收益率统计:")
    print(f"  平均收益率：{df['returns'].mean()*100:.4f}%")
    print(f"  收益率标准差：{df['returns'].std()*100:.4f}%")
    print(f"  总收益率：{(1 + df['returns']).prod()*100 - 100:.2f}%")

    # 退出原因分析
    print(f"\n退出原因详细分析:")
    for reason in df['exit_reason'].unique():
        subset = df[df['exit_reason'] == reason]
        win_rate = len(subset[subset['pnl'] > 0]) / len(subset) * 100 if len(subset) > 0 else 0
        avg_ret = subset['returns'].mean() * 100 if len(subset) > 0 else 0
        print(f"  {reason}:")
        print(f"    交易数：{len(subset)} ({len(subset)/len(df)*100:.1f}%)")
        print(f"    胜率：{win_rate:.1f}%")
        print(f"    平均收益率：{avg_ret:.4f}%")

    # 持仓时间分析
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    df['hold_minutes'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 60

    print(f"\n持仓时间分析:")
    print(f"  平均持仓：{df['hold_minutes'].mean():.1f} 分钟")
    print(f"  中位数持仓：{df['hold_minutes'].median():.1f} 分钟")
    print(f"  最长持仓：{df['hold_minutes'].max():.1f} 分钟")
    print(f"  最短持仓：{df['hold_minutes'].min():.1f} 分钟")

    # 按持仓时间分组
    df['hold_bucket'] = pd.cut(df['hold_minutes'], bins=[0, 30, 60, 120, 240, float('inf')],
                               labels=['0-30min', '30-60min', '60-120min', '120-240min', '>240min'])
    print(f"\n按持仓时间分组:")
    for bucket in df['hold_bucket'].cat.categories:
        subset = df[df['hold_bucket'] == bucket]
        if len(subset) > 0:
            win_rate = len(subset[subset['pnl'] > 0]) / len(subset) * 100
            avg_ret = subset['returns'].mean() * 100
            print(f"  {bucket}: {len(subset)} 交易，胜率 {win_rate:.1f}%, 平均收益率 {avg_ret:.4f}%")

    # 按股票分析
    print(f"\n按股票分析 (按总盈亏排序):")
    stock_stats = df.groupby('stock_code').agg({
        'pnl': ['count', 'sum', 'mean'],
        'returns': 'mean',
        'exit_reason': lambda x: x.value_counts().index[0] if len(x) > 0 else 'N/A'
    }).round(4)
    stock_stats.columns = ['trades', 'total_pnl', 'avg_pnl', 'avg_return', 'main_exit']
    stock_stats = stock_stats.sort_values('total_pnl', ascending=False)
    print(stock_stats.head(20).to_string())

    # 最佳和最差股票
    print(f"\n最佳股票 (前 5):")
    for stock in stock_stats.head(5).index:
        stats = stock_stats.loc[stock]
        print(f"  {stock}: {stats['trades']} 交易，总盈利 {stats['total_pnl']:.2f} 元，平均收益 {stats['avg_return']*100:.4f}%")

    print(f"\n最差股票 (后 5):")
    for stock in stock_stats.tail(5).index:
        stats = stock_stats.loc[stock]
        print(f"  {stock}: {stats['trades']} 交易，总亏损 {stats['total_pnl']:.2f} 元，平均收益 {stats['avg_return']*100:.4f}%")


if __name__ == "__main__":
    main()
