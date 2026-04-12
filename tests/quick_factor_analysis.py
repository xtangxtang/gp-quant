"""
快速分析因子分布
"""

import os
import glob
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.tick_entropy import build_tick_entropy_features


def main():
    data_dir = "/nvme5/xtang/gp-workspace/gp-data/trade"

    # 获取几只股票
    stock_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    stock_dirs = sorted(stock_dirs)[:3]

    print(f"分析股票：{stock_dirs}")

    all_factors = []

    for stock_code in stock_dirs:
        print(f"\n处理 {stock_code}...")

        # 加载 2025 年 4 月数据（快速测试）
        stock_dir = os.path.join(data_dir, stock_code)
        csv_files = sorted(glob.glob(os.path.join(stock_dir, "2025-04-*.csv")))[:5]  # 只取 5 天

        if not csv_files:
            print(f"  无数据")
            continue

        dfs = []
        for f in csv_files:
            try:
                df = pd.read_csv(f)
                df['stock_code'] = stock_code
                df['trade_time'] = pd.to_datetime(df['时间'])
                dfs.append(df)
            except:
                continue

        if not dfs:
            continue

        data = pd.concat(dfs, ignore_index=True)
        data = data.rename(columns={
            '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low',
            '成交量 (手)': 'volume', '成交额 (元)': 'amount',
            '均价': 'avg_price', '换手率 (%)': 'turnover_rate'
        })

        if len(data) < 300:
            print(f"  数据不足")
            continue

        # 使用 build_tick_entropy_features
        input_df = pd.DataFrame({
            'trade_time': data['trade_time'],
            'price': data['close'],
            'turnover_rate': data.get('turnover_rate', pd.Series(1.0, index=data.index))
        })

        features = build_tick_entropy_features(input_df, windows={'path_irrev': 240, 'perm_entropy': 240, 'turnover': 240})

        if features.empty:
            continue

        all_factors.append(features)
        print(f"  完成，{len(features)} 条记录")

    if not all_factors:
        print("无数据")
        return

    combined = pd.concat(all_factors, ignore_index=True)

    print("\n" + "=" * 60)
    print("熵因子分布统计 (窗口=240 分钟)")
    print("=" * 60)

    # 路径不可逆性
    pi = combined['path_irreversibility'].dropna()
    if len(pi) > 0:
        print(f"\n路径不可逆性熵:")
        print(f"  均值：{pi.mean():.4f}")
        print(f"  标准差：{pi.std():.4f}")
        print(f"  最小值：{pi.min():.4f}")
        print(f"  最大值：{pi.max():.4f}")
        print(f"  分位数:")
        for q in [0.25, 0.5, 0.75, 0.9]:
            print(f"    {int(q*100)}%: {pi.quantile(q):.4f}")
        for thresh in [0.01, 0.05, 0.1, 0.2]:
            print(f"  > {thresh}: {(pi > thresh).mean() * 100:.1f}%")

    # 排列熵
    pe = combined['permutation_entropy'].dropna()
    if len(pe) > 0:
        print(f"\n排列熵:")
        print(f"  均值：{pe.mean():.4f}")
        print(f"  标准差：{pe.std():.4f}")
        print(f"  最小值：{pe.min():.4f}")
        print(f"  最大值：{pe.max():.4f}")
        for q in [0.25, 0.5, 0.75, 0.9]:
            print(f"    {int(q*100)}%: {pe.quantile(q):.4f}")
        for thresh in [0.5, 0.6, 0.7, 0.8]:
            print(f"  < {thresh}: {(pe < thresh).mean() * 100:.1f}%")

    # 主导特征值
    de = combined.get('dominant_eigenvalue')
    if de is not None:
        de = de.dropna()
        if len(de) > 0:
            print(f"\n主导特征值:")
            print(f"  均值：{de.mean():.4f}")
            print(f"  标准差：{de.std():.4f}")
            print(f"  最小值：{de.min():.4f}")
            print(f"  最大值：{de.max():.4f}")
            for q in [0.25, 0.5, 0.75, 0.9]:
                print(f"    {int(q*100)}%: {de.quantile(q):.4f}")

    # 市场状态分布
    if 'market_state' in combined.columns:
        print(f"\n市场状态分布:")
        state_counts = combined['market_state'].value_counts()
        for state, count in state_counts.items():
            print(f"  {state}: {count} ({count / len(combined) * 100:.1f}%)")

    # 买入条件分析
    print(f"\n" + "=" * 60)
    print("买入条件命中率测试")
    print("=" * 60)

    # 根据实际分布调整测试范围
    for pi_thresh in [0.01, 0.05, 0.1]:
        for pe_thresh in [0.95, 0.97, 0.99]:
            c1 = combined['path_irreversibility'] > pi_thresh
            c2 = combined['permutation_entropy'] < pe_thresh
            c3 = combined['market_state'].isin(['ordered', 'weak_chaos'])
            if 'dominant_eigenvalue' in combined.columns:
                c4 = combined['dominant_eigenvalue'].abs() < 0.9
            else:
                c4 = True

            all_c = c1 & c2 & c3 & c4
            pct = all_c.mean() * 100 if len(all_c) > 0 else 0
            print(f"pi > {pi_thresh}, pe < {pe_thresh}: {pct:.2f}% ({int(all_c.sum())} 条)")


if __name__ == "__main__":
    main()
