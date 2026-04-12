"""
测试实际因子值 - 使用 backtest_multi_day.py 中的计算函数
"""

import os
import glob
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def calc_path_irreversibility(returns: np.ndarray) -> float:
    """计算路径不可逆性熵"""
    if len(returns) < 50:
        return np.nan

    # 三态离散化
    sigma = np.std(returns)
    if sigma < 1e-10:
        return 0.0

    threshold = 0.5 * sigma
    states = np.zeros(len(returns), dtype=np.int64)
    states[returns < -threshold] = -1
    states[returns > threshold] = 1

    # 状态转移计数
    n_states = 3
    counts = np.zeros((n_states, n_states), dtype=np.float64)

    for t in range(len(states) - 1):
        i, j = int(states[t] + 1), int(states[t + 1] + 1)  # 映射到 0,1,2
        if 0 <= i < n_states and 0 <= j < n_states:
            counts[i, j] += 1.0

    # KL 散度
    total = counts.sum()
    if total < 10:
        return np.nan

    forward = counts / total
    backward = counts.T / total

    mask = (forward > 1e-10) & (backward > 1e-10)
    if not np.any(mask):
        return 0.0

    kl_div = np.sum(forward[mask] * np.log(forward[mask] / backward[mask]))
    return max(0.0, kl_div)


def calc_permutation_entropy(values: np.ndarray, order: int = 3) -> float:
    """计算排列熵"""
    import math

    if len(values) < order + 10:
        return np.nan

    values = values[np.isfinite(values)]
    if len(values) < order + 10:
        return np.nan

    # 计算序列表
    counts = {}
    for idx in range(len(values) - order + 1):
        pattern = tuple(np.argsort(values[idx:idx+order], kind='mergesort'))
        counts[pattern] = counts.get(pattern, 0) + 1

    if not counts:
        return np.nan

    # Shannon 熵
    freq = np.array(list(counts.values()), dtype=np.float64)
    prob = freq / freq.sum()
    prob = prob[prob > 0]

    entropy = -np.sum(prob * np.log(prob))

    # 归一化
    normalizer = np.log(math.factorial(order))
    if normalizer <= 0:
        return np.nan

    return entropy / normalizer


def market_state_classifier_original(
    path_irrev: float,
    perm_entropy: float,
    dominant_eig: float = None,
    turnover_entropy: float = None
) -> str:
    """原始的市场状态分类器"""
    path_irrev = path_irrev if np.isfinite(path_irrev) else 0.5
    perm_entropy = perm_entropy if np.isfinite(perm_entropy) else 0.5

    # 高不可逆性 + 低排列熵 = 强有序（趋势/主力控盘）
    if path_irrev > 0.3 and perm_entropy < 0.4:
        return 'ordered'

    # 低不可逆性 + 高排列熵 = 强混沌（无序波动/散户博弈）
    if path_irrev < 0.1 and perm_entropy > 0.7:
        return 'strong_chaos'

    # 中间状态 = 弱混沌
    return 'weak_chaos'


def main():
    data_dir = "/nvme5/xtang/gp-workspace/gp-data/trade"

    # 获取几只股票
    stock_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    stock_dirs = sorted(stock_dirs)[:3]

    print(f"分析股票：{stock_dirs}")

    all_pi = []
    all_pe = []
    all_states_original = []

    for stock_code in stock_dirs:
        print(f"\n处理 {stock_code}...")

        # 加载 2025 年 4 月数据
        stock_dir = os.path.join(data_dir, stock_code)
        csv_files = sorted(glob.glob(os.path.join(stock_dir, "2025-04-*.csv")))[:5]

        if not csv_files:
            continue

        dfs = []
        for f in csv_files:
            try:
                df = pd.read_csv(f)
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
            continue

        # 计算收益率
        data['log_ret'] = np.log(data['close']).diff()

        window_size = 240  # 240 分钟

        for i in range(window_size, len(data)):
            window_ret = data['log_ret'].iloc[i-window_size:i].dropna()

            if len(window_ret) > 50:
                pi = calc_path_irreversibility(window_ret.values)
                pe = calc_permutation_entropy(window_ret.values)

                if np.isfinite(pi) and np.isfinite(pe):
                    all_pi.append(pi)
                    all_pe.append(pe)
                    state = market_state_classifier_original(pi, pe)
                    all_states_original.append(state)

        print(f"  完成，{len(all_pi)} 条记录")

    # 统计
    print("\n" + "=" * 60)
    print("因子分布统计")
    print("=" * 60)

    print(f"\n路径不可逆性熵:")
    print(f"  均值：{np.mean(all_pi):.4f}")
    print(f"  标准差：{np.std(all_pi):.4f}")
    print(f"  最小值：{np.min(all_pi):.4f}")
    print(f"  最大值：{np.max(all_pi):.4f}")
    for q in [0.25, 0.5, 0.75, 0.9]:
        print(f"  {int(q*100)}% 分位：{np.quantile(all_pi, q):.4f}")

    print(f"\n排列熵:")
    print(f"  均值：{np.mean(all_pe):.4f}")
    print(f"  标准差：{np.std(all_pe):.4f}")
    print(f"  最小值：{np.min(all_pe):.4f}")
    print(f"  最大值：{np.max(all_pe):.4f}")
    for q in [0.25, 0.5, 0.75, 0.9]:
        print(f"  {int(q*100)}% 分位：{np.quantile(all_pe, q):.4f}")

    print(f"\n市场状态分布 (原始分类器):")
    from collections import Counter
    state_counts = Counter(all_states_original)
    for state, count in state_counts.items():
        print(f"  {state}: {count} ({count / len(all_states_original) * 100:.1f}%)")

    # 测试不同阈值的效果
    print("\n" + "=" * 60)
    print("测试不同市场状态分类阈值")
    print("=" * 60)

    # 根据实际分布调整：排列熵均值约 0.97，所以"低"排列熵应该是 < 0.95
    # 路径不可逆性均值约 0.08，所以"高"路径不可逆性应该是 > 0.1

    for pi_thresh in [0.05, 0.1, 0.15]:
        for pe_thresh in [0.9, 0.95, 0.97]:
            ordered_count = sum(1 for pi, pe in zip(all_pi, all_pe) if pi > pi_thresh and pe < pe_thresh)
            chaos_count = sum(1 for pi, pe in zip(all_pi, all_pe) if pi < 0.05 and pe > 0.8)
            weak_count = len(all_pi) - ordered_count - chaos_count

            print(f"\npi > {pi_thresh}, pe < {pe_thresh} -> ordered:")
            print(f"  ordered: {ordered_count} ({ordered_count/len(all_pi)*100:.1f}%)")
            print(f"  strong_chaos (pi<0.05, pe>0.8): {chaos_count} ({chaos_count/len(all_pi)*100:.1f}%)")
            print(f"  weak_chaos: {weak_count} ({weak_count/len(all_pi)*100:.1f}%)")


if __name__ == "__main__":
    main()
