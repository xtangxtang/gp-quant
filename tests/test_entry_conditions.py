"""
测试买入条件命中率
"""

import os
import glob
import numpy as np
import pandas as pd


def calc_path_irreversibility(returns: np.ndarray) -> float:
    """计算路径不可逆性熵"""
    if len(returns) < 50:
        return np.nan

    sigma = np.std(returns)
    if sigma < 1e-10:
        return 0.0

    threshold = 0.5 * sigma
    states = np.zeros(len(returns), dtype=np.int64)
    states[returns < -threshold] = -1
    states[returns > threshold] = 1

    n_states = 3
    counts = np.zeros((n_states, n_states), dtype=np.float64)

    for t in range(len(states) - 1):
        i, j = int(states[t] + 1), int(states[t + 1] + 1)
        if 0 <= i < n_states and 0 <= j < n_states:
            counts[i, j] += 1.0

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

    counts = {}
    for idx in range(len(values) - order + 1):
        pattern = tuple(np.argsort(values[idx:idx+order], kind='mergesort'))
        counts[pattern] = counts.get(pattern, 0) + 1

    if not counts:
        return np.nan

    freq = np.array(list(counts.values()), dtype=np.float64)
    prob = freq / freq.sum()
    prob = prob[prob > 0]

    entropy = -np.sum(prob * np.log(prob))

    normalizer = np.log(math.factorial(order))
    if normalizer <= 0:
        return np.nan

    return entropy / normalizer


def dominant_eigenvalue_from_autocorr(values: np.ndarray, order: int = 2) -> float:
    """计算主导特征值"""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) < max(12, order + 6):
        return np.nan

    centered = values - float(np.mean(values))
    if float(np.std(centered)) <= 1e-12:
        return np.nan

    acov = []
    for lag in range(order + 1):
        left = centered[:len(centered) - lag]
        right = centered[lag:]
        if len(right) == 0:
            return np.nan
        acov.append(float(np.dot(left, right)) / float(len(right)))

    system = np.asarray(
        [[acov[abs(i - j)] for j in range(order)] for i in range(order)],
        dtype=np.float64
    )
    rhs = np.asarray(acov[1: order + 1], dtype=np.float64)

    try:
        phi = np.linalg.solve(system + np.eye(order, dtype=np.float64) * 1e-8, rhs)
    except np.linalg.LinAlgError:
        return np.nan

    companion = np.zeros((order, order), dtype=np.float64)
    companion[0, :] = phi
    for i in range(1, order):
        companion[i, i - 1] = 1.0

    eigs = np.linalg.eigvals(companion)
    dominant = float(np.max(np.abs(eigs)))
    return dominant


def main():
    data_dir = "/nvme5/xtang/gp-workspace/gp-data/trade"
    stock_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])[:3]

    print(f"分析股票：{stock_dirs}")

    all_pi, all_pe, all_de, all_states = [], [], [], []

    for stock_code in stock_dirs:
        print(f"\n处理 {stock_code}...")

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

        if len(data) < 1300:
            continue

        data['log_ret'] = np.log(data['close']).diff()
        window_size = 240  # 改用 1 天窗口测试

        count = 0
        for i in range(window_size, len(data)):
            window_ret = data['log_ret'].iloc[i-window_size:i].dropna()

            if len(window_ret) > 50:
                pi = calc_path_irreversibility(window_ret.values)
                pe = calc_permutation_entropy(window_ret.values)
                de = dominant_eigenvalue_from_autocorr(window_ret.values)

                if np.isfinite(pi) and np.isfinite(pe):
                    # 改进的市场状态分类
                    if pi > 0.01 and pe < 0.97:
                        state = 'ordered'
                    elif pi < 0.005 and pe > 0.98:
                        state = 'strong_chaos'
                    else:
                        state = 'weak_chaos'

                    all_pi.append(pi)
                    all_pe.append(pe)
                    all_de.append(de if np.isfinite(de) else 0.9)
                    all_states.append(state)
                    count += 1

        print(f"  完成，{count} 条记录")

    # 分析条件命中率
    print("\n" + "=" * 60)
    print("条件命中率分析")
    print("=" * 60)

    pi_arr = np.array(all_pi)
    pe_arr = np.array(all_pe)
    de_arr = np.array(all_de)
    states = np.array(all_states)

    # 各条件
    c1 = pi_arr > 0.01
    c2 = pe_arr < 0.97
    c3 = np.isin(states, ['ordered', 'weak_chaos'])
    c4 = np.abs(de_arr) < 0.9

    print(f"\n总记录数：{len(all_pi)}")
    print(f"\n各条件命中率:")
    print(f"  c1 (path_irrev > 0.01): {c1.mean() * 100:.1f}% ({c1.sum()} 条)")
    print(f"  c2 (perm_ent < 0.97): {c2.mean() * 100:.1f}% ({c2.sum()} 条)")
    print(f"  c3 (state in ordered/weak_chaos): {c3.mean() * 100:.1f}% ({c3.sum()} 条)")
    print(f"  c4 (|dom_eig| < 0.9): {c4.mean() * 100:.1f}% ({c4.sum()} 条)")

    # 组合条件
    all_c = c1 & c2 & c3 & c4
    print(f"\n全部条件同时满足：{all_c.mean() * 100:.4f}% ({all_c.sum()} 条)")

    # 逐步放宽测试
    print("\n" + "=" * 60)
    print("放宽条件测试")
    print("=" * 60)

    for pi_t in [0.005, 0.01, 0.02]:
        for pe_t in [0.97, 0.98, 0.99]:
            c1_test = pi_arr > pi_t
            c2_test = pe_arr < pe_t
            combined = c1_test & c2_test & c3 & c4
            print(f"pi > {pi_t}, pe < {pe_t}: {combined.mean() * 100:.2f}% ({combined.sum()} 条)")


if __name__ == "__main__":
    main()
