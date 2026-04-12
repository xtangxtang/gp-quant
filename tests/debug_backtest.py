"""
调试回测因子值
"""

import os
import glob
import numpy as np
import pandas as pd


def calc_path_irreversibility(returns: np.ndarray) -> float:
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


def turnover_rate_entropy(turnover: np.ndarray, n_bins: int = 10) -> float:
    if len(turnover) < n_bins:
        return np.nan
    arr = np.asarray(turnover, dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if len(arr) < n_bins:
        return np.nan
    counts, _ = np.histogram(arr, bins=n_bins)
    prob = counts / counts.sum()
    prob = prob[prob > 0]
    return -np.sum(prob * np.log(prob)) / np.log(n_bins)


def dominant_eigenvalue_from_autocorr(values: np.ndarray, order: int = 2) -> float:
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
    system = np.asarray([[acov[abs(i - j)] for j in range(order)] for i in range(order)], dtype=np.float64)
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
    return float(np.max(np.abs(eigs)))


def load_and_calculate(stock_code: str):
    data_dir = "/nvme5/xtang/gp-workspace/gp-data/trade"
    stock_dir = os.path.join(data_dir, stock_code)

    csv_files = sorted(glob.glob(os.path.join(stock_dir, "2025-04-*.csv")))[:5]
    if not csv_files:
        return None, None

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            df['trade_time'] = pd.to_datetime(df['时间'])
            dfs.append(df)
        except:
            continue

    if not dfs:
        return None, None

    data = pd.concat(dfs, ignore_index=True)
    data = data.rename(columns={
        '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low',
        '成交量 (手)': 'volume', '成交额 (元)': 'amount',
        '均价': 'avg_price', '换手率 (%)': 'turnover_rate'
    })

    # 处理缺失的换手率
    if 'turnover_rate' not in data.columns or data['turnover_rate'].isna().all():
        vol_col = None
        for col in ['volume', '成交量 (手)', '成交量']:
            if col in data.columns:
                vol_col = col
                break
        if vol_col:
            data['turnover_rate'] = data[vol_col].pct_change().fillna(0.01).abs() + 0.01
        else:
            data['turnover_rate'] = 1.0

    if len(data) < 300:
        return None, None

    data['log_ret'] = np.log(data['close']).diff()
    window_size = 240

    results = {'trade_time': data['trade_time'], 'close': data['close']}

    # 计算因子
    path_irrev, perm_ent, turn_ent, dom_eig, market_state = [], [], [], [], []

    for i in range(window_size, len(data)):
        window_ret = data['log_ret'].iloc[i-window_size:i].dropna()
        if len(window_ret) > 50:
            pi = calc_path_irreversibility(window_ret.values)
            pe = calc_permutation_entropy(window_ret.values)
            te = turnover_rate_entropy(data['turnover_rate'].iloc[i-window_size:i].values)
            de = dominant_eigenvalue_from_autocorr(window_ret.values)

            # 市场状态
            if np.isnan(pi) or np.isnan(pe):
                state = 'unknown'
            elif pi > 0.01 and pe < 0.97:
                state = 'ordered'
            elif pi < 0.005 and pe > 0.98:
                state = 'strong_chaos'
            else:
                state = 'weak_chaos'

            path_irrev.append(pi)
            perm_ent.append(pe)
            turn_ent.append(te)
            dom_eig.append(de)
            market_state.append(state)
        else:
            path_irrev.append(np.nan)
            perm_ent.append(np.nan)
            turn_ent.append(np.nan)
            dom_eig.append(np.nan)
            market_state.append('unknown')

    # 填充前面的 NaN
    path_irrev = [np.nan] * window_size + path_irrev
    perm_ent = [np.nan] * window_size + perm_ent
    turn_ent = [np.nan] * window_size + turn_ent
    dom_eig = [np.nan] * window_size + dom_eig
    market_state = ['unknown'] * window_size + market_state

    results['path_irreversibility'] = path_irrev
    results['permutation_entropy'] = perm_ent
    results['turnover_entropy'] = turn_ent
    results['dominant_eigenvalue'] = dom_eig
    results['market_state'] = market_state

    return pd.DataFrame(data), pd.DataFrame(results)


def test_backtest(data: pd.DataFrame, factors: pd.DataFrame):
    """模拟回测检查条件"""
    print(f"\ndata 索引前 5 条：{list(data['trade_time'].head())}")
    print(f"factors 索引前 5 条：{list(factors['trade_time'].head())}")

    # 正确的合并方式
    merged = factors.copy()
    merged = merged.merge(data[['trade_time', 'close']], on='trade_time', how='left')

    print(f"\n合并后数据量：{len(merged)}")
    print(f"close 列 NaN 数量：{merged['close'].isna().sum()}")
    print(f"因子列：{list(factors.columns)}")

    # 检查前 100 条有非 NaN 因子的记录
    count = 0
    for idx, row in merged.iterrows():
        path_irrev = row.get('path_irreversibility', np.nan)
        perm_ent = row.get('permutation_entropy', np.nan)
        market_state = row.get('market_state', 'unknown')
        dom_eig = row.get('dominant_eigenvalue', np.nan)

        if not np.isfinite(path_irrev):
            continue

        count += 1
        if count <= 20:  # 打印前 20 条
            c1 = path_irrev > 0.01
            c2 = perm_ent < 0.97
            c3 = market_state in ['ordered', 'weak_chaos']
            c4 = np.isfinite(dom_eig) and abs(dom_eig) < 0.9

            print(f"\n时间：{row['trade_time']}, 价格：{row['close']:.2f}")
            print(f"  path_irrev={path_irrev:.4f} (>0.01: {c1})")
            print(f"  perm_ent={perm_ent:.4f} (<0.97: {c2})")
            print(f"  market_state={market_state} (ordered/weak: {c3})")
            print(f"  dom_eig={dom_eig:.4f} (<0.9: {c4})")
            print(f"  全部满足：{c1 and c2 and c3 and c4}")

        if count >= 100:
            break

    print(f"\n总共有 {count} 条非 NaN 记录")


def main():
    stock_code = "bj920000"
    print(f"测试 {stock_code}...")

    data, factors = load_and_calculate(stock_code)

    if data is None or factors is None:
        print("数据加载失败")
        return

    print(f"数据量：{len(data)}, 因子量：{len(factors)}")

    # 因子统计
    print("\n因子统计:")
    print(f"  path_irrev: mean={factors['path_irreversibility'].mean():.4f}, std={factors['path_irreversibility'].std():.4f}")
    print(f"  perm_ent: mean={factors['permutation_entropy'].mean():.4f}, std={factors['permutation_entropy'].std():.4f}")
    print(f"  market_state: {factors['market_state'].value_counts().to_dict()}")

    test_backtest(data, factors)


if __name__ == "__main__":
    main()
