"""
分析熵因子分布 - 用于调整阈值
"""

import os
import glob
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.tick_entropy import (
    rolling_path_irreversibility,
    rolling_permutation_entropy,
    turnover_rate_entropy,
    market_state_classifier,
    dominant_eigenvalue_from_autocorr
)


def load_stock_data(data_dir: str, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """加载单只股票数据"""
    stock_dir = os.path.join(data_dir, stock_code)
    if not os.path.exists(stock_dir):
        return pd.DataFrame()

    csv_files = glob.glob(os.path.join(stock_dir, "*.csv"))
    dfs = []

    for f in csv_files:
        date_str = os.path.basename(f).replace('.csv', '')
        if date_str < start_date or date_str > end_date:
            continue
        try:
            df = pd.read_csv(f)
            df['stock_code'] = stock_code
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    data = pd.concat(dfs, ignore_index=True)
    data['trade_time'] = pd.to_datetime(data['时间'])

    # 重命名列
    data = data.rename(columns={
        '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low',
        '成交量 (手)': 'volume', '成交额 (元)': 'amount',
        '均价': 'avg_price', '换手率 (%)': 'turnover_rate'
    })

    data = data.sort_values(['stock_code', 'trade_time']).reset_index(drop=True)
    return data


def calculate_factors(data: pd.DataFrame, window_size: int = 1200) -> pd.DataFrame:
    """计算熵因子"""
    if len(data) < window_size + 10:
        return pd.DataFrame()

    data = data.sort_values('trade_time').reset_index(drop=True)
    data['log_ret'] = np.log(data['close']).diff()

    results = {
        'trade_time': data['trade_time'],
        'close': data['close'],
        'stock_code': data['stock_code'].iloc[0]
    }

    # 1. 路径不可逆性熵
    print("  计算路径不可逆性熵...")
    path_irrev = []
    for i in range(len(data)):
        if i < window_size:
            path_irrev.append(np.nan)
        else:
            window_ret = data['log_ret'].iloc[i-window_size:i].dropna()
            if len(window_ret) > 50:
                pi = rolling_path_irreversibility(pd.Series(window_ret.values))
                path_irrev.append(pi)
            else:
                path_irrev.append(np.nan)
    results['path_irreversibility'] = path_irrev

    # 2. 排列熵
    print("  计算排列熵...")
    perm_ent = []
    for i in range(len(data)):
        if i < window_size:
            perm_ent.append(np.nan)
        else:
            window_ret = data['log_ret'].iloc[i-window_size:i].dropna()
            if len(window_ret) > 50:
                pe = rolling_permutation_entropy(pd.Series(window_ret.values))
                perm_ent.append(pe)
            else:
                perm_ent.append(np.nan)
    results['permutation_entropy'] = perm_ent

    # 3. 换手率熵
    print("  计算换手率熵...")
    turn_ent = []
    for i in range(len(data)):
        if i < window_size:
            turn_ent.append(np.nan)
        else:
            window_turn = data['turnover_rate'].iloc[i-window_size:i].dropna()
            if len(window_turn) > 50:
                te = turnover_rate_entropy(window_turn.values, n_bins=10)
                turn_ent.append(te)
            else:
                turn_ent.append(np.nan)
    results['turnover_entropy'] = turn_ent

    # 4. 主导特征值
    print("  计算主导特征值...")
    dom_eig = []
    for i in range(len(data)):
        if i < window_size:
            dom_eig.append(np.nan)
        else:
            window_ret = data['log_ret'].iloc[i-window_size:i].dropna()
            if len(window_ret) > 50:
                de = dominant_eigenvalue_from_autocorr(window_ret.values, order=2)
                dom_eig.append(de)
            else:
                dom_eig.append(np.nan)
    results['dominant_eigenvalue'] = dom_eig

    # 5. 市场状态
    print("  计算市场状态...")
    market_state = []
    for i in range(len(data)):
        pi = results['path_irreversibility'][i]
        pe = results['permutation_entropy'][i]
        te = results['turnover_entropy'][i]
        de = results['dominant_eigenvalue'][i]

        if np.isnan(pi) or np.isnan(pe):
            market_state.append('unknown')
        else:
            state = market_state_classifier(pi, pe, de, te)
            market_state.append(state)
    results['market_state'] = market_state

    return pd.DataFrame(results)


def main():
    data_dir = "/nvme5/xtang/gp-workspace/gp-data/trade"

    # 获取几只股票
    stock_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    stock_dirs = sorted(stock_dirs)[:5]  # 只分析前 5 只股票

    print(f"分析股票：{stock_dirs}")

    all_factors = []

    for stock_code in stock_dirs:
        print(f"\n处理 {stock_code}...")

        # 加载 2025 年 4 月 -12 月数据
        data = load_stock_data(data_dir, stock_code, "2025-04-01", "2025-12-31")

        if len(data) < 1300:
            print(f"  数据不足，跳过")
            continue

        # 计算因子
        factors = calculate_factors(data, window_size=1200)

        if factors.empty:
            print(f"  因子计算失败")
            continue

        all_factors.append(factors)
        print(f"  完成，{len(factors)} 条记录")

    if not all_factors:
        print("无数据")
        return

    combined = pd.concat(all_factors, ignore_index=True)

    print("\n" + "=" * 60)
    print("熵因子分布统计")
    print("=" * 60)

    # 路径不可逆性
    pi = combined['path_irreversibility'].dropna()
    print(f"\n路径不可逆性熵 (path_irreversibility):")
    print(f"  均值：{pi.mean():.4f}")
    print(f"  标准差：{pi.std():.4f}")
    print(f"  最小值：{pi.min():.4f}")
    print(f"  最大值：{pi.max():.4f}")
    print(f"  25% 分位：{pi.quantile(0.25):.4f}")
    print(f"  50% 分位：{pi.quantile(0.50):.4f}")
    print(f"  75% 分位：{pi.quantile(0.75):.4f}")
    print(f"  90% 分位：{pi.quantile(0.90):.4f}")
    print(f"  > 0.2 的比例：{(pi > 0.2).mean() * 100:.1f}%")
    print(f"  > 0.1 的比例：{(pi > 0.1).mean() * 100:.1f}%")
    print(f"  > 0.05 的比例：{(pi > 0.05).mean() * 100:.1f}%")

    # 排列熵
    pe = combined['permutation_entropy'].dropna()
    print(f"\n排列熵 (permutation_entropy):")
    print(f"  均值：{pe.mean():.4f}")
    print(f"  标准差：{pe.std():.4f}")
    print(f"  最小值：{pe.min():.4f}")
    print(f"  最大值：{pe.max():.4f}")
    print(f"  25% 分位：{pe.quantile(0.25):.4f}")
    print(f"  50% 分位：{pe.quantile(0.50):.4f}")
    print(f"  75% 分位：{pe.quantile(0.75):.4f}")
    print(f"  < 0.6 的比例：{(pe < 0.6).mean() * 100:.1f}%")
    print(f"  < 0.7 的比例：{(pe < 0.7).mean() * 100:.1f}%")
    print(f"  < 0.8 的比例：{(pe < 0.8).mean() * 100:.1f}%")

    # 主导特征值
    de = combined['dominant_eigenvalue'].dropna()
    print(f"\n主导特征值 (dominant_eigenvalue):")
    print(f"  均值：{de.mean():.4f}")
    print(f"  标准差：{de.std():.4f}")
    print(f"  最小值：{de.min():.4f}")
    print(f"  最大值：{de.max():.4f}")
    print(f"  25% 分位：{de.quantile(0.25):.4f}")
    print(f"  50% 分位：{de.quantile(0.50):.4f}")
    print(f"  75% 分位：{de.quantile(0.75):.4f}")
    print(f"  |val| < 0.8 的比例：{((de > -0.8) & (de < 0.8)).mean() * 100:.1f}%")
    print(f"  |val| < 0.9 的比例：{((de > -0.9) & (de < 0.9)).mean() * 100:.1f}%")
    print(f"  |val| > 0.9 的比例：{((de > 0.9) | (de < -0.9)).mean() * 100:.1f}%")

    # 市场状态分布
    print(f"\n市场状态分布:")
    state_counts = combined['market_state'].value_counts()
    for state, count in state_counts.items():
        print(f"  {state}: {count} ({count / len(combined) * 100:.1f}%)")

    # 当前买入条件的命中率
    print(f"\n" + "=" * 60)
    print("当前买入条件命中率分析")
    print("=" * 60)

    condition_1 = combined['path_irreversibility'] > 0.2
    condition_2 = combined['permutation_entropy'] < 0.6
    condition_3 = combined['market_state'].isin(['ordered', 'weak_chaos'])
    condition_4 = (combined['dominant_eigenvalue'].abs() < 0.8) & np.isfinite(combined['dominant_eigenvalue'])

    print(f"condition_1 (path_irrev > 0.2): {condition_1.mean() * 100:.1f}%")
    print(f"condition_2 (perm_ent < 0.6): {condition_2.mean() * 100:.1f}%")
    print(f"condition_3 (state in ordered/weak_chaos): {condition_3.mean() * 100:.1f}%")
    print(f"condition_4 (|dom_eig| < 0.8): {condition_4.mean() * 100:.1f}%")

    all_conditions = condition_1 & condition_2 & condition_3 & condition_4
    print(f"\n全部条件同时满足：{all_conditions.mean() * 100:.4f}%")
    print(f"满足全部条件的记录数：{all_conditions.sum()}")

    # 宽松条件的命中率
    print(f"\n" + "=" * 60)
    print("宽松条件命中率")
    print("=" * 60)

    cond_1_loose = combined['path_irreversibility'] > 0.05
    cond_2_loose = combined['permutation_entropy'] < 0.8
    cond_4_loose = (combined['dominant_eigenvalue'].abs() < 0.95) & np.isfinite(combined['dominant_eigenvalue'])

    all_loose = cond_1_loose & cond_2_loose & condition_3 & cond_4_loose
    print(f"宽松条件 (path_irrev > 0.05, perm_ent < 0.8, |dom_eig| < 0.95):")
    print(f"  命中率：{all_loose.mean() * 100:.4f}%")
    print(f"  满足记录数：{all_loose.sum()}")


if __name__ == "__main__":
    main()
