"""
熵因子分钟数据回测 - 多天窗口版本

策略改进:
1. 使用多天 (5 天) 的 1 分钟数据作为滚动窗口计算熵因子
2. 买入条件：三个熵指标同时满足有序状态
3. 卖出条件：临界转变预警 (dominant_eig > 0.9) 或 止盈止损
4. 时间范围：2025 年 4 月 -2025 年 12 月
"""

import os
import glob
import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.tick_entropy import (
    rolling_path_irreversibility,
    rolling_permutation_entropy,
    turnover_rate_entropy,
    market_state_classifier,
    dominant_eigenvalue_from_autocorr
)


# =============================================================================
# 配置
# =============================================================================

@dataclass
class Config:
    data_dir: str = "/nvme5/xtang/gp-workspace/gp-data/trade"
    output_dir: str = "/nvme5/xtang/gp-workspace/gp-quant/results/entropy_backtest_v2"

    # 时间范围：2025 年 4 月 -12 月
    start_date: str = "2025-04-01"
    end_date: str = "2025-12-31"

    # 窗口配置：使用 1 天（240 分钟）而不是 5 天
    # 原因：测试发现 1200 分钟窗口下 path_irrev 值过低（几乎全为 0）
    # 240 分钟窗口能产生更有效的熵指标分布
    window_days: int = 1
    minutes_per_day: int = 240
    window_size: int = 240  # 1 天窗口

    # 股票池配置
    max_stocks: int = 50
    min_market_cap: float = 0  # 可以加市值过滤

    # 交易配置
    initial_capital: float = 1_000_000.0
    position_size: float = 0.1
    max_positions: int = 10

    # 费用
    commission: float = 0.0003
    stamp_tax: float = 0.001
    slippage: float = 0.001


# =============================================================================
# 数据加载
# =============================================================================

def load_stock_data(data_dir: str, stock_code: str,
                    start_date: str, end_date: str) -> pd.DataFrame:
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

    # 处理换手率
    if 'turnover_rate' not in data.columns or data['turnover_rate'].isna().all():
        # 尝试从成交量或其他列获取
        vol_col = None
        for col in ['volume', '成交量 (手)', '成交量']:
            if col in data.columns:
                vol_col = col
                break
        if vol_col:
            data['turnover_rate'] = data[vol_col].pct_change().fillna(0.01).abs() + 0.01
        else:
            data['turnover_rate'] = 0.01

    data = data.sort_values(['stock_code', 'trade_time']).reset_index(drop=True)
    return data


# =============================================================================
# 因子计算 - 多天窗口
# =============================================================================

def calculate_multi_day_factors(data: pd.DataFrame, window_size: int = 1200) -> pd.DataFrame:
    """
    使用多天窗口计算熵因子

    对每个时间点，使用过去 window_size 个分钟数据计算：
    1. 路径不可逆性熵
    2. 排列熵
    3. 换手率熵
    4. 主导特征值 (用于临界预警)
    5. 市场状态
    """
    if len(data) < window_size + 10:
        return pd.DataFrame()

    data = data.sort_values('trade_time').reset_index(drop=True)

    # 计算收益率
    data['log_ret'] = np.log(data['close']).diff()

    results = {
        'trade_time': data['trade_time'],
        'close': data['close'],
        'stock_code': data['stock_code'].iloc[0]
    }

    # 1. 路径不可逆性熵 (滚动窗口)
    print("  计算路径不可逆性熵...")
    path_irrev = []
    for i in range(len(data)):
        if i < window_size:
            path_irrev.append(np.nan)
        else:
            window_ret = data['log_ret'].iloc[i-window_size:i].dropna()
            if len(window_ret) > 50:
                pi = calc_path_irreversibility(window_ret.values)
                path_irrev.append(pi)
            else:
                path_irrev.append(np.nan)
    results['path_irreversibility'] = path_irrev

    # 2. 排列熵 (滚动窗口)
    print("  计算排列熵...")
    perm_ent = []
    for i in range(len(data)):
        if i < window_size:
            perm_ent.append(np.nan)
        else:
            window_ret = data['log_ret'].iloc[i-window_size:i].dropna()
            if len(window_ret) > 50:
                pe = calc_permutation_entropy(window_ret.values)
                perm_ent.append(pe)
            else:
                perm_ent.append(np.nan)
    results['permutation_entropy'] = perm_ent

    # 3. 换手率熵 (滚动窗口)
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

    # 4. 主导特征值 (用于临界预警)
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
            # 使用改进的分类逻辑（基于实际数据分布）
            # 路径不可逆性均值 0.01，排列熵均值 0.97
            # ordered: 相对高 path_irrev (>0.01) 且相对低 perm_ent (<0.97)
            # strong_chaos: 很低 path_irrev (<0.005) 且很高 perm_ent (>0.98)
            if pi > 0.01 and pe < 0.97:
                state = 'ordered'
            elif pi < 0.005 and pe > 0.98:
                state = 'strong_chaos'
            else:
                state = 'weak_chaos'
            market_state.append(state)
    results['market_state'] = market_state

    return pd.DataFrame(results)


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


# =============================================================================
# 简化的单股票回测
# =============================================================================

def backtest_single_stock(data: pd.DataFrame, factors: pd.DataFrame,
                          config: Config) -> List[Dict]:
    """对单只股票运行回测"""
    trades = []
    position = None
    capital = config.initial_capital / config.max_stocks

    # factors 已经包含 close 列，直接使用
    merged = factors.copy()

    # 删除 NaN 价格
    merged = merged.dropna(subset=['close'])

    if len(merged) < config.window_size + 10:
        return []

    # 回测主循环
    for idx, row in merged.iterrows():
        trade_time = row['trade_time']
        close_price = row['close']

        # 获取因子值
        path_irrev = row.get('path_irreversibility', np.nan)
        perm_ent = row.get('permutation_entropy', np.nan)
        turn_ent = row.get('turnover_entropy', np.nan)
        dom_eig = row.get('dominant_eigenvalue', np.nan)
        market_state = row.get('market_state', 'unknown')

        # 检查是否需要平仓
        if position is not None:
            should_exit = False
            exit_reason = None

            returns = (close_price - position['entry_price']) / position['entry_price']

            # 卖出条件 1: 临界转变预警
            if np.isfinite(dom_eig) and abs(dom_eig) > 0.9:
                should_exit = True
                exit_reason = "critical_warning"

            # 卖出条件 2: 止盈
            elif returns > 0.05:  # +5% 止盈
                should_exit = True
                exit_reason = "take_profit_+5%"

            # 卖出条件 3: 止损
            elif returns < -0.03:  # -3% 止损
                should_exit = True
                exit_reason = "stop_loss_-3%"

            # 卖出条件 4: 状态恶化
            elif market_state == 'strong_chaos':
                should_exit = True
                exit_reason = "state_deterioration"

            # 执行平仓
            if should_exit:
                shares = position['shares']
                exit_price = close_price

                # 计算收益
                proceeds = shares * exit_price
                commission = max(5, proceeds * config.commission)
                stamp_tax = proceeds * config.stamp_tax
                slip = proceeds * config.slippage
                net_proceeds = proceeds - commission - stamp_tax - slip

                # 成本
                cost = shares * position['entry_price']
                entry_comm = max(5, cost * config.commission)
                entry_slip = cost * config.slippage
                total_cost = cost + entry_comm + entry_slip

                pnl = net_proceeds - total_cost
                trade_ret = pnl / total_cost

                trades.append({
                    'stock_code': data['stock_code'].iloc[0],
                    'entry_time': position['entry_time'],
                    'exit_time': trade_time,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'shares': shares,
                    'pnl': pnl,
                    'returns': trade_ret,
                    'exit_reason': exit_reason
                })

                capital += net_proceeds
                position = None

        # 检查是否可以开仓
        if position is None and np.isfinite(path_irrev):
            # 买入条件：三个熵同时满足有序状态
            # 根据实际因子分布调整（path_irrev 均值 0.01，perm_ent 均值 0.97）：
            condition_1 = path_irrev > 0.01   # 路径不可逆性高于中位数
            condition_2 = perm_ent < 0.97     # 排列熵低于中位数（相对有序）
            condition_3 = market_state in ['ordered', 'weak_chaos']  # 非强混沌状态
            condition_4 = np.isfinite(dom_eig) and abs(dom_eig) < 0.9  # 非临界状态

            if condition_1 and condition_2 and condition_3 and condition_4:
                if close_price > 0 and np.isfinite(close_price):
                    # 计算买入数量
                    pos_value = capital * config.position_size
                    shares = int(pos_value / close_price / 100) * 100

                    if shares >= 100:
                        cost = shares * close_price
                        comm = max(5, cost * config.commission)
                        slip = cost * config.slippage
                        total = cost + comm + slip

                        if total <= capital:
                            capital -= total
                            position = {
                                'entry_time': trade_time,
                                'entry_price': close_price,
                                'shares': shares
                            }

    # 期末平仓
    if position is not None and len(merged) > 0:
        last_row = merged.iloc[-1]
        exit_price = last_row['close']
        shares = position['shares']

        proceeds = shares * exit_price
        commission = max(5, proceeds * config.commission)
        stamp_tax = proceeds * config.stamp_tax
        slip = proceeds * config.slippage
        net_proceeds = proceeds - commission - stamp_tax - slip

        cost = shares * position['entry_price']
        entry_comm = max(5, cost * config.commission)
        entry_slip = cost * config.slippage
        total_cost = cost + entry_comm + entry_slip

        pnl = net_proceeds - total_cost
        trades.append({
            'stock_code': data['stock_code'].iloc[0],
            'entry_time': position['entry_time'],
            'exit_time': last_row['trade_time'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'shares': shares,
            'pnl': pnl,
            'returns': pnl / total_cost,
            'exit_reason': 'end_of_period'
        })

    return trades


# =============================================================================
# 主函数
# =============================================================================

def run_backtest():
    """运行回测"""
    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)

    # 获取股票池
    stock_dirs = [d for d in os.listdir(config.data_dir)
                  if os.path.isdir(os.path.join(config.data_dir, d))]
    stock_dirs = sorted(stock_dirs)[:config.max_stocks]

    print(f"股票池：{len(stock_dirs)} 只股票", flush=True)
    print(f"时间范围：{config.start_date} - {config.end_date}", flush=True)
    print(f"窗口大小：{config.window_size} 分钟 ({config.window_days}天)", flush=True)

    all_trades = []

    for stock_code in stock_dirs:
        print(f"\n处理 {stock_code}...", flush=True)

        # 加载数据 (需要额外的历史数据用于计算窗口)
        start_load = "2025-01-01"  # 提前加载用于计算窗口
        data = load_stock_data(config.data_dir, stock_code, start_load, config.end_date)

        if len(data) < config.window_size + 100:
            print(f"  数据不足，跳过", flush=True)
            continue

        # 计算多天窗口的熵因子
        print(f"  计算熵因子...", flush=True)
        factors = calculate_multi_day_factors(data, config.window_size)

        if factors.empty:
            print(f"  因子计算失败，跳过", flush=True)
            continue

        # 运行回测
        print(f"  运行回测...", flush=True)
        trades = backtest_single_stock(data, factors, config)
        all_trades.extend(trades)

        print(f"  完成，产生 {len(trades)} 笔交易", flush=True)

    # 汇总结果
    print(f"\n{'='*60}")
    print("回测结果汇总")
    print(f"{'='*60}")

    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_df.to_csv(os.path.join(config.output_dir, "trades_all.csv"), index=False)

        total_trades = len(trades_df)
        winning = len(trades_df[trades_df['pnl'] > 0])
        win_rate = winning / total_trades if total_trades > 0 else 0
        total_pnl = trades_df['pnl'].sum()
        total_return = (1 + trades_df['returns']).prod() - 1
        avg_return = trades_df['returns'].mean()

        print(f"总交易次数：{total_trades}")
        print(f"盈利次数：{winning}")
        print(f"亏损次数：{total_trades - winning}")
        print(f"胜率：{win_rate*100:.2f}%")
        print(f"总盈亏：{total_pnl:.2f}元")
        print(f"总收益率：{total_return*100:.2f}%")
        print(f"平均收益率：{avg_return*100:.4f}%")

        # 退出原因统计
        print(f"\n退出原因分布:")
        for reason, count in trades_df['exit_reason'].value_counts().items():
            print(f"  {reason}: {count} ({count/total_trades*100:.1f}%)")

        # 按股票统计
        print(f"\n按股票统计 (前 10):")
        stock_stats = trades_df.groupby('stock_code').agg({
            'pnl': ['count', 'sum'],
            'returns': 'mean'
        }).round(3)
        print(stock_stats.head(10))

    else:
        print("无交易!")

    return all_trades


if __name__ == "__main__":
    run_backtest()
