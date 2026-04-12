"""
熵因子分钟数据回测 - 修复版本

修复了退出价格获取的 bug
"""

import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.tick_entropy import build_tick_entropy_features


# =============================================================================
# 配置
# =============================================================================

@dataclass
class BacktestConfig:
    """回测配置"""
    data_dir: str = "/nvme5/xtang/gp-workspace/gp-data/trade"
    output_dir: str = "/nvme5/xtang/gp-workspace/gp-quant/results/entropy_backtest"

    max_stocks: int = 20
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"
    window: int = 60

    initial_capital: float = 1_000_000.0
    position_size: float = 0.1
    max_positions: int = 5

    commission: float = 0.0003
    stamp_tax: float = 0.001
    slippage: float = 0.001


# =============================================================================
# 数据处理
# =============================================================================

def load_and_merge_data(data_dir: str, stock_codes: List[str],
                        start_date: str, end_date: str) -> pd.DataFrame:
    """加载并合并多只股票数据"""
    all_data = []

    for code in stock_codes:
        stock_dir = os.path.join(data_dir, code)
        if not os.path.exists(stock_dir):
            continue

        csv_files = glob.glob(os.path.join(stock_dir, "*.csv"))

        for f in csv_files:
            date_str = os.path.basename(f).replace('.csv', '')
            if date_str < start_date or date_str > end_date:
                continue

            try:
                df = pd.read_csv(f)
                df['stock_code'] = code
                all_data.append(df)
            except Exception:
                continue

    if not all_data:
        return pd.DataFrame()

    data = pd.concat(all_data, ignore_index=True)
    data['trade_time'] = pd.to_datetime(data['时间'])

    # 重命名列
    data = data.rename(columns={
        '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low',
        '成交量 (手)': 'volume', '成交额 (元)': 'amount',
        '均价': 'avg_price', '换手率 (%)': 'turnover_rate'
    })

    # 处理缺失的换手率数据
    if 'turnover_rate' in data.columns:
        data['turnover_rate'] = data['turnover_rate'].fillna(0.01)
        # 将百分比转换为小数 (如果原始数据是百分比形式如 0.05 表示 5%)
        # 检查数据范围，如果都小于 1，可能是百分比小数形式
        if data['turnover_rate'].max() < 1:
            data['turnover_rate'] = data['turnover_rate'] * 100
    else:
        # 如果没有换手率，用成交量近似
        data['turnover_rate'] = 1.0

    data = data.sort_values(['stock_code', 'trade_time']).reset_index(drop=True)
    return data


def calculate_factors(data: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """计算熵因子"""
    results = []

    # 检查必要列是否存在
    if 'turnover_rate' not in data.columns:
        # 尝试从原始列名映射
        if '换手率 (%)' in data.columns:
            data = data.rename(columns={'换手率 (%)': 'turnover_rate'})
        else:
            # 如果没有换手率，用成交量代替
            data['turnover_rate'] = data['volume'] / data['volume'].mean() if 'volume' in data.columns else 1.0

    for stock_code in data['stock_code'].unique():
        stock_data = data[data['stock_code'] == stock_code].copy()
        stock_data = stock_data.sort_values('trade_time').reset_index(drop=True)

        if len(stock_data) < window + 10:
            continue

        # 检查换手率是否有有效数据
        turnover = stock_data['turnover_rate']
        if turnover.isna().all() or (turnover == 0).all():
            # 用成交量变化率作为替代
            stock_data['turnover_rate'] = stock_data['volume'].pct_change().fillna(0.01).abs() + 0.01

        input_df = pd.DataFrame({
            'trade_time': stock_data['trade_time'],
            'price': stock_data['close'],
            'turnover_rate': stock_data['turnover_rate']
        })

        entropy_features = build_tick_entropy_features(
            input_df,
            windows={'path_irrev': window, 'perm_entropy': window, 'turnover': window}
        )

        if entropy_features.empty:
            continue

        entropy_features['stock_code'] = stock_code
        results.append(entropy_features)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


# =============================================================================
# 简化的事件驱动回测
# =============================================================================

class SimpleBacktester:
    """简化的事件驱动回测器"""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.capital = config.initial_capital
        self.positions = {}  # stock_code -> {entry_time, entry_price, shares}
        self.trades = []
        self.equity_curve = []

    def run_single_stock(self, stock_data: pd.DataFrame,
                         factors: pd.DataFrame,
                         exit_scheme: str) -> List[Dict]:
        """对单只股票运行回测"""
        trades = []
        position = None
        capital = self.config.initial_capital / self.config.max_stocks

        # 合并数据和因子
        merged = factors.merge(
            stock_data[['trade_time', 'close']],
            on='trade_time',
            suffixes=('', '_market')
        )

        if len(merged) < self.config.window + 10:
            return []

        # 索引用于查找退出价格
        price_index = stock_data.set_index('trade_time')['close'].to_dict()

        for idx, row in merged.iterrows():
            trade_time = row['trade_time']
            close_price = row['close']

            # 检查是否需要平仓
            if position is not None:
                should_exit = False
                exit_reason = None

                # 计算当前收益
                if trade_time in price_index:
                    current_price = price_index[trade_time]
                else:
                    current_price = close_price

                returns = (current_price - position['entry_price']) / position['entry_price']
                hold_minutes = (trade_time - position['entry_time']).total_seconds() / 60

                # 根据不同方案判断退出
                if exit_scheme == 'A':
                    if hold_minutes >= 30:
                        should_exit = True
                        exit_reason = "fixed_hold_30min"

                elif exit_scheme == 'B':
                    current_state = row.get('market_state', 'weak_chaos')
                    path_irrev = row.get('path_irreversibility', 0)
                    perm_ent = row.get('permutation_entropy', 0.5)

                    if current_state == 'strong_chaos':
                        should_exit = True
                        exit_reason = "state_deterioration"
                    elif path_irrev < 0.1:
                        should_exit = True
                        exit_reason = "path_irrev_drop"
                    elif perm_ent > 0.8:
                        should_exit = True
                        exit_reason = "perm_ent_high"
                    elif hold_minutes >= 120:
                        should_exit = True
                        exit_reason = "max_hold_time"

                elif exit_scheme == 'C':
                    if returns < -0.02:
                        should_exit = True
                        exit_reason = "stop_loss_-2%"
                    elif returns > 0.03:
                        should_exit = True
                        exit_reason = "take_profit_+3%"
                    elif hold_minutes >= 60 and returns <= 0:
                        should_exit = True
                        exit_reason = "time_stop_60min"
                    elif hold_minutes >= 120:
                        should_exit = True
                        exit_reason = "max_hold_120min"

                elif exit_scheme == 'D':
                    if returns < -0.02:
                        should_exit = True
                        exit_reason = "stop_loss_-2%"
                    elif returns > 0.03:
                        should_exit = True
                        exit_reason = "take_profit_+3%"
                    else:
                        current_state = row.get('market_state', 'weak_chaos')
                        if current_state == 'strong_chaos':
                            should_exit = True
                            exit_reason = "state_deterioration"

                    if not should_exit and hold_minutes >= 60 and returns <= 0:
                        should_exit = True
                        exit_reason = "time_stop_60min"
                    elif not should_exit and hold_minutes >= 120:
                        should_exit = True
                        exit_reason = "max_hold_120min"

                # 执行平仓
                if should_exit:
                    exit_price = current_price
                    shares = position['shares']

                    # 计算收益
                    proceeds = shares * exit_price
                    commission = max(5, proceeds * self.config.commission)
                    stamp_tax = proceeds * self.config.stamp_tax
                    slippage = proceeds * self.config.slippage
                    net_proceeds = proceeds - commission - stamp_tax - slippage

                    # 计算成本
                    cost = shares * position['entry_price']
                    entry_commission = max(5, cost * self.config.commission)
                    entry_slippage = cost * self.config.slippage
                    total_cost = cost + entry_commission + entry_slippage

                    pnl = net_proceeds - total_cost
                    trade_return = pnl / total_cost

                    trades.append({
                        'stock_code': stock_data['stock_code'].iloc[0],
                        'entry_time': position['entry_time'],
                        'exit_time': trade_time,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'shares': shares,
                        'pnl': pnl,
                        'returns': trade_return,
                        'exit_reason': exit_reason,
                        'exit_scheme': exit_scheme
                    })

                    capital += net_proceeds
                    position = None

            # 检查是否可以开仓
            if position is None:
                path_irrev = row.get('path_irreversibility', 0)
                perm_ent = row.get('permutation_entropy', 0.5)
                market_state = row.get('market_state', 'weak_chaos')

                # 买入条件
                if path_irrev > 0.15 and perm_ent < 0.7 and market_state in ['ordered', 'weak_chaos']:
                    if close_price > 0 and np.isfinite(close_price):
                        # 计算买入数量
                        position_value = capital * self.config.position_size
                        shares = int(position_value / close_price / 100) * 100

                        if shares >= 100:
                            cost = shares * close_price
                            commission = max(5, cost * self.config.commission)
                            slippage = cost * self.config.slippage
                            total_cost = cost + commission + slippage

                            if total_cost <= capital:
                                capital -= total_cost
                                position = {
                                    'entry_time': trade_time,
                                    'entry_price': close_price,
                                    'shares': shares
                                }

        # 如果还有持仓，在最后一个时间点平仓
        if position is not None and len(merged) > 0:
            last_row = merged.iloc[-1]
            exit_time = last_row['trade_time']
            exit_price = last_row['close']
            shares = position['shares']

            proceeds = shares * exit_price
            commission = max(5, proceeds * self.config.commission)
            stamp_tax = proceeds * self.config.stamp_tax
            slippage = proceeds * self.config.slippage
            net_proceeds = proceeds - commission - stamp_tax - slippage

            cost = shares * position['entry_price']
            entry_commission = max(5, cost * self.config.commission)
            entry_slippage = cost * self.config.slippage
            total_cost = cost + entry_commission + entry_slippage

            pnl = net_proceeds - total_cost

            trades.append({
                'stock_code': stock_data['stock_code'].iloc[0],
                'entry_time': position['entry_time'],
                'exit_time': exit_time,
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'shares': shares,
                'pnl': pnl,
                'returns': pnl / total_cost,
                'exit_reason': 'end_of_data',
                'exit_scheme': exit_scheme
            })

        return trades


def run_backtest():
    """运行回测"""
    config = BacktestConfig(max_stocks=20, start_date="2024-01-01", end_date="2024-12-31")
    os.makedirs(config.output_dir, exist_ok=True)

    # 获取股票池
    stock_dirs = [d for d in os.listdir(config.data_dir)
                  if os.path.isdir(os.path.join(config.data_dir, d))]
    stock_dirs = sorted(stock_dirs)[:config.max_stocks]

    print(f"股票池：{len(stock_dirs)} 只股票")

    # 加载数据
    print("加载数据...")
    data = load_and_merge_data(config.data_dir, stock_dirs,
                               config.start_date, config.end_date)
    print(f"数据量：{len(data)} 条")

    # 计算因子
    print("计算熵因子...")
    factors = calculate_factors(data, config.window)
    print(f"因子数据：{len(factors)} 条")

    # 运行 4 种方案
    schemes = ['A', 'B', 'C', 'D']
    all_trades = {s: [] for s in schemes}

    for scheme in schemes:
        print(f"\n{'='*60}")
        print(f"开始回测 - 方案 {scheme}")
        print(f"{'='*60}")

        backtester = SimpleBacktester(config)
        scheme_trades = []

        for stock_code in stock_dirs:
            stock_data = data[data['stock_code'] == stock_code]
            stock_factors = factors[factors['stock_code'] == stock_code]

            if len(stock_data) < 100 or len(stock_factors) < 10:
                continue

            trades = backtester.run_single_stock(stock_data, stock_factors, scheme)
            scheme_trades.extend(trades)

        all_trades[scheme] = scheme_trades

        # 统计
        if scheme_trades:
            trades_df = pd.DataFrame(scheme_trades)
            total_trades = len(trades_df)
            winning = len(trades_df[trades_df['pnl'] > 0])
            win_rate = winning / total_trades if total_trades > 0 else 0
            total_pnl = trades_df['pnl'].sum()
            avg_return = trades_df['returns'].mean()
            total_return = (1 + trades_df['returns']).prod() - 1

            print(f"总交易次数：{total_trades}")
            print(f"盈利次数：{winning}")
            print(f"亏损次数：{total_trades - winning}")
            print(f"胜率：{win_rate*100:.2f}%")
            print(f"总盈亏：{total_pnl:.2f}元")
            print(f"平均收益率：{avg_return*100:.2f}%")
            print(f"总收益率：{total_return*100:.2f}%")

            # 保存结果
            trades_df.to_csv(os.path.join(config.output_dir, f"trades_scheme_{scheme}.csv"), index=False)
            print(f"交易记录已保存")
        else:
            print("无交易!")

    # 对比
    print(f"\n{'='*60}")
    print("4 种方案对比")
    print(f"{'='*60}")
    print(f"{'方案':<10} {'交易次数':<12} {'胜率':<10} {'总盈亏 (元)':<15} {'总收益率':<12}")
    print(f"{'-'*60}")

    for scheme in schemes:
        trades = all_trades[scheme]
        if trades:
            df = pd.DataFrame(trades)
            total = len(df)
            win_rate = len(df[df['pnl'] > 0]) / total if total > 0 else 0
            total_pnl = df['pnl'].sum()
            total_ret = (1 + df['returns']).prod() - 1
            print(f"{scheme:<10} {total:<12} {win_rate*100:<10.1f}% {total_pnl:<15.2f} {total_ret*100:<12.1f}%")
        else:
            print(f"{scheme:<10} 0            0.0%       0.00           0.0%")

    print(f"{'='*60}")

    return all_trades


if __name__ == "__main__":
    run_backtest()
