"""
日线级别熵因子策略回测

基于因子验证结果的发现：
1. 换手率熵在日线级别有显著预测能力（IC = -0.16, IC IR = -1.15）
2. 低换手熵 → 主力集中 → 未来正收益
3. 高换手熵 → 散户博弈 → 未来负收益

策略逻辑：
- 买入：低换手率熵 + 有序市场状态
- 卖出：高换手率熵（散户博弈）+ 止盈止损
"""

import os
import glob
import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.tick_entropy import (
    rolling_path_irreversibility,
    rolling_permutation_entropy,
    turnover_rate_entropy,
    market_state_classifier,
    dominant_eigenvalue_from_autocorr,
)


# =============================================================================
# 配置
# =============================================================================

@dataclass
class Config:
    # 数据路径
    minute_data_dir: str = "/nvme5/xtang/gp-workspace/gp-data/trade"
    output_dir: str = "/nvme5/xtang/gp-workspace/gp-quant/results/daily_backtest"

    # 时间范围：2025 年 1 月 -2026 年 4 月
    start_date: str = "2025-01-01"
    end_date: str = "2026-04-03"

    # 股票池配置
    max_stocks: int = 50
    min_market_cap: float = 0

    # 因子计算窗口（日线）
    window: int = 20  # 20 个交易日

    # 交易配置
    initial_capital: float = 1_000_000.0
    position_size: float = 0.1  # 单只股票最大仓位
    max_positions: int = 10  # 最大持仓数

    # 费用
    commission: float = 0.0003  # 万三
    stamp_tax: float = 0.001    # 千一（卖出收取）
    slippage: float = 0.001     # 千一滑点

    # 止盈止损
    take_profit: float = 0.08   # 8% 止盈
    stop_loss: float = 0.05     # 5% 止损
    max_holding_days: int = 20  # 最长持仓 20 天

    # 策略参数
    turnover_entropy_low: float = 0.7   # 低换手熵阈值（买入）
    turnover_entropy_high: float = 0.85  # 高换手熵阈值（卖出）
    path_irrev_min: float = 0.0         # 路径不可逆性最小值
    perm_entropy_max: float = 1.0       # 排列熵最大值


# =============================================================================
# 数据加载
# =============================================================================

def load_daily_data(
    data_dir: str,
    stock_code: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """从分钟数据聚合日线数据"""
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
    data['date'] = data['trade_time'].dt.date

    # 重命名列
    data = data.rename(columns={
        '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low',
        '成交量(手)': 'volume', '成交额 (元)': 'amount',
        '均价': 'avg_price', '换手率(%)': 'turnover_rate'
    })

    # 按天聚合
    agg_cols = {
        'stock_code': 'first',
        'open': 'first',
        'close': 'last',
        'high': 'max',
        'low': 'min',
    }
    if 'volume' in data.columns and not data['volume'].isna().all():
        agg_cols['volume'] = 'sum'
    if 'turnover_rate' in data.columns and not data['turnover_rate'].isna().all():
        agg_cols['turnover_rate'] = 'sum'

    daily = data.groupby('date').agg(agg_cols).reset_index()
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.sort_values(['stock_code', 'date']).reset_index(drop=True)

    return daily


# =============================================================================
# 因子计算
# =============================================================================

def calculate_daily_factors(
    data: pd.DataFrame,
    window: int = 20
) -> pd.DataFrame:
    """计算日线级别的熵因子"""
    if len(data) < window + 10:
        return pd.DataFrame()

    data = data.sort_values('date').reset_index(drop=True)
    data['log_ret'] = np.log(data['close']).diff()

    results = pd.DataFrame({
        'date': data['date'],
        'close': data['close'],
        'open': data['open'],
        'high': data['high'],
        'low': data['low'],
        'volume': data.get('volume', pd.Series([0] * len(data))),
        'stock_code': data['stock_code'].iloc[0]
    })

    # 1. 路径不可逆性熵
    print("  计算路径不可逆性熵...")
    results['path_irreversibility'] = rolling_path_irreversibility(
        data['log_ret'], window=window
    )

    # 2. 排列熵
    print("  计算排列熵...")
    results['permutation_entropy'] = rolling_permutation_entropy(
        data['log_ret'], window=window, order=3
    )

    # 3. 换手率熵
    print("  计算换手率熵...")
    if 'turnover_rate' in data.columns:
        results['turnover_entropy'] = data['turnover_rate'].rolling(
            window=window, min_periods=10
        ).apply(lambda x: turnover_rate_entropy(x.values, n_bins=10), raw=False)
    else:
        results['turnover_entropy'] = np.nan

    # 4. 主导特征值
    print("  计算主导特征值...")
    def calc_dominant_eig(x):
        if isinstance(x, np.ndarray):
            return dominant_eigenvalue_from_autocorr(x, order=2)
        return dominant_eigenvalue_from_autocorr(x.values, order=2)

    results['dominant_eigenvalue'] = data['log_ret'].rolling(
        window=window, min_periods=10
    ).apply(calc_dominant_eig, raw=False)

    # 5. 市场状态
    print("  计算市场状态...")
    results['market_state'] = results.apply(
        lambda row: market_state_classifier(
            row['path_irreversibility'],
            row['permutation_entropy'],
            turnover_entropy=row.get('turnover_entropy', np.nan)
        ),
        axis=1
    )

    return results


def process_single_stock(args: tuple) -> tuple:
    """处理单只股票（用于并行化）"""
    stock_code, data_dir, start_date, end_date, window = args

    try:
        data = load_daily_data(data_dir, stock_code, start_date, end_date)

        if len(data) < 30:
            return stock_code, pd.DataFrame()

        factors = calculate_daily_factors(data, window)

        if factors.empty:
            return stock_code, pd.DataFrame()

        return stock_code, factors
    except Exception as e:
        print(f"  {stock_code} 处理失败：{e}")
        return stock_code, pd.DataFrame()


# =============================================================================
# 回测引擎
# =============================================================================

class DailyBacktester:
    """日线级别回测器"""

    def __init__(self, config: Config):
        self.config = config
        self.capital = config.initial_capital
        self.positions = {}  # stock_code -> {entry_date, entry_price, shares, entry_turnover_entropy}

    def run_backtest(self, all_factors: List[pd.DataFrame]) -> pd.DataFrame:
        """运行回测"""
        # 合并所有因子数据
        combined = pd.concat(all_factors, ignore_index=True)
        combined = combined.dropna(subset=['close', 'turnover_entropy', 'path_irreversibility'])
        combined = combined.sort_values('date')

        # 计算未来 N 日收益（用于分析）
        for days in [1, 3, 5]:
            combined[f'future_{days}d_ret'] = combined.groupby('stock_code')['close'].shift(-days) / combined['close'] - 1

        trades = []
        self.capital = self.config.initial_capital
        self.positions = {}

        dates = combined['date'].unique()

        print(f"\n开始回测，共 {len(dates)} 个交易日...")

        for current_date in tqdm(dates, desc="回测中"):
            day_data = combined[combined['date'] == current_date]

            # 1. 检查现有持仓是否需要平仓
            for stock_code in list(self.positions.keys()):
                position = self.positions[stock_code]
                stock_data = day_data[day_data['stock_code'] == stock_code]

                if len(stock_data) == 0:
                    continue

                row = stock_data.iloc[0]
                current_price = row['close']
                current_turnover_entropy = row['turnover_entropy']

                # 计算收益
                returns = (current_price - position['entry_price']) / position['entry_price']

                # 检查平仓条件
                should_exit = False
                exit_reason = None

                # 止盈
                if returns >= self.config.take_profit:
                    should_exit = True
                    exit_reason = "take_profit"

                # 止损
                elif returns <= -self.config.stop_loss:
                    should_exit = True
                    exit_reason = "stop_loss"

                # 高换手熵（散户博弈信号）
                elif current_turnover_entropy > self.config.turnover_entropy_high:
                    should_exit = True
                    exit_reason = "high_turnover_entropy"

                # 持仓时间到期
                holding_days = (current_date - position['entry_date']).days
                if holding_days >= self.config.max_holding_days:
                    should_exit = True
                    exit_reason = "max_holding_days"

                if should_exit:
                    trade = self._exit_position(
                        stock_code=stock_code,
                        position=position,
                        exit_date=current_date,
                        exit_price=current_price,
                        exit_reason=exit_reason,
                        entry_turnover_entropy=position.get('entry_turnover_entropy', np.nan)
                    )
                    trades.append(trade)
                    del self.positions[stock_code]

            # 2. 检查是否可以开新仓
            if len(self.positions) < self.config.max_positions:
                # 筛选买入候选
                buy_signals = day_data[
                    (day_data['turnover_entropy'] < self.config.turnover_entropy_low) &
                    (day_data['path_irreversibility'] > self.config.path_irrev_min) &
                    (day_data['permutation_entropy'] < self.config.perm_entropy_max) &
                    (~day_data['market_state'].isin(['strong_chaos']))
                ]

                # 按换手熵排序，选择最低的
                buy_signals = buy_signals.sort_values('turnover_entropy')

                for _, row in buy_signals.iterrows():
                    if len(self.positions) >= self.config.max_positions:
                        break

                    stock_code = row['stock_code']
                    if stock_code in self.positions:
                        continue

                    current_price = row['close']
                    if current_price <= 0:
                        continue

                    # 计算买入数量
                    position_value = self.capital * self.config.position_size
                    shares = int(position_value / current_price / 100) * 100

                    if shares < 100:
                        continue

                    # 计算成本
                    cost = shares * current_price
                    commission = max(5, cost * self.config.commission)
                    slippage = cost * self.config.slippage
                    total_cost = cost + commission + slippage

                    if total_cost > self.capital:
                        continue

                    # 开仓
                    self.capital -= total_cost
                    self.positions[stock_code] = {
                        'entry_date': current_date,
                        'entry_price': current_price,
                        'shares': shares,
                        'entry_turnover_entropy': row['turnover_entropy']
                    }

        # 3. 期末平仓
        print("\n期末平仓...")
        last_date = combined['date'].max()
        last_day_data = combined[combined['date'] == last_date]

        for stock_code in list(self.positions.keys()):
            position = self.positions[stock_code]
            stock_data = last_day_data[last_day_data['stock_code'] == stock_code]

            if len(stock_data) == 0:
                continue

            row = stock_data.iloc[0]
            trade = self._exit_position(
                stock_code=stock_code,
                position=position,
                exit_date=last_date,
                exit_price=row['close'],
                exit_reason="end_of_period",
                entry_turnover_entropy=position.get('entry_turnover_entropy', np.nan)
            )
            trades.append(trade)

        return pd.DataFrame(trades)

    def _exit_position(
        self,
        stock_code: str,
        position: dict,
        exit_date: pd.Timestamp,
        exit_price: float,
        exit_reason: str,
        entry_turnover_entropy: float
    ) -> dict:
        """平仓并返回交易记录"""
        shares = position['shares']
        entry_price = position['entry_price']

        # 计算收益
        proceeds = shares * exit_price
        commission = max(5, proceeds * self.config.commission)
        stamp_tax = proceeds * self.config.stamp_tax
        slippage = proceeds * self.config.slippage
        net_proceeds = proceeds - commission - stamp_tax - slippage

        # 成本
        cost = shares * entry_price
        entry_comm = max(5, cost * self.config.commission)
        entry_slip = cost * self.config.slippage
        total_cost = cost + entry_comm + entry_slip

        pnl = net_proceeds - total_cost
        trade_ret = pnl / total_cost

        self.capital += net_proceeds

        return {
            'stock_code': stock_code,
            'entry_date': position['entry_date'],
            'exit_date': exit_date,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'shares': shares,
            'entry_turnover_entropy': entry_turnover_entropy,
            'pnl': pnl,
            'returns': trade_ret,
            'exit_reason': exit_reason,
            'holding_days': (exit_date - position['entry_date']).days
        }


# =============================================================================
# 结果分析
# =============================================================================

def analyze_results(trades: pd.DataFrame) -> None:
    """分析回测结果"""
    if trades.empty:
        print("无交易记录")
        return

    print("\n" + "=" * 80)
    print("回测结果分析")
    print("=" * 80)

    # 基础统计
    total_trades = len(trades)
    winning = len(trades[trades['pnl'] > 0])
    win_rate = winning / total_trades * 100 if total_trades > 0 else 0

    print(f"\n基础统计:")
    print(f"  总交易数：{total_trades}")
    print(f"  盈利交易：{winning} ({win_rate:.1f}%)")
    print(f"  亏损交易：{total_trades - winning} ({100 - win_rate:.1f}%)")

    # 盈亏统计
    total_pnl = trades['pnl'].sum()
    total_return = (1 + trades['returns']).prod() - 1

    winning_trades = trades[trades['pnl'] > 0]
    losing_trades = trades[trades['pnl'] <= 0]

    avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0

    print(f"\n盈亏统计:")
    print(f"  总盈亏：{total_pnl:.2f} 元")
    print(f"  总收益率：{total_return * 100:.2f}%")
    print(f"  平均盈利：{avg_win:.2f} 元")
    print(f"  平均亏损：{avg_loss:.2f} 元")

    if avg_loss != 0:
        profit_loss_ratio = abs(avg_win / avg_loss)
        print(f"  盈亏比：{profit_loss_ratio:.2f}")

    # 退出原因分布
    print(f"\n退出原因分布:")
    exit_stats = trades.groupby('exit_reason').agg({
        'pnl': ['count', 'mean'],
        'returns': 'mean'
    }).round(4)
    exit_stats.columns = ['count', 'avg_pnl', 'avg_return']
    exit_stats['pct'] = exit_stats['count'] / total_trades * 100
    print(exit_stats.to_string())

    # 持仓时间分析
    print(f"\n持仓时间分析:")
    print(f"  平均持仓：{trades['holding_days'].mean():.1f} 天")
    print(f"  中位数持仓：{trades['holding_days'].median():.1f} 天")

    # 按持仓时间分组
    trades['holding_bucket'] = pd.cut(
        trades['holding_days'],
        bins=[0, 3, 5, 10, 20, float('inf')],
        labels=['1-3 天', '4-5 天', '6-10 天', '11-20 天', '>20 天']
    )

    for bucket in trades['holding_bucket'].cat.categories:
        subset = trades[trades['holding_bucket'] == bucket]
        if len(subset) > 0:
            bucket_win_rate = len(subset[subset['pnl'] > 0]) / len(subset) * 100
            bucket_ret = subset['returns'].mean() * 100
            print(f"  {bucket}: {len(subset)} 交易，胜率 {bucket_win_rate:.1f}%, 平均收益 {bucket_ret:.3f}%")

    # 按入场换手熵分组
    print(f"\n按入场换手熵分组:")
    trades['entry_te_bucket'] = pd.qcut(
        trades['entry_turnover_entropy'].rank(method='first'),
        q=5,
        labels=['Q1(最低)', 'Q2', 'Q3', 'Q4', 'Q5(最高)']
    )

    for bucket in trades['entry_te_bucket'].cat.categories:
        subset = trades[trades['entry_te_bucket'] == bucket]
        if len(subset) > 0:
            bucket_win_rate = len(subset[subset['pnl'] > 0]) / len(subset) * 100
            bucket_ret = subset['returns'].mean() * 100
            print(f"  {bucket}: {len(subset)} 交易，胜率 {bucket_win_rate:.1f}%, 平均收益 {bucket_ret:.3f}%")

    # 月度收益
    print(f"\n月度收益:")
    trades['month'] = pd.to_datetime(trades['exit_date']).dt.to_period('M')
    monthly = trades.groupby('month').agg({
        'pnl': 'sum',
        'returns': lambda x: (1 + x).prod() - 1
    }).round(4)
    monthly.columns = ['pnl', 'return']
    print(monthly.to_string())

    # 绘制权益曲线数据
    print(f"\n权益曲线数据 (前 20 个交易日):")
    trades_sorted = trades.sort_values('exit_date')
    trades_sorted['cum_return'] = (1 + trades_sorted['returns']).cumprod() - 1
    print(trades_sorted[['exit_date', 'pnl', 'cum_return']].head(20).to_string())


# =============================================================================
# 主函数
# =============================================================================

def main():
    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)

    print("=" * 80)
    print("日线级别熵因子策略回测")
    print("=" * 80)
    print(f"\n配置参数:")
    print(f"  时间范围：{config.start_date} - {config.end_date}")
    print(f"  股票数量：{config.max_stocks}")
    print(f"  因子窗口：{config.window} 天")
    print(f"  初始资金：{config.initial_capital:,.0f} 元")
    print(f"  止盈/止损：{config.take_profit*100:.1f}% / {config.stop_loss*100:.1f}%")
    print(f"  换手熵阈值：<{config.turnover_entropy_low} 买入，>{config.turnover_entropy_high} 卖出")

    # 获取股票池
    stock_dirs = [d for d in os.listdir(config.minute_data_dir)
                  if os.path.isdir(os.path.join(config.minute_data_dir, d))]
    stock_dirs = sorted(stock_dirs)[:config.max_stocks]

    print(f"\n股票池：{len(stock_dirs)} 只股票")

    all_factors = []

    # 使用多进程并行处理
    tasks = [
        (stock_code, config.minute_data_dir, config.start_date, config.end_date, config.window)
        for stock_code in stock_dirs
    ]

    print(f"\n开始处理 {len(tasks)} 只股票...")

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_single_stock, task): task[0] for task in tasks}

        for future in tqdm(as_completed(futures), total=len(futures), desc="计算因子"):
            stock_code, factors = future.result()

            if factors.empty:
                continue

            all_factors.append(factors)

    if not all_factors:
        print("\n没有有效数据，回测失败")
        return

    print(f"\n成功处理 {len(all_factors)} 只股票")

    # 运行回测
    backtester = DailyBacktester(config)
    trades = backtester.run_backtest(all_factors)

    # 保存结果
    trades.to_csv(os.path.join(config.output_dir, "trades_daily.csv"), index=False)
    print(f"\n交易记录已保存：{os.path.join(config.output_dir, 'trades_daily.csv')}")

    # 分析结果
    analyze_results(trades)

    # 保存策略汇总
    summary = {
        'total_trades': len(trades),
        'win_rate': len(trades[trades['pnl'] > 0]) / len(trades) * 100 if len(trades) > 0 else 0,
        'total_pnl': trades['pnl'].sum(),
        'total_return': (1 + trades['returns']).prod() - 1,
        'avg_win': trades[trades['pnl'] > 0]['pnl'].mean() if len(trades[trades['pnl'] > 0]) > 0 else 0,
        'avg_loss': trades[trades['pnl'] <= 0]['pnl'].mean() if len(trades[trades['pnl'] <= 0]) > 0 else 0,
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(config.output_dir, "summary.csv"), index=False)

    print(f"\n回测完成！结果已保存至：{config.output_dir}")


if __name__ == "__main__":
    main()
