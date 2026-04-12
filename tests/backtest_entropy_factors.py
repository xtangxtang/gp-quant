"""
熵因子分钟数据回测框架

验证 3 个核心熵因子在分钟 K 线数据上对未来收益率的预测能力
对比 4 种卖出方案的表现
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

from core.tick_entropy import (
    build_tick_entropy_features,
    rolling_path_irreversibility,
    rolling_permutation_entropy,
    rolling_turnover_entropy,
    market_state_classifier
)


# =============================================================================
# 配置
# =============================================================================

@dataclass
class BacktestConfig:
    """回测配置"""
    data_dir: str = "/nvme5/xtang/gp-workspace/gp-data/trade"
    output_dir: str = "/nvme5/xtang/gp-workspace/gp-quant/results/entropy_backtest"

    # 股票池配置
    stock_pool: List[str] = field(default_factory=list)  # 留空则使用全市场
    max_stocks: int = 100  # 最多测试多少只股票

    # 时间范围
    start_date: str = "2024-01-01"
    end_date: str = "2025-12-31"

    # 因子配置
    window: int = 60  # 滚动窗口 (分钟)

    # 交易配置
    initial_capital: float = 1_000_000.0
    position_size: float = 0.1  # 单次持仓比例 10%
    max_positions: int = 10  # 最多同时持仓数

    # 费用配置
    commission: float = 0.0003  # 万三佣金
    stamp_tax: float = 0.001  # 千一印花税 (卖出时收取)
    slippage: float = 0.001  # 千一滑点


# =============================================================================
# 数据加载器
# =============================================================================

class DataLoader:
    """分钟数据加载器"""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def get_stock_codes(self) -> List[str]:
        """获取所有股票代码"""
        dirs = [d for d in os.listdir(self.data_dir)
                if os.path.isdir(os.path.join(self.data_dir, d))]
        return sorted(dirs)

    def load_stock_data(self, stock_code: str,
                        start_date: str = None,
                        end_date: str = None) -> pd.DataFrame:
        """
        加载单只股票的分钟数据

        Returns:
            DataFrame with columns:
            - stock_code
            - trade_time (datetime)
            - open, high, low, close
            - volume, amount, avg_price
            - turnover_rate
        """
        stock_dir = os.path.join(self.data_dir, stock_code)
        if not os.path.exists(stock_dir):
            return pd.DataFrame()

        # 获取所有 CSV 文件
        csv_files = glob.glob(os.path.join(stock_dir, "*.csv"))
        if not csv_files:
            return pd.DataFrame()

        # 过滤日期范围
        dfs = []
        for f in csv_files:
            # 从文件名提取日期 (YYYY-MM-DD.csv)
            date_str = os.path.basename(f).replace('.csv', '')
            if start_date and date_str < start_date:
                continue
            if end_date and date_str > end_date:
                continue

            try:
                df = pd.read_csv(f)
                df['stock_code'] = stock_code
                dfs.append(df)
            except Exception:
                continue

        if not dfs:
            return pd.DataFrame()

        # 合并数据
        data = pd.concat(dfs, ignore_index=True)

        # 处理时间列
        data['trade_time'] = pd.to_datetime(data['时间'])

        # 标准化列名
        data = data.rename(columns={
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量 (手)': 'volume',
            '成交额 (元)': 'amount',
            '均价': 'avg_price',
            '换手率 (%)': 'turnover_rate'
        })

        # 检查必要列是否存在
        required_cols = ['stock_code', 'trade_time', 'open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                data[col] = np.nan

        # 添加可选列
        if 'volume' not in data.columns:
            data['volume'] = np.nan
        if 'amount' not in data.columns:
            data['amount'] = np.nan
        if 'avg_price' not in data.columns:
            data['avg_price'] = np.nan
        if 'turnover_rate' not in data.columns:
            data['turnover_rate'] = np.nan

        # 排序
        data = data.sort_values(['stock_code', 'trade_time']).reset_index(drop=True)

        return data[['stock_code', 'trade_time', 'open', 'high', 'low',
                     'close', 'volume', 'amount', 'avg_price', 'turnover_rate']]

    def filter_active_stocks(self, stock_codes: List[str],
                             min_days: int = 200) -> List[str]:
        """过滤活跃股票 (至少有 min_days 个交易日)"""
        active = []
        for code in stock_codes:
            stock_dir = os.path.join(self.data_dir, code)
            if not os.path.exists(stock_dir):
                continue
            days = len([f for f in os.listdir(stock_dir) if f.endswith('.csv')])
            if days >= min_days:
                active.append(code)
        return active


# =============================================================================
# 因子计算
# =============================================================================

class EntropyFactorCalculator:
    """熵因子计算器"""

    def __init__(self, window: int = 60):
        self.window = window

    def calculate_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 3 个核心熵因子

        Input DataFrame columns:
        - stock_code, trade_time, close, turnover_rate

        Output DataFrame columns:
        - stock_code, trade_time
        - path_irreversibility
        - permutation_entropy
        - turnover_entropy
        - market_state
        - close (收盘价)
        """
        results = []

        for stock_code in df['stock_code'].unique():
            stock_data = df[df['stock_code'] == stock_code].copy()
            stock_data = stock_data.sort_values('trade_time').reset_index(drop=True)

            if len(stock_data) < self.window + 10:
                continue

            # 准备输入数据
            input_df = pd.DataFrame({
                'trade_time': stock_data['trade_time'],
                'price': stock_data['close'],
                'turnover_rate': stock_data['turnover_rate']
            })

            # 计算熵因子
            entropy_features = build_tick_entropy_features(
                input_df,
                windows={
                    'path_irrev': self.window,
                    'perm_entropy': self.window,
                    'turnover': self.window
                }
            )

            if entropy_features.empty:
                continue

            entropy_features['stock_code'] = stock_code
            # 保留收盘价用于回测
            entropy_features['close'] = stock_data['close'].values[:len(entropy_features)]
            results.append(entropy_features)

        if not results:
            return pd.DataFrame()

        result_df = pd.concat(results, ignore_index=True)

        # 标准化因子值 (按股票)
        result_df = self._standardize_factors(result_df)

        return result_df

    def _standardize_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """对因子进行标准化 (按股票)"""
        factors = ['path_irreversibility', 'permutation_entropy', 'turnover_entropy']

        for factor in factors:
            if factor not in df.columns:
                continue

            # 按股票分组标准化
            def standardize(group):
                values = group[factor]
                mean = values.mean()
                std = values.std()
                if std > 0:
                    group[f'{factor}_std'] = (values - mean) / std
                else:
                    group[f'{factor}_std'] = 0
                return group

            df = df.groupby('stock_code', group_keys=False).apply(standardize)

        return df


# =============================================================================
# 信号生成器
# =============================================================================

@dataclass
class Signal:
    """交易信号"""
    stock_code: str
    trade_time: pd.Timestamp
    action: str  # 'buy' or 'sell'
    price: float
    reason: str
    signal_type: str = 'entry'  # 'entry' or 'exit'


class SignalGenerator:
    """信号生成器"""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.window = config.window

    def generate_entry_signals(self, factors_df: pd.DataFrame) -> List[Signal]:
        """
        生成买入信号

        买入条件:
        1. path_irreversibility > 0.15 (主力开始控盘)
        2. permutation_entropy < 0.7 (非高度无序)
        3. market_state in ['ordered', 'weak_chaos']
        4. 无 critical 预警
        """
        signals = []

        for idx, row in factors_df.iterrows():
            if pd.isna(row.get('path_irreversibility')):
                continue

            # 获取因子值
            path_irrev = row.get('path_irreversibility', 0)
            perm_ent = row.get('permutation_entropy', 0.5)
            market_state = row.get('market_state', 'weak_chaos')
            close_price = row.get('close', 0)

            # 过滤无效价格
            if close_price is None or close_price <= 0 or np.isnan(close_price):
                continue

            # 买入条件判断
            condition_1 = path_irrev > 0.15
            condition_2 = perm_ent < 0.7
            condition_3 = market_state in ['ordered', 'weak_chaos']

            if condition_1 and condition_2 and condition_3:
                signals.append(Signal(
                    stock_code=row['stock_code'],
                    trade_time=row['trade_time'],
                    action='buy',
                    price=close_price,
                    reason=f"state={market_state}, path_irrev={path_irrev:.3f}",
                    signal_type='entry'
                ))

        return signals

    def generate_exit_signals(self,
                              entry_signal: Signal,
                              current_data: pd.DataFrame,
                              current_factors: pd.DataFrame,
                              exit_scheme: str) -> Tuple[bool, str]:
        """
        生成卖出信号

        4 种方案:
        A: 固定持有 30 分钟
        B: 因子驱动 (熵因子恶化)
        C: 止盈止损 (+3%/-2%)
        D: 组合方案 (B+C 取最先触发)
        """
        entry_time = entry_signal.trade_time
        entry_price = entry_signal.price

        # 获取当前数据
        current_row = current_data.iloc[-1] if len(current_data) > 0 else None
        current_time = current_row['trade_time'] if current_row is not None else entry_time
        current_price = current_row['close'] if current_row is not None else entry_price

        # 计算持仓时间 (分钟)
        hold_minutes = (current_time - entry_time).total_seconds() / 60

        # 计算收益率
        returns = (current_price - entry_price) / entry_price

        # ========== 方案 A: 固定持有 ==========
        if exit_scheme == 'A':
            if hold_minutes >= 30:
                return True, "fixed_hold_30min"
            return False, "holding"

        # ========== 方案 B: 因子驱动 ==========
        elif exit_scheme == 'B':
            if len(current_factors) > 0:
                last_factor = current_factors.iloc[-1]
                current_state = last_factor.get('market_state', 'weak_chaos')
                path_irrev = last_factor.get('path_irreversibility', 0)
                perm_ent = last_factor.get('permutation_entropy', 0.5)

                # 状态恶化
                if current_state == 'strong_chaos':
                    return True, "state_deterioration"

                # 路径不可逆性跌破
                if path_irrev < 0.1:
                    return True, "path_irrev_drop"

                # 排列熵突破 (进入无序)
                if perm_ent > 0.8:
                    return True, "perm_ent_high"

            # 默认持有
            if hold_minutes < 120:  # 最多持有 120 分钟
                return False, "holding"
            return True, "max_hold_time"

        # ========== 方案 C: 止盈止损 ==========
        elif exit_scheme == 'C':
            # 止损
            if returns < -0.02:
                return True, "stop_loss_-2%"

            # 止盈
            if returns > 0.03:
                return True, "take_profit_+3%"

            # 时间止损
            if hold_minutes >= 60 and returns <= 0:
                return True, "time_stop_60min"

            # 最大持有时间
            if hold_minutes >= 120:
                return True, "max_hold_120min"

            return False, "holding"

        # ========== 方案 D: 组合方案 ==========
        elif exit_scheme == 'D':
            # 1. 止损优先 (无条件)
            if returns < -0.02:
                return True, "stop_loss_-2%"

            # 2. 止盈
            if returns > 0.03:
                return True, "take_profit_+3%"

            # 3. 因子恶化
            if len(current_factors) > 0:
                last_factor = current_factors.iloc[-1]
                current_state = last_factor.get('market_state', 'weak_chaos')
                path_irrev = last_factor.get('path_irreversibility', 0)

                if current_state == 'strong_chaos':
                    return True, "state_deterioration"
                if path_irrev < 0.1:
                    return True, "path_irrev_drop"

            # 4. 时间止损
            if hold_minutes >= 60 and returns <= 0:
                return True, "time_stop_60min"

            # 5. 最大持有时间
            if hold_minutes >= 120:
                return True, "max_hold_120min"

            return False, "holding"

        return False, "unknown"


# =============================================================================
# 回测引擎
# =============================================================================

@dataclass
class Position:
    """持仓记录"""
    stock_code: str
    entry_time: pd.Timestamp
    entry_price: float
    shares: int
    signal: Signal = None


@dataclass
class Trade:
    """交易记录"""
    stock_code: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    returns: float
    exit_reason: str
    exit_scheme: str


class Backtester:
    """回测引擎"""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.data_loader = DataLoader(config.data_dir)
        self.factor_calculator = EntropyFactorCalculator(config.window)
        self.signal_generator = SignalGenerator(config)

        # 回测状态
        self.cash = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []

    def run(self, exit_scheme: str) -> Dict:
        """
        运行回测

        Args:
            exit_scheme: 卖出方案 ('A', 'B', 'C', 'D')

        Returns:
            回测结果统计
        """
        print(f"\n{'='*60}")
        print(f"开始回测 - 卖出方案 {exit_scheme}")
        print(f"{'='*60}")

        # 重置状态
        self.cash = self.config.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []

        # 获取股票池
        stock_codes = self.config.stock_pool
        if not stock_codes:
            all_stocks = self.data_loader.get_stock_codes()
            stock_codes = self.data_loader.filter_active_stocks(all_stocks, min_days=200)
            stock_codes = stock_codes[:self.config.max_stocks]

        print(f"股票池数量：{len(stock_codes)}")

        # 加载数据 (多只股票合并)
        print("加载数据...")
        all_data_list = []
        for code in stock_codes[:20]:  # 限制股票数量加速回测
            data = self.data_loader.load_stock_data(
                code, self.config.start_date, self.config.end_date
            )
            if len(data) > 0:
                all_data_list.append(data)
        all_data = pd.concat(all_data_list, ignore_index=True) if all_data_list else pd.DataFrame()

        if all_data.empty:
            print("无数据!")
            return {}

        print(f"数据量：{len(all_data)} 条")

        # 计算因子
        print("计算熵因子...")
        factors_df = self.factor_calculator.calculate_factors(all_data)

        if factors_df.empty:
            print("因子计算失败!")
            return {}

        print(f"因子数据量：{len(factors_df)} 条")

        # 获取唯一时间点
        trade_times = sorted(factors_df['trade_time'].unique())
        print(f"交易时间点：{len(trade_times)} 个")

        # 按时间步进回测
        print("开始回测...")
        pending_exits = []  # 待处理的退出信号

        for time_idx, current_time in enumerate(trade_times):
            # 获取当前时间点的数据
            current_factors = factors_df[factors_df['trade_time'] == current_time]

            # 检查现有持仓的退出条件
            positions_to_close = []
            for stock_code, position in list(self.positions.items()):
                # 获取该股票的当前因子
                stock_factors = current_factors[current_factors['stock_code'] == stock_code]

                # 获取从买入到当前的所有数据
                stock_data = all_data[
                    (all_data['stock_code'] == stock_code) &
                    (all_data['trade_time'] >= position.entry_time) &
                    (all_data['trade_time'] <= current_time)
                ]

                if len(stock_factors) == 0 or len(stock_data) == 0:
                    continue

                # 检查退出条件
                should_exit, exit_reason = self.signal_generator.generate_exit_signals(
                    position.signal,
                    stock_data,
                    stock_factors,
                    exit_scheme
                )

                if should_exit:
                    positions_to_close.append((stock_code, exit_reason))

            # 执行平仓
            for stock_code, exit_reason in positions_to_close:
                self._close_position(stock_code, current_time, exit_reason, exit_scheme)

            # 生成买入信号
            entry_signals = self._filter_entry_signals(
                current_factors,
                exclude_stocks=list(self.positions.keys())
            )

            # 执行买入
            for signal in entry_signals:
                self._open_position(signal)

            # 记录权益曲线
            total_value = self._calculate_total_value(current_factors)
            self.equity_curve.append({
                'trade_time': current_time,
                'cash': self.cash,
                'total_value': total_value,
                'positions': len(self.positions)
            })

            # 进度显示
            if time_idx % 100 == 0:
                print(f"  时间 {time_idx}/{len(trade_times)}, "
                      f"持仓 {len(self.positions)}, "
                      f"交易 {len(self.trades)}")

        # 平掉所有剩余持仓
        print("平掉剩余持仓...")
        for stock_code in list(self.positions.keys()):
            self._close_position(stock_code, trade_times[-1], "end_of_backtest", exit_scheme)

        # 生成统计报告
        stats = self._generate_statistics(exit_scheme)

        return stats

    def _filter_entry_signals(self,
                              factors_df: pd.DataFrame,
                              exclude_stocks: List[str]) -> List[Signal]:
        """过滤买入信号 (排除已持仓股票和资金不足)"""
        # 生成所有买入信号
        all_signals = self.signal_generator.generate_entry_signals(factors_df)

        # 过滤
        filtered = []
        for signal in all_signals:
            # 排除已持仓
            if signal.stock_code in exclude_stocks:
                continue

            # 检查持仓数量限制
            if len(self.positions) >= self.config.max_positions:
                break

            # 检查资金
            required_cash = signal.price * 100 * self.config.position_size
            if required_cash > self.cash * 0.95:  # 留 5% 缓冲
                continue

            filtered.append(signal)

        return filtered

    def _open_position(self, signal: Signal):
        """开仓"""
        # 检查价格有效性
        if signal.price <= 0 or np.isnan(signal.price):
            return

        # 计算买入数量 (100 股的整数倍)
        position_value = self.cash * self.config.position_size
        shares = int(position_value / signal.price / 100) * 100

        if shares < 100:
            return

        # 计算费用
        cost = shares * signal.price
        commission = max(5, cost * self.config.commission)  # 最低 5 元
        slippage_cost = cost * self.config.slippage
        total_cost = cost + commission + slippage_cost

        if total_cost > self.cash:
            return

        # 更新资金
        self.cash -= total_cost

        # 创建持仓
        position = Position(
            stock_code=signal.stock_code,
            entry_time=signal.trade_time,
            entry_price=signal.price,
            shares=shares,
            signal=signal
        )

        self.positions[signal.stock_code] = position

        # print(f"  买入 {signal.stock_code} @ {signal.price:.2f}, {shares}股")

    def _close_position(self, stock_code: str, exit_time: pd.Timestamp,
                        exit_reason: str, exit_scheme: str):
        """平仓"""
        if stock_code not in self.positions:
            return

        position = self.positions.pop(stock_code)

        # 获取退出价格 (使用当前时间的收盘价)
        exit_price = position.entry_price  # 默认，如果找不到就用买入价

        # 更新资金
        proceeds = position.shares * exit_price
        commission = max(5, proceeds * self.config.commission)
        stamp_tax = proceeds * self.config.stamp_tax  # 印花税
        slippage_cost = proceeds * self.config.slippage
        net_proceeds = proceeds - commission - stamp_tax - slippage_cost

        self.cash += net_proceeds

        # 计算盈亏
        pnl = net_proceeds - (position.shares * position.entry_price +
                              max(5, position.shares * position.entry_price * self.config.commission) +
                              position.shares * position.entry_price * self.config.slippage)
        returns = pnl / (position.shares * position.entry_price)

        # 记录交易
        trade = Trade(
            stock_code=stock_code,
            entry_time=position.entry_time,
            exit_time=exit_time,
            entry_price=position.entry_price,
            exit_price=exit_price,
            shares=position.shares,
            pnl=pnl,
            returns=returns,
            exit_reason=exit_reason,
            exit_scheme=exit_scheme
        )

        self.trades.append(trade)

        # print(f"  卖出 {stock_code} @ {exit_price:.2f}, 盈亏 {pnl:.2f} ({returns*100:.2f}%), 原因：{exit_reason}")

    def _calculate_total_value(self, factors_df: pd.DataFrame) -> float:
        """计算总资产"""
        total = self.cash

        for stock_code, position in self.positions.items():
            # 获取最新价格
            stock_factors = factors_df[factors_df['stock_code'] == stock_code]
            if len(stock_factors) > 0:
                # 这里简化处理，实际需要获取最新价格
                price = position.entry_price  # TODO: 获取实际价格
            else:
                price = position.entry_price

            total += position.shares * price

        return total

    def _generate_statistics(self, exit_scheme: str) -> Dict:
        """生成统计报告"""
        if not self.trades:
            return {}

        trades_df = pd.DataFrame([{
            'stock_code': t.stock_code,
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'shares': t.shares,
            'pnl': t.pnl,
            'returns': t.returns,
            'exit_reason': t.exit_reason,
            'exit_scheme': t.exit_scheme
        } for t in self.trades])

        # 基础统计
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # 盈亏统计
        total_pnl = trades_df['pnl'].sum()
        avg_pnl = trades_df['pnl'].mean()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].mean()) if losing_trades > 0 else 0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')

        # 收益率统计
        avg_return = trades_df['returns'].mean()
        total_return = (1 + trades_df['returns']).prod() - 1

        # 权益曲线统计
        equity_df = pd.DataFrame(self.equity_curve)
        if len(equity_df) > 0:
            peak = equity_df['total_value'].cummax()
            drawdown = (equity_df['total_value'] - peak) / peak
            max_drawdown = drawdown.min()
            final_value = equity_df['total_value'].iloc[-1]
        else:
            max_drawdown = 0
            final_value = self.cash

        # 按退出原因分组
        exit_reason_stats = trades_df.groupby('exit_reason').agg({
            'pnl': ['count', 'sum', 'mean'],
            'returns': 'mean'
        }).round(4)

        stats = {
            'exit_scheme': exit_scheme,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_return': avg_return,
            'total_return': total_return,
            'initial_capital': self.config.initial_capital,
            'final_value': final_value,
            'max_drawdown': max_drawdown,
            'exit_reason_stats': exit_reason_stats,
            'trades_df': trades_df,
            'equity_curve': equity_df
        }

        # 打印报告
        self._print_statistics(stats)

        return stats

    def _print_statistics(self, stats: Dict):
        """打印统计报告"""
        print(f"\n{'='*60}")
        print(f"回测结果 - 方案 {stats['exit_scheme']}")
        print(f"{'='*60}")
        print(f"总交易次数：{stats['total_trades']}")
        print(f"盈利次数：{stats['winning_trades']}")
        print(f"亏损次数：{stats['losing_trades']}")
        print(f"胜率：{stats['win_rate']*100:.2f}%")
        print(f"总盈亏：{stats['total_pnl']:.2f}")
        print(f"平均盈亏：{stats['avg_pnl']:.2f}")
        print(f"平均盈利：{stats['avg_win']:.2f}")
        print(f"平均亏损：{stats['avg_loss']:.2f}")
        print(f"盈亏比：{stats['profit_factor']:.2f}")
        print(f"平均收益率：{stats['avg_return']*100:.2f}%")
        print(f"总收益率：{stats['total_return']*100:.2f}%")
        print(f"初始资金：{stats['initial_capital']:.2f}")
        print(f"最终资金：{stats['final_value']:.2f}")
        print(f"最大回撤：{stats['max_drawdown']*100:.2f}%")
        print(f"{'='*60}")


# =============================================================================
# 主函数
# =============================================================================

def run_backtest():
    """运行完整回测 (4 种方案对比)"""
    # 配置
    config = BacktestConfig(
        max_stocks=20,  # 先测试 20 只股票
        start_date="2024-01-01",
        end_date="2024-12-31",
        window=60,
        max_positions=5
    )

    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)

    # 运行 4 种方案
    schemes = ['A', 'B', 'C', 'D']
    all_stats = {}

    for scheme in schemes:
        backtester = Backtester(config)
        stats = backtester.run(exit_scheme=scheme)
        all_stats[scheme] = stats

        # 保存结果
        if stats and 'trades_df' in stats:
            trades_file = os.path.join(config.output_dir, f"trades_scheme_{scheme}.csv")
            stats['trades_df'].to_csv(trades_file, index=False)
            print(f"交易记录已保存：{trades_file}")

        if stats and 'equity_curve' in stats and len(stats['equity_curve']) > 0:
            equity_file = os.path.join(config.output_dir, f"equity_scheme_{scheme}.csv")
            pd.DataFrame(stats['equity_curve']).to_csv(equity_file, index=False)
            print(f"权益曲线已保存：{equity_file}")

    # 对比报告
    print(f"\n{'='*60}")
    print("4 种方案对比")
    print(f"{'='*60}")
    print(f"{'方案':<10} {'交易次数':<12} {'胜率':<10} {'总收益':<12} {'最大回撤':<12}")
    print(f"{'-'*60}")

    for scheme in schemes:
        stats = all_stats.get(scheme, {})
        trades = stats.get('total_trades', 0)
        win_rate = f"{stats.get('win_rate', 0)*100:.1f}%"
        total_return = f"{stats.get('total_return', 0)*100:.1f}%"
        max_dd = f"{stats.get('max_drawdown', 0)*100:.1f}%"
        print(f"{scheme:<10} {trades:<12} {win_rate:<10} {total_return:<12} {max_dd:<12}")

    print(f"{'='*60}")

    return all_stats


if __name__ == "__main__":
    results = run_backtest()
