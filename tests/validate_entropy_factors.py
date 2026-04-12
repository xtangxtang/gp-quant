"""
熵因子有效性综合验证脚本

验证方法：
1. IC 分析（Information Coefficient）- 因子与未来收益的相关性
2. 因子分层回测（Quantile Backtest）- 按因子值分组的累积收益
3. 事件研究法（Event Study）- 信号后的平均收益路径

适用数据：
- 分钟级别：验证短期预测能力
- 日线级别：验证中长期预测能力
"""

import os
import glob
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from scipy import stats
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.tick_entropy import (
    rolling_path_irreversibility,
    rolling_permutation_entropy,
    rolling_turnover_entropy,
    market_state_classifier,
    dominant_eigenvalue_from_autocorr,
)


# =============================================================================
# 配置
# =============================================================================

@dataclass
class Config:
    # 数据路径
    tick_data_dir: str = "/nvme5/xtang/gp-workspace/gp-data/tick"
    minute_data_dir: str = "/nvme5/xtang/gp-workspace/gp-data/trade"
    output_dir: str = "/nvme5/xtang/gp-workspace/gp-quant/results/factor_validation"

    # 时间范围
    start_date: str = "2025-04-01"
    end_date: str = "2025-12-31"

    # 股票池配置
    max_stocks: int = 30

    # 分钟数据窗口
    minute_window: int = 240  # 240 分钟 = 1 天

    # IC 分析的前瞻期（分钟）
    ic_horizons: List[int] = None

    # 多线程配置
    n_workers: int = 8

    def __post_init__(self):
        if self.ic_horizons is None:
            self.ic_horizons = [30, 60, 120]  # 30/60/120 分钟


# =============================================================================
# 数据加载
# =============================================================================

def load_minute_data(
    data_dir: str,
    stock_code: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """加载单只股票的分钟数据"""
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


def load_daily_data(
    data_dir: str,
    stock_code: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """加载单只股票的日线数据（从分钟数据聚合）"""
    minute_data = load_minute_data(data_dir, stock_code, start_date, end_date)

    if minute_data.empty:
        return pd.DataFrame()

    # 按天聚合
    minute_data['date'] = minute_data['trade_time'].dt.date

    # 检查存在的列
    agg_cols = {
        'stock_code': 'first',
        'open': 'first',
        'close': 'last',
        'high': 'max',
        'low': 'min',
    }
    if 'volume' in minute_data.columns:
        agg_cols['volume'] = 'sum'
    if 'amount' in minute_data.columns:
        agg_cols['amount'] = 'sum'
    if 'turnover_rate' in minute_data.columns:
        agg_cols['turnover_rate'] = 'sum'

    daily = minute_data.groupby('date').agg(agg_cols).reset_index()
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.sort_values(['stock_code', 'date']).reset_index(drop=True)

    return daily


# =============================================================================
# 因子计算
# =============================================================================

def process_single_stock_minute(args: Tuple) -> Tuple[str, pd.DataFrame]:
    """处理单只股票的分钟数据（用于并行化）"""
    stock_code, data_dir, start_date, end_date, window = args

    try:
        data = load_minute_data(data_dir, stock_code, start_date, end_date)

        if len(data) < window + 50:
            return stock_code, pd.DataFrame()

        factors = calculate_factors_minute(data, window)

        if factors.empty:
            return stock_code, pd.DataFrame()

        return stock_code, factors
    except Exception as e:
        print(f"  {stock_code} 处理失败：{e}")
        return stock_code, pd.DataFrame()


def process_single_stock_daily(args: Tuple) -> Tuple[str, pd.DataFrame]:
    """处理单只股票的日线数据（用于并行化）"""
    stock_code, data_dir, start_date, end_date, window = args

    try:
        data = load_daily_data(data_dir, stock_code, start_date, end_date)

        if len(data) < 30:
            return stock_code, pd.DataFrame()

        factors = calculate_factors_daily(data, window)

        if factors.empty:
            return stock_code, pd.DataFrame()

        return stock_code, factors
    except Exception as e:
        print(f"  {stock_code} 处理失败：{e}")
        return stock_code, pd.DataFrame()

def calculate_factors_minute(
    data: pd.DataFrame,
    window: int = 240
) -> pd.DataFrame:
    """计算分钟级别的熵因子"""
    if len(data) < window + 50:
        return pd.DataFrame()

    data = data.sort_values('trade_time').reset_index(drop=True)
    data['log_ret'] = np.log(data['close']).diff()

    results = pd.DataFrame({
        'trade_time': data['trade_time'],
        'close': data['close'],
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
    results['turnover_entropy'] = rolling_turnover_entropy(
        data['turnover_rate'], window=window, n_bins=10
    )

    # 4. 主导特征值（临界预警）
    print("  计算主导特征值...")
    def calc_dominant_eig(x):
        if isinstance(x, np.ndarray):
            return dominant_eigenvalue_from_autocorr(x, order=2)
        return dominant_eigenvalue_from_autocorr(x.values, order=2)

    results['dominant_eigenvalue'] = data['log_ret'].rolling(
        window=window, min_periods=50
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


def calculate_factors_daily(
    data: pd.DataFrame,
    window: int = 20
) -> pd.DataFrame:
    """计算日级别的熵因子"""
    if len(data) < window + 10:
        return pd.DataFrame()

    data = data.sort_values('date').reset_index(drop=True)
    data['log_ret'] = np.log(data['close']).diff()

    results = pd.DataFrame({
        'date': data['date'],
        'close': data['close'],
        'stock_code': data['stock_code'].iloc[0]
    })

    # 1. 路径不可逆性熵（20 日窗口）
    print("  计算路径不可逆性熵（日线）...")
    results['path_irreversibility'] = rolling_path_irreversibility(
        data['log_ret'], window=window
    )

    # 2. 排列熵
    print("  计算排列熵（日线）...")
    results['permutation_entropy'] = rolling_permutation_entropy(
        data['log_ret'], window=window, order=3
    )

    # 3. 换手率熵
    print("  计算换手率熵（日线）...")
    results['turnover_entropy'] = rolling_turnover_entropy(
        data['turnover_rate'], window=window, n_bins=10
    )

    # 4. 主导特征值
    print("  计算主导特征值（日线）...")
    def calc_dominant_eig_daily(x):
        if isinstance(x, np.ndarray):
            return dominant_eigenvalue_from_autocorr(x, order=2)
        return dominant_eigenvalue_from_autocorr(x.values, order=2)

    results['dominant_eigenvalue'] = data['log_ret'].rolling(
        window=window, min_periods=10
    ).apply(calc_dominant_eig_daily, raw=False)

    # 5. 市场状态
    print("  计算市场状态（日线）...")
    results['market_state'] = results.apply(
        lambda row: market_state_classifier(
            row['path_irreversibility'],
            row['permutation_entropy'],
            turnover_entropy=row.get('turnover_entropy', np.nan)
        ),
        axis=1
    )

    return results


# =============================================================================
# 1. IC 分析（Information Coefficient）
# =============================================================================

def calculate_future_returns(
    prices: pd.Series,
    horizons: List[int] = [30, 60, 120]
) -> pd.DataFrame:
    """计算未来 N 期的收益率"""
    results = pd.DataFrame(index=prices.index)

    for h in horizons:
        # 未来 h 期的收益率
        future_ret = prices.shift(-h) / prices - 1
        results[f'return_{h}'] = future_ret

    return results


def ic_analysis_single_stock(
    factors: pd.DataFrame,
    time_col: str,
    horizons: List[int] = [30, 60, 120]
) -> Dict:
    """对单只股票进行 IC 分析"""
    factors = factors.copy()
    factors = factors.dropna(subset=['path_irreversibility', 'permutation_entropy', 'turnover_entropy'])

    if len(factors) < 100:
        return None

    # 计算未来收益率
    future_rets = calculate_future_returns(factors['close'], horizons)
    factors = pd.concat([factors, future_rets], axis=1)
    factors = factors.dropna()

    if len(factors) < 50:
        return None

    # 计算各因子与各期收益的 IC
    factor_cols = ['path_irreversibility', 'permutation_entropy', 'turnover_entropy', 'dominant_eigenvalue']
    ic_results = {}

    for factor in factor_cols:
        if factor not in factors.columns:
            continue
        ic_results[factor] = {}
        for h in horizons:
            ret_col = f'return_{h}'
            if ret_col not in factors.columns:
                continue

            # Pearson IC
            ic, pvalue = stats.pearsonr(factors[factor], factors[ret_col])

            # Spearman IC (rank correlation)
            sic, spvalue = stats.spearmanr(factors[factor], factors[ret_col])

            ic_results[factor][h] = {
                'pearson_ic': ic,
                'pearson_pvalue': pvalue,
                'spearman_ic': sic,
                'spearman_pvalue': spvalue,
                'n_obs': len(factors)
            }

    return ic_results


def ic_analysis_panel(
    all_factors: List[pd.DataFrame],
    time_col: str,
    horizons: List[int] = [30, 60, 120]
) -> pd.DataFrame:
    """面板数据的 IC 分析"""
    all_ics = {'pearson': [], 'spearman': []}

    for factors in all_factors:
        ic_result = ic_analysis_single_stock(factors, time_col, horizons)
        if ic_result is None:
            continue

        for factor, ics in ic_result.items():
            for h, metrics in ics.items():
                all_ics['pearson'].append({
                    'factor': factor,
                    'horizon': h,
                    'ic': metrics['pearson_ic'],
                    'pvalue': metrics['pearson_pvalue'],
                    'type': 'pearson'
                })
                all_ics['spearman'].append({
                    'factor': factor,
                    'horizon': h,
                    'ic': metrics['spearman_ic'],
                    'pvalue': metrics['spearman_pvalue'],
                    'type': 'spearman'
                })

    # 汇总统计
    pearson_df = pd.DataFrame(all_ics['pearson'])
    spearman_df = pd.DataFrame(all_ics['spearman'])

    summary = pearson_df.groupby(['factor', 'horizon']).agg({
        'ic': ['mean', 'std', 'count'],
        'pvalue': 'mean'
    }).round(4)
    summary.columns = ['mean_pearson_ic', 'std_pearson_ic', 'n_stocks', 'mean_pvalue']
    summary['ic_ir'] = summary['mean_pearson_ic'] / summary['std_pearson_ic']  # IC IR

    return summary, pearson_df, spearman_df


# =============================================================================
# 2. 因子分层回测（Quantile Backtest）
# =============================================================================

def quantile_backtest(
    all_factors: List[pd.DataFrame],
    time_col: str,
    n_quantiles: int = 5,
    holding_period: int = 60
) -> pd.DataFrame:
    """
    因子分层回测

    将股票按因子值分成 N 组，计算每组的累积收益
    """
    # 合并所有数据
    combined = pd.concat(all_factors, ignore_index=True)
    combined = combined.dropna(subset=['path_irreversibility', 'permutation_entropy', 'turnover_entropy', 'close'])

    # 计算未来收益
    combined = combined.sort_values([time_col])
    combined['future_return'] = combined.groupby('stock_code')['close'].shift(-holding_period) / combined['close'] - 1
    combined = combined.dropna(subset=['future_return'])

    results = {}

    for factor in ['path_irreversibility', 'permutation_entropy', 'turnover_entropy']:
        if factor not in combined.columns:
            continue

        # 按因子值分位数分组
        combined['quantile'] = pd.qcut(
            combined[factor].rank(method='first'),
            q=n_quantiles,
            labels=[f'Q{i+1}' for i in range(n_quantiles)]
        )

        # 计算每组的平均未来收益
        quantile_returns = combined.groupby('quantile')['future_return'].agg(['mean', 'std', 'count'])
        quantile_returns.columns = [f'{factor}_mean_ret', f'{factor}_std_ret', f'{factor}_count']

        results[factor] = quantile_returns

    return pd.concat(results.values(), axis=1) if results else None


def quantile_cumulative_returns(
    all_factors: List[pd.DataFrame],
    time_col: str,
    factor: str = 'path_irreversibility',
    n_quantiles: int = 5,
    holding_period: int = 60
) -> pd.DataFrame:
    """计算各分位组的累积收益曲线"""
    combined = pd.concat(all_factors, ignore_index=True)
    combined = combined.dropna(subset=[factor, 'close'])
    combined = combined.sort_values(time_col)

    # 每天调仓
    combined['date'] = pd.to_datetime(combined[time_col]).dt.date

    # 按日期和因子分位数分组
    combined['quantile'] = pd.qcut(
        combined[factor].rank(method='first'),
        q=n_quantiles,
        labels=[f'Q{i+1}' for i in range(n_quantiles)]
    )

    # 计算每组的日收益
    daily_ret = combined.groupby(['date', 'quantile']).apply(
        lambda x: x['close'].iloc[-1] / x['close'].iloc[0] - 1 if len(x) > 0 else 0
    ).reset_index()
    daily_ret.columns = ['date', 'quantile', 'daily_return']

    # 累积收益
    daily_ret['cum_return'] = daily_ret.groupby('quantile')['daily_return'].transform(
        lambda x: (1 + x).cumprod() - 1
    )

    return daily_ret


# =============================================================================
# 3. 事件研究法（Event Study）
# =============================================================================

def event_study_analysis(
    all_factors: List[pd.DataFrame],
    time_col: str,
    event_window: Tuple[int, int] = (-10, 50),
    buy_threshold: Dict = None
) -> pd.DataFrame:
    """
    事件研究法 - 分析买入信号后的平均收益路径

    event_window: (事前窗口，事后窗口)，单位：分钟/天
    buy_threshold: 买入条件字典
    """
    if buy_threshold is None:
        buy_threshold = {
            'path_irreversibility': ('gt', 0.01),
            'permutation_entropy': ('lt', 0.97),
            'market_state': ('in', ['ordered', 'weak_chaos'])
        }

    combined = pd.concat(all_factors, ignore_index=True)
    combined = combined.dropna(subset=['path_irreversibility', 'permutation_entropy', 'close'])
    combined = combined.sort_values([time_col])

    # 识别买入信号
    def is_buy_signal(row):
        for factor, (op, thresh) in buy_threshold.items():
            if factor not in row.index:
                continue
            val = row[factor]
            if op == 'gt' and not (val > thresh):
                return False
            elif op == 'lt' and not (val < thresh):
                return False
            elif op == 'in' and val not in thresh:
                return False
        return True

    combined['is_signal'] = combined.apply(is_buy_signal, axis=1)

    # 提取信号事件
    events = combined[combined['is_signal']].copy()

    if len(events) == 0:
        print("  未找到符合条件的买入信号")
        return None

    print(f"  找到 {len(events)} 个买入信号")

    # 计算每个信号后的收益路径
    post_returns = []
    pre_min, post_max = event_window

    for idx, event in events.iterrows():
        stock = event.get('stock_code')
        if stock is None:
            # 如果没有 stock_code，假设是单只股票数据
            stock_data = combined
        else:
            stock_data = combined[combined['stock_code'] == stock]

        event_time = event[time_col]
        stock_data = stock_data.sort_values(time_col)

        # 找到事件时间点
        event_idx = stock_data[stock_data[time_col] == event_time]
        if len(event_idx) == 0:
            continue
        event_pos = event_idx.index[0]

        # 获取事件前后的索引范围
        all_idx = stock_data.index.tolist()
        try:
            pos_in_list = all_idx.index(event_pos)
        except ValueError:
            continue

        start_pos = max(0, pos_in_list + pre_min)
        end_pos = min(len(all_idx), pos_in_list + post_max + 1)

        if start_pos >= end_pos:
            continue

        # 提取价格路径
        subset = stock_data.iloc[start_pos:end_pos].copy()
        subset['relative_time'] = range(-pos_in_list + start_pos, -pos_in_list + end_pos)
        base_price = subset['close'].iloc[pos_in_list - start_pos]
        subset['cum_return'] = subset['close'] / base_price - 1

        post_returns.append(subset[['relative_time', 'cum_return', time_col]])

    if not post_returns:
        return None

    # 合并所有事件的路径
    all_paths = pd.concat(post_returns, ignore_index=True)

    # 按相对时间计算平均收益路径
    avg_path = all_paths.groupby('relative_time')['cum_return'].agg(['mean', 'std', 'count']).reset_index()
    avg_path.columns = ['relative_time', 'mean_return', 'std_return', 'n_events']
    avg_path['mean_return'] = avg_path['mean_return'].fillna(0)

    return avg_path


# =============================================================================
# 主验证流程
# =============================================================================

def run_validation_minute(config: Config):
    """运行分钟级别的因子验证"""
    print("=" * 80)
    print("熵因子有效性验证 - 分钟级别")
    print("=" * 80)

    os.makedirs(config.output_dir, exist_ok=True)

    # 获取股票池
    stock_dirs = [d for d in os.listdir(config.minute_data_dir)
                  if os.path.isdir(os.path.join(config.minute_data_dir, d))]
    stock_dirs = sorted(stock_dirs)[:config.max_stocks]

    print(f"\n股票池：{len(stock_dirs)} 只股票")
    print(f"时间范围：{config.start_date} - {config.end_date}")
    print(f"窗口大小：{config.minute_window} 分钟")
    print(f"并行 worker 数：{config.n_workers}")

    all_factors = []

    # 使用多进程并行处理
    tasks = [
        (stock_code, config.minute_data_dir, config.start_date, config.end_date, config.minute_window)
        for stock_code in stock_dirs
    ]

    print(f"\n开始并行处理 {len(tasks)} 只股票...")

    with ProcessPoolExecutor(max_workers=config.n_workers) as executor:
        futures = {executor.submit(process_single_stock_minute, task): task[0] for task in tasks}

        for future in tqdm(as_completed(futures), total=len(futures), desc="处理股票"):
            stock_code, factors = future.result()

            if factors.empty:
                print(f"  {stock_code}: 跳过")
                continue

            all_factors.append(factors)
            print(f"  {stock_code}: {len(factors)} 条记录")

    if not all_factors:
        print("\n没有有效数据，验证失败")
        return

    print(f"\n{'=' * 60}")
    print("1. IC 分析结果")
    print(f"{'=' * 60}")

    ic_summary, ic_pearson, ic_spearman = ic_analysis_panel(
        all_factors, 'trade_time', config.ic_horizons
    )

    print("\nIC 汇总（Pearson）:")
    print(ic_summary.to_string())

    # 保存 IC 结果
    ic_summary.to_csv(os.path.join(config.output_dir, "ic_summary_minute.csv"))
    ic_pearson.to_csv(os.path.join(config.output_dir, "ic_pearson_minute.csv"), index=False)
    ic_spearman.to_csv(os.path.join(config.output_dir, "ic_spearman_minute.csv"), index=False)

    print(f"\n{'=' * 60}")
    print("2. 因子分层回测结果")
    print(f"{'=' * 60}")

    quantile_results = quantile_backtest(
        all_factors, 'trade_time', n_quantiles=5, holding_period=60
    )

    if quantile_results is not None:
        print("\n各分位组的平均未来收益:")
        print(quantile_results.to_string())
        quantile_results.to_csv(os.path.join(config.output_dir, "quantile_minute.csv"))

    # 计算累积收益曲线
    for factor in ['path_irreversibility', 'permutation_entropy', 'turnover_entropy']:
        cum_ret = quantile_cumulative_returns(
            all_factors, 'trade_time', factor=factor, n_quantiles=5, holding_period=60
        )
        if cum_ret is not None and len(cum_ret) > 0:
            cum_ret.to_csv(os.path.join(config.output_dir, f"cumulative_{factor}_minute.csv"), index=False)
            print(f"\n{factor} 各分位组累积收益已保存")

    print(f"\n{'=' * 60}")
    print("3. 事件研究结果")
    print(f"{'=' * 60}")

    event_result = event_study_analysis(
        all_factors, 'trade_time',
        event_window=(-10, 50),
        buy_threshold={
            'path_irreversibility': ('gt', 0.01),
            'permutation_entropy': ('lt', 0.97),
            'market_state': ('in', ['ordered', 'weak_chaos'])
        }
    )

    if event_result is not None:
        print("\n买入信号后的平均收益路径:")
        print(event_result.to_string())
        event_result.to_csv(os.path.join(config.output_dir, "event_study_minute.csv"), index=False)

    print(f"\n{'=' * 60}")
    print("验证完成！结果已保存到:", config.output_dir)
    print(f"{'=' * 60}")

    return all_factors


def run_validation_daily(config: Config):
    """运行日线级别的因子验证"""
    print("=" * 80)
    print("熵因子有效性验证 - 日线级别")
    print("=" * 80)

    daily_output_dir = os.path.join(config.output_dir, "daily")
    os.makedirs(daily_output_dir, exist_ok=True)

    # 获取股票池
    stock_dirs = [d for d in os.listdir(config.minute_data_dir)
                  if os.path.isdir(os.path.join(config.minute_data_dir, d))]
    stock_dirs = sorted(stock_dirs)[:config.max_stocks]

    print(f"\n股票池：{len(stock_dirs)} 只股票")
    print(f"时间范围：{config.start_date} - {config.end_date}")
    print(f"并行 worker 数：{config.n_workers}")

    all_factors = []

    # 使用多进程并行处理
    tasks = [
        (stock_code, config.minute_data_dir, config.start_date, config.end_date, 20)
        for stock_code in stock_dirs
    ]

    print(f"\n开始并行处理 {len(tasks)} 只股票...")

    with ProcessPoolExecutor(max_workers=config.n_workers) as executor:
        futures = {executor.submit(process_single_stock_daily, task): task[0] for task in tasks}

        for future in tqdm(as_completed(futures), total=len(futures), desc="处理股票"):
            stock_code, factors = future.result()

            if factors.empty:
                print(f"  {stock_code}: 跳过")
                continue

            all_factors.append(factors)
            print(f"  {stock_code}: {len(factors)} 条记录")

    if not all_factors:
        print("\n没有有效数据，验证失败")
        return

    print(f"\n{'=' * 60}")
    print("1. IC 分析结果（日线）")
    print(f"{'=' * 60}")

    # 日线 IC 分析（前瞻 1/3/5 天）
    ic_summary, ic_pearson, ic_spearman = ic_analysis_panel(
        all_factors, 'date', horizons=[1, 3, 5]
    )

    print("\nIC 汇总（Pearson）:")
    print(ic_summary.to_string())

    ic_summary.to_csv(os.path.join(daily_output_dir, "ic_summary_daily.csv"))
    ic_pearson.to_csv(os.path.join(daily_output_dir, "ic_pearson_daily.csv"), index=False)
    ic_spearman.to_csv(os.path.join(daily_output_dir, "ic_spearman_daily.csv"), index=False)

    print(f"\n{'=' * 60}")
    print("2. 因子分层回测结果（日线）")
    print(f"{'=' * 60}")

    quantile_results = quantile_backtest(
        all_factors, 'date', n_quantiles=5, holding_period=3
    )

    if quantile_results is not None:
        print("\n各分位组的平均未来收益（3 日）:")
        print(quantile_results.to_string())
        quantile_results.to_csv(os.path.join(daily_output_dir, "quantile_daily.csv"))

    # 累积收益曲线
    for factor in ['path_irreversibility', 'permutation_entropy', 'turnover_entropy']:
        cum_ret = quantile_cumulative_returns(
            all_factors, 'date', factor=factor, n_quantiles=5, holding_period=3
        )
        if cum_ret is not None and len(cum_ret) > 0:
            cum_ret.to_csv(os.path.join(daily_output_dir, f"cumulative_{factor}_daily.csv"), index=False)
            print(f"\n{factor} 各分位组累积收益已保存")

    print(f"\n{'=' * 60}")
    print("3. 事件研究结果（日线）")
    print(f"{'=' * 60}")

    event_result = event_study_analysis(
        all_factors, 'date',
        event_window=(-5, 10),
        buy_threshold={
            'path_irreversibility': ('gt', 0.01),
            'permutation_entropy': ('lt', 0.97),
            'market_state': ('in', ['ordered', 'weak_chaos'])
        }
    )

    if event_result is not None:
        print("\n买入信号后的平均收益路径:")
        print(event_result.to_string())
        event_result.to_csv(os.path.join(daily_output_dir, "event_study_daily.csv"), index=False)

    print(f"\n{'=' * 60}")
    print("日线验证完成！结果已保存到:", daily_output_dir)
    print(f"{'=' * 60}")

    return all_factors


def main():
    config = Config()

    # 运行分钟级别验证
    minute_factors = run_validation_minute(config)

    # 运行日线级别验证
    daily_factors = run_validation_daily(config)

    print("\n" + "=" * 80)
    print("全部验证完成！")
    print("=" * 80)
    print(f"\n结果目录：{config.output_dir}")
    print("  - ic_summary_*.csv: IC 分析汇总")
    print("  - quantile_*.csv: 因子分层回测结果")
    print("  - cumulative_*.csv: 各分位组累积收益曲线")
    print("  - event_study_*.csv: 事件研究结果")


if __name__ == "__main__":
    main()
