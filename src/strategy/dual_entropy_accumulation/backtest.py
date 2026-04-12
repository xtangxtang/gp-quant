"""
双熵共振策略 - 回测引擎 (v2 优化版)

回测逻辑：
1. 周频买入扫描：每周最后一个交易日，全市场扫描 buy 信号
2. 日频卖出检查：每个交易日，对持仓做卖出信号检测
3. 执行：信号日收盘后决策，T+1 开盘价成交
4. 约束：最多 max_positions 只持仓，等权分配
5. 费用：佣金 + 滑点

优化要点（v2）：
- 多进程并行计算日线熵和日内熵
- 日线熵结果缓存（5日内复用）
- 卖出扫描每3天做一次完整熵扫描，其余日只做止损止盈检查

用法::

    python -m src.strategy.dual_entropy_accumulation.backtest \\
        --start 2025-01-01 --end 2025-12-31 \\
        --daily_data_dir ../gp-data/tushare-daily-full \\
        --minute_data_dir ../gp-data/trade \\
        --workers 8
"""

import os
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta

from .config import Config, DailyEntropyConfig
from .daily_entropy import DailyEntropy, DailyEntropyResult
from .intraday_entropy import IntradayEntropyAnalyzer, IntradayEntropyResult
from .fusion_signal import FusionSignal, FusionResult
from .sell_signal import SellSignalEngine, SellResult
from .bifurcation import (
    BifurcationDetector, BifurcationConfig, BifurcationResult,
    TrendHoldEvaluator, TrendHoldConfig,
)


# ================================================================
#  回测配置
# ================================================================

@dataclass
class BacktestConfig:
    """回测参数"""

    # 时间范围
    start_date: str = '2025-01-01'
    end_date: str = '2025-12-31'

    # 资金
    initial_capital: float = 1_000_000.0

    # 仓位
    max_positions: int = 10
    min_buy_score: float = 0.55       # 只有 buy 信号才建仓
    min_sell_score: float = 0.50      # sell 信号触发平仓
    sell_warning_days: int = 3        # warning 信号连续 N 天触发平仓

    # 费用
    commission_rate: float = 0.0015   # 单边佣金 0.15%
    slippage_rate: float = 0.001      # 滑点 0.1%
    stamp_tax_rate: float = 0.0005    # 印花税（卖出单边）

    # 扫描频率
    buy_scan_frequency: str = 'weekly'  # weekly = 每周最后交易日做买入扫描
    sell_scan_frequency: str = 'daily'  # daily = 每天都检查卖出

    # 止损止盈
    stop_loss: float = -0.08          # 止损 -8%（收紧）
    take_profit: float = 0.30         # 止盈 +30%（仅对普通持仓）

    # 持仓上限天数（防止长期锁仓）
    max_hold_days: int = 90

    # 分岔预警模式
    enable_bifurcation: bool = True
    bifurcation_max_close: float = 500.0  # 分岔候选股价上限（放宽）
    bifurcation_extra_positions: int = 5   # 分岔可超出 max_positions 的额外仓位
    bifurcation_scan_interval: int = 1     # 分岔扫描间隔（交易日），1=每天

    # 并行进程数
    workers: int = 8

    # 卖出熵扫描间隔（天）
    sell_entropy_interval: int = 3


# ================================================================
#  交易记录
# ================================================================

@dataclass
class Trade:
    """单笔交易"""
    stock_code: str
    entry_date: str
    entry_price: float
    entry_score: float
    exit_date: str = ''
    exit_price: float = 0.0
    exit_reason: str = ''
    shares: int = 0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    hold_days: int = 0
    is_bifurcation: bool = False


@dataclass
class Position:
    """当前持仓"""
    stock_code: str
    entry_date: str
    entry_price: float
    shares: int
    cost: float              # 总成本（含费用）
    entry_score: float
    warning_count: int = 0   # 连续 warning 天数
    hold_days: int = 0
    is_bifurcation: bool = False  # 分岔预警买入标记
    peak_price: float = 0.0       # 持仓期间最高价（用于追踪止盈）


# ================================================================
#  Worker 函数（顶层，可序列化到子进程）
# ================================================================

def _worker_compute_daily_entropy(args):
    """子进程: 从预加载的日线数据计算截断到 cutoff 的日线熵。"""
    stock_code, close_arr, vol_arr, trade_dates, cutoff_int, config_dict = args

    try:
        idx = np.searchsorted(trade_dates, cutoff_int, side='right')
        if idx < 70:
            return None

        close_sub = close_arr[:idx]
        vol_sub = vol_arr[:idx]

        last_close = close_sub[-1]
        if last_close < config_dict.get('min_close', 3.0):
            return None
        if last_close > config_dict.get('max_close', 100.0):
            return None

        # 只取最近 250 天
        start = max(0, len(close_sub) - 250)
        close_sub = close_sub[start:]
        vol_sub = vol_sub[start:]

        if len(close_sub) < 70:
            return None

        # 构建 Series
        prices = pd.Series(close_sub, index=np.arange(len(close_sub)))
        volumes = pd.Series(vol_sub, index=np.arange(len(vol_sub)))

        calculator = DailyEntropy(DailyEntropyConfig())
        result = calculator.compute(prices, volumes)

        if result is None:
            return None

        return (stock_code, result)
    except Exception:
        return None


def _worker_compute_intraday_entropy(args):
    """子进程: 计算日内熵。"""
    stock_code, minute_dir, scan_date, intraday_config = args

    try:
        if not os.path.isdir(minute_dir):
            return None

        analyzer = IntradayEntropyAnalyzer(
            rolling_window=intraday_config['rolling_window'],
            perm_order=intraday_config['perm_order'],
            irrev_threshold_sigma=intraday_config['irrev_threshold_sigma'],
            ar_order=intraday_config['ar_order'],
        )

        csv_files = sorted(f for f in os.listdir(minute_dir) if f.endswith('.csv'))
        if not csv_files:
            return None

        target_file = f'{scan_date}.csv'
        files_before = [f for f in csv_files if f <= target_file]
        if not files_before:
            return None

        today_file = files_before[-1]
        if not today_file.startswith(scan_date):
            return None

        today_path = os.path.join(minute_dir, today_file)
        today_result = analyzer.analyze_day(stock_code, today_path)
        if today_result is None:
            return None

        lookback = intraday_config.get('lookback_days', 10)
        recent_files = files_before[max(0, len(files_before) - lookback - 1):-1]

        recent_results = []
        for f in recent_files:
            r = analyzer.analyze_day(stock_code, os.path.join(minute_dir, f))
            if r is not None:
                recent_results.append(r)

        return (stock_code, today_result, recent_results)
    except Exception:
        return None


def _worker_full_buy_eval(args):
    """
    子进程: 完整的买入评估（日线熵 + 日内熵 + 融合信号）。

    合并所有计算到一个 worker，减少进程间通信开销。
    """
    (stock_code, close_arr, vol_arr, trade_dates, cutoff_int,
     minute_base_dir, scan_date,
     daily_config_dict, intraday_config, fusion_config_dict) = args

    try:
        # ---- 日线熵 ----
        idx = np.searchsorted(trade_dates, cutoff_int, side='right')
        if idx < 70:
            return None

        close_sub = close_arr[:idx]
        vol_sub = vol_arr[:idx]

        last_close = close_sub[-1]
        if last_close < daily_config_dict.get('min_close', 3.0):
            return None
        if last_close > daily_config_dict.get('max_close', 100.0):
            return None

        start = max(0, len(close_sub) - 250)
        close_sub = close_sub[start:]
        vol_sub = vol_sub[start:]

        if len(close_sub) < 70:
            return None

        prices = pd.Series(close_sub, index=np.arange(len(close_sub)))
        volumes = pd.Series(vol_sub, index=np.arange(len(vol_sub)))

        calculator = DailyEntropy(DailyEntropyConfig())
        daily_result = calculator.compute(prices, volumes)
        if daily_result is None:
            return None

        # 检查是否符合压缩态门槛
        ep = daily_result.entropy_percentile
        pe = daily_result.perm_entropy_20
        ep_max = fusion_config_dict.get('daily_entropy_percentile_max', 0.40) + 0.10
        pe_max = fusion_config_dict.get('daily_perm_entropy_max', 0.85) + 0.05
        if not (np.isfinite(ep) and ep <= ep_max):
            return None
        if not (np.isfinite(pe) and pe <= pe_max):
            return None

        # ---- 日内熵 ----
        minute_dir = os.path.join(minute_base_dir, stock_code)
        if not os.path.isdir(minute_dir):
            return None

        analyzer = IntradayEntropyAnalyzer(
            rolling_window=intraday_config['rolling_window'],
            perm_order=intraday_config['perm_order'],
            irrev_threshold_sigma=intraday_config['irrev_threshold_sigma'],
            ar_order=intraday_config['ar_order'],
        )

        csv_files = sorted(f for f in os.listdir(minute_dir) if f.endswith('.csv'))
        if not csv_files:
            return None

        target_file = f'{scan_date}.csv'
        files_before = [f for f in csv_files if f <= target_file]
        if not files_before or not files_before[-1].startswith(scan_date):
            return None

        today_path = os.path.join(minute_dir, files_before[-1])
        today_intraday = analyzer.analyze_day(stock_code, today_path)
        if today_intraday is None:
            return None

        lookback = intraday_config.get('lookback_days', 10)
        recent_files = files_before[max(0, len(files_before) - lookback - 1):-1]
        recent_intraday = []
        for f in recent_files:
            r = analyzer.analyze_day(stock_code, os.path.join(minute_dir, f))
            if r is not None:
                recent_intraday.append(r)

        # ---- 融合信号 ----
        from .config import FusionSignalConfig
        fusion_engine = FusionSignal(FusionSignalConfig())
        fusion_result = fusion_engine.evaluate(
            stock_code, daily_result, today_intraday, recent_intraday,
        )

        if fusion_result.signal == 'buy':
            return fusion_result
        return None
    except Exception:
        return None


def _worker_bifurcation_eval(args):
    """
    子进程: 分岔预警评估 v2（仅需日线熵，不需要日内数据）。

    双层设计:
      第一层: 硬性门槛 (|λ|, 价格加速度, 流动性, 动量) → 快速排除 99%
      第二层: 熵综合打分 → 在剩下的 ~1% 候选中精选

    对 t, t-1, t-2 三个时点分别检测，取最高分 buy 信号。
    """
    (stock_code, close_arr, vol_arr, trade_dates, cutoff_int,
     max_close_bifurcation) = args

    try:
        base_idx = np.searchsorted(trade_dates, cutoff_int, side='right')
        if base_idx < 80:
            return None

        last_close = close_arr[base_idx - 1]
        if last_close < 3.0:
            return None
        if last_close > max_close_bifurcation:
            return None

        calculator = DailyEntropy(DailyEntropyConfig())
        detector = BifurcationDetector()
        best_br = None

        for offset in range(3):
            idx = base_idx - offset
            if idx < 80:
                continue

            cur_close = close_arr[idx - 1]

            # ---- 价格结构特征 (第一层预筛需要) ----
            returns = np.diff(np.log(close_arr[max(0, idx - 25):idx]))
            if len(returns) < 20:
                continue

            ret20 = returns[-20:]
            mom20 = float(np.sum(ret20))

            # 价格加速度: 近5日收益 - 前5日收益
            if len(ret20) >= 10:
                ret5_recent = np.sum(ret20[-5:])
                ret5_prev = np.sum(ret20[-10:-5])
                price_accel = ret5_recent - ret5_prev
            else:
                price_accel = 0.0

            # 流动性代理: close × 20日均量
            avg_vol_20 = np.mean(vol_arr[max(0, idx - 20):idx])
            liquidity = cur_close * avg_vol_20

            # ---- 日线熵计算 ----
            start = max(0, idx - 250)
            close_sub = close_arr[start:idx]
            vol_sub = vol_arr[start:idx]
            if len(close_sub) < 70:
                continue

            prices = pd.Series(close_sub, index=np.arange(len(close_sub)))
            volumes = pd.Series(vol_sub, index=np.arange(len(vol_sub)))
            result = calculator.compute(prices, volumes)
            if result is None:
                continue

            # ---- 第一层: 硬性门槛 ----
            if not detector.passes_prescreen(
                dominant_eigenvalue=result.dominant_eigenvalue,
                price_accel=price_accel,
                liquidity=liquidity,
                mom20=mom20,
            ):
                continue

            # ---- 5天前日线熵（计算速度） ----
            idx5 = max(0, idx - 5)
            if idx5 < 70:
                result5 = result
            else:
                start5 = max(0, idx5 - 250)
                close_sub5 = close_arr[start5:idx5]
                vol_sub5 = vol_arr[start5:idx5]
                if len(close_sub5) >= 70:
                    prices5 = pd.Series(close_sub5, index=np.arange(len(close_sub5)))
                    volumes5 = pd.Series(vol_sub5, index=np.arange(len(vol_sub5)))
                    r5 = calculator.compute(prices5, volumes5)
                    result5 = r5 if r5 is not None else result
                else:
                    result5 = result

            # ---- 第二层: 熵综合打分 ----
            d_int = int(trade_dates[idx - 1])
            trade_date_str = f'{d_int // 10000}-{(d_int % 10000) // 100:02d}-{d_int % 100:02d}'

            br = detector.evaluate(
                stock_code=stock_code,
                trade_date=trade_date_str,
                perm_entropy_20=result.perm_entropy_20,
                perm_entropy_60=result.perm_entropy_60,
                entropy_gap=result.entropy_gap,
                entropy_percentile=result.entropy_percentile,
                path_irreversibility=result.path_irreversibility,
                dominant_eigenvalue=result.dominant_eigenvalue,
                var_lift=result.var_lift,
                price_accel=price_accel,
                mom20=mom20,
                liquidity=liquidity,
                dominant_eigenvalue_prev5=result5.dominant_eigenvalue,
                path_irrev_prev5=result5.path_irreversibility,
            )

            if br.signal == 'buy':
                if best_br is None or br.total_score > best_br.total_score:
                    best_br = br

        return best_br
    except Exception:
        return None


# ================================================================
#  回测引擎 (v2 — 多进程 + 缓存)
# ================================================================

class Backtester:
    """
    双熵共振策略回测引擎（v2 优化版）。

    优化：
    - 买入扫描用 ProcessPoolExecutor 并行（每个 worker 做完整的 日线熵+日内熵+融合）
    - 卖出熵扫描每 N 天做一次，中间只做止损止盈
    - 日线数据预加载为 numpy 数组（比 DataFrame 快）
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        bt_config: Optional[BacktestConfig] = None,
    ):
        self.config = config or Config()
        self.bt = bt_config or BacktestConfig()

        # 状态
        self.cash: float = self.bt.initial_capital
        self.positions: Dict[str, Position] = {}  # code → Position
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[str, float]] = []  # [(date, total_equity)]

        # 缓存：DataFrame + numpy 数组
        self._daily_data_cache: Dict[str, pd.DataFrame] = {}  # code → full daily df
        self._daily_arrays: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        # code → (trade_dates, close_arr, vol_arr, open_arr)
        self._trading_dates: List[str] = []  # YYYY-MM-DD
        self._weekly_scan_dates: Set[str] = set()

        # 日线熵结果缓存: (stock_code, date_str) → DailyEntropyResult
        self._entropy_cache: Dict[Tuple[str, str], DailyEntropyResult] = {}

        # 分岔预警
        self._trend_hold = TrendHoldEvaluator() if self.bt.enable_bifurcation else None

    # ================================================================
    #  数据预加载
    # ================================================================

    def _load_all_daily_data(self):
        """把所有日线 CSV 预加载到内存。"""
        daily_dir = self.config.scanner.daily_data_dir
        print(f'[预加载] 读取日线数据: {daily_dir}')

        count = 0
        for fname in sorted(os.listdir(daily_dir)):
            if not fname.endswith('.csv'):
                continue
            code = fname.replace('.csv', '')
            if not (code.startswith('sh') or code.startswith('sz')):
                continue

            path = os.path.join(daily_dir, fname)
            try:
                df = pd.read_csv(path)
            except Exception:
                continue

            required = {'trade_date', 'close', 'vol', 'open'}
            if not required.issubset(df.columns):
                continue

            df['trade_date'] = pd.to_numeric(df['trade_date'], errors='coerce')
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['vol'] = pd.to_numeric(df['vol'], errors='coerce')
            df['open'] = pd.to_numeric(df['open'], errors='coerce')
            df = df.dropna(subset=['trade_date', 'close', 'vol', 'open'])
            df = df.sort_values('trade_date').drop_duplicates('trade_date', keep='last')
            df = df.reset_index(drop=True)

            self._daily_data_cache[code] = df
            # 提取 numpy 数组（避免 worker 里多次 .values）
            self._daily_arrays[code] = (
                df['trade_date'].values.astype(np.int64),
                df['close'].values.astype(np.float64),
                df['vol'].values.astype(np.float64),
                df['open'].values.astype(np.float64),
            )
            count += 1

        print(f'      加载 {count} 只股票日线数据')

    def _build_trading_calendar(self):
        """从某只活跃股票的日线数据提取交易日历。"""
        ref_code = 'sh600000'
        if ref_code not in self._daily_data_cache:
            ref_code = next(iter(self._daily_data_cache))

        df = self._daily_data_cache[ref_code]
        start_int = int(self.bt.start_date.replace('-', ''))
        end_int = int(self.bt.end_date.replace('-', ''))

        mask = (df['trade_date'] >= start_int) & (df['trade_date'] <= end_int)
        dates_int = df.loc[mask, 'trade_date'].astype(int).tolist()
        self._trading_dates = [
            f'{d // 10000}-{(d % 10000) // 100:02d}-{d % 100:02d}'
            for d in sorted(dates_int)
        ]

        self._weekly_scan_dates = set()
        week_groups: Dict[str, List[str]] = {}
        for d_str in self._trading_dates:
            dt = datetime.strptime(d_str, '%Y-%m-%d')
            week_key = dt.strftime('%Y-W%W')
            week_groups.setdefault(week_key, []).append(d_str)

        for week, dates in week_groups.items():
            self._weekly_scan_dates.add(dates[-1])

        print(f'[日历] {self._trading_dates[0]} ~ {self._trading_dates[-1]}, '
              f'共 {len(self._trading_dates)} 个交易日, '
              f'{len(self._weekly_scan_dates)} 个周扫描日')

    # ================================================================
    #  快速辅助
    # ================================================================

    def _get_next_open(self, stock_code: str, date_str: str) -> Optional[float]:
        arrays = self._daily_arrays.get(stock_code)
        if arrays is None:
            return None
        trade_dates, close_arr, vol_arr, open_arr = arrays
        cutoff = int(date_str.replace('-', ''))
        idx = np.searchsorted(trade_dates, cutoff, side='right')
        if idx >= len(trade_dates):
            return None
        return float(open_arr[idx])

    def _get_close(self, stock_code: str, date_str: str) -> Optional[float]:
        arrays = self._daily_arrays.get(stock_code)
        if arrays is None:
            return None
        trade_dates, close_arr, vol_arr, open_arr = arrays
        target = int(date_str.replace('-', ''))
        idx = np.searchsorted(trade_dates, target, side='left')
        if idx >= len(trade_dates) or trade_dates[idx] != target:
            return None
        return float(close_arr[idx])

    # ================================================================
    #  日线熵计算（带缓存）
    # ================================================================

    def _compute_daily_entropy_at_date(
        self,
        stock_code: str,
        date_str: str,
        max_close_override: Optional[float] = None,
    ) -> Optional[DailyEntropyResult]:
        """从缓存中截断数据计算日线熵，带结果缓存。"""
        cache_key = (stock_code, date_str)
        if cache_key in self._entropy_cache:
            return self._entropy_cache[cache_key]

        arrays = self._daily_arrays.get(stock_code)
        if arrays is None:
            return None

        trade_dates, close_arr, vol_arr, open_arr = arrays
        cutoff = int(date_str.replace('-', ''))
        idx = np.searchsorted(trade_dates, cutoff, side='right')
        if idx < 70:
            return None

        last_close = close_arr[idx - 1]
        if last_close < self.config.scanner.min_close:
            return None
        max_close = max_close_override if max_close_override is not None else self.config.scanner.max_close
        if last_close > max_close:
            return None

        start = max(0, idx - 250)
        close_sub = close_arr[start:idx]
        vol_sub = vol_arr[start:idx]

        if len(close_sub) < 70:
            return None

        prices = pd.Series(close_sub, index=np.arange(len(close_sub)))
        volumes = pd.Series(vol_sub, index=np.arange(len(vol_sub)))

        calculator = DailyEntropy(self.config.daily)
        result = calculator.compute(prices, volumes)

        if result is not None:
            self._entropy_cache[cache_key] = result
        return result

    # ================================================================
    #  日内熵（单进程版，用于卖出扫描）
    # ================================================================

    def _compute_intraday_entropy(
        self,
        stock_code: str,
        scan_date: str,
    ) -> Tuple[Optional[IntradayEntropyResult], List[IntradayEntropyResult]]:
        minute_dir = os.path.join(self.config.scanner.minute_data_dir, stock_code)
        if not os.path.isdir(minute_dir):
            return None, []

        analyzer = IntradayEntropyAnalyzer(
            rolling_window=self.config.intraday.rolling_window,
            perm_order=self.config.intraday.perm_order,
            irrev_threshold_sigma=self.config.intraday.irrev_threshold_sigma,
            ar_order=self.config.intraday.ar_order,
        )

        csv_files = sorted(f for f in os.listdir(minute_dir) if f.endswith('.csv'))
        if not csv_files:
            return None, []

        target_file = f'{scan_date}.csv'
        files_before = [f for f in csv_files if f <= target_file]
        if not files_before:
            return None, []

        today_file = files_before[-1]
        if not today_file.startswith(scan_date):
            return None, []

        today_path = os.path.join(minute_dir, today_file)
        today_result = analyzer.analyze_day(stock_code, today_path)
        if today_result is None:
            return None, []

        lookback = self.config.intraday.lookback_days
        recent_files = files_before[max(0, len(files_before) - lookback - 1):-1]

        recent_results = []
        for f in recent_files:
            r = analyzer.analyze_day(stock_code, os.path.join(minute_dir, f))
            if r is not None:
                recent_results.append(r)

        return today_result, recent_results

    # ================================================================
    #  买入扫描（多进程版）
    # ================================================================

    def _fast_prescreen(self, date_str: str) -> List[str]:
        """快速预筛：低波动 + 有分钟数据。"""
        cutoff = int(date_str.replace('-', ''))
        minute_dir = self.config.scanner.minute_data_dir
        candidates = []

        for code, arrays in self._daily_arrays.items():
            trade_dates, close_arr, vol_arr, open_arr = arrays
            idx = np.searchsorted(trade_dates, cutoff, side='right')
            if idx < 120:
                continue

            last_close = close_arr[idx - 1]
            if last_close < self.config.scanner.min_close:
                continue
            if last_close > self.config.scanner.max_close:
                continue

            minute_path = os.path.join(minute_dir, code, f'{date_str}.csv')
            if not os.path.isfile(minute_path):
                continue

            # 20日对数收益标准差在120日的百分位
            start = max(0, idx - 120)
            closes = close_arr[start:idx]
            if len(closes) < 60:
                continue

            returns = np.diff(np.log(closes))
            std_20 = np.std(returns[-20:])

            rolling_stds = np.array([
                np.std(returns[i-20:i]) for i in range(20, len(returns))
            ])
            if len(rolling_stds) == 0:
                continue

            percentile = np.sum(rolling_stds < std_20) / len(rolling_stds)
            if percentile <= 0.50:
                candidates.append((code, percentile))

        candidates.sort(key=lambda x: x[1])
        max_pre = self.config.scanner.max_stocks * 3
        return [c[0] for c in candidates[:max_pre]]

    def _scan_buy_at_date(self, date_str: str) -> List[FusionResult]:
        """多进程并行买入扫描。"""
        candidates = self._fast_prescreen(date_str)
        if not candidates:
            return []

        cutoff_int = int(date_str.replace('-', ''))
        minute_base_dir = self.config.scanner.minute_data_dir

        intraday_config = {
            'rolling_window': self.config.intraday.rolling_window,
            'perm_order': self.config.intraday.perm_order,
            'irrev_threshold_sigma': self.config.intraday.irrev_threshold_sigma,
            'ar_order': self.config.intraday.ar_order,
            'lookback_days': self.config.intraday.lookback_days,
        }
        daily_config_dict = {
            'min_close': self.config.scanner.min_close,
            'max_close': self.config.scanner.max_close,
        }
        fusion_config_dict = {
            'daily_entropy_percentile_max': self.config.fusion.daily_entropy_percentile_max,
            'daily_perm_entropy_max': self.config.fusion.daily_perm_entropy_max,
        }

        # 构建 worker 参数
        tasks = []
        for code in candidates[:self.config.scanner.max_stocks * 3]:
            arrays = self._daily_arrays.get(code)
            if arrays is None:
                continue
            trade_dates, close_arr, vol_arr, open_arr = arrays
            tasks.append((
                code, close_arr, vol_arr, trade_dates, cutoff_int,
                minute_base_dir, date_str,
                daily_config_dict, intraday_config, fusion_config_dict,
            ))

        results = []
        n_workers = min(self.bt.workers, len(tasks))

        if n_workers <= 1 or len(tasks) <= 5:
            # 少量任务就串行
            for task in tasks:
                r = _worker_full_buy_eval(task)
                if r is not None:
                    results.append(r)
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_worker_full_buy_eval, t): t[0] for t in tasks}
                for future in as_completed(futures):
                    try:
                        r = future.result()
                        if r is not None:
                            results.append(r)
                    except Exception:
                        pass

        results.sort(key=lambda r: r.total_score, reverse=True)
        return results

    # ================================================================
    #  分岔预警扫描（多进程版）
    # ================================================================

    def _bifurcation_prescreen(self, date_str: str) -> List[str]:
        """分岔预警预筛 v2：流动性 + 动量基础过滤。
        
        硬性门槛 (|λ|, 价格加速度) 在 worker 中检查,
        这里只做最基本的价格和流动性过滤, 大幅缩小候选池。
        """
        cutoff = int(date_str.replace('-', ''))
        candidates = []

        bif_cfg = BifurcationConfig()

        for code, arrays in self._daily_arrays.items():
            if code in self.positions:
                continue

            trade_dates, close_arr, vol_arr, open_arr = arrays
            idx = np.searchsorted(trade_dates, cutoff, side='right')
            if idx < 80:
                continue

            last_close = close_arr[idx - 1]
            if last_close < self.config.scanner.min_close:
                continue
            if last_close > self.bt.bifurcation_max_close:
                continue

            # 流动性门槛: close × 20日均量
            avg_vol_20 = np.mean(vol_arr[max(0, idx - 20):idx])
            liquidity = last_close * avg_vol_20
            if liquidity < bif_cfg.liquidity_gate:
                continue

            # 20日动量下限: 排除深度下跌
            if idx >= 21:
                mom20 = np.log(close_arr[idx - 1] / close_arr[idx - 21])
                if mom20 < bif_cfg.momentum_floor:
                    continue

            candidates.append(code)

        return candidates

    def _scan_bifurcation_at_date(self, date_str: str) -> List[BifurcationResult]:
        """多进程并行分岔预警扫描。"""
        if not self.bt.enable_bifurcation:
            return []

        candidates = self._bifurcation_prescreen(date_str)
        if not candidates:
            return []

        cutoff_int = int(date_str.replace('-', ''))

        tasks = []
        for code in candidates:
            arrays = self._daily_arrays.get(code)
            if arrays is None:
                continue
            trade_dates, close_arr, vol_arr, open_arr = arrays
            tasks.append((
                code, close_arr, vol_arr, trade_dates, cutoff_int,
                self.bt.bifurcation_max_close,
            ))

        results = []
        n_workers = min(self.bt.workers, len(tasks))

        if n_workers <= 1 or len(tasks) <= 5:
            for task in tasks:
                r = _worker_bifurcation_eval(task)
                if r is not None:
                    results.append(r)
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_worker_bifurcation_eval, t): t[0] for t in tasks}
                for future in as_completed(futures):
                    try:
                        r = future.result()
                        if r is not None:
                            results.append(r)
                    except Exception:
                        pass

        results.sort(key=lambda r: r.total_score, reverse=True)
        return results

    # ================================================================
    #  卖出扫描（单进程，只扫持仓）
    # ================================================================

    def _scan_sell_at_date(self, date_str: str) -> Dict[str, SellResult]:
        if not self.positions:
            return {}

        sell_engine = SellSignalEngine(self.config.sell)
        results = {}

        for code in list(self.positions.keys()):
            pos = self.positions[code]
            max_close_ovr = self.bt.bifurcation_max_close if pos.is_bifurcation else None
            daily_result = self._compute_daily_entropy_at_date(code, date_str, max_close_override=max_close_ovr)
            if daily_result is None:
                continue

            today_intraday, recent_intraday = self._compute_intraday_entropy(code, date_str)
            if today_intraday is None:
                continue

            sell_result = sell_engine.evaluate(
                code, daily_result, today_intraday, recent_intraday,
            )
            results[code] = sell_result

        return results

    # ================================================================
    #  交易执行
    # ================================================================

    def _execute_buy(self, code: str, date_str: str, score: float,
                     is_bifurcation: bool = False) -> bool:
        """
        以 T+1 开盘价买入。

        返回是否成功。
        """
        if code in self.positions:
            return False

        # 分岔和普通持仓使用独立额度
        if is_bifurcation:
            bif_count = sum(1 for p in self.positions.values() if p.is_bifurcation)
            if bif_count >= self.bt.bifurcation_extra_positions:
                return False
        else:
            non_bif_count = sum(1 for p in self.positions.values() if not p.is_bifurcation)
            if non_bif_count >= self.bt.max_positions:
                return False

        next_open = self._get_next_open(code, date_str)
        if next_open is None or next_open <= 0:
            return False

        # 等权分配: 基于总仓位上限
        total_cap = self.bt.max_positions + self.bt.bifurcation_extra_positions
        n_available = total_cap - len(self.positions)
        alloc = self.cash / max(1, n_available)
        alloc = min(alloc, self.cash)

        # 考虑佣金和滑点的买入价
        buy_price = next_open * (1 + self.bt.slippage_rate)
        shares = int(alloc / buy_price / 100) * 100  # 整手
        if shares <= 0:
            return False

        cost = shares * buy_price
        commission = cost * self.bt.commission_rate
        total_cost = cost + commission

        if total_cost > self.cash:
            shares -= 100
            if shares <= 0:
                return False
            cost = shares * buy_price
            commission = cost * self.bt.commission_rate
            total_cost = cost + commission

        self.cash -= total_cost

        # 执行日期 = T+1
        exec_date = self._next_trading_date(date_str)

        self.positions[code] = Position(
            stock_code=code,
            entry_date=exec_date or date_str,
            entry_price=buy_price,
            shares=shares,
            cost=total_cost,
            entry_score=score,
            is_bifurcation=is_bifurcation,
            peak_price=buy_price,
        )
        return True

    def _execute_sell(self, code: str, date_str: str, reason: str) -> Optional[Trade]:
        """
        以 T+1 开盘价卖出。

        返回 Trade 记录。
        """
        pos = self.positions.get(code)
        if pos is None:
            return None

        next_open = self._get_next_open(code, date_str)
        if next_open is None or next_open <= 0:
            return None

        sell_price = next_open * (1 - self.bt.slippage_rate)
        proceeds = pos.shares * sell_price
        commission = proceeds * self.bt.commission_rate
        stamp_tax = proceeds * self.bt.stamp_tax_rate
        net_proceeds = proceeds - commission - stamp_tax

        self.cash += net_proceeds

        pnl = net_proceeds - pos.cost
        pnl_pct = pnl / pos.cost if pos.cost > 0 else 0.0

        exec_date = self._next_trading_date(date_str)

        trade = Trade(
            stock_code=code,
            entry_date=pos.entry_date,
            entry_price=pos.entry_price,
            entry_score=pos.entry_score,
            exit_date=exec_date or date_str,
            exit_price=sell_price,
            exit_reason=reason,
            shares=pos.shares,
            pnl=pnl,
            pnl_pct=pnl_pct,
            hold_days=pos.hold_days,
            is_bifurcation=pos.is_bifurcation,
        )
        self.trades.append(trade)
        del self.positions[code]
        return trade

    def _next_trading_date(self, date_str: str) -> Optional[str]:
        """获取 date_str 之后的下一个交易日。"""
        try:
            idx = self._trading_dates.index(date_str)
            if idx + 1 < len(self._trading_dates):
                return self._trading_dates[idx + 1]
        except ValueError:
            pass
        return None

    # ================================================================
    #  持仓估值
    # ================================================================

    def _portfolio_value(self, date_str: str) -> float:
        """计算当前总资产 = 现金 + 持仓市值。"""
        total = self.cash
        for code, pos in self.positions.items():
            close = self._get_close(code, date_str)
            if close is not None:
                total += pos.shares * close
            else:
                total += pos.shares * pos.entry_price
        return total

    # ================================================================
    #  主回测循环
    # ================================================================

    def run(self) -> pd.DataFrame:
        """
        执行回测。

        返回
        ----
        equity_df : pd.DataFrame
            日度净值曲线
        """
        print('=' * 80)
        print('  双熵共振策略 回测引擎 v2 (多进程优化)')
        print('=' * 80)
        print(f'  期间: {self.bt.start_date} ~ {self.bt.end_date}')
        print(f'  初始资金: {self.bt.initial_capital:,.0f}')
        print(f'  最大持仓: {self.bt.max_positions} 只')
        print(f'  止损/止盈: {self.bt.stop_loss:.0%} / {self.bt.take_profit:.0%}')
        print(f'  并行进程: {self.bt.workers}')
        print(f'  卖出熵扫描间隔: 每{self.bt.sell_entropy_interval}天')
        print(f'  分岔预警: {"ON (extra=" + str(self.bt.bifurcation_extra_positions) + ", interval=" + str(self.bt.bifurcation_scan_interval) + "d)" if self.bt.enable_bifurcation else "OFF"}')
        print(f'  费用: 佣金 {self.bt.commission_rate:.2%}, 滑点 {self.bt.slippage_rate:.2%}, '
              f'印花税 {self.bt.stamp_tax_rate:.2%}')
        print()

        # 预加载
        self._load_all_daily_data()
        self._build_trading_calendar()

        if not self._trading_dates:
            print('错误: 无交易日数据!')
            return pd.DataFrame()

        total_days = len(self._trading_dates)
        buy_scan_count = 0
        sell_scan_count = 0
        total_buys = 0
        total_sells = 0

        import time
        t0 = time.time()

        for day_idx, date_str in enumerate(self._trading_dates):
            # ---- 更新持仓天数 ----
            for pos in self.positions.values():
                pos.hold_days += 1

            # ---- 止损止盈检查（每天，纯价格比较，快速） ----
            codes_to_sell = []
            for code, pos in self.positions.items():
                close = self._get_close(code, date_str)
                if close is None:
                    continue
                current_pnl_pct = (close - pos.entry_price) / pos.entry_price

                # 更新峰值价格（用于追踪止盈）
                if close > pos.peak_price:
                    pos.peak_price = close

                # 分岔持仓延长最大持仓天数
                effective_max_hold = self.bt.max_hold_days
                if pos.is_bifurcation and self._trend_hold:
                    effective_max_hold += self._trend_hold.config.max_hold_days_extension

                if current_pnl_pct <= self.bt.stop_loss:
                    codes_to_sell.append((code, f'止损 {current_pnl_pct:.1%}'))
                elif pos.is_bifurcation and self._trend_hold:
                    # 分岔持仓: 使用追踪止盈代替固定止盈
                    trailing_pct = self._trend_hold.config.trailing_stop_pct
                    if pos.peak_price > 0:
                        drawdown_from_peak = (close - pos.peak_price) / pos.peak_price
                        if current_pnl_pct > 0.15 and drawdown_from_peak <= -trailing_pct:
                            codes_to_sell.append((code,
                                f'追踪止盈(峰{pos.peak_price:.1f}→{close:.1f} '
                                f'回撤{drawdown_from_peak:.1%} 总盈{current_pnl_pct:.1%})'))
                    if pos.hold_days >= effective_max_hold:
                        codes_to_sell.append((code, f'持仓超限 {pos.hold_days}天'))
                elif (self.bt.take_profit > 0
                      and current_pnl_pct >= self.bt.take_profit):
                    codes_to_sell.append((code, f'止盈 {current_pnl_pct:.1%}'))
                elif pos.hold_days >= effective_max_hold:
                    codes_to_sell.append((code, f'持仓超限 {pos.hold_days}天'))

            for code, reason in codes_to_sell:
                trade = self._execute_sell(code, date_str, reason)
                if trade:
                    total_sells += 1

            # ---- 卖出信号扫描（每N天做一次完整熵扫描） ----
            do_entropy_sell = (
                self.positions
                and (day_idx % self.bt.sell_entropy_interval == 0
                     or date_str in self._weekly_scan_dates)
            )

            if do_entropy_sell:
                sell_results = self._scan_sell_at_date(date_str)
                sell_scan_count += 1

                for code, sr in sell_results.items():
                    if code not in self.positions:
                        continue
                    pos = self.positions[code]

                    if sr.signal == 'sell':
                        # 分岔持仓: 检查趋势是否仍存活，若在则抑制卖出
                        if pos.is_bifurcation and self._trend_hold:
                            cached = self._entropy_cache.get((code, date_str))
                            if cached is not None:
                                close = self._get_close(code, date_str)
                                pnl_pct = ((close - pos.entry_price) / pos.entry_price) if close else 0
                                if self._trend_hold.is_trend_alive(
                                    cached.dominant_eigenvalue,
                                    cached.path_irreversibility,
                                    cached.entropy_percentile,
                                    pnl_pct,
                                ):
                                    pos.warning_count = 0  # 趋势存活，重置
                                    continue  # 抑制卖出

                        trade = self._execute_sell(code, date_str, f'熵卖出({sr.sell_type}) score={sr.total_score:.3f}')
                        if trade:
                            total_sells += 1
                    elif sr.signal == 'warning':
                        pos.warning_count += 1
                        if pos.warning_count >= self.bt.sell_warning_days:
                            # 分岔持仓: 连续预警也受趋势持有抑制
                            if pos.is_bifurcation and self._trend_hold:
                                cached = self._entropy_cache.get((code, date_str))
                                if cached is not None:
                                    close = self._get_close(code, date_str)
                                    pnl_pct = ((close - pos.entry_price) / pos.entry_price) if close else 0
                                    if self._trend_hold.is_trend_alive(
                                        cached.dominant_eigenvalue,
                                        cached.path_irreversibility,
                                        cached.entropy_percentile,
                                        pnl_pct,
                                    ):
                                        pos.warning_count = 0  # 趋势存活，忽略预警
                                        continue

                            trade = self._execute_sell(code, date_str,
                                                       f'连续预警{pos.warning_count}天({sr.sell_type})')
                            if trade:
                                total_sells += 1
                    else:
                        pos.warning_count = 0

            # ---- 买入扫描（周频，使用普通持仓额度） ----
            is_buy_scan_day = date_str in self._weekly_scan_dates
            non_bif_count = sum(1 for p in self.positions.values() if not p.is_bifurcation)
            if is_buy_scan_day and non_bif_count < self.bt.max_positions:
                buy_results = self._scan_buy_at_date(date_str)
                buy_scan_count += 1

                buy_results = [r for r in buy_results if r.stock_code not in self.positions]

                n_slots = self.bt.max_positions - non_bif_count
                for r in buy_results[:n_slots]:
                    if self._execute_buy(r.stock_code, date_str, r.total_score):
                        total_buys += 1

            # ---- 分岔预警扫描（独立额度，不与普通持仓争抢） ----
            if self.bt.enable_bifurcation and (day_idx % self.bt.bifurcation_scan_interval == 0):
                bif_results = self._scan_bifurcation_at_date(date_str)
                bif_results = [r for r in bif_results if r.stock_code not in self.positions]

                bif_count = sum(1 for p in self.positions.values() if p.is_bifurcation)

                if bif_count < self.bt.bifurcation_extra_positions:
                    # 有空槽，直接买入
                    n_slots = self.bt.bifurcation_extra_positions - bif_count
                    for br in bif_results[:n_slots]:
                        if self._execute_buy(br.stock_code, date_str, br.total_score,
                                             is_bifurcation=True):
                            total_buys += 1
                elif bif_results:
                    # 槽满: 如果新候选得分远高于现有亏损持仓，替换之
                    bif_positions = [(c, p) for c, p in self.positions.items() if p.is_bifurcation]
                    for br in bif_results:
                        if br.total_score < 0.55:  # 新候选必须足够强
                            break
                        # 找最差的分岔持仓（亏损+低入场分）
                        worst_code, worst_pos = None, None
                        worst_metric = float('inf')
                        for c, p in bif_positions:
                            close = self._get_close(c, date_str)
                            if close is None:
                                continue
                            cur_pnl = (close - p.entry_price) / p.entry_price
                            if cur_pnl >= 0:
                                continue  # 不替换盈利持仓
                            metric = cur_pnl + p.entry_score * 0.1  # 越差越应替换
                            if metric < worst_metric:
                                worst_metric = metric
                                worst_code = c
                                worst_pos = p
                        if worst_code and br.total_score - worst_pos.entry_score > 0.10:
                            trade = self._execute_sell(worst_code, date_str,
                                f'分岔换仓(新{br.stock_code} sc={br.total_score:.3f})')
                            if trade:
                                total_sells += 1
                                bif_positions = [(c, p) for c, p in bif_positions if c != worst_code]
                                if self._execute_buy(br.stock_code, date_str, br.total_score,
                                                     is_bifurcation=True):
                                    total_buys += 1
                                    bif_positions.append((br.stock_code, self.positions[br.stock_code]))

            # ---- 记录净值 ----
            equity = self._portfolio_value(date_str)
            self.equity_curve.append((date_str, equity))

            # ---- 进度 ----
            if (day_idx + 1) % 10 == 0 or day_idx == total_days - 1:
                nav = equity / self.bt.initial_capital
                elapsed = time.time() - t0
                speed = (day_idx + 1) / elapsed if elapsed > 0 else 0
                eta = (total_days - day_idx - 1) / speed if speed > 0 else 0
                print(f'  [{day_idx + 1:>3d}/{total_days}] {date_str}  '
                      f'NAV={nav:.4f}  持仓={len(self.positions)}  '
                      f'现金={self.cash:,.0f}  '
                      f'买={total_buys} 卖={total_sells}  '
                      f'{elapsed:.0f}s (ETA {eta:.0f}s)')

        # ---- 收尾：平掉所有持仓 ----
        last_date = self._trading_dates[-1]
        for code in list(self.positions.keys()):
            trade = self._execute_sell(code, last_date, '回测结束强平')
            if trade:
                total_sells += 1

        final_equity = self.cash
        self.equity_curve.append((f'{last_date}_final', final_equity))

        total_time = time.time() - t0
        print()
        print(f'  总耗时: {total_time:.0f}s ({total_time/60:.1f}min)')
        print(f'  买入扫描: {buy_scan_count} 次, 卖出扫描: {sell_scan_count} 次')
        print(f'  买入: {total_buys} 笔, 卖出: {total_sells} 笔')
        print(f'  熵缓存命中: {len(self._entropy_cache)} 条目')
        print()

        # 构建净值 DataFrame
        equity_df = pd.DataFrame(self.equity_curve, columns=['date', 'equity'])
        equity_df = equity_df[~equity_df['date'].str.contains('_')]
        equity_df['nav'] = equity_df['equity'] / self.bt.initial_capital
        equity_df['daily_return'] = equity_df['nav'].pct_change()

        return equity_df

    # ================================================================
    #  绩效分析
    # ================================================================

    def report(self, equity_df: pd.DataFrame, output_dir: str = '') -> Dict:
        """生成回测绩效报告。"""
        if equity_df.empty:
            print('无回测数据。')
            return {}

        output_dir = output_dir or self.config.scanner.output_dir
        os.makedirs(output_dir, exist_ok=True)

        nav = equity_df['nav'].values
        returns = equity_df['daily_return'].dropna().values
        final_nav = nav[-1]
        total_return = final_nav - 1.0

        # 年化收益率
        n_days = len(nav)
        annual_return = (final_nav ** (252 / max(1, n_days))) - 1.0

        # 最大回撤
        peak = np.maximum.accumulate(nav)
        drawdown = (nav - peak) / peak
        max_drawdown = float(np.min(drawdown))

        # 夏普比率
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0.0

        # 胜率
        if self.trades:
            wins = [t for t in self.trades if t.pnl > 0]
            losses = [t for t in self.trades if t.pnl <= 0]
            win_rate = len(wins) / len(self.trades)
            avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
            avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0
            profit_factor = (
                abs(sum(t.pnl for t in wins)) / abs(sum(t.pnl for t in losses))
                if losses and sum(t.pnl for t in losses) != 0
                else float('inf')
            )
            avg_hold_days = np.mean([t.hold_days for t in self.trades])
        else:
            win_rate = avg_win = avg_loss = profit_factor = avg_hold_days = 0

        # Calmar 比率
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'calmar_ratio': calmar,
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'profit_factor': profit_factor,
            'avg_hold_days': avg_hold_days,
        }

        # ---- 打印 ----
        print('=' * 60)
        print('  回测绩效报告')
        print('=' * 60)
        print(f'  期间:        {self.bt.start_date} ~ {self.bt.end_date}')
        print(f'  初始资金:    {self.bt.initial_capital:>15,.0f}')
        print(f'  最终资金:    {self.cash:>15,.0f}')
        print(f'  总收益率:    {total_return:>14.2%}')
        print(f'  年化收益率:  {annual_return:>14.2%}')
        print(f'  最大回撤:    {max_drawdown:>14.2%}')
        print(f'  夏普比率:    {sharpe:>14.4f}')
        print(f'  Calmar比率:  {calmar:>14.4f}')
        print(f'  ---')
        print(f'  总交易笔数:  {len(self.trades):>14d}')
        print(f'  胜率:        {win_rate:>14.2%}')
        print(f'  平均盈利:    {avg_win:>14.2%}')
        print(f'  平均亏损:    {avg_loss:>14.2%}')
        print(f'  盈亏比:      {profit_factor:>14.2f}')
        print(f'  平均持仓天:  {avg_hold_days:>14.1f}')
        print('=' * 60)

        # ---- 交易明细按卖出原因统计 ----
        if self.trades:
            print('\n  卖出原因统计:')
            reason_groups: Dict[str, List[Trade]] = {}
            for t in self.trades:
                key = t.exit_reason.split('(')[0].strip() if '(' in t.exit_reason else t.exit_reason
                # 归类
                if '止损' in t.exit_reason:
                    key = '止损'
                elif '止盈' in t.exit_reason:
                    key = '止盈'
                elif '熵卖出' in t.exit_reason:
                    key = '熵卖出'
                elif '预警' in t.exit_reason:
                    key = '连续预警'
                elif '超限' in t.exit_reason:
                    key = '持仓超限'
                elif '强平' in t.exit_reason:
                    key = '结束强平'
                else:
                    key = '其他'
                reason_groups.setdefault(key, []).append(t)

            for reason, trades in sorted(reason_groups.items()):
                avg_pnl = np.mean([t.pnl_pct for t in trades])
                total_pnl = sum(t.pnl for t in trades)
                print(f'    {reason:<12s}: {len(trades):>3d}笔, '
                      f'平均收益={avg_pnl:>7.2%}, '
                      f'总盈亏={total_pnl:>12,.0f}')

        # ---- 月度收益 ----
        if len(equity_df) > 1:
            print('\n  月度收益:')
            equity_df_copy = equity_df.copy()
            equity_df_copy['month'] = equity_df_copy['date'].str[:7]
            monthly = equity_df_copy.groupby('month').agg(
                start_nav=('nav', 'first'),
                end_nav=('nav', 'last'),
            )
            monthly['return'] = monthly['end_nav'] / monthly['start_nav'] - 1.0
            for month, row in monthly.iterrows():
                bar = '█' * max(0, int(row['return'] * 100))
                neg_bar = '░' * max(0, int(-row['return'] * 100))
                print(f'    {month}: {row["return"]:>7.2%}  '
                      f'{"" if row["return"] >= 0 else "-"}{bar}{neg_bar}')

        # ---- 保存 ----
        equity_df.to_csv(os.path.join(output_dir, 'backtest_equity.csv'),
                         index=False, encoding='utf-8-sig')

        trades_rows = [
            {
                'stock_code': t.stock_code,
                'entry_date': t.entry_date,
                'entry_price': round(t.entry_price, 3),
                'entry_score': round(t.entry_score, 4),
                'exit_date': t.exit_date,
                'exit_price': round(t.exit_price, 3),
                'exit_reason': t.exit_reason,
                'shares': t.shares,
                'pnl': round(t.pnl, 2),
                'pnl_pct': round(t.pnl_pct, 4),
                'hold_days': t.hold_days,
                'is_bifurcation': t.is_bifurcation,
            }
            for t in self.trades
        ]
        pd.DataFrame(trades_rows).to_csv(
            os.path.join(output_dir, 'backtest_trades.csv'),
            index=False, encoding='utf-8-sig',
        )

        print(f'\n  结果已保存: {output_dir}/')
        print(f'    backtest_equity.csv')
        print(f'    backtest_trades.csv')

        return metrics


# ================================================================
#  CLI 入口
# ================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='双熵共振策略 - 回测')

    parser.add_argument('--start', type=str, default='2025-01-01',
                        help='回测开始日期')
    parser.add_argument('--end', type=str, default='2025-12-31',
                        help='回测结束日期')
    parser.add_argument('--capital', type=float, default=1_000_000,
                        help='初始资金')
    parser.add_argument('--max_positions', type=int, default=10,
                        help='最大持仓数')
    parser.add_argument('--max_stocks', type=int, default=200,
                        help='每次扫描评估的最大股票数')
    parser.add_argument('--workers', type=int, default=8,
                        help='并行进程数')
    parser.add_argument('--sell_entropy_interval', type=int, default=3,
                        help='卖出熵扫描间隔（天）')
    parser.add_argument('--daily_data_dir', type=str, default='',
                        help='日线数据目录')
    parser.add_argument('--minute_data_dir', type=str, default='',
                        help='分钟数据目录')
    parser.add_argument('--output_dir', type=str, default='',
                        help='输出目录')
    parser.add_argument('--stop_loss', type=float, default=-0.10,
                        help='止损比例')
    parser.add_argument('--take_profit', type=float, default=0.30,
                        help='止盈比例')
    parser.add_argument('--max_hold_days', type=int, default=60,
                        help='最大持仓天数')
    parser.add_argument('--enable_bifurcation', action='store_true',
                        help='启用分岔预警模式（捕捉爆发型大牛股）')
    parser.add_argument('--bifurcation_max_close', type=float, default=500.0,
                        help='分岔候选股价上限')
    parser.add_argument('--bifurcation_extra_positions', type=int, default=3,
                        help='分岔可超出max_positions的额外仓位数')
    parser.add_argument('--bifurcation_scan_interval', type=int, default=3,
                        help='分岔扫描间隔（交易日）')

    return parser.parse_args()


def main():
    args = parse_args()

    config = Config()
    if args.daily_data_dir:
        config.scanner.daily_data_dir = args.daily_data_dir
    if args.minute_data_dir:
        config.scanner.minute_data_dir = args.minute_data_dir
    if args.output_dir:
        config.scanner.output_dir = args.output_dir
    else:
        config.scanner.output_dir = '../../results/backtest'
    config.scanner.max_stocks = args.max_stocks

    bt_config = BacktestConfig(
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        max_positions=args.max_positions,
        stop_loss=args.stop_loss,
        workers=args.workers,
        sell_entropy_interval=args.sell_entropy_interval,
        take_profit=args.take_profit,
        max_hold_days=args.max_hold_days,
        enable_bifurcation=args.enable_bifurcation,
        bifurcation_max_close=args.bifurcation_max_close,
        bifurcation_extra_positions=args.bifurcation_extra_positions,
        bifurcation_scan_interval=args.bifurcation_scan_interval,
    )

    backtester = Backtester(config, bt_config)
    equity_df = backtester.run()
    metrics = backtester.report(equity_df)


if __name__ == '__main__':
    main()
