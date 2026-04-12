"""
双熵共振策略 - 全市场扫描器

流程：
1. 加载日线数据 → 计算日线熵 → 预筛出日线压缩态股票
2. 对预筛股票加载日内分钟数据 → 计算日内熵
3. 双熵融合评分 → 输出 buy / watch / skip
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

from .config import Config
from .daily_entropy import DailyEntropy, DailyEntropyResult
from .intraday_entropy import IntradayEntropyAnalyzer, IntradayEntropyResult
from .fusion_signal import FusionSignal, FusionResult
from .sell_signal import SellSignalEngine, SellResult


# ================================================================
#  Worker 函数（ProcessPoolExecutor 需要顶层可 pickle 的函数）
# ================================================================

def _daily_entropy_worker(args) -> Optional[Tuple[str, dict, np.ndarray, np.ndarray]]:
    """
    并行计算单只股票的日线熵。

    返回 (stock_code, daily_result_dict, prices_array, dates_array) 或 None。
    """
    stock_code, csv_path, config_dict = args

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    required = {'trade_date', 'close', 'vol'}
    if not required.issubset(df.columns):
        return None

    df['trade_date'] = pd.to_numeric(df['trade_date'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['vol'] = pd.to_numeric(df['vol'], errors='coerce')
    df = df.dropna(subset=['trade_date', 'close', 'vol'])
    df = df.sort_values('trade_date').drop_duplicates('trade_date', keep='last')

    if len(df) < config_dict.get('min_data_days', 60):
        return None

    # 价格过滤
    last_close = df['close'].iloc[-1]
    if last_close < config_dict.get('min_close', 3.0):
        return None
    if last_close > config_dict.get('max_close', 100.0):
        return None

    dates = pd.to_datetime(df['trade_date'].astype(int).astype(str), format='%Y%m%d', errors='coerce')
    df = df[dates.notna()]
    dates = dates[dates.notna()]

    if len(df) < config_dict.get('min_data_days', 60):
        return None

    prices = pd.Series(df['close'].values, index=dates.values)
    volumes = pd.Series(df['vol'].values, index=dates.values)

    from .daily_entropy import DailyEntropy, DailyEntropyResult
    from .config import DailyEntropyConfig

    de_config = DailyEntropyConfig(**{k: v for k, v in config_dict.get('daily_config', {}).items()})
    calculator = DailyEntropy(de_config)
    result = calculator.compute(prices, volumes)

    if result is None:
        return None

    # 返回可序列化的结果
    result_dict = {
        'perm_entropy_20': result.perm_entropy_20,
        'perm_entropy_60': result.perm_entropy_60,
        'entropy_gap': result.entropy_gap,
        'entropy_percentile': result.entropy_percentile,
        'path_irreversibility': result.path_irreversibility,
        'dominant_eigenvalue': result.dominant_eigenvalue,
        'var_lift': result.var_lift,
        'is_compressed': result.is_compressed,
        'compression_days': result.compression_days,
        'entropy_velocity_5': result.entropy_velocity_5,
        'path_irrev_velocity_5': result.path_irrev_velocity_5,
        'entropy_percentile_prev5': result.entropy_percentile_prev5,
    }

    return (
        stock_code,
        result_dict,
        prices.values[-120:],  # 最近 120 天价格
        dates.values[-1],      # 最新日期
    )


class Scanner:
    """
    全市场扫描器。

    典型用法::

        from src.strategy.dual_entropy_accumulation.scanner import Scanner
        from src.strategy.dual_entropy_accumulation.config import Config

        config = Config()
        config.scanner.daily_data_dir = '../gp-data/tushare-daily-full'
        config.scanner.minute_data_dir = '../gp-data/trade'

        scanner = Scanner(config)
        results = scanner.scan()
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

    # ================================================================
    #  辅助：从 dict 重建 DailyEntropyResult
    # ================================================================

    @staticmethod
    def _rebuild_daily_result(d: dict) -> DailyEntropyResult:
        return DailyEntropyResult(
            perm_entropy_20=d['perm_entropy_20'],
            perm_entropy_60=d['perm_entropy_60'],
            entropy_gap=d['entropy_gap'],
            entropy_percentile=d['entropy_percentile'],
            path_irreversibility=d['path_irreversibility'],
            dominant_eigenvalue=d['dominant_eigenvalue'],
            var_lift=d['var_lift'],
            is_compressed=d['is_compressed'],
            compression_days=d['compression_days'],
            entropy_velocity_5=d.get('entropy_velocity_5', 0.0),
            path_irrev_velocity_5=d.get('path_irrev_velocity_5', 0.0),
            entropy_percentile_prev5=d.get('entropy_percentile_prev5', 0.5),
        )

    # ================================================================
    #  步骤 1：日线预筛
    # ================================================================

    def _list_stocks(self) -> List[Tuple[str, str]]:
        """列出所有日线数据文件，返回 [(stock_code, csv_path), ...]"""
        daily_dir = self.config.scanner.daily_data_dir
        if not os.path.isdir(daily_dir):
            raise FileNotFoundError(f'日线数据目录不存在: {daily_dir}')

        pairs = []
        for fname in os.listdir(daily_dir):
            if not fname.endswith('.csv'):
                continue
            stock_code = fname.replace('.csv', '')
            csv_path = os.path.join(daily_dir, fname)
            pairs.append((stock_code, csv_path))

        return sorted(pairs)

    def _compute_daily_entropy_batch(
        self,
        stock_list: List[Tuple[str, str]],
    ) -> Dict[str, Tuple[dict, np.ndarray]]:
        """
        并行计算日线熵，返回 {stock_code: (daily_result_dict, last_date)}
        """
        from dataclasses import asdict

        config_dict = {
            'min_data_days': self.config.scanner.min_data_days,
            'min_close': self.config.scanner.min_close,
            'max_close': self.config.scanner.max_close,
            'daily_config': asdict(self.config.daily),
        }

        tasks = [(code, path, config_dict) for code, path in stock_list]
        results = {}

        workers = self.config.scanner.workers
        if workers <= 1:
            for task in tasks:
                out = _daily_entropy_worker(task)
                if out is not None:
                    code, result_dict, prices, last_date = out
                    results[code] = (result_dict, last_date)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_daily_entropy_worker, t): t[0] for t in tasks}
                for future in as_completed(futures):
                    try:
                        out = future.result()
                    except Exception:
                        continue
                    if out is not None:
                        code, result_dict, prices, last_date = out
                        results[code] = (result_dict, last_date)

        return results

    def _filter_compressed(
        self,
        daily_results: Dict[str, Tuple[dict, np.ndarray]],
    ) -> Dict[str, dict]:
        """筛选日线处于压缩态的股票。"""
        cfg = self.config.fusion
        compressed = {}

        for code, (result_dict, _) in daily_results.items():
            ep = result_dict['entropy_percentile']
            pe = result_dict['perm_entropy_20']

            # 宽松预筛（最终硬性门槛在 fusion_signal 中检查）
            if np.isfinite(ep) and ep <= cfg.daily_entropy_percentile_max + 0.10:
                if np.isfinite(pe) and pe <= cfg.daily_perm_entropy_max + 0.05:
                    compressed[code] = result_dict

        return compressed

    # ================================================================
    #  步骤 2：日内熵计算
    # ================================================================

    def _compute_intraday_entropy(
        self,
        stock_code: str,
        scan_date: str,
    ) -> Tuple[Optional[IntradayEntropyResult], List[IntradayEntropyResult]]:
        """
        计算单只股票的日内熵。

        返回 (today_result, recent_results)
        """
        minute_dir = os.path.join(self.config.scanner.minute_data_dir, stock_code)
        if not os.path.isdir(minute_dir):
            return None, []

        analyzer = IntradayEntropyAnalyzer(
            rolling_window=self.config.intraday.rolling_window,
            perm_order=self.config.intraday.perm_order,
            irrev_threshold_sigma=self.config.intraday.irrev_threshold_sigma,
            ar_order=self.config.intraday.ar_order,
        )

        # 找到 scan_date 及之前的 CSV 文件
        csv_files = sorted(f for f in os.listdir(minute_dir) if f.endswith('.csv'))
        if not csv_files:
            return None, []

        # 定位目标日期
        if scan_date:
            target_file = f'{scan_date}.csv'
            target_files_before = [f for f in csv_files if f <= target_file]
        else:
            target_files_before = csv_files

        if not target_files_before:
            return None, []

        # 今日
        today_file = target_files_before[-1]
        today_path = os.path.join(minute_dir, today_file)
        today_result = analyzer.analyze_day(stock_code, today_path)

        if today_result is None:
            return None, []

        # 近期（不含今日）
        lookback = self.config.intraday.lookback_days
        recent_files = target_files_before[max(0, len(target_files_before) - lookback - 1):-1]

        recent_results = []
        for f in recent_files:
            r = analyzer.analyze_day(stock_code, os.path.join(minute_dir, f))
            if r is not None:
                recent_results.append(r)

        return today_result, recent_results

    # ================================================================
    #  步骤 3：融合评分
    # ================================================================

    def scan(self, scan_date: str = '') -> List[FusionResult]:
        """
        执行全市场扫描。

        参数
        ----
        scan_date : str
            扫描日期（YYYY-MM-DD 格式），空字符串 = 最新

        返回
        ----
        List[FusionResult]
            按 total_score 降序排列的融合结果
        """
        scan_date = scan_date or self.config.scanner.scan_date

        # 步骤 1：日线预筛
        print('[1/3] 加载日线数据并计算日线熵...')
        stock_list = self._list_stocks()
        print(f'      共 {len(stock_list)} 只股票')

        daily_results = self._compute_daily_entropy_batch(stock_list)
        print(f'      有效日线数据: {len(daily_results)} 只')

        compressed = self._filter_compressed(daily_results)
        print(f'      日线压缩态预筛: {len(compressed)} 只')

        # 限制数量
        max_stocks = self.config.scanner.max_stocks
        if len(compressed) > max_stocks:
            # 按 entropy_percentile 排序取 top
            sorted_codes = sorted(
                compressed.keys(),
                key=lambda c: compressed[c].get('entropy_percentile', 1.0),
            )
            compressed = {c: compressed[c] for c in sorted_codes[:max_stocks]}
            print(f'      截取前 {max_stocks} 只')

        # 步骤 2：日内熵
        print(f'[2/3] 计算日内熵（{len(compressed)} 只）...')
        fusion_engine = FusionSignal(self.config.fusion)
        results: List[FusionResult] = []
        skipped = 0

        for i, (code, daily_dict) in enumerate(compressed.items()):
            if (i + 1) % 50 == 0:
                print(f'      进度: {i + 1}/{len(compressed)}')

            today_intraday, recent_intraday = self._compute_intraday_entropy(code, scan_date)
            if today_intraday is None:
                skipped += 1
                continue

            # 重建 DailyEntropyResult（不含序列）
            daily_result = self._rebuild_daily_result(daily_dict)

            # 步骤 3：融合评分
            fusion_result = fusion_engine.evaluate(
                code, daily_result, today_intraday, recent_intraday
            )
            results.append(fusion_result)

        print(f'      完成，跳过 {skipped} 只（无分钟数据）')

        # 按得分排序
        results.sort(key=lambda r: r.total_score, reverse=True)

        # 汇总
        buys = [r for r in results if r.signal == 'buy']
        watches = [r for r in results if r.signal == 'watch']
        print(f'[3/3] 扫描完成:')
        print(f'      buy={len(buys)}, watch={len(watches)}, skip={len(results) - len(buys) - len(watches)}')

        return results

    def save_results(
        self,
        results: List[FusionResult],
        output_dir: Optional[str] = None,
    ) -> str:
        """
        保存扫描结果到 CSV。

        返回输出目录路径。
        """
        output_dir = output_dir or self.config.scanner.output_dir
        os.makedirs(output_dir, exist_ok=True)

        rows = [r.to_dict() for r in results]
        df = pd.DataFrame(rows)

        # 全部结果
        all_path = os.path.join(output_dir, 'scan_results.csv')
        df.to_csv(all_path, index=False, encoding='utf-8-sig')

        # buy 信号
        buy_df = df[df['signal'] == 'buy'].copy()
        buy_path = os.path.join(output_dir, 'buy_signals.csv')
        buy_df.to_csv(buy_path, index=False, encoding='utf-8-sig')

        # watch 信号
        watch_df = df[df['signal'] == 'watch'].copy()
        watch_path = os.path.join(output_dir, 'watch_signals.csv')
        watch_df.to_csv(watch_path, index=False, encoding='utf-8-sig')

        print(f'结果已保存到: {output_dir}')
        print(f'  scan_results.csv  ({len(df)} 只)')
        print(f'  buy_signals.csv   ({len(buy_df)} 只)')
        print(f'  watch_signals.csv ({len(watch_df)} 只)')

        return output_dir

    # ================================================================
    #  卖出扫描
    # ================================================================

    def scan_sell(
        self,
        watchlist: Optional[List[str]] = None,
        scan_date: str = '',
    ) -> List[SellResult]:
        """
        对持仓/观察列表执行卖出信号扫描。

        参数
        ----
        watchlist : List[str] | None
            待检查股票代码列表，如 ['sh600000', 'sz000001']。
            为 None 时扫描所有有日线数据的股票（注意：耗时较长）。
        scan_date : str
            扫描日期（YYYY-MM-DD 格式），空字符串 = 最新

        返回
        ----
        List[SellResult]
            按 total_score 降序排列的卖出信号
        """
        scan_date = scan_date or self.config.scanner.scan_date

        # 步骤 1：确定候选列表 & 计算日线熵
        print('[1/3] 加载日线数据并计算日线熵...')
        all_stocks = self._list_stocks()

        if watchlist is not None:
            watchset = set(watchlist)
            stock_list = [(c, p) for c, p in all_stocks if c in watchset]
            print(f'      观察列表: {len(stock_list)} 只')
        else:
            stock_list = all_stocks
            print(f'      全市场: {len(stock_list)} 只')

        daily_results = self._compute_daily_entropy_batch(stock_list)
        print(f'      有效日线数据: {len(daily_results)} 只')

        # 卖出扫描不做压缩态预筛 — 检查所有候选

        # 步骤 2：日内熵 + 卖出评分
        print(f'[2/3] 计算日内熵 & 卖出评分（{len(daily_results)} 只）...')
        sell_engine = SellSignalEngine(self.config.sell)
        results: List[SellResult] = []
        skipped = 0

        for i, (code, (daily_dict, last_date)) in enumerate(daily_results.items()):
            if (i + 1) % 50 == 0:
                print(f'      进度: {i + 1}/{len(daily_results)}')

            today_intraday, recent_intraday = self._compute_intraday_entropy(code, scan_date)
            if today_intraday is None:
                skipped += 1
                continue

            daily_result = self._rebuild_daily_result(daily_dict)

            sell_result = sell_engine.evaluate(
                code, daily_result, today_intraday, recent_intraday
            )
            results.append(sell_result)

        print(f'      完成，跳过 {skipped} 只（无分钟数据）')

        # 按得分排序
        results.sort(key=lambda r: r.total_score, reverse=True)

        # 汇总
        sells = [r for r in results if r.signal == 'sell']
        warnings = [r for r in results if r.signal == 'warning']
        print(f'[3/3] 卖出扫描完成:')
        print(f'      sell={len(sells)}, warning={len(warnings)}, hold={len(results) - len(sells) - len(warnings)}')

        return results

    def save_sell_results(
        self,
        results: List[SellResult],
        output_dir: Optional[str] = None,
    ) -> str:
        """保存卖出扫描结果到 CSV。"""
        output_dir = output_dir or self.config.scanner.output_dir
        os.makedirs(output_dir, exist_ok=True)

        rows = [r.to_dict() for r in results]
        df = pd.DataFrame(rows)

        all_path = os.path.join(output_dir, 'sell_scan_results.csv')
        df.to_csv(all_path, index=False, encoding='utf-8-sig')

        sell_df = df[df['signal'] == 'sell'].copy()
        sell_path = os.path.join(output_dir, 'sell_signals.csv')
        sell_df.to_csv(sell_path, index=False, encoding='utf-8-sig')

        warn_df = df[df['signal'] == 'warning'].copy()
        warn_path = os.path.join(output_dir, 'sell_warnings.csv')
        warn_df.to_csv(warn_path, index=False, encoding='utf-8-sig')

        print(f'卖出结果已保存到: {output_dir}')
        print(f'  sell_scan_results.csv ({len(df)} 只)')
        print(f'  sell_signals.csv      ({len(sell_df)} 只)')
        print(f'  sell_warnings.csv     ({len(warn_df)} 只)')

        return output_dir
