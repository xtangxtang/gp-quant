"""
四层熵交易系统 - 核心整合模块

整合四层架构：
1. 市场门控层
2. 个股状态层
3. 执行成本层
4. 实验模型层

输出最终交易决策。
"""

import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .config import Config
from .market_gate import MarketGate, MarketGateOutput, MarketState
from .stock_state import StockState, StockStateOutput, StateFlow
from .execution_cost import ExecutionCost, ExecutionCostOutput
from .experimental_model import ExperimentalModel, ExperimentalOutput


def _load_stock_data_worker(task_data) -> Optional[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Load a single stock's daily CSV and return date, price, and volume arrays.

    Reads from tushare-daily-full format:
        ts_code, trade_date, open, high, low, close, ..., vol, amount, ...
    """
    stock_code, csv_path = task_data

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    required_cols = {'trade_date', 'close', 'vol'}
    if not required_cols.issubset(df.columns):
        return None

    df['trade_date'] = pd.to_numeric(df['trade_date'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['vol'] = pd.to_numeric(df['vol'], errors='coerce')
    df = df.dropna(subset=['trade_date', 'close', 'vol'])

    if df.empty or len(df) < 30:
        return None

    df = df.sort_values('trade_date')
    df = df.drop_duplicates(subset='trade_date', keep='last')

    dates = pd.to_datetime(df['trade_date'].astype(int).astype(str), format='%Y%m%d', errors='coerce')
    df = df[dates.notna()]
    dates = dates[dates.notna()]

    if len(df) < 30:
        return None

    return (
        stock_code,
        dates.to_numpy(dtype='datetime64[ns]'),
        df['close'].to_numpy(dtype=np.float64, copy=False),
        df['vol'].to_numpy(dtype=np.float64, copy=False),
    )


@dataclass
class StockDecision:
    """单只股票的最终决策"""

    stock_code: str
    stock_name: str

    # 各层输出
    market_gate: MarketGateOutput
    stock_state: StockStateOutput
    execution: ExecutionCostOutput
    experimental: ExperimentalOutput

    # 最终决策
    action: str  # buy/sell/hold/wait
    position_size: float  # 建议仓位金额
    confidence: float  # 置信度

    # 详细信息
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'stock_code': self.stock_code,
            'stock_name': self.stock_name,
            'action': self.action,
            'position_size': self.position_size,
            'confidence': self.confidence,
            'market_gate': self.market_gate.to_dict() if self.market_gate else {},
            'stock_state': self.stock_state.to_dict() if self.stock_state else {},
            'execution': self.execution.to_dict() if self.execution else {},
            'experimental': self.experimental.to_dict() if self.experimental else {},
        }


def _evaluate_stock_worker(
    task_data,
    market_gate_output: MarketGateOutput,
    config: Config,
) -> Optional[StockDecision]:
    """
    单个股票的评估 worker 函数（用于 multiprocessing.Pool）

    这个函数必须在模块级别定义，以便 spawn 模式可以 pickle 它。
    """
    import pandas as pd

    stock_code, dates_arr, prices_arr, volumes_arr = task_data

    stock_index = pd.to_datetime(dates_arr)

    # 转换为 pandas Series
    stock_prices = pd.Series(prices_arr, index=stock_index)
    stock_volumes = pd.Series(volumes_arr, index=stock_index)
    stock_returns = np.log(stock_prices / stock_prices.shift(1)).dropna()

    if len(stock_prices) < 30:
        return None

    # 创建临时的评估器实例
    from .stock_state import StockState
    from .execution_cost import ExecutionCost
    from .experimental_model import ExperimentalModel

    stock_state_eval = StockState(config.stock_state, config.layer)
    execution_eval = ExecutionCost(config.execution, config.execution.initial_capital)
    experimental_eval = ExperimentalModel(config.experimental)

    # Layer 2: 个股状态
    stock_state_output = stock_state_eval.evaluate(
        prices=stock_prices,
        volumes=stock_volumes,
        dates=stock_prices.index.to_series(),
        gate_open=not market_gate_output.abandonment_flag,
    )

    # Layer 3: 执行成本
    execution_output = execution_eval.evaluate(
        signal_strength=stock_state_output.total_score,
        market_gate_score=market_gate_output.gate_score,
        market_state=market_gate_output.state,
        stock_state=stock_state_output.current_state,
        current_price=stock_prices.iloc[-1],
        entry_price=None,
        current_return=0.0,
        volatility=stock_returns.std(),
        noise_cost=market_gate_output.noise_cost,
    )

    # Layer 4: 实验模型
    experimental_output = experimental_eval.evaluate(
        stock_returns, stock_volumes
    )

    # 综合决策
    action, confidence = _compute_final_action_static(
        stock_state_output, market_gate_output, execution_output, config.layer
    )

    decision = StockDecision(
        stock_code=stock_code,
        stock_name=stock_code,
        market_gate=market_gate_output,
        stock_state=stock_state_output,
        execution=execution_output,
        experimental=experimental_output,
        action=action,
        position_size=execution_output.position_size if action == 'buy' else 0,
        confidence=confidence,
        details={
            'entry_mode': execution_output.entry_mode,
            'staged_days': execution_output.staged_entry_days,
        }
    )

    return decision


def _compute_final_action_static(
    stock_state: StockStateOutput,
    market_gate: MarketGateOutput,
    execution: ExecutionCostOutput,
    layer_weights,
) -> Tuple[str, float]:
    """
    静态版本的 compute_final_action，用于 worker 进程
    """
    # 个股状态信号
    state_signal = stock_state.signal
    state_score = stock_state.total_score

    # 市场门控信号
    gate_score = market_gate.gate_score
    gate_open = not market_gate.abandonment_flag

    # 执行层信号
    exec_mode = execution.entry_mode
    abandon_flag = execution.abandonment_flag

    # 综合置信度
    confidence = (
        state_score * layer_weights.stock_state_weight +
        gate_score * layer_weights.market_gate_weight +
        execution.position_scale * layer_weights.execution_weight
    )

    # 最终决策逻辑
    if not gate_open or abandon_flag:
        action = 'wait'
        confidence = 0.0

    elif exec_mode == 'skip':
        action = 'wait'

    elif state_signal == 'buy' and gate_open:
        action = 'buy'

    elif state_signal == 'sell':
        action = 'sell'

    elif state_signal == 'hold':
        action = 'hold'

    else:
        action = 'wait'

    return action, confidence


def evaluate_single_stock(
    stock_code: str,
    stock_prices,  # np.ndarray or pd.Series
    stock_volumes,  # np.ndarray or pd.Series
    stock_returns,  # np.ndarray or pd.Series
    market_gate_output: MarketGateOutput,
    stock_state_fn,
    execution_cost_fn,
    experimental_fn,
    final_action_fn,
) -> Optional["StockDecision"]:
    """
    评估单只股票（用于并行计算）
    """
    import pandas as pd

    # 转换为 pandas Series（如果需要）
    if isinstance(stock_prices, np.ndarray):
        stock_prices = pd.Series(stock_prices)
    if isinstance(stock_volumes, np.ndarray):
        stock_volumes = pd.Series(stock_volumes)
    if isinstance(stock_returns, np.ndarray):
        stock_returns = pd.Series(stock_returns)

    if len(stock_prices) < 30:
        return None

    # Layer 2: 个股状态
    stock_state_output = stock_state_fn(
        prices=stock_prices,
        volumes=stock_volumes,
        dates=pd.Series(range(len(stock_prices))),  # 使用索引代替日期
        gate_open=not market_gate_output.abandonment_flag,
    )

    # Layer 3: 执行成本
    execution_output = execution_cost_fn(
        signal_strength=stock_state_output.total_score,
        market_gate_score=market_gate_output.gate_score,
        market_state=market_gate_output.state,
        stock_state=stock_state_output.current_state,
        current_price=stock_prices.iloc[-1],
        entry_price=None,
        current_return=0.0,
        volatility=stock_returns.std(),
        noise_cost=market_gate_output.noise_cost,
    )

    # Layer 4: 实验模型
    experimental_output = experimental_fn(
        stock_returns, stock_volumes
    )

    # 综合决策
    action, confidence = final_action_fn(
        stock_state_output, market_gate_output, execution_output
    )

    decision = StockDecision(
        stock_code=stock_code,
        stock_name=stock_code,
        market_gate=market_gate_output,
        stock_state=stock_state_output,
        execution=execution_output,
        experimental=experimental_output,
        action=action,
        position_size=execution_output.position_size if action == 'buy' else 0,
        confidence=confidence,
        details={
            'entry_mode': execution_output.entry_mode,
            'staged_days': execution_output.staged_entry_days,
        }
    )

    return decision


@dataclass
class SystemOutput:
    """系统总输出"""

    scan_date: str
    market_gate: MarketGateOutput
    decisions: List[StockDecision]

    # 汇总统计
    total_buy: int
    total_sell: int
    total_hold: int
    total_wait: int

    # 建议总仓位
    recommended_total_position: float

    def to_dataframe(self) -> pd.DataFrame:
        """转换为 DataFrame"""
        records = []
        for d in self.deisions:
            rec = d.to_dict()
            rec.update({
                'market_state': self.market_gate.state,
                'gate_score': self.market_gate.gate_score,
            })
            records.append(rec)
        return pd.DataFrame(records)


class FourLayerEntropySystem:
    """
    四层熵交易系统

    基于 GP-QUANT 12 篇复杂系统论文实现的完整交易架构。
    """

    def __init__(self, config: Config):
        self.config = config

        # 初始化四层
        self.market_gate = MarketGate(config.market_gate)
        self.stock_state = StockState(config.stock_state, config.layer)
        self.execution_cost = ExecutionCost(config.execution, config.execution.initial_capital)
        self.experimental = ExperimentalModel(config.experimental)

        # 行业映射缓存
        self.industry_map: Optional[Dict[str, str]] = None

    def load_data(
        self,
        data_dir: str,
        basic_path: str,
        max_stocks: int = 100,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, str]]:
        """
        加载股票数据

        返回
        ----
        (prices, volumes, returns, industry_map)
        """
        # 加载股票基本信息
        basic = pd.read_csv(basic_path)
        industry_map = dict(zip(basic['ts_code'], basic['industry']))

        # 获取股票列表（日线 CSV 文件，每只股票一个文件）
        csv_files = sorted(f for f in os.listdir(data_dir) if f.endswith('.csv'))
        csv_files = csv_files[:max_stocks]

        num_workers = self.config.num_workers
        if num_workers <= 0:
            num_workers = os.cpu_count() or 1
        num_workers = min(num_workers, max(1, len(csv_files)))

        # 加载各股票数据
        all_prices = {}
        all_volumes = {}

        load_tasks = [
            (os.path.splitext(csv_file)[0], os.path.join(data_dir, csv_file))
            for csv_file in csv_files
        ]

        if self.config.use_parallel and num_workers > 1:
            print(f"并行加载股票数据（共{len(load_tasks)}只，并发数：{num_workers}）...", flush=True)
            futures = {}
            loaded_count = 0

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                for task in load_tasks:
                    futures[executor.submit(_load_stock_data_worker, task)] = task[0]

                for future in as_completed(futures):
                    stock_code = futures[future]
                    try:
                        result = future.result()
                    except Exception:
                        continue

                    if result is None:
                        continue

                    _, dates_arr, prices_arr, volumes_arr = result
                    stock_index = pd.DatetimeIndex(dates_arr)
                    all_prices[stock_code] = pd.Series(prices_arr, index=stock_index)
                    all_volumes[stock_code] = pd.Series(volumes_arr, index=stock_index)

                    loaded_count += 1
                    if loaded_count % 100 == 0 or loaded_count == len(load_tasks):
                        print(f"  已加载：{loaded_count}/{len(load_tasks)}", flush=True)
        else:
            print(f"串行加载股票数据（共{len(load_tasks)}只）...", flush=True)
            for task in load_tasks:
                result = _load_stock_data_worker(task)
                if result is None:
                    continue

                stock_code, dates_arr, prices_arr, volumes_arr = result
                stock_index = pd.DatetimeIndex(dates_arr)
                all_prices[stock_code] = pd.Series(prices_arr, index=stock_index)
                all_volumes[stock_code] = pd.Series(volumes_arr, index=stock_index)

        ordered_stock_codes = [task[0] for task in load_tasks if task[0] in all_prices]
        all_prices = {stock_code: all_prices[stock_code] for stock_code in ordered_stock_codes}
        all_volumes = {stock_code: all_volumes[stock_code] for stock_code in ordered_stock_codes}

        if not ordered_stock_codes:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

        # 全市场分钟历史存在上市时间差与停牌缺口，使用并集对齐而非全量交集。
        prices_df = pd.concat(
            [all_prices[stock_code].rename(stock_code) for stock_code in ordered_stock_codes],
            axis=1,
            sort=True,
        ).sort_index().dropna(how='all')
        volumes_df = pd.concat(
            [all_volumes[stock_code].rename(stock_code) for stock_code in ordered_stock_codes],
            axis=1,
            sort=True,
        ).reindex(prices_df.index)
        returns_df = pd.concat(
            [np.log(all_prices[stock_code] / all_prices[stock_code].shift(1)).rename(stock_code) for stock_code in ordered_stock_codes],
            axis=1,
            sort=True,
        ).reindex(prices_df.index)

        if prices_df.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

        return prices_df, volumes_df, returns_df, industry_map

    def evaluate_market_gate(
        self,
        returns: pd.DataFrame,
        volumes: pd.DataFrame,
        prices: pd.DataFrame,
        industry_map: Dict[str, str],
    ) -> MarketGateOutput:
        """
        Layer 1: 评估市场门控
        """
        return self.market_gate.evaluate(returns, volumes, prices, industry_map)

    def evaluate_stock_state(
        self,
        prices: pd.Series,
        volumes: pd.Series,
        dates: pd.Series,
        gate_open: bool,
    ) -> StockStateOutput:
        """
        Layer 2: 评估个股状态
        """
        return self.stock_state.evaluate(prices, volumes, dates, gate_open)

    def evaluate_execution_cost(
        self,
        signal_strength: float,
        market_gate_score: float,
        market_state: str,
        stock_state: str,
        current_price: float,
        entry_price: Optional[float],
        current_return: float,
        volatility: float,
        noise_cost: float,
    ) -> ExecutionCostOutput:
        """
        Layer 3: 评估执行成本
        """
        return self.execution_cost.evaluate(
            signal_strength=signal_strength,
            market_gate_score=market_gate_score,
            market_state=market_state,
            stock_state=stock_state,
            current_price=current_price,
            entry_price=entry_price,
            current_return=current_return,
            volatility=volatility,
            noise_cost=noise_cost,
        )

    def evaluate_experimental(
        self,
        returns: pd.Series,
        volumes: pd.Series,
    ) -> ExperimentalOutput:
        """
        Layer 4: 评估实验模型
        """
        return self.experimental.evaluate(returns, volumes)

    def compute_final_action(
        self,
        stock_state: StockStateOutput,
        market_gate: MarketGateOutput,
        execution: ExecutionCostOutput,
    ) -> Tuple[str, float]:
        """
        计算最终动作和置信度

        基于各层权重的综合决策。
        """
        layer_weights = self.config.layer

        # 个股状态信号
        state_signal = stock_state.signal
        state_score = stock_state.total_score

        # 市场门控信号
        gate_score = market_gate.gate_score
        gate_open = not market_gate.abandonment_flag

        # 执行层信号
        exec_mode = execution.entry_mode
        abandon_flag = execution.abandonment_flag

        # 综合置信度
        confidence = (
            state_score * layer_weights.stock_state_weight +
            gate_score * layer_weights.market_gate_weight +
            execution.position_scale * layer_weights.execution_weight
        )

        # 最终决策逻辑
        if not gate_open or abandon_flag:
            action = 'wait'
            confidence = 0.0

        elif exec_mode == 'skip':
            action = 'wait'

        elif state_signal == 'buy' and gate_open:
            action = 'buy'

        elif state_signal == 'sell':
            action = 'sell'

        elif state_signal == 'hold':
            action = 'hold'

        else:
            action = 'wait'

        return action, confidence

    def scan(
        self,
        data_dir: str,
        basic_path: str,
        scan_date: Optional[str] = None,
        max_stocks: int = 100,
    ) -> SystemOutput:
        """
        执行全市场扫描

        参数
        ----
        data_dir : str
            数据目录
        basic_path : str
            股票基本信息路径
        scan_date : str, optional
            扫描日期
        max_stocks : int
            最大股票数

        返回
        ----
        SystemOutput
            系统扫描结果
        """
        print("加载数据...")
        prices, volumes, returns, industry_map = self.load_data(
            data_dir, basic_path, max_stocks
        )

        if prices.empty:
            raise ValueError("未找到有效数据")

        print(f"加载完成：{len(prices)} 交易日，{len(prices.columns)} 只股票")

        # Layer 1: 市场门控评估
        print("评估市场门控...")
        market_gate_output = self.evaluate_market_gate(
            returns, volumes, prices, industry_map
        )
        print(f"  市场状态：{market_gate_output.state}")
        print(f"  门控得分：{market_gate_output.gate_score:.2f}")

        gate_open = not market_gate_output.abandonment_flag

        # 逐股评估
        decisions = []
        stock_codes = list(prices.columns)

        # 确定工作进程数量
        import multiprocessing
        num_workers = self.config.num_workers
        if num_workers <= 0:
            num_workers = multiprocessing.cpu_count()

        print(f"评估个股状态（共{len(stock_codes)}只，并发数：{num_workers}）...")

        if self.config.use_parallel and num_workers > 1:
            # 并行计算模式 - 使用 multiprocessing.Pool
            import multiprocessing as mp
            import tempfile
            import pickle
            from functools import partial

            # 创建临时文件存储结果
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
            temp_results_path = temp_file.name
            temp_file.close()

            # 准备并行计算的任务数据
            tasks = []
            for stock_code in stock_codes:
                stock_prices = prices[stock_code].dropna()
                stock_volumes = volumes[stock_code].dropna()
                if len(stock_prices) >= 30:
                    tasks.append((
                        stock_code,
                        stock_prices.index.to_numpy(),
                        stock_prices.values,
                        stock_volumes.values,
                    ))

            # 创建进程池 - 使用 spawn 模式
            ctx = mp.get_context('spawn')

            # 使用 imap_unordered 边处理边返回结果
            with ctx.Pool(processes=num_workers) as pool:
                # 使用 partial 包装函数
                worker_fn = partial(
                    _evaluate_stock_worker,
                    market_gate_output=market_gate_output,
                    config=self.config,
                )

                # 分批处理，每批 100 个股票
                batch_size = 100
                total_tasks = len(tasks)

                with open(temp_results_path, 'ab') as temp_f:
                    for i in range(0, total_tasks, batch_size):
                        batch = tasks[i:i + batch_size]
                        for result in pool.imap_unordered(worker_fn, batch, chunksize=4):
                            if result is not None:
                                pickle.dump(result, temp_f)
                                temp_f.flush()

                        progress = min(i + batch_size, total_tasks)
                        print(f"  进度：{progress}/{total_tasks}")

            # 从临时文件读取结果
            with open(temp_results_path, 'rb') as temp_f:
                while True:
                    try:
                        result = pickle.load(temp_f)
                        decisions.append(result)
                    except EOFError:
                        break

            # 删除临时文件
            import os
            os.unlink(temp_results_path)

        else:
            # 串行计算模式
            for i, stock_code in enumerate(stock_codes):
                if (i + 1) % 10 == 0:
                    print(f"  进度：{i + 1}/{len(stock_codes)}")

                stock_prices = prices[stock_code].dropna()
                stock_volumes = volumes[stock_code].dropna()
                stock_returns = returns[stock_code].dropna()

                if len(stock_prices) < 30:
                    continue

                # Layer 2: 个股状态
                stock_state_output = self.evaluate_stock_state(
                    prices=stock_prices,
                    volumes=stock_volumes,
                    dates=stock_prices.index.to_series(),
                    gate_open=gate_open,
                )

                # Layer 3: 执行成本
                execution_output = self.evaluate_execution_cost(
                    signal_strength=stock_state_output.total_score,
                    market_gate_score=market_gate_output.gate_score,
                    market_state=market_gate_output.state,
                    stock_state=stock_state_output.current_state,
                    current_price=stock_prices.iloc[-1],
                    entry_price=None,
                    current_return=0.0,
                    volatility=stock_returns.std(),
                    noise_cost=market_gate_output.noise_cost,
                )

                # Layer 4: 实验模型
                experimental_output = self.evaluate_experimental(
                    stock_returns, stock_volumes
                )

                # 综合决策
                action, confidence = self.compute_final_action(
                    stock_state_output, market_gate_output, execution_output
                )

                decision = StockDecision(
                    stock_code=stock_code,
                    stock_name=stock_code,
                    market_gate=market_gate_output,
                    stock_state=stock_state_output,
                    execution=execution_output,
                    experimental=experimental_output,
                    action=action,
                    position_size=execution_output.position_size if action == 'buy' else 0,
                    confidence=confidence,
                    details={
                        'entry_mode': execution_output.entry_mode,
                        'staged_days': execution_output.staged_entry_days,
                    }
                )

                decisions.append(decision)

        # 汇总统计
        total_buy = sum(1 for d in decisions if d.action == 'buy')
        total_sell = sum(1 for d in decisions if d.action == 'sell')
        total_hold = sum(1 for d in decisions if d.action == 'hold')
        total_wait = sum(1 for d in decisions if d.action == 'wait')

        # 建议总仓位
        recommended_position = sum(d.position_size for d in decisions if d.action == 'buy')

        # 确定扫描日期
        if scan_date is None:
            scan_date = str(prices.index[-1].date())

        output = SystemOutput(
            scan_date=scan_date,
            market_gate=market_gate_output,
            decisions=decisions,
            total_buy=total_buy,
            total_sell=total_sell,
            total_hold=total_hold,
            total_wait=total_wait,
            recommended_total_position=recommended_position,
        )

        return output
