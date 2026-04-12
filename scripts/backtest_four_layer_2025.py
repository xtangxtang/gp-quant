"""
四层熵交易系统 2025年回归测试

逐日模拟：
- 每个交易日运行 scan，获取 buy/sell/wait 信号
- 跟踪持仓、计算收益、扣除交易费用
- 最终输出全年盈亏
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# 添加项目根目录到 path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.strategy.four_layer_entropy_system import FourLayerEntropySystem, Config
from src.strategy.four_layer_entropy_system.core_system import _load_stock_data_worker


# ── 配置 ──────────────────────────────────────────────────────────────
DATA_DIR = "/nvme5/xtang/gp-workspace/gp-data/tushare-daily-full"
BASIC_PATH = "/nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv"
TRADE_CAL_PATH = "/nvme5/xtang/gp-workspace/gp-data/tushare-trade_cal/trade_cal.csv"

BACKTEST_START = "20250101"
BACKTEST_END = "20251231"
INITIAL_CAPITAL = 1_000_000.0

# 回测参数
MAX_POSITIONS = 10           # 最大同时持仓数
HOLD_DAYS_MIN = 3            # 最短持有天数（避免噪声）
HOLD_DAYS_MAX = 60           # 最长持有天数（兜底防护）
MAX_STOCKS_SCAN = 500        # 扫描股票数（SH/SZ主板优先）
LOOKBACK_DAYS = 250          # 回看天数（需要约1年历史数据计算熵指标）
MIN_AVG_AMOUNT = 1000        # 最近20日日均成交额下限（万元），过滤流动性差的票

# 交易费用
COMMISSION = 0.0003          # 万三
STAMP_TAX = 0.001            # 千一（卖出）
SLIPPAGE = 0.001             # 千一滑点

# 止损（安全网，不设固定止盈，完全靠熵信号+移动止损退出）
STOP_LOSS = 0.08             # 8% 止损
TRAILING_STOP = 0.05         # 5% 移动止损（从最高点回撤5%离场，自动锁利）
TRAILING_ACTIVATE = 0.03     # 移动止损激活阈值：盈利超过3%后开启移动止损

# 熵退出参数
ENTROPY_EXIT_PERCENTILE = 0.65  # 熵百分位超过此值时考虑退出（有序→无序）
ENTROPY_EXIT_GAP = -0.005       # 熵差低于此值退出（短期熵>长期=扩散）
ENTROPY_EXIT_ACCEL = 0.005      # 熵加速度超过此值退出（熵加速扩张）
ENTROPY_EXIT_VOTES = 2          # 需要至少N个熵指标同时触发才卖出

# 扫描频率
SCAN_INTERVAL = 5            # 每周扫描一次
ENTROPY_CHECK_INTERVAL = 1   # 每天检查持仓熵状态

# 抑制 FutureWarning
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


@dataclass
class Position:
    """持仓记录"""
    stock_code: str
    entry_date: str
    entry_price: float
    shares: int
    cost: float                # 含费用的总成本
    highest_price: float       # 用于跟踪止盈
    hold_days: int = 0
    entry_mode: str = 'full'
    # 买入时的熵状态（基准）
    entry_entropy_percentile: float = 0.0
    entry_perm_entropy: float = 0.0
    entry_entropy_gap: float = 0.0


@dataclass
class Trade:
    """交易记录"""
    date: str
    stock_code: str
    action: str                # buy / sell
    price: float
    shares: int
    amount: float
    fee: float
    pnl: float = 0.0          # 卖出时的盈亏


def load_trading_calendar(cal_path: str, start: str, end: str) -> List[str]:
    """加载交易日历，返回目标时间段内的交易日列表"""
    cal = pd.read_csv(cal_path)
    cal = cal[cal['exchange'] == 'SSE']
    cal = cal[cal['is_open'] == 1]
    cal['cal_date'] = cal['cal_date'].astype(str)
    mask = (cal['cal_date'] >= start) & (cal['cal_date'] <= end)
    return sorted(cal.loc[mask, 'cal_date'].tolist())


def load_all_stock_data(data_dir: str, max_stocks: int) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """一次性加载所有股票数据到内存，仅SH/SZ主板+创业板，按成交额排序"""
    all_csv = sorted(f for f in os.listdir(data_dir) if f.endswith('.csv'))

    # 仅加载SH/SZ主板+创业板（排除BJ北交所、科创板sh688）
    sh_main = [f for f in all_csv if f.startswith('sh60')]      # 沪市主板
    sz_main = [f for f in all_csv if f.startswith('sz00')]      # 深市主板
    sz_gem  = [f for f in all_csv if f.startswith('sz30')]      # 创业板
    eligible = sh_main + sz_main + sz_gem

    print(f"  股票池: 沪主板 {len(sh_main)}, 深主板 {len(sz_main)}, 创业板 {len(sz_gem)}")

    # 先加载全部，然后按近期日均成交额排序取 top N
    stock_data = {}
    avg_amounts = {}  # stock_code -> avg daily amount (近20日)
    for csv_file in eligible:
        stock_code = os.path.splitext(csv_file)[0]
        csv_path = os.path.join(data_dir, csv_file)
        result = _load_stock_data_worker((stock_code, csv_path))
        if result is not None:
            _, dates_arr, prices_arr, volumes_arr = result
            stock_data[stock_code] = (dates_arr, prices_arr, volumes_arr)
            # 用最近20日成交量×价格近似成交额 (vol单位是手=100股)
            n = min(20, len(prices_arr))
            avg_amt = np.mean(prices_arr[-n:] * volumes_arr[-n:] * 100)  # 元
            avg_amounts[stock_code] = avg_amt

    # 按成交额降序排列，取流动性最好的 top N
    ranked = sorted(avg_amounts.keys(), key=lambda sc: avg_amounts[sc], reverse=True)
    # 过滤低流动性
    ranked = [sc for sc in ranked if avg_amounts[sc] >= MIN_AVG_AMOUNT * 1e4]  # 万元转元
    selected = ranked[:max_stocks]

    stock_data = {sc: stock_data[sc] for sc in selected}
    print(f"  流动性过滤后: {len(stock_data)} 只 (日均成交额 >= {MIN_AVG_AMOUNT}万)")
    if selected:
        top_amt = avg_amounts[selected[0]] / 1e8
        bot_amt = avg_amounts[selected[-1]] / 1e8
        print(f"  成交额范围: {bot_amt:.1f}亿 ~ {top_amt:.1f}亿")

    return stock_data


def build_truncated_dataframes(
    stock_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    cutoff_date: pd.Timestamp,
    lookback_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """构建截止到 cutoff_date 的价格/成交量/收益率 DataFrame"""
    all_prices = {}
    all_volumes = {}

    for stock_code, (dates_arr, prices_arr, volumes_arr) in stock_data.items():
        idx = pd.DatetimeIndex(dates_arr)
        mask = idx <= cutoff_date
        if mask.sum() < 30:
            continue

        # 只取最近 lookback_days 天
        sel_idx = idx[mask]
        sel_prices = prices_arr[mask]
        sel_volumes = volumes_arr[mask]

        if lookback_days > 0 and len(sel_idx) > lookback_days:
            sel_idx = sel_idx[-lookback_days:]
            sel_prices = sel_prices[-lookback_days:]
            sel_volumes = sel_volumes[-lookback_days:]

        all_prices[stock_code] = pd.Series(sel_prices, index=sel_idx)
        all_volumes[stock_code] = pd.Series(sel_volumes, index=sel_idx)

    if not all_prices:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    ordered = sorted(all_prices.keys())
    prices_df = pd.concat(
        [all_prices[sc].rename(sc) for sc in ordered],
        axis=1, sort=True,
    ).sort_index().dropna(how='all')

    volumes_df = pd.concat(
        [all_volumes[sc].rename(sc) for sc in ordered],
        axis=1, sort=True,
    ).reindex(prices_df.index)

    returns_df = pd.concat(
        [np.log(all_prices[sc] / all_prices[sc].shift(1)).rename(sc) for sc in ordered],
        axis=1, sort=True,
    ).reindex(prices_df.index)

    return prices_df, volumes_df, returns_df


def get_price_on_date(
    stock_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    stock_code: str,
    target_date: pd.Timestamp,
) -> Optional[float]:
    """获取某只股票在某日期的收盘价"""
    if stock_code not in stock_data:
        return None
    dates_arr, prices_arr, _ = stock_data[stock_code]
    idx = pd.DatetimeIndex(dates_arr)
    mask = idx == target_date
    if mask.any():
        return float(prices_arr[mask][-1])
    # 找最近的前一日
    mask = idx <= target_date
    if mask.any():
        return float(prices_arr[mask][-1])
    return None


def calculate_buy_fee(amount: float) -> float:
    """计算买入费用"""
    return amount * (COMMISSION + SLIPPAGE)


def calculate_sell_fee(amount: float) -> float:
    """计算卖出费用"""
    return amount * (COMMISSION + STAMP_TAX + SLIPPAGE)


def evaluate_entropy_exit(
    stock_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    stock_code: str,
    cutoff_date: pd.Timestamp,
    system: FourLayerEntropySystem,
    pos: Position,
) -> Tuple[bool, str, Dict]:
    """
    评估持仓的熵状态，判断是否应该退出

    核心逻辑：买入时处于低熵（有序），当熵回升到无序状态时卖出。

    返回: (should_exit, reason, details)
    """
    if stock_code not in stock_data:
        return False, '', {}

    dates_arr, prices_arr, volumes_arr = stock_data[stock_code]
    idx = pd.DatetimeIndex(dates_arr)
    mask = idx <= cutoff_date
    if mask.sum() < 30:
        return False, '', {}

    sel_idx = idx[mask]
    sel_prices = prices_arr[mask]
    sel_volumes = volumes_arr[mask]

    # 只取最近 LOOKBACK_DAYS 天
    if len(sel_idx) > LOOKBACK_DAYS:
        sel_idx = sel_idx[-LOOKBACK_DAYS:]
        sel_prices = sel_prices[-LOOKBACK_DAYS:]
        sel_volumes = sel_volumes[-LOOKBACK_DAYS:]

    stock_prices = pd.Series(sel_prices, index=sel_idx)
    stock_volumes = pd.Series(sel_volumes, index=sel_idx)

    try:
        stock_state = system.evaluate_stock_state(
            prices=stock_prices,
            volumes=stock_volumes,
            dates=stock_prices.index.to_series(),
            gate_open=True,
        )
    except Exception:
        return False, '', {}

    # 收集当前熵指标
    details = {
        'state': stock_state.current_state,
        'perm_entropy': stock_state.permutation_entropy,
        'entropy_percentile': stock_state.entropy_percentile,
        'entropy_gap': stock_state.entropy_gap,
        'entropy_accel': stock_state.entropy_accel,
        'entropy_quality': stock_state.entropy_quality,
        'signal': stock_state.signal,
    }

    # ── 熵退出投票机制 ──
    votes = 0
    reasons = []

    # 1. 状态直接判定：已进入扩散/衰竭态
    if stock_state.current_state in ('diffusion', 'exhaustion'):
        votes += 2  # 双票权重
        reasons.append(f'状态={stock_state.current_state}')

    # 2. 熵百分位从低位升到高位（有序→无序）
    if stock_state.entropy_percentile > ENTROPY_EXIT_PERCENTILE:
        votes += 1
        reasons.append(f'熵百分位={stock_state.entropy_percentile:.2f}>{ENTROPY_EXIT_PERCENTILE}')

    # 3. 熵差变负（短期熵 > 长期熵 = 短期扩散）
    if stock_state.entropy_gap < ENTROPY_EXIT_GAP:
        votes += 1
        reasons.append(f'熵差={stock_state.entropy_gap:.4f}<{ENTROPY_EXIT_GAP}')

    # 4. 熵加速度正（熵加速扩张）
    if stock_state.entropy_accel > ENTROPY_EXIT_ACCEL:
        votes += 1
        reasons.append(f'熵加速={stock_state.entropy_accel:.4f}>{ENTROPY_EXIT_ACCEL}')

    # 5. 系统发出 sell 信号
    if stock_state.signal == 'sell':
        votes += 1
        reasons.append('信号=sell')

    should_exit = votes >= ENTROPY_EXIT_VOTES
    reason_str = '熵退出(' + ','.join(reasons) + ')' if should_exit else ''

    return should_exit, reason_str, details


def run_backtest():
    """运行回测主逻辑"""

    print("=" * 70)
    print("四层熵交易系统 - 2025年回归测试")
    print("=" * 70)
    print(f"回测区间: {BACKTEST_START} ~ {BACKTEST_END}")
    print(f"初始资金: {INITIAL_CAPITAL:,.0f}")
    print(f"最大持仓数: {MAX_POSITIONS}")
    print(f"持仓天数: {HOLD_DAYS_MIN}~{HOLD_DAYS_MAX} (熵驱动退出)")
    print(f"止损: {STOP_LOSS*100:.0f}% (无固定止盈，完全熵驱动退出)")
    print(f"移动止损: 盈利>{TRAILING_ACTIVATE*100:.0f}%后激活, 从高点回撤{TRAILING_STOP*100:.0f}%离场")
    print(f"熵退出: 百分位>{ENTROPY_EXIT_PERCENTILE}, 熵差<{ENTROPY_EXIT_GAP}, 需{ENTROPY_EXIT_VOTES}票")
    print(f"扫描间隔: 每 {SCAN_INTERVAL} 个交易日")
    print(f"扫描股票数: {MAX_STOCKS_SCAN}")
    print()

    # 1. 加载交易日历
    print("加载交易日历...")
    trading_days = load_trading_calendar(TRADE_CAL_PATH, BACKTEST_START, BACKTEST_END)
    print(f"  2025年交易日数: {len(trading_days)}")

    # 2. 一次性加载全部股票数据
    print("加载股票数据（全量加载到内存）...")
    t0 = time.time()
    stock_data = load_all_stock_data(DATA_DIR, MAX_STOCKS_SCAN)
    print(f"  加载完成: {len(stock_data)} 只股票, 耗时 {time.time() - t0:.1f}s")

    # 3. 加载 basic 信息（行业映射）
    basic = pd.read_csv(BASIC_PATH)
    if 'industry' not in basic.columns:
        basic['industry'] = 'unknown'
    ts_code_col = 'ts_code' if 'ts_code' in basic.columns else basic.columns[0]
    industry_map = dict(zip(basic[ts_code_col], basic['industry']))

    # 4. 初始化系统
    config = Config()
    config.use_parallel = False       # 回测模式串行，稳定可复现
    config.max_stocks = MAX_STOCKS_SCAN
    system = FourLayerEntropySystem(config)

    # 5. 回测状态
    cash = INITIAL_CAPITAL
    positions: Dict[str, Position] = {}   # stock_code -> Position
    trades: List[Trade] = []
    daily_nav: List[Tuple[str, float]] = []  # (date, net_asset_value)
    scan_results_cache = {}                  # date -> decisions (buy list)

    # 统计
    total_trades = 0
    winning_trades = 0
    losing_trades = 0
    total_pnl = 0.0

    print()
    print("开始逐日回测...")
    print("-" * 70)

    for day_idx, date_str in enumerate(trading_days):
        target_date = pd.Timestamp(date_str)

        # ── 更新持仓信息 ──
        for sc, pos in list(positions.items()):
            pos.hold_days += 1
            current_price = get_price_on_date(stock_data, sc, target_date)
            if current_price is not None and current_price > pos.highest_price:
                pos.highest_price = current_price

        # ── 检查止盈止损 + 熵状态退出 ──
        stocks_to_sell = []
        for sc, pos in list(positions.items()):
            current_price = get_price_on_date(stock_data, sc, target_date)
            if current_price is None:
                continue

            ret = (current_price - pos.entry_price) / pos.entry_price
            trail_from_high = (pos.highest_price - current_price) / pos.highest_price if pos.highest_price > 0 else 0

            sell_reason = None

            # 1. 硬性止损 + 移动止损（安全网）
            if ret <= -STOP_LOSS:
                sell_reason = "止损"
            elif trail_from_high >= TRAILING_STOP and ret > TRAILING_ACTIVATE:
                sell_reason = f"移动止损(最高+{((pos.highest_price-pos.entry_price)/pos.entry_price*100):.1f}%,回撤{trail_from_high*100:.1f}%)"
            # 2. 最长持有天数兜底
            elif pos.hold_days >= HOLD_DAYS_MAX:
                sell_reason = "超时"
            # 3. 熵驱动退出（核心逻辑：有序→无序时卖出）
            elif pos.hold_days >= HOLD_DAYS_MIN:
                should_exit, reason, entropy_details = evaluate_entropy_exit(
                    stock_data, sc, target_date, system, pos
                )
                if should_exit:
                    sell_reason = reason

            if sell_reason:
                stocks_to_sell.append((sc, current_price, sell_reason))

        # 执行卖出
        for sc, sell_price, reason in stocks_to_sell:
            pos = positions[sc]
            sell_amount = sell_price * pos.shares
            fee = calculate_sell_fee(sell_amount)
            pnl = sell_amount - fee - pos.cost

            trade = Trade(
                date=date_str,
                stock_code=sc,
                action=f'sell_{reason}',
                price=sell_price,
                shares=pos.shares,
                amount=sell_amount,
                fee=fee,
                pnl=pnl,
            )
            trades.append(trade)

            cash += sell_amount - fee
            total_pnl += pnl
            total_trades += 1
            if pnl > 0:
                winning_trades += 1
            else:
                losing_trades += 1

            del positions[sc]

        # ── 定期扫描（每 SCAN_INTERVAL 日） ──
        if day_idx % SCAN_INTERVAL == 0:
            t_scan = time.time()

            # 构建截止到当日的数据
            prices_df, volumes_df, returns_df = build_truncated_dataframes(
                stock_data, target_date, LOOKBACK_DAYS
            )

            if not prices_df.empty:
                # 评估市场门控
                market_gate_output = system.evaluate_market_gate(
                    returns_df, volumes_df, prices_df, industry_map
                )

                gate_open = not market_gate_output.abandonment_flag

                # 逐股评估（只评估不在持仓中的股票）
                buy_candidates = []
                for sc in prices_df.columns:
                    if sc in positions:
                        continue

                    stock_prices = prices_df[sc].dropna()
                    stock_volumes = volumes_df[sc].dropna()

                    if len(stock_prices) < 30:
                        continue

                    stock_returns = np.log(stock_prices / stock_prices.shift(1)).dropna()

                    try:
                        stock_state_output = system.evaluate_stock_state(
                            prices=stock_prices,
                            volumes=stock_volumes,
                            dates=stock_prices.index.to_series(),
                            gate_open=gate_open,
                        )

                        execution_output = system.evaluate_execution_cost(
                            signal_strength=stock_state_output.total_score,
                            market_gate_score=market_gate_output.gate_score,
                            market_state=market_gate_output.state,
                            stock_state=stock_state_output.current_state,
                            current_price=stock_prices.iloc[-1],
                            entry_price=None,
                            current_return=0.0,
                            volatility=stock_returns.std() if len(stock_returns) > 0 else 0.01,
                            noise_cost=market_gate_output.noise_cost,
                        )

                        action, confidence = system.compute_final_action(
                            stock_state_output, market_gate_output, execution_output
                        )

                        if action == 'buy':
                            buy_candidates.append({
                                'stock_code': sc,
                                'confidence': confidence,
                                'total_score': stock_state_output.total_score,
                                'entry_mode': execution_output.entry_mode,
                                'position_size': execution_output.position_size,
                                'price': stock_prices.iloc[-1],
                                # 记录买入时的熵状态（用于对比退出时的变化）
                                'entry_entropy_percentile': stock_state_output.entropy_percentile,
                                'entry_perm_entropy': stock_state_output.permutation_entropy,
                                'entry_entropy_gap': stock_state_output.entropy_gap,
                            })
                    except Exception:
                        continue

                # 按置信度排序，取 top N
                buy_candidates.sort(key=lambda x: x['confidence'], reverse=True)
                scan_results_cache[date_str] = buy_candidates

            scan_time = time.time() - t_scan

            # 输出扫描结果
            n_buy = len(scan_results_cache.get(date_str, []))
            gate_state = market_gate_output.state if not prices_df.empty else 'N/A'
            if day_idx % (SCAN_INTERVAL * 4) == 0 or n_buy > 0:
                print(
                    f"[{date_str}] 扫描完成 | 市场状态: {gate_state} | "
                    f"买入候选: {n_buy} | 持仓数: {len(positions)} | "
                    f"耗时: {scan_time:.1f}s"
                )

        # ── 执行买入 ──
        buy_list = scan_results_cache.get(date_str, [])
        available_slots = MAX_POSITIONS - len(positions)

        if available_slots > 0 and buy_list:
            for candidate in buy_list[:available_slots]:
                sc = candidate['stock_code']
                if sc in positions:
                    continue

                price = get_price_on_date(stock_data, sc, target_date)
                if price is None:
                    continue

                # 计算买入金额 — 等权分配可用资金到可用仓位槽
                slots_available = MAX_POSITIONS - len(positions)
                if slots_available <= 0:
                    break
                target_per_slot = cash * 0.90 / slots_available
                # 根据信号强度微调
                if candidate['confidence'] >= 0.7:
                    size_mult = 1.0
                elif candidate['confidence'] >= 0.5:
                    size_mult = 0.7
                else:
                    size_mult = 0.5

                buy_amount = min(
                    target_per_slot * size_mult,
                    cash * 0.90,
                )

                if buy_amount < 10000:  # 最小买入 1 万
                    continue

                fee = calculate_buy_fee(buy_amount)
                shares = int(buy_amount / price / 100) * 100  # A股100股整数
                if shares <= 0:
                    continue

                actual_amount = shares * price
                actual_fee = calculate_buy_fee(actual_amount)

                if actual_amount + actual_fee > cash:
                    continue

                pos = Position(
                    stock_code=sc,
                    entry_date=date_str,
                    entry_price=price,
                    shares=shares,
                    cost=actual_amount + actual_fee,
                    highest_price=price,
                    hold_days=0,
                    entry_mode=candidate['entry_mode'],
                    entry_entropy_percentile=candidate.get('entry_entropy_percentile', 0.0),
                    entry_perm_entropy=candidate.get('entry_perm_entropy', 0.0),
                    entry_entropy_gap=candidate.get('entry_entropy_gap', 0.0),
                )
                positions[sc] = pos
                cash -= (actual_amount + actual_fee)

                trade = Trade(
                    date=date_str,
                    stock_code=sc,
                    action='buy',
                    price=price,
                    shares=shares,
                    amount=actual_amount,
                    fee=actual_fee,
                )
                trades.append(trade)

        # ── 计算日终资产净值 ──
        portfolio_value = cash
        for sc, pos in positions.items():
            current_price = get_price_on_date(stock_data, sc, target_date)
            if current_price is not None:
                portfolio_value += current_price * pos.shares
            else:
                portfolio_value += pos.entry_price * pos.shares

        daily_nav.append((date_str, portfolio_value))

        # 每月打印一次净值
        if day_idx > 0 and (
            date_str[:6] != trading_days[day_idx - 1][:6]
            or day_idx == len(trading_days) - 1
        ):
            nav_ret = (portfolio_value / INITIAL_CAPITAL - 1) * 100
            print(
                f"  [{date_str}] 净值: {portfolio_value:>12,.0f} | "
                f"累计收益: {nav_ret:>+7.2f}% | "
                f"持仓数: {len(positions)} | 现金: {cash:>12,.0f}"
            )

    # ── 回测结束，强制清仓 ──
    final_date = trading_days[-1]
    final_date_ts = pd.Timestamp(final_date)
    for sc, pos in list(positions.items()):
        current_price = get_price_on_date(stock_data, sc, final_date_ts)
        if current_price is None:
            current_price = pos.entry_price
        sell_amount = current_price * pos.shares
        fee = calculate_sell_fee(sell_amount)
        pnl = sell_amount - fee - pos.cost

        trades.append(Trade(
            date=final_date,
            stock_code=sc,
            action='sell_final',
            price=current_price,
            shares=pos.shares,
            amount=sell_amount,
            fee=fee,
            pnl=pnl,
        ))
        cash += sell_amount - fee
        total_pnl += pnl
        total_trades += 1
        if pnl > 0:
            winning_trades += 1
        else:
            losing_trades += 1

    positions.clear()

    # ── 计算最终结果 ──
    final_nav = cash
    total_return = (final_nav / INITIAL_CAPITAL - 1) * 100
    total_fees = sum(t.fee for t in trades)

    # 计算最大回撤
    nav_series = pd.Series(
        [nav for _, nav in daily_nav],
        index=[d for d, _ in daily_nav],
    )
    peak = nav_series.expanding().max()
    drawdown = (nav_series - peak) / peak
    max_drawdown = drawdown.min() * 100

    # 月度收益
    nav_df = pd.DataFrame(daily_nav, columns=['date', 'nav'])
    nav_df['month'] = nav_df['date'].str[:6]
    monthly = nav_df.groupby('month').last()
    monthly['return'] = monthly['nav'].pct_change() * 100
    # 修正首月
    monthly.iloc[0, monthly.columns.get_loc('return')] = (
        (monthly.iloc[0]['nav'] / INITIAL_CAPITAL - 1) * 100
    )

    # ── 打印报告 ──
    print()
    print("=" * 70)
    print("回测结果汇总")
    print("=" * 70)
    print(f"回测区间:        {BACKTEST_START} ~ {BACKTEST_END}")
    print(f"初始资金:        {INITIAL_CAPITAL:>15,.0f}")
    print(f"期末资金:        {final_nav:>15,.0f}")
    print(f"总盈亏:          {total_pnl:>+15,.0f}")
    print(f"总收益率:        {total_return:>+14.2f}%")
    print(f"最大回撤:        {max_drawdown:>14.2f}%")
    print(f"总交易费用:      {total_fees:>15,.0f}")
    print(f"交易次数(卖出):  {total_trades:>15}")
    print(f"盈利交易:        {winning_trades:>15}")
    print(f"亏损交易:        {losing_trades:>15}")
    if total_trades > 0:
        win_rate = winning_trades / total_trades * 100
        print(f"胜率:            {win_rate:>14.1f}%")
        avg_win = sum(t.pnl for t in trades if t.pnl > 0 and t.action != 'buy') / max(winning_trades, 1)
        avg_loss = sum(t.pnl for t in trades if t.pnl <= 0 and t.action != 'buy') / max(losing_trades, 1)
        print(f"平均盈利:        {avg_win:>+15,.0f}")
        print(f"平均亏损:        {avg_loss:>+15,.0f}")
        if avg_loss != 0:
            profit_factor = abs(avg_win / avg_loss)
            print(f"盈亏比:          {profit_factor:>14.2f}")

    # 年化收益率（假设243交易日）
    annual_return = total_return  # 已经是全年
    # 夏普比率
    nav_returns = nav_series.pct_change().dropna()
    if len(nav_returns) > 0 and nav_returns.std() > 0:
        sharpe = nav_returns.mean() / nav_returns.std() * np.sqrt(243)
        print(f"夏普比率:        {sharpe:>14.2f}")

    print()
    print("月度收益:")
    print("-" * 30)
    for idx, row in monthly.iterrows():
        print(f"  {idx}: {row['return']:>+7.2f}%")

    # 保存结果
    result_dir = os.path.join(PROJECT_ROOT, 'results', 'four_layer_backtest_2025_v4_pure_entropy')
    os.makedirs(result_dir, exist_ok=True)

    # 保存净值曲线
    nav_df.to_csv(os.path.join(result_dir, 'daily_nav.csv'), index=False)

    # 保存交易记录
    trades_data = [{
        'date': t.date,
        'stock_code': t.stock_code,
        'action': t.action,
        'price': t.price,
        'shares': t.shares,
        'amount': t.amount,
        'fee': t.fee,
        'pnl': t.pnl,
    } for t in trades]
    pd.DataFrame(trades_data).to_csv(os.path.join(result_dir, 'trades.csv'), index=False)

    # 保存汇总
    summary = {
        'backtest_period': f'{BACKTEST_START} ~ {BACKTEST_END}',
        'initial_capital': INITIAL_CAPITAL,
        'final_capital': final_nav,
        'total_pnl': total_pnl,
        'total_return_pct': total_return,
        'max_drawdown_pct': max_drawdown,
        'total_fees': total_fees,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': winning_trades / max(total_trades, 1) * 100,
        'scan_interval': SCAN_INTERVAL,
        'max_positions': MAX_POSITIONS,
        'max_stocks_scan': MAX_STOCKS_SCAN,
    }
    with open(os.path.join(result_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── 个股交易明细 ──
    print()
    print("=" * 110)
    print("个股交易明细")
    print("=" * 110)
    print(f"{'股票代码':<12} {'买入日':>10} {'买价':>8} {'卖出日':>10} {'卖价':>8} "
          f"{'持仓天数':>6} {'收益率':>8} {'盈亏':>10} {'退出原因'}")
    print("-" * 110)

    trades_df = pd.DataFrame(trades_data)
    buy_df = trades_df[trades_df['action'] == 'buy'].copy()
    sell_df = trades_df[trades_df['action'].str.startswith('sell')].copy()

    # 按股票配对买卖
    pair_records = []
    for sc in sell_df['stock_code'].unique():
        sc_buys = buy_df[buy_df['stock_code'] == sc].sort_values('date').reset_index(drop=True)
        sc_sells = sell_df[sell_df['stock_code'] == sc].sort_values('date').reset_index(drop=True)
        for i in range(min(len(sc_buys), len(sc_sells))):
            b = sc_buys.iloc[i]
            s = sc_sells.iloc[i]
            ret_pct = (s['price'] - b['price']) / b['price'] * 100
            days = (pd.Timestamp(str(s['date'])) - pd.Timestamp(str(b['date']))).days
            # 简化退出原因
            reason = s['action'].replace('sell_', '')
            if len(reason) > 40:
                reason = reason[:40] + '...'
            pair_records.append({
                'stock': sc, 'buy_date': str(b['date']), 'buy_price': b['price'],
                'sell_date': str(s['date']), 'sell_price': s['price'],
                'days': days, 'ret_pct': ret_pct, 'pnl': s['pnl'], 'reason': reason,
            })

    # 按买入日期排序
    pair_records.sort(key=lambda x: x['buy_date'])
    for r in pair_records:
        pnl_str = f"{r['pnl']:>+10,.0f}"
        ret_str = f"{r['ret_pct']:>+7.2f}%"
        print(f"{r['stock']:<12} {r['buy_date']:>10} {r['buy_price']:>8.2f} "
              f"{r['sell_date']:>10} {r['sell_price']:>8.2f} "
              f"{r['days']:>6} {ret_str} {pnl_str} {r['reason']}")

    print("-" * 110)
    print(f"合计: {len(pair_records)} 笔交易")

    # 按退出原因汇总
    print()
    print("=" * 70)
    print("按退出原因汇总")
    print("=" * 70)
    print(f"{'退出原因':<30} {'笔数':>5} {'胜率':>7} {'平均盈亏':>12} {'总盈亏':>12}")
    print("-" * 70)

    reason_groups = {}
    for r in pair_records:
        # 归类简化
        raw = r['reason']
        if raw.startswith('止损'):
            key = '止损'
        elif raw.startswith('移动止损'):
            key = '移动止损'
        elif raw.startswith('超时'):
            key = '超时'
        elif raw.startswith('final'):
            key = '年末清仓'
        elif '熵退出' in raw:
            key = '熵退出'
        else:
            key = raw
        if key not in reason_groups:
            reason_groups[key] = []
        reason_groups[key].append(r)

    for key in ['熵退出', '移动止损', '止损', '超时', '年末清仓']:
        if key not in reason_groups:
            continue
        group = reason_groups[key]
        n = len(group)
        wins = sum(1 for r in group if r['pnl'] > 0)
        wr = wins / n * 100
        avg_pnl = sum(r['pnl'] for r in group) / n
        total_pnl_group = sum(r['pnl'] for r in group)
        print(f"{key:<30} {n:>5} {wr:>6.1f}% {avg_pnl:>+12,.0f} {total_pnl_group:>+12,.0f}")

    # 其他未归类的
    for key, group in reason_groups.items():
        if key in ['熵退出', '移动止损', '止损', '超时', '年末清仓']:
            continue
        n = len(group)
        wins = sum(1 for r in group if r['pnl'] > 0)
        wr = wins / n * 100
        avg_pnl = sum(r['pnl'] for r in group) / n
        total_pnl_group = sum(r['pnl'] for r in group)
        print(f"{key:<30} {n:>5} {wr:>6.1f}% {avg_pnl:>+12,.0f} {total_pnl_group:>+12,.0f}")

    # 保存配对交易表
    pd.DataFrame(pair_records).to_csv(os.path.join(result_dir, 'trade_pairs.csv'), index=False)

    print()
    print(f"结果已保存到: {result_dir}")

    return summary


if __name__ == '__main__':
    run_backtest()
