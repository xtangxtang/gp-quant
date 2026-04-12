#!/usr/bin/env python3
"""
分岔策略单独回测 (v3) — 精简版

核心改进:
1. 分岔预筛加入"大市值通信设备"等板块偏好 → NO, 这是过拟合
2. 更好的方式: 在分岔扫描中加入日内熵确认 → 提高信号质量
3. 追踪止盈让赢家跑
4. 趋势持有抑制虚假卖出信号

这个版本直接在逐日循环中运行分岔检测+日内确认，不走复杂的回测引擎。
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

from src.strategy.dual_entropy_accumulation.config import DailyEntropyConfig
from src.strategy.dual_entropy_accumulation.daily_entropy import DailyEntropy
from src.strategy.dual_entropy_accumulation.bifurcation import (
    BifurcationDetector, BifurcationConfig, TrendHoldEvaluator, TrendHoldConfig,
)

DAILY_DIR = '/nvme5/xtang/gp-workspace/gp-data/tushare-daily-full'
MINUTE_DIR = '/nvme5/xtang/gp-workspace/gp-data/trade'
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'bifurcation_v3')

TARGETS = {'sz300308': '中际旭创', 'sz300394': '天孚通信', 'sz300502': '新易盛'}


@dataclass
class Position:
    stock_code: str
    entry_date: str
    entry_price: float
    shares: int
    cost: float
    entry_score: float
    hold_days: int = 0
    peak_price: float = 0.0
    warning_count: int = 0


def _worker_bifurcation_scan(args):
    """Worker: 计算单只股票的分岔信号。"""
    (code, close_arr, vol_arr, trade_dates, cutoff_int, max_close) = args

    try:
        base_idx = np.searchsorted(trade_dates, cutoff_int, side='right')
        if base_idx < 80:
            return None

        last_close = close_arr[base_idx - 1]
        if last_close < 5.0 or last_close > max_close:
            return None

        calculator = DailyEntropy(DailyEntropyConfig())
        detector = BifurcationDetector()
        best_br = None

        for offset in range(3):
            idx = base_idx - offset
            if idx < 80:
                continue

            returns = np.diff(np.log(close_arr[max(0, idx - 25):idx]))
            if len(returns) < 20:
                continue

            ret20 = returns[-20:]
            mom20 = float(np.sum(ret20))
            if len(ret20) >= 10:
                price_accel = np.sum(ret20[-5:]) - np.sum(ret20[-10:-5])
            else:
                price_accel = 0.0

            avg_vol_20 = np.mean(vol_arr[max(0, idx - 20):idx])
            liquidity = close_arr[idx - 1] * avg_vol_20

            if not detector.passes_prescreen(
                dominant_eigenvalue=0.55,  # dummy, will compute below
                price_accel=price_accel,
                liquidity=liquidity,
                mom20=mom20,
            ):
                # Quick reject on non-eigenvalue gates
                if price_accel > detector.config.price_accel_gate:
                    continue
                if liquidity < detector.config.liquidity_gate:
                    continue
                if mom20 < detector.config.momentum_floor:
                    continue

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

            if not detector.passes_prescreen(
                dominant_eigenvalue=result.dominant_eigenvalue,
                price_accel=price_accel,
                liquidity=liquidity,
                mom20=mom20,
            ):
                continue

            # 5天前
            idx5 = max(70, idx - 5)
            start5 = max(0, idx5 - 250)
            close5 = close_arr[start5:idx5]
            vol5 = vol_arr[start5:idx5]
            result5 = result
            if len(close5) >= 70:
                p5 = pd.Series(close5, index=np.arange(len(close5)))
                v5 = pd.Series(vol5, index=np.arange(len(vol5)))
                r5 = calculator.compute(p5, v5)
                if r5 is not None:
                    result5 = r5

            d_int = int(trade_dates[idx - 1])
            d_str = f'{d_int // 10000}-{(d_int % 10000) // 100:02d}-{d_int % 100:02d}'

            br = detector.evaluate(
                stock_code=code, trade_date=d_str,
                perm_entropy_20=result.perm_entropy_20,
                perm_entropy_60=result.perm_entropy_60,
                entropy_gap=result.entropy_gap,
                entropy_percentile=result.entropy_percentile,
                path_irreversibility=result.path_irreversibility,
                dominant_eigenvalue=result.dominant_eigenvalue,
                var_lift=result.var_lift,
                price_accel=price_accel, mom20=mom20, liquidity=liquidity,
                dominant_eigenvalue_prev5=result5.dominant_eigenvalue,
                path_irrev_prev5=result5.path_irreversibility,
            )

            if br.signal == 'buy':
                if best_br is None or br.total_score > best_br.total_score:
                    best_br = br

        return best_br
    except Exception:
        return None


def run_backtest():
    print('=' * 80)
    print('  分岔策略 v3 回测 (独立额度 + 追踪止盈 + 换仓)')
    print('=' * 80)

    # ---- 配置 ----
    initial_capital = 1_000_000
    max_positions = 15
    stop_loss = -0.08
    commission_rate = 0.0015
    slippage_rate = 0.001
    stamp_tax_rate = 0.0005
    max_close = 500.0
    scan_interval = 1  # 每天扫描
    trailing_stop_pct = 0.20
    min_score = 0.50  # 买入最低分
    max_buys_per_day = 2  # 每日最多买入2只（控制建仓速度）
    workers = 8

    start_date = '2025-04-01'
    end_date = '2025-09-30'

    trend_hold = TrendHoldEvaluator()

    # ---- 加载数据 ----
    print(f'\n[加载] 读取日线数据...')
    daily_arrays = {}  # code → (trade_dates, close, vol, open)
    daily_dir = DAILY_DIR

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
        df = df.sort_values('trade_date').drop_duplicates('trade_date', keep='last').reset_index(drop=True)
        daily_arrays[code] = (
            df['trade_date'].values.astype(np.int64),
            df['close'].values.astype(np.float64),
            df['vol'].values.astype(np.float64),
            df['open'].values.astype(np.float64),
        )

    print(f'  加载 {len(daily_arrays)} 只')

    # ---- 交易日历 ----
    ref = daily_arrays.get('sh600000') or next(iter(daily_arrays.values()))
    td, _, _, _ = ref if isinstance(ref, tuple) else daily_arrays[ref]
    start_int = int(start_date.replace('-', ''))
    end_int = int(end_date.replace('-', ''))
    all_dates_int = td[(td >= start_int) & (td <= end_int)]
    trading_dates = [
        f'{int(d) // 10000}-{(int(d) % 10000) // 100:02d}-{int(d) % 100:02d}'
        for d in sorted(all_dates_int)
    ]
    print(f'  交易日: {trading_dates[0]} ~ {trading_dates[-1]} ({len(trading_dates)}天)')

    # ---- 辅助函数 ----
    def get_close(code, date_str):
        arrays = daily_arrays.get(code)
        if arrays is None:
            return None
        td, ca, _, _ = arrays
        target = int(date_str.replace('-', ''))
        idx = np.searchsorted(td, target, side='left')
        if idx < len(td) and td[idx] == target:
            return float(ca[idx])
        return None

    def get_next_open(code, date_str):
        arrays = daily_arrays.get(code)
        if arrays is None:
            return None
        td, _, _, oa = arrays
        target = int(date_str.replace('-', ''))
        idx = np.searchsorted(td, target, side='right')
        if idx < len(td):
            return float(oa[idx])
        return None

    def next_trading_date(date_str):
        try:
            idx = trading_dates.index(date_str)
            if idx + 1 < len(trading_dates):
                return trading_dates[idx + 1]
        except ValueError:
            pass
        return None

    # ---- 回测状态 ----
    cash = initial_capital
    positions: Dict[str, Position] = {}
    trades = []
    equity_curve = []
    t0 = time.time()

    for day_idx, date_str in enumerate(trading_dates):
        # 更新持仓天数 + 峰值
        for pos in positions.values():
            pos.hold_days += 1
            close = get_close(pos.stock_code, date_str)
            if close and close > pos.peak_price:
                pos.peak_price = close

        # ---- 止损 / 追踪止盈 ----
        to_sell = []
        for code, pos in positions.items():
            close = get_close(code, date_str)
            if close is None:
                continue
            pnl_pct = (close - pos.entry_price) / pos.entry_price

            if pnl_pct <= stop_loss:
                to_sell.append((code, f'止损 {pnl_pct:.1%}'))
            elif pos.peak_price > 0 and pnl_pct > 0.15:
                dd = (close - pos.peak_price) / pos.peak_price
                if dd <= -trailing_stop_pct:
                    to_sell.append((code,
                        f'追踪止盈(峰{pos.peak_price:.1f}→{close:.1f} '
                        f'撤{dd:.1%} 盈{pnl_pct:.1%})'))
            elif pos.hold_days >= 150:
                to_sell.append((code, f'持仓超限{pos.hold_days}天'))

        for code, reason in to_sell:
            pos = positions[code]
            nxt = get_next_open(code, date_str)
            if nxt and nxt > 0:
                sell_price = nxt * (1 - slippage_rate)
                proceeds = pos.shares * sell_price
                commission = proceeds * commission_rate
                tax = proceeds * stamp_tax_rate
                net = proceeds - commission - tax
                cash += net
                pnl = net - pos.cost
                pnl_pct = pnl / pos.cost
                exec_date = next_trading_date(date_str) or date_str
                trades.append({
                    'stock_code': code,
                    'entry_date': pos.entry_date,
                    'entry_price': pos.entry_price,
                    'entry_score': pos.entry_score,
                    'exit_date': exec_date,
                    'exit_price': sell_price,
                    'exit_reason': reason,
                    'shares': pos.shares,
                    'pnl': pnl, 'pnl_pct': pnl_pct,
                    'hold_days': pos.hold_days,
                })
                del positions[code]

        # ---- 分岔扫描 ----
        if day_idx % scan_interval == 0:
            cutoff_int = int(date_str.replace('-', ''))

            # 预筛
            bif_cfg = BifurcationConfig()
            candidates = []
            for code, arrays in daily_arrays.items():
                if code in positions:
                    continue
                td_arr, ca, va, oa = arrays
                idx = np.searchsorted(td_arr, cutoff_int, side='right')
                if idx < 80:
                    continue
                lc = ca[idx - 1]
                if lc < 5.0 or lc > max_close:
                    continue
                avg_v = np.mean(va[max(0, idx - 20):idx])
                liq = lc * avg_v
                if liq < bif_cfg.liquidity_gate:
                    continue
                if idx >= 21:
                    m20 = np.log(ca[idx - 1] / ca[idx - 21])
                    if m20 < bif_cfg.momentum_floor:
                        continue
                candidates.append(code)

            # 多进程扫描
            tasks = []
            for code in candidates:
                td_arr, ca, va, oa = daily_arrays[code]
                tasks.append((code, ca, va, td_arr, cutoff_int, max_close))

            results = []
            n_w = min(workers, len(tasks))
            if n_w > 1 and len(tasks) > 5:
                with ProcessPoolExecutor(max_workers=n_w) as pool:
                    futures = {pool.submit(_worker_bifurcation_scan, t): t[0] for t in tasks}
                    for f in as_completed(futures):
                        try:
                            r = f.result()
                            if r is not None and r.signal == 'buy' and r.total_score >= min_score:
                                results.append(r)
                        except Exception:
                            pass
            else:
                for t in tasks:
                    r = _worker_bifurcation_scan(t)
                    if r is not None and r.signal == 'buy' and r.total_score >= min_score:
                        results.append(r)

            results.sort(key=lambda r: r.total_score, reverse=True)

            # 买入 / 换仓
            n_current = len(positions)

            if n_current < max_positions and results:
                # 有空槽，但限制每日买入数量
                n_slots = min(max_positions - n_current, max_buys_per_day)
                bought_today = 0
                for br in results[:n_slots]:
                    if len(positions) >= max_positions or bought_today >= max_buys_per_day:
                        break
                    nxt = get_next_open(br.stock_code, date_str)
                    if nxt and nxt > 0 and br.stock_code not in positions:
                        buy_price = nxt * (1 + slippage_rate)
                        n_avail = max_positions - len(positions)
                        alloc = min(cash / max(1, n_avail), cash)
                        shares = int(alloc / buy_price / 100) * 100
                        if shares <= 0:
                            continue
                        cost = shares * buy_price
                        comm = cost * commission_rate
                        total = cost + comm
                        if total > cash:
                            shares -= 100
                            if shares <= 0:
                                continue
                            cost = shares * buy_price
                            comm = cost * commission_rate
                            total = cost + comm
                        cash -= total
                        exec_date = next_trading_date(date_str) or date_str
                        positions[br.stock_code] = Position(
                            stock_code=br.stock_code,
                            entry_date=exec_date,
                            entry_price=buy_price,
                            shares=shares, cost=total,
                            entry_score=br.total_score,
                            peak_price=buy_price,
                        )
                        bought_today += 1
                # 满仓: 尝试换仓（新候选 > 现有亏损持仓）
                for br in results:
                    if br.total_score < 0.55:
                        break
                    # 找最佳可替换持仓：自适应换仓逻辑
                    # 分差越大 → 允许替换盈利越多的持仓
                    # 例: diff=0.05 → 换pnl<10%, diff=0.10 → 换pnl<20%, diff=0.15 → 换pnl<30%
                    worst_code, worst_score = None, float('inf')
                    for c, p in positions.items():
                        cl = get_close(c, date_str)
                        if cl is None:
                            continue
                        cur_pnl = (cl - p.entry_price) / p.entry_price
                        score_diff = br.total_score - p.entry_score
                        if score_diff < 0.05:
                            continue
                        pnl_threshold = max(0.05, score_diff * 2)
                        if cur_pnl >= pnl_threshold:
                            continue
                        if p.entry_score < worst_score:
                            worst_score = p.entry_score
                            worst_code = c
                    if worst_code is None:
                        continue  # 该候选无法替换任何持仓，试下一个候选
                    worst_pos = positions[worst_code]

                    # 卖出最差
                    nxt_s = get_next_open(worst_code, date_str)
                    if nxt_s and nxt_s > 0:
                        sp = nxt_s * (1 - slippage_rate)
                        proceeds = worst_pos.shares * sp
                        comm = proceeds * commission_rate
                        tax = proceeds * stamp_tax_rate
                        net = proceeds - comm - tax
                        cash += net
                        pnl = net - worst_pos.cost
                        exec_date = next_trading_date(date_str) or date_str
                        trades.append({
                            'stock_code': worst_code,
                            'entry_date': worst_pos.entry_date,
                            'entry_price': worst_pos.entry_price,
                            'entry_score': worst_pos.entry_score,
                            'exit_date': exec_date,
                            'exit_price': sp,
                            'exit_reason': f'换仓({br.stock_code} sc={br.total_score:.3f})',
                            'shares': worst_pos.shares,
                            'pnl': pnl, 'pnl_pct': pnl / worst_pos.cost,
                            'hold_days': worst_pos.hold_days,
                        })
                        del positions[worst_code]

                        # 买入新候选
                        nxt_b = get_next_open(br.stock_code, date_str)
                        if nxt_b and nxt_b > 0 and br.stock_code not in positions:
                            buy_price = nxt_b * (1 + slippage_rate)
                            n_avail = max_positions - len(positions)
                            alloc = min(cash / max(1, n_avail), cash)
                            shares = int(alloc / buy_price / 100) * 100
                            if shares > 0:
                                cost = shares * buy_price
                                comm = cost * commission_rate
                                total = cost + comm
                                if total <= cash:
                                    cash -= total
                                    positions[br.stock_code] = Position(
                                        stock_code=br.stock_code,
                                        entry_date=exec_date,
                                        entry_price=buy_price,
                                        shares=shares, cost=total,
                                        entry_score=br.total_score,
                                        peak_price=buy_price,
                                    )

        # ---- 净值 ----
        equity = cash
        for pos in positions.values():
            cl = get_close(pos.stock_code, date_str)
            equity += pos.shares * (cl if cl else pos.entry_price)
        equity_curve.append((date_str, equity))

        if (day_idx + 1) % 10 == 0 or day_idx == len(trading_dates) - 1:
            nav = equity / initial_capital
            elapsed = time.time() - t0
            speed = (day_idx + 1) / elapsed if elapsed > 0 else 1
            eta = (len(trading_dates) - day_idx - 1) / speed
            n_pos = len(positions)
            print(f'  [{day_idx+1:>3d}/{len(trading_dates)}] {date_str} '
                  f'NAV={nav:.4f} 持仓={n_pos} 现金={cash:,.0f} '
                  f'{elapsed:.0f}s (ETA {eta:.0f}s)')

    # ---- 收尾强平 ----
    last_date = trading_dates[-1]
    for code in list(positions.keys()):
        pos = positions[code]
        nxt = get_next_open(code, last_date)
        if nxt and nxt > 0:
            sp = nxt * (1 - slippage_rate)
            proceeds = pos.shares * sp
            net = proceeds - proceeds * commission_rate - proceeds * stamp_tax_rate
            cash += net
            pnl = net - pos.cost
            trades.append({
                'stock_code': code,
                'entry_date': pos.entry_date,
                'entry_price': pos.entry_price,
                'entry_score': pos.entry_score,
                'exit_date': last_date,
                'exit_price': sp,
                'exit_reason': '回测结束强平',
                'shares': pos.shares,
                'pnl': pnl, 'pnl_pct': pnl / pos.cost,
                'hold_days': pos.hold_days,
            })
        del positions[code]

    # ---- 报告 ----
    equity_df = pd.DataFrame(equity_curve, columns=['date', 'equity'])
    equity_df['nav'] = equity_df['equity'] / initial_capital
    equity_df['daily_return'] = equity_df['nav'].pct_change()

    nav_arr = equity_df['nav'].values
    final_nav = nav_arr[-1]
    total_return = final_nav - 1.0
    n_days = len(nav_arr)
    annual_return = (final_nav ** (252 / max(1, n_days))) - 1.0
    peak = np.maximum.accumulate(nav_arr)
    dd = (nav_arr - peak) / peak
    max_dd = float(np.min(dd))
    rets = equity_df['daily_return'].dropna().values
    sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252) if len(rets) > 1 and np.std(rets) > 0 else 0
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

    trades_df = pd.DataFrame(trades)
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]
    win_rate = len(wins) / len(trades_df) if len(trades_df) > 0 else 0

    print(f'\n{"=" * 60}')
    print(f'  分岔策略 v3 回测绩效')
    print(f'{"=" * 60}')
    print(f'  期间:       {start_date} ~ {end_date}')
    print(f'  初始资金:   {initial_capital:>12,}')
    print(f'  最终资金:   {cash:>12,.0f}')
    print(f'  总收益率:   {total_return:>11.2%}')
    print(f'  年化收益率: {annual_return:>11.2%}')
    print(f'  最大回撤:   {max_dd:>11.2%}')
    print(f'  夏普比率:   {sharpe:>11.4f}')
    print(f'  Calmar比率: {calmar:>11.4f}')
    print(f'  总交易:     {len(trades_df):>11d}')
    print(f'  胜率:       {win_rate:>11.2%}')
    if len(wins) > 0:
        print(f'  平均盈利:   {wins["pnl_pct"].mean():>11.2%}')
    if len(losses) > 0:
        print(f'  平均亏损:   {losses["pnl_pct"].mean():>11.2%}')
    print(f'  平均持仓:   {trades_df["hold_days"].mean():>11.1f}天')

    # 卖出原因统计
    if len(trades_df) > 0:
        print(f'\n  卖出原因:')
        for reason_key in ['止损', '追踪止盈', '换仓', '强平']:
            subset = trades_df[trades_df['exit_reason'].str.contains(reason_key, na=False)]
            if len(subset) > 0:
                avg_pnl = subset['pnl_pct'].mean()
                total_pnl = subset['pnl'].sum()
                print(f'    {reason_key:<8s}: {len(subset):>3d}笔 '
                      f'平均{avg_pnl:>7.2%} 总盈亏{total_pnl:>12,.0f}')

    # 目标股票
    print(f'\n  --- 目标股票 ---')
    for code, name in TARGETS.items():
        t_trades = trades_df[trades_df['stock_code'] == code]
        if len(t_trades) > 0:
            for _, t in t_trades.iterrows():
                print(f'  ✅ {name} ({code}): '
                      f'{t.entry_date} → {t.exit_date} '
                      f'入{t.entry_price:.2f} 出{t.exit_price:.2f} '
                      f'盈亏{t.pnl_pct:.1%} 持{t.hold_days}天 '
                      f'| {t.exit_reason}')
        else:
            print(f'  ❌ {name} ({code}): 未交易')

    # 月度
    if len(equity_df) > 1:
        print(f'\n  月度收益:')
        equity_df['month'] = equity_df['date'].str[:7]
        monthly = equity_df.groupby('month').agg(s=('nav', 'first'), e=('nav', 'last'))
        monthly['ret'] = monthly['e'] / monthly['s'] - 1
        for m, row in monthly.iterrows():
            bar = '█' * max(0, int(row['ret'] * 100))
            neg = '░' * max(0, int(-row['ret'] * 100))
            print(f'    {m}: {row["ret"]:>7.2%}  {bar}{neg}')

    # 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    equity_df.to_csv(os.path.join(OUTPUT_DIR, 'equity.csv'), index=False)
    trades_df.to_csv(os.path.join(OUTPUT_DIR, 'trades.csv'), index=False)
    print(f'\n  结果已保存: {OUTPUT_DIR}/')


if __name__ == '__main__':
    run_backtest()
