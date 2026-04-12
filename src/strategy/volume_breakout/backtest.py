#!/usr/bin/env python3
"""
量价突破策略 - 回测引擎

与分岔策略 v3 回测结构一致:
  - 逐日循环
  - 多进程扫描
  - 追踪止盈 + 量能衰竭退出
  - 自适应换仓
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

from src.strategy.volume_breakout.config import (
    BreakoutDetectorConfig, BreakoutExitConfig, BreakoutBacktestConfig,
)
from src.strategy.volume_breakout.detector import BreakoutDetector, BreakoutResult

DAILY_DIR = '/nvme5/xtang/gp-workspace/gp-data/tushare-daily-full'
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'results', 'volume_breakout')

TARGETS = {
    'sz300450': '先导智能',
    'sz300748': '利元亨',
    'sh603200': '上海洗霸',
}


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
    entry_vol_5d: float = 0.0   # 入场时5日均量 (用于量能衰竭检测)


# ================================================================
#  Worker 函数 (顶层, 可 pickle)
# ================================================================

def _worker_breakout_scan(args) -> Optional[BreakoutResult]:
    """子进程: 对单只股票做突破检测"""
    stock_code, close_arr, high_arr, low_arr, vol_arr, td_arr, cutoff_int, max_close = args
    try:
        idx = np.searchsorted(td_arr, cutoff_int, side='right')
        if idx < 70:
            return None
        c = close_arr[:idx]
        h = high_arr[:idx]
        l = low_arr[:idx]
        v = vol_arr[:idx]

        last_c = c[-1]
        if last_c > max_close:
            return None

        cfg = BreakoutDetectorConfig()
        det = BreakoutDetector(cfg)

        if not det.passes_prescreen(c, v):
            return None

        return det.evaluate(stock_code, c, h, l, v)
    except Exception:
        return None


# ================================================================
#  辅助函数
# ================================================================

def _load_daily_data(daily_dir: str):
    """加载全市场日线数据到内存"""
    daily_arrays = {}
    for fname in sorted(os.listdir(daily_dir)):
        if not fname.endswith('.csv'):
            continue
        code = fname.replace('.csv', '')
        if not (code.startswith('sh') or code.startswith('sz')):
            continue
        path = os.path.join(daily_dir, fname)
        try:
            df = pd.read_csv(path, usecols=['trade_date', 'open', 'high', 'low', 'close', 'vol'])
        except Exception:
            continue
        for col in ['trade_date', 'open', 'high', 'low', 'close', 'vol']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna().sort_values('trade_date')
        if len(df) < 70:
            continue
        daily_arrays[code] = (
            df['trade_date'].values.astype(np.int64),
            df['close'].values.astype(np.float64),
            df['high'].values.astype(np.float64),
            df['low'].values.astype(np.float64),
            df['vol'].values.astype(np.float64),
            df['open'].values.astype(np.float64),
        )
    return daily_arrays


def _get_trading_dates(daily_arrays: dict, start: str, end: str) -> List[str]:
    """从数据中提取交易日列表"""
    s_int = int(start.replace('-', ''))
    e_int = int(end.replace('-', ''))
    # 用大盘股提取交易日
    for code in ['sh000001', 'sh600000', 'sz000001']:
        if code in daily_arrays:
            td = daily_arrays[code][0]
            mask = (td >= s_int) & (td <= e_int)
            dates = td[mask]
            return [str(d) for d in sorted(dates)]
    # fallback: 用任意一只
    for code, arrs in daily_arrays.items():
        td = arrs[0]
        mask = (td >= s_int) & (td <= e_int)
        if mask.sum() > 50:
            dates = td[mask]
            return [str(d) for d in sorted(dates)]
    return []


def _date_str(d_int) -> str:
    d = int(d_int)
    return f'{d // 10000}-{(d % 10000) // 100:02d}-{d % 100:02d}'


def _next_trading_date(date_str: str, trading_dates: List[str]) -> Optional[str]:
    try:
        idx = trading_dates.index(date_str)
        if idx + 1 < len(trading_dates):
            return trading_dates[idx + 1]
    except ValueError:
        pass
    return None


# ================================================================
#  回测主函数
# ================================================================

def run_backtest():
    bt_cfg = BreakoutBacktestConfig()
    exit_cfg = BreakoutExitConfig()
    det_cfg = BreakoutDetectorConfig()

    print('=' * 80)
    print('  量价突破策略 回测')
    print('=' * 80)

    # ---- 加载数据 ----
    print(f'\n[加载] 读取日线数据...')
    daily_arrays = _load_daily_data(DAILY_DIR)
    print(f'  加载 {len(daily_arrays)} 只')

    trading_dates = _get_trading_dates(daily_arrays, bt_cfg.start_date, bt_cfg.end_date)
    if not trading_dates:
        print('  无交易日')
        return
    print(f'  交易日: {_date_str(trading_dates[0])} ~ {_date_str(trading_dates[-1])} ({len(trading_dates)}天)')

    # ---- 回测状态 ----
    cash = bt_cfg.initial_capital
    positions: Dict[str, Position] = {}
    trades: List[dict] = []
    equity_curve: List[Tuple[str, float]] = []
    t0 = time.time()

    # ---- 辅助函数 ----
    def get_close(code, date_str):
        d_int = int(date_str.replace('-', ''))
        if code not in daily_arrays:
            return None
        td, ca, ha, la, va, oa = daily_arrays[code]
        idx = np.searchsorted(td, d_int, side='right') - 1
        if idx >= 0 and td[idx] == d_int:
            return float(ca[idx])
        return None

    def get_open(code, date_str):
        d_int = int(date_str.replace('-', ''))
        if code not in daily_arrays:
            return None
        td, ca, ha, la, va, oa = daily_arrays[code]
        idx = np.searchsorted(td, d_int, side='right') - 1
        if idx >= 0 and td[idx] == d_int:
            return float(oa[idx])
        return None

    def get_vol_5d(code, date_str):
        d_int = int(date_str.replace('-', ''))
        if code not in daily_arrays:
            return 0.0
        td, ca, ha, la, va, oa = daily_arrays[code]
        idx = np.searchsorted(td, d_int, side='right')
        if idx < 5:
            return 0.0
        return float(np.mean(va[idx - 5:idx]))

    def get_next_open(code, date_str):
        nxt = _next_trading_date(date_str, [_date_str(d) for d in trading_dates])
        if nxt:
            return get_open(code, nxt)
        return None

    fmt_dates = [_date_str(d) for d in trading_dates]

    # ---- 逐日循环 ----
    for day_idx, raw_date in enumerate(trading_dates):
        date_str = _date_str(raw_date)
        date_int = int(raw_date)

        # ---- 持仓更新 (止损/追踪止盈/量能衰竭/超时) ----
        for code in list(positions.keys()):
            pos = positions[code]
            pos.hold_days += 1

            cl = get_close(code, date_str)
            if cl is None:
                continue

            # 更新峰值
            if cl > pos.peak_price:
                pos.peak_price = cl

            cur_pnl = (cl - pos.entry_price) / pos.entry_price
            exit_reason = ''

            # 硬止损
            if cur_pnl <= exit_cfg.stop_loss:
                exit_reason = f'止损 {cur_pnl:.1%}'

            # 追踪止盈
            if not exit_reason and pos.peak_price > 0:
                drawdown = (pos.peak_price - cl) / pos.peak_price
                if drawdown >= exit_cfg.trailing_stop_pct and cur_pnl >= exit_cfg.trailing_min_profit:
                    exit_reason = (f'追踪止盈(峰{pos.peak_price:.1f}→{cl:.1f} '
                                   f'撤{-drawdown:.1%} 盈{cur_pnl:.1%})')

            # 量能衰竭
            if not exit_reason and pos.entry_vol_5d > 0 and cur_pnl >= exit_cfg.vol_exhaustion_min_profit:
                cur_v5 = get_vol_5d(code, date_str)
                if cur_v5 / pos.entry_vol_5d < exit_cfg.vol_exhaustion_ratio:
                    exit_reason = f'量能衰竭(vol比={cur_v5 / pos.entry_vol_5d:.2f} 盈{cur_pnl:.1%})'

            # 超时
            if not exit_reason and pos.hold_days >= exit_cfg.max_hold_days:
                exit_reason = f'持仓超时({pos.hold_days}天)'

            if exit_reason:
                # T+1 卖出
                nxt = _next_trading_date(date_str, fmt_dates)
                sell_price_raw = get_open(code, nxt) if nxt else cl
                if sell_price_raw is None or sell_price_raw <= 0:
                    sell_price_raw = cl
                sp = sell_price_raw * (1 - bt_cfg.slippage_rate)
                proceeds = pos.shares * sp
                comm = proceeds * bt_cfg.commission_rate
                tax = proceeds * bt_cfg.stamp_tax_rate
                net = proceeds - comm - tax
                cash += net
                pnl = net - pos.cost
                trades.append({
                    'stock_code': code,
                    'entry_date': pos.entry_date,
                    'entry_price': pos.entry_price,
                    'entry_score': pos.entry_score,
                    'exit_date': nxt or date_str,
                    'exit_price': sp,
                    'exit_reason': exit_reason,
                    'shares': pos.shares,
                    'pnl': pnl,
                    'pnl_pct': pnl / pos.cost,
                    'hold_days': pos.hold_days,
                })
                del positions[code]

        # ---- 扫描 (按 scan_interval) ----
        if day_idx % bt_cfg.scan_interval != 0:
            pass  # 跳过非扫描日
        else:
            # 构建候选
            tasks = []
            for code, arrs in daily_arrays.items():
                if code in positions:
                    continue
                td, ca, ha, la, va, oa = arrs
                idx = np.searchsorted(td, date_int, side='right')
                if idx < 70:
                    continue
                last_c = ca[idx - 1]
                if last_c < det_cfg.min_close or last_c > det_cfg.max_close:
                    continue
                avg_vol = np.mean(va[max(0, idx - 20):idx])
                if last_c * avg_vol < det_cfg.min_liquidity:
                    continue
                tasks.append((code, ca, ha, la, va, td, date_int, det_cfg.max_close))

            # 多进程扫描
            results: List[BreakoutResult] = []
            n_w = min(bt_cfg.workers, len(tasks))
            if n_w > 1 and len(tasks) > 5:
                with ProcessPoolExecutor(max_workers=n_w) as pool:
                    futures = {pool.submit(_worker_breakout_scan, t): t[0] for t in tasks}
                    for f in as_completed(futures):
                        try:
                            r = f.result()
                            if r is not None and r.signal == 'buy' and r.total_score >= det_cfg.buy_score_min:
                                results.append(r)
                        except Exception:
                            pass
            else:
                for t in tasks:
                    r = _worker_breakout_scan(t)
                    if r is not None and r.signal == 'buy' and r.total_score >= det_cfg.buy_score_min:
                        results.append(r)

            results.sort(key=lambda r: r.total_score, reverse=True)

            # ---- 买入 / 换仓 ----
            n_current = len(positions)

            if n_current < bt_cfg.max_positions and results:
                n_slots = min(bt_cfg.max_positions - n_current, bt_cfg.max_buys_per_day)
                bought = 0
                for br in results[:n_slots]:
                    if len(positions) >= bt_cfg.max_positions or bought >= bt_cfg.max_buys_per_day:
                        break
                    nxt = _next_trading_date(date_str, fmt_dates)
                    if not nxt:
                        continue
                    nxt_open = get_open(br.stock_code, nxt)
                    if not nxt_open or nxt_open <= 0 or br.stock_code in positions:
                        continue
                    buy_price = nxt_open * (1 + bt_cfg.slippage_rate)
                    n_avail = bt_cfg.max_positions - len(positions)
                    alloc = min(cash / max(1, n_avail), cash)
                    shares = int(alloc / buy_price / 100) * 100
                    if shares <= 0:
                        continue
                    cost = shares * buy_price
                    comm = cost * bt_cfg.commission_rate
                    total = cost + comm
                    if total > cash:
                        shares -= 100
                        if shares <= 0:
                            continue
                        cost = shares * buy_price
                        comm = cost * bt_cfg.commission_rate
                        total = cost + comm
                    cash -= total
                    v5 = get_vol_5d(br.stock_code, date_str)
                    positions[br.stock_code] = Position(
                        stock_code=br.stock_code,
                        entry_date=nxt,
                        entry_price=buy_price,
                        shares=shares,
                        cost=total,
                        entry_score=br.total_score,
                        peak_price=buy_price,
                        entry_vol_5d=v5,
                    )
                    bought += 1

            elif results:
                # 满仓换仓: PnL-based — 换掉亏损持仓
                swapped = 0
                for br in results:
                    if swapped >= bt_cfg.max_buys_per_day:
                        break
                    # 找亏损最多且持仓≥5天的持仓
                    worst_code, worst_pnl = None, float('inf')
                    for c, p in positions.items():
                        if p.hold_days < 5:
                            continue
                        cl = get_close(c, date_str)
                        if cl is None:
                            continue
                        cur_pnl = (cl - p.entry_price) / p.entry_price
                        # 只换掉亏损或微利(<2%)的持仓
                        if cur_pnl >= 0.02:
                            continue
                        if cur_pnl < worst_pnl:
                            worst_pnl = cur_pnl
                            worst_code = c
                    if worst_code is None:
                        break
                    worst_pos = positions[worst_code]
                    nxt = _next_trading_date(date_str, fmt_dates)
                    if not nxt:
                        continue
                    sell_open = get_open(worst_code, nxt)
                    if not sell_open or sell_open <= 0:
                        continue
                    sp = sell_open * (1 - bt_cfg.slippage_rate)
                    proceeds = worst_pos.shares * sp
                    comm = proceeds * bt_cfg.commission_rate
                    tax = proceeds * bt_cfg.stamp_tax_rate
                    net = proceeds - comm - tax
                    cash += net
                    pnl = net - worst_pos.cost
                    trades.append({
                        'stock_code': worst_code,
                        'entry_date': worst_pos.entry_date,
                        'entry_price': worst_pos.entry_price,
                        'entry_score': worst_pos.entry_score,
                        'exit_date': nxt,
                        'exit_price': sp,
                        'exit_reason': f'换仓({br.stock_code} sc={br.total_score:.3f})',
                        'shares': worst_pos.shares,
                        'pnl': pnl,
                        'pnl_pct': pnl / worst_pos.cost,
                        'hold_days': worst_pos.hold_days,
                    })
                    del positions[worst_code]

                    buy_open = get_open(br.stock_code, nxt)
                    if buy_open and buy_open > 0 and br.stock_code not in positions:
                        buy_price = buy_open * (1 + bt_cfg.slippage_rate)
                        n_avail = bt_cfg.max_positions - len(positions)
                        alloc = min(cash / max(1, n_avail), cash)
                        shares = int(alloc / buy_price / 100) * 100
                        if shares > 0:
                            cost_b = shares * buy_price
                            comm_b = cost_b * bt_cfg.commission_rate
                            total_b = cost_b + comm_b
                            if total_b <= cash:
                                cash -= total_b
                                v5 = get_vol_5d(br.stock_code, date_str)
                                positions[br.stock_code] = Position(
                                    stock_code=br.stock_code,
                                    entry_date=nxt,
                                    entry_price=buy_price,
                                    shares=shares,
                                    cost=total_b,
                                    entry_score=br.total_score,
                                    peak_price=buy_price,
                                    entry_vol_5d=v5,
                                )
                                swapped += 1

        # ---- 净值 ----
        equity = cash
        for pos in positions.values():
            cl = get_close(pos.stock_code, date_str)
            equity += pos.shares * (cl if cl else pos.entry_price)
        equity_curve.append((date_str, equity))

        if (day_idx + 1) % 10 == 0 or day_idx == len(trading_dates) - 1:
            nav = equity / bt_cfg.initial_capital
            elapsed = time.time() - t0
            speed = (day_idx + 1) / elapsed if elapsed > 0 else 1
            eta = (len(trading_dates) - day_idx - 1) / speed
            print(f'  [{day_idx + 1:>3d}/{len(trading_dates)}] {date_str} '
                  f'NAV={nav:.4f} 持仓={len(positions)} 现金={cash:,.0f} '
                  f'{elapsed:.0f}s (ETA {eta:.0f}s)')

    # ---- 强平 ----
    last_date = _date_str(trading_dates[-1])
    for code, pos in list(positions.items()):
        cl = get_close(code, last_date)
        if cl is None:
            cl = pos.entry_price
        sp = cl
        proceeds = pos.shares * sp
        comm = proceeds * bt_cfg.commission_rate
        tax = proceeds * bt_cfg.stamp_tax_rate
        net = proceeds - comm - tax
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
            'pnl': pnl,
            'pnl_pct': pnl / pos.cost,
            'hold_days': pos.hold_days,
        })
    positions.clear()

    # ---- 绩效报告 ----
    eq_df = pd.DataFrame(equity_curve, columns=['date', 'equity'])
    eq_df['nav'] = eq_df['equity'] / bt_cfg.initial_capital
    eq_df['ret'] = eq_df['nav'].pct_change().fillna(0)

    total_return = eq_df['nav'].iloc[-1] - 1
    trading_days = len(eq_df)
    ann_return = (1 + total_return) ** (252 / max(trading_days, 1)) - 1

    # 最大回撤
    cummax = eq_df['nav'].cummax()
    drawdown = (eq_df['nav'] - cummax) / cummax
    max_dd = float(drawdown.min())

    # 夏普
    daily_ret = eq_df['ret'].values
    sharpe = float(np.mean(daily_ret) / max(np.std(daily_ret), 1e-8) * np.sqrt(252))

    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

    trade_df = pd.DataFrame(trades)
    n_trades = len(trade_df)
    win_rate = len(trade_df[trade_df['pnl_pct'] > 0]) / max(n_trades, 1)
    avg_win = float(trade_df[trade_df['pnl_pct'] > 0]['pnl_pct'].mean()) if (trade_df['pnl_pct'] > 0).any() else 0
    avg_loss = float(trade_df[trade_df['pnl_pct'] <= 0]['pnl_pct'].mean()) if (trade_df['pnl_pct'] <= 0).any() else 0
    avg_hold = float(trade_df['hold_days'].mean()) if n_trades > 0 else 0

    print(f'\n{"=" * 60}')
    print(f'  量价突破策略 回测绩效')
    print(f'{"=" * 60}')
    print(f'  期间:       {bt_cfg.start_date} ~ {bt_cfg.end_date}')
    print(f'  初始资金:      {bt_cfg.initial_capital:,.0f}')
    print(f'  最终资金:      {eq_df["equity"].iloc[-1]:,.0f}')
    print(f'  总收益率:        {total_return:.2%}')
    print(f'  年化收益率:      {ann_return:.2%}')
    print(f'  最大回撤:       {max_dd:.2%}')
    print(f'  夏普比率:        {sharpe:.4f}')
    print(f'  Calmar比率:      {calmar:.4f}')
    print(f'  总交易:              {n_trades}')
    print(f'  胜率:            {win_rate:.2%}')
    print(f'  平均盈利:        {avg_win:.2%}')
    print(f'  平均亏损:        {avg_loss:.2%}')
    print(f'  平均持仓:          {avg_hold:.1f}天')

    # 卖出原因
    if n_trades > 0:
        print(f'\n  卖出原因:')
        for reason_prefix in ['止损', '追踪止盈', '量能衰竭', '持仓超时', '换仓', '回测结束']:
            mask = trade_df['exit_reason'].str.startswith(reason_prefix)
            sub = trade_df[mask]
            if len(sub) > 0:
                avg_pnl = sub['pnl_pct'].mean()
                total_pnl = sub['pnl'].sum()
                print(f'    {reason_prefix:<8s}:  {len(sub):>2d}笔 平均{avg_pnl:>7.2%} 总盈亏 {total_pnl:>12,.0f}')

    # 目标股票
    print(f'\n  --- 目标股票 ---')
    for code, name in TARGETS.items():
        t_trades = trade_df[trade_df['stock_code'] == code]
        if len(t_trades) > 0:
            for _, r in t_trades.iterrows():
                print(f'  ✅ {name} ({code}): {r["entry_date"]} → {r["exit_date"]} '
                      f'入{r["entry_price"]:.2f} 出{r["exit_price"]:.2f} '
                      f'盈亏{r["pnl_pct"]:.1%} 持{r["hold_days"]}天 | {r["exit_reason"]}')
        else:
            print(f'  ❌ {name} ({code}): 未交易')

    # 月度收益
    print(f'\n  月度收益:')
    eq_df['month'] = eq_df['date'].str[:7]
    monthly = eq_df.groupby('month')['ret'].sum()
    for m, r in monthly.items():
        bar = '█' * int(abs(r) * 100) if r >= 0 else '░' * int(abs(r) * 100)
        sign = '' if r >= 0 else '-'
        print(f'    {m}: {r:>7.2%}  {sign}{bar}')

    # 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    eq_df.to_csv(os.path.join(OUTPUT_DIR, 'equity.csv'), index=False)
    trade_df.to_csv(os.path.join(OUTPUT_DIR, 'trades.csv'), index=False)
    print(f'\n  结果已保存: {OUTPUT_DIR}/')


if __name__ == '__main__':
    run_backtest()
