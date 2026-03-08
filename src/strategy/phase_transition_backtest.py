import os
import glob
import time
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count

START_DATE = "20250101"
END_DATE = "20251230"

def calc_entropy(x):
    """Calculate Shannon Entropy of the distribution of daily returns."""
    # Drop NaNs
    x = x[~np.isnan(x)]
    if len(x) < 5:
        return 999.0
    hist, _ = np.histogram(x, bins=10)
    hist = hist / np.sum(hist)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log(hist))

def fast_hurst(ts):
    """
    A simplified, extremely fast Hurst exponent estimator using mean squared displacement.
    If E[(x_{t+tau} - x_t)^2] ~ tau^(2H), then the slope of log(MSD) vs log(tau) is 2H.
    H = 0.5 is random walk. H > 0.5 is trending. H < 0.5 is mean-reverting.
    """
    ts = np.asarray(ts)
    if len(ts) < 10:
        return 0.5
    lags = [2, 4, 8, 16]
    tau = []
    msd = []
    for lag in lags:
        if lag >= len(ts):
            break
        diffs = ts[lag:] - ts[:-lag]
        val = np.mean(diffs**2)
        if val > 0:
            msd.append(val)
            tau.append(lag)
    
    if len(tau) < 2:
        return 0.5
    m = np.polyfit(np.log(tau), np.log(msd), 1)
    H = m[0] / 2.0
    return H

def process_single_stock(file_path):
    """
    Process a single stock's historical data to find 'Phase Transition' signals.
    """
    try:
        df = pd.read_csv(file_path)
    except Exception:
        return []
        
    if df.empty or 'trade_date' not in df.columns:
        return []
    
    df['trade_date_str'] = df['trade_date'].astype(str)
    # We need data before 20230101 for the 60-day moving averages
    df = df[df['trade_date_str'] >= '20220901'].copy()
    if df.empty:
        return []
    
    df = df.sort_values('trade_date').reset_index(drop=True)
    
    # Needs to be a dataframe with these key columns
    required_cols = ['close', 'open', 'pct_chg', 'net_mf_amount']
    for col in required_cols:
        if col not in df.columns:
            return []
            
    # Calculate prerequisites
    df['ma60'] = df['close'].rolling(window=60).mean()
    df['mf_5d'] = df['net_mf_amount'].fillna(0).rolling(window=5).sum()
    
    # We only care about triggering trades between START_DATE and END_DATE
    out_trades = []
    
    # Use array for speed
    closes = df['close'].values
    pct_chgs = df['pct_chg'].values
    dates = df['trade_date_str'].values

    # Last index within END_DATE for this stock (inclusive)
    end_cut_idx = int(np.searchsorted(dates, END_DATE, side='right') - 1)
    if end_cut_idx < 0:
        return []
    
    for i in range(60, len(df)):
        current_date = dates[i]
        
        if current_date < START_DATE or current_date > END_DATE:
            continue
            
        # Fast pre-condition: stock must be breaking out a bit (e.g. > 5% today) to signal standard "symmetry breaking"
        if pct_chgs[i] < 3.0:
            continue
            
        # Fast pre-condition: Close must be above MA60 (it's an upward phase transition)
        if closes[i] < df['ma60'].iloc[i]:
            continue
            
        # Fast pre-condition: Moneyflow (Energy) is positive over last 5 days
        if df['mf_5d'].iloc[i] <= 0:
            continue
            
        # Passed fast filters. Now calculate the physics metrics (computational)
        window_60 = closes[i-60:i+1] # Long term chaotic base
        window_20 = closes[i-20:i+1] # Short term order emerging
        ret_60 = pct_chgs[i-60:i+1]
        ret_20 = pct_chgs[i-20:i+1]
        
        hurst_60 = fast_hurst(window_60)
        hurst_20 = fast_hurst(window_20)
        
        entropy_60 = calc_entropy(ret_60)
        entropy_20 = calc_entropy(ret_20)
        
        # --- PHASE TRANSITION LOGIC ---
        # 1. 60-day was chaotic or mean-reverting: Hurst ~ 0.2 to 0.60
        if not (0.20 <= hurst_60 <= 0.60):
            continue
            
        # 2. 20-day is suddenly showing trend/order: Hurst > 0.45 or jumping significantly
        if hurst_20 < 0.45:
            continue
            
        # 3. 20-day entropy is slightly less than or similar to 60-day entropy (Order increasing / Entropy resisting increase)
        if entropy_20 > entropy_60 + 0.1:
            continue
            
        # SIGNAL GENERATED!
        ts_code = df['ts_code'].iloc[0]
        
        # Calculate forward returns utilizing physical stop-loss and take-profit.
        # No time-stop exit: if neither sell condition triggers, hold until END_DATE.
        if i + 1 < len(df) and (i + 1) <= end_cut_idx:
            entry_price = df['open'].iloc[i+1]  # Enter at next day open
            max_forward_idx = end_cut_idx + 1

            # Arrays from entry until END_DATE (inclusive)
            f_closes = df['close'].values[i+1:max_forward_idx]
            f_ma20 = df['close'].rolling(window=20).mean().values[i+1:max_forward_idx]
            f_pct_chg = df['pct_chg'].values[i+1:max_forward_idx]

            if len(f_closes) > 0:
                exit_idx = len(f_closes) - 1
                exit_reason = "End Date"
                
                # Iterate through holding period to check for early exit signals
                for j in range(len(f_closes)):
                    p_close = f_closes[j]
                    current_return = (p_close - entry_price) / entry_price
                    
                    # 1. STOP LOSS: Hard stop loss down -8% or dropping significantly below MA20 (False breakout/Heat Death)
                    if current_return <= -0.08 or (p_close < f_ma20[j] * 0.95):
                        exit_idx = j
                        exit_reason = "Stop Loss"
                        break
                    
                    # 2. TAKE PROFIT / CHAOS RETURN: 
                    # If we have held for at least 5 days and prices are very profitable, 
                    # and short-term entropy violently spikes, it means the trending structure is breaking down into chaos (Distribution phase).
                    if j >= 5 and current_return > 0.15:
                        # Re-calculate short-term entropy on the fly to detect if order is lost
                        recent_ret = f_pct_chg[max(0, j-10):j+1]
                        if len(recent_ret) >= 5:
                            current_entropy = calc_entropy(recent_ret)
                            # If current 10-day entropy strongly exceeds the base 60-day entropy, the ordered trend has dissolved
                            if current_entropy > entropy_60 * 1.2:
                                exit_idx = j
                                exit_reason = "Take Profit (Entropy Spike)"
                                break
                
                max_close = np.max(f_closes[:exit_idx+1])
                exit_close = f_closes[exit_idx]
                
                max_return = (max_close - entry_price) / entry_price * 100.0
                hold_return = (exit_close - entry_price) / entry_price * 100.0
                
                out_trades.append({
                    'ts_code': ts_code,
                    'signal_date': current_date,
                    'entry_date': dates[i+1],
                    'exit_reason': exit_reason,
                    'hold_days': exit_idx + 1,
                    'hurst_60': round(hurst_60, 3),
                    'hurst_20': round(hurst_20, 3),
                    'entropy_drop_pct': round((entropy_60 - entropy_20)/entropy_60 * 100, 2),
                    'energy_mf': round(df['mf_5d'].iloc[i], 2),
                    'max_hold_return(%)': round(max_return, 2),
                    'actual_return(%)': round(hold_return, 2)
                })
                
    return out_trades

def main():
    data_dir = "/nvme5/xtang/gp-workspace/gp-data/tushare-daily-full/"
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    print(f"Found {len(files)} stock CSV files. Starting 'Phase Transition' backtest...")
    
    start_time = time.time()
    
    # Multiprocessing to process stocks quickly
    pool = Pool(cpu_count() - 1)
    results = pool.map(process_single_stock, files)
    pool.close()
    pool.join()
    
    all_trades = []
    for r in results:
        all_trades.extend(r)
        
    df_trades = pd.DataFrame(all_trades)
    
    if df_trades.empty:
        print("No trades triggered under these strict conditions.")
        return
        
    # Load basic info for names
    basic_info_path = '/nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv'
    if os.path.exists(basic_info_path):
        df_basic = pd.read_csv(basic_info_path)
        name_map = dict(zip(df_basic['ts_code'], df_basic['name']))
        df_trades.insert(1, 'name', df_trades['ts_code'].map(name_map))
        
    # Sort by date
    df_trades = df_trades.sort_values('signal_date').reset_index(drop=True)
    
    # Calculate global metrics
    total_trades = len(df_trades)
    win_trades = len(df_trades[df_trades['actual_return(%)'] > 0])
    win_rate = win_trades / total_trades * 100
    avg_max_ret = df_trades['max_hold_return(%)'].mean()
    avg_hold_ret = df_trades['actual_return(%)'].mean()
    median_hold_ret = df_trades['actual_return(%)'].median()
    avg_hold_days = df_trades['hold_days'].mean()
    
    # Loss analysis
    loss_trades = df_trades[df_trades['actual_return(%)'] < 0]
    avg_loss = loss_trades['actual_return(%)'].mean() if not loss_trades.empty else 0
    max_loss = loss_trades['actual_return(%)'].min() if not loss_trades.empty else 0
    
    # Exit Reason Breakdown
    exit_counts = df_trades['exit_reason'].value_counts().to_dict()
    stop_loss_pct = exit_counts.get('Stop Loss', 0) / total_trades * 100
    take_proft_pct = exit_counts.get('Take Profit (Entropy Spike)', 0) / total_trades * 100
    end_date_pct = exit_counts.get('End Date', 0) / total_trades * 100

    trades_path = "/nvme5/xtang/gp-workspace/gp-quant/backtest_trades_2025_hold_to_end.csv"
    df_trades.to_csv(trades_path, index=False)

    bull = (
        df_trades.sort_values('actual_return(%)', ascending=False)
        .groupby('ts_code', as_index=False)
        .head(1)
        .head(30)
    )
    bull_table = bull.to_markdown(index=False)
    
    summary = f"""
# A股从混沌到有序（相变）策略回测报告（仅 2025，去掉 20 日时间卖出）

* **基于理论**: 经济物理学 / 耗散结构 / 非线性分岔
* **核心核心指标 (买入信号)**:
  - `Hurst 60` 介于 0.2 - 0.6 (长期一直处于震荡/混沌)
  - `Hurst 20` > 0.45 (短期迅速转为趋势)
  - 股价突破 MA60 且资金净流入 (对称性破缺)
* **卖出离场信号 (最新补充)**:
  1. 止损退出：跌幅达到 -8% 或显著跌穿MA20 (由于能量不足导致的假突破/系统热寂退化现象)
  2. 乱序逃顶：在巨幅盈利后，短期香农熵激增超长期标准的20%，意味着大资金分歧加大、系统趋势开始瓦解回归混沌。
* **回测期间**: {START_DATE} 到 {END_DATE}

## 回测总体表现
* **信号触发总次数**: {total_trades}
* **胜率 (绝对收益为正)**: {win_rate:.2f}%
* **单次交易平均最大浮盈 (持有期内)**: {avg_max_ret:.2f}%
* **平均平仓收益率**: {avg_hold_ret:.2f}%
* **中位数收益率**: {median_hold_ret:.2f}%
* **平均持仓交易日天数**: {avg_hold_days:.1f} 天

### 卖出信号触发情况分析
* **打中止损条件离场 (Stop Loss)**: {stop_loss_pct:.1f}% （截断了左侧致命长尾风险）
* **在高位触发混沌逃顶 (Take Profit)**: {take_proft_pct:.1f}% （鱼尾行情瓦解时保住主升浪利润）
* **持仓到 {END_DATE} 自然到期平仓 (End Date)**: {end_date_pct:.1f}%

### 亏损情况分析
* **亏损交易占比**: {100 - win_rate:.2f}%
* **严格止损后的平均亏损幅度**: {avg_loss:.2f}%
* **单次交易最大亏损幅度**: {max_loss:.2f}% （因一字跌停或极其极端缺口等导致未能如期抛出）

## 2025 牛股榜（按标的最佳一笔收益率 Top 30）
{bull_table}


## 经典相变触发案例 (收益最高的 20 次交易)
"""
    
    top_20 = df_trades.sort_values('actual_return(%)', ascending=False).head(20)
    summary += top_20.to_markdown(index=False)
    
    # Save report
    report_path = "/nvme5/xtang/gp-workspace/gp-quant/backtest_report_phase_transition_2025_hold_to_end.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(summary)
        
    print(f"\nBacktest completed in {time.time() - start_time:.2f} seconds.")
    print("="*60)
    print(f"Total Signals: {total_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Avg Max Return: {avg_max_ret:.2f}%")
    print(f"Avg Holding Return: {avg_hold_ret:.2f}%")
    print("="*60)
    print(f"Full report exported to: {report_path}")
    print(f"Full trades exported to: {trades_path}")

if __name__ == "__main__":
    main()
