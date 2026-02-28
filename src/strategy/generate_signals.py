import os
import glob
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count

# Re-use the existing math functions
def calc_entropy(x):
    x = x[~np.isnan(x)]
    if len(x) < 5: return 999.0
    hist, _ = np.histogram(x, bins=10)
    hist = hist / np.sum(hist)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log(hist))

def fast_hurst(ts):
    ts = np.asarray(ts)
    if len(ts) < 10: return 0.5
    lags = [2, 4, 8, 16]
    tau, msd = [], []
    for lag in lags:
        if lag >= len(ts): break
        diffs = ts[lag:] - ts[:-lag]
        val = np.mean(diffs**2)
        if val > 0:
            msd.append(val)
            tau.append(lag)
    if len(tau) < 2: return 0.5
    m = np.polyfit(np.log(tau), np.log(msd), 1)
    return m[0] / 2.0

def scan_latest_signal(file_path):
    try:
        df = pd.read_csv(file_path)
    except: return None
    
    if len(df) < 61: return None
    
    df = df.sort_values('trade_date').reset_index(drop=True)
    df['trade_date_str'] = df['trade_date'].astype(str)
    
    required_cols = ['close', 'open', 'pct_chg', 'net_mf_amount']
    if not all(c in df.columns for c in required_cols): return None
        
    df['ma60'] = df['close'].rolling(window=60).mean()
    df['mf_5d'] = df['net_mf_amount'].fillna(0).rolling(window=5).sum()
    
    # Only look at the very last day (or last 3 days to catch recent signals)
    recent_signals = []
    
    closes = df['close'].values
    pct_chgs = df['pct_chg'].values
    dates = df['trade_date_str'].values
    
    # Check last 3 trading days
    for i in range(len(df)-3, len(df)):
        if i < 60: continue
            
        if pct_chgs[i] < 3.0: continue
        if closes[i] < df['ma60'].iloc[i]: continue
        if df['mf_5d'].iloc[i] <= 0: continue
            
        window_60 = closes[i-60:i+1]
        window_20 = closes[i-20:i+1]
        ret_60 = pct_chgs[i-60:i+1]
        ret_20 = pct_chgs[i-20:i+1]
        
        h60 = fast_hurst(window_60)
        h20 = fast_hurst(window_20)
        e60 = calc_entropy(ret_60)
        e20 = calc_entropy(ret_20)
        
        if not (0.20 <= h60 <= 0.60): continue
        if h20 < 0.45: continue
        if e20 > e60 + 0.1: continue
            
        recent_signals.append({
            'ts_code': df['ts_code'].iloc[0],
            'signal_date': dates[i],
            'close_price': closes[i],
            'h60': round(h60, 3),
            'h20': round(h20, 3)
        })
        
    return recent_signals

def main():
    data_dir = "/nvme5/xtang/gp-workspace/gp-data/tushare-daily-full/"
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    pool = Pool(cpu_count() - 1)
    results = pool.map(scan_latest_signal, files)
    
    all_sigs = []
    for r in results:
        if r: all_sigs.extend(r)
        
    if not all_sigs:
        print("No recent signals found.")
        return
        
    df_sig = pd.DataFrame(all_sigs)
    
    basic_path = '/nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv'
    if os.path.exists(basic_path):
        basic = pd.read_csv(basic_path)
        name_map = dict(zip(basic['ts_code'], basic['name']))
        df_sig.insert(1, 'name', df_sig['ts_code'].map(name_map))
        
    df_sig = df_sig.sort_values('signal_date', ascending=False)
    print("=== 最新交易日发现的买入信号 (相变临界点) ===")
    print(df_sig.to_markdown(index=False))

if __name__ == "__main__":
    main()
