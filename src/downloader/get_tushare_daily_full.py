import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import pandas as pd
import tushare as ts
import threading


def _infer_last_open_trade_date(pro, today_yyyymmdd: str, exchange: str = "SSE", lookback_days: int = 120) -> str:
    """Return the last open trade date <= today.

    This avoids treating weekends/holidays as missing data.
    Falls back to today on any error.
    """

    try:
        end_dt = datetime.strptime(today_yyyymmdd, "%Y%m%d")
        start_dt = end_dt - timedelta(days=int(lookback_days))
        df = pro.trade_cal(exchange=exchange, start_date=start_dt.strftime("%Y%m%d"), end_date=today_yyyymmdd)
        if df is None or df.empty:
            return today_yyyymmdd
        if "is_open" not in df.columns or "cal_date" not in df.columns:
            return today_yyyymmdd

        open_df = df[df["is_open"].astype(int) == 1]
        if open_df.empty:
            return today_yyyymmdd

        v = open_df["cal_date"].max()
        return str(v)
    except Exception:
        return today_yyyymmdd

class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.time()
            self.calls = [t for t in self.calls if now - t < self.period]
            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (now - self.calls[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                now = time.time()
                self.calls = [t for t in self.calls if now - t < self.period]
            self.calls.append(now)

# Tushare limit is 200 per minute. Let's use 150 per 60 seconds to be safe.
api_limiter = RateLimiter(150, 60)

def convert_to_ts_code(symbol: str) -> str:
    """Convert sh600000 to 600000.SH"""
    symbol = symbol.strip().lower()
    if symbol.startswith("sh"):
        return f"{symbol[2:]}.SH"
    elif symbol.startswith("sz"):
        return f"{symbol[2:]}.SZ"
    elif symbol.startswith("bj"):
        return f"{symbol[2:]}.BJ"
    return symbol

def get_last_date(csv_path: str) -> str:
    """Get the last trade_date from the existing CSV."""
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path, usecols=['trade_date'])
        if df.empty:
            return None
        return str(df['trade_date'].max())
    except Exception:
        return None

def fetch_data_in_chunks(pro, api_name, ts_code, start_date, end_date, chunk_years=10):
    """Fetch data from Tushare in chunks to avoid row limits (usually 5000 rows)."""
    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    
    all_dfs = []
    current_start = start_dt
    
    while current_start <= end_dt:
        current_end = current_start + timedelta(days=365 * chunk_years)
        if current_end > end_dt:
            current_end = end_dt
            
        s_date = current_start.strftime("%Y%m%d")
        e_date = current_end.strftime("%Y%m%d")
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                api_limiter.wait()
                if api_name == 'daily':
                    df = pro.daily(ts_code=ts_code, start_date=s_date, end_date=e_date)
                elif api_name == 'daily_basic':
                    df = pro.daily_basic(ts_code=ts_code, start_date=s_date, end_date=e_date)
                elif api_name == 'moneyflow':
                    df = pro.moneyflow(ts_code=ts_code, start_date=s_date, end_date=e_date)
                else:
                    raise ValueError(f"Unknown api_name: {api_name}")
                    
                if df is not None and not df.empty:
                    all_dfs.append(df)
                break # Success, break retry loop
            except Exception as e:
                err_msg = str(e)
                if "每分钟最多访问" in err_msg:
                    print(f"Rate limit hit for {api_name} {ts_code}. Sleeping 30s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(30)
                else:
                    print(f"Error fetching {api_name} for {ts_code} from {s_date} to {e_date}: {e}. Retrying in 5s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(5)
                if attempt == max_retries - 1:
                    print(f"Failed to fetch {api_name} for {ts_code} after {max_retries} attempts.")
            
        current_start = current_end + timedelta(days=1)
        
    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)

def download_one_symbol(pro, symbol: str, out_dir: str, start_date: str, end_date: str) -> tuple[str, str]:
    ts_code = convert_to_ts_code(symbol)
    csv_path = os.path.join(out_dir, f"{symbol}.csv")
    
    last_date = get_last_date(csv_path)
    
    fetch_start = start_date
    if last_date:
        # If we have data up to last_date, start fetching from the next day
        last_dt = datetime.strptime(str(last_date), "%Y%m%d")
        fetch_start = (last_dt + timedelta(days=1)).strftime("%Y%m%d")
        
    if fetch_start > end_date:
        return symbol, "up-to-date"
        
    # Fetch the three datasets
    df_daily = fetch_data_in_chunks(pro, 'daily', ts_code, fetch_start, end_date)
    if df_daily.empty:
        # Common case: today is not a trading day (weekend/holiday), so there is
        # nothing new to append. Treat as up-to-date if we already have history.
        if last_date:
            return symbol, "up-to-date"
        return symbol, "no daily data"
        
    df_basic = fetch_data_in_chunks(pro, 'daily_basic', ts_code, fetch_start, end_date)
    df_moneyflow = fetch_data_in_chunks(pro, 'moneyflow', ts_code, fetch_start, end_date)
    
    # Merge them
    # daily_basic has many columns, we might only want specific ones or all of them.
    # The user asked for: k线 (daily), 换手率 (daily_basic), 成交额 (daily), 资金流向 (moneyflow)
    # Let's merge all available columns, but avoid duplicating overlapping columns like 'close'
    
    df = df_daily
    
    if not df_basic.empty:
        cols_to_use = df_basic.columns.difference(df.columns).tolist() + ['ts_code', 'trade_date']
        df = pd.merge(df, df_basic[cols_to_use], on=['ts_code', 'trade_date'], how='left')
        
    if not df_moneyflow.empty:
        cols_to_use = df_moneyflow.columns.difference(df.columns).tolist() + ['ts_code', 'trade_date']
        df = pd.merge(df, df_moneyflow[cols_to_use], on=['ts_code', 'trade_date'], how='left')
        
    # Sort by date ascending
    df = df.sort_values('trade_date', ascending=True)
    
    # Save to CSV
    if last_date and os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
        return symbol, f"appended {len(df)} rows"
    else:
        df.to_csv(csv_path, index=False)
        return symbol, f"wrote {len(df)} rows"

def main():
    p = argparse.ArgumentParser(description="Download full-history daily data from Tushare")
    p.add_argument("--output_dir", type=str, required=True, help="Output directory root")
    p.add_argument("--token", type=str, required=True, help="Tushare API token")
    p.add_argument("--threads", type=int, default=4, help="Worker threads (default: 4)")
    p.add_argument("--start_date", type=str, default="19900101", help="Start date YYYYMMDD")
    p.add_argument("--end_date", type=str, default="", help="End date YYYYMMDD (default: today)")
    p.add_argument("--symbols", type=str, default="", help="Optional comma-separated symbol list")
    p.add_argument("--list_file", type=str, default="tushare_gplist.json", help="JSON file containing symbol list")
    
    args = p.parse_args()
    
    ts.set_token(args.token)
    pro = ts.pro_api()
    
    out_dir = args.output_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(os.getcwd(), out_dir)
        
    target_dir = os.path.join(out_dir, "tushare-daily-full")
    os.makedirs(target_dir, exist_ok=True)
    
    today_yyyymmdd = datetime.today().strftime("%Y%m%d")
    if args.end_date.strip():
        end_date = args.end_date.strip()
    else:
        end_date = _infer_last_open_trade_date(pro, today_yyyymmdd)
    
    if args.symbols.strip():
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        list_path = os.path.join(out_dir, args.list_file)
        if not os.path.exists(list_path):
            print(f"Error: List file not found at {list_path}")
            return
        with open(list_path, "r", encoding="utf-8") as f:
            symbols = json.load(f)
            
    print(f"Output dir: {target_dir}")
    print(f"Symbols: {len(symbols)}")
    print(f"Date range: {args.start_date} to {end_date}")
    
    max_workers = max(1, args.threads)
    ok = 0
    up_to_date = 0
    failed = 0
    
    t0 = time.time()
    
    def _task(sym):
        try:
            return download_one_symbol(pro, sym, target_dir, args.start_date, end_date)
        except Exception as e:
            return sym, f"failed: {e}"
            
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_task, s): s for s in symbols}
        for i, fut in enumerate(as_completed(futures), 1):
            sym, status = fut.result()
            
            if "wrote" in status or "appended" in status:
                ok += 1
            elif status == "up-to-date":
                up_to_date += 1
            else:
                failed += 1
                
            if i % 50 == 0 or i == len(symbols):
                elapsed = int(time.time() - t0)
                print(f"Progress {i}/{len(symbols)} | ok={ok} up-to-date={up_to_date} failed={failed} | elapsed_s={elapsed}")
                
            if status not in {"up-to-date"} and not ("wrote" in status or "appended" in status):
                print(f"{sym}: {status}")
                
    elapsed_s = int(time.time() - t0)
    print(f"Done. ok={ok} up-to-date={up_to_date} failed={failed} elapsed_s={elapsed_s}")

if __name__ == "__main__":
    main()
