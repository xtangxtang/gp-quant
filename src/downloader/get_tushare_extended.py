import argparse
import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import pandas as pd
import tushare as ts

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

# Tushare limit is usually 100-500 per minute depending on the endpoint.
# We use 100 per 60 seconds to be safe for financial data.
api_limiter = RateLimiter(100, 60)

def convert_to_ts_code(symbol: str) -> str:
    symbol = symbol.strip().lower()
    if symbol.startswith("sh"):
        return f"{symbol[2:]}.SH"
    elif symbol.startswith("sz"):
        return f"{symbol[2:]}.SZ"
    elif symbol.startswith("bj"):
        return f"{symbol[2:]}.BJ"
    return symbol

def fetch_data_in_chunks(pro, api_name, ts_code, start_date, end_date, chunk_years=10):
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
                func = getattr(pro, api_name)
                df = func(ts_code=ts_code, start_date=s_date, end_date=e_date)
                if df is not None and not df.empty:
                    all_dfs.append(df)
                break
            except Exception as e:
                err_msg = str(e)
                if "每分钟最多访问" in err_msg:
                    time.sleep(30)
                else:
                    time.sleep(5)
                if attempt == max_retries - 1:
                    print(f"Failed to fetch {api_name} for {ts_code} after {max_retries} attempts.")
            
        current_start = current_end + timedelta(days=1)
        
    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)

def fetch_data_single(pro, api_name, ts_code):
    max_retries = 5
    for attempt in range(max_retries):
        try:
            api_limiter.wait()
            func = getattr(pro, api_name)
            df = func(ts_code=ts_code)
            return df
        except Exception as e:
            err_msg = str(e)
            if "每分钟最多访问" in err_msg:
                time.sleep(30)
            else:
                time.sleep(5)
            if attempt == max_retries - 1:
                print(f"Failed to fetch {api_name} for {ts_code} after {max_retries} attempts.")
    return pd.DataFrame()

def download_one_symbol(pro, symbol: str, out_dir: str, dataset: str, start_date: str, end_date: str) -> tuple[str, str]:
    ts_code = convert_to_ts_code(symbol)
    csv_path = os.path.join(out_dir, f"{symbol}.csv")
    
    if os.path.exists(csv_path):
        return symbol, "up-to-date"
        
    # Datasets that need chunking due to row limits
    chunked_datasets = ['adj_factor', 'stk_limit']
    
    if dataset in chunked_datasets:
        df = fetch_data_in_chunks(pro, dataset, ts_code, start_date, end_date)
    else:
        df = fetch_data_single(pro, dataset, ts_code)
        
    if df is None or df.empty:
        return symbol, "no data"
        
    # Sort by date if applicable
    if 'trade_date' in df.columns:
        df = df.sort_values('trade_date', ascending=True)
    elif 'end_date' in df.columns:
        df = df.sort_values('end_date', ascending=True)
    elif 'ann_date' in df.columns:
        df = df.sort_values('ann_date', ascending=True)
        
    df.to_csv(csv_path, index=False)
    return symbol, f"wrote {len(df)} rows"

def download_global_data(pro, out_dir: str, dataset: str):
    csv_path = os.path.join(out_dir, f"{dataset}.csv")
    if os.path.exists(csv_path):
        print(f"{dataset} already exists at {csv_path}")
        return
        
    print(f"Fetching global dataset: {dataset}...")
    max_retries = 5
    for attempt in range(max_retries):
        try:
            api_limiter.wait()
            func = getattr(pro, dataset)
            df = func()
            if df is not None and not df.empty:
                df.to_csv(csv_path, index=False)
                print(f"Successfully saved {dataset} to {csv_path}")
            else:
                print(f"No data returned for {dataset}")
            return
        except Exception as e:
            err_msg = str(e)
            if "每分钟最多访问" in err_msg:
                time.sleep(30)
            else:
                time.sleep(5)
    print(f"Failed to fetch {dataset}")

def main():
    p = argparse.ArgumentParser(description="Download extended data from Tushare")
    p.add_argument("--output_dir", type=str, required=True, help="Output directory root")
    p.add_argument("--token", type=str, required=True, help="Tushare API token")
    p.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., adj_factor, income, trade_cal)")
    p.add_argument("--threads", type=int, default=4, help="Worker threads (default: 4)")
    p.add_argument("--start_date", type=str, default="19900101", help="Start date YYYYMMDD")
    p.add_argument("--end_date", type=str, default="", help="End date YYYYMMDD (default: today)")
    p.add_argument("--list_file", type=str, default="tushare_gplist.json", help="JSON file containing symbol list")
    
    args = p.parse_args()
    
    ts.set_token(args.token)
    pro = ts.pro_api()
    
    out_dir = args.output_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(os.getcwd(), out_dir)
        
    target_dir = os.path.join(out_dir, f"tushare-{args.dataset}")
    os.makedirs(target_dir, exist_ok=True)
    
    # Global datasets that don't need symbol iteration
    global_datasets = ['trade_cal']
    if args.dataset in global_datasets:
        download_global_data(pro, target_dir, args.dataset)
        return
    
    end_date = args.end_date.strip() or datetime.today().strftime("%Y%m%d")
    
    list_path = os.path.join(out_dir, args.list_file)
    if not os.path.exists(list_path):
        print(f"Error: List file not found at {list_path}")
        return
    with open(list_path, "r", encoding="utf-8") as f:
        symbols = json.load(f)
            
    print(f"Dataset: {args.dataset}")
    print(f"Output dir: {target_dir}")
    print(f"Symbols: {len(symbols)}")
    
    max_workers = max(1, args.threads)
    ok = 0
    up_to_date = 0
    failed = 0
    
    t0 = time.time()
    
    def _task(sym):
        try:
            return download_one_symbol(pro, sym, target_dir, args.dataset, args.start_date, end_date)
        except Exception as e:
            return sym, f"failed: {e}"
            
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_task, s): s for s in symbols}
        for i, fut in enumerate(as_completed(futures), 1):
            sym, status = fut.result()
            
            if "wrote" in status:
                ok += 1
            elif status == "up-to-date":
                up_to_date += 1
            else:
                failed += 1
                
            if i % 50 == 0 or i == len(symbols):
                elapsed = int(time.time() - t0)
                print(f"Progress {i}/{len(symbols)} | ok={ok} up-to-date={up_to_date} failed={failed} | elapsed_s={elapsed}")
                
            if status not in {"up-to-date", "no data"} and not "wrote" in status:
                print(f"{sym}: {status}")
                
    elapsed_s = int(time.time() - t0)
    print(f"Done. ok={ok} up-to-date={up_to_date} failed={failed} elapsed_s={elapsed_s}")

if __name__ == "__main__":
    main()
