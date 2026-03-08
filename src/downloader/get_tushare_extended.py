import argparse
import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import pandas as pd
import tushare as ts


def _infer_last_open_trade_date(pro, today_yyyymmdd: str, exchange: str = "SSE", lookback_days: int = 120) -> str:
    """Return the last open trade date <= today (YYYYMMDD).

    Used to avoid pointless requests on weekends/holidays.
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
        return str(open_df["cal_date"].max())
    except Exception:
        return today_yyyymmdd


def _detect_date_col(csv_path: str) -> str | None:
    try:
        header = pd.read_csv(csv_path, nrows=0)
        cols = set(header.columns)
    except Exception:
        return None
    for c in ("trade_date", "end_date", "ann_date", "cal_date"):
        if c in cols:
            return c
    return None


def _get_last_date(csv_path: str, date_col: str) -> str | None:
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path, usecols=[date_col])
        if df.empty:
            return None
        v = df[date_col].dropna().astype(str).max()
        return str(v) if v else None
    except Exception:
        return None


def _safe_call_api(pro, api_name: str, **params) -> pd.DataFrame:
    """Call a tushare pro endpoint with best-effort parameter compatibility."""
    func = getattr(pro, api_name)
    params = {k: v for k, v in params.items() if v is not None and str(v) != ""}
    return func(**params)


def _fetch_incremental(pro, dataset: str, ts_code: str, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    """Fetch data for a symbol, preferring start/end params when supported."""

    # Try with date filters first (many endpoints accept these).
    if start_date or end_date:
        try:
            api_limiter.wait()
            return _safe_call_api(pro, dataset, ts_code=ts_code, start_date=start_date, end_date=end_date)
        except Exception:
            pass

    # Fallback to a plain query.
    api_limiter.wait()
    return _safe_call_api(pro, dataset, ts_code=ts_code)

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

    date_col = None
    last_date = None
    if os.path.exists(csv_path):
        date_col = _detect_date_col(csv_path)
        if date_col:
            last_date = _get_last_date(csv_path, date_col)
        
    # Datasets that need chunking due to row limits
    chunked_datasets = ['adj_factor', 'stk_limit']

    fetch_start = start_date
    if last_date:
        if date_col == "trade_date":
            try:
                last_dt = datetime.strptime(str(last_date), "%Y%m%d")
                fetch_start = (last_dt + timedelta(days=1)).strftime("%Y%m%d")
            except Exception:
                fetch_start = start_date
        else:
            # For quarterly/announcement datasets, overlap by 1 to reduce the chance
            # of missing late updates (we'll deduplicate when rewriting).
            fetch_start = str(last_date)

    # If we are already beyond the requested end_date, no need to call the API.
    if last_date and date_col == "trade_date" and str(fetch_start) > str(end_date):
        return symbol, "up-to-date"

    if dataset in chunked_datasets:
        df = fetch_data_in_chunks(pro, dataset, ts_code, fetch_start, end_date)
    else:
        # Best-effort incremental: try passing start/end, else fall back.
        max_retries = 5
        df = pd.DataFrame()
        for attempt in range(max_retries):
            try:
                df = _fetch_incremental(pro, dataset, ts_code, fetch_start if last_date else start_date, end_date)
                break
            except Exception as e:
                err_msg = str(e)
                if "每分钟最多访问" in err_msg:
                    time.sleep(30)
                else:
                    time.sleep(5)
                if attempt == max_retries - 1:
                    print(f"Failed to fetch {dataset} for {ts_code} after {max_retries} attempts.")
        
    if df is None or df.empty:
        return symbol, "up-to-date" if last_date else "no data"
        
    # Sort by date if applicable
    if 'trade_date' in df.columns:
        df = df.sort_values('trade_date', ascending=True)
    elif 'end_date' in df.columns:
        df = df.sort_values('end_date', ascending=True)
    elif 'ann_date' in df.columns:
        df = df.sort_values('ann_date', ascending=True)
        
    # Persist
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
        return symbol, f"wrote {len(df)} rows"

    # If we fetched strictly after the last trade_date, we can append safely.
    if date_col == "trade_date" and last_date:
        df.to_csv(csv_path, mode='a', header=False, index=False)
        return symbol, f"appended {len(df)} rows"

    # Otherwise, rewrite with dedup to handle overlap.
    try:
        old = pd.read_csv(csv_path)
        merged = pd.concat([old, df], ignore_index=True)
        merged = merged.drop_duplicates()
        if 'trade_date' in merged.columns:
            merged = merged.sort_values('trade_date', ascending=True)
        elif 'end_date' in merged.columns:
            merged = merged.sort_values('end_date', ascending=True)
        elif 'ann_date' in merged.columns:
            merged = merged.sort_values('ann_date', ascending=True)
        elif 'cal_date' in merged.columns:
            merged = merged.sort_values('cal_date', ascending=True)
        merged.to_csv(csv_path, index=False)
        return symbol, f"updated (+{len(df)} rows, deduped)"
    except Exception:
        # Fallback: overwrite
        df.to_csv(csv_path, index=False)
        return symbol, f"overwrote {len(df)} rows"

def download_global_data(pro, out_dir: str, dataset: str):
    csv_path = os.path.join(out_dir, f"{dataset}.csv")
    print(f"Fetching global dataset: {dataset}...")
    max_retries = 5
    for attempt in range(max_retries):
        try:
            api_limiter.wait()
            func = getattr(pro, dataset)
            if dataset == 'trade_cal':
                today_yyyymmdd = datetime.today().strftime("%Y%m%d")
                end_yyyymmdd = _infer_last_open_trade_date(pro, today_yyyymmdd)
                start_yyyymmdd = "19900101"
                if os.path.exists(csv_path):
                    date_col = _detect_date_col(csv_path) or "cal_date"
                    last_date = _get_last_date(csv_path, date_col)
                    if last_date:
                        try:
                            last_dt = datetime.strptime(str(last_date), "%Y%m%d")
                            start_yyyymmdd = (last_dt + timedelta(days=1)).strftime("%Y%m%d")
                        except Exception:
                            start_yyyymmdd = str(last_date)
                df = func(exchange='', start_date=start_yyyymmdd, end_date=end_yyyymmdd)
            else:
                df = func()

            if df is not None and not df.empty:
                if os.path.exists(csv_path):
                    # Append with dedup
                    try:
                        old = pd.read_csv(csv_path)
                        merged = pd.concat([old, df], ignore_index=True).drop_duplicates()
                        if 'cal_date' in merged.columns:
                            merged = merged.sort_values('cal_date', ascending=True)
                        merged.to_csv(csv_path, index=False)
                        print(f"Successfully updated {dataset} at {csv_path}")
                    except Exception:
                        df.to_csv(csv_path, index=False)
                        print(f"Successfully overwrote {dataset} at {csv_path}")
                else:
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
    
    today_yyyymmdd = datetime.today().strftime("%Y%m%d")
    if args.end_date.strip():
        end_date = args.end_date.strip()
    else:
        end_date = _infer_last_open_trade_date(pro, today_yyyymmdd)
    
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
