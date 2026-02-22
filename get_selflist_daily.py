import requests
import json
import pandas as pd
import time
from datetime import datetime
import os
import threading
import argparse
import shutil
import chinese_calendar as calendar
import random
import re


def _symbol_to_em_secid(symbol: str) -> str:
    s = symbol.strip().lower()
    if s.startswith("sh"):
        return f"1.{s[2:]}"
    if s.startswith("sz"):
        return f"0.{s[2:]}"
    # Fallback: infer from first digit (6/9 for SH; others SZ)
    market_code = 1 if s.startswith(("6", "9")) else 0
    return f"{market_code}.{s}"


def _fetch_em_1m_as_ticklike(symbol: str, date_yyyy_mm_dd: str) -> pd.DataFrame:
    """Fetch Eastmoney 1-minute klines for a single trading day.

    Returns a DataFrame compatible with the old Sina tick schema (tick-like per minute).
    """
    date_yyyymmdd = date_yyyy_mm_dd.replace("-", "")
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "klt": "1",
        "fqt": "0",
        "secid": _symbol_to_em_secid(symbol),
        "beg": date_yyyymmdd,
        "end": date_yyyymmdd,
    }
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Referer": "https://quote.eastmoney.com/",
        "Accept": "application/json,text/plain,*/*",
    }
    r = requests.get(url, params=params, headers=headers, timeout=20)
    r.raise_for_status()
    data_json = r.json()
    klines = (data_json.get("data") or {}).get("klines") or []
    if not klines:
        return pd.DataFrame(
            columns=["成交时间", "成交价", "涨跌幅", "价格变动", "成交量(手)", "成交额(元)", "性质"]
        )

    rows = []
    # kline format: f51..f61, example:
    # 2026-02-13 09:31,10.96,10.96,10.98,10.95,12459,13660888.00,0.27,0.00,0.00,0.01
    for item in klines:
        parts = item.split(",")
        if len(parts) < 7:
            continue
        dt = parts[0]
        hhmmss = dt.split(" ", 1)[1] if " " in dt else dt
        close_price = parts[2]
        volume = parts[5]
        amount = parts[6]
        amp = parts[7] if len(parts) > 7 else ""
        change_pct = parts[8] if len(parts) > 8 else ""
        change_abs = parts[9] if len(parts) > 9 else ""
        rows.append(
            {
                "成交时间": hhmmss + (":00" if len(hhmmss) == 5 else ""),
                "成交价": close_price,
                "涨跌幅": change_pct,
                "价格变动": change_abs,
                "成交量(手)": volume,
                "成交额(元)": amount,
                "性质": "中性盘",
                "_振幅": amp,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Best-effort numeric conversion
    for col in ["成交价", "涨跌幅", "价格变动", "成交量(手)", "成交额(元)"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["成交时间"] = df["成交时间"].astype(str)
    df = df.sort_values("成交时间").reset_index(drop=True)
    return df[["成交时间", "成交价", "涨跌幅", "价格变动", "成交量(手)", "成交额(元)", "性质"]]


def _decode_sina_html(content: bytes) -> str:
    # Sina pages are typically GBK/GB2312; gb18030 is the safest superset.
    return content.decode('gb18030', errors='ignore')


def _extract_page_date(html_text: str) -> str | None:
    m = re.search(r'var\s+pageDate\s*=\s*\"([^\"]+)\"', html_text)
    return m.group(1) if m else None


def _find_trade_detail_table(tables: list[pd.DataFrame]) -> pd.DataFrame | None:
    required = {"成交时间", "成交价", "成交量(手)", "成交额(元)", "性质"}
    for df in tables:
        cols = {str(c).strip() for c in df.columns}
        if required.issubset(cols):
            return df
    # Fallback: some versions may omit a column or rename slightly
    for df in tables:
        cols = {str(c).strip() for c in df.columns}
        if "成交时间" in cols and "成交价" in cols and any("成交量" in c for c in cols):
            return df
    return None

def record_failure(symbol, date, output_dir):
    fail_file = os.path.join(output_dir, "failed_tasks.json")
    failed_tasks = []
    if os.path.exists(fail_file):
        try:
            with open(fail_file, 'r') as f:
                failed_tasks = json.load(f)
        except:
            pass
    task = {"symbol": symbol, "date": date}
    if task not in failed_tasks:
        failed_tasks.append(task)
    with open(fail_file, 'w') as f:
        json.dump(failed_tasks, f, indent=4)

def remove_failure(symbol, date, output_dir):
    fail_file = os.path.join(output_dir, "failed_tasks.json")
    if os.path.exists(fail_file):
        try:
            with open(fail_file, 'r') as f:
                failed_tasks = json.load(f)
            task = {"symbol": symbol, "date": date}
            if task in failed_tasks:
                failed_tasks.remove(task)
                with open(fail_file, 'w') as f:
                    json.dump(failed_tasks, f, indent=4)
        except:
            pass

def get_daily(tasks, working_path, output_dir):
    colnames=["成交时间", "成交价", "涨跌幅", "价格变动", "成交量(手)", "成交额(元)", "性质"] 
    
    for task in tasks:
        sysmbol = task["symbol"]
        today_time = task["date"]
        
        os.chdir(working_path)
        os.chdir(output_dir)
        
        print(f"Processing: {sysmbol} for {today_time}")
        time.sleep(random.uniform(2, 5)) # 增加随机延迟
        
        csv_dir = os.path.join(output_dir, sysmbol)
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        os.chdir(csv_dir)
        
        csv_file = f"{csv_dir}/{today_time}.csv"
        if os.path.exists(csv_file):
            try:
                tmp_df = pd.read_csv(csv_file, delimiter=",")
                if not tmp_df.empty :
                    last_row = tmp_df.iloc[-1]
                    if str(last_row["成交时间"]) < "14:55:00":
                        print(f"{csv_file} last time {last_row['成交时间']} is early, will re-download")
                        os.remove(csv_file)                
                    else:
                        print(f"{sysmbol} {today_time} already complete.")
                        remove_failure(sysmbol, today_time, output_dir)
                        continue
                else:
                    os.remove(csv_file)
            except Exception:
                os.remove(csv_file)
                
        total_detail_df = pd.DataFrame(columns=colnames)
        total_detail_df = total_detail_df.set_index("成交时间")

        day_success = True
        try:
            em_df = _fetch_em_1m_as_ticklike(sysmbol, today_time)
            if not em_df.empty:
                total_detail_df = em_df.set_index("成交时间")
                print(f"Fetched {len(em_df)} 1-minute rows from Eastmoney for {sysmbol} {today_time}")
            else:
                print(f"No data found for {sysmbol} on {today_time}")
        except Exception as e:
            msg = str(e)
            if len(msg) > 200:
                msg = msg[:200] + "..."
            print(f"Eastmoney fetch failed for {sysmbol} {today_time}: {type(e).__name__} - {msg}")
            day_success = False
                
        if day_success:
            if not total_detail_df.empty:
                total_detail_df = total_detail_df.sort_index() # Chronological order by time
                total_detail_df.to_csv(csv_file)
                print(f"finish {sysmbol} {today_time}")
            remove_failure(sysmbol, today_time, output_dir)
        else:
            record_failure(sysmbol, today_time, output_dir)
            
    os.chdir(working_path)

def create_folder(symbols):
    for sysmbol in symbols:
        if not os.path.exists(f"gp_daily/{sysmbol}"):
            os.makedirs(f"gp_daily/{sysmbol}")

def rename_file():
    dir_list = os.listdir("gp_daily")
    cwd = os.getcwd()
    os.chdir("gp_daily")
    for symb_dir in dir_list:
        os.chdir(symb_dir)
        if os.path.exists("2023-6-2.csv"):
            os.rename("2023-6-2.csv", "2023-06-02.csv")
        os.chdir("../")
    os.chdir(cwd)    

def remove_files_not_self_list(self_gplist):
    dir_list = os.listdir("gp_daily")
    cwd = os.getcwd()
    os.chdir("gp_daily")
    for file in dir_list:
        if file in self_gplist:            
            continue
        else:
            if os.path.exists(file):
                shutil.rmtree(file)
                print("remove :" + file)    

    os.chdir(cwd)

def divide_chunks(l, n):     
    for i in range(0, len(l), n):
        yield l[i:i + n]                    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Download daily stock data.')
    parser.add_argument('--start_date', type=str, help='Start date in YYYY-MM-DD format', default="")
    parser.add_argument('--end_date', type=str, help='End date in YYYY-MM-DD format', default="")
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Directory to save the downloaded data (default: ./gp_daily)',
        default=None,
    )
    args = parser.parse_args()

    chunks_num = 1 # 降低并发，原来是 5
    working_path = os.getcwd()
    
    # Set up output directory
    # - If user explicitly provides --output_dir, use it
    # - Otherwise default to ./gp_daily
    output_dir = args.output_dir if args.output_dir else "gp_daily"
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(working_path, output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Data will be saved to: {output_dir}")

    # Self list location:
    # - If user provided --output_dir, read self_gplist.json from that directory
    # - Otherwise read it from current working directory
    self_list_dir = output_dir if args.output_dir else working_path
    json_file = os.path.join(self_list_dir, "self_gplist.json")
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            self_gplist = json.load(f)
        print(f"Loaded {len(self_gplist)} symbols from {json_file}")
    else:
        self_gplist = ["sz002409", "sz301323", "sh688114", "sh688508"]
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self_gplist, f, indent=4)
        print(f"Created default {json_file}")
    
    # Determine dates to process
    dates_to_process = []
    if args.start_date:
        try:
            start = datetime.strptime(args.start_date, '%Y-%m-%d')
            if args.end_date:
                end = datetime.strptime(args.end_date, '%Y-%m-%d')
            else:
                end = datetime.today()
                
            date_generated = pd.date_range(start, end)
            # Filter out weekends and Chinese holidays
            for d in date_generated:
                if calendar.is_workday(d) and d.weekday() < 5:
                    dates_to_process.append(d.strftime('%Y-%m-%d'))
            print(f"Processing dates: {dates_to_process}")
        except ValueError:
            print("Error: Dates must be in YYYY-MM-DD format.")
            exit(1)
    elif args.end_date:
        print("Error: Cannot provide --end_date without --start_date.")
        exit(1)
    else:
        # Default behavior: today
        today_str = datetime.today().strftime('%Y-%m-%d')
        if calendar.is_workday(datetime.today()) and datetime.today().weekday() < 5:
            dates_to_process = [today_str]
        else:
            dates_to_process = []
            print("Today is not a trading day.")

    # Load failed tasks
    fail_file = os.path.join(output_dir, "failed_tasks.json")
    failed_tasks = []
    if os.path.exists(fail_file):
        try:
            with open(fail_file, 'r') as f:
                failed_tasks = json.load(f)
        except:
            pass

    tasks_to_run = []
    for target_date in dates_to_process:
        for sym in self_gplist:
            tasks_to_run.append({"symbol": sym, "date": target_date})
            
    for ft in failed_tasks:
        if ft not in tasks_to_run and ft["symbol"] in self_gplist:
            tasks_to_run.append(ft)

    if not tasks_to_run:
        print("No tasks to run.")
        exit(0)

    # Divide tasks into chunks
    chunk_size = max(1, int(len(tasks_to_run) / chunks_num))
    tasks_cks = list(divide_chunks(tasks_to_run, chunk_size))
    actual_chunks_num = len(tasks_cks)
    
    threads = []
    for i in range(actual_chunks_num):    
        t = threading.Thread(target=get_daily, args=(tasks_cks[i], working_path, output_dir))
        t.name = f"worker_{i}"
        print(t.name)
        threads.append(t)

    for t in threads:
        t.start()
        
    for t in threads:
        t.join()
