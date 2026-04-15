"""
Download all Tushare data available with 2000-point token.

Data categories:
1. Per-stock datasets (iterate over all symbols)
2. Per-date datasets (iterate over trade dates)
3. Global datasets (single API call or paginated)
4. Index datasets
5. Fund datasets
6. Futures datasets
7. Bond datasets
8. Macro datasets
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import pandas as pd
import tushare as ts


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
class RateLimiter:
    def __init__(self, max_calls: int, period: float):
        self.max_calls = int(max_calls)
        self.period = float(period)
        self.calls: list[float] = []
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.time()
            self.calls = [t for t in self.calls if now - t < self.period]
            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (now - self.calls[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                now = time.time()
                self.calls = [t for t in self.calls if now - t < self.period]
            self.calls.append(time.time())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def safe_api_call(pro, api_name: str, rate_limiter: RateLimiter, max_retries: int = 5, **params) -> pd.DataFrame:
    """Call a Tushare API with rate limiting and exponential backoff."""
    params = {k: v for k, v in params.items() if v is not None and str(v).strip() != ""}
    func = getattr(pro, api_name)
    for attempt in range(max_retries):
        try:
            rate_limiter.wait()
            df = func(**params)
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            err = str(e)
            if "每分钟最多访问" in err:
                wait = 15 * (attempt + 1)
                time.sleep(wait)
            elif "没有访问该接口的权限" in err or "权限" in err:
                return pd.DataFrame()
            else:
                time.sleep(5 * (attempt + 1))
            if attempt == max_retries - 1:
                raise
    return pd.DataFrame()


def get_trade_dates(pro, rate_limiter, start_date: str, end_date: str) -> list[str]:
    """Get list of open trade dates in range."""
    df = safe_api_call(pro, "trade_cal", rate_limiter,
                       exchange="SSE", start_date=start_date, end_date=end_date)
    if df.empty:
        return []
    open_df = df[df["is_open"].astype(int) == 1]
    dates = sorted(open_df["cal_date"].astype(str).tolist())
    return dates


def load_symbols(data_dir: str) -> list[str]:
    """Load symbol list from gplist.json."""
    path = os.path.join(data_dir, "tushare_gplist.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def symbol_to_ts_code(sym: str) -> str:
    s = sym.strip().lower()
    if s.startswith("sh"):
        return f"{s[2:]}.SH"
    if s.startswith("sz"):
        return f"{s[2:]}.SZ"
    if s.startswith("bj"):
        return f"{s[2:]}.BJ"
    return s


def save_csv(df: pd.DataFrame, path: str, sort_col: str | None = None, append: bool = False):
    """Save dataframe to CSV with optional append and dedup."""
    if df.empty:
        return
    if sort_col and sort_col in df.columns:
        df[sort_col] = pd.to_numeric(df[sort_col], errors='coerce')
        df = df.sort_values(sort_col, ascending=True)

    if append and os.path.exists(path):
        old = pd.read_csv(path)
        if sort_col and sort_col in old.columns:
            old[sort_col] = pd.to_numeric(old[sort_col], errors='coerce')
        frames = [f.dropna(axis=1, how='all') for f in [old, df] if not f.empty]
        merged = pd.concat(frames, ignore_index=True).drop_duplicates() if frames else old
        if sort_col and sort_col in merged.columns:
            merged[sort_col] = pd.to_numeric(merged[sort_col], errors='coerce')
            merged = merged.sort_values(sort_col, ascending=True)
        merged.to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)


def get_last_date_in_csv(path: str, date_col: str) -> str | None:
    """Read the max date from a CSV file."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, usecols=[date_col])
        if df.empty:
            return None
        v = df[date_col].dropna().astype(str).max()
        return str(v) if v else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Download functions per category
# ---------------------------------------------------------------------------

def download_per_stock_dataset(pro, rl, symbols, out_dir, api_name, dir_name,
                               sort_col="trade_date", threads=4, start_date="19900101",
                               end_date=None):
    """Download a dataset that requires iterating over stock ts_codes."""
    target_dir = os.path.join(out_dir, dir_name)
    os.makedirs(target_dir, exist_ok=True)

    if end_date is None:
        end_date = datetime.today().strftime("%Y%m%d")

    ok = 0
    skip = 0
    fail = 0
    t0 = time.time()

    def _task(sym):
        ts_code = symbol_to_ts_code(sym)
        csv_path = os.path.join(target_dir, f"{sym}.csv")

        fetch_start = start_date
        if os.path.exists(csv_path):
            last = get_last_date_in_csv(csv_path, sort_col)
            if last:
                try:
                    dt = datetime.strptime(str(last).split(".")[0], "%Y%m%d")
                    fetch_start = (dt + timedelta(days=1)).strftime("%Y%m%d")
                except Exception:
                    fetch_start = str(last)
                if str(fetch_start) > str(end_date):
                    return sym, "up-to-date"

        try:
            df = safe_api_call(pro, api_name, rl, ts_code=ts_code,
                               start_date=fetch_start, end_date=end_date)
            if df.empty:
                return sym, "no-data"
            save_csv(df, csv_path, sort_col=sort_col, append=os.path.exists(csv_path))
            return sym, f"wrote {len(df)}"
        except Exception as e:
            return sym, f"fail: {e}"

    first_fail_logged = False
    with ThreadPoolExecutor(max_workers=max(1, threads)) as ex:
        futs = {ex.submit(_task, s): s for s in symbols}
        for i, fut in enumerate(as_completed(futs), 1):
            sym, status = fut.result()
            if "wrote" in status:
                ok += 1
            elif status in ("up-to-date", "no-data"):
                skip += 1
            else:
                fail += 1
                if not first_fail_logged:
                    print(f"  {api_name} first fail: {sym} → {status}")
                    first_fail_logged = True
            if i % 200 == 0 or i == len(symbols):
                elapsed = int(time.time() - t0)
                print(f"  {api_name} progress {i}/{len(symbols)} ok={ok} skip={skip} fail={fail} elapsed={elapsed}s")

    elapsed = int(time.time() - t0)
    print(f"  {api_name} done ok={ok} skip={skip} fail={fail} elapsed={elapsed}s")


def download_per_date_dataset(pro, rl, out_dir, api_name, dir_name,
                               sort_col="trade_date", start_date="20100101",
                               end_date=None):
    """Download a dataset that accepts trade_date as param (batch by date, single CSV)."""
    target_dir = os.path.join(out_dir, dir_name)
    os.makedirs(target_dir, exist_ok=True)
    csv_path = os.path.join(target_dir, f"{api_name}.csv")

    if end_date is None:
        end_date = datetime.today().strftime("%Y%m%d")

    # Check last downloaded date
    last = get_last_date_in_csv(csv_path, sort_col) if os.path.exists(csv_path) else None
    if last:
        try:
            dt = datetime.strptime(str(last).split(".")[0], "%Y%m%d")
            fetch_start = (dt + timedelta(days=1)).strftime("%Y%m%d")
        except Exception:
            fetch_start = start_date
    else:
        fetch_start = start_date

    if str(fetch_start) > str(end_date):
        print(f"  {api_name} up-to-date")
        return

    trade_dates = get_trade_dates(pro, rl, fetch_start, end_date)
    if not trade_dates:
        print(f"  {api_name} no trade dates to fetch")
        return

    all_dfs = []
    t0 = time.time()
    for i, td in enumerate(trade_dates):
        try:
            df = safe_api_call(pro, api_name, rl, trade_date=td)
            if not df.empty:
                all_dfs.append(df)
        except Exception as e:
            print(f"  {api_name} date={td} error: {e}")

        if (i + 1) % 50 == 0:
            elapsed = int(time.time() - t0)
            print(f"  {api_name} progress {i+1}/{len(trade_dates)} elapsed={elapsed}s")

    if all_dfs:
        merged = pd.concat(all_dfs, ignore_index=True)
        save_csv(merged, csv_path, sort_col=sort_col, append=os.path.exists(csv_path))
        print(f"  {api_name} done rows={len(merged)} dates={len(trade_dates)} elapsed={int(time.time()-t0)}s")
    else:
        print(f"  {api_name} no data returned")


def download_global_dataset(pro, rl, out_dir, api_name, dir_name,
                             sort_col=None, **extra_params):
    """Download a single-call global dataset."""
    target_dir = os.path.join(out_dir, dir_name)
    os.makedirs(target_dir, exist_ok=True)
    csv_path = os.path.join(target_dir, f"{api_name}.csv")

    try:
        df = safe_api_call(pro, api_name, rl, **extra_params)
        if df.empty:
            print(f"  {api_name} no data")
            return
        save_csv(df, csv_path, sort_col=sort_col, append=False)
        print(f"  {api_name} done rows={len(df)}")
    except Exception as e:
        print(f"  {api_name} error: {e}")


def download_date_range_dataset(pro, rl, out_dir, api_name, dir_name,
                                 sort_col="trade_date", start_date="20100101",
                                 end_date=None, chunk_days=365):
    """Download dataset by date range chunks into a single CSV."""
    target_dir = os.path.join(out_dir, dir_name)
    os.makedirs(target_dir, exist_ok=True)
    csv_path = os.path.join(target_dir, f"{api_name}.csv")

    if end_date is None:
        end_date = datetime.today().strftime("%Y%m%d")

    last = get_last_date_in_csv(csv_path, sort_col) if os.path.exists(csv_path) else None
    if last:
        try:
            dt = datetime.strptime(str(last).split(".")[0], "%Y%m%d")
            fetch_start = (dt + timedelta(days=1)).strftime("%Y%m%d")
        except Exception:
            fetch_start = start_date
    else:
        fetch_start = start_date

    if str(fetch_start) > str(end_date):
        print(f"  {api_name} up-to-date")
        return

    start_dt = datetime.strptime(fetch_start, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    all_dfs = []
    t0 = time.time()
    current = start_dt

    while current <= end_dt:
        chunk_end = min(current + timedelta(days=chunk_days), end_dt)
        s = current.strftime("%Y%m%d")
        e = chunk_end.strftime("%Y%m%d")
        try:
            df = safe_api_call(pro, api_name, rl, start_date=s, end_date=e)
            if not df.empty:
                all_dfs.append(df)
        except Exception as ex:
            print(f"  {api_name} chunk {s}-{e} error: {ex}")
        current = chunk_end + timedelta(days=1)

    if all_dfs:
        merged = pd.concat(all_dfs, ignore_index=True)
        save_csv(merged, csv_path, sort_col=sort_col, append=os.path.exists(csv_path))
        print(f"  {api_name} done rows={len(merged)} elapsed={int(time.time()-t0)}s")
    else:
        print(f"  {api_name} no new data")


def download_index_daily(pro, rl, out_dir, end_date=None):
    """Download index daily data for major indices."""
    dir_name = "tushare-index-daily"
    target_dir = os.path.join(out_dir, dir_name)
    os.makedirs(target_dir, exist_ok=True)

    if end_date is None:
        end_date = datetime.today().strftime("%Y%m%d")

    # Major A-share indices
    indices = [
        "000001.SH",  # 上证指数
        "000300.SH",  # 沪深300
        "000016.SH",  # 上证50
        "000905.SH",  # 中证500
        "000852.SH",  # 中证1000
        "399001.SZ",  # 深证成指
        "399006.SZ",  # 创业板指
        "399106.SZ",  # 深证综指
        "399005.SZ",  # 中小板指
        "399673.SZ",  # 创业板50
        "000688.SH",  # 科创50
        "000015.SH",  # 红利指数
        "000010.SH",  # 上证180
        "000009.SH",  # 上证380
        "000903.SH",  # 中证100
        "000906.SH",  # 中证800
        "399303.SZ",  # 国证2000
    ]

    for ts_code in indices:
        fname = ts_code.replace(".", "_").lower() + ".csv"
        csv_path = os.path.join(target_dir, fname)

        fetch_start = "19900101"
        if os.path.exists(csv_path):
            last = get_last_date_in_csv(csv_path, "trade_date")
            if last:
                try:
                    dt = datetime.strptime(str(last).split(".")[0], "%Y%m%d")
                    fetch_start = (dt + timedelta(days=1)).strftime("%Y%m%d")
                except Exception:
                    pass
                if str(fetch_start) > str(end_date):
                    continue

        try:
            df = safe_api_call(pro, "index_daily", rl, ts_code=ts_code,
                               start_date=fetch_start, end_date=end_date)
            if not df.empty:
                save_csv(df, csv_path, sort_col="trade_date", append=os.path.exists(csv_path))
                print(f"  index_daily {ts_code} rows={len(df)}")
        except Exception as e:
            print(f"  index_daily {ts_code} error: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Download all Tushare data (2000-pt token)")
    parser.add_argument("-o", "--output-dir", required=True, help="Data root dir")
    parser.add_argument("--token", default="", help="Tushare token (env TUSHARE_TOKEN fallback)")
    parser.add_argument("--threads", type=int, default=4, help="Worker threads for per-stock tasks")
    parser.add_argument("--rate", type=int, default=180, help="API calls per minute")
    parser.add_argument("--category", default="all",
                        help="Category to download: all, stock, financial, index, fund, futures, bond, macro, market")
    parser.add_argument("--start-from", default="",
                        help="When --category=all, skip categories before this one. "
                             "Order: stock,financial,market,index,fund,futures,bond,macro,fx,hk,option")
    parser.add_argument("--start-date", default="", help="Override start date YYYYMMDD")
    parser.add_argument("--end-date", default="", help="Override end date YYYYMMDD")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip datasets already fully downloaded")
    args = parser.parse_args()

    token = args.token or os.environ.get("TUSHARE_TOKEN", "") or os.environ.get("GP_TUSHARE_TOKEN", "")
    if not token:
        print("Error: --token or TUSHARE_TOKEN env required")
        sys.exit(1)

    ts.set_token(token)
    pro = ts.pro_api()
    rl = RateLimiter(args.rate, 60)

    out_dir = args.output_dir
    end_date = args.end_date or datetime.today().strftime("%Y%m%d")
    cat = args.category.lower()
    threads = args.threads

    # --start-from: skip categories before the specified one when using --category=all
    _CAT_ORDER = ["stock", "financial", "market", "index", "fund", "futures", "bond", "macro", "fx", "hk", "option"]
    start_from = args.start_from.lower() if args.start_from else ""
    _skip_cats = set()
    if start_from and cat == "all":
        if start_from in _CAT_ORDER:
            _skip_cats = set(_CAT_ORDER[:_CAT_ORDER.index(start_from)])
            print(f"[--start-from {start_from}] Skipping categories: {', '.join(sorted(_skip_cats))}")
        else:
            print(f"Warning: unknown --start-from category '{start_from}', ignoring")

    def _should_run(category_name):
        """Check if a category should be executed based on cat and --start-from."""
        if cat != "all" and cat != category_name:
            return False
        if category_name in _skip_cats:
            return False
        return True

    # Load symbols for per-stock datasets
    symbols = []
    gplist_path = os.path.join(out_dir, "tushare_gplist.json")
    if os.path.exists(gplist_path):
        with open(gplist_path, "r") as f:
            symbols = json.load(f)
    else:
        print(f"Warning: {gplist_path} not found, per-stock datasets will be skipped")

    # ======================================================================
    # 1. STOCK DATA — per-stock iteration
    # ======================================================================
    if _should_run("stock"):
        print("\n" + "=" * 60)
        print("  STOCK DATA (per-stock)")
        print("=" * 60)

        # daily_basic — PE/PB/换手率/市值 etc. (2000分)
        print("\n[daily_basic] 每日指标 (PE/PB/换手率/总市值/流通市值)...")
        download_per_stock_dataset(pro, rl, symbols, out_dir,
                                    api_name="daily_basic", dir_name="tushare-daily_basic",
                                    sort_col="trade_date", threads=threads,
                                    start_date=args.start_date or "20040101", end_date=end_date)

        # moneyflow — 个股资金流向 (2000分)
        print("\n[moneyflow] 个股资金流向...")
        download_per_stock_dataset(pro, rl, symbols, out_dir,
                                    api_name="moneyflow", dir_name="tushare-moneyflow",
                                    sort_col="trade_date", threads=threads,
                                    start_date=args.start_date or "20100101", end_date=end_date)

        # stk_holdernumber — 股东人数 (2000分)
        print("\n[stk_holdernumber] 股东人数...")
        download_per_stock_dataset(pro, rl, symbols, out_dir,
                                    api_name="stk_holdernumber", dir_name="tushare-stk_holdernumber",
                                    sort_col="end_date", threads=threads,
                                    start_date=args.start_date or "20100101", end_date=end_date)

    # ======================================================================
    # 2. FINANCIAL DATA — per-stock iteration
    # ======================================================================
    if _should_run("financial"):
        print("\n" + "=" * 60)
        print("  FINANCIAL DATA (per-stock)")
        print("=" * 60)

        # forecast — 业绩预告 (2000分)
        print("\n[forecast] 业绩预告...")
        download_per_stock_dataset(pro, rl, symbols, out_dir,
                                    api_name="forecast", dir_name="tushare-forecast",
                                    sort_col="ann_date", threads=threads,
                                    start_date=args.start_date or "20000101", end_date=end_date)

        # express — 业绩快报 (2000分)
        print("\n[express] 业绩快报...")
        download_per_stock_dataset(pro, rl, symbols, out_dir,
                                    api_name="express", dir_name="tushare-express",
                                    sort_col="ann_date", threads=threads,
                                    start_date=args.start_date or "20000101", end_date=end_date)

        # fina_audit — 财务审计意见 (2000分)
        print("\n[fina_audit] 财务审计意见...")
        download_per_stock_dataset(pro, rl, symbols, out_dir,
                                    api_name="fina_audit", dir_name="tushare-fina_audit",
                                    sort_col="ann_date", threads=threads,
                                    start_date=args.start_date or "20000101", end_date=end_date)

        # fina_mainbz — 主营业务构成 (2000分)
        print("\n[fina_mainbz] 主营业务构成...")
        download_per_stock_dataset(pro, rl, symbols, out_dir,
                                    api_name="fina_mainbz", dir_name="tushare-fina_mainbz",
                                    sort_col="end_date", threads=threads,
                                    start_date=args.start_date or "20000101", end_date=end_date)

        # disclosure_date — 财报披露计划 (2000分)
        print("\n[disclosure_date] 财报披露计划...")
        download_per_stock_dataset(pro, rl, symbols, out_dir,
                                    api_name="disclosure_date", dir_name="tushare-disclosure_date",
                                    sort_col="end_date", threads=threads,
                                    start_date=args.start_date or "20100101", end_date=end_date)

    # ======================================================================
    # 3. MARKET DATA — per-date iteration (trade_date param)
    # ======================================================================
    if _should_run("market"):
        print("\n" + "=" * 60)
        print("  MARKET DATA (per-date)")
        print("=" * 60)

        # top_list — 龙虎榜 (2000分)
        print("\n[top_list] 龙虎榜每日明细...")
        download_per_date_dataset(pro, rl, out_dir,
                                   api_name="top_list", dir_name="tushare-top_list",
                                   sort_col="trade_date",
                                   start_date=args.start_date or "20050101", end_date=end_date)

        # top_inst — 龙虎榜机构 (2000分)
        print("\n[top_inst] 龙虎榜机构交易明细...")
        download_per_date_dataset(pro, rl, out_dir,
                                   api_name="top_inst", dir_name="tushare-top_inst",
                                   sort_col="trade_date",
                                   start_date=args.start_date or "20050101", end_date=end_date)

        # margin — 融资融券汇总 (2000分)
        print("\n[margin] 融资融券交易汇总...")
        download_per_date_dataset(pro, rl, out_dir,
                                   api_name="margin", dir_name="tushare-margin",
                                   sort_col="trade_date",
                                   start_date=args.start_date or "20100101", end_date=end_date)

        # margin_detail — 融资融券明细 (2000分)
        print("\n[margin_detail] 融资融券交易明细...")
        download_per_date_dataset(pro, rl, out_dir,
                                   api_name="margin_detail", dir_name="tushare-margin_detail",
                                   sort_col="trade_date",
                                   start_date=args.start_date or "20100101", end_date=end_date)

        # hk_hold — 沪深股通持股明细 (2000分)
        print("\n[hk_hold] 沪深股通持股明细...")
        download_per_date_dataset(pro, rl, out_dir,
                                   api_name="hk_hold", dir_name="tushare-hk_hold",
                                   sort_col="trade_date",
                                   start_date=args.start_date or "20170301", end_date=end_date)

        # block_trade — 大宗交易 (2000分)
        print("\n[block_trade] 大宗交易...")
        download_per_date_dataset(pro, rl, out_dir,
                                   api_name="block_trade", dir_name="tushare-block_trade",
                                   sort_col="trade_date",
                                   start_date=args.start_date or "20100101", end_date=end_date)

        # stk_holdertrade — 股东增减持 (2000分)
        print("\n[stk_holdertrade] 股东增减持...")
        download_per_date_dataset(pro, rl, out_dir,
                                   api_name="stk_holdertrade", dir_name="tushare-stk_holdertrade",
                                   sort_col="ann_date",
                                   start_date=args.start_date or "20100101", end_date=end_date)

        # repurchase — 股票回购 (2000分)
        print("\n[repurchase] 股票回购...")
        download_date_range_dataset(pro, rl, out_dir,
                                     api_name="repurchase", dir_name="tushare-repurchase",
                                     sort_col="ann_date",
                                     start_date=args.start_date or "20110101", end_date=end_date,
                                     chunk_days=365)

        # pledge_stat — 股权质押统计 (2000分)
        print("\n[pledge_stat] 股权质押统计...")
        download_per_date_dataset(pro, rl, out_dir,
                                   api_name="pledge_stat", dir_name="tushare-pledge_stat",
                                   sort_col="end_date",
                                   start_date=args.start_date or "20140101", end_date=end_date)

        # pledge_detail — 股权质押明细 (需要 ts_code 参数, 跳过)
        # print("\n[pledge_detail] 股权质押明细...")
        # download_per_date_dataset(pro, rl, out_dir,
        #                            api_name="pledge_detail", dir_name="tushare-pledge_detail",
        #                            sort_col="ann_date",
        #                            start_date=args.start_date or "20040101", end_date=end_date)

        # new_share — IPO新股列表 (120分)
        print("\n[new_share] IPO新股列表...")
        download_global_dataset(pro, rl, out_dir,
                                 api_name="new_share", dir_name="tushare-new_share",
                                 sort_col="ipo_date")

    # ======================================================================
    # 4. INDEX DATA
    # ======================================================================
    if _should_run("index"):
        print("\n" + "=" * 60)
        print("  INDEX DATA")
        print("=" * 60)

        # index_basic — 指数基本信息 (2000分)
        print("\n[index_basic] 指数基本信息...")
        for market in ["SSE", "SZSE", "CSI", "CICC", "SW", "OTH"]:
            download_global_dataset(pro, rl, out_dir,
                                     api_name="index_basic", dir_name="tushare-index_basic",
                                     sort_col="list_date", market=market)

        # index_daily — 指数日线 (major indices)
        print("\n[index_daily] 指数日线行情...")
        download_index_daily(pro, rl, out_dir, end_date=end_date)

        # index_weekly — 指数周线 (2000分)
        print("\n[index_weekly] 指数周线行情...")
        for ts_code in ["000001.SH", "000300.SH", "399001.SZ", "399006.SZ", "000905.SH"]:
            target_dir = os.path.join(out_dir, "tushare-index_weekly")
            os.makedirs(target_dir, exist_ok=True)
            fname = ts_code.replace(".", "_").lower() + ".csv"
            csv_path = os.path.join(target_dir, fname)
            try:
                df = safe_api_call(pro, "index_weekly", rl, ts_code=ts_code,
                                    start_date="19900101", end_date=end_date)
                if not df.empty:
                    save_csv(df, csv_path, sort_col="trade_date")
                    print(f"  index_weekly {ts_code} rows={len(df)}")
            except Exception as e:
                print(f"  index_weekly {ts_code} error: {e}")

        # index_monthly — 指数月线 (2000分)
        print("\n[index_monthly] 指数月线行情...")
        for ts_code in ["000001.SH", "000300.SH", "399001.SZ", "399006.SZ", "000905.SH"]:
            target_dir = os.path.join(out_dir, "tushare-index_monthly")
            os.makedirs(target_dir, exist_ok=True)
            fname = ts_code.replace(".", "_").lower() + ".csv"
            csv_path = os.path.join(target_dir, fname)
            try:
                df = safe_api_call(pro, "index_monthly", rl, ts_code=ts_code,
                                    start_date="19900101", end_date=end_date)
                if not df.empty:
                    save_csv(df, csv_path, sort_col="trade_date")
                    print(f"  index_monthly {ts_code} rows={len(df)}")
            except Exception as e:
                print(f"  index_monthly {ts_code} error: {e}")

        # index_weight — 指数成分和权重 (2000分)
        print("\n[index_weight] 指数成分和权重...")
        for idx_code in ["000300.SH", "000905.SH", "000852.SH", "000016.SH"]:
            target_dir = os.path.join(out_dir, "tushare-index_weight")
            os.makedirs(target_dir, exist_ok=True)
            fname = idx_code.replace(".", "_").lower() + ".csv"
            csv_path = os.path.join(target_dir, fname)
            try:
                df = safe_api_call(pro, "index_weight", rl, index_code=idx_code,
                                    start_date="20050101", end_date=end_date)
                if not df.empty:
                    save_csv(df, csv_path, sort_col="trade_date")
                    print(f"  index_weight {idx_code} rows={len(df)}")
            except Exception as e:
                print(f"  index_weight {idx_code} error: {e}")

        # index_classify — 申万行业分类 (2000分)
        print("\n[index_classify] 申万行业分类...")
        download_global_dataset(pro, rl, out_dir,
                                 api_name="index_classify", dir_name="tushare-index_classify",
                                 sort_col=None, src="SW2021")

        # index_member_all — 申万行业成分 (2000分)
        print("\n[index_member_all] 申万行业成分...")
        download_global_dataset(pro, rl, out_dir,
                                 api_name="index_member_all", dir_name="tushare-index_member_all",
                                 sort_col=None)

    # ======================================================================
    # 5. FUND DATA
    # ======================================================================
    if _should_run("fund"):
        print("\n" + "=" * 60)
        print("  FUND DATA (skipped)")
        print("=" * 60)
        pass  # 基金数据跳过

    # ======================================================================
    # 6. FUTURES DATA
    # ======================================================================
    if _should_run("futures"):
        print("\n" + "=" * 60)
        print("  FUTURES DATA")
        print("=" * 60)

        # fut_basic — 期货合约列表 (2000分)
        print("\n[fut_basic] 期货合约列表...")
        for exchange in ["CFFEX", "DCE", "CZCE", "SHFE", "INE", "GFEX"]:
            download_global_dataset(pro, rl, out_dir,
                                     api_name="fut_basic", dir_name="tushare-fut_basic",
                                     sort_col=None, exchange=exchange)

        # fut_daily — 期货日线行情 (2000分) — by date range
        print("\n[fut_daily] 期货日线行情...")
        download_date_range_dataset(pro, rl, out_dir,
                                     api_name="fut_daily", dir_name="tushare-fut_daily",
                                     sort_col="trade_date",
                                     start_date=args.start_date or "20200101", end_date=end_date,
                                     chunk_days=30)

        # fut_holding — 每日成交持仓排名 (2000分)
        print("\n[fut_holding] 每日成交持仓排名...")
        download_per_date_dataset(pro, rl, out_dir,
                                     api_name="fut_holding", dir_name="tushare-fut_holding",
                                     sort_col="trade_date",
                                     start_date=args.start_date or "20200101", end_date=end_date)

        # fut_wsr — 仓单日报 (2000分)
        print("\n[fut_wsr] 仓单日报...")
        download_per_date_dataset(pro, rl, out_dir,
                                     api_name="fut_wsr", dir_name="tushare-fut_wsr",
                                     sort_col="trade_date",
                                     start_date=args.start_date or "20200101", end_date=end_date)

        # fut_settle — 结算参数 (2000分)
        print("\n[fut_settle] 期货结算参数...")
        download_per_date_dataset(pro, rl, out_dir,
                                     api_name="fut_settle", dir_name="tushare-fut_settle",
                                     sort_col="trade_date",
                                     start_date=args.start_date or "20200101", end_date=end_date)

    # ======================================================================
    # 7. BOND DATA
    # ======================================================================
    if _should_run("bond"):
        print("\n" + "=" * 60)
        print("  BOND DATA (可转债)")
        print("=" * 60)

        # cb_basic — 可转债基础信息 (2000分)
        print("\n[cb_basic] 可转债基础信息...")
        download_global_dataset(pro, rl, out_dir,
                                 api_name="cb_basic", dir_name="tushare-cb_basic",
                                 sort_col=None)

        # cb_issue — 可转债发行数据 (2000分)
        print("\n[cb_issue] 可转债发行数据...")
        download_global_dataset(pro, rl, out_dir,
                                 api_name="cb_issue", dir_name="tushare-cb_issue",
                                 sort_col=None)

        # cb_daily — 可转债日线 (2000分)
        print("\n[cb_daily] 可转债日线数据...")
        download_date_range_dataset(pro, rl, out_dir,
                                     api_name="cb_daily", dir_name="tushare-cb_daily",
                                     sort_col="trade_date",
                                     start_date=args.start_date or "20180101", end_date=end_date,
                                     chunk_days=30)

    # ======================================================================
    # 8. MACRO DATA
    # ======================================================================
    if _should_run("macro"):
        print("\n" + "=" * 60)
        print("  MACRO DATA")
        print("=" * 60)

        # shibor — SHIBOR利率 (2000分)
        print("\n[shibor] SHIBOR利率...")
        download_date_range_dataset(pro, rl, out_dir,
                                     api_name="shibor", dir_name="tushare-shibor",
                                     sort_col="date",
                                     start_date=args.start_date or "20060101", end_date=end_date,
                                     chunk_days=365)

        # shibor_quote — SHIBOR报价 (2000分)
        print("\n[shibor_quote] SHIBOR报价数据...")
        download_date_range_dataset(pro, rl, out_dir,
                                     api_name="shibor_quote", dir_name="tushare-shibor_quote",
                                     sort_col="date",
                                     start_date=args.start_date or "20060101", end_date=end_date,
                                     chunk_days=365)

        # shibor_lpr — LPR贷款基础利率 (120分)
        print("\n[shibor_lpr] LPR贷款基础利率...")
        download_date_range_dataset(pro, rl, out_dir,
                                     api_name="shibor_lpr", dir_name="tushare-shibor_lpr",
                                     sort_col="date",
                                     start_date=args.start_date or "20130101", end_date=end_date,
                                     chunk_days=365)

        # libor — LIBOR拆借利率 (120分)
        print("\n[libor] LIBOR拆借利率...")
        download_date_range_dataset(pro, rl, out_dir,
                                     api_name="libor", dir_name="tushare-libor",
                                     sort_col="date",
                                     start_date=args.start_date or "19860101", end_date=end_date,
                                     chunk_days=365)

        # hibor — HIBOR拆借利率 (120分)
        print("\n[hibor] HIBOR拆借利率...")
        download_date_range_dataset(pro, rl, out_dir,
                                     api_name="hibor", dir_name="tushare-hibor",
                                     sort_col="date",
                                     start_date=args.start_date or "20020101", end_date=end_date,
                                     chunk_days=365)

    # ======================================================================
    # 9. FX DATA
    # ======================================================================
    if _should_run("fx"):
        print("\n" + "=" * 60)
        print("  FX DATA (外汇)")
        print("=" * 60)

        # fx_obasic — 外汇基础信息 (2000分)
        print("\n[fx_obasic] 外汇基础信息...")
        download_global_dataset(pro, rl, out_dir,
                                 api_name="fx_obasic", dir_name="tushare-fx",
                                 sort_col=None, exchange="FXCM")

        # fx_daily — 外汇日线 (2000分)
        print("\n[fx_daily] 外汇日线行情...")
        target_dir = os.path.join(out_dir, "tushare-fx")
        os.makedirs(target_dir, exist_ok=True)
        fx_pairs = ["USDCNY", "USDCNH", "USDJPY", "EURUSD", "GBPUSD"]
        for pair in fx_pairs:
            csv_path = os.path.join(target_dir, f"fx_{pair.lower()}.csv")
            try:
                df = safe_api_call(pro, "fx_daily", rl, ts_code=pair,
                                    start_date="20100101", end_date=end_date)
                if not df.empty:
                    save_csv(df, csv_path, sort_col="trade_date")
                    print(f"  fx_daily {pair} rows={len(df)}")
            except Exception as e:
                print(f"  fx_daily {pair} error: {e}")

    # ======================================================================
    # 10. HK STOCK DATA
    # ======================================================================
    if _should_run("hk"):
        print("\n" + "=" * 60)
        print("  HK STOCK DATA")
        print("=" * 60)

        # hk_basic — 港股列表 (2000分)
        print("\n[hk_basic] 港股列表...")
        download_global_dataset(pro, rl, out_dir,
                                 api_name="hk_basic", dir_name="tushare-hk_basic",
                                 sort_col=None)

    # ======================================================================
    # 11. OPTION DATA
    # ======================================================================
    if _should_run("option"):
        print("\n" + "=" * 60)
        print("  OPTION DATA")
        print("=" * 60)

        # opt_basic — 期权合约列表 (2000分)
        print("\n[opt_basic] 期权合约列表...")
        for exchange in ["SSE", "SZSE", "CFFEX", "DCE", "CZCE", "SHFE"]:
            download_global_dataset(pro, rl, out_dir,
                                     api_name="opt_basic", dir_name="tushare-opt_basic",
                                     sort_col=None, exchange=exchange)

    print("\n" + "=" * 60)
    print("  ALL DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
