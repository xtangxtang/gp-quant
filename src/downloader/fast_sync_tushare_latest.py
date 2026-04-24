import argparse
import csv
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import pandas as pd
import tushare as ts


STOCK_BASIC_FIELDS = "ts_code,symbol,name,area,industry,market,list_date"

DATE_BATCH_CONFIG = {
    "daily_full": {
        "dir_name": "tushare-daily-full",
        "date_col": "trade_date",
    },
    "adj_factor": {
        "dir_name": "tushare-adj_factor",
        "date_col": "trade_date",
    },
    "stk_limit": {
        "dir_name": "tushare-stk_limit",
        "date_col": "trade_date",
    },
    "suspend_d": {
        "dir_name": "tushare-suspend_d",
        "date_col": "trade_date",
    },
    "dividend": {
        "dir_name": "tushare-dividend",
        "date_col": "ann_date",
    },
}

FINANCIAL_DATASETS = {
    "income": {
        "dir_name": "tushare-income",
        "date_col": "ann_date",
    },
    "balancesheet": {
        "dir_name": "tushare-balancesheet",
        "date_col": "ann_date",
    },
    "cashflow": {
        "dir_name": "tushare-cashflow",
        "date_col": "ann_date",
    },
    "fina_indicator": {
        "dir_name": "tushare-fina_indicator",
        "date_col": "ann_date",
    },
}


class RateLimiter:
    """Thread-safe sliding window rate limiter with dynamic server-side backoff."""

    RATE_LIMIT_ERROR_MIN = 20
    RATE_LIMIT_RECOVERY_STEP = 2

    def __init__(self, max_calls: int, period: float):
        self.initial_max = int(max_calls)
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

    def reduce_rate(self) -> None:
        """Halve the rate on server-side rate limit errors, with a floor."""
        with self.lock:
            new_max = max(self.max_calls // 2, self.RATE_LIMIT_ERROR_MIN)
            if new_max != self.max_calls:
                print(f"[RateLimiter] rate limit error: max_calls {self.max_calls} -> {new_max}")
            self.max_calls = new_max
            # Flush stale calls from the window to immediately throttle
            cutoff = time.time() - self.period
            self.calls = [t for t in self.calls if t > cutoff]

    def recover_rate(self) -> None:
        """Gradually increase the rate toward the initial value on sustained success."""
        with self.lock:
            if self.max_calls < self.initial_max:
                self.max_calls = min(self.max_calls + self.RATE_LIMIT_RECOVERY_STEP, self.initial_max)


def convert_ts_code_to_symbol(ts_code: str) -> str:
    code, market = ts_code.split(".")
    return f"{market.lower()}{code}"


def detect_last_date(path: str, date_col: str) -> str | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if not header:
                return None
            resolved_date_col = resolve_date_col_name(header, date_col)
            if resolved_date_col not in header:
                return None
            date_idx = header.index(resolved_date_col)
            last_value = None
            for row in reader:
                if len(row) > date_idx and row[date_idx]:
                    last_value = str(row[date_idx]).strip()
            return last_value
    except Exception:
        return None


def build_last_date_map(directory: str, date_col: str) -> dict[str, str]:
    result: dict[str, str] = {}
    if not os.path.isdir(directory):
        return result
    for name in os.listdir(directory):
        if not name.endswith(".csv"):
            continue
        symbol = os.path.splitext(name)[0]
        path = os.path.join(directory, name)
        last_date = detect_last_date(path, date_col)
        if last_date:
            result[symbol] = str(last_date)
    return result


def resolve_date_col_name(columns: list[str] | pd.Index, preferred: str) -> str:
    values = list(columns)
    if preferred in values:
        return preferred
    aliases = {
        "trade_date": ["suspend_date"],
        "suspend_date": ["trade_date"],
    }
    for alias in aliases.get(preferred, []):
        if alias in values:
            return alias
    return preferred


def ensure_trade_cal(pro, output_dir: str) -> None:
    target_dir = os.path.join(output_dir, "tushare-trade_cal")
    os.makedirs(target_dir, exist_ok=True)
    path = os.path.join(target_dir, "trade_cal.csv")

    today = datetime.today().strftime("%Y%m%d")
    start_date = "19900101"
    last_cal_date = detect_last_date(path, "cal_date")
    if last_cal_date:
        try:
            start_date = (datetime.strptime(last_cal_date, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
        except Exception:
            start_date = str(last_cal_date)

    if start_date > "20261231":
        return

    df = pro.trade_cal(exchange="", start_date=start_date, end_date="20261231")
    if df is None or df.empty:
        return

    if os.path.exists(path):
        old = pd.read_csv(path)
        merged = pd.concat([old, df], ignore_index=True).drop_duplicates()
        merged = merged.sort_values(["exchange", "cal_date"], ascending=[True, True])
        merged.to_csv(path, index=False)
    else:
        df.sort_values(["exchange", "cal_date"], ascending=[True, True]).to_csv(path, index=False)


def infer_last_available_trade_date(pro, today_yyyymmdd: str) -> str:
    end_dt = datetime.strptime(today_yyyymmdd, "%Y%m%d")
    start_dt = end_dt - timedelta(days=30)
    cal = pro.trade_cal(exchange="SSE", start_date=start_dt.strftime("%Y%m%d"), end_date=today_yyyymmdd)
    if cal is None or cal.empty:
        return today_yyyymmdd
    open_dates = cal.loc[cal["is_open"].astype(int) == 1, "cal_date"].astype(str).sort_values().tolist()
    for date_value in reversed(open_dates):
        df = pro.daily(trade_date=str(date_value))
        if df is not None and not df.empty:
            return str(date_value)
    return today_yyyymmdd


def fetch_with_retry(pro, limiter: RateLimiter, dataset: str, **params) -> pd.DataFrame:
    max_retries = 8
    func = getattr(pro, dataset)
    for attempt in range(1, max_retries + 1):
        try:
            limiter.wait()
            cleaned = {k: v for k, v in params.items() if v is not None and str(v) != ""}
            df = func(**cleaned)
            limiter.recover_rate()
            return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        except Exception as exc:
            message = str(exc)
            is_rate_limit = (
                "频率超限" in message
                or "每分钟最多访问" in message
                or "访问过于频繁" in message
            )
            if is_rate_limit:
                limiter.reduce_rate()
                sleep_seconds = min(60, 15 + attempt * 5)
            else:
                sleep_seconds = min(20, 2 + attempt * 2)
            print(f"retry dataset={dataset} params={cleaned} attempt={attempt}/{max_retries} sleep={sleep_seconds}s err={message}")
            time.sleep(sleep_seconds)
    raise RuntimeError(f"dataset={dataset} params={params} failed after retries")


def append_grouped_rows(df: pd.DataFrame, target_dir: str, date_col: str, last_date_map: dict[str, str]) -> tuple[int, int]:
    if df is None or df.empty:
        return 0, 0

    resolved_date_col = resolve_date_col_name(df.columns, date_col)
    if resolved_date_col not in df.columns:
        raise KeyError(date_col)

    written_files = 0
    written_rows = 0
    for ts_code, group in df.groupby("ts_code"):
        symbol = convert_ts_code_to_symbol(str(ts_code))
        path = os.path.join(target_dir, f"{symbol}.csv")
        group = group.sort_values(resolved_date_col, ascending=True)
        last_date = str(last_date_map.get(symbol) or "")
        if last_date:
            group = group[group[resolved_date_col].astype(str) > last_date]
        if group.empty:
            continue
        header = not os.path.exists(path)
        group.to_csv(path, mode="a" if not header else "w", header=header, index=False)
        last_date_map[symbol] = str(group[resolved_date_col].astype(str).max())
        written_files += 1
        written_rows += len(group)
    return written_files, written_rows


def update_stock_basic_and_list(pro, output_dir: str) -> list[str]:
    df = pro.stock_basic(exchange="", list_status="L", fields=STOCK_BASIC_FIELDS)
    if df is None or df.empty:
        raise RuntimeError("stock_basic returned no data")
    df = df.sort_values(["ts_code"], ascending=[True]).reset_index(drop=True)
    stock_basic_path = os.path.join(output_dir, "tushare_stock_basic.csv")
    df.to_csv(stock_basic_path, index=False)

    symbols = [convert_ts_code_to_symbol(ts_code) for ts_code in df["ts_code"].astype(str).tolist()]
    list_path = os.path.join(output_dir, "tushare_gplist.json")
    with open(list_path, "w", encoding="utf-8") as handle:
        json.dump(symbols, handle, indent=2, ensure_ascii=False)
    return symbols


def update_daily_full(pro, output_dir: str, trade_dates: list[str], limiter: RateLimiter) -> None:
    if not trade_dates:
        print("daily_full already up-to-date")
        return

    target_dir = os.path.join(output_dir, DATE_BATCH_CONFIG["daily_full"]["dir_name"])
    os.makedirs(target_dir, exist_ok=True)
    last_date_map = build_last_date_map(target_dir, "trade_date")

    total_files = 0
    total_rows = 0
    for trade_date in trade_dates:
        df_daily = fetch_with_retry(pro, limiter, "daily", trade_date=trade_date)
        df_basic = fetch_with_retry(pro, limiter, "daily_basic", trade_date=trade_date)
        df_moneyflow = fetch_with_retry(pro, limiter, "moneyflow", trade_date=trade_date)

        if df_daily is None or df_daily.empty:
            print(f"daily_full trade_date={trade_date} no rows")
            continue

        merged = df_daily.copy()
        if df_basic is not None and not df_basic.empty:
            cols = df_basic.columns.difference(merged.columns).tolist() + ["ts_code", "trade_date"]
            merged = pd.merge(merged, df_basic[cols], on=["ts_code", "trade_date"], how="left")
        if df_moneyflow is not None and not df_moneyflow.empty:
            cols = df_moneyflow.columns.difference(merged.columns).tolist() + ["ts_code", "trade_date"]
            merged = pd.merge(merged, df_moneyflow[cols], on=["ts_code", "trade_date"], how="left")

        files_written, rows_written = append_grouped_rows(merged, target_dir, "trade_date", last_date_map)
        total_files += files_written
        total_rows += rows_written
        print(f"daily_full trade_date={trade_date} files_written={files_written} rows_written={rows_written}")

    print(f"daily_full finished files_written={total_files} rows_written={total_rows}")


def update_trade_date_dataset(pro, output_dir: str, dataset: str, trade_dates: list[str], limiter: RateLimiter) -> None:
    if not trade_dates:
        print(f"{dataset} already up-to-date")
        return

    cfg = DATE_BATCH_CONFIG[dataset]
    target_dir = os.path.join(output_dir, cfg["dir_name"])
    os.makedirs(target_dir, exist_ok=True)
    last_date_map = build_last_date_map(target_dir, cfg["date_col"])

    total_files = 0
    total_rows = 0
    for trade_date in trade_dates:
        df = fetch_with_retry(pro, limiter, dataset, trade_date=trade_date)
        if df is None or df.empty:
            print(f"{dataset} trade_date={trade_date} no rows")
            continue
        files_written, rows_written = append_grouped_rows(df, target_dir, cfg["date_col"], last_date_map)
        total_files += files_written
        total_rows += rows_written
        print(f"{dataset} trade_date={trade_date} files_written={files_written} rows_written={rows_written}")

    print(f"{dataset} finished files_written={total_files} rows_written={total_rows}")


def update_dividend(pro, output_dir: str, start_ann_date: str, end_ann_date: str, limiter: RateLimiter) -> None:
    if start_ann_date > end_ann_date:
        print("dividend already up-to-date")
        return

    target_dir = os.path.join(output_dir, DATE_BATCH_CONFIG["dividend"]["dir_name"])
    os.makedirs(target_dir, exist_ok=True)
    last_date_map = build_last_date_map(target_dir, "ann_date")

    current = datetime.strptime(start_ann_date, "%Y%m%d")
    end_dt = datetime.strptime(end_ann_date, "%Y%m%d")
    total_files = 0
    total_rows = 0
    while current <= end_dt:
        ann_date = current.strftime("%Y%m%d")
        df = fetch_with_retry(pro, limiter, "dividend", ann_date=ann_date)
        if df is not None and not df.empty:
            files_written, rows_written = append_grouped_rows(df, target_dir, "ann_date", last_date_map)
            total_files += files_written
            total_rows += rows_written
            print(f"dividend ann_date={ann_date} files_written={files_written} rows_written={rows_written}")
        current += timedelta(days=1)
    print(f"dividend finished files_written={total_files} rows_written={total_rows}")


def update_financial_dataset(
    pro,
    output_dir: str,
    dataset: str,
    symbols: list[str],
    start_date: str,
    end_date: str,
    limiter: RateLimiter,
    threads: int,
) -> None:
    cfg = FINANCIAL_DATASETS[dataset]
    target_dir = os.path.join(output_dir, cfg["dir_name"])
    os.makedirs(target_dir, exist_ok=True)
    last_date_map = build_last_date_map(target_dir, cfg["date_col"])

    def task(symbol: str) -> tuple[str, str]:
        ts_code = symbol[2:] + "." + symbol[:2].upper()
        path = os.path.join(target_dir, f"{symbol}.csv")
        last_ann_date = last_date_map.get(symbol)
        query_start = start_date
        if last_ann_date:
            query_start = str(last_ann_date)
        try:
            df = fetch_with_retry(pro, limiter, dataset, ts_code=ts_code, start_date=query_start, end_date=end_date)
        except Exception as exc:
            return symbol, f"failed: {exc}"

        if df is None or df.empty:
            return symbol, "up-to-date"

        if cfg["date_col"] in df.columns:
            df[cfg["date_col"]] = df[cfg["date_col"]].astype(str)
            df = df.sort_values(cfg["date_col"], ascending=True)

        if os.path.exists(path):
            try:
                old = pd.read_csv(path)
                merged = pd.concat([old, df], ignore_index=True).drop_duplicates()
                if cfg["date_col"] in merged.columns:
                    merged[cfg["date_col"]] = merged[cfg["date_col"]].astype(str)
                    merged = merged.sort_values(cfg["date_col"], ascending=True)
                merged.to_csv(path, index=False)
            except Exception:
                df.to_csv(path, index=False)
        else:
            df.to_csv(path, index=False)

        last_value = None
        if cfg["date_col"] in df.columns and not df.empty:
            last_value = str(df[cfg["date_col"]].astype(str).max())
        if last_value:
            last_date_map[symbol] = last_value
        return symbol, f"updated {len(df)} rows"

    ok = 0
    up_to_date = 0
    failed = 0
    start_ts = time.time()

    with ThreadPoolExecutor(max_workers=max(1, int(threads))) as executor:
        futures = {executor.submit(task, symbol): symbol for symbol in symbols}
        for index, future in enumerate(as_completed(futures), 1):
            symbol, status = future.result()
            if status.startswith("updated"):
                ok += 1
            elif status == "up-to-date":
                up_to_date += 1
            else:
                failed += 1
                print(f"{dataset} {symbol} {status}")
            if index % 200 == 0 or index == len(symbols):
                elapsed = int(time.time() - start_ts)
                print(
                    f"{dataset} progress {index}/{len(symbols)} ok={ok} up_to_date={up_to_date} failed={failed} elapsed_s={elapsed}"
                )

    print(f"{dataset} finished ok={ok} up_to_date={up_to_date} failed={failed}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast latest-data sync for gp-data Tushare datasets")
    parser.add_argument("--output-dir", required=True, help="gp-data root directory")
    parser.add_argument("--token", required=True, help="Tushare token")
    parser.add_argument("--financial-threads", type=int, default=16, help="Thread count for per-symbol financial sync")
    parser.add_argument("--batch-rate", type=int, default=240, help="Max batch API calls per 60 seconds")
    parser.add_argument("--financial-rate", type=int, default=180, help="Max financial API calls per 60 seconds")
    parser.add_argument("--backfill-open-days", type=int, default=0, help="Force-refresh the latest N open trading days across all per-symbol date-based datasets")
    parser.add_argument("--skip-date-based", action="store_true", help="Skip stock_basic, trade_cal, daily_full, adj_factor, stk_limit, suspend_d, and dividend")
    parser.add_argument("--skip-financials", action="store_true", help="Skip per-symbol financial datasets")
    parser.add_argument("--financial-datasets", default="income,balancesheet,cashflow,fina_indicator", help="Comma-separated financial datasets to run")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    ts.set_token(args.token)
    pro = ts.pro_api()

    symbols = update_stock_basic_and_list(pro, output_dir)
    print(f"stock_basic and gplist updated symbols={len(symbols)}")

    selected_financials = [
        item.strip() for item in args.financial_datasets.split(",") if item.strip() in FINANCIAL_DATASETS
    ]
    if not selected_financials and not args.skip_financials:
        raise RuntimeError("No valid financial datasets selected")

    today = datetime.today().strftime("%Y%m%d")
    last_available_trade_date = infer_last_available_trade_date(pro, today)
    print(f"last_available_trade_date={last_available_trade_date}")

    if not args.skip_date_based:
        ensure_trade_cal(pro, output_dir)
        print("trade_cal updated")

        trade_dates: list[str] = []
        if int(args.backfill_open_days) > 0:
            cal = pro.trade_cal(exchange="SSE", start_date="19900101", end_date=last_available_trade_date)
            if cal is not None and not cal.empty:
                open_dates = cal.loc[cal["is_open"].astype(int) == 1, "cal_date"].astype(str).sort_values().tolist()
                trade_dates = open_dates[-int(args.backfill_open_days) :]
        else:
            daily_dir = os.path.join(output_dir, DATE_BATCH_CONFIG["daily_full"]["dir_name"])
            daily_last_date_map = build_last_date_map(daily_dir, "trade_date")
            current_daily_max = max(daily_last_date_map.values()) if daily_last_date_map else "19900101"
            start_trade_date = (datetime.strptime(current_daily_max, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
            if start_trade_date <= last_available_trade_date:
                cal = pro.trade_cal(exchange="SSE", start_date=start_trade_date, end_date=last_available_trade_date)
                if cal is not None and not cal.empty:
                    trade_dates = (
                        cal.loc[cal["is_open"].astype(int) == 1, "cal_date"].astype(str).sort_values().tolist()
                    )
        print(f"trade_dates_to_sync={trade_dates}")

        batch_limiter = RateLimiter(args.batch_rate, 60)
        update_daily_full(pro, output_dir, trade_dates, batch_limiter)
        update_trade_date_dataset(pro, output_dir, "adj_factor", trade_dates, batch_limiter)
        update_trade_date_dataset(pro, output_dir, "stk_limit", trade_dates, batch_limiter)
        update_trade_date_dataset(pro, output_dir, "suspend_d", trade_dates, batch_limiter)

        dividend_dir = os.path.join(output_dir, DATE_BATCH_CONFIG["dividend"]["dir_name"])
        dividend_last_map = build_last_date_map(dividend_dir, "ann_date")
        current_dividend_max = max(dividend_last_map.values()) if dividend_last_map else "19900101"
        start_ann_date = (datetime.strptime(current_dividend_max, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
        update_dividend(pro, output_dir, start_ann_date, today, batch_limiter)

    if not args.skip_financials:
        financial_limiter = RateLimiter(args.financial_rate, 60)
        financial_start_date = (datetime.strptime(today, "%Y%m%d") - timedelta(days=20)).strftime("%Y%m%d")
        for dataset in selected_financials:
            update_financial_dataset(
                pro,
                output_dir,
                dataset,
                symbols,
                financial_start_date,
                today,
                financial_limiter,
                args.financial_threads,
            )


if __name__ == "__main__":
    main()