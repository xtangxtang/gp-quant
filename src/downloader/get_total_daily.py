import argparse
import json
import os
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import time

import chinese_calendar as calendar
import pandas as pd
import requests


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from eastmoney_universe import fetch_all_symbols_eastmoney, fetch_listing_date_eastmoney, last_trading_day
from downloader_common import ProxyConnectivityError, run_tasks_in_threads


def _load_total_list(path: str) -> list[str] | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and isinstance(obj.get("symbols"), list):
            return obj["symbols"]
    except Exception:
        return None
    return None


def _ensure_total_list(path: str) -> list[str]:
    asof = last_trading_day()
    cached = _load_total_list(path) if os.path.exists(path) else None
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict) and obj.get("asof") == asof and isinstance(obj.get("symbols"), list):
                return obj["symbols"]
        except Exception:
            pass

    try:
        payload = fetch_all_symbols_eastmoney()
    except requests.exceptions.ProxyError as e:
        if cached:
            print(f"[WARN] ProxyError while refreshing total list; using cached symbols from {path}")
            return cached
        print(f"[FATAL] ProxyError while fetching total list: {e}")
        raise SystemExit(86)
    except Exception as e:
        if cached:
            print(f"[WARN] Failed to refresh total list; using cached symbols from {path}: {type(e).__name__}: {str(e)[:160]}")
            return cached
        raise
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload["symbols"]


def _load_json(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _ensure_listing_date_cache(cache_path: str, symbols: list[str], threads: int) -> dict[str, str]:
    """Ensure cache contains listing dates (YYYY-MM-DD) for as many symbols as possible."""

    cache = _load_json(cache_path)
    if not isinstance(cache, dict):
        cache = {}

    missing = [s for s in symbols if s not in cache]
    if not missing:
        return cache

    max_workers = max(1, int(threads))
    print(f"Fetching listing dates for {len(missing)} symbols (workers={max_workers}) ...")

    def _worker(sym: str):
        # Small jitter to avoid bursting the same endpoint
        time.sleep(random.uniform(0.02, 0.12))
        try:
            d = fetch_listing_date_eastmoney(sym)
            return sym, d
        except Exception:
            return sym, None

    updated = 0
    ex = ThreadPoolExecutor(max_workers=max_workers)
    futures = []
    interrupted = False
    try:
        futures = [ex.submit(_worker, s) for s in missing]
        for fut in as_completed(futures):
            sym, d = fut.result()
            if d:
                cache[sym] = d
                updated += 1
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted while fetching listing dates; saving partial cache...")
        try:
            ex.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        _save_json(cache_path, cache)
        raise
    finally:
        if not interrupted:
            try:
                ex.shutdown(wait=True)
            except Exception:
                pass

    _save_json(cache_path, cache)
    print(f"Listing-date cache updated: +{updated} (total_cached={len(cache)})")
    return cache


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download daily stock data for ALL symbols.")
    parser.add_argument("--start_date", type=str, help="Start date in YYYY-MM-DD format", default="")
    parser.add_argument("--end_date", type=str, help="End date in YYYY-MM-DD format", default="")
    parser.add_argument("--threads", type=int, help="Number of worker threads (default: 1)", default=1)
    parser.add_argument(
        "--no_ipo_filter",
        action="store_true",
        help="Disable IPO/listing-date filtering (will request all symbols for all dates)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save the downloaded data (default: ./gp_daily)",
        default=None,
    )
    parser.add_argument(
        "--adj",
        type=str,
        default="qfq",
        help="Price adjustment mode for minute data: qfq(前复权, default) | none(不复权) | hfq(后复权)",
    )
    args = parser.parse_args()

    chunks_num = max(1, int(args.threads))
    working_path = os.getcwd()

    output_dir = args.output_dir if args.output_dir else "gp_daily"
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(working_path, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Data will be saved to: {output_dir}")

    # Total list location rule: follow the same rule as self-list
    # - explicit --output_dir: list lives inside output_dir
    # - no --output_dir: list lives in current working directory
    list_dir = output_dir if args.output_dir else working_path
    total_list_file = os.path.join(list_dir, "total_gplist.json")
    symbols = _ensure_total_list(total_list_file)
    print(f"Loaded {len(symbols)} symbols from {total_list_file}")

    # Determine dates to process
    dates_to_process: list[str] = []
    if args.start_date:
        try:
            start = datetime.strptime(args.start_date, "%Y-%m-%d")
            if args.end_date:
                end = datetime.strptime(args.end_date, "%Y-%m-%d")
            else:
                end = datetime.today()

            date_generated = pd.date_range(start, end)
            for d in date_generated:
                if calendar.is_workday(d) and d.weekday() < 5:
                    dates_to_process.append(d.strftime("%Y-%m-%d"))
            print(f"Processing dates: {dates_to_process}")
        except ValueError:
            print("Error: Dates must be in YYYY-MM-DD format.")
            raise SystemExit(1)
    elif args.end_date:
        print("Error: Cannot provide --end_date without --start_date.")
        raise SystemExit(1)
    else:
        today_str = datetime.today().strftime("%Y-%m-%d")
        if calendar.is_workday(datetime.today()) and datetime.today().weekday() < 5:
            dates_to_process = [today_str]
        else:
            dates_to_process = []
            print("Today is not a trading day.")

    if not dates_to_process:
        print("No dates to process.")
        raise SystemExit(0)

    # Filter out tasks before IPO/listing date to avoid wasted requests.
    # Cache file lives alongside total_gplist.json (same dir rule).
    listing_dates: dict[str, str] = {}
    listing_cache_file = os.path.join(list_dir, "total_listing_dates.json")
    if not args.no_ipo_filter:
        listing_dates = _ensure_listing_date_cache(listing_cache_file, symbols, chunks_num)

    tasks_to_run = []
    for sym in symbols:
        ipo_date = listing_dates.get(sym)
        for target_date in dates_to_process:
            if ipo_date and target_date < ipo_date:
                continue
            tasks_to_run.append({"symbol": sym, "date": target_date})

    if not args.no_ipo_filter:
        skipped = len(symbols) * len(dates_to_process) - len(tasks_to_run)
        if skipped > 0:
            print(f"Skipped {skipped} pre-IPO tasks using listing-date cache")

    try:
        run_tasks_in_threads(tasks_to_run, chunks_num, working_path, output_dir, fqt=args.adj)
    except ProxyConnectivityError as e:
        print(f"[FATAL] Proxy connectivity error: {e}")
        raise SystemExit(86)
