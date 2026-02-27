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


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from downloader_common import ProxyConnectivityError, run_tasks_in_threads
from eastmoney_universe import fetch_listing_date_eastmoney


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
    cache = _load_json(cache_path)
    if not isinstance(cache, dict):
        cache = {}

    missing = [s for s in symbols if s not in cache]
    if not missing:
        return cache

    max_workers = max(1, int(threads))
    print(f"Fetching listing dates for {len(missing)} self symbols (workers={max_workers}) ...")

    def _worker(sym: str):
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Download daily stock data.")
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

    # Self list location:
    # - If user provided --output_dir, read self_gplist.json from that directory
    # - Otherwise read it from current working directory
    self_list_dir = output_dir if args.output_dir else working_path
    json_file = os.path.join(self_list_dir, "self_gplist.json")
    if os.path.exists(json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            self_gplist = json.load(f)
        print(f"Loaded {len(self_gplist)} symbols from {json_file}")
    else:
        self_gplist = ["sz002409", "sz301323", "sh688114", "sh688508"]
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(self_gplist, f, indent=4)
        print(f"Created default {json_file}")

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
            print("Today is not a trading day.")
            dates_to_process = []

    fail_file = os.path.join(output_dir, "failed_tasks.json")
    failed_tasks: list[dict] = []
    if os.path.exists(fail_file):
        try:
            with open(fail_file, "r", encoding="utf-8") as f:
                failed_tasks = json.load(f)
        except Exception:
            failed_tasks = []

    tasks_to_run: list[dict] = []
    for target_date in dates_to_process:
        for sym in self_gplist:
            tasks_to_run.append({"symbol": sym, "date": target_date})

    for ft in failed_tasks:
        if ft not in tasks_to_run and ft.get("symbol") in self_gplist:
            tasks_to_run.append(ft)

    # Filter out tasks before IPO/listing date to avoid wasted requests.
    # Cache lives alongside self_gplist.json (same dir rule).
    if dates_to_process and self_gplist and not args.no_ipo_filter:
        listing_cache_file = os.path.join(self_list_dir, "self_listing_dates.json")
        listing_dates = _ensure_listing_date_cache(listing_cache_file, self_gplist, chunks_num)
        filtered = []
        for t in tasks_to_run:
            sym = t.get("symbol")
            d = t.get("date")
            ipo = listing_dates.get(sym)
            if ipo and d and d < ipo:
                continue
            filtered.append(t)
        skipped = len(tasks_to_run) - len(filtered)
        tasks_to_run = filtered
        if skipped > 0:
            print(f"Skipped {skipped} pre-IPO tasks using listing-date cache")

    if not tasks_to_run:
        print("No tasks to run.")
        raise SystemExit(0)

    try:
        run_tasks_in_threads(tasks_to_run, chunks_num, working_path, output_dir, fqt=args.adj)
    except ProxyConnectivityError as e:
        print(f"[FATAL] Proxy connectivity error: {e}")
        raise SystemExit(86)


if __name__ == "__main__":
    main()
