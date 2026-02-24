import argparse
import bisect
import csv
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import requests
import chinese_calendar as calendar

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from eastmoney_universe import fetch_all_symbols_eastmoney, fetch_listing_date_eastmoney, last_trading_day, symbol_to_em_secid


def _normalize_adj(adj: str) -> str:
    a = (adj or "none").strip().lower()
    if a in {"none", "0", "raw", "nfq"}:
        return "0"
    if a in {"qfq", "1", "forward"}:
        return "1"
    if a in {"hfq", "2", "backward"}:
        return "2"
    raise ValueError(f"Unknown adj: {adj}")


def _em_headers() -> dict:
    return {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Referer": "https://quote.eastmoney.com/",
        "Accept": "application/json,text/plain,*/*",
    }


def _fetch_em_daily_klines(symbol: str, beg_yyyymmdd: str, end_yyyymmdd: str, fqt: str) -> list[str]:
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "klt": "101",
        "fqt": fqt,
        "secid": symbol_to_em_secid(symbol),
        "beg": beg_yyyymmdd,
        "end": end_yyyymmdd,
        # Eastmoney supports lmt to increase returned rows; harmless if ignored.
        "lmt": "1000000",
    }

    r = requests.get(url, params=params, headers=_em_headers(), timeout=30)
    r.raise_for_status()
    j = r.json()
    return (j.get("data") or {}).get("klines") or []


def _parse_kline_line(line: str) -> dict | None:
    # expected: YYYY-MM-DD,open,close,high,low,volume,amount,amplitude,pctchg,chg,turnover
    if not line or not isinstance(line, str):
        return None
    parts = line.split(",")
    if len(parts) < 7:
        return None

    def sf(v: str):
        try:
            return float(v)
        except Exception:
            return None

    d = parts[0].strip()
    if not d:
        return None

    return {
        "date": d,
        "open": sf(parts[1]),
        "close": sf(parts[2]),
        "high": sf(parts[3]),
        "low": sf(parts[4]),
        "volume": sf(parts[5]),
        "amount": sf(parts[6]),
        "amplitude": sf(parts[7]) if len(parts) > 7 else None,
        "pctchg": sf(parts[8]) if len(parts) > 8 else None,
        "chg": sf(parts[9]) if len(parts) > 9 else None,
        "turnover": sf(parts[10]) if len(parts) > 10 else None,
    }


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


def _load_total_list(path: str) -> list[str] | None:
    obj = _load_json(path)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("symbols"), list):
        return obj["symbols"]
    return None


def _ensure_total_list(path: str) -> list[str]:
    asof = last_trading_day()
    if os.path.exists(path):
        try:
            obj = _load_json(path)
            if isinstance(obj, dict) and obj.get("asof") == asof and isinstance(obj.get("symbols"), list):
                return obj["symbols"]
        except Exception:
            pass

    payload = fetch_all_symbols_eastmoney()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    _save_json(path, payload)
    return payload["symbols"]


def _ensure_listing_date_cache(cache_path: str, symbols: list[str], threads: int) -> dict[str, str]:
    cache = _load_json(cache_path)
    if not isinstance(cache, dict):
        cache = {}

    missing = [s for s in symbols if s not in cache]
    if not missing:
        return cache

    max_workers = max(1, int(threads))
    print(f"Fetching listing dates for {len(missing)} symbols (workers={max_workers}) ...")

    def _worker(sym: str):
        time.sleep(random.uniform(0.02, 0.12))
        try:
            d = fetch_listing_date_eastmoney(sym)
            return sym, d
        except Exception:
            return sym, None

    updated = 0
    ex = ThreadPoolExecutor(max_workers=max_workers)
    try:
        futures = [ex.submit(_worker, s) for s in missing]
        for fut in as_completed(futures):
            sym, d = fut.result()
            if d:
                cache[sym] = d
                updated += 1
    finally:
        try:
            ex.shutdown(wait=True)
        except Exception:
            pass

    _save_json(cache_path, cache)
    print(f"Listing-date cache updated: +{updated} (total_cached={len(cache)})")
    return cache


def _yyyymmdd(date_yyyy_mm_dd: str) -> str:
    return date_yyyy_mm_dd.replace("-", "")


def _safe_next_day(date_yyyy_mm_dd: str) -> str:
    d = datetime.strptime(date_yyyy_mm_dd, "%Y-%m-%d")
    return (d + timedelta(days=1)).strftime("%Y-%m-%d")


def _safe_prev_day(date_yyyy_mm_dd: str) -> str:
    d = datetime.strptime(date_yyyy_mm_dd, "%Y-%m-%d")
    return (d - timedelta(days=1)).strftime("%Y-%m-%d")


def _is_trading_day(d: datetime) -> bool:
    # Keep consistent with other downloaders in this repo: only weekdays.
    # chinese_calendar only supports a limited year range (commonly 2004..current);
    # for years outside the supported window, fall back to weekday-only (Mon-Fri).
    y = d.year
    if y < 2004 or y > 2100:
        return d.weekday() < 5
    try:
        return calendar.is_workday(d) and d.weekday() < 5
    except NotImplementedError:
        return d.weekday() < 5


def _trading_days_between(start_date: str, end_date: str) -> list[str]:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    days: list[str] = []
    cur = start
    one = timedelta(days=1)
    while cur <= end:
        if _is_trading_day(cur):
            days.append(cur.strftime("%Y-%m-%d"))
        cur += one
    return days


def _read_last_date(csv_path: str) -> str | None:
    if not os.path.exists(csv_path):
        return None

    # Read last non-empty line efficiently
    with open(csv_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        if size == 0:
            return None

        # read tail chunk
        chunk = 4096
        pos = max(0, size - chunk)
        f.seek(pos)
        data = f.read().decode("utf-8", errors="ignore")

    lines = [ln.strip() for ln in data.splitlines() if ln.strip()]
    if not lines:
        return None

    # If we didn't capture full last line (rare), fall back to full read
    last = lines[-1]
    if "," not in last:
        try:
            with open(csv_path, "r", encoding="utf-8") as tf:
                all_lines = [ln.strip() for ln in tf.read().splitlines() if ln.strip()]
            if all_lines:
                last = all_lines[-1]
        except Exception:
            return None

    # Skip header
    if last.lower().startswith("date,"):
        return None

    return last.split(",", 1)[0].strip() or None


def _load_existing_rows(csv_path: str) -> tuple[dict[str, dict], list[str]]:
    """Load existing CSV into a date->row map and sorted list of dates."""
    if not os.path.exists(csv_path):
        return {}, []

    rows_by_date: dict[str, dict] = {}
    dates: list[str] = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if not r:
                    continue
                d = (r.get("date") or "").strip()
                if not d:
                    continue
                if d not in rows_by_date:
                    dates.append(d)
                rows_by_date[d] = dict(r)
    except Exception:
        # If file is malformed, treat as empty so we can rebuild.
        return {}, []

    dates.sort()
    return rows_by_date, dates


def _write_all_rows(csv_path: str, rows_by_date: dict[str, dict]) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    tmp = csv_path + ".tmp"
    fieldnames = [
        "date",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "amplitude",
        "pctchg",
        "chg",
        "turnover",
    ]
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for d in sorted(rows_by_date.keys()):
            w.writerow(rows_by_date[d])
    os.replace(tmp, csv_path)


def _compress_missing_ranges(expected: list[str], have: set[str]) -> list[tuple[str, str]]:
    """Compress missing expected dates into contiguous ranges (by expected list adjacency)."""
    ranges: list[tuple[str, str]] = []
    cur_start = None
    cur_end = None
    for d in expected:
        if d in have:
            if cur_start is not None:
                ranges.append((cur_start, cur_end or cur_start))
                cur_start = None
                cur_end = None
            continue

        if cur_start is None:
            cur_start = d
            cur_end = d
        else:
            cur_end = d

    if cur_start is not None:
        ranges.append((cur_start, cur_end or cur_start))
    return ranges


def _split_range_by_days(start: str, end: str, max_days: int = 260) -> list[tuple[str, str]]:
    """Split a (start,end) date range into smaller chunks by calendar days to avoid huge responses."""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    if s > e:
        return []
    chunks: list[tuple[str, str]] = []
    cur = s
    while cur <= e:
        nxt = min(e, cur + timedelta(days=max_days))
        chunks.append((cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
        cur = nxt + timedelta(days=1)
    return chunks


def _append_rows(csv_path: str, rows: list[dict]) -> None:
    if not rows:
        return

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    exists = os.path.exists(csv_path)

    fieldnames = [
        "date",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "amplitude",
        "pctchg",
        "chg",
        "turnover",
    ]

    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def _download_one_symbol(
    symbol: str,
    listing_date: str | None,
    end_date: str,
    out_dir: str,
    fqt: str,
    trading_days_all: list[str],
) -> tuple[str, str]:
    root = os.path.join(out_dir, "total-daily-trade")
    csv_path = os.path.join(root, f"{symbol}.csv")

    desired_start = listing_date or "1990-01-01"
    if desired_start > end_date:
        return symbol, "up-to-date"

    # Load existing data to scan continuity and fill missing segments.
    rows_by_date, existing_dates = _load_existing_rows(csv_path)
    have: set[str] = set(existing_dates)

    # Expected trading days for this symbol range.
    # Use precomputed trading_days_all and slice using bisect.
    start_idx = bisect.bisect_left(trading_days_all, desired_start)
    end_idx = bisect.bisect_right(trading_days_all, end_date)
    expected = trading_days_all[start_idx:end_idx]

    if not expected:
        return symbol, "up-to-date"

    missing_ranges = _compress_missing_ranges(expected, have)
    if not missing_ranges:
        return symbol, "up-to-date"

    # Track whether we can append-only (i.e., all new rows are strictly after max existing date).
    max_existing = existing_dates[-1] if existing_dates else None

    total_new = 0
    filled_ranges = 0

    def _download_range(range_start: str, range_end: str) -> list[dict]:
        beg = _yyyymmdd(range_start)
        end = _yyyymmdd(range_end)

        time.sleep(random.uniform(0.05, 0.25))
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                klines = _fetch_em_daily_klines(symbol, beg, end, fqt)
                parsed: list[dict] = []
                for line in klines:
                    obj = _parse_kline_line(line)
                    if not obj:
                        continue
                    d = obj.get("date")
                    if not d:
                        continue
                    if d < range_start or d > range_end:
                        continue
                    parsed.append(obj)
                parsed.sort(key=lambda x: x["date"])
                return parsed
            except Exception as e:
                last_exc = e
                time.sleep(0.8 * (attempt + 1) + random.uniform(0.0, 0.4))
        raise last_exc or RuntimeError("download failed")

    # Download missing ranges; split into smaller calendar chunks to reduce timeouts.
    new_rows: list[dict] = []
    try:
        for rs, re in missing_ranges:
            for cs, ce in _split_range_by_days(rs, re, max_days=260):
                parsed = _download_range(cs, ce)
                if not parsed:
                    continue
                filled_ranges += 1
                for r in parsed:
                    d = r.get("date")
                    if not d:
                        continue
                    if d in rows_by_date:
                        continue
                    rows_by_date[d] = r
                    new_rows.append(r)
                    total_new += 1
    except Exception as e:
        return symbol, f"failed {type(e).__name__}: {str(e)[:180]}"

    if total_new <= 0:
        # We attempted to fill gaps, but Eastmoney may not have data for some expected days (e.g. suspension).
        return symbol, "ok +0"

    # Decide append vs rewrite to keep CSV strictly chronological.
    need_rewrite = False
    if max_existing is None:
        need_rewrite = True
    else:
        for r in new_rows:
            d = r.get("date")
            if d and d <= max_existing:
                need_rewrite = True
                break

    if need_rewrite:
        _write_all_rows(csv_path, rows_by_date)
    else:
        new_rows.sort(key=lambda x: x["date"])
        _append_rows(csv_path, new_rows)

    return symbol, f"ok +{total_new} (filled_ranges={filled_ranges})"


def main() -> None:
    p = argparse.ArgumentParser(
        description="Download full-history daily K线 for ALL current symbols (IPO -> now) into --output_dir/total-daily-trade/"
    )
    p.add_argument("--output_dir", type=str, default=None, help="Output directory root (required)")
    p.add_argument("--threads", type=int, default=8, help="Worker threads (default: 8)")
    p.add_argument(
        "--adj",
        type=str,
        default="none",
        help="Daily price adjustment: none|qfq|hfq (default: none)",
    )
    p.add_argument(
        "--no_ipo_filter",
        action="store_true",
        help="Disable IPO/listing-date logic (will start from 1990-01-01 for missing listing dates)",
    )
    p.add_argument(
        "--end_date",
        type=str,
        default="",
        help="End date YYYY-MM-DD (default: today; Eastmoney will naturally stop at latest trading day)",
    )
    p.add_argument(
        "--symbols",
        type=str,
        default="",
        help="Optional comma-separated symbol list for testing (e.g. sh600000,sz000001)",
    )

    args = p.parse_args()

    if not args.output_dir:
        print("Error: --output_dir is required")
        raise SystemExit(1)

    working_path = os.getcwd()
    out_dir = args.output_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(working_path, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    end_date = args.end_date.strip() or datetime.today().strftime("%Y-%m-%d")
    try:
        datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        print("Error: --end_date must be YYYY-MM-DD")
        raise SystemExit(1)

    list_dir = out_dir
    total_list_file = os.path.join(list_dir, "total_gplist.json")

    if args.symbols.strip():
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = _ensure_total_list(total_list_file)

    print(f"Output root: {out_dir}")
    print(f"Symbols: {len(symbols)}")
    print(f"End date: {end_date}")

    listing_dates: dict[str, str] = {}
    if not args.no_ipo_filter and not args.symbols.strip():
        cache_path = os.path.join(list_dir, "total_listing_dates.json")
        listing_dates = _ensure_listing_date_cache(cache_path, symbols, max(1, args.threads))

    fqt = _normalize_adj(args.adj)

    # Ensure target dir exists
    os.makedirs(os.path.join(out_dir, "total-daily-trade"), exist_ok=True)

    # Precompute trading calendar once; reused by all symbols for continuity scan.
    # Global start: earliest possible IPO date used by this script.
    global_start = "1990-01-01"
    if not args.no_ipo_filter and listing_dates:
        try:
            global_start = min(listing_dates.values())
        except Exception:
            global_start = "1990-01-01"

    try:
        trading_days_all = _trading_days_between(global_start, end_date)
    except Exception:
        trading_days_all = _trading_days_between("1990-01-01", end_date)

    max_workers = max(1, int(args.threads))
    ok = 0
    up_to_date = 0
    failed = 0

    t0 = time.time()

    def _task(sym: str):
        ipo = listing_dates.get(sym)
        if args.no_ipo_filter:
            ipo = ipo or "1990-01-01"
        return _download_one_symbol(sym, ipo, end_date, out_dir, fqt, trading_days_all)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_task, s): s for s in symbols}
        for i, fut in enumerate(as_completed(futures), 1):
            sym, status = fut.result()
            if status.startswith("ok"):
                ok += 1
            elif status == "up-to-date":
                up_to_date += 1
            else:
                failed += 1

            if i % 50 == 0 or i == len(symbols):
                elapsed = int((time.time() - t0) * 1000)
                print(
                    f"Progress {i}/{len(symbols)} | ok={ok} up-to-date={up_to_date} failed={failed} | elapsed_ms={elapsed}"
                )

            if status not in {"up-to-date", "ok +0"} and not status.startswith("ok +"):
                print(f"{sym}: {status}")

    elapsed_s = int(time.time() - t0)
    print(f"Done. ok={ok} up-to-date={up_to_date} failed={failed} elapsed_s={elapsed_s}")


if __name__ == "__main__":
    main()
