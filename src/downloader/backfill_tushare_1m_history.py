import argparse
import bisect
import csv
import fcntl
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, time as dt_time, timedelta

import pandas as pd


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from tushare_provider import fetch_ts_float_share_map, fetch_ts_minute_range_df, ts_code_to_symbol


CSV_COLUMNS = ["时间", "开盘", "收盘", "最高", "最低", "成交量(手)", "成交额(元)", "均价", "换手率(%)"]


class RateLimiter:
    def __init__(self, max_calls: int, period: float):
        self.max_calls = max(1, int(max_calls))
        self.period = float(period)
        self.calls: list[float] = []
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.time()
            self.calls = [item for item in self.calls if now - item < self.period]
            if len(self.calls) >= self.max_calls:
                sleep_seconds = self.period - (now - self.calls[0])
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
                now = time.time()
                self.calls = [item for item in self.calls if now - item < self.period]
            self.calls.append(time.time())


class TradeDateFloatShareCache:
    def __init__(self, daily_basic_limiter: RateLimiter):
        self.daily_basic_limiter = daily_basic_limiter
        self.lock = threading.Lock()
        self.cache: dict[str, dict[str, float]] = {}

    def preload(self, trade_dates: list[str]) -> None:
        for trade_date in trade_dates:
            self.get(trade_date)

    def get(self, trade_date: str) -> dict[str, float]:
        trade_date_compact = trade_date.replace("-", "")
        with self.lock:
            cached = self.cache.get(trade_date_compact)
            if cached is not None:
                return cached

        self.daily_basic_limiter.wait()
        mapping = fetch_ts_float_share_map(trade_date_compact)
        with self.lock:
            self.cache[trade_date_compact] = mapping
        return mapping


def log(message: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def _require_token(token_arg: str) -> str:
    token = (token_arg or "").strip()
    if token:
        return token
    token = (os.getenv("TUSHARE_TOKEN", "") or "").strip()
    if token:
        return token
    token = (os.getenv("GP_TUSHARE_TOKEN", "") or "").strip()
    if token:
        return token
    raise RuntimeError("Tushare token is required. Pass --token or set TUSHARE_TOKEN/GP_TUSHARE_TOKEN.")


def _normalize_date(value: str) -> str:
    text = (value or "").strip()
    if not text:
        raise ValueError("Empty date")
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        datetime.strptime(text, "%Y-%m-%d")
        return text
    if len(text) == 8 and text.isdigit():
        return datetime.strptime(text, "%Y%m%d").strftime("%Y-%m-%d")
    raise ValueError(f"Unsupported date format: {value}")


def _state_path(output_dir: str) -> str:
    return os.path.join(output_dir, ".tushare_1m_history_backfill_state.json")


def _failed_chunks_path(output_dir: str) -> str:
    return os.path.join(output_dir, ".tushare_1m_history_failed_chunks.json")


def _lock_path(output_dir: str) -> str:
    return os.path.join(output_dir, ".tushare_1m_history_backfill.lock")


def _trade_cal_path(output_dir: str) -> str:
    return os.path.join(output_dir, "tushare-trade_cal", "trade_cal.csv")


def _stock_basic_path(output_dir: str, stock_basic_file: str) -> str:
    return os.path.join(output_dir, stock_basic_file)


def _load_open_trade_dates(output_dir: str, start_date: str, end_date: str) -> list[str]:
    path = _trade_cal_path(output_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(f"trade_cal.csv not found: {path}")

    start_compact = start_date.replace("-", "")
    end_compact = end_date.replace("-", "")
    dates: list[str] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            exchange = str(row.get("exchange") or "").strip().upper()
            if exchange not in {"", "SSE"}:
                continue
            is_open = str(row.get("is_open") or "").strip()
            if is_open not in {"1", "1.0", "true", "True"}:
                continue
            cal_date = str(row.get("cal_date") or "").strip().replace(".0", "")
            if not cal_date or cal_date < start_compact or cal_date > end_compact:
                continue
            dates.append(f"{cal_date[0:4]}-{cal_date[4:6]}-{cal_date[6:8]}")
    if not dates:
        raise RuntimeError(f"No open trade dates resolved for range {start_date}..{end_date}")
    return dates


def _resolve_default_end_date(output_dir: str) -> str:
    today = datetime.now()
    today_dash = today.strftime("%Y-%m-%d")
    include_today = today.time() >= dt_time(16, 0)
    path = _trade_cal_path(output_dir)
    if not os.path.exists(path):
        fallback = today if include_today else today - timedelta(days=1)
        return fallback.strftime("%Y-%m-%d")

    open_dates: list[str] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            exchange = str(row.get("exchange") or "").strip().upper()
            if exchange not in {"", "SSE"}:
                continue
            is_open = str(row.get("is_open") or "").strip()
            if is_open not in {"1", "1.0", "true", "True"}:
                continue
            cal_date = str(row.get("cal_date") or "").strip().replace(".0", "")
            if not cal_date:
                continue
            open_dates.append(f"{cal_date[0:4]}-{cal_date[4:6]}-{cal_date[6:8]}")
    open_dates.sort()
    for trade_date in reversed(open_dates):
        if trade_date < today_dash:
            return trade_date
        if include_today and trade_date == today_dash:
            return trade_date
    raise RuntimeError("Unable to resolve a closed trade date from trade_cal.csv")


def _load_symbol_rows(output_dir: str, stock_basic_file: str, symbols_arg: str) -> list[dict]:
    path = _stock_basic_path(output_dir, stock_basic_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"stock basic file not found: {path}")

    selected = {item.strip().lower() for item in symbols_arg.split(",") if item.strip()} if symbols_arg.strip() else set()
    result: list[dict] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ts_code = str(row.get("ts_code") or "").strip()
            list_date = str(row.get("list_date") or "").strip().replace(".0", "")
            if not ts_code or len(list_date) != 8:
                continue
            symbol = ts_code_to_symbol(ts_code)
            if selected and symbol not in selected:
                continue
            result.append(
                {
                    "symbol": symbol,
                    "list_date": f"{list_date[0:4]}-{list_date[4:6]}-{list_date[6:8]}",
                }
            )
    if not result:
        raise RuntimeError("No symbols resolved from stock basic file")
    return result


def _chunk_dates(open_dates: list[str], chunk_open_days: int) -> list[list[str]]:
    size = max(1, int(chunk_open_days))
    return [open_dates[index : index + size] for index in range(0, len(open_dates), size)]


def _estimate_work(symbol_rows: list[dict], open_dates: list[str], chunk_open_days: int) -> tuple[int, int]:
    total_trade_days = 0
    total_chunk_calls = 0
    for row in symbol_rows:
        list_date = row["list_date"]
        start_index = bisect.bisect_left(open_dates, list_date)
        if start_index >= len(open_dates):
            continue
        symbol_trade_days = len(open_dates) - start_index
        total_trade_days += symbol_trade_days
        total_chunk_calls += math.ceil(symbol_trade_days / max(1, int(chunk_open_days)))
    return total_trade_days, total_chunk_calls


def _load_state(output_dir: str) -> dict:
    path = _state_path(output_dir)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _save_state(output_dir: str, state: dict) -> None:
    path = _state_path(output_dir)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _load_failed_chunks(output_dir: str) -> list[dict]:
    path = _failed_chunks_path(output_dir)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return []
    return data if isinstance(data, list) else []


def _save_failed_chunks(output_dir: str, items: list[dict]) -> None:
    path = _failed_chunks_path(output_dir)
    if not items:
        if os.path.exists(path):
            os.remove(path)
        return
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(items, handle, indent=2, ensure_ascii=False)


def _record_failed_chunk(output_dir: str, item: dict) -> None:
    items = _load_failed_chunks(output_dir)
    key = (item.get("symbol"), item.get("chunk_start"), item.get("chunk_end"))
    deduped = []
    replaced = False
    for existing in items:
        existing_key = (existing.get("symbol"), existing.get("chunk_start"), existing.get("chunk_end"))
        if existing_key == key:
            deduped.append(item)
            replaced = True
        else:
            deduped.append(existing)
    if not replaced:
        deduped.append(item)
    _save_failed_chunks(output_dir, deduped)


def _remove_failed_chunk(output_dir: str, symbol: str, chunk_start: str, chunk_end: str) -> None:
    items = _load_failed_chunks(output_dir)
    filtered = [
        item
        for item in items
        if (item.get("symbol"), item.get("chunk_start"), item.get("chunk_end")) != (symbol, chunk_start, chunk_end)
    ]
    _save_failed_chunks(output_dir, filtered)


def _acquire_lock(output_dir: str):
    path = _lock_path(output_dir)
    handle = open(path, "a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError(f"Historical Tushare minute backfill is already running: {path}") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def _all_files_exist(output_dir: str, symbol: str, trade_dates: list[str]) -> bool:
    symbol_dir = os.path.join(output_dir, "trade", symbol)
    if not os.path.isdir(symbol_dir):
        return False
    for trade_date in trade_dates:
        if not os.path.exists(os.path.join(symbol_dir, f"{trade_date}.csv")):
            return False
    return True


def _sleep_seconds_for_attempt(attempt: int, message: str) -> float:
    if "每分钟最多访问" in message or "访问过于频繁" in message:
        return min(60.0, 10.0 + attempt * 8.0)
    return min(20.0, 2.0 + attempt * 3.0)


def _write_symbol_chunk(
    output_dir: str,
    symbol: str,
    eligible_dates: list[str],
    minute_df: pd.DataFrame,
    float_share_cache: TradeDateFloatShareCache,
    force: bool,
) -> tuple[int, int, int]:
    if minute_df.empty:
        return 0, 0, len(eligible_dates)

    symbol_dir = os.path.join(output_dir, "trade", symbol)
    os.makedirs(symbol_dir, exist_ok=True)

    df = minute_df.copy()
    df["trade_date"] = df["时间"].astype(str).str.slice(0, 10)
    date_set = set(eligible_dates)
    df = df[df["trade_date"].isin(date_set)]
    if df.empty:
        return 0, 0, len(eligible_dates)

    files_written = 0
    rows_written = 0
    written_dates: set[str] = set()
    for trade_date, group in df.groupby("trade_date"):
        written_dates.add(trade_date)
        csv_path = os.path.join(symbol_dir, f"{trade_date}.csv")
        if os.path.exists(csv_path) and not force:
            continue
        day_df = group.copy().sort_values("时间").reset_index(drop=True)
        float_share_map = float_share_cache.get(trade_date)
        float_shares = float_share_map.get(symbol)
        if float_shares and float_shares > 0:
            day_df["换手率(%)"] = pd.to_numeric(day_df["成交量(手)"], errors="coerce") * 10000.0 / float_shares
        else:
            day_df["换手率(%)"] = pd.NA
        for column in CSV_COLUMNS:
            if column not in day_df.columns:
                day_df[column] = pd.NA
        day_df[CSV_COLUMNS].to_csv(csv_path, index=False)
        files_written += 1
        rows_written += len(day_df)

    return files_written, rows_written, len(set(eligible_dates) - written_dates)


def _process_symbol_chunk(
    symbol_row: dict,
    chunk_dates: list[str],
    args,
    minute_limiter: RateLimiter,
    float_share_cache: TradeDateFloatShareCache,
) -> dict:
    symbol = symbol_row["symbol"]
    list_date = symbol_row["list_date"]
    eligible_dates = [trade_date for trade_date in chunk_dates if trade_date >= list_date]
    if not eligible_dates:
        return {"symbol": symbol, "status": "pre-listing", "files_written": 0, "rows_written": 0, "missing_dates": 0}

    if not args.force and _all_files_exist(args.output_dir, symbol, eligible_dates):
        return {"symbol": symbol, "status": "already-complete", "files_written": 0, "rows_written": 0, "missing_dates": 0}

    start_datetime = eligible_dates[0] + " 09:00:00"
    end_datetime = eligible_dates[-1] + " 19:00:00"
    last_error = ""
    minute_df = pd.DataFrame()
    for attempt in range(1, args.max_retries + 1):
        try:
            minute_limiter.wait()
            minute_df = fetch_ts_minute_range_df(symbol, start_datetime, end_datetime, args.fqt)
            break
        except Exception as exc:
            last_error = str(exc)
            if attempt >= args.max_retries:
                return {
                    "symbol": symbol,
                    "status": "failed",
                    "error": last_error,
                    "chunk_start": eligible_dates[0],
                    "chunk_end": eligible_dates[-1],
                    "files_written": 0,
                    "rows_written": 0,
                    "missing_dates": len(eligible_dates),
                }
            sleep_seconds = _sleep_seconds_for_attempt(attempt, last_error)
            log(
                f"retry symbol={symbol} chunk={eligible_dates[0]}..{eligible_dates[-1]} "
                f"attempt={attempt}/{args.max_retries} sleep_s={sleep_seconds} err={last_error}"
            )
            time.sleep(sleep_seconds)

    files_written, rows_written, missing_dates = _write_symbol_chunk(
        args.output_dir,
        symbol,
        eligible_dates,
        minute_df,
        float_share_cache,
        args.force,
    )
    return {
        "symbol": symbol,
        "status": "success",
        "chunk_start": eligible_dates[0],
        "chunk_end": eligible_dates[-1],
        "files_written": files_written,
        "rows_written": rows_written,
        "missing_dates": missing_dates,
    }


def _run_chunk_round(
    symbol_rows: list[dict],
    chunk_dates: list[str],
    args,
    minute_limiter: RateLimiter,
    float_share_cache: TradeDateFloatShareCache,
    threads: int,
) -> tuple[list[dict], dict[str, int]]:
    failures: list[dict] = []
    summary = {
        "completed": 0,
        "failed": 0,
        "files_written": 0,
        "rows_written": 0,
        "missing_dates": 0,
        "skipped": 0,
    }
    with ThreadPoolExecutor(max_workers=max(1, int(threads))) as executor:
        futures = {
            executor.submit(_process_symbol_chunk, row, chunk_dates, args, minute_limiter, float_share_cache): row
            for row in symbol_rows
        }
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            status = result.get("status")
            if status == "failed":
                failures.append(result)
                summary["failed"] += 1
            else:
                summary["completed"] += 1
                if status in {"pre-listing", "already-complete"}:
                    summary["skipped"] += 1
            summary["files_written"] += int(result.get("files_written") or 0)
            summary["rows_written"] += int(result.get("rows_written") or 0)
            summary["missing_dates"] += int(result.get("missing_dates") or 0)
            if index % max(1, int(args.log_every)) == 0 or index == len(futures):
                log(
                    f"chunk_progress {chunk_dates[0]}..{chunk_dates[-1]} done={index}/{len(futures)} "
                    f"files_written={summary['files_written']} rows_written={summary['rows_written']} "
                    f"failed={summary['failed']} skipped={summary['skipped']}"
                )
    return failures, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill historical 1-minute A-share data via Tushare stk_mins")
    parser.add_argument("--output_dir", required=True, help="gp-data root directory")
    parser.add_argument("--token", default="", help="Tushare token; falls back to env vars")
    parser.add_argument("--symbols", default="", help="Optional comma-separated symbols, e.g. sh600000,sz000001")
    parser.add_argument("--stock_basic_file", default="tushare_stock_basic.csv", help="Stock basic CSV file")
    parser.add_argument("--start_date", default="1990-12-19", help="Global start date, YYYYMMDD or YYYY-MM-DD")
    parser.add_argument("--end_date", default="", help="Global end date; default is latest closed trade date")
    parser.add_argument("--chunk_open_days", type=int, default=30, help="Open trading days per stk_mins request")
    parser.add_argument("--threads", type=int, default=4, help="Worker threads per trade-date chunk")
    parser.add_argument("--minute_api_rate", type=int, default=120, help="Historical minute API calls per 60 seconds")
    parser.add_argument("--daily_basic_rate", type=int, default=120, help="daily_basic calls per 60 seconds")
    parser.add_argument("--max_retries", type=int, default=4, help="Max retries per symbol chunk")
    parser.add_argument("--retry_failed_rounds", type=int, default=1, help="Extra retry rounds for failed symbol chunks")
    parser.add_argument("--retry_sleep_seconds", type=float, default=15.0, help="Sleep between failed chunk retry rounds")
    parser.add_argument("--fqt", default="0", help="Adjustment mode: 0/raw, 1/qfq, 2/hfq")
    parser.add_argument("--force", action="store_true", help="Overwrite existing daily minute CSV files")
    parser.add_argument("--log_every", type=int, default=100, help="Log progress every N finished symbol chunks")
    parser.add_argument("--reset_state", action="store_true", help="Ignore and overwrite previous backfill state")
    parser.add_argument("--max_chunks", type=int, default=0, help="Optional limit for testing; 0 means all chunks")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "trade"), exist_ok=True)

    token = _require_token(args.token)
    os.environ["TUSHARE_TOKEN"] = token
    start_date = _normalize_date(args.start_date)
    end_date = _normalize_date(args.end_date) if args.end_date.strip() else _resolve_default_end_date(args.output_dir)
    if start_date > end_date:
        raise ValueError("start_date must be <= end_date")

    lock_handle = _acquire_lock(args.output_dir)
    minute_limiter = RateLimiter(args.minute_api_rate, 60.0)
    daily_basic_limiter = RateLimiter(args.daily_basic_rate, 60.0)
    float_share_cache = TradeDateFloatShareCache(daily_basic_limiter)

    try:
        symbol_rows = _load_symbol_rows(args.output_dir, args.stock_basic_file, args.symbols)
        open_dates = _load_open_trade_dates(args.output_dir, start_date, end_date)
        chunks = _chunk_dates(open_dates, args.chunk_open_days)
        if args.max_chunks and args.max_chunks > 0:
            chunks = chunks[: int(args.max_chunks)]

        estimated_trade_days, estimated_chunk_calls = _estimate_work(symbol_rows, open_dates, args.chunk_open_days)
        state = {} if args.reset_state else _load_state(args.output_dir)
        start_chunk_index = int(state.get("current_chunk_index") or 0) if state else 0
        start_chunk_index = max(0, min(start_chunk_index, len(chunks)))

        run_state = {
            "status": "running",
            "started_at": state.get("started_at") or datetime.now().isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "current_chunk_index": start_chunk_index,
            "total_chunks": len(chunks),
            "chunk_open_days": args.chunk_open_days,
            "threads": args.threads,
            "minute_api_rate": args.minute_api_rate,
            "daily_basic_rate": args.daily_basic_rate,
            "symbols_total": len(symbol_rows),
            "estimated_trade_days": estimated_trade_days,
            "estimated_symbol_chunk_calls": estimated_chunk_calls,
            "files_written": int(state.get("files_written") or 0) if state and not args.reset_state else 0,
            "rows_written": int(state.get("rows_written") or 0) if state and not args.reset_state else 0,
            "missing_dates": int(state.get("missing_dates") or 0) if state and not args.reset_state else 0,
            "chunks_failed": int(state.get("chunks_failed") or 0) if state and not args.reset_state else 0,
            "last_error": "",
        }
        _save_state(args.output_dir, run_state)

        log(
            f"historical_backfill started symbols={len(symbol_rows)} open_dates={len(open_dates)} chunks={len(chunks)} "
            f"estimated_trade_days={estimated_trade_days} estimated_symbol_chunk_calls={estimated_chunk_calls} "
            f"range={start_date}..{end_date} start_chunk_index={start_chunk_index}"
        )

        for chunk_index in range(start_chunk_index, len(chunks)):
            chunk_dates = chunks[chunk_index]
            run_state.update(
                {
                    "current_chunk_index": chunk_index,
                    "current_chunk_start": chunk_dates[0],
                    "current_chunk_end": chunk_dates[-1],
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
            _save_state(args.output_dir, run_state)
            log(f"chunk start index={chunk_index + 1}/{len(chunks)} range={chunk_dates[0]}..{chunk_dates[-1]}")

            float_share_cache.preload(chunk_dates)
            failed_rows, summary = _run_chunk_round(
                symbol_rows,
                chunk_dates,
                args,
                minute_limiter,
                float_share_cache,
                args.threads,
            )

            pending_failures = failed_rows
            for retry_round in range(1, max(0, int(args.retry_failed_rounds)) + 1):
                if not pending_failures:
                    break
                log(
                    f"chunk retry round={retry_round} range={chunk_dates[0]}..{chunk_dates[-1]} "
                    f"pending_failures={len(pending_failures)}"
                )
                time.sleep(max(0.0, float(args.retry_sleep_seconds)))
                retry_symbol_rows = [
                    {"symbol": item["symbol"], "list_date": chunk_dates[0]}
                    for item in pending_failures
                ]
                pending_failures, retry_summary = _run_chunk_round(
                    retry_symbol_rows,
                    chunk_dates,
                    args,
                    minute_limiter,
                    float_share_cache,
                    max(1, int(args.threads) // 2),
                )
                for key in ["completed", "failed", "files_written", "rows_written", "missing_dates", "skipped"]:
                    summary[key] = summary.get(key, 0) + retry_summary.get(key, 0)

            for item in pending_failures:
                run_state["chunks_failed"] = int(run_state.get("chunks_failed") or 0) + 1
                failure_record = {
                    "symbol": item["symbol"],
                    "chunk_start": item.get("chunk_start") or chunk_dates[0],
                    "chunk_end": item.get("chunk_end") or chunk_dates[-1],
                    "error": item.get("error") or "unknown error",
                    "failed_at": datetime.now().isoformat(timespec="seconds"),
                }
                _record_failed_chunk(args.output_dir, failure_record)

            if not pending_failures:
                for row in symbol_rows:
                    _remove_failed_chunk(args.output_dir, row["symbol"], chunk_dates[0], chunk_dates[-1])

            run_state["files_written"] = int(run_state.get("files_written") or 0) + int(summary.get("files_written") or 0)
            run_state["rows_written"] = int(run_state.get("rows_written") or 0) + int(summary.get("rows_written") or 0)
            run_state["missing_dates"] = int(run_state.get("missing_dates") or 0) + int(summary.get("missing_dates") or 0)
            run_state["updated_at"] = datetime.now().isoformat(timespec="seconds")
            _save_state(args.output_dir, run_state)
            log(
                f"chunk done index={chunk_index + 1}/{len(chunks)} range={chunk_dates[0]}..{chunk_dates[-1]} "
                f"files_written={summary.get('files_written', 0)} rows_written={summary.get('rows_written', 0)} "
                f"missing_dates={summary.get('missing_dates', 0)} remaining_failures={len(pending_failures)}"
            )

        run_state.update(
            {
                "status": "success",
                "current_chunk_index": len(chunks),
                "current_chunk_start": "",
                "current_chunk_end": "",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "last_error": "",
            }
        )
        _save_state(args.output_dir, run_state)
        log(
            f"historical_backfill finished files_written={run_state['files_written']} rows_written={run_state['rows_written']} "
            f"missing_dates={run_state['missing_dates']} chunks_failed={run_state['chunks_failed']}"
        )
    except Exception as exc:
        state = _load_state(args.output_dir)
        state.update(
            {
                "status": "failed",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "last_error": str(exc),
            }
        )
        _save_state(args.output_dir, state)
        raise
    finally:
        try:
            lock_handle.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()