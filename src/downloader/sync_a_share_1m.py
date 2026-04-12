import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from downloader_common import run_tasks_in_threads


MARKET_CLOSE_CUTOFF = "14:55:00"


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


def _to_compact_date(value: str) -> str:
    return _normalize_date(value).replace("-", "")


def _trade_cal_path(output_dir: str) -> str:
    return os.path.join(output_dir, "tushare-trade_cal", "trade_cal.csv")


def _load_trade_cal_open_dates(output_dir: str) -> list[str]:
    path = _trade_cal_path(output_dir)
    if not os.path.exists(path):
        return []

    df = pd.read_csv(path, usecols=["cal_date", "is_open"])
    if df.empty:
        return []

    df["cal_date"] = df["cal_date"].astype(str).str.replace(r"\.0$", "", regex=True)
    df = df[df["is_open"].astype(str).isin(["1", "1.0"])]
    if df.empty:
        return []

    dates = sorted({datetime.strptime(v, "%Y%m%d").strftime("%Y-%m-%d") for v in df["cal_date"].tolist()})
    return dates


def _recent_weekdays(end_date: str, count: int) -> list[str]:
    dates: list[str] = []
    current = datetime.strptime(end_date, "%Y-%m-%d")
    while len(dates) < max(1, int(count)):
        if current.weekday() < 5:
            dates.append(current.strftime("%Y-%m-%d"))
        current -= timedelta(days=1)
    dates.sort()
    return dates


def _select_target_dates(output_dir: str, start_date: str, end_date: str, recent_open_days: int) -> list[str]:
    today = datetime.today().strftime("%Y-%m-%d")
    resolved_end = _normalize_date(end_date) if end_date else today
    open_dates = _load_trade_cal_open_dates(output_dir)

    if start_date:
        resolved_start = _normalize_date(start_date)
        if resolved_start > resolved_end:
            raise ValueError("start_date must be <= end_date")
        if open_dates:
            return [d for d in open_dates if resolved_start <= d <= resolved_end]

        selected: list[str] = []
        current = datetime.strptime(resolved_start, "%Y-%m-%d")
        final_dt = datetime.strptime(resolved_end, "%Y-%m-%d")
        while current <= final_dt:
            if current.weekday() < 5:
                selected.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        return selected

    if open_dates:
        eligible = [d for d in open_dates if d <= resolved_end]
        return eligible[-max(1, int(recent_open_days)) :]

    return _recent_weekdays(resolved_end, recent_open_days)


def _load_symbols(output_dir: str, symbols_arg: str, list_file: str) -> list[str]:
    if symbols_arg.strip():
        symbols = [item.strip().lower() for item in symbols_arg.split(",") if item.strip()]
        if not symbols:
            raise ValueError("No valid symbols parsed from --symbols")
        return symbols

    list_path = os.path.join(output_dir, list_file)
    if not os.path.exists(list_path):
        raise FileNotFoundError(
            f"Symbol list file not found: {list_path}. Run stock list sync first or pass --symbols explicitly."
        )

    with open(list_path, "r", encoding="utf-8") as handle:
        symbols = json.load(handle)
    if not isinstance(symbols, list) or not symbols:
        raise RuntimeError(f"Symbol list file is empty or invalid: {list_path}")
    return [str(item).strip().lower() for item in symbols if str(item).strip()]


def _minute_csv_path(output_dir: str, symbol: str, trade_date: str) -> str:
    return os.path.join(output_dir, "trade", symbol, f"{trade_date}.csv")


def _is_complete_minute_file(csv_path: str) -> bool:
    if not os.path.exists(csv_path):
        return False

    try:
        df = pd.read_csv(csv_path, usecols=["时间"])
    except Exception:
        return False

    if df.empty:
        return False

    last_value = str(df.iloc[-1]["时间"] or "").strip()
    if not last_value:
        return False

    last_time = last_value.split(" ", 1)[1] if " " in last_value else last_value
    if len(last_time) == 5:
        last_time = last_time + ":00"
    return last_time >= MARKET_CLOSE_CUTOFF


def _build_tasks(output_dir: str, symbols: list[str], trade_dates: list[str], force: bool) -> tuple[list[dict], int]:
    tasks: list[dict] = []
    skipped_complete = 0
    for trade_date in trade_dates:
        for symbol in symbols:
            csv_path = _minute_csv_path(output_dir, symbol, trade_date)
            if not force and _is_complete_minute_file(csv_path):
                skipped_complete += 1
                continue
            tasks.append({"symbol": symbol, "date": trade_date})
    return tasks, skipped_complete


def _failed_tasks_path(output_dir: str) -> str:
    return os.path.join(output_dir, "failed_tasks.json")


def _task_key(task: dict) -> tuple[str, str]:
    return str(task.get("symbol") or "").lower(), str(task.get("date") or "")


def _dedupe_tasks(tasks: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for task in tasks:
        key = _task_key(task)
        if not key[0] or not key[1] or key in seen:
            continue
        seen.add(key)
        deduped.append({"symbol": key[0], "date": key[1]})
    return deduped


def _load_failed_tasks(output_dir: str) -> list[dict]:
    path = _failed_tasks_path(output_dir)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            tasks = json.load(handle)
    except Exception:
        return []
    if not isinstance(tasks, list):
        return []
    normalized: list[dict] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        symbol = str(task.get("symbol") or "").strip().lower()
        trade_date = str(task.get("date") or "").strip()
        if not symbol or not trade_date:
            continue
        normalized.append({"symbol": symbol, "date": trade_date})
    return _dedupe_tasks(normalized)


def _save_failed_tasks(output_dir: str, tasks: list[dict]) -> None:
    path = _failed_tasks_path(output_dir)
    if not tasks:
        if os.path.exists(path):
            os.remove(path)
        return
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_dedupe_tasks(tasks), handle, indent=4, ensure_ascii=False)


def _filter_tasks_for_scope(
    tasks: list[dict], output_dir: str, symbols: list[str], trade_dates: list[str], force: bool
) -> list[dict]:
    symbol_set = {item.strip().lower() for item in symbols}
    trade_date_set = set(trade_dates)
    scoped: list[dict] = []
    for task in tasks:
        symbol, trade_date = _task_key(task)
        if symbol not in symbol_set or trade_date not in trade_date_set:
            continue
        if not force and _is_complete_minute_file(_minute_csv_path(output_dir, symbol, trade_date)):
            continue
        scoped.append({"symbol": symbol, "date": trade_date})
    return _dedupe_tasks(scoped)


def _prune_completed_failed_tasks(output_dir: str, symbols: list[str], trade_dates: list[str], force: bool) -> list[dict]:
    all_failed_tasks = _load_failed_tasks(output_dir)
    if not all_failed_tasks:
        return []

    scoped_keys = {(symbol.strip().lower(), trade_date) for symbol in symbols for trade_date in trade_dates}
    remaining_tasks: list[dict] = []
    for task in all_failed_tasks:
        key = _task_key(task)
        if key in scoped_keys and not force and _is_complete_minute_file(_minute_csv_path(output_dir, key[0], key[1])):
            continue
        remaining_tasks.append(task)
    _save_failed_tasks(output_dir, remaining_tasks)
    return _filter_tasks_for_scope(remaining_tasks, output_dir, symbols, trade_dates, force)


def _retry_threads(initial_threads: int, round_index: int) -> int:
    value = max(1, int(initial_threads))
    for _ in range(max(0, int(round_index))):
        value = max(1, value // 2)
    return value


def _round_primary_source(base_source: str, round_index: int) -> str:
    source = str(base_source or "ts").strip().lower()
    return source


def _run_retry_rounds(
    tasks: list[dict],
    output_dir: str,
    threads: int,
    fqt: str,
    source: str,
    force: bool,
    retry_failed_rounds: int,
    retry_sleep_seconds: float,
    selected_symbols: list[str],
    selected_dates: list[str],
) -> list[dict]:
    pending_tasks = _dedupe_tasks(tasks)
    if not pending_tasks:
        return []

    total_rounds = 1 + max(0, int(retry_failed_rounds))
    for round_index in range(total_rounds):
        round_threads = _retry_threads(threads, round_index)
        round_label = "initial" if round_index == 0 else f"retry-{round_index}"
        round_source = _round_primary_source(source, round_index)
        os.environ["GP_MINUTE_SOURCE"] = round_source
        print(f"round={round_label} tasks={len(pending_tasks)} threads={round_threads} source={round_source}")
        run_tasks_in_threads(
            pending_tasks,
            round_threads,
            os.getcwd(),
            output_dir,
            fqt=fqt,
            force=force,
            abort_on_proxy=False,
        )
        remaining_tasks = _filter_tasks_for_scope(
            _load_failed_tasks(output_dir), output_dir, selected_symbols, selected_dates, force
        )
        if not remaining_tasks:
            print(f"round={round_label} all pending tasks completed")
            return []
        pending_keys = {_task_key(task) for task in pending_tasks}
        pending_tasks = [task for task in remaining_tasks if _task_key(task) in pending_keys]
        if not pending_tasks:
            print(f"round={round_label} no scoped failures remain")
            return []
        if round_index < total_rounds - 1 and float(retry_sleep_seconds) > 0:
            print(f"round={round_label} remaining_failures={len(pending_tasks)} sleep_s={retry_sleep_seconds}")
            time.sleep(float(retry_sleep_seconds))
    return pending_tasks


def _warn_if_free_source_window_risky(source: str, trade_dates: list[str]) -> None:
    if source == "ts" or not trade_dates:
        return
    latest = max(trade_dates)
    earliest = min(trade_dates)
    latest_dt = datetime.strptime(latest, "%Y-%m-%d")
    earliest_dt = datetime.strptime(earliest, "%Y-%m-%d")
    if (latest_dt - earliest_dt).days > 7:
        print(
            "[WARN] Eastmoney/Tencent free minute endpoints usually only cover the most recent few trading days. "
            "Older dates may fail unless you switch to --source ts and have minute permission."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Incrementally sync A-share 1-minute data into gp-data/trade")
    parser.add_argument("--output_dir", required=True, help="gp-data root directory")
    parser.add_argument("--threads", type=int, default=4, help="Worker threads")
    parser.add_argument("--symbols", default="", help="Optional comma-separated symbol list, e.g. sh600000,sz000001")
    parser.add_argument("--list_file", default="tushare_gplist.json", help="JSON file containing symbol list")
    parser.add_argument("--start_date", default="", help="Start date, YYYYMMDD or YYYY-MM-DD")
    parser.add_argument("--end_date", default="", help="End date, YYYYMMDD or YYYY-MM-DD")
    parser.add_argument("--recent_open_days", type=int, default=3, help="When start/end are omitted, sync the latest N open days")
    parser.add_argument("--fqt", default="0", help="Adjustment mode: 0/raw, 1/qfq, 2/hfq")
    parser.add_argument("--source", choices=["em", "tx", "ts"], default="ts", help="Primary minute source")
    parser.add_argument("--token", default="", help="Optional Tushare token, used when --source ts")
    parser.add_argument("--force", action="store_true", help="Re-download even when a day file already looks complete")
    parser.add_argument(
        "--retry_failed_rounds",
        type=int,
        default=2,
        help="Additional retry rounds for failed tasks after the initial run",
    )
    parser.add_argument(
        "--retry_sleep_seconds",
        type=float,
        default=8.0,
        help="Sleep seconds between retry rounds",
    )
    parser.add_argument(
        "--retry_failed_only",
        action="store_true",
        help="Only re-run tasks currently recorded in failed_tasks.json within the selected scope",
    )
    parser.add_argument(
        "--no_resume_failures",
        action="store_true",
        help="Ignore existing failed_tasks.json entries when building the initial task list",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    os.environ["GP_MINUTE_SOURCE"] = args.source
    if args.token.strip():
        os.environ["TUSHARE_TOKEN"] = args.token.strip()

    symbols = _load_symbols(output_dir, args.symbols, args.list_file)
    trade_dates = _select_target_dates(output_dir, args.start_date, args.end_date, args.recent_open_days)
    if not trade_dates:
        raise RuntimeError("No target trade dates resolved. Check trade_cal data or the requested date range.")

    _warn_if_free_source_window_risky(args.source, trade_dates)

    tasks, skipped_complete = _build_tasks(output_dir, symbols, trade_dates, args.force)
    scoped_failed_tasks = _prune_completed_failed_tasks(output_dir, symbols, trade_dates, args.force)
    if args.retry_failed_only:
        tasks = scoped_failed_tasks
    elif not args.no_resume_failures:
        tasks = _dedupe_tasks(scoped_failed_tasks + tasks)

    print(f"output_dir={output_dir}")
    print(f"source={args.source} fqt={args.fqt}")
    print(f"symbols={len(symbols)} trade_dates={trade_dates[0]}..{trade_dates[-1]} ({len(trade_dates)} days)")
    print(f"skipped_complete={skipped_complete} resumed_failures={len(scoped_failed_tasks)} pending_tasks={len(tasks)}")

    if not tasks:
        print("All target minute files already look complete.")
        return

    remaining_tasks = _run_retry_rounds(
        tasks,
        output_dir,
        max(1, int(args.threads)),
        str(args.fqt),
        str(args.source),
        bool(args.force),
        int(args.retry_failed_rounds),
        float(args.retry_sleep_seconds),
        symbols,
        trade_dates,
    )
    if remaining_tasks:
        print(f"Unresolved failed tasks after retries: {len(remaining_tasks)}")
        print(f"See failure queue: {_failed_tasks_path(output_dir)}")
        raise RuntimeError("Minute sync finished with unresolved failed tasks")


if __name__ == "__main__":
    main()