import argparse
import csv
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, time as dt_time, timedelta


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))


def log(message: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def _resolve_token(token_arg: str) -> str:
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


def _parse_schedule_time(value: str) -> dt_time:
    text = (value or "").strip()
    try:
        parsed = datetime.strptime(text, "%H:%M")
    except ValueError as exc:
        raise ValueError(f"Invalid schedule time: {value}. Expected HH:MM.") from exc
    return parsed.time()


def _state_path(output_dir: str) -> str:
    return os.path.join(output_dir, ".eod_data_scheduler_state.json")


def _lock_path(output_dir: str) -> str:
    return os.path.join(output_dir, ".eod_data_scheduler.lock")


def _trade_cal_path(output_dir: str) -> str:
    return os.path.join(output_dir, "tushare-trade_cal", "trade_cal.csv")


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


def _parse_timestamp(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _state_text(value: object, default: str = "-") -> str:
    text = str(value or "").strip()
    return text or default


def _all_done_text(state: dict) -> str:
    if "all_done" not in state:
        return "-"
    return "yes" if bool(state.get("all_done")) else "no"


def _state_summary(state: dict) -> str:
    summary = (
        f"attempt_trade_date={_state_text(state.get('last_attempt_trade_date'))} "
        f"success_trade_date={_state_text(state.get('last_success_trade_date'))} "
        f"overall={_state_text(state.get('last_status'), 'idle')} "
        f"phase={_state_text(state.get('current_phase'))} "
        f"fast_sync={_state_text(state.get('fast_sync_status'))} "
        f"minute_sync={_state_text(state.get('minute_sync_status'))} "
        f"all_done={_all_done_text(state)} "
        f"started_at={_state_text(state.get('last_started_at'))} "
        f"finished_at={_state_text(state.get('last_finished_at'))}"
    )
    last_error = _state_text(state.get("last_error"))
    if last_error != "-":
        summary += f" last_error={last_error}"
    return summary


def _log_state(prefix: str, state: dict) -> None:
    log(f"{prefix}: {_state_summary(state)}")


def _remaining_seconds_until_schedule(current_dt: datetime, schedule_time: dt_time) -> int:
    scheduled_dt = datetime.combine(current_dt.date(), schedule_time)
    return max(0, int((scheduled_dt - current_dt).total_seconds()))


def _remaining_failure_retry_seconds(state: dict, current_dt: datetime, failure_retry_interval_seconds: int) -> int:
    finished_at = _parse_timestamp(state.get("last_finished_at"))
    if finished_at is None:
        return 0
    elapsed = int((current_dt - finished_at).total_seconds())
    return max(0, int(failure_retry_interval_seconds) - elapsed)


def _maybe_log_status(message: str, tracker: dict, current_dt: datetime, interval_seconds: int) -> None:
    last_message = str(tracker.get("message") or "")
    last_logged_at = tracker.get("logged_at")
    should_log = message != last_message
    if not should_log and isinstance(last_logged_at, datetime):
        should_log = (current_dt - last_logged_at).total_seconds() >= max(1, int(interval_seconds))
    elif not should_log:
        should_log = True
    if should_log:
        log(message)
        tracker["message"] = message
        tracker["logged_at"] = current_dt


def _lookup_trade_cal_status(output_dir: str, trade_date_yyyymmdd: str) -> bool | None:
    path = _trade_cal_path(output_dir)
    if not os.path.exists(path):
        return None

    fallback_status: bool | None = None
    try:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                cal_date = str(row.get("cal_date") or "").strip().replace(".0", "")
                if cal_date != trade_date_yyyymmdd:
                    continue
                exchange = str(row.get("exchange") or "").strip().upper()
                is_open = str(row.get("is_open") or "").strip()
                status = is_open in {"1", "1.0", "true", "True"}
                if exchange == "SSE":
                    return status
                if fallback_status is None:
                    fallback_status = status
    except Exception:
        return None
    return fallback_status


def _fallback_latest_open_weekday(today: datetime) -> str:
    cursor = today
    while cursor.weekday() >= 5:
        cursor -= timedelta(days=1)
    return cursor.strftime("%Y%m%d")


def _latest_open_trade_date(output_dir: str, today_yyyymmdd: str) -> str:
    path = _trade_cal_path(output_dir)
    if os.path.exists(path):
        latest_open: str | None = None
        try:
            with open(path, "r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    cal_date = str(row.get("cal_date") or "").strip().replace(".0", "")
                    if not cal_date or cal_date > today_yyyymmdd:
                        continue
                    exchange = str(row.get("exchange") or "").strip().upper()
                    if exchange not in {"", "SSE"}:
                        continue
                    is_open = str(row.get("is_open") or "").strip()
                    if is_open not in {"1", "1.0", "true", "True"}:
                        continue
                    latest_open = cal_date
        except Exception:
            latest_open = None
        if latest_open:
            return latest_open

    return _fallback_latest_open_weekday(datetime.strptime(today_yyyymmdd, "%Y%m%d"))


def _is_trading_day(output_dir: str, current_dt: datetime) -> bool:
    trade_date = current_dt.strftime("%Y%m%d")
    status = _lookup_trade_cal_status(output_dir, trade_date)
    if status is None:
        return current_dt.weekday() < 5
    return bool(status)


def _resolve_run_trade_date(output_dir: str, current_dt: datetime, allow_non_trading_day: bool) -> str | None:
    today_yyyymmdd = current_dt.strftime("%Y%m%d")
    if _is_trading_day(output_dir, current_dt):
        return today_yyyymmdd
    if not allow_non_trading_day:
        return None
    return _latest_open_trade_date(output_dir, today_yyyymmdd)


def _should_run(
    state: dict,
    trade_date_yyyymmdd: str,
    current_dt: datetime,
    failure_retry_interval_seconds: int,
) -> bool:
    last_success = str(state.get("last_success_trade_date") or "").strip()
    if last_success == trade_date_yyyymmdd:
        return False

    last_attempt_trade_date = str(state.get("last_attempt_trade_date") or "").strip()
    if last_attempt_trade_date != trade_date_yyyymmdd:
        return True

    last_status = str(state.get("last_status") or "").strip().lower()
    if last_status in {"", "running"}:
        return True
    if last_status == "success":
        return False

    finished_at = _parse_timestamp(state.get("last_finished_at"))
    if finished_at is None:
        return True
    return (current_dt - finished_at).total_seconds() >= max(1, int(failure_retry_interval_seconds))


def _build_fast_sync_command(args, token: str) -> list[str]:
    command = [
        sys.executable,
        os.path.join(CURRENT_DIR, "fast_sync_tushare_latest.py"),
        "--output-dir",
        args.output_dir,
        "--token",
        token,
        "--financial-threads",
        str(args.financial_threads),
        "--batch-rate",
        str(args.batch_rate),
        "--financial-rate",
        str(args.financial_rate),
        "--backfill-open-days",
        str(args.backfill_open_days),
    ]
    if args.skip_financials:
        command.append("--skip-financials")
    if args.skip_date_based:
        command.append("--skip-date-based")
    if args.financial_datasets.strip():
        command.extend(["--financial-datasets", args.financial_datasets.strip()])
    return command


def _build_minute_sync_command(args, token: str, trade_date_yyyymmdd: str) -> list[str]:
    command = [
        sys.executable,
        os.path.join(CURRENT_DIR, "sync_a_share_1m.py"),
        "--output_dir",
        args.output_dir,
        "--start_date",
        trade_date_yyyymmdd,
        "--end_date",
        trade_date_yyyymmdd,
        "--threads",
        str(args.minute_threads),
        "--source",
        args.minute_source,
        "--fqt",
        str(args.minute_fqt),
        "--retry_failed_rounds",
        str(args.minute_retry_failed_rounds),
        "--retry_sleep_seconds",
        str(args.minute_retry_sleep_seconds),
        "--no_resume_failures",
    ]
    if args.minute_force:
        command.append("--force")
    if args.minute_source == "ts":
        command.extend(["--token", token])
    return command


def _run_command(command: list[str], dry_run: bool) -> None:
    printable = " ".join(command)
    log(f"run: {printable}")
    if dry_run:
        return
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(command, cwd=WORKSPACE_DIR, env=env, check=True)


def _run_job(args, token: str, trade_date_yyyymmdd: str, state: dict) -> None:
    if args.dry_run:
        dry_state = {
            "last_attempt_trade_date": trade_date_yyyymmdd,
            "last_status": "dry-run",
            "current_phase": "fast_sync",
            "fast_sync_status": "planned",
            "minute_sync_status": "planned",
            "all_done": False,
            "last_started_at": datetime.now().isoformat(timespec="seconds"),
            "last_finished_at": "",
            "last_error": "",
            "last_success_trade_date": state.get("last_success_trade_date") or "",
        }
        _log_state("job status", dry_state)
        dry_state.update({"fast_sync_status": "running", "current_phase": "fast_sync"})
        _log_state("phase start", dry_state)
        _run_command(_build_fast_sync_command(args, token), dry_run=True)
        dry_state.update({"fast_sync_status": "success", "current_phase": "minute_sync"})
        _log_state("phase done", dry_state)
        dry_state.update({"minute_sync_status": "running", "current_phase": "minute_sync"})
        _log_state("phase start", dry_state)
        _run_command(_build_minute_sync_command(args, token, trade_date_yyyymmdd), dry_run=True)
        dry_state.update(
            {
                "last_status": "success",
                "current_phase": "completed",
                "minute_sync_status": "success",
                "all_done": True,
                "last_finished_at": datetime.now().isoformat(timespec="seconds"),
                "last_success_trade_date": trade_date_yyyymmdd,
            }
        )
        _log_state("job status", dry_state)
        return

    started_at = datetime.now()
    state.update(
        {
            "last_attempt_trade_date": trade_date_yyyymmdd,
            "last_status": "running",
            "current_phase": "fast_sync",
            "fast_sync_status": "pending",
            "minute_sync_status": "pending",
            "all_done": False,
            "last_started_at": started_at.isoformat(timespec="seconds"),
            "last_finished_at": "",
            "last_error": "",
        }
    )
    _save_state(args.output_dir, state)
    _log_state("job status", state)

    try:
        state.update({"current_phase": "fast_sync", "fast_sync_status": "running"})
        _save_state(args.output_dir, state)
        _log_state("phase start", state)
        _run_command(_build_fast_sync_command(args, token), dry_run=args.dry_run)
        state.update({"fast_sync_status": "success", "current_phase": "minute_sync"})
        _save_state(args.output_dir, state)
        _log_state("phase done", state)

        state.update({"current_phase": "minute_sync", "minute_sync_status": "running"})
        _save_state(args.output_dir, state)
        _log_state("phase start", state)
        _run_command(_build_minute_sync_command(args, token, trade_date_yyyymmdd), dry_run=args.dry_run)
    except Exception as exc:
        finished_at = datetime.now()
        if str(state.get("current_phase") or "") == "fast_sync":
            state["fast_sync_status"] = "failed"
        else:
            state["minute_sync_status"] = "failed"
        state.update(
            {
                "last_status": "failed",
                "current_phase": "failed",
                "all_done": False,
                "last_finished_at": finished_at.isoformat(timespec="seconds"),
                "last_error": str(exc),
            }
        )
        _save_state(args.output_dir, state)
        _log_state("job status", state)
        raise

    finished_at = datetime.now()
    state.update(
        {
            "last_status": "success",
            "current_phase": "completed",
            "fast_sync_status": "success",
            "minute_sync_status": "success",
            "all_done": True,
            "last_finished_at": finished_at.isoformat(timespec="seconds"),
            "last_error": "",
            "last_success_trade_date": trade_date_yyyymmdd,
        }
    )
    _save_state(args.output_dir, state)
    _log_state("job status", state)


def _acquire_lock(output_dir: str):
    path = _lock_path(output_dir)
    handle = open(path, "a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError(f"Scheduler is already running: {path}") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def _sleep_with_log(seconds: int) -> None:
    sleep_seconds = max(1, int(seconds))
    log(f"sleep {sleep_seconds}s")
    time.sleep(sleep_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run end-of-day gp-data sync on each trading day at a fixed local time")
    parser.add_argument("--output-dir", required=True, help="gp-data root directory")
    parser.add_argument("--token", default="", help="Tushare token; falls back to TUSHARE_TOKEN/GP_TUSHARE_TOKEN")
    parser.add_argument("--schedule-time", default="16:00", help="Local trigger time in HH:MM, default 16:00")
    parser.add_argument("--check-interval-seconds", type=int, default=30, help="Polling interval while waiting")
    parser.add_argument(
        "--status-log-interval-seconds",
        type=int,
        default=300,
        help="Heartbeat interval for logging idle scheduler status",
    )
    parser.add_argument(
        "--failure-retry-interval-seconds",
        type=int,
        default=1800,
        help="Cooldown before retrying the same trade date after a failed scheduled run",
    )
    parser.add_argument("--backfill-open-days", type=int, default=1, help="Open days to refresh in the daily incremental sync")
    parser.add_argument("--financial-threads", type=int, default=16, help="Thread count for financial incremental sync")
    parser.add_argument("--batch-rate", type=int, default=240, help="Tushare batch API rate limit per 60 seconds")
    parser.add_argument("--financial-rate", type=int, default=360, help="Tushare financial API rate limit per 60 seconds")
    parser.add_argument(
        "--financial-datasets",
        default="income,balancesheet,cashflow,fina_indicator",
        help="Comma-separated financial datasets for fast_sync_tushare_latest.py",
    )
    parser.add_argument("--skip-date-based", action="store_true", help="Skip daily/date-based Tushare datasets")
    parser.add_argument("--skip-financials", action="store_true", help="Skip financial Tushare datasets")
    parser.add_argument("--minute-source", choices=["tx", "em", "ts"], default="ts", help="Primary source for current-day 1-minute sync")
    parser.add_argument("--minute-threads", type=int, default=6, help="Worker threads for current-day minute sync")
    parser.add_argument("--minute-fqt", default="0", help="Adjustment mode for minute sync: 0/raw, 1/qfq, 2/hfq")
    parser.add_argument("--minute-force", action="store_true", help="Force refresh current-day minute files")
    parser.add_argument(
        "--minute-retry-failed-rounds",
        type=int,
        default=2,
        help="Retry rounds inside the current-day minute sync",
    )
    parser.add_argument(
        "--minute-retry-sleep-seconds",
        type=float,
        default=2.0,
        help="Sleep seconds between minute retry rounds",
    )
    parser.add_argument("--run-now", action="store_true", help="Run immediately instead of waiting for the next scheduled time")
    parser.add_argument("--run-once", action="store_true", help="Exit after one run or one scheduling check")
    parser.add_argument(
        "--allow-non-trading-day",
        action="store_true",
        help="Only with --run-now: use the latest open trade date when today is not a trading day",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    token = _resolve_token(args.token)
    schedule_time = _parse_schedule_time(args.schedule_time)
    lock_handle = _acquire_lock(args.output_dir)
    status_tracker = {"message": "", "logged_at": None}

    log(
        "scheduler started: "
        f"output_dir={args.output_dir} schedule_time={schedule_time.strftime('%H:%M')} "
        f"check_interval_seconds={args.check_interval_seconds} "
        f"status_log_interval_seconds={args.status_log_interval_seconds} "
        f"state_file={_state_path(args.output_dir)}"
    )

    try:
        state = _load_state(args.output_dir)
        _log_state("loaded state", state)
        while True:
            now = datetime.now()
            if args.run_now:
                trade_date = _resolve_run_trade_date(args.output_dir, now, args.allow_non_trading_day)
                if not trade_date:
                    log("today is not a trading day; nothing to run")
                else:
                    log(f"run-now for trade_date={trade_date}")
                    _run_job(args, token, trade_date, state)
                return

            if not _is_trading_day(args.output_dir, now):
                _maybe_log_status(
                    f"idle: trading_day=no trade_date={now.strftime('%Y%m%d')} {_state_summary(state)}",
                    status_tracker,
                    now,
                    args.status_log_interval_seconds,
                )
                if args.run_once:
                    log("today is not a trading day; exit without running")
                    return
                _sleep_with_log(args.check_interval_seconds)
                continue

            if now.time() < schedule_time:
                _maybe_log_status(
                    "idle: "
                    f"trading_day=yes trade_date={now.strftime('%Y%m%d')} "
                    f"waiting_for_schedule=yes scheduled_time={schedule_time.strftime('%H:%M')} "
                    f"remaining_seconds={_remaining_seconds_until_schedule(now, schedule_time)} "
                    f"{_state_summary(state)}",
                    status_tracker,
                    now,
                    args.status_log_interval_seconds,
                )
                if args.run_once:
                    log("scheduled time not reached yet; exit without running")
                    return
                _sleep_with_log(args.check_interval_seconds)
                continue

            trade_date = now.strftime("%Y%m%d")
            state = _load_state(args.output_dir)
            if not _should_run(state, trade_date, now, args.failure_retry_interval_seconds):
                last_attempt_trade_date = str(state.get("last_attempt_trade_date") or "").strip()
                last_status = str(state.get("last_status") or "").strip().lower()
                if last_attempt_trade_date == trade_date and last_status == "failed":
                    wait_message = (
                        "idle: "
                        f"trade_date={trade_date} retry_pending=yes "
                        f"retry_in_seconds={_remaining_failure_retry_seconds(state, now, args.failure_retry_interval_seconds)} "
                        f"{_state_summary(state)}"
                    )
                else:
                    wait_message = f"idle: trade_date={trade_date} all_done=yes {_state_summary(state)}"
                _maybe_log_status(wait_message, status_tracker, now, args.status_log_interval_seconds)
                if args.run_once:
                    log(f"trade_date={trade_date} already handled; exit")
                    return
                _sleep_with_log(args.check_interval_seconds)
                continue

            log(f"start scheduled sync for trade_date={trade_date}")
            _run_job(args, token, trade_date, state)
            log(f"scheduled sync finished for trade_date={trade_date}")
            status_tracker = {"message": "", "logged_at": None}
            if args.run_once:
                return
    finally:
        try:
            lock_handle.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()