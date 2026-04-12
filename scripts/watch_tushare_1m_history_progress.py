#!/usr/bin/env python3

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime


WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

CHUNK_PROGRESS_RE = re.compile(
    r"^\[(?P<ts>[^\]]+)\] chunk_progress (?P<range>\S+) "
    r"done=(?P<done>\d+)/(?P<total>\d+) files_written=(?P<files>\d+) "
    r"rows_written=(?P<rows>\d+) failed=(?P<failed>\d+) skipped=(?P<skipped>\d+)"
)
CHUNK_START_RE = re.compile(
    r"^\[(?P<ts>[^\]]+)\] chunk start index=(?P<index>\d+)/(?P<total>\d+) range=(?P<range>\S+)"
)
CHUNK_DONE_RE = re.compile(
    r"^\[(?P<ts>[^\]]+)\] chunk done index=(?P<index>\d+)/(?P<total>\d+) range=(?P<range>\S+) "
    r"files_written=(?P<files>\d+) rows_written=(?P<rows>\d+) "
    r"missing_dates=(?P<missing_dates>\d+) remaining_failures=(?P<remaining_failures>\d+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch historical Tushare 1-minute backfill progress.")
    parser.add_argument("--output-dir", default=_default_output_dir(), help="gp-data root directory")
    parser.add_argument("--refresh-seconds", type=float, default=20.0, help="Refresh interval in seconds")
    parser.add_argument("--bar-width", type=int, default=32, help="ASCII progress bar width")
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit")
    return parser.parse_args()


def _default_output_dir() -> str:
    candidate = os.path.abspath(os.path.join(WORKSPACE_DIR, "..", "gp-data"))
    return candidate


def _state_path(output_dir: str) -> str:
    return os.path.join(output_dir, ".tushare_1m_history_backfill_state.json")


def _lock_path(output_dir: str) -> str:
    return os.path.join(output_dir, ".tushare_1m_history_backfill.lock")


def _log_path() -> str:
    return os.path.join(WORKSPACE_DIR, "results", "logs", "tushare_1m_history_backfill.log")


def _load_state(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _find_lock_holders(lock_path: str) -> list[dict]:
    holders: list[dict] = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        fd_dir = f"/proc/{pid}/fd"
        try:
            for fd in os.listdir(fd_dir):
                if os.readlink(os.path.join(fd_dir, fd)) == lock_path:
                    cmdline = (
                        open(f"/proc/{pid}/cmdline", "rb")
                        .read()
                        .replace(b"\x00", b" ")
                        .decode("utf-8", "replace")
                        .strip()
                    )
                    holders.append({"pid": int(pid), "cmdline": cmdline})
                    break
        except Exception:
            continue
    holders.sort(key=lambda item: item["pid"])
    return holders


def _read_recent_markers(log_path: str, max_lines: int = 600) -> list[str]:
    if not os.path.exists(log_path):
        return []
    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        lines = [line.rstrip("\n") for line in handle.readlines()[-max_lines:]]
    markers = [
        line
        for line in lines
        if "chunk start" in line
        or "chunk_progress" in line
        or "chunk done" in line
        or "historical_backfill finished" in line
        or "historical_backfill failed" in line
    ]
    return markers


def _parse_current_chunk(markers: list[str], current_range: str) -> tuple[dict | None, dict | None, str]:
    current_progress = None
    current_start = None
    latest_marker = markers[-1] if markers else ""

    for line in reversed(markers):
        match = CHUNK_PROGRESS_RE.match(line)
        if match and match.group("range") == current_range:
            current_progress = {
                "ts": match.group("ts"),
                "range": match.group("range"),
                "done": int(match.group("done")),
                "total": int(match.group("total")),
                "files": int(match.group("files")),
                "rows": int(match.group("rows")),
                "failed": int(match.group("failed")),
                "skipped": int(match.group("skipped")),
                "line": line,
            }
            break

    for line in reversed(markers):
        match = CHUNK_START_RE.match(line)
        if match and match.group("range") == current_range:
            current_start = {
                "ts": match.group("ts"),
                "index": int(match.group("index")),
                "total": int(match.group("total")),
                "range": match.group("range"),
                "line": line,
            }
            break

    return current_start, current_progress, latest_marker


def _render_bar(ratio: float, width: int) -> str:
    clamped = max(0.0, min(1.0, ratio))
    filled = int(round(clamped * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _format_int(value: int) -> str:
    return f"{int(value):,}"


def _snapshot(args: argparse.Namespace) -> dict:
    state_path = _state_path(args.output_dir)
    lock_path = _lock_path(args.output_dir)
    log_path = _log_path()

    state = _load_state(state_path)
    holders = _find_lock_holders(lock_path)
    markers = _read_recent_markers(log_path)
    current_range = f"{state.get('current_chunk_start', '')}..{state.get('current_chunk_end', '')}"
    current_start, current_progress, latest_marker = _parse_current_chunk(markers, current_range)

    completed_chunks = int(state.get("current_chunk_index", 0))
    total_chunks = max(1, int(state.get("total_chunks", 0) or 0))
    current_done = int(current_progress["done"]) if current_progress else 0
    current_total = max(1, int(current_progress["total"])) if current_progress else max(1, int(state.get("symbols_total", 1)))
    overall_ratio = min(1.0, (completed_chunks + current_done / current_total) / total_chunks)
    chunk_ratio = min(1.0, current_done / current_total)
    current_chunk_number = current_start["index"] if current_start else min(total_chunks, completed_chunks + (1 if state.get("status") == "running" else 0))

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "state": state,
        "holders": holders,
        "current_range": current_range,
        "current_start": current_start,
        "current_progress": current_progress,
        "latest_marker": latest_marker,
        "overall_ratio": overall_ratio,
        "chunk_ratio": chunk_ratio,
        "completed_chunks": completed_chunks,
        "current_chunk_number": current_chunk_number,
        "total_chunks": total_chunks,
        "current_done": current_done,
        "current_total": current_total,
    }


def _render_tty(snapshot: dict, bar_width: int) -> str:
    state = snapshot["state"]
    progress = snapshot["current_progress"]
    holders = snapshot["holders"]
    overall_bar = _render_bar(snapshot["overall_ratio"], bar_width)
    chunk_bar = _render_bar(snapshot["chunk_ratio"], bar_width)
    active_pid = holders[0]["pid"] if holders else "-"
    lines = [
        f"[{snapshot['timestamp']}] status={state.get('status')} active_pid={active_pid}",
        f"overall {overall_bar} {snapshot['overall_ratio'] * 100:6.2f}%  chunks={snapshot['completed_chunks']}/{snapshot['total_chunks']} done current={snapshot['current_chunk_number']}/{snapshot['total_chunks']}",
        f"current {chunk_bar} {snapshot['chunk_ratio'] * 100:6.2f}%  symbols={snapshot['current_done']}/{snapshot['current_total']}  range={snapshot['current_range']}",
        f"totals  files={_format_int(state.get('files_written', 0))} rows={_format_int(state.get('rows_written', 0))} missing_dates={_format_int(state.get('missing_dates', 0))} chunks_failed={_format_int(state.get('chunks_failed', 0))}",
        f"detail  failed={progress['failed'] if progress else 0} skipped={progress['skipped'] if progress else 0} latest={snapshot['latest_marker']}",
    ]
    return "\n".join(lines)


def _render_log_line(snapshot: dict, bar_width: int) -> str:
    state = snapshot["state"]
    progress = snapshot["current_progress"]
    holders = snapshot["holders"]
    overall_bar = _render_bar(snapshot["overall_ratio"], bar_width)
    chunk_bar = _render_bar(snapshot["chunk_ratio"], bar_width)
    return (
        f"[{snapshot['timestamp']}] status={state.get('status')} active_pid={holders[0]['pid'] if holders else '-'} "
        f"overall={overall_bar} {snapshot['overall_ratio'] * 100:5.1f}% chunks={snapshot['completed_chunks']}/{snapshot['total_chunks']} "
        f"current={snapshot['current_chunk_number']}/{snapshot['total_chunks']} chunk={chunk_bar} {snapshot['chunk_ratio'] * 100:5.1f}% "
        f"symbols={snapshot['current_done']}/{snapshot['current_total']} range={snapshot['current_range']} "
        f"files={state.get('files_written', 0)} rows={state.get('rows_written', 0)} missing_dates={state.get('missing_dates', 0)} "
        f"chunk_failed={progress['failed'] if progress else 0} skipped={progress['skipped'] if progress else 0}"
    )


def _snapshot_key(snapshot: dict) -> tuple:
    state = snapshot["state"]
    progress = snapshot["current_progress"] or {}
    holders = snapshot["holders"]
    return (
        state.get("status"),
        holders[0]["pid"] if holders else None,
        snapshot["completed_chunks"],
        snapshot["current_chunk_number"],
        snapshot["total_chunks"],
        snapshot["current_done"],
        snapshot["current_total"],
        snapshot["current_range"],
        state.get("files_written", 0),
        state.get("rows_written", 0),
        state.get("missing_dates", 0),
        state.get("chunks_failed", 0),
        progress.get("failed", 0),
        progress.get("skipped", 0),
    )


def main() -> int:
    args = parse_args()
    last_output = None
    last_key = None
    while True:
        snapshot = _snapshot(args)
        snapshot_key = _snapshot_key(snapshot)
        if sys.stdout.isatty():
            rendered = _render_tty(snapshot, args.bar_width)
            if snapshot_key != last_key:
                sys.stdout.write("\x1b[2J\x1b[H")
                sys.stdout.write(rendered + "\n")
                sys.stdout.flush()
                last_output = rendered
                last_key = snapshot_key
        else:
            rendered = _render_log_line(snapshot, args.bar_width)
            if snapshot_key != last_key:
                print(rendered, flush=True)
                last_output = rendered
                last_key = snapshot_key

        if args.once or snapshot["state"].get("status") != "running":
            return 0
        time.sleep(max(1.0, float(args.refresh_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())