"""Supervisor agent — monitors, schedules, retries, and alerts for all data agents.

Usage:
  # Show status dashboard
  python supervisor.py --data-dir /path/to/gp-data status

  # Run daily scheduled sync (replaces eod_data_scheduler)
  python supervisor.py --data-dir /path/to/gp-data run --token <token>

  # Run a specific agent
  python supervisor.py --data-dir /path/to/gp-data run --token <token> --agent daily_financial

  # Daemon mode: wait until schedule_time, then run
  python supervisor.py --data-dir /path/to/gp-data daemon --token <token> --schedule-time 16:00
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from base_agent import AgentState

# Agent dependency graph: {name: [depends_on]}
AGENT_REGISTRY = {
    "stock_list": {
        "depends_on": [],
        "priority": 0,
        "module": "agent_stock_list",
        "class": "StockListAgent",
        "description": "同步股票列表",
        "daily": True,
    },
    "daily_financial": {
        "depends_on": ["stock_list"],
        "priority": 1,
        "module": "agent_daily_financial",
        "class": "DailyFinancialAgent",
        "description": "日线行情 + 财务数据",
        "daily": True,
    },
    "market_data": {
        "depends_on": ["stock_list"],
        "priority": 1,
        "module": "agent_market_data",
        "class": "MarketDataAgent",
        "description": "资金流 / 指数 / 市场数据",
        "daily": True,
    },
    "minute": {
        "depends_on": ["stock_list"],
        "priority": 2,
        "module": "agent_minute",
        "class": "MinuteAgent",
        "description": "1 分钟 K 线数据",
        "daily": True,
    },
    "derived": {
        "depends_on": ["daily_financial"],
        "priority": 3,
        "module": "agent_derived",
        "class": "DerivedDataAgent",
        "description": "衍生数据（周线等）",
        "daily": True,
    },
    "market_trend": {
        "depends_on": ["derived", "market_data"],
        "priority": 4,
        "module": "agent_market_trend",
        "class": "MarketTrendAgent",
        "description": "大盘趋势判断（7 维度评分 + 报告）",
        "daily": True,
    },
}

ALERT_LOG = ".agent_alerts.log"


def _load_agent_state(data_dir: str, name: str) -> dict:
    s = AgentState(data_dir, name)
    return s.data


def _is_trading_day(data_dir: str) -> bool:
    cal_file = os.path.join(data_dir, "tushare-trade_cal", "trade_cal.csv")
    today = datetime.now().strftime("%Y%m%d")
    if os.path.exists(cal_file):
        try:
            import csv
            with open(cal_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("cal_date") == today:
                        return str(row.get("is_open", "0")) == "1"
        except Exception:
            pass
    # Fallback: weekday check
    return datetime.now().weekday() < 5


def cmd_status(data_dir: str):
    """Print status dashboard for all agents."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'=' * 76}")
    print(f"  Agent Supervisor Dashboard — {now}")
    print(f"{'=' * 76}")
    print(f"  {'Agent':<20} {'Status':<10} {'Progress':<20} {'Last Success':<20} {'Fail#'}")
    print(f"  {'-' * 20} {'-' * 10} {'-' * 20} {'-' * 20} {'-' * 5}")

    for name in AGENT_REGISTRY:
        state = _load_agent_state(data_dir, name)
        status = state.get("status", "idle")
        progress = state.get("progress", {})
        phase = progress.get("phase", "")
        pct = progress.get("pct", 0)
        last_ok = state.get("last_success_at", "") or "—"
        fails = state.get("consecutive_failures", 0)
        dur = state.get("stats", {}).get("duration_seconds", 0)

        # Status color indicator
        icon = {"idle": "⏸", "running": "▶", "success": "✓", "failed": "✗"}.get(status, "?")

        prog_str = f"{phase} {pct:.0f}%" if phase else "—"
        if status == "success" and dur:
            prog_str = f"done ({dur:.0f}s)"

        print(f"  {name:<20} {icon} {status:<8} {prog_str:<20} {last_ok:<20} {fails}")

    # Show last error for failed agents
    has_errors = False
    for name in AGENT_REGISTRY:
        state = _load_agent_state(data_dir, name)
        if state.get("status") == "failed" and state.get("last_error"):
            if not has_errors:
                print(f"\n  Errors:")
                has_errors = True
            err = state["last_error"][:100]
            print(f"    [{name}] {err}")

    print(f"{'=' * 76}\n")


def _run_agent(data_dir: str, name: str, token: str, extra_kwargs: dict | None = None) -> bool:
    """Run a single agent in-process."""
    reg = AGENT_REGISTRY[name]
    module_name = reg["module"]
    class_name = reg["class"]

    # Dynamic import
    import importlib
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    agent = cls(data_dir, token)
    return agent.execute(**(extra_kwargs or {}))


def _alert(data_dir: str, agent_name: str, status: str, message: str):
    """Write alert to log file."""
    ts = datetime.now().isoformat(timespec="seconds")
    line = f"[{ts}] [{agent_name}] {status}: {message}\n"
    log_path = os.path.join(data_dir, ALERT_LOG)
    with open(log_path, "a") as f:
        f.write(line)
    print(f"  ALERT: {line.strip()}")


def cmd_run(data_dir: str, token: str, agent_name: str | None = None,
            max_retries: int = 3, retry_interval: int = 300, daily_only: bool = True):
    """Run agents in dependency order with retry."""

    if agent_name:
        # Run single agent
        if agent_name not in AGENT_REGISTRY:
            print(f"Unknown agent: {agent_name}. Available: {', '.join(AGENT_REGISTRY)}")
            sys.exit(1)
        ok = _run_agent(data_dir, agent_name, token)
        if not ok:
            _alert(data_dir, agent_name, "FAILED", _load_agent_state(data_dir, agent_name).get("last_error", ""))
        return ok

    # Run all agents in dependency order
    # Topological sort by priority
    sorted_agents = sorted(AGENT_REGISTRY.items(), key=lambda x: x[1]["priority"])

    results: dict[str, bool] = {}
    for name, reg in sorted_agents:
        if daily_only and not reg["daily"]:
            print(f"[supervisor] Skipping {name} (not daily)")
            continue

        # Check dependencies
        deps_ok = all(results.get(d, False) for d in reg["depends_on"])
        if not deps_ok:
            failed_deps = [d for d in reg["depends_on"] if not results.get(d, False)]
            print(f"[supervisor] Skipping {name}: dependency failed ({', '.join(failed_deps)})")
            _alert(data_dir, name, "SKIPPED", f"dependency failed: {', '.join(failed_deps)}")
            results[name] = False
            continue

        # Run with retries
        ok = False
        for attempt in range(1, max_retries + 1):
            print(f"\n[supervisor] ▶ Running {name} — {reg['description']} (attempt {attempt}/{max_retries})")
            ok = _run_agent(data_dir, name, token)
            state = _load_agent_state(data_dir, name)
            dur = state.get("stats", {}).get("duration_seconds", 0)
            if ok:
                print(f"[supervisor] ✓ {name} completed in {dur:.0f}s")
                break
            if attempt < max_retries:
                _alert(data_dir, name, "RETRY", f"attempt {attempt} failed: {state.get('last_error', '')}")
                print(f"[supervisor] {name} failed, retry in {retry_interval}s...")
                time.sleep(retry_interval)

        results[name] = ok
        if not ok:
            state = _load_agent_state(data_dir, name)
            _alert(data_dir, name, "FAILED", state.get("last_error", ""))

    # Summary
    print(f"\n{'=' * 60}")
    print(f"[supervisor] Run summary:")
    print(f"  {'Agent':<20} {'Status':<10} {'Duration'}")
    print(f"  {'-' * 20} {'-' * 10} {'-' * 10}")
    for name, ok in results.items():
        icon = "✓" if ok else "✗"
        state = _load_agent_state(data_dir, name)
        dur = state.get("stats", {}).get("duration_seconds", 0)
        dur_str = f"{dur:.0f}s" if dur < 3600 else f"{dur/3600:.1f}h"
        print(f"  {name:<20} {icon} {'ok':<8} {dur_str}" if ok else f"  {name:<20} {icon} {'FAIL':<8} {dur_str}")
    print(f"{'=' * 60}")

    all_ok = all(results.values())
    if all_ok:
        _alert(data_dir, "supervisor", "SUCCESS", "All agents completed successfully")
    else:
        failed = [n for n, ok in results.items() if not ok]
        _alert(data_dir, "supervisor", "PARTIAL_FAILURE", f"Failed agents: {', '.join(failed)}")

    return all_ok


def cmd_daemon(data_dir: str, token: str, schedule_time: str = "16:00",
               max_retries: int = 3, retry_interval: int = 300):
    """Daemon mode: run daily at schedule_time on trading days."""
    print(f"[supervisor] Daemon started. Schedule: {schedule_time} on trading days.")
    print(f"[supervisor] Data dir: {data_dir}")

    while True:
        now = datetime.now()
        target_hm = schedule_time.split(":")
        target_h, target_m = int(target_hm[0]), int(target_hm[1])

        # Calculate seconds until next run
        target = now.replace(hour=target_h, minute=target_m, second=0, microsecond=0)
        if now >= target:
            # Already past today's schedule, wait for tomorrow
            from datetime import timedelta
            target += timedelta(days=1)

        wait = (target - now).total_seconds()
        print(f"[supervisor] Next run at {target.strftime('%Y-%m-%d %H:%M')} (in {wait/3600:.1f}h)")

        # Wait in 60s chunks to allow status checks
        while wait > 0:
            time.sleep(min(60, wait))
            wait -= 60

        # Check trading day
        if not _is_trading_day(data_dir):
            print(f"[supervisor] {datetime.now().strftime('%Y-%m-%d')} is not a trading day, skip.")
            continue

        print(f"\n[supervisor] === Daily sync starting at {datetime.now().isoformat(timespec='seconds')} ===")
        cmd_run(data_dir, token, max_retries=max_retries, retry_interval=retry_interval)
        cmd_status(data_dir)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Agent Supervisor — monitor, schedule, and manage data agents")
    parser.add_argument("--data-dir", required=True, help="gp-data root directory")
    sub = parser.add_subparsers(dest="command", required=True)

    # status
    sub.add_parser("status", help="Show agent status dashboard")

    # run
    run_p = sub.add_parser("run", help="Run agents (all or specific)")
    run_p.add_argument("--token", default="", help="Tushare token")
    run_p.add_argument("--agent", default=None, help="Run specific agent")
    run_p.add_argument("--max-retries", type=int, default=3)
    run_p.add_argument("--retry-interval", type=int, default=300, help="Seconds between retries")
    run_p.add_argument("--all", action="store_true", help="Include non-daily agents (market_data)")

    # daemon
    daemon_p = sub.add_parser("daemon", help="Daemon mode: daily scheduled runs")
    daemon_p.add_argument("--token", required=True, help="Tushare token")
    daemon_p.add_argument("--schedule-time", default="16:00", help="HH:MM trigger time")
    daemon_p.add_argument("--max-retries", type=int, default=3)
    daemon_p.add_argument("--retry-interval", type=int, default=300)

    args = parser.parse_args()
    data_dir = os.path.abspath(args.data_dir)

    if args.command == "status":
        cmd_status(data_dir)
    elif args.command == "run":
        token = args.token or os.getenv("TUSHARE_TOKEN", "") or os.getenv("GP_TUSHARE_TOKEN", "")
        ok = cmd_run(data_dir, token, agent_name=args.agent, max_retries=args.max_retries,
                     retry_interval=args.retry_interval, daily_only=not args.all)
        sys.exit(0 if ok else 1)
    elif args.command == "daemon":
        cmd_daemon(data_dir, args.token, schedule_time=args.schedule_time,
                   max_retries=args.max_retries, retry_interval=args.retry_interval)


if __name__ == "__main__":
    main()
