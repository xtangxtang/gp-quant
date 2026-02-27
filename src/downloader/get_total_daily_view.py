import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

_WEB_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "web"))
if _WEB_DIR not in sys.path:
    sys.path.insert(0, _WEB_DIR)

from eastmoney_daily import default_data_dir, fetch_latest_daily_summary
from csv_utils import write_rows_csv


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _cache_dir(output_dir: str) -> str:
    return os.path.join(os.path.abspath(output_dir), "total-daily-view")


def _save_rows_to_csv(path: str, rows: list[dict]) -> None:
    fieldnames = [
        "symbol",
        "symbol_name",
        "date",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "pctchg",
        "turnover",
        "chg",
        "amplitude",
    ]
    write_rows_csv(path, fieldnames=fieldnames, rows=rows)


def _load_symbols_from_file(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return [str(x).strip() for x in obj if str(x).strip()]
    if isinstance(obj, dict) and isinstance(obj.get("symbols"), list):
        return [str(x).strip() for x in obj["symbols"] if str(x).strip()]
    return []


def _resolve_symbol_list(data_dir: str, mode: str) -> tuple[list[str], str]:
    data_dir = os.path.abspath(data_dir)
    total_path = os.path.join(data_dir, "total_gplist.json")
    self_path = os.path.join(data_dir, "self_gplist.json")

    m = (mode or "auto").strip().lower()
    if m == "total":
        return _load_symbols_from_file(total_path), total_path
    if m == "self":
        return _load_symbols_from_file(self_path), self_path

    # auto
    if os.path.exists(total_path):
        syms = _load_symbols_from_file(total_path)
        if syms:
            return syms, total_path
    if os.path.exists(self_path):
        return _load_symbols_from_file(self_path), self_path
    return [], ""


def main() -> None:
    p = argparse.ArgumentParser(
        description="Download latest trading-day daily summaries for all symbols in total_gplist.json/self_gplist.json"
    )
    p.add_argument(
        "--output_dir",
        dest="output_dir",
        default=None,
        help="Directory that contains total_gplist.json (and used to store total-daily-view cache)",
    )
    p.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        help="Alias of --output_dir",
    )
    p.add_argument("--list", dest="list_mode", default="total", choices=["auto", "total", "self"])
    p.add_argument("--threads", type=int, default=20)
    p.add_argument("--adj", default="none", choices=["none", "qfq", "hfq"], help="Daily kline adj mode")
    p.add_argument("--force", action="store_true", help="Force refresh even if cache exists")
    args = p.parse_args()

    out_dir = args.output_dir or default_data_dir()
    out_dir = os.path.abspath(out_dir)

    syms, from_file = _resolve_symbol_list(out_dir, args.list_mode)
    effective_mode = (args.list_mode or "auto").strip().lower()
    if not syms and effective_mode != "self":
        syms, from_file = _resolve_symbol_list(out_dir, "self")
        if syms:
            effective_mode = "self"

    if not syms:
        raise SystemExit(
            f"No symbols found. Checked output_dir={out_dir} file={from_file or '(none)'}; "
            "please run total download first (to generate total_gplist.json) or create self_gplist.json."
        )

    t0 = time.time()
    benchmark_date = None
    try:
        bench = fetch_latest_daily_summary(syms[0], adj=args.adj)
        if bench and bench.get("date"):
            benchmark_date = str(bench.get("date"))
    except Exception:
        benchmark_date = None

    cache_file = None
    if benchmark_date:
        cache_file = os.path.join(_cache_dir(out_dir), f"{benchmark_date}.csv")
        if os.path.exists(cache_file) and not args.force:
            print(f"Cache exists, skip: {cache_file}")
            raise SystemExit(0)

    max_workers = max(1, int(args.threads))
    rows: list[dict] = []
    errors: list[dict] = []

    def _work(sym: str):
        s = sym.strip()
        if not s:
            return None
        try:
            r = fetch_latest_daily_summary(s, adj=args.adj)
            if not r:
                return {"symbol": s, "_error": "No data"}
            return r
        except Exception as e:
            return {"symbol": s, "_error": f"{type(e).__name__}: {str(e)[:180]}"}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_work, s) for s in syms]
        for fut in as_completed(futures):
            r = fut.result()
            if not r:
                continue
            if r.get("_error"):
                errors.append(r)
                continue
            rows.append(r)

    rows.sort(key=lambda x: str(x.get("symbol") or ""))

    date_for_file = benchmark_date
    if not date_for_file and rows:
        counts: dict[str, int] = {}
        for r in rows:
            d = str(r.get("date") or "")
            if not d:
                continue
            counts[d] = counts.get(d, 0) + 1
        if counts:
            date_for_file = max(counts.items(), key=lambda kv: kv[1])[0]

    if not date_for_file:
        raise SystemExit("Unable to determine trading date from fetched data")

    cache_file = os.path.join(_cache_dir(out_dir), f"{date_for_file}.csv")
    _save_rows_to_csv(cache_file, rows)

    print(
        json.dumps(
            {
                "ok": True,
                "output_dir": out_dir,
                "symbols": len(syms),
                "threads": max_workers,
                "adj": args.adj,
                "list_mode": effective_mode,
                "list_file": from_file,
                "date": date_for_file,
                "cache_file": cache_file,
                "errors": len(errors),
                "elapsed_ms": int((time.time() - t0) * 1000),
            },
            ensure_ascii=False,
        )
    )

    if errors:
        # Print a small preview to help debugging without flooding.
        print(f"Errors (showing up to 10/{len(errors)}):")
        for e in errors[:10]:
            print(e)


if __name__ == "__main__":
    main()
