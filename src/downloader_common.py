import json
import os
import random
import threading
import time
from datetime import datetime, timedelta

import pandas as pd
import requests


_FAIL_LOCK = threading.Lock()


class UnsupportedMinuteHistoryError(RuntimeError):
    pass


def record_failure(symbol: str, date: str, output_dir: str) -> None:
    fail_file = os.path.join(output_dir, "failed_tasks.json")
    with _FAIL_LOCK:
        failed_tasks = []
        if os.path.exists(fail_file):
            try:
                with open(fail_file, "r") as f:
                    failed_tasks = json.load(f)
            except Exception:
                failed_tasks = []
        task = {"symbol": symbol, "date": date}
        if task not in failed_tasks:
            failed_tasks.append(task)
        with open(fail_file, "w") as f:
            json.dump(failed_tasks, f, indent=4)


def remove_failure(symbol: str, date: str, output_dir: str) -> None:
    fail_file = os.path.join(output_dir, "failed_tasks.json")
    with _FAIL_LOCK:
        if not os.path.exists(fail_file):
            return
        try:
            with open(fail_file, "r") as f:
                failed_tasks = json.load(f)
            task = {"symbol": symbol, "date": date}
            if task in failed_tasks:
                failed_tasks.remove(task)
                with open(fail_file, "w") as f:
                    json.dump(failed_tasks, f, indent=4)
        except Exception:
            return


def divide_chunks(items: list, chunk_size: int):
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def _symbol_to_em_secid(symbol: str) -> str:
    s = symbol.strip().lower()
    if s.startswith("sh"):
        return f"1.{s[2:]}"
    if s.startswith("sz"):
        return f"0.{s[2:]}"
    if s.startswith("bj"):
        # Beijing exchange symbols map to market_code=0 on Eastmoney push2his
        return f"0.{s[2:]}"
    # Fallback: infer from first digit (6/9 for SH; others SZ)
    # Note: 92xxxx are Beijing codes on Eastmoney (market_code=0)
    if s.startswith("92"):
        market_code = 0
    else:
        market_code = 1 if s.startswith(("6", "9")) else 0
    return f"{market_code}.{s}"


def _normalize_em_fqt(fqt: int | str) -> str:
    """Normalize fqt to Eastmoney expected string.

    Eastmoney kline/get uses:
    - 0: no adjustment
    - 1: forward adjustment (前复权)
    - 2: backward adjustment (后复权)
    """

    if isinstance(fqt, int):
        return str(fqt)
    s = str(fqt).strip().lower()
    if s in {"0", "none", "raw", "nfq"}:
        return "0"
    if s in {"1", "qfq", "forward"}:
        return "1"
    if s in {"2", "hfq", "backward"}:
        return "2"
    raise ValueError(f"Unknown fqt/adj mode: {fqt}")


def _em_headers() -> dict:
    return {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Referer": "https://quote.eastmoney.com/",
        "Accept": "application/json,text/plain,*/*",
    }


def _fetch_em_trends2(symbol: str, ndays: int = 5) -> dict:
    """Fetch recent minute trends via Eastmoney trends2/get.

    Note: This endpoint appears to support only the most recent ~5 trading days.
    """

    url = "https://push2his.eastmoney.com/api/qt/stock/trends2/get"
    params = {
        "secid": _symbol_to_em_secid(symbol),
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "fields1": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58",
        "ndays": str(ndays),
        "iscr": "0",
        "iscca": "0",
    }
    r = requests.get(url, params=params, headers=_em_headers(), timeout=20)
    r.raise_for_status()
    j = r.json()
    data = j.get("data") or {}
    return data


def _fetch_em_daily_klines(symbol: str, beg_yyyymmdd: str, end_yyyymmdd: str, fqt: str) -> list[str]:
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "klt": "101",
        "fqt": fqt,
        "secid": _symbol_to_em_secid(symbol),
        "beg": beg_yyyymmdd,
        "end": end_yyyymmdd,
    }
    r = requests.get(url, params=params, headers=_em_headers(), timeout=20)
    r.raise_for_status()
    j = r.json()
    return (j.get("data") or {}).get("klines") or []


def _daily_close_and_prev_close(symbol: str, date_yyyy_mm_dd: str, fqt: str) -> tuple[float | None, float | None]:
    """Return (close_today, close_prev_trading_day) for the given fqt series."""

    target_dt = datetime.strptime(date_yyyy_mm_dd, "%Y-%m-%d")
    beg = (target_dt - timedelta(days=90)).strftime("%Y%m%d")
    end = target_dt.strftime("%Y%m%d")
    klines = _fetch_em_daily_klines(symbol, beg, end, fqt)
    if not klines:
        return None, None

    # Each kline: YYYY-MM-DD,open,close,high,low,volume,amount,...
    dates: list[str] = []
    closes: list[float] = []
    for item in klines:
        parts = item.split(",")
        if len(parts) < 3:
            continue
        d = parts[0]
        try:
            c = float(parts[2])
        except Exception:
            continue
        dates.append(d)
        closes.append(c)

    if not dates:
        return None, None

    try:
        idx = dates.index(date_yyyy_mm_dd)
    except ValueError:
        return None, None

    close_today = closes[idx]
    close_prev = closes[idx - 1] if idx - 1 >= 0 else None
    return close_today, close_prev


def fetch_em_1m(symbol: str, date_yyyy_mm_dd: str, fqt: int | str = 1) -> pd.DataFrame:
    """Fetch 1-minute bars for a single trading day (Eastmoney-like columns).

    Output columns match Eastmoney minute-bar semantics (from `trends2/get`):
    - 时间 (YYYY-MM-DD HH:MM)
    - 开盘
    - 收盘
    - 最高
    - 最低
    - 成交量(手)
    - 成交额(元)
    - 均价

    Notes:
    - Minute history is only available for the most recent ~5 trading days.
      If the requested date is outside that window, raises UnsupportedMinuteHistoryError.
    - `--adj qfq/hfq` is applied via a daily adjustment factor
      (daily_close_adj / daily_close_raw) for the requested date.
    """

    fqt_str = _normalize_em_fqt(fqt)
    data = _fetch_em_trends2(symbol, ndays=5)
    trends = data.get("trends") or []
    if not trends:
        return pd.DataFrame(columns=["时间", "开盘", "收盘", "最高", "最低", "成交量(手)", "成交额(元)", "均价"])

    available_dates = sorted({str(line).split(" ", 1)[0] for line in trends if isinstance(line, str)})
    if date_yyyy_mm_dd not in available_dates:
        if available_dates:
            raise UnsupportedMinuteHistoryError(
                f"Minute history only available for recent days ({available_dates[0]} ~ {available_dates[-1]}). "
                f"Requested={date_yyyy_mm_dd}."
            )
        raise UnsupportedMinuteHistoryError(f"Minute history unavailable for requested date={date_yyyy_mm_dd}.")

    rows = []
    for item in trends:
        if not isinstance(item, str):
            continue
        parts = item.split(",")
        if len(parts) < 8:
            continue
        dt = parts[0]
        if not dt.startswith(date_yyyy_mm_dd + " "):
            continue
        # trends2 row format (empirical):
        # datetime, open, close, high, low, volume(手), amount(元), avg_price
        try:
            o = float(parts[1])
            c = float(parts[2])
            h = float(parts[3])
            l = float(parts[4])
            v = float(parts[5])
            amt = float(parts[6])
            avg = float(parts[7])
        except Exception:
            continue
        rows.append(
            {
                "时间": dt,
                "_open_raw": o,
                "_close_raw": c,
                "_high_raw": h,
                "_low_raw": l,
                "成交量(手)": v,
                "_amount_raw": amt,
                "_avg_raw": avg,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["时间", "开盘", "收盘", "最高", "最低", "成交量(手)", "成交额(元)", "均价"])

    df = pd.DataFrame(rows)
    df["时间"] = df["时间"].astype(str)

    factor = 1.0
    close_raw_today, _prev_close_raw = _daily_close_and_prev_close(symbol, date_yyyy_mm_dd, "0")
    if fqt_str != "0":
        close_adj_today, _prev_close_adj = _daily_close_and_prev_close(symbol, date_yyyy_mm_dd, fqt_str)
        if close_raw_today and close_adj_today and close_raw_today > 0:
            factor = float(close_adj_today) / float(close_raw_today)

    df["开盘"] = df["_open_raw"].astype(float) * factor
    df["收盘"] = df["_close_raw"].astype(float) * factor
    df["最高"] = df["_high_raw"].astype(float) * factor
    df["最低"] = df["_low_raw"].astype(float) * factor
    df["均价"] = df["_avg_raw"].astype(float) * factor
    df["成交额(元)"] = df["_amount_raw"].astype(float) * factor

    df["成交量(手)"] = pd.to_numeric(df["成交量(手)"], errors="coerce")
    for col in ["开盘", "收盘", "最高", "最低", "成交额(元)", "均价"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("时间").reset_index(drop=True)
    return df[["时间", "开盘", "收盘", "最高", "最低", "成交量(手)", "成交额(元)", "均价"]]


def get_daily(tasks: list[dict], working_path: str, output_dir: str, fqt: int | str = 1) -> None:
    colnames = ["时间", "开盘", "收盘", "最高", "最低", "成交量(手)", "成交额(元)", "均价"]

    for task in tasks:
        symbol = task["symbol"]
        date_str = task["date"]

        print(f"Processing: {symbol} for {date_str}")
        time.sleep(random.uniform(2, 5))

        csv_dir = os.path.join(output_dir, symbol)
        csv_file = f"{csv_dir}/{date_str}.csv"

        if os.path.exists(csv_file):
            try:
                tmp_df = pd.read_csv(csv_file, delimiter=",")
                if not tmp_df.empty:
                    last_row = tmp_df.iloc[-1]
                    last_time = str(last_row.get("时间", ""))
                    last_hhmmss = last_time.split(" ", 1)[1] if " " in last_time else last_time
                    if last_hhmmss and len(last_hhmmss) == 5:
                        last_hhmmss = last_hhmmss + ":00"
                    if last_hhmmss < "14:55:00":
                        print(f"{csv_file} last time {last_time} is early, will re-download")
                        os.remove(csv_file)
                    else:
                        print(f"{symbol} {date_str} already complete.")
                        remove_failure(symbol, date_str, output_dir)
                        continue
                else:
                    os.remove(csv_file)
            except Exception:
                os.remove(csv_file)

        total_detail_df = pd.DataFrame(columns=colnames)
        day_success = True
        try:
            em_df = fetch_em_1m(symbol, date_str, fqt=fqt)
            if not em_df.empty:
                total_detail_df = em_df
                print(f"Fetched {len(em_df)} 1-minute rows from Eastmoney for {symbol} {date_str}")
            else:
                print(f"No data found for {symbol} on {date_str}")
        except Exception as e:
            msg = str(e)
            if len(msg) > 200:
                msg = msg[:200] + "..."
            print(f"Eastmoney fetch failed for {symbol} {date_str}: {type(e).__name__} - {msg}")
            day_success = False

        if day_success:
            if not total_detail_df.empty:
                os.makedirs(csv_dir, exist_ok=True)
                total_detail_df = total_detail_df.sort_values("时间")
                total_detail_df.to_csv(csv_file, index=False)
                print(f"finish {symbol} {date_str}")
            remove_failure(symbol, date_str, output_dir)
        else:
            record_failure(symbol, date_str, output_dir)


def run_tasks_in_threads(
    tasks_to_run: list[dict],
    threads: int,
    working_path: str,
    output_dir: str,
    fqt: int | str = 1,
) -> None:
    if not tasks_to_run:
        print("No tasks to run.")
        return

    chunks_num = max(1, int(threads))
    chunk_size = max(1, (len(tasks_to_run) + chunks_num - 1) // chunks_num)
    tasks_cks = list(divide_chunks(tasks_to_run, chunk_size))

    workers: list[threading.Thread] = []
    for i, chunk in enumerate(tasks_cks):
        t = threading.Thread(target=get_daily, args=(chunk, working_path, output_dir, fqt))
        t.name = f"worker_{i}"
        print(t.name)
        workers.append(t)

    for t in workers:
        t.start()
    for t in workers:
        t.join()
