import json
import os
import random
import threading
import _thread
import time
import pandas as pd
import requests

try:
    from tushare_provider import fetch_ts_float_share_map
except Exception:
    fetch_ts_float_share_map = None


_FAIL_LOCK = threading.Lock()
_FLOAT_SHARES_LOCK = threading.Lock()
_FLOAT_SHARES_CACHE: dict[str, float | None] = {}
_FLOAT_SHARES_BY_DATE_CACHE: dict[str, dict[str, float]] = {}
_PROXY_ABORT_LOCK = threading.Lock()
_PROXY_ABORT_EVENT = threading.Event()
_PROXY_ABORT_MESSAGE = ""

_TS_DISABLE_LOCK = threading.Lock()
_TS_MINUTE_DISABLED = False
_TS_MINUTE_DISABLED_REASON = ""
_MISSING_DEP_WARNINGS: set[str] = set()


_NO_PROXY = os.getenv("GP_NO_PROXY", "").strip().lower() in {"1", "true", "yes"}


_HTTP_LOCAL = threading.local()


def _get_http_session() -> requests.Session:
    s = getattr(_HTTP_LOCAL, "session", None)
    if s is None:
        s = requests.Session()
        if _NO_PROXY:
            s.trust_env = False
        _HTTP_LOCAL.session = s
    return s


def _warn_missing_dependency(name: str, reason: str) -> None:
    with _FAIL_LOCK:
        if name in _MISSING_DEP_WARNINGS:
            return
        _MISSING_DEP_WARNINGS.add(name)
    print(f"[WARN] Optional dependency {name} unavailable: {reason}")


_HTTP_MAX_INFLIGHT = max(1, int(os.getenv("GP_HTTP_MAX_INFLIGHT", "4") or "4"))
_HTTP_SEM = threading.BoundedSemaphore(_HTTP_MAX_INFLIGHT)

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
}


def _http_get_json(url: str, params: dict, timeout: int = 20, retries: int = 3, headers: dict | None = None) -> dict:
    last_err: Exception | None = None
    for i in range(max(1, int(retries))):
        try:
            with _HTTP_SEM:
                r = _get_http_session().get(url, params=params, headers=headers or _DEFAULT_HEADERS, timeout=timeout)
                r.raise_for_status()
                return r.json()
        except requests.exceptions.ProxyError as e:
            last_err = e
            # If the proxy tunnel hiccups, retry with backoff. Direct connect is usually blocked
            # in corporate networks, so we don't automatically switch to direct here.
        except requests.exceptions.RequestException as e:
            last_err = e

        if i < retries - 1:
            # Exponential-ish backoff with jitter to reduce proxy burst failures.
            time.sleep(min(8.0, 0.6 * (2**i)) + random.uniform(0.2, 1.0))

    if last_err is None:
        raise RuntimeError("HTTP request failed for unknown reason")
    raise last_err


class UnsupportedMinuteHistoryError(RuntimeError):
    pass


class ProxyConnectivityError(RuntimeError):
    pass


class RateLimitedError(RuntimeError):
    pass


def _looks_like_tushare_rate_limit(exc: Exception) -> bool:
    # Tushare SDK sometimes prints the Chinese message to stdout and raises OSError('ERROR').
    if isinstance(exc, OSError) and str(exc).strip().upper() == "ERROR":
        return True
    msg = str(exc)
    return (
        "最多访问" in msg
        or "权限" in msg and "详情" in msg
        or "tushare" in msg.lower() and "token" in msg.lower()
    )


def _disable_tushare_minute(reason: str) -> None:
    global _TS_MINUTE_DISABLED, _TS_MINUTE_DISABLED_REASON
    with _TS_DISABLE_LOCK:
        if not _TS_MINUTE_DISABLED:
            _TS_MINUTE_DISABLED = True
            _TS_MINUTE_DISABLED_REASON = reason
            print(f"[WARN] Disabling tushare minute source for this run: {reason}")


def _tushare_minute_disabled() -> bool:
    with _TS_DISABLE_LOCK:
        return _TS_MINUTE_DISABLED


def _is_proxy_error(exc: Exception) -> bool:
    if isinstance(exc, requests.exceptions.ProxyError):
        return True
    msg = str(exc).lower()
    return "proxy" in msg and "error" in msg


def _set_proxy_abort(message: str) -> None:
    global _PROXY_ABORT_MESSAGE
    with _PROXY_ABORT_LOCK:
        if not _PROXY_ABORT_EVENT.is_set():
            _PROXY_ABORT_MESSAGE = message
        _PROXY_ABORT_EVENT.set()


def _get_proxy_abort_message() -> str:
    with _PROXY_ABORT_LOCK:
        return _PROXY_ABORT_MESSAGE


def _get_float_shares_cached(symbol: str) -> float | None:
    with _FLOAT_SHARES_LOCK:
        if symbol in _FLOAT_SHARES_CACHE:
            return _FLOAT_SHARES_CACHE[symbol]
    return None


def _get_tushare_float_share_map_cached(date_yyyy_mm_dd: str) -> dict[str, float]:
    trade_date = str(date_yyyy_mm_dd or "").strip().replace("-", "")
    if not trade_date:
        return {}

    with _FLOAT_SHARES_LOCK:
        cached = _FLOAT_SHARES_BY_DATE_CACHE.get(trade_date)
        if cached is not None:
            return cached

    mapping: dict[str, float] = {}
    if _has_tushare_token():
        try:
            if fetch_ts_float_share_map is None:
                _warn_missing_dependency(
                    "tushare_provider",
                    "historical minute turnover will fall back to current float shares",
                )
            else:
                mapping = fetch_ts_float_share_map(trade_date)
        except Exception:
            mapping = {}

    with _FLOAT_SHARES_LOCK:
        _FLOAT_SHARES_BY_DATE_CACHE[trade_date] = mapping
    return mapping


def _get_float_shares_for_trade_date(symbol: str, date_yyyy_mm_dd: str) -> float | None:
    trade_date_map = _get_tushare_float_share_map_cached(date_yyyy_mm_dd)
    float_shares = trade_date_map.get(symbol)
    if float_shares is not None and float_shares > 0:
        return float_shares
    return _get_float_shares_cached(symbol)


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


def _normalize_em_fqt(fqt: int | str) -> str:
    """Normalize fqt string.

    Values:
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


def _tx_headers() -> dict:
    return {
        "User-Agent": _DEFAULT_HEADERS["User-Agent"],
        "Referer": "https://gu.qq.com/",
        "Accept": "application/json,text/plain,*/*",
    }


def _symbol_to_tx_code(symbol: str) -> str:
    s = (symbol or "").strip().lower()
    if s.startswith(("sh", "sz", "bj")):
        return s
    if s.startswith("92"):
        return "bj" + s
    if s.startswith(("6", "9")):
        return "sh" + s
    return "sz" + s


def minute_source() -> str:
    """Return minute data source selector.

    Controlled by env var GP_MINUTE_SOURCE:
      - ts/tushare
      - tx/tencent
    """

    s = (os.getenv("GP_MINUTE_SOURCE", "") or "").strip().lower()
    if s in {"ts", "tushare"} or not s:
        return "ts"
    if s in {"tx", "tencent"}:
        return "tx"
    return "ts"


def _has_tushare_token() -> bool:
    token = (os.getenv("TUSHARE_TOKEN", "") or "").strip()
    if token:
        return True
    token = (os.getenv("GP_TUSHARE_TOKEN", "") or "").strip()
    return bool(token)


def _fetch_tx_minute_query(symbol: str) -> tuple[str, list[str]]:
    """Fetch latest 1-minute data via Tencent minute/query.

    Returns:
      (date_yyyymmdd, rows)

    Row format (empirical): "HHMM price volume amount".
    """

    url = "https://web.ifzq.gtimg.cn/appstock/app/minute/query"
    code = _symbol_to_tx_code(symbol)
    j = _http_get_json(url, params={"code": code}, timeout=20, retries=6, headers=_tx_headers())
    node = (j.get("data") or {}).get(code) or {}
    core = node.get("data") or {}
    date = str(core.get("date") or "").strip()
    rows = core.get("data") or []
    if not isinstance(rows, list):
        rows = []
    rows = [str(x) for x in rows if isinstance(x, (str, int, float))]
    return date, rows


def fetch_tx_1m(symbol: str, date_yyyy_mm_dd: str, fqt: int | str = 1) -> pd.DataFrame:
    """Fetch 1-minute bars for a single trading day via Tencent.

    Output columns:
    - 时间 (YYYY-MM-DD HH:MM)
    - 开盘/收盘/最高/最低/均价 (腾讯 minute/query only provides one price per minute;
      we map it to OHLC=均价=price)
    - 成交量(手), 成交额(元)
    """

    fqt_str = _normalize_em_fqt(fqt)
    date_yyyymmdd, raw_rows = _fetch_tx_minute_query(symbol)
    expected_yyyymmdd = date_yyyy_mm_dd.replace("-", "")
    if date_yyyymmdd and date_yyyymmdd != expected_yyyymmdd:
        raise UnsupportedMinuteHistoryError(
            f"Tencent minute/query only returned date={date_yyyymmdd} for symbol={symbol}; requested={expected_yyyymmdd}."
        )

    rows = []
    for item in raw_rows:
        parts = str(item).strip().split()
        if len(parts) < 2:
            continue
        hhmm = parts[0]
        if len(hhmm) != 4 or not hhmm.isdigit():
            continue
        try:
            price = float(parts[1])
        except Exception:
            continue

        vol = None
        amt = None
        if len(parts) >= 3:
            try:
                vol = float(parts[2])
            except Exception:
                vol = None
        if len(parts) >= 4:
            try:
                amt = float(parts[3])
            except Exception:
                amt = None

        dt = f"{date_yyyy_mm_dd} {hhmm[0:2]}:{hhmm[2:4]}"
        rows.append(
            {
                "时间": dt,
                "_open_raw": price,
                "_close_raw": price,
                "_high_raw": price,
                "_low_raw": price,
                "成交量(手)": vol,
                "_amount_raw": amt,
                "_avg_raw": price,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["时间", "开盘", "收盘", "最高", "最低", "成交量(手)", "成交额(元)", "均价"])

    df = pd.DataFrame(rows)
    df["时间"] = df["时间"].astype(str)

    df["开盘"] = pd.to_numeric(df["_open_raw"], errors="coerce")
    df["收盘"] = pd.to_numeric(df["_close_raw"], errors="coerce")
    df["最高"] = pd.to_numeric(df["_high_raw"], errors="coerce")
    df["最低"] = pd.to_numeric(df["_low_raw"], errors="coerce")
    df["均价"] = pd.to_numeric(df["_avg_raw"], errors="coerce")
    df["成交额(元)"] = pd.to_numeric(df["_amount_raw"], errors="coerce")

    df["成交量(手)"] = pd.to_numeric(df["成交量(手)"], errors="coerce")

    df = df.sort_values("时间").reset_index(drop=True)
    return df[["时间", "开盘", "收盘", "最高", "最低", "成交量(手)", "成交额(元)", "均价"]]


def fetch_ts_1m(symbol: str, date_yyyy_mm_dd: str, fqt: int | str = 1) -> pd.DataFrame:
    from tushare_provider import fetch_ts_minute_df

    try:
        df = fetch_ts_minute_df(symbol, date_yyyy_mm_dd, fqt=fqt)
    except Exception as e:
        if _looks_like_tushare_rate_limit(e):
            _disable_tushare_minute(
                "rate-limit/permission reached (tushare minute often requires separate permission; trial is ~2 calls/min)"
            )
            raise RateLimitedError("tushare minute rate-limited/permission denied") from e
        raise
    if df is None or df.empty:
        return pd.DataFrame(columns=["时间", "开盘", "收盘", "最高", "最低", "成交量(手)", "成交额(元)", "均价"])
    # Ensure canonical column set.
    for c in ["时间", "开盘", "收盘", "最高", "最低", "成交量(手)", "成交额(元)", "均价"]:
        if c not in df.columns:
            df[c] = pd.NA
    return df[["时间", "开盘", "收盘", "最高", "最低", "成交量(手)", "成交额(元)", "均价"]]


def fetch_1m(symbol: str, date_yyyy_mm_dd: str, fqt: int | str = 1) -> tuple[pd.DataFrame, str]:
    """Fetch 1-minute bars with source selection and fallback.

    Primary source is controlled by GP_MINUTE_SOURCE (default: ts).
    Fallback tries the other source on ProxyError or UnsupportedMinuteHistoryError.
    """

    primary = minute_source()
    candidates = ["ts"]
    if not _has_tushare_token():
        raise RuntimeError("Tushare token required for minute data. Set TUSHARE_TOKEN env or pass --token.")
    if _tushare_minute_disabled():
        raise RuntimeError("Tushare minute data source has been disabled due to rate-limit/permission issues.")
    order = [primary] + [x for x in candidates if x != primary]

    first_err: Exception | None = None
    for src in order:
        try:
            if src == "ts":
                return fetch_ts_1m(symbol, date_yyyy_mm_dd, fqt=fqt), "ts"
            if src == "tx":
                return fetch_tx_1m(symbol, date_yyyy_mm_dd, fqt=fqt), "tx"
        except Exception as e:
            if first_err is None:
                first_err = e
            if _is_proxy_error(e) or isinstance(e, UnsupportedMinuteHistoryError) or isinstance(e, RateLimitedError):
                continue
            raise

    if first_err is None:
        raise RuntimeError("Minute fetch failed for unknown reason")
    raise first_err


def get_daily(
    tasks: list[dict],
    working_path: str,
    output_dir: str,
    fqt: int | str = 1,
    force: bool = False,
    abort_on_proxy: bool = True,
) -> None:
    colnames = ["时间", "开盘", "收盘", "最高", "最低", "成交量(手)", "成交额(元)", "均价", "换手率(%)"]

    trade_root = os.path.join(output_dir, "trade")

    for task in tasks:
        if _PROXY_ABORT_EVENT.is_set():
            break

        symbol = task["symbol"]
        date_str = task["date"]

        print(f"Processing: {symbol} for {date_str}")
        wait_seconds = random.uniform(2, 5)
        # Wake early when proxy abort is signaled to minimize tail latency.
        if _PROXY_ABORT_EVENT.wait(timeout=wait_seconds):
            break

        csv_dir = os.path.join(trade_root, symbol)
        csv_file = f"{csv_dir}/{date_str}.csv"

        if os.path.exists(csv_file) and not force:
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
            minute_df, src = fetch_1m(symbol, date_str, fqt=fqt)
            if not minute_df.empty:
                float_shares = _get_float_shares_for_trade_date(symbol, date_str)
                if float_shares and float_shares > 0:
                    # turnover(%) = volume_shares / float_shares * 100
                    # volume_shares = volume_hands * 100
                    minute_df["换手率(%)"] = (
                        pd.to_numeric(minute_df["成交量(手)"], errors="coerce") * 10000.0 / float_shares
                    )
                else:
                    minute_df["换手率(%)"] = pd.NA
                total_detail_df = minute_df
                provider = "Tushare" if src == "ts" else "Tencent"
                print(f"Fetched {len(minute_df)} 1-minute rows from {provider} for {symbol} {date_str}")
            else:
                print(f"No data found for {symbol} on {date_str}")
        except Exception as e:
            msg = str(e)
            if len(msg) > 200:
                msg = msg[:200] + "..."
            print(f"Minute fetch failed for {symbol} {date_str}: {type(e).__name__} - {msg}")
            day_success = False
            if _is_proxy_error(e):
                record_failure(symbol, date_str, output_dir)
                if abort_on_proxy:
                    _set_proxy_abort(f"{type(e).__name__}: {msg}")
                    print("[ABORT] Proxy error detected, stop all remaining tasks immediately.")
                    try:
                        _thread.interrupt_main()
                    except Exception:
                        pass
                    break
                print("[WARN] Proxy error detected, record failure and continue this round.")

        if day_success:
            if not total_detail_df.empty:
                os.makedirs(csv_dir, exist_ok=True)
                total_detail_df = total_detail_df.sort_values("时间")
                for c in colnames:
                    if c not in total_detail_df.columns:
                        total_detail_df[c] = pd.NA
                total_detail_df = total_detail_df[colnames]
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
    force: bool = False,
    abort_on_proxy: bool = True,
) -> None:
    _PROXY_ABORT_EVENT.clear()
    _set_proxy_abort_message = False
    with _PROXY_ABORT_LOCK:
        global _PROXY_ABORT_MESSAGE
        _PROXY_ABORT_MESSAGE = ""

    if not tasks_to_run:
        print("No tasks to run.")
        return

    chunks_num = max(1, int(threads))
    chunk_size = max(1, (len(tasks_to_run) + chunks_num - 1) // chunks_num)
    tasks_cks = list(divide_chunks(tasks_to_run, chunk_size))

    workers: list[threading.Thread] = []
    for i, chunk in enumerate(tasks_cks):
        t = threading.Thread(target=get_daily, args=(chunk, working_path, output_dir, fqt, force, abort_on_proxy))
        t.name = f"worker_{i}"
        t.daemon = True
        print(t.name)
        workers.append(t)

    for t in workers:
        t.start()
    try:
        for t in workers:
            t.join()
    except KeyboardInterrupt:
        if _PROXY_ABORT_EVENT.is_set():
            msg = _get_proxy_abort_message() or "Proxy error detected"
            raise ProxyConnectivityError(msg)
        print("Interrupted. Exiting without waiting for workers...")
        return

    if _PROXY_ABORT_EVENT.is_set():
        msg = _get_proxy_abort_message() or "Proxy error detected"
        raise ProxyConnectivityError(msg)
