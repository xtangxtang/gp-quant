import glob
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .multitimeframe_evaluation import build_resonance_daily_frame
from .multitimeframe_feature_engine import aggregate_stock_bars, compute_physics_state_features, to_trade_date_str
from .multitimeframe_physics_utils import build_index_monthly_regime_by_date
from .multitimeframe_report_writer import write_scan_outputs


@dataclass(frozen=True)
class ScanConfig:
    data_dir: str
    out_dir: str
    scan_date: str = ""
    top_n: int = 30
    symbols: str = ""
    index_path: str = ""
    basic_path: str = ""
    lookback_years: int = 5
    entry_threshold: float = 0.18
    persist_bars: int = 3
    energy_min: float = -0.10
    order_min: float = 0.05
    phase_min: float = 0.00
    gate_index: bool = False
    daily_ws: int = 20
    daily_wl: int = 60
    weekly_ws: int = 12
    weekly_wl: int = 36
    monthly_ws: int = 12
    monthly_wl: int = 36
    resonance_threshold: float = 0.22
    resonance_min_count: int = 2
    resonance_persist_days: int = 2
    weekly_support_threshold: float = 0.10
    monthly_support_threshold: float = 0.08
    min_amount: float = 500000.0
    min_turnover: float = 1.0
    exclude_st: bool = True
    backtest_start_date: str = ""
    backtest_end_date: str = ""
    hold_days: int = 5
    max_positions: int = 10
    max_positions_per_industry: int = 2


@dataclass
class PreparedSymbol:
    symbol: str
    ts_code: str
    name: str
    area: str
    industry: str
    market: str
    is_st: bool
    resonance_daily: pd.DataFrame
    date_to_index: dict[str, int]
    close_by_date: dict[str, float]


def _industry_bucket(row: dict[str, Any]) -> str:
    return str(row.get("industry") or "UNKNOWN")


def _load_basic_info_map(basic_path: str) -> dict[str, dict[str, str]]:
    if not basic_path or not os.path.exists(basic_path):
        return {}
    try:
        df = pd.read_csv(basic_path, usecols=["ts_code", "name", "area", "industry", "market"])
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return {}
        df = df.fillna("")
        return {
            str(row["ts_code"]): {
                "name": str(row.get("name", "") or ""),
                "area": str(row.get("area", "") or ""),
                "industry": str(row.get("industry", "") or ""),
                "market": str(row.get("market", "") or ""),
            }
            for _, row in df.iterrows()
        }
    except Exception:
        return {}


def _is_st_name(name: str) -> bool:
    normalized = str(name or "").strip().upper()
    return normalized.startswith("ST") or normalized.startswith("*ST")


def _load_index_regime(index_path: str) -> dict[str, str]:
    index_regime_by_date: dict[str, str] = {}
    if index_path:
        df_idx = pd.read_csv(index_path)
        df_idx["trade_date_str"] = df_idx["trade_date"].astype(str)
        df_idx = df_idx.sort_values("trade_date_str").reset_index(drop=True)
        _, index_regime_by_date = build_index_monthly_regime_by_date(df_idx)
    return index_regime_by_date


def _resolve_files(data_dir: str, symbols_arg: str) -> list[str]:
    symbols = [symbol.strip() for symbol in str(symbols_arg).split(",") if symbol.strip()]
    if symbols:
        files = [os.path.join(data_dir, f"{symbol}.csv") for symbol in symbols if os.path.exists(os.path.join(data_dir, f"{symbol}.csv"))]
    else:
        files = glob.glob(os.path.join(data_dir, "*.csv"))
    return sorted(files)


def _read_last_non_empty_line(file_path: str) -> str:
    with open(file_path, "rb") as file_obj:
        file_obj.seek(0, os.SEEK_END)
        position = file_obj.tell()
        buffer = bytearray()
        while position > 0:
            position -= 1
            file_obj.seek(position)
            char = file_obj.read(1)
            if char == b"\n" and buffer:
                break
            if char != b"\n":
                buffer.extend(char)
        return bytes(reversed(buffer)).decode("utf-8", errors="ignore").strip()


def _latest_trade_date_for_file(file_path: str) -> str | None:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file_obj:
            header_line = file_obj.readline().strip()
        last_line = _read_last_non_empty_line(file_path)
    except OSError:
        return None
    if not header_line or not last_line:
        return None
    header_parts = [part.strip() for part in header_line.split(",")]
    if "trade_date" not in header_parts:
        return None
    trade_date_idx = header_parts.index("trade_date")
    parts = last_line.split(",")
    if trade_date_idx >= len(parts):
        return None
    trade_date = str(parts[trade_date_idx]).strip()
    return trade_date if trade_date.isdigit() else None


def _infer_scan_date(files: list[str], requested_scan_date: str) -> str:
    if requested_scan_date:
        return str(requested_scan_date)
    latest_dates = [_latest_trade_date_for_file(file_path) for file_path in files]
    latest_dates = [trade_date for trade_date in latest_dates if trade_date]
    if not latest_dates:
        raise SystemExit("Unable to infer scan_date from CSV files.")
    return max(latest_dates)


def _build_symbol_from_ts_code(ts_code: str) -> str:
    if not ts_code or "." not in ts_code:
        return str(ts_code).lower()
    code, exch = ts_code.split(".", 1)
    return f"{exch.lower()}{code}"


def _build_snapshot_row(prepared: PreparedSymbol, scan_date: str, index_regime_by_date: dict[str, str]) -> dict[str, Any] | None:
    idx = prepared.date_to_index.get(str(scan_date))
    if idx is None:
        return None
    row = prepared.resonance_daily.iloc[idx]
    index_regime = str(index_regime_by_date.get(str(scan_date), "BASE")) if index_regime_by_date else "BASE"
    return {
        "symbol": prepared.symbol,
        "ts_code": prepared.ts_code,
        "name": prepared.name,
        "area": prepared.area,
        "industry": prepared.industry,
        "market": prepared.market,
        "is_st": prepared.is_st,
        "scan_date": str(scan_date),
        "signal_date": str(row["bar_end"]),
        "daily_bar_end": str(row["bar_end"]),
        "weekly_bar_end": str(row["weekly_bar_end"]) if pd.notna(row.get("weekly_bar_end")) else None,
        "monthly_bar_end": str(row["monthly_bar_end"]) if pd.notna(row.get("monthly_bar_end")) else None,
        "close": float(row["close"]) if pd.notna(row.get("close")) else None,
        "amount": float(row["amount_sum"]) if pd.notna(row.get("amount_sum")) else None,
        "turnover_rate": float(row["turnover_mean"]) if pd.notna(row.get("turnover_mean")) else None,
        "daily_state": bool(row["daily_state"]) if pd.notna(row.get("daily_state")) else False,
        "weekly_state": bool(row["weekly_state"]) if pd.notna(row.get("weekly_state")) else False,
        "monthly_state": bool(row["monthly_state"]) if pd.notna(row.get("monthly_state")) else False,
        "resonance_state": bool(row["resonance_state"]) if pd.notna(row.get("resonance_state")) else False,
        "support_count": int(row["support_count"]) if pd.notna(row.get("support_count")) else 0,
        "daily_score": float(row["daily_score_ctx"]) if pd.notna(row.get("daily_score_ctx")) else None,
        "weekly_score": float(row["weekly_score_ctx"]) if pd.notna(row.get("weekly_score_ctx")) else None,
        "monthly_score": float(row["monthly_score_ctx"]) if pd.notna(row.get("monthly_score_ctx")) else None,
        "resonance_score": float(row["resonance_score"]) if pd.notna(row.get("resonance_score")) else None,
        "energy_term": float(row["energy_term"]) if pd.notna(row.get("energy_term")) else None,
        "temperature_term": float(row["temperature_term"]) if pd.notna(row.get("temperature_term")) else None,
        "order_term": float(row["order_term"]) if pd.notna(row.get("order_term")) else None,
        "phase_term": float(row["phase_term"]) if pd.notna(row.get("phase_term")) else None,
        "switch_term": float(row["switch_term"]) if pd.notna(row.get("switch_term")) else None,
        "index_regime": index_regime,
    }


def _passes_candidate_filters(row: dict[str, Any], config: ScanConfig) -> bool:
    if not row or not bool(row.get("resonance_state")):
        return False
    if bool(config.exclude_st) and bool(row.get("is_st")):
        return False
    amount = float(row.get("amount") or 0.0)
    turnover_rate = float(row.get("turnover_rate") or 0.0)
    if amount < float(config.min_amount):
        return False
    if turnover_rate < float(config.min_turnover):
        return False
    if bool(config.gate_index) and str(row.get("index_regime") or "BASE") not in {"BULL", "BASE"}:
        return False
    return True


def _prepare_symbol_state(file_path: str, config: ScanConfig, basic_info_map: dict[str, dict[str, str]]) -> PreparedSymbol | None:
    try:
        df = pd.read_csv(file_path)
    except Exception:
        return None
    if df.empty or "trade_date" not in df.columns:
        return None

    df = to_trade_date_str(df)
    if df.empty:
        return None

    max_needed_date = str(config.backtest_end_date or config.scan_date)
    if len(max_needed_date) == 8:
        start_keep = f"{int(max_needed_date[:4]) - int(config.lookback_years)}0101"
    else:
        start_keep = "20000101"
    df = df[(df["trade_date_str"] >= start_keep) & (df["trade_date_str"] <= max_needed_date)].copy()
    if df.empty:
        return None

    ts_code = str(df["ts_code"].iloc[0]) if "ts_code" in df.columns else ""
    symbol = _build_symbol_from_ts_code(ts_code or os.path.splitext(os.path.basename(file_path))[0])
    basic_info = basic_info_map.get(ts_code, {})
    name = str(basic_info.get("name", "") or "")
    area = str(basic_info.get("area", "") or "")
    industry = str(basic_info.get("industry", "") or "")
    market = str(basic_info.get("market", "") or "")
    is_st = _is_st_name(name)

    daily_bars = aggregate_stock_bars(df, "D")
    weekly_bars = aggregate_stock_bars(df, "W")
    monthly_bars = aggregate_stock_bars(df, "M")
    if daily_bars.empty or weekly_bars.empty or monthly_bars.empty:
        return None

    daily_feat = compute_physics_state_features(
        daily_bars,
        window_s=int(config.daily_ws),
        window_l=int(config.daily_wl),
        entry_threshold=float(config.entry_threshold),
        persist_bars=int(config.persist_bars),
        energy_min=float(config.energy_min),
        order_min=float(config.order_min),
        phase_min=float(config.phase_min),
    )
    weekly_feat = compute_physics_state_features(
        weekly_bars,
        window_s=int(config.weekly_ws),
        window_l=int(config.weekly_wl),
        entry_threshold=float(config.weekly_support_threshold),
        persist_bars=max(1, int(config.persist_bars) - 1),
        energy_min=float(config.energy_min),
        order_min=float(config.order_min) - 0.05,
        phase_min=float(config.phase_min) - 0.05,
    )
    monthly_feat = compute_physics_state_features(
        monthly_bars,
        window_s=int(config.monthly_ws),
        window_l=int(config.monthly_wl),
        entry_threshold=float(config.monthly_support_threshold),
        persist_bars=1,
        energy_min=float(config.energy_min) - 0.05,
        order_min=float(config.order_min) - 0.05,
        phase_min=float(config.phase_min) - 0.05,
    )
    if daily_feat.empty or weekly_feat.empty or monthly_feat.empty:
        return None

    resonance_daily = build_resonance_daily_frame(
        daily_feat,
        weekly_feat,
        monthly_feat,
        daily_threshold=float(config.entry_threshold),
        weekly_threshold=float(config.weekly_support_threshold),
        monthly_threshold=float(config.monthly_support_threshold),
        resonance_threshold=float(config.resonance_threshold),
        resonance_min_count=int(config.resonance_min_count),
        resonance_persist_days=int(config.resonance_persist_days),
    )
    if resonance_daily.empty:
        return None

    resonance_daily = resonance_daily.reset_index(drop=True)
    resonance_daily["bar_end"] = resonance_daily["bar_end"].astype(str)
    date_to_index = {trade_date: idx for idx, trade_date in enumerate(resonance_daily["bar_end"].tolist())}
    close_by_date = {
        str(trade_date): float(close)
        for trade_date, close in zip(resonance_daily["bar_end"].tolist(), resonance_daily["close"].tolist(), strict=False)
        if pd.notna(close)
    }
    return PreparedSymbol(
        symbol=symbol,
        ts_code=ts_code,
        name=name,
        area=area,
        industry=industry,
        market=market,
        is_st=is_st,
        resonance_daily=resonance_daily,
        date_to_index=date_to_index,
        close_by_date=close_by_date,
    )


def _prepare_all_symbols(files: list[str], config: ScanConfig, basic_info_map: dict[str, dict[str, str]]) -> list[PreparedSymbol]:
    prepared: list[PreparedSymbol] = []
    for file_path in files:
        item = _prepare_symbol_state(file_path, config, basic_info_map)
        if item is not None:
            prepared.append(item)
    return prepared


def _sort_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            float(row.get("resonance_score") or -999.0),
            int(row.get("support_count") or 0),
            float(row.get("daily_score") or -999.0),
            float(row.get("amount") or 0.0),
        ),
        reverse=True,
    )


def _select_candidate_rows(
    candidate_rows: list[dict[str, Any]],
    target_count: int,
    max_positions_per_industry: int,
) -> tuple[list[dict[str, Any]], int]:
    selected_rows: list[dict[str, Any]] = []
    skipped_industry = 0
    industry_counts: dict[str, int] = {}

    for row in candidate_rows:
        if len(selected_rows) >= int(target_count):
            break
        industry = _industry_bucket(row)
        if int(max_positions_per_industry) > 0 and industry_counts.get(industry, 0) >= int(max_positions_per_industry):
            skipped_industry += 1
            continue
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
        selected_rows.append(dict(row))

    return selected_rows, skipped_industry


def _build_scan_rows(
    prepared_symbols: list[PreparedSymbol],
    config: ScanConfig,
    index_regime_by_date: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    market_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []

    for prepared in prepared_symbols:
        market_row = _build_snapshot_row(prepared, str(config.scan_date), index_regime_by_date)
        if market_row is None:
            continue
        market_rows.append(market_row)
        if _passes_candidate_filters(market_row, config):
            candidate_rows.append(dict(market_row))

    candidate_rows = _sort_candidate_rows(candidate_rows)
    selected_candidates, _ = _select_candidate_rows(
        candidate_rows,
        int(config.top_n),
        int(config.max_positions_per_industry),
    )
    for rank, row in enumerate(selected_candidates, start=1):
        selected_row = dict(row)
        selected_row["selected_rank"] = rank
        selected_rows.append(selected_row)

    return market_rows, candidate_rows, selected_rows


def _trade_plan_from_snapshot(prepared: PreparedSymbol, scan_date: str, hold_days: int) -> dict[str, Any] | None:
    idx = prepared.date_to_index.get(str(scan_date))
    if idx is None:
        return None
    entry_idx = idx + 1
    if entry_idx >= len(prepared.resonance_daily):
        return None
    exit_idx = min(entry_idx + max(1, int(hold_days)) - 1, len(prepared.resonance_daily) - 1)

    entry_row = prepared.resonance_daily.iloc[entry_idx]
    exit_row = prepared.resonance_daily.iloc[exit_idx]
    entry_price = float(entry_row["open"])
    exit_price = float(exit_row["close"])
    if entry_price <= 0:
        return None
    max_close = float(prepared.resonance_daily.iloc[entry_idx : exit_idx + 1]["close"].max())
    return {
        "entry_date": str(entry_row["bar_start"]),
        "exit_date": str(exit_row["bar_end"]),
        "entry_scan_date": str(scan_date),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "hold_days_realized": int(exit_idx - entry_idx + 1),
        "max_runup_pct": (max_close / entry_price - 1.0) * 100.0,
        "return_pct": (exit_price / entry_price - 1.0) * 100.0,
    }


def _daily_mark_to_market_return(prepared: PreparedSymbol, trade: dict[str, Any], current_date: str) -> float | None:
    if str(current_date) < str(trade["entry_date"]) or str(current_date) > str(trade["exit_date"]):
        return None

    current_close = prepared.close_by_date.get(str(current_date))
    if current_close is None or current_close <= 0:
        return None

    if str(current_date) == str(trade["entry_date"]):
        entry_price = float(trade.get("entry_price") or 0.0)
        if entry_price <= 0:
            return None
        return current_close / entry_price - 1.0

    current_idx = prepared.date_to_index.get(str(current_date))
    if current_idx is None or current_idx <= 0:
        return None
    prev_date = str(prepared.resonance_daily.iloc[current_idx - 1]["bar_end"])
    prev_close = prepared.close_by_date.get(prev_date)
    if prev_close is None or prev_close <= 0:
        return None
    return current_close / prev_close - 1.0


def _run_forward_backtest(
    prepared_symbols: list[PreparedSymbol],
    config: ScanConfig,
    index_regime_by_date: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    if not config.backtest_start_date or not config.backtest_end_date:
        return [], [], None

    scan_dates: set[str] = set()
    for prepared in prepared_symbols:
        for trade_date in prepared.date_to_index:
            if str(config.backtest_start_date) <= trade_date <= str(config.backtest_end_date):
                scan_dates.add(trade_date)
    ordered_scan_dates = sorted(scan_dates)
    if not ordered_scan_dates:
        return [], [], None

    symbol_lookup = {prepared.symbol: prepared for prepared in prepared_symbols}
    active_positions: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    nav = 1.0

    for scan_date in ordered_scan_dates:
        active_positions = [pos for pos in active_positions if str(pos["exit_date"]) >= scan_date]
        candidate_rows_day: list[dict[str, Any]] = []
        for prepared in prepared_symbols:
            market_row = _build_snapshot_row(prepared, scan_date, index_regime_by_date)
            if market_row is None:
                continue
            if _passes_candidate_filters(market_row, config):
                candidate_rows_day.append(dict(market_row))

        candidate_rows_day = _sort_candidate_rows(candidate_rows_day)[: int(config.top_n)]
        accepted_today = 0
        skipped_full = 0
        skipped_industry = 0

        for candidate in candidate_rows_day:
            prepared = symbol_lookup.get(str(candidate["symbol"]))
            if prepared is None:
                continue
            plan = _trade_plan_from_snapshot(prepared, scan_date, int(config.hold_days))
            if plan is None:
                continue

            entry_date = str(plan["entry_date"])
            active_positions = [pos for pos in active_positions if str(pos["exit_date"]) >= entry_date]
            if any(str(pos["symbol"]) == str(candidate["symbol"]) for pos in active_positions):
                continue
            if len(active_positions) >= int(config.max_positions):
                skipped_full += 1
                continue
            industry = _industry_bucket(candidate)
            industry_open_count = sum(1 for pos in active_positions if _industry_bucket(pos) == industry)
            if int(config.max_positions_per_industry) > 0 and industry_open_count >= int(config.max_positions_per_industry):
                skipped_industry += 1
                continue

            trade = {**candidate, **plan, "scan_date": scan_date}
            trades.append(trade)
            active_positions.append(
                {
                    "symbol": trade["symbol"],
                    "ts_code": trade["ts_code"],
                    "industry": trade.get("industry"),
                    "entry_date": entry_date,
                    "exit_date": str(trade["exit_date"]),
                    "entry_price": float(trade["entry_price"]),
                }
            )
            accepted_today += 1

        active_trade_returns: list[float] = []
        for position in active_positions:
            prepared = symbol_lookup.get(str(position["symbol"]))
            if prepared is None:
                continue
            trade_return = _daily_mark_to_market_return(prepared, position, scan_date)
            if trade_return is not None:
                active_trade_returns.append(float(trade_return))

        strategy_daily_return = float(sum(active_trade_returns) / len(active_trade_returns)) if active_trade_returns else 0.0
        nav *= 1.0 + strategy_daily_return
        realized_trades = [trade for trade in trades if str(trade["exit_date"]) == scan_date]

        daily_rows.append(
            {
                "scan_date": scan_date,
                "n_candidates": len(candidate_rows_day),
                "n_selected": accepted_today,
                "n_skipped_full": skipped_full,
                "n_skipped_industry": skipped_industry,
                "active_positions": int(len(active_trade_returns)),
                "realized_trades": int(len(realized_trades)),
                "strategy_daily_return": float(strategy_daily_return * 100.0),
                "nav": float(nav),
            }
        )

    if not trades:
        return daily_rows, [], {
            "backtest_start_date": str(config.backtest_start_date),
            "backtest_end_date": str(config.backtest_end_date),
            "hold_days": int(config.hold_days),
            "top_n": int(config.top_n),
            "max_positions": int(config.max_positions),
            "max_positions_per_industry": int(config.max_positions_per_industry),
            "n_trades": 0,
            "final_nav": float(nav),
            "total_return_pct": float((nav - 1.0) * 100.0),
        }

    df_trades = pd.DataFrame(trades)
    df_daily = pd.DataFrame(daily_rows)
    max_drawdown_pct = None
    if not df_daily.empty and "nav" in df_daily.columns:
        nav_series = df_daily["nav"].astype(float)
        running_peak = nav_series.cummax()
        drawdown = nav_series / running_peak - 1.0
        max_drawdown_pct = float(drawdown.min() * 100.0)
    return daily_rows, trades, {
        "backtest_start_date": str(config.backtest_start_date),
        "backtest_end_date": str(config.backtest_end_date),
        "hold_days": int(config.hold_days),
        "top_n": int(config.top_n),
        "max_positions": int(config.max_positions),
        "max_positions_per_industry": int(config.max_positions_per_industry),
        "n_trades": int(len(df_trades)),
        "win_rate": float((df_trades["return_pct"] > 0).mean()),
        "avg_return_pct": float(df_trades["return_pct"].mean()),
        "median_return_pct": float(df_trades["return_pct"].median()),
        "avg_max_runup_pct": float(df_trades["max_runup_pct"].mean()),
        "final_nav": float(nav),
        "total_return_pct": float((nav - 1.0) * 100.0),
        "max_drawdown_pct": max_drawdown_pct,
        "total_skipped_industry": int(df_daily["n_skipped_industry"].sum()) if not df_daily.empty and "n_skipped_industry" in df_daily.columns else 0,
    }


def run_multitimeframe_scan(config: ScanConfig) -> list[str]:
    files = _resolve_files(config.data_dir, config.symbols)
    if not files:
        raise SystemExit(f"No CSVs found in data_dir={config.data_dir}")

    effective_scan_date = _infer_scan_date(files, config.scan_date)
    if config.backtest_end_date and str(config.backtest_end_date) < effective_scan_date:
        effective_scan_date = str(config.backtest_end_date)
    config = ScanConfig(**{**config.__dict__, "scan_date": effective_scan_date})

    basic_info_map = _load_basic_info_map(config.basic_path)
    index_regime_by_date = _load_index_regime(config.index_path)
    prepared_symbols = _prepare_all_symbols(files, config, basic_info_map)

    market_rows, candidate_rows, selected_rows = _build_scan_rows(prepared_symbols, config, index_regime_by_date)
    backtest_daily_rows, backtest_trade_rows, backtest_summary = _run_forward_backtest(
        prepared_symbols,
        config,
        index_regime_by_date,
    )

    return write_scan_outputs(
        out_dir=config.out_dir,
        scan_date=str(config.scan_date),
        top_n=int(config.top_n),
        market_rows=market_rows,
        candidate_rows=candidate_rows,
        selected_rows=selected_rows,
        backtest_daily_rows=backtest_daily_rows,
        backtest_trade_rows=backtest_trade_rows,
        backtest_summary=backtest_summary,
    )
