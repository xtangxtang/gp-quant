import glob
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .entropy_bifurcation_feature_engine import build_entropy_bifurcation_feature_frame
from .entropy_bifurcation_report_writer import write_entropy_bifurcation_outputs
from .entropy_bifurcation_signal_models import VALID_STRATEGIES, apply_strategy_model


@dataclass(frozen=True)
class EntropyBifurcationScanConfig:
    strategy_name: str = "entropy_bifurcation_setup"
    data_dir: str = ""
    out_dir: str = ""
    scan_date: str = ""
    top_n: int = 30
    symbols: str = ""
    basic_path: str = ""
    lookback_years: int = 5
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
    signal_daily: pd.DataFrame
    date_to_index: dict[str, int]
    close_by_date: dict[str, float]


def _industry_bucket(row: dict[str, Any]) -> str:
    return str(row.get("industry") or "UNKNOWN")


def _load_basic_info_map(basic_path: str) -> dict[str, dict[str, str]]:
    if not basic_path or not os.path.exists(basic_path):
        return {}
    try:
        df = pd.read_csv(basic_path, usecols=["ts_code", "name", "area", "industry", "market"])
        if df.empty:
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


def _prepare_symbol_state(file_path: str, config: EntropyBifurcationScanConfig, basic_info_map: dict[str, dict[str, str]]) -> PreparedSymbol | None:
    try:
        df = pd.read_csv(file_path)
    except Exception:
        return None
    if df.empty or "trade_date" not in df.columns:
        return None

    max_needed_date = str(config.backtest_end_date or config.scan_date)
    if len(max_needed_date) == 8:
        start_keep = f"{int(max_needed_date[:4]) - int(config.lookback_years)}0101"
        df = df[(df["trade_date"].astype(str) >= start_keep) & (df["trade_date"].astype(str) <= max_needed_date)].copy()
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

    features = build_entropy_bifurcation_feature_frame(df)
    if features.empty:
        return None
    signal_daily = apply_strategy_model(features, config.strategy_name)
    if signal_daily.empty:
        return None

    signal_daily = signal_daily.reset_index(drop=True)
    signal_daily["bar_end"] = signal_daily["bar_end"].astype(str)
    date_to_index = {trade_date: idx for idx, trade_date in enumerate(signal_daily["bar_end"].tolist())}
    close_by_date = {
        str(trade_date): float(close)
        for trade_date, close in zip(signal_daily["bar_end"].tolist(), signal_daily["close"].tolist(), strict=False)
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
        signal_daily=signal_daily,
        date_to_index=date_to_index,
        close_by_date=close_by_date,
    )


def _prepare_all_symbols(
    files: list[str], config: EntropyBifurcationScanConfig, basic_info_map: dict[str, dict[str, str]]
) -> list[PreparedSymbol]:
    prepared: list[PreparedSymbol] = []
    for file_path in files:
        item = _prepare_symbol_state(file_path, config, basic_info_map)
        if item is not None:
            prepared.append(item)
    return prepared


def _build_snapshot_row(prepared: PreparedSymbol, scan_date: str) -> dict[str, Any] | None:
    idx = prepared.date_to_index.get(str(scan_date))
    if idx is None:
        return None
    row = prepared.signal_daily.iloc[idx]
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
        "close": float(row["close"]) if pd.notna(row.get("close")) else None,
        "amount": float(row["amount"]) if pd.notna(row.get("amount")) else None,
        "turnover_rate": float(row["turnover_rate"]) if pd.notna(row.get("turnover_rate")) else None,
        "strategy_name": str(row.get("strategy_name") or ""),
        "strategy_state": bool(row["strategy_state"]) if pd.notna(row.get("strategy_state")) else False,
        "strategy_score": float(row["strategy_score"]) if pd.notna(row.get("strategy_score")) else None,
        "strategy_component_a": float(row["strategy_component_a"]) if pd.notna(row.get("strategy_component_a")) else None,
        "strategy_component_b": float(row["strategy_component_b"]) if pd.notna(row.get("strategy_component_b")) else None,
        "strategy_component_c": float(row["strategy_component_c"]) if pd.notna(row.get("strategy_component_c")) else None,
        "entropy_quality": float(row["entropy_quality"]) if pd.notna(row.get("entropy_quality")) else None,
        "bifurcation_quality": float(row["bifurcation_quality"]) if pd.notna(row.get("bifurcation_quality")) else None,
        "trigger_quality": float(row["trigger_quality"]) if pd.notna(row.get("trigger_quality")) else None,
        "entropy_percentile_120": float(row["entropy_percentile_120"]) if pd.notna(row.get("entropy_percentile_120")) else None,
        "entropy_gap": float(row["entropy_gap"]) if pd.notna(row.get("entropy_gap")) else None,
        "perm_entropy_20_norm": float(row["perm_entropy_20_norm"]) if pd.notna(row.get("perm_entropy_20_norm")) else None,
        "perm_entropy_60_norm": float(row["perm_entropy_60_norm"]) if pd.notna(row.get("perm_entropy_60_norm")) else None,
        "ar1_20": float(row["ar1_20"]) if pd.notna(row.get("ar1_20")) else None,
        "recovery_rate_20": float(row["recovery_rate_20"]) if pd.notna(row.get("recovery_rate_20")) else None,
        "state_skew_20": float(row["state_skew_20"]) if pd.notna(row.get("state_skew_20")) else None,
        "var_lift_10_20": float(row["var_lift_10_20"]) if pd.notna(row.get("var_lift_10_20")) else None,
        "breakout_10": float(row["breakout_10"]) if pd.notna(row.get("breakout_10")) else None,
        "breakout_20": float(row["breakout_20"]) if pd.notna(row.get("breakout_20")) else None,
        "volume_impulse_5_20": float(row["volume_impulse_5_20"]) if pd.notna(row.get("volume_impulse_5_20")) else None,
        "flow_impulse_5_20": float(row["flow_impulse_5_20"]) if pd.notna(row.get("flow_impulse_5_20")) else None,
        "energy_impulse": float(row["energy_impulse"]) if pd.notna(row.get("energy_impulse")) else None,
        "order_alignment": float(row["order_alignment"]) if pd.notna(row.get("order_alignment")) else None,
        "mf_z_60": float(row["mf_z_60"]) if pd.notna(row.get("mf_z_60")) else None,
    }


def _augment_cross_section(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return rows
    df = pd.DataFrame(rows)
    df["strategy_score"] = pd.to_numeric(df.get("strategy_score"), errors="coerce")
    df["entropy_quality"] = pd.to_numeric(df.get("entropy_quality"), errors="coerce")
    df["bifurcation_quality"] = pd.to_numeric(df.get("bifurcation_quality"), errors="coerce")
    df["trigger_quality"] = pd.to_numeric(df.get("trigger_quality"), errors="coerce")
    df["breakout_10"] = pd.to_numeric(df.get("breakout_10"), errors="coerce")
    df["strategy_state"] = pd.Series(df.get("strategy_state"), dtype="boolean").fillna(False).astype(bool)
    df["score_pct_rank"] = df["strategy_score"].rank(pct=True, method="average")
    df["entropy_pct_rank"] = df["entropy_quality"].rank(pct=True, method="average")
    df["bifurcation_pct_rank"] = df["bifurcation_quality"].rank(pct=True, method="average")
    df["trigger_pct_rank"] = df["trigger_quality"].rank(pct=True, method="average")
    df["breakout_pct_rank"] = df["breakout_10"].rank(pct=True, method="average")
    return df.to_dict(orient="records")


def _passes_candidate_filters(row: dict[str, Any], config: EntropyBifurcationScanConfig) -> bool:
    if not row or not bool(row.get("strategy_state")):
        return False
    if bool(config.exclude_st) and bool(row.get("is_st")):
        return False
    amount = float(row.get("amount") or 0.0)
    turnover_rate = float(row.get("turnover_rate") or 0.0)
    if amount < float(config.min_amount):
        return False
    if turnover_rate < float(config.min_turnover):
        return False
    if float(row.get("score_pct_rank") or 0.0) < 0.80:
        return False
    if float(row.get("bifurcation_pct_rank") or 0.0) < 0.70:
        return False
    return True


def _sort_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            float(row.get("strategy_score") or -999.0),
            float(row.get("bifurcation_quality") or -999.0),
            float(row.get("entropy_quality") or -999.0),
            float(row.get("amount") or 0.0),
        ),
        reverse=True,
    )


def _select_candidate_rows(
    candidate_rows: list[dict[str, Any]], target_count: int, max_positions_per_industry: int
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
    prepared_symbols: list[PreparedSymbol], config: EntropyBifurcationScanConfig
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    market_rows = [row for row in (_build_snapshot_row(prepared, str(config.scan_date)) for prepared in prepared_symbols) if row]
    market_rows = _augment_cross_section(market_rows)
    candidate_rows = [dict(row) for row in market_rows if _passes_candidate_filters(row, config)]
    candidate_rows = _sort_candidate_rows(candidate_rows)
    selected_candidates, _ = _select_candidate_rows(candidate_rows, int(config.top_n), int(config.max_positions_per_industry))
    selected_rows: list[dict[str, Any]] = []
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
    if entry_idx >= len(prepared.signal_daily):
        return None
    exit_idx = min(entry_idx + max(1, int(hold_days)) - 1, len(prepared.signal_daily) - 1)
    entry_row = prepared.signal_daily.iloc[entry_idx]
    exit_row = prepared.signal_daily.iloc[exit_idx]
    entry_price = float(entry_row["open"])
    exit_price = float(exit_row["close"])
    if entry_price <= 0:
        return None
    max_close = float(prepared.signal_daily.iloc[entry_idx : exit_idx + 1]["close"].max())
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
    prev_date = str(prepared.signal_daily.iloc[current_idx - 1]["bar_end"])
    prev_close = prepared.close_by_date.get(prev_date)
    if prev_close is None or prev_close <= 0:
        return None
    return current_close / prev_close - 1.0


def _run_forward_backtest(
    prepared_symbols: list[PreparedSymbol], config: EntropyBifurcationScanConfig
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
        market_rows_day = [row for row in (_build_snapshot_row(prepared, scan_date) for prepared in prepared_symbols) if row is not None]
        market_rows_day = _augment_cross_section(market_rows_day)
        candidate_rows_day = [dict(row) for row in market_rows_day if _passes_candidate_filters(row, config)]
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
                "strategy_name": str(config.strategy_name),
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
            "strategy_name": str(config.strategy_name),
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
        "strategy_name": str(config.strategy_name),
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
        "total_skipped_industry": int(df_daily["n_skipped_industry"].sum()) if not df_daily.empty else 0,
    }


def run_entropy_bifurcation_scan(config: EntropyBifurcationScanConfig) -> list[str]:
    if str(config.strategy_name) not in VALID_STRATEGIES:
        supported = ", ".join(sorted(VALID_STRATEGIES))
        raise SystemExit(f"Unsupported strategy_name={config.strategy_name}. Supported: {supported}")

    files = _resolve_files(config.data_dir, config.symbols)
    if not files:
        raise SystemExit(f"No CSVs found in data_dir={config.data_dir}")

    effective_scan_date = _infer_scan_date(files, config.scan_date)
    if config.backtest_end_date and str(config.backtest_end_date) < effective_scan_date:
        effective_scan_date = str(config.backtest_end_date)
    config = EntropyBifurcationScanConfig(**{**config.__dict__, "scan_date": effective_scan_date})

    basic_info_map = _load_basic_info_map(config.basic_path)
    prepared_symbols = _prepare_all_symbols(files, config, basic_info_map)
    market_rows, candidate_rows, selected_rows = _build_scan_rows(prepared_symbols, config)
    backtest_daily_rows, backtest_trade_rows, backtest_summary = _run_forward_backtest(prepared_symbols, config)

    return write_entropy_bifurcation_outputs(
        out_dir=config.out_dir,
        strategy_name=str(config.strategy_name),
        scan_date=str(config.scan_date),
        top_n=int(config.top_n),
        market_rows=market_rows,
        candidate_rows=candidate_rows,
        selected_rows=selected_rows,
        backtest_daily_rows=backtest_daily_rows,
        backtest_trade_rows=backtest_trade_rows,
        backtest_summary=backtest_summary,
    )
