import glob
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .continuous_decline_recovery_feature_engine import build_continuous_decline_recovery_feature_frame
from .continuous_decline_recovery_report_writer import write_continuous_decline_recovery_outputs


def _scaled_scalar(value: Any, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if pd.isna(number) or high <= low:
        return 0.0
    return float(np.clip((number - float(low)) / float(high - low), 0.0, 1.0))


def _clip01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if pd.isna(number):
        return 0.0
    return float(np.clip(number, 0.0, 1.0))


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if pd.isna(number):
        return float(default)
    return float(number)


@dataclass(frozen=True)
class ContinuousDeclineRecoveryConfig:
    strategy_name: str = "continuous_decline_recovery"
    data_dir: str = ""
    out_dir: str = ""
    scan_date: str = ""
    top_n: int = 30
    top_sectors: int = 6
    symbols: str = ""
    basic_path: str = ""
    lookback_years: int = 4
    min_amount: float = 600000.0
    min_turnover: float = 1.0
    exclude_st: bool = True
    market_window: int = 6
    min_sector_members: int = 4
    min_rebound_from_low: float = 0.03
    max_rebound_from_low: float = 0.15
    backtest_start_date: str = ""
    backtest_end_date: str = ""
    hold_days: int = 8
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
    daily: pd.DataFrame
    trade_dates: np.ndarray
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
        return sorted(
            [os.path.join(data_dir, f"{symbol}.csv") for symbol in symbols if os.path.exists(os.path.join(data_dir, f"{symbol}.csv"))]
        )
    return sorted(glob.glob(os.path.join(data_dir, "*.csv")))


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


def _prepare_symbol_state(
    file_path: str,
    min_keep_date: str,
    max_needed_date: str,
    basic_info_map: dict[str, dict[str, str]],
) -> PreparedSymbol | None:
    try:
        df = pd.read_csv(file_path)
    except Exception:
        return None
    if df.empty or "trade_date" not in df.columns:
        return None

    trade_dates = df["trade_date"].astype(str)
    if min_keep_date:
        df = df[trade_dates >= str(min_keep_date)].copy()
        trade_dates = df["trade_date"].astype(str)
    if max_needed_date:
        df = df[trade_dates <= str(max_needed_date)].copy()
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

    daily = build_continuous_decline_recovery_feature_frame(df)
    if daily.empty:
        return None

    trade_dates = daily["trade_date_str"].astype(str).to_numpy(dtype=object)
    date_to_index = {str(trade_date): idx for idx, trade_date in enumerate(trade_dates)}
    close_by_date = {
        str(trade_date): _as_float(close_value, np.nan)
        for trade_date, close_value in zip(trade_dates, daily["close"].to_numpy(dtype=np.float64), strict=False)
    }
    return PreparedSymbol(
        symbol=symbol,
        ts_code=ts_code,
        name=name,
        area=area,
        industry=industry,
        market=market,
        is_st=is_st,
        daily=daily,
        trade_dates=trade_dates,
        date_to_index=date_to_index,
        close_by_date=close_by_date,
    )


def _master_trade_dates(prepared_symbols: list[PreparedSymbol]) -> np.ndarray:
    if not prepared_symbols:
        return np.asarray([], dtype=object)
    longest = max(prepared_symbols, key=lambda item: len(item.trade_dates))
    return longest.trade_dates.astype(object)


def _resolve_scan_date_from_master(master_dates: np.ndarray, requested_scan_date: str) -> str:
    if len(master_dates) == 0:
        raise SystemExit("No trade dates available in prepared symbols.")
    target = str(requested_scan_date)
    position = int(np.searchsorted(master_dates, target, side="right") - 1)
    if position < 0:
        raise SystemExit(f"No available trade date on or before {target}.")
    return str(master_dates[position])


def _decorate_row(prepared: PreparedSymbol, idx: int, config: ContinuousDeclineRecoveryConfig) -> dict[str, Any]:
    row = prepared.daily.iloc[int(idx)].to_dict()
    row.update(
        {
            "symbol": prepared.symbol,
            "ts_code": prepared.ts_code,
            "name": prepared.name,
            "area": prepared.area,
            "industry": prepared.industry or "UNKNOWN",
            "market": prepared.market,
            "is_st": bool(prepared.is_st),
        }
    )
    amount = _as_float(row.get("amount"), 0.0)
    turnover_rate = _as_float(row.get("turnover_rate"), 0.0)
    liquidity_ok = amount >= float(config.min_amount) and turnover_rate >= float(config.min_turnover)
    row["liquidity_ok"] = bool(liquidity_ok)
    row["tradable_ok"] = bool(liquidity_ok and (not config.exclude_st or not prepared.is_st))
    return row


def _rows_for_date(prepared_symbols: list[PreparedSymbol], scan_date: str, config: ContinuousDeclineRecoveryConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prepared in prepared_symbols:
        idx = prepared.date_to_index.get(str(scan_date))
        if idx is None:
            continue
        rows.append(_decorate_row(prepared, idx, config))
    return rows


def _build_market_aggregate_cache(
    prepared_symbols: list[PreparedSymbol],
    master_dates: np.ndarray,
    config: ContinuousDeclineRecoveryConfig,
) -> pd.DataFrame:
    if len(master_dates) == 0:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for trade_date in master_dates:
        current_rows = _rows_for_date(prepared_symbols, str(trade_date), config)
        rows.append(_aggregate_market_row(current_rows, str(trade_date)))

    market_cache = pd.DataFrame(rows)
    if market_cache.empty:
        return market_cache
    market_cache["trade_date"] = market_cache["trade_date"].astype(str)
    return market_cache.sort_values(["trade_date"], ascending=[True]).reset_index(drop=True)


def _aggregate_market_row(rows: list[dict[str, Any]], trade_date: str) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    if df.empty:
        return {
            "trade_date": str(trade_date),
            "n_universe": 0,
            "eq_ret_1": np.nan,
            "eq_ret_3": np.nan,
            "eq_ret_5": np.nan,
            "eq_ret_10": np.nan,
            "eq_ret_20": np.nan,
            "below_ma20_share": np.nan,
            "downtrend_share": np.nan,
            "deep_drawdown_share": np.nan,
            "repair_share": np.nan,
            "early_entry_share": np.nan,
            "flow_positive_share": np.nan,
            "amount_expand_share": np.nan,
            "overheat_share": np.nan,
            "washout_score": np.nan,
            "repair_score": np.nan,
            "overheat_score": np.nan,
        }

    universe = df[df["tradable_ok"]].copy()
    if universe.empty:
        universe = df.copy()

    eq_ret_1 = float(universe["ret_1"].mean())
    eq_ret_3 = float(universe["ret_3"].mean())
    eq_ret_5 = float(universe["ret_5"].mean())
    eq_ret_10 = float(universe["ret_10"].mean())
    eq_ret_20 = float(universe["ret_20"].mean())
    below_ma20_share = float(universe["close_below_ma20"].mean())
    downtrend_share = float(((universe["damage_score"] >= 0.32) | (universe["ret_10"] <= -0.06)).mean())
    deep_drawdown_share = float((universe["drawdown_20"] <= -0.12).mean())
    repair_share = float(universe["repair_flag"].mean())
    early_entry_share = float(universe["early_entry_flag"].mean())
    flow_positive_share = float((universe["flow_ratio_5"] > 0.0).mean())
    amount_expand_share = float((universe["amount_ratio_20"] >= 1.05).mean())
    overheat_share = float(universe["overheat_flag"].mean())

    washout_score = _clip01(
        0.30 * _scaled_scalar(-eq_ret_10, 0.01, 0.08)
        + 0.25 * _scaled_scalar(below_ma20_share, 0.30, 0.85)
        + 0.25 * _scaled_scalar(deep_drawdown_share, 0.08, 0.55)
        + 0.20 * _scaled_scalar(downtrend_share, 0.20, 0.75)
    )
    repair_score = _clip01(
        0.25 * _scaled_scalar(eq_ret_3, -0.005, 0.03)
        + 0.25 * _scaled_scalar(repair_share, 0.20, 0.60)
        + 0.20 * _scaled_scalar(early_entry_share, 0.03, 0.25)
        + 0.15 * _scaled_scalar(flow_positive_share, 0.40, 0.70)
        + 0.15 * _scaled_scalar(amount_expand_share, 0.20, 0.55)
    )
    overheat_score = _clip01(
        0.60 * _scaled_scalar(overheat_share, 0.10, 0.35)
        + 0.40 * _scaled_scalar(eq_ret_5, 0.02, 0.08)
    )

    return {
        "trade_date": str(trade_date),
        "n_universe": int(len(universe)),
        "eq_ret_1": eq_ret_1,
        "eq_ret_3": eq_ret_3,
        "eq_ret_5": eq_ret_5,
        "eq_ret_10": eq_ret_10,
        "eq_ret_20": eq_ret_20,
        "below_ma20_share": below_ma20_share,
        "downtrend_share": downtrend_share,
        "deep_drawdown_share": deep_drawdown_share,
        "repair_share": repair_share,
        "early_entry_share": early_entry_share,
        "flow_positive_share": flow_positive_share,
        "amount_expand_share": amount_expand_share,
        "overheat_share": overheat_share,
        "washout_score": washout_score,
        "repair_score": repair_score,
        "overheat_score": overheat_score,
    }


def _build_market_frame(
    market_cache: pd.DataFrame,
    master_dates: np.ndarray,
    scan_date: str,
    config: ContinuousDeclineRecoveryConfig,
) -> pd.DataFrame:
    if len(master_dates) == 0 or market_cache.empty:
        return pd.DataFrame()
    position = int(np.searchsorted(master_dates, str(scan_date), side="right") - 1)
    if position < 0:
        return pd.DataFrame()

    start = max(0, position - int(config.market_window) + 1)
    window_dates = [str(value) for value in master_dates[start : position + 1]]
    return market_cache[market_cache["trade_date"].isin(window_dates)].copy().reset_index(drop=True)


def _derive_market_timing(market_frame: pd.DataFrame) -> dict[str, Any]:
    if market_frame.empty:
        return {
            "scan_date": "",
            "market_buy_state": "no_setup",
            "market_buy_score": 0.0,
            "market_washout_score": 0.0,
            "market_repair_score": 0.0,
            "market_overheat_score": 0.0,
            "recent_washout_peak": 0.0,
            "market_can_buy": False,
            "eq_ret_5": 0.0,
            "reason": "市场窗口数据为空，无法判断连续下跌后的修复状态。",
        }

    current = market_frame.iloc[-1]
    recent_washout_peak = float(market_frame["washout_score"].max())
    market_buy_score = _clip01(
        0.55 * recent_washout_peak
        + 0.45 * _as_float(current.get("repair_score"), 0.0)
        - 0.35 * _as_float(current.get("overheat_score"), 0.0)
    )

    market_buy_state = "selloff"
    market_can_buy = False
    if recent_washout_peak < 0.50:
        market_buy_state = "no_setup"
    elif _as_float(current.get("overheat_score"), 0.0) >= 0.58:
        market_buy_state = "rebound_crowded"
    elif (
        _as_float(current.get("repair_score"), 0.0) >= 0.44
        and _as_float(current.get("early_entry_share"), 0.0) >= 0.03
        and market_buy_score >= 0.50
    ):
        market_buy_state = "buy_window"
        market_can_buy = True
    elif (
        recent_washout_peak >= 0.50
        and _as_float(current.get("repair_score"), 0.0) >= 0.30
        and _as_float(current.get("eq_ret_1"), -1.0) >= -0.01
        and market_buy_score >= 0.38
    ):
        market_buy_state = "repair_watch"
        market_can_buy = True

    if market_buy_state == "no_setup":
        reason = "最近窗口里没有形成足够强的连续下跌压力，当前更像普通波动，而不是跌后修复买点。"
    elif market_buy_state == "selloff":
        reason = "近期下跌压力已经建立，但修复广度和资金回流还不够，市场仍更接近下跌段而不是买入段。"
    elif market_buy_state == "repair_watch":
        reason = "连续下跌后的修复已经开始，但广度和强度还在早期，适合先观察或小仓试探。"
    elif market_buy_state == "buy_window":
        reason = "连续下跌后的修复已经从观察期进入可执行窗口，广度、量能和资金回流同时改善。"
    else:
        reason = "修复已经展开，但短线反弹开始拥挤，继续追价的性价比显著下降。"

    return {
        "scan_date": str(current.get("trade_date") or ""),
        "market_buy_state": market_buy_state,
        "market_buy_score": market_buy_score,
        "market_washout_score": _as_float(current.get("washout_score"), 0.0),
        "market_repair_score": _as_float(current.get("repair_score"), 0.0),
        "market_overheat_score": _as_float(current.get("overheat_score"), 0.0),
        "recent_washout_peak": recent_washout_peak,
        "market_can_buy": bool(market_can_buy),
        "eq_ret_1": _as_float(current.get("eq_ret_1"), 0.0),
        "eq_ret_3": _as_float(current.get("eq_ret_3"), 0.0),
        "eq_ret_5": _as_float(current.get("eq_ret_5"), 0.0),
        "eq_ret_10": _as_float(current.get("eq_ret_10"), 0.0),
        "eq_ret_20": _as_float(current.get("eq_ret_20"), 0.0),
        "repair_share": _as_float(current.get("repair_share"), 0.0),
        "early_entry_share": _as_float(current.get("early_entry_share"), 0.0),
        "flow_positive_share": _as_float(current.get("flow_positive_share"), 0.0),
        "amount_expand_share": _as_float(current.get("amount_expand_share"), 0.0),
        "reason": reason,
    }


def _rank_sectors(
    current_rows: list[dict[str, Any]],
    market_timing: dict[str, Any],
    config: ContinuousDeclineRecoveryConfig,
) -> list[dict[str, Any]]:
    df = pd.DataFrame([row for row in current_rows if row.get("tradable_ok")])
    if df.empty:
        return []

    sector_rows: list[dict[str, Any]] = []
    market_eq_ret_5 = _as_float(market_timing.get("eq_ret_5"), 0.0)
    for industry, group in df.groupby("industry"):
        if len(group) < int(config.min_sector_members):
            continue
        sector_damage = float(group["damage_score"].mean())
        sector_repair = float(group["repair_score"].mean())
        sector_entry_share = float(group["base_candidate_flag"].mean())
        sector_flow = float(group["flow_support_score"].mean())
        sector_overheat = float(group["overheat_score"].mean())
        sector_ret_3 = float(group["ret_3"].mean())
        sector_ret_5 = float(group["ret_5"].mean())
        sector_rel_5 = sector_ret_5 - market_eq_ret_5
        sector_rel_score = _scaled_scalar(sector_rel_5, -0.02, 0.05)
        sector_score = _clip01(
            0.20 * sector_damage
            + 0.28 * sector_repair
            + 0.17 * sector_entry_share
            + 0.15 * sector_flow
            + 0.10 * sector_rel_score
            + 0.10 * (1.0 - sector_overheat)
        )
        if sector_score >= 0.62 and sector_repair >= 0.50:
            sector_state = "leading_repair"
        elif sector_score >= 0.48:
            sector_state = "watch"
        else:
            sector_state = "lagging"
        sector_rows.append(
            {
                "scan_date": str(df["trade_date_str"].iloc[0]),
                "industry": str(industry or "UNKNOWN"),
                "member_count": int(len(group)),
                "sector_score": sector_score,
                "sector_state": sector_state,
                "sector_damage_score": sector_damage,
                "sector_repair_score": sector_repair,
                "sector_entry_share": sector_entry_share,
                "sector_flow_score": sector_flow,
                "sector_overheat_score": sector_overheat,
                "sector_ret_3": sector_ret_3,
                "sector_ret_5": sector_ret_5,
                "sector_relative_strength_5": sector_rel_5,
            }
        )

    sector_rows = sorted(
        sector_rows,
        key=lambda row: (
            -_as_float(row.get("sector_score"), 0.0),
            -_as_float(row.get("sector_relative_strength_5"), 0.0),
            -_as_float(row.get("sector_repair_score"), 0.0),
            -int(row.get("member_count", 0)),
        ),
    )
    for rank, row in enumerate(sector_rows, start=1):
        row["sector_rank"] = int(rank)
        row["sector_selected"] = bool(
            row["sector_rank"] <= int(config.top_sectors)
            and _as_float(row.get("sector_score"), 0.0) >= 0.48
            and _as_float(row.get("sector_repair_score"), 0.0) >= 0.45
        )
    return sector_rows


def _execution_cost_state(row: dict[str, Any], config: ContinuousDeclineRecoveryConfig) -> str:
    amount = _as_float(row.get("amount"), 0.0)
    turnover_rate = _as_float(row.get("turnover_rate"), 0.0)
    if amount < float(config.min_amount) or turnover_rate < float(config.min_turnover):
        return "blocked"
    if _as_float(row.get("amount_ratio_20"), 0.0) < 0.90 or _as_float(row.get("atr_ratio_14"), 0.0) > 0.08:
        return "cautious"
    if _as_float(row.get("overheat_score"), 0.0) > 0.45:
        return "cautious"
    return "normal"


def _conditional_relax_rank_limit(config: ContinuousDeclineRecoveryConfig) -> int:
    return max(1, min(3, int(config.top_sectors)))


def _strong_repair_profile(row: dict[str, Any], sector: dict[str, Any], config: ContinuousDeclineRecoveryConfig) -> bool:
    return bool(
        int(sector.get("sector_rank", 999) or 999) <= _conditional_relax_rank_limit(config)
        and _as_float(row.get("repair_score"), 0.0) >= 0.60
        and _as_float(row.get("entry_window_score"), 0.0) >= 0.55
        and _as_float(row.get("flow_support_score"), 0.0) >= 0.45
        and _as_float(row.get("stability_score"), 0.0) >= 0.38
        and _as_float(sector.get("sector_repair_score"), 0.0) >= 0.55
    )


def _damage_threshold(row: dict[str, Any], sector: dict[str, Any], config: ContinuousDeclineRecoveryConfig) -> float:
    if _strong_repair_profile(row, sector, config):
        return 0.24
    return 0.32


def _sector_score_threshold(row: dict[str, Any], sector: dict[str, Any], config: ContinuousDeclineRecoveryConfig) -> float:
    if _strong_repair_profile(row, sector, config) and _as_float(sector.get("sector_relative_strength_5"), 0.0) >= -0.01:
        return 0.40
    return 0.48


def _local_repair_override_allowed(
    row: dict[str, Any],
    sector: dict[str, Any],
    market_timing: dict[str, Any],
    strategy_score: float,
    config: ContinuousDeclineRecoveryConfig,
) -> bool:
    market_state = str(market_timing.get("market_buy_state") or "")
    if market_state not in {"no_setup", "selloff"}:
        return False
    if _as_float(market_timing.get("market_buy_score"), 0.0) < 0.34:
        return False
    if int(sector.get("sector_rank", 999) or 999) > _conditional_relax_rank_limit(config):
        return False
    if _as_float(sector.get("sector_score"), 0.0) < _sector_score_threshold(row, sector, config):
        return False
    if not _strong_repair_profile(row, sector, config):
        return False
    if _as_float(row.get("overheat_score"), 0.0) > 0.20:
        return False
    if _as_float(row.get("rebound_from_low_10"), 0.0) > min(float(config.max_rebound_from_low), 0.13):
        return False
    return bool(strategy_score >= 0.52)


def _strategy_score_threshold(market_state: str) -> float:
    if str(market_state) in {"buy_window", "repair_watch"}:
        return 0.46
    return 0.50


def _entry_plan(row: dict[str, Any], market_timing: dict[str, Any], config: ContinuousDeclineRecoveryConfig) -> tuple[str, float, int, str]:
    if not bool(row.get("strategy_state")):
        return "skip", 0.0, 0, "abandon"

    market_state = str(market_timing.get("market_buy_state") or "")
    market_buy_score = _as_float(market_timing.get("market_buy_score"), 0.0)
    stock_score = _as_float(row.get("strategy_score"), 0.0)
    execution_state = str(row.get("execution_cost_state") or "blocked")
    market_access_mode = str(row.get("market_access_mode") or "")

    if market_access_mode == "local_repair_override":
        position_scale = 0.25 if execution_state == "normal" else 0.15
        return "probe", position_scale, 1, "reduce"

    if market_state == "repair_watch":
        position_scale = 0.35 if execution_state == "normal" else 0.25
        return "probe", position_scale, 1, "reduce"

    if market_state == "buy_window" and market_buy_score >= 0.68 and stock_score >= 0.76 and execution_state == "normal":
        return "full", 1.0, 1, "trail"

    if market_state == "buy_window":
        position_scale = 0.65 if execution_state == "normal" else 0.50
        return "staged", position_scale, 2, "trail"

    return "skip", 0.0, 0, "abandon"


def _reason_text(row: dict[str, Any]) -> str:
    return (
        f"近10日跌幅 {_as_float(row.get('ret_10')):.1%}、20日高点回撤 {_as_float(row.get('drawdown_20')):.1%} 后，"
        f"当前自10日低点反弹 {_as_float(row.get('rebound_from_low_10')):.1%}，"
        f"量能放大 {_as_float(row.get('amount_ratio_20')):.2f} 倍，"
        f"5日资金比 {_as_float(row.get('flow_ratio_5')):.2%}，"
        f"所在板块排名 #{int(row.get('sector_rank', 0) or 0)}。"
    )


def _advice_text(row: dict[str, Any], market_timing: dict[str, Any], config: ContinuousDeclineRecoveryConfig) -> str:
    market_state = str(market_timing.get("market_buy_state") or "")
    market_access_mode = str(row.get("market_access_mode") or "")
    if not bool(row.get("strategy_state")):
        if not bool(market_timing.get("market_can_buy")):
            return f"当前市场处于 {market_state}，更适合继续等修复确认，而不是提前抄底。"
        return "当前标的还没有同时满足板块领先、修复进入窗口和不过热三项条件，更适合放入观察池。"

    entry_mode = str(row.get("entry_mode") or "skip")
    if market_access_mode == "local_repair_override":
        return f"当前市场仍处于 {market_state}，但该股修复强度、行业位置和资金回流明显领先，只适合用试探仓位跟踪 {int(config.hold_days)} 个交易日。"
    if entry_mode == "probe":
        return f"当前更适合试探建仓，基础持有周期先按 {int(config.hold_days)} 个交易日估计，若两天内不能继续放量走强则减仓。"
    if entry_mode == "staged":
        return f"当前适合分两段建仓，基础持有周期按 {int(config.hold_days)} 个交易日估计，若回落跌破 5 日线则先收缩仓位。"
    if entry_mode == "full":
        return f"当前处于连续下跌后的高质量修复窗口，可以直接建仓并用 {int(config.hold_days)} 个交易日跟踪，随后用 5 日线或量价转弱做退出。"
    return "当前不建议执行。"


def _enrich_rows(
    current_rows: list[dict[str, Any]],
    sector_rows: list[dict[str, Any]],
    market_timing: dict[str, Any],
    config: ContinuousDeclineRecoveryConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sector_map = {str(row.get("industry") or "UNKNOWN"): row for row in sector_rows}
    market_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    for row in current_rows:
        enriched = dict(row)
        industry = _industry_bucket(enriched)
        sector = sector_map.get(
            industry,
            {
                "sector_rank": 999,
                "sector_score": 0.0,
                "sector_state": "unranked",
                "sector_repair_score": 0.0,
                "sector_entry_share": 0.0,
                "sector_relative_strength_5": 0.0,
                "sector_selected": False,
                "member_count": 0,
                "industry": industry,
            },
        )
        relative_strength_vs_sector = _as_float(enriched.get("ret_5"), 0.0) - _as_float(sector.get("sector_ret_5"), 0.0)
        relative_strength_score = _scaled_scalar(relative_strength_vs_sector, -0.04, 0.08)
        strategy_score = _clip01(
            0.18 * _as_float(enriched.get("damage_score"), 0.0)
            + 0.24 * _as_float(enriched.get("repair_score"), 0.0)
            + 0.16 * _as_float(enriched.get("entry_window_score"), 0.0)
            + 0.14 * _as_float(enriched.get("flow_support_score"), 0.0)
            + 0.08 * _as_float(enriched.get("stability_score"), 0.0)
            + 0.12 * _as_float(sector.get("sector_score"), 0.0)
            + 0.08 * relative_strength_score
            - 0.10 * _as_float(enriched.get("overheat_score"), 0.0)
        )

        dynamic_damage_threshold = _damage_threshold(enriched, sector, config)
        dynamic_sector_score_threshold = _sector_score_threshold(enriched, sector, config)

        setup_valid = bool(
            enriched.get("tradable_ok")
            and _as_float(enriched.get("damage_score"), 0.0) >= dynamic_damage_threshold
            and _as_float(enriched.get("repair_score"), 0.0) >= 0.48
            and _as_float(enriched.get("entry_window_score"), 0.0) >= 0.40
            and _as_float(enriched.get("flow_support_score"), 0.0) >= 0.35
            and _as_float(enriched.get("rebound_from_low_10"), 0.0) >= float(config.min_rebound_from_low)
            and _as_float(enriched.get("rebound_from_low_10"), 0.0) <= float(config.max_rebound_from_low)
            and _as_float(sector.get("sector_score"), 0.0) >= dynamic_sector_score_threshold
            and int(sector.get("sector_rank", 999)) <= int(config.top_sectors)
        )
        market_state = str(market_timing.get("market_buy_state") or "")
        broad_market_allows = bool(market_timing.get("market_can_buy"))
        local_repair_override = bool(
            setup_valid and _local_repair_override_allowed(enriched, sector, market_timing, strategy_score, config)
        )
        market_access_mode = "blocked"
        if broad_market_allows:
            market_access_mode = "broad_market"
        elif local_repair_override:
            market_access_mode = "local_repair_override"
        market_allows_trade = bool(broad_market_allows or local_repair_override)
        threshold = _strategy_score_threshold(market_state)
        strategy_state = bool(setup_valid and market_allows_trade and strategy_score >= threshold)

        enriched.update(
            {
                "strategy_name": str(config.strategy_name),
                "scan_date": str(market_timing.get("scan_date") or enriched.get("trade_date_str") or ""),
                "market_buy_state": str(market_timing.get("market_buy_state") or ""),
                "market_buy_score": _as_float(market_timing.get("market_buy_score"), 0.0),
                "market_washout_score": _as_float(market_timing.get("market_washout_score"), 0.0),
                "market_repair_score": _as_float(market_timing.get("market_repair_score"), 0.0),
                "market_overheat_score": _as_float(market_timing.get("market_overheat_score"), 0.0),
                "market_reason": str(market_timing.get("reason") or ""),
                "sector_rank": int(sector.get("sector_rank", 999) or 999),
                "sector_score": _as_float(sector.get("sector_score"), 0.0),
                "sector_state": str(sector.get("sector_state") or "unranked"),
                "sector_repair_score": _as_float(sector.get("sector_repair_score"), 0.0),
                "sector_entry_share": _as_float(sector.get("sector_entry_share"), 0.0),
                "sector_relative_strength_5": _as_float(sector.get("sector_relative_strength_5"), 0.0),
                "sector_selected": bool(sector.get("sector_selected")),
                "strategy_score": strategy_score,
                "strategy_state": bool(strategy_state),
                "candidate_flag": bool(setup_valid),
                "damage_threshold": float(dynamic_damage_threshold),
                "sector_score_threshold": float(dynamic_sector_score_threshold),
                "strategy_score_threshold": float(threshold),
                "local_repair_override": bool(local_repair_override),
                "market_access_mode": str(market_access_mode),
                "market_allows_trade": bool(market_allows_trade),
                "relative_strength_vs_sector_5": relative_strength_vs_sector,
                "relative_strength_score": relative_strength_score,
                "execution_cost_state": _execution_cost_state(enriched, config),
            }
        )
        entry_mode, position_scale, staged_entry_days, exit_mode = _entry_plan(enriched, market_timing, config)
        enriched["entry_mode"] = entry_mode
        enriched["position_scale"] = position_scale
        enriched["staged_entry_days"] = int(staged_entry_days)
        enriched["exit_mode"] = exit_mode
        enriched["reason"] = _reason_text(enriched)
        enriched["action_advice"] = _advice_text(enriched, market_timing, config)

        market_rows.append(enriched)
        if setup_valid:
            candidate_rows.append(enriched)

    return market_rows, candidate_rows


def _select_portfolio(candidate_rows: list[dict[str, Any]], config: ContinuousDeclineRecoveryConfig) -> list[dict[str, Any]]:
    eligible = [row for row in candidate_rows if bool(row.get("strategy_state")) and str(row.get("entry_mode")) != "skip"]
    eligible = sorted(
        eligible,
        key=lambda row: (
            -_as_float(row.get("sector_score"), 0.0),
            -_as_float(row.get("strategy_score"), 0.0),
            -_as_float(row.get("repair_score"), 0.0),
            -_as_float(row.get("flow_support_score"), 0.0),
            -_as_float(row.get("amount"), 0.0),
        ),
    )

    selected: list[dict[str, Any]] = []
    industry_counts: dict[str, int] = {}
    for row in eligible:
        industry = _industry_bucket(row)
        if len(selected) >= int(config.max_positions):
            break
        if industry_counts.get(industry, 0) >= int(config.max_positions_per_industry):
            continue
        selected_row = dict(row)
        selected_row["selected_rank"] = int(len(selected) + 1)
        selected.append(selected_row)
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
    return selected


def _trade_plan_from_snapshot(prepared: PreparedSymbol, snapshot_row: dict[str, Any], scan_date: str, hold_days: int) -> dict[str, Any] | None:
    idx = prepared.date_to_index.get(str(scan_date))
    if idx is None:
        return None
    entry_idx = idx + 1
    if entry_idx >= len(prepared.daily):
        return None

    exit_idx = min(entry_idx + max(1, int(hold_days)) - 1, len(prepared.daily) - 1)
    entry_row = prepared.daily.iloc[entry_idx]
    exit_row = prepared.daily.iloc[exit_idx]
    entry_price = _as_float(entry_row.get("open"), 0.0)
    exit_price = _as_float(exit_row.get("close"), 0.0)
    if entry_price <= 0.0 or exit_price <= 0.0:
        return None

    holding_window = prepared.daily.iloc[entry_idx : exit_idx + 1]
    max_high = _as_float(holding_window["high"].max(), np.nan)
    min_low = _as_float(holding_window["low"].min(), np.nan)
    return {
        "entry_date": str(entry_row.get("trade_date_str") or ""),
        "exit_date": str(exit_row.get("trade_date_str") or ""),
        "entry_scan_date": str(scan_date),
        "entry_price": float(entry_price),
        "exit_price": float(exit_price),
        "hold_days_realized": int(exit_idx - entry_idx + 1),
        "full_hold_reached": bool(exit_idx == entry_idx + max(1, int(hold_days)) - 1),
        "max_runup_pct": float((max_high / entry_price - 1.0) * 100.0) if np.isfinite(max_high) and max_high > 0.0 else np.nan,
        "max_drawdown_pct": float((min_low / entry_price - 1.0) * 100.0) if np.isfinite(min_low) and min_low > 0.0 else np.nan,
        "return_pct": float((exit_price / entry_price - 1.0) * 100.0),
        "scan_to_entry_gap_days": int(entry_idx - idx),
    }


def _daily_mark_to_market_return(prepared: PreparedSymbol, trade: dict[str, Any], current_date: str) -> float | None:
    if str(current_date) < str(trade.get("entry_date") or "") or str(current_date) > str(trade.get("exit_date") or ""):
        return None

    current_close = prepared.close_by_date.get(str(current_date))
    if current_close is None or not np.isfinite(current_close) or current_close <= 0.0:
        return None

    if str(current_date) == str(trade.get("entry_date") or ""):
        entry_price = _as_float(trade.get("entry_price"), 0.0)
        if entry_price <= 0.0:
            return None
        return float(current_close / entry_price - 1.0)

    current_idx = prepared.date_to_index.get(str(current_date))
    if current_idx is None or current_idx <= 0:
        return None
    prev_date = str(prepared.daily.iloc[current_idx - 1]["trade_date_str"])
    prev_close = prepared.close_by_date.get(prev_date)
    if prev_close is None or not np.isfinite(prev_close) or prev_close <= 0.0:
        return None
    return float(current_close / prev_close - 1.0)


def _position_live_scale(prepared: PreparedSymbol, trade: dict[str, Any], current_date: str) -> float:
    if str(current_date) < str(trade.get("entry_date") or "") or str(current_date) > str(trade.get("exit_date") or ""):
        return 0.0

    target_scale = _as_float(trade.get("position_scale"), 0.0)
    if target_scale <= 0.0:
        return 0.0

    mode = str(trade.get("entry_mode") or "full")
    if mode == "full":
        return target_scale

    current_idx = prepared.date_to_index.get(str(current_date))
    entry_idx = prepared.date_to_index.get(str(trade.get("entry_date") or ""))
    if current_idx is None or entry_idx is None:
        return target_scale

    held_days = max(0, current_idx - entry_idx)
    ramp_days = max(1, int(trade.get("staged_entry_days") or 1))
    fraction = min(1.0, float(held_days + 1) / float(ramp_days))
    if mode == "probe":
        fraction = min(fraction, 1.0)
    return float(target_scale * fraction)


def _run_forward_backtest(
    prepared_symbols: list[PreparedSymbol],
    master_dates: np.ndarray,
    market_cache: pd.DataFrame,
    config: ContinuousDeclineRecoveryConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    if not config.backtest_start_date or not config.backtest_end_date:
        return [], [], None
    if len(master_dates) == 0:
        return [], [], None

    start_pos = int(np.searchsorted(master_dates, str(config.backtest_start_date), side="left"))
    end_pos = int(np.searchsorted(master_dates, str(config.backtest_end_date), side="right") - 1)
    if start_pos >= len(master_dates) or end_pos < 0 or start_pos > end_pos:
        return [], [], None

    eval_end_pos = min(len(master_dates) - 1, end_pos + max(1, int(config.hold_days)))
    evaluation_dates = [str(value) for value in master_dates[start_pos : eval_end_pos + 1]]
    scan_dates = set(str(value) for value in master_dates[start_pos : end_pos + 1])
    if not evaluation_dates:
        return [], [], None

    symbol_lookup = {prepared.symbol: prepared for prepared in prepared_symbols}
    active_positions: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    nav = 1.0

    for current_date in evaluation_dates:
        active_positions = [position for position in active_positions if str(position.get("exit_date") or "") >= str(current_date)]
        candidate_rows_day: list[dict[str, Any]] = []
        selected_rows_day: list[dict[str, Any]] = []
        sector_rows_day: list[dict[str, Any]] = []
        summary_day: dict[str, Any] = {}
        skipped_full = 0
        skipped_industry = 0
        skipped_duplicate = 0
        accepted_today = 0
        scan_enabled = current_date in scan_dates

        if scan_enabled:
            _, sector_rows_day, candidate_rows_day, selected_rows_day, summary_day = _evaluate_scan_date(
                prepared_symbols,
                master_dates,
                market_cache,
                current_date,
                config,
            )
            for selected_row in selected_rows_day:
                prepared = symbol_lookup.get(str(selected_row.get("symbol") or ""))
                if prepared is None:
                    continue
                trade_plan = _trade_plan_from_snapshot(prepared, selected_row, current_date, int(config.hold_days))
                if trade_plan is None:
                    continue

                entry_date = str(trade_plan.get("entry_date") or "")
                active_positions = [position for position in active_positions if str(position.get("exit_date") or "") >= entry_date]
                if any(str(position.get("symbol") or "") == str(selected_row.get("symbol") or "") for position in active_positions):
                    skipped_duplicate += 1
                    continue
                if len(active_positions) >= int(config.max_positions):
                    skipped_full += 1
                    continue

                industry = _industry_bucket(selected_row)
                industry_open_count = sum(1 for position in active_positions if _industry_bucket(position) == industry)
                if int(config.max_positions_per_industry) > 0 and industry_open_count >= int(config.max_positions_per_industry):
                    skipped_industry += 1
                    continue

                trade = {**selected_row, **trade_plan, "scan_date": str(current_date)}
                trades.append(trade)
                active_positions.append(
                    {
                        "symbol": trade["symbol"],
                        "ts_code": trade["ts_code"],
                        "name": trade.get("name"),
                        "industry": trade.get("industry"),
                        "market": trade.get("market"),
                        "entry_date": entry_date,
                        "exit_date": str(trade.get("exit_date") or ""),
                        "entry_price": float(trade.get("entry_price") or 0.0),
                        "position_scale": float(trade.get("position_scale") or 0.0),
                        "entry_mode": str(trade.get("entry_mode") or "full"),
                        "staged_entry_days": int(trade.get("staged_entry_days") or 1),
                    }
                )
                accepted_today += 1

        weighted_return_sum = 0.0
        gross_exposure = 0.0
        live_positions = 0
        for position in active_positions:
            prepared = symbol_lookup.get(str(position.get("symbol") or ""))
            if prepared is None:
                continue
            live_scale = _position_live_scale(prepared, position, current_date)
            trade_return = _daily_mark_to_market_return(prepared, position, current_date)
            if live_scale > 0.0:
                gross_exposure += float(live_scale)
                live_positions += 1
            if live_scale > 0.0 and trade_return is not None:
                weighted_return_sum += float(live_scale) * float(trade_return)

        strategy_daily_return = float(weighted_return_sum / max(float(config.max_positions), 1.0))
        nav *= 1.0 + strategy_daily_return
        realized_trades = [trade for trade in trades if str(trade.get("exit_date") or "") == str(current_date)]

        market_state = str(summary_day.get("market_buy_state") or "") if summary_day else ""
        market_score = _as_float(summary_day.get("market_buy_score"), np.nan) if summary_day else np.nan
        top_sector = str(summary_day.get("top_sector") or "") if summary_day else ""

        daily_rows.append(
            {
                "strategy_name": str(config.strategy_name),
                "scan_date": str(current_date),
                "is_scan_day": bool(scan_enabled),
                "market_buy_state": market_state,
                "market_buy_score": market_score,
                "top_sector": top_sector,
                "n_sector_ranked": int(len(sector_rows_day)) if scan_enabled else 0,
                "n_candidates": int(len(candidate_rows_day)) if scan_enabled else 0,
                "n_selected": int(accepted_today),
                "n_skipped_full": int(skipped_full),
                "n_skipped_industry": int(skipped_industry),
                "n_skipped_duplicate": int(skipped_duplicate),
                "active_positions": int(live_positions),
                "gross_exposure": float(gross_exposure),
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
            "top_sectors": int(config.top_sectors),
            "max_positions": int(config.max_positions),
            "max_positions_per_industry": int(config.max_positions_per_industry),
            "n_scan_days": int(len(scan_dates)),
            "n_eval_days": int(len(evaluation_dates)),
            "n_trades": 0,
            "final_nav": float(nav),
            "total_return_pct": float((nav - 1.0) * 100.0),
        }

    df_daily = pd.DataFrame(daily_rows)
    df_trades = pd.DataFrame(trades)
    max_drawdown_pct = np.nan
    if not df_daily.empty and "nav" in df_daily.columns:
        nav_series = df_daily["nav"].astype(float)
        running_peak = nav_series.cummax()
        drawdown = nav_series / running_peak - 1.0
        max_drawdown_pct = float(drawdown.min() * 100.0)

    if not df_daily.empty:
        state_counts = df_daily[df_daily["is_scan_day"] == True]["market_buy_state"].value_counts().to_dict()
    else:
        state_counts = {}

    return daily_rows, trades, {
        "strategy_name": str(config.strategy_name),
        "backtest_start_date": str(config.backtest_start_date),
        "backtest_end_date": str(config.backtest_end_date),
        "hold_days": int(config.hold_days),
        "top_n": int(config.top_n),
        "top_sectors": int(config.top_sectors),
        "max_positions": int(config.max_positions),
        "max_positions_per_industry": int(config.max_positions_per_industry),
        "n_scan_days": int(len(scan_dates)),
        "n_eval_days": int(len(evaluation_dates)),
        "n_days_with_signal": int((df_daily["n_selected"] > 0).sum()) if not df_daily.empty else 0,
        "n_trades": int(len(df_trades)),
        "win_rate": float((df_trades["return_pct"] > 0.0).mean()),
        "avg_return_pct": float(df_trades["return_pct"].mean()),
        "median_return_pct": float(df_trades["return_pct"].median()),
        "avg_max_runup_pct": float(df_trades["max_runup_pct"].mean()),
        "avg_max_drawdown_pct": float(df_trades["max_drawdown_pct"].mean()),
        "avg_hold_days_realized": float(df_trades["hold_days_realized"].mean()),
        "avg_position_scale": float(df_trades["position_scale"].mean()) if "position_scale" in df_trades.columns else np.nan,
        "avg_candidates_per_scan_day": float(df_daily.loc[df_daily["is_scan_day"] == True, "n_candidates"].mean()) if not df_daily.empty else np.nan,
        "avg_selected_per_scan_day": float(df_daily.loc[df_daily["is_scan_day"] == True, "n_selected"].mean()) if not df_daily.empty else np.nan,
        "avg_gross_exposure": float(df_daily["gross_exposure"].mean()) if not df_daily.empty else np.nan,
        "buy_window_days": int(state_counts.get("buy_window", 0)),
        "repair_watch_days": int(state_counts.get("repair_watch", 0)),
        "selloff_days": int(state_counts.get("selloff", 0)),
        "rebound_crowded_days": int(state_counts.get("rebound_crowded", 0)),
        "total_skipped_full": int(df_daily["n_skipped_full"].sum()) if not df_daily.empty else 0,
        "total_skipped_industry": int(df_daily["n_skipped_industry"].sum()) if not df_daily.empty else 0,
        "total_skipped_duplicate": int(df_daily["n_skipped_duplicate"].sum()) if not df_daily.empty else 0,
        "final_nav": float(nav),
        "total_return_pct": float((nav - 1.0) * 100.0),
        "max_drawdown_pct": max_drawdown_pct,
    }


def _evaluate_scan_date(
    prepared_symbols: list[PreparedSymbol],
    master_dates: np.ndarray,
    market_cache: pd.DataFrame,
    scan_date: str,
    config: ContinuousDeclineRecoveryConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    current_rows = _rows_for_date(prepared_symbols, scan_date, config)
    market_frame = _build_market_frame(market_cache, master_dates, scan_date, config)
    market_timing = _derive_market_timing(market_frame)
    market_timing["scan_date"] = str(scan_date)
    sector_rows = _rank_sectors(current_rows, market_timing, config)
    market_rows, candidate_rows = _enrich_rows(current_rows, sector_rows, market_timing, config)
    selected_rows = _select_portfolio(candidate_rows, config)

    summary = {
        "strategy_name": str(config.strategy_name),
        "scan_date": str(scan_date),
        "market_buy_state": str(market_timing.get("market_buy_state") or ""),
        "market_buy_score": _as_float(market_timing.get("market_buy_score"), 0.0),
        "market_washout_score": _as_float(market_timing.get("market_washout_score"), 0.0),
        "market_repair_score": _as_float(market_timing.get("market_repair_score"), 0.0),
        "market_overheat_score": _as_float(market_timing.get("market_overheat_score"), 0.0),
        "recent_washout_peak": _as_float(market_timing.get("recent_washout_peak"), 0.0),
        "market_can_buy": bool(market_timing.get("market_can_buy")),
        "market_reason": str(market_timing.get("reason") or ""),
        "n_scanned": int(len(market_rows)),
        "n_candidates": int(len(candidate_rows)),
        "n_selected": int(len(selected_rows)),
        "n_selected_industries": int(len({row.get('industry') for row in selected_rows})),
        "top_sector": str(sector_rows[0]["industry"]) if sector_rows else "",
        "top_sector_score": _as_float(sector_rows[0].get("sector_score"), np.nan) if sector_rows else np.nan,
        "avg_candidate_score": float(pd.DataFrame(candidate_rows)["strategy_score"].mean()) if candidate_rows else np.nan,
        "avg_selected_score": float(pd.DataFrame(selected_rows)["strategy_score"].mean()) if selected_rows else np.nan,
    }
    return market_rows, sector_rows, candidate_rows, selected_rows, summary


def run_continuous_decline_recovery_scan(config: ContinuousDeclineRecoveryConfig) -> list[str]:
    files = _resolve_files(config.data_dir, config.symbols)
    if not files:
        raise SystemExit(f"No CSV files found under {config.data_dir}")

    if bool(config.backtest_start_date) != bool(config.backtest_end_date):
        raise SystemExit("backtest_start_date and backtest_end_date must be provided together.")

    requested_scan_date = _infer_scan_date(files, config.scan_date)
    latest_available_date = _infer_scan_date(files, "")
    history_anchor_date = str(config.backtest_start_date or requested_scan_date)
    min_keep_date = f"{int(history_anchor_date[:4]) - int(config.lookback_years)}0101"
    max_needed_date = latest_available_date if config.backtest_end_date else requested_scan_date
    basic_info_map = _load_basic_info_map(config.basic_path)
    prepared_symbols: list[PreparedSymbol] = []
    for file_path in files:
        prepared = _prepare_symbol_state(file_path, min_keep_date, max_needed_date, basic_info_map)
        if prepared is not None:
            prepared_symbols.append(prepared)
    if not prepared_symbols:
        raise SystemExit("Unable to prepare any symbols for continuous_decline_recovery.")

    master_dates = _master_trade_dates(prepared_symbols)
    market_cache = _build_market_aggregate_cache(prepared_symbols, master_dates, config)
    scan_date = _resolve_scan_date_from_master(master_dates, requested_scan_date)
    resolved_backtest_start = ""
    resolved_backtest_end = ""
    if config.backtest_start_date and config.backtest_end_date:
        resolved_backtest_start = _resolve_scan_date_from_master(master_dates, config.backtest_start_date)
        resolved_backtest_end = _resolve_scan_date_from_master(master_dates, config.backtest_end_date)

    runtime_config = ContinuousDeclineRecoveryConfig(
        **{
            **vars(config),
            "scan_date": scan_date,
            "backtest_start_date": resolved_backtest_start,
            "backtest_end_date": resolved_backtest_end,
        }
    )

    market_rows, sector_rows, candidate_rows, selected_rows, summary = _evaluate_scan_date(
        prepared_symbols,
        master_dates,
        market_cache,
        scan_date,
        runtime_config,
    )

    backtest_daily_rows, backtest_trade_rows, backtest_summary = _run_forward_backtest(
        prepared_symbols,
        master_dates,
        market_cache,
        runtime_config,
    )

    if backtest_summary is None:
        backtest_summary = {
            "strategy_name": str(runtime_config.strategy_name),
            "scan_date": str(scan_date),
            "status": "not_run",
            "hold_days": int(runtime_config.hold_days),
            "message": "未提供 backtest_start_date/backtest_end_date，已跳过滚动前瞻回测。",
        }
    else:
        backtest_summary = {**backtest_summary, "scan_date": str(scan_date), "status": "completed"}

    return write_continuous_decline_recovery_outputs(
        out_dir=runtime_config.out_dir,
        strategy_name=runtime_config.strategy_name,
        scan_date=scan_date,
        top_n=runtime_config.top_n,
        market_rows=market_rows,
        sector_rows=sector_rows,
        candidate_rows=candidate_rows,
        selected_rows=selected_rows,
        summary=summary,
        backtest_daily_rows=backtest_daily_rows,
        backtest_trade_rows=backtest_trade_rows,
        backtest_summary=backtest_summary,
    )