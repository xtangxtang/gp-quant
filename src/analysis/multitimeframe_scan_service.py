import glob
import os
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from .multitimeframe_evaluation import (
    build_resonance_daily_frame,
    compute_year_return_for_file,
    eval_first_entry_in_year,
    eval_first_resonance_in_year,
)
from .multitimeframe_feature_engine import (
    aggregate_stock_bars,
    compute_physics_state_features,
    to_trade_date_str,
)
from .multitimeframe_physics_utils import build_index_monthly_regime_by_date
from .multitimeframe_report_writer import write_scan_outputs


@dataclass(frozen=True)
class ScanConfig:
    data_dir: str
    out_dir: str
    test_year: int = 2025
    top_n: int = 300
    symbols: str = ""
    index_path: str = ""
    basic_path: str = ""
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
    resonance_min_count: int = 3
    resonance_persist_days: int = 2
    weekly_support_threshold: float = 0.10
    monthly_support_threshold: float = 0.08


def _load_basic_name_map(basic_path: str) -> dict[str, str]:
    if not basic_path or not os.path.exists(basic_path):
        return {}
    try:
        df = pd.read_csv(basic_path, usecols=["ts_code", "name"])
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return {}
        return dict(zip(df["ts_code"].astype(str), df["name"].astype(str)))
    except Exception:
        return {}


def _load_index_regime(index_path: str, gate_index: bool) -> tuple[dict[str, str], set[str] | None]:
    index_regime_by_date: dict[str, str] = {}
    allow_index_regimes = {"BULL", "BASE"} if bool(gate_index) else None
    if index_path:
        df_idx = pd.read_csv(index_path)
        df_idx["trade_date_str"] = df_idx["trade_date"].astype(str)
        df_idx = df_idx.sort_values("trade_date_str").reset_index(drop=True)
        _, index_regime_by_date = build_index_monthly_regime_by_date(df_idx)
    return index_regime_by_date, allow_index_regimes


def _resolve_files(data_dir: str, symbols_arg: str) -> list[str]:
    symbols = [s.strip() for s in str(symbols_arg).split(",") if s.strip()]
    if symbols:
        files = [os.path.join(data_dir, f"{s}.csv") for s in symbols if os.path.exists(os.path.join(data_dir, f"{s}.csv"))]
    else:
        files = glob.glob(os.path.join(data_dir, "*.csv"))
    return sorted(files)


def _rank_bull_stocks(files: list[str], year: int) -> pd.DataFrame:
    with Pool(max(1, cpu_count() - 1)) as pool:
        rows = list(pool.starmap(compute_year_return_for_file, [(fp, year) for fp in files]))
    rows = [r for r in rows if r]
    if not rows:
        raise SystemExit("No stocks with sufficient data to compute yearly return.")
    return pd.DataFrame(rows).sort_values("year_return", ascending=False).reset_index(drop=True)


def _compute_single_symbol_outputs(
    df: pd.DataFrame,
    config: ScanConfig,
    index_regime_by_date: dict[str, str],
    allow_index_regimes: set[str] | None,
    row: pd.Series,
    name_map: dict[str, str],
) -> tuple[list[dict], dict | None, list[dict]]:
    out_rows: list[dict] = []

    start_keep = f"{int(config.test_year) - 5}0101"
    end_keep = f"{int(config.test_year)}1231"
    df = df[(df["trade_date_str"] >= start_keep) & (df["trade_date_str"] <= end_keep)].copy()
    if df.empty:
        return [], None, []

    ts_code = str(df["ts_code"].iloc[0]) if "ts_code" in df.columns else str(row.get("ts_code") or "")
    name = str(name_map.get(ts_code, "") or "")
    symbol = os.path.splitext(os.path.basename(str(row["file_path"])))[0]

    daily_bars = aggregate_stock_bars(df, "D")
    weekly_bars = aggregate_stock_bars(df, "W")
    monthly_bars = aggregate_stock_bars(df, "M")
    if daily_bars.empty or weekly_bars.empty or monthly_bars.empty:
        return [], None, []

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
        return [], None, []

    evals = [
        eval_first_entry_in_year(daily_feat, "state", "bar_end", "bar_start", index_regime_by_date, int(config.test_year), allow_index_regimes, "D"),
        eval_first_entry_in_year(weekly_feat, "state", "bar_end", "bar_start", index_regime_by_date, int(config.test_year), allow_index_regimes, "W"),
        eval_first_entry_in_year(monthly_feat, "state", "bar_end", "bar_start", index_regime_by_date, int(config.test_year), allow_index_regimes, "M"),
    ]

    for ev in evals:
        out_rows.append(
            {
                "symbol": symbol,
                "ts_code": ts_code,
                "name": name,
                "freq": ev.freq,
                "year_return": float(row["year_return"]),
                "n_signals": int(ev.n_signals),
                "first_signal_date": ev.first_signal_date,
                "entry_date": ev.entry_date,
                "entry_price": ev.entry_price,
                "score": ev.score,
                "score_min_persist": ev.score_min_persist,
                "energy_term": ev.energy_term,
                "temperature_term": ev.temperature_term,
                "order_term": ev.order_term,
                "phase_term": ev.phase_term,
                "switch_term": ev.switch_term,
                "index_regime": ev.index_regime,
                "ret_to_year_end": ev.ret_to_year_end,
                "max_runup_to_year_end": ev.max_runup_to_year_end,
            }
        )

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
        return out_rows, None, []

    resonance_eval = eval_first_resonance_in_year(
        resonance_daily,
        index_regime_by_date=index_regime_by_date,
        year=int(config.test_year),
        allow_index_regimes=allow_index_regimes,
    )
    resonance_row = {
        "symbol": symbol,
        "ts_code": ts_code,
        "name": name,
        "year_return": float(row["year_return"]),
        "n_signals": int(resonance_eval.n_signals),
        "first_signal_date": resonance_eval.first_signal_date,
        "entry_date": resonance_eval.entry_date,
        "entry_price": resonance_eval.entry_price,
        "daily_score": resonance_eval.daily_score,
        "weekly_score": resonance_eval.weekly_score,
        "monthly_score": resonance_eval.monthly_score,
        "resonance_score": resonance_eval.resonance_score,
        "support_count": resonance_eval.support_count,
        "index_regime": resonance_eval.index_regime,
        "ret_to_year_end": resonance_eval.ret_to_year_end,
        "max_runup_to_year_end": resonance_eval.max_runup_to_year_end,
    }

    y0 = f"{int(config.test_year)}0101"
    y1 = f"{int(config.test_year)}1231"
    sig_df = resonance_daily[
        resonance_daily["bar_end"].astype(str).between(y0, y1) & resonance_daily["resonance_state"].fillna(False)
    ].copy()
    signal_rows = []
    for _, sr in sig_df.iterrows():
        signal_rows.append(
            {
                "symbol": symbol,
                "ts_code": ts_code,
                "name": name,
                "signal_date": str(sr["bar_end"]),
                "daily_score": float(sr["daily_score_ctx"]) if np.isfinite(sr["daily_score_ctx"]) else None,
                "weekly_score": float(sr["weekly_score_ctx"]) if np.isfinite(sr["weekly_score_ctx"]) else None,
                "monthly_score": float(sr["monthly_score_ctx"]) if np.isfinite(sr["monthly_score_ctx"]) else None,
                "resonance_score": float(sr["resonance_score"]) if np.isfinite(sr["resonance_score"]) else None,
                "support_count": int(sr["support_count"]),
                "energy_term": float(sr["energy_term"]) if np.isfinite(sr["energy_term"]) else None,
                "temperature_term": float(sr["temperature_term"]) if np.isfinite(sr["temperature_term"]) else None,
                "order_term": float(sr["order_term"]) if np.isfinite(sr["order_term"]) else None,
                "phase_term": float(sr["phase_term"]) if np.isfinite(sr["phase_term"]) else None,
            }
        )

    return out_rows, resonance_row, signal_rows


def run_multitimeframe_scan(config: ScanConfig) -> list[str]:
    index_regime_by_date, allow_index_regimes = _load_index_regime(config.index_path, config.gate_index)
    name_map = _load_basic_name_map(config.basic_path)

    files = _resolve_files(config.data_dir, config.symbols)
    if not files:
        raise SystemExit(f"No CSVs found in data_dir={config.data_dir}")

    df_ret = _rank_bull_stocks(files, int(config.test_year))

    df_bull = df_ret.head(int(config.top_n)).copy().reset_index(drop=True)
    df_bull["name"] = df_bull["ts_code"].astype(str).map(name_map).fillna("")

    out_rows: list[dict] = []
    resonance_rows: list[dict] = []
    resonance_signal_rows: list[dict] = []

    for _, row in df_bull.iterrows():
        try:
            df = pd.read_csv(str(row["file_path"]))
        except Exception:
            continue
        if df.empty or "trade_date" not in df.columns:
            continue
        df = to_trade_date_str(df)
        if df.empty:
            continue

        stock_out_rows, resonance_row, signal_rows = _compute_single_symbol_outputs(
            df=df,
            config=config,
            index_regime_by_date=index_regime_by_date,
            allow_index_regimes=allow_index_regimes,
            row=row,
            name_map=name_map,
        )
        out_rows.extend(stock_out_rows)
        if resonance_row:
            resonance_rows.append(resonance_row)
        resonance_signal_rows.extend(signal_rows)

    return write_scan_outputs(
        out_dir=config.out_dir,
        test_year=int(config.test_year),
        top_n=int(config.top_n),
        df_ret=df_ret,
        df_bull=df_bull,
        out_rows=out_rows,
        resonance_rows=resonance_rows,
        resonance_signal_rows=resonance_signal_rows,
    )