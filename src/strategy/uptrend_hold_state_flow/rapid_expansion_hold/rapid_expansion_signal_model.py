import numpy as np
import pandas as pd


def _bool_series(values: pd.Series | np.ndarray) -> pd.Series:
    return pd.Series(values, dtype="boolean").fillna(False).astype(bool)


def evaluate_rapid_expansion_path(
    df: pd.DataFrame,
    start_date: str,
    scan_date: str = "",
    exit_persist_days: int = 2,
) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    out = df.copy().sort_values("bar_end_dt").reset_index(drop=True)
    out["bar_end"] = out["bar_end"].astype(str)
    start_date = str(start_date or "").strip()
    scan_date = str(scan_date or "").strip()
    if not start_date:
        raise ValueError("start_date is required")

    if scan_date:
        out = out[out["bar_end"] <= scan_date].copy()
    out = out[out["bar_end"] >= start_date].copy().reset_index(drop=True)
    if out.empty:
        raise ValueError("No trading rows found on or after start_date")

    persist_days = max(1, int(exit_persist_days))
    exit_seed = _bool_series(out["exit_seed"])
    exit_persist = exit_seed.rolling(window=persist_days, min_periods=persist_days).sum() >= persist_days
    out["exit_persist"] = exit_persist.to_numpy(dtype=bool)

    hold_path_state = np.ones(len(out), dtype=bool)
    if bool(out["exit_persist"].any()):
        first_exit_idx = int(np.flatnonzero(out["exit_persist"].to_numpy(dtype=bool))[0])
        hold_path_state[first_exit_idx:] = False
    out["hold_path_state"] = hold_path_state
    out["strategy_state"] = hold_path_state
    out["hold_score"] = pd.to_numeric(out["expansion_hold_score"], errors="coerce").fillna(0.0)
    out["strategy_score"] = out["hold_score"]
    out["strategy_name"] = "rapid_expansion_hold"
    return out