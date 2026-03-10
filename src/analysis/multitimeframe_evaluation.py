from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EntryEval:
    freq: str
    first_signal_date: str | None
    entry_date: str | None
    entry_price: float | None
    score: float | None
    score_min_persist: float | None
    energy_term: float | None
    temperature_term: float | None
    order_term: float | None
    phase_term: float | None
    switch_term: float | None
    index_regime: str | None
    ret_to_year_end: float | None
    max_runup_to_year_end: float | None
    n_signals: int


@dataclass(frozen=True)
class ResonanceEval:
    first_signal_date: str | None
    entry_date: str | None
    entry_price: float | None
    daily_score: float | None
    weekly_score: float | None
    monthly_score: float | None
    resonance_score: float | None
    support_count: int
    index_regime: str | None
    ret_to_year_end: float | None
    max_runup_to_year_end: float | None
    n_signals: int


@dataclass(frozen=True)
class SignalSnapshot:
    signal_date: str | None
    daily_bar_end: str | None
    weekly_bar_end: str | None
    monthly_bar_end: str | None
    daily_state: bool
    weekly_state: bool
    monthly_state: bool
    resonance_state: bool
    support_count: int
    daily_score: float | None
    weekly_score: float | None
    monthly_score: float | None
    resonance_score: float | None
    energy_term: float | None
    temperature_term: float | None
    order_term: float | None
    phase_term: float | None
    switch_term: float | None


def eval_first_entry_in_year(
    features: pd.DataFrame,
    state_col: str,
    date_col_end: str,
    date_col_start: str,
    index_regime_by_date: dict[str, str] | None,
    year: int,
    allow_index_regimes: set[str] | None,
    freq: str,
) -> EntryEval:
    if features is None or not isinstance(features, pd.DataFrame) or features.empty:
        return EntryEval(freq, None, None, None, None, None, None, None, None, None, None, None, None, None, 0)

    df = features.copy().reset_index(drop=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df = df.dropna(subset=["open", "close"]).reset_index(drop=True)
    if df.empty:
        return EntryEval(freq, None, None, None, None, None, None, None, None, None, None, None, None, None, 0)

    y0 = f"{int(year)}0101"
    y1 = f"{int(year)}1231"

    if index_regime_by_date:
        df["index_regime"] = df[date_col_end].astype(str).map(index_regime_by_date).fillna("BASE")
    else:
        df["index_regime"] = "BASE"

    ok = df[state_col].fillna(False).to_numpy(dtype=bool)
    ok = ok & df[date_col_end].astype(str).between(y0, y1).to_numpy()
    if allow_index_regimes:
        ok = ok & df["index_regime"].isin(set(allow_index_regimes)).to_numpy()

    sig_idx = np.where(ok)[0]
    n_signals = int(len(sig_idx))
    if n_signals == 0:
        return EntryEval(freq, None, None, None, None, None, None, None, None, None, None, None, None, None, 0)

    i = int(sig_idx[0])
    entry_i = i + 1
    score = float(df.loc[i, "score"]) if np.isfinite(df.loc[i, "score"]) else None
    score_min_persist = float(df.loc[i, "score_min_persist"]) if np.isfinite(df.loc[i, "score_min_persist"]) else None
    energy_term = float(df.loc[i, "energy_term"]) if np.isfinite(df.loc[i, "energy_term"]) else None
    temperature_term = float(df.loc[i, "temperature_term"]) if np.isfinite(df.loc[i, "temperature_term"]) else None
    order_term = float(df.loc[i, "order_term"]) if np.isfinite(df.loc[i, "order_term"]) else None
    phase_term = float(df.loc[i, "phase_term"]) if np.isfinite(df.loc[i, "phase_term"]) else None
    switch_term = float(df.loc[i, "switch_term"]) if np.isfinite(df.loc[i, "switch_term"]) else None
    first_signal_date = str(df.loc[i, date_col_end])
    index_regime = str(df.loc[i, "index_regime"])

    if entry_i >= len(df):
        return EntryEval(
            freq,
            first_signal_date,
            None,
            None,
            score,
            score_min_persist,
            energy_term,
            temperature_term,
            order_term,
            phase_term,
            switch_term,
            index_regime,
            None,
            None,
            n_signals,
        )

    entry_price = float(df.loc[entry_i, "open"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return EntryEval(
            freq,
            first_signal_date,
            None,
            None,
            score,
            score_min_persist,
            energy_term,
            temperature_term,
            order_term,
            phase_term,
            switch_term,
            index_regime,
            None,
            None,
            n_signals,
        )

    end_mask = df[date_col_end].astype(str) <= y1
    end_df = df[end_mask].copy().reset_index(drop=True)
    if end_df.empty or entry_i >= len(end_df):
        return EntryEval(
            freq,
            first_signal_date,
            str(df.loc[entry_i, date_col_start]),
            entry_price,
            score,
            score_min_persist,
            energy_term,
            temperature_term,
            order_term,
            phase_term,
            switch_term,
            index_regime,
            None,
            None,
            n_signals,
        )

    end_price = float(end_df.loc[len(end_df) - 1, "close"])
    ret_to_year_end = (end_price / entry_price - 1.0) if np.isfinite(end_price) and end_price > 0 else np.nan
    runup = np.nan
    closes = end_df.loc[entry_i:, "close"].to_numpy(dtype=np.float64)
    if closes.size > 0 and np.isfinite(closes).any():
        runup = float(np.nanmax(closes) / entry_price - 1.0)

    return EntryEval(
        freq=freq,
        first_signal_date=first_signal_date,
        entry_date=str(df.loc[entry_i, date_col_start]),
        entry_price=float(entry_price),
        score=score,
        score_min_persist=score_min_persist,
        energy_term=energy_term,
        temperature_term=temperature_term,
        order_term=order_term,
        phase_term=phase_term,
        switch_term=switch_term,
        index_regime=index_regime,
        ret_to_year_end=float(ret_to_year_end) if np.isfinite(ret_to_year_end) else None,
        max_runup_to_year_end=float(runup) if np.isfinite(runup) else None,
        n_signals=n_signals,
    )


def build_resonance_daily_frame(
    daily_feat: pd.DataFrame,
    weekly_feat: pd.DataFrame,
    monthly_feat: pd.DataFrame,
    daily_threshold: float,
    weekly_threshold: float,
    monthly_threshold: float,
    resonance_threshold: float,
    resonance_min_count: int,
    resonance_persist_days: int,
) -> pd.DataFrame:
    if daily_feat is None or daily_feat.empty:
        return pd.DataFrame()

    base = daily_feat.copy().sort_values("bar_end_dt").reset_index(drop=True)

    def prep_context(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        cols = [
            "bar_end_dt",
            "bar_end",
            "score",
            "score_min_persist",
            "state",
            "energy_term",
            "temperature_term",
            "order_term",
            "phase_term",
            "switch_term",
        ]
        out = df[cols].copy().sort_values("bar_end_dt").reset_index(drop=True)
        rename = {c: f"{prefix}_{c}" for c in cols if c != "bar_end_dt"}
        return out.rename(columns=rename)

    weekly_ctx = prep_context(weekly_feat, "weekly") if weekly_feat is not None and not weekly_feat.empty else pd.DataFrame()
    monthly_ctx = prep_context(monthly_feat, "monthly") if monthly_feat is not None and not monthly_feat.empty else pd.DataFrame()

    if not weekly_ctx.empty:
        base = pd.merge_asof(base, weekly_ctx, on="bar_end_dt", direction="backward")
    if not monthly_ctx.empty:
        base = pd.merge_asof(base, monthly_ctx, on="bar_end_dt", direction="backward")

    base["daily_state"] = pd.Series(base["state"], dtype="boolean").fillna(False).to_numpy(dtype=bool)
    base["weekly_state"] = base.get("weekly_state", False)
    base["monthly_state"] = base.get("monthly_state", False)

    if "weekly_state" in base.columns:
        base["weekly_state"] = pd.Series(base["weekly_state"], dtype="boolean").fillna(False).to_numpy(dtype=bool)
    else:
        base["weekly_state"] = False
    if "monthly_state" in base.columns:
        base["monthly_state"] = pd.Series(base["monthly_state"], dtype="boolean").fillna(False).to_numpy(dtype=bool)
    else:
        base["monthly_state"] = False

    base["support_count"] = (
        base["daily_state"].astype(int) + base["weekly_state"].astype(int) + base["monthly_state"].astype(int)
    )

    base["daily_score_ctx"] = pd.to_numeric(base["score_min_persist"], errors="coerce")
    base["weekly_score_ctx"] = pd.to_numeric(base.get("weekly_score_min_persist"), errors="coerce")
    base["monthly_score_ctx"] = pd.to_numeric(base.get("monthly_score_min_persist"), errors="coerce")

    alignment_bonus = (
        (pd.to_numeric(base["order_term"], errors="coerce") > 0).astype(int)
        + (pd.to_numeric(base.get("weekly_order_term"), errors="coerce") > 0).astype(int)
        + (pd.to_numeric(base.get("monthly_order_term"), errors="coerce") > 0).astype(int)
        + (pd.to_numeric(base["phase_term"], errors="coerce") > 0).astype(int)
        + (pd.to_numeric(base.get("weekly_phase_term"), errors="coerce") > 0).astype(int)
        + (pd.to_numeric(base.get("monthly_phase_term"), errors="coerce") > 0).astype(int)
    ) / 6.0
    base["alignment_bonus"] = np.clip(alignment_bonus, 0.0, 1.0)

    raw_resonance = (
        0.50 * pd.to_numeric(base["daily_score_ctx"], errors="coerce").fillna(-1.0)
        + 0.30 * pd.to_numeric(base["weekly_score_ctx"], errors="coerce").fillna(-1.0)
        + 0.20 * pd.to_numeric(base["monthly_score_ctx"], errors="coerce").fillna(-1.0)
        + 0.10 * base["alignment_bonus"]
    )
    base["resonance_score"] = np.clip(raw_resonance, -1.0, 1.0)

    base["resonance_ready_raw"] = (
        base["daily_state"]
        & (pd.to_numeric(base["daily_score_ctx"], errors="coerce") >= float(daily_threshold))
        & (pd.to_numeric(base["weekly_score_ctx"], errors="coerce") >= float(weekly_threshold))
        & (pd.to_numeric(base["monthly_score_ctx"], errors="coerce") >= float(monthly_threshold))
        & (base["support_count"] >= int(resonance_min_count))
        & (pd.to_numeric(base["resonance_score"], errors="coerce") >= float(resonance_threshold))
    )

    persist = max(1, int(resonance_persist_days))
    if persist <= 1:
        base["resonance_state"] = base["resonance_ready_raw"]
    else:
        base["resonance_state"] = (
            base["resonance_ready_raw"].astype(int).rolling(window=persist, min_periods=persist).min().fillna(0) >= 1
        )

    return base


def eval_first_resonance_in_year(
    daily_resonance: pd.DataFrame,
    index_regime_by_date: dict[str, str] | None,
    year: int,
    allow_index_regimes: set[str] | None,
) -> ResonanceEval:
    if daily_resonance is None or not isinstance(daily_resonance, pd.DataFrame) or daily_resonance.empty:
        return ResonanceEval(None, None, None, None, None, None, None, 0, None, None, None, 0)

    df = daily_resonance.copy().reset_index(drop=True)
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["open", "close"]).reset_index(drop=True)
    if df.empty:
        return ResonanceEval(None, None, None, None, None, None, None, 0, None, None, None, 0)

    y0 = f"{int(year)}0101"
    y1 = f"{int(year)}1231"

    if index_regime_by_date:
        df["index_regime"] = df["bar_end"].astype(str).map(index_regime_by_date).fillna("BASE")
    else:
        df["index_regime"] = "BASE"

    ok = df["resonance_state"].fillna(False).to_numpy(dtype=bool)
    ok = ok & df["bar_end"].astype(str).between(y0, y1).to_numpy()
    if allow_index_regimes:
        ok = ok & df["index_regime"].isin(set(allow_index_regimes)).to_numpy()

    sig_idx = np.where(ok)[0]
    n_signals = int(len(sig_idx))
    if n_signals == 0:
        return ResonanceEval(None, None, None, None, None, None, None, 0, None, None, None, 0)

    i = int(sig_idx[0])
    entry_i = i + 1
    if entry_i >= len(df):
        return ResonanceEval(
            first_signal_date=str(df.loc[i, "bar_end"]),
            entry_date=None,
            entry_price=None,
            daily_score=float(df.loc[i, "daily_score_ctx"]) if np.isfinite(df.loc[i, "daily_score_ctx"]) else None,
            weekly_score=float(df.loc[i, "weekly_score_ctx"]) if np.isfinite(df.loc[i, "weekly_score_ctx"]) else None,
            monthly_score=float(df.loc[i, "monthly_score_ctx"]) if np.isfinite(df.loc[i, "monthly_score_ctx"]) else None,
            resonance_score=float(df.loc[i, "resonance_score"]) if np.isfinite(df.loc[i, "resonance_score"]) else None,
            support_count=int(df.loc[i, "support_count"]),
            index_regime=str(df.loc[i, "index_regime"]),
            ret_to_year_end=None,
            max_runup_to_year_end=None,
            n_signals=n_signals,
        )

    entry_price = float(df.loc[entry_i, "open"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return ResonanceEval(
            first_signal_date=str(df.loc[i, "bar_end"]),
            entry_date=None,
            entry_price=None,
            daily_score=float(df.loc[i, "daily_score_ctx"]) if np.isfinite(df.loc[i, "daily_score_ctx"]) else None,
            weekly_score=float(df.loc[i, "weekly_score_ctx"]) if np.isfinite(df.loc[i, "weekly_score_ctx"]) else None,
            monthly_score=float(df.loc[i, "monthly_score_ctx"]) if np.isfinite(df.loc[i, "monthly_score_ctx"]) else None,
            resonance_score=float(df.loc[i, "resonance_score"]) if np.isfinite(df.loc[i, "resonance_score"]) else None,
            support_count=int(df.loc[i, "support_count"]),
            index_regime=str(df.loc[i, "index_regime"]),
            ret_to_year_end=None,
            max_runup_to_year_end=None,
            n_signals=n_signals,
        )

    end_df = df[df["bar_end"].astype(str) <= y1].copy().reset_index(drop=True)
    if end_df.empty or entry_i >= len(end_df):
        return ResonanceEval(
            first_signal_date=str(df.loc[i, "bar_end"]),
            entry_date=str(df.loc[entry_i, "bar_start"]),
            entry_price=entry_price,
            daily_score=float(df.loc[i, "daily_score_ctx"]) if np.isfinite(df.loc[i, "daily_score_ctx"]) else None,
            weekly_score=float(df.loc[i, "weekly_score_ctx"]) if np.isfinite(df.loc[i, "weekly_score_ctx"]) else None,
            monthly_score=float(df.loc[i, "monthly_score_ctx"]) if np.isfinite(df.loc[i, "monthly_score_ctx"]) else None,
            resonance_score=float(df.loc[i, "resonance_score"]) if np.isfinite(df.loc[i, "resonance_score"]) else None,
            support_count=int(df.loc[i, "support_count"]),
            index_regime=str(df.loc[i, "index_regime"]),
            ret_to_year_end=None,
            max_runup_to_year_end=None,
            n_signals=n_signals,
        )

    end_price = float(end_df.loc[len(end_df) - 1, "close"])
    ret_to_year_end = (end_price / entry_price - 1.0) if np.isfinite(end_price) and end_price > 0 else np.nan
    closes = end_df.loc[entry_i:, "close"].to_numpy(dtype=np.float64)
    runup = float(np.nanmax(closes) / entry_price - 1.0) if closes.size > 0 and np.isfinite(closes).any() else np.nan

    return ResonanceEval(
        first_signal_date=str(df.loc[i, "bar_end"]),
        entry_date=str(df.loc[entry_i, "bar_start"]),
        entry_price=entry_price,
        daily_score=float(df.loc[i, "daily_score_ctx"]) if np.isfinite(df.loc[i, "daily_score_ctx"]) else None,
        weekly_score=float(df.loc[i, "weekly_score_ctx"]) if np.isfinite(df.loc[i, "weekly_score_ctx"]) else None,
        monthly_score=float(df.loc[i, "monthly_score_ctx"]) if np.isfinite(df.loc[i, "monthly_score_ctx"]) else None,
        resonance_score=float(df.loc[i, "resonance_score"]) if np.isfinite(df.loc[i, "resonance_score"]) else None,
        support_count=int(df.loc[i, "support_count"]),
        index_regime=str(df.loc[i, "index_regime"]),
        ret_to_year_end=float(ret_to_year_end) if np.isfinite(ret_to_year_end) else None,
        max_runup_to_year_end=float(runup) if np.isfinite(runup) else None,
        n_signals=n_signals,
    )


def build_latest_signal_snapshot(
    daily_resonance: pd.DataFrame,
    scan_date: str,
    require_exact_date: bool = True,
) -> SignalSnapshot | None:
    if daily_resonance is None or not isinstance(daily_resonance, pd.DataFrame) or daily_resonance.empty:
        return None

    df = daily_resonance.copy().reset_index(drop=True)
    df["bar_end"] = df["bar_end"].astype(str)
    df = df[df["bar_end"] <= str(scan_date)].copy().reset_index(drop=True)
    if df.empty:
        return None

    row = df.iloc[-1]
    if require_exact_date and str(row["bar_end"]) != str(scan_date):
        return None

    return SignalSnapshot(
        signal_date=str(row["bar_end"]),
        daily_bar_end=str(row["bar_end"]),
        weekly_bar_end=str(row["weekly_bar_end"]) if "weekly_bar_end" in row.index and pd.notna(row["weekly_bar_end"]) else None,
        monthly_bar_end=str(row["monthly_bar_end"]) if "monthly_bar_end" in row.index and pd.notna(row["monthly_bar_end"]) else None,
        daily_state=bool(row["daily_state"]) if pd.notna(row["daily_state"]) else False,
        weekly_state=bool(row["weekly_state"]) if pd.notna(row["weekly_state"]) else False,
        monthly_state=bool(row["monthly_state"]) if pd.notna(row["monthly_state"]) else False,
        resonance_state=bool(row["resonance_state"]) if pd.notna(row["resonance_state"]) else False,
        support_count=int(row["support_count"]) if pd.notna(row["support_count"]) else 0,
        daily_score=float(row["daily_score_ctx"]) if np.isfinite(row["daily_score_ctx"]) else None,
        weekly_score=float(row["weekly_score_ctx"]) if np.isfinite(row["weekly_score_ctx"]) else None,
        monthly_score=float(row["monthly_score_ctx"]) if np.isfinite(row["monthly_score_ctx"]) else None,
        resonance_score=float(row["resonance_score"]) if np.isfinite(row["resonance_score"]) else None,
        energy_term=float(row["energy_term"]) if np.isfinite(row["energy_term"]) else None,
        temperature_term=float(row["temperature_term"]) if np.isfinite(row["temperature_term"]) else None,
        order_term=float(row["order_term"]) if np.isfinite(row["order_term"]) else None,
        phase_term=float(row["phase_term"]) if np.isfinite(row["phase_term"]) else None,
        switch_term=float(row["switch_term"]) if np.isfinite(row["switch_term"]) else None,
    )


def compute_year_return_for_file(file_path: str, year: int) -> dict | None:
    try:
        df = pd.read_csv(file_path, usecols=["trade_date", "close", "ts_code"])
    except Exception:
        return None
    if df.empty or "trade_date" not in df.columns or "close" not in df.columns:
        return None

    df["trade_date_str"] = df["trade_date"].astype(str)
    df["dt"] = pd.to_datetime(df["trade_date_str"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    if df.empty:
        return None

    ts_code = str(df["ts_code"].iloc[0]) if "ts_code" in df.columns else None

    y0 = f"{int(year)}0101"
    y1 = f"{int(year)}1231"
    pre = df[df["trade_date_str"] < y0]
    cur = df[(df["trade_date_str"] >= y0) & (df["trade_date_str"] <= y1)]
    if pre.empty or cur.empty:
        return None

    start_close = float(pre.iloc[-1]["close"])
    end_close = float(cur.iloc[-1]["close"])
    if not np.isfinite(start_close) or start_close <= 0 or not np.isfinite(end_close) or end_close <= 0:
        return None

    ret = end_close / start_close - 1.0
    return {
        "symbol": file_path.split("/")[-1].rsplit(".", 1)[0],
        "ts_code": ts_code,
        "year": int(year),
        "start_date": str(pre.iloc[-1]["trade_date_str"]),
        "end_date": str(cur.iloc[-1]["trade_date_str"]),
        "start_close": start_close,
        "end_close": end_close,
        "year_return": float(ret),
        "file_path": file_path,
    }