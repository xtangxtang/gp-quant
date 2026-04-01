import difflib
import os

import pandas as pd

try:
    from pypinyin import lazy_pinyin
except ImportError:
    lazy_pinyin = None


def build_symbol_from_ts_code(ts_code: str) -> str:
    if not ts_code or "." not in ts_code:
        return str(ts_code).lower()
    code, exch = ts_code.split(".", 1)
    return f"{exch.lower()}{code}"


def infer_ts_code_from_numeric(code: str) -> str:
    c = str(code).strip()
    if c.startswith("92"):
        return f"{c}.BJ"
    if c.startswith(("6", "9")):
        return f"{c}.SH"
    return f"{c}.SZ"


def load_basic_frame(basic_path: str) -> pd.DataFrame:
    if not basic_path or not os.path.exists(basic_path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(basic_path, usecols=["ts_code", "symbol", "name", "area", "industry", "market"])
        return df.fillna("") if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def resolve_target(symbol_or_name: str, data_dir: str, basic_path: str) -> dict[str, str]:
    raw = str(symbol_or_name or "").strip()
    if not raw:
        raise ValueError("symbol_or_name is required")

    basic_df = load_basic_frame(basic_path)
    lower = raw.lower()
    ts_code = ""
    symbol = ""
    name = ""
    area = ""
    industry = ""
    market = ""

    if lower.startswith(("sh", "sz", "bj")):
        symbol = lower
        ts_code = infer_ts_code_from_numeric(lower[2:]) if "." not in lower else lower.upper()
    elif raw.endswith((".SH", ".SZ", ".BJ", ".sh", ".sz", ".bj")):
        ts_code = raw.upper()
        symbol = build_symbol_from_ts_code(ts_code)
    elif raw.isdigit():
        ts_code = infer_ts_code_from_numeric(raw)
        symbol = build_symbol_from_ts_code(ts_code)
    else:
        if basic_df.empty:
            raise ValueError("basic_path is required when resolving by stock name")
        row = _resolve_name_row(raw, basic_df, data_dir)
        ts_code = str(row.get("ts_code") or "")
        symbol = build_symbol_from_ts_code(ts_code)
        name = str(row.get("name") or "")
        area = str(row.get("area") or "")
        industry = str(row.get("industry") or "")
        market = str(row.get("market") or "")

    if not basic_df.empty and ts_code:
        matched = basic_df[basic_df["ts_code"].astype(str) == ts_code]
        if not matched.empty:
            row = matched.iloc[0]
            name = str(row.get("name") or name)
            area = str(row.get("area") or area)
            industry = str(row.get("industry") or industry)
            market = str(row.get("market") or market)

    file_path = os.path.join(data_dir, f"{symbol}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV not found for {raw}: {file_path}")

    return {
        "symbol": symbol,
        "ts_code": ts_code,
        "name": name,
        "area": area,
        "industry": industry,
        "market": market,
        "file_path": file_path,
    }


def _resolve_name_row(raw: str, basic_df: pd.DataFrame, data_dir: str) -> pd.Series:
    names = basic_df["name"].astype(str).str.strip()

    exact = basic_df[names == raw]
    if not exact.empty:
        return _select_row(exact, raw, data_dir)

    partial = basic_df[names.str.contains(raw, regex=False)]
    if not partial.empty:
        return _select_row(partial, raw, data_dir)

    pinyin_row = _resolve_pinyin_row(raw, basic_df, data_dir)
    if pinyin_row is not None:
        return pinyin_row

    message = f"Unable to resolve stock name: {raw}"
    suggestions = _build_suggestions(raw, basic_df)
    if suggestions:
        message = f"{message}. Did you mean: {suggestions}?"
    raise ValueError(message)


def _resolve_pinyin_row(raw: str, basic_df: pd.DataFrame, data_dir: str) -> pd.Series | None:
    raw_key = _name_pinyin_key(raw)
    if not raw_key:
        return None

    names = basic_df["name"].astype(str).str.strip()
    matched_names = sorted({name for name in names if name and _name_pinyin_key(name) == raw_key})
    if len(matched_names) != 1:
        return None
    return _select_row(basic_df[names == matched_names[0]], raw, data_dir)


def _select_row(matches: pd.DataFrame, raw: str, data_dir: str) -> pd.Series:
    if len(matches) == 1:
        return matches.iloc[0]

    local_matches = _filter_matches_with_local_data(matches, data_dir)
    if len(local_matches) == 1:
        return local_matches.iloc[0]

    candidate_frame = local_matches if not local_matches.empty else matches
    candidates = _format_candidates(candidate_frame)
    raise ValueError(
        f"Stock name '{raw}' matched multiple candidates: {candidates}. "
        "Please use ts_code or market-prefixed symbol."
    )


def _filter_matches_with_local_data(matches: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    if matches.empty:
        return matches
    filtered = matches.copy()
    filtered["_symbol"] = filtered["ts_code"].astype(str).map(build_symbol_from_ts_code)
    filtered["_file_path"] = filtered["_symbol"].map(lambda symbol: os.path.join(data_dir, f"{symbol}.csv"))
    filtered = filtered[filtered["_file_path"].map(os.path.exists)].copy()
    return filtered.drop(columns=["_symbol", "_file_path"], errors="ignore")


def _build_suggestions(raw: str, basic_df: pd.DataFrame, limit: int = 3) -> str:
    name_to_code: dict[str, str] = {}
    for _, row in basic_df.iterrows():
        name = str(row.get("name") or "").strip()
        ts_code = str(row.get("ts_code") or "").strip()
        if name and name not in name_to_code:
            name_to_code[name] = ts_code

    if not name_to_code:
        return ""

    suggestions: list[str] = []
    raw_key = _name_pinyin_key(raw)
    if raw_key:
        for name in sorted(name_to_code):
            if _name_pinyin_key(name) == raw_key:
                suggestions.append(name)

    close_matches = difflib.get_close_matches(raw, list(name_to_code), n=limit * 2, cutoff=0.5)
    for name in close_matches:
        if name not in suggestions:
            suggestions.append(name)

    formatted = [f"{name} ({name_to_code[name]})" for name in suggestions[:limit]]
    return ", ".join(formatted)


def _format_candidates(matches: pd.DataFrame, limit: int = 5) -> str:
    candidates: list[str] = []
    for _, row in matches.head(limit).iterrows():
        name = str(row.get("name") or "").strip()
        ts_code = str(row.get("ts_code") or "").strip()
        if name or ts_code:
            candidates.append(f"{name} ({ts_code})")
    return ", ".join(candidates)


def _name_pinyin_key(name: str) -> str:
    if lazy_pinyin is None:
        return ""
    text = str(name or "").strip()
    if not text:
        return ""
    return "".join(lazy_pinyin(text))