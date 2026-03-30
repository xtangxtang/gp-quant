import re


def normalize_trade_date(value: object, allow_empty: bool = True) -> str:
    text = str(value or "").strip()
    if not text:
        if allow_empty:
            return ""
        raise ValueError("trade date is required")
    digits = re.sub(r"\D", "", text)
    if len(digits) != 8:
        raise ValueError(f"Invalid trade date format: {text}")
    return digits


def safe_file_component(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    return re.sub(r"[^0-9A-Za-z._-]+", "_", text)