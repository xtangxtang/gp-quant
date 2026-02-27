import csv
import os
from typing import Iterable


def write_rows_csv(path: str, fieldnames: list[str], rows: Iterable[dict]) -> None:
    """Write rows to CSV with a stable schema.

    - Always writes header.
    - Ensures column order matches fieldnames.
    - Converts None to empty string.
    """

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def _normalize_row(r: dict) -> dict:
        out: dict = {}
        for k in fieldnames:
            v = r.get(k)
            out[k] = "" if v is None else v
        return out

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            if not isinstance(r, dict):
                continue
            w.writerow(_normalize_row(r))
