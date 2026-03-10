import argparse
import sys
from pathlib import Path

if __package__:
    from .multitimeframe_scan_service import ScanConfig, run_multitimeframe_scan
else:
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from src.analysis.multitimeframe_scan_service import ScanConfig, run_multitimeframe_scan


def _build_argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--test_year", type=int, default=2025)
    p.add_argument("--top_n", type=int, default=300)
    p.add_argument("--symbols", type=str, default="")
    p.add_argument("--index_path", type=str, default="")
    p.add_argument("--basic_path", type=str, default="")

    p.add_argument("--entry_threshold", type=float, default=0.18)
    p.add_argument("--persist_bars", type=int, default=3)
    p.add_argument("--energy_min", type=float, default=-0.10)
    p.add_argument("--order_min", type=float, default=0.05)
    p.add_argument("--phase_min", type=float, default=0.00)
    p.add_argument("--gate_index", action="store_true")

    p.add_argument("--daily_ws", type=int, default=20)
    p.add_argument("--daily_wl", type=int, default=60)
    p.add_argument("--weekly_ws", type=int, default=12)
    p.add_argument("--weekly_wl", type=int, default=36)
    p.add_argument("--monthly_ws", type=int, default=12)
    p.add_argument("--monthly_wl", type=int, default=36)

    p.add_argument("--resonance_threshold", type=float, default=0.22)
    p.add_argument("--resonance_min_count", type=int, default=3)
    p.add_argument("--resonance_persist_days", type=int, default=2)
    p.add_argument("--weekly_support_threshold", type=float, default=0.10)
    p.add_argument("--monthly_support_threshold", type=float, default=0.08)
    return p


def main() -> None:
    args = _build_argument_parser().parse_args()
    config = ScanConfig(**vars(args))
    output_paths = run_multitimeframe_scan(config)
    print("Wrote:")
    for path in output_paths:
        print(" -", path)


if __name__ == "__main__":
    main()
