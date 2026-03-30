import argparse
import sys
from pathlib import Path

if __package__:
    from .entropy_hold_scan_service import EntropyHoldConfig, run_entropy_hold_scan
else:
    _PROJECT_ROOT = Path(__file__).resolve().parents[4]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from src.strategy.uptrend_hold_state_flow.entropy_hold_judgement.entropy_hold_scan_service import EntropyHoldConfig, run_entropy_hold_scan


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--symbol_or_name", required=True)
    parser.add_argument("--start_date", required=True)
    parser.add_argument("--scan_date", type=str, default="")
    parser.add_argument("--basic_path", type=str, default="")
    parser.add_argument("--lookback_years", type=int, default=5)
    parser.add_argument("--exit_persist_days", type=int, default=3)
    return parser


def main() -> None:
    args = _build_argument_parser().parse_args()
    config = EntropyHoldConfig(**vars(args))
    output_paths = run_entropy_hold_scan(config)
    print("Wrote:")
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()