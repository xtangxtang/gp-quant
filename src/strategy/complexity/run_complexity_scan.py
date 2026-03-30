import argparse
import sys
from pathlib import Path

if __package__:
    from .complexity_scan_service import ComplexityScanConfig, run_complexity_scan
    from .complexity_signal_models import VALID_STRATEGIES
else:
    _PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from src.strategy.complexity.complexity_scan_service import ComplexityScanConfig, run_complexity_scan
    from src.strategy.complexity.complexity_signal_models import VALID_STRATEGIES


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy_name", required=True, choices=sorted(VALID_STRATEGIES))
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--scan_date", type=str, default="")
    parser.add_argument("--top_n", type=int, default=30)
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument("--basic_path", type=str, default="")
    parser.add_argument("--lookback_years", type=int, default=5)
    parser.add_argument("--min_amount", type=float, default=500000.0)
    parser.add_argument("--min_turnover", type=float, default=1.0)
    parser.add_argument("--exclude_st", action="store_true", default=True)
    parser.add_argument("--include_st", action="store_false", dest="exclude_st")
    parser.add_argument("--backtest_start_date", type=str, default="")
    parser.add_argument("--backtest_end_date", type=str, default="")
    parser.add_argument("--hold_days", type=int, default=5)
    parser.add_argument("--max_positions", type=int, default=10)
    parser.add_argument("--max_positions_per_industry", type=int, default=2)
    return parser


def main() -> None:
    args = _build_argument_parser().parse_args()
    config = ComplexityScanConfig(**vars(args))
    output_paths = run_complexity_scan(config)
    print("Wrote:")
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()