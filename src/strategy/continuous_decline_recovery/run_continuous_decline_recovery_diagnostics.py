import argparse
import sys
from pathlib import Path

if __package__:
    from .continuous_decline_recovery_diagnostics import run_continuous_decline_recovery_diagnostics
    from .continuous_decline_recovery_scan_service import ContinuousDeclineRecoveryConfig
else:
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.strategy.continuous_decline_recovery.continuous_decline_recovery_diagnostics import (
        run_continuous_decline_recovery_diagnostics,
    )
    from src.strategy.continuous_decline_recovery.continuous_decline_recovery_scan_service import (
        ContinuousDeclineRecoveryConfig,
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--scan_date", type=str, default="")
    parser.add_argument("--top_n", type=int, default=30)
    parser.add_argument("--top_sectors", type=int, default=6)
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument("--basic_path", type=str, default="")
    parser.add_argument("--lookback_years", type=int, default=4)
    parser.add_argument("--min_amount", type=float, default=600000.0)
    parser.add_argument("--min_turnover", type=float, default=1.0)
    parser.add_argument("--exclude_st", action="store_true", default=True)
    parser.add_argument("--include_st", action="store_false", dest="exclude_st")
    parser.add_argument("--market_window", type=int, default=6)
    parser.add_argument("--min_sector_members", type=int, default=4)
    parser.add_argument("--min_rebound_from_low", type=float, default=0.03)
    parser.add_argument("--max_rebound_from_low", type=float, default=0.15)
    parser.add_argument("--backtest_start_date", type=str, default="")
    parser.add_argument("--backtest_end_date", type=str, default="")
    parser.add_argument("--hold_days", type=int, default=8)
    parser.add_argument("--max_positions", type=int, default=10)
    parser.add_argument("--max_positions_per_industry", type=int, default=2)
    parser.add_argument("--min_near_miss_strategy_score", type=float, default=0.45)
    parser.add_argument("--top_sample_count", type=int, default=100)
    return parser


def main() -> None:
    args = _build_argument_parser().parse_args()
    arg_dict = vars(args).copy()
    min_near_miss_strategy_score = float(arg_dict.pop("min_near_miss_strategy_score"))
    top_sample_count = int(arg_dict.pop("top_sample_count"))
    config = ContinuousDeclineRecoveryConfig(**arg_dict)
    output_paths = run_continuous_decline_recovery_diagnostics(
        config,
        out_dir=config.out_dir,
        min_near_miss_strategy_score=min_near_miss_strategy_score,
        top_sample_count=top_sample_count,
    )
    print("Wrote:")
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()