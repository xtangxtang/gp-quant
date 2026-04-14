"""
大盘趋势判断策略 - CLI 入口

用法:
  python -m src.strategy.market_trend.run_market_trend \
    --data_dir /path/to/tushare-daily-full \
    --index_dir /path/to/tushare-index-daily \
    --out_dir ./results/market_trend \
    --start_date 20240101 --end_date 20260410
"""

import argparse
import sys
from pathlib import Path

if __package__:
    from .config import MarketTrendConfig
    from .trend_engine import run_market_trend_scan
else:
    _PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from src.strategy.market_trend.config import MarketTrendConfig
    from src.strategy.market_trend.trend_engine import run_market_trend_scan


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="大盘趋势判断 - 从小见大")

    # 数据路径
    p.add_argument("--data_dir", required=True,
                   help="tushare-daily-full 目录")
    p.add_argument("--index_dir", default="",
                   help="tushare-index-daily 目录")
    p.add_argument("--stk_limit_dir", default="",
                   help="tushare-stk_limit 目录")
    p.add_argument("--margin_path", default="",
                   help="tushare-margin/margin.csv 路径")
    p.add_argument("--shibor_path", default="",
                   help="tushare-shibor/shibor.csv 路径")
    p.add_argument("--index_member_path", default="",
                   help="tushare-index_member_all/index_member_all.csv 路径")
    p.add_argument("--basic_path", default="",
                   help="tushare_stock_basic.csv 路径")
    p.add_argument("--out_dir", required=True,
                   help="输出目录")

    # 扫描范围
    p.add_argument("--start_date", default="",
                   help="开始日期 YYYYMMDD")
    p.add_argument("--end_date", default="",
                   help="结束日期 YYYYMMDD")
    p.add_argument("--index_code", default="000001_sh",
                   help="指数代码 (默认: 上证综指)")

    # 微观参数
    p.add_argument("--ma_short", type=int, default=20)
    p.add_argument("--ma_long", type=int, default=60)
    p.add_argument("--entropy_order", type=int, default=3)
    p.add_argument("--entropy_window", type=int, default=20)
    p.add_argument("--momentum_window", type=int, default=20)

    # 趋势判定
    p.add_argument("--strong_up_threshold", type=float, default=0.5)
    p.add_argument("--up_threshold", type=float, default=0.2)
    p.add_argument("--down_threshold", type=float, default=-0.2)
    p.add_argument("--strong_down_threshold", type=float, default=-0.5)

    # 性能
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--min_bars", type=int, default=60)

    # 报告
    p.add_argument("--report", action="store_true",
                   help="生成每日诊断报告 (Markdown)")
    p.add_argument("--report_date", default="",
                   help="报告日期 YYYYMMDD (空=最后一个交易日)")

    return p


def main() -> None:
    args = _build_parser().parse_args()

    cfg = MarketTrendConfig(
        data_dir=args.data_dir,
        index_dir=args.index_dir,
        stk_limit_dir=args.stk_limit_dir,
        margin_path=args.margin_path,
        shibor_path=args.shibor_path,
        index_member_path=args.index_member_path,
        basic_path=args.basic_path,
        out_dir=args.out_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        index_code=args.index_code,
        ma_short=args.ma_short,
        ma_long=args.ma_long,
        entropy_order=args.entropy_order,
        entropy_window=args.entropy_window,
        momentum_window=args.momentum_window,
        strong_up_threshold=args.strong_up_threshold,
        up_threshold=args.up_threshold,
        down_threshold=args.down_threshold,
        strong_down_threshold=args.strong_down_threshold,
        workers=args.workers,
        min_bars=args.min_bars,
        report=args.report,
        report_date=args.report_date,
    )

    run_market_trend_scan(cfg)


if __name__ == "__main__":
    main()
