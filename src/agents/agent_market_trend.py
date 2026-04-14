"""Agent 6: Market Trend — 每日大盘趋势判断（在所有数据下载完成后运行）。"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from base_agent import BaseAgent


DATA_ROOT_DEFAULT = "/nvme5/xtang/gp-workspace/gp-data"
OUT_DIR_DEFAULT = os.path.join(
    os.path.dirname(__file__), "..", "..", "results", "market_trend"
)


class MarketTrendAgent(BaseAgent):
    name = "market_trend"
    description = "每日大盘趋势判断（从小见大 7 维度评分）"

    def run(self, **kwargs):
        from strategy.market_trend.config import MarketTrendConfig
        from strategy.market_trend.trend_engine import run_market_trend_scan

        data_root = self.data_dir
        out_dir = kwargs.get("out_dir", OUT_DIR_DEFAULT)
        out_dir = os.path.abspath(out_dir)

        # 默认只跑最近 120 个交易日（足够预热指标），减少每日开销
        lookback_days = kwargs.get("lookback_days", 120)
        today = datetime.now().strftime("%Y%m%d")

        # 计算 start_date: 往前推 lookback_days 自然日 * 1.5 覆盖交易日
        from datetime import timedelta
        start_dt = datetime.now() - timedelta(days=int(lookback_days * 1.5))
        start_date = kwargs.get("start_date", start_dt.strftime("%Y%m%d"))
        end_date = kwargs.get("end_date", today)

        self.update_progress("building_config", 0, 4)

        cfg = MarketTrendConfig(
            data_dir=os.path.join(data_root, "tushare-daily-full"),
            index_dir=os.path.join(data_root, "tushare-index-daily"),
            stk_limit_dir=os.path.join(data_root, "tushare-stk_limit"),
            margin_path=os.path.join(data_root, "tushare-margin", "margin.csv"),
            shibor_path=os.path.join(data_root, "tushare-shibor", "shibor.csv"),
            index_member_path=os.path.join(
                data_root, "tushare-index_member_all", "index_member_all.csv"
            ),
            basic_path=os.path.join(data_root, "tushare_stock_basic.csv"),
            out_dir=out_dir,
            start_date=start_date,
            end_date=end_date,
            workers=8,
            report=True,
        )

        self.update_progress("loading_data", 1, 4)
        os.makedirs(out_dir, exist_ok=True)

        self.update_progress("scanning", 2, 4)
        results = run_market_trend_scan(cfg)

        self.update_progress("done", 4, 4)

        stats = {"days_scanned": len(results)}
        if results:
            last = results[-1]
            stats["latest_date"] = last.date
            stats["latest_trend"] = last.trend
            stats["latest_score"] = last.composite_score
            print(f"[market_trend] 最新趋势: {last.date} → {last.trend} "
                  f"(score={last.composite_score:.3f})")

        self.state.set_success(0, stats)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="gp-data root dir")
    p.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    p.add_argument("--lookback-days", type=int, default=120)
    p.add_argument("--start-date", default="")
    p.add_argument("--end-date", default="")
    args = p.parse_args()

    agent = MarketTrendAgent(args.data_dir)
    extra = {"out_dir": args.out_dir, "lookback_days": args.lookback_days}
    if args.start_date:
        extra["start_date"] = args.start_date
    if args.end_date:
        extra["end_date"] = args.end_date
    ok = agent.execute(**extra)
    sys.exit(0 if ok else 1)
