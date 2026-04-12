"""Agent 3: Market data — moneyflow, margin, block_trade, HK holdings, indices, etc."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "downloader"))

from base_agent import BaseAgent

# Categories from download_all_tushare.py that belong to "market data"
MARKET_CATEGORIES = ["stock", "market", "index", "fund", "futures", "bond", "macro", "fx", "hk", "option"]


class MarketDataAgent(BaseAgent):
    name = "market_data"
    description = "全量同步资金流、指数、两融、大宗交易等市场数据"

    def run(self, **kwargs):
        import tushare as ts
        from download_all_tushare import main as download_main

        categories = kwargs.get("categories", MARKET_CATEGORIES)
        rate = kwargs.get("rate", 180)
        threads = kwargs.get("threads", 4)

        # download_all_tushare.main() uses argparse — we build sys.argv
        argv_backup = sys.argv
        try:
            for i, cat in enumerate(categories):
                self.update_progress(f"category_{cat}", i, len(categories))
                sys.argv = [
                    "download_all_tushare.py",
                    "-o", self.data_dir,
                    "--token", self.token,
                    "--threads", str(threads),
                    "--rate", str(rate),
                    "--category", cat,
                ]
                try:
                    download_main()
                except SystemExit:
                    pass
            self.update_progress("done", len(categories), len(categories))
        finally:
            sys.argv = argv_backup


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--token", default="")
    p.add_argument("--rate", type=int, default=180)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--categories", default=",".join(MARKET_CATEGORIES),
                    help="Comma-separated categories to download")
    args = p.parse_args()
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    agent = MarketDataAgent(args.data_dir, args.token)
    ok = agent.execute(categories=cats, rate=args.rate, threads=args.threads)
    sys.exit(0 if ok else 1)
