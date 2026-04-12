"""Agent 1: Stock list sync — downloads tushare_stock_basic.csv and tushare_gplist.json."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "downloader"))

from base_agent import BaseAgent


class StockListAgent(BaseAgent):
    name = "stock_list"
    description = "同步股票列表 (stock_basic + gplist)"

    def run(self, **kwargs):
        import tushare as ts
        from fast_sync_tushare_latest import update_stock_basic_and_list

        ts.set_token(self.token)
        pro = ts.pro_api()

        self.update_progress("sync_stock_list")
        symbols = update_stock_basic_and_list(pro, self.data_dir)
        self.update_progress("done", len(symbols), len(symbols))
        self.state.set_success(0, {"files_written": 2, "rows_written": len(symbols)})


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--token", default="")
    args = p.parse_args()
    agent = StockListAgent(args.data_dir, args.token)
    ok = agent.execute()
    sys.exit(0 if ok else 1)
