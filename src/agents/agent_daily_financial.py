"""Agent 2: Daily + Financial data — incremental daily klines, adj_factor, financials."""

import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "downloader"))

from base_agent import BaseAgent


class DailyFinancialAgent(BaseAgent):
    name = "daily_financial"
    description = "增量同步日线行情 + 财务数据"

    def run(self, **kwargs):
        import tushare as ts
        from fast_sync_tushare_latest import (
            RateLimiter,
            build_last_date_map,
            ensure_trade_cal,
            infer_last_available_trade_date,
            update_daily_full,
            update_dividend,
            update_financial_dataset,
            update_stock_basic_and_list,
            update_trade_date_dataset,
            DATE_BATCH_CONFIG,
            FINANCIAL_DATASETS,
        )

        batch_rate = kwargs.get("batch_rate", 240)
        financial_rate = kwargs.get("financial_rate", 360)
        financial_threads = kwargs.get("financial_threads", 16)
        backfill_days = kwargs.get("backfill_open_days", 0)
        skip_financials = kwargs.get("skip_financials", False)

        ts.set_token(self.token)
        pro = ts.pro_api()

        # Phase 1: stock list
        self.update_progress("stock_list")
        symbols = update_stock_basic_and_list(pro, self.data_dir)

        today = datetime.today().strftime("%Y%m%d")
        last_available = infer_last_available_trade_date(pro, today)

        # Phase 2: trade cal + daily
        self.update_progress("trade_cal")
        ensure_trade_cal(pro, self.data_dir)

        trade_dates: list[str] = []
        if backfill_days > 0:
            cal = pro.trade_cal(exchange="SSE", start_date="19900101", end_date=last_available)
            if cal is not None and not cal.empty:
                open_dates = cal.loc[cal["is_open"].astype(int) == 1, "cal_date"].astype(str).sort_values().tolist()
                trade_dates = open_dates[-backfill_days:]
        else:
            daily_dir = os.path.join(self.data_dir, DATE_BATCH_CONFIG["daily_full"]["dir_name"])
            daily_last_map = build_last_date_map(daily_dir, "trade_date")
            current_max = max(daily_last_map.values()) if daily_last_map else "19900101"
            start_date = (datetime.strptime(current_max, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
            if start_date <= last_available:
                cal = pro.trade_cal(exchange="SSE", start_date=start_date, end_date=last_available)
                if cal is not None and not cal.empty:
                    trade_dates = cal.loc[cal["is_open"].astype(int) == 1, "cal_date"].astype(str).sort_values().tolist()

        batch_limiter = RateLimiter(batch_rate, 60)

        self.update_progress("daily_full", 0, len(trade_dates))
        update_daily_full(pro, self.data_dir, trade_dates, batch_limiter)
        self.update_progress("daily_full", len(trade_dates), len(trade_dates))

        # Phase 3: adj_factor, stk_limit, suspend_d
        for ds in ["adj_factor", "stk_limit", "suspend_d"]:
            self.update_progress(ds)
            update_trade_date_dataset(pro, self.data_dir, ds, trade_dates, batch_limiter)

        # Phase 4: dividend
        self.update_progress("dividend")
        dividend_dir = os.path.join(self.data_dir, DATE_BATCH_CONFIG["dividend"]["dir_name"])
        dividend_last_map = build_last_date_map(dividend_dir, "ann_date")
        current_div_max = max(dividend_last_map.values()) if dividend_last_map else "19900101"
        start_ann = (datetime.strptime(current_div_max, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
        update_dividend(pro, self.data_dir, start_ann, today, batch_limiter)

        # Phase 5: financials
        if not skip_financials:
            financial_limiter = RateLimiter(financial_rate, 60)
            fin_start = (datetime.strptime(today, "%Y%m%d") - timedelta(days=20)).strftime("%Y%m%d")
            fin_datasets = [d for d in FINANCIAL_DATASETS]
            for i, ds in enumerate(fin_datasets):
                self.update_progress(f"financial_{ds}", i + 1, len(fin_datasets))
                update_financial_dataset(pro, self.data_dir, ds, symbols, fin_start, today, financial_limiter, financial_threads)
            self.update_progress("financial_done", len(fin_datasets), len(fin_datasets))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--token", default="")
    p.add_argument("--batch-rate", type=int, default=240)
    p.add_argument("--financial-rate", type=int, default=360)
    p.add_argument("--financial-threads", type=int, default=16)
    p.add_argument("--backfill-open-days", type=int, default=0)
    p.add_argument("--skip-financials", action="store_true")
    args = p.parse_args()
    agent = DailyFinancialAgent(args.data_dir, args.token)
    ok = agent.execute(
        batch_rate=args.batch_rate,
        financial_rate=args.financial_rate,
        financial_threads=args.financial_threads,
        backfill_open_days=args.backfill_open_days,
        skip_financials=args.skip_financials,
    )
    sys.exit(0 if ok else 1)
