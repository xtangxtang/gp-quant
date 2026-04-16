"""Agent: 熵惜售分岔突破策略 — 每日扫描

依赖: derived (周线), market_data (资金流/指数), daily_financial (日线)
在 supervisor DAG 中位于 market_trend 同级或之后.
"""

import os
import sys
from datetime import datetime

# 项目根目录 (agents/ → src/ → project root)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

sys.path.insert(0, os.path.dirname(__file__))
from base_agent import BaseAgent


class EntropyScanAgent(BaseAgent):
    name = "entropy_scan"
    description = "熵惜售分岔突破策略每日扫描"

    def run(self, **kwargs):
        from src.strategy.entropy_accumulation_breakout.scan_service import (
            ScanConfig,
            run_scan,
            write_results,
        )

        data_dir = self.data_dir
        daily_dir = os.path.join(data_dir, "tushare-daily-full")
        basic_path = os.path.join(data_dir, "tushare_stock_basic.csv")
        cache_dir = os.path.join(data_dir, "feature-cache")

        # 自动检测最新交易日 (从数据文件推断, 而非用系统日期)
        scan_date = self._detect_latest_trade_date(daily_dir)

        # 项目根目录 (supervisor.py 所在目录的上两级)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        out_dir = os.path.join(project_root, "results", "entropy_accumulation_breakout")

        self.state.update_progress("配置", 0, 0)
        print(f"[{self.name}] scan_date={scan_date}, cache_dir={cache_dir}")

        cfg = ScanConfig(
            data_dir=daily_dir,
            data_root=data_dir,
            basic_path=basic_path if os.path.exists(basic_path) else "",
            out_dir=out_dir,
            scan_date=scan_date,
            feature_cache_dir=cache_dir,
        )

        self.state.update_progress("扫描", 0, 1)
        all_result, top_picks = run_scan(cfg)

        n_total = len(all_result) if len(all_result) > 0 else 0
        n_breakout = len(top_picks) if len(top_picks) > 0 else 0

        self.state.update_progress("写出", 1, 1)
        write_results(out_dir, scan_date, all_result, top_picks)

        print(f"[{self.name}] 扫描日期: {scan_date}, 有效股票: {n_total}, 突破候选: {n_breakout}")

    @staticmethod
    def _detect_latest_trade_date(daily_dir: str) -> str:
        """从日线 CSV 文件推断最新交易日."""
        import pandas as pd
        # 抽样几只大盘股
        for sym in ("sh600000", "sz000001", "sh601318", "sh600519"):
            fpath = os.path.join(daily_dir, f"{sym}.csv")
            if os.path.exists(fpath):
                try:
                    df = pd.read_csv(fpath, usecols=["trade_date"])
                    return str(df["trade_date"].max())
                except Exception:
                    continue
        # fallback: 今天
        return datetime.now().strftime("%Y%m%d")
