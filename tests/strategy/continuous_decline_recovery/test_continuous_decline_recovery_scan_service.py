import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.strategy.continuous_decline_recovery.continuous_decline_recovery_scan_service import (
    ContinuousDeclineRecoveryConfig,
    run_continuous_decline_recovery_scan,
)


class ContinuousDeclineRecoveryScanServiceTests(unittest.TestCase):
    def test_scan_service_identifies_leading_sector_after_selloff(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            data_dir, out_dir, basic_path, scan_date, _ = self._write_fixture(temp_path)

            config = ContinuousDeclineRecoveryConfig(
                data_dir=str(data_dir),
                out_dir=str(out_dir),
                scan_date=scan_date,
                basic_path=str(basic_path),
                top_n=10,
                top_sectors=1,
                min_sector_members=3,
                min_amount=600000.0,
                min_turnover=1.0,
                max_positions=2,
                max_positions_per_industry=2,
                min_rebound_from_low=0.02,
                max_rebound_from_low=0.20,
            )

            output_paths = run_continuous_decline_recovery_scan(config)
            for path in output_paths:
                self.assertTrue(Path(path).exists(), path)

            sector_path = out_dir / f"sector_ranking_{config.strategy_name}_{scan_date}.csv"
            summary_path = out_dir / f"strategy_summary_{config.strategy_name}_{scan_date}.csv"
            selected_path = out_dir / f"selected_portfolio_{config.strategy_name}_{scan_date}_top{config.top_n}.csv"

            sectors = pd.read_csv(sector_path)
            summary = pd.read_csv(summary_path).iloc[0]
            selected = pd.read_csv(selected_path)

            self.assertFalse(sectors.empty)
            self.assertEqual(str(sectors.iloc[0]["industry"]), "半导体")
            self.assertIn(str(summary["market_buy_state"]), {"repair_watch", "buy_window"})
            self.assertFalse(selected.empty)
            self.assertTrue((selected["industry"] == "半导体").all())

    def test_scan_service_runs_forward_backtest_and_writes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            data_dir, out_dir, basic_path, scan_date, dates = self._write_fixture(temp_path)
            backtest_start_date = str(dates[94].strftime("%Y%m%d"))
            backtest_end_date = str(dates[96].strftime("%Y%m%d"))

            config = ContinuousDeclineRecoveryConfig(
                data_dir=str(data_dir),
                out_dir=str(out_dir),
                scan_date=scan_date,
                basic_path=str(basic_path),
                top_n=10,
                top_sectors=1,
                min_sector_members=3,
                min_amount=600000.0,
                min_turnover=1.0,
                max_positions=2,
                max_positions_per_industry=2,
                min_rebound_from_low=0.02,
                max_rebound_from_low=0.20,
                hold_days=5,
                backtest_start_date=backtest_start_date,
                backtest_end_date=backtest_end_date,
            )

            output_paths = run_continuous_decline_recovery_scan(config)
            for path in output_paths:
                self.assertTrue(Path(path).exists(), path)

            daily_path = out_dir / f"forward_backtest_daily_{config.strategy_name}_{scan_date}.csv"
            trades_path = out_dir / f"forward_backtest_trades_{config.strategy_name}_{scan_date}.csv"
            summary_path = out_dir / f"forward_backtest_summary_{config.strategy_name}_{scan_date}.csv"
            yearly_path = out_dir / f"forward_backtest_yearly_performance_{config.strategy_name}_{scan_date}.csv"
            interval_path = out_dir / f"forward_backtest_interval_performance_{config.strategy_name}_{scan_date}.csv"

            daily = pd.read_csv(daily_path)
            trades = pd.read_csv(trades_path)
            summary = pd.read_csv(summary_path).iloc[0]
            yearly = pd.read_csv(yearly_path)
            interval = pd.read_csv(interval_path)

            self.assertFalse(daily.empty)
            self.assertFalse(trades.empty)
            self.assertFalse(yearly.empty)
            self.assertFalse(interval.empty)
            self.assertIn("nav", daily.columns)
            self.assertIn("return_pct", trades.columns)
            self.assertIn("period_label", yearly.columns)
            self.assertIn("total_return_pct", yearly.columns)
            self.assertIn("period_label", interval.columns)
            self.assertIn("annualized_return_pct", interval.columns)
            self.assertEqual(str(summary["status"]), "completed")
            self.assertGreater(int(summary["n_trades"]), 0)
            self.assertGreater(float(summary["final_nav"]), 0.0)
            self.assertIn("max_drawdown_pct", summary.index)
            self.assertTrue((daily["nav"] > 0.0).all())

    @staticmethod
    def _build_symbol(ts_code: str) -> str:
        code, exch = ts_code.split(".", 1)
        return f"{exch.lower()}{code}"

    def _write_fixture(self, temp_path: Path) -> tuple[Path, Path, Path, str, pd.DatetimeIndex]:
        data_dir = temp_path / "daily"
        out_dir = temp_path / "out"
        data_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        dates = pd.bdate_range("2024-01-02", periods=110)
        scan_date = str(dates[96].strftime("%Y%m%d"))

        basic_rows: list[dict[str, str]] = []
        specs = [
            ("600101.SH", "半导体", 0.0110, 0.070, 2.30, 0.10),
            ("600102.SH", "半导体", 0.0106, 0.065, 2.10, 0.25),
            ("600103.SH", "半导体", 0.0102, 0.062, 2.00, 0.40),
            ("600201.SH", "医药", 0.0055, 0.025, 1.30, 0.15),
            ("600202.SH", "医药", 0.0050, 0.022, 1.25, 0.30),
            ("600203.SH", "医药", 0.0048, 0.020, 1.20, 0.45),
        ]

        for ts_code, industry, rebound_strength, flow_scale, amount_boost, phase_shift in specs:
            symbol = self._build_symbol(ts_code)
            frame = self._build_daily_frame(dates, rebound_strength, flow_scale, amount_boost, phase_shift)
            frame.insert(0, "ts_code", ts_code)
            frame.to_csv(data_dir / f"{symbol}.csv", index=False)
            basic_rows.append(
                {
                    "ts_code": ts_code,
                    "name": f"{industry}{symbol[-3:]}",
                    "area": "测试",
                    "industry": industry,
                    "market": "主板",
                }
            )

        basic_path = temp_path / "basic.csv"
        pd.DataFrame(basic_rows).to_csv(basic_path, index=False)
        return data_dir, out_dir, basic_path, scan_date, dates

    @staticmethod
    def _build_daily_frame(
        dates: pd.DatetimeIndex,
        rebound_strength: float,
        flow_scale: float,
        amount_boost: float,
        phase_shift: float,
    ) -> pd.DataFrame:
        periods = len(dates)
        close = np.zeros(periods, dtype=np.float64)
        price = 18.5 + phase_shift
        for idx in range(periods):
            if idx < 70:
                ret = 0.0012 + 0.0010 * np.sin((idx + phase_shift) / 9.0)
            elif idx < 90:
                ret = -0.018 + 0.0012 * np.cos((idx + phase_shift) / 3.8)
            else:
                ret = rebound_strength + 0.0010 * np.sin((idx + phase_shift) / 4.5)
            price *= 1.0 + ret
            close[idx] = price

        open_ = close * (1.0 - 0.004 * np.cos((np.arange(periods) + phase_shift) / 6.0))
        high = np.maximum(open_, close) * (1.0 + 0.014)
        low = np.minimum(open_, close) * (1.0 - 0.013)

        amount = 7.2e5 + 6.5e4 * np.sin((np.arange(periods) + phase_shift) / 4.0)
        amount[90:] *= amount_boost
        turnover_rate = 1.35 + 0.20 * np.cos((np.arange(periods) + phase_shift) / 5.5)
        turnover_rate[90:] += 0.55

        net_mf_amount = amount * -0.03
        net_mf_amount[90:] = amount[90:] * flow_scale

        return pd.DataFrame(
            {
                "trade_date": dates.strftime("%Y%m%d"),
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "amount": amount,
                "turnover_rate": turnover_rate,
                "net_mf_amount": net_mf_amount,
            }
        )


if __name__ == "__main__":
    unittest.main()