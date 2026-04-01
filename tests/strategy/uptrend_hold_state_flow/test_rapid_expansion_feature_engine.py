import unittest

import numpy as np
import pandas as pd

from src.strategy.uptrend_hold_state_flow.rapid_expansion_exhaustion_exit.rapid_expansion_exhaustion_feature_engine import (
    build_rapid_expansion_exhaustion_feature_frame,
)
from src.strategy.uptrend_hold_state_flow.rapid_expansion_hold.rapid_expansion_feature_engine import (
    build_rapid_expansion_feature_frame,
)


class RapidExpansionFeatureEngineTests(unittest.TestCase):
    def test_feature_builders_derive_missing_market_columns(self) -> None:
        daily = self._build_daily_frame(220)

        rapid = build_rapid_expansion_feature_frame(daily)
        self.assertFalse(rapid.empty)

        for column in ["ret_5", "range_state_20", "trend_strength", "atr_ratio_20", "expansion_hold_score"]:
            self.assertIn(column, rapid.columns)

        latest_rapid = rapid.iloc[-1]
        for column in ["ret_5", "range_state_20", "trend_strength", "atr_ratio_20", "expansion_hold_score"]:
            self.assertFalse(pd.isna(latest_rapid[column]), column)

        exhaustion = build_rapid_expansion_exhaustion_feature_frame(daily)
        self.assertFalse(exhaustion.empty)
        latest_exhaustion = exhaustion.iloc[-1]
        for column in ["ret_5", "range_state_20", "atr_ratio_20", "peak_extension_score", "exhaustion_exit_score"]:
            self.assertIn(column, exhaustion.columns)
            self.assertFalse(pd.isna(latest_exhaustion[column]), column)

    @staticmethod
    def _build_daily_frame(periods: int) -> pd.DataFrame:
        dates = pd.bdate_range("2024-01-02", periods=periods)
        index = np.arange(periods, dtype=np.float64)

        close = 12.0 + 0.08 * index + 0.55 * np.sin(index / 6.0)
        open_ = close * (1.0 - 0.004 * np.cos(index / 5.0))
        high = np.maximum(open_, close) * (1.0 + 0.012 + 0.001 * np.sin(index / 9.0))
        low = np.minimum(open_, close) * (1.0 - 0.011 - 0.001 * np.cos(index / 8.0))
        amount = 6.0e7 + 1.2e7 * np.sin(index / 4.0) + 8.0e5 * index
        turnover_rate = 2.1 + 0.35 * np.cos(index / 7.0)
        net_mf_amount = 4.5e6 * np.sin(index / 5.5) + 6.0e5 * index

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