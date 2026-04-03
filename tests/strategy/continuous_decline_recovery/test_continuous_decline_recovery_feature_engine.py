import unittest

import numpy as np
import pandas as pd

from src.strategy.continuous_decline_recovery.continuous_decline_recovery_feature_engine import (
    build_continuous_decline_recovery_feature_frame,
)


class ContinuousDeclineRecoveryFeatureEngineTests(unittest.TestCase):
    def test_feature_builder_creates_selloff_repair_fields(self) -> None:
        daily = self._build_daily_frame(periods=100, rebound_strength=0.011, flow_scale=0.07, amount_boost=2.2)

        features = build_continuous_decline_recovery_feature_frame(daily)
        self.assertFalse(features.empty)

        required_columns = [
            "damage_score",
            "repair_score",
            "entry_window_score",
            "flow_support_score",
            "stability_score",
            "overheat_score",
            "rebound_from_low_10",
            "amount_ratio_20",
            "base_candidate_flag",
        ]
        for column in required_columns:
            self.assertIn(column, features.columns)

        latest = features.iloc[-1]
        for column in required_columns[:-1]:
            self.assertFalse(pd.isna(latest[column]), column)
        self.assertTrue(bool(latest["base_candidate_flag"]))

    @staticmethod
    def _build_daily_frame(periods: int, rebound_strength: float, flow_scale: float, amount_boost: float) -> pd.DataFrame:
        dates = pd.bdate_range("2024-01-02", periods=periods)
        close = np.zeros(periods, dtype=np.float64)
        price = 18.0
        for idx in range(periods):
            if idx < 72:
                ret = 0.0015 + 0.0010 * np.sin(idx / 9.0)
            elif idx < 92:
                ret = -0.018 + 0.0015 * np.cos(idx / 4.0)
            else:
                ret = rebound_strength + 0.0010 * np.sin(idx / 5.0)
            price *= 1.0 + ret
            close[idx] = price

        open_ = close * (1.0 - 0.004 * np.cos(np.arange(periods) / 7.0))
        high = np.maximum(open_, close) * (1.0 + 0.014)
        low = np.minimum(open_, close) * (1.0 - 0.013)

        amount = 7.5e5 + 6.0e4 * np.sin(np.arange(periods) / 4.0)
        amount[92:] *= amount_boost
        turnover_rate = 1.4 + 0.25 * np.cos(np.arange(periods) / 6.0)
        turnover_rate[92:] += 0.55

        net_mf_amount = amount * -0.03
        net_mf_amount[92:] = amount[92:] * flow_scale

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