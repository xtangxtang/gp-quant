"""
Holding Position Model — 因子计算

复用 adaptive_state_machine/feature.py 的 FactorCalculator。
这里做一层薄封装，确保接口一致。
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FactorCalculator:
    """持仓模型因子计算器 — 复用 adaptive_state_machine 的 FactorCalculator"""

    def __init__(
        self,
        daily_dir: str,
        data_root: str = "",
        cache_dir: str = "",
        max_workers: int = 28,
    ):
        self.daily_dir = daily_dir
        self.data_root = data_root
        self.cache_dir = cache_dir
        self.max_workers = max_workers

    def compute_one(
        self,
        symbol: str,
        scan_date: str = "",
    ) -> tuple[str, Optional[pd.DataFrame]]:
        """计算单只股票的因子序列"""
        from src.strategy.entropy_accumulation_breakout.feature_engine import build_features

        fpath = os.path.join(self.daily_dir, f"{symbol}.csv")
        if not os.path.exists(fpath):
            return symbol, None

        try:
            df = pd.read_csv(fpath, on_bad_lines="skip")
        except Exception:
            return symbol, None

        if "trade_date" not in df.columns or "close" not in df.columns:
            return symbol, None

        df["trade_date"] = df["trade_date"].astype(str)
        if scan_date:
            df = df[df["trade_date"] <= scan_date]

        if len(df) < 80:
            return symbol, None

        try:
            result = build_features(
                df_daily=df, symbol=symbol, data_root=self.data_root,
            )
            df_featured = result.get("daily")
            df_weekly = result.get("weekly")

            if df_featured is None or len(df_featured) < 80:
                return symbol, None

            # 追加周线独有特征
            weekly_cols = [
                "pe_ttm_pctl", "pb_pctl",
                "weekly_big_net_cumsum",
                "weekly_turnover_shrink", "weekly_turnover_ma4",
            ]
            for col in weekly_cols:
                k = f"w_{col}"
                if df_weekly is not None and not df_weekly.empty and col in df_weekly.columns:
                    val = df_weekly.iloc[-1].get(col)
                    df_featured[k] = float(val) if pd.notna(val) else np.nan
                else:
                    df_featured[k] = np.nan

            return symbol, df_featured
        except Exception as e:
            logger.debug(f"Error computing {symbol}: {e}")
            return symbol, None

    def get_factor_sequence(
        self,
        symbol: str,
        current_date: str,
        seq_len: int = 60,
        factor_names: Optional[list[str]] = None,
        standardize_mean: Optional[np.ndarray] = None,
        standardize_std: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        获取单只股票最近 seq_len 天的因子序列。

        Returns:
            (seq_len, n_factors) 已标准化的 numpy 数组，或 None
        """
        sym, df = self.compute_one(symbol, scan_date=current_date)
        if df is None:
            return None

        factor_cols = [c for c in factor_names if c in df.columns] if factor_names else [
            c for c in df.columns if df[c].dtype in (np.float32, np.float64, np.int32, np.int64)
            and c not in ("trade_date", "open", "high", "low", "close", "vol", "amount",
                          "turnover_rate", "net_mf_amount", "symbol")
        ]

        if len(factor_cols) < 10:
            return None

        factor_values = df[factor_cols].select_dtypes(include=[np.number]).values.astype(np.float32)
        if len(factor_values) < seq_len:
            return None

        seq = factor_values[-seq_len:]

        # 标准化
        if standardize_mean is not None and standardize_std is not None:
            seq = (seq - standardize_mean) / standardize_std
            seq = np.clip(seq, -5.0, 5.0)

        return seq
