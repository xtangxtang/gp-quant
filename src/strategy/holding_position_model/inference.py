"""
Holding Position Model — 推理

单只股票 + 持仓信息 → stay_prob / collapse_risk / expected_days → 拿住/减仓/走人
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from .config import (
    SEQ_LEN, HOLDING_DIM,
    DECISION_STAY_THRESHOLD, DECISION_COLLAPSE_THRESHOLD, DECISION_DAYS_THRESHOLD,
)
from .model import HoldingPositionModel

logger = logging.getLogger(__name__)

# ── 从 adaptive_state_machine 导入因子去重 ──
from src.strategy.adaptive_state_machine.train_attention import _dedup_factors


def _compute_holding_features(
    entry_price: float,
    current_price: float,
    peak_price: float,
    days_held: int,
    recent_closes: np.ndarray,
) -> np.ndarray:
    """计算 5 维持仓特征"""
    unrealized_pnl = (current_price - entry_price) / max(abs(entry_price), 1e-8)
    max_pnl = (peak_price - entry_price) / max(abs(entry_price), 1e-8)
    drawdown = (current_price - peak_price) / max(abs(peak_price), 1e-8)
    entry_pos = float((recent_closes < entry_price).sum() + 1) / max(len(recent_closes), 1)

    return np.array([
        days_held / 20.0,
        unrealized_pnl,
        max_pnl,
        drawdown,
        entry_pos,
    ], dtype=np.float32)


class HoldingInference:
    """持仓推理引擎"""

    def __init__(
        self,
        model_path: str,
        daily_dir: str,
        data_root: str = "",
        seq_len: int = SEQ_LEN,
    ):
        self.model_path = model_path
        self.daily_dir = daily_dir
        self.data_root = data_root
        self.seq_len = seq_len
        self.model: Optional[HoldingPositionModel] = None
        self._standardize_mean = None
        self._standardize_std = None
        self._factor_names = None

    def load(self) -> bool:
        """加载模型"""
        if not os.path.exists(self.model_path):
            logger.error(f"Model not found: {self.model_path}")
            return False

        self.model, state = HoldingPositionModel.load_from_checkpoint(self.model_path)
        self._factor_names = state.get("factor_names")
        self.seq_len = state.get("seq_len", self.seq_len)

        # 加载标准化参数
        stats_path = self.model_path.replace(".pt", "_stats.npy")
        if os.path.exists(stats_path):
            stats = np.load(stats_path, allow_pickle=True).item()
            self._standardize_mean = stats.get("mean")
            self._standardize_std = stats.get("std")
            logger.info("Loaded standardize params")
        else:
            logger.warning("No standardize params found, will compute from input")

        return True

    def _get_factor_sequence(
        self,
        symbol: str,
        current_date: str,
    ) -> Optional[tuple[np.ndarray, pd.DataFrame]]:
        """获取因子序列 + 原始 DataFrame"""
        from src.strategy.holding_position_model.feature import FactorCalculator
        calc = FactorCalculator(
            daily_dir=self.daily_dir,
            data_root=self.data_root,
        )
        sym, df = calc.compute_one(symbol, scan_date=current_date)
        if df is None:
            return None

        factor_cols = [c for c in self._factor_names if c in df.columns] if self._factor_names else [
            c for c in df.columns if df[c].dtype in (np.float32, np.float64, np.int32, np.int64)
            and c not in ("trade_date", "open", "high", "low", "close", "vol", "amount",
                          "turnover_rate", "net_mf_amount", "symbol")
        ]

        if len(factor_cols) < 10:
            return None

        factor_values = df[factor_cols].select_dtypes(include=[np.number]).values.astype(np.float32)
        if len(factor_values) < self.seq_len:
            return None

        seq = factor_values[-self.seq_len:]

        if self._standardize_mean is not None and self._standardize_std is not None:
            seq = (seq - self._standardize_mean) / self._standardize_std
            seq = np.clip(seq, -5.0, 5.0)

        return seq, df

    def predict(
        self,
        symbol: str,
        entry_price: float,
        entry_date: str,
        current_date: str = "",
    ) -> dict:
        """
        对单只持仓股票做决策。

        Returns:
            {
                "symbol": str,
                "entry_price": float,
                "current_price": float,
                "peak_price": float,
                "days_held": int,
                "unrealized_pnl": float,
                "drawdown": float,
                "stay_prob": float,
                "collapse_risk": float,
                "expected_days": float,
                "recommendation": "持有" | "减仓" | "走人",
            }
        """
        if self.model is None and not self.load():
            raise RuntimeError("Model not loaded")

        # 获取因子序列
        result = self._get_factor_sequence(symbol, current_date)
        if result is None:
            raise ValueError(f"No valid factor data for {symbol}")
        factor_seq, df = result

        # 找到 current_date 和 entry_date 在 df 中的位置
        sym_dates = df["trade_date"].tolist()
        close_prices = df["close"].values.astype(np.float64)
        high_prices = df["high"].values.astype(np.float64)

        entry_idx = None
        current_idx = len(sym_dates) - 1  # 默认最新

        for i, d in enumerate(sym_dates):
            if d == str(entry_date)[:8]:
                entry_idx = i
            if current_date and d == str(current_date)[:8]:
                current_idx = i

        if entry_idx is None:
            # 用最近的日期
            entry_idx = 0

        entry_price_actual = float(df.iloc[entry_idx]["open"]) if "open" in df.columns else entry_price
        current_price = close_prices[current_idx]
        price_path = close_prices[entry_idx: current_idx + 1]
        peak_price = price_path.max()
        high_path = high_prices[entry_idx: current_idx + 1]
        peak_high = high_path.max()

        # 用用户传入的 entry_price (如果提供了) 或实际开盘价
        effective_entry = entry_price if entry_price > 0 else entry_price_actual
        days_held = current_idx - entry_idx

        # 持仓特征
        holding_feat = _compute_holding_features(
            effective_entry, current_price, max(peak_price, peak_high),
            days_held, close_prices[max(0, current_idx - self.seq_len): current_idx],
        )

        # 推理
        factor_seq_batch = factor_seq[np.newaxis, :, :]
        holding_feat_batch = holding_feat[np.newaxis, :]

        output = self.model.forward(factor_seq_batch, holding_feat_batch, training=False)

        stay_prob = float(output["stay_prob"][0])
        collapse_risk = float(output["collapse_risk"][0])
        expected_days = float(output["expected_days"][0])

        # 决策
        if collapse_risk > DECISION_COLLAPSE_THRESHOLD:
            recommendation = "走人"
        elif stay_prob > DECISION_STAY_THRESHOLD:
            recommendation = "持有"
        elif expected_days < DECISION_DAYS_THRESHOLD:
            recommendation = "减仓"
        else:
            recommendation = "观望"

        unrealized_pnl = (current_price - effective_entry) / max(abs(effective_entry), 1e-8)
        drawdown = (current_price - max(peak_price, peak_high)) / max(abs(max(peak_price, peak_high)), 1e-8)

        return {
            "symbol": symbol,
            "entry_price": round(effective_entry, 2),
            "current_price": round(current_price, 2),
            "peak_price": round(max(peak_price, peak_high), 2),
            "days_held": days_held,
            "unrealized_pnl": round(unrealized_pnl * 100, 2),
            "drawdown": round(drawdown * 100, 2),
            "stay_prob": round(stay_prob, 3),
            "collapse_risk": round(collapse_risk, 3),
            "expected_days": round(expected_days, 1),
            "recommendation": recommendation,
        }

    def predict_batch(
        self,
        positions: list[dict],
    ) -> list[dict]:
        """
        批量推理。

        Args:
            positions: [{"symbol": str, "entry_price": float, "entry_date": str}, ...]

        Returns:
            list of prediction dicts
        """
        results = []
        for pos in positions:
            try:
                result = self.predict(
                    symbol=pos["symbol"],
                    entry_price=pos["entry_price"],
                    entry_date=pos["entry_date"],
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to predict for {pos['symbol']}: {e}")
                results.append({
                    "symbol": pos["symbol"],
                    "error": str(e),
                })
        return results
