"""
Bull Hunter v3 — Prediction Tracker (预测跟踪器)

职责:
  1. 记录 Agent 3 每日 Top 5 预测到 tracking 表
  2. 每日更新 tracking 表的实际涨幅 (max_gain, current_gain)
  3. 30 天到期后做最终评估
  4. 提供近期预测记录 (Agent 3 去重用)
  5. 输出到期评估结果 (Agent 4 反馈用)

持久化:
  results/bull_hunter/tracking/
    active.csv        # 活跃跟踪项 (未到期)
    history.csv       # 已到期历史记录
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 跟踪窗口 (交易日)
TRACKING_DAYS = 30
# 评估阈值
EVAL_SUCCESS_GAIN = 0.30   # max_gain >= 30% 视为成功
EVAL_FAIL_GAIN = 0.10      # max_gain < 10% 视为失败
EVAL_LOSS_THRESHOLD = -0.10  # current_gain < -10% 视为亏损

ACTIVE_COLS = [
    "symbol", "name", "industry", "scan_date", "entry_date", "entry_close",
    "prob_200", "prob_100", "rank", "model_date",
    "days_held", "current_close", "current_gain", "max_gain",
]
HISTORY_COLS = ACTIVE_COLS + ["final_gain", "max_gain_final", "eval_result"]


class PredictionTracker:
    """管理 Agent 3 预测的跟踪、更新和到期评估。"""

    def __init__(self, tracking_dir: str):
        self.tracking_dir = tracking_dir
        self.active_path = os.path.join(tracking_dir, "active.csv")
        self.history_path = os.path.join(tracking_dir, "history.csv")
        os.makedirs(tracking_dir, exist_ok=True)

    def record_predictions(
        self,
        predictions: pd.DataFrame,
        scan_date: str,
        data_dir: str,
        calendar: list[str],
        model_date: str = "",
    ) -> int:
        """
        记录 Agent 3 当日预测到 active tracking 表。

        Args:
            predictions: Agent 3 输出 (含 symbol, prob_200, prob_100, rank)
            scan_date: 预测日期
            data_dir: 日线数据目录 (获取 entry_close)
            calendar: 交易日历
            model_date: 模型训练日期

        Returns:
            新增记录数
        """
        if predictions.empty:
            return 0

        active = self._load_active()

        # 避免重复: 同一 scan_date + symbol 不重复记录
        existing_keys = set()
        if not active.empty:
            existing_keys = set(
                active["symbol"] + "_" + active["scan_date"]
            )

        # 次日开盘价作为 entry
        date_set = set(calendar)
        if scan_date in date_set:
            sd_idx = calendar.index(scan_date)
            entry_date_idx = sd_idx + 1
            entry_date = calendar[entry_date_idx] if entry_date_idx < len(calendar) else scan_date
        else:
            entry_date = scan_date

        new_rows = []
        for _, row in predictions.iterrows():
            sym = row["symbol"]
            key = f"{sym}_{scan_date}"
            if key in existing_keys:
                continue

            entry_close = _get_price(data_dir, sym, entry_date, "open")
            if entry_close is None or entry_close <= 0:
                entry_close = _get_price(data_dir, sym, scan_date, "close")
            if entry_close is None or entry_close <= 0:
                continue

            new_rows.append({
                "symbol": sym,
                "name": row.get("name", ""),
                "industry": row.get("industry", ""),
                "scan_date": scan_date,
                "entry_date": entry_date,
                "entry_close": round(float(entry_close), 2),
                "prob_200": round(float(row.get("prob_200", 0)), 6),
                "prob_100": round(float(row.get("prob_100", 0)), 6),
                "rank": int(row.get("rank", 0)),
                "model_date": model_date,
                "days_held": 0,
                "current_close": round(float(entry_close), 2),
                "current_gain": 0.0,
                "max_gain": 0.0,
            })

        if new_rows:
            new_df = pd.DataFrame(new_rows)
            active = pd.concat([active, new_df], ignore_index=True)
            self._save_active(active)
            logger.info(f"Tracker: 新增 {len(new_rows)} 条跟踪 (scan_date={scan_date})")

        return len(new_rows)

    def update_prices(
        self,
        current_date: str,
        data_dir: str,
        calendar: list[str],
    ) -> int:
        """
        更新所有活跃跟踪项的当前价格和涨幅。

        Returns:
            更新的记录数
        """
        active = self._load_active()
        if active.empty:
            return 0

        date_set = set(calendar)
        n_updated = 0

        for idx, row in active.iterrows():
            sym = row["symbol"]
            entry_close = float(row["entry_close"])
            entry_date = row["entry_date"]

            # 计算持有天数
            if entry_date in date_set and current_date in date_set:
                e_idx = calendar.index(entry_date)
                c_idx = calendar.index(current_date)
                days = c_idx - e_idx
            else:
                days = int(row["days_held"])

            # 获取当前价格
            current_close = _get_price(data_dir, sym, current_date, "close")
            if current_close is None or current_close <= 0:
                continue

            current_gain = (current_close - entry_close) / entry_close
            old_max = float(row["max_gain"])
            max_gain = max(old_max, current_gain)

            active.at[idx, "days_held"] = days
            active.at[idx, "current_close"] = round(current_close, 2)
            active.at[idx, "current_gain"] = round(current_gain, 4)
            active.at[idx, "max_gain"] = round(max_gain, 4)
            n_updated += 1

        self._save_active(active)
        if n_updated > 0:
            logger.info(f"Tracker: 更新 {n_updated} 条价格 (date={current_date})")
        return n_updated

    def evaluate_expired(self, current_date: str, calendar: list[str]) -> pd.DataFrame:
        """
        评估到期项: days_held >= TRACKING_DAYS 的移到 history。

        评估标准:
          - success: max_gain >= 30%
          - fail: max_gain < 10%
          - loss: current_gain < -10%
          - neutral: 其他

        Returns:
            到期评估结果 DataFrame
        """
        active = self._load_active()
        if active.empty:
            return pd.DataFrame()

        expired_mask = active["days_held"] >= TRACKING_DAYS
        expired = active[expired_mask].copy()
        remaining = active[~expired_mask].copy()

        if expired.empty:
            return pd.DataFrame()

        # 评估
        expired["final_gain"] = expired["current_gain"]
        expired["max_gain_final"] = expired["max_gain"]

        conditions = [
            expired["max_gain"] >= EVAL_SUCCESS_GAIN,
            expired["current_gain"] < EVAL_LOSS_THRESHOLD,
            expired["max_gain"] < EVAL_FAIL_GAIN,
        ]
        choices = ["success", "loss", "fail"]
        expired["eval_result"] = np.select(conditions, choices, default="neutral")

        # 追加到 history
        history = self._load_history()
        history = pd.concat([history, expired], ignore_index=True)
        self._save_history(history)

        # 更新 active (移除到期项)
        self._save_active(remaining)

        n_success = (expired["eval_result"] == "success").sum()
        n_fail = (expired["eval_result"] == "fail").sum()
        n_loss = (expired["eval_result"] == "loss").sum()
        n_neutral = (expired["eval_result"] == "neutral").sum()
        logger.info(
            f"Tracker: {len(expired)} 条到期 — "
            f"成功={n_success} 失败={n_fail} 亏损={n_loss} 中性={n_neutral}"
        )

        return expired

    def get_recent_predictions(self, scan_date: str, calendar: list[str], window_days: int = 3) -> pd.DataFrame:
        """获取近 window_days 天的预测记录 (Agent 3 去重用)。"""
        active = self._load_active()
        if active.empty:
            return pd.DataFrame()

        date_set = set(calendar)
        if scan_date not in date_set:
            return active  # 保守: 返回全部

        cur_idx = calendar.index(scan_date)
        cutoff_idx = max(0, cur_idx - window_days)
        cutoff_date = calendar[cutoff_idx]

        recent = active[active["scan_date"] >= cutoff_date]
        return recent

    def get_active_count(self) -> int:
        """当前活跃跟踪数。"""
        active = self._load_active()
        return len(active)

    def get_tracking_summary(self) -> dict:
        """获取跟踪状态摘要 (Agent 4 用)。"""
        active = self._load_active()
        history = self._load_history()

        summary = {
            "n_active": len(active),
            "n_history": len(history),
        }

        if not active.empty:
            summary["active_avg_gain"] = round(float(active["current_gain"].mean()), 4)
            summary["active_avg_max_gain"] = round(float(active["max_gain"].mean()), 4)
            summary["active_avg_days"] = round(float(active["days_held"].mean()), 1)

        if not history.empty:
            n = len(history)
            summary["history_success_rate"] = round(float((history["eval_result"] == "success").mean()), 4)
            summary["history_fail_rate"] = round(float((history["eval_result"] == "fail").mean()), 4)
            summary["history_loss_rate"] = round(float((history["eval_result"] == "loss").mean()), 4)
            summary["history_avg_max_gain"] = round(float(history["max_gain_final"].mean()), 4)
            summary["history_avg_final_gain"] = round(float(history["final_gain"].mean()), 4)

        return summary

    def get_expired_for_review(self, n_recent: int = 50) -> pd.DataFrame:
        """获取最近 n_recent 条到期记录 (Agent 4 review 用)。"""
        history = self._load_history()
        if history.empty:
            return pd.DataFrame()
        return history.tail(n_recent)

    # ── 内部方法 ──

    def _load_active(self) -> pd.DataFrame:
        if os.path.exists(self.active_path):
            try:
                df = pd.read_csv(self.active_path, dtype={"scan_date": str, "entry_date": str})
                return df
            except Exception:
                pass
        return pd.DataFrame(columns=ACTIVE_COLS)

    def _save_active(self, df: pd.DataFrame):
        df.to_csv(self.active_path, index=False, encoding="utf-8-sig")

    def _load_history(self) -> pd.DataFrame:
        if os.path.exists(self.history_path):
            try:
                df = pd.read_csv(self.history_path, dtype={"scan_date": str, "entry_date": str})
                return df
            except Exception:
                pass
        return pd.DataFrame(columns=HISTORY_COLS)

    def _save_history(self, df: pd.DataFrame):
        df.to_csv(self.history_path, index=False, encoding="utf-8-sig")


# ── 辅助函数 ──

def _get_price(data_dir: str, symbol: str, date: str, col: str = "close") -> float | None:
    """获取某只股票在某日的价格。"""
    fpath = os.path.join(data_dir, f"{symbol}.csv")
    if not os.path.exists(fpath):
        return None
    try:
        df = pd.read_csv(fpath, usecols=["trade_date", col], dtype={"trade_date": str})
        row = df[df["trade_date"] == date]
        if row.empty:
            return None
        return float(row.iloc[0][col])
    except Exception:
        return None
