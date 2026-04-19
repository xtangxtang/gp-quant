"""
因子模型选股 — Agent 3: 回测验证

职责: 对 Agent 2 选出的 Top N 股票, 模拟交易计算实际收益。
输入: PipelineState.selections (来自 Agent 2)
输出: PipelineState.trades, PipelineState.metrics

交易规则:
  - 买入: scan_date 次日开盘价
  - 卖出: 持有 hold_days 天后收盘价
  - 等权分配
"""

from __future__ import annotations

import glob
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_validation(
    selections: dict[str, pd.DataFrame],
    data_dir: str,
    scan_date: str,
    calendar: list[str] | None = None,
) -> dict[str, dict]:
    """
    对选出的股票模拟交易, 返回每个 horizon 的交易记录和绩效指标。

    Args:
        selections: {horizon: DataFrame with symbol, pred_score, ...}
        data_dir: 日线 CSV 目录
        scan_date: 信号日
        calendar: 交易日历列表 (可选, 不传则自动构建)

    Returns:
        {horizon: {
            "trades": [dict],
            "metrics": dict,
            "entry_date": str,
            "exit_date": str,
        }}
    """
    hold_map = {"1d": 1, "3d": 3, "5d": 5, "1w": 5, "3w": 15, "5w": 25}

    # ── 构建交易日历 ──
    if calendar is None:
        calendar = _build_calendar(data_dir)

    if scan_date not in calendar:
        # 找最近的交易日
        earlier = [d for d in calendar if d <= scan_date]
        if earlier:
            scan_date = earlier[-1]
        else:
            logger.error(f"scan_date {scan_date} not in calendar")
            return {}

    cal_idx = calendar.index(scan_date)

    # ── 加载原始日线 (lazy) ──
    raw_cache: dict[str, pd.DataFrame] = {}

    results = {}
    for h, sel_df in selections.items():
        if sel_df.empty:
            continue

        hold_days = hold_map.get(h, 5)

        # 确定买卖日期
        if cal_idx + 1 >= len(calendar):
            logger.warning(f"  {h}: 无法确定入场日 (日历不够)")
            continue
        entry_date = calendar[cal_idx + 1]

        if cal_idx + 1 + hold_days >= len(calendar):
            logger.warning(f"  {h}: 无法确定退场日 (日历不够)")
            continue
        exit_date = calendar[cal_idx + 1 + hold_days]

        trades = []
        for _, row in sel_df.iterrows():
            sym = row["symbol"]
            rdf = _get_raw_data(raw_cache, data_dir, sym)
            if rdf is None:
                continue

            entry_rows = rdf[rdf["trade_date"] == entry_date]
            if entry_rows.empty:
                continue
            entry_price = float(entry_rows.iloc[0]["open"])
            if entry_price <= 0:
                continue

            exit_rows = rdf[rdf["trade_date"] == exit_date]
            if exit_rows.empty:
                continue
            exit_price = float(exit_rows.iloc[0]["close"])

            pnl_pct = (exit_price - entry_price) / entry_price * 100

            trades.append({
                "signal_date": scan_date,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "symbol": sym,
                "name": row.get("name", ""),
                "industry": row.get("industry", ""),
                "pred_score": row.get("pred_score", 0),
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "pnl_pct": round(pnl_pct, 2),
                "hold_days": hold_days,
            })

        metrics = _compute_metrics(trades) if trades else {"n_trades": 0}

        results[h] = {
            "trades": trades,
            "metrics": metrics,
            "entry_date": entry_date,
            "exit_date": exit_date,
        }

        n_win = sum(1 for t in trades if t["pnl_pct"] > 0)
        logger.info(f"  {h}: {len(trades)} 笔交易, "
                    f"胜率 {n_win}/{len(trades)}, "
                    f"均收益 {metrics.get('avg_pnl', 0):+.2f}%, "
                    f"买入 {entry_date}, 卖出 {exit_date}")

    logger.info(f"Agent 3 完成: {sum(len(r['trades']) for r in results.values())} 笔交易")
    return results


def _build_calendar(data_dir: str) -> list[str]:
    """从日线数据构建交易日历。"""
    all_dates: set[str] = set()
    csvs = sorted(glob.glob(os.path.join(data_dir, "*.csv")))[:50]
    for fpath in csvs:
        try:
            df = pd.read_csv(fpath, usecols=["trade_date"])
            all_dates.update(df["trade_date"].astype(str).tolist())
        except Exception:
            continue
    return sorted(all_dates)


def _get_raw_data(
    cache: dict[str, pd.DataFrame],
    data_dir: str,
    symbol: str,
) -> pd.DataFrame | None:
    """加载并缓存原始日线数据。"""
    if symbol in cache:
        return cache[symbol]
    fpath = os.path.join(data_dir, f"{symbol}.csv")
    if not os.path.exists(fpath):
        cache[symbol] = None
        return None
    try:
        df = pd.read_csv(fpath, usecols=["trade_date", "open", "close", "amount"])
        df["trade_date"] = df["trade_date"].astype(str)
        df = df.sort_values("trade_date").reset_index(drop=True)
        cache[symbol] = df
        return df
    except Exception:
        cache[symbol] = None
        return None


def _compute_metrics(trades: list[dict]) -> dict:
    """计算回测绩效指标。"""
    if not trades:
        return {"n_trades": 0}

    pnls = np.array([t["pnl_pct"] for t in trades])
    n = len(pnls)
    wins = int((pnls > 0).sum())

    return {
        "n_trades": n,
        "win_rate": round(wins / n, 4) if n > 0 else 0,
        "avg_pnl": round(float(pnls.mean()), 4),
        "median_pnl": round(float(np.median(pnls)), 4),
        "total_pnl": round(float(pnls.sum()), 2),
        "best_trade": round(float(pnls.max()), 2),
        "worst_trade": round(float(pnls.min()), 2),
        "std_pnl": round(float(pnls.std()), 4),
    }
