"""
因子模型选股 — Agent 3: 条件退出回测 (v2)

替代固定持有期:
  - 买入: scan_date 次日开盘价 (同旧版)
  - 退出: 条件触发 (结构崩塌信号) 或 达到最大持有天数 (安全网)

退出条件 (>=2 个同时触发):
  1. 缩量度飙升 (vol_shrink > 1.5) — 先放量后急缩, 动能衰竭
  2. 散户占比骤降 (mf_sm_proportion < 0.25) — 散户已跑光
  3. 突破幅度回落 (breakout_range < 0.10) — 价格跌回中轨
  4. 极度无序 (perm_entropy_m > 0.98) — 完全随机

最大持有天数由 horizon 决定:
  3d -> 3, 5d -> 5, 1w -> 5, 3w -> 15, 5w -> 25
"""

from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExitConfig:
    """退出条件配置。"""
    vol_shrink_exit: float = 1.5          # 缩量度飙升
    mf_sm_proportion_exit: float = 0.25   # 散户占比骤降
    breakout_range_exit: float = 0.10     # 突破幅度回落
    perm_entropy_exit: float = 0.98       # 极度无序
    min_exit_signals: int = 2             # 至少 N 个退出信号同时触发
    min_hold_days: int = 1                # 最少持有天数 (避免当日进出)


def run_validation(
    selections: dict[str, pd.DataFrame],
    data_dir: str,
    scan_date: str,
    calendar: list[str] | None = None,
    cache_dir: str = "",
    exit_cfg: ExitConfig | None = None,
) -> dict[str, dict]:
    """
    条件退出回测。

    Args:
        selections: {horizon: DataFrame with symbol, composite_score, ...}
        data_dir: 日线 CSV 目录
        scan_date: 信号日
        calendar: 交易日历
        cache_dir: 特征缓存目录 (用于加载退出条件因子)
        exit_cfg: 退出条件配置

    Returns:
        {horizon: {
            "trades": [dict],
            "metrics": dict,
            "entry_date": str,
            "exit_date": str,
        }}
    """
    hold_map = {"1d": 1, "3d": 3, "5d": 5, "1w": 5, "3w": 15, "5w": 25}
    exit_cfg = exit_cfg or ExitConfig()

    # ── 构建交易日历 ──
    if calendar is None:
        calendar = _build_calendar(data_dir)

    if scan_date not in calendar:
        earlier = [d for d in calendar if d <= scan_date]
        if earlier:
            scan_date = earlier[-1]
        else:
            logger.error(f"scan_date {scan_date} not in calendar")
            return {}

    cal_idx = calendar.index(scan_date)

    # ── 数据缓存 ──
    raw_cache: dict[str, pd.DataFrame | None] = {}
    factor_cache: dict[str, pd.DataFrame | None] = {}

    results = {}
    for h, sel_df in selections.items():
        if sel_df.empty:
            continue

        max_hold_days = hold_map.get(h, 5)

        # 确定入场日
        if cal_idx + 1 >= len(calendar):
            logger.warning(f"  {h}: 日历不足, 无法入场")
            continue
        entry_date = calendar[cal_idx + 1]

        if cal_idx + 1 + max_hold_days >= len(calendar):
            logger.warning(f"  {h}: 日历不足, 无法完成持有期")
            continue
        max_exit_date = calendar[cal_idx + 1 + max_hold_days]

        trades = []
        for _, row in sel_df.iterrows():
            sym = row["symbol"]
            rdf = _get_raw_data(raw_cache, data_dir, sym)
            if rdf is None:
                continue

            # 入场价
            entry_rows = rdf[rdf["trade_date"] == entry_date]
            if entry_rows.empty:
                continue
            entry_price = float(entry_rows.iloc[0]["open"])
            if entry_price <= 0:
                continue

            # ── 条件退出: 逐日检查 ──
            actual_exit_date = max_exit_date
            exit_reason = "max_hold"
            actual_hold_days = max_hold_days

            if cache_dir:
                fdf = _get_factor_data(factor_cache, cache_dir, sym)
                if fdf is not None:
                    for day_offset in range(exit_cfg.min_hold_days, max_hold_days + 1):
                        check_idx = cal_idx + 1 + day_offset
                        if check_idx >= len(calendar):
                            break
                        check_date = calendar[check_idx]

                        n_signals = _check_exit_conditions(fdf, check_date, exit_cfg)
                        if n_signals >= exit_cfg.min_exit_signals:
                            actual_exit_date = check_date
                            exit_reason = "condition"
                            actual_hold_days = day_offset
                            break

            # 退场价
            exit_rows = rdf[rdf["trade_date"] == actual_exit_date]
            if exit_rows.empty:
                # 找最近可用日
                fwd = rdf[rdf["trade_date"] >= actual_exit_date].head(1)
                if fwd.empty:
                    continue
                actual_exit_date = str(fwd.iloc[0]["trade_date"])
                exit_rows = fwd

            exit_price = float(exit_rows.iloc[0]["close"])
            pnl_pct = (exit_price - entry_price) / entry_price * 100

            trades.append({
                "signal_date": scan_date,
                "entry_date": entry_date,
                "exit_date": actual_exit_date,
                "symbol": sym,
                "name": row.get("name", ""),
                "industry": row.get("industry", ""),
                "composite_score": row.get("composite_score", 0),
                "phase": row.get("phase", ""),
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "pnl_pct": round(pnl_pct, 2),
                "hold_days": actual_hold_days,
                "max_hold_days": max_hold_days,
                "exit_reason": exit_reason,
            })

        metrics = _compute_metrics(trades) if trades else {"n_trades": 0}

        results[h] = {
            "trades": trades,
            "metrics": metrics,
            "entry_date": entry_date,
            "exit_date": max_exit_date,
        }

        n_win = sum(1 for t in trades if t["pnl_pct"] > 0)
        n_cond = sum(1 for t in trades if t["exit_reason"] == "condition")
        avg_hold = (np.mean([t["hold_days"] for t in trades])
                    if trades else 0)
        logger.info(f"  {h}: {len(trades)} 笔, 胜率 {n_win}/{len(trades)}, "
                    f"条件退出 {n_cond} 笔, 均持有 {avg_hold:.1f} 天, "
                    f"均收益 {metrics.get('avg_pnl', 0):+.2f}%")

    logger.info(f"Agent 3 完成: {sum(len(r['trades']) for r in results.values())} 笔交易")
    return results


def _check_exit_conditions(
    fdf: pd.DataFrame,
    check_date: str,
    cfg: ExitConfig,
) -> int:
    """检查单日退出条件, 返回触发的信号数。"""
    rows = fdf[fdf["trade_date"] == check_date]
    if rows.empty:
        return 0

    row = rows.iloc[0]
    signals = 0

    if "vol_shrink" in fdf.columns:
        v = row.get("vol_shrink")
        if pd.notna(v) and v > cfg.vol_shrink_exit:
            signals += 1

    if "mf_sm_proportion" in fdf.columns:
        v = row.get("mf_sm_proportion")
        if pd.notna(v) and v < cfg.mf_sm_proportion_exit:
            signals += 1

    if "breakout_range" in fdf.columns:
        v = row.get("breakout_range")
        if pd.notna(v) and v < cfg.breakout_range_exit:
            signals += 1

    if "perm_entropy_m" in fdf.columns:
        v = row.get("perm_entropy_m")
        if pd.notna(v) and v > cfg.perm_entropy_exit:
            signals += 1

    return signals


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
    cache: dict[str, pd.DataFrame | None],
    data_dir: str,
    symbol: str,
) -> pd.DataFrame | None:
    """加载原始日线数据 (用于计算 PnL)。"""
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


def _get_factor_data(
    cache: dict[str, pd.DataFrame | None],
    cache_dir: str,
    symbol: str,
) -> pd.DataFrame | None:
    """加载因子时序 (用于退出条件检测)。"""
    if symbol in cache:
        return cache[symbol]

    fpath = os.path.join(cache_dir, "daily", f"{symbol}.csv")
    if not os.path.exists(fpath):
        cache[symbol] = None
        return None

    try:
        # 只加载退出相关列
        exit_cols = ["trade_date", "vol_shrink", "mf_sm_proportion",
                     "breakout_range", "perm_entropy_m"]
        available = pd.read_csv(fpath, nrows=0).columns.tolist()
        use_cols = [c for c in exit_cols if c in available]
        if "trade_date" not in use_cols:
            use_cols.insert(0, "trade_date")

        df = pd.read_csv(fpath, usecols=use_cols)
        df["trade_date"] = df["trade_date"].astype(str)
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
    hold_days = np.array([t["hold_days"] for t in trades])
    n = len(pnls)
    wins = int((pnls > 0).sum())
    n_cond = sum(1 for t in trades if t["exit_reason"] == "condition")

    return {
        "n_trades": n,
        "win_rate": round(wins / n, 4) if n > 0 else 0,
        "avg_pnl": round(float(pnls.mean()), 4),
        "median_pnl": round(float(np.median(pnls)), 4),
        "total_pnl": round(float(pnls.sum()), 2),
        "best_trade": round(float(pnls.max()), 2),
        "worst_trade": round(float(pnls.min()), 2),
        "std_pnl": round(float(pnls.std()), 4),
        "avg_hold_days": round(float(hold_days.mean()), 1),
        "condition_exit_rate": round(n_cond / n, 4) if n > 0 else 0,
    }
