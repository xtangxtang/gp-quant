"""
扫描服务 + 回测引擎

全市场扫描 → 候选筛选 → 前瞻回测
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .feature_engine import build_features
from .signal_detector import (
    DetectorConfig,
    SymbolSignal,
    detect_structural_collapse,
    evaluate_symbol,
)

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════
# 配置
# ═════════════════════════════════════════════════════════

@dataclass
class ScanConfig:
    data_dir: str = ""
    basic_path: str = ""
    out_dir: str = ""
    scan_date: str = ""
    symbols: list[str] = field(default_factory=list)
    top_n: int = 30
    lookback_days: int = 500

    # 过滤
    min_amount: float = 5000.0   # 最低日均成交额(万元)
    min_turnover: float = 0.5    # 最低换手率
    exclude_st: bool = True

    # 回测
    backtest_start_date: str = ""
    backtest_end_date: str = ""
    hold_days: int = 5
    max_positions: int = 10
    max_per_industry: int = 2

    # 检测器参数
    detector: DetectorConfig = field(default_factory=DetectorConfig)

    def __post_init__(self):
        """归一化日期字段: 2025/01/01 或 2025-01-01 → 20250101"""
        for attr in ("scan_date", "backtest_start_date", "backtest_end_date"):
            v = getattr(self, attr, "").strip()
            for sep in ("/", "-"):
                if sep in v:
                    parts = v.split(sep)
                    if len(parts) == 3 and all(p.isdigit() for p in parts):
                        v = f"{int(parts[0]):04d}{int(parts[1]):02d}{int(parts[2]):02d}"
                    break
            setattr(self, attr, v)


# ═════════════════════════════════════════════════════════
# 数据加载
# ═════════════════════════════════════════════════════════

def _load_basic_info(basic_path: str) -> dict[str, dict]:
    """加载股票基本信息"""
    if not basic_path or not os.path.exists(basic_path):
        return {}
    try:
        df = pd.read_csv(basic_path, dtype=str)
    except Exception:
        return {}
    info = {}
    for _, row in df.iterrows():
        ts = str(row.get("ts_code", ""))
        sym = ts.replace(".", "").lower()
        if ts.endswith(".SH"):
            sym = "sh" + ts[:6]
        elif ts.endswith(".SZ"):
            sym = "sz" + ts[:6]
        elif ts.endswith(".BJ"):
            sym = "bj" + ts[:6]
        info[sym] = {
            "ts_code": ts,
            "name": str(row.get("name", "")),
            "industry": str(row.get("industry", "")),
            "market": str(row.get("market", "")),
            "list_date": str(row.get("list_date", "")),
        }
    return info


def _resolve_symbols(data_dir: str, symbols: list[str] | None) -> list[str]:
    """解析符号列表"""
    if symbols:
        return symbols
    csvs = sorted(Path(data_dir).glob("*.csv"))
    return [f.stem for f in csvs]


def _load_daily(data_dir: str, symbol: str) -> pd.DataFrame | None:
    """加载单只股票日线数据"""
    fpath = os.path.join(data_dir, f"{symbol}.csv")
    if not os.path.exists(fpath):
        return None
    try:
        df = pd.read_csv(fpath)
    except Exception:
        return None
    if "trade_date" not in df.columns or "close" not in df.columns:
        return None
    df = df.sort_values("trade_date").reset_index(drop=True)
    return df


def _should_skip(
    df: pd.DataFrame,
    info: dict,
    cfg: ScanConfig,
    scan_dt: str,
) -> bool:
    """预过滤: ST / 流动性不足 / 上市太短"""
    if cfg.exclude_st:
        name = info.get("name", "")
        if "ST" in name or "退" in name:
            return True
    if len(df) < 120:
        return True
    # 截取到扫描日
    if scan_dt:
        df_cut = df[df["trade_date"].astype(str) <= scan_dt]
    else:
        df_cut = df
    if len(df_cut) < 120:
        return True
    recent = df_cut.tail(20)
    avg_amount = recent["amount"].mean() if "amount" in recent.columns else 0
    if avg_amount < cfg.min_amount:
        return True
    return False


# ═════════════════════════════════════════════════════════
# 核心扫描
# ═════════════════════════════════════════════════════════

def scan_single_symbol(
    data_dir: str,
    symbol: str,
    cfg: ScanConfig,
    scan_dt: str,
    basic_info: dict,
) -> SymbolSignal | None:
    """扫描单只股票"""
    df = _load_daily(data_dir, symbol)
    if df is None:
        return None

    info = basic_info.get(symbol, {})
    if _should_skip(df, info, cfg, scan_dt):
        return None

    # 截取数据
    if scan_dt:
        df = df[df["trade_date"].astype(str) <= scan_dt].copy()
    df = df.tail(cfg.lookback_days).reset_index(drop=True)

    if len(df) < 60:
        return None

    # 构建特征
    feats = build_features(df)
    df_daily = feats["daily"]
    df_weekly = feats.get("weekly")

    # 评估
    signal = evaluate_symbol(df_daily, df_weekly, symbol, cfg.detector)
    signal.details["name"] = info.get("name", "")
    signal.details["industry"] = info.get("industry", "")
    return signal


def run_scan(cfg: ScanConfig) -> pd.DataFrame:
    """
    全市场扫描, 返回候选股票表.
    """
    basic_info = _load_basic_info(cfg.basic_path)
    symbols = _resolve_symbols(cfg.data_dir, cfg.symbols if cfg.symbols else None)

    scan_dt = cfg.scan_date
    if not scan_dt:
        # 推断最新日期
        for sym in symbols[:10]:
            df = _load_daily(cfg.data_dir, sym)
            if df is not None and len(df) > 0:
                scan_dt = str(df["trade_date"].iloc[-1])
                break

    logger.info("Scanning %d symbols for date %s", len(symbols), scan_dt)

    results = []
    for i, sym in enumerate(symbols):
        if (i + 1) % 500 == 0:
            logger.info("  progress: %d / %d", i + 1, len(symbols))
        sig = scan_single_symbol(cfg.data_dir, sym, cfg, scan_dt, basic_info)
        if sig is not None:
            results.append(sig)

    if not results:
        logger.warning("No valid symbols found.")
        return pd.DataFrame()

    # 构建结果表
    rows = []
    for s in results:
        row = {
            "symbol": s.symbol,
            "trade_date": s.trade_date,
            "phase": s.phase,
            "accum_quality": s.accum_quality,
            "bifurc_quality": s.bifurc_quality,
            "composite_score": s.composite_score,
            "entry_signal": s.entry_signal,
            "name": s.details.get("name", ""),
            "industry": s.details.get("industry", ""),
        }
        for k, v in s.details.items():
            if k not in ("name", "industry"):
                row[k] = v
        rows.append(row)

    df_result = pd.DataFrame(rows)

    # 排序并截取 top N
    df_result = df_result.sort_values("composite_score", ascending=False).reset_index(drop=True)

    # 只保留有信号的
    df_signals = df_result[df_result["phase"].isin(["breakout", "accumulation"])].copy()
    df_breakout = df_signals[df_signals["phase"] == "breakout"].head(cfg.top_n)

    return df_result, df_breakout


# ═════════════════════════════════════════════════════════
# 前瞻回测
# ═════════════════════════════════════════════════════════

@dataclass
class Trade:
    symbol: str
    entry_date: str
    entry_price: float
    exit_date: str = ""
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl_pct: float = 0.0
    hold_days: int = 0


def run_backtest(
    cfg: ScanConfig,
    scan_dates: list[str] | None = None,
) -> tuple[pd.DataFrame, list[Trade]]:
    """
    前瞻回测: 在每个 scan_date 扫描 → 选股 → 持有 hold_days → 退出.
    退出策略包含结构崩塌检测.
    """
    basic_info = _load_basic_info(cfg.basic_path)
    all_symbols = _resolve_symbols(cfg.data_dir, None)

    # 交易日历
    cal_dates = set()
    for sym in all_symbols[:5]:
        df = _load_daily(cfg.data_dir, sym)
        if df is not None:
            for d in df["trade_date"].astype(str):
                cal_dates.add(d)
    cal_dates = sorted(cal_dates)

    # 确定回测日期范围
    start = cfg.backtest_start_date
    end = cfg.backtest_end_date
    if not start:
        start = cal_dates[max(0, len(cal_dates) - 250)]
    if not end:
        end = cal_dates[-1]
    bt_dates = [d for d in cal_dates if start <= d <= end]

    if scan_dates is None:
        # 每 hold_days 天扫描一次
        scan_dates = bt_dates[::cfg.hold_days]

    trades: list[Trade] = []
    equity = [1.0]
    equity_dates = [start]

    for sd in scan_dates:
        if sd > end:
            break

        cfg_scan = ScanConfig(
            data_dir=cfg.data_dir,
            basic_path=cfg.basic_path,
            scan_date=sd,
            symbols=cfg.symbols,
            lookback_days=cfg.lookback_days,
            min_amount=cfg.min_amount,
            exclude_st=cfg.exclude_st,
            top_n=cfg.top_n,
            detector=cfg.detector,
        )
        try:
            all_result, top_picks = run_scan(cfg_scan)
        except Exception as e:
            logger.warning("Scan failed for %s: %s", sd, e)
            continue

        if len(top_picks) == 0:
            continue

        # 取前 max_positions 只
        picks = top_picks.head(cfg.max_positions)

        # 模拟持有
        batch_pnls = []
        for _, row in picks.iterrows():
            sym = row["symbol"]
            df_full = _load_daily(cfg.data_dir, sym)
            if df_full is None:
                continue

            df_after = df_full[df_full["trade_date"].astype(str) > sd].head(cfg.hold_days + 1)
            if len(df_after) < 2:
                continue

            entry_price = float(df_after["close"].iloc[0])

            # 持有期间做结构崩塌检测
            exit_idx = len(df_after) - 1
            exit_reason = "hold_days"

            # 简易崩塌检测: 看持有期间熵是否飙升
            if len(df_after) >= 5:
                df_hold_ctx = df_full[df_full["trade_date"].astype(str) <= str(df_after["trade_date"].iloc[-1])].tail(cfg.lookback_days)
                if len(df_hold_ctx) >= 60:
                    try:
                        feats_hold = build_features(df_hold_ctx)
                        collapse = detect_structural_collapse(
                            feats_hold["daily"], len(feats_hold["daily"]) - len(df_after), cfg.detector
                        )
                        # 找到第一个崩塌信号
                        collapse_tail = collapse.iloc[-len(df_after):]
                        collapse_idx = collapse_tail[collapse_tail].index
                        if len(collapse_idx) > 0:
                            first_collapse = collapse_idx[0]
                            relative = first_collapse - collapse_tail.index[0]
                            if relative < exit_idx:
                                exit_idx = relative
                                exit_reason = "collapse"
                    except Exception:
                        pass

            exit_price = float(df_after["close"].iloc[min(exit_idx, len(df_after) - 1)])
            pnl = (exit_price - entry_price) / entry_price

            trade = Trade(
                symbol=sym,
                entry_date=str(df_after["trade_date"].iloc[0]),
                entry_price=entry_price,
                exit_date=str(df_after["trade_date"].iloc[min(exit_idx, len(df_after) - 1)]),
                exit_price=exit_price,
                exit_reason=exit_reason,
                pnl_pct=round(pnl * 100, 2),
                hold_days=min(exit_idx, len(df_after) - 1),
            )
            trades.append(trade)
            batch_pnls.append(pnl)

        if batch_pnls:
            avg_pnl = np.mean(batch_pnls)
            new_eq = equity[-1] * (1 + avg_pnl)
            equity.append(new_eq)
            equity_dates.append(sd)

    # 汇总
    df_equity = pd.DataFrame({"date": equity_dates, "equity": equity})
    return df_equity, trades


# ═════════════════════════════════════════════════════════
# 输出
# ═════════════════════════════════════════════════════════

def write_results(
    out_dir: str,
    scan_date: str,
    all_result: pd.DataFrame,
    top_picks: pd.DataFrame,
    df_equity: pd.DataFrame | None = None,
    trades: list[Trade] | None = None,
):
    """将结果写入 CSV"""
    os.makedirs(out_dir, exist_ok=True)

    if len(all_result) > 0:
        all_result.to_csv(
            os.path.join(out_dir, f"market_snapshot_{scan_date}.csv"),
            index=False,
        )

    if len(top_picks) > 0:
        top_picks.to_csv(
            os.path.join(out_dir, f"breakout_candidates_{scan_date}.csv"),
            index=False,
        )

    if df_equity is not None and len(df_equity) > 0:
        df_equity.to_csv(
            os.path.join(out_dir, f"backtest_equity_{scan_date}.csv"),
            index=False,
        )

    if trades:
        rows = [
            {
                "symbol": t.symbol,
                "entry_date": t.entry_date,
                "entry_price": t.entry_price,
                "exit_date": t.exit_date,
                "exit_price": t.exit_price,
                "exit_reason": t.exit_reason,
                "pnl_pct": t.pnl_pct,
                "hold_days": t.hold_days,
            }
            for t in trades
        ]
        pd.DataFrame(rows).to_csv(
            os.path.join(out_dir, f"backtest_trades_{scan_date}.csv"),
            index=False,
        )

    logger.info("Results written to %s", out_dir)
