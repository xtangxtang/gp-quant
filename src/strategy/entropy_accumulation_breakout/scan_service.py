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
from .signal_detector_v2 import (
    DetectorConfigV2 as DetectorConfig,
    SymbolSignal,
    detect_exit_v2 as detect_structural_collapse,
    evaluate_symbol,
)
from src.core.tick_entropy import permutation_entropy as _pe_fast

logger = logging.getLogger(__name__)

# ── 多进程共享状态 (fork COW) ──
_BT_CACHE: dict[str, pd.DataFrame] = {}
_BT_BASIC: dict[str, dict] = {}
_BT_DETECTOR: "DetectorConfig | None" = None
_BT_DATA_ROOT: str = ""
_BT_CACHE_DIR: str = ""


# ═════════════════════════════════════════════════════════
# 配置
# ═════════════════════════════════════════════════════════

@dataclass
class ScanConfig:
    data_dir: str = ""
    data_root: str = ""  # 数据根目录 (包含 tushare-weekly-5d/, trade/, tushare-moneyflow/)
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

    # 特征缓存
    feature_cache_dir: str = ""  # 为空时不使用缓存

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

        # 自动推断 data_root: data_dir 通常是 .../tushare-daily-full
        if not self.data_root and self.data_dir:
            import os
            parent = os.path.dirname(self.data_dir.rstrip("/"))
            # 验证 parent 包含周线/分钟/资金流目录之一
            for sub in ("tushare-weekly-5d", "trade", "tushare-moneyflow"):
                if os.path.isdir(os.path.join(parent, sub)):
                    self.data_root = parent
                    break


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
    feats = build_features(df, data_root=cfg.data_root, symbol=symbol, cache_dir=cfg.feature_cache_dir)
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


def _bt_scan_worker(args: tuple) -> "SymbolSignal | None":
    """Worker: 在子进程中对单只股票做快速特征计算+信号检测 (回测专用)"""
    sym, sd, bt_lookback, min_amount = args
    df_full = _BT_CACHE.get(sym)
    if df_full is None:
        return None

    df_cut = df_full[df_full["trade_date"].astype(str) <= sd]
    if len(df_cut) < 120:
        return None

    # 流动性过滤
    recent = df_cut.tail(20)
    if "amount" in recent.columns and recent["amount"].mean() < min_amount:
        return None

    # 快速 PE 预过滤
    tail_close = df_cut["close"].astype(np.float64).values[-20:]
    if len(tail_close) >= 20:
        quick_pe = _pe_fast(tail_close, order=3)
        if quick_pe is not None and not np.isnan(quick_pe) and quick_pe > 0.85:
            return None

    df_slice = df_cut.tail(bt_lookback).reset_index(drop=True)
    if len(df_slice) < 60:
        return None

    try:
        feats = build_features(df_slice, skip_weekly=True, data_root=_BT_DATA_ROOT, symbol=sym, cache_dir=_BT_CACHE_DIR)
        sig = evaluate_symbol(feats["daily"], feats.get("weekly"), sym, _BT_DETECTOR)
        info = _BT_BASIC.get(sym, {})
        sig.details["name"] = info.get("name", "")
        sig.details["industry"] = info.get("industry", "")
        if sig.phase in ("breakout", "accumulation"):
            return sig
    except Exception:
        pass
    return None


def run_backtest(
    cfg: ScanConfig,
    scan_dates: list[str] | None = None,
) -> tuple[pd.DataFrame, list[Trade]]:
    """
    前瞻回测: 在每个 scan_date 扫描 → 选股 → 持有 hold_days → 退出.

    优化: 预加载 CSV 到内存, 避免每轮重复磁盘 IO.
    """
    import time as _time
    import multiprocessing as _mp

    basic_info = _load_basic_info(cfg.basic_path)
    all_symbols = _resolve_symbols(cfg.data_dir, cfg.symbols if cfg.symbols else None)

    # ── 预加载 CSV (一次 IO, 多轮复用) ──
    t_load = _time.time()
    logger.info("Pre-loading %d symbol CSVs into memory …", len(all_symbols))
    data_cache: dict[str, pd.DataFrame] = {}
    for sym in all_symbols:
        df = _load_daily(cfg.data_dir, sym)
        if df is None or len(df) < 120:
            continue
        info = basic_info.get(sym, {})
        name = info.get("name", "")
        if cfg.exclude_st and ("ST" in name or "退" in name):
            continue
        data_cache[sym] = df
    logger.info("Loaded %d symbols in %.1fs", len(data_cache), _time.time() - t_load)

    # 交易日历
    cal_dates: set[str] = set()
    for sym in list(data_cache.keys())[:10]:
        for d in data_cache[sym]["trade_date"].astype(str):
            cal_dates.add(d)
    cal_dates_sorted = sorted(cal_dates)

    # 日期范围
    start = cfg.backtest_start_date
    end = cfg.backtest_end_date
    if not start:
        start = cal_dates_sorted[max(0, len(cal_dates_sorted) - 250)]
    if not end:
        end = cal_dates_sorted[-1]
    bt_dates = [d for d in cal_dates_sorted if start <= d <= end]

    if scan_dates is None:
        scan_dates = bt_dates[::cfg.hold_days]

    logger.info("Backtest %s→%s, %d rounds (hold_days=%d), %d symbols",
                start, end, len(scan_dates), cfg.hold_days, len(data_cache))

    # ── 将缓存写入模块级全局变量 (fork COW 共享) ──
    global _BT_CACHE, _BT_BASIC, _BT_DETECTOR, _BT_DATA_ROOT, _BT_CACHE_DIR
    _BT_CACHE = data_cache
    _BT_BASIC = basic_info
    _BT_DETECTOR = cfg.detector
    _BT_DATA_ROOT = cfg.data_root
    _BT_CACHE_DIR = cfg.feature_cache_dir

    n_workers = min(16, max(1, (_mp.cpu_count() or 4) // 2))
    pool = _mp.Pool(n_workers)
    logger.info("Using %d worker processes", n_workers)

    trades: list[Trade] = []
    equity = [1.0]
    equity_dates = [start]

    for round_idx, sd in enumerate(scan_dates):
        if sd > end:
            break

        t_round = _time.time()

        # ── 并行扫描: 用 multiprocessing Pool ──
        bt_lookback = 200
        tasks = [(sym, sd, bt_lookback, cfg.min_amount) for sym in data_cache.keys()]
        raw_results = pool.map(_bt_scan_worker, tasks, chunksize=64)
        signals = [sig for sig in raw_results if sig is not None]

        elapsed_round = _time.time() - t_round
        logger.info("  round %d/%d  date=%s  signals=%d  %.1fs",
                     round_idx + 1, len(scan_dates), sd, len(signals), elapsed_round)

        if not signals:
            continue

        # 按综合分排序, 优先 breakout
        signals.sort(key=lambda s: s.composite_score, reverse=True)
        picks = [s for s in signals if s.phase == "breakout"][:cfg.max_positions]
        if not picks:
            picks = signals[:cfg.max_positions]

        # ── 模拟持有 ──
        batch_pnls = []
        for sig in picks:
            sym = sig.symbol
            df_full = data_cache[sym]
            df_after = df_full[df_full["trade_date"].astype(str) > sd].head(cfg.hold_days + 1)
            if len(df_after) < 2:
                continue

            entry_price = float(df_after["close"].iloc[0])
            exit_idx = len(df_after) - 1
            exit_reason = "hold_days"

            # 止损
            for i in range(1, len(df_after)):
                pnl_i = (float(df_after["close"].iloc[i]) - entry_price) / entry_price
                if pnl_i <= -0.10:
                    exit_idx = i
                    exit_reason = "stop_loss"
                    break

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

    pool.close()
    pool.join()
    # 清理全局引用
    _BT_CACHE.clear()

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
