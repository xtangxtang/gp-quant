"""
个股因子有效性画像 (Per-Stock Factor Profiling)

从特征缓存数据分析每只股票各因子对未来涨跌的预测能力。
详见 FACTOR_PROFILING.md。

用法:
  python -m src.strategy.entropy_accumulation_breakout.factor_profiling \
    --cache_dir /path/to/feature-cache \
    --out_dir results/factor_profiling \
    [--forward_days 3] [--decay_lambda 0.007] [--symbols sh600519] [--top_n 50]
"""

import argparse
import json
import os
import sys
import glob
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 因子列表 — 与策略特征对齐
# ---------------------------------------------------------------------------

CORE_FACTORS = [
    "perm_entropy_s", "perm_entropy_m", "perm_entropy_l",
    "entropy_slope", "entropy_accel",
    "path_irrev_m", "path_irrev_l",
    "dom_eig_m", "dom_eig_l",
    "turnover_entropy_m", "turnover_entropy_l",
    "volatility_m", "volatility_l",
    "vol_compression", "bbw_pctl",
]

VOLUME_FACTORS = [
    "vol_ratio_s", "vol_impulse", "vol_shrink", "breakout_range",
]

MONEYFLOW_FACTORS = [
    "mf_big_net", "mf_big_net_ratio",
    "mf_big_cumsum_s", "mf_big_cumsum_m", "mf_big_cumsum_l",
    "mf_sm_proportion", "mf_flow_imbalance",
    "mf_big_momentum", "mf_big_streak",
]

QUANTUM_FACTORS = [
    "coherence_l1", "purity_norm", "von_neumann_entropy", "coherence_decay_rate",
]

WEEKLY_EXTRA_FACTORS = [
    "pe_ttm_pctl", "pb_pctl",
    "weekly_big_net", "weekly_big_net_cumsum",
    "weekly_turnover_ma4", "weekly_turnover_shrink",
]

ALL_FACTORS = CORE_FACTORS + VOLUME_FACTORS + MONEYFLOW_FACTORS + QUANTUM_FACTORS

# 周线分析用的因子：核心因子 + 周线特有因子
WEEKLY_FACTORS = CORE_FACTORS + VOLUME_FACTORS + QUANTUM_FACTORS + WEEKLY_EXTRA_FACTORS


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class ProfilingConfig:
    cache_dir: str = ""
    out_dir: str = "results/factor_profiling"
    forward_days: list = field(default_factory=lambda: [1, 3, 5])
    forward_weeks: list = field(default_factory=lambda: [1, 3, 5])
    decay_lambda: float = 0.007       # ~100天半衰期
    weekly_decay_lambda: float = 0.035 # ~20周半衰期 (与日线等效)
    min_rows: int = 120               # 最少数据行数
    min_weekly_rows: int = 30          # 周线最少行数
    ic_threshold: float = 0.05        # IC 绝对值门槛
    ic_ir_threshold: float = 0.5      # IC_IR 稳定性门槛
    rolling_window: int = 60          # 滚动 IC 窗口
    weekly_rolling_window: int = 15   # 周线滚动 IC 窗口
    n_quintiles: int = 5              # 分箱数
    workers: int = 16                 # 并行进程数
    symbols: list = field(default_factory=list)
    top_n: int = 0                    # >0: 只分析成交额 top N
    basic_path: str = ""              # 股票基本信息 CSV


# ---------------------------------------------------------------------------
# 核心计算
# ---------------------------------------------------------------------------

def weighted_rank_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """加权 Spearman Rank IC。"""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 30:
        return np.nan
    x, y, w = x[mask], y[mask], w[mask]
    # 转为秩
    rx = pd.Series(x).rank().values
    ry = pd.Series(y).rank().values
    # 加权相关
    w_sum = w.sum()
    mx = np.average(rx, weights=w)
    my = np.average(ry, weights=w)
    cov = np.sum(w * (rx - mx) * (ry - my)) / w_sum
    sx = np.sqrt(np.sum(w * (rx - mx) ** 2) / w_sum)
    sy = np.sqrt(np.sum(w * (ry - my) ** 2) / w_sum)
    if sx < 1e-12 or sy < 1e-12:
        return np.nan
    return cov / (sx * sy)


def compute_rolling_ic(factor_vals: np.ndarray, ret_vals: np.ndarray,
                       window: int = 60) -> np.ndarray:
    """滚动窗口 Rank IC 序列（无加权，用于 IC_IR 计算）。"""
    n = len(factor_vals)
    min_valid = min(20, max(8, int(window * 0.6)))
    ics = []
    for i in range(window, n):
        x = factor_vals[i - window:i]
        y = ret_vals[i - window:i]
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < min_valid:
            ics.append(np.nan)
            continue
        rx = pd.Series(x[mask]).rank().values
        ry = pd.Series(y[mask]).rank().values
        corr = np.corrcoef(rx, ry)[0, 1]
        ics.append(corr)
    return np.array(ics)


def quintile_returns(factor_vals: np.ndarray, ret_vals: np.ndarray,
                     w: np.ndarray, n_q: int = 5) -> dict:
    """分箱分析：按因子值分 n_q 档，计算每档加权平均收益。"""
    mask = np.isfinite(factor_vals) & np.isfinite(ret_vals)
    if mask.sum() < n_q * 10:
        return {}
    f, r, ww = factor_vals[mask], ret_vals[mask], w[mask]
    # 分箱
    try:
        bins = pd.qcut(pd.Series(f), n_q, labels=False, duplicates="drop")
    except ValueError:
        return {}
    result = {}
    for q in sorted(bins.unique()):
        idx = bins.values == q
        if idx.sum() > 0:
            result[f"Q{q+1}"] = float(np.average(r[idx], weights=ww[idx]))
    return result


def analyze_one_factor(df: pd.DataFrame, factor: str, ret_col: str,
                       weights: np.ndarray, cfg: ProfilingConfig) -> Optional[dict]:
    """分析单个因子对单只股票的预测能力。"""
    if factor not in df.columns:
        return None

    f_vals = df[factor].values.astype(float)
    r_vals = df[ret_col].values.astype(float)

    # 全局加权 IC
    ic = weighted_rank_corr(f_vals, r_vals, weights)
    if np.isnan(ic):
        return None

    # 滚动 IC → IC_IR
    rolling_ics = compute_rolling_ic(f_vals, r_vals, cfg.rolling_window)
    valid_ics = rolling_ics[np.isfinite(rolling_ics)]
    if len(valid_ics) >= 3:
        ic_ir = float(np.mean(valid_ics) / (np.std(valid_ics) + 1e-12))
    else:
        ic_ir = 0.0

    # 近期 IC（最近 60 天，无衰减）
    recent_n = min(60, len(f_vals))
    recent_f = f_vals[-recent_n:]
    recent_r = r_vals[-recent_n:]
    mask = np.isfinite(recent_f) & np.isfinite(recent_r)
    if mask.sum() >= 20:
        rx = pd.Series(recent_f[mask]).rank().values
        ry = pd.Series(recent_r[mask]).rank().values
        recent_ic = float(np.corrcoef(rx, ry)[0, 1])
    else:
        recent_ic = np.nan

    # 分箱分析
    q_ret = quintile_returns(f_vals, r_vals, weights, cfg.n_quintiles)

    # 单调性检测
    monotonic = ""
    if len(q_ret) >= 4:
        vals = list(q_ret.values())
        diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
        if all(d > 0 for d in diffs):
            monotonic = "increasing"
        elif all(d < 0 for d in diffs):
            monotonic = "decreasing"
        elif len(diffs) >= 3:
            mid = len(vals) // 2
            if vals[0] > vals[mid] < vals[-1]:
                monotonic = "U-shape"
            elif vals[0] < vals[mid] > vals[-1]:
                monotonic = "inverted-U"

    return {
        "factor": factor,
        "ic": round(ic, 4),
        "ic_ir": round(ic_ir, 4),
        "recent_ic": round(recent_ic, 4) if np.isfinite(recent_ic) else None,
        "abs_ic": round(abs(ic), 4),
        "monotonic": monotonic,
        "quintile_returns": {k: round(v, 4) for k, v in q_ret.items()},
    }


# ---------------------------------------------------------------------------
# 个股分析
# ---------------------------------------------------------------------------

def profile_one_stock(args: tuple) -> Optional[dict]:
    """分析单只股票所有因子（日线+周线）。在子进程中运行。"""
    csv_path, cfg_dict = args
    cfg = ProfilingConfig(**cfg_dict)

    symbol = os.path.basename(csv_path).replace(".csv", "")
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    if len(df) < cfg.min_rows:
        return None

    df = df.sort_values("trade_date").reset_index(drop=True)

    # 计算前瞻收益
    for fd in cfg.forward_days:
        df[f"fwd_ret_{fd}d"] = df["close"].shift(-fd) / df["close"] - 1

    # 时间衰减权重
    T = len(df) - 1
    t = np.arange(len(df), dtype=float)
    weights = np.exp(-cfg.decay_lambda * (T - t))

    # 主分析：使用第一个 forward_days
    primary_fd = cfg.forward_days[0] if len(cfg.forward_days) == 1 else cfg.forward_days[1]  # 默认用 3d
    ret_col = f"fwd_ret_{primary_fd}d"

    # 分析所有因子
    factor_results = []
    for factor in ALL_FACTORS:
        res = analyze_one_factor(df, factor, ret_col, weights, cfg)
        if res is not None:
            factor_results.append(res)

    if not factor_results:
        return None

    # 排序: 按 |IC| 降序
    factor_results.sort(key=lambda x: x["abs_ic"], reverse=True)

    # 分类
    positive = [r for r in factor_results
                if r["ic"] > cfg.ic_threshold and r["ic_ir"] > cfg.ic_ir_threshold]
    negative = [r for r in factor_results
                if r["ic"] < -cfg.ic_threshold and r["ic_ir"] > cfg.ic_ir_threshold]
    unstable = [r for r in factor_results
                if abs(r["ic"]) > cfg.ic_threshold and abs(r["ic_ir"]) <= cfg.ic_ir_threshold]

    # 多窗口 IC 汇总
    multi_fd_ic = {}
    for fd in cfg.forward_days:
        rc = f"fwd_ret_{fd}d"
        fd_ics = {}
        for factor in ALL_FACTORS:
            if factor in df.columns:
                ic_val = weighted_rank_corr(
                    df[factor].values.astype(float),
                    df[rc].values.astype(float),
                    weights,
                )
                if np.isfinite(ic_val):
                    fd_ics[factor] = round(ic_val, 4)
        multi_fd_ic[f"{fd}d"] = fd_ics

    # ---- 周线分析 ----
    weekly_profile = _profile_weekly(symbol, cfg)

    result = {
        "symbol": symbol,
        "data_range": [str(df["trade_date"].iloc[0]), str(df["trade_date"].iloc[-1])],
        "n_rows": len(df),
        "primary_forward_days": primary_fd,
        "decay_lambda": cfg.decay_lambda,
        "n_effective_positive": len(positive),
        "n_effective_negative": len(negative),
        "n_unstable": len(unstable),
        "top_positive_factors": positive[:5],
        "top_negative_factors": negative[:5],
        "unstable_factors": unstable[:3],
        "all_factor_ic": {r["factor"]: r["ic"] for r in factor_results},
        "multi_forward_ic": multi_fd_ic,
        "all_details": factor_results,
    }

    if weekly_profile is not None:
        result["weekly"] = weekly_profile

    return result


def _profile_weekly(symbol: str, cfg: ProfilingConfig) -> Optional[dict]:
    """分析单只股票的周线因子。"""
    weekly_csv = os.path.join(cfg.cache_dir, "weekly", f"{symbol}.csv")
    if not os.path.exists(weekly_csv):
        return None

    try:
        wdf = pd.read_csv(weekly_csv)
    except Exception:
        return None

    if len(wdf) < cfg.min_weekly_rows:
        return None

    wdf = wdf.sort_values("trade_date").reset_index(drop=True)

    # 计算前瞻周收益
    for fw in cfg.forward_weeks:
        wdf[f"fwd_ret_{fw}w"] = wdf["close"].shift(-fw) / wdf["close"] - 1

    # 时间衰减权重（周尺度）
    T = len(wdf) - 1
    t = np.arange(len(wdf), dtype=float)
    w_weights = np.exp(-cfg.weekly_decay_lambda * (T - t))

    # 用 cfg 但调整 rolling_window
    wcfg = ProfilingConfig(**{
        k: getattr(cfg, k) for k in cfg.__dataclass_fields__
    })
    wcfg.rolling_window = cfg.weekly_rolling_window

    # 主分析窗口：默认用 3w
    primary_fw = cfg.forward_weeks[0] if len(cfg.forward_weeks) == 1 else cfg.forward_weeks[1]
    ret_col = f"fwd_ret_{primary_fw}w"

    # 分析周线因子
    factor_results = []
    for factor in WEEKLY_FACTORS:
        res = analyze_one_factor(wdf, factor, ret_col, w_weights, wcfg)
        if res is not None:
            factor_results.append(res)

    if not factor_results:
        return None

    factor_results.sort(key=lambda x: x["abs_ic"], reverse=True)

    positive = [r for r in factor_results
                if r["ic"] > cfg.ic_threshold and r["ic_ir"] > cfg.ic_ir_threshold]
    negative = [r for r in factor_results
                if r["ic"] < -cfg.ic_threshold and r["ic_ir"] > cfg.ic_ir_threshold]

    # 多周期 IC
    multi_fw_ic = {}
    for fw in cfg.forward_weeks:
        rc = f"fwd_ret_{fw}w"
        fw_ics = {}
        for factor in WEEKLY_FACTORS:
            if factor in wdf.columns:
                ic_val = weighted_rank_corr(
                    wdf[factor].values.astype(float),
                    wdf[rc].values.astype(float),
                    w_weights,
                )
                if np.isfinite(ic_val):
                    fw_ics[factor] = round(ic_val, 4)
        multi_fw_ic[f"{fw}w"] = fw_ics

    return {
        "data_range": [str(wdf["trade_date"].iloc[0]), str(wdf["trade_date"].iloc[-1])],
        "n_rows": len(wdf),
        "primary_forward_weeks": primary_fw,
        "n_effective_positive": len(positive),
        "n_effective_negative": len(negative),
        "top_positive_factors": positive[:5],
        "top_negative_factors": negative[:5],
        "all_factor_ic": {r["factor"]: r["ic"] for r in factor_results},
        "multi_forward_ic": multi_fw_ic,
        "all_details": factor_results,
    }


# ---------------------------------------------------------------------------
# 全市场汇总
# ---------------------------------------------------------------------------

def aggregate_results(all_results: list, cfg: ProfilingConfig, basic_df: Optional[pd.DataFrame]):
    """汇总 → 全市场 CSV + 行业汇总。"""

    # ------ 因子有效性统计 ------
    factor_stats = {f: {"positive": 0, "negative": 0, "unstable": 0,
                        "insignificant": 0, "ic_list": [], "ic_ir_list": []}
                    for f in ALL_FACTORS}

    for res in all_results:
        for detail in res["all_details"]:
            f = detail["factor"]
            if f not in factor_stats:
                continue
            ic = detail["ic"]
            ic_ir = detail["ic_ir"]
            factor_stats[f]["ic_list"].append(ic)
            factor_stats[f]["ic_ir_list"].append(ic_ir)
            if abs(ic) <= cfg.ic_threshold:
                factor_stats[f]["insignificant"] += 1
            elif abs(ic_ir) <= cfg.ic_ir_threshold:
                factor_stats[f]["unstable"] += 1
            elif ic > 0:
                factor_stats[f]["positive"] += 1
            else:
                factor_stats[f]["negative"] += 1

    rows = []
    for f in ALL_FACTORS:
        s = factor_stats[f]
        total = s["positive"] + s["negative"] + s["unstable"] + s["insignificant"]
        ics = np.array(s["ic_list"])
        ic_irs = np.array(s["ic_ir_list"])
        effective = s["positive"] + s["negative"]
        rows.append({
            "factor": f,
            "n_positive": s["positive"],
            "n_negative": s["negative"],
            "n_unstable": s["unstable"],
            "n_insignificant": s["insignificant"],
            "n_total": total,
            "pct_effective": round(effective / total, 4) if total > 0 else 0,
            "median_ic": round(float(np.median(ics)), 4) if len(ics) > 0 else 0,
            "median_abs_ic": round(float(np.median(np.abs(ics))), 4) if len(ics) > 0 else 0,
            "mean_ic": round(float(np.mean(ics)), 4) if len(ics) > 0 else 0,
            "median_ic_ir": round(float(np.median(ic_irs)), 4) if len(ic_irs) > 0 else 0,
            "direction_consensus": "positive" if s["positive"] > s["negative"] * 2
                                   else ("negative" if s["negative"] > s["positive"] * 2
                                         else "mixed"),
        })

    factor_summary = pd.DataFrame(rows).sort_values("pct_effective", ascending=False)

    # ------ 行业汇总 ------
    industry_summary = None
    if basic_df is not None:
        # 构建 symbol → industry 映射
        sym_industry = {}
        for _, row in basic_df.iterrows():
            ts = str(row.get("ts_code", ""))
            ind = str(row.get("industry", ""))
            if ts and ind:
                # ts_code 格式: 600519.SH → sh600519
                parts = ts.split(".")
                if len(parts) == 2:
                    sym = parts[1].lower() + parts[0]
                    sym_industry[sym] = ind

        # 每个行业每个因子的平均 IC
        ind_rows = []
        for res in all_results:
            ind = sym_industry.get(res["symbol"], "unknown")
            for detail in res["all_details"]:
                ind_rows.append({
                    "industry": ind,
                    "factor": detail["factor"],
                    "ic": detail["ic"],
                    "ic_ir": detail["ic_ir"],
                })

        if ind_rows:
            ind_df = pd.DataFrame(ind_rows)
            industry_summary = ind_df.groupby(["industry", "factor"]).agg(
                mean_ic=("ic", "mean"),
                median_ic=("ic", "median"),
                mean_ic_ir=("ic_ir", "mean"),
                n_stocks=("ic", "count"),
            ).reset_index()
            industry_summary = industry_summary.round(4)

    return factor_summary, industry_summary


def aggregate_weekly_results(weekly_results: list, cfg: ProfilingConfig) -> pd.DataFrame:
    """汇总周线因子有效性统计。"""
    factor_stats = {f: {"positive": 0, "negative": 0, "unstable": 0,
                        "insignificant": 0, "ic_list": [], "ic_ir_list": []}
                    for f in WEEKLY_FACTORS}

    for res in weekly_results:
        w = res["weekly"]
        for detail in w.get("all_details", []):
            f = detail["factor"]
            if f not in factor_stats:
                continue
            ic = detail["ic"]
            ic_ir = detail["ic_ir"]
            factor_stats[f]["ic_list"].append(ic)
            factor_stats[f]["ic_ir_list"].append(ic_ir)
            if abs(ic) <= cfg.ic_threshold:
                factor_stats[f]["insignificant"] += 1
            elif abs(ic_ir) <= cfg.ic_ir_threshold:
                factor_stats[f]["unstable"] += 1
            elif ic > 0:
                factor_stats[f]["positive"] += 1
            else:
                factor_stats[f]["negative"] += 1

    rows = []
    for f in WEEKLY_FACTORS:
        s = factor_stats[f]
        total = len(s["ic_list"])
        ics = np.array(s["ic_list"])
        effective = s["positive"] + s["negative"]
        rows.append({
            "factor": f,
            "n_positive": s["positive"],
            "n_negative": s["negative"],
            "n_unstable": s["unstable"],
            "n_insignificant": s["insignificant"],
            "n_total": total,
            "pct_effective": round(effective / total, 4) if total > 0 else 0,
            "median_ic": round(float(np.median(ics)), 4) if len(ics) > 0 else 0,
            "median_abs_ic": round(float(np.median(np.abs(ics))), 4) if len(ics) > 0 else 0,
            "mean_ic": round(float(np.mean(ics)), 4) if len(ics) > 0 else 0,
            "direction_consensus": "positive" if s["positive"] > s["negative"] * 2
                                   else ("negative" if s["negative"] > s["positive"] * 2
                                         else "mixed"),
        })

    return pd.DataFrame(rows).sort_values("pct_effective", ascending=False)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def run_profiling(cfg: ProfilingConfig):
    """执行全市场因子画像分析。"""
    os.makedirs(cfg.out_dir, exist_ok=True)

    # 收集 CSV 文件
    daily_dir = os.path.join(cfg.cache_dir, "daily")
    all_csvs = sorted(glob.glob(os.path.join(daily_dir, "*.csv")))
    logger.info(f"找到 {len(all_csvs)} 只股票的特征缓存")

    # 过滤
    if cfg.symbols:
        sym_set = set(cfg.symbols)
        all_csvs = [p for p in all_csvs
                    if os.path.basename(p).replace(".csv", "") in sym_set]
        logger.info(f"过滤后 {len(all_csvs)} 只")

    # top_n 过滤：按最近一行 amount 排序
    if cfg.top_n > 0 and not cfg.symbols:
        amounts = []
        for p in all_csvs:
            try:
                tail = pd.read_csv(p, skiprows=lambda i: i not in [0] and i < max(0, sum(1 for _ in open(p)) - 3),
                                   nrows=3)
                amt = tail["amount"].iloc[-1] if "amount" in tail.columns else 0
                amounts.append((p, amt))
            except Exception:
                amounts.append((p, 0))
        amounts.sort(key=lambda x: x[1], reverse=True)
        all_csvs = [p for p, _ in amounts[:cfg.top_n]]
        logger.info(f"取成交额 Top {cfg.top_n}, 实际 {len(all_csvs)} 只")

    if not all_csvs:
        logger.error("无可分析股票")
        return

    # 加载行业信息
    basic_df = None
    if cfg.basic_path and os.path.exists(cfg.basic_path):
        basic_df = pd.read_csv(cfg.basic_path, dtype=str)

    # 并行分析
    cfg_dict = {
        "cache_dir": cfg.cache_dir,
        "out_dir": cfg.out_dir,
        "forward_days": cfg.forward_days,
        "forward_weeks": cfg.forward_weeks,
        "decay_lambda": cfg.decay_lambda,
        "weekly_decay_lambda": cfg.weekly_decay_lambda,
        "min_rows": cfg.min_rows,
        "min_weekly_rows": cfg.min_weekly_rows,
        "ic_threshold": cfg.ic_threshold,
        "ic_ir_threshold": cfg.ic_ir_threshold,
        "rolling_window": cfg.rolling_window,
        "weekly_rolling_window": cfg.weekly_rolling_window,
        "n_quintiles": cfg.n_quintiles,
        "workers": cfg.workers,
        "symbols": cfg.symbols,
        "top_n": cfg.top_n,
        "basic_path": cfg.basic_path,
    }

    all_results = []
    n_total = len(all_csvs)
    logger.info(f"开始分析 {n_total} 只股票 (workers={cfg.workers})")

    with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
        futures = {pool.submit(profile_one_stock, (p, cfg_dict)): p for p in all_csvs}
        done = 0
        for future in as_completed(futures):
            done += 1
            if done % 500 == 0 or done == n_total:
                logger.info(f"  进度: {done}/{n_total}")
            try:
                res = future.result()
                if res is not None:
                    all_results.append(res)
            except Exception as e:
                sym = os.path.basename(futures[future])
                logger.warning(f"  {sym} 失败: {e}")

    logger.info(f"成功分析 {len(all_results)}/{n_total} 只")

    if not all_results:
        logger.error("无有效结果")
        return

    # ------ 输出个股 JSON (存到 feature-cache) ------
    today_str = datetime.now().strftime("%Y%m%d")
    stock_dir = os.path.join(cfg.cache_dir, "factor_profile", today_str)
    os.makedirs(stock_dir, exist_ok=True)
    for res in all_results:
        fpath = os.path.join(stock_dir, f"{res['symbol']}.json")
        # 只输出摘要，不输出完整 all_details (日线 + 周线)
        output = {k: v for k, v in res.items() if k != "all_details"}
        if "weekly" in output and isinstance(output["weekly"], dict):
            output["weekly"] = {k: v for k, v in output["weekly"].items()
                                if k != "all_details"}
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"个股画像已写入 {stock_dir}/ ({len(all_results)} 只)")

    # ------ 全市场汇总 ------
    factor_summary, industry_summary = aggregate_results(all_results, cfg, basic_df)
    factor_summary.to_csv(os.path.join(cfg.out_dir, "factor_summary.csv"), index=False)
    logger.info(f"因子汇总已写入 {cfg.out_dir}/factor_summary.csv")

    if industry_summary is not None:
        industry_summary.to_csv(os.path.join(cfg.out_dir, "industry_factor_summary.csv"), index=False)
        logger.info(f"行业汇总已写入 {cfg.out_dir}/industry_factor_summary.csv")

    # ------ 个股 IC 矩阵 (wide format) ------
    ic_rows = []
    for res in all_results:
        row = {"symbol": res["symbol"]}
        row.update(res["all_factor_ic"])
        ic_rows.append(row)
    ic_matrix = pd.DataFrame(ic_rows)
    ic_matrix.to_csv(os.path.join(cfg.out_dir, "stock_factor_ic_matrix.csv"), index=False)
    logger.info(f"IC 矩阵已写入 {cfg.out_dir}/stock_factor_ic_matrix.csv")

    # ------ 周线汇总 ------
    weekly_results = [res for res in all_results if res.get("weekly")]
    weekly_factor_summary = None
    weekly_ic_matrix = None
    if weekly_results:
        weekly_factor_summary = aggregate_weekly_results(weekly_results, cfg)
        weekly_factor_summary.to_csv(
            os.path.join(cfg.out_dir, "weekly_factor_summary.csv"), index=False)
        logger.info(f"周线因子汇总已写入 {cfg.out_dir}/weekly_factor_summary.csv")

        wic_rows = []
        for res in weekly_results:
            row = {"symbol": res["symbol"]}
            row.update(res["weekly"]["all_factor_ic"])
            wic_rows.append(row)
        weekly_ic_matrix = pd.DataFrame(wic_rows)
        weekly_ic_matrix.to_csv(
            os.path.join(cfg.out_dir, "weekly_ic_matrix.csv"), index=False)
        logger.info(f"周线 IC 矩阵已写入 {cfg.out_dir}/weekly_ic_matrix.csv")

    # ------ 打印摘要 ------
    print("\n" + "=" * 70)
    print("因子有效性汇总 (按全市场有效率排序)")
    print("=" * 70)
    print(f"{'因子':<25} {'有效率':>7} {'正向':>5} {'负向':>5} {'不稳定':>6} {'中位IC':>8} {'方向':>10}")
    print("-" * 70)
    for _, row in factor_summary.head(20).iterrows():
        print(f"{row['factor']:<25} {row['pct_effective']:>7.1%} "
              f"{row['n_positive']:>5} {row['n_negative']:>5} {row['n_unstable']:>6} "
              f"{row['median_ic']:>8.4f} {row['direction_consensus']:>10}")

    # 有效因子 Top 5
    print("\n--- 全市场最有效因子 Top 5 ---")
    for i, (_, row) in enumerate(factor_summary.head(5).iterrows()):
        direction = "↑涨" if row["direction_consensus"] == "positive" else (
            "↓跌" if row["direction_consensus"] == "negative" else "混合")
        print(f"  {i+1}. {row['factor']}: 有效率 {row['pct_effective']:.1%}, "
              f"中位IC {row['median_ic']:.4f}, {direction}")

    # 个股差异度
    ic_std_per_factor = {}
    for f in ALL_FACTORS:
        ics = [res["all_factor_ic"].get(f, np.nan) for res in all_results]
        ics = [x for x in ics if np.isfinite(x)]
        if ics:
            ic_std_per_factor[f] = np.std(ics)
    most_divergent = sorted(ic_std_per_factor.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\n--- 个股差异最大的因子 (IC 标准差最高) ---")
    for f, std in most_divergent:
        print(f"  {f}: IC std = {std:.4f} → 个性化空间大")

    # ------ 生成总结报告 ------
    generate_summary_report(cfg.out_dir, factor_summary, industry_summary,
                            ic_matrix, all_results, ic_std_per_factor,
                            weekly_factor_summary=weekly_factor_summary,
                            weekly_ic_matrix=weekly_ic_matrix)


# ---------------------------------------------------------------------------
# 总结报告生成
# ---------------------------------------------------------------------------

def generate_summary_report(out_dir: str, factor_summary: pd.DataFrame,
                            industry_summary: Optional[pd.DataFrame],
                            ic_matrix: pd.DataFrame,
                            all_results: list,
                            ic_std_per_factor: dict,
                            weekly_factor_summary: Optional[pd.DataFrame] = None,
                            weekly_ic_matrix: Optional[pd.DataFrame] = None):
    """从三个 CSV 生成 Markdown 总结报告。"""

    today = os.path.basename(out_dir) if os.path.basename(out_dir).isdigit() else datetime.now().strftime("%Y%m%d")
    n_stocks = len(all_results)

    lines = []
    lines.append(f"# 因子有效性画像日报 — {today}")
    lines.append("")
    lines.append(f"> 分析日期: {today} | 股票数: {n_stocks} | 因子数: {len(ALL_FACTORS)}")
    lines.append("")

    # =========== 1. 全市场因子有效率排名 ===========
    lines.append("## 1. 全市场因子有效率排名")
    lines.append("")
    lines.append("| 排名 | 因子 | 有效率 | 正向 | 负向 | 不稳定 | 中位IC | 方向共识 |")
    lines.append("|------|------|--------|------|------|--------|--------|----------|")
    for i, (_, row) in enumerate(factor_summary.iterrows()):
        direction_zh = {"positive": "正向↑", "negative": "负向↓", "mixed": "混合↔"}
        d = direction_zh.get(row["direction_consensus"], row["direction_consensus"])
        lines.append(
            f"| {i+1} | `{row['factor']}` | {row['pct_effective']:.1%} | "
            f"{row['n_positive']} | {row['n_negative']} | {row['n_unstable']} | "
            f"{row['median_ic']:.4f} | {d} |"
        )
    lines.append("")

    # =========== 2. 关键发现 ===========
    lines.append("## 2. 关键发现")
    lines.append("")

    # 强普适因子
    strong_universal = factor_summary[factor_summary["pct_effective"] >= 0.25]
    if len(strong_universal) > 0:
        lines.append("### 2.1 强普适因子 (有效率 ≥ 25%)")
        lines.append("")
        for _, row in strong_universal.iterrows():
            d = "越大越涨" if row["direction_consensus"] == "positive" else (
                "越大越跌" if row["direction_consensus"] == "negative" else "方向不一致")
            lines.append(f"- **`{row['factor']}`**: 有效率 {row['pct_effective']:.1%}, "
                         f"中位IC {row['median_ic']:+.4f} → {d}")
        lines.append("")

    # 个股差异大的因子
    lines.append("### 2.2 个股差异最大因子 (IC 标准差)")
    lines.append("")
    lines.append("这些因子在不同股票上表现截然不同，最适合做个性化参数。")
    lines.append("")
    divergent_sorted = sorted(ic_std_per_factor.items(), key=lambda x: x[1], reverse=True)
    for f, std in divergent_sorted[:10]:
        # 找到该因子的有效率
        row_match = factor_summary[factor_summary["factor"] == f]
        eff = row_match.iloc[0]["pct_effective"] if len(row_match) > 0 else 0
        lines.append(f"- `{f}`: IC std = {std:.4f}, 有效率 {eff:.1%}")
    lines.append("")

    # 策略核心因子评估
    lines.append("### 2.3 策略核心因子评估")
    lines.append("")
    lines.append("对比策略当前使用的信号条件与实际因子有效性：")
    lines.append("")
    strategy_factors = {
        "perm_entropy_m": "惜售检测 (< 0.65)",
        "path_irrev_m": "定向力量 (> 0.05)",
        "dom_eig_m": "临界减速 (> 0.85)",
        "mf_flow_imbalance": "资金流不平衡 (> 0.3)",
        "mf_big_cumsum_s": "大单累计 (> 0)",
        "mf_big_streak": "大单连续 (≥ 3天)",
        "mf_big_momentum": "大单动量 (> 0)",
        "coherence_decay_rate": "退相干速率 (< 0)",
        "purity_norm": "纯度 (> 0.6)",
        "vol_impulse": "量能脉冲 (> 1.8×)",
    }
    lines.append("| 因子 | 策略用途 | 全市场有效率 | 中位IC | IC_IR | 评价 |")
    lines.append("|------|---------|-------------|--------|-------|------|")
    for f, usage in strategy_factors.items():
        row_match = factor_summary[factor_summary["factor"] == f]
        if len(row_match) > 0:
            r = row_match.iloc[0]
            eff = r["pct_effective"]
            mic = r["median_ic"]
            mir = r["median_ic_ir"]
            if eff >= 0.25:
                verdict = "✅ 强有效"
            elif eff >= 0.15:
                verdict = "⚠️ 一般"
            else:
                verdict = "❌ 弱"
            lines.append(f"| `{f}` | {usage} | {eff:.1%} | {mic:+.4f} | {mir:+.4f} | {verdict} |")
    lines.append("")

    # =========== 3. 因子分组分析 ===========
    lines.append("## 3. 因子分组分析")
    lines.append("")

    groups = {
        "熵因子": ["perm_entropy_s", "perm_entropy_m", "perm_entropy_l", "entropy_slope", "entropy_accel"],
        "不可逆因子": ["path_irrev_m", "path_irrev_l"],
        "临界因子": ["dom_eig_m", "dom_eig_l"],
        "波动率因子": ["volatility_m", "volatility_l", "vol_compression", "bbw_pctl"],
        "量因子": ["vol_ratio_s", "vol_impulse", "vol_shrink"],
        "资金流因子": ["mf_big_net", "mf_big_net_ratio", "mf_big_cumsum_s", "mf_big_cumsum_m",
                     "mf_big_cumsum_l", "mf_sm_proportion", "mf_flow_imbalance",
                     "mf_big_momentum", "mf_big_streak"],
        "量子相干因子": ["coherence_l1", "purity_norm", "von_neumann_entropy", "coherence_decay_rate"],
    }

    for gname, gfactors in groups.items():
        gdf = factor_summary[factor_summary["factor"].isin(gfactors)]
        if len(gdf) == 0:
            continue
        avg_eff = gdf["pct_effective"].mean()
        best = gdf.iloc[0]  # 已按有效率排序
        lines.append(f"- **{gname}**: 平均有效率 {avg_eff:.1%}, "
                     f"最佳 `{best['factor']}` ({best['pct_effective']:.1%})")
    lines.append("")

    # =========== 4. 行业洞察 ===========
    if industry_summary is not None and len(industry_summary) > 0:
        lines.append("## 4. 行业洞察")
        lines.append("")
        lines.append("各行业最有效因子 (按行业内 |mean_ic| 排序, 取 Top 3):")
        lines.append("")

        # 每个行业最有效的因子
        ind_top = (industry_summary
                   .assign(abs_mean_ic=lambda d: d["mean_ic"].abs())
                   .sort_values("abs_mean_ic", ascending=False)
                   .groupby("industry").head(3))

        # 选取股票数 > 20 的行业
        big_industries = ind_top[ind_top["n_stocks"] >= 20]
        if len(big_industries) > 0:
            # 按行业分组展示
            top_industries = (big_industries.groupby("industry")["abs_mean_ic"]
                              .max().sort_values(ascending=False).head(15).index)
            for ind in top_industries:
                sub = big_industries[big_industries["industry"] == ind].head(3)
                factors_str = " | ".join(
                    f"`{r['factor']}`({r['mean_ic']:+.3f})" for _, r in sub.iterrows()
                )
                lines.append(f"- **{ind}** ({sub.iloc[0]['n_stocks']}只): {factors_str}")
            lines.append("")

    # =========== 5. IC 矩阵统计 ===========
    lines.append("## 5. IC 矩阵统计")
    lines.append("")

    # 去除 symbol 列
    ic_cols = [c for c in ic_matrix.columns if c != "symbol"]
    ic_data = ic_matrix[ic_cols]

    # 因子间相关性 — 找高度相关的因子对
    if len(ic_data) > 50:
        corr_mat = ic_data.corr()
        high_corr_pairs = []
        for i, c1 in enumerate(ic_cols):
            for j, c2 in enumerate(ic_cols):
                if j > i:
                    c = corr_mat.loc[c1, c2]
                    if abs(c) > 0.7:
                        high_corr_pairs.append((c1, c2, c))
        high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        if high_corr_pairs:
            lines.append("### 高相关因子对 (|ρ| > 0.7)")
            lines.append("")
            lines.append("这些因子可能存在信息冗余，考虑合并或去除：")
            lines.append("")
            for c1, c2, c in high_corr_pairs[:10]:
                lines.append(f"- `{c1}` ↔ `{c2}`: ρ = {c:.3f}")
            lines.append("")

    # IC 分布统计
    lines.append("### 因子 IC 分布")
    lines.append("")
    lines.append("| 因子 | IC 均值 | IC 中位 | IC 标准差 | IC>0 占比 | IC>0.1 占比 |")
    lines.append("|------|---------|---------|----------|----------|------------|")
    for f in ic_cols:
        vals = ic_data[f].dropna().values
        if len(vals) < 10:
            continue
        lines.append(
            f"| `{f}` | {np.mean(vals):+.4f} | {np.median(vals):+.4f} | "
            f"{np.std(vals):.4f} | {(vals > 0).mean():.1%} | {(vals > 0.1).mean():.1%} |"
        )
    lines.append("")

    # =========== 6. 策略建议 ===========
    lines.append("## 6. 策略优化建议")
    lines.append("")

    # 基于数据的建议
    top5 = factor_summary.head(5)
    for _, row in top5.iterrows():
        f = row["factor"]
        if f in strategy_factors:
            lines.append(f"- ✅ `{f}` 是策略已使用因子，有效率 {row['pct_effective']:.1%}，继续保持")
        else:
            lines.append(f"- 💡 `{f}` 有效率 {row['pct_effective']:.1%} 但策略未直接使用，"
                         f"考虑纳入信号条件")

    # 无效策略因子
    for f, usage in strategy_factors.items():
        row_match = factor_summary[factor_summary["factor"] == f]
        if len(row_match) > 0 and row_match.iloc[0]["pct_effective"] < 0.12:
            lines.append(f"- ⚠️ `{f}` ({usage}) 全市场有效率仅 "
                         f"{row_match.iloc[0]['pct_effective']:.1%}，建议降低权重或改为可选条件")
    lines.append("")

    # =========== 7. 周线因子分析 ===========
    if weekly_factor_summary is not None and len(weekly_factor_summary) > 0:
        weekly_results_with_data = [r for r in all_results if r.get("weekly")]
        n_weekly = len(weekly_results_with_data)

        lines.append("## 7. 周线因子分析")
        lines.append("")
        lines.append(f"> 覆盖 {n_weekly} 只股票 | 因子数: {len(WEEKLY_FACTORS)} | "
                     f"前瞻窗口: 1w/3w/5w")
        lines.append("")

        # 7.1 周线因子有效率排名
        lines.append("### 7.1 周线因子有效率排名")
        lines.append("")
        lines.append("| 排名 | 因子 | 有效率 | 正向 | 负向 | 不稳定 | 中位IC | 方向共识 |")
        lines.append("|------|------|--------|------|------|--------|--------|----------|")
        direction_zh = {"positive": "正向↑", "negative": "负向↓", "mixed": "混合↔"}
        for i, (_, row) in enumerate(weekly_factor_summary.iterrows()):
            d = direction_zh.get(row["direction_consensus"], row["direction_consensus"])
            lines.append(
                f"| {i+1} | `{row['factor']}` | {row['pct_effective']:.1%} | "
                f"{row['n_positive']} | {row['n_negative']} | {row['n_unstable']} | "
                f"{row['median_ic']:.4f} | {d} |"
            )
        lines.append("")

        # 7.2 周线强普适因子
        strong_weekly = weekly_factor_summary[weekly_factor_summary["pct_effective"] >= 0.15]
        if len(strong_weekly) > 0:
            lines.append("### 7.2 周线强普适因子 (有效率 ≥ 15%)")
            lines.append("")
            for _, row in strong_weekly.iterrows():
                d = "越大越涨" if row["direction_consensus"] == "positive" else (
                    "越大越跌" if row["direction_consensus"] == "negative" else "方向不一致")
                lines.append(f"- **`{row['factor']}`**: 有效率 {row['pct_effective']:.1%}, "
                             f"中位IC {row['median_ic']:+.4f} → {d}")
            lines.append("")

        # 7.3 日线 vs 周线对比
        lines.append("### 7.3 日线 vs 周线因子对比")
        lines.append("")
        lines.append("共有因子在两个时间框架下的表现对比：")
        lines.append("")
        lines.append("| 因子 | 日线有效率 | 日线中位IC | 周线有效率 | 周线中位IC | 变化 |")
        lines.append("|------|----------|----------|----------|----------|------|")
        common_factors = [f for f in WEEKLY_FACTORS if f in ALL_FACTORS]
        for f in common_factors:
            d_match = factor_summary[factor_summary["factor"] == f]
            w_match = weekly_factor_summary[weekly_factor_summary["factor"] == f]
            if len(d_match) == 0 or len(w_match) == 0:
                continue
            d_eff = d_match.iloc[0]["pct_effective"]
            d_ic = d_match.iloc[0]["median_ic"]
            w_eff = w_match.iloc[0]["pct_effective"]
            w_ic = w_match.iloc[0]["median_ic"]
            diff = w_eff - d_eff
            arrow = "📈" if diff > 0.05 else ("📉" if diff < -0.05 else "➡️")
            lines.append(
                f"| `{f}` | {d_eff:.1%} | {d_ic:+.4f} | "
                f"{w_eff:.1%} | {w_ic:+.4f} | {arrow} {diff:+.1%} |"
            )
        lines.append("")

        # 7.4 周线特有因子
        weekly_only = [f for f in WEEKLY_EXTRA_FACTORS]
        wo_df = weekly_factor_summary[weekly_factor_summary["factor"].isin(weekly_only)]
        if len(wo_df) > 0:
            lines.append("### 7.4 周线特有因子")
            lines.append("")
            for _, row in wo_df.iterrows():
                d = direction_zh.get(row["direction_consensus"], "?")
                lines.append(f"- `{row['factor']}`: 有效率 {row['pct_effective']:.1%}, "
                             f"中位IC {row['median_ic']:+.4f}, {d}")
            lines.append("")

        # 7.5 周线 IC 分布
        if weekly_ic_matrix is not None and len(weekly_ic_matrix) > 50:
            lines.append("### 7.5 周线 IC 分布")
            lines.append("")
            lines.append("| 因子 | IC 均值 | IC 中位 | IC 标准差 | IC>0 占比 |")
            lines.append("|------|---------|---------|----------|----------|")
            wic_cols = [c for c in weekly_ic_matrix.columns if c != "symbol"]
            for f in wic_cols:
                vals = weekly_ic_matrix[f].dropna().values
                if len(vals) < 10:
                    continue
                lines.append(
                    f"| `{f}` | {np.mean(vals):+.4f} | {np.median(vals):+.4f} | "
                    f"{np.std(vals):.4f} | {(vals > 0).mean():.1%} |"
                )
            lines.append("")

    # 写入文件
    report_path = os.path.join(out_dir, "summary_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"总结报告已写入 {report_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="个股因子有效性画像")
    parser.add_argument("--cache_dir", required=True, help="特征缓存目录")
    parser.add_argument("--out_dir", default="", help="输出目录 (默认: src/strategy/.../reports/YYYYMMDD)")
    parser.add_argument("--forward_days", type=str, default="1,3,5",
                        help="日线前瞻天数列表, 逗号分隔")
    parser.add_argument("--forward_weeks", type=str, default="1,3,5",
                        help="周线前瞻周数列表, 逗号分隔")
    parser.add_argument("--decay_lambda", type=float, default=0.007,
                        help="日线时间衰减参数 (默认 0.007, ~100天半衰期)")
    parser.add_argument("--weekly_decay_lambda", type=float, default=0.035,
                        help="周线时间衰减参数 (默认 0.035, ~20周半衰期)")
    parser.add_argument("--min_rows", type=int, default=120,
                        help="最少数据行数")
    parser.add_argument("--ic_threshold", type=float, default=0.05)
    parser.add_argument("--ic_ir_threshold", type=float, default=0.5)
    parser.add_argument("--rolling_window", type=int, default=60)
    parser.add_argument("--n_quintiles", type=int, default=5)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--symbols", type=str, default="",
                        help="逗号分隔股票列表 (如 sh600519,sz000001)")
    parser.add_argument("--top_n", type=int, default=0,
                        help="只分析成交额 Top N")
    parser.add_argument("--basic_path", type=str,
                        default="/nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv",
                        help="股票基本信息 CSV")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # 默认输出到 reports/YYYYMMDD
    out_dir = args.out_dir
    if not out_dir:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        today = datetime.now().strftime("%Y%m%d")
        out_dir = os.path.join(script_dir, "reports", today)

    cfg = ProfilingConfig(
        cache_dir=args.cache_dir,
        out_dir=out_dir,
        forward_days=[int(x) for x in args.forward_days.split(",")],
        forward_weeks=[int(x) for x in args.forward_weeks.split(",")],
        decay_lambda=args.decay_lambda,
        weekly_decay_lambda=args.weekly_decay_lambda,
        min_rows=args.min_rows,
        ic_threshold=args.ic_threshold,
        ic_ir_threshold=args.ic_ir_threshold,
        rolling_window=args.rolling_window,
        n_quintiles=args.n_quintiles,
        workers=args.workers,
        symbols=[s.strip() for s in args.symbols.split(",") if s.strip()],
        top_n=args.top_n,
        basic_path=args.basic_path,
    )

    run_profiling(cfg)


if __name__ == "__main__":
    main()
