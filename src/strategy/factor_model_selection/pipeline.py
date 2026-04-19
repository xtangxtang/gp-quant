"""
因子模型选股 — 4-Agent Pipeline 编排器

DAG:
  Agent 1 (因子计算) → Agent 2 (模型选股) → Agent 3 (回测验证) → Agent 4 (策略分析)

支持两种模式:
  - 单次运行: 对一个 scan_date 跑完整流水线
  - 滚动运行: 按时间窗口逐日滚动, 累计所有分析结果

用法:
  # 单次运行
  python -m src.strategy.factor_model_selection.pipeline \
    --cache_dir /path/to/feature-cache \
    --data_dir /path/to/tushare-daily-full \
    --scan_date 20260410

  # 滚动运行 (2025Q1)
  python -m src.strategy.factor_model_selection.pipeline \
    --cache_dir /path/to/feature-cache \
    --data_dir /path/to/tushare-daily-full \
    --start_date 20250101 --end_date 20250331 \
    --scan_interval 5
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 各 horizon 对应的持有天数 (交易日)
HOLD_MAP = {"1d": 1, "3d": 3, "5d": 5, "1w": 5, "3w": 15, "5w": 25}


@dataclass
class PipelineConfig:
    """流水线配置。"""
    cache_dir: str = ""           # 特征缓存根目录
    data_dir: str = ""            # 日线 CSV 目录
    basic_path: str = ""          # tushare_stock_basic.csv
    out_dir: str = "results/factor_model_selection"

    # 时间参数
    scan_date: str = ""           # 单次模式: 信号日
    start_date: str = ""          # 滚动模式: 起始日期
    end_date: str = ""            # 滚动模式: 结束日期
    scan_interval: int = 5        # 扫描间隔 (交易日)

    # 模型参数
    top_n: int = 5                # 每个 horizon 选几只
    horizons: list[str] = field(default_factory=lambda: ["5d"])

    # 自动推断
    def __post_init__(self):
        for attr in ("scan_date", "start_date", "end_date"):
            v = getattr(self, attr, "").strip()
            for sep in ("/", "-"):
                if sep in v:
                    parts = v.split(sep)
                    if len(parts) == 3 and all(p.isdigit() for p in parts):
                        v = f"{int(parts[0]):04d}{int(parts[1]):02d}{int(parts[2]):02d}"
                    break
            setattr(self, attr, v)

        if not self.basic_path and self.data_dir:
            candidate = os.path.join(os.path.dirname(self.data_dir.rstrip("/")), "tushare_stock_basic.csv")
            if os.path.exists(candidate):
                self.basic_path = candidate


def build_calendar(data_dir: str) -> list[str]:
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


def run_pipeline_single(cfg: PipelineConfig) -> dict:
    """
    单次流水线: 对一个 scan_date 执行 4 个 Agent。

    Returns:
        完整分析结果 dict
    """
    from .agent_factor import run_factor_snapshot
    from .agent_selection import run_selection, StateConfig
    from .agent_validation import run_validation
    from .agent_analysis import run_analysis

    scan_date = cfg.scan_date

    # 构建日历
    calendar = build_calendar(cfg.data_dir)
    if scan_date not in calendar:
        earlier = [d for d in calendar if d <= scan_date]
        if earlier:
            scan_date = earlier[-1]
            logger.info(f"scan_date 调整为最近交易日: {scan_date}")
        else:
            logger.error(f"scan_date {cfg.scan_date} 无法匹配日历")
            return {}

    t0 = time.time()

    # ── Agent 1: 因子计算 ──
    print(f"\n{'─'*60}")
    print(f"  Agent 1: 因子计算 (scan_date={scan_date})")
    print(f"{'─'*60}")
    factor_snapshot = run_factor_snapshot(
        cache_dir=cfg.cache_dir,
        scan_date=scan_date,
        basic_path=cfg.basic_path,
    )
    if factor_snapshot.empty:
        logger.error("Agent 1 失败: 无因子数据")
        return {}
    t1 = time.time()
    print(f"  → {len(factor_snapshot)} 只股票, {t1 - t0:.1f}s")

    # ── Agent 2: 因子状态机选股 ──
    print(f"\n{'─'*60}")
    print(f"  Agent 2: 因子状态机选股 (scan_date={scan_date})")
    print(f"{'─'*60}")
    selections = run_selection(
        factor_snapshot=factor_snapshot,
        cache_dir=cfg.cache_dir,
        scan_date=scan_date,
        top_n=cfg.top_n,
        horizons=cfg.horizons,
        cfg=StateConfig(),
    )
    # 只保留请求的 horizon
    selections = {h: df for h, df in selections.items() if h in cfg.horizons}
    t2 = time.time()

    n_sel = sum(len(df) for df in selections.values())
    print(f"  → {n_sel} 条推荐, {t2 - t1:.1f}s")

    if n_sel == 0:
        logger.warning("Agent 2 无推荐, 流水线终止")
        return {}

    # ── Agent 3: 条件退出回测 ──
    print(f"\n{'─'*60}")
    print(f"  Agent 3: 条件退出回测")
    print(f"{'─'*60}")
    validation_results = run_validation(
        selections=selections,
        data_dir=cfg.data_dir,
        scan_date=scan_date,
        calendar=calendar,
        cache_dir=cfg.cache_dir,
    )
    t3 = time.time()
    n_trades = sum(len(r["trades"]) for r in validation_results.values())
    print(f"  → {n_trades} 笔交易, {t3 - t2:.1f}s")

    # ── Agent 4: 策略分析 ──
    print(f"\n{'─'*60}")
    print(f"  Agent 4: 策略分析")
    print(f"{'─'*60}")
    analysis = run_analysis(
        factor_snapshot=factor_snapshot,
        selections=selections,
        validation_results=validation_results,
        data_dir=cfg.data_dir,
        scan_date=scan_date,
        calendar=calendar,
    )
    t4 = time.time()
    print(f"  → 分析完成, {t4 - t3:.1f}s")
    print(f"\n  总耗时: {t4 - t0:.1f}s")

    return {
        "scan_date": scan_date,
        "n_stocks": len(factor_snapshot),
        "selections": {h: df.to_dict("records") for h, df in selections.items()},
        "validation": {h: v["metrics"] for h, v in validation_results.items()},
        "analysis": analysis,
    }


def run_pipeline_rolling(cfg: PipelineConfig) -> list[dict]:
    """
    滚动流水线: 逐日执行, 累计结果。

    每个 horizon 按自身持有天数决定扫描间隔, 保证持仓期不重叠:
      - 3d → 每 3 个交易日扫描一次
      - 5d/1w → 每 5 个交易日
      - 3w → 每 15 个交易日
      - 5w → 每 25 个交易日

    scan_interval 参数作为最小间隔下限 (默认 0, 即完全由 hold_days 决定)。
    """
    calendar = build_calendar(cfg.data_dir)
    all_dates = [d for d in calendar if cfg.start_date <= d <= cfg.end_date]

    # ── 为每个 horizon 生成独立扫描日序列 ──
    horizon_schedules: dict[str, list[str]] = {}
    for h in cfg.horizons:
        hold_days = HOLD_MAP.get(h, 5)
        interval = max(hold_days, cfg.scan_interval)
        horizon_schedules[h] = all_dates[::interval]

    # 合并所有扫描日 (去重+排序)
    scan_dates_set: set[str] = set()
    for dates in horizon_schedules.values():
        scan_dates_set.update(dates)
    scan_dates = sorted(scan_dates_set)

    # 打印调度计划
    print(f"\n{'='*70}")
    print(f"  滚动流水线: {cfg.start_date} ~ {cfg.end_date}")
    print(f"  调度计划:")
    for h in cfg.horizons:
        hold_days = HOLD_MAP.get(h, 5)
        interval = max(hold_days, cfg.scan_interval)
        print(f"    {h}: 持有 {hold_days} 天, 间隔 {interval} 天, {len(horizon_schedules[h])} 个扫描日")
    print(f"  总扫描日: {len(scan_dates)} 个 (去重后)")
    print(f"{'='*70}\n")

    all_results = []

    for i, scan_date in enumerate(scan_dates):
        # 确定本次扫描哪些 horizons
        active_horizons = [h for h in cfg.horizons if scan_date in horizon_schedules[h]]

        print(f"\n{'='*70}")
        print(f"  [{i+1}/{len(scan_dates)}] scan_date = {scan_date}")
        print(f"  Active horizons: {', '.join(active_horizons)}")
        print(f"{'='*70}")

        single_cfg = PipelineConfig(
            cache_dir=cfg.cache_dir,
            data_dir=cfg.data_dir,
            basic_path=cfg.basic_path,
            out_dir=cfg.out_dir,
            scan_date=scan_date,
            top_n=cfg.top_n,
            horizons=active_horizons,
        )

        result = run_pipeline_single(single_cfg)
        if result:
            all_results.append(result)

    # ── 汇总打印 ──
    if all_results:
        _print_rolling_summary(all_results, cfg)

    # ── 保存 ──
    if cfg.out_dir:
        _save_rolling_results(all_results, cfg)

    return all_results


def _print_rolling_summary(results: list[dict], cfg: PipelineConfig):
    """打印滚动流水线汇总。"""
    print(f"\n{'='*70}")
    print(f"  滚动流水线汇总 ({len(results)} 个扫描日)")
    print(f"{'='*70}")

    for h in cfg.horizons:
        # 只统计包含该 horizon 的结果
        h_results = [r for r in results if h in r.get("validation", {})]
        metrics_list = [
            r["validation"][h]
            for r in h_results
            if r["validation"][h].get("n_trades", 0) > 0
        ]

        hold_days = HOLD_MAP.get(h, 5)
        interval = max(hold_days, cfg.scan_interval)

        if not metrics_list:
            print(f"\n  {h} (持有{hold_days}天, 间隔{interval}天): 无有效结果")
            continue

        avg_wr = np.mean([m["win_rate"] for m in metrics_list])
        avg_pnl = np.mean([m["avg_pnl"] for m in metrics_list])
        total_trades = sum(m["n_trades"] for m in metrics_list)
        effective_count = sum(
            1 for r in h_results
            if r.get("analysis", {}).get("per_horizon", {}).get(h, {}).get("effectiveness", {}).get("effective", False)
        )

        print(f"\n  {h} (持有{hold_days}天, 间隔{interval}天):")
        print(f"    扫描日: {len(h_results)}, 有效: {len(metrics_list)}")
        print(f"    总交易数: {total_trades}")
        print(f"    平均胜率: {avg_wr:.1%}")
        print(f"    平均收益: {avg_pnl:+.2f}%")
        avg_cond = np.mean([m.get("condition_exit_rate", 0) for m in metrics_list])
        avg_hold = np.mean([m.get("avg_hold_days", hold_days) for m in metrics_list])
        print(f"    条件退出率: {avg_cond:.1%}, 均持有 {avg_hold:.1f} 天")
        print(f"    策略有效次数: {effective_count}/{len(h_results)}")

    # 大牛股捕获率汇总
    for h in cfg.horizons:
        total_bull = 0
        total_caught = 0
        for r in results:
            ca = r.get("analysis", {}).get("per_horizon", {}).get(h, {}).get("catch_analysis", {})
            total_bull += ca.get("n_bull", 0)
            total_caught += ca.get("n_caught", 0)

        if total_bull > 0:
            print(f"\n  {h} 大牛股捕获: {total_caught}/{total_bull} ({total_caught/total_bull:.1%})")


def _save_rolling_results(results: list[dict], cfg: PipelineConfig):
    """保存滚动结果。"""
    os.makedirs(cfg.out_dir, exist_ok=True)

    # 汇总表
    summary_rows = []
    for r in results:
        for h in cfg.horizons:
            m = r.get("validation", {}).get(h, {})
            if m.get("n_trades", 0) == 0:
                continue
            eff = r.get("analysis", {}).get("per_horizon", {}).get(h, {}).get("effectiveness", {})
            ca = r.get("analysis", {}).get("per_horizon", {}).get(h, {}).get("catch_analysis", {})
            summary_rows.append({
                "scan_date": r["scan_date"],
                "horizon": h,
                "n_trades": m["n_trades"],
                "win_rate": m["win_rate"],
                "avg_pnl": m["avg_pnl"],
                "best_trade": m.get("best_trade", 0),
                "worst_trade": m.get("worst_trade", 0),
                "effective": eff.get("effective", False),
                "bull_caught": ca.get("n_caught", 0),
                "bull_total": ca.get("n_bull", 0),
            })

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        fpath = os.path.join(cfg.out_dir, f"pipeline_summary_{cfg.start_date}_{cfg.end_date}.csv")
        df.to_csv(fpath, index=False)
        logger.info(f"汇总表 → {fpath}")

    # 完整结果 JSON
    # 精简版: 去掉大体积的 factor_snapshot
    slim = []
    for r in results:
        slim.append({
            "scan_date": r["scan_date"],
            "n_stocks": r.get("n_stocks", 0),
            "selections": r.get("selections", {}),
            "validation": r.get("validation", {}),
            "analysis": {
                "effectiveness": r.get("analysis", {}).get("effectiveness", {}),
                "summary": r.get("analysis", {}).get("summary", ""),
                "per_horizon": {
                    h: {
                        "effectiveness": hr.get("effectiveness", {}),
                        "catch_analysis": {
                            k: v for k, v in hr.get("catch_analysis", {}).items()
                            if k != "missed"  # missed 列表太长, 不存
                        },
                    }
                    for h, hr in r.get("analysis", {}).get("per_horizon", {}).items()
                },
            },
        })

    jpath = os.path.join(cfg.out_dir, f"pipeline_detail_{cfg.start_date}_{cfg.end_date}.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(slim, f, ensure_ascii=False, indent=2)
    logger.info(f"详细结果 → {jpath}")


# ═════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Factor Model 4-Agent Pipeline")

    p.add_argument("--cache_dir", required=True, help="特征缓存根目录")
    p.add_argument("--data_dir", required=True, help="日线 CSV 目录")
    p.add_argument("--basic_path", type=str, default="", help="tushare_stock_basic.csv")
    p.add_argument("--out_dir", type=str, default="results/factor_model_selection", help="输出目录")

    p.add_argument("--scan_date", type=str, default="", help="单次模式: 信号日")
    p.add_argument("--start_date", type=str, default="", help="滚动模式: 起始日期")
    p.add_argument("--end_date", type=str, default="", help="滚动模式: 结束日期")
    p.add_argument("--scan_interval", type=int, default=5, help="扫描间隔 (交易日)")

    p.add_argument("--top_n", type=int, default=5, help="每个 horizon 选几只")
    p.add_argument("--horizons", type=str, default="5d", help="逗号分隔 horizons")
    p.add_argument("--verbose", action="store_true")

    return p


def main():
    args = _build_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    horizons = [h.strip() for h in args.horizons.split(",")]

    cfg = PipelineConfig(
        cache_dir=args.cache_dir,
        data_dir=args.data_dir,
        basic_path=args.basic_path,
        out_dir=args.out_dir,
        scan_date=args.scan_date,
        start_date=args.start_date,
        end_date=args.end_date,
        scan_interval=args.scan_interval,
        top_n=args.top_n,
        horizons=horizons,
    )

    if cfg.start_date and cfg.end_date:
        run_pipeline_rolling(cfg)
    elif cfg.scan_date:
        run_pipeline_single(cfg)
    else:
        print("请指定 --scan_date (单次) 或 --start_date + --end_date (滚动)")


if __name__ == "__main__":
    main()
