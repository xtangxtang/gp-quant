"""
大盘趋势判断策略 - 趋势引擎

整合 7 维指标 → 综合评分 → 趋势判定。
主入口: run_market_trend_scan()
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import MarketTrendConfig, TrendState
from .data_loader import (
    get_trading_dates,
    load_all_stocks,
    load_basic_names,
    load_index,
    load_industry_map,
    load_margin,
    load_shibor,
    load_stk_limit,
)
from .macro_indicators import MacroIndicatorEngine, MacroSnapshot
from .micro_indicators import MicroIndicatorEngine, MicroSnapshot, SectorFlow


# ---------------------------------------------------------------------------
# 评分函数: 将原始指标映射到 [-1, 1]
# ---------------------------------------------------------------------------

def _score_breadth(snap: MicroSnapshot) -> float:
    """广度评分。

    核心逻辑:
    - advance_ratio 0.5 是中性，0.7+ 是强势
    - above_ma20_ratio 0.5 是中性
    - 涨停多于跌停 → 正分
    - 广度推力 → 额外加分
    """
    # advance_ratio: [0.3, 0.7] → [-1, 1]
    ar = np.clip((snap.advance_ratio - 0.5) / 0.2, -1.0, 1.0)
    # above_ma20: [0.3, 0.7] → [-1, 1]
    ma20 = np.clip((snap.above_ma20_ratio - 0.5) / 0.2, -1.0, 1.0)
    # above_ma60: [0.3, 0.7] → [-1, 1]
    ma60 = np.clip((snap.above_ma60_ratio - 0.5) / 0.2, -1.0, 1.0)
    # new_high_ratio: [0, 0.1] → [0, 1]
    nh = np.clip(snap.new_high_ratio / 0.1, 0.0, 1.0)
    # 涨跌停差: [-50, 50] → [-1, 1]
    limit_diff = snap.limit_up_count - snap.limit_down_count
    limit_score = np.clip(limit_diff / 50.0, -1.0, 1.0)

    score = 0.30 * ar + 0.25 * ma20 + 0.15 * ma60 + 0.15 * nh + 0.15 * limit_score
    if snap.breadth_thrust:
        score = min(score + 0.15, 1.0)
    return float(np.clip(score, -1.0, 1.0))


def _score_money_flow(snap: MicroSnapshot) -> float:
    """资金流评分。"""
    # net_inflow_ratio: [0.3, 0.7] → [-1, 1]
    inflow = np.clip((snap.net_inflow_ratio - 0.5) / 0.2, -1.0, 1.0)
    # big_order_net_sum: 大单净额 (万元)，粗略 [-5亿, +5亿] → [-1, 1]
    big = np.clip(snap.big_order_net_sum / 5e5, -1.0, 1.0)
    score = 0.6 * inflow + 0.4 * big
    return float(np.clip(score, -1.0, 1.0))


def _score_volatility(snap: MicroSnapshot) -> float:
    """波动结构评分。

    低波动 + 低恐慌 → 正分 (有利市场)
    高波动 + 高恐慌 → 负分
    """
    # vol_median: 低波好。[0.01, 0.04] → [1, -1]
    vol = np.clip(1.0 - (snap.vol_median - 0.01) / 0.03 * 2.0, -1.0, 1.0)
    # panic_ratio: [0, 0.05] → [0, -1]
    panic = np.clip(-snap.panic_ratio / 0.05, -1.0, 0.0)
    score = 0.5 * vol + 0.5 * panic
    return float(np.clip(score, -1.0, 1.0))


def _score_entropy(snap: MicroSnapshot) -> float:
    """熵/有序度评分。

    ordering_ratio 高 → 有序 → 趋势明确 → 正分
    """
    # ordering_ratio: [0.2, 0.7] → [-1, 1]
    order = np.clip((snap.ordering_ratio - 0.45) / 0.25, -1.0, 1.0)
    # entropy_median 低 → 有序。max entropy for order=3 is ln(6) ≈ 1.79
    # [0.8, 1.6] → [1, -1]
    ent = np.clip(1.0 - (snap.entropy_median - 0.8) / 0.8 * 2.0, -1.0, 1.0)
    score = 0.5 * order + 0.5 * ent
    return float(np.clip(score, -1.0, 1.0))


def _score_momentum(snap: MicroSnapshot) -> float:
    """动量扩散评分。

    sector_momentum_std 低 → 行业一致上涨 → 正分
    trend_alignment 高 → 动量持续 → 正分
    """
    # trend_alignment: [0.4, 0.7] → [-1, 1]
    align = np.clip((snap.trend_alignment - 0.55) / 0.15, -1.0, 1.0)
    # sector_momentum_std: 低离散 → 好。[0.02, 0.08] → [1, -1]
    disp = np.clip(1.0 - (snap.sector_momentum_std - 0.02) / 0.06 * 2.0, -1.0, 1.0)
    score = 0.6 * align + 0.4 * disp
    return float(np.clip(score, -1.0, 1.0))


def _score_leverage(macro: MacroSnapshot) -> float:
    """杠杆资金评分。

    融资余额趋势向上 → 正分 (增量资金入场)
    融资净买入 > 0 → 正分
    """
    # margin_balance_chg_pct: [-0.02, 0.02] → [-1, 1]
    chg = np.clip(macro.margin_balance_chg_pct / 0.02, -1.0, 1.0)
    # margin MA5 vs MA20 金叉/死叉
    if macro.margin_balance_ma20 > 0:
        ma_ratio = (macro.margin_balance_ma5 / macro.margin_balance_ma20) - 1.0
        ma_score = np.clip(ma_ratio / 0.02, -1.0, 1.0)
    else:
        ma_score = 0.0
    # 净买入: [-50亿, 50亿] → [-1, 1]
    net = np.clip(macro.margin_net_buy / 50.0, -1.0, 1.0)
    score = 0.35 * chg + 0.35 * ma_score + 0.30 * net
    return float(np.clip(score, -1.0, 1.0))


def _score_liquidity(macro: MacroSnapshot) -> float:
    """流动性评分。

    SHIBOR 低且稳 → 正分 (流动性充裕)
    SHIBOR 飙升 → 负分 (资金面紧张)
    """
    # shibor_on: 低利率好。[1.0, 3.0] → [1, -1]
    level = np.clip(1.0 - (macro.shibor_on - 1.0) / 2.0 * 2.0, -1.0, 1.0)
    # on vs ma20: 低于均值好
    if macro.shibor_on_ma20 > 0:
        rel = (macro.shibor_on / macro.shibor_on_ma20) - 1.0
        rel_score = np.clip(-rel / 0.2, -1.0, 1.0)
    else:
        rel_score = 0.0
    # 日变化: 飙升 → 负分。[-0.3, 0.3] → [1, -1]
    spike = np.clip(-macro.shibor_on_change / 0.3, -1.0, 1.0)
    score = 0.30 * level + 0.30 * rel_score + 0.40 * spike
    return float(np.clip(score, -1.0, 1.0))


def _determine_trend(score: float, cfg: MarketTrendConfig,
                     breadth_thrust: bool, panic_ratio: float) -> str:
    """判定趋势状态。"""
    if score >= cfg.strong_up_threshold and breadth_thrust:
        return "STRONG_UP"
    if score >= cfg.strong_up_threshold:
        return "UP"
    if score >= cfg.up_threshold:
        return "UP"
    if score <= cfg.strong_down_threshold and panic_ratio > 0.03:
        return "STRONG_DOWN"
    if score <= cfg.down_threshold:
        return "DOWN"
    return "NEUTRAL"


# ---------------------------------------------------------------------------
# 股票特征预计算 worker (用于多进程)
# ---------------------------------------------------------------------------

def _worker_compute_features(args: Tuple) -> Optional[Tuple[str, pd.DataFrame]]:
    """多进程 worker: 计算单只股票的特征列。"""
    symbol, df_bytes, cfg = args
    try:
        df = pd.read_pickle(df_bytes)  # from BytesIO
        engine = MicroIndicatorEngine(cfg)
        df = engine.compute_stock_features(df)
        return (symbol, df)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 主引擎
# ---------------------------------------------------------------------------

def run_market_trend_scan(cfg: MarketTrendConfig) -> List[TrendState]:
    """执行大盘趋势扫描。

    流程:
    1. 加载所有数据
    2. 预计算每只股票的特征列 (多进程)
    3. 遍历交易日，聚合微观+宏观 → 评分 → 趋势判定
    4. 输出 CSV
    """
    t0 = time.time()
    print(f"[MarketTrend] 加载数据...")

    # 1. 加载
    stocks_raw = load_all_stocks(cfg)
    print(f"  个股: {len(stocks_raw)} 只")

    index_df = load_index(cfg)
    print(f"  指数: {len(index_df)} 行")

    limit_data = load_stk_limit(cfg)
    print(f"  涨跌停: {len(limit_data)} 只")

    margin_df = load_margin(cfg)
    print(f"  两融: {len(margin_df)} 行")

    shibor_df = load_shibor(cfg)
    print(f"  SHIBOR: {len(shibor_df)} 行")

    industry_map = load_industry_map(cfg)
    print(f"  行业映射: {len(industry_map)} 只")

    names_map = load_basic_names(cfg)
    print(f"  股票名称: {len(names_map)} 只")

    t1 = time.time()
    print(f"  数据加载耗时: {t1 - t0:.1f}s")

    # 2. 预计算特征
    print(f"[MarketTrend] 预计算个股特征 ({len(stocks_raw)} 只)...")
    micro_engine = MicroIndicatorEngine(cfg)
    stocks: Dict[str, pd.DataFrame] = {}
    for symbol, df in stocks_raw.items():
        df = micro_engine.compute_stock_features(df)
        stocks[symbol] = df
    del stocks_raw
    t2 = time.time()
    print(f"  特征计算耗时: {t2 - t1:.1f}s")

    # 3. 宏观引擎
    macro_engine = MacroIndicatorEngine(cfg, margin_df, shibor_df)

    # 4. 交易日序列
    trading_dates = get_trading_dates(stocks, cfg.start_date, cfg.end_date)
    print(f"[MarketTrend] 交易日: {len(trading_dates)} 天 "
          f"({trading_dates[0]} ~ {trading_dates[-1]})")

    # 指数查找表
    idx_close_map: Dict[int, float] = {}
    idx_pct_map: Dict[int, float] = {}
    if not index_df.empty:
        for _, row in index_df.iterrows():
            td = int(row["trade_date"])
            idx_close_map[td] = float(row["close"])
            pct = row.get("pct_chg", np.nan)
            idx_pct_map[td] = float(pct) if not np.isnan(pct) else 0.0

    # 5. 逐日扫描
    print(f"[MarketTrend] 逐日扫描...")
    results: List[TrendState] = []
    ema_advance: Optional[float] = None
    ema_alpha = 2.0 / (cfg.breadth_thrust_window + 1)

    # 确定报告日期
    report_date_int = int(cfg.report_date) if cfg.report_date else 0
    report_sector_flows: List[SectorFlow] = []

    for i, date in enumerate(trading_dates):
        date_int = int(date)

        # 微观
        micro = micro_engine.aggregate_daily(
            date_int, stocks, limit_data, industry_map, names_map,
            ema_advance=ema_advance,
        )
        if micro is None:
            continue

        # 保存报告日期的板块流向
        if cfg.report:
            if report_date_int == 0 or date_int == report_date_int:
                report_sector_flows = micro.sector_flows

        # 更新 EMA
        if ema_advance is None:
            ema_advance = micro.advance_ratio
        else:
            ema_advance = ema_alpha * micro.advance_ratio + (1 - ema_alpha) * ema_advance

        # 宏观
        macro = macro_engine.snapshot(date_int)

        # 评分
        w = cfg.weights
        bs = _score_breadth(micro)
        mfs = _score_money_flow(micro)
        vs = _score_volatility(micro)
        es = _score_entropy(micro)
        ms = _score_momentum(micro)
        ls = _score_leverage(macro)
        lqs = _score_liquidity(macro)

        composite = (
            w["breadth"] * bs
            + w["money_flow"] * mfs
            + w["volatility"] * vs
            + w["entropy"] * es
            + w["momentum"] * ms
            + w["leverage"] * ls
            + w["liquidity"] * lqs
        )
        composite = float(np.clip(composite, -1.0, 1.0))

        # 趋势
        trend = _determine_trend(composite, cfg, micro.breadth_thrust, micro.panic_ratio)

        state = TrendState(
            date=str(date_int),
            trend=trend,
            composite_score=round(composite, 4),
            breadth_score=round(bs, 4),
            money_flow_score=round(mfs, 4),
            volatility_score=round(vs, 4),
            entropy_score=round(es, 4),
            momentum_score=round(ms, 4),
            leverage_score=round(ls, 4),
            liquidity_score=round(lqs, 4),
            advance_ratio=round(micro.advance_ratio, 4),
            above_ma20_ratio=round(micro.above_ma20_ratio, 4),
            above_ma60_ratio=round(micro.above_ma60_ratio, 4),
            new_high_ratio=round(micro.new_high_ratio, 4),
            limit_up_count=micro.limit_up_count,
            limit_down_count=micro.limit_down_count,
            breadth_thrust=micro.breadth_thrust,
            net_inflow_ratio=round(micro.net_inflow_ratio, 4),
            big_order_net_sum=round(micro.big_order_net_sum / 1e4, 2),  # 转亿
            vol_median=round(micro.vol_median, 6),
            panic_ratio=round(micro.panic_ratio, 4),
            entropy_median=round(micro.entropy_median, 4),
            ordering_ratio=round(micro.ordering_ratio, 4),
            sector_momentum_std=round(micro.sector_momentum_std, 6),
            trend_alignment=round(micro.trend_alignment, 4),
            margin_balance=round(macro.margin_balance, 2),
            margin_net_buy=round(macro.margin_net_buy, 2),
            shibor_on=round(macro.shibor_on, 4),
            shibor_on_change=round(macro.shibor_on_change, 4),
            index_close=idx_close_map.get(date_int, 0.0),
            index_pct_chg=idx_pct_map.get(date_int, 0.0),
            total_stocks=micro.total_stocks,
        )
        results.append(state)

        if (i + 1) % 50 == 0 or i == len(trading_dates) - 1:
            print(f"  [{i + 1}/{len(trading_dates)}] {date_int} "
                  f"trend={trend} score={composite:.3f} "
                  f"adv={micro.advance_ratio:.2%} stocks={micro.total_stocks}")

    t3 = time.time()
    print(f"[MarketTrend] 扫描完成: {len(results)} 天, 耗时 {t3 - t2:.1f}s")

    # 6. 输出
    if cfg.out_dir:
        os.makedirs(cfg.out_dir, exist_ok=True)
        _write_results(results, cfg)

    # 7. 生成诊断报告
    if cfg.report and results:
        from .report_generator import generate_daily_report, write_report
        # 找到报告对应的 TrendState
        if report_date_int > 0:
            target = [r for r in results if r.date == str(report_date_int)]
            report_state = target[0] if target else results[-1]
        else:
            report_state = results[-1]
        report_text = generate_daily_report(
            report_state, report_sector_flows, results, cfg,
        )
        write_report(report_text, report_sector_flows,
                     report_state.date, cfg.out_dir)
        # 也输出到控制台
        print(f"\n{'='*60}")
        print(report_text)

    total_time = time.time() - t0
    print(f"[MarketTrend] 总耗时: {total_time:.1f}s")
    return results


def _write_results(results: List[TrendState], cfg: MarketTrendConfig) -> None:
    """输出结果 CSV。"""
    if not results:
        return

    import dataclasses
    rows = [dataclasses.asdict(s) for s in results]
    df = pd.DataFrame(rows)

    # 主结果
    path = os.path.join(cfg.out_dir, "market_trend_daily.csv")
    df.to_csv(path, index=False)
    print(f"  → {path} ({len(df)} 行)")

    # 趋势统计
    trend_counts = df["trend"].value_counts()
    summary_path = os.path.join(cfg.out_dir, "market_trend_summary.csv")
    summary = pd.DataFrame({
        "trend": trend_counts.index,
        "days": trend_counts.values,
        "pct": (trend_counts.values / len(df) * 100).round(1),
    })
    summary.to_csv(summary_path, index=False)
    print(f"  → {summary_path}")
    print(f"\n  趋势统计:")
    for _, row in summary.iterrows():
        print(f"    {row['trend']:12s} {int(row['days']):4d} 天 ({row['pct']:.1f}%)")

    # 和指数的收益率对比
    if "index_pct_chg" in df.columns:
        for t in ["STRONG_UP", "UP", "NEUTRAL", "DOWN", "STRONG_DOWN"]:
            sub = df[df["trend"] == t]
            if len(sub) > 0:
                avg_idx_ret = sub["index_pct_chg"].mean()
                print(f"    {t:12s} 时指数日均涨幅: {avg_idx_ret:.3f}%")
