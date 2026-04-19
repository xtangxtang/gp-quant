"""
因子模型选股 — Agent 4: 策略分析

职责:
  1. 评判策略是否有效 (胜率/收益/Sharpe 是否达标)
  2. 发现该时段的大牛股 (全市场涨幅 Top 30)
  3. 分析策略是否抓住了大牛股; 没抓住的原因是什么
  4. 抓住的牛股贡献了多少收益

输入: PipelineState (完整的 factor_snapshot, selections, validation_results)
输出: PipelineState.analysis (结构化分析报告)
"""

from __future__ import annotations

import glob
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── 策略有效性判定阈值 ──
EFFECTIVENESS_THRESHOLDS = {
    "min_win_rate": 0.50,       # 胜率 > 50%
    "min_avg_pnl": 0.30,       # 平均收益 > 0.3%
    "min_profit_loss": 1.0,    # 盈亏比 > 1.0
    "max_worst_trade": -15.0,  # 最差单笔不超过 -15%
}

BULL_TOP_N = 30  # 全市场涨幅 Top 30 作为"大牛股"


def run_analysis(
    factor_snapshot: pd.DataFrame,
    selections: dict[str, pd.DataFrame],
    validation_results: dict[str, dict],
    data_dir: str,
    scan_date: str,
    calendar: list[str] | None = None,
) -> dict:
    """
    全面分析策略表现。

    Returns:
        {
            "effectiveness": dict,     # 策略是否有效
            "bull_stocks": dict,       # 大牛股分析
            "per_horizon": dict,       # 每个 horizon 的详细分析
            "summary": str,            # 一句话总结
        }
    """
    hold_map = {"1d": 1, "3d": 3, "5d": 5, "1w": 5, "3w": 15, "5w": 25}

    # ── 构建交易日历 ──
    if calendar is None:
        calendar = _build_calendar(data_dir)

    analysis = {
        "effectiveness": {},
        "bull_stocks": {},
        "per_horizon": {},
        "summary": "",
    }

    # ── 逐 horizon 分析 ──
    for h, vr in validation_results.items():
        trades = vr.get("trades", [])
        metrics = vr.get("metrics", {})
        entry_date = vr.get("entry_date", "")
        exit_date = vr.get("exit_date", "")
        hold_days = hold_map.get(h, 5)

        if not entry_date or not exit_date:
            continue

        # 1. 策略有效性判定
        eff = _judge_effectiveness(metrics)

        # 2. 找出全市场同期大牛股
        bull_df = _find_bull_stocks(data_dir, entry_date, exit_date, top_n=BULL_TOP_N)

        # 3. 分析是否抓住了大牛股
        selected_syms = set(t["symbol"] for t in trades)
        catch_analysis = _analyze_bull_catch(
            bull_df, selected_syms, trades, factor_snapshot
        )

        # 4. 被选中股票中的贡献分析
        contribution = _analyze_contribution(trades)

        horizon_result = {
            "metrics": metrics,
            "effectiveness": eff,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "hold_days": hold_days,
            "bull_stocks": bull_df.to_dict("records") if not bull_df.empty else [],
            "catch_analysis": catch_analysis,
            "contribution": contribution,
        }
        analysis["per_horizon"][h] = horizon_result

    # ── 全局总结 ──
    analysis["effectiveness"] = _overall_effectiveness(analysis["per_horizon"])
    analysis["summary"] = _generate_summary(analysis)

    # ── 打印报告 ──
    _print_report(analysis, scan_date)

    logger.info("Agent 4 完成: 策略分析报告已生成")
    return analysis


def _judge_effectiveness(metrics: dict) -> dict:
    """判定单个 horizon 的策略有效性。"""
    th = EFFECTIVENESS_THRESHOLDS
    n = metrics.get("n_trades", 0)
    if n == 0:
        return {"effective": False, "reason": "无交易", "score": 0}

    checks = {
        "win_rate": metrics.get("win_rate", 0) >= th["min_win_rate"],
        "avg_pnl": metrics.get("avg_pnl", 0) >= th["min_avg_pnl"],
        "worst_ok": metrics.get("worst_trade", -100) >= th["max_worst_trade"],
    }

    # 盈亏比
    avg_win = metrics.get("avg_pnl", 0)  # 简化: 用平均收益代替
    pl_ok = avg_win > 0
    checks["positive_expectation"] = pl_ok

    score = sum(checks.values()) / len(checks)
    passed = score >= 0.5  # 至少一半条件通过

    reasons = []
    if not checks["win_rate"]:
        reasons.append(f"胜率 {metrics.get('win_rate', 0):.1%} < {th['min_win_rate']:.0%}")
    if not checks["avg_pnl"]:
        reasons.append(f"均收益 {metrics.get('avg_pnl', 0):+.2f}% < {th['min_avg_pnl']:.1f}%")
    if not checks["worst_ok"]:
        reasons.append(f"最差 {metrics.get('worst_trade', 0):.1f}% 超限")

    return {
        "effective": passed,
        "score": round(score, 2),
        "checks": checks,
        "reasons": reasons if not passed else [],
    }


def _find_bull_stocks(
    data_dir: str,
    entry_date: str,
    exit_date: str,
    top_n: int = 30,
) -> pd.DataFrame:
    """找出全市场 entry_date ~ exit_date 期间涨幅最大的 Top N 股票。"""
    csvs = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    rows = []

    for fpath in csvs:
        sym = os.path.basename(fpath).replace(".csv", "")
        try:
            df = pd.read_csv(fpath, usecols=["trade_date", "open", "close", "amount"])
            df["trade_date"] = df["trade_date"].astype(str)
        except Exception:
            continue

        entry_rows = df[df["trade_date"] == entry_date]
        exit_rows = df[df["trade_date"] == exit_date]

        if entry_rows.empty or exit_rows.empty:
            continue

        entry_price = float(entry_rows.iloc[0]["open"])
        exit_price = float(exit_rows.iloc[0]["close"])

        if entry_price <= 0:
            continue

        # 排除成交额太低的
        mask = (df["trade_date"] >= entry_date) & (df["trade_date"] <= exit_date)
        if "amount" in df.columns:
            avg_amt = df[mask]["amount"].mean() if mask.sum() > 0 else 0
            if avg_amt < 5000:
                continue

        pnl_pct = (exit_price - entry_price) / entry_price * 100
        rows.append({
            "symbol": sym,
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "pnl_pct": round(pnl_pct, 2),
        })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).sort_values("pnl_pct", ascending=False)
    return result.head(top_n).reset_index(drop=True)


def _analyze_bull_catch(
    bull_df: pd.DataFrame,
    selected_syms: set[str],
    trades: list[dict],
    factor_snapshot: pd.DataFrame,
) -> dict:
    """分析策略是否抓住了大牛股，以及没抓住的原因。"""
    if bull_df.empty:
        return {"caught": [], "missed": [], "catch_rate": 0}

    caught = []
    missed = []

    for _, row in bull_df.iterrows():
        sym = row["symbol"]
        bull_pnl = row["pnl_pct"]

        if sym in selected_syms:
            # 抓住了: 找对应交易记录
            trade = next((t for t in trades if t["symbol"] == sym), None)
            actual_pnl = trade["pnl_pct"] if trade else bull_pnl
            caught.append({
                "symbol": sym,
                "bull_pnl": bull_pnl,
                "actual_pnl": actual_pnl,
            })
        else:
            # 没抓住: 分析原因
            reason = _diagnose_miss(sym, factor_snapshot)
            missed.append({
                "symbol": sym,
                "bull_pnl": bull_pnl,
                "miss_reason": reason,
            })

    catch_rate = len(caught) / len(bull_df) if len(bull_df) > 0 else 0

    # 如果抓住了, 计算贡献
    caught_pnl = sum(c["actual_pnl"] for c in caught)

    return {
        "caught": caught,
        "missed": missed,
        "catch_rate": round(catch_rate, 4),
        "n_bull": len(bull_df),
        "n_caught": len(caught),
        "caught_total_pnl": round(caught_pnl, 2),
    }


def _diagnose_miss(sym: str, factor_snapshot: pd.DataFrame) -> str:
    """诊断为什么没选中某只牛股。"""
    if sym not in factor_snapshot.index:
        return "不在因子截面中 (数据缺失/流动性过低/ST)"

    row = factor_snapshot.loc[sym]

    # 检查常见遗漏原因
    reasons = []

    avg_amt = row.get("_avg_amount_20", 0)
    if pd.notna(avg_amt) and avg_amt < 5000:
        reasons.append(f"成交额不足({avg_amt:.0f})")

    # 检查关键因子是否缺失
    key_factors = ["perm_entropy_m", "path_irrev_m", "vol_shrink", "mf_sm_proportion"]
    missing = [f for f in key_factors if f in factor_snapshot.columns and pd.isna(row.get(f))]
    if missing:
        reasons.append(f"因子缺失: {', '.join(missing)}")

    if not reasons:
        reasons.append("因子值在截面中排名不够高 (模型评分低)")

    return "; ".join(reasons)


def _analyze_contribution(trades: list[dict]) -> dict:
    """分析被选中股票的收益贡献。"""
    if not trades:
        return {}

    sorted_trades = sorted(trades, key=lambda t: t["pnl_pct"], reverse=True)
    total_pnl = sum(t["pnl_pct"] for t in trades)

    top_contributor = sorted_trades[0]
    worst = sorted_trades[-1]

    return {
        "total_pnl": round(total_pnl, 2),
        "top_contributor": {
            "symbol": top_contributor["symbol"],
            "name": top_contributor.get("name", ""),
            "pnl_pct": top_contributor["pnl_pct"],
        },
        "worst_performer": {
            "symbol": worst["symbol"],
            "name": worst.get("name", ""),
            "pnl_pct": worst["pnl_pct"],
        },
        "pnl_distribution": {
            ">5%": sum(1 for t in trades if t["pnl_pct"] > 5),
            "2~5%": sum(1 for t in trades if 2 <= t["pnl_pct"] <= 5),
            "0~2%": sum(1 for t in trades if 0 <= t["pnl_pct"] < 2),
            "-2~0%": sum(1 for t in trades if -2 <= t["pnl_pct"] < 0),
            "<-2%": sum(1 for t in trades if t["pnl_pct"] < -2),
        },
    }


def _overall_effectiveness(per_horizon: dict) -> dict:
    """跨 horizon 汇总策略有效性。"""
    if not per_horizon:
        return {"effective": False, "reason": "无回测结果"}

    effective_count = sum(
        1 for h in per_horizon.values()
        if h.get("effectiveness", {}).get("effective", False)
    )
    total = len(per_horizon)

    return {
        "effective": effective_count > total / 2,
        "effective_horizons": effective_count,
        "total_horizons": total,
        "ratio": round(effective_count / total, 2) if total > 0 else 0,
    }


def _generate_summary(analysis: dict) -> str:
    """生成一句话总结。"""
    eff = analysis.get("effectiveness", {})
    parts = []

    if eff.get("effective"):
        parts.append(f"策略有效 ({eff.get('effective_horizons')}/{eff.get('total_horizons')} 个 horizon 达标)")
    else:
        parts.append(f"策略待优化 ({eff.get('effective_horizons', 0)}/{eff.get('total_horizons', 0)} 个 horizon 达标)")

    # 大牛股捕获率
    for h, hr in analysis.get("per_horizon", {}).items():
        ca = hr.get("catch_analysis", {})
        if ca.get("n_bull", 0) > 0:
            parts.append(f"{h}: 捕获 {ca['n_caught']}/{ca['n_bull']} 只大牛股 ({ca['catch_rate']:.0%})")
            break  # 只报一个 horizon

    return "; ".join(parts)


def _print_report(analysis: dict, scan_date: str):
    """打印完整分析报告。"""
    print(f"\n{'='*70}")
    print(f"  策略分析报告 — scan_date: {scan_date}")
    print(f"{'='*70}")

    # 总结
    print(f"\n  总结: {analysis['summary']}")

    # 策略有效性
    eff = analysis["effectiveness"]
    print(f"\n  策略有效性: {'✓ 有效' if eff.get('effective') else '✗ 待优化'} "
          f"({eff.get('effective_horizons', 0)}/{eff.get('total_horizons', 0)})")

    for h, hr in analysis.get("per_horizon", {}).items():
        m = hr.get("metrics", {})
        e = hr.get("effectiveness", {})
        ca = hr.get("catch_analysis", {})
        contrib = hr.get("contribution", {})

        print(f"\n  {'─'*50}")
        print(f"  {h} (持有{hr.get('hold_days')}天, {hr.get('entry_date')}→{hr.get('exit_date')})")
        print(f"  {'─'*50}")

        # 绩效
        print(f"    交易: {m.get('n_trades', 0)} 笔, "
              f"胜率 {m.get('win_rate', 0):.1%}, "
              f"均收益 {m.get('avg_pnl', 0):+.2f}%, "
              f"{'✓' if e.get('effective') else '✗'}")

        if e.get("reasons"):
            for r in e["reasons"]:
                print(f"    ⚠ {r}")

        # 大牛股分析
        if ca.get("n_bull", 0) > 0:
            print(f"\n    大牛股捕获: {ca['n_caught']}/{ca['n_bull']} ({ca['catch_rate']:.0%})")

            if ca.get("caught"):
                print(f"    ✓ 抓住的:")
                for c in ca["caught"][:5]:
                    print(f"      {c['symbol']}: 大盘涨幅 {c['bull_pnl']:+.1f}%, "
                          f"实际收益 {c['actual_pnl']:+.1f}%")

            if ca.get("missed"):
                print(f"    ✗ 错过的 (Top 5):")
                for c in ca["missed"][:5]:
                    print(f"      {c['symbol']}: 涨幅 {c['bull_pnl']:+.1f}%, "
                          f"原因: {c['miss_reason']}")

        # 收益贡献
        if contrib:
            top = contrib.get("top_contributor", {})
            worst = contrib.get("worst_performer", {})
            print(f"\n    最大贡献: {top.get('symbol', '')} {top.get('name', '')} "
                  f"{top.get('pnl_pct', 0):+.1f}%")
            print(f"    最差表现: {worst.get('symbol', '')} {worst.get('name', '')} "
                  f"{worst.get('pnl_pct', 0):+.1f}%")

            dist = contrib.get("pnl_distribution", {})
            if dist:
                print(f"    收益分布: >5%={dist.get('>5%', 0)}, "
                      f"2~5%={dist.get('2~5%', 0)}, "
                      f"0~2%={dist.get('0~2%', 0)}, "
                      f"-2~0%={dist.get('-2~0%', 0)}, "
                      f"<-2%={dist.get('<-2%', 0)}")

    print()


def _build_calendar(data_dir: str) -> list[str]:
    all_dates: set[str] = set()
    csvs = sorted(glob.glob(os.path.join(data_dir, "*.csv")))[:50]
    for fpath in csvs:
        try:
            df = pd.read_csv(fpath, usecols=["trade_date"])
            all_dates.update(df["trade_date"].astype(str).tolist())
        except Exception:
            continue
    return sorted(all_dates)
