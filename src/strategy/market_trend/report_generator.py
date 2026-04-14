"""
大盘趋势判断策略 - 每日诊断报告生成器

生成包含趋势判定、7 维度分析、板块资金流向、关键信号的 Markdown 报告。
"""

import os
from typing import List

from .config import MarketTrendConfig, TrendState
from .micro_indicators import SectorFlow


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

_TREND_EMOJI = {
    "STRONG_UP": "🔴 强势上涨",
    "UP": "🟠 上涨",
    "NEUTRAL": "🟡 中性震荡",
    "DOWN": "🟢 下跌",
    "STRONG_DOWN": "🔵 强势下跌",
}

_SCORE_LABEL = {
    'breadth': '广度 Breadth',
    'money_flow': '资金流 Money Flow',
    'volatility': '波动结构 Volatility',
    'entropy': '熵/有序度 Entropy',
    'momentum': '动量扩散 Momentum',
    'leverage': '杠杆资金 Leverage',
    'liquidity': '流动性 Liquidity',
}


def _fmt_date(d: str) -> str:
    return f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 else d


def _score_status(s: float) -> str:
    if s >= 0.3:
        return "强正面"
    if s >= 0.1:
        return "偏正面"
    if s > -0.1:
        return "中性"
    if s > -0.3:
        return "偏负面"
    return "强负面"


def _score_bar(s: float) -> str:
    """[-1, 1] → ASCII bar."""
    n = int(round((s + 1) / 2 * 10))
    n = max(0, min(10, n))
    return "█" * n + "░" * (10 - n)


# ---------------------------------------------------------------------------
# 综合判断文字 (自动生成)
# ---------------------------------------------------------------------------

def _auto_summary(state: TrendState, sector_flows: List[SectorFlow]) -> str:
    parts: List[str] = []

    # 趋势
    trend_cn = _TREND_EMOJI.get(state.trend, state.trend)
    parts.append(f"市场处于 **{trend_cn}** 状态，综合得分 {state.composite_score:+.3f}。")

    # 广度
    if state.advance_ratio < 0.4:
        parts.append(f"市场广度偏弱，仅 {state.advance_ratio:.0%} 个股上涨。")
    elif state.advance_ratio > 0.6:
        parts.append(f"市场广度良好，{state.advance_ratio:.0%} 个股上涨。")

    # 资金流
    net_billion = state.big_order_net_sum  # 已是亿
    if net_billion > 10:
        parts.append(f"大单资金净流入 {net_billion:.1f} 亿，主力积极进场。")
    elif net_billion < -10:
        parts.append(f"大单资金净流出 {abs(net_billion):.1f} 亿，主力离场。")

    # 流动性
    if state.shibor_on < 1.5:
        parts.append(f"隔夜 SHIBOR {state.shibor_on:.2f}% 处于低位，流动性充裕。")
    elif state.shibor_on > 2.5:
        parts.append(f"隔夜 SHIBOR {state.shibor_on:.2f}% 偏高，资金面偏紧。")

    # 两融
    if state.leverage_score > 0.3:
        parts.append("杠杆资金持续流入，市场做多情绪升温。")
    elif state.leverage_score < -0.3:
        parts.append("杠杆资金回落，增量资金不足。")

    # 板块方向
    if sector_flows:
        top = sector_flows[0]
        bot = sector_flows[-1]
        if top.big_order_net > 0:
            parts.append(f"资金集中流入 **{top.sector_name}**"
                         f"（净额 {top.big_order_net / 1e4:+.1f} 亿）。")
        if bot.big_order_net < 0:
            parts.append(f"**{bot.sector_name}** 资金流出最大"
                         f"（净额 {bot.big_order_net / 1e4:+.1f} 亿）。")

    # 动量一致性
    if state.momentum_score < -0.3:
        parts.append("板块动量分化，机构方向不明确。")
    elif state.momentum_score > 0.3:
        parts.append("板块动量一致，市场方向明确。")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 核心报告生成
# ---------------------------------------------------------------------------

def generate_daily_report(
    state: TrendState,
    sector_flows: List[SectorFlow],
    recent_states: List[TrendState],
    cfg: MarketTrendConfig,
) -> str:
    """生成单日综合诊断报告 (Markdown)。"""

    date_str = _fmt_date(state.date)
    lines: List[str] = []

    # ---- 标题 ----
    lines.append(f"# 市场趋势诊断报告 — {date_str}\n")

    # ---- 综合判断 ----
    lines.append("## 综合判断\n")
    lines.append(_auto_summary(state, sector_flows))
    lines.append("")

    # ---- 核心数据面板 ----
    lines.append("## 核心数据\n")
    lines.append(f"| 指标 | 值 |")
    lines.append(f"|------|------|")
    lines.append(f"| 趋势状态 | **{state.trend}** |")
    lines.append(f"| 综合得分 | {state.composite_score:+.4f} |")
    if state.index_close > 0:
        lines.append(f"| 上证综指 | {state.index_close:.2f} ({state.index_pct_chg:+.2f}%) |")
    lines.append(f"| 上涨 / 总数 | {int(state.advance_ratio * state.total_stocks)}"
                 f" / {state.total_stocks} ({state.advance_ratio:.1%}) |")
    lines.append(f"| 涨停 / 跌停 | {state.limit_up_count} / {state.limit_down_count} |")
    lines.append(f"| 大单净额 | {state.big_order_net_sum:+.1f} 亿 |")
    lines.append(f"| 两融余额 | {state.margin_balance:.0f} 亿 |")
    lines.append(f"| 隔夜 SHIBOR | {state.shibor_on:.4f}% |")
    lines.append(f"| 广度推力 | {'触发' if state.breadth_thrust else '未触发'} |")
    lines.append("")

    # ---- 7 维度分析 ----
    lines.append("## 7 维度分析\n")

    dim_data = [
        ('breadth', state.breadth_score),
        ('money_flow', state.money_flow_score),
        ('volatility', state.volatility_score),
        ('entropy', state.entropy_score),
        ('momentum', state.momentum_score),
        ('leverage', state.leverage_score),
        ('liquidity', state.liquidity_score),
    ]

    lines.append("| 维度 | 得分 | 权重 | 加权贡献 | 状态 | 强度 |")
    lines.append("|------|------|------|----------|------|------|")
    for key, score in dim_data:
        w = cfg.weights.get(key, 0.0)
        contrib = score * w
        label = _SCORE_LABEL.get(key, key)
        status = _score_status(score)
        bar = _score_bar(score)
        lines.append(f"| {label} | {score:+.3f} | {w:.0%} | {contrib:+.4f} | {status} | {bar} |")
    lines.append("")

    # 维度解读
    lines.append("### 维度解读\n")
    _add_dimension_insights(lines, state)
    lines.append("")

    # ---- 板块资金流向 ----
    lines.append("## 板块资金流向\n")
    top_n = cfg.report_top_n

    if sector_flows:
        # 净流入 Top
        inflows = [sf for sf in sector_flows if sf.big_order_net > 0]
        outflows = [sf for sf in sector_flows if sf.big_order_net < 0]

        if inflows:
            lines.append(f"### 净流入 Top {min(top_n, len(inflows))}\n")
            lines.append("| 板块 | 大单净额(亿) | 上涨比 | 均涨幅(%) | 20日均收益(%) | 股票数 |")
            lines.append("|------|-------------|--------|----------|--------------|--------|")
            for sf in inflows[:top_n]:
                lines.append(
                    f"| {sf.sector_name} "
                    f"| {sf.big_order_net / 1e4:+.2f} "
                    f"| {sf.advance_ratio:.0%} "
                    f"| {sf.avg_pct_chg:+.2f} "
                    f"| {sf.avg_ret_20d * 100:+.1f} "
                    f"| {sf.stock_count} |"
                )
            lines.append("")

        if outflows:
            lines.append(f"### 净流出 Top {min(top_n, len(outflows))}\n")
            lines.append("| 板块 | 大单净额(亿) | 上涨比 | 均涨幅(%) | 20日均收益(%) | 股票数 |")
            lines.append("|------|-------------|--------|----------|--------------|--------|")
            for sf in sorted(outflows, key=lambda x: x.big_order_net)[:top_n]:
                lines.append(
                    f"| {sf.sector_name} "
                    f"| {sf.big_order_net / 1e4:+.2f} "
                    f"| {sf.advance_ratio:.0%} "
                    f"| {sf.avg_pct_chg:+.2f} "
                    f"| {sf.avg_ret_20d * 100:+.1f} "
                    f"| {sf.stock_count} |"
                )
            lines.append("")

        # 板块全景
        lines.append("### 板块全景一览\n")
        lines.append("| 板块 | 净额(亿) | 上涨比 | 均涨幅 | 方向 |")
        lines.append("|------|---------|--------|--------|------|")
        for sf in sector_flows:
            net_b = sf.big_order_net / 1e4
            arrow = "↑" if net_b > 0.5 else ("↓" if net_b < -0.5 else "→")
            lines.append(
                f"| {sf.sector_name} "
                f"| {net_b:+.2f} "
                f"| {sf.advance_ratio:.0%} "
                f"| {sf.avg_pct_chg:+.2f}% "
                f"| {arrow} |"
            )
        lines.append("")
    else:
        lines.append("*无行业分类数据，跳过板块分析*\n")

    # ---- 关键信号 ----
    lines.append("## 关键信号与风险提示\n")
    _add_signals(lines, state, sector_flows)
    lines.append("")

    # ---- 近期趋势对比 ----
    if recent_states and len(recent_states) > 1:
        lines.append("## 近期趋势变化\n")
        lines.append("| 日期 | 趋势 | 综合得分 | 广度 | 资金流 | 动量 | 杠杆 | 流动性 |")
        lines.append("|------|------|---------|------|--------|------|------|--------|")
        for rs in recent_states[-10:]:
            lines.append(
                f"| {_fmt_date(rs.date)} "
                f"| {rs.trend} "
                f"| {rs.composite_score:+.3f} "
                f"| {rs.breadth_score:+.2f} "
                f"| {rs.money_flow_score:+.2f} "
                f"| {rs.momentum_score:+.2f} "
                f"| {rs.leverage_score:+.2f} "
                f"| {rs.liquidity_score:+.2f} |"
            )
        lines.append("")

        # 趋势分布
        recent_10 = recent_states[-10:]
        from collections import Counter
        cnt = Counter(rs.trend for rs in recent_10)
        lines.append(f"最近 {len(recent_10)} 天趋势分布: " + " / ".join(
            f"{t} {c}天" for t, c in cnt.most_common()
        ))
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 维度解读
# ---------------------------------------------------------------------------

def _add_dimension_insights(lines: List[str], state: TrendState) -> None:
    # 广度
    lines.append(f"- **广度**: 上涨占比 {state.advance_ratio:.1%}，"
                 f"MA20 以上 {state.above_ma20_ratio:.1%}，"
                 f"MA60 以上 {state.above_ma60_ratio:.1%}，"
                 f"创新高 {state.new_high_ratio:.1%}")

    # 资金流
    lines.append(f"- **资金流**: 净流入正的个股占 {state.net_inflow_ratio:.1%}，"
                 f"全市场大单净额 {state.big_order_net_sum:+.1f} 亿")

    # 波动
    lines.append(f"- **波动**: 中位数波动率 {state.vol_median:.4f}，"
                 f"恐慌度(跌>5%) {state.panic_ratio:.2%}")

    # 熵
    lines.append(f"- **熵**: 排列熵中位数 {state.entropy_median:.3f}，"
                 f"有序股票占比 {state.ordering_ratio:.1%}")

    # 动量
    lines.append(f"- **动量**: 行业动量离散 {state.sector_momentum_std:.4f}，"
                 f"方向一致性 {state.trend_alignment:.1%}")

    # 杠杆
    lines.append(f"- **杠杆**: 两融余额 {state.margin_balance:.0f} 亿，"
                 f"融资净买入 {state.margin_net_buy:+.1f} 亿")

    # 流动性
    lines.append(f"- **流动性**: SHIBOR 隔夜 {state.shibor_on:.4f}%，"
                 f"日变化 {state.shibor_on_change:+.4f}%")


# ---------------------------------------------------------------------------
# 信号检测
# ---------------------------------------------------------------------------

def _add_signals(lines: List[str], state: TrendState,
                 sector_flows: List[SectorFlow]) -> None:
    signals: List[str] = []

    # 广度信号
    if state.breadth_thrust:
        signals.append("✅ **广度推力触发** — 短期强势信号")
    if state.advance_ratio < 0.35:
        signals.append(f"⚠️ **广度极弱** — 仅 {state.advance_ratio:.0%} 股上涨，市场极度分化")
    elif state.advance_ratio < 0.45:
        signals.append(f"⚠️ **广度不足** — {state.advance_ratio:.0%} 股上涨，缺乏赚钱效应")
    if state.above_ma20_ratio < 0.3:
        signals.append(f"⚠️ **均线破位** — 仅 {state.above_ma20_ratio:.0%} 股在 MA20 以上")

    # 涨跌停
    if state.limit_down_count > 30:
        signals.append(f"🔴 **跌停潮** — {state.limit_down_count} 家跌停")
    if state.limit_up_count > 50:
        signals.append(f"✅ **涨停活跃** — {state.limit_up_count} 家涨停")

    # 资金流
    net_b = state.big_order_net_sum
    if net_b < -50:
        signals.append(f"🔴 **大单巨额流出** — 净流出 {abs(net_b):.0f} 亿")
    elif net_b < -20:
        signals.append(f"⚠️ **大单净流出** — 净流出 {abs(net_b):.0f} 亿")
    elif net_b > 20:
        signals.append(f"✅ **大单净流入** — 净流入 {net_b:.0f} 亿")

    # 恐慌
    if state.panic_ratio > 0.05:
        signals.append(f"🔴 **恐慌性抛售** — {state.panic_ratio:.1%} 个股跌超 5%")
    elif state.panic_ratio > 0.02:
        signals.append(f"⚠️ **局部恐慌** — {state.panic_ratio:.1%} 个股跌超 5%")

    # 流动性
    if state.shibor_on < 1.2:
        signals.append(f"✅ **流动性极度宽松** — SHIBOR {state.shibor_on:.2f}%")
    if state.shibor_on_change > 0.3:
        signals.append(f"⚠️ **SHIBOR 飙升** — 日变化 +{state.shibor_on_change:.2f}%，资金面收紧")

    # 杠杆
    if state.margin_net_buy > 30:
        signals.append(f"✅ **融资大幅净买入** — {state.margin_net_buy:+.0f} 亿")
    elif state.margin_net_buy < -30:
        signals.append(f"⚠️ **融资大幅净偿还** — {state.margin_net_buy:+.0f} 亿")

    # 动量分化
    if state.momentum_score < -0.5:
        signals.append("⚠️ **板块严重分化** — 动量方向不一致，缺乏主线")
    elif state.momentum_score > 0.5:
        signals.append("✅ **板块共振** — 各行业动量一致向上")

    # 板块集中度
    if sector_flows and len(sector_flows) >= 5:
        top3_net = sum(sf.big_order_net for sf in sector_flows[:3])
        total_net = sum(abs(sf.big_order_net) for sf in sector_flows)
        if total_net > 0:
            concentration = abs(top3_net) / total_net
            if concentration > 0.5:
                top3_names = "、".join(sf.sector_name for sf in sector_flows[:3])
                signals.append(f"📌 **资金高度集中** — Top3({top3_names}) 占全市场"
                               f" {concentration:.0%}")

    if not signals:
        signals.append("无显著信号")

    for s in signals:
        lines.append(f"- {s}")


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def write_report(
    report_text: str,
    sector_flows: List[SectorFlow],
    date: str,
    out_dir: str,
) -> None:
    """写入报告文件和板块流向 CSV。"""
    os.makedirs(out_dir, exist_ok=True)

    # Markdown 报告
    report_path = os.path.join(out_dir, f"market_trend_report_{date}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"  → 诊断报告: {report_path}")

    # 板块流向 CSV
    if sector_flows:
        import pandas as pd
        rows = []
        for sf in sector_flows:
            rows.append({
                "sector": sf.sector_name,
                "stock_count": sf.stock_count,
                "advance_count": sf.advance_count,
                "advance_ratio": round(sf.advance_ratio, 4),
                "big_order_net_wan": round(sf.big_order_net, 0),
                "big_order_net_yi": round(sf.big_order_net / 1e4, 2),
                "net_inflow_count": sf.net_inflow_count,
                "net_inflow_ratio": round(sf.net_inflow_ratio, 4),
                "avg_pct_chg": round(sf.avg_pct_chg, 4),
                "avg_ret_20d": round(sf.avg_ret_20d, 6),
            })
        df = pd.DataFrame(rows)
        csv_path = os.path.join(out_dir, f"sector_flows_{date}.csv")
        df.to_csv(csv_path, index=False)
        print(f"  → 板块流向: {csv_path} ({len(df)} 板块)")
