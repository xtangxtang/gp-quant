"""
Bull Hunter v3 — CLI 入口

用法:
  # 单日扫描
  python -m src.strategy.factor_model_selection.v3_bull_hunter.run_bull_hunter \
      --scan_date 20260416

  # 滚动回测
  python -m src.strategy.factor_model_selection.v3_bull_hunter.run_bull_hunter \
      --rolling --start_date 20250101 --end_date 20250630

  # 自定义路径
  python -m src.strategy.factor_model_selection.v3_bull_hunter.run_bull_hunter \
      --scan_date 20260416 \
      --cache_dir /path/to/feature-cache \
      --data_dir /path/to/tushare-daily-full
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from .pipeline import PipelineConfig, run_backtest, run_rolling, run_scan
from .agent2_train import TrainConfig
from .agent3_predict import PredictConfig


def main():
    parser = argparse.ArgumentParser(description="Bull Hunter v3 — 大牛股预测系统")

    # 模式
    parser.add_argument("--scan_date", type=str, default="",
                        help="单日扫描日期 (YYYYMMDD)")
    parser.add_argument("--rolling", action="store_true",
                        help="滚动回测模式")
    parser.add_argument("--backtest", action="store_true",
                        help="回测模式 (滚动扫描 + 实际收益验证)")
    parser.add_argument("--start_date", type=str, default="",
                        help="滚动/回测起始日期")
    parser.add_argument("--end_date", type=str, default="",
                        help="滚动/回测结束日期")

    # 路径
    parser.add_argument("--cache_dir", type=str,
                        default="/nvme5/xtang/gp-workspace/gp-data/feature-cache",
                        help="特征缓存目录")
    parser.add_argument("--data_dir", type=str,
                        default="/nvme5/xtang/gp-workspace/gp-data/tushare-daily-full",
                        help="日线数据目录")
    parser.add_argument("--basic_path", type=str,
                        default="/nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv",
                        help="股票基本信息 CSV")
    parser.add_argument("--results_dir", type=str,
                        default="results/bull_hunter",
                        help="结果输出目录")

    # 训练参数
    parser.add_argument("--lookback_months", type=int, default=12,
                        help="训练回看月数")
    parser.add_argument("--n_estimators", type=int, default=800,
                        help="LightGBM 树数")
    parser.add_argument("--learning_rate", type=float, default=0.03,
                        help="LightGBM 学习率")

    # 预测阈值
    parser.add_argument("--threshold_30", type=float, default=0.15)
    parser.add_argument("--threshold_100", type=float, default=0.15)
    parser.add_argument("--threshold_200", type=float, default=0.15)
    parser.add_argument("--top_n", type=int, default=20)

    # 滚动间隔
    parser.add_argument("--interval_days", type=int, default=20,
                        help="滚动回测间隔天数")

    args = parser.parse_args()

    # 日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # 构建配置
    train_cfg = TrainConfig(
        lookback_months=args.lookback_months,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
    )
    predict_cfg = PredictConfig(
        threshold_30pct=args.threshold_30,
        threshold_100pct=args.threshold_100,
        threshold_200pct=args.threshold_200,
        top_n_per_grade=args.top_n,
    )
    pipe_cfg = PipelineConfig(
        cache_dir=args.cache_dir,
        data_dir=args.data_dir,
        basic_path=args.basic_path,
        results_dir=args.results_dir,
        train_cfg=train_cfg,
        predict_cfg=predict_cfg,
    )

    if args.backtest:
        if not args.start_date or not args.end_date:
            print("ERROR: --backtest 模式需要 --start_date 和 --end_date")
            sys.exit(1)
        result = run_backtest(
            args.start_date, args.end_date, pipe_cfg,
            args.interval_days, args.top_n,
        )
        if not result.empty:
            _print_backtest_summary(result, args)
    elif args.rolling:
        if not args.start_date or not args.end_date:
            print("ERROR: --rolling 模式需要 --start_date 和 --end_date")
            sys.exit(1)
        run_rolling(args.start_date, args.end_date, pipe_cfg, args.interval_days)
    elif args.scan_date:
        result = run_scan(args.scan_date, pipe_cfg)
        if not result.empty:
            # 读取健康报告
            import json as _json
            health_path = os.path.join(args.results_dir, args.scan_date, "health_report.json")
            health = {}
            if os.path.exists(health_path):
                with open(health_path) as _f:
                    health = _json.load(_f)

            # 读取模型 meta
            meta_path = os.path.join(args.cache_dir, "bull_models", args.scan_date, "meta.json")
            meta = {}
            if os.path.exists(meta_path):
                with open(meta_path) as _f:
                    meta = _json.load(_f)

            grade_label = {"A": "大牛股(200%)", "B": "翻倍股(100%)", "C": "短线强势(30%)"}
            prob_label = {"prob_30": "2周涨30%", "prob_100": "2月翻倍", "prob_200": "6月涨200%"}

            n_a = (result["grade"] == "A").sum()
            n_b = (result["grade"] == "B").sum()
            n_c = (result["grade"] == "C").sum()

            print(f"\n{'='*60}")
            print(f"  Bull Hunter v3 — 大牛股预测报告")
            print(f"  扫描日期: {args.scan_date}")
            print(f"{'='*60}")

            # 模型质量摘要
            if meta:
                print(f"\n📊 模型质量:")
                for tname in ["30pct", "100pct", "200pct"]:
                    m = meta.get(tname, {})
                    auc = m.get("val_auc", 0)
                    prec = m.get("val_precision", 0)
                    rec = m.get("val_recall", 0)
                    th = m.get("best_threshold", 0)
                    n_tr = m.get("n_train", 0)
                    pos_r = m.get("pos_rate_train", 0)
                    top3 = [f["name"] for f in m.get("top_features", [])[:3]]
                    print(f"  {tname:8s}: AUC={auc:.3f}  精确率={prec:.1%}  "
                          f"召回率={rec:.1%}  最优阈值={th:.2f}  "
                          f"训练={n_tr}条(正样本{pos_r:.1%})")
                    if top3:
                        print(f"            关键因子: {', '.join(top3)}")

            # 健康状态
            status = health.get("status", "unknown")
            status_cn = {"healthy": "✅ 健康", "warning": "⚠️ 警告", "critical": "🔴 严重"}.get(status, status)
            print(f"\n🏥 系统状态: {status_cn}")
            suggestions = health.get("suggestions", [])
            if suggestions:
                for s in suggestions:
                    print(f"  💡 {s}")

            # 候选列表
            print(f"\n🎯 候选列表: 共 {len(result)} 只 (A级={n_a}, B级={n_b}, C级={n_c})")
            print(f"{'-'*60}")

            for grade in ["A", "B", "C"]:
                subset = result[result["grade"] == grade]
                if subset.empty:
                    continue
                print(f"\n  【{grade}级 — {grade_label.get(grade, '')}】")
                for _, row in subset.iterrows():
                    sym = row["symbol"]
                    name = row.get("name", "")
                    parts = []
                    for c, label in prob_label.items():
                        if c in row:
                            parts.append(f"{label}={row[c]:.1%}")
                    print(f"    {sym} {name:<8s} | {' | '.join(parts)}")

            print(f"\n{'='*60}")
    else:
        print("ERROR: 需要 --scan_date, --rolling 或 --backtest")
        sys.exit(1)


def _print_backtest_summary(result, args):
    """打印回测结果中文摘要。"""
    import numpy as np

    n_dates = result["scan_date"].nunique()
    n_total = len(result)

    print(f"\n{'='*70}")
    print(f"  Bull Hunter v3 — 回测报告")
    print(f"  回测区间: {args.start_date} ~ {args.end_date}")
    print(f"  扫描间隔: 每 {args.interval_days} 个交易日")
    print(f"  扫描次数: {n_dates} 次, 总预测: {n_total} 条")
    print(f"{'='*70}")

    # 按等级统计
    for grade, label in [("A", "大牛股(200%)"), ("B", "翻倍股(100%)"),
                         ("C", "短线强势(30%)"), ("ALL", "全部")]:
        subset = result if grade == "ALL" else result[result["grade"] == grade]
        if subset.empty:
            continue

        n = len(subset)
        print(f"\n  【{grade}级 — {label}】 {n} 条预测")
        print(f"  {'-'*66}")

        for fwd_days, target, fwd_label in [
            (10, 0.30, "10日(短线)"),
            (40, 1.00, "40日(中线)"),
            (120, 2.00, "120日(长线)"),
        ]:
            col = f"actual_{fwd_days}d"
            if col not in subset.columns:
                continue
            valid = subset[col].dropna()
            if len(valid) == 0:
                print(f"    {fwd_label}: 无数据")
                continue

            avg = valid.mean()
            med = valid.median()
            win_rate = (valid > 0).mean()
            hit_rate = (valid >= target).mean()
            max_gain = valid.max()
            max_loss = valid.min()

            print(f"    {fwd_label}: 均值={avg:+.1%}  中位数={med:+.1%}  "
                  f"胜率={win_rate:.0%}  命中率(≥{target:.0%})={hit_rate:.0%}  "
                  f"最大涨={max_gain:+.1%}  最大亏={max_loss:+.1%}  "
                  f"(有效={len(valid)}/{n})")

        # 40d 最大涨幅
        if "actual_max_40d" in subset.columns:
            max40 = subset["actual_max_40d"].dropna()
            if len(max40) > 0:
                print(f"    40日最大涨幅: 均值={max40.mean():+.1%}  "
                      f">=30%比例={float((max40 >= 0.30).mean()):.0%}  "
                      f">=100%比例={float((max40 >= 1.00).mean()):.0%}")

    # 按扫描日期的时间序列
    print(f"\n  📅 按扫描日期:")
    print(f"  {'扫描日':>10s}  {'数量':>4s}  {'10d均值':>8s}  {'10d胜率':>7s}  "
          f"{'40d均值':>8s}  {'40d胜率':>7s}")
    print(f"  {'-'*58}")
    for sd in sorted(result["scan_date"].unique()):
        sub = result[result["scan_date"] == sd]
        n = len(sub)
        parts = [f"  {sd:>10s}  {n:>4d}"]
        for fwd in [10, 40]:
            col = f"actual_{fwd}d"
            if col in sub.columns:
                valid = sub[col].dropna()
                if len(valid) > 0:
                    parts.append(f"  {valid.mean():>+7.1%}")
                    parts.append(f"  {(valid > 0).mean():>6.0%}")
                else:
                    parts.append(f"  {'N/A':>7s}")
                    parts.append(f"  {'N/A':>6s}")
        print("".join(parts))

    out_dir = os.path.join(args.results_dir, f"backtest_{args.start_date}_{args.end_date}")
    print(f"\n  📁 结果目录: {out_dir}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
