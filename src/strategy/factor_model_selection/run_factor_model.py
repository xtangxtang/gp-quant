"""
因子模型选股策略 — 主入口

三种选股模式:
  ic_scoring    — IC 加权多 horizon 截面评分 (线性)
  lgb_scoring   — LightGBM 模型选股 (非线性)
  profiling     — 因子有效性画像 (IC 分析)

用法:
  # IC 加权选股
  python -m src.strategy.factor_model_selection.run_factor_model \
    --mode ic_scoring \
    --feature_cache_dir /path/to/feature-cache \
    --scan_date 20260417

  # LightGBM 训练
  python -m src.strategy.factor_model_selection.run_factor_model \
    --mode lgb_train \
    --feature_cache_dir /path/to/feature-cache

  # LightGBM 选股
  python -m src.strategy.factor_model_selection.run_factor_model \
    --mode lgb_scoring \
    --feature_cache_dir /path/to/feature-cache \
    --scan_date 20260417

  # LightGBM walk-forward 回测
  python -m src.strategy.factor_model_selection.run_factor_model \
    --mode lgb_backtest_wf \
    --feature_cache_dir /path/to/feature-cache \
    --data_dir /path/to/tushare-daily-full \
    --backtest_start_date 20250101 --backtest_end_date 20250630

  # 因子画像
  python -m src.strategy.factor_model_selection.run_factor_model \
    --mode profiling \
    --feature_cache_dir /path/to/feature-cache
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Factor Model Selection Strategy")

    p.add_argument("--mode", required=True,
                   choices=["ic_scoring", "lgb_train", "lgb_scoring", "lgb_backtest", "lgb_backtest_wf", "profiling"],
                   help="运行模式")

    # 数据路径
    p.add_argument("--feature_cache_dir", required=True, help="特征缓存目录")
    p.add_argument("--data_dir", type=str, default="", help="日线 CSV 目录 (回测用)")
    p.add_argument("--basic_path", type=str, default="", help="tushare_stock_basic.csv")
    p.add_argument("--out_dir", type=str, default="results/factor_model_selection", help="输出目录")

    # 扫描参数
    p.add_argument("--scan_date", type=str, default="", help="扫描日期 YYYYMMDD")
    p.add_argument("--top_n", type=int, default=5, help="每个 horizon 选 N 只")

    # LightGBM 参数
    p.add_argument("--model_dir", type=str, default="", help="模型目录 (默认 cache_dir/lgb_models)")
    p.add_argument("--train_end", type=str, default="", help="训练截止日期")
    p.add_argument("--val_days", type=int, default=60, help="验证集天数")
    p.add_argument("--decay_lambda", type=float, default=0.007, help="时间衰减参数")
    p.add_argument("--n_estimators", type=int, default=500, help="LightGBM 树数")
    p.add_argument("--max_depth", type=int, default=5, help="最大深度")
    p.add_argument("--learning_rate", type=float, default=0.05, help="学习率")
    p.add_argument("--max_stocks", type=int, default=0, help="最大股票数 (0=全部)")
    p.add_argument("--workers", type=int, default=8, help="并行进程数")

    # 回测参数
    p.add_argument("--backtest_start_date", type=str, default="", help="回测起始日期")
    p.add_argument("--backtest_end_date", type=str, default="", help="回测结束日期")
    p.add_argument("--horizons", type=str, default="5d", help="回测 horizon (逗号分隔)")
    p.add_argument("--scan_interval", type=int, default=0, help="扫描间隔 (0=自动)")
    p.add_argument("--retrain_every", type=int, default=60, help="重训周期(交易日)")

    # profiling 参数
    p.add_argument("--forward_days", type=str, default="1,3,5", help="前瞻天数 (逗号分隔)")
    p.add_argument("--forward_weeks", type=str, default="1,3,5", help="前瞻周数 (逗号分隔)")

    p.add_argument("--verbose", action="store_true")
    return p


def _load_basic_info(basic_path: str) -> dict[str, dict]:
    if not basic_path or not os.path.exists(basic_path):
        return {}
    df = pd.read_csv(basic_path, dtype=str)
    info = {}
    for _, row in df.iterrows():
        ts = str(row.get("ts_code", ""))
        parts = ts.split(".")
        if len(parts) == 2:
            sym = parts[1].lower() + parts[0]
            info[sym] = {"name": str(row.get("name", "")), "industry": str(row.get("industry", ""))}
    return info


def _find_latest_profile_dir(cache_dir: str) -> str:
    profile_root = os.path.join(cache_dir, "factor_profile")
    if not os.path.isdir(profile_root):
        return ""
    subdirs = sorted([
        d for d in os.listdir(profile_root)
        if os.path.isdir(os.path.join(profile_root, d)) and d.isdigit()
    ])
    if not subdirs:
        return ""
    return os.path.join(profile_root, subdirs[-1])


def run_ic_scoring_mode(args):
    from .ic_scoring import (
        ScoringConfig, run_ic_scoring, load_market_ic_weights, print_ic_weights, ALL_HORIZONS,
    )

    profile_dir = _find_latest_profile_dir(args.feature_cache_dir)
    if not profile_dir:
        logger.error(f"No factor_profile directory found under {args.feature_cache_dir}")
        return

    daily_cache = os.path.join(args.feature_cache_dir, "daily")
    weekly_cache = os.path.join(args.feature_cache_dir, "weekly")
    basic_info = _load_basic_info(args.basic_path)

    scoring_cfg = ScoringConfig(
        top_n_per_horizon=args.top_n,
        min_amount=5000.0,
        exclude_st=True,
    )

    horizon_tops, combined = run_ic_scoring(
        profile_dir=profile_dir,
        daily_cache_dir=daily_cache,
        weekly_cache_dir=weekly_cache,
        scan_date=args.scan_date,
        basic_info=basic_info,
        cfg=scoring_cfg,
    )

    print(f"\n{'='*60}")
    print(f"  扫描日期: {args.scan_date or 'auto'}")
    print(f"  模式: IC 加权多 horizon 选股")
    print(f"  候选总数: {len(combined)} 只 ({combined['symbol'].nunique() if len(combined) > 0 else 0} 去重)")
    print(f"{'='*60}\n")

    # 打印 IC 权重
    ic_w = load_market_ic_weights(profile_dir)
    print_ic_weights(ic_w)

    for h in ALL_HORIZONS:
        df_h = horizon_tops.get(h, pd.DataFrame())
        if len(df_h) == 0:
            print(f"\n  {h}: (无候选)")
            continue
        print(f"\n  ── {h} Top {len(df_h)} ──")
        show_cols = [c for c in ["rank", "symbol", "name", "industry", "score", "top_factors"] if c in df_h.columns]
        print(df_h[show_cols].to_string(index=False))

    # 写出
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        scan_date = args.scan_date or "latest"
        if len(combined) > 0:
            combined.to_csv(os.path.join(args.out_dir, f"ic_top_candidates_{scan_date}.csv"), index=False)
        for h, df in horizon_tops.items():
            if len(df) > 0:
                df.to_csv(os.path.join(args.out_dir, f"ic_top_{h}_{scan_date}.csv"), index=False)
        print(f"\nResults → {args.out_dir}")


def run_lgb_mode(args, mode: str):
    from .factor_model import ModelConfig, train_all, run_lgb_scoring, run_backtest, run_backtest_wf, ALL_HORIZONS

    cfg = ModelConfig(
        cache_dir=args.feature_cache_dir,
        model_dir=args.model_dir,
        scan_date=args.scan_date,
        train_end=args.train_end,
        val_days=args.val_days,
        decay_lambda=args.decay_lambda,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        max_stocks=args.max_stocks,
        workers=args.workers,
    )
    model_dir = cfg.model_dir or os.path.join(cfg.cache_dir, "lgb_models")

    if mode == "lgb_train":
        train_all(cfg)

    elif mode == "lgb_scoring":
        basic_info = _load_basic_info(args.basic_path)
        horizon_tops, combined = run_lgb_scoring(
            model_dir=model_dir,
            daily_cache_dir=os.path.join(cfg.cache_dir, "daily"),
            weekly_cache_dir=os.path.join(cfg.cache_dir, "weekly"),
            scan_date=cfg.scan_date,
            basic_info=basic_info,
            cfg=cfg,
        )
        print(f"\n{'='*60}")
        print(f"  扫描日期: {cfg.scan_date or 'auto'}")
        print(f"  模式: LightGBM 模型选股")
        print(f"  候选总数: {len(combined)} 只 ({combined['symbol'].nunique() if len(combined) > 0 else 0} 去重)")
        print(f"{'='*60}\n")
        for h in ALL_HORIZONS:
            df_h = horizon_tops.get(h, pd.DataFrame())
            if len(df_h) == 0:
                print(f"\n  {h}: (无候选)")
                continue
            print(f"\n  ── {h} Top {len(df_h)} ──")
            print(df_h.to_string(index=False))
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            combined.to_csv(os.path.join(args.out_dir, f"lgb_candidates_{cfg.scan_date or 'latest'}.csv"), index=False)
            print(f"\nResults → {args.out_dir}")

    elif mode == "lgb_backtest":
        if not args.data_dir:
            logger.error("--data_dir required for backtest mode")
            return
        horizons = [h.strip() for h in args.horizons.split(",")]
        run_backtest(
            cache_dir=cfg.cache_dir,
            data_dir=args.data_dir,
            model_dir=model_dir,
            start_date=args.backtest_start_date,
            end_date=args.backtest_end_date,
            horizons=horizons,
            top_n=args.top_n,
            scan_interval=args.scan_interval,
            basic_path=args.basic_path,
            out_dir=args.out_dir,
        )

    elif mode == "lgb_backtest_wf":
        if not args.data_dir:
            logger.error("--data_dir required for backtest_wf mode")
            return
        horizons = [h.strip() for h in args.horizons.split(",")]
        run_backtest_wf(
            cache_dir=cfg.cache_dir,
            data_dir=args.data_dir,
            start_date=args.backtest_start_date,
            end_date=args.backtest_end_date,
            horizons=horizons,
            top_n=args.top_n,
            scan_interval=args.scan_interval,
            retrain_every=args.retrain_every,
            basic_path=args.basic_path,
            out_dir=args.out_dir,
            cfg=cfg,
        )


def run_profiling_mode(args):
    from .factor_profiling import ProfilingConfig, run_profiling

    forward_days = [int(x) for x in args.forward_days.split(",")]
    forward_weeks = [int(x) for x in args.forward_weeks.split(",")]

    pcfg = ProfilingConfig(
        cache_dir=args.feature_cache_dir,
        out_dir=args.out_dir,
        forward_days=forward_days,
        forward_weeks=forward_weeks,
        decay_lambda=args.decay_lambda,
        max_stocks=args.max_stocks if args.max_stocks > 0 else 0,
        workers=args.workers,
    )
    run_profiling(pcfg)


def main():
    args = _build_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.mode == "ic_scoring":
        run_ic_scoring_mode(args)
    elif args.mode.startswith("lgb_"):
        run_lgb_mode(args, args.mode)
    elif args.mode == "profiling":
        run_profiling_mode(args)


if __name__ == "__main__":
    main()
