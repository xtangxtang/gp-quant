"""
训练 Attention Learner 模型 (Phase 1)

用法:
  # 全量训练
  python -m src.strategy.adaptive_state_machine.train_attention \
    --data-dir /path/to/gp-data/tushare-daily-full \
    --model-path models/attention_model.pt

  # Walk-Forward 训练 (推荐, 避免未来信息泄漏)
  python -m src.strategy.adaptive_state_machine.train_attention \
    --data-dir ... \
    --model-path models/attention_model.pt \
    --walk-forward \
    --train-years 2024,2025 \
    --eval-year 2026
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _compute_one_symbol_factors(
    daily_dir: str,
    symbol: str,
    scan_date: str = "",
) -> tuple[str, pd.DataFrame]:
    """计算单只股票的因子序列"""
    try:
        from src.strategy.entropy_accumulation_breakout.feature_engine import (
            build_features,
        )
    except ImportError:
        return symbol, pd.DataFrame()

    fpath = os.path.join(daily_dir, f"{symbol}.csv")
    if not os.path.exists(fpath):
        return symbol, pd.DataFrame()

    try:
        df = pd.read_csv(fpath, on_bad_lines="skip")
    except Exception:
        return symbol, pd.DataFrame()

    if "trade_date" not in df.columns or "close" not in df.columns:
        return symbol, pd.DataFrame()

    df["trade_date"] = df["trade_date"].astype(str)
    if scan_date:
        df = df[df["trade_date"] <= scan_date]

    if len(df) < 80:
        return symbol, pd.DataFrame()

    try:
        result = build_features(df_daily=df, symbol=symbol)
        df_featured = result.get("daily")
        if df_featured is None or len(df_featured) < 80:
            return symbol, pd.DataFrame()
        return symbol, df_featured
    except Exception as e:
        logger.debug(f"Error computing {symbol}: {e}")
        return symbol, pd.DataFrame()


def build_training_data(
    daily_dir: str,
    max_stocks: int = 500,
    scan_date: str = "",
    seq_len: int = 60,
    forward_days: int = 10,
) -> tuple:
    """
    构建训练数据: 计算因子 + 构建序列 + 生成标签。

    Returns:
        (X, y_reg, y_cls, sample_weights, factor_names, trade_dates)
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    csv_files = glob.glob(os.path.join(daily_dir, "*.csv"))
    symbols = [os.path.basename(f).replace(".csv", "") for f in csv_files]

    if len(symbols) > max_stocks:
        np.random.seed(42)
        symbols = list(np.random.choice(symbols, max_stocks, replace=False))

    logger.info(f"Computing factors for {len(symbols)} stocks...")

    max_workers = min(28, os.cpu_count() or 4)
    logger.info(f"Using {max_workers} workers")

    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_compute_one_symbol_factors, daily_dir, sym, scan_date): sym
            for sym in symbols
        }

        for future in as_completed(futures):
            sym, df = future.result()
            if not df.empty:
                results[sym] = df

    logger.info(f"Computed factors for {len(results)}/{len(symbols)} stocks")
    if not results:
        raise ValueError("No valid factor data")

    # 构建全局交易日期列表
    all_dates = set()
    for sym, df in results.items():
        all_dates.update(df["trade_date"].tolist())
    trade_dates = sorted(all_dates)
    date_to_idx = {d: i for i, d in enumerate(trade_dates)}

    # 构建因子名列表
    all_factor_cols = set()
    for df in results.values():
        exclude = {"trade_date", "open", "high", "low", "close", "vol", "amount",
                    "turnover_rate", "net_mf_amount", "symbol"}
        cols = {c for c in df.columns if c not in exclude}
        all_factor_cols = all_factor_cols.intersection(cols) if all_factor_cols else cols

    factor_names = sorted(all_factor_cols)
    # Filter to only numeric columns (exclude ts_code etc.)
    sample_df = next(iter(results.values()))
    numeric_cols = set(sample_df.select_dtypes(include=[np.number]).columns)
    factor_names = [c for c in factor_names if c in numeric_cols]
    # Filter to only numeric columns (exclude ts_code etc.)
    sample_df = next(iter(results.values()))
    numeric_cols = set(sample_df.select_dtypes(include=[np.number]).columns)
    factor_names = [c for c in factor_names if c in numeric_cols]
    logger.info(f"Numeric factor columns: {len(factor_names)}")

    from src.strategy.adaptive_state_machine.attention_learner import _standardize_factors

    X_list = []
    y_reg_list = []
    y_cls_list = []
    sample_weights_list = []

    current_year = 2026
    time_decay_lambda = 0.02

    for sym, df in results.items():
        if len(df) < seq_len + forward_days:
            continue

        factor_cols = [c for c in factor_names if c in df.columns]
        if len(factor_cols) < 10:
            continue

        factor_values = df[factor_cols].select_dtypes(include=[np.number]).values
        sym_dates = df["trade_date"].tolist()

        for i in range(seq_len, len(factor_values) - forward_days):
            x_seq = factor_values[i - seq_len:i]
            if np.any(np.isnan(x_seq)):
                continue

            entry_date = str(sym_dates[i])
            future_idx = date_to_idx.get(entry_date, -1)
            if future_idx < 0:
                continue
            target_idx = future_idx + forward_days
            if target_idx >= len(trade_dates):
                continue

            try:
                target_date = trade_dates[target_idx]
                target_pos = sym_dates.index(target_date) if target_date in sym_dates else -1
                if target_pos < 0:
                    continue

                entry_price = float(df.iloc[i]["close"])
                exit_price = float(df.iloc[target_pos]["close"])

                if entry_price <= 0 or exit_price <= 0:
                    continue

                ret = (exit_price - entry_price) / entry_price
                if abs(ret) > 0.5:
                    continue

                X_list.append(x_seq)
                y_reg_list.append(ret)
                y_cls_list.append(1 if ret > 0 else 0)

                # 时间衰减权重
                year = int(entry_date[:4]) if len(entry_date) >= 4 else current_year
                years_ago = current_year - year
                weight = np.exp(-time_decay_lambda * years_ago * 365)
                sample_weights_list.append(weight)

            except (KeyError, TypeError, ValueError):
                continue

    if not X_list:
        raise ValueError("No valid training samples")

    X = np.array(X_list, dtype=np.float32)
    y_reg = np.array(y_reg_list, dtype=np.float32)
    y_cls = np.array(y_cls_list, dtype=np.int64)
    sample_weights = np.array(sample_weights_list, dtype=np.float32)

    # 标准化
    X_flat = X.reshape(-1, X.shape[-1])
    X_std, mean, std = _standardize_factors(X_flat)
    X = X_std.reshape(X.shape)

    logger.info(f"Training data: {len(X)} samples, {X.shape[-1]} factors, "
                 f"seq_len={seq_len}")
    logger.info(f"  Return stats: mean={y_reg.mean():.4f}, "
                 f"std={y_reg.std():.4f}, "
                 f"up_ratio={(y_cls==1).mean():.1%}")

    return X, y_reg, y_cls, sample_weights, factor_names, trade_dates, mean, std


def _finalize_samples(X_list, y_reg_list, y_cls_list, w_list):
    """将列表转换为 numpy 数组并标准化"""
    from src.strategy.adaptive_state_machine.attention_learner import _standardize_factors
    X = np.array(X_list, dtype=np.float32)
    y_reg = np.array(y_reg_list, dtype=np.float32)
    y_cls = np.array(y_cls_list, dtype=np.int64)
    w = np.array(w_list, dtype=np.float32)
    X_flat = X.reshape(-1, X.shape[-1])
    X_std, mean, std = _standardize_factors(X_flat)
    return X_std.reshape(X.shape), y_reg, y_cls, w, mean, std


def build_cross_sectional_data(
    daily_dir: str,
    train_years: list[int],
    eval_year: int,
    max_stocks: int = 800,
    seq_len: int = 60,
    forward_days: int = 10,
) -> tuple:
    """
    Walk-Forward 截面数据构建: 按日期分组。

    Returns:
        (train_sections, eval_sections, factor_names, train_mean, train_std)
        sections = {date_str: {"X": (N, seq_len, n_factors), "y_reg": (N,), "y_cls": (N,)}}
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    csv_files = glob.glob(os.path.join(daily_dir, "*.csv"))
    symbols = [os.path.basename(f).replace(".csv", "") for f in csv_files]

    if len(symbols) > max_stocks:
        np.random.seed(42)
        symbols = list(np.random.choice(symbols, max_stocks, replace=False))

    logger.info(f"Computing factors for {len(symbols)} stocks...")

    max_workers = min(28, os.cpu_count() or 4)
    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_compute_one_symbol_factors, daily_dir, sym, ""): sym
            for sym in symbols
        }
        for future in as_completed(futures):
            sym, df = future.result()
            if not df.empty:
                results[sym] = df

    logger.info(f"Computed factors for {len(results)}/{len(symbols)} stocks")

    # 全局交易日期
    all_dates = set()
    for df in results.values():
        all_dates.update(df["trade_date"].tolist())
    trade_dates = sorted(all_dates)
    date_to_idx = {d: i for i, d in enumerate(trade_dates)}

    # 因子名
    all_factor_cols = set()
    for df in results.values():
        exclude = {"trade_date", "open", "high", "low", "close", "vol", "amount",
                    "turnover_rate", "net_mf_amount", "symbol"}
        cols = {c for c in df.columns if c not in exclude}
        all_factor_cols = all_factor_cols.intersection(cols) if all_factor_cols else cols
    factor_names = sorted(all_factor_cols)
    # Filter to only numeric columns (exclude ts_code etc.)
    sample_df = next(iter(results.values()))
    numeric_cols = set(sample_df.select_dtypes(include=[np.number]).columns)
    factor_names = [c for c in factor_names if c in numeric_cols]

    # 按日期分组
    train_dates_set = {d for d in trade_dates if int(d[:4]) in train_years}
    eval_dates_set = {d for d in trade_dates if int(d[:4]) == eval_year}

    # 每只股票预计算因子值
    stock_data = {}
    for sym, df in results.items():
        if len(df) < seq_len + forward_days:
            continue
        fcols = [c for c in factor_names if c in df.columns]
        if len(fcols) < 10:
            continue
        fvals = df[fcols].select_dtypes(include=[np.number]).values
        sym_dates = df["trade_date"].tolist()
        stock_data[sym] = (fvals, sym_dates)

    # 按日期组织截面数据
    train_sections = {}
    eval_sections = {}

    current_year = eval_year
    time_decay_lambda = 0.02

    # 预构建每只股票的 date→index 映射 + close 数组（避免 O(N×M×L) 的 list.index 调用）
    stock_lookup = {}
    for sym, (fvals, sym_dates) in stock_data.items():
        date_to_i = {d: i for i, d in enumerate(sym_dates)}
        close_arr = results[sym]["close"].values.astype(np.float64)
        stock_lookup[sym] = (fvals, sym_dates, date_to_i, close_arr)

    logger.info(f"Building cross-sections over {len(trade_dates)} dates...")

    for entry_date in trade_dates:
        future_idx = date_to_idx.get(entry_date, -1)
        if future_idx < 0:
            continue
        target_idx = future_idx + forward_days
        if target_idx >= len(trade_dates):
            continue
        target_date = trade_dates[target_idx]

        section_X = []
        section_y_reg = []
        section_y_cls = []
        section_weight = []

        year = int(entry_date[:4])
        weight = np.exp(-time_decay_lambda * (current_year - year) * 365)

        for sym, (fvals, sym_dates, date_to_i, close_arr) in stock_lookup.items():
            i = date_to_i.get(entry_date)
            if i is None or i < seq_len:
                continue

            # 找目标日期在目标股票中的位置
            target_pos = date_to_i.get(target_date)
            if target_pos is None:
                continue

            entry_price = float(close_arr[i])
            exit_price = float(close_arr[target_pos])
            if entry_price <= 0 or exit_price <= 0:
                continue
            ret = (exit_price - entry_price) / entry_price
            if abs(ret) > 0.5:
                continue

            x_seq = fvals[i - seq_len:i]
            if np.any(np.isnan(x_seq)):
                continue

            section_X.append(x_seq)
            section_y_reg.append(ret)
            section_y_cls.append(1 if ret > 0 else 0)
            section_weight.append(weight)

        if len(section_X) < 10:
            continue

        section = {
            "X": np.array(section_X, dtype=np.float32),
            "y_reg": np.array(section_y_reg, dtype=np.float32),
            "y_cls": np.array(section_y_cls, dtype=np.int64),
            "weight": np.array(section_weight, dtype=np.float32),
        }

        if entry_date in train_dates_set:
            train_sections[entry_date] = section
        elif entry_date in eval_dates_set:
            eval_sections[entry_date] = section

    if not train_sections:
        raise ValueError("No valid training sections")

    # 标准化: 用训练集全局 mean/std
    all_train_X = np.concatenate([s["X"] for s in train_sections.values()], axis=0)
    from src.strategy.adaptive_state_machine.attention_learner import _standardize_factors
    _, train_mean, train_std = _standardize_factors(all_train_X.reshape(-1, all_train_X.shape[-1]))
    logger.info(f"Cross-sectional data: {len(train_sections)} train dates, "
                f"{len(eval_sections)} eval dates")
    logger.info(f"  Avg stocks per train date: "
                f"{np.mean([s['X'].shape[0] for s in train_sections.values()]):.0f}")
    sys.stdout.flush()

    # 应用标准化
    for date in train_sections:
        sec = train_sections[date]
        sec["X"] = (sec["X"] - train_mean) / train_std
        sec["X"] = np.clip(sec["X"], -5.0, 5.0)

    for date in eval_sections:
        sec = eval_sections[date]
        sec["X"] = (sec["X"] - train_mean) / train_std
        sec["X"] = np.clip(sec["X"], -5.0, 5.0)

    return train_sections, eval_sections, factor_names, train_mean, train_std


def build_walk_forward_data(
    daily_dir: str,
    train_years: list[int],
    eval_year: int,
    max_stocks: int = 500,
    seq_len: int = 60,
    forward_days: int = 10,
) -> tuple:
    """
    Walk-Forward 数据构建: 按年份切分训练/评估集。

    Returns:
        (X_train, y_reg_train, y_cls_train, w_train, X_eval, y_reg_eval, y_cls_eval, w_eval, factor_names, trade_dates)
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    csv_files = glob.glob(os.path.join(daily_dir, "*.csv"))
    symbols = [os.path.basename(f).replace(".csv", "") for f in csv_files]

    if len(symbols) > max_stocks:
        np.random.seed(42)
        symbols = list(np.random.choice(symbols, max_stocks, replace=False))

    logger.info(f"Computing factors for {len(symbols)} stocks...")

    max_workers = min(28, os.cpu_count() or 4)
    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_compute_one_symbol_factors, daily_dir, sym, ""): sym
            for sym in symbols
        }
        for future in as_completed(futures):
            sym, df = future.result()
            if not df.empty:
                results[sym] = df

    logger.info(f"Computed factors for {len(results)}/{len(symbols)} stocks")

    # 按年份过滤
    all_dates = set()
    for df in results.values():
        all_dates.update(df["trade_date"].tolist())
    trade_dates = sorted(all_dates)
    date_to_idx = {d: i for i, d in enumerate(trade_dates)}

    train_dates_set = set()
    eval_dates_set = set()
    for d in trade_dates:
        yr = int(d[:4])
        if yr in train_years:
            train_dates_set.add(d)
        elif yr == eval_year:
            eval_dates_set.add(d)

    # 构建因子名
    all_factor_cols = set()
    for df in results.values():
        exclude = {"trade_date", "open", "high", "low", "close", "vol", "amount",
                    "turnover_rate", "net_mf_amount", "symbol"}
        cols = {c for c in df.columns if c not in exclude}
        all_factor_cols = all_factor_cols.intersection(cols) if all_factor_cols else cols
    factor_names = sorted(all_factor_cols)
    # Filter to only numeric columns (exclude ts_code etc.)
    sample_df = next(iter(results.values()))
    numeric_cols = set(sample_df.select_dtypes(include=[np.number]).columns)
    factor_names = [c for c in factor_names if c in numeric_cols]

    from src.strategy.adaptive_state_machine.attention_learner import _standardize_factors

    train_X, train_y_reg, train_y_cls, train_w = [], [], [], []
    eval_X, eval_y_reg, eval_y_cls, eval_w = [], [], [], []

    current_year = eval_year
    time_decay_lambda = 0.02

    for sym, df in results.items():
        if len(df) < seq_len + forward_days:
            continue
        factor_cols = [c for c in factor_names if c in df.columns]
        if len(factor_cols) < 10:
            continue
        factor_values = df[factor_cols].select_dtypes(include=[np.number]).values
        sym_dates = df["trade_date"].tolist()

        for i in range(seq_len, len(factor_values) - forward_days):
            x_seq = factor_values[i - seq_len:i]
            if np.any(np.isnan(x_seq)):
                continue
            entry_date = str(sym_dates[i])

            future_idx = date_to_idx.get(entry_date, -1)
            if future_idx < 0:
                continue
            target_idx = future_idx + forward_days
            if target_idx >= len(trade_dates):
                continue

            try:
                target_date = trade_dates[target_idx]
                target_pos = sym_dates.index(target_date) if target_date in sym_dates else -1
                if target_pos < 0:
                    continue
                entry_price = float(df.iloc[i]["close"])
                exit_price = float(df.iloc[target_pos]["close"])
                if entry_price <= 0 or exit_price <= 0:
                    continue
                ret = (exit_price - entry_price) / entry_price
                if abs(ret) > 0.5:
                    continue

                year = int(entry_date[:4])
                weight = np.exp(-time_decay_lambda * (current_year - year) * 365)

                if entry_date in train_dates_set:
                    train_X.append(x_seq)
                    train_y_reg.append(ret)
                    train_y_cls.append(1 if ret > 0 else 0)
                    train_w.append(weight)
                elif entry_date in eval_dates_set:
                    eval_X.append(x_seq)
                    eval_y_reg.append(ret)
                    eval_y_cls.append(1 if ret > 0 else 0)
                    eval_w.append(weight)
            except (KeyError, TypeError, ValueError):
                continue

    if not train_X:
        raise ValueError("No valid training samples")

    def _finalize(X_list, y_reg_list, y_cls_list, w_list):
        X = np.array(X_list, dtype=np.float32)
        y_reg = np.array(y_reg_list, dtype=np.float32)
        y_cls = np.array(y_cls_list, dtype=np.int64)
        w = np.array(w_list, dtype=np.float32)
        X_flat = X.reshape(-1, X.shape[-1])
        X_std, mean, std = _standardize_factors(X_flat)
        return X_std.reshape(X.shape), y_reg, y_cls, w, mean, std

    train_X, train_y_reg, train_y_cls, train_w, train_mean, train_std = _finalize(train_X, train_y_reg, train_y_cls, train_w)

    logger.info(f"Train: {len(train_X)} samples, Eval: {len(eval_X) if eval_X else 0} samples")
    logger.info(f"  Return stats: mean={train_y_reg.mean():.4f}, std={train_y_reg.std():.4f}")

    if eval_X:
        eval_X, eval_y_reg, eval_y_cls, eval_w, _, _ = _finalize(eval_X, eval_y_reg, eval_y_cls, eval_w)
        return train_X, train_y_reg, train_y_cls, train_w, eval_X, eval_y_reg, eval_y_cls, eval_w, factor_names, trade_dates, train_mean, train_std

    return train_X, train_y_reg, train_y_cls, train_w, None, None, None, None, factor_names, trade_dates, train_mean, train_std


def main():
    parser = argparse.ArgumentParser(description="Train Attention Learner (Phase 2)")
    parser.add_argument("--data-dir", required=True, help="tushare-daily-full directory")
    parser.add_argument("--model-path", required=True, help="Output model path")
    parser.add_argument("--max-stocks", type=int, default=800, help="Max stocks")
    parser.add_argument("--seq-len", type=int, default=60, help="Sequence length")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--d-model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n-heads", type=int, default=8, help="Attention heads")
    parser.add_argument("--n-layers", type=int, default=4, help="Transformer layers")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--scan-date", default="", help="Limit data to this date")

    # Walk-Forward 训练
    parser.add_argument("--walk-forward", action="store_true", help="Walk-forward mode")
    parser.add_argument("--train-years", default="", help="Training years (comma-separated, e.g. 2024,2025)")
    parser.add_argument("--eval-year", type=int, default=2026, help="Evaluation year")

    # Phase 2.1: 正则化 (defaults reverted to Phase 1 levels)
    parser.add_argument("--cross-sectional", action="store_true", help="Enable cross-sectional encoder")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate (Phase 1: 0.1)")
    parser.add_argument("--weight-decay", type=float, default=0.0001, help="Weight decay (Phase 1: 1e-4)")
    parser.add_argument("--stock-dropout", type=float, default=0.05, help="Stock dropout ratio per cross-section")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing for classification")

    args = parser.parse_args()

    from .attention_learner import AttentionTrainer, TrainConfig

    config = TrainConfig(
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_epochs=args.epochs,
        device=args.device,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        cross_sectional=args.cross_sectional,
        stock_dropout=args.stock_dropout,
        label_smoothing=args.label_smoothing,
    )

    if args.walk_forward:
        # Walk-Forward 模式
        train_years = [int(y) for y in args.train_years.split(",")] if args.train_years else [2024, 2025]
        logger.info(f"Walk-Forward mode: train={train_years}, eval={args.eval_year}")

        if args.cross_sectional:
            # 截面模式
            logger.info("Cross-Sectional mode: enabled")
            data = build_cross_sectional_data(
                args.data_dir,
                train_years=train_years,
                eval_year=args.eval_year,
                max_stocks=args.max_stocks,
                seq_len=args.seq_len,
            )
            train_sections, eval_sections, factor_names, train_mean, train_std = data

            config.factor_names = factor_names

            trainer = AttentionTrainer(config)
            model = trainer.train_cross_sectional(
                train_sections, eval_sections,
                save_path=args.model_path,
                standardize_mean=train_mean, standardize_std=train_std,
            )

            # 评估
            if eval_sections:
                logger.info("Evaluating on walk-forward holdout...")
                trainer.model.eval_mode()
                all_pred_reg = []
                all_pred_cls = []
                all_y_reg = []
                all_y_cls = []
                for date in sorted(eval_sections.keys()):
                    sec = eval_sections[date]
                    X_e = sec["X"]
                    import torch as _eval_torch
                    with _eval_torch.no_grad():
                        # Phase 2.1 fix: use cross-sectional forward path when model has CS encoder
                        has_cs = hasattr(trainer.model, 'cross_sectional_transformer')
                        if has_cs:
                            out = trainer.model.forward_cross_sectional(X_e, training=False)
                        else:
                            out = trainer.model.forward(X_e, training=False)
                    all_pred_reg.append(out["regression"])
                    all_pred_cls.append(out["classification"])
                    all_y_reg.append(sec["y_reg"])
                    all_y_cls.append(sec["y_cls"])
                pred_reg = np.concatenate(all_pred_reg)
                pred_cls = np.concatenate(all_pred_cls)
                y_reg_eval = np.concatenate(all_y_reg)
                y_cls_eval = np.concatenate(all_y_cls)
                corr = np.corrcoef(pred_reg, y_reg_eval)[0, 1]
                acc = (np.argmax(pred_cls, axis=1) == y_cls_eval).mean()
                logger.info(f"  Eval IC (Pearson): {corr:.4f}")
                logger.info(f"  Eval direction accuracy: {acc:.1%}")
        else:
            # 非截面模式 (Phase 1 兼容)
            data = build_walk_forward_data(
                args.data_dir,
                train_years=train_years,
                eval_year=args.eval_year,
                max_stocks=args.max_stocks,
                seq_len=args.seq_len,
            )

            X_train, y_reg_train, y_cls_train, w_train = data[0], data[1], data[2], data[3]
            X_eval, y_reg_eval, y_cls_eval, w_eval = data[4], data[5], data[6], data[7]
            factor_names = data[8]
            train_mean, train_std = data[10], data[11]

            config.factor_names = factor_names

            trainer = AttentionTrainer(config)
            model = trainer.train(
                X_train, y_reg_train, y_cls_train, w_train,
                save_path=args.model_path,
                standardize_mean=train_mean, standardize_std=train_std,
            )

            # 评估 (分 batch 避免 OOM)
            if X_eval is not None and len(X_eval) > 0:
                logger.info("Evaluating on walk-forward holdout...")
                trainer.model.eval_mode()
                all_pred_reg = []
                all_pred_cls = []
                eval_batch_size = 512
                import torch as _eval_torch
                with _eval_torch.no_grad():
                    for bs in range(0, len(X_eval), eval_batch_size):
                        be = min(bs + eval_batch_size, len(X_eval))
                        out = trainer.model.forward(X_eval[bs:be], training=False)
                        all_pred_reg.append(out["regression"])
                        all_pred_cls.append(out["classification"])
                pred_reg = np.concatenate(all_pred_reg)
                pred_cls = np.concatenate(all_pred_cls)
                corr = np.corrcoef(pred_reg, y_reg_eval)[0, 1]
                acc = (np.argmax(pred_cls, axis=1) == y_cls_eval).mean()
                logger.info(f"  Eval IC (Pearson): {corr:.4f}")
                logger.info(f"  Eval direction accuracy: {acc:.1%}")

    else:
        # 全量训练模式
        logger.info("Full training mode...")
        start = time.time()
        X, y_reg, y_cls, sample_weights, factor_names, trade_dates, train_mean, train_std = build_training_data(
            args.data_dir,
            max_stocks=args.max_stocks,
            scan_date=args.scan_date,
            seq_len=args.seq_len,
        )
        logger.info(f"Data built in {time.time() - start:.1f}s")

        config.factor_names = factor_names

        trainer = AttentionTrainer(config)
        model = trainer.train(
            X, y_reg, y_cls, sample_weights,
            save_path=args.model_path,
            standardize_mean=train_mean, standardize_std=train_std,
        )

    # Summary
    logger.info("Training summary:")
    for h in trainer.train_history[-5:]:
        logger.info(f"  Epoch {h['epoch']}: train={h['train_loss']:.6f}, "
                     f"val={h['val_loss']:.6f}, val_ic={h.get('val_ic', 0):.4f}")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    # Sanity check: extract factor weights
    logger.info("Sanity check: extracting factor weights...")
    from .attention_learner import AttentionLearner

    test_X = X if 'X' in dir() else None
    if test_X is None and 'X_train' in dir():
        test_X = X_train
    if test_X is None:
        # 截面模式: 从 train_sections 取一个截面
        test_X = list(train_sections.values())[0]["X"][:50]

    learner = AttentionLearner(
        model_path=args.model_path,
        seq_len=args.seq_len,
    )
    weights = learner.extract_weights(test_X[:50])
    top5 = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]
    logger.info("Top 5 factors by attention weight:")
    for f, w in top5:
        logger.info(f"  {f}: {w:.4f}")


if __name__ == "__main__":
    main()
