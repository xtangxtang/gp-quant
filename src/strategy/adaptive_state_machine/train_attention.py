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




def _neutralize_market_cap(
    X: np.ndarray,
    factor_names: list[str],
    market_cap_col: str = "total_mv",
) -> np.ndarray:
    """
    截面市值中性化: 每个时间步, 将每个因子对 log(total_mv) 做截面回归, 取残差。

    Args:
        X: (n_samples, seq_len, n_factors) 标准化前的原始因子值
        factor_names: 因子名列表
        market_cap_col: 市值因子名 (必须在 factor_names 中)

    Returns:
        X_neutralized: 市值中性化后的因子值 (原始形状)
    """
    if market_cap_col not in factor_names:
        logger.warning(f"Market cap factor '{market_cap_col}' not found, skipping neutralization")
        return X

    mc_idx = factor_names.index(market_cap_col)
    X_copy = X.copy()
    n_samples, seq_len, n_factors = X.shape

    # 对每个样本的最后一个时间步 (当前截面) 做中性化
    for i in range(n_samples):
        mc_values = X_copy[i, :, mc_idx]  # (seq_len,) 市值时序
        if np.any(np.isnan(mc_values)):
            continue

        for j in range(n_factors):
            if j == mc_idx:
                continue
            factor_ts = X_copy[i, :, j]
            if np.any(np.isnan(factor_ts)):
                continue

            # 对整个时序做中性化 (回归 log(mc) → factor)
            log_mc = np.log(np.abs(mc_values) + 1e-6)
            # 简单 OLS: factor = alpha + beta * log_mc + residual
            # 用 numpy 做逐时间步的截面中性化
            mask = ~(np.isnan(factor_ts) | np.isnan(log_mc))
            if mask.sum() < 5:
                continue

            X_mc = np.column_stack([np.ones(mask.sum()), log_mc[mask]])
            y = factor_ts[mask]
            try:
                beta = np.linalg.lstsq(X_mc, y, rcond=None)[0]
                residual = y - X_mc @ beta
                factor_ts[mask] = residual
            except np.linalg.LinAlgError:
                pass

    return X_copy



def build_walk_forward_data(
    daily_dir: str,
    train_years: list[int],
    eval_year: int,
    max_stocks: int = 500,
    seq_len: int = 60,
    forward_days: int = 10,
    neutralize_col: str = "",
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

    def _finalize(X_list, y_reg_list, y_cls_list, w_list, factor_names=None, neutralize_col=None):
        X = np.array(X_list, dtype=np.float32)
        y_reg = np.array(y_reg_list, dtype=np.float32)
        y_cls = np.array(y_cls_list, dtype=np.int64)
        w = np.array(w_list, dtype=np.float32)

        # Market cap neutralization (before standardization)
        if neutralize_col and factor_names and neutralize_col in factor_names:
            logger.info(f"  Neutralizing against '{neutralize_col}' ...")
            X = _neutralize_market_cap(X, factor_names, neutralize_col)
            logger.info(f"  Neutralization done")

        X_flat = X.reshape(-1, X.shape[-1])
        X_std, mean, std = _standardize_factors(X_flat)
        return X_std.reshape(X.shape), y_reg, y_cls, w, mean, std

    train_X, train_y_reg, train_y_cls, train_w, train_mean, train_std = _finalize(
        train_X, train_y_reg, train_y_cls, train_w,
        factor_names=factor_names, neutralize_col=neutralize_col,
    )

    logger.info(f"Train: {len(train_X)} samples, Eval: {len(eval_X) if eval_X else 0} samples")
    logger.info(f"  Return stats: mean={train_y_reg.mean():.4f}, std={train_y_reg.std():.4f}")

    if eval_X:
        eval_X, eval_y_reg, eval_y_cls, eval_w, _, _ = _finalize(eval_X, eval_y_reg, eval_y_cls, eval_w)
        return train_X, train_y_reg, train_y_cls, train_w, eval_X, eval_y_reg, eval_y_cls, eval_w, factor_names, trade_dates, train_mean, train_std

    return train_X, train_y_reg, train_y_cls, train_w, None, None, None, None, factor_names, trade_dates, train_mean, train_std




def _build_daily_to_weekly_sequences_multi(
    daily_dir: str,
    train_years: list[int],
    eval_year: int,
    max_stocks: int = 500,
    daily_seq_len: int = 60,
    weekly_seq_len: int = 12,
    forward_days: int = 10,
    neutralize_col: str = "",
) -> tuple:
    """
    Multi-scale 数据构建: 日线 + 周线双通道。

    周线聚合: 每 5 个交易日取均值作为一周。
    日线 60 天 + 周线 12 周 → 双通道输入。

    Returns:
        (X_daily_train, X_weekly_train, y_reg_train, y_cls_train, w_train,
         X_daily_eval, X_weekly_eval, y_reg_eval, y_cls_eval, w_eval,
         None, None, None, factor_names, d_mean, d_std, w_mean, w_std)
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    csv_files = glob.glob(os.path.join(daily_dir, "*.csv"))
    symbols = [os.path.basename(f).replace(".csv", "") for f in csv_files]

    if len(symbols) > max_stocks:
        np.random.seed(42)
        symbols = list(np.random.choice(symbols, max_stocks, replace=False))

    logger.info(f"Computing factors for {len(symbols)} stocks (multi-scale)...")

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
    sample_df = next(iter(results.values()))
    numeric_cols = set(sample_df.select_dtypes(include=[np.number]).columns)
    factor_names = [c for c in factor_names if c in numeric_cols]

    train_years_set = set(train_years)
    current_year = eval_year
    time_decay_lambda = 0.02

    train_daily_X, train_weekly_X, train_y_reg, train_y_cls, train_w = [], [], [], [], []
    eval_daily_X, eval_weekly_X, eval_y_reg, eval_y_cls, eval_w = [], [], [], [], []

    # 预计算查找表和 close 数组
    stock_lookup = {}
    for sym, df in results.items():
        if len(df) < daily_seq_len + forward_days:
            continue
        fvals = df[factor_names].select_dtypes(include=[np.number]).values
        sym_dates = df["trade_date"].tolist()
        if len(fvals) < daily_seq_len + forward_days:
            continue
        date_to_i = {d: i for i, d in enumerate(sym_dates)}
        close_arr = df["close"].values.astype(np.float64)
        stock_lookup[sym] = (fvals, sym_dates, date_to_i, close_arr)

    logger.info(f"Building multi-scale sequences over {len(trade_dates)} dates...")

    for entry_date in trade_dates:
        yr = int(entry_date[:4])
        if yr not in train_years_set and yr != eval_year:
            continue

        future_idx = date_to_idx.get(entry_date, -1)
        if future_idx < 0:
            continue
        target_idx = future_idx + forward_days
        if target_idx >= len(trade_dates):
            continue
        target_date = trade_dates[target_idx]

        weight = np.exp(-time_decay_lambda * (current_year - yr) * 365)

        daily_seqs, weekly_seqs, y_regs, y_clss = [], [], [], []

        for sym, (fvals, sym_dates, date_to_i, close_arr) in stock_lookup.items():
            i = date_to_i.get(entry_date)
            if i is None or i < daily_seq_len:
                continue
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

            x_daily = fvals[i - daily_seq_len:i]
            if np.any(np.isnan(x_daily)):
                continue

            # 周线聚合: 从日线序列中每 5 天取均值
            x_weekly = []
            valid = True
            for w in range(weekly_seq_len):
                start = i - (weekly_seq_len - w) * 5
                end = i - (weekly_seq_len - w - 1) * 5
                if start < 0 or end > i:
                    valid = False
                    break
                week_slice = fvals[start:end]
                if np.any(np.isnan(week_slice)):
                    valid = False
                    break
                x_weekly.append(np.nanmean(week_slice, axis=0))

            if not valid:
                continue

            daily_seqs.append(x_daily)
            weekly_seqs.append(np.array(x_weekly, dtype=np.float32))
            y_regs.append(ret)
            y_clss.append(1 if ret > 0 else 0)

        if len(daily_seqs) < 10:
            continue

        if yr in train_years_set:
            train_daily_X.extend(daily_seqs)
            train_weekly_X.extend(weekly_seqs)
            train_y_reg.extend(y_regs)
            train_y_cls.extend(y_clss)
            train_w.extend([weight] * len(daily_seqs))
        elif yr == eval_year:
            eval_daily_X.extend(daily_seqs)
            eval_weekly_X.extend(weekly_seqs)
            eval_y_reg.extend(y_regs)
            eval_y_cls.extend(y_clss)
            eval_w.extend([weight] * len(daily_seqs))

    def _finalize_ms(Xd_list, Xw_list, y_reg_list, y_cls_list, w_list):
        X_daily = np.array(Xd_list, dtype=np.float32)
        X_weekly = np.array(Xw_list, dtype=np.float32)
        y_reg = np.array(y_reg_list, dtype=np.float32)
        y_cls = np.array(y_cls_list, dtype=np.int64)
        w = np.array(w_list, dtype=np.float32)

        from src.strategy.adaptive_state_machine.attention_learner import _standardize_factors
        Xd_flat = X_daily.reshape(-1, X_daily.shape[-1])
        Xd_std, d_mean, d_std = _standardize_factors(Xd_flat)
        X_daily = Xd_std.reshape(X_daily.shape)

        Xw_flat = X_weekly.reshape(-1, X_weekly.shape[-1])
        Xw_std, w_mean, w_std = _standardize_factors(Xw_flat)
        X_weekly = Xw_std.reshape(X_weekly.shape)

        return X_daily, X_weekly, y_reg, y_cls, w, d_mean, d_std, w_mean, w_std

    if train_daily_X:
        train_res = _finalize_ms(train_daily_X, train_weekly_X, train_y_reg, train_y_cls, train_w)
    else:
        train_res = (None,) * 9

    if eval_daily_X:
        eval_res = _finalize_ms(eval_daily_X, eval_weekly_X, eval_y_reg, eval_y_cls, eval_w)
    else:
        eval_res = (None,) * 9

    return (*train_res, *eval_res, factor_names)


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
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate (Phase 1: 0.1)")
    parser.add_argument("--weight-decay", type=float, default=0.0001, help="Weight decay (Phase 1: 1e-4)")
    parser.add_argument("--neutralize", type=str, default="", help="Market cap neutralization column (e.g. total_mv)")
    parser.add_argument("--multi-scale", action="store_true", help="Use multi-scale (daily+weekly) model")

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
    )

    if args.walk_forward:
        train_years = [int(y) for y in args.train_years.split(",")] if args.train_years else [2024, 2025]
        logger.info(f"Walk-Forward mode: train={train_years}, eval={args.eval_year}")

        # 固定随机种子，确保可复现
        import torch as _seed_torch
        _seed_torch.manual_seed(42)
        np.random.seed(42)

        neutralize_col = args.neutralize if args.neutralize else ""

        if args.multi_scale:
            # Multi-scale mode
            logger.info("Multi-scale mode: daily + weekly")
            data = _build_daily_to_weekly_sequences_multi(
                args.data_dir,
                train_years=train_years,
                eval_year=args.eval_year,
                max_stocks=args.max_stocks,
                daily_seq_len=args.seq_len,
                weekly_seq_len=12,
                neutralize_col=neutralize_col,
            )

            X_daily_train, X_weekly_train, y_reg_train, y_cls_train, w_train = data[0], data[1], data[2], data[3], data[4]
            X_daily_eval, X_weekly_eval, y_reg_eval, y_cls_eval, w_eval = data[9], data[10], data[11], data[12], data[13]
            factor_names = data[18]
            train_mean, train_std = data[14], data[15]
            weekly_mean, weekly_std = data[16], data[17]

            from src.strategy.adaptive_state_machine.attention_learner import (
                MultiScaleAttentionModel,
            )

            config.factor_names = factor_names

            trainer = AttentionTrainer(config)
            model = trainer.train_multi_scale(
                X_daily_train, X_weekly_train, y_reg_train, y_cls_train, w_train,
                save_path=args.model_path,
                standardize_mean=train_mean, standardize_std=train_std,
                weekly_mean=weekly_mean, weekly_std=weekly_std,
            )

            # Eval
            if X_daily_eval is not None and len(X_daily_eval) > 0:
                logger.info("Evaluating multi-scale model on holdout...")
                trainer.model.eval_mode()
                all_pred_reg, all_pred_cls = [], []
                eval_batch_size = 512
                import torch as _eval_torch
                with _eval_torch.no_grad():
                    for bs in range(0, len(X_daily_eval), eval_batch_size):
                        be = min(bs + eval_batch_size, len(X_daily_eval))
                        out = trainer.model.forward(
                            X_daily_eval[bs:be], X_weekly_eval[bs:be], training=False
                        )
                        all_pred_reg.append(out["regression"])
                        all_pred_cls.append(out["classification"])
                pred_reg = np.concatenate(all_pred_reg).flatten()
                pred_cls = np.concatenate(all_pred_cls)
                from scipy.stats import spearmanr
                pearson_ic = np.corrcoef(pred_reg, y_reg_eval)[0, 1]
                spearman_ic, _ = spearmanr(pred_reg, y_reg_eval)
                acc = (np.argmax(pred_cls, axis=1) == y_cls_eval).mean()
                logger.info(f"  Eval IC (Pearson): {pearson_ic:.4f}")
                logger.info(f"  Eval IC (Spearman): {spearman_ic:.4f}")
                logger.info(f"  Eval direction accuracy: {acc:.1%}")
            return  # Multi-scale done
        else:
            # Standard single-scale
            data = build_walk_forward_data(
                args.data_dir,
                train_years=train_years,
                eval_year=args.eval_year,
                max_stocks=args.max_stocks,
                seq_len=args.seq_len,
                neutralize_col=neutralize_col,
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

        # 评估
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
            from scipy.stats import spearmanr
            pearson_ic = np.corrcoef(pred_reg, y_reg_eval)[0, 1]
            spearman_ic, _ = spearmanr(pred_reg, y_reg_eval)
            acc = (np.argmax(pred_cls, axis=1) == y_cls_eval).mean()
            logger.info(f"  Eval IC (Pearson): {pearson_ic:.4f}")
            logger.info(f"  Eval IC (Spearman): {spearman_ic:.4f}")
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
