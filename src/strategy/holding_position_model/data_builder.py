"""
Holding Position Model — 训练数据构造 (memmap 版本)

核心逻辑：模拟持仓轨迹 + 用未来走势打标签

防 OOM 策略:
  - 先统计样本总数 → 预分配 memmap 文件
  - 两遍遍历: 第一遍统计, 第二遍写数据
  - 内存峰值 < 5GB (无论样本量多少)
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HOLDING_FEATURE_NAMES = [
    "days_since_entry", "unrealized_pnl", "max_pnl_since_entry",
    "drawdown_from_peak", "entry_price_position",
]
HOLDING_DIM = len(HOLDING_FEATURE_NAMES)


def _compute_one_symbol_factors(
    daily_dir: str,
    symbol: str,
    scan_date: str = "",
    data_root: str = "",
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
        result = build_features(df_daily=df, symbol=symbol, data_root=data_root)
        df_featured = result.get("daily")
        df_weekly = result.get("weekly")

        if df_featured is None or len(df_featured) < 80:
            return symbol, pd.DataFrame()

        weekly_cols = ["pe_ttm_pctl", "pb_pctl", "weekly_big_net_cumsum",
                       "weekly_turnover_shrink", "weekly_turnover_ma4"]
        for col in weekly_cols:
            k = f"w_{col}"
            if df_weekly is not None and not df_weekly.empty and col in df_weekly.columns:
                val = df_weekly.iloc[-1].get(col)
                df_featured[k] = float(val) if pd.notna(val) else np.nan
            else:
                df_featured[k] = np.nan

        return symbol, df_featured
    except Exception as e:
        logger.debug(f"Error computing {symbol}: {e}")
        return symbol, pd.DataFrame()


def _build_holding_features(
    entry_price: float, current_price: float, peak_price: float,
    days_held: int, entry_price_position: float,
) -> np.ndarray:
    return np.array([
        days_held / 20.0,
        (current_price - entry_price) / max(abs(entry_price), 1e-8),
        (peak_price - entry_price) / max(abs(entry_price), 1e-8),
        (current_price - peak_price) / max(abs(peak_price), 1e-8),
        entry_price_position,
    ], dtype=np.float32)


def _assign_labels(
    future_closes: np.ndarray, current_price: float, peak_price: float,
    max_hold_days: int = 20,
    forward_stay: int = 10, forward_collapse: int = 5,
    stay_drawdown_thresh: float = 0.05,
    collapse_drop_thresh: float = 0.05,
    collapse_drawdown_thresh: float = 0.10,
) -> tuple[int, int, float]:
    future_n = len(future_closes)
    if future_n == 0:
        return 0, 0, float(max_hold_days)

    stay_window = future_closes[:min(forward_stay, future_n)]
    stay_return = (stay_window[-1] - current_price) / max(abs(current_price), 1e-8)
    stay_drawdown = (stay_window.min() - current_price) / max(abs(current_price), 1e-8)
    stay_label = int(stay_return > 0 and abs(stay_drawdown) < stay_drawdown_thresh)

    collapse_window = future_closes[:min(forward_collapse, future_n)]
    collapse_drop = (collapse_window.min() - current_price) / max(abs(current_price), 1e-8)
    current_drawdown = (current_price - peak_price) / max(abs(peak_price), 1e-8)
    collapse_label = int(
        collapse_drop < -collapse_drop_thresh or current_drawdown < -collapse_drawdown_thresh
    )

    days_label = float(max_hold_days)
    for d in range(min(forward_collapse, future_n)):
        fut_drop = (future_closes[d] - current_price) / max(abs(current_price), 1e-8)
        if fut_drop < -collapse_drop_thresh:
            days_label = float(d + 1)
            break

    return stay_label, collapse_label, days_label


def build_training_data(
    daily_dir: str,
    data_root: str = "",
    max_stocks: int = 500,
    scan_date: str = "",
    seq_len: int = 60,
    max_hold_days: int = 20,
    entry_sample_ratio: float = 0.3,
    train_ratio: float = 0.8,
    output_dir: str = "",
) -> dict:
    """
    构建持仓模型训练数据 (memmap 版本)。

    流程:
      1. 计算因子 (CPU 并行)
      2. 第一遍: 统计 train/eval 样本数
      3. 预分配 memmap 文件
      4. 第二遍: 写入数据到 memmap (不占内存)
      5. 标准化: 在训练集 memmap 上计算 mean/std 并原地写入
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # ── Step 1: 计算因子 ──
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
            pool.submit(_compute_one_symbol_factors, daily_dir, sym, scan_date, data_root): sym
            for sym in symbols
        }
        for future in as_completed(futures):
            sym, df = future.result()
            if not df.empty:
                results[sym] = df

    logger.info(f"Computed factors for {len(results)}/{len(symbols)} stocks")
    if not results:
        raise ValueError("No valid factor data")

    # ── Step 2: 提取统一因子名（交集 + 去冗余） ──
    exclude = {"trade_date", "open", "high", "low", "close", "vol", "amount",
               "turnover_rate", "net_mf_amount", "symbol"}
    common_cols = None
    for df in results.values():
        numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
        cols = {c for c in numeric_cols if c not in exclude}
        common_cols = cols if common_cols is None else common_cols & cols

    if not common_cols:
        raise ValueError("No common factor columns across all stocks")

    from src.strategy.adaptive_state_machine.train_attention import _dedup_factors
    factor_names = _dedup_factors(sorted(common_cols))
    n_factors = len(factor_names)
    logger.info(f"Factor columns: {n_factors} (after dedup)")

    # ── Step 3: 全局交易日期 + 时间 split ──
    all_dates = set()
    for df in results.values():
        all_dates.update(df["trade_date"].tolist())
    global_dates = sorted(all_dates)
    split_idx = int(len(global_dates) * train_ratio)
    train_dates = set(global_dates[:split_idx])
    eval_dates = set(global_dates[split_idx:])
    logger.info(f"Time split: {len(train_dates)} train / {len(eval_dates)} eval dates")

    current_year = 2026
    time_decay_lambda = 0.02

    # ── Step 4: 第一遍 — 统计样本数 ──
    np.random.seed(42)
    n_train = 0
    n_eval = 0

    for sym, df in results.items():
        if len(df) < seq_len + max_hold_days + max(max_hold_days, 10):
            continue

        n_rows = len(df)
        min_entry = seq_len
        max_entry = n_rows - max_hold_days - 10
        if max_entry <= min_entry:
            continue

        all_entries = list(range(min_entry, max_entry + 1))
        n_sample = max(1, int(len(all_entries) * entry_sample_ratio))
        sampled_entries = np.random.choice(all_entries, n_sample, replace=False)

        for entry_idx in sampled_entries:
            entry_date = df.iloc[entry_idx]["trade_date"]
            is_train = entry_date in train_dates

            for hold_day in range(1, max_hold_days + 1):
                if entry_idx + hold_day >= n_rows:
                    break
                if is_train:
                    n_train += 1
                else:
                    n_eval += 1

    logger.info(f"Sample counts: {n_train} train / {n_eval} eval")

    if n_train == 0:
        raise ValueError("No valid training samples")

    # ── Step 5: 预分配 memmap 文件 ──
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.path.join(os.path.dirname(__file__), "data_cache")
        os.makedirs(output_dir, exist_ok=True)

    memmaps = {}
    mmap_files = {}

    def _create_memmap(name, shape, dtype):
        path = os.path.join(output_dir, f"{name}.mmap")
        mmap = np.memmap(path, dtype=dtype, mode="w+", shape=shape)
        memmaps[name] = mmap
        mmap_files[name] = path
        return mmap

    # 预分配
    m_factors_train = _create_memmap("X_factors_train", (n_train, seq_len, n_factors), np.float32)
    m_holding_train = _create_memmap("X_holding_train", (n_train, HOLDING_DIM), np.float32)
    m_stay_train = _create_memmap("y_stay_train", (n_train,), np.float32)
    m_collapse_train = _create_memmap("y_collapse_train", (n_train,), np.float32)
    m_days_train = _create_memmap("y_days_train", (n_train,), np.float32)
    m_weights_train = _create_memmap("sample_weights_train", (n_train,), np.float32)

    m_factors_eval = _create_memmap("X_factors_eval", (n_eval, seq_len, n_factors), np.float32)
    m_holding_eval = _create_memmap("X_holding_eval", (n_eval, HOLDING_DIM), np.float32)
    m_stay_eval = _create_memmap("y_stay_eval", (n_eval,), np.float32)
    m_collapse_eval = _create_memmap("y_collapse_eval", (n_eval,), np.float32)
    m_days_eval = _create_memmap("y_days_eval", (n_eval,), np.float32)
    m_weights_eval = _create_memmap("sample_weights_eval", (n_eval,), np.float32)

    # ── Step 6: 第二遍 — 写入数据到 memmap ──
    np.random.seed(42)
    i_train = 0
    i_eval = 0
    stock_count = 0
    FLUSH_INTERVAL = 100  # flush every N stocks to prevent dirty page accumulation

    for sym, df in results.items():
        n_rows = len(df)
        if n_rows < seq_len + max_hold_days + max(max_hold_days, 10):
            continue

        # 对齐因子列
        factor_values = np.full((n_rows, n_factors), np.nan, dtype=np.float32)
        for j, col in enumerate(factor_names):
            if col in df.columns and df[col].dtype in (np.float32, np.float64, np.int32, np.int64):
                factor_values[:, j] = df[col].values.astype(np.float32)

        if (~np.isnan(factor_values)).sum(axis=1).mean() < n_factors * 0.8:
            continue

        sym_dates = df["trade_date"].tolist()
        close_prices = df["close"].values.astype(np.float64)
        open_prices = df["open"].values.astype(np.float64)

        min_entry = seq_len
        max_entry = n_rows - max_hold_days - 10
        if max_entry <= min_entry:
            continue

        all_entries = list(range(min_entry, max_entry + 1))
        n_sample = max(1, int(len(all_entries) * entry_sample_ratio))
        sampled_entries = np.random.choice(all_entries, n_sample, replace=False)

        for entry_idx in sampled_entries:
            entry_price = open_prices[entry_idx]
            if entry_price <= 0:
                continue

            factor_window = factor_values[entry_idx - seq_len: entry_idx]
            if np.any(np.isnan(factor_window)):
                continue

            recent_closes = close_prices[entry_idx - seq_len: entry_idx]
            entry_pos = float((recent_closes < entry_price).sum() + 1) / seq_len

            for hold_day in range(1, max_hold_days + 1):
                current_idx = entry_idx + hold_day
                if current_idx >= n_rows:
                    break

                current_price = close_prices[current_idx]
                if current_price <= 0:
                    continue

                price_path = close_prices[entry_idx: current_idx + 1]
                peak_price = price_path.max()
                holding_feat = _build_holding_features(
                    entry_price, current_price, peak_price, hold_day, entry_pos,
                )
                future_end = min(current_idx + 1 + max_hold_days, n_rows)
                future_closes = close_prices[current_idx + 1: future_end]
                stay_label, collapse_label, days_label = _assign_labels(
                    future_closes, current_price, peak_price, max_hold_days=max_hold_days,
                )
                entry_date = sym_dates[entry_idx]
                year = int(entry_date[:4]) if len(entry_date) >= 4 else current_year
                weight = np.exp(-time_decay_lambda * (current_year - year) * 365)

                is_train = entry_date in train_dates

                if is_train:
                    idx = i_train
                    m_factors_train[idx] = factor_window
                    m_holding_train[idx] = holding_feat
                    m_stay_train[idx] = stay_label
                    m_collapse_train[idx] = collapse_label
                    m_days_train[idx] = days_label
                    m_weights_train[idx] = weight
                    i_train += 1
                else:
                    idx = i_eval
                    m_factors_eval[idx] = factor_window
                    m_holding_eval[idx] = holding_feat
                    m_stay_eval[idx] = stay_label
                    m_collapse_eval[idx] = collapse_label
                    m_days_eval[idx] = days_label
                    m_weights_eval[idx] = weight
                    i_eval += 1

        # 定期释放单只股票内存
        del factor_values
        stock_count += 1

        # 定期 flush memmap 防止 dirty page 积累
        if stock_count % FLUSH_INTERVAL == 0:
            m_factors_train.flush()
            m_holding_train.flush()
            m_stay_train.flush()
            m_collapse_train.flush()
            m_days_train.flush()
            m_weights_train.flush()
            if n_eval > 0:
                m_factors_eval.flush()
                m_holding_eval.flush()
                m_stay_eval.flush()
                m_collapse_eval.flush()
                m_days_eval.flush()
                m_weights_eval.flush()
            logger.info(f"  Progress: {stock_count}/{len(results)} stocks, "
                        f"{i_train} train / {i_eval} eval samples written")

    # Final flush
    m_factors_train.flush()
    m_holding_train.flush()
    m_stay_train.flush()
    m_collapse_train.flush()
    m_days_train.flush()
    m_weights_train.flush()
    if n_eval > 0:
        m_factors_eval.flush()
        m_holding_eval.flush()
        m_stay_eval.flush()
        m_collapse_eval.flush()
        m_days_eval.flush()
        m_weights_eval.flush()

    # 裁剪到实际写入的样本数 (可能有少量被过滤)
    n_train = i_train
    n_eval = i_eval
    logger.info(f"Actual written: {n_train} train / {n_eval} eval samples")

    def _resize(name, new_n):
        old = memmaps[name]
        old.flush()
        del memmaps[name]
        path = mmap_files[name]
        new_shape = (new_n,) + old.shape[1:]
        expected_size = int(np.prod(new_shape) * np.dtype(old.dtype).itemsize)
        # Truncate file to actual data size
        with open(path, "r+b") as f:
            f.truncate(expected_size)
        new_mmap = np.memmap(path, dtype=old.dtype, mode="r+", shape=new_shape)
        memmaps[name] = new_mmap

    if n_train < m_factors_train.shape[0]:
        _resize("X_factors_train", n_train)
        _resize("X_holding_train", n_train)
        _resize("y_stay_train", n_train)
        _resize("y_collapse_train", n_train)
        _resize("y_days_train", n_train)
        _resize("sample_weights_train", n_train)

    if n_eval < m_factors_eval.shape[0]:
        _resize("X_factors_eval", n_eval)
        _resize("X_holding_eval", n_eval)
        _resize("y_stay_eval", n_eval)
        _resize("y_collapse_eval", n_eval)
        _resize("y_days_eval", n_eval)
        _resize("sample_weights_eval", n_eval)

    # ── Step 7: 标准化 (在训练集 memmap 上) ──
    # 计算 mean/std: 分块读取避免 OOM
    chunk_size = 100000
    sum_vals = np.zeros(n_factors, dtype=np.float64)
    sum_sq = np.zeros(n_factors, dtype=np.float64)
    count = 0

    for start in range(0, n_train, chunk_size):
        end = min(start + chunk_size, n_train)
        chunk = memmaps["X_factors_train"][start:end]  # (chunk, seq_len, n_factors)
        chunk_flat = chunk.reshape(-1, n_factors)
        sum_vals += np.nansum(chunk_flat, axis=0)
        sum_sq += np.nansum(chunk_flat ** 2, axis=0)
        count += chunk_flat.shape[0]

    mean = sum_vals / count
    var = sum_sq / count - mean ** 2
    var = np.maximum(var, 0.0)  # clamp negative values from floating point errors
    std = np.sqrt(var)
    std[std < 1e-8] = 1.0

    # 原地标准化
    for start in range(0, n_train, chunk_size):
        end = min(start + chunk_size, n_train)
        chunk = memmaps["X_factors_train"][start:end]
        chunk = ((chunk - mean) / std).astype(np.float32)
        chunk = np.clip(chunk, -5.0, 5.0)
        memmaps["X_factors_train"][start:end] = chunk
        memmaps["X_factors_train"].flush()

    # eval 集也标准化
    if n_eval > 0:
        for start in range(0, n_eval, chunk_size):
            end = min(start + chunk_size, n_eval)
            chunk = memmaps["X_factors_eval"][start:end]
            chunk = ((chunk - mean) / std).astype(np.float32)
            chunk = np.clip(chunk, -5.0, 5.0)
            memmaps["X_factors_eval"][start:end] = chunk
            memmaps["X_factors_eval"].flush()

    logger.info(f"Standardization done. Train: {n_train}, Eval: {n_eval}")
    logger.info(f"  Factors: {n_factors}, seq_len: {seq_len}")
    logger.info(f"  Stay ratio: {memmaps['y_stay_train'][:n_train].mean():.1%}, "
                f"Collapse ratio: {memmaps['y_collapse_train'][:n_train].mean():.1%}")

    return {
        "X_factors_train": memmaps["X_factors_train"][:n_train],
        "X_factors_eval": memmaps["X_factors_eval"][:n_eval],
        "X_holding_train": memmaps["X_holding_train"][:n_train],
        "X_holding_eval": memmaps["X_holding_eval"][:n_eval],
        "y_stay_train": memmaps["y_stay_train"][:n_train],
        "y_stay_eval": memmaps["y_stay_eval"][:n_eval],
        "y_collapse_train": memmaps["y_collapse_train"][:n_train],
        "y_collapse_eval": memmaps["y_collapse_eval"][:n_eval],
        "y_days_train": memmaps["y_days_train"][:n_train],
        "y_days_eval": memmaps["y_days_eval"][:n_eval],
        "sample_weights_train": memmaps["sample_weights_train"][:n_train],
        "sample_weights_eval": memmaps["sample_weights_eval"][:n_eval],
        "factor_names": factor_names,
        "standardize_mean": mean,
        "standardize_std": std,
        "memmap_dir": output_dir,
        "memmap_files": mmap_files,
        "n_train": n_train,
        "n_eval": n_eval,
        "n_factors": n_factors,
    }
