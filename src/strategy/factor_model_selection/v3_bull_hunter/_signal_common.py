"""
Bull Hunter v6 — Shared helpers for Agent 6 (exit signal) / Agent 8 (buy signal).

提取 agent6/agent8 之间的重复代码:
  - 历史交易后 N 日收益计算
  - 价格 / 成交量序列加载
  - 规则权重持久化 (按 filename 区分)
  - 自动权重调整 (相关性自适应, 通过 sign 区分买/卖方向)
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MIN_WEIGHT = 0.01
_MAX_WEIGHT = 0.25
_ADJUST_LR = 0.3
_MIN_ADJUST_SAMPLES = 10


# ─── 价格 / 收益 ───

def get_post_trade_return(
    data_dir: str,
    symbol: str,
    trade_date: str,
    calendar: list[str],
    forward_days: int = 20,
) -> float | None:
    """计算交易日后 N 个交易日的收盘涨幅。"""
    fpath = os.path.join(data_dir, f"{symbol}.csv")
    if not os.path.exists(fpath):
        return None
    try:
        df = pd.read_csv(fpath, usecols=["trade_date", "close"], dtype={"trade_date": str})
        df = df.sort_values("trade_date").reset_index(drop=True)
        match = df[df["trade_date"] == trade_date]
        if match.empty:
            return None
        idx = match.index[0]
        if idx + forward_days >= len(df):
            return None
        price_at_trade = df.at[idx, "close"]
        price_after = df.at[idx + forward_days, "close"]
        if price_at_trade <= 0:
            return None
        return (price_after - price_at_trade) / price_at_trade
    except Exception:
        return None


def load_recent_prices(
    data_dir: str,
    symbol: str,
    current_date: str,
    calendar: list[str],
    lookback: int = 60,
) -> np.ndarray | None:
    """加载 current_date 及之前的 N 个交易日收盘价。"""
    fpath = os.path.join(data_dir, f"{symbol}.csv")
    if not os.path.exists(fpath):
        return None
    try:
        df = pd.read_csv(fpath, usecols=["trade_date", "close"], dtype={"trade_date": str})
        df = df[df["trade_date"] <= current_date].sort_values("trade_date")
        if len(df) < lookback:
            return df["close"].values if len(df) > 10 else None
        return df["close"].values[-lookback:]
    except Exception:
        return None


def load_recent_volumes(
    data_dir: str,
    symbol: str,
    current_date: str,
    lookback: int = 60,
) -> np.ndarray | None:
    """加载 current_date 及之前的 N 个交易日成交量。"""
    fpath = os.path.join(data_dir, f"{symbol}.csv")
    if not os.path.exists(fpath):
        return None
    try:
        df = pd.read_csv(fpath, usecols=["trade_date", "vol"], dtype={"trade_date": str})
        df = df[df["trade_date"] <= current_date].sort_values("trade_date")
        if len(df) < lookback:
            return df["vol"].values if len(df) > 10 else None
        return df["vol"].values[-lookback:]
    except Exception:
        return None


# ─── 规则权重持久化 ───

def load_rule_weights(model_dir: str, weights_filename: str, default: dict) -> dict:
    """加载规则引擎权重 (优先读持久化文件, 否则返回 default 副本)。"""
    weights_path = os.path.join(model_dir, weights_filename)
    if os.path.exists(weights_path):
        try:
            with open(weights_path) as f:
                return json.load(f)
        except Exception:
            pass
    return dict(default)


def save_rule_weights_to(model_dir: str, weights_filename: str, weights: dict, label: str = ""):
    """保存规则引擎权重。"""
    os.makedirs(model_dir, exist_ok=True)
    weights_path = os.path.join(model_dir, weights_filename)
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(weights, f, ensure_ascii=False, indent=2)
    if label:
        logger.info(f"{label}: 规则权重已更新 → {weights_path}")


# ─── 模型预测覆盖 ───

def apply_model_override(
    df: pd.DataFrame,
    model_dir: str,
    model_filename: str,
    feature_prefix: str,
    output_col: str,
    label: str,
    extra_exclude_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    """如果有训练好的模型, 用模型预测覆盖规则评分。"""
    import pickle
    model_path = os.path.join(model_dir, model_filename)
    if not os.path.exists(model_path):
        return df
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        feat_cols = [
            c for c in df.columns
            if c.startswith(feature_prefix) and c not in extra_exclude_cols
        ]
        if not feat_cols:
            return df
        X = df[feat_cols].fillna(0).values
        proba = model.predict_proba(X)[:, 1]
        df[output_col] = np.round(proba, 4)
        logger.info(f"{label}: 使用训练模型覆盖规则评分")
    except Exception as e:
        logger.warning(f"{label}: 模型预测失败, 使用规则评分: {e}")
    return df


def save_daily_snapshot(df: pd.DataFrame, current_date: str, portfolio_dir: str, subdir: str):
    """保存当日因子快照 CSV (供后续训练 / 自动调权)。"""
    out_dir = os.path.join(portfolio_dir, subdir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{current_date}.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")


# ─── 自动权重调整 (买卖通用) ───

def auto_adjust_signal_weights(
    *,
    portfolio_dir: str,
    data_dir: str,
    calendar: list[str],
    direction: str,                 # "buy" 或 "sell"
    factors: list[str],             # 因子名列表 (不含前缀)
    default_weights: dict,          # 默认权重 (无持久化时使用)
    feature_prefix: str,            # 因子列前缀: "buy_" / "exit_"
    snapshot_subdir: str,           # 因子快照目录: "buy_quality" / "sell_weights"
    model_subdir: str,              # 模型目录: "buy_models" / "exit_models"
    weights_filename: str,          # 权重文件: "buy_rule_weights.json" / "rule_weights.json"
    label: str,                     # 日志标签: "Agent 8" / "Agent 6"
    forward_days: int = 20,
) -> dict:
    """
    基于历史交易因子 vs 实际收益的相关性, 自动调整规则引擎权重。

    买入方向 (direction='buy'):
      - 因子与 post_return 正相关 → 因子有效 → 提权 (sign=+1)
    卖出方向 (direction='sell'):
      - 因子与 post_return 负相关 → 因子有效 (因子高=卖后跌=卖对了) → 提权 (sign=-1)
    """
    if direction not in ("buy", "sell"):
        raise ValueError(f"direction must be 'buy' or 'sell', got {direction!r}")

    sign = +1.0 if direction == "buy" else -1.0
    model_dir = os.path.join(portfolio_dir, model_subdir)
    old_weights = load_rule_weights(model_dir, weights_filename, default_weights)

    trades_path = os.path.join(portfolio_dir, "trades.csv")
    if not os.path.exists(trades_path):
        logger.info(f"{label} 权重调整: 无交易记录, 跳过")
        return {"adjusted": False, "reason": "no_trades"}

    trades = pd.read_csv(trades_path, dtype={"trade_date": str})
    rows_dir = trades[trades["direction"] == direction]
    if len(rows_dir) < _MIN_ADJUST_SAMPLES:
        logger.info(f"{label} 权重调整: {direction} {len(rows_dir)} 笔 < {_MIN_ADJUST_SAMPLES}, 跳过")
        return {"adjusted": False, "reason": f"insufficient_samples({len(rows_dir)})"}

    snap_dir = os.path.join(portfolio_dir, snapshot_subdir)
    feat_cols = [f"{feature_prefix}{f}" for f in factors]

    rows = []
    for _, trade_row in rows_dir.iterrows():
        sym = trade_row["symbol"]
        tdate = str(trade_row["trade_date"])

        snap_path = os.path.join(snap_dir, f"{tdate}.csv")
        if not os.path.exists(snap_path):
            continue
        try:
            snap_df = pd.read_csv(snap_path)
        except Exception:
            continue
        sym_row = snap_df[snap_df["symbol"] == sym]
        if sym_row.empty:
            continue

        rec = {}
        for col in feat_cols:
            rec[col] = float(sym_row[col].iloc[0]) if col in sym_row.columns else 0.0

        ret = get_post_trade_return(data_dir, sym, tdate, calendar, forward_days)
        if ret is None:
            continue

        rec["post_return"] = ret
        rows.append(rec)

    if len(rows) < _MIN_ADJUST_SAMPLES:
        logger.info(f"{label} 权重调整: 有效样本 {len(rows)} 条 < {_MIN_ADJUST_SAMPLES}, 跳过")
        return {"adjusted": False, "reason": f"insufficient_valid({len(rows)})"}

    df = pd.DataFrame(rows)

    # Spearman 相关性
    correlations: dict[str, float] = {}
    for col in feat_cols:
        fname = col[len(feature_prefix):]
        if df[col].std() < 1e-9:
            correlations[fname] = 0.0
            continue
        corr = df[col].corr(df["post_return"], method="spearman")
        correlations[fname] = round(corr if not np.isnan(corr) else 0.0, 4)

    # 有效性: 买入取 +corr, 卖出取 -corr
    corr_arr = np.array([correlations.get(f, 0) for f in factors])
    effectiveness = sign * corr_arr
    clipped = np.clip(effectiveness, -0.5, 0.5)
    shifted = np.maximum(clipped + 0.5, 0.05)
    target_raw = shifted / shifted.sum()
    target_clipped = np.clip(target_raw, _MIN_WEIGHT, _MAX_WEIGHT)
    target_weights = target_clipped / target_clipped.sum()

    # 平滑混合
    new_weights: dict[str, float] = {}
    changes: dict[str, float] = {}
    for i, fname in enumerate(factors):
        old_w = old_weights.get(fname, default_weights.get(fname, 0.05))
        target_w = float(target_weights[i])
        new_w = old_w * (1 - _ADJUST_LR) + target_w * _ADJUST_LR
        new_w = max(_MIN_WEIGHT, min(_MAX_WEIGHT, new_w))
        new_weights[fname] = round(new_w, 6)
        changes[fname] = round(new_w - old_w, 6)

    total = sum(new_weights.values())
    new_weights = {k: round(v / total, 6) for k, v in new_weights.items()}

    save_rule_weights_to(model_dir, weights_filename, new_weights, label=label)

    sorted_changes = sorted(changes.items(), key=lambda x: abs(x[1]), reverse=True)
    top_changes = [(k, v) for k, v in sorted_changes[:5] if abs(v) > 0.005]
    if top_changes:
        change_str = ", ".join(f"{k}:{v:+.3f}" for k, v in top_changes)
        logger.info(f"{label} 权重调整 ({len(rows)} 样本): {change_str}")
    else:
        logger.info(f"{label} 权重调整: {len(rows)} 样本, 权重变化微小")

    return {
        "adjusted": True,
        "n_samples": len(rows),
        "correlations": correlations,
        "old_weights": {k: round(v, 6) for k, v in old_weights.items()},
        "new_weights": new_weights,
        "changes": {k: v for k, v in changes.items() if abs(v) > 0.001},
    }
