"""
Bull Hunter v4 — Agent 6: Exit Signal Trainer (卖出因子训练 Agent)

职责:
  1. 每日对已持仓股票计算退出因子
  2. 训练/更新卖出信号模型 (基于历史卖出成功/失败经验)
  3. 输出每只持仓的 sell_weight (0~1) 和 sell_reason
  4. 接收 Agent 7 Supervisor 的调参反馈

退出因子 (EXIT_FACTORS):
  - 动量衰减: momentum 由正转负, 加速度为负
  - 波动率扩大: 短期波动率显著高于长期
  - 熵无序化: permutation_entropy 上升, 路径不可逆消失
  - 资金流反转: 主力净流出加速
  - 技术支撑: 跌破关键均线, ATR 异常
  - 盈利回撤: 从最高点回撤过大

模型:
  - 初期: 规则引擎 (加权评分)
  - 积累足够交易历史后: LightGBM 二分类 (label=1: 该卖, label=0: 该继续持有)

交互:
  Agent 6 → sell_weights → Agent 5 (Portfolio)
  Agent 7 → directives  → Agent 6 (调整因子权重/阈值)

持久化:
  results/bull_hunter/portfolio/exit_models/
    rule_weights.json       # 规则引擎权重
    exit_model.pkl          # 训练后的 LightGBM 模型 (可选)
    exit_training_log.csv   # 训练样本记录
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── 退出因子 ──

EXIT_FACTORS = [
    # 动量衰减
    "momentum_decay",       # 20d 动量变化率 (由正转负 → 卖)
    "momentum_accel",       # 动量加速度 (负值越大越该卖)
    # 波动率异常
    "vol_expansion",        # 短期/长期波动率比 (>2.0 → 异常)
    "vol_spike",            # 单日波动是否异常 (>3 sigma)
    # 熵信号反转
    "entropy_disorder",     # 置换熵变化 (上升 = 无序化)
    "irrev_collapse",       # 路径不可逆从高位下降
    # 资金流
    "mf_outflow_accel",     # 主力净流出加速度
    "mf_streak_negative",   # 连续净流出天数
    # 技术
    "below_ma20",           # 跌破 20 日均线
    "below_ma60",           # 跌破 60 日均线
    "atr_anomaly",          # ATR 异常扩大
    # 盈利保护
    "drawdown_from_peak",   # 从最高涨幅的回撤
    "gain_vs_target",       # 当前涨幅 / 目标涨幅 (接近目标 → 考虑止盈)
    "holding_days_norm",    # 持有天数归一化 (越久权重越大)
]

# 规则引擎默认权重
DEFAULT_RULE_WEIGHTS = {
    "momentum_decay": 0.12,
    "momentum_accel": 0.10,
    "vol_expansion": 0.08,
    "vol_spike": 0.06,
    "entropy_disorder": 0.10,
    "irrev_collapse": 0.08,
    "mf_outflow_accel": 0.10,
    "mf_streak_negative": 0.06,
    "below_ma20": 0.05,
    "below_ma60": 0.08,
    "atr_anomaly": 0.04,
    "drawdown_from_peak": 0.08,
    "gain_vs_target": 0.03,
    "holding_days_norm": 0.02,
}

# 卖出阈值 (加权得分 > 此值 → sell_weight 高)
SELL_SCORE_THRESHOLD = 0.50
# LightGBM 训练最小样本数
MIN_TRAIN_SAMPLES = 50


@dataclass
class ExitSignalConfig:
    """Agent 6 配置 (可被 Agent 7 调整)。"""
    rule_weights: dict = field(default_factory=lambda: dict(DEFAULT_RULE_WEIGHTS))
    use_model: bool = False       # 是否使用训练模型 (积累足够样本后启用)
    retrain_interval: int = 20    # 多少笔新交易后重训
    target_gain: float = 2.00     # 目标涨幅 (200%)
    max_hold_days: int = 120      # 最大持有天数 (超过后权重增加)


def run_exit_signal(
    positions: pd.DataFrame,
    factor_snapshot: pd.DataFrame,
    current_date: str,
    data_dir: str,
    calendar: list[str],
    portfolio_dir: str,
    cfg: ExitSignalConfig | None = None,
) -> pd.DataFrame:
    """
    Agent 6: 为每只持仓计算卖出权重。

    Args:
        positions: 当前持仓 DataFrame (来自 Portfolio)
        factor_snapshot: Agent 1 全市场因子快照
        current_date: 当日日期
        data_dir: 日线数据目录
        calendar: 交易日历
        portfolio_dir: 持仓目录 (读写模型/权重)
        cfg: Agent 6 配置

    Returns:
        DataFrame (symbol, sell_weight, sell_reason, + 各退出因子得分)
    """
    cfg = cfg or ExitSignalConfig()

    if positions.empty:
        logger.info("Agent 6: 无持仓, 跳过")
        return pd.DataFrame(columns=["symbol", "sell_weight", "sell_reason"])

    model_dir = os.path.join(portfolio_dir, "exit_models")
    os.makedirs(model_dir, exist_ok=True)

    # 加载规则权重 (可能被 Agent 7 更新过)
    weights = _load_rule_weights(model_dir, cfg)

    # ── 计算每只持仓的退出因子 ──
    results = []
    for _, pos in positions.iterrows():
        sym = pos["symbol"]
        factors = _compute_exit_factors(
            sym=sym,
            pos=pos,
            factor_snapshot=factor_snapshot,
            data_dir=data_dir,
            current_date=current_date,
            calendar=calendar,
            cfg=cfg,
        )

        # 加权评分
        score = 0.0
        for fname, fval in factors.items():
            w = weights.get(fname, 0)
            score += w * fval

        # 归一化到 0~1
        sell_weight = min(max(score, 0.0), 1.0)

        # 生成卖出原因
        sell_reason = _generate_sell_reason(factors, weights, sell_weight)

        row = {"symbol": sym, "sell_weight": round(sell_weight, 4), "sell_reason": sell_reason}
        row.update({f"exit_{k}": round(v, 4) for k, v in factors.items()})
        results.append(row)

    df = pd.DataFrame(results)

    # 尝试用 LightGBM 模型覆盖 (如果可用)
    if cfg.use_model:
        df = _apply_model_override(df, model_dir)

    n_high = (df["sell_weight"] >= 0.6).sum() if not df.empty else 0
    logger.info(f"Agent 6 完成: {len(df)} 只持仓评估, {n_high} 只卖出权重 >= 0.6")

    # 保存当日卖出权重
    _save_daily_weights(df, current_date, portfolio_dir)

    return df


def train_exit_model(
    portfolio_dir: str,
    data_dir: str,
    calendar: list[str],
    cfg: ExitSignalConfig | None = None,
    min_samples: int | None = None,
) -> dict:
    """
    训练卖出信号模型 (基于历史交易的卖出时机好坏)。

    样本构造:
      正例 (label=1, 该卖):
        - 卖出后 20 天股价没有回升 (后续涨幅 < 5%)
      负例 (label=0, 不该卖/卖早了):
        - 卖出后 20 天股价大幅上涨 (后续涨幅 >= 5%)
      补充负例:
        - 持有期间每日 sell_weight 快照, 且持有未卖时涨幅还在增长

    Args:
        min_samples: 最少卖出样本数 (默认 MIN_TRAIN_SAMPLES=50, 回测可降低)

    Returns:
        {"trained": bool, "n_samples": int, "auc": float}
    """
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    cfg = cfg or ExitSignalConfig()
    required_samples = min_samples if min_samples is not None else MIN_TRAIN_SAMPLES
    model_dir = os.path.join(portfolio_dir, "exit_models")
    os.makedirs(model_dir, exist_ok=True)

    trades_path = os.path.join(portfolio_dir, "trades.csv")
    if not os.path.exists(trades_path):
        logger.info("Agent 6 训练: 无交易记录, 跳过")
        return {"trained": False, "reason": "no_trades"}

    trades = pd.read_csv(trades_path, dtype={"trade_date": str})
    sells = trades[trades["direction"] == "sell"]

    if len(sells) < required_samples:
        logger.info(f"Agent 6 训练: 卖出记录 {len(sells)} 条, 不足 {required_samples}, 跳过")
        return {"trained": False, "reason": f"insufficient_samples({len(sells)})"}

    logger.info(f"Agent 6 训练: {len(sells)} 条卖出记录, 构建训练集...")

    # ── 1. 从 sell_weights 目录加载退出因子快照 ──
    sw_dir = os.path.join(portfolio_dir, "sell_weights")
    exit_cols = [f"exit_{f}" for f in EXIT_FACTORS]

    samples_X = []
    samples_y = []

    # ── 2. 正例/负例: 卖出事件 ──
    POST_SELL_DAYS = 20
    GOOD_SELL_THRESHOLD = 0.05  # 卖出后 20 天涨幅 < 5% → 卖对了

    for _, sell_row in sells.iterrows():
        sym = sell_row["symbol"]
        sell_date = str(sell_row["trade_date"])

        # 加载卖出当天的退出因子
        sw_path = os.path.join(sw_dir, f"{sell_date}.csv")
        if not os.path.exists(sw_path):
            continue
        sw_df = pd.read_csv(sw_path)
        sym_row = sw_df[sw_df["symbol"] == sym]
        if sym_row.empty:
            continue

        features = []
        for col in exit_cols:
            features.append(float(sym_row[col].iloc[0]) if col in sym_row.columns else 0.0)

        # 计算卖出后 20 天涨幅
        post_sell_return = _get_post_trade_return(
            data_dir, sym, sell_date, calendar, POST_SELL_DAYS
        )
        if post_sell_return is None:
            continue

        # label: 1=该卖(后续没涨), 0=卖早了(后续涨了)
        label = 1 if post_sell_return < GOOD_SELL_THRESHOLD else 0

        samples_X.append(features)
        samples_y.append(label)

    # ── 3. 补充负例: 持有中未卖出且后续上涨的快照 ──
    if os.path.isdir(sw_dir):
        sw_files = sorted([f for f in os.listdir(sw_dir) if f.endswith(".csv")])
        # 采样最多 200 条, 避免负例过多
        import random
        sample_files = random.sample(sw_files, min(len(sw_files), 50))

        for sw_file in sample_files:
            date = sw_file.replace(".csv", "")
            sw_path = os.path.join(sw_dir, sw_file)
            try:
                sw_df = pd.read_csv(sw_path)
            except Exception:
                continue

            # 只取 sell_weight 低的 (实际没卖出的)
            held = sw_df[sw_df["sell_weight"] < 0.4]
            for _, row in held.iterrows():
                sym = row["symbol"]
                post_return = _get_post_trade_return(
                    data_dir, sym, date, calendar, POST_SELL_DAYS
                )
                if post_return is None:
                    continue

                # 持有期间涨了 → 不该卖 (label=0)
                if post_return > 0.05:
                    features = []
                    for col in exit_cols:
                        features.append(float(row[col]) if col in row.index else 0.0)
                    samples_X.append(features)
                    samples_y.append(0)

    if len(samples_X) < MIN_TRAIN_SAMPLES:
        logger.info(f"Agent 6 训练: 有效样本 {len(samples_X)} 条, 不足, 跳过")
        return {"trained": False, "reason": f"insufficient_valid_samples({len(samples_X)})"}

    X = np.array(samples_X, dtype=np.float64)
    y = np.array(samples_y, dtype=np.int32)
    X = np.nan_to_num(X, nan=0.0)

    pos_count = y.sum()
    neg_count = len(y) - pos_count
    logger.info(f"Agent 6 训练: {len(X)} 样本 (正例={pos_count}, 负例={neg_count})")

    if pos_count < 3 or neg_count < 3:
        logger.info("Agent 6 训练: 正/负例不足, 跳过")
        return {"trained": False, "reason": "class_imbalance"}

    # ── 4. 训练 LightGBM ──
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scale = neg_count / max(pos_count, 1)
    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale,
        random_state=42,
        verbosity=-1,
    )
    model.fit(X_train, y_train)

    # 评估
    y_pred = model.predict_proba(X_val)[:, 1]
    try:
        auc = roc_auc_score(y_val, y_pred)
    except ValueError:
        auc = 0.5

    logger.info(f"Agent 6 训练完成: AUC={auc:.3f}")

    # ── 5. 保存模型 ──
    model_path = os.path.join(model_dir, "exit_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # 保存训练元数据
    meta = {
        "n_samples": len(X),
        "n_positive": int(pos_count),
        "n_negative": int(neg_count),
        "auc": round(auc, 4),
        "feature_names": exit_cols,
    }
    meta_path = os.path.join(model_dir, "exit_model_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 如果 AUC 足够好, 启用模型
    if auc >= 0.55:
        cfg.use_model = True
        logger.info(f"Agent 6: 模型 AUC {auc:.3f} >= 0.55, 启用模型预测")
    else:
        logger.info(f"Agent 6: 模型 AUC {auc:.3f} < 0.55, 保持规则引擎")

    return {"trained": True, "n_samples": len(X), "auc": round(auc, 4)}


def _get_post_trade_return(
    data_dir: str,
    symbol: str,
    trade_date: str,
    calendar: list[str],
    forward_days: int = 20,
) -> float | None:
    """计算交易后 N 个交易日的涨幅。"""
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


# ─── 退出因子计算 ───

def _compute_exit_factors(
    sym: str,
    pos: pd.Series,
    factor_snapshot: pd.DataFrame,
    data_dir: str,
    current_date: str,
    calendar: list[str],
    cfg: ExitSignalConfig,
) -> dict[str, float]:
    """计算单只持仓的全部退出因子 (值域 0~1, 越大越该卖)。"""
    factors = {}

    # 从 factor_snapshot 获取当日因子
    snap = {}
    if not factor_snapshot.empty and sym in factor_snapshot.index:
        snap = factor_snapshot.loc[sym].to_dict()

    # 从持仓获取基础数据
    buy_price = float(pos.get("buy_price", 0))
    current_gain = float(pos.get("current_gain", 0))
    max_gain = float(pos.get("max_gain", 0))
    days_held = int(pos.get("days_held", 0))

    # 加载近期价格序列
    prices = _load_recent_prices(data_dir, sym, current_date, calendar, lookback=60)

    # ── 1. 动量衰减 ──
    mom_20d = float(snap.get("momentum_20d", 0)) if snap else 0
    mom_60d = float(snap.get("momentum_60d", 0)) if snap else 0
    # 动量从正转负 → 高分
    if mom_20d < 0 and mom_60d > 0:
        factors["momentum_decay"] = min(abs(mom_20d) * 5, 1.0)
    elif mom_20d < 0 and mom_60d < 0:
        factors["momentum_decay"] = min(abs(mom_20d) * 3, 1.0)
    elif mom_20d < -0.1:
        factors["momentum_decay"] = min(abs(mom_20d) * 2, 1.0)
    else:
        factors["momentum_decay"] = 0.0

    # ── 2. 动量加速度 ──
    if prices is not None and len(prices) >= 30:
        ret_10d = prices[-1] / prices[-11] - 1 if prices[-11] > 0 else 0
        ret_prev_10d = prices[-11] / prices[-21] - 1 if len(prices) > 20 and prices[-21] > 0 else 0
        accel = ret_10d - ret_prev_10d
        factors["momentum_accel"] = min(max(-accel * 5, 0), 1.0)  # 减速越大分越高
    else:
        factors["momentum_accel"] = 0.0

    # ── 3. 波动率异常 ──
    vol_ratio = float(snap.get("volatility_ratio", 1.0)) if snap else 1.0
    factors["vol_expansion"] = min(max((vol_ratio - 1.5) / 1.5, 0), 1.0)

    # ── 4. 波动率突刺 ──
    if prices is not None and len(prices) >= 20:
        daily_ret = np.diff(prices) / prices[:-1]
        recent_ret = abs(daily_ret[-1]) if len(daily_ret) > 0 else 0
        std_20d = np.std(daily_ret[-20:]) if len(daily_ret) >= 20 else 0.01
        z_score = recent_ret / (std_20d + 1e-8)
        factors["vol_spike"] = min(max((z_score - 2.0) / 3.0, 0), 1.0)
    else:
        factors["vol_spike"] = 0.0

    # ── 5. 熵无序化 ──
    pe_s = float(snap.get("perm_entropy_s", 0.5)) if snap else 0.5
    pe_m = float(snap.get("perm_entropy_m", 0.5)) if snap else 0.5
    # 短期熵高于中期 → 无序化加剧
    entropy_diff = pe_s - pe_m
    factors["entropy_disorder"] = min(max(entropy_diff * 3 + 0.2 * (pe_s - 0.6), 0), 1.0)

    # ── 6. 路径不可逆崩塌 ──
    irrev_m = float(snap.get("path_irrev_m", 0)) if snap else 0
    irrev_l = float(snap.get("path_irrev_l", 0)) if snap else 0
    # 不可逆下降 → 主力撤退
    if irrev_m < 0.05 and irrev_l > 0.10:
        factors["irrev_collapse"] = min((irrev_l - irrev_m) * 5, 1.0)
    else:
        factors["irrev_collapse"] = 0.0

    # ── 7. 资金流出加速 ──
    mf_momentum = float(snap.get("mf_big_momentum", 0)) if snap else 0
    factors["mf_outflow_accel"] = min(max(-mf_momentum * 3, 0), 1.0)

    # ── 8. 连续净流出天数 ──
    mf_streak = float(snap.get("mf_big_streak", 0)) if snap else 0
    factors["mf_streak_negative"] = min(max(-mf_streak / 5.0, 0), 1.0) if mf_streak < 0 else 0.0

    # ── 9. 跌破 MA20 ──
    price_vs_ma20 = float(snap.get("price_vs_ma20", 0)) if snap else 0
    factors["below_ma20"] = min(max(-price_vs_ma20 * 3, 0), 1.0) if price_vs_ma20 < 0 else 0.0

    # ── 10. 跌破 MA60 ──
    close_vs_high_60 = float(snap.get("close_vs_high_60d", 0)) if snap else 0
    factors["below_ma60"] = min(max(-close_vs_high_60, 0), 1.0) if close_vs_high_60 < -0.15 else 0.0

    # ── 11. ATR 异常 ──
    atr = float(snap.get("atr_20d", 0)) if snap else 0
    factors["atr_anomaly"] = min(max((atr - 0.05) * 10, 0), 1.0)

    # ── 12. 从最高点回撤 ──
    dd = max_gain - current_gain if max_gain > 0 else abs(min(current_gain, 0))
    factors["drawdown_from_peak"] = min(dd / 0.30, 1.0)

    # ── 13. 涨幅 vs 目标 ──
    if cfg.target_gain > 0 and current_gain > 0:
        ratio = current_gain / cfg.target_gain
        factors["gain_vs_target"] = min(ratio, 1.0)  # 接近目标 → 可以止盈
    else:
        factors["gain_vs_target"] = 0.0

    # ── 14. 持有天数归一化 ──
    factors["holding_days_norm"] = min(days_held / cfg.max_hold_days, 1.0)

    return factors


def _generate_sell_reason(factors: dict, weights: dict, sell_weight: float) -> str:
    """根据最高权重因子生成卖出原因描述。"""
    if sell_weight < 0.3:
        return ""

    # 找出贡献最大的因子
    contributions = [(k, factors.get(k, 0) * weights.get(k, 0)) for k in factors]
    contributions.sort(key=lambda x: x[1], reverse=True)
    top3 = [f"{k}({v:.2f})" for k, v in contributions[:3] if v > 0.01]

    if not top3:
        return ""
    return "|".join(top3)


def _load_recent_prices(
    data_dir: str,
    symbol: str,
    current_date: str,
    calendar: list[str],
    lookback: int = 60,
) -> np.ndarray | None:
    """加载近期收盘价序列。"""
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


def _load_rule_weights(model_dir: str, cfg: ExitSignalConfig) -> dict:
    """加载规则引擎权重 (优先读持久化文件, 否则用配置默认)。"""
    weights_path = os.path.join(model_dir, "rule_weights.json")
    if os.path.exists(weights_path):
        try:
            with open(weights_path) as f:
                return json.load(f)
        except Exception:
            pass
    return dict(cfg.rule_weights)


def save_rule_weights(model_dir: str, weights: dict):
    """保存规则引擎权重 (Agent 7 调优后调用)。"""
    weights_path = os.path.join(model_dir, "rule_weights.json")
    os.makedirs(model_dir, exist_ok=True)
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(weights, f, ensure_ascii=False, indent=2)
    logger.info(f"Agent 6: 规则权重已更新 → {weights_path}")


def _apply_model_override(df: pd.DataFrame, model_dir: str) -> pd.DataFrame:
    """如果有训练好的模型, 用模型预测覆盖规则评分。"""
    model_path = os.path.join(model_dir, "exit_model.pkl")
    if not os.path.exists(model_path):
        return df

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        exit_cols = [c for c in df.columns if c.startswith("exit_")]
        if not exit_cols:
            return df

        X = df[exit_cols].fillna(0).values
        proba = model.predict_proba(X)[:, 1]
        df["sell_weight"] = np.round(proba, 4)
        logger.info("Agent 6: 使用训练模型覆盖规则评分")
    except Exception as e:
        logger.warning(f"Agent 6: 模型预测失败, 使用规则评分: {e}")

    return df


def _save_daily_weights(df: pd.DataFrame, current_date: str, portfolio_dir: str):
    """保存每日卖出权重 (供 Agent 7 分析)。"""
    out_dir = os.path.join(portfolio_dir, "sell_weights")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{current_date}.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
