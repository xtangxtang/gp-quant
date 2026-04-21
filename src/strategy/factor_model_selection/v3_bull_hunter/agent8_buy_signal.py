"""
Bull Hunter v6 — Agent 8: Buy Signal Quality (买入时机评估 Agent)

职责:
  1. 对 Agent 3 Top N 候选计算买入时机因子 (当天是否好买点)
  2. 训练/更新买入质量模型 (基于历史买入后 5/10/20 天表现)
  3. 输出 buy_quality (0~1) 和 buy_reason, 供 Agent 5 过滤或排序
  4. 与 Agent 6 (卖出信号) 对称

买入时机因子 (BUY_FACTORS):
  - 动量启动: 短期动量由负转正, 加速度为正
  - 波动率压缩: 长期横盘后波动率收窄 (布林带/ATR)
  - 熵有序化: 置换熵下降, 路径不可逆上升 → 主力控盘
  - 资金流入: 主力连续净流入, 流入加速
  - 技术支撑: 站上均线, 突破前期高点
  - 量价配合: 放量上涨, 缩量回调

模型:
  - 初期: 规则引擎 (加权评分)
  - 积累足够交易历史后: LightGBM 二分类 (label=1: 好买点, label=0: 差买点)

交互:
  Agent 3 → candidates → Agent 8 → buy_quality → Agent 5 (Portfolio)

持久化:
  results/bull_hunter/portfolio/buy_models/
    buy_rule_weights.json   # 规则引擎权重
    buy_model.pkl           # 训练后的 LightGBM 模型 (可选)
    buy_model_meta.json     # 训练元数据
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import _signal_common as _sc

logger = logging.getLogger(__name__)

# ── 买入时机因子 ──

BUY_FACTORS = [
    # 动量启动
    "momentum_ignition",    # 20d 动量由负转正 → 启动信号
    "momentum_accel",       # 动量加速度 (正值越大越好)
    # 波动率压缩
    "vol_compression",      # 波动率压缩度 (ATR 收窄)
    "bbw_squeeze",          # 布林带宽收窄 (低位百分位)
    # 熵有序化
    "entropy_ordering",     # 置换熵下降 → 有序化
    "irrev_buildup",        # 路径不可逆上升 → 主力进场
    # 资金流入
    "mf_inflow_accel",      # 主力净流入加速度
    "mf_streak_positive",   # 连续净流入天数
    # 技术支撑
    "above_ma20",           # 站上 20 日均线
    "above_ma60",           # 站上 60 日均线
    "breakout_strength",    # 突破前期阻力位的力度
    # 量价配合
    "vol_price_synergy",    # 量价协同 (放量上涨)
    "volume_expansion",     # 成交量放大倍数
    "pullback_depth",       # 回调深度 (适度回调 → 好买点)
]

# 规则引擎默认权重
DEFAULT_BUY_WEIGHTS = {
    "momentum_ignition": 0.10,
    "momentum_accel": 0.08,
    "vol_compression": 0.10,
    "bbw_squeeze": 0.08,
    "entropy_ordering": 0.12,
    "irrev_buildup": 0.10,
    "mf_inflow_accel": 0.10,
    "mf_streak_positive": 0.06,
    "above_ma20": 0.05,
    "above_ma60": 0.04,
    "breakout_strength": 0.05,
    "vol_price_synergy": 0.05,
    "volume_expansion": 0.04,
    "pullback_depth": 0.03,
}

# LightGBM 训练最小样本数
MIN_TRAIN_SAMPLES = 30


@dataclass
class BuySignalConfig:
    """Agent 8 配置。"""
    rule_weights: dict = field(default_factory=lambda: dict(DEFAULT_BUY_WEIGHTS))
    use_model: bool = False       # 是否使用训练模型
    min_buy_quality: float = 0.3  # 低于此 quality 不推荐买入
    min_prob_200: float = 0.30    # Agent 3 预测概率硬门槛 (V9: 0.20 → 0.30)
    retrain_interval: int = 20    # 多少笔新交易后重训


def run_buy_signal(
    candidates: pd.DataFrame,
    factor_snapshot: pd.DataFrame,
    current_date: str,
    data_dir: str,
    calendar: list[str],
    portfolio_dir: str,
    cfg: BuySignalConfig | None = None,
) -> pd.DataFrame:
    """
    Agent 8: 为每只候选股票计算买入时机质量。

    Args:
        candidates: Agent 3 Top N 候选 (symbol, prob_200, prob_100, rank, ...)
        factor_snapshot: Agent 1 全市场因子快照
        current_date: 当日日期
        data_dir: 日线数据目录
        calendar: 交易日历
        portfolio_dir: 持仓目录 (读写模型/权重)
        cfg: Agent 8 配置

    Returns:
        输入 candidates DataFrame 附加列:
          buy_quality: float (0~1, 越大越是好买点)
          buy_reason: str (主要买入理由)
          buy_{factor}: 各因子得分
    """
    cfg = cfg or BuySignalConfig()

    if candidates.empty:
        logger.info("Agent 8: 无候选, 跳过")
        return candidates

    model_dir = os.path.join(portfolio_dir, "buy_models")
    os.makedirs(model_dir, exist_ok=True)

    weights = _sc.load_rule_weights(model_dir, "buy_rule_weights.json", DEFAULT_BUY_WEIGHTS)

    results = []
    for _, row in candidates.iterrows():
        sym = row["symbol"]
        factors = _compute_buy_factors(
            sym=sym,
            factor_snapshot=factor_snapshot,
            data_dir=data_dir,
            current_date=current_date,
            calendar=calendar,
        )

        # 加权评分
        score = 0.0
        for fname, fval in factors.items():
            w = weights.get(fname, 0)
            score += w * fval

        buy_quality = min(max(score, 0.0), 1.0)
        buy_reason = _generate_buy_reason(factors, weights, buy_quality)

        entry = row.to_dict()
        entry["buy_quality"] = round(buy_quality, 4)
        entry["buy_reason"] = buy_reason
        entry.update({f"buy_{k}": round(v, 4) for k, v in factors.items()})
        results.append(entry)

    df = pd.DataFrame(results)

    # 尝试用 LightGBM 模型覆盖
    if cfg.use_model:
        df = _sc.apply_model_override(
            df, model_dir,
            model_filename="buy_model.pkl",
            feature_prefix="buy_",
            output_col="buy_quality",
            label="Agent 8",
            extra_exclude_cols=("buy_quality", "buy_reason"),
        )

    # 按 buy_quality 降序排序 (保留原有排名信息)
    if not df.empty:
        df = df.sort_values("buy_quality", ascending=False).reset_index(drop=True)

    n_good = (df["buy_quality"] >= cfg.min_buy_quality).sum() if not df.empty else 0
    logger.info(f"Agent 8 完成: {len(df)} 只候选评估, {n_good} 只 buy_quality >= {cfg.min_buy_quality}")

    # 保存当日买入质量快照
    _sc.save_daily_snapshot(df, current_date, portfolio_dir, subdir="buy_quality")

    return df


def train_buy_model(
    portfolio_dir: str,
    data_dir: str,
    calendar: list[str],
    cfg: BuySignalConfig | None = None,
    min_samples: int | None = None,
) -> dict:
    """
    训练买入信号模型 (基于历史买入后的表现好坏)。

    样本构造:
      正例 (label=1, 好买点):
        - 买入后 20 天涨幅 >= 5%
      负例 (label=0, 差买点):
        - 买入后 20 天涨幅 < 0% (亏损)

    Returns:
        {"trained": bool, "n_samples": int, "auc": float}
    """
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    cfg = cfg or BuySignalConfig()
    required_samples = min_samples if min_samples is not None else MIN_TRAIN_SAMPLES
    model_dir = os.path.join(portfolio_dir, "buy_models")
    os.makedirs(model_dir, exist_ok=True)

    trades_path = os.path.join(portfolio_dir, "trades.csv")
    if not os.path.exists(trades_path):
        logger.info("Agent 8 训练: 无交易记录, 跳过")
        return {"trained": False, "reason": "no_trades"}

    trades = pd.read_csv(trades_path, dtype={"trade_date": str})
    buys = trades[trades["direction"] == "buy"]

    if len(buys) < required_samples:
        logger.info(f"Agent 8 训练: 买入记录 {len(buys)} 条, 不足 {required_samples}, 跳过")
        return {"trained": False, "reason": f"insufficient_samples({len(buys)})"}

    logger.info(f"Agent 8 训练: {len(buys)} 条买入记录, 构建训练集...")

    bq_dir = os.path.join(portfolio_dir, "buy_quality")
    buy_cols = [f"buy_{f}" for f in BUY_FACTORS]

    samples_X = []
    samples_y = []

    POST_BUY_DAYS = 20
    GOOD_BUY_THRESHOLD = 0.05   # 买入后 20 天涨幅 >= 5% → 好买点
    BAD_BUY_THRESHOLD = 0.0     # 买入后 20 天涨幅 < 0% → 差买点

    for _, buy_row in buys.iterrows():
        sym = buy_row["symbol"]
        buy_date = str(buy_row["trade_date"])

        # 加载买入当天的买入因子快照
        bq_path = os.path.join(bq_dir, f"{buy_date}.csv")
        if not os.path.exists(bq_path):
            continue
        bq_df = pd.read_csv(bq_path)
        sym_row = bq_df[bq_df["symbol"] == sym]
        if sym_row.empty:
            continue

        features = []
        for col in buy_cols:
            features.append(float(sym_row[col].iloc[0]) if col in sym_row.columns else 0.0)

        # 计算买入后收益
        post_buy_return = _sc.get_post_trade_return(
            data_dir, sym, buy_date, calendar, POST_BUY_DAYS
        )
        if post_buy_return is None:
            continue

        # label: 1=好买点, 0=差买点; 中间区域跳过
        if post_buy_return >= GOOD_BUY_THRESHOLD:
            label = 1
        elif post_buy_return < BAD_BUY_THRESHOLD:
            label = 0
        else:
            continue  # 0~5% 模糊区, 不参与训练

        samples_X.append(features)
        samples_y.append(label)

    if len(samples_X) < MIN_TRAIN_SAMPLES:
        logger.info(f"Agent 8 训练: 有效样本 {len(samples_X)} 条, 不足, 跳过")
        return {"trained": False, "reason": f"insufficient_valid_samples({len(samples_X)})"}

    X = np.array(samples_X, dtype=np.float64)
    y = np.array(samples_y, dtype=np.int32)
    X = np.nan_to_num(X, nan=0.0)

    pos_count = y.sum()
    neg_count = len(y) - pos_count
    logger.info(f"Agent 8 训练: {len(X)} 样本 (好买点={pos_count}, 差买点={neg_count})")

    if pos_count < 3 or neg_count < 3:
        logger.info("Agent 8 训练: 正/负例不足, 跳过")
        return {"trained": False, "reason": "class_imbalance"}

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

    y_pred = model.predict_proba(X_val)[:, 1]
    try:
        auc = roc_auc_score(y_val, y_pred)
    except ValueError:
        auc = 0.5

    logger.info(f"Agent 8 训练完成: AUC={auc:.3f}")

    model_path = os.path.join(model_dir, "buy_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    meta = {
        "n_samples": len(X),
        "n_positive": int(pos_count),
        "n_negative": int(neg_count),
        "auc": round(auc, 4),
        "feature_names": buy_cols,
    }
    meta_path = os.path.join(model_dir, "buy_model_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if auc >= 0.55:
        cfg.use_model = True
        logger.info(f"Agent 8: 模型 AUC {auc:.3f} >= 0.55, 启用模型预测")
    else:
        logger.info(f"Agent 8: 模型 AUC {auc:.3f} < 0.55, 保持规则引擎")

    return {"trained": True, "n_samples": len(X), "auc": round(auc, 4)}


# ─── 买入因子计算 ───

def _compute_buy_factors(
    sym: str,
    factor_snapshot: pd.DataFrame,
    data_dir: str,
    current_date: str,
    calendar: list[str],
) -> dict[str, float]:
    """计算单只候选股的全部买入因子 (值域 0~1, 越大越是好买点)。"""
    factors = {}

    # 从 factor_snapshot 获取当日因子
    snap = {}
    if not factor_snapshot.empty and sym in factor_snapshot.index:
        snap = factor_snapshot.loc[sym].to_dict()

    # 加载近期价格序列
    prices = _sc.load_recent_prices(data_dir, sym, current_date, calendar, lookback=60)

    # 加载近期成交量
    volumes = _sc.load_recent_volumes(data_dir, sym, current_date, lookback=60)

    # ── 1. 动量启动 ──
    mom_20d = float(snap.get("momentum_20d", 0)) if snap else 0
    mom_60d = float(snap.get("momentum_60d", 0)) if snap else 0
    # 动量从负转正 → 启动信号
    if mom_20d > 0 and mom_60d < 0:
        factors["momentum_ignition"] = min(mom_20d * 5, 1.0)
    elif mom_20d > 0.05:
        factors["momentum_ignition"] = min(mom_20d * 3, 1.0)
    else:
        factors["momentum_ignition"] = max(mom_20d * 2, 0.0) if mom_20d > 0 else 0.0

    # ── 2. 动量加速度 ──
    if prices is not None and len(prices) >= 30:
        ret_10d = prices[-1] / prices[-11] - 1 if prices[-11] > 0 else 0
        ret_prev_10d = prices[-11] / prices[-21] - 1 if len(prices) > 20 and prices[-21] > 0 else 0
        accel = ret_10d - ret_prev_10d
        factors["momentum_accel"] = min(max(accel * 5, 0), 1.0)  # 加速为正 → 高分
    else:
        factors["momentum_accel"] = 0.0

    # ── 3. 波动率压缩 ──
    vol_comp = float(snap.get("vol_compression", 0)) if snap else 0
    # vol_compression 越大 → 波动率越收窄 → 越好
    factors["vol_compression"] = min(max(vol_comp, 0), 1.0)

    # ── 4. 布林带收窄 ──
    bbw = float(snap.get("bbw_pctl", 0.5)) if snap else 0.5
    # bbw_pctl 低 → 布林带窄 → 即将突破
    factors["bbw_squeeze"] = min(max(1.0 - bbw, 0), 1.0)

    # ── 5. 熵有序化 ──
    pe_s = float(snap.get("perm_entropy_s", 0.5)) if snap else 0.5
    pe_m = float(snap.get("perm_entropy_m", 0.5)) if snap else 0.5
    entropy_slope = float(snap.get("entropy_slope", 0)) if snap else 0
    # 熵下降 (有序化) → 高分
    ordering = 0.0
    if pe_s < pe_m:
        ordering += min((pe_m - pe_s) * 3, 0.5)
    if entropy_slope < 0:
        ordering += min(abs(entropy_slope) * 2, 0.5)
    factors["entropy_ordering"] = min(ordering, 1.0)

    # ── 6. 路径不可逆上升 (主力进场) ──
    irrev_m = float(snap.get("path_irrev_m", 0)) if snap else 0
    irrev_l = float(snap.get("path_irrev_l", 0)) if snap else 0
    # 中期不可逆高于长期 → 主力正在加仓
    if irrev_m > 0.10:
        factors["irrev_buildup"] = min(irrev_m * 3, 1.0)
    elif irrev_m > irrev_l and irrev_m > 0.05:
        factors["irrev_buildup"] = min((irrev_m - irrev_l) * 5 + 0.2, 1.0)
    else:
        factors["irrev_buildup"] = 0.0

    # ── 7. 资金流入加速 ──
    mf_momentum = float(snap.get("mf_big_momentum", 0)) if snap else 0
    factors["mf_inflow_accel"] = min(max(mf_momentum * 3, 0), 1.0)

    # ── 8. 连续净流入天数 ──
    mf_streak = float(snap.get("mf_big_streak", 0)) if snap else 0
    factors["mf_streak_positive"] = min(max(mf_streak / 5.0, 0), 1.0) if mf_streak > 0 else 0.0

    # ── 9. 站上 MA20 ──
    price_vs_ma20 = float(snap.get("price_vs_ma20", 0)) if snap else 0
    factors["above_ma20"] = min(max(price_vs_ma20 * 3, 0), 1.0) if price_vs_ma20 > 0 else 0.0

    # ── 10. 站上 MA60 ──
    close_vs_high_60 = float(snap.get("close_vs_high_60d", 0)) if snap else 0
    # close_vs_high_60d 接近 1.0 → 在 60 日高点附近 → 强势
    if close_vs_high_60 > 0.85:
        factors["above_ma60"] = min((close_vs_high_60 - 0.85) * 6, 1.0)
    else:
        factors["above_ma60"] = 0.0

    # ── 11. 突破力度 ──
    breakout = float(snap.get("breakout_range", 0)) if snap else 0
    factors["breakout_strength"] = min(max(breakout * 2, 0), 1.0)

    # ── 12. 量价协同 ──
    vps = float(snap.get("vol_price_synergy", 0)) if snap else 0
    factors["vol_price_synergy"] = min(max(vps * 5, 0), 1.0)

    # ── 13. 成交量放大 ──
    if volumes is not None and len(volumes) >= 20:
        vol_5 = np.mean(volumes[-5:])
        vol_20 = np.mean(volumes[-20:])
        ratio = vol_5 / (vol_20 + 1e-8)
        # 1.5~3 倍 → 好 (温和放量); >3 → 过热
        if 1.3 <= ratio <= 3.0:
            factors["volume_expansion"] = min((ratio - 1.0) / 2.0, 1.0)
        elif ratio > 3.0:
            factors["volume_expansion"] = max(1.0 - (ratio - 3.0) * 0.3, 0.3)
        else:
            factors["volume_expansion"] = 0.0
    else:
        factors["volume_expansion"] = 0.0

    # ── 14. 回调深度 (适度回调 → 好买点) ──
    if prices is not None and len(prices) >= 20:
        high_20 = np.max(prices[-20:])
        current = prices[-1]
        if high_20 > 0:
            pullback = 1.0 - current / high_20
            # 5~15% 回调 → 最佳; >25% → 太深
            if 0.03 <= pullback <= 0.15:
                factors["pullback_depth"] = min(pullback * 5, 1.0)
            elif pullback < 0.03:
                factors["pullback_depth"] = 0.1  # 几乎没回调
            else:
                factors["pullback_depth"] = max(0.5 - (pullback - 0.15) * 3, 0.0)
        else:
            factors["pullback_depth"] = 0.0
    else:
        factors["pullback_depth"] = 0.0

    return factors


def _generate_buy_reason(factors: dict, weights: dict, buy_quality: float) -> str:
    """根据最高权重因子生成买入理由描述。"""
    if buy_quality < 0.2:
        return ""

    contributions = [(k, factors.get(k, 0) * weights.get(k, 0)) for k in factors]
    contributions.sort(key=lambda x: x[1], reverse=True)
    top3 = [f"{k}({v:.2f})" for k, v in contributions[:3] if v > 0.01]

    if not top3:
        return ""
    return "|".join(top3)


# ─── 自动权重调整 ───

def auto_adjust_buy_weights(
    portfolio_dir: str,
    data_dir: str,
    calendar: list[str],
    cfg: BuySignalConfig | None = None,
) -> dict:
    """
    基于历史买入因子 vs 买后收益的 Spearman 相关性, 自动调整规则引擎权重。
    薄包装: 委托给 _signal_common.auto_adjust_signal_weights (buy 方向)。
    """
    cfg = cfg or BuySignalConfig()
    return _sc.auto_adjust_signal_weights(
        portfolio_dir=portfolio_dir,
        data_dir=data_dir,
        calendar=calendar,
        direction="buy",
        factors=BUY_FACTORS,
        default_weights=DEFAULT_BUY_WEIGHTS,
        feature_prefix="buy_",
        snapshot_subdir="buy_quality",
        model_subdir="buy_models",
        weights_filename="buy_rule_weights.json",
        label="Agent 8",
    )

