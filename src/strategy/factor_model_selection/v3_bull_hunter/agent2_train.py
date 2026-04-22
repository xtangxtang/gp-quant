"""
Bull Hunter v3 — Agent 2: 大牛股分类器训练

v4 重构: 聚焦 A 类大牛股 (200%) 单模型训练, 辅以 100% 模型做排序参考。

目标窗口:
  200% (主模型) → 6 个月 (120 个交易日)
  100% (辅助)   → 2 个月 (40 个交易日)

训练触发:
  1. 定时: 每周 (5 个交易日) 自动重训
  2. 事件: Agent 4 反馈 tuning_directives 时立即重训

训练数据:
  回看 1 年的滚动样本 — 每 sample_interval_days 取因子快照,
  计算对应窗口后的实际涨幅, 标记 label=1 / label=0。

输出:
  feature-cache/bull_models/{model_date}/
    model_200pct.pkl   (主模型)
    model_100pct.pkl   (辅助排序)
    meta.json
  feature-cache/bull_models/latest -> {model_date}/  (符号链接)

版本管理:
  - 保留最近 8 个模型版本 (~2 个月)
  - latest 符号链接始终指向当前生效模型
  - Agent 4 触发的重训写入 {date}_retrain/ 目录
"""

from __future__ import annotations

import glob
import json
import logging
import os
import pickle
from dataclasses import dataclass, field, replace

import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── 目标配置 (v4: 只保留 200pct 主模型 + 100pct 辅助) ──
TARGETS = {
    "200pct": {"threshold": 2.00, "forward_days": 120, "label": "6月涨200%"},
    "100pct": {"threshold": 1.00, "forward_days": 40, "label": "2月涨100%"},
}

# 训练间隔 (交易日数)
TRAIN_INTERVAL_DAYS = 5  # 每周
# 最大保留模型版本数
MAX_MODEL_VERSIONS = 20

DAILY_FACTORS = [
    "perm_entropy_s", "perm_entropy_m", "perm_entropy_l",
    "entropy_slope", "entropy_accel",
    "path_irrev_m", "path_irrev_l",
    "dom_eig_m", "dom_eig_l",
    "turnover_entropy_m", "turnover_entropy_l",
    "volatility_m", "volatility_l",
    "vol_compression", "bbw_pctl",
    "vol_ratio_s", "vol_impulse", "vol_shrink", "breakout_range",
    "mf_big_net", "mf_big_net_ratio",
    "mf_big_cumsum_s", "mf_big_cumsum_m", "mf_big_cumsum_l",
    "mf_sm_proportion", "mf_flow_imbalance",
    "mf_big_momentum", "mf_big_streak",
    "coherence_l1", "purity_norm", "von_neumann_entropy", "coherence_decay_rate",
    # ── v2: factor_advisor 建议新增的 9 个衍生因子 ──
    "momentum_5d", "momentum_20d", "momentum_60d", "price_vs_ma20",
    "vol_price_synergy", "volatility_ratio", "mf_reversal_zscore",
    "atr_20d", "close_vs_high_60d",
    # ── v10: 行业动量/共振因子 (截面聚合) ──
    "industry_mom_5d", "industry_mom_20d", "industry_breadth_5d",
    "industry_rs_20d", "industry_vol_surge",
]


@dataclass
class TrainConfig:
    """训练配置。"""
    lookback_months: int = 12       # 回看多少个月构造训练样本
    sample_interval_days: int = 5   # 采样间隔 (每周一次, 增加样本密度)
    val_ratio: float = 0.2          # 验证集比例 (时间末尾)
    max_scale_pos_weight: float = 5.0   # 上限, 避免过高导致loss震荡
    ewma_halflife_days: int = 90    # P2: EWMA 样本权重半衰期 (天数, 0=不用)
    # LightGBM (v4 回退为默认, V1 回测表现最优)
    n_estimators: int = 800
    max_depth: int = 5
    num_leaves: int = 31
    learning_rate: float = 0.03
    subsample: float = 0.7
    colsample_bytree: float = 0.7
    min_child_samples: int = 50
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    workers: int = 8
    # ── Agent 4 反馈: 因子增删 ──
    drop_factors: list[str] = field(default_factory=list)
    add_factors: list[str] = field(default_factory=list)
    # ── Agent 4 反馈: 模型切换 ──
    model_type: str = "lgbm"   # lgbm | xgboost | random_forest
    # ── 训练触发元数据 ──
    trigger: str = "weekly"     # weekly | agent4_feedback | manual
    trigger_reason: str = ""    # Agent 4 触发时的原因
    # ── 强制重训 (忽略已有模型缓存) ──
    force_retrain: bool = False


def run_training(
    cache_dir: str,
    scan_date: str,
    cfg: TrainConfig | None = None,
    basic_path: str = "",
) -> dict[str, dict]:
    """
    训练大牛股分类器 (200pct 主模型 + 100pct 辅助).

    Args:
        cache_dir: 特征缓存根目录 (含 daily/ 子目录)
        scan_date: 训练日期 (YYYYMMDD)
        cfg: 训练配置
        basic_path: 股票基本信息 (tushare_stock_basic.csv), 用于行业因子

    Returns:
        {target_name: {"model_path": str, "meta": dict}}
    """
    cfg = cfg or TrainConfig()

    # 确定模型目录名
    if cfg.trigger == "agent4_feedback":
        model_subdir = f"{scan_date}_retrain"
    else:
        model_subdir = scan_date

    model_dir = os.path.join(cache_dir, "bull_models", model_subdir)
    meta_path = os.path.join(model_dir, "meta.json")

    # 检查是否已有模型 (force_retrain 时跳过)
    if not cfg.force_retrain and os.path.exists(meta_path):
        logger.info(f"Agent 2: 模型已存在 {model_dir}, 跳过训练")
        with open(meta_path) as f:
            existing_meta = json.load(f)
        results = {}
        for tname in TARGETS:
            mp = os.path.join(model_dir, f"model_{tname}.pkl")
            if os.path.exists(mp):
                results[tname] = {
                    "model_path": mp,
                    "meta": existing_meta.get(tname, {}),
                }
        if results:
            _update_latest_link(cache_dir, model_subdir)
            return results

    # ── 构建训练日历 ──
    calendar = _build_calendar(cache_dir)
    if scan_date not in calendar:
        earlier = [d for d in calendar if d <= scan_date]
        if not earlier:
            logger.error(f"scan_date {scan_date} 不在日历中")
            return {}
        scan_date = earlier[-1]

    scan_idx = calendar.index(scan_date)

    # ── 确定采样日期 ──
    # 回看 lookback_months 个月, 每 sample_interval_days 采样一次
    lookback_days = cfg.lookback_months * 20  # 粗略换算
    max_forward = max(t["forward_days"] for t in TARGETS.values())

    # 采样起点: 至少需要 max_forward 天的前瞻空间
    sample_end_idx = scan_idx - max_forward
    sample_start_idx = max(0, sample_end_idx - lookback_days)

    if sample_end_idx <= sample_start_idx:
        logger.error(f"数据不足: scan_idx={scan_idx}, 需要 {lookback_days + max_forward} 天历史")
        return {}

    sample_dates = []
    for i in range(sample_start_idx, sample_end_idx, cfg.sample_interval_days):
        sample_dates.append(calendar[i])

    logger.info(f"训练采样: {len(sample_dates)} 个日期, "
                f"{calendar[sample_start_idx]} ~ {calendar[sample_end_idx - 1]}")

    # ── 构建训练 panel ──
    panel = _build_training_panel(cache_dir, calendar, sample_dates, scan_idx,
                                  basic_path=basic_path)
    if panel.empty:
        logger.error("训练 panel 为空")
        return {}

    logger.info(f"训练 panel: {len(panel)} 行, {panel['symbol'].nunique()} 只股票")

    # ── 训练每个目标 ──
    os.makedirs(model_dir, exist_ok=True)
    all_meta = {}
    results = {}

    for tname, tspec in TARGETS.items():
        threshold = tspec["threshold"]
        fwd_days = tspec["forward_days"]

        label_col = f"gain_{fwd_days}d"
        if label_col not in panel.columns:
            logger.warning(f"  {tname}: 缺少 {label_col} 列, 跳过")
            continue

        # 构造二分类标签
        labels = (panel[label_col] >= threshold).astype(int)
        n_pos = labels.sum()
        n_total = len(labels)
        pos_rate = n_pos / n_total if n_total > 0 else 0

        if n_pos < 10:
            logger.warning(f"  {tname}: 正样本仅 {n_pos} 个, 跳过")
            continue

        logger.info(f"  {tname}: {n_total} 样本, {n_pos} 正样本 ({pos_rate:.1%})")

        # 训练
        model, meta = _train_one_target(
            panel, labels, tname, threshold, fwd_days, cfg
        )

        if model is not None:
            model_path = os.path.join(model_dir, f"model_{tname}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            all_meta[tname] = meta
            results[tname] = {"model_path": model_path, "meta": meta}
            logger.info(f"  {tname}: 训练完成, val_auc={meta.get('val_auc', 0):.4f}, "
                        f"saved → {model_path}")

    # 保存 meta
    all_meta["_train_info"] = {
        "trigger": cfg.trigger,
        "trigger_reason": cfg.trigger_reason,
        "model_type": cfg.model_type,
        "scan_date": scan_date,
        "model_subdir": model_subdir,
        "drop_factors": cfg.drop_factors,
        "add_factors": cfg.add_factors,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=2)

    # 更新 latest 符号链接
    _update_latest_link(cache_dir, model_subdir)

    # 清理旧版本 (保留最近 MAX_MODEL_VERSIONS 个)
    _cleanup_old_models(cache_dir)

    logger.info(f"Agent 2 完成: {len(results)} 个模型 (trigger={cfg.trigger}), "
                f"存储于 {model_dir}")
    return results


def needs_training(cache_dir: str, scan_date: str) -> bool:
    """
    判断是否需要训练: 距离上次训练 >= TRAIN_INTERVAL_DAYS 个交易日。

    用于 pipeline 判断 weekly 训练触发条件。
    """
    latest_dir = os.path.join(cache_dir, "bull_models", "latest")
    if not os.path.exists(latest_dir):
        return True

    meta_path = os.path.join(latest_dir, "meta.json")
    if not os.path.exists(meta_path):
        return True

    with open(meta_path) as f:
        meta = json.load(f)

    last_scan = meta.get("_train_info", {}).get("scan_date", "")
    if not last_scan:
        return True

    # 用日历计算距离
    calendar = _build_calendar(cache_dir)
    if scan_date not in set(calendar) or last_scan not in set(calendar):
        return True

    last_idx = calendar.index(last_scan)
    cur_idx = calendar.index(scan_date)
    days_since = cur_idx - last_idx

    return days_since >= TRAIN_INTERVAL_DAYS


def get_latest_model_dir(cache_dir: str) -> str | None:
    """获取 latest 模型目录路径, 不存在返回 None。"""
    latest = os.path.join(cache_dir, "bull_models", "latest")
    if os.path.exists(latest):
        return os.path.realpath(latest)
    # fallback: 最新日期目录
    root = os.path.join(cache_dir, "bull_models")
    if not os.path.exists(root):
        return None
    dirs = [d for d in sorted(os.listdir(root)) if d != "latest" and os.path.isdir(os.path.join(root, d))]
    if dirs:
        return os.path.join(root, dirs[-1])
    return None


def _is_model_loadable(model_dir: str) -> bool:
    """检查模型是否可加载 (有 pkl 文件且 model_type 兼容当前环境)。"""
    # 至少需要一个 model pkl 文件
    has_pkl = any(
        f.endswith(".pkl") for f in os.listdir(model_dir)
        if os.path.isfile(os.path.join(model_dir, f))
    )
    if not has_pkl:
        return False
    meta_path = os.path.join(model_dir, "meta.json")
    if not os.path.exists(meta_path):
        return True  # 无 meta 默认可用
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        # 检查任意一个 target 的 model_type
        for key in ("30pct", "100pct", "200pct"):
            if key in meta and isinstance(meta[key], dict):
                mtype = meta[key].get("model_type", "lgbm")
                if mtype == "xgboost":
                    try:
                        import xgboost  # noqa: F401
                    except ImportError:
                        return False
                break
    except Exception:
        pass
    return True


def get_model_for_date(cache_dir: str, scan_date: str) -> str | None:
    """
    获取 scan_date 可用的最新模型 (训练日期 <= scan_date, 避免前瞻偏差)。

    在回测中使用: 逐日选择当时已训练好的模型。
    支持目录名格式: 纯日期 (20250101) 或 Agent 4 重训 (20250103_retrain)。
    自动跳过当前环境无法加载的模型 (如 xgboost 未安装)。
    """
    root = os.path.join(cache_dir, "bull_models")
    if not os.path.exists(root):
        return None

    def _extract_date(dirname: str) -> str | None:
        """从目录名提取日期部分: '20250103' → '20250103', '20250103_retrain' → '20250103'"""
        base = dirname.split("_")[0]
        if base.isdigit() and len(base) == 8:
            return base
        return None

    # 收集所有模型目录及其日期
    candidates = []
    for d in os.listdir(root):
        if d == "latest" or not os.path.isdir(os.path.join(root, d)):
            continue
        date_part = _extract_date(d)
        if date_part and date_part <= scan_date:
            candidates.append((date_part, d))

    if not candidates:
        return None

    # 按日期排序, 同日期下 _retrain 优先于原始模型
    candidates.sort(key=lambda x: (x[0], "_retrain" in x[1]))

    # 从最新往回找, 跳过不可加载的模型 (如 xgboost 未安装)
    for _, dirname in reversed(candidates):
        full_path = os.path.join(root, dirname)
        if _is_model_loadable(full_path):
            return full_path
        else:
            logger.warning(f"  跳过不兼容模型: {dirname} (需要 xgboost)")

    return None


def _update_latest_link(cache_dir: str, model_subdir: str):
    """更新 latest 符号链接指向新模型。"""
    root = os.path.join(cache_dir, "bull_models")
    latest = os.path.join(root, "latest")
    target = os.path.join(root, model_subdir)

    if not os.path.isdir(target):
        return

    # 删除旧链接
    if os.path.islink(latest):
        os.unlink(latest)
    elif os.path.exists(latest):
        import shutil
        shutil.rmtree(latest)

    os.symlink(target, latest)
    logger.info(f"  latest → {model_subdir}")


def _cleanup_old_models(cache_dir: str):
    """保留最近 MAX_MODEL_VERSIONS 个模型, 删除旧的。保护 latest 指向的目录。"""
    root = os.path.join(cache_dir, "bull_models")
    if not os.path.exists(root):
        return

    # 获取 latest 指向的目录名, 绝对不能删除
    latest_link = os.path.join(root, "latest")
    protected = None
    if os.path.islink(latest_link):
        protected = os.path.basename(os.path.realpath(latest_link))

    dirs = sorted([
        d for d in os.listdir(root)
        if d != "latest" and os.path.isdir(os.path.join(root, d))
    ])
    if len(dirs) <= MAX_MODEL_VERSIONS:
        return

    import shutil
    to_remove = dirs[:len(dirs) - MAX_MODEL_VERSIONS]
    for d in to_remove:
        if d == protected:
            continue
        path = os.path.join(root, d)
        shutil.rmtree(path)
        logger.info(f"  清理旧模型: {d}")


def _build_training_panel(
    cache_dir: str,
    calendar: list[str],
    sample_dates: list[str],
    scan_idx: int,
    basic_path: str = "",
) -> pd.DataFrame:
    """
    构建训练 panel: 对每个采样日期, 取全市场因子快照 + 计算前瞻涨幅。

    Returns:
        DataFrame: columns = [symbol, sample_date] + factors + [gain_10d, gain_40d, gain_120d]
    """
    daily_dir = os.path.join(cache_dir, "daily")
    csv_files = sorted(glob.glob(os.path.join(daily_dir, "*.csv")))

    # 预计算日期→日历索引映射 (O(1) 查找)
    date_to_idx = {d: i for i, d in enumerate(calendar)}
    sample_set = set(sample_dates)

    # 预计算每个 forward_days 的目标日期映射
    fwd_target_dates = {}
    for fwd_days in [40, 120]:
        mapping = {}
        for sd in sample_dates:
            sd_idx = date_to_idx.get(sd)
            if sd_idx is None:
                continue
            target_idx = sd_idx + fwd_days
            if target_idx < len(calendar):
                mapping[sd] = calendar[target_idx]
        fwd_target_dates[fwd_days] = mapping

    # 所有需要的日期 (采样日 + 目标日)
    needed_dates = set(sample_dates)
    for mapping in fwd_target_dates.values():
        needed_dates.update(mapping.values())

    # 预加载所有股票 (一次性读取, 计算衍生因子后按 trade_date 索引)
    logger.info(f"加载 {len(csv_files)} 只股票的因子时序...")
    all_panels = []
    n_loaded = 0
    cols_needed = ["trade_date", "close", "high", "low", "amount", "net_mf_amount"] + DAILY_FACTORS

    from .agent1_factor import compute_derived_factors

    for fp in csv_files:
        sym = os.path.basename(fp).replace(".csv", "")
        if sym.startswith("bj"):
            continue
        try:
            df = pd.read_csv(fp)
            if len(df) < 60:
                continue
            df["trade_date"] = df["trade_date"].astype(str)
            # 先计算衍生因子 (需要完整时间序列做 rolling)
            df = compute_derived_factors(df)
            # 然后只保留需要的日期行 (大幅减少内存)
            df = df[df["trade_date"].isin(needed_dates)]
            if df.empty:
                continue
            # 只保留需要的列
            avail_cols = [c for c in cols_needed if c in df.columns]
            df = df[avail_cols].copy()
            df["symbol"] = sym
            df = df.set_index("trade_date")
            all_panels.append(df)
            n_loaded += 1
        except Exception:
            continue

    logger.info(f"加载完成: {n_loaded} 只股票")

    if not all_panels:
        return pd.DataFrame()

    # 合并为大宽表 (trade_date 为 index, symbol 列)
    big = pd.concat(all_panels, axis=0)
    big = big.reset_index()

    # 提取采样日快照
    snapshots = big[big["trade_date"].isin(sample_set)].copy()
    snapshots = snapshots.rename(columns={"trade_date": "sample_date"})

    # 计算前瞻涨幅 — 向量化 merge
    for fwd_days in [40, 120]:
        mapping = fwd_target_dates[fwd_days]
        snapshots[f"_target_date_{fwd_days}"] = snapshots["sample_date"].map(mapping)

        target_closes = big[["trade_date", "symbol", "close"]].rename(
            columns={"trade_date": f"_target_date_{fwd_days}", "close": f"_target_close_{fwd_days}"}
        )
        snapshots = snapshots.merge(
            target_closes,
            on=["symbol", f"_target_date_{fwd_days}"],
            how="left",
        )

        sd_close = snapshots["close"]
        tgt_close = snapshots[f"_target_close_{fwd_days}"]
        valid = (sd_close > 0) & sd_close.notna() & tgt_close.notna() & (tgt_close > 0)
        snapshots[f"gain_{fwd_days}d"] = np.where(
            valid, (tgt_close - sd_close) / sd_close, np.nan
        )
        snapshots.drop(columns=[f"_target_date_{fwd_days}", f"_target_close_{fwd_days}"], inplace=True)

    # 过滤无效行 (close <= 0)
    snapshots = snapshots[snapshots["close"].gt(0) & snapshots["close"].notna()]

    # ── v10: 行业因子 (截面聚合) ──
    from .agent1_factor import _load_basic_info, compute_industry_factors, INDUSTRY_FACTORS
    basic_info = _load_basic_info(basic_path)
    if basic_info:
        snapshots["_industry"] = snapshots["symbol"].map(
            lambda s: basic_info.get(s, {}).get("industry", ""))
        # 按采样日分组计算行业因子 (避免跨日泄漏)
        parts = []
        for sd, grp in snapshots.groupby("sample_date"):
            grp = compute_industry_factors(grp)
            parts.append(grp)
        snapshots = pd.concat(parts, axis=0)
        snapshots.drop(columns=["_industry"], inplace=True, errors="ignore")

    # 整理列
    factor_cols = [f for f in DAILY_FACTORS if f in snapshots.columns]
    keep_cols = ["symbol", "sample_date"] + factor_cols + ["gain_40d", "gain_120d"]
    keep_cols = [c for c in keep_cols if c in snapshots.columns]
    result = snapshots[keep_cols].reset_index(drop=True)

    logger.info(f"训练 panel 构建完成: {len(result)} 行")
    return result


def _train_one_target(
    panel: pd.DataFrame,
    labels: pd.Series,
    target_name: str,
    threshold: float,
    fwd_days: int,
    cfg: TrainConfig,
) -> tuple:
    """训练单个目标的分类模型 (支持 lgbm/xgboost/random_forest)。"""
    # 特征列 — 统一因子剔除 (v4 不再分层)
    base_factors = [f for f in DAILY_FACTORS if f not in cfg.drop_factors]
    # 新增因子需已在 panel 中 (由 Agent 1 衍生因子生成)
    extra_factors = [f for f in cfg.add_factors if f in panel.columns]
    all_factors = base_factors + extra_factors

    feature_cols = [f for f in all_factors if f in panel.columns
                    and panel[f].notna().sum() > 100]
    if not feature_cols:
        return None, {}

    if cfg.drop_factors:
        logger.info(f"    因子剔除 ({target_name}): {cfg.drop_factors}")
    if extra_factors:
        logger.info(f"    因子新增: {extra_factors}")
    logger.info(f"    有效因子: {len(feature_cols)} 个 (model_type={cfg.model_type})")

    # 去除 label 为 NaN 的行
    gain_col = f"gain_{fwd_days}d"
    valid_mask = panel[gain_col].notna()
    X = panel.loc[valid_mask, feature_cols].fillna(0).values
    y = labels[valid_mask].values

    if len(X) < 200:
        return None, {}

    # P2: EWMA 样本权重 (近期样本权重更高)
    sample_weights = None
    if cfg.ewma_halflife_days > 0 and "sample_date" in panel.columns:
        sample_dates = panel.loc[valid_mask, "sample_date"].values
        # 转换为距最新日期的天数差
        unique_dates = sorted(set(sample_dates))
        max_date = unique_dates[-1]
        date_to_idx = {d: i for i, d in enumerate(unique_dates)}
        max_idx = date_to_idx[max_date]
        decay = np.log(2) / cfg.ewma_halflife_days
        raw_weights = np.array([np.exp(-decay * (max_idx - date_to_idx[d]) * cfg.sample_interval_days)
                                for d in sample_dates])
        # 归一化到均值 1.0 (不改变有效样本量的数量级)
        sample_weights = raw_weights / raw_weights.mean()
        logger.info(f"    EWMA 权重: halflife={cfg.ewma_halflife_days}d, "
                     f"min={sample_weights.min():.3f}, max={sample_weights.max():.3f}")

    # 时间分割: 最后 val_ratio 的样本做验证
    n_val = max(int(len(X) * cfg.val_ratio), 50)
    X_train, X_val = X[:-n_val], X[-n_val:]
    y_train, y_val = y[:-n_val], y[-n_val:]
    w_train = sample_weights[:-n_val] if sample_weights is not None else None

    n_pos_train = int(y_train.sum())
    n_neg_train = len(y_train) - n_pos_train
    if n_pos_train < 5:
        return None, {}

    raw_ratio = n_neg_train / max(n_pos_train, 1)
    scale_pos = min(raw_ratio, cfg.max_scale_pos_weight)
    logger.info(f"    scale_pos_weight: {scale_pos:.1f} (raw={raw_ratio:.1f}, cap={cfg.max_scale_pos_weight})")

    # ── 构建模型 (支持 lgbm / xgboost / random_forest) ──
    if cfg.model_type == "xgboost":
        try:
            import xgboost as xgb
        except ImportError:
            logger.warning("    xgboost 未安装, 回退到 lgbm")
            cfg = replace(cfg, model_type="lgbm")

    if cfg.model_type == "xgboost":
        import xgboost as xgb
        model = xgb.XGBClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            min_child_weight=cfg.min_child_samples,
            reg_alpha=cfg.reg_alpha,
            reg_lambda=cfg.reg_lambda,
            scale_pos_weight=scale_pos,
            random_state=42,
            n_jobs=cfg.workers,
            verbosity=0,
            eval_metric="auc",
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            sample_weight=w_train,
            verbose=False,
        )
    elif cfg.model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        # RF 不支持 scale_pos_weight, 用 class_weight
        model = RandomForestClassifier(
            n_estimators=min(cfg.n_estimators, 500),
            max_depth=cfg.max_depth + 3,
            min_samples_leaf=cfg.min_child_samples,
            max_features="sqrt",
            class_weight={0: 1.0, 1: scale_pos},
            random_state=42,
            n_jobs=cfg.workers,
        )
        model.fit(X_train, y_train, sample_weight=w_train)
    else:
        # 默认: LightGBM
        model = lgb.LGBMClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            num_leaves=cfg.num_leaves,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            min_child_samples=cfg.min_child_samples,
            reg_alpha=cfg.reg_alpha,
            reg_lambda=cfg.reg_lambda,
            scale_pos_weight=scale_pos,
            random_state=42,
            n_jobs=cfg.workers,
            verbose=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            sample_weight=w_train,
            callbacks=[
                lgb.log_evaluation(period=200),
            ],
        )

    # 评估 — 用最优 F1 阈值而非固定 0.5
    from sklearn.metrics import roc_auc_score, precision_score, recall_score
    val_proba = model.predict_proba(X_val)[:, 1]

    try:
        val_auc = roc_auc_score(y_val, val_proba)
    except ValueError:
        val_auc = 0.5

    # 搜索最优阈值 (正样本稀少, 固定 0.5 不合理)
    best_f1, best_th = 0, 0.5
    for th in np.arange(0.01, 0.50, 0.01):
        pred_th = (val_proba > th).astype(int)
        tp = ((pred_th == 1) & (y_val == 1)).sum()
        fp = ((pred_th == 1) & (y_val == 0)).sum()
        fn = ((pred_th == 0) & (y_val == 1)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best_f1:
            best_f1, best_th = f1, th

    val_pred = (val_proba > best_th).astype(int)
    val_precision = precision_score(y_val, val_pred, zero_division=0)
    val_recall = recall_score(y_val, val_pred, zero_division=0)

    # 特征重要性 (不同模型取法不同)
    if hasattr(model, "feature_importances_"):
        importance = dict(zip(feature_cols, model.feature_importances_.tolist()))
    else:
        importance = {f: 0 for f in feature_cols}
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]

    # best_iteration (lgbm/xgb 有, RF 没有)
    best_iter = getattr(model, "best_iteration_", cfg.n_estimators)

    meta = {
        "target": target_name,
        "threshold": threshold,
        "forward_days": fwd_days,
        "model_type": cfg.model_type,
        "n_features": len(feature_cols),
        "drop_factors": cfg.drop_factors,
        "add_factors": [f for f in cfg.add_factors if f in feature_cols],
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_pos_train": n_pos_train,
        "pos_rate_train": round(n_pos_train / len(y_train), 4),
        "n_pos_val": int(y_val.sum()),
        "pos_rate_val": round(int(y_val.sum()) / len(y_val), 4),
        "scale_pos_weight": round(scale_pos, 2),
        "val_auc": round(val_auc, 4),
        "val_precision": round(val_precision, 4),
        "val_recall": round(val_recall, 4),
        "best_threshold": round(best_th, 3),
        "best_f1": round(best_f1, 4),
        "best_iteration": best_iter,
        "feature_cols": feature_cols,
        "top_features": [{"name": n, "importance": v} for n, v in top_features],
    }

    return model, meta


def _build_calendar(cache_dir: str) -> list[str]:
    """从 daily cache 构建交易日历。"""
    daily_dir = os.path.join(cache_dir, "daily")
    csvs = sorted(glob.glob(os.path.join(daily_dir, "*.csv")))[:50]
    all_dates: set[str] = set()
    for fp in csvs:
        try:
            df = pd.read_csv(fp, usecols=["trade_date"])
            all_dates.update(df["trade_date"].astype(str).tolist())
        except Exception:
            continue
    return sorted(all_dates)
