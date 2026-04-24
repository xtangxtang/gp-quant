"""
Adaptive State Machine — 共享配置

定义:
  - 5 个状态枚举
  - 全部 32 个因子列表（按类别分组）
  - 默认阈值配置（从 signal_detector.py 提取）
  - AdaptiveConfig dataclass（支持序列化/反序列化）
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


# ═════════════════════════════════════════════════════════
# 状态定义
# ═════════════════════════════════════════════════════════

class StockState(str, Enum):
    IDLE = "idle"                    # 无信号
    ACCUMULATION = "accumulation"    # 惜售吸筹中
    BREAKOUT = "breakout"            # 分岔突破
    HOLD = "hold"                    # 持仓中（突破后未退出）
    COLLAPSE = "collapse"            # 结构崩塌（退出信号）


# ═════════════════════════════════════════════════════════
# 因子列表 (32 个, 与 feature_engine.py 对齐)
# ═════════════════════════════════════════════════════════

# 熵指标 (9)
ENTROPY_FACTORS = [
    "perm_entropy_s", "perm_entropy_m", "perm_entropy_l",
    "entropy_slope", "entropy_accel",
    "path_irrev_m", "path_irrev_l",
    "turnover_entropy_m", "turnover_entropy_l",
]

# 波动率 (8)
VOLATILITY_FACTORS = [
    "volatility_m", "volatility_l", "vol_compression",
    "bbw", "bbw_pctl",
    "vol_ratio_s", "vol_impulse", "vol_shrink",
]

# 主特征值 (2)
EIGENVALUE_FACTORS = [
    "dom_eig_m", "dom_eig_l",
]

# 量子相干性 (4)
COHERENCE_FACTORS = [
    "coherence_l1", "purity_norm",
    "von_neumann_entropy", "coherence_decay_rate",
]

# 资金流 (7) — 来自 tushare-moneyflow
MONEYFLOW_FACTORS = [
    "mf_big_net", "mf_big_net_ratio",
    "mf_big_cumsum_s", "mf_big_cumsum_m", "mf_big_cumsum_l",
    "mf_sm_proportion", "mf_flow_imbalance",
    "mf_big_momentum", "mf_big_streak",
]

# 价格位置 (2)
PRICE_FACTORS = [
    "breakout_range",
]

# 分钟线微观 (4)
MINUTE_FACTORS = [
    "intraday_perm_entropy", "intraday_path_irrev",
    "intraday_vol_concentration", "intraday_range_ratio",
]

# 全部因子
ALL_FACTORS = (
    ENTROPY_FACTORS
    + VOLATILITY_FACTORS
    + EIGENVALUE_FACTORS
    + COHERENCE_FACTORS
    + MONEYFLOW_FACTORS
    + PRICE_FACTORS
    + MINUTE_FACTORS
)

# 按类别分组的因子（用于 AQ / BQ 评分）
AQ_FACTORS = [
    "perm_entropy_m",      # 置换熵
    "path_irrev_m",        # 路径不可逆性
    "big_net_ratio_ma",    # 大单净额（日线自带）
    "purity_norm",         # 密度矩阵纯度
    "mf_big_streak",       # 大单连续流入
    "mf_flow_imbalance",   # 资金流不平衡
]

BQ_FACTORS = [
    "dom_eig_m",           # 主特征值
    "vol_impulse",         # 量能脉冲
    "perm_entropy_m",      # 有序度
    "path_irrev_m",        # 方向性
    "coherence_decay_rate",  # 退相干速率
    "mf_big_momentum",     # 大单资金动量
    "mf_big_net_ratio",    # 大单净额占比
]

# ═════════════════════════════════════════════════════════
# 默认阈值 (从 signal_detector.py DetectorConfig 提取)
# ═════════════════════════════════════════════════════════

DEFAULT_THRESHOLDS = {
    # ── Accumulation ──
    "perm_entropy_acc": 0.65,
    "path_irrev_acc": 0.05,
    "turnover_entropy_acc": 0.60,
    "accum_min_days": 5,
    "mf_flow_imbalance_min": 0.3,
    "mf_big_streak_min": 3,

    # ── Breakout ──
    "dom_eig_breakout": 0.85,
    "vol_impulse_breakout": 1.8,
    "perm_entropy_breakout_max": 0.75,

    # ── Collapse ──
    "perm_entropy_collapse": 0.90,
    "path_irrev_collapse": 0.01,
    "entropy_accel_collapse": 0.05,
    "vol_exhaustion_ratio": 0.3,
    "purity_collapse_max": 0.3,
    "collapse_need_n": 3,   # 5 个信号中需要 N 个

    # ── Coherence ──
    "purity_accum_min": 0.6,
    "coherence_decay_breakout": -0.005,

    # ── Weekly ──
    "weekly_perm_entropy_max": 0.75,

    # ── Minute ──
    "intraday_entropy_low": 0.70,

    # ── AQ/BQ Composite ──
    "aq_bq_weight": 0.4,   # composite = 0.4*AQ + 0.6*BQ
}

# 阈值搜索范围（基准值 ±20%）
THRESHOLD_SEARCH_RANGES = {
    "perm_entropy_acc": (0.52, 0.78),
    "path_irrev_acc": (0.04, 0.06),
    "accum_min_days": (3, 7),
    "dom_eig_breakout": (0.68, 1.02),
    "vol_impulse_breakout": (1.44, 2.16),
    "perm_entropy_breakout_max": (0.60, 0.90),
    "perm_entropy_collapse": (0.72, 1.00),
    "path_irrev_collapse": (0.005, 0.02),
    "entropy_accel_collapse": (0.03, 0.08),
    "collapse_need_n": (2, 4),
}


# ═════════════════════════════════════════════════════════
# 自适应配置 (可序列化, 持久化到 JSON)
# ═════════════════════════════════════════════════════════

@dataclass
class AdaptiveConfig:
    """
    动态参数配置。Agent 2 更新后持久化到 JSON，
    Agent 3 / Agent 4 读取使用。
    """
    # 因子权重 (初始为均匀权重, Agent 2 更新)
    factor_weights: dict = field(default_factory=lambda: {f: 1.0 for f in ALL_FACTORS})

    # 当前阈值 (初始为 DEFAULT_THRESHOLDS, Agent 2 更新)
    thresholds: dict = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))

    # 学习率 (Agent 4 动态调整)
    learning_rate: float = 0.1

    # 每因子绩效分 (Agent 4 写入)
    factor_scores: dict = field(default_factory=dict)

    # AQ 内部权重 (6 个 AQ 因子, Agent 2 更新)
    aq_weights: dict = field(default_factory=lambda: {
        "perm_entropy_m": 0.25,
        "path_irrev_m": 0.20,
        "big_net_ratio_ma": 0.15,
        "purity_norm": 0.15,
        "mf_big_streak": 0.15,
        "mf_flow_imbalance": 0.10,
    })

    # BQ 内部权重 (7 个 BQ 因子, Agent 2 更新)
    bq_weights: dict = field(default_factory=lambda: {
        "dom_eig_m": 0.20,
        "vol_impulse": 0.20,
        "perm_entropy_m": 0.10,
        "path_irrev_m": 0.10,
        "coherence_decay_rate": 0.15,
        "mf_big_momentum": 0.15,
        "mf_big_net_ratio": 0.10,
    })

    # AQ/BQ 组合权重
    aq_bq_weight: float = 0.4  # composite = aq_bq_weight * AQ + (1 - aq_bq_weight) * BQ

    # 元数据
    version: int = 0
    last_updated: str = ""
    update_count: int = 0

    def save(self, path: str):
        """持久化到 JSON 文件"""
        data = asdict(self)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "AdaptiveConfig":
        """从 JSON 文件加载"""
        if not os.path.exists(path):
            cfg = cls()
            cfg.save(path)
            return cfg
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def clamp_thresholds(self):
        """将阈值约束在搜索范围内，防止漂移"""
        for key, (lo, hi) in THRESHOLD_SEARCH_RANGES.items():
            if key in self.thresholds:
                self.thresholds[key] = max(lo, min(hi, self.thresholds[key]))
        # 整数阈值取整
        for key in ("accum_min_days", "collapse_need_n"):
            if key in self.thresholds:
                self.thresholds[key] = int(round(self.thresholds[key]))

    def smooth_update(self, old: "AdaptiveConfig", alpha: float = 0.2):
        """平滑过渡: new = (1-alpha) * old + alpha * current"""
        for key in self.thresholds:
            if key in old.thresholds:
                old_val = old.thresholds[key]
                cur_val = self.thresholds[key]
                if isinstance(old_val, float):
                    self.thresholds[key] = (1 - alpha) * old_val + alpha * cur_val
                else:
                    pass  # 整数阈值不平滑
