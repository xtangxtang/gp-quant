"""
Holding Position Model — 配置

定义：模型超参、持仓特征名、训练配置、决策阈值
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import json
import os


# ═════════════════════════════════════════════════════════
# 持仓特征 (5 维)
# ═════════════════════════════════════════════════════════

HOLDING_FEATURES = [
    "days_since_entry",       # 入场后第几天
    "unrealized_pnl",         # 当前浮盈/浮亏比例
    "max_pnl_since_entry",    # 入场后最高浮盈
    "drawdown_from_peak",     # 从持仓最高点的回撤
    "entry_price_position",   # 入场价在近 60 天的分位
]
HOLDING_DIM = len(HOLDING_FEATURES)  # 5

# ═════════════════════════════════════════════════════════
# 模型超参
# ═════════════════════════════════════════════════════════

SEQ_LEN = 60          # 因子时序长度
D_MODEL = 128         # Transformer 维度
N_HEADS = 8           # Attention 头数
N_LAYERS = 4          # Transformer 层数
HOLDING_HIDDEN = 32   # 持仓特征投影维度
DROPOUT = 0.1         # Dropout 率
WEIGHT_DECAY = 1e-4   # 权重衰减

# ═════════════════════════════════════════════════════════
# 训练配置
# ═════════════════════════════════════════════════════════

MAX_STOCKS = 500           # 最大股票数
ENTRY_SAMPLE_RATIO = 0.3   # entry_day 采样比例 (30%)
MAX_HOLD_DAYS = 20         # 最大模拟持有天数
FORWARD_DAYS_STAY = 10     # stay_label 未来窗口
FORWARD_DAYS_COLLAPSE = 5  # collapse_label 未来窗口
STAY_DRAWDOWN_THRESHOLD = 0.05   # stay_label: 最大回撤 < 5%
COLLAPSE_DROP_THRESHOLD = 0.05   # collapse_label: 跌幅 > 5%
COLLAPSE_DRAWDOWN_THRESHOLD = 0.10  # collapse_label: 回撤 > 10%
TRAIN_RATIO = 0.8          # 时间 split: 80% 训练, 20% 测试

# Loss 权重
LOSS_STAY_WEIGHT = 1.0
LOSS_COLLAPSE_WEIGHT = 1.0
LOSS_DAYS_WEIGHT = 0.5

# 训练参数
N_EPOCHS = 30
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
TIME_DECAY_LAMBDA = 0.02   # 时间衰减因子

# ═════════════════════════════════════════════════════════
# 决策阈值 (推理时使用)
# ═════════════════════════════════════════════════════════

DECISION_STAY_THRESHOLD = 0.7       # stay_prob > 0.7 → 拿住
DECISION_COLLAPSE_THRESHOLD = 0.3   # collapse_risk > 0.3 → 走人
DECISION_DAYS_THRESHOLD = 3         # expected_days < 3 → 减仓


@dataclass
class HoldingConfig:
    """持仓模型配置 (可持久化)"""
    seq_len: int = SEQ_LEN
    d_model: int = D_MODEL
    n_heads: int = N_HEADS
    n_layers: int = N_LAYERS
    holding_dim: int = HOLDING_DIM
    holding_hidden: int = HOLDING_HIDDEN
    dropout: float = DROPOUT
    weight_decay: float = WEIGHT_DECAY
    n_epochs: int = N_EPOCHS
    batch_size: int = BATCH_SIZE
    lr: float = LEARNING_RATE
    loss_stay_weight: float = LOSS_STAY_WEIGHT
    loss_collapse_weight: float = LOSS_COLLAPSE_WEIGHT
    loss_days_weight: float = LOSS_DAYS_WEIGHT
    factor_names: Optional[list[str]] = None
    standardize_mean: Optional[list[float]] = None
    standardize_std: Optional[list[float]] = None

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "HoldingConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
