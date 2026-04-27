"""
Attention Learner — 动态因子权重提取 + 多任务预测

模型架构 (Phase 1):
  - 输入: (seq_len=60, n_factors) 每只股票的因子时序
  - 嵌入: Linear(n_factors → d_model=128) + 可学习位置编码
  - Transformer: 4 层 encoder, 8 attention heads, d_model=128
  - 多任务头:
      * 回归: 预测未来收益率 (MSE)
      * 分类: 预测涨/跌 (CrossEntropy)
      * 分位数回归: 9 个分位点 (10/20/...90%) (Quantile Loss)
  - 输出: 因子重要性分数 = mean attention over sequence → softmax

训练: 多任务 loss (MSE + CE + Quantile + IC Loss), 时间衰减
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── Lazy import torch ───
_torch = None


def _import_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch

try:
    import torch as _torch_mod
    _torch = _torch_mod
except ImportError:
    pass


# ═════════════════════════════════════════════════════════
# 纯 numpy 前置: 数据准备
# ═════════════════════════════════════════════════════════

FACTOR_COLUMNS = [
    # 熵 & 不可逆性 (8)
    "perm_entropy_s", "perm_entropy_m", "perm_entropy_l",
    "path_irrev_m", "path_irrev_l",
    "entropy_accel",
    "turnover_entropy_m", "turnover_entropy_l",
    # 波动率 (8)
    "volatility_m", "volatility_l",
    "vol_compression", "bbw", "bbw_pctl",
    "vol_ratio_s", "vol_shrink", "vol_impulse",
    # 特征值 (2)
    "dom_eig_m", "dom_eig_l",
    # 相干性 (2)
    "coherence_l1", "coherence_decay_rate",
    # 资金流 (12)
    "mf_big_net", "mf_big_net_ratio",
    "mf_big_cumsum_s", "mf_big_cumsum_m", "mf_big_cumsum_l",
    "mf_sm_proportion", "mf_flow_imbalance",
    "mf_big_momentum", "mf_big_streak",
    "mf_cumsum_s", "mf_cumsum_m",
    "mf_impulse", "net_mf_vol",
    # 价格 & 量价 (3)
    "breakout_range", "volume_ratio", "purity",
    # 估值/市值 (6)
    "pe", "pe_ttm", "pb", "ps",
    "dv_ttm", "circ_mv",
    # 周线独有特征 (5)
    "w_pe_ttm_pctl", "w_pb_pctl",
    "w_weekly_big_net_cumsum",
    "w_weekly_turnover_shrink", "w_weekly_turnover_ma4",
    # 其他 (5)
    "turnover_rate_f", "big_net_ratio_ma",
    "buy_sm_amount", "buy_sm_vol",
    "entropy_accel",
]

# 分位数回归的分位点
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N_QUANTILES = len(QUANTILE_LEVELS)

# 因子分组: 7 个语义组，每组独立投影后 concat → Linear → d_model
# 基于 Vision 文档 v2 第 2 项 + 实际 65 个因子名
FACTOR_GROUPS = {
    # 熵 & 不可逆性 (8)
    "entropy": [
        "perm_entropy_s", "perm_entropy_m", "perm_entropy_l",
        "entropy_accel",
        "path_irrev_m", "path_irrev_l",
        "turnover_entropy_m", "turnover_entropy_l",
    ],
    # 波动率 (8)
    "volatility": [
        "volatility_m", "volatility_l",
        "vol_compression", "bbw", "bbw_pctl",
        "vol_ratio_s", "vol_shrink", "vol_impulse",
    ],
    # 特征值 (2)
    "eigenvalue": [
        "dom_eig_m", "dom_eig_l",
    ],
    # 相干性 (2)
    "coherence": [
        "coherence_l1", "coherence_decay_rate",
    ],
    # 资金流 — 资金流向 (12)
    "money_flow": [
        "mf_big_net", "mf_big_net_ratio",
        "mf_big_cumsum_s", "mf_big_cumsum_m", "mf_big_cumsum_l",
        "mf_sm_proportion", "mf_flow_imbalance",
        "mf_big_momentum", "mf_big_streak",
        "mf_cumsum_s", "mf_cumsum_m",
        "mf_impulse", "net_mf_vol",
    ],
    # 资金流 — 订单流 (2)
    "order_flow": [
        "buy_sm_amount", "buy_sm_vol",
    ],
    # 价格 & 量价 (3)
    "price": [
        "breakout_range", "volume_ratio", "purity",
    ],
    # 估值 & 市值 (6)
    "valuation": [
        "pe", "pe_ttm", "pb", "ps",
        "dv_ttm", "circ_mv",
    ],
    # 周线独有特征 (5)
    "weekly": [
        "w_pe_ttm_pctl", "w_pb_pctl",
        "w_weekly_big_net_cumsum",
        "w_weekly_turnover_shrink", "w_weekly_turnover_ma4",
    ],
    # 其他 (5)
    "meta": [
        "turnover_rate_f", "big_net_ratio_ma",
    ],
}


def _standardize_factors(values: np.ndarray) -> np.ndarray:
    """按因子标准化: (x - mean) / std，clip 到 [-5, 5]"""
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    std[std < 1e-8] = 1.0
    result = (values - mean) / std
    return np.clip(result, -5.0, 5.0), mean, std


# ═════════════════════════════════════════════════════════
# PyTorch 模型定义
# ═════════════════════════════════════════════════════════

class FactorAttentionModel(_torch.nn.Module if _torch is not None else object):
    """Transformer 因子注意力模型 (Phase 1 升级)。

    多任务学习: 回归 + 分类 + 分位数回归
    """

    def __init__(
        self,
        n_factors: int = 32,
        seq_len: int = 60,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        n_quantiles: int = N_QUANTILES,
        factor_names: list[str] | None = None,
    ):
        torch = _import_torch()
        super().__init__()

        self.n_factors = n_factors
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.n_quantiles = n_quantiles
        self._factor_names = list(factor_names) if factor_names is not None else []

        # 因子嵌入: 简单 Linear(n_factors → d_model)
        self.embedding = torch.nn.Linear(n_factors, d_model)

        # 可学习位置编码
        self.pos_encoding = torch.nn.Parameter(
            torch.randn(1, seq_len, d_model) * 0.02
        )

        # Transformer encoder 层
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
        )

        # 可学习 summary token
        self.summary_token = torch.nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # 回归头: 预测未来收益率
        self.regression_head = torch.nn.Sequential(
            torch.nn.Linear(d_model, 32),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(32, 1),
        )

        # 分类头: 预测涨/跌
        self.classification_head = torch.nn.Sequential(
            torch.nn.Linear(d_model, 32),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(32, 2),
        )

        # 分位数回归头: 基准 + (n_quantiles-1) 个增量
        self.quantile_base_head = torch.nn.Sequential(
            torch.nn.Linear(d_model, 64),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, 1),  # 最低分位点 (q10)
        )
        self.quantile_delta_head = torch.nn.Sequential(
            torch.nn.Linear(d_model, 64),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, n_quantiles - 1),  # 8 个正增量
        )

        # 因子重要性投影
        self.factor_proj = torch.nn.Linear(d_model, n_factors)

    def _embed(self, x: np.ndarray) -> "torch.Tensor":
        """因子嵌入: (batch, seq_len, n_factors) → (batch, seq_len, d_model)"""
        import torch
        device = next(self.embedding.parameters()).device
        x_tensor = torch.from_numpy(x.astype(np.float32)).to(device)
        return self.embedding(x_tensor)

    def forward(self, x: np.ndarray, training: bool = False) -> dict:
        """前向传播。

        Args:
            x: (batch, seq_len, n_factors) 标准化后的因子值
            training: 是否启用 dropout

        Returns:
            dict with keys: regression, classification, quantiles, attention_weights, factor_importance
        """
        import torch
        x_tensor = torch.from_numpy(x.astype(np.float32))

        # 嵌入 + 位置编码
        h = self._embed(x) + self.pos_encoding

        # 拼接 summary token
        batch_size = h.shape[0]
        summary = self.summary_token.expand(batch_size, -1, -1)
        h = torch.cat([summary, h], dim=1)  # (batch, seq_len+1, d_model)

        # Transformer
        encoder_output = self.transformer(h)  # (batch, seq_len+1, d_model)

        # summary token 的输出 (位置 0)
        summary_out = encoder_output[:, 0, :]  # (batch, d_model)

        # 回归 + 分类 + 分位数 (单调性约束)
        regression = self.regression_head(summary_out).squeeze(-1)
        classification = self.classification_head(summary_out)
        quantiles = self._monotone_quantiles(summary_out)

        # 因子重要性
        factor_importance = self.factor_proj(summary_out)  # (batch, n_factors)

        return {
            "regression": regression.detach().cpu().numpy(),
            "classification": classification.detach().cpu().numpy(),
            "quantiles": quantiles.detach().cpu().numpy(),
            "factor_importance": factor_importance.detach().cpu().numpy(),
        }

    def forward_tensors(self, x: np.ndarray) -> dict:
        """前向传播，返回原始 tensor（用于训练时的 loss 计算）"""
        import torch
        h = self._embed(x) + self.pos_encoding

        batch_size = h.shape[0]
        summary = self.summary_token.expand(batch_size, -1, -1)
        h = torch.cat([summary, h], dim=1)

        encoder_output = self.transformer(h)
        summary_out = encoder_output[:, 0, :]

        regression = self.regression_head(summary_out).squeeze(-1)
        classification = self.classification_head(summary_out)
        quantiles = self._monotone_quantiles(summary_out)
        factor_importance = self.factor_proj(summary_out)

        return {
            "regression": regression,
            "classification": classification,
            "quantiles": quantiles,
            "factor_importance": factor_importance,
        }

    def _monotone_quantiles(self, summary_out) -> "torch.Tensor":
        """强制分位数单调性: base + cumsum(softplus(deltas))"""
        import torch
        base = self.quantile_base_head(summary_out)  # (batch, 1)
        deltas = torch.nn.functional.softplus(self.quantile_delta_head(summary_out))  # (batch, n_q-1), 保证 >= 0
        cumulative = torch.cumsum(deltas, dim=1)  # (batch, n_q-1)
        return torch.cat([base, base + cumulative], dim=1)  # (batch, n_quantiles)

    def _temporal_encode(self, x_tensor) -> "torch.Tensor":
        """时序编码: (N, seq_len, n_factors) → (N, d_model) summary tokens"""
        import torch
        h = self.embedding(x_tensor) + self.pos_encoding
        batch_size = h.shape[0]
        summary = self.summary_token.expand(batch_size, -1, -1)
        h = torch.cat([summary, h], dim=1)
        encoder_output = self.transformer(h)
        return encoder_output[:, 0, :]  # (N, d_model)

    def get_attention_weights(self, x: np.ndarray) -> np.ndarray:
        """提取 attention weights 作为因子重要性。"""
        import torch
        x_tensor = torch.from_numpy(x.astype(np.float32))
        h = self._embed(x) + self.pos_encoding

        batch_size = h.shape[0]
        summary = self.summary_token.expand(batch_size, -1, -1)
        h = torch.cat([summary, h], dim=1)

        # 取最后一层的 self-attention
        last_layer = self.transformer.layers[-1]
        h_norm = last_layer.norm1(h)

        # 计算 Q, K
        W_qkv = last_layer.self_attn.in_proj_weight
        Q = (W_qkv[:self.d_model] @ h_norm[0].T).T  # (seq_len+1, d_model)
        K = (W_qkv[self.d_model:2*self.d_model] @ h_norm[0].T).T

        # summary token (index 0) 对所有位置的 attention
        q_summary = Q[0:1, :]
        attn_scores = torch.matmul(q_summary, K.T) / (self.d_model ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # 取对原始序列的 attention (去掉 summary token)
        seq_attn = attn_weights[0, 1:]

        # attention 加权平均原始因子
        x_original = x_tensor
        weights = seq_attn.unsqueeze(-1).unsqueeze(0).expand(batch_size, -1, -1)
        weighted_factors = (x_original * weights).sum(dim=1)

        return weighted_factors.detach().cpu().numpy()

    def train_mode(self):
        self.train()

    def eval_mode(self):
        self.eval()

    def save(self, path: str, standardize_mean: np.ndarray = None, standardize_std: np.ndarray = None):
        """保存模型 (含标准化参数)"""
        import torch
        state = {
            "model_state_dict": self.state_dict(),
            "n_factors": self.n_factors,
            "seq_len": self.seq_len,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "n_quantiles": self.n_quantiles,
            "factor_names": self._factor_names,
        }
        if standardize_mean is not None:
            state["standardize_mean"] = standardize_mean
        if standardize_std is not None:
            state["standardize_std"] = standardize_std
        torch.save(state, path)
        logger.info(f"Attention model saved to {path}")

    def load(self, path: str):
        """加载模型"""
        import torch
        if not os.path.exists(path):
            logger.warning(f"Model file not found: {path}, using random init")
            return False
        state = torch.load(path, weights_only=False, map_location="cpu")
        sd = state.get("model_state_dict", {})
        try:
            self.load_state_dict(sd, strict=False)
        except Exception as e:
            logger.warning(f"Partial state dict load: {e}")
            self.load_state_dict(sd, strict=False)

        logger.info(f"Attention model loaded from {path}")
        return True


# ═════════════════════════════════════════════════════════
# PatchTST 价格编码器 (Feature 3)
# ═════════════════════════════════════════════════════════

class PatchTSTEncoder(_torch.nn.Module if _torch is not None else object):
    """PatchTST 风格的 OHLCV 编码器。

    将原始价格序列分割为 patches, 用 Transformer 编码。
    输入: (batch, seq_len, 5) → (batch, d_patch)
    5 个通道: open, high, low, close, volume (标准化后)
    """

    def __init__(self, seq_len=60, patch_len=10, stride=5, d_model=128, dropout=0.1):
        torch = _import_torch()
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (seq_len - patch_len) // stride + 1

        # Patch embedding: 每个 patch 展平后投影
        self.patch_embed = torch.nn.Linear(patch_len * 5, d_model)
        self.pos_embed = torch.nn.Parameter(torch.randn(1, self.n_patches + 1, d_model) * 0.02)
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x):
        """x: (batch, seq_len, 5) → (batch, d_model)"""
        torch = _import_torch()
        batch = x.shape[0]

        # 构建 patches: (batch, n_patches, patch_len * 5)
        patches = []
        for i in range(self.n_patches):
            start = i * self.stride
            end = start + self.patch_len
            patch = x[:, start:end, :].reshape(batch, -1)
            patches.append(patch)
        x_patches = torch.stack(patches, dim=1)  # (batch, n_patches, patch_len*5)

        h = self.patch_embed(x_patches)  # (batch, n_patches, d_model)
        cls = self.cls_token.expand(batch, -1, -1)
        h = torch.cat([cls, h], dim=1) + self.pos_embed
        h = self.transformer(h)
        return h[:, 0, :]  # CLS token output


# ═════════════════════════════════════════════════════════
# Multi-Scale Attention Model (Feature 2)
# ═════════════════════════════════════════════════════════

class MultiScaleAttentionModel(_torch.nn.Module if _torch is not None else object):
    """双通道注意力模型: 日线 + 周线。

    日线通道: (batch, 60, n_factors) → Transformer
    周线通道: (batch, 12, n_factors) → Transformer
    输出: concat(summary_daily, summary_weekly) → 预测头
    """

    def __init__(
        self,
        n_factors: int = 32,
        daily_seq_len: int = 60,
        weekly_seq_len: int = 12,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        n_quantiles: int = N_QUANTILES,
        factor_names: list[str] | None = None,
    ):
        torch = _import_torch()
        super().__init__()

        self.n_factors = n_factors
        self.daily_seq_len = daily_seq_len
        self.weekly_seq_len = weekly_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.n_quantiles = n_quantiles
        self._factor_names = list(factor_names) if factor_names is not None else []

        # 共享嵌入权重 (因子相同)
        self.embedding = torch.nn.Linear(n_factors, d_model)

        # 日线 Transformer
        self.daily_pos = torch.nn.Parameter(torch.randn(1, daily_seq_len, d_model) * 0.02)
        self.daily_summary = torch.nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        daily_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.daily_transformer = torch.nn.TransformerEncoder(daily_layer, num_layers=n_layers)

        # 周线 Transformer
        self.weekly_pos = torch.nn.Parameter(torch.randn(1, weekly_seq_len, d_model) * 0.02)
        self.weekly_summary = torch.nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        weekly_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.weekly_transformer = torch.nn.TransformerEncoder(weekly_layer, num_layers=n_layers)

        # 融合层: concat(daily_summary, weekly_summary) → d_model
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(d_model * 2, d_model),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )

        # 预测头 (与 FactorAttentionModel 相同)
        self.regression_head = torch.nn.Sequential(
            torch.nn.Linear(d_model, 32), torch.nn.GELU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(32, 1),
        )
        self.classification_head = torch.nn.Sequential(
            torch.nn.Linear(d_model, 32), torch.nn.GELU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(32, 2),
        )
        self.quantile_base_head = torch.nn.Sequential(
            torch.nn.Linear(d_model, 64), torch.nn.GELU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(64, 1),
        )
        self.quantile_delta_head = torch.nn.Sequential(
            torch.nn.Linear(d_model, 64), torch.nn.GELU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(64, n_quantiles - 1),
        )
        self.factor_proj = torch.nn.Linear(d_model, n_factors)

    def _encode_daily(self, x_daily) -> "torch.Tensor":
        torch = _import_torch()
        if isinstance(x_daily, np.ndarray):
            x_daily = torch.from_numpy(x_daily.astype(np.float32)).to(self.embedding.weight.device)
        h = self.embedding(x_daily) + self.daily_pos
        batch = h.shape[0]
        summary = self.daily_summary.expand(batch, -1, -1)
        h = torch.cat([summary, h], dim=1)
        out = self.daily_transformer(h)
        return out[:, 0, :]

    def _encode_weekly(self, x_weekly) -> "torch.Tensor":
        torch = _import_torch()
        if isinstance(x_weekly, np.ndarray):
            x_weekly = torch.from_numpy(x_weekly.astype(np.float32)).to(self.embedding.weight.device)
        h = self.embedding(x_weekly) + self.weekly_pos
        batch = h.shape[0]
        summary = self.weekly_summary.expand(batch, -1, -1)
        h = torch.cat([summary, h], dim=1)
        out = self.weekly_transformer(h)
        return out[:, 0, :]

    def _monotone_quantiles(self, summary_out) -> "torch.Tensor":
        torch = _import_torch()
        base = self.quantile_base_head(summary_out)
        deltas = torch.nn.functional.softplus(self.quantile_delta_head(summary_out))
        cumulative = torch.cumsum(deltas, dim=1)
        return torch.cat([base, base + cumulative], dim=1)

    def forward(self, x_daily: np.ndarray, x_weekly: np.ndarray, training: bool = False) -> dict:
        torch = _import_torch()
        device = self.embedding.weight.device
        x_d = torch.from_numpy(x_daily.astype(np.float32)).to(device)
        x_w = torch.from_numpy(x_weekly.astype(np.float32)).to(device)

        daily_sum = self._encode_daily(x_d)
        weekly_sum = self._encode_weekly(x_w)
        fused = self.fusion(torch.cat([daily_sum, weekly_sum], dim=1))

        regression = self.regression_head(fused).squeeze(-1)
        classification = self.classification_head(fused)
        quantiles = self._monotone_quantiles(fused)
        factor_importance = self.factor_proj(fused)

        return {
            "regression": regression.detach().cpu().numpy(),
            "classification": classification.detach().cpu().numpy(),
            "quantiles": quantiles.detach().cpu().numpy(),
            "factor_importance": factor_importance.detach().cpu().numpy(),
        }

    def forward_tensors(self, x_daily: np.ndarray, x_weekly: np.ndarray) -> dict:
        torch = _import_torch()
        device = self.embedding.weight.device
        x_d = torch.from_numpy(x_daily.astype(np.float32)).to(device)
        x_w = torch.from_numpy(x_weekly.astype(np.float32)).to(device)

        daily_sum = self._encode_daily(x_d)
        weekly_sum = self._encode_weekly(x_w)
        fused = self.fusion(torch.cat([daily_sum, weekly_sum], dim=1))

        return {
            "regression": self.regression_head(fused).squeeze(-1),
            "classification": self.classification_head(fused),
            "quantiles": self._monotone_quantiles(fused),
            "factor_importance": self.factor_proj(fused),
        }

    def train_mode(self):
        self.train()

    def eval_mode(self):
        self.eval()

    def parameters(self, recurse=True):
        return list(super().parameters(recurse=recurse))

    def save(self, path: str, standardize_mean: np.ndarray = None,
             standardize_std: np.ndarray = None,
             weekly_mean: np.ndarray = None, weekly_std: np.ndarray = None):
        torch = _import_torch()
        state = {
            "model_state_dict": self.state_dict(),
            "n_factors": self.n_factors,
            "daily_seq_len": self.daily_seq_len,
            "weekly_seq_len": self.weekly_seq_len,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "n_quantiles": self.n_quantiles,
            "factor_names": self._factor_names,
            "model_type": "multi_scale",
        }
        if standardize_mean is not None:
            state["standardize_mean"] = standardize_mean
            state["standardize_std"] = standardize_std
        if weekly_mean is not None:
            state["weekly_mean"] = weekly_mean
            state["weekly_std"] = weekly_std
        torch.save(state, path)
        logger.info(f"Multi-scale model saved to {path}")

    def load(self, path: str):
        torch = _import_torch()
        if not os.path.exists(path):
            logger.warning(f"Model file not found: {path}")
            return False
        state = torch.load(path, weights_only=False, map_location="cpu")
        sd = state.get("model_state_dict", {})
        try:
            self.load_state_dict(sd, strict=False)
        except Exception as e:
            logger.warning(f"Partial state dict load: {e}")
            self.load_state_dict(sd, strict=False)
        logger.info(f"Multi-scale model loaded from {path}")
        return True


# ═════════════════════════════════════════════════════════
# 分位数损失
# ═════════════════════════════════════════════════════════

def _quantile_loss(pred: "torch.Tensor", target: "torch.Tensor", tau: float) -> "torch.Tensor":
    """
    分位数损失: pinball loss
    L(y, q) = max(tau * (y - q), (tau - 1) * (y - q))
    """
    import torch
    diff = target - pred
    return torch.mean(torch.max(tau * diff, (tau - 1) * diff))


# ═════════════════════════════════════════════════════════
# 可微 Spearman 相关 (IC Loss)
# ═════════════════════════════════════════════════════════

def _rank_data(x: "torch.Tensor") -> "torch.Tensor":
    """可微近似排序 (用 sigmoid 平滑)"""
    import torch
    # 用 pairwise 比较做可微排序
    n = x.shape[0]
    # 对每个元素，计算它比其他元素大的比例
    x_expanded = x.unsqueeze(1)  # (n, 1)
    comparisons = torch.sigmoid(10.0 * (x_expanded - x.unsqueeze(0)))  # (n, n)
    ranks = comparisons.sum(dim=1)  # (n,)
    return ranks


def _spearman_correlation(pred: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
    """可微 Spearman 相关 (负值 = loss)"""
    import torch
    pred_ranks = _rank_data(pred)
    target_ranks = _rank_data(target)

    pred_centered = pred_ranks - pred_ranks.mean()
    target_centered = target_ranks - target_ranks.mean()

    cov = (pred_centered * target_centered).mean()
    std_pred = torch.sqrt((pred_centered ** 2).mean() + 1e-8)
    std_target = torch.sqrt((target_centered ** 2).mean() + 1e-8)

    return cov / (std_pred * std_target)


# ═════════════════════════════════════════════════════════
# 训练配置
# ═════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    """训练配置 (Phase 2)"""
    seq_len: int = 60           # 序列长度 (交易日)
    d_model: int = 128          # 模型维度
    n_heads: int = 8            # attention heads
    n_layers: int = 4           # transformer 层数
    dropout: float = 0.1        # dropout (Phase 1)
    learning_rate: float = 1e-3
    n_epochs: int = 30          # 训练轮数 (Phase 2.1)
    batch_size: int = 256
    forward_days: int = 10      # 预测未来 N 天收益率
    time_decay_lambda: float = 0.02  # 时间衰减系数
    # Loss 权重
    regression_weight: float = 0.15
    classification_weight: float = 0.15
    quantile_weight: float = 0.4     # 分位数回归是主要目标
    ic_weight: float = 0.3           # IC 最大化
    val_split: float = 0.2      # 验证集比例
    early_stop_patience: int = 5
    device: str = "cpu"         # 默认 CPU
    # Phase 2: 正则化
    weight_decay: float = 1e-4      # Phase 2.1: back to Phase 1
    # 因子名列表 (用于分组嵌入)
    factor_names: list[str] | None = None


class AttentionTrainer:
    """训练 Attention Learner 模型 (Phase 1)。

    多任务 loss: 回归 + 分类 + 分位数回归 + IC Loss
    """

    def __init__(self, config: Optional[TrainConfig] = None):
        self.config = config or TrainConfig()
        self.model: Optional[FactorAttentionModel] = None
        self.train_history: list[dict] = []

    def train(
        self,
        X: np.ndarray,
        y_reg: np.ndarray,
        y_cls: np.ndarray,
        sample_weights: np.ndarray,
        y_state: Optional[np.ndarray] = None,  # 向后兼容, 忽略
        save_path: Optional[str] = None,
        standardize_mean: Optional[np.ndarray] = None,
        standardize_std: Optional[np.ndarray] = None,
    ) -> FactorAttentionModel:
        """
        训练模型 (Phase 1: 回归 + 分类 + 分位数 + IC)。

        Args:
            X: (n_samples, seq_len, n_factors)
            y_reg: (n_samples,) 未来收益率
            y_cls: (n_samples,) 涨跌标签
            sample_weights: (n_samples,) 时间衰减权重
            y_state: 忽略 (向后兼容)
            save_path: 模型保存路径

        Returns:
            训练好的模型
        """
        torch = _import_torch()
        nn = torch.nn
        F = torch.nn.functional

        config = self.config
        device = config.device

        # 初始化模型
        self.model = FactorAttentionModel(
            n_factors=X.shape[-1],
            seq_len=config.seq_len,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            dropout=config.dropout,
            factor_names=config.factor_names,
        )
        self.model.to(device)
        self.model.train_mode()

        # 优化器
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.n_epochs, eta_min=1e-5
        )

        # 划分训练集/验证集
        n = len(X)
        n_val = int(n * config.val_split)
        indices = np.random.permutation(n)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        X_train = X[train_idx]
        y_reg_train = y_reg[train_idx]
        y_cls_train = y_cls[train_idx]
        w_train = sample_weights[train_idx]

        X_val = X[val_idx]
        y_reg_val = y_reg[val_idx]
        y_cls_val = y_cls[val_idx]
        w_val = sample_weights[val_idx]

        logger.info(f"Training: {len(train_idx)} train, {len(val_idx)} val samples")
        logger.info(f"  Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"  Loss weights: reg={config.regression_weight}, "
                     f"cls={config.classification_weight}, "
                     f"quantile={config.quantile_weight}, "
                     f"ic={config.ic_weight}")

        mse_loss = nn.MSELoss(reduction="none")
        ce_loss = nn.CrossEntropyLoss(reduction="none", label_smoothing=0.0)

        best_val_loss = float("inf")
        patience_counter = 0
        batch_size = config.batch_size

        for epoch in range(config.n_epochs):
            epoch_start = time.time()

            # 训练
            total_loss = 0.0
            n_batches = 0
            total_reg_loss = 0.0
            total_cls_loss = 0.0
            total_quantile_loss = 0.0
            total_ic_loss_val = 0.0

            perm = np.random.permutation(len(X_train))
            for start in range(0, len(X_train), batch_size):
                batch_idx = perm[start:start + batch_size]
                if len(batch_idx) < 8:
                    continue

                x_batch = X_train[batch_idx]
                y_reg_batch = y_reg_train[batch_idx]
                y_cls_batch = y_cls_train[batch_idx]
                w_batch = w_train[batch_idx]

                # 前向传播
                output = self.model.forward_tensors(x_batch)

                w_tensor = torch.from_numpy(w_batch).to(device)

                # 回归 loss
                pred_reg = output["regression"]
                reg_loss = (mse_loss(pred_reg, torch.from_numpy(y_reg_batch).float().to(device)) * w_tensor).mean()

                # 分类 loss
                pred_cls = output["classification"]
                cls_loss = (ce_loss(pred_cls, torch.from_numpy(y_cls_batch).to(device)) * w_tensor).mean()

                # 分位数回归 loss
                pred_quantiles = output["quantiles"]  # (batch, n_quantiles)
                y_reg_tensor = torch.from_numpy(y_reg_batch).float().to(device)
                quantile_loss_val = torch.tensor(0.0, device=device)
                for q_idx, tau in enumerate(QUANTILE_LEVELS):
                    q_pred = pred_quantiles[:, q_idx]
                    quantile_loss_val = quantile_loss_val + _quantile_loss(q_pred, y_reg_tensor, tau)
                quantile_loss_val = quantile_loss_val / N_QUANTILES

                # IC loss: 最大化 batch 内预测与真实收益的 Spearman 相关
                ic_loss_val = torch.tensor(0.0, device=device)
                if len(batch_idx) > 10:  # 需要足够样本计算相关
                    spearman_corr = _spearman_correlation(pred_reg, y_reg_tensor.to(device))
                    ic_loss_val = -spearman_corr  # 负相关 = loss

                # 总 loss
                loss = (config.regression_weight * reg_loss +
                        config.classification_weight * cls_loss +
                        config.quantile_weight * quantile_loss_val +
                        config.ic_weight * ic_loss_val)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1
                total_reg_loss += reg_loss.item()
                total_cls_loss += cls_loss.item()
                total_quantile_loss += quantile_loss_val.item()
                total_ic_loss_val += ic_loss_val.item()

            avg_train_loss = total_loss / max(n_batches, 1)

            # 验证
            self.model.eval_mode()
            with torch.no_grad():
                val_output = self.model.forward_tensors(X_val)
                val_pred_reg = val_output["regression"]
                val_pred_cls = val_output["classification"]
                val_w = torch.from_numpy(w_val).to(device)
                y_reg_tensor_val = torch.from_numpy(y_reg_val).float().to(device)

                val_reg_loss = (mse_loss(val_pred_reg, y_reg_tensor_val) * val_w).mean()
                val_cls_loss = (ce_loss(val_pred_cls, torch.from_numpy(y_cls_val).to(device)) * val_w).mean()

                # 验证分位数 loss
                val_quantile_loss = torch.tensor(0.0, device=device)
                val_pred_q = val_output["quantiles"]
                for q_idx, tau in enumerate(QUANTILE_LEVELS):
                    val_quantile_loss = val_quantile_loss + _quantile_loss(val_pred_q[:, q_idx], y_reg_tensor_val, tau)
                val_quantile_loss = val_quantile_loss / N_QUANTILES

                # 验证 IC
                val_ic_loss = torch.tensor(0.0, device=device)
                if len(X_val) > 10:
                    val_ic_loss = -_spearman_correlation(val_pred_reg, y_reg_tensor_val)

            val_loss = (config.regression_weight * val_reg_loss +
                        config.classification_weight * val_cls_loss +
                        config.quantile_weight * val_quantile_loss +
                        config.ic_weight * val_ic_loss)

            # 验证集 Spearman IC (绝对值, 用于监控)
            with torch.no_grad():
                val_ic_abs = abs(_spearman_correlation(val_pred_reg, y_reg_tensor_val).item())

            self.model.train_mode()
            scheduler.step()

            elapsed = time.time() - epoch_start
            self.train_history.append({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss.item(),
                "reg_loss": total_reg_loss / max(n_batches, 1),
                "cls_loss": total_cls_loss / max(n_batches, 1),
                "quantile_loss": total_quantile_loss / max(n_batches, 1),
                "ic_loss": total_ic_loss_val / max(n_batches, 1),
                "val_ic": val_ic_abs,
                "lr": scheduler.get_last_lr()[0],
            })

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"  Epoch {epoch+1}/{config.n_epochs}: "
                            f"train={avg_train_loss:.6f}, val={val_loss.item():.6f}, "
                            f"val_ic={val_ic_abs:.4f}, "
                            f"lr={scheduler.get_last_lr()[0]:.6f}, "
                            f"time={elapsed:.1f}s")

            # Early stopping
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.early_stop_patience:
                    logger.info(f"  Early stopping at epoch {epoch+1}")
                    break

        # 保存模型 (含标准化参数)
        if save_path:
            self.model.save(save_path, standardize_mean=standardize_mean, standardize_std=standardize_std)

        logger.info(f"Training done. Best val loss: {best_val_loss:.6f}")
        return self.model

    def train_cross_sectional(
        self,
        train_sections: dict,
        eval_sections: dict,
        save_path: Optional[str] = None,
        standardize_mean: Optional[np.ndarray] = None,
        standardize_std: Optional[np.ndarray] = None,
    ) -> FactorAttentionModel:
        """
        截面训练循环 (Phase 2)。每个 batch = 一个交易日的全部股票。

        Args:
            train_sections: {date: {"X": (N,60,47), "y_reg": (N,), "y_cls": (N,)}}
            eval_sections: 同上，用于验证
            save_path: 模型保存路径

        Returns:
            训练好的模型
        """
        torch = _import_torch()
        nn = torch.nn

        config = self.config
        device = config.device

        # 初始化模型
        self.model = FactorAttentionModel(
            n_factors=train_sections[next(iter(train_sections))]["X"].shape[-1],
            seq_len=config.seq_len,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            dropout=config.dropout,
            factor_names=config.factor_names,
        )
        self.model.to(device)
        self.model.train_mode()

        # 优化器
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.n_epochs, eta_min=1e-5
        )

        # 验证集划分: 从 train_sections 中取 20% 日期作为 val
        all_dates = list(train_sections.keys())
        np.random.shuffle(all_dates)
        n_val = int(len(all_dates) * config.val_split)
        val_dates = set(all_dates[:n_val])
        train_dates = all_dates[n_val:]

        logger.info(f"Training: {len(train_dates)} train dates, {len(val_dates)} val dates")
        logger.info(f"  Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"  Loss weights: reg={config.regression_weight}, "
                     f"cls={config.classification_weight}, "
                     f"quantile={config.quantile_weight}, "
                     f"ic={config.ic_weight}")

        mse_loss = nn.MSELoss(reduction="none")
        ce_loss = nn.CrossEntropyLoss(reduction="none", label_smoothing=0.0)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(config.n_epochs):
            epoch_start = time.time()

            # 训练: shuffle 日期列表
            epoch_dates = list(train_dates)
            np.random.shuffle(epoch_dates)

            total_loss = 0.0
            n_batches = 0
            total_reg_loss = 0.0
            total_cls_loss = 0.0
            total_quantile_loss = 0.0
            total_ic_loss_val = 0.0

            for date in epoch_dates:
                section = train_sections[date]
                X = section["X"]
                y_reg = section["y_reg"]
                y_cls = section["y_cls"]

                if len(X) < 8:
                    continue

                # 数据增强 1: Stock dropout
                mask = None
                if config.stock_dropout > 0 and self.model.training:
                    mask = np.random.rand(len(X)) > config.stock_dropout
                    if mask.sum() < 8:
                        continue
                    X, y_reg, y_cls = X[mask], y_reg[mask], y_cls[mask]

                # 前向传播
                output = self.model.forward_tensors_cross_sectional(X)

                w_batch = section.get("weight", np.ones(len(y_reg), dtype=np.float32))
                if mask is not None and len(mask) == len(w_batch):
                    w_batch = w_batch[mask]
                w_tensor = torch.from_numpy(w_batch.astype(np.float32)).to(device)

                # 回归 loss
                pred_reg = output["regression"]
                reg_loss = (mse_loss(pred_reg, torch.from_numpy(y_reg).float().to(device)) * w_tensor).mean()

                # 分类 loss
                pred_cls = output["classification"]
                cls_loss = (ce_loss(pred_cls, torch.from_numpy(y_cls).to(device)) * w_tensor).mean()

                # 分位数回归 loss
                pred_quantiles = output["quantiles"]
                y_reg_tensor = torch.from_numpy(y_reg).float().to(device)
                quantile_loss_val = torch.tensor(0.0, device=device)
                for q_idx, tau in enumerate(QUANTILE_LEVELS):
                    q_pred = pred_quantiles[:, q_idx]
                    quantile_loss_val = quantile_loss_val + _quantile_loss(q_pred, y_reg_tensor, tau)
                quantile_loss_val = quantile_loss_val / N_QUANTILES

                # IC loss: 天然截面内
                ic_loss_val = torch.tensor(0.0, device=device)
                if len(X) > 10:
                    spearman_corr = _spearman_correlation(pred_reg, y_reg_tensor)
                    ic_loss_val = -spearman_corr

                # 总 loss
                loss = (config.regression_weight * reg_loss +
                        config.classification_weight * cls_loss +
                        config.quantile_weight * quantile_loss_val +
                        config.ic_weight * ic_loss_val)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1
                total_reg_loss += reg_loss.item()
                total_cls_loss += cls_loss.item()
                total_quantile_loss += quantile_loss_val.item()
                total_ic_loss_val += ic_loss_val.item()

            avg_train_loss = total_loss / max(n_batches, 1)

            # 验证: 对每个 val date 分别算 Spearman IC，取均值
            self.model.eval_mode()
            val_ics = []
            with torch.no_grad():
                for vdate in val_dates:
                    vsec = train_sections[vdate]
                    vX = vsec["X"]
                    if len(vX) < 10:
                        continue
                    vout = self.model.forward_tensors_cross_sectional(vX)
                    v_pred = vout["regression"]
                    v_y = torch.from_numpy(vsec["y_reg"]).float()
                    ic = _spearman_correlation(v_pred, v_y)
                    val_ics.append(ic.item())

            if val_ics:
                val_ic_mean = np.mean(val_ics)
                val_ic_abs = abs(val_ic_mean)
                val_loss = torch.tensor(-val_ic_abs)
            else:
                val_loss = torch.tensor(best_val_loss)
                val_ic_abs = 0.0

            self.model.train_mode()
            scheduler.step()

            elapsed = time.time() - epoch_start
            self.train_history.append({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss.item(),
                "val_ic": val_ic_abs,
                "lr": scheduler.get_last_lr()[0],
            })

            logger.info(f"  Epoch {epoch+1}/{config.n_epochs}: "
                        f"train={avg_train_loss:.6f}, val={val_loss.item():.6f}, "
                        f"val_ic={val_ic_abs:.4f}, "
                        f"lr={scheduler.get_last_lr()[0]:.6f}, "
                        f"time={elapsed:.1f}s")
            # Dual flush: print + direct write to output file
            msg = f"  Epoch {epoch+1}/{config.n_epochs}: train={avg_train_loss:.6f}, val={val_loss.item():.6f}, val_ic={val_ic_abs:.4f}, lr={scheduler.get_last_lr()[0]:.6f}, time={elapsed:.1f}s"
            print(msg, flush=True)
            out_file = os.environ.get("PHASE2_OUTPUT_FILE")
            if out_file:
                with open(out_file, "a") as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} [INFO] {msg}\n")
                    f.flush()

            # Early stopping on val_loss
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.early_stop_patience:
                    logger.info(f"  Early stopping at epoch {epoch+1}")
                    break

        # 保存模型
        if save_path:
            self.model.save(save_path, standardize_mean=standardize_mean, standardize_std=standardize_std)

        logger.info(f"Training done. Best val loss: {best_val_loss:.6f}")
        return self.model


# ═════════════════════════════════════════════════════════
# Attention Learner — 扫描时推理
# ═════════════════════════════════════════════════════════


    def train_multi_scale(
        self,
        X_daily: np.ndarray,
        X_weekly: np.ndarray,
        y_reg: np.ndarray,
        y_cls: np.ndarray,
        sample_weights: np.ndarray,
        save_path: str = None,
        standardize_mean: np.ndarray = None,
        standardize_std: np.ndarray = None,
        weekly_mean: np.ndarray = None,
        weekly_std: np.ndarray = None,
    ):
        """
        训练 Multi-Scale 模型 (日线 + 周线双通道)。

        Args:
            X_daily: (n_samples, daily_seq_len, n_factors)
            X_weekly: (n_samples, weekly_seq_len, n_factors)
            y_reg: (n_samples,) 未来收益率
            y_cls: (n_samples,) 涨跌标签
            sample_weights: (n_samples,) 时间衰减权重
        """
        torch = _import_torch()
        nn = torch.nn
        config = self.config
        device = config.device

        # 初始化 MultiScaleAttentionModel
        self.model = MultiScaleAttentionModel(
            n_factors=X_daily.shape[-1],
            daily_seq_len=X_daily.shape[1],
            weekly_seq_len=X_weekly.shape[1],
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            dropout=config.dropout,
            factor_names=config.factor_names,
        )
        self.model.to(device)
        self.model.train_mode()

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.n_epochs, eta_min=1e-5
        )

        # 划分训练/验证
        n = len(X_daily)
        n_val = int(n * config.val_split)
        indices = np.random.permutation(n)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        Xd_train = X_daily[train_idx]
        Xw_train = X_weekly[train_idx]
        y_reg_train = y_reg[train_idx]
        y_cls_train = y_cls[train_idx]
        w_train = sample_weights[train_idx]

        Xd_val = X_daily[val_idx]
        Xw_val = X_weekly[val_idx]
        y_reg_val = y_reg[val_idx]
        y_cls_val = y_cls[val_idx]
        w_val = sample_weights[val_idx]

        logger.info(f"Multi-scale training: {len(train_idx)} train, {len(val_idx)} val samples")
        logger.info(f"  Daily: {Xd_train.shape}, Weekly: {Xw_train.shape}")
        logger.info(f"  Model params: {sum(p.numel() for p in self.model.parameters()):,}")

        mse_loss = nn.MSELoss(reduction="none")
        ce_loss = nn.CrossEntropyLoss(reduction="none")

        best_val_loss = float("inf")
        patience_counter = 0
        batch_size = config.batch_size

        for epoch in range(config.n_epochs):
            epoch_start = time.time()

            perm = np.random.permutation(len(Xd_train))
            total_loss = 0.0
            n_batches = 0

            for start in range(0, len(Xd_train), batch_size):
                batch_idx = perm[start:start + batch_size]
                if len(batch_idx) < 8:
                    continue

                xd_batch = Xd_train[batch_idx]
                xw_batch = Xw_train[batch_idx]
                y_reg_batch = y_reg_train[batch_idx]
                y_cls_batch = y_cls_train[batch_idx]
                w_batch = w_train[batch_idx]

                output = self.model.forward_tensors(xd_batch, xw_batch)

                w_tensor = torch.from_numpy(w_batch).to(device)

                pred_reg = output["regression"]
                reg_loss = (mse_loss(pred_reg, torch.from_numpy(y_reg_batch).float().to(device)) * w_tensor).mean()

                pred_cls = output["classification"]
                cls_loss = (ce_loss(pred_cls, torch.from_numpy(y_cls_batch).to(device)) * w_tensor).mean()

                pred_quantiles = output["quantiles"]
                y_reg_tensor = torch.from_numpy(y_reg_batch).float().to(device)
                quantile_loss_val = torch.tensor(0.0, device=device)
                for q_idx, tau in enumerate(QUANTILE_LEVELS):
                    q_pred = pred_quantiles[:, q_idx]
                    quantile_loss_val = quantile_loss_val + _quantile_loss(q_pred, y_reg_tensor, tau)
                quantile_loss_val = quantile_loss_val / N_QUANTILES

                ic_loss_val = torch.tensor(0.0, device=device)
                if len(batch_idx) > 10:
                    spearman_corr = _spearman_correlation(pred_reg, y_reg_tensor)
                    ic_loss_val = -spearman_corr

                loss = (config.regression_weight * reg_loss +
                        config.classification_weight * cls_loss +
                        config.quantile_weight * quantile_loss_val +
                        config.ic_weight * ic_loss_val)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_train_loss = total_loss / max(n_batches, 1)

            # Validation
            self.model.eval_mode()
            with torch.no_grad():
                val_output = self.model.forward_tensors(Xd_val, Xw_val)
                val_pred_reg = val_output["regression"]
                val_pred_cls = val_output["classification"]
                val_w = torch.from_numpy(w_val).to(device)
                y_reg_tensor_val = torch.from_numpy(y_reg_val).float().to(device)

                val_reg_loss = (mse_loss(val_pred_reg, y_reg_tensor_val) * val_w).mean()
                val_cls_loss = (ce_loss(val_pred_cls, torch.from_numpy(y_cls_val).to(device)) * val_w).mean()

                val_quantile_loss = torch.tensor(0.0, device=device)
                val_pred_q = val_output["quantiles"]
                for q_idx, tau in enumerate(QUANTILE_LEVELS):
                    val_quantile_loss = val_quantile_loss + _quantile_loss(val_pred_q[:, q_idx], y_reg_tensor_val, tau)
                val_quantile_loss = val_quantile_loss / N_QUANTILES

                val_ic_loss = torch.tensor(0.0, device=device)
                if len(Xd_val) > 10:
                    val_ic_loss = -_spearman_correlation(val_pred_reg, y_reg_tensor_val)

                val_loss = (config.regression_weight * val_reg_loss +
                            config.classification_weight * val_cls_loss +
                            config.quantile_weight * val_quantile_loss +
                            config.ic_weight * val_ic_loss)

                val_ic_abs = abs(_spearman_correlation(val_pred_reg, y_reg_tensor_val).item())

            self.model.train_mode()
            scheduler.step()

            elapsed = time.time() - epoch_start
            self.train_history.append({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss.item(),
                "val_ic": val_ic_abs,
                "lr": scheduler.get_last_lr()[0],
            })

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"  Epoch {epoch+1}/{config.n_epochs}: "
                            f"train={avg_train_loss:.6f}, val={val_loss.item():.6f}, "
                            f"val_ic={val_ic_abs:.4f}, "
                            f"lr={scheduler.get_last_lr()[0]:.6f}, "
                            f"time={elapsed:.1f}s")

            # Early stopping
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.early_stop_patience:
                    logger.info(f"  Early stopping at epoch {epoch+1}")
                    break

        if save_path:
            self.model.save(save_path, standardize_mean=standardize_mean,
                          standardize_std=standardize_std,
                          weekly_mean=weekly_mean, weekly_std=weekly_std)

        logger.info(f"Multi-scale training done. Best val loss: {best_val_loss:.6f}")
        return self.model



class AttentionLearner:
    """动态因子权重提取器。

    每次扫描时:
      1. 加载训练好的模型
      2. 输入最近 N 天的因子数据
      3. 提取 attention weights → 因子权重
      4. 返回 {factor: weight} 给 pipeline 使用
    """

    def __init__(
        self,
        model_path: str,
        seq_len: int = 60,
        n_factors: int = 32,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.seq_len = seq_len
        self.n_factors = n_factors
        self.device = device
        self.model: Optional[FactorAttentionModel] = None
        self._standardize_params: Optional[tuple] = None  # (mean, std)

    def load_model(self) -> bool:
        """加载训练好的模型 (含标准化参数)"""
        if not os.path.exists(self.model_path):
            logger.warning(f"Model file not found: {self.model_path}")
            return False

        import torch as _tmp_torch
        state = _tmp_torch.load(self.model_path, weights_only=False, map_location="cpu")
        actual_n_factors = state.get("n_factors", self.n_factors)
        actual_seq_len = state.get("seq_len", self.seq_len)

        self.model = FactorAttentionModel(
            n_factors=actual_n_factors,
            seq_len=actual_seq_len,
            d_model=state.get("d_model", 128),
            n_heads=state.get("n_heads", 8),
            n_layers=state.get("n_layers", 4),
            factor_names=state.get("factor_names"),
        )
        success = self.model.load(self.model_path)
        if success:
            self.model.to(self.device)
            self.model.eval_mode()
            self.n_factors = actual_n_factors
            self.seq_len = actual_seq_len
            # 从 checkpoint 加载标准化参数
            if "standardize_mean" in state and "standardize_std" in state:
                self._standardize_params = (state["standardize_mean"], state["standardize_std"])
                logger.info("Loaded standardize params from checkpoint")
            else:
                logger.warning("No standardize params in checkpoint, will compute from input")
        return success

    def extract_weights(
        self,
        recent_factors: np.ndarray,
    ) -> dict[str, float]:
        """
        从因子时序中提取动态因子权重。

        Args:
            recent_factors: (seq_len, n_factors) 或 (n_stocks, seq_len, n_factors)

        Returns:
            {factor_name: weight} 归一化后的因子权重
        """
        torch = _import_torch()

        if self.model is None:
            if not self.load_model():
                logger.warning("Cannot load model, returning uniform weights")
                return {f: 1.0 for f in FACTOR_COLUMNS[:self.n_factors]}

        # 确保输入形状
        if recent_factors.ndim == 2:
            recent_factors = recent_factors[np.newaxis, :, :]

        # 标准化
        if self._standardize_params is None:
            flat = recent_factors.reshape(-1, recent_factors.shape[-1])
            mean = np.nanmean(flat, axis=0)
            std = np.nanstd(flat, axis=0)
            std[std < 1e-8] = 1.0
            self._standardize_params = (mean, std)

        mean, std = self._standardize_params
        standardized = (recent_factors - mean) / std
        standardized = np.clip(standardized, -5.0, 5.0)

        # 推理
        output = self.model.forward(standardized, training=False)

        # 因子重要性
        importance = output["factor_importance"]  # (n_stocks, n_factors)
        avg_importance = importance.mean(axis=0)

        # 取绝对值并归一化
        abs_importance = np.abs(avg_importance)
        total = abs_importance.sum()
        if total > 0:
            weights = abs_importance / total * self.model.n_factors
        else:
            weights = np.ones(self.model.n_factors)

        # 映射到因子名: 优先使用模型保存的实际因子名 (避免 FACTOR_COLUMNS 硬编码错位)
        actual_n_factors = self.model.n_factors
        factor_names = self.model._factor_names if self.model._factor_names else []
        if len(factor_names) != actual_n_factors:
            factor_names = list(FACTOR_COLUMNS[:actual_n_factors])
            while len(factor_names) < actual_n_factors:
                factor_names.append(f"factor_{len(factor_names)}")
        result = {factor_names[i]: float(weights[i]) for i in range(actual_n_factors)}

        logger.info(f"Attention weights extracted: {len(result)} factors, "
                     f"top3={sorted(result.items(), key=lambda x: x[1], reverse=True)[:3]}")

        return result

    def predict(
        self,
        recent_factors: np.ndarray,
    ) -> dict:
        """
        完整预测: 返回收益率预测 + 涨跌分类 + 分位数预测 + 因子权重。

        Args:
            recent_factors: (seq_len, n_factors) 或 (n_stocks, seq_len, n_factors)

        Returns:
            {regression, classification, quantiles, factor_weights}
        """
        torch = _import_torch()

        if self.model is None:
            if not self.load_model():
                return {"regression": None, "classification": None, "quantiles": None, "factor_weights": None}

        if recent_factors.ndim == 2:
            recent_factors = recent_factors[np.newaxis, :, :]

        if self._standardize_params is None:
            flat = recent_factors.reshape(-1, recent_factors.shape[-1])
            mean = np.nanmean(flat, axis=0)
            std = np.nanstd(flat, axis=0)
            std[std < 1e-8] = 1.0
            self._standardize_params = (mean, std)

        mean, std = self._standardize_params
        standardized = (recent_factors - mean) / std
        standardized = np.clip(standardized, -5.0, 5.0)

        output = self.model.forward(standardized, training=False)

        factor_names = self.model._factor_names if self.model._factor_names else FACTOR_COLUMNS[:self.model.n_factors]
        importance = output["factor_importance"]
        abs_importance = np.abs(importance.mean(axis=0))
        total = abs_importance.sum()
        if total > 0:
            weights = abs_importance / total * self.model.n_factors
        else:
            weights = np.ones(self.model.n_factors)

        factor_weights = {factor_names[i]: float(weights[i]) for i in range(self.model.n_factors)}

        return {
            "regression": output["regression"].tolist(),
            "classification": output["classification"].tolist(),
            "quantiles": output["quantiles"].tolist(),
            "factor_weights": factor_weights,
        }
