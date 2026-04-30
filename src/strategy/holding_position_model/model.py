"""
Holding Position Model — Transformer + 持仓特征双分支架构

架构:
  Input A: 因子时序 (seq_len × n_factors) → Transformer Encoder → summary_vector
  Input B: 持仓特征 (5 维) → Linear → holding_vector
  Concat → 三个头:
    stay_prob:     sigmoid → [0, 1]  趋势延续概率
    collapse_risk: sigmoid → [0, 1]  崩塌风险
    expected_days: relu   → [0, inf] 预期持有天数
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ─── Lazy import torch ───
_torch = None
_nn = None


def _import_torch():
    global _torch, _nn
    if _torch is None:
        import torch
        import torch.nn as nn
        _torch = torch
        _nn = nn
    return _torch, _nn


# ─── Ensure torch/nn available at class definition time ───
import torch
import torch.nn as nn
_torch = torch
_nn = nn


class HoldingPositionModel(nn.Module):
    """持仓决策模型 — Transformer + 持仓特征双分支"""

    def __init__(
        self,
        n_factors: int = 32,
        seq_len: int = 60,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        holding_dim: int = 5,
        holding_hidden: int = 32,
        dropout: float = 0.1,
        factor_names: Optional[list[str]] = None,
    ):
        super().__init__()

        self.n_factors = n_factors
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.holding_dim = holding_dim
        self.holding_hidden = holding_hidden
        self._factor_names = factor_names

        # ── Factor 分支: Transformer Encoder ──
        self.embedding = nn.Linear(n_factors, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Summary token: 对序列做 mean pooling + CLS 的融合
        self.summary_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # ── Holding 分支 ──
        self.holding_proj = nn.Sequential(
            nn.Linear(holding_dim, holding_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── 融合 + 三个头 ──
        fused_dim = d_model + holding_hidden
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.stay_head = nn.Linear(fused_dim, 1)
        self.collapse_head = nn.Linear(fused_dim, 1)
        self.days_head = nn.Linear(fused_dim, 1)

    def forward(
        self,
        factor_seq: np.ndarray,
        holding_feat: np.ndarray,
        training: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        Args:
            factor_seq: (batch, seq_len, n_factors)
            holding_feat: (batch, 5)
            training: 是否 training 模式 (影响 dropout)

        Returns:
            {
                "stay_prob": (batch,),      # sigmoid [0, 1]
                "collapse_risk": (batch,),  # sigmoid [0, 1]
                "expected_days": (batch,),  # relu [0, inf]
            }
        """
        torch = _import_torch()[0]

        self.train(training)
        with torch.set_grad_enabled(training):
            x_factors = torch.tensor(factor_seq, dtype=torch.float32)
            x_holding = torch.tensor(holding_feat, dtype=torch.float32)

            # Factor 分支
            x = self.embedding(x_factors)
            x = x + self.pos_encoding[:, :x.size(1), :]
            x = self.dropout(x)
            x = self.transformer(x)

            # Summary: mean pooling + learnable projection
            summary = self.summary_proj(x.mean(dim=1))  # (batch, d_model)

            # Holding 分支
            h = self.holding_proj(x_holding)  # (batch, holding_hidden)

            # 融合
            fused = torch.cat([summary, h], dim=-1)
            fused = self.fusion(fused)

            # 三个头
            stay_logit = self.stay_head(fused).squeeze(-1)
            collapse_logit = self.collapse_head(fused).squeeze(-1)
            days_raw = self.days_head(fused).squeeze(-1)

            stay_prob = torch.sigmoid(stay_logit)
            collapse_risk = torch.sigmoid(collapse_logit)
            expected_days = torch.relu(days_raw)

            return {
                "stay_prob": stay_prob.detach().cpu().numpy(),
                "collapse_risk": collapse_risk.detach().cpu().numpy(),
                "expected_days": expected_days.detach().cpu().numpy(),
            }

    def save(self, path: str):
        """保存模型权重"""
        import torch as _tmp_torch
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = {
            "model_state_dict": self.state_dict(),
            "n_factors": self.n_factors,
            "seq_len": self.seq_len,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "holding_dim": self.holding_dim,
            "holding_hidden": self.holding_hidden,
            "factor_names": self._factor_names,
        }
        _tmp_torch.save(state, path)
        logger.info(f"Saved model to {path}")

    @classmethod
    def load_from_checkpoint(cls, path: str) -> tuple["HoldingPositionModel", dict]:
        """从 checkpoint 加载模型"""
        import torch as _tmp_torch
        state = _tmp_torch.load(path, weights_only=False, map_location="cpu")

        model = cls(
            n_factors=state.get("n_factors", 32),
            seq_len=state.get("seq_len", 60),
            d_model=state.get("d_model", 128),
            n_heads=state.get("n_heads", 8),
            n_layers=state.get("n_layers", 4),
            holding_dim=state.get("holding_dim", 5),
            holding_hidden=state.get("holding_hidden", 32),
            factor_names=state.get("factor_names"),
        )

        # Load state dict
        model_state = state["model_state_dict"]
        model.load_state_dict(model_state, strict=False)

        logger.info(f"Loaded model from {path}")
        return model, state


def compute_loss(
    outputs: dict[str, np.ndarray],
    y_stay: np.ndarray,
    y_collapse: np.ndarray,
    y_days: np.ndarray,
    sample_weights: np.ndarray,
    loss_stay_weight: float = 1.0,
    loss_collapse_weight: float = 1.0,
    loss_days_weight: float = 0.5,
) -> tuple[float, dict[str, float]]:
    """
    计算多任务 loss (纯 numpy, 用于训练循环中的 logging)。
    实际训练用 torch 的 BCE/MSE。
    """
    torch = _import_torch()[0]

    stay_pred = torch.tensor(outputs["stay_prob"], dtype=torch.float32)
    collapse_pred = torch.tensor(outputs["collapse_risk"], dtype=torch.float32)
    days_pred = torch.tensor(outputs["expected_days"], dtype=torch.float32)

    w = torch.tensor(sample_weights, dtype=torch.float32)

    # BCE loss for stay and collapse
    stay_target = torch.tensor(y_stay, dtype=torch.float32)
    collapse_target = torch.tensor(y_collapse, dtype=torch.float32)

    stay_loss = torch.nn.functional.binary_cross_entropy(stay_pred, stay_target, reduction="none")
    stay_loss = (stay_loss * w).mean() * loss_stay_weight

    collapse_loss = torch.nn.functional.binary_cross_entropy(collapse_pred, collapse_target, reduction="none")
    collapse_loss = (collapse_loss * w).mean() * loss_collapse_weight

    # MSE loss for days
    days_target = torch.tensor(y_days, dtype=torch.float32)
    days_loss = torch.nn.functional.mse_loss(days_pred, days_target, reduction="none")
    days_loss = (days_loss * w).mean() * loss_days_weight

    total_loss = stay_loss + collapse_loss + days_loss

    component_losses = {
        "stay_loss": stay_loss.item(),
        "collapse_loss": collapse_loss.item(),
        "days_loss": days_loss.item(),
        "total_loss": total_loss.item(),
    }

    return total_loss.item(), component_losses
