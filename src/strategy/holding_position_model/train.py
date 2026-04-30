"""
Holding Position Model — 训练脚本 (memmap/DataLoader 版本)

多任务训练: stay_prob (BCE) + collapse_risk (BCE) + expected_days (MSE)

memmap 策略:
  - X_factors_train 等是 memmap 数组 (在磁盘上)
  - 每个 epoch 随机采样 index → 只把当前 batch 读到 GPU
  - 内存峰值 = batch_size × seq_len × n_factors × 4 bytes
"""

from __future__ import annotations

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)

_torch = None


def _import_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


class HoldingTrainer:
    """持仓模型训练器 — DataLoader 模式"""

    def __init__(self, model, lr=1e-3, weight_decay=1e-4, device="cpu"):
        torch = _import_torch()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters()], lr=lr, weight_decay=weight_decay,
        )
        self.train_history = []

    def _batch_iter(
        self, n_samples, indices, batch_size,
        X_factors, X_holding, y_stay, y_collapse, y_days, sample_weights,
    ):
        """生成器: 按 batch 读取 memmap 数据 → 送到 GPU"""
        torch = _import_torch()

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]

            # memmap 随机访问: 只把当前 batch 读到内存
            f_batch = torch.tensor(X_factors[batch_idx], dtype=torch.float32, device=self.device)
            h_batch = torch.tensor(X_holding[batch_idx], dtype=torch.float32, device=self.device)
            stay_t = torch.tensor(y_stay[batch_idx], dtype=torch.float32, device=self.device)
            collapse_t = torch.tensor(y_collapse[batch_idx], dtype=torch.float32, device=self.device)
            days_t = torch.tensor(y_days[batch_idx], dtype=torch.float32, device=self.device)
            w_t = torch.tensor(sample_weights[batch_idx], dtype=torch.float32, device=self.device)

            yield f_batch, h_batch, stay_t, collapse_t, days_t, w_t

    def train_epoch(
        self, X_factors, X_holding, y_stay, y_collapse, y_days, sample_weights,
        batch_size=256,
        loss_stay_weight=1.0, loss_collapse_weight=1.0, loss_days_weight=0.5,
    ):
        torch = _import_torch()
        nn = torch.nn

        self.model.train(True)
        n = len(X_factors)
        indices = np.random.permutation(n)

        total_loss = 0.0
        n_batches = 0

        for f_batch, h_batch, stay_t, collapse_t, days_t, w_t in self._batch_iter(
            n, indices, batch_size,
            X_factors, X_holding, y_stay, y_collapse, y_days, sample_weights,
        ):
            # Forward
            x = self.model.embedding(f_batch)
            x = x + self.model.pos_encoding[:, :x.size(1), :]
            x = self.model.dropout(x)
            x = self.model.transformer(x)
            summary = self.model.summary_proj(x.mean(dim=1))
            h = self.model.holding_proj(h_batch)
            fused = self.model.fusion(torch.cat([summary, h], dim=-1))

            stay_pred = torch.sigmoid(self.model.stay_head(fused).squeeze(-1))
            collapse_pred = torch.sigmoid(self.model.collapse_head(fused).squeeze(-1))
            days_pred = torch.relu(self.model.days_head(fused).squeeze(-1))

            # Loss
            loss = (
                nn.functional.binary_cross_entropy(stay_pred, stay_t, reduction="none").mean()
                * loss_stay_weight
                + nn.functional.binary_cross_entropy(collapse_pred, collapse_t, reduction="none").mean()
                * loss_collapse_weight
                + nn.functional.mse_loss(days_pred, days_t, reduction="none").mean()
                * loss_days_weight
            )

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            # 清理 GPU 缓存
            del f_batch, h_batch, stay_t, collapse_t, days_t, w_t
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return {"loss": total_loss / max(n_batches, 1)}

    def evaluate(
        self, X_factors, X_holding, y_stay, y_collapse, y_days, batch_size=512,
    ):
        torch = _import_torch()
        nn = torch.nn

        self.model.train(False)
        n = len(X_factors)
        if n == 0:
            return {"eval_loss": 0.0, "stay_acc": 0.0, "collapse_acc": 0.0, "days_mae": 0.0}

        all_stay_pred = []
        all_collapse_pred = []
        all_days_pred = []
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)

                f_tensor = torch.tensor(X_factors[start:end], dtype=torch.float32, device=self.device)
                h_tensor = torch.tensor(X_holding[start:end], dtype=torch.float32, device=self.device)
                stay_t = torch.tensor(y_stay[start:end], dtype=torch.float32, device=self.device)
                collapse_t = torch.tensor(y_collapse[start:end], dtype=torch.float32, device=self.device)
                days_t = torch.tensor(y_days[start:end], dtype=torch.float32, device=self.device)

                x = self.model.embedding(f_tensor)
                x = x + self.model.pos_encoding[:, :x.size(1), :]
                x = self.model.transformer(x)
                summary = self.model.summary_proj(x.mean(dim=1))
                h = self.model.holding_proj(h_tensor)
                fused = self.model.fusion(torch.cat([summary, h], dim=-1))

                stay_pred = torch.sigmoid(self.model.stay_head(fused).squeeze(-1))
                collapse_pred = torch.sigmoid(self.model.collapse_head(fused).squeeze(-1))
                days_pred = torch.relu(self.model.days_head(fused).squeeze(-1))

                all_stay_pred.append(stay_pred.cpu().numpy())
                all_collapse_pred.append(collapse_pred.cpu().numpy())
                all_days_pred.append(days_pred.cpu().numpy())

                loss = (
                    nn.functional.binary_cross_entropy(stay_pred, stay_t).item()
                    + nn.functional.binary_cross_entropy(collapse_pred, collapse_t).item()
                    + nn.functional.mse_loss(days_pred, days_t).item() * 0.5
                )
                total_loss += loss
                n_batches += 1

                del f_tensor, h_tensor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        stay_pred_all = np.concatenate(all_stay_pred)
        collapse_pred_all = np.concatenate(all_collapse_pred)
        days_pred_all = np.concatenate(all_days_pred)

        return {
            "eval_loss": total_loss / max(n_batches, 1),
            "stay_acc": ((stay_pred_all > 0.5) == y_stay).mean(),
            "collapse_acc": ((collapse_pred_all > 0.5) == y_collapse).mean(),
            "days_mae": np.abs(days_pred_all - y_days).mean(),
        }

    def train(
        self,
        X_factors_train, X_holding_train, y_stay_train, y_collapse_train, y_days_train, w_train,
        X_factors_eval=None, X_holding_eval=None, y_stay_eval=None, y_collapse_eval=None, y_days_eval=None,
        n_epochs=30, batch_size=256, save_path="",
        standardize_mean=None, standardize_std=None,
        loss_stay_weight=1.0, loss_collapse_weight=1.0, loss_days_weight=0.5,
    ):
        logger.info(
            f"Training: {len(X_factors_train)} samples, "
            f"eval: {len(X_factors_eval) if X_factors_eval is not None else 0} samples"
        )

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            train_result = self.train_epoch(
                X_factors_train, X_holding_train, y_stay_train, y_collapse_train, y_days_train, w_train,
                batch_size=batch_size,
                loss_stay_weight=loss_stay_weight,
                loss_collapse_weight=loss_collapse_weight,
                loss_days_weight=loss_days_weight,
            )

            epoch_info = {"epoch": epoch, "train_loss": train_result["loss"], "time": time.time() - t0}

            if X_factors_eval is not None and len(X_factors_eval) > 0:
                eval_result = self.evaluate(
                    X_factors_eval, X_holding_eval, y_stay_eval, y_collapse_eval, y_days_eval,
                )
                epoch_info.update(eval_result)

            self.train_history.append(epoch_info)

            if epoch % 5 == 0 or epoch == 1:
                eval_str = ""
                if "eval_loss" in epoch_info:
                    eval_str = (
                        f" eval_loss={epoch_info['eval_loss']:.4f} "
                        f"stay_acc={epoch_info['stay_acc']:.3f} "
                        f"collapse_acc={epoch_info['collapse_acc']:.3f} "
                        f"days_mae={epoch_info['days_mae']:.1f}"
                    )
                logger.info(
                    f"Epoch {epoch}/{n_epochs}: train_loss={epoch_info['train_loss']:.4f}{eval_str} "
                    f"({epoch_info['time']:.1f}s)"
                )

        result = {"history": self.train_history}
        if save_path:
            self.model.save(save_path)
            result["model_path"] = save_path
            if standardize_mean is not None:
                np.save(save_path.replace(".pt", "_stats.npy"), {
                    "mean": standardize_mean, "std": standardize_std,
                })

        # Flush all handlers to ensure output is visible
        for handler in logging.getLogger().handlers:
            handler.flush()

        return result
