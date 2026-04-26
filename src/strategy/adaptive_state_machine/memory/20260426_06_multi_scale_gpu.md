# Multi-Scale 模型 + 市值中性化 + GPU 训练

**日期**: 2026-04-26
**前置**: `20260426_04_experiment_a_review.md` (建议的 3 个方向)

---

## 一、代码修复清单

### 1. 设备迁移 Bug (GPU 支持)
训练循环中 `torch.from_numpy()` 创建的 tensor 在 CPU 上，但模型已 `.to("cuda")`，导致 `RuntimeError: Expected all tensors to be on the same device`。

修复位置 (attention_learner.py)：

| 方法 | 修复前 | 修复后 |
|------|--------|--------|
| `AttentionModel._embed()` | `torch.from_numpy(x)` | `.to(device)` via `next(self.embedding.parameters()).device` |
| `MultiScaleAttentionModel.forward()` | `torch.from_numpy(x)` | `.to(self.embedding.weight.device)` |
| `MultiScaleAttentionModel.forward_tensors()` | `torch.from_numpy(x)` | `.to(self.embedding.weight.device)` |
| `MultiScaleAttentionModel._encode_daily()` | `.astype()` on tensor | 先 `isinstance()` 检查，tensor 直接跳过转换 |
| `MultiScaleAttentionModel._encode_weekly()` | 同上 | 同上 |

### 2. numpy 数组布尔值 Bug
`factor_names` 是 numpy array，`factor_names or []` 报错 `ValueError: The truth value of an array...`

```python
# 修复前
self._factor_names = factor_names or []
# 修复后
self._factor_names = list(factor_names) if factor_names is not None else []
```

### 3. Multi-scale Eval 数据索引错误
`_build_daily_to_weekly_sequences_multi()` 返回 `(*train_res(9), *eval_res(9), factor_names)` 共 19 个元素。

`main()` 中 eval 数据索引从 `[5:10]` 读到了 train_res 的尾部，导致 `y_reg_eval` 是 shape=(47,) 的权重均值。

```python
# 修复前
X_daily_eval = data[5], X_weekly_eval = data[6], y_reg_eval = data[7] ...
# 修复后
X_daily_eval = data[9], X_weekly_eval = data[10], y_reg_eval = data[11], y_cls_eval = data[12], w_eval = data[13]
factor_names = data[18]
```

### 4. Eval corrcoef 维度不匹配
`pred_reg` shape=(N, 1) vs `y_reg_eval` shape=(N,) → `np.corrcoef` 拼接失败。

```python
pred_reg = np.concatenate(all_pred_reg).flatten()
```

### 5. Multi-scale 训练后 fall-through
Multi-scale 评估完成后代码继续执行到 `else:` 分支（标准单通道训练），因 `TrainConfig` 已移除 `label_smoothing` 字段而崩溃。

```python
# 修复：在 multi-scale eval 后加 return
if args.multi_scale:
    ...
    logger.info(f"  Eval direction accuracy: {acc:.1%}")
    return  # Multi-scale done
```

### 6. Python 环境切换
- 机器有 Python 3.11 和 3.12，torch/pandas 只装在了 3.11
- 3.12 缺少 pandas, scikit-learn, torch → 切换代理 `child-prc.intel.com:913` 安装
- `sudo https_proxy=http://child-prc.intel.com:913 http_proxy=http://child-prc.intel.com:913 python3 -m pip install pandas scikit-learn`
- `sudo https_proxy=http://child-prc.intel.com:913 http_proxy=http://child-prc.intel.com:913 python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`
- 最终使用 **Python 3.12 + torch 2.6.0+cu124** (1x A100 40GB)

---

## 二、新增功能

### Multi-Scale Attention Model
- **输入**: 日线 (60 天, 47 因子) + 周线 (12 周, 47 因子)
- **架构**: 共享 embedding + 两个独立 Transformer encoder → concat summary → fusion layer → heads
- **参数量**: 1,139,867 (vs 单通道 575,387)
- **训练**: `train_multi_scale()` 方法，相同多任务 loss 结构
- **数据**: `_build_daily_to_weekly_sequences_multi()` 周线由日线每 5 日均值聚合

### 市值中性化
- `--neutralize total_mv` 参数
- 横截面 OLS 回归：`factor ~ log(total_mv)`，取残差替代原始因子值
- 集成在 `build_walk_forward_data()` 和 `_build_daily_to_weekly_sequences_multi()` 中

---

## 三、训练结果

### Multi-Scale (日线 + 周线双通道)
| 阶段 | 指标 | 值 |
|------|------|-----|
| 训练 (2024-2025) | 参数量 | 1.14M |
| 训练 (2024-2025) | val IC (Spearman) | 0.878 |
| 评估 (2026 holdout, 24,091 样本) | IC (Pearson) | **0.002** |
| 评估 (2026 holdout) | 方向准确率 | **50.3%** |
| 训练速度 | ~8.5s/epoch | 24091 val samples |

### 对比：Phase 1 Linear 嵌入 (历史结果)
| 阶段 | 指标 | 值 |
|------|------|-----|
| 评估 (2026 holdout) | IC (Spearman) | **0.058** |

---

## 四、结论

### Multi-Scale 过拟合严重
- 训练 val IC 0.88 → 评估 IC 0.002，**过拟合程度远超预期**
- 参数量翻倍 (1.14M vs 575K)，但 OOS IC 反而不如单通道 Linear (0.058)
- 方向准确率 50.3% ≈ 随机猜测

### 可能原因
1. **模型容量过大**: 1.14M 参数对 96K 训练样本，且因子只有 47 个
2. **周线信息冗余**: 周线 = 日线每 5 日均值，信息高度重叠，额外通道可能学到噪声
3. **正则化不足**: dropout=0.1, weight_decay=0.0001 对 1.14M 参数模型太弱

### 后续方向
1. **市值中性化训练** (`--neutralize total_mv`) — 降低市值偏差
2. **Multi-scale 正则化调优**: 加大 dropout (0.3-0.5)、weight_decay (0.01)、早停
3. **简化 multi-scale**: 共享 encoder 而非双 encoder，减少参数量
4. **确认单通道基线**: 用清理后代码重跑 `--walk-forward` 确认 IC≈0.058 可复现
