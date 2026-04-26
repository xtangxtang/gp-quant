# Transformer Phase 2 — 修改记录与训练结果

> 日期: 2026-04-25
> 状态: 训练完成，v1/v2 均不理想

## 一、改动清单

### attention_learner.py

| 改动项 | Phase 1 | Phase 2 |
|--------|---------|---------|
| CrossSectionalEncoder | 无 | 新增: 2层, 8 heads, d=128 |
| forward_cross_sectional | 无 | 新增: 时序编码 + 截面编码 + Residual |
| forward_tensors_cross_sectional | 无 | 新增: 截面训练用 |
| dropout | 0.1 | 0.2 |
| weight_decay | 1e-4 | 0.01 |
| label_smoothing | 0 | 0.1 |
| stock_dropout | 无 | 0.2 |
| 数据增强 | 无 | 时序jitter ±2天, 因子噪声 σ=0.05 |
| 总参数 | 575,387 | 840,475 |

### train_attention.py

| 改动项 | Phase 1 | Phase 2 |
|--------|---------|---------|
| 数据管道 | 扁平样本数组 | 按日期分组的截面字典 |
| build_cross_sectional_data() | 无 | 新增: 按日期组织数据 |
| train_cross_sectional() | 无 | 新增: 截面训练循环 |
| CLI 参数 | 基础 | 新增 --cross-sectional, --dropout, --weight-decay, --stock-dropout, --label-smoothing |
| --epochs 默认 | 30 | 50 |
| --max-stocks 默认 | 500 | 800 |

### strategy.py

| 改动项 | Phase 1 | Phase 2 |
|--------|---------|---------|
| 推理路径 | 固定 forward() | 自动检测: hasattr → forward_cross_sectional 或 forward |

## 二、训练执行（Phase 2 v1 — 强正则化）

### 命令

```bash
taskset -c 0-27 .venv/bin/python -m src.strategy.adaptive_state_machine.train_attention \
  --data-dir /home/xtang/gp-workspace/gp-data/tushare-daily-full \
  --model-path src/strategy/adaptive_state_machine/models/attention_phase2.pt \
  --walk-forward --train-years 2024,2025 --eval-year 2026 \
  --max-stocks 800 --epochs 50 \
  --cross-sectional \
  --dropout 0.2 --weight-decay 0.01
```

### 数据

- 股票: 792/800
- 训练日期: 485 天 (388 train / 97 val)
- 评估日期: 55 天
- 平均每天: 407 只股票

### Epoch 轨迹

| Epoch | Train Loss | Val Loss | Val IC | LR | 耗时 |
|-------|-----------|----------|--------|---------|------|
| 1 | 0.002434 | -0.075161 | 0.0752 | 0.000999 | 518.0s |
| 2 | -0.005739 | -0.068191 | 0.0682 | - | 488.2s |
| 3 | -0.004528 | -0.014969 | 0.0150 | - | - |
| 4 | 0.012427 | -0.009503 | 0.0095 | - | - |
| 5 | 0.011872 | -0.016523 | 0.0165 | - | - |
| 6 | 0.012623 | -0.017870 | 0.0179 | - | - |

**Early stopping at epoch 6** — val_loss 从未超过 epoch 1 的 -0.075

### 最终评估 (2026 Holdout)

| 指标 | Phase 1 | Phase 2 v1 | 变化 |
|------|---------|-----------|------|
| Eval IC | 0.0581 | 0.0550 | -5% |
| 方向准确率 | 50.8% | 46.5% | -4.3pp |
| Val IC (best) | 0.906 | 0.075 | - |
| 参数量 | 575K | 840K | +46% |

### Top 5 因子 (v1)
1. mf_big_cumsum_l: 2.6458
2. dom_eig_l: 2.1316
3. perm_entropy_s: 2.1177
4. factor_40: 2.0546
5. volatility_l: 2.0373

## 三、Phase 2 v2 — 中等正则化

### 命令

```bash
taskset -c 0-27 .venv/bin/python -m src.strategy.adaptive_state_machine.train_attention \
  --data-dir /home/xtang/gp-workspace/gp-data/tushare-daily-full \
  --model-path src/strategy/adaptive_state_machine/models/attention_phase2_v2.pt \
  --walk-forward --train-years 2024,2025 --eval-year 2026 \
  --max-stocks 800 --epochs 50 \
  --cross-sectional \
  --dropout 0.15 --weight-decay 0.001 \
  --stock-dropout 0.1 --label-smoothing 0.05
```

### Epoch 轨迹

| Epoch | Train Loss | Val Loss | Val IC | LR | 耗时 |
|-------|-----------|----------|--------|---------|------|
| 1 | -0.002011 | -0.092044 | 0.0920 | 0.000999 | 533.1s |
| 2 | -0.004794 | -0.070482 | 0.0705 | - | 532.0s |
| 3 | 0.004497 | -0.013738 | 0.0137 | - | - |
| 4 | 0.012062 | -0.008607 | 0.0086 | - | - |
| 5 | 0.011858 | -0.002403 | 0.0024 | - | - |
| 6 | 0.011527 | -0.009032 | 0.0090 | - | - |

**Early stopping at epoch 6** — val_loss 从未超过 epoch 1 的 -0.092

### 最终评估 (2026 Holdout)

| 指标 | Phase 1 | Phase 2 v1 (强正则) | Phase 2 v2 (中等正则) |
|------|---------|-------------------|---------------------|
| Eval IC | 0.0581 | 0.0550 | **-0.0017** |
| 方向准确率 | 50.8% | 46.5% | 46.5% |
| Val IC (best) | 0.906 | 0.075 | 0.092 |
| 参数量 | 575K | 840K | 840K |

### Top 5 因子 (v2)
1. path_irrev_m: 3.4108
2. perm_entropy_l: 2.5300
3. factor_42: 2.3442
4. bbw_pctl: 2.2880
5. intraday_path_irrev: 1.9135

## 四、综合分析与诊断

### 问题 1：正则化过度导致欠拟合

两次 Phase 2 训练都在 epoch 6 early stopping，且 Eval IC 不升反降：

| 参数 | Phase 1 | Phase 2 v1 | Phase 2 v2 |
|------|---------|-----------|-----------|
| dropout | 0.1 | 0.2 | 0.15 |
| weight_decay | 1e-4 | 0.01 | 0.001 |
| stock_dropout | 无 | 0.2 | 0.1 |
| 因子噪声 | 无 | σ=0.05 | σ=0.05 |
| label_smoothing | 0 | 0.1 | 0.05 |

v2 虽然降低了正则化强度，但 Eval IC 从 0.0581 跌到 -0.0017，说明问题不仅仅是正则化强度。

### 问题 2：CrossSectionalEncoder 效果不佳

- 增加 265K 参数（总计 840K）但没有带来有效信号
- 截面结构 (stocks 互相 attend) 在截面数据管道中已经通过 IC loss 隐含利用了
- 额外的 CrossSectionalEncoder 可能引入了不必要的复杂度

### 问题 3：训练循环设计

- Train loss 从 Epoch 1 的负值快速变正，说明模型没有持续学习
- Val IC 从 Epoch 1 的最高点单调下降
- 三种数据增强（stock_dropout + jitter + 噪声）叠加，可能破坏了有效信号

### 诊断总结

Phase 1 的问题是严重过拟合 (Val IC=0.906 vs Eval IC=0.058)，但 Phase 2 的修正方向过于激进。
截面数据管道本身是一个好的改进（IC loss 天然截面内计算），但 CrossSectionalEncoder + 多重数据增强组合导致了欠拟合。

## 五、代码级 Bug 诊断（补充分析）

### Bug 1：评估用了错误的前向路径

`train_attention.py` 评估代码 (line ~634):

```python
out = trainer.model.forward(X_e, training=False)  # ← 用的 forward()
```

训练用 `forward_tensors_cross_sectional()`（经过截面编码器），但评估用 `forward()`（跳过截面编码器）。模型的 heads 已经适配了截面编码器输出的分布，却在评估时收到了不同分布的输入。**训练和评估的前向路径不一致。**

### Bug 2：`np.roll` jitter 破坏时序

`attention_learner.py` `train_cross_sectional()` 中:

```python
X = np.roll(X, jitter, axis=1)
```

`np.roll` 是循环移位 — 末尾数据会绕到开头。对时序来说这创造了不可能的模式（第 58~60 天的数据出现在第 1~2 天位置）。这不是 jitter，是数据污染。正确做法是改变序列的起始切片位置，不是 roll。

### Bug 3：CrossSectionalEncoder 位置编码无效

```python
self.pos_encoding = torch.nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
```

shape `(1, 1, d_model)` 广播到所有 stock 位置后是同一个值。模型无法区分不同 stock 的位置 — 截面 Transformer 等效于一个没有位置信息的 set function（对 stock 排列不变）。截面 Transformer 对 set 也能工作，但无法学到"排序相关"的截面效应。

### Bug 4：Val IC 计算方式不正确

`train_cross_sectional()` 中 val IC 是把所有日期的预测拼接后算一个 Spearman。这混合了截面和时序维度。正确的截面 IC 应该**每天算一个 Spearman，再取均值**（mean IC across dates）。

### Bug 5：三重数据增强叠加过度

stock_dropout(20%) + jitter(±2) + noise(σ=0.05) 三者叠加。388 个训练日期 × ~400 stocks = ~155K 有效样本，但每个 epoch 只有 388 次梯度更新（vs Phase 1 的 ~471 次 batch），信号/噪声比太低。

---

## 六、下一步建议：Phase 2.1 — 最小化消融验证

不要同时改多个东西。先做**消融实验**：用截面数据管道 + Phase 1 模型（无 CrossSectionalEncoder），只看数据管道改进本身有没有效果。

### 核心原则

Phase 1 → Phase 2 只改了一个东西：截面数据管道。其他所有正则化和增强回退到 Phase 1。

### 具体方案

| 项目 | Phase 2 v1/v2 | Phase 2.1 建议 |
|------|--------------|---------------|
| CrossSectionalEncoder | 启用 | **关闭**（先验证数据管道） |
| np.roll jitter | 启用 | **删除** |
| 因子噪声 σ=0.05 | 启用 | **删除** |
| stock_dropout | 0.2/0.1 | **0.05**（极轻微） |
| dropout | 0.2/0.15 | **0.1**（回到 Phase 1） |
| weight_decay | 0.01/0.001 | **1e-4**（回到 Phase 1） |
| label_smoothing | 0.1/0.05 | **0**（回到 Phase 1） |
| 训练循环 | 截面 batch (388步/epoch) | 截面 batch |
| 评估前向路径 | `forward()` ← bug | **修复：用对应的截面/非截面路径** |
| Val IC 计算 | 全日期拼接 | **每日 IC 均值** |

### 需要修改的代码

1. **`train_attention.py` 评估部分**: `forward()` → `forward_cross_sectional()` 或根据 flag 选择
2. **`train_cross_sectional()` 数据增强**: 删除 jitter 和 noise，stock_dropout 降到 0.05
3. **`train_cross_sectional()` val IC**: 改为每日 Spearman 均值
4. **CLI 默认值**: dropout/weight_decay/label_smoothing 回退到 Phase 1 水平

### 预期命令

```bash
taskset -c 0-27 .venv/bin/python -m src.strategy.adaptive_state_machine.train_attention \
  --data-dir /home/xtang/gp-workspace/gp-data/tushare-daily-full \
  --model-path src/strategy/adaptive_state_machine/models/attention_phase2_1.pt \
  --walk-forward --train-years 2024,2025 --eval-year 2026 \
  --max-stocks 800 --epochs 30 \
  --cross-sectional \
  --dropout 0.1 --weight-decay 0.0001 \
  --stock-dropout 0.05 --label-smoothing 0
```

### 判断标准

- 如果 Phase 2.1 Eval IC > Phase 1 (0.058) → 截面数据管道有价值，之后再加 CrossSectionalEncoder
- 如果 Phase 2.1 Eval IC ≤ Phase 1 → 截面 batch 方式对这个规模的数据没优势，回退到 Phase 1 的 shuffle 方式
- 之后再独立验证 CrossSectionalEncoder 的效果（修好位置编码后）
