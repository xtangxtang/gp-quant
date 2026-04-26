# Transformer Phase 2 修改总结与实验回顾

> 日期: 2026-04-25
> 状态: 全部实验完成，Phase 2 整体失败
> 结论: CrossSectionalEncoder 引入后导致 Eval IC 全面恶化

## 一、全部代码修改清单

### 1. attention_learner.py

#### 新增 CrossSectionalEncoder 类

```python
class CrossSectionalEncoder(torch.nn.Module):
    """截面编码器：stocks 互相 attend"""
```
- 2 层 TransformerEncoder, 8 heads, d_model=128, dropout=0.2
- 可学习位置编码 shape `(1, 1, d_model)` — **后续发现此为 Bug**（对所有 stock 广播同一向量）
- 参数增量 ~265K（总参数 575K → 840K）

#### FactorAttentionModel 新增字段

```python
self.cross_sectional_transformer = CrossSectionalEncoder(
    d_model=d_model, n_heads=n_heads, n_layers=2, dropout=dropout,
)
```

#### 新增方法

- `forward_cross_sectional()` — 截面推理：时序编码 → 截面编码 → Residual → Heads
- `forward_tensors_cross_sectional()` — 截面训练用（返回原始 tensor 用于 loss 计算）
- `save()` 添加 `has_cross_sectional=True` 标记

#### TrainConfig 新增字段

```python
cross_sectional: bool = False
weight_decay: float = 0.01      # Phase 2 v1 默认
stock_dropout: float = 0.2      # 截面随机丢弃比例
label_smoothing: float = 0.1    # 分类头 smoothing
```

#### TrainConfig 默认值演变

| 参数 | Phase 1 | Phase 2 v1 | Phase 2 v2 | Phase 2.1 |
|------|---------|-----------|-----------|-----------|
| dropout | 0.1 | 0.2 | 0.15 | **0.1** |
| weight_decay | 1e-4 | 0.01 | 0.001 | **1e-4** |
| stock_dropout | 无 | 0.2 | 0.1 | **0.05** |
| label_smoothing | 0 | 0.1 | 0.05 | **0** |
| n_epochs | 30 | 50 | 50 | 30 |

### 2. train_attention.py

#### 新增 build_cross_sectional_data()

按日期组织截面数据，替代原有的扁平 (stock, date) 数组。

**性能优化**：预建 date-to-index 字典，避免 O(N×M×L) 的 list.index 调用。

```python
# 优化前（极慢）:
if entry_date not in sym_dates: continue  # O(N) 每次
i = sym_dates.index(entry_date)           # O(N) 每次

# 优化后:
stock_lookup[sym] = (fvals, sym_dates, date_to_i, close_arr)
i = date_to_i.get(entry_date)             # O(1)
```

#### 新增 train_cross_sectional() 训练循环

截面 batch 训练，每次梯度更新处理一整天的所有股票。

**数据增强（Phase 2 v1/v2 启用，Phase 2.1 删除）**:
- stock_dropout: 随机丢弃 20% 股票
- np.roll jitter: 序列起始 ±2 天随机偏移 — **后续发现为 Bug**（循环移位创造不可能模式）
- 因子噪声: Gaussian σ=0.05

**Bug 修复**:
- Val IC 改为每日 Spearman 均值（Phase 2 v1/v2 用全日期拼接，混合截面和时序维度）
- mask 变量初始化为 None，避免 UnboundLocalError

#### 评估路径修复

```python
# Bug: 训练用 forward_tensors_cross_sectional(), 评估用 forward()
# 修复:
has_cs = hasattr(trainer.model, 'cross_sectional_transformer')
if has_cs:
    out = trainer.model.forward_cross_sectional(X_e, training=False)
else:
    out = trainer.model.forward(X_e, training=False)
```

#### CLI 新增参数

`--cross-sectional`, `--dropout`, `--weight-decay`, `--stock-dropout`, `--label-smoothing`

### 3. strategy.py

`_predict_states_from_model()` 自动检测截面模型并调用对应前向路径。

## 二、发现的 5 个 Bug

| # | Bug | 影响 | 修复 |
|---|-----|------|------|
| 1 | **评估前向路径不一致**：训练用 forward_tensors_cross_sectional()，评估用 forward() | 训练和评估接收不同分布的输入 | 自动检测模型类型选择路径 |
| 2 | **np.roll 破坏时序**：循环移位把末尾数据绕到开头，创造不可能的时间模式 | 数据污染 | Phase 2.1 删除 |
| 3 | **CrossSectionalEncoder 位置编码无效**：shape (1,1,d_model) 对所有 stock 广播同一值 | 模型无法区分不同 stock 位置 | 未修复 |
| 4 | **Val IC 计算不正确**：全日期拼接后算 Spearman，混合截面和时序 | 验证指标不可靠 | 改为每日 Spearman 均值 |
| 5 | **三重数据增强叠加过度**：stock_dropout(20%) + jitter(±2) + noise(σ=0.05) | 信号/噪声比太低 | Phase 2.1 删除 jitter+noise，stock_dropout 降到 0.05 |

## 三、全部训练结果对比

### Phase 1（Baseline）

- 模型: Phase 1 (575K 参数), 无 CrossSectionalEncoder
- 数据管道: 扁平 (stock, date) 数组
- 正则化: dropout=0.1, weight_decay=1e-4

| 指标 | 值 |
|------|-----|
| **Eval IC** | **0.0581** |
| 方向准确率 | 50.8% |
| Val IC | 0.906（过拟合标志） |

### Phase 2 v1（强正则化）

- 模型: +CrossSectionalEncoder (840K 参数)
- 正则化: dropout=0.2, weight_decay=0.01, stock_dropout=0.2, label_smoothing=0.1
- 数据增强: jitter + 噪声

| Epoch | Train Loss | Val Loss | Val IC |
|-------|-----------|----------|--------|
| 1 | 0.002434 | -0.075161 | 0.0752 |
| 2 | -0.005739 | -0.068191 | 0.0682 |
| 3 | -0.004528 | -0.014969 | 0.0150 |
| 4 | 0.012427 | -0.009503 | 0.0095 |
| 5 | 0.011872 | -0.016523 | 0.0165 |
| 6 | 0.012623 | -0.017870 | 0.0179 |

**Early stopping at epoch 6**

| 指标 | 值 | vs Phase 1 |
|------|-----|-----------|
| Eval IC | 0.0550 | -5% |
| 方向准确率 | 46.5% | -4.3pp |

### Phase 2 v2（中等正则化）

- 正则化: dropout=0.15, weight_decay=0.001, stock_dropout=0.1, label_smoothing=0.05

| Epoch | Train Loss | Val Loss | Val IC |
|-------|-----------|----------|--------|
| 1 | -0.002011 | -0.092044 | 0.0920 |
| 2 | -0.004794 | -0.070482 | 0.0705 |
| 3 | 0.004497 | -0.013738 | 0.0137 |
| 4 | 0.012062 | -0.008607 | 0.0086 |
| 5 | 0.011858 | -0.002403 | 0.0024 |
| 6 | 0.011527 | -0.009032 | 0.0090 |

**Early stopping at epoch 6**

| 指标 | 值 | vs Phase 1 |
|------|-----|-----------|
| Eval IC | **-0.0017** | 完全失效 |
| 方向准确率 | 46.5% | -4.3pp |

### Phase 2.1（最小化消融 — 保留 CrossSectionalEncoder）

- 正则化回退到 Phase 1: dropout=0.1, weight_decay=1e-4
- 删除破坏性数据增强: 无 jitter, 无噪声
- 修复 Bug: 评估路径, Val IC 计算

| Epoch | Train Loss | Val Loss | Val IC | 状态 |
|-------|-----------|----------|--------|------|
| 1 | -0.001821 | -0.067219 | 0.0672 | |
| 2 | -0.015919 | -0.108393 | 0.1084 | ↑ |
| **3** | **-0.020752** | **-0.128780** | **0.1288** | **Best** |
| 4 | -0.024538 | -0.113246 | 0.1132 | ↓ |
| 5 | 0.006478 | -0.009028 | 0.0090 | ↓↓ |
| 6 | 0.011359 | -0.015366 | 0.0154 | ↓ |
| 7 | 0.012400 | -0.014961 | 0.0150 | ↓ |
| 8 | 0.012387 | -0.006749 | 0.0067 | Early Stop |

**Early stopping at epoch 8**

| 指标 | 值 | vs Phase 1 |
|------|-----|-----------|
| Eval IC | **-0.0254** | **反相关，比随机还差** |
| 方向准确率 | 46.5% | -4.3pp |

**Top 5 因子 (Phase 2.1)**: turnover_entropy_l (4.17), bbw_pctl (3.24), factor_42 (2.92), mf_big_streak (2.44), mf_big_cumsum_s (2.25)

### 全阶段对比

| 指标 | Phase 1 | Phase 2 v1 | Phase 2 v2 | Phase 2.1 |
|------|---------|-----------|-----------|-----------|
| CrossSectionalEncoder | 无 | 有 | 有 | 有 |
| Val IC (best) | 0.906 | 0.075 | 0.092 | 0.1288 |
| **Eval IC** | **0.0581** | 0.0550 | -0.0017 | **-0.0254** |
| 方向准确率 | **50.8%** | 46.5% | 46.5% | 46.5% |
| Early Stop | - | epoch 6 | epoch 6 | epoch 8 |
| Val→Eval 关系 | 过拟合 | 接近 | 反相关 | **反相关** |

## 四、核心结论

### 1. CrossSectionalEncoder 是根本问题

所有启用 CrossSectionalEncoder 的实验（v1, v2, 2.1）Eval IC 均不如 Phase 1。即使 Phase 2.1 回退了所有正则化到 Phase 1 水平并修复了所有 Bug，Eval IC 仍然恶化到 -0.0254。

**根因**：
- 位置编码 shape `(1, 1, d_model)` 对所有 stock 广播同一向量，模型无法区分 stock 位置
- 截面 Transformer 等效于一个没有位置信息的 set function，对 stock 排列不变
- 840K 参数 vs 575K 参数，训练数据量不变，容量翻倍但信号不足

### 2. Val IC 与 Eval IC 可能反相关

Phase 2 v2 和 2.1 的 Val IC 看起来合理（0.092, 0.1288），但 Eval IC 为负值（-0.0017, -0.0254）。说明截面训练优化的目标（同一天内股票的相对排序）与真实泛化能力（跨天的 Pearson 相关）不一致。

### 3. 数据增强破坏信号

Phase 2 v1/v2 中 np.roll jitter 和因子噪声叠加导致严重的信号破坏。Phase 2.1 删除后情况有所改善（至少训练到 epoch 8 而非 epoch 6），但仍无法改善 Eval IC。

## 五、已修改的文件

| 文件 | 改动 |
|------|------|
| `attention_learner.py` | 新增 CrossSectionalEncoder, forward_cross_sectional, forward_tensors_cross_sectional, TrainConfig 新字段, 正则化参数默认值调整, 数据增强删除, Val IC 修复 |
| `train_attention.py` | 新增 build_cross_sectional_data(), train_cross_sectional(), 评估路径修复, CLI 参数, 性能优化 |
| `strategy.py` | _predict_states_from_model() 自动检测截面模型 |

## 六、下一步方向（未实施）

### Phase 2.2（建议）— 纯截面数据管道，无 CrossSectionalEncoder

- 关闭 CrossSectionalEncoder，使用 Phase 1 的 575K 参数模型
- 保留截面数据管道（按日期分组）
- 训练循环用 Phase 1 的 forward() 路径
- 目标：验证截面数据管道本身（IC loss 天然截面内计算）是否有价值

### 长期：修复 CrossSectionalEncoder

- 使用 rank-based 位置编码（按股票截面排名分配位置）
- 或使用 permutation-invariant 的 set function（Deep Sets）

## 七、路线 A 执行：Phase 1 回测验证

按照 higher_level_review 的建议，停止 Phase 2 优化，用 Phase 1 模型跑完整回测。

### 修复的 Bug

**strategy.py torch-to-numpy 转换**：模型输出为 torch Tensor，后续 numpy 操作报错。
添加 `.detach().cpu().numpy()` 转换。

### 回测命令

```bash
python -m src.strategy.adaptive_state_machine.run_adaptive_state_machine \
  --backtest --start_date 20250101 --end_date 20260331 \
  --interval_days 20 \
  --attention_model src/strategy/adaptive_state_machine/models/attention_phase1.pt \
  --cls_mode model
```

### 回测结果

- 15 个扫描日期（2025-01-02 → 2026-03-05）
- 总信号: 63,743
- 状态分布: HOLD 87.5% / COLLAPSE 12.5% / ACCUMULATION 0% / BREAKOUT 0%
- Top 因子: turnover_entropy_m (2.50), coherence_decay_rate (2.40), factor_37 (2.33)

### 问题

分类头未校准，`pred_up_prob` 整体偏低，导致没有买入信号（ACCUMULATION/BREAKOUT 全为 0）。
模型的排序能力存在（IC=0.058），但分类头的概率输出无法用于状态判定。

### 结论

Phase 1 模型可以正常推理，因子重要性排序合理，但需要通过调整状态判定阈值或改用分位数回归输出（而非分类头）来生成买入信号。
