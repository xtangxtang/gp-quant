# Transformer 状态分类器实验记录

> 日期: 2026-04-24

## 一、改动概述

将规则判定 (state_evaluator.py) 替换为 Transformer 直接输出 5 类状态分类 (idle / accumulation / breakout / hold / collapse)，替代原有的阈值条件计数 + AQ/BQ 打分机制。

### 新增/修改的文件

| 文件 | 改动 |
|------|------|
| `train_attention.py` | 新增 `--mode state_cls` 选项，标签改为收益分位 + 熵值判定 |
| `attention_learner.py` | 新增 `state_cls_head` (5 分类头)，训练器三任务 loss |
| `strategy.py` | 新增 `_predict_states_from_model()`，支持 `--cls_mode model/rules` |
| `pipeline.py` | 新增 `cls_mode` 参数 |
| `run_adaptive_state_machine.py` | 新增 `--cls_mode` CLI 参数 |
| `state_evaluator.py` | 保留不变 (rules 模式仍使用) |

## 二、标签定义

从未来收益分位反推状态标签（非规则输出）：

| 状态 | 条件 |
|------|------|
| **breakout** | 未来收益 ≥ 80th 分位 |
| **accumulation** | 收益 50-80th 分位 + 熵值下降 (t-20 ~ t 熵值差 < -0.05) |
| **hold** | 收益 20-50th 分位 |
| **collapse** | 收益 ≤ 20th 分位 |
| **idle** | 收益 50-80th 分位但熵值未下降 |

训练数据分布 (500 只股票 × 多扫描日):
- idle: ~25%
- accumulation: ~18%
- breakout: ~20%
- hold: ~25%
- collapse: ~12%

## 三、模型架构改动

在 `FactorAttentionModel` 中新增状态分类头：

```python
self.state_cls_head = torch.nn.Sequential(
    torch.nn.Linear(d_model, 32),
    torch.nn.GELU(),
    torch.nn.Dropout(dropout),
    torch.nn.Linear(32, 5),
)
```

Loss 组成: 30% MSE(回归) + 30% CE(涨跌) + 40% CE(5 类状态)

## 四、真实历史序列实现

### 问题

最初使用伪序列 (19 天均值填充 + 1 天真实值) 进行推理，导致模型推理分布与训练分布不一致，全部 500 只股票预测为 breakout。

### 解决

在 `strategy.py` 中新增 `_factor_history` 缓冲区和 `_build_real_sequences()` 方法：
- 每次扫描后将当日截面数据追加到 `_factor_history`
- 推理时从缓存中取最近 19 天真实数据 + 当天 = 20 天真实序列
- 缺失值前向填充，仍缺失则用列均值

### 效果

- 伪序列: 500/500 = 100% breakout
- 真实序列 (首次扫描无历史): 仍然 100% breakout（历史重复当天数据）
- 真实序列 (后续扫描有历史): 93.7% breakout (5021/5358)，略有改善但仍严重失衡

## 五、回测结果

### 设置

| 参数 | 值 |
|------|-----|
| 回测区间 | 2025-01-01 ~ 2026-03-31 |
| 扫描间隔 | 20 个交易日 |
| 扫描次数 | 15 次 |
| 模式 | Transformer 状态分类 (`--cls_mode model`) |
| 序列类型 | 真实 20 天历史 |

### 结果

| 指标 | 值 |
|------|-----|
| 总验证数 | 23,754 |
| 正确 | 12,622 |
| 错误 | 11,132 |
| **准确率** | **53.14%** |
| 规则基线 | 52.18% |

### 状态分布 (全量)

| 状态 | 数量 | 占比 |
|------|------|------|
| breakout | 23,309 | 98.1% |
| hold | 403 | 1.7% |
| accumulation | 42 | 0.2% |
| idle | 0 | 0% |
| collapse | 0 | 0% |

### 最终扫描 (20260305) 状态分布

| 状态 | 数量 |
|------|------|
| breakout | 5,021 |
| idle | 269 |
| hold | 56 |
| accumulation | 12 |
| collapse | 0 |

## 六、问题分析

### 核心问题：类别崩塌

模型将 98% 以上的股票预测为 breakout，实质退化为二分类器 (breakout vs not-breakout)。

### 可能原因

1. **Softmax 校准不足**: 模型输出的 5 类概率分布不均，breakout 类的 softmax 值系统性偏高
2. **训练/推理分布差异**: 即使使用真实序列，推理时的因子分布与训练时仍有偏移
3. **边界模糊**: idle / accumulation / hold 三类在收益分位定义上重叠度高，模型难以区分
4. **Loss 权重**: 40% 状态分类 loss 可能不足以让模型学到 5 类边界，尤其当回归和分类任务更容易优化时
5. **首次扫描问题**: 第一轮扫描无历史数据，20 天全是同一天的重复值，导致预测全偏

### 与规则基线的对比

| 维度 | 规则基线 | Transformer 分类 |
|------|----------|------------------|
| 准确率 | 52.18% | 53.14% (+0.96%) |
| 状态多样性 | 4 种状态均有分布 | 几乎只有 breakout |
| 实用性 | 可用 | 不可用 (无区分度) |

虽然准确率略有提升，但由于状态分布极度不均衡，实际上无法用于多状态选股。

## 七、待尝试方案

1. **类别加权损失**: `CrossEntropyLoss(weight=class_weights)`，按训练集中各类别频率的倒数加权
2. **温度缩放**: 推理时对 logits 除以温度系数 (T > 1) 使概率分布更均匀
3. **阈值替代 argmax**: 用验证集调优每类的独立阈值，而非取最大概率
4. **更严格的 breakout 标签**: 将 breakout 标签从 80th 分位提高到 90th 分位，减少训练集中 breakout 样本
5. **混合模式**: Transformer 只负责 breakout vs non-breakout 二分类，其他状态用规则判定
