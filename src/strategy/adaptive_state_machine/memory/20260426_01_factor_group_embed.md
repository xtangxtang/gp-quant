# 因子分组嵌入（Factor Group Embedding）实现与训练结果

> 日期: 2026-04-26
> 前置: `memory/20260425_13_absolute_filter.md` — 绝对信号过滤改动

## 一、实现内容

### 1. 绝对信号过滤（strategy.py, 2 行改动）

在 `_predict_states_from_model()` 中，BREAKOUT/ACCUMULATION 条件加了 `and pred_return > 0`：

```python
if pct > 0.9 and q50_pctile > 0.8 and pred_return > 0:
    state = StockState.BREAKOUT
elif pct > 0.7 and q50_pctile > 0.5 and pred_return > 0:
    state = StockState.ACCUMULATION
```

### 2. 因子分组嵌入（attention_learner.py）

**FACTOR_GROUPS 定义**：11 个语义组（entropy, volatility, eigenvalue, coherence, moneyFlow, price, intraday, valuation, capital, order_flow, meta），覆盖 64 个因子。

**动态构建**：模型初始化时根据实际 factor_names 构建 active_groups，而非静态硬编码。每组独立 `Linear(n_factors → d_group=16)` → concat → `Linear(11×16 → d_model=128)`。

**向后兼容**：`load()` 检测旧模型（有 `embedding.weight` 无 `group_embeddings.*`），动态创建 single embedding 层，`_group_embed()` 走旧路径。

### 3. 训练脚本改动（train_attention.py）

- factor_names 数值列过滤（排除 ts_code 等非数值列）
- `config.factor_names` 赋值（3 个数据构建函数中）

## 二、训练配置

| 参数 | 值 |
|------|-----|
| 模式 | Walk-Forward: train=[2024,2025], eval=2026 |
| 股票数 | 800 |
| 因子数 | 47（800 只股票的公共因子） |
| 活跃组 | 10（order_flow 组无公共因子被排除） |
| 参数量 | 855,851 |
| Epoch | 50 |
| 设备 | CPU 28 核 |
| Dropout | 0.2 |
| Weight decay | 0.01 |
| 模型文件 | `models/attention_group_embed.pt` |

## 三、训练结果

### 训练曲线

| Epoch | train_loss | val_loss | val_ic (Spearman) |
|-------|-----------|----------|-------------------|
| 1 | -0.046 | -0.080 | 0.303 |
| 5 | -0.158 | -0.164 | 0.578 |
| 10 | -0.215 | -0.216 | 0.744 |
| 15 | -0.238 | -0.239 | 0.820 |
| 20 | -0.250 | -0.248 | 0.847 |
| 25 | -0.258 | -0.254 | 0.865 |
| 30 | -0.264 | -0.258 | 0.879 |
| 35 | -0.268 | -0.261 | 0.889 |
| 40 | -0.271 | -0.263 | 0.896 |
| 45 | -0.273 | -0.264 | 0.899 |
| 50 | -0.274 | -0.265 | **0.900** |

### 2026 Eval 集（前向检验）

| 指标 | Phase 1 | Phase 2 (分组嵌入) |
|------|---------|-------------------|
| Eval IC (Pearson) | 0.058 | 0.017 |
| 方向准确率 | ~50% | 50.0% |

### Top 5 Attention 因子

1. `von_neumann_entropy`: 3.5953
2. `bbw`: 2.9503
3. `factor_44`: 2.7244
4. `mf_sm_proportion`: 2.6091
5. `coherence_decay_rate`: 1.9385

## 四、关键观察

1. **Val Spearman IC = 0.900** — 训练集上排序能力极强，Train/Val 差距仅 0.009，无过拟合
2. **Eval Pearson IC = 0.017** — 远低于 Phase 1 (0.058)，2026 前向数据上模型几乎无排序能力
3. **方向准确率 50.0%** — 恰好随机猜测水平
4. **因子分组嵌入** 让模型学到了因子语义（von_neumann_entropy、bbw 等被高权重），但尚未转化为更好的前向预测能力

## 五、后续步骤

- 需要检查 eval 前向检验的代码路径（Pearson IC 计算是否正确）
- 对比 Phase 1 模型在相同 eval 集上的表现
- P&L 验证（用新模型跑回测生成信号 → verify_pnl.py）
