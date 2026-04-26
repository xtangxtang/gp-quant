# Transformer 完全化改造建议

> 日期: 2026-04-25

当前 `FactorAttentionModel` 基础已具备，但局限明显（d=64/2 层、单股票独立、伪标签）。本方案从架构、数据、训练、推理四块提出具体改造建议。

---

## 一、架构升级：从 Encoder-Only 到时空双塔

当前模型只看单股票时序，丢失截面信息。建议改为 **Temporal Encoder + Cross-Sectional Encoder** 双塔：

```
Input: (N_stocks, T=60, F=37)
        ↓
[Temporal Transformer]  — 每只股票独立，捕获时间动态
        ↓ (N_stocks, d_model)
[Cross-Sectional Transformer] — 同一时刻所有股票互相 attention，学截面排序
        ↓
[Multi-Task Heads]: 回归 + 分类 + 状态 + 未来 N 日收益分布(分位数)
```

**理由**：A 股有强烈的截面排名效应（同板块共振、资金跷跷板），单股票模型无法捕获"为什么今天选 A 而不是 B"。可参考 `factor_model_selection/v3_bull_hunter` 的截面思路。

**规模建议**（5300 股 × 多年数据）：
- `d_model=128~256`, `n_heads=8`, `n_layers=4` (temporal) + `2` (cross-sectional)
- 总参数 1~5M，A100 单卡可训

---

## 二、输入表示重构

### 1. 多尺度时间窗

```
当前: seq_len=20 单一窗口
建议: 短中长三窗口拼接
  short:  T=20  日线
  medium: T=60  日线（采样）
  long:   T=20  周线
  weekly_features: 加入 weekly summary token
```

### 2. 因子分组嵌入（Factor Group Embedding）

不把因子直接 Linear 投影。按 7 类（熵/波动/特征值/相干/资金流/价格/分钟）做**分组嵌入**，每组独立 Linear → concat，类似 NLP 的 segment embedding。这样模型能学到"资金流类整体重要"。

### 3. 价格 Patch（参考 PatchTST）

把 OHLCV 做 patching（每 5 天一个 patch），用 vanilla Transformer 处理价格序列，再与因子序列 cross-attention 融合。这是 2023 后时序 SOTA 范式。

---

## 三、训练目标重设计

当前 `assign_state_label` 用未来收益分位 + 熵变化打 5 类标签 — 这是**自我循环的伪标签**，准确率上不去的主因。

### 建议改用纯监督目标：

| 任务 | 输出 | Loss |
|------|------|------|
| **未来收益分布** | 9 个分位数 (10/20/...90%) | Quantile Loss |
| **未来 5/10/20 日方向** | 3 个二分类头 | BCE |
| **超额收益 vs 行业均值** | 回归 | Huber Loss |
| **波动率** | 回归 | MSE |
| **状态分类** | 5 类 | **删除**（伪标签噪声大） |

> 状态名称（accumulation/breakout/...）只在**推理后由规则给收益分位赋予**，训练时不监督。

### Loss 加权

- 用 **Sharpe-aware loss**：`loss = -E[r·sign(pred)] / Var[r·sign(pred)]^0.5`
- 加 **IC Loss**：直接最大化 batch 内预测与真实收益的 Spearman 相关

---

## 四、训练策略

1. **滚动窗口训练（Walk-Forward）**
   - 训练窗 2 年 → 预测下 20 日 → 滚动重训
   - 避免未来信息泄漏，符合 v3_bull_hunter 已验证范式

2. **股票 ID 嵌入（可选）**
   - 5300 个 embedding 容易过拟，建议用**行业 + 市值分桶**嵌入替代

3. **预训练 + 微调**
   - 阶段1：自监督 masked factor reconstruction（类 BERT）
   - 阶段2：多任务监督

4. **Mixup / 时序数据增强**
   - 同行业内股票特征做 mixup
   - 时序 jitter / scaling

5. **Early Stopping 用 IC**，不用 loss

---

## 五、推理与决策

完全去掉 state_evaluator.py 的硬编码阈值：

```python
# 推理
pred_returns_quantile = model(features)  # (N, 9)
expected_return = mean(quantile)
prob_up = P(quantile > 0)
risk = quantile[80%] - quantile[20%]  # IQR

# 选股 = 截面排序，不再用阈值
score = expected_return / risk  # Sharpe-like
top_n = score.argsort()[-N:]

# 状态标签（仅可视化用）
if score > p90 and prob_up > 0.7: "breakout"
elif score > p70 and entropy_low: "accumulation"
elif score < p10: "collapse"
```

---

## 六、保留 vs 抛弃

| 当前模块 | 建议 |
|----------|------|
| feature.py | **保留**，因子计算复用 |
| weight.py (IC + 阈值搜索) | **抛弃**，模型端到端学权重 |
| validator.py (奖惩反馈) | **抛弃**，训练时监督就够了 |
| state_evaluator.py | **简化**为推理后排序 + 可选规则贴标 |
| attention_learner.py | **重写**为更大的双塔模型 |
| config.py 阈值 | **删除**，仅保留路径配置 |

---

## 七、实施路径

**阶段 1**（1-2 周）：在现有 attention_learner.py 上：
- 删除 `state_cls_head`，只保留回归+分类
- 改用 Quantile Loss + IC Loss
- 扩到 `d_model=128, n_layers=4, seq_len=60`
- Walk-Forward 训练 2025 年数据

**阶段 2**：加 Cross-Sectional Encoder

**阶段 3**：PatchTST 风格价格分支 + 多尺度

---

## 八、与现有实验的关系

| 实验 | 结果 | 本方案的处理 |
|------|------|-------------|
| Attention 权重注入 state_evaluator | 52.18% 准确率 | 阶段 1 保留兼容，阶段 2 抛弃 |
| Transformer 5 类状态分类 | 53.14%，98% breakout 崩塌 | 删除状态分类头，改用纯监督 |
| 真实 20 天历史序列 | 已实现，略微改善 | 升级到 seq_len=60 |
