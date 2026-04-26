# Adaptive State Machine — 策略总结

> 最后更新: 2026-04-24

## 一、策略概览

Adaptive State Machine 是一个基于 A 股日线数据的量化选股系统，采用 **四组件闭环架构**：因子计算 → 权重学习 → 状态判定 → 验证反馈。

与传统策略不同，它引入了 **Transformer 注意力模型** 来动态判断"当前市场状态下该关注哪些因子"，替代传统的固定 IC 权重，使策略能够自适应市场环境变化。

### 核心文件

```
adaptive_state_machine/
├── config.py              # 配置: 状态枚举、因子列表、阈值、AdaptiveConfig
├── feature.py             # 因子计算: 全市场并行计算 79 个因子
├── weight.py              # 权重学习: IC 计算 / Attention 权重注入、阈值搜索
├── state_evaluator.py     # 状态判定: AQ/BQ 打分、分数竞争、模型信号校准
├── validator.py           # 验证反馈: 历史预测验证、因子绩效评分
├── attention_learner.py   # Attention 模型: Transformer 架构、训练、推理
├── strategy.py            # 统一入口: AdaptiveStateMachine 类 (scan + backtest)
├── pipeline.py            # 向后兼容的 run_scan / run_backtest 函数
├── train_attention.py     # Attention 模型训练脚本
└── run_adaptive_state_machine.py  # CLI 入口
```

---

## 二、架构详解

### 2.1 因子计算 (feature.py)

复用 `entropy_accumulation_breakout/feature_engine.py`，计算 79 个因子（实际参与状态判定约 37 个核心因子）：

| 类别 | 因子数 | 示例 |
|------|--------|------|
| 熵指标 | 9 | perm_entropy_m, path_irrev_m/l, entropy_accel |
| 波动率 | 8 | vol_impulse, vol_compression, bbw |
| 主特征值 | 2 | dom_eig_m, dom_eig_l |
| 量子相干性 | 4 | coherence_l1, purity_norm, coherence_decay_rate |
| 资金流 | 9 | mf_big_momentum, mf_big_streak, mf_flow_imbalance |
| 价格位置 | 1 | breakout_range |
| 分钟线微观 | 4 | intraday_perm_entropy, intraday_path_irrev |

因子计算使用 `ProcessPoolExecutor(max_workers=28)` 并行处理，全市场约 5300 只股票。

### 2.2 Attention 模型 (attention_learner.py)

**架构**: 小型 Transformer (76K 参数)
- 输入: (seq_len=20, n_factors=47) 每只股票的因子时序
- 嵌入: Linear(47 → 64) + 可学习位置编码
- Transformer: 2 层 encoder, 4 attention heads, d_model=64
- 输出头: 回归(未来收益率) + 分类(涨/跌) + 因子重要性投影
- 可学习 summary token 聚合序列信息

**训练**:
- 多任务 loss: 30% MSE(回归) + 50% CrossEntropy(分类) + 20% 一致性损失
- 时间衰减权重: 近年样本权重更高 (lambda=0.02)
- 早停机制: patience=5 轮
- 当前模型: 300 只股票 × 15 epochs, ~330K 样本

**推理时提取两个输出**:
1. **注意力权重**: 通过 factor_proj 层将 summary token 输出投影到因子空间 → 归一化为平均 1.0 的因子权重
2. **模型预测**: 回归头输出预期收益率，分类头输出上涨概率

### 2.3 权重学习 (weight.py)

权重学习器有两种模式：

**模式 A — Attention 模式** (当前默认):
- 从 Attention 模型提取因子权重
- 写入 `config.attention_weights` 和 `config.factor_weights`
- **跳过 IC 计算**，完全依赖 Attention 的动态权重

**模式 B — IC 模式** (Attention 不可用时回退):
- 计算截面 Spearman Rank IC: IC(factor, forward_return)
- 滚动多日数据 (lookback=60 天, forward_window=10 天)
- 权重更新: w += lr * IC * (1 + validator_reward)

两种模式都会：
- **阈值搜索**: 用历史数据搜索最优阈值组合 (perm_entropy_acc, path_irrev_acc, dom_eig_breakout 等 10 个可调阈值)
- **AQ/BQ 权重更新**: 根据因子重要性重新分配 AQ(积累质量) 和 BQ(突破质量) 内部权重
- **平滑过渡**: 新旧阈值按 0.2 权重平滑混合，防止突变

### 2.4 状态判定 (state_evaluator.py)

每只股票经过 4 个独立检测器打分，然后 **分数竞争** 决定最终状态：

| 状态 | 检测条件 | 分数计算 |
|------|----------|----------|
| **accumulation** (吸筹) | perm_entropy_m 低, path_irrev_m 高, mf_flow_imbalance 正, mf_big_streak 连续 | 通过条件数/总条件数 |
| **breakout** (突破) | dom_eig_m 高, vol_impulse 大, perm_entropy_m 低, mf_big_momentum 正 | 通过条件数/总条件数 |
| **collapse** (崩塌) | perm_entropy_m 极高, path_irrev_m 极低, entropy_accel 快, purity_norm 低 | 触发信号数/需要数(4) |
| **hold** (持仓) | 之前已进入 breakout 且未超时 | 跨扫描日跟踪 |
| **idle** (无信号) | 以上都不满足 | — |

**AQ (积累质量) 分数计算**:
```
AQ = Σ w_i × normalize(factor_i)
```
- 6 个 AQ 因子: perm_entropy_m, path_irrev_m, big_net_ratio_ma, purity_norm, mf_big_streak, mf_flow_imbalance
- 当 `config.attention_weights` 存在时，权重从 Attention 模型动态提取；否则用固定权重

**BQ (突破质量) 分数计算**:
```
BQ = Σ w_j × normalize(factor_j)
```
- 7 个 BQ 因子: dom_eig_m, vol_impulse, perm_entropy_m, path_irrev_m, coherence_decay_rate, mf_big_momentum, mf_big_net_ratio
- 同上，Attention 权重优先

**综合置信度**:
```
confidence = aq_bq_weight × AQ + (1 - aq_bq_weight) × BQ
默认 aq_bq_weight = 0.4 (更看重 BQ)
```

**模型预测校准** (方案 2):
Transformer 的上涨概率 `pred_up_prob` 作为校准因子影响分数竞争：
- `pred_up_prob < 0.3` 且 collapse_score > 0.5 → collapse 判定放宽 (×1.3)
- `0.3 ≤ pred_up_prob < 0.4` → accum/breakout 分数打折 (×0.7)
- `pred_up_prob > 0.6` → accum/breakout 分数上浮 (×1.1, 上限 1.0)

### 2.5 验证反馈 (validator.py)

- 保存每次扫描的预测结果
- 到期后用实际价格验证 (forward_window=10 天后)
- 统计准确率 (correct / total_verified)
- 按状态、按因子分别统计
- 为 Weight Learner 提供奖励/惩罚信号

---

## 三、数据流

```
日线 CSV (5300+ 只)
    ↓
[FactorCalculator] → 79 列因子矩阵 + 周线数据
    ↓
[AttentionLearner] → (1) 因子权重 {factor: weight}
                     (2) 每只股票的 {pred_return, pred_up_prob}
    ↓
[WeightLearner] → 更新 AdaptiveConfig:
                   - factor_weights / attention_weights
                   - aq_weights / bq_weights
                   - thresholds (10 个可调阈值)
    ↓
[StateEvaluator] → 每只股票: {state, confidence, aq_score, bq_score, pred_return, pred_up_prob}
    ↓
[Validator] → 保存预测，到期验证 → 反馈给下次扫描
```

---

## 四、回测结果

### 4.1 回测设置

| 参数 | 值 |
|------|-----|
| 回测区间 | 2025-01-01 ~ 2026-03-31 |
| 扫描间隔 | 每 20 个交易日 (约 1 个月) |
| 扫描次数 | 15 次 |
| 覆盖股票 | ~5,300 只/次 (全市场) |
| 权重模式 | Attention (IC 跳过) |
| 注入方式 | Attention 权重 + 模型预测校准 |

### 4.2 准确率对比

| 方案 | 验证数 | 正确 | 错误 | 准确率 |
|------|--------|------|------|--------|
| 纯 IC (基线) | 5,117 | 2,605 | 2,512 | 50.90% |
| 纯 Attention (无注入) | 5,117 | 2,606 | 2,511 | 50.93% |
| **Attention 注入 + 模型校准** | **7,743** | **4,040** | **3,703** | **52.18%** |

Attention 注入方案比基线 **+1.28 个百分点**。

### 4.3 按状态分解

| 状态 | 纯 IC | 纯 Attention (无注入) | Attention 注入 | 变化 |
|------|-------|----------------------|----------------|------|
| breakout | 50.1% (639/1275) | — | 50.1% (1135/2264) | 持平 |
| accumulation | 51.3% (445/867) | — | 52.0% (625/1201) | +0.7% |
| **hold** | 51.8% (976/1884) | — | **54.4% (1734/3187)** | **+2.6%** |
| collapse | 50.0% (546/1091) | — | 50.0% (546/1091) | 持平 |

**hold 状态是主要贡献者**，Attention 权重注入 AQ/BQ 打分后，持仓判断更准确。

### 4.4 信号分布

15 次扫描共生成 8,667 个信号：

| 状态 | 数量 | 占比 |
|------|------|------|
| hold | 2,213 | 25.5% |
| breakout | 1,363 | 15.7% |
| collapse | 1,340 | 15.5% |
| accumulation | 940 | 10.8% |
| idle (未输出) | ~2,811 | 32.5% |

各扫描日期信号分布（仅含信号状态）:

| 日期 | accumulation | breakout | hold | collapse | 备注 |
|------|-------------|----------|------|----------|------|
| 2025-01-02 | 59 | 124 | 0 | 0 | 年初，首次扫描无 hold |
| 2025-02-07 | 42 | 49 | 124 | 0 | |
| 2025-03-07 | 48 | 55 | 169 | 0 | |
| 2025-04-07 | 14 | 68 | 101 | 0 | 吸筹信号最少 |
| 2025-05-08 | 80 | 110 | 121 | 0 | |
| 2025-06-06 | 76 | 32 | 167 | 275 | collapse 首次出现，市场转弱 |
| 2025-07-04 | 61 | 36 | 134 | 183 | |
| 2025-08-01 | 92 | 125 | 64 | 100 | breakout 反弹 |
| 2025-08-29 | 85 | 97 | 155 | 63 | |
| 2025-09-26 | 64 | 43 | 209 | 129 | |
| 2025-11-03 | 56 | 82 | 129 | 86 | |
| 2025-12-01 | 65 | 129 | 120 | 112 | 年末 breakout 增多 |
| 2025-12-29 | 51 | 80 | 206 | 41 | |
| 2026-01-28 | 81 | 261 | 194 | 104 | breakout 激增 |
| 2026-03-05 | 66 | 72 | 320 | 247 | hold 累积最多 |

### 4.5 Attention 权重趋势

最后一轮扫描 (2026-03-05) 的 Top 5 Attention 权重：

| 因子 | 权重 | 含义 |
|------|------|------|
| path_irrev_l | 2.67 | 长周期路径不可逆性 (趋势持续性) |
| purity_norm | 2.22 | 密度矩阵纯度 (信号噪声比) |
| entropy_accel | 2.17 | 熵值加速度 (无序度变化速度) |
| mf_big_streak | 2.11 | 大单连续流入天数 |
| von_neumann_entropy | 2.06 | 冯·诺依曼熵 (系统无序度) |

---

## 五、关键设计决策

1. **为什么用 Attention 替代 IC**: IC 是线性相关，Attention 能捕捉因子间的非线性交互和市场状态依赖。实验表明纯 Attention 与纯 IC 基线持平，但注入状态判定后准确率提升 1.28%。

2. **为什么需要注入**: Attention 权重只更新 `config.factor_weights` 不会影响状态判定，因为状态判定基于阈值条件计数。通过注入 `config.attention_weights` 到 AQ/BQ 打分，以及模型预测校准，Attention 才能真正影响结果。

3. **为什么用伪序列**: 每次扫描只有单日截面快照，没有真实历史序列。用全市场因子均值填充历史 19 天、真实值作为最新一天，让模型至少能利用当日的截面信息。

4. **为什么 hold 状态提升最大**: Attention 权重的动态性使 AQ 打分更准确地反映了"哪些因子在当前市场下更重要"，而 hold 状态依赖跨日 AQ/BQ 累积，受益最大。

---

## 七、Transformer 状态分类器实验 (2026-04-24 后续)

尝试用 Transformer 直接输出 5 类状态分类替代规则判定，详见 `transformer_state_cls_20260424.md`。

**结论**: 准确率 53.14%（略高于规则基线 52.18%），但存在严重类别崩塌（98% 预测为 breakout），实际不可用。待尝试类别加权损失、温度缩放等修复方案。

---

## 八、运行命令

```bash
cd /home/xtang/gp-workspace/gp-quant

# 单日扫描
taskset -c 0-27 .venv/bin/python -m src.strategy.adaptive_state_machine.run_adaptive_state_machine \
  --scan_date 20260424

# 历史回测
taskset -c 0-27 .venv/bin/python -m src.strategy.adaptive_state_machine.run_adaptive_state_machine \
  --backtest --start_date 20250101 --end_date 20260331 --interval_days 20 \
  --attention_model src/strategy/adaptive_state_machine/models/attention_model.pt
```

输出目录: `results/adaptive_state_machine/`
