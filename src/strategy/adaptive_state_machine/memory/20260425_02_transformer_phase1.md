# Transformer Phase 1 改造记录

> 日期: 2026-04-25

## 一、改造目标

根据 `transformer_fully_vision_20260425.md` 中的建议，实施**阶段 1**：
- 删除 `state_cls_head`（伪标签噪声大）
- 改用分位数回归 + IC Loss
- 扩大模型规模 (d_model=128, n_layers=4, seq_len=60)

## 二、改动清单

### attention_learner.py (重写)

| 改动 | 之前 | 之后 |
|------|------|------|
| d_model | 64 | 128 |
| n_heads | 4 | 8 |
| n_layers | 2 | 4 |
| seq_len | 20 | 60 |
| state_cls_head | 5 分类头 | **删除** |
| quantile_head | 无 | **新增**: 9 分位点回归 |
| IC Loss | 无 | **新增**: 可微 Spearman 相关 |
| Loss 组成 | 30% MSE + 50% CE + 20% 一致性 | 15% MSE + 15% CE + 40% Quantile + 30% IC |
| 总参数 | 76K | **564K** |

### train_attention.py (重写)

| 改动 | 之前 | 之后 |
|------|------|------|
| 默认 seq_len | 20 | 60 |
| 默认 d_model | 64 | 128 |
| 默认 n_heads | 4 | 8 |
| 默认 n_layers | 2 | 4 |
| mode 参数 | default / state_cls | **删除 state_cls** |
| Walk-Forward | 不支持 | **支持**: `--walk-forward --train-years 2024,2025 --eval-year 2026` |

### strategy.py (修改)

| 改动 | 说明 |
|------|------|
| `_seq_len` | 新增字段, 默认 60, 从模型加载时自动同步 |
| `_build_real_sequences()` | 支持动态 seq_len |
| `_extract_attention_from_cross_section()` | 使用 attn_learner.seq_len |
| `_predict_states_from_model()` | **重写**: 用分位数回归输出做截面排序, 不再用 state_cls_logits |

### run_adaptive_state_machine.py (修改)

| 改动 | 说明 |
|------|------|
| `--cls_mode` 默认值 | "model" → "rules" (model 模式改为实验性) |

## 三、新模型架构

```
Input: (batch, 60, 47)
  ↓
Linear(47 → 128) + 位置编码
  ↓
TransformerEncoder (4 层, 8 heads, d=128)
  ↓
Summary Token (位置 0)
  ↓
┌──────────────┬──────────────┬──────────────┬──────────────┐
│ Regression   │ Classification │ Quantile    │ Factor Proj  │
│ Linear→1     │ Linear→2      │ Linear→9     │ Linear→47    │
│ (收益率)     │ (涨/跌)       │ (9 分位点)   │ (因子重要性) │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

## 四、Loss 设计

```
total_loss = 0.15 * MSE(regression, target)
           + 0.15 * CrossEntropy(classification, target)
           + 0.40 * mean(QuantileLoss(q_tau, target) for tau in [0.1..0.9])
           + 0.30 * (-SpearmanCorrelation(regression, target))
```

- **分位数回归** (40%): 预测 9 个分位点, 用 pinball loss, 输出完整收益分布
- **IC Loss** (30%): 直接最大化 batch 内预测与真实收益的 Spearman 秩相关
- **回归/分类** (各 15%): 保留作为辅助任务

## 五、推理流程

```
model(features) → {
    regression: float,          # 预期收益率
    quantiles: [q10, q20, ..., q90],  # 收益分布
    up_prob: float,             # 上涨概率
    factor_weights: {f: w}      # 因子重要性
}

# 截面排序
expected_return = regression
risk = q80 - q20  # IQR
score = expected_return / risk  # Sharpe-like

# 状态标签 (仅用于可视化, 不再由模型直接分类)
if score > p90 and up_prob > 0.6: "breakout"
elif score > p70 and up_prob > 0.5: "accumulation"
elif score < p10 and up_prob < 0.4: "collapse"
elif score > p30: "hold"
else: "idle"
```

## 六、训练命令

```bash
cd /home/xtang/gp-workspace/gp-quant

# 全量训练
taskset -c 0-27 .venv/bin/python src/strategy/adaptive_state_machine/train_attention.py \
  --data-dir /home/xtang/gp-workspace/gp-data/tushare-daily-full \
  --model-path src/strategy/adaptive_state_machine/models/attention_phase1.pt \
  --seq-len 60 --d-model 128 --n-layers 4 --n-heads 8 --epochs 30

# Walk-Forward 训练 (推荐)
taskset -c 0-27 .venv/bin/python src/strategy/adaptive_state_machine/train_attention.py \
  --data-dir /home/xtang/gp-workspace/gp-data/tushare-daily-full \
  --model-path src/strategy/adaptive_state_machine/models/attention_phase1.pt \
  --walk-forward --train-years 2024,2025 --eval-year 2026

# 扫描 (rules 模式)
.venv/bin/python -m src.strategy.adaptive_state_machine.run_adaptive_state_machine \
  --scan_date 20260425 \
  --attention_model src/strategy/adaptive_state_machine/models/attention_phase1.pt

# 扫描 (model 模式: 分位数回归排名)
.venv/bin/python -m src.strategy.adaptive_state_machine.run_adaptive_state_machine \
  --scan_date 20260425 \
  --attention_model src/strategy/adaptive_state_machine/models/attention_phase1.pt \
  --cls_mode model
```

## 七、与之前的对比

| 维度 | 之前 (Phase 0) | Phase 1 |
|------|---------------|---------|
| 模型参数 | 76K | 564K (7.4x) |
| 序列长度 | 20 | 60 (3x) |
| 训练目标 | 回归+涨跌+5类状态 | 回归+涨跌+9分位+IC |
| 状态判定 | 模型直接分类 (伪标签) | 分位数排名 + 规则贴标 |
| 数据泄漏防护 | 无 | Walk-Forward 模式 |
| 选股方式 | 状态分类 argmax | Sharpe-like 截面排序 |

## 八、后续修复 (2026-04-25 第二轮)

初始实现后发现 4 个问题并全部修复：

### 问题 2: Walk-Forward 评估 OOM

**现象**: `train_attention.py` 中 `trainer.model.forward_tensors(X_eval)` 一次性对全部 eval 数据前向传播，数据量大时 OOM。

**修复**: 分 batch 前向传播 (batch_size=512)，结果拼接。
```python
for bs in range(0, len(X_eval), eval_batch_size):
    be = min(bs + eval_batch_size, len(X_eval))
    out = trainer.model.forward(X_eval[bs:be], training=False)
```

### 问题 3: 分位数索引硬编码

**现象**: `strategy.py` 中 `pred_quantiles[:, 1]` / `[:, 7]` 硬编码假设 `QUANTILE_LEVELS` 顺序不变。

**修复**: 从常量反查索引。
```python
from src.strategy.adaptive_state_machine.attention_learner import QUANTILE_LEVELS
_q_idx = {q: i for i, q in enumerate(QUANTILE_LEVELS)}
q_low = pred_quantiles[:, _q_idx[0.2]]   # 20th
q_high = pred_quantiles[:, _q_idx[0.8]]  # 80th
```

### 问题 4: 标准化参数不一致

**现象**: 训练时 `_standardize_factors` 在全局 flatten 后标准化，但推理时 `AttentionLearner._standardize_params` 只从当前批次计算。训练/推理的标准化分布不同，导致分布偏移。

**修复**: 训练时将 `(mean, std)` 保存到 checkpoint，推理时优先加载使用。
- `FactorAttentionModel.save()` 新增 `standardize_mean`, `standardize_std` 参数
- `AttentionTrainer.train()` 接收并传递 `standardize_mean/std` 到 `model.save()`
- `AttentionLearner.load_model()` 从 checkpoint 加载 `standardize_mean/std` 到 `_standardize_params`

### 问题 5: 分位数单调性约束

**现象**: 9 个分位数输出应满足 `q10 ≤ q20 ≤ ... ≤ q90`，但直接用 `Linear` 输出无法保证。

**修复**: 改为 base + cumsum(softplus(deltas)) 结构。
```python
# 模型:
self.quantile_base_head = Linear(..., 1)       # 最低分位点 (q10)
self.quantile_delta_head = Linear(..., n_q-1)  # 8 个增量

def _monotone_quantiles(self, summary_out):
    base = self.quantile_base_head(summary_out)  # (batch, 1)
    deltas = F.softplus(self.quantile_delta_head(summary_out))  # ≥ 0
    cumulative = torch.cumsum(deltas, dim=1)
    return torch.cat([base, base + cumulative], dim=1)
```
通过 `softplus(≥0)` + `cumsum(递增)` 强制保证单调性。

## 九、Pipeline 精简：摘除 weight.py 和 validator.py

全面 Transformer 化后，因子权重由模型 attention 端到端学习，选股由分位数排名决定，不再需要外部 IC 计算和验证反馈。

### 区分两种 IC

| IC 的位置 | 作用 | 状态 |
|-----------|------|------|
| weight.py 截面 IC 权重 | 4-Agent 闭环中更新因子权重 | **删掉** |
| validator.py 因子奖惩 | 状态判定验证后给 WeightLearner 反馈 | **删掉** |
| 训练 loss 中 IC Loss | 可微 Spearman 相关，优化排名一致性 | **保留** (Phase 1) |
| 评估指标 Eval IC | 监控模型预测质量 | **永远保留** |

### 改动

**strategy.py**:
- 删除 `from .weight import WeightLearner` 和 `from .validator import Validator`
- `scan()` 方法签名简化：移除 `price_df`, `trade_dates`, `validator`, `learner` 参数
- 删除 weight update 整个代码块 (validator.verify + learner.update)
- 删除 `validator.ingest_predictions()` 调用
- `backtest()` 方法：删除 `load_price_matrix()` 调用、validator/learner 实例化、`get_performance_summary()` 输出
- docstring 更新：反映新流程

**weight.py / validator.py**: **已删除**（无外部引用）

**config.py**: 清理 stale 注释 (Weight Learner / Validator 引用)

### 精简后流程

```
因子计算 (feature.py)
    ↓
Transformer 推理 (attention_learner.py)
    ├──→ attention 权重 → 因子重要性
    └──→ 分位数回归 → 截面排序
    ↓
状态判定
    ├── model 模式: 分位数排名 → 规则贴标
    └── rules 模式: state_evaluator.py (fallback)
    ↓
输出信号 CSV
```

### 已删除模块

| 模块 | 状态 |
|------|------|
| weight.py | **已删除** |
| validator.py | **已删除** |

### 保留模块

| 模块 | 状态 |
|------|------|
| state_evaluator.py | rules 模式仍使用 |
| config.py | 清理了 stale 注释，保留 |
