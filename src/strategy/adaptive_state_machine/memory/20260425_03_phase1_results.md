# Transformer Phase 1 — 修改记录与训练结果

> 日期: 2026-04-25
> 状态: 训练完成，模型已保存

## 一、改造依据

根据 `20260425_01_transformer_fully_vision.md` 中的建议，实施 **阶段 1 改造**。

详细内容见 `20260425_02_transformer_phase1.md`，此处为训练执行与结果记录。

## 二、改动清单

### attention_learner.py (重写)

| 改动项 | Phase 0 (之前) | Phase 1 (之后) |
|--------|----------------|----------------|
| d_model | 64 | 128 |
| n_heads | 4 | 8 |
| n_layers | 2 | 4 |
| seq_len | 20 | 60 |
| state_cls_head | 5 分类头 | **删除** |
| quantile_head | 无 | **新增**: 9 分位点回归 (monotonicity 约束) |
| IC Loss | 无 | **新增**: 可微 Spearman 相关 |
| Loss 组成 | 30% MSE + 50% CE + 20% 一致性 | 15% MSE + 15% CE + 40% Quantile + 30% IC |
| 总参数 | 76K | 575,387 |

### train_attention.py (重写)

| 改动项 | Phase 0 | Phase 1 |
|--------|---------|---------|
| 默认 seq_len | 20 | 60 |
| 默认 d_model | 64 | 128 |
| Walk-Forward | 不支持 | 支持: `--walk-forward --train-years 2024,2025 --eval-year 2026` |
| mode 参数 | default / state_cls | 删除 state_cls |

### strategy.py (修改)

- 新增 `_seq_len` 字段，默认 60，从模型加载时自动同步
- `_build_real_sequences()` 支持动态 seq_len
- `_predict_states_from_model()` 重写：用分位数回归输出做截面排序，不再用 state_cls_logits

### run_adaptive_state_machine.py (修改)

- `--cls_mode` 默认值改为 `"rules"`（`model` 模式改为实验性）

### weight.py / validator.py

- **已删除**（Transformer 化后不再需要外部 IC 计算和验证反馈）

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

**分位数单调性约束**: 改为 `base + cumsum(softplus(deltas))` 结构，保证 `q10 ≤ q20 ≤ ... ≤ q90`。

## 五、训练执行

### 命令

```bash
cd /home/xtang/gp-workspace/gp-quant
taskset -c 0-27 .venv/bin/python -m src.strategy.adaptive_state_machine.train_attention \
  --data-dir /home/xtang/gp-workspace/gp-data/tushare-daily-full \
  --model-path src/strategy/adaptive_state_machine/models/attention_phase1.pt \
  --walk-forward --train-years 2024,2025 --eval-year 2026 \
  --max-stocks 500 --epochs 30
```

- CPU: taskset 绑定 28 核 (0-27)
- 数据: 495 只股票，Train 120,457 样本 / Eval 11,324 样本
- 收益率统计: mean=0.0072, std=0.0927

### 训练时间线

| 时间 | 事件 |
|------|------|
| 09:02 | 训练开始，因子计算 |
| 09:07 | 数据准备完成，开始训练 |
| 09:12 | Epoch 1 |
| 09:31 | Epoch 5 |
| 09:55 | Epoch 10 |
| 10:19 | Epoch 15 |
| 10:43 | Epoch 20 |
| 11:07 | Epoch 25 |
| 11:31 | Epoch 30，训练完成，模型保存 |

### 遇到的问题

1. **ImportError**: 直接运行 `python src/.../train_attention.py` 报 `attempted relative import with no known parent package`
   - **修复**: 改用模块方式 `python -m src.strategy.adaptive_state_machine.train_attention`

2. **日志缓冲**: Python logging 输出缓存，epochs 16-29 在训练过程中不可见（文件未增长），但不影响训练本身

3. **OOM 风险**: eval 全量前向传播可能 OOM
   - **修复**: 分 batch 前向传播 (batch_size=512)

4. **标准化参数不一致**: 训练/推理标准化分布不同
   - **修复**: 训练时将 `(mean, std)` 保存到 checkpoint，推理时加载

## 六、Epoch 轨迹

| Epoch | Train Loss | Val Loss | Val IC | LR | 耗时 |
|-------|-----------|----------|--------|---------|------|
| 1 | -0.051603 | -0.091739 | 0.3405 | 0.000997 | 288.3s |
| 5 | -0.206941 | -0.209329 | 0.7226 | 0.000934 | 288.2s |
| 10 | -0.250142 | -0.246301 | 0.8406 | 0.000753 | 288.6s |
| 15 | -0.264809 | -0.256572 | 0.8731 | 0.000505 | 289.7s |
| 20 | -0.273287 | -0.262921 | 0.8930 | 0.000258 | 289.5s |
| 25 | -0.277880 | -0.266492 | 0.9038 | 0.000076 | 288.5s |
| 26 | -0.278692 | -0.266564 | 0.9041 | - | 288.5s |
| 27 | -0.279115 | -0.266670 | 0.9047 | - | 288.5s |
| 28 | -0.279374 | -0.267016 | 0.9055 | - | 288.5s |
| 29 | -0.279687 | -0.267114 | 0.9060 | - | 288.5s |
| 30 | -0.279902 | -0.267142 | 0.9060 | 0.000010 | 288.6s |

### 收敛特征

- Train loss 单调递减：-0.052 → -0.280
- Val loss 单调递减：-0.092 → -0.267
- Val IC 单调递增：0.34 → 0.91
- Epoch 25 后基本收敛（Val IC 从 0.9038 → 0.9060，提升微弱）
- LR 从 0.001 衰减到 0.00001，调度正常

## 七、最终评估结果 (2026 Walk-Forward Holdout)

| 指标 | 值 | 基线/阈值 | 评估 |
|------|-----|-----------|------|
| **Eval IC (Pearson)** | **0.0581** | > 0.03 → Phase 2 | 通过 |
| **方向准确率** | **50.8%** | 50% (随机) | 略优 |
| **模型参数量** | **575,387** | - | - |

### Top 5 因子 (按 Attention 权重)

| 排名 | 因子 | 权重 |
|------|------|------|
| 1 | turnover_entropy_m | 2.3786 |
| 2 | coherence_decay_rate | 2.3184 |
| 3 | factor_37 | 2.2788 |
| 4 | mf_big_cumsum_l | 2.1063 |
| 5 | perm_entropy_l | 1.8881 |

### 分析

1. **Val IC vs Eval IC 差距大**: Val IC=0.906 但 Eval IC=0.058，说明模型在训练/验证集上过度拟合，泛化能力有限
2. **方向准确率仅 50.8%**: 略高于随机基线 (50%)，在单天收益率预测任务中属于正常水平
3. **Top 因子**: turnover_entropy_m（换手率熵）和 coherence_decay_rate（退相干速率）权重最高，符合物理直觉
4. **factor_37**: 实际训练使用了 47 个因子（feature_engine.py 输出的所有数值列），但 config.py 中只命名了 37 个，所以索引 37+ 的因子显示为未命名。后续应在 feature_engine.py 中统一因子名称

## 八、Pipeline 精简后的流程

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

## 九、下一步决策

根据预设标准：
- IC > 0.03 → **进入 Phase 2** ✓
- IC 0.01-0.03 → 输入优化
- IC < 0.01 → 优先 Phase 2

**结论**: Eval IC = 0.0581 > 0.03，**直接进入 Phase 2 (Cross-Sectional Encoder)**。

Phase 2 重点解决的问题：
1. 缩小 Val/Eval IC 差距（正则化、数据增强、早停）
2. 提升方向准确率（当前 50.8% → 目标 52%+）
3. 修复 factor_37 未命名问题
4. 加入时序一致性信号

## 十、模型文件

- `src/strategy/adaptive_state_machine/models/attention_phase1.pt`
- 包含: 模型权重 + standardize_mean + standardize_std
