# 代码清理 — 回退到 Phase 1

> 日期: 2026-04-26
> 前置: `20260426_04_experiment_a_review.md`

## 完成的改动

### 1. 修复 `extract_weights()` 因子名映射 bug
- `attention_learner.py` 中 `extract_weights()` 和 `predict()` 改用 `self.model._factor_names`
- 之前: 硬编码 `FACTOR_COLUMNS[:n]`（37 个手写顺序，全部错位）
- 修复后: 优先使用模型 checkpoint 中的实际因子名（47 个字母序），fallback 才用 FACTOR_COLUMNS

### 2. 回退到 Phase 1 Linear 嵌入
- `FactorAttentionModel.__init__`: 删除分组嵌入逻辑，改为 `Linear(n_factors, d_model)`
- `_group_embed()` → `_embed()`: 简化为单 `self.embedding(x_tensor)`

### 3. 删除冗余 CrossSectionalEncoder
- 删除类定义、截面 forward 方法、训练/推理/CLI 中所有引用
- 消除 ~265K 死参数

### 4. 简化 `load()` 方法
- 删除向后兼容逻辑

### 5. 修正诊断文档
- 标注"因子集不同"假设已证伪

## 下一步

1. 用 GPU 重训验证回退后 IC 恢复到 0.058
2. 多尺度时间窗（周线/日线融合）
3. 市值中性化（`total_mv` 高权重暗示市值偏差）
