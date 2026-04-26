# 实验 A 结果审查 — 三个问题 + 建议

**日期**: 2026-04-26
**前置**: `20260426_03_experiment_a_results.md`

---

## 问题 1: `extract_weights()` 因子名映射 bug — Top 5 全错

`extract_weights()` 使用硬编码 `FACTOR_COLUMNS`（37 个旧因子名，手动排列顺序）按位置映射，但模型实际 47 个因子是**字母序排列**的。

**47/47 全部错位。**

文档报告的 Top 5 → 实际因子：

| 报告名 | 权重 | 实际因子 | 说明 |
|--------|------|---------|------|
| `turnover_entropy_m` | 3.01 | **`dom_eig_l`** | 主特征值长周期 |
| `von_neumann_entropy` | 2.63 | **`path_irrev_m`** | 路径不可逆中周期 |
| `intraday_path_irrev` | 2.42 | **`total_mv`** | 总市值 ⚠️ |
| `factor_41` | 2.21 | **`vol_ratio_s`** | 短期量比 |
| `factor_43` | 2.20 | **`volatility_l`** | 长周期波动率 |

`total_mv` 排第三值得警惕 — 市值因子是 A 股最强但最容易过拟合的因子。

**根因**: `FACTOR_COLUMNS` 只有 37 个条目，比模型实际 47 因子少 10 个，且顺序完全不同（手写 vs 字母序）。

**修复**: `extract_weights()` 应使用 `self.model._factor_names` 而非 `FACTOR_COLUMNS`：
```python
# 修复前
factor_names = FACTOR_COLUMNS[:actual_n_factors]
# 修复后
factor_names = self.model._factor_names if self.model._factor_names else FACTOR_COLUMNS[:actual_n_factors]
```

---

## 问题 2: 02 诊断文档"因子集不同"假设被证伪

实测 v1 (800 股) 和 v2 (500 股) 的 47 因子**完全相同**（都是字母序同一组）。

`max_stocks` 只影响训练样本量，不影响因子列。`build_walk_forward_data()` 排除非数值列后，因子集由 `feature_engine` 固定产出决定，与股票数量无关。

因此 `20260426_02_group_embed_diagnosis.md` 中"因子集不同导致 IC 下降"的假设**不成立**。v1 (IC=0.017) → v2 (IC=0.055) 的改善**全部归因于正则化配置修复**（dropout 0.2→0.1, weight_decay 0.01→0.0001）。

---

## 问题 3: "可以保留"的结论偏乐观

IC 0.055 ≈ 0.058，分组嵌入没有带来任何提升，但增加了：

| 维度 | Phase 1 (Linear) | 分组嵌入 | 评价 |
|------|-----------------|---------|------|
| 嵌入参数 | 47×128 = 6,016 | 10 组 Linear + fusion ≈ 10K+ | 更多 |
| 推理开销 | 1 次矩阵乘 | dict 查找 + 10 次切片 + concat + fusion | 更慢 |
| 代码复杂度 | 无额外定义 | FACTOR_GROUPS (65 行) + 动态分组逻辑 | 更复杂 |
| Eval IC | 0.058 | 0.055 | 持平或略低 |

**奥卡姆剃刀：性能相同时选更简单的方案。**

---

## 建议

### 立即行动
1. **修复 `extract_weights()` bug** — 改用 `self.model._factor_names`
2. **修正 02 诊断文档** — 标注"因子集不同"假设已证伪

### 策略决策
3. **回退到 Phase 1 Linear 嵌入** — 分组嵌入增加复杂度但无 IC 收益
4. **保留 FACTOR_GROUPS 定义** — 作为因子语义文档有参考价值，但不用于模型嵌入

### 后续方向（精力应投入）
5. **Multi-scale time windows** — 不同窗口捕捉不同周期信号
6. **因子质量提升** — `total_mv` 排第三暗示模型可能依赖市值偏差，考虑市值中性化
7. **PatchTST price branch** — 引入原始价格序列作为补充输入
