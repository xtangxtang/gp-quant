# Factor Group Embedding 诊断报告

**日期**: 2026-04-26
**结论**: Eval IC 从 0.058 降到 0.017，**非苹果对苹果比较**，无法归因到分组嵌入本身

---

## 代码审查结果 ✅

| 检查项 | 结果 |
|--------|------|
| FACTOR_GROUPS 覆盖率 | 47/47 因子全部命中，0 遗漏 |
| factor_names 持久化 | checkpoint 正确保存/恢复 |
| 向后兼容 (Phase 1 旧模型) | `_use_old_embedding` flag 正确切换 |
| `_group_embed()` 列索引 | 按因子名查 idx，不依赖位置顺序，逻辑正确 |
| 活跃分组数 | 10/11（order_flow 无匹配因子，已排除） |
| group_embeddings 权重保存 | 10 组 Linear 层 + fusion 层均在 state_dict 中 |

**代码层面无 bug。**

---

## IC 下降根因分析

### Val IC 0.900 vs Eval IC 0.017 — 过拟合差距比 Phase 1 更大

| 指标 | Phase 1 | Group Embed | 差异 |
|------|---------|-------------|------|
| Val IC | 0.906 | 0.900 | ≈ 相同 |
| Eval IC | 0.058 | 0.017 | **-70%** |
| 参数量 | 575K | 855K | +49% |
| dropout | 0.1 | 0.2 | Phase 2 配置 |
| weight_decay | 1e-4 | 0.01 | Phase 2 配置 (100x) |
| max_stocks | 500 | 800 | 不同股票集 |

### 三个混淆变量

#### 1. 因子集不同（max_stocks 500 → 800）

不同 `max_stocks` 导致公共因子列不同。500 只股票的因子交集 vs 800 只股票的因子交集可能有差异。Phase 1 的 47 因子和 Group Embed 的 47 因子**虽然数量相同，但具体因子可能不完全一致**。

更多股票 → 更多缺失值 → 更多因子被过滤 → 可能丢失 Phase 1 中有效的因子，替换为信息量较低的因子。

#### 2. 正则化配置错误

使用了 Phase 2 Cross-Sectional 失败时的默认值：
- `dropout=0.2`（Phase 1 成功用 0.1）
- `weight_decay=0.01`（Phase 1 成功用 1e-4，差 100 倍）

Phase 2 的 5 次实验全部失败，当时的正则化配置已被证明无效。

#### 3. 冗余参数（CrossSectionalEncoder）

`__init__` 中 `self.cross_sectional_transformer = CrossSectionalEncoder(...)` **始终被创建**，即使 `forward()` 不使用它。这导致：
- 855K 参数中约 265K 是死权重
- `state_dict` 包含这些冗余参数
- 对训练本身无影响（不参与梯度计算），但增加了 checkpoint 体积

---

## 建议：控制变量重训

在判断分组嵌入是否有效之前，必须控制变量：

### 实验 A：最小变量对比

```bash
python -m src.strategy.adaptive_state_machine.train_attention \
    --max-stocks 500 \
    --dropout 0.1 \
    --weight-decay 0.0001 \
    --save-path src/strategy/adaptive_state_machine/models/attention_group_embed_v2.pt
```

与 Phase 1 唯一区别：`Linear(47→128)` → `分组嵌入(11组×Linear→16→concat→fusion→128)`

**预期**：
- 如果 Eval IC ≥ 0.058 → 分组嵌入至少不劣于单层 Linear，可继续优化
- 如果 Eval IC < 0.058 → 分组嵌入增加了不必要的复杂度，回退到 Phase 1

### 实验 B（可选）：移除 CrossSectionalEncoder

在 `__init__` 中删除 `self.cross_sectional_transformer` 的创建，减少 265K 冗余参数。虽然不影响训练，但保持代码整洁。

---

## 结论

当前 Group Embed IC=0.017 的结果**不能说明分组嵌入无效**，因为实验改变了 3 个变量（因子集、正则化、参数量）。需要先做控制变量实验 A 才能下结论。
