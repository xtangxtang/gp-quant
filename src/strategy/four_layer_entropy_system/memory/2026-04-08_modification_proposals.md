# 四层熵交易系统修改建议

**日期**: 2026-04-08  
**依据**: `docs/complexity_research_12_papers_conclusion.md`（12 篇论文统一结论）、`docs/complexity_theory_notes.md`（理论笔记与研发路线图）、`memory/entropy_backtest_results.md`（分钟级回测 -100% 结果）、`src/strategy/entropy_bifurcation_setup/`（基线特征引擎）

---

## 一、数据层面：从分钟转日线（P0，最优先）

`memory/entropy_backtest_results.md` 已证实分钟级熵因子**完全无效（总收益 -100%）**：1719 笔交易、胜率 35.4%、交易成本致命。但 `core_system.py` 的 `load_data()` 仍在读分钟级 CSV（列名"时间/收盘/成交量"）。

**建议**：让四层系统直接复用 `entropy_bifurcation_feature_engine.build_entropy_bifurcation_feature_frame()` 的日线特征输出，而不是自己从分钟数据重新计算。基线特征引擎已有 30+ 个日线特征（含 `entropy_accel_5`、`coarse_entropy_lb_20`、`experimental_tda_score` 等），四层系统不需要重复造轮子。

**这是根本性问题——在分钟数据上跑这套系统，无论怎么调参都不会有效。**

---

## 二、个股状态层（Layer 2）：补缺特征（P1）

当前 `stock_state.py` 只用了 4 个核心特征。对照文档建议和基线引擎已有的字段：

| 特征 | 基线引擎已有 | 四层系统已有 | 文档建议优先级 |
|------|:---:|:---:|:---:|
| `path_irreversibility_20` | ✅ | ✅ | 1 |
| `dominant_eig_20` | ✅ | ✅ | 2 |
| `perm_entropy_20_norm` | ✅ | ✅ | — |
| `phase_adjusted_ar1_20` | ✅ | ✅ | — |
| `entropy_accel_5` | ✅ | ❌ | 3 |
| `coarse_entropy_lb_20` | ✅ | ❌ | — |
| `entropy_gap` | ✅ | ❌ | — |
| `entropy_percentile_120` | ✅ | ❌ | — |
| `var_lift_10_20` | ✅ | ❌ | — |
| `mse_slope_20_60` | ❌ | ❌ | 4（多尺度熵斜率）|

### 具体建议

1. **`entropy_accel_5`**（熵的二阶变化率）应纳入 `bifurcation_quality` 计算——12 篇论文结论明确要求识别"结构切换速度"。
2. **`entropy_gap`** 和 **`entropy_percentile_120`** 应纳入 `entropy_quality`——基线引擎的五门条件已证明 `compression_state` 需要 `entropy_percentile ≤ 0.40 AND entropy_gap ≥ 0.015`，四层系统缺了这层过滤。
3. **`var_lift_10_20`** 应纳入 `bifurcation_quality`——方差抬升是临界减速的经典指标，基线模型给了 0.10 权重。

---

## 三、Market Gate（Layer 1）：加入周期去偏（P2）

论文 12（Statistical warning indicators for abrupt transitions with slow periodic forcing）**直接修正了 early warning signals 的乐观假设**。当前 `market_gate.py` 的 `compute_phase_state()` 仅做了 coupling × noise 的联合判断矩阵，没有任何季节/财报/节假日周期校正。

### 具体建议

1. 在 `compute_phase_state()` 中加入 A 股日历效应因素（春节前后、两会、季报窗口期、国庆），对应 `market_phase_weights` 配置项已存在但未使用。
2. 对 `coupling_entropy` 做 phase-aware 校正：同一季节、同一季报阶段的历史基线去偏，避免把"正常的季报同步波动"误判为系统性耦合失稳。

---

## 四、执行成本层（Layer 3）：Strategic Abandonment 阈值偏保守（P1）

当前 `execution_cost.py` 的 `abandonment_score > 0.7` 才触发跳过，但论文 9（Communication-Induced Bifurcation）的核心结论是：**高噪声环境中"少做"几乎总是比"更用力做"更优**，且保守策略最多只差最优的 2 倍（论文 3）。

### 具体建议

1. 将 abandonment 阈值从 **0.7 降到 0.6**，更早触发"战略性放弃"。
2. `determine_entry_mode()` 中 `signal_strength < 0.5` 时当前用 `probe`（25% 仓位），建议改为 **`skip`**——回测显示低信号交易的手续费+滑点累积是主要亏损来源。
3. 止盈/止损参数（+8%/-5%）与回测（+5%/-3%）不一致。回测显示 +5% 止盈有 100% 胜率但触发太少，建议维持 +8% 但增加**分段止盈**（+4% 减半仓、+8% 清仓）。

---

## 五、实验层（Layer 4）：权重保持 0%（P3）

当前三个实验模块都是近似替代品：

- **TDA**：用相关矩阵特征值熵而非真正的持久同调
- **Reservoir**：用固定随机稀疏矩阵（100 节点）而非论文推荐的 500+ 节点
- **Structure**：手写规则而非学习型状态模型

文档建议权重保持 0%，只做观察层——这一点当前实现是正确的。

### 具体建议

1. 保持权重 = 0 不变。
2. 将实验层的计算结果**写入日志而非参与决策**，用于离线分析这些指标是否在 out-of-sample 中有增量信息。
3. 未来升级路径：用 `gudhi` 或 `ripser` 实现真正的 TDA（README 中已提及但未执行）。

---

## 六、权重体系与决策逻辑（P2）

当前权重：stock_state **70%** / market_gate **20%** / execution **10%** / experimental **0%**。

`compute_final_action` 的实际逻辑是布尔门控而非连续加权——`gate_closed` 或 `execution_abandoned` 直接 → `wait`。这意味着 market_gate 的实际影响力远大于 20% 权重所暗示的，实际上是**硬性一票否决**。

### 具体建议

1. 布尔门控逻辑本身是正确的，符合"市场门控层"的定位。但应把权重含义改清楚——当前 20% 权重只影响 confidence 计算，不影响 action 决策，容易误导。
2. 基线引擎的**五门条件**（compression + instability + phase + trigger + quality）比四层系统的连续分数更稳健——建议四层系统也引入类似的布尔门控，至少要求 `entropy_quality ≥ 0.62` 和 `bifurcation_quality ≥ 0.20` 作为硬性最低门槛。

---

## 七、优先级总结

| 优先级 | 修改项 | 预期收益 |
|:---:|---|---|
| **P0** | 数据层从分钟切日线，复用基线特征引擎 | 避免在已证伪的时间尺度上浪费计算 |
| **P1** | Layer 2 补入 `entropy_accel_5`、`entropy_gap`、`var_lift_10_20` | 对齐基线引擎已验证的特征集 |
| **P1** | Layer 3 降低 abandonment 阈值，弱信号直接 skip | 减少无效交易的成本侵蚀 |
| **P2** | Layer 1 加入日历周期去偏 | 减少 early warning 的相位假信号 |
| **P2** | Layer 2 增加硬性门槛条件 | 对齐基线的五门过滤逻辑 |
| **P3** | Layer 3 分段止盈 | 改善盈利交易的兑现率 |
| **P3** | Layer 4 输出改为纯日志 | 为未来 OOS 验证积累数据 |
