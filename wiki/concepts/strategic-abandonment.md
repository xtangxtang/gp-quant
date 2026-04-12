---
title: 战略放弃
tags: [control-theory, noise, position-management]
confidence: high
status: active
sources: [communication-induced-bifurcation]
created: 2026-04-12
updated: 2026-04-12
---

# 战略放弃 (Strategic Abandonment)

## 核心思想

来自控制理论的洞见（见 [communication-induced-bifurcation](../sources/communication-induced-bifurcation.md)）：

> 在噪声环境中实现精确控制需要消耗**大量信息处理能量**。  
> 当噪声成本超过潜在收益时，**不交易是最优策略**。

这不是「认输」，而是资源配置的数学最优解。

## 在本项目中的实现

### [four-layer-system](../entities/four-layer-system.md) 第一层（市场门）

```
market_gate_state = abandon | distorted | expansion | transition | compression | neutral
```

当 `coupling_entropy` 高 + `noise_cost` 高 → `market_gate_state = 'abandon'` → 触发 `strategic_abandonment = True`

此时系统层面直接跳过所有交易，无论个股信号多强。

### [four-layer-system](../entities/four-layer-system.md) 第三层（执行成本）

即使市场门没有触发完全放弃，执行成本层也可能根据个股的噪声水平给出：
- `entry_mode = 'skip'`: 跳过
- `entry_mode = 'probe'`: 只用 25% 仓位试探
- `position_scale`: 根据噪声调整仓位大小

## 共识 #7

> 信息处理有真实成本；高噪声 = 战略放弃。  
> 见 [12-papers-synthesis](../sources/12-papers-synthesis.md)

## 相关概念

- [entropy](entropy.md) — 高熵 = 高噪声 → 考虑放弃
- [bifurcation](bifurcation.md) — 只在临界区域投入资源
- [four-layer-system](../entities/four-layer-system.md) — 战略放弃的系统实现
