---
title: 四层选股系统
tags: [strategy, entropy-bifurcation, four-layer]
confidence: high
status: active
created: 2026-04-12
updated: 2026-04-12
---

# 四层选股系统 (Four-Layer System)

**源码**: `src/strategy/entropy_bifurcation_setup/`  
**运行**: `scripts/run_entropy_bifurcation_setup.sh`

## 架构

基于 [12-papers-synthesis](../sources/12-papers-synthesis.md) 的 8 个共识，实现 4 层渐进式选股：

```
Layer 1: 市场门 (Market Gate)
    ↓ 通过
Layer 2: 股票状态 (Stock State)
    ↓ 通过
Layer 3: 执行成本 (Execution Cost)
    ↓ 通过
Layer 4: 实验层 (Experimental) — 权重 0%
```

## Layer 1 — 市场门

评估整体市场是否适合交易。

| 输入 | 含义 |
|------|------|
| `coupling_entropy` | 市场整体耦合熵 |
| `noise_cost` | 噪声成本（信息处理代价） |

| 输出状态 | 含义 |
|----------|------|
| `abandon` | [strategic-abandonment](../concepts/strategic-abandonment.md) — 完全不交易 |
| `distorted` | 市场扭曲，极度谨慎 |
| `expansion` | 扩张期 |
| `transition` | 转变期 |
| `compression` | 压缩期 |
| `neutral` | 中性 |

## Layer 2 — 股票状态

对通过市场门的个股，评估其微观结构。

| 指标 | 来源 | 含义 |
|------|------|------|
| `path_irreversibility_20` | [path-irreversibility](../concepts/path-irreversibility.md) | 20 日不可逆性 |
| `dominant_eig_20` | [dominant-eigenvalue](../concepts/dominant-eigenvalue.md) | 20 日主特征值 |
| `perm_entropy` | [permutation-entropy](../concepts/permutation-entropy.md) | 置换熵 |
| `phase_adjusted_ar1_20` | [bifurcation](../concepts/bifurcation.md) | 相位校正 AR(1) |

输出三个质量分数：
- `entropy_quality`: 熵质量
- `bifurcation_quality`: 分岔质量
- `trigger_quality`: 触发质量

## Layer 3 — 执行成本

根据噪声水平调整仓位。

| 输出 | 含义 |
|------|------|
| `execution_cost_state`: normal/cautious/blocked | 执行可行性 |
| `entry_mode`: skip/probe/staged/full | 入场方式 |
| `position_scale`: 25%/50%/80% | 仓位比例 |

## Layer 4 — 实验层

TDA（拓扑数据分析）、储层计算、结构约束潜变量。

> 当前权重 = 0%，纯实验性质。  
> 共识 #5: TDA 和储层计算适合作为**门控/过滤器**，不适合作为主信号。

## 相关实体

- [tick-entropy-module](tick-entropy-module.md) — 核心计算引擎
- [multitimeframe-scanner](multitimeframe-scanner.md) — 互补策略
- [hold-exit-system](hold-exit-system.md) — 买入后的持仓管理

## 相关概念

- [entropy](../concepts/entropy.md), [bifurcation](../concepts/bifurcation.md), [strategic-abandonment](../concepts/strategic-abandonment.md), [path-irreversibility](../concepts/path-irreversibility.md)
