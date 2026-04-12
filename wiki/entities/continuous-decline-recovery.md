---
title: 连续下跌恢复策略
tags: [strategy, recovery, decline]
confidence: medium
status: active
created: 2026-04-12
updated: 2026-04-12
---

# 连续下跌恢复策略 (Continuous Decline Recovery)

**源码**: `src/strategy/continuous_decline_recovery/`

## 核心理念

> 不在暴跌底部抄底，而在**第一次可持续性恢复**时买入。

## 三层架构

### Layer 1 — 市场层
检测全市场是否经历了连续性卖出，以及恢复是否正在进行。

### Layer 2 — 行业层
哪些行业受伤早但修复快？选择领先恢复的行业。

### Layer 3 — 个股层
在恢复行业中，选择：
- 确实受损过（不是没跌的）
- 正在恢复（从低点反弹 3%-15%）
- 处于早期窗口（不是已经涨太多）
- 未过热

## 市场状态机

```
no_setup → selloff → repair_watch → buy_window → rebound_crowded
```

只在 `buy_window` 状态时交易。

## 关键参数

| 参数 | 默认值 |
|------|--------|
| 市场回望天数 | 6 |
| 最小行业成员数 | 4 |
| 最小反弹幅度 | 3% |
| 最大反弹幅度 | 15% |
| 前哨行业数 | 6 |

## 相关实体

- [data-pipeline](data-pipeline.md) — 数据源
- [hold-exit-system](hold-exit-system.md) — 买入后的持仓管理

## 相关概念

- [dissipative-structure](../concepts/dissipative-structure.md) — 恢复 = 新的有序结构形成
- [bifurcation](../concepts/bifurcation.md) — 从熊市吸引子向恢复吸引子的转变
