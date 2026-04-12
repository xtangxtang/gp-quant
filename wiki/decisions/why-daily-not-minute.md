---
title: "决策：为什么选日线而非分钟线"
tags: [decision, timeframe, entropy]
confidence: high
status: active
created: 2026-04-12
updated: 2026-04-12
---

# 为什么选日线而非分钟线

## 决策

熵因子（路径不可逆性、置换熵、主特征值等）在**日线或周线**级别应用，不在分钟级别。

## 依据

### 实证 — 分钟级回测失败

[entropy-backtest-minute](../experiments/entropy-backtest-minute.md) 的 50 只股票 × 9 个月回测证明：
- 240 分钟窗口的熵因子只有 **35% 胜率**
- 交易成本致命（1,719 次交易）
- 状态退出触发太频繁（52.6%）

### 理论 — Seifert 粗粒化下界

[seifert-2025-entropy-bounds](../sources/seifert-2025-entropy-bounds.md) 证明粗粒化观测给出下界。日线 vs 分钟线：
- 日线是更高层的粗粒化 → 下界更松 → 但信号更稳定
- 分钟线更细但噪声更大 → 信噪比太低

### 实践 — 信息处理成本

[strategic-abandonment](../concepts/strategic-abandonment.md): 分钟级交易的信息处理成本（监控、决策频率、心理压力）远超日线。

## 影响

- [tick-entropy-module](../entities/tick-entropy-module.md) 窗口参数设为 20 天（日线）
- [four-layer-system](../entities/four-layer-system.md) 使用 `*_20` 后缀的日线指标
- [multitimeframe-scanner](../entities/multitimeframe-scanner.md) 基于日/周/月 K 线

## 开放问题

分钟级数据仍有价值：
- 可用于盘中**执行优化**（不是择时信号）
- 可提供 tick 级别的成交量分布信息
