---
title: 多时间框架共振
tags: [multitimeframe, resonance, fractal]
confidence: high
status: active
created: 2026-04-12
updated: 2026-04-12
---

# 多时间框架共振 (Multi-Timeframe Resonance)

## 核心定义

当多个时间尺度的趋势信号**同时对齐**时，信号可靠性大幅提升。

```
日线开始趋势 + 周线确认方向 + 月线不矛盾 → 共振信号
```

这是 [fractal](fractal.md) 自相似性思想的直接应用：如果分形结构在多个尺度上一致，则趋势更可能持续。

## 物理特征（5 维）

[multitimeframe-scanner](../entities/multitimeframe-scanner.md) 为每个时间框架计算 5 个物理特征：

| 特征 | 含义 |
|------|------|
| Energy（能量） | 成交量 × 波动率，市场投入的能量 |
| Temperature（温度） | 波动率标准化度量 |
| Order（有序度） | 基于 Hurst 指数的趋势有序程度 |
| Phase（相位） | 当前处于趋势周期的哪个阶段 |
| Switch（切换） | 相位切换信号 |

## 共振评分

```python
resonance_score = weighted_sum(daily_score, weekly_score, monthly_score)
```

- `resonance_threshold = 0.22`: 最低共振分数
- `resonance_min_count = 2`: 至少 2 个时间框架给出正向信号

## 生产参数

| 参数 | 默认值 |
|------|--------|
| `top_n` | 30 |
| `hold_days` | 5 |
| `max_positions` | 10 |
| `max_positions_per_industry` | 2 |
| `min_amount` | 500,000 |
| `min_turnover` | 1.0% |

## 相关概念

- [fractal](fractal.md) — 理论基础：多尺度自相似性
- [entropy](entropy.md) — 每个时间框架的熵度量
- [multitimeframe-scanner](../entities/multitimeframe-scanner.md) — 实现此概念的系统
