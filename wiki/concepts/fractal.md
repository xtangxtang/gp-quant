---
title: 分形与多尺度结构
tags: [fractal, multiscale, self-similarity]
confidence: high
status: active
created: 2026-04-12
updated: 2026-04-12
---

# 分形 (Fractal)

## 核心定义

价格序列在不同时间尺度上呈现**自相似性**：1 分钟 K 线的形态与日线、月线在统计上相似。

## 物理解释

市场参与者跨时间尺度分布：
- 高频程序 → 秒级
- 日内交易者 → 分钟/小时级
- 波段交易者 → 日/周级
- 机构 → 月/季级
- 长期基金 → 年级

这些嵌套的震荡时间框架创造了收益率的分形结构。

## Hurst 指数

$$H = \frac{\log(R/S)}{\log(n)}$$

- $H > 0.5$: 持续性（趋势跟踪有效）
- $H = 0.5$: 随机游走
- $H < 0.5$: 均值回复

在 [multitimeframe-scanner](../entities/multitimeframe-scanner.md) 的物理特征中使用 Hurst 指数评估趋势持续性。

## 在本项目中的应用

**多时间框架共振** ([multitimeframe-resonance](multitimeframe-resonance.md)) 就是分形思想的直接应用：
- 日线开始趋势 + 周线确认 + 月线不矛盾 → 共振信号
- 本质上是检测分形结构中的一致性

## 相关概念

- [multitimeframe-resonance](multitimeframe-resonance.md) — 多时间框架共振
- [entropy](entropy.md) — 不同尺度的熵度量
- [dissipative-structure](dissipative-structure.md) — 尺度间的能量传递
