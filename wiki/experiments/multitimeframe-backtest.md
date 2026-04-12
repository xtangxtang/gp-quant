---
title: 多时间框架共振回测
tags: [experiment, backtest, multitimeframe]
confidence: medium
status: active
created: 2026-04-12
updated: 2026-04-12
open-questions:
  - 前瞻回测结果的系统性分析待补充
---

# 多时间框架共振回测

**输出**: `results/multitimeframe_resonance/`  
**系统**: [multitimeframe-scanner](../entities/multitimeframe-scanner.md)

## 方法

- 全市场扫描，选出 Top 30 共振股票
- 等权持有 5 天后轮换
- 最多 10 个持仓，每行业最多 2 个

## 输出文件

- `forward_backtest_daily_<date>.csv` — 日净值曲线
- `forward_backtest_trades_<date>.csv` — 逐笔交易

## 状态

前瞻回测（forward backtest）结果持续产出中，系统性绩效分析待整理。

## 相关实体

- [multitimeframe-scanner](../entities/multitimeframe-scanner.md) — 被测系统

## 相关概念

- [multitimeframe-resonance](../concepts/multitimeframe-resonance.md) — 理论基础
- [fractal](../concepts/fractal.md) — 多尺度一致性
