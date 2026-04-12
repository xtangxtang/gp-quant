---
title: 四层系统 2025 回测
tags: [experiment, backtest, four-layer]
confidence: medium
status: active
created: 2026-04-12
updated: 2026-04-12
open-questions:
  - 各版本 (v1-v4) 的详细对比结果待整理
---

# 四层系统 2025 年回测

**输出**: `results/four_layer_backtest_2025*/`  
**脚本**: `scripts/backtest_four_layer_2025.py`

## 版本演进

| 版本 | 目录 | 特点 |
|------|------|------|
| v1 | `four_layer_backtest_2025/` | 基础四层系统 |
| v2 | `four_layer_backtest_2025_v2/` | 参数优化 |
| v3 | `four_layer_backtest_2025_v3_entropy_exit/` | 加入熵退出信号 |
| v4 | `four_layer_backtest_2025_v4_pure_entropy/` | 纯熵驱动入场/退出 |

## 状态

回测结果存在，但详细分析报告待整理。

## 相关实体

- [[four-layer-system]] — 被测系统
- [[tick-entropy-module]] — 核心计算

## 相关实验

- [[entropy-backtest-minute]] — 分钟级失败促使转向日线
