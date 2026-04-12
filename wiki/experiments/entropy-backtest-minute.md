---
title: "分钟级熵因子回测（失败）"
tags: [experiment, backtest, entropy, failure]
confidence: high
status: active
created: 2026-04-12
updated: 2026-04-12
---

# 分钟级熵因子回测

**结果**: ❌ 失败  
**来源**: `memory/entropy_backtest_results.md`  
**输出**: `results/entropy_backtest/`

## 设置

- 50 只股票
- 2025 年 4-12 月
- 240 分钟（≈1 交易日）滑动窗口
- 基于熵的入场 / 临界转变退出

## 关键发现

| 指标 | 数值 | 结论 |
|------|------|------|
| 胜率 | 35% | ❌ 远低于 50% 随机水平 |
| 止盈退出 | 19%（100% 胜率） | ✅ 有效但触发太少 |
| 止损退出 | 27.9%（0% 胜率） | ❌ 止损形同虚设 |
| 状态退出 | 52.6%（31% 胜率） | ⚠️ state → strong_chaos 太频繁 |
| 换手 | 1,719 次 | 💀 交易成本致命 |

## 根本原因

1. 熵因子在 240 分钟窗口下**没有预测力**
2. 分钟级噪声太大，掩盖了真实的熵信号
3. 高换手率导致交易成本吞噬微薄利润
4. 止损位（-3%）在分钟级波动下轻易被触发

## 产出决策

→ [[why-daily-not-minute]]: 切换到日线/周线应用  
→ [[tick-entropy-module]]: 增加时间尺度警告

## 概念链接

- [[entropy]], [[permutation-entropy]], [[path-irreversibility]]
