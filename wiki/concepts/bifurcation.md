---
title: 分岔与临界转变
tags: [bifurcation, critical-transition, early-warning]
confidence: high
status: active
sources: [period-doubling-eigenvalue, reservoir-computing-tipping, statistical-warning-indicators]
created: 2026-04-12
updated: 2026-04-12
open-questions:
  - Hopf/周期倍化/折叠分岔在金融中只是类比，精确分类尚不成熟
---

# 分岔 (Bifurcation)

## 核心定义

系统参数缓慢漂移到**临界点**时，系统状态发生**质变**——从一个吸引子跳到另一个。

在 A 股市场：横盘积累能量 → 突破临界阈值 → 趋势启动（牛市/熊市吸引子切换）。

## 临界减速 (Critical Slowing Down)

分岔前最重要的预警信号：

- 系统恢复均衡的速度变慢
- 自相关增强，方差增大
- 数学表达：主特征值 $|\lambda_{dom}| \to 1$

> **AR(1) + 方差不够！**  
> 论文证明：单一 AR(1) 系数过于简化，需要从自相关结构提取 [dominant-eigenvalue](dominant-eigenvalue.md)。  
> 见 [period-doubling-eigenvalue](../sources/period-doubling-eigenvalue.md)

## 周期性背景问题

> 传统 CSD 指标在**周期性驱动**下会失效（季节性、财报周期、节假日效应）。  
> 必须做**相位去趋势**：`phase_adjusted_ar1` 指标在 [four-layer-system](../entities/four-layer-system.md) 中实现。  
> 见源页 [period-doubling-eigenvalue](../sources/period-doubling-eigenvalue.md)

## 在本项目中的应用

| 系统 | 用法 |
|------|------|
| [four-layer-system](../entities/four-layer-system.md) 第二层 | `dominant_eig_20` > 0.9 → 分岔临近，触发信号 |
| [tick-entropy-module](../entities/tick-entropy-module.md) | `dominant_eigenvalue_from_autocorr()` 提取 AR 伴随矩阵特征值 |
| [multitimeframe-scanner](../entities/multitimeframe-scanner.md) | 物理特征中的 `phase` 指标 |

## 6 个「不可直接套用」之一

> Hopf / 周期倍化 / 折叠分岔只是**类比**，不是精确匹配。  
> 金融市场有突然跳跃 + 反身性 ≠ 缓慢参数漂移。  
> 见 [12-papers-synthesis](../sources/12-papers-synthesis.md)

## 相关概念

- [dominant-eigenvalue](dominant-eigenvalue.md) — 主特征值的详细说明
- [entropy](entropy.md) — 熵降低预示分岔
- [dissipative-structure](dissipative-structure.md) — 相变理论基础
- [path-irreversibility](path-irreversibility.md) — 分岔后路径不可逆增强
