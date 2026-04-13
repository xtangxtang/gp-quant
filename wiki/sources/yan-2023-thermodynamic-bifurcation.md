---
title: "Yan et al. (2023) — 热力学预测分岔与非平衡相变"
tags: [thermodynamics, bifurcation, entropy-production, phase-transition]
confidence: high
status: active
source_file: "Communications Physics 6:16"
created: 2026-04-13
updated: 2026-04-13
---

# Yan et al. (2023) — Thermodynamic and dynamical predictions for bifurcations and non-equilibrium phase transitions

**期刊**: Communications Physics, 6, 16  
**DOI**: 10.1038/s42005-022-01113-9

## 核心结论

1. **熵产生率在分岔点达到峰值**：可作为分岔预测指标
2. 热力学量（熵产生、自由能耗散）比纯动力学量更早发出分岔预警
3. 方法适用于鞍结点分岔、Hopf 分岔、跨临界分岔等多种类型

## 关键公式思想

在分岔点附近：

$$\dot{S}_{\text{prod}} \to \text{peak}$$

熵产生率（entropy production rate）达到极大值 → 系统即将从一个稳态跳到另一个稳态。

## 方法

- 基于 Fokker-Planck 方程分析随机动力系统
- 计算稳态概率分布的熵产生率
- 比较热力学预测 vs 传统动力学稳定性分析的时效性

## 对本项目的应用

→ 为 [entropy-accumulation-breakout](../entities/entropy-accumulation-breakout.md) Phase 2（分岔突破）提供理论依据：
- 在突破发生前，应该观察到路径不可逆性（熵产生代理）的上升
- 同时主特征值 → 1（动力学指标也接近临界）
- 两个指标同时满足 → 分岔信号更加可靠

→ 支撑了策略中**同时要求** `path_irrev > 0.05` **和** `dom_eig > 0.85` 的设计：
- `path_irrev` ≈ 热力学预测（不可逆性 / 熵产生代理）
- `dom_eig` ≈ 动力学预测（临界减速 / 自相关）
- 论文证明两者在分岔点附近同时达到极值

## 概念链接

- [bifurcation](../concepts/bifurcation.md), [entropy](../concepts/entropy.md)
- [path-irreversibility](../concepts/path-irreversibility.md), [dominant-eigenvalue](../concepts/dominant-eigenvalue.md)
