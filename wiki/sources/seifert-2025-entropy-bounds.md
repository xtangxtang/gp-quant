---
title: "Seifert (2025) — 粗粒化熵产生下界"
tags: [entropy, thermodynamics, coarse-graining]
confidence: high
status: active
source_file: docs/papers/entropy_bounds_coarse_grained.pdf
created: 2026-04-12
updated: 2026-04-12
---

# Seifert (2025) — Entropy Production Bounds from Coarse-Grained Trajectories

## 核心结论

从**部分可观测**（粗粒化）的轨迹中，只能计算出真实熵产生的**下界**。

## 对本项目的意义

1. 我们用日 K 线数据计算的 `path_irreversibility` 是真实市场微观不可逆性的**下界**
2. 这意味着：如果下界已经显著 > 0，真实的主力控盘程度**至少**这么强
3. 但也意味着：我们的测量会**漏报**（false negative）一些实际存在的不可逆性

## 关键公式思想

$$\sigma_{obs} \leq \sigma_{true}$$

$\sigma_{obs}$: 从日线数据可观测的熵产生  
$\sigma_{true}$: 包含所有 tick 级信息的真实熵产生

## 实际影响

| 影响 | 说明 |
|------|------|
| 信号设计 | `path_irreversibility` 是保守估计，阈值不宜设太高 |
| 置信度 | 检测到的信号可信（至少这么不可逆），未检测到不代表没有 |
| 时间尺度 | 更细粒度的数据 → 更紧的下界 → 但分钟级回测已证明无效 ([why-daily-not-minute](../decisions/why-daily-not-minute.md)) |

## 共识映射

→ 共识 #1: 市场熵 = 不可逆性代理，非真正热力学量  
→ 见 [12-papers-synthesis](12-papers-synthesis.md)

## 概念链接

- [entropy](../concepts/entropy.md), [path-irreversibility](../concepts/path-irreversibility.md), [tick-entropy-module](../entities/tick-entropy-module.md)
