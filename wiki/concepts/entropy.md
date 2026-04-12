---
title: 熵与市场无序度
tags: [entropy, thermodynamics, information-theory]
confidence: high
status: active
sources: [seifert-2025, entropy-bounds-paper, entropy-diagnostics-paper]
created: 2026-04-12
updated: 2026-04-12
open-questions:
  - 日线级别的最佳熵计算窗口（20 天 vs 60 天）仍需更多回测验证
---

# 熵 (Entropy)

## 核心定义

熵度量系统的**无序度**。在金融市场语境下：

- **低熵** = 有序 = 资本定向流入 = 趋势正在形成
- **高熵** = 无序 = 随机噪声 = 无方向性

## 关键区分

> **市场熵 ≠ 物理熵**  
> 我们计算的是不可逆性代理量（proxy），不是真实热力学熵产生。  
> Seifert (2025) 证明：粗粒化观测（日 K 线）只能给出熵产生的**下界**。  
> 见 [[seifert-2025-entropy-bounds]]

## 本项目中的熵指标

| 指标 | 含义 | 模块 |
|------|------|------|
| `path_irreversibility` | 前向/反向转移概率的 KL 散度 | [[tick-entropy-module]] |
| `permutation_entropy` | 基于序型的复杂度（Bandt & Pompe 2002） | [[tick-entropy-module]] |
| `waiting_time_entropy` | 交易间隔分布的熵 | [[tick-entropy-module]] |
| `turnover_rate_entropy` | 换手率分布的熵 | [[tick-entropy-module]] |
| `coupling_entropy` | 市场整体耦合熵（[[four-layer-system]] 第一层） | [[four-layer-system]] |

## 交易含义

- `permutation_entropy` 0.5–0.6 → 趋势状态（可交易）
- `permutation_entropy` > 0.9 → 随机游走（不可交易）
- `path_irreversibility` > 0.3 → 强烈主力控盘信号

## 8 个共识之一

> 市场熵 = 不可逆性代理，非真正热力学量。  
> 时变耦合结构 > 单一资产熵。  
> 见 [[12-papers-synthesis]]

## 相关概念

- [[path-irreversibility]] — 路径不可逆性
- [[permutation-entropy]] — 置换熵的详细说明
- [[bifurcation]] — 熵降低预示分岔临近
- [[dissipative-structure]] — 耗散结构中的熵流
- [[strategic-abandonment]] — 高熵时的最优退出策略
