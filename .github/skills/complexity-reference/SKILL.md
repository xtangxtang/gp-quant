---
name: complexity-reference
description: "查阅复杂性理论、经济物理学知识和12篇论文洞见。Use when: 复杂性理论, complexity theory, 熵, entropy, 分岔, bifurcation, Hurst 指数, 耗散结构, 相变, phase transition, 论文, papers, 物理模型, 非线性, fractal, 分形, 置换熵, permutation entropy, 理论基础。"
argument-hint: "搜索的概念，例如：熵、分岔前兆、Hurst 指数"
---

# 复杂性理论参考 (Complexity Reference)

查阅本项目的理论基础——复杂性理论、经济物理学概念、以及 12 篇核心论文的综合洞见。

## 使用场景

- 解释策略中某个物理指标的含义
- 查阅论文中特定概念的交易应用
- 理解策略设计背后的理论依据
- 在开发新策略时参考理论框架

## 参考文档

### 1. 复杂性理论笔记

```
docs/complexity_theory_notes.md
```

主要内容：
- 耗散结构与远离平衡态系统
- Shannon 熵与置换熵
- 分岔理论（Hopf 分岔、倍周期分岔）
- Hurst 指数（趋势/均值回归/随机）
- 分形几何与自相似性
- 临界减速（Critical Slowing Down）
- 物理特征在 A 股中的映射

### 2. 12 篇论文综合分析

```
docs/complexity_research_12_papers_conclusion.md
```

核心结论摘要：

| 论文主题 | 关键洞见 | 交易因子 |
|----------|----------|----------|
| 随机热力学（时变网络） | 市场不可逆性 ∝ 资源流动强度 | `path_irreversibility_20` |
| 部分可观测熵界 | 只能测量真实熵的下界 | 使用粗粒化下界代理 |
| 拓扑分岔检测 | Hopf/倍周期分岔有拓扑签名 | `experimental_tda_score` |
| 自相关倍周期 | 分岔前兆 = 主特征值 → -1 | `dominant_eig_20` |
| 储层计算早期预警 | 多指标动态融合 > 单指标 | 4 层融合 |
| 混沌 vs 不稳定性 | 高 Lyapunov ≠ 真混沌 | 区分局部不稳定 vs 黑天鹅 |
| 控制成本理论 | 高噪声 → 低频决策最优 | `strategic_abandonment` |
| PINN vs Neural ODE | 物理约束 > 纯学习 | 灰箱模型 |

### 3. 12 篇论文深度分析

```
docs/papers/12_papers_deep_analysis.md
```

逐篇详细分析、公式推导、交易映射。

## 核心概念速查

### 熵 (Entropy)
- **Shannon 熵**: $H = -\sum p_i \log p_i$，衡量收益率分布的不确定性
- **置换熵 (Permutation Entropy)**: 基于时间序列符号化的序列复杂度
- **低熵** = 有序收敛（趋势积蓄） → 买入前兆
- **高熵** = 混乱发散 → 避免入场

### Hurst 指数
- $H > 0.5$: 趋势性（persistent）
- $H = 0.5$: 随机游走
- $H < 0.5$: 均值回归（anti-persistent）
- 计算：对数-对数 MSD 斜率

### 分岔 (Bifurcation)
- **Hopf 分岔**: 稳定点 → 极限环振荡
- **倍周期分岔**: 周期翻倍 → 混沌前奏
- **检测**: 主特征值趋近 -1 → 分岔即将发生

### 物理五维特征
每个时间框架（D/W/M）计算 5 个物理量：
1. **energy** — 资金流强度
2. **temperature** — 换手率 + 波动性
3. **order** — 趋势结构（均线排列、突破）
4. **phase** — 熵 + Hurst（有序 vs 混沌）
5. **switch** — 混沌→有序的转换触发

### 市场三阶段假说
1. **低熵压缩** (ordering) — 能量积蓄
2. **分岔/临界减速** (phase transition precursor) — 相变前兆
3. **再序化** (rupture/expansion) — 趋势爆发
