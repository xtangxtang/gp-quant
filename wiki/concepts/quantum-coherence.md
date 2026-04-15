---
title: 量子相干性与市场状态坍缩
tags: [quantum, coherence, density-matrix, decoherence, bifurcation]
confidence: medium
status: experimental
sources: [baaquie-quantum-finance, busemeyer-quantum-cognition]
created: 2026-04-14
updated: 2026-04-14
open-questions:
  - 密度矩阵的最佳构造方式（特征向量投影 vs softmax 概率）
  - coherence_decay_rate 的最优窗口（5日 vs 10日）
  - 与现有 bifurcation_quality 的最佳融合权重需回测确认
---

# 量子相干性与市场状态坍缩

## 核心思想

借用量子力学的**密度矩阵形式**来编码市场的"多空未决"状态，比传统的离散状态分类器（`ordered | weak_chaos | strong_chaos | critical`）提供更多信息。

关键新增量：**非对角元的衰减速率**——度量市场从"多空叠加"到"方向确认"的收敛速度。

## 与现有框架的关系

### 量子概念 → 已有系统的映射

| 量子力学概念 | 金融市场含义 | 已有系统对应 | 新增信息 |
|-------------|------------|------------|---------|
| 叠加态 (superposition) | 多空力量共存，方向未决 | Phase 1: 惜售吸筹 | 对角元概率分布 |
| **相干性 (coherence)** | **状态之间的"不确定性程度"** | **无直接对应** | **非对角元 ρ_ij** |
| 波函数坍缩 (collapse) | 方向确认，趋势启动 | Phase 2: 分岔突破 | 坍缩速度 |
| 退相干 (decoherence) | 共识瓦解，趋势消散 | Phase 3: 结构崩塌 | 退相干速率 |
| 纯态 (pure state) | 市场高度一致 | perm_entropy 极低 | purity = Tr(ρ²) |
| 混合态 (mixed state) | 市场极度分裂 | perm_entropy 极高 | von Neumann 熵 |

### 关键区别

> **已有指标告诉你"系统在哪"，密度矩阵告诉你"系统正以多快的速度变化"。**
>
> - `market_state_classifier` → 离散标签（当前状态）
> - `density_matrix` → 连续分布（状态概率 + 状态间相干性）
> - `coherence_decay_rate` → 变化速率（方向确认的速度）

## 数学框架

### 1. 密度矩阵构造

定义 4 个基态（与 `market_state_classifier` 一致）：

$$|0\rangle = |\text{compressed}\rangle, \quad |1\rangle = |\text{transitioning}\rangle, \quad |2\rangle = |\text{trending}\rangle, \quad |3\rangle = |\text{chaotic}\rangle$$

从每日的特征向量 $(PE, PI, DE)$ 构造状态向量 $|\psi_t\rangle$：

1. 用 softmax 将特征映射为基态的概率幅
2. 在滚动窗口内取混合态密度矩阵：

$$\rho = \frac{1}{N} \sum_{t=1}^{N} |\psi_t\rangle\langle\psi_t|$$

### 2. 核心指标公式

**l1-范数相干性度量**（Baumgratz et al. 2014）：

$$C_{l_1}(\rho) = \sum_{i \neq j} |\rho_{ij}|$$

- 范围: $[0, d-1]$，其中 $d=4$
- 归一化后: $[0, 1]$

**纯度 (Purity)**：

$$\gamma = \text{Tr}(\rho^2) \in [1/d, 1]$$

- $\gamma = 1$: 纯态（市场高度一致）
- $\gamma = 1/d$: 最大混合态

**von Neumann 熵**：

$$S(\rho) = -\text{Tr}(\rho \ln \rho) = -\sum_i \lambda_i \ln \lambda_i$$

- 推广了 Shannon 熵
- $S = 0$: 纯态；$S = \ln d$: 最大混合态

**相干衰减速率**：

$$\dot{C} = \frac{dC_{l_1}}{dt} \approx \frac{C_{l_1}(t) - C_{l_1}(t-\Delta t)}{\Delta t}$$

- $\dot{C} > 0$: 被动退相干/方向正在确认
- $\dot{C} < 0$: 不确定性增加
- $|\dot{C}|$ 大: 状态快速变化

## 集成策略（推荐）

### 不单独建策略的原因

1. **80% 代码重复**：数据加载、基础特征、回测引擎与 `entropy_accumulation_breakout` 完全相同
2. **信号源重叠**：密度矩阵的对角元与 `market_state_classifier` 等价
3. **唯一新信息是变化速率**：值得作为增强因子，不值得撑起独立策略

### 集成方案

```
src/core/quantum_coherence.py          ← 新增核心模块
src/core/tick_entropy.py               ← 已有，不动
src/strategy/entropy_accumulation_breakout/
    feature_engine.py                  ← 增加 coherence 列
    signal_detector.py                 ← 用 coherence_decay_rate 增强 bifurcation_quality
```

#### bifurcation_quality 增强

```python
# 原始权重:
# 0.35 × norm_dom_eig + 0.30 × norm_vol_impulse + 0.20 × (1-norm_entropy) + 0.15 × norm_path_irrev

# 增强后:
# 0.25 × norm_dom_eig
# + 0.25 × norm_vol_impulse
# + 0.15 × (1-norm_entropy)
# + 0.15 × norm_path_irrev
# + 0.20 × norm_coherence_decay_rate    ← 退相干速度越快，突破质量越高
```

#### Phase 判定增强

| Phase | 现有条件 | + 量子相干增强 |
|-------|---------|--------------|
| Accumulation | PE < 0.65, path_irrev > 0.05 | + purity > 0.6（状态纯度高 = 筹码集中） |
| Breakout | dom_eig > 0.85, vol_impulse > 1.8 | + coherence_decay_rate > 阈值（退相干加速 = 方向确认中） |
| Collapse | PE > 0.90, path_irrev < 0.01 | + purity < 0.3（状态极度混合 = 共识瓦解） |

### 何时考虑独立策略

只有回测证明 `coherence_decay_rate` 有独立 alpha 时（即：单独用退相干速率选股的胜率 > 基线），才值得拆出为独立策略。

## 物理直觉类比

| 物理过程 | 市场过程 | 可观测指标 |
|---------|---------|-----------|
| 粒子在势阱中随机运动 | 股价在箱体内震荡 | 高 perm_entropy, 低 path_irrev |
| 量子隧穿穿越势垒 | 价格突破关键阻力位 | dom_eig → 1, vol_impulse 放大 |
| 测量导致波函数坍缩 | 大单突破确认方向 | coherence_decay_rate 骤增 |
| 退相干使量子态变经典 | 趋势共识形成 | purity → 1, von_neumann_entropy → 0 |
| 热涨落破坏相干 | 获利盘分歧/散户涌入 | purity → 1/d, coherence 回升 |

## 参考文献

- Baaquie, B.E. (2004). *Quantum Finance: Path Integrals and Hamiltonians for Options and Interest Rates*. Cambridge University Press.
- Busemeyer, J.R. & Bruza, P.D. (2012). *Quantum Models of Cognition and Decision*. Cambridge University Press.
- Baumgratz, T., Cramer, M. & Plenio, M.B. (2014). Quantifying Coherence. *Physical Review Letters*, 113(14), 140401.
- Haven, E. & Khrennikov, A. (2013). *Quantum Social Science*. Cambridge University Press.

## 实现位置

- 核心模块: `src/core/quantum_coherence.py`
- 依赖: `src/core/tick_entropy.py`（permutation_entropy, path_irreversibility_entropy, dominant_eigenvalue_from_autocorr）
