---
title: 主特征值与临界减速
tags: [bifurcation, eigenvalue, critical-slowing-down]
confidence: high
status: active
sources: [period-doubling-eigenvalue]
created: 2026-04-12
updated: 2026-04-12
---

# 主特征值 (Dominant Eigenvalue)

## 核心思想

从价格序列的自相关结构中提取 AR 伴随矩阵的**主特征值** $\lambda_{dom}$：

$$|\lambda_{dom}| \to 1 \implies \text{系统逼近不稳定边界（分岔临近）}$$

这比简单的 AR(1) 系数更加鲁棒，因为它捕获了完整的自相关结构。

## 为什么比 AR(1) 更好

| 方法 | 问题 |
|------|------|
| AR(1) + 方差 | 过于简化；周期性背景下产生假信号 |
| 主特征值 | 包含高阶自相关信息；可检测周期倍化分岔 |
| 相位校正主特征值 | 去除季节/财报周期影响后更准确 |

## 实现

```python
# src/core/tick_entropy.py
dominant_eigenvalue_from_autocorr(series, p=5)
```

步骤：
1. 计算 $p$ 阶自相关系数
2. 构建 AR 伴随矩阵
3. 求特征值
4. 返回模最大的特征值 $|\lambda_{dom}|$

## 在本项目中的应用

| 系统 | 用途 | 阈值 |
|------|------|------|
| [[four-layer-system]] 第二层 | `dominant_eig_20`（20 日窗口） | > 0.9 → 分岔临近 |
| [[tick-entropy-module]] | `market_state_classifier()` 输入之一 | 参与 `critical` 状态判定 |
| [[four-layer-system]] 第二层 | `phase_adjusted_ar1_20` 与之互补 | 去除周期性驱动 |

## 相关概念

- [[bifurcation]] — 主特征值检测的目标现象
- [[permutation-entropy]] — 与之互补的复杂度指标
- [[entropy]] — 低熵 + 高特征值 → 强信号
