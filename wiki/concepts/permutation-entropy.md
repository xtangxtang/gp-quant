---
title: 置换熵
tags: [entropy, complexity, permutation]
confidence: high
status: active
sources: [bandt-pompe-2002]
created: 2026-04-12
updated: 2026-04-12
---

# 置换熵 (Permutation Entropy)

## 定义

Bandt & Pompe (2002) 提出的基于**序型**（ordinal pattern）的复杂度度量。

对长度为 $m$ 的滑动窗口，提取其排列模式（共 $m!$ 种可能），然后计算排列分布的 Shannon 熵：

$$H_{PE} = -\sum_{i=1}^{m!} p_i \log p_i$$

归一化后 $H_{PE} \in [0, 1]$。

## 优势

- **抗噪声**: 只比较相对大小，不受幅度影响
- **计算高效**: $O(n \cdot m)$
- **无参数假设**: 不需要假设数据分布

## 交易解读

| 值域 | 含义 | 操作 |
|------|------|------|
| 0.5–0.6 | 趋势状态，排列模式集中 | ✅ 可交易 |
| 0.6–0.8 | 弱混沌，有一定规律 | ⚠️ 谨慎 |
| > 0.9 | 接近随机游走，排列均匀分布 | ❌ 不交易 |

## 在本项目中的实现

```python
# src/core/tick_entropy.py
permutation_entropy(series, m=3, delay=1)
```

在 [[tick-entropy-module]] 中，`permutation_entropy` 是 `market_state_classifier()` 的三个输入之一。

在 [[four-layer-system]] 第二层中，`perm_entropy` 参与计算 `entropy_quality` 分数。

## 相关概念

- [[entropy]] — 熵的总体框架
- [[path-irreversibility]] — 另一个基于转移概率的熵指标
- [[dominant-eigenvalue]] — 与置换熵互补的分岔指标
