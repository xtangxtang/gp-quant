---
title: 路径不可逆性
tags: [entropy, irreversibility, market-microstructure]
confidence: high
status: active
sources: [seifert-2025]
created: 2026-04-12
updated: 2026-04-12
---

# 路径不可逆性 (Path Irreversibility)

## 核心定义

在随机游走（纯效率市场）下，价格序列的前向和反向转移概率**完全对称**。

当路径不可逆性显著 > 0 时，说明存在**定向力量**（主力资金、政策驱动）在系统性地推动价格。

$$D_{KL}(P_{forward} \| P_{reverse}) > 0 \implies \text{主力控盘}$$

## 计算方法

1. 将收益率和成交流量离散化为三态：-1, 0, +1
2. 统计前向状态转移矩阵 $T_{forward}$
3. 构建反向转移矩阵 $T_{reverse}$
4. 计算 KL 散度

```python
# src/core/tick_entropy.py
path_irreversibility_entropy(states, window=60)
```

## 解读

| 值域 | 含义 |
|------|------|
| ≈ 0 | 可逆/随机，无方向性 |
| 0.1–0.3 | 弱不可逆，有一定主力参与 |
| > 0.3 | 强烈主力控盘信号 |

## 在本项目中的应用

- [[tick-entropy-module]]: `market_state_classifier()` 的核心输入
- [[four-layer-system]] 第二层: `path_irreversibility_20`（20 日窗口）参与 `entropy_quality` 计算
- 与 [[permutation-entropy]] 互补：路径不可逆检测方向性，置换熵检测复杂度

## 关键约束

> Seifert (2025) 证明：从粗粒化观测计算的不可逆性只是真实熵产生的**下界**。  
> 我们看到的是「至少这么不可逆」，实际可能更强。  
> 见 [[seifert-2025-entropy-bounds]]

## 相关概念

- [[entropy]] — 路径不可逆是熵的一种度量
- [[permutation-entropy]] — 互补指标
- [[strategic-abandonment]] — 低不可逆 + 高噪声 → 放弃
