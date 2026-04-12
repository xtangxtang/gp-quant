---
title: 核心熵计算模块
tags: [entropy, core-module, implementation]
confidence: high
status: active
created: 2026-04-12
updated: 2026-04-12
---

# Tick Entropy Module

**源码**: `src/core/tick_entropy.py`  
**文档**: `src/core/TICK_ENTROPY_MODULE_README.md`

## 功能

提供 5 个熵指标和 1 个状态分类器，是 [[four-layer-system]] 和 [[multitimeframe-scanner]] 的核心计算引擎。

## 5 个熵指标

| 函数 | 输出 | 理论来源 |
|------|------|----------|
| `path_irreversibility_entropy()` | KL 散度 ≥ 0 | Seifert 2025 |
| `permutation_entropy()` | [0, 1] 归一化 | Bandt & Pompe 2002 |
| `waiting_time_entropy()` | 交易间隔分布熵 | 信息论 |
| `turnover_rate_entropy()` | 换手率分布熵 | 信息论 |
| `dominant_eigenvalue_from_autocorr()` | $|\lambda_{dom}| \in [0, 1]$ | AR 伴随矩阵 |

## 状态分类器

```python
market_state_classifier(path_irrev, perm_entropy, turnover_entropy)
# → 'ordered' | 'weak_chaos' | 'strong_chaos' | 'critical'
```

组合三个指标判断当前市场微观状态。

## 主入口

```python
df_result = build_tick_entropy_features(
    df_tick, 
    windows={'path_irrev': 60, 'perm_entropy': 60, 'turnover': 60}
)
```

## 时间尺度警告

> ⚠️ **分钟级无效**: 回测证明熵因子在 240 分钟窗口下只有 35% 胜率。  
> ✅ **日线有效**: 需要在日线或周线级别应用。  
> 见 [[why-daily-not-minute]], [[entropy-backtest-minute]]

## 相关实体

- [[four-layer-system]] — 使用本模块进行股票状态评估
- [[multitimeframe-scanner]] — 物理特征计算中使用 Hurst 等指标
- [[hold-exit-system]] — 熵储备判断使用本模块输出

## 相关概念

- [[entropy]], [[permutation-entropy]], [[path-irreversibility]], [[dominant-eigenvalue]]
