---
title: "Dmitriev et al. (2025) — 股票市场自组织到相变边缘"
tags: [self-organization, phase-transition, entropy, order-parameter, sandpile]
confidence: high
status: active
source_file: "Frontiers in Physics 12:1508465"
created: 2026-04-13
updated: 2026-04-13
---

# Dmitriev et al. (2025) — Self-organization of the stock exchange to the edge of a phase transition

**期刊**: Frontiers in Physics, 12, 1508465  
**全文**: https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2024.1508465

## 核心结论

1. **熵是控制参数**：信息熵决定了市场距离相变的远近
2. **成交量是序参量**：成交量是市场状态相变的关键可观测量
3. **沙堆模型**：市场像 Bak-Tang-Wiesenfeld 沙堆一样自组织到临界态（SOC）
4. **最有效的早期预警测度**：方差、AR(1)、峰度、偏度

## 理论框架

### 控制参数与序参量

```
控制参数: 信息熵 S
    ↓ 调节
序参量: 成交量 V
    ↓ 表征
系统状态: 有序 / 临界 / 混沌
```

当熵低于临界值时 → 成交量突然增大（类似相变中的序参量突变）→ 趋势形成。

### 自组织临界性 (SOC)

- 市场持续自发演化到「临界态边缘」
- 小的扰动可以引发大的级联反应（幂律分布）
- 类比沙堆：每粒沙（每笔交易）都可能触发雪崩（大幅波动）

### 早期预警信号

| 预警指标 | 原理 | 有效性 |
|---------|------|--------|
| 方差增大 | 临界减速 → 涨落增大 | ★★★ |
| AR(1) → 1 | 自相关增强 → 恢复变慢 | ★★★ |
| 峰度增大 | 尾部事件增多 | ★★ |
| 偏度变化 | 分布不对称 | ★★ |

## 对本项目的应用

→ 直接支撑了 [entropy-accumulation-breakout](../entities/entropy-accumulation-breakout.md) 的设计理念：
- **Phase 1**: 熵（控制参数）持续降低 → 系统接近相变
- **Phase 2**: 成交量（序参量）突变 → 相变发生（突破）
- **dom_eig 主特征值**: 对应 AR(1) → 1 的临界减速信号

→ 也支撑了 [dominant-eigenvalue](../concepts/dominant-eigenvalue.md) 作为分岔预测指标的有效性。

## 与本项目其他来源的关系

| 来源 | 关系 |
|------|------|
| [seifert-2025-entropy-bounds](seifert-2025-entropy-bounds.md) | 互补：Seifert 给出理论下界，Dmitriev 给出实证证据 |
| [fan-2025-irreversibility](fan-2025-irreversibility.md) | 互补：Fan 聚焦不可逆性，Dmitriev 聚焦整体相变 |
| [12-papers-synthesis](12-papers-synthesis.md) | 扩展了综合分析中缺少的 SOC 视角 |

## 概念链接

- [entropy](../concepts/entropy.md), [bifurcation](../concepts/bifurcation.md), [dissipative-structure](../concepts/dissipative-structure.md)
- [dominant-eigenvalue](../concepts/dominant-eigenvalue.md)
