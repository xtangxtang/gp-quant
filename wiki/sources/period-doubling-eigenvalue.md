---
title: "主特征值检测周期倍化分岔 (2026)"
tags: [bifurcation, eigenvalue, critical-slowing-down]
confidence: high
status: active
source_file: docs/papers/period_doubling_dominant_eigenvalue.pdf
created: 2026-04-12
updated: 2026-04-12
---

# Period-Doubling Bifurcation via Dominant Eigenvalue (2026)

## 核心结论

1. AR(1) + 方差作为早期预警信号（EWS）**过于简化**
2. 从自相关结构提取的**主特征值** $|\lambda_{dom}|$ 更准确
3. 可检测周期倍化分岔（而非仅 fold bifurcation）
4. 在周期性背景噪声下，需要**相位去趋势**才能避免假信号

## 方法

1. 对时间序列拟合 $p$ 阶 AR 模型
2. 构建 $p \times p$ 伴随矩阵
3. 计算特征值
4. 追踪 $|\lambda_{dom}|$ 随时间的变化

$$|\lambda_{dom}| \to 1 \implies \text{临界减速 → 分岔临近}$$

## 对本项目的影响

| 影响 | 实现 |
|------|------|
| 核心指标 | `dominant_eigenvalue_from_autocorr()` in [tick-entropy-module](../entities/tick-entropy-module.md) |
| 四层系统 | `dominant_eig_20` 是第二层的关键输入 ([four-layer-system](../entities/four-layer-system.md)) |
| 相位校正 | `phase_adjusted_ar1_20` 去除周期性驱动 |
| 阈值 | > 0.9 标记为分岔临近 |

## 共识映射

→ 共识 #3: AR(1) + 方差不足以做真正的 EWS  
→ 共识 #4: 所有 EWS 必须处理周期性背景  
→ 见 [12-papers-synthesis](12-papers-synthesis.md)

## 概念链接

- [dominant-eigenvalue](../concepts/dominant-eigenvalue.md), [bifurcation](../concepts/bifurcation.md), [permutation-entropy](../concepts/permutation-entropy.md)
