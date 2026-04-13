---
title: "Fan et al. (2025) — KLD 不可逆性检测金融不稳定"
tags: [entropy, irreversibility, KLD, financial-instability]
confidence: high
status: active
source_file: "Entropy 27(4):402"
created: 2026-04-13
updated: 2026-04-13
---

# Fan et al. (2025) — Instability of Financial Time Series Revealed by Irreversibility Analysis

**期刊**: Entropy, 27(4), 402  
**全文**: https://www.mdpi.com/1099-4300/27/4/402

## 核心结论

1. **KLD（Kullback-Leibler 散度）时间不可逆性**在金融危机前显著上升，是有效的早期预警信号
2. 基于 DHVG（Directed Horizontal Visibility Graph）的 KLD 方法优于传统矩统计量（方差、偏度、峰度）
3. 滑动窗口方法能捕捉局部状态变化，适合实时监控

## 方法

- 将时间序列映射为有向水平可见图（DHVG）
- 计算前向和反向图的度分布
- 用 KLD 度量两个方向的不对称性 → 时间不可逆性

$$\text{KLD} = \sum_k P_{\text{in}}(k) \ln \frac{P_{\text{in}}(k)}{P_{\text{out}}(k)}$$

## 关键发现

| 发现 | 含义 |
|------|------|
| KLD 在市场不稳定前上升 | 不可逆性增强 = 市场偏离均衡 |
| KLD > 矩统计量 | 信息论方法的检测灵敏度更高 |
| 滑动窗口有效 | 可以做成实时指标 |
| 多市场验证 | 美股、欧股、亚股均有效 |

## 对本项目的应用

→ 直接支撑了 `rolling_path_irreversibility()` 函数的设计  
→ 在 [entropy-accumulation-breakout](../entities/entropy-accumulation-breakout.md) 策略中：
- **Phase 1 (惜售)**: 不可逆性上升 = 有定向力量在运作
- **Phase 3 (崩塌)**: 不可逆性骤降 = 定向力量消失（主力撤离）

## 概念链接

- [path-irreversibility](../concepts/path-irreversibility.md), [entropy](../concepts/entropy.md)
- [tick-entropy-module](../entities/tick-entropy-module.md)
