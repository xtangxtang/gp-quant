---
title: 熵惜售分岔突破策略
tags: [strategy, entropy, bifurcation, dissipative-structure, accumulation, breakout]
confidence: high
status: active
created: 2026-04-13
updated: 2026-04-13
---

# 熵惜售分岔突破策略 (Entropy-Accumulation-Breakout)

**源码**: `src/strategy/entropy_accumulation_breakout/`  
**运行**: `scripts/run_entropy_accumulation_breakout.sh`  
**详细文档**: `src/strategy/entropy_accumulation_breakout/README.md`

## 核心思想

用信息熵检测惜售吸筹，用分岔理论检测突破，用耗散结构理论检测趋势衰竭退出。三个阶段对应一笔完整交易的生命周期。

## 三阶段状态机

```
idle → accumulation → breakout → hold → collapse
         (买入准备)     (买入)           (卖出)
```

### Phase 1 — 惜售吸筹 (Accumulation)

大资金持有者不愿卖出 → 换手率下降 → 流动性收缩 → 价格序列变得有序（局部熵降）。

检测条件（至少满足 N-1 个，持续 ≥ 5 天）：

| 条件 | 阈值 | 概念来源 |
|------|------|---------|
| 置换熵低位 | `perm_entropy_m < 0.65` | [permutation-entropy](../concepts/permutation-entropy.md) |
| 路径不可逆性高 | `path_irrev_m > 0.05` | [path-irreversibility](../concepts/path-irreversibility.md) |
| 成交量萎缩 | `vol_shrink < 0.7` | 流动性收缩 |
| 波动率压缩 | `vol_compression < 0.8` | 布林带收窄 |

物理解释：资本流入创造局部有序 → [dissipative-structure](../concepts/dissipative-structure.md) 正在形成。

### Phase 2 — 分岔突破 (Bifurcation Breakout)

能量积累到临界点 → 主特征值 → 1（临界减速）→ 放量打破对称性 → 新趋势形成。

检测条件：

| 条件 | 阈值 | 概念来源 |
|------|------|---------|
| 近期有惜售 | 过去 10 天内 accumulation = True | Phase 1 |
| 临界减速 | `dom_eig_m > 0.85` | [dominant-eigenvalue](../concepts/dominant-eigenvalue.md) |
| 量能脉冲 | `vol_impulse > 1.8` | 能量注入 |
| 价格位置 | `breakout_range > 0.8` | 区间高位 |
| 有序突破 | `perm_entropy_m < 0.75` | 排除噪声驱动 |

关键约束：突破时熵仍需保持低位——这是区分「有序突破」和「噪声假突破」的核心逻辑。

### Phase 3 — 结构崩塌退出 (Structural Collapse)

趋势是 [dissipative-structure](../concepts/dissipative-structure.md)，需要持续能量输入维持。能量停止 → 熵增主导 → 结构瓦解。

任意 2/4 信号同时触发即退出：

| 信号 | 阈值 | 含义 |
|------|------|------|
| 熵飙升 | `perm_entropy_m > 0.85` | 有序结构瓦解 |
| 不可逆性骤降 | `path_irrev_m < 0.02` | 主力撤离 |
| 熵加速 | `entropy_accel > 0.03` | 不可控的熵膨胀 |
| 量能衰竭 | `vol / peak_vol < 0.4` | 能量供给不足 |

安全网：持仓 > 20 天强制退出。

## 多时间框架确认

日线发出突破信号后，需通过周线确认：

| 周线条件 | 阈值 |
|---------|------|
| 周线置换熵 | `< 0.75` |
| 周线趋势 | 收盘价 ≥ 8 周均线 |

依据 [multitimeframe-resonance](../concepts/multitimeframe-resonance.md)：高频（1 分钟）噪声可制造日线级别的虚假有序性，周线过滤可排除此类伪信号。

## 特征体系

从日线/周线 OHLCV 数据提取 25+ 特征，9 大类：

1. **置换熵** — perm_entropy_{s,m,l}, entropy_slope, entropy_accel
2. **路径不可逆性** — path_irrev_{m,l}
3. **主特征值** — dom_eig_{m,l}
4. **换手率熵** — turnover_entropy_{m,l}
5. **波动率** — volatility_{m,l}, vol_compression, bbw, bbw_pctl
6. **成交量** — vol_ratio_s, vol_impulse, vol_shrink
7. **价格突破** — breakout_up, breakout_range
8. **资金流** — mf_cumsum_{s,m}, mf_impulse
9. **大单** — big_net_ratio, big_net_ratio_ma

依赖 [tick-entropy-module](tick-entropy-module.md) 的核心计算函数。

## 综合评分

$$\text{composite\_score} = 0.4 \times \text{accum\_quality} + 0.6 \times \text{bifurc\_quality}$$

权重偏向分岔质量，因为实际交易触发依赖突破的发生。

## 文件结构

| 文件 | 功能 |
|------|------|
| `feature_engine.py` | 特征引擎：25+ 特征，日线 + 周线双时间框架 |
| `signal_detector.py` | 三阶段状态检测器 + 质量评分 |
| `scan_service.py` | 全市场扫描 + 前瞻回测引擎 |
| `run_entropy_accumulation_breakout.py` | CLI 入口 |
| `README.md` | 详细策略文档（含数学公式与完整参数表） |

## 理论来源

- [fan-2025-irreversibility](../sources/fan-2025-irreversibility.md) — KLD 不可逆性检测金融不稳定
- [dmitriev-2025-self-organization](../sources/dmitriev-2025-self-organization.md) — 熵=控制参数，成交量=序参量
- [yan-2023-thermodynamic-bifurcation](../sources/yan-2023-thermodynamic-bifurcation.md) — 熵产生率预测分岔点
- [seifert-2025-entropy-bounds](../sources/seifert-2025-entropy-bounds.md) — 粗粒化熵产生下界
- Bielinskyi et al. (2025) — 加权置换熵预警市场崩溃
- Ardakani (2025) — 尾部加权熵检测泡沫

## 与其他策略的关系

| 策略 | 异同 |
|------|------|
| [four-layer-system](four-layer-system.md) | 共用 tick_entropy 模块；四层是多条件筛选，本策略是三阶段状态机 |
| [multitimeframe-scanner](multitimeframe-scanner.md) | 共用多时间框架理念；本策略更聚焦于熵的时序演化 |
| [hold-exit-system](hold-exit-system.md) | 退出逻辑类似（熵扩散 = 衰竭退出）；本策略增加了结构崩塌的多信号综合判定 |

## 概念链接

- [entropy](../concepts/entropy.md), [bifurcation](../concepts/bifurcation.md), [dissipative-structure](../concepts/dissipative-structure.md)
- [permutation-entropy](../concepts/permutation-entropy.md), [path-irreversibility](../concepts/path-irreversibility.md), [dominant-eigenvalue](../concepts/dominant-eigenvalue.md)
- [multitimeframe-resonance](../concepts/multitimeframe-resonance.md)
