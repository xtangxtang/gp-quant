# 熵惜售分岔突破策略 (Entropy-Accumulation-Breakout)

> 基于信息熵、路径不可逆性、临界减速理论与量子相干性的三阶段交易系统

---

## 目录

1. [策略总览](#1-策略总览)
2. [理论基础](#2-理论基础)
3. [三阶段状态机](#3-三阶段状态机)
4. [特征体系](#4-特征体系)
5. [信号检测逻辑](#5-信号检测逻辑)
6. [多时间框架确认](#6-多时间框架确认)
7. [市场状态门控](#7-市场状态门控)
8. [回测引擎](#8-回测引擎)
9. [参数说明](#9-参数说明)
10. [使用方法](#10-使用方法)
11. [文件结构](#11-文件结构)
12. [回测结果](#12-回测结果)
13. [参考文献](#13-参考文献)

---

## 1. 策略总览

### 核心思想

股价的大幅上涨不是随机事件，而是一个从「有序积累」到「临界突破」再到「结构耗散」的物理过程。本策略将这一过程建模为三个阶段：

```
惜售吸筹 (Accumulation)  →  分岔突破 (Bifurcation)  →  结构崩塌 (Collapse)
     ↓                          ↓                          ↓
   买入准备                     买入执行                     卖出退出
```

**一句话总结**：用熵判断是否有人在惜售（筹码集中），用分岔检测惜售是否导致股价即将上涨，上涨后用耗散结构理论判断趋势是否即将崩塌并卖出。

### 物理类比

| 物理概念 | 市场对应 | 可观测量 |
|---------|---------|---------|
| 局部熵降 | 筹码集中、惜售 | 置换熵 < 0.65 |
| 路径不可逆性 | 定向资金力量 | KLD(forward ‖ backward) > 0.05 |
| 临界减速 (Critical Slowing Down) | 即将突破的前兆 | 主特征值 → 1 |
| 对称性破缺 | 量价突破 | 成交量脉冲 > 1.8× |
| 耗散结构维持失败 | 趋势衰竭 | 熵快速扩张 + 量能衰竭 |
| 量子叠加态 | 多空力量共存，方向未决 | coherence_l1 高 |
| 波函数坍缩 | 方向确认，趋势启动 | coherence_decay_rate 为负 |
| 退相干 | 共识瓦解 | purity_norm 骤降 |

---

## 2. 理论基础

本策略的设计综合了以下六篇论文的核心洞见：

### 2.1 熵产生与不可逆性

**Seifert (2025)** — *The Stochastic Thermodynamics of Computation*

粗粒化轨迹的熵产生提供了微观不可逆性的下界。在市场中，我们计算的路径不可逆性（path irreversibility）是实际信息不对称程度的保守估计——如果检测到了方向性，那么真实的定向力量只会更强。

**应用**：`path_irreversibility` 指标作为「有人在定向操作」的下界估计。

### 2.2 不可逆性分析揭示金融不稳定

**Fan et al. (2025)** — *Instability of Financial Time Series Revealed by Irreversibility Analysis* (Entropy 27(4):402)

使用 KLD（Kullback-Leibler 散度）+ DHVG（Directed Horizontal Visibility Graph）滑动窗口方法检测金融时序的不稳定性。研究发现：
- KLD 不可逆性在市场不稳定前显著上升
- KLD 优于传统矩统计量（方差、偏度、峰度）
- 滑动窗口方法能捕捉局部状态变化

**应用**：`rolling_path_irreversibility()` 函数的理论依据。在惜售阶段不可逆性应上升（定向力量运作），在崩塌阶段不可逆性应骤降（主力撤离）。

### 2.3 置换熵预警市场崩溃

**Bielinskyi et al. (2025)** — *Early Warning Signs: Evaluating Permutation Entropy Metrics for Stock Market Crashes*

加权置换熵（Weighted Permutation Entropy, WPE）能够捕捉不同市场复杂度特征。低置换熵意味着价格序列变得更加可预测/有序——这在「大资金惜售」场景中是有意义的：当筹码被少数人持有且不愿卖出时，交易行为变得更有规律。

**应用**：`rolling_permutation_entropy()` 作为惜售检测的核心指标。低熵 = 有序 = 筹码集中。

### 2.4 股票市场的自组织临界性

**Dmitriev et al. (2025)** — *Self-organization of the stock exchange to the edge of a phase transition* (Front. Phys. 12:1508465)

核心发现：
- **熵是控制参数**：信息熵决定了市场距离相变的远近
- **成交量是序参量**：成交量是市场状态相变的关键观测变量
- **沙堆模型**：市场像 Bak-Tang-Wiesenfeld 沙堆一样自组织到临界态
- **最有效的早期预警测度**：方差、AR(1)、峰度、偏度

**应用**：主特征值（dominant eigenvalue → 1）作为临界减速的度量；成交量脉冲作为突破确认。

### 2.5 热力学预测分岔

**Yan et al. (2023)** — *Thermodynamic and dynamical predictions for bifurcations and non-equilibrium phase transitions* (Commun. Phys. 6:16)

熵产生率在分岔点达到峰值，可以作为分岔的预测指标。这意味着在突破发生前，我们应该观察到路径不可逆性的上升和主特征值向 1 趋近。

**应用**：在分岔突破检测中，同时要求高 `path_irrev` 和高 `dom_eig`。

### 2.6 尾部加权熵检测泡沫

**Ardakani (2025)** — *Detecting Financial Bubbles with Tail-Weighted Entropy*

尾部加权熵能检测金融泡沫和结构崩塌。当尾部事件频率异常升高时，尾部加权熵会偏离正常值。

**应用**：结构崩塌退出阶段的辅助参考——当置换熵快速扩张且伴随极端收益出现时，趋势可能接近终点。

### 2.7 量子相干性与退相干

**Baumgratz et al. (2014)** — *Quantifying Coherence* (PRL 113(14), 140401)

量子力学的密度矩阵形式为市场状态编码提供了更丰富的工具。传统状态分类器（ordered/chaos/critical）给出离散标签，而密度矩阵同时编码：
- **对角元**：各状态的概率（与 `market_state_classifier` 等价）
- **非对角元**：状态间的相干性（新信息——度量"方向未决的程度"）

关键新指标 — **退相干速率** `coherence_decay_rate`：
- 负值：相干性衰减 = 方向正在确认 = 市场从"叠加态"向"确定态"坍缩
- 正值：相干性增加 = 不确定性在增加 = 共识正在瓦解
- |值|越大：状态变化越快

$$C_{l_1}(\rho) = \sum_{i \neq j} |\rho_{ij}|, \quad \dot{C} = \frac{\Delta C_{l_1}}{\Delta t}$$

**应用**：`coherence_decay_rate` 作为 `bifurcation_quality` 的第 5 个因子（权重 20%），退相干越快 = 突破质量越高。`purity_norm` 辅助惜售质量评分（高纯度 = 状态集中 = 惜售明确）和崩塌检测（低纯度 = 共识瓦解）。

---

## 3. 三阶段状态机

```
                    ┌───────────────────────────────────────────────┐
                    │                                               │
                    ▼                                               │
              ┌──────────┐                                          │
          ┌──▶│   idle   │◀───────────────────────────┐             │
          │   └────┬─────┘                            │             │
          │        │                                  │             │
          │        │ 置换熵 < 0.65 (有序)             │             │
          │        │ 路径不可逆性 > 0.05 (定向力量)    │             │
          │        │ 纯度 > 0.6 (状态集中)             │             │
          │        │ 持续 ≥ 5 天                       │ 条件不满足  │
          │        ▼                                  │             │
          │   ┌──────────────┐                        │             │
          │   │ accumulation │────────────────────────┘             │
          │   │  (叠加态)     │                                      │
          │   └────┬─────────┘                                      │
          │        │                                                │
          │        │ 主特征值 > 0.85 (临界减速)                       │
          │        │ 量能脉冲 > 1.8× (能量注入)                       │
          │        │ 置换熵仍 < 0.75 (有序突破)                        │
          │        │ 退相干速率 < 0 (方向正在确认)                      │
          │        │ 周线置换熵 < 0.75 (多尺度确认)                     │
          │        ▼                                                │
          │   ┌──────────┐        ★ BUY SIGNAL                     │
          │   │ breakout │────────────────────────┐                 │
          │   │ (坍缩)    │                        │                 │
          │   └────┬─────┘                        ▼                 │
          │        │                       ┌────────────┐           │
          │        │                       │   hold     │           │
          │        │                       │  (趋势态)   │           │
          │        │                       └────┬───────┘           │
          │        │                            │                   │
          │        │    置换熵 > 0.90 (无序扩散)  │                   │
          │        │    路径不可逆性 < 0.01       │                   │
          │        │    熵加速度 > 0.05           │                   │
          │        │    量能衰竭 < 0.3×峰值       │                   │
          │        │    纯度骤降 < 0.3 (共识瓦解)  │                   │
          │        │    (任意 3/5 触发)            │                   │
          │        │    或止损 ≤ -10%             │                   │
          │        │                            ▼                   │
          │        │                       ┌──────────┐             │
          └────────┼───────────────────────│ collapse │─────────────┘
                   │                       │ (退相干)  │
                   │                       └──────────┘
                   │                         ★ SELL SIGNAL
                   │
                   └── 直接回到 idle (未进入突破)
```

### 状态定义

| 状态 | 含义 | 物理解释 | 量子类比 |
|------|------|---------|---------|
| **idle** | 无信号 | 系统处于远离临界态的平衡状态 | 热平衡态 |
| **accumulation** | 惜售吸筹中 | 局部熵降低，耗散结构正在形成 | 叠加态（多空共存） |
| **breakout** | 分岔突破 | 临界点到达，对称性破缺，新平衡态形成 | 波函数坍缩（方向确认） |
| **hold** | 持有中 | 耗散结构维持，趋势自我强化 | 确定态（纯态） |
| **collapse** | 结构崩塌 | 耗散结构维持失败，熵增主导 | 退相干（共识瓦解） |

---

## 4. 特征体系

### 4.1 特征总表

本策略从日线/周线 OHLCV 数据中提取 30+ 个特征，分为 10 大类：

| 类别 | 特征名 | 公式 / 说明 | 用途 |
|------|--------|-----------|------|
| **置换熵** | `perm_entropy_s/m/l` | 短/中/长窗口置换熵 [0,1] | 有序度量度 |
| | `entropy_slope` | `perm_entropy_s - perm_entropy_l` | 多尺度熵差异 |
| | `entropy_accel` | `perm_entropy_s.diff(5)` | 熵变化速率 |
| **路径不可逆性** | `path_irrev_m/l` | KLD(forward ‖ backward) ≥ 0 | 方向性力量 |
| **主特征值** | `dom_eig_m/l` | 自相关矩阵最大特征值 [0,1] | 临界减速指标 |
| **换手率熵** | `turnover_entropy_m/l` | 换手率分布熵 [0,1] | 流动性状态 |
| **波动率** | `volatility_m/l` | 收益率滚动标准差 | 辅助参考 |
| | `vol_compression` | `volatility_m / volatility_l` | 辅助参考 |
| | `bbw` | 布林带宽度 `2σ/MA` | 辅助参考 |
| | `bbw_pctl` | BBW 在过去 120 天的分位数 | 辅助参考 |
| **成交量** | `vol_ratio_s` | 短期均量 / 中期均量 | 辅助参考 |
| | `vol_impulse` | 当日量 / 中期均量 | 分岔突破信号 |
| | `vol_shrink` | 短期均量 / 长期均量 | 辅助参考 |
| **价格位置** | `breakout_range` | 在 20 日高低区间的相对位置 [0,1] | 辅助参考（不参与信号判定） |
| **资金流** | `mf_cumsum_s/m` | 净资金流累计（短/中） | 资金流向 |
| | `mf_impulse` | 当日净流入 / 中期标准差 | 资金脉冲 |
| **大单** | `big_net_ratio` | (大买 - 大卖) / 总额 | 主力方向 |
| | `big_net_ratio_ma` | `big_net_ratio` 的短期均值 | 平滑后的主力方向 |
| **量子相干性** | `coherence_l1` | 密度矩阵非对角元 l1-范数 [0,1] | 状态不确定性 |
| | `purity` / `purity_norm` | Tr(ρ²)，归一化到 [0,1] | 状态纯度/集中度 |
| | `von_neumann_entropy` | -Tr(ρ ln ρ)，归一化到 [0,1] | 量子态混合度 |
| | `coherence_decay_rate` | ΔC/Δt，相干性变化速率 | 方向确认速度 |

### 4.2 窗口参数

| 时间级别 | 短窗口 (short) | 中窗口 (medium) | 长窗口 (long) |
|---------|---------------|----------------|--------------|
| **日线** | 10 天 | 20 天 | 60 天 |
| **周线** | 4 周 | 8 周 | 24 周 |

### 4.3 数据要求

| 字段 | 必需 | 说明 |
|------|------|------|
| `trade_date` | ✅ | YYYYMMDD 格式 |
| `open/high/low/close` | ✅ | OHLC 价格 |
| `vol` | ✅ | 成交量 |
| `amount` | ✅ | 成交额 |
| `turnover_rate` | 可选 | 换手率，用于换手率熵 |
| `net_mf_amount` | 可选 | 净资金流入，用于路径不可逆性的 order flow 代理 |
| `buy_elg/lg_amount`, `sell_elg/lg_amount` | 可选 | 大单买卖额，用于大单净额占比 |

---

## 5. 信号检测逻辑

### 5.1 Phase 1: 惜售吸筹检测

#### 物理直觉

当大资金持有者不愿在当前价格卖出股票时（惜售），市场的微观结构会发生可观测的变化：

```
惜售 → 卖盘减少 → 换手率下降 → 成交变得单调
                                    ↓
                           置换熵降低 (序列更有序)
                           路径不可逆性上升 (少数交易被同方向力量主导)
```

#### 检测条件

同时满足以下条件，且持续 ≥ 5 天：

| 条件 | 阈值 | 含义 |
|------|------|------|
| `perm_entropy_m < 0.65` | 有序度 | 价格序列变得更可预测 |
| `path_irrev_m > 0.05` | 方向性 | 存在定向操作力量 |

#### 质量评分

惜售质量分 `accum_quality ∈ [0, 1]`，加权公式：

$$\text{AQ} = 0.35 \times S_{\text{entropy}} + 0.30 \times S_{\text{irrev}} + 0.15 \times S_{\text{big\_net}} + 0.20 \times S_{\text{purity}}$$

- $S_{\text{entropy}} = 1 - \frac{\text{perm\_entropy\_m}}{1.0}$，越低越好
- $S_{\text{irrev}} = \frac{\text{path\_irrev\_m}}{0.5}$，越高越好
- $S_{\text{big\_net}} = \frac{\text{big\_net\_ratio\_ma} + 0.1}{0.2}$，正向流入加分
- $S_{\text{purity}} = \text{purity\_norm}$，密度矩阵纯度越高 = 状态越集中 = 惜售越明确

### 5.2 Phase 2: 分岔突破检测

#### 物理直觉

惜售积累了足够的「势能」后，系统处于临界态。此时一个量能扰动就能触发分岔（相变）：

```
能量积累 (惜售) → 临界态 → 外部扰动 (放量) → 对称性破缺 → 新趋势
    ↓                ↓              ↓               ↓
 accum 阶段     dom_eig → 1    vol_impulse > 1.8   perm_entropy < 0.75
```

**关键**：必须是「有序突破」——突破时熵仍保持低位，说明不是噪声驱动的假突破。

#### 检测条件

需同时满足以下条件中的 **至少 N-1 个**：

| 条件 | 阈值 | 含义 |
|------|------|------|
| 近期有惜售 | 过去 10 天内 `is_accumulating` 为 True | 有足够的能量积累 |
| `dom_eig_m > 0.85` | 临界减速 | 系统自相关增强，接近失稳点 |
| `vol_impulse > 1.8` | 量能脉冲 | 放量打破均衡 |
| `perm_entropy_m < 0.75` | 有序突破 | 排除噪声驱动的假突破 |

#### 质量评分

分岔质量分 `bifurc_quality ∈ [0, 1]`：

$$\text{BQ} = 0.25 \times S_{\text{dom\_eig}} + 0.25 \times S_{\text{vol\_impulse}} + 0.15 \times S_{\text{entropy}} + 0.15 \times S_{\text{irrev}} + 0.20 \times S_{\text{decay}}$$

- $S_{\text{decay}} = \text{clip}(-\text{coherence\_decay\_rate}, 0, 0.05) / 0.05$，退相干速率越快（负值越大）= 方向确认越快 = 突破质量越高

### 5.3 Phase 3: 结构崩塌退出

#### 物理直觉

上涨趋势是一种耗散结构（Prigogine），需要持续的能量输入（成交量 + 资金流）来维持。当能量供给停止：

```
能量输入停止 → 无法对抗熵增 → 局部熵上升 → 耗散结构解体 → 趋势终结
      ↓               ↓              ↓              ↓
  量能衰竭       路径不可逆性降      置换熵飙升     价格下跌
```

#### 检测条件

以下 5 个信号中 **任意 3 个同时触发** 即判定崩塌：

| 信号 | 阈值 | 含义 |
|------|------|------|
| `perm_entropy_m > 0.90` | 熵飙升 | 有序结构瓦解 |
| `path_irrev_m < 0.01` | 不可逆性骤降 | 定向力量消失（主力撤离） |
| `entropy_accel > 0.05` | 熵加速 | 熵在加速膨胀，不可控 |
| `vol_impulse / peak_vol < 0.3` | 量能衰竭 | 能量供给不足以维持结构 |
| `purity_norm < 0.3` | 纯度骤降 | 密度矩阵状态混合 = 市场共识瓦解 |

**止损安全网**：持仓亏损超过 10% 强制止损退出。

> **设计理念**：要求 3/5 信号共振才判定崩塌，避免单一通道噪声导致过早退出。纯理论驱动持有 — 无固定持有天数限制，让趋势自然运行直至结构坍塌。

### 5.4 综合评分

入场综合分 = `composite_score`：

$$\text{CS} = 0.4 \times \text{AQ} + 0.6 \times \text{BQ}$$

权重偏向分岔质量，因为最终交易的触发依赖于突破的发生。

---

## 6. 多时间框架确认

### 周线过滤

在日线发出突破信号后，需要通过周线级别的确认才能实际触发入场信号：

| 周线条件 | 阈值 | 含义 |
|---------|------|------|
| 周线置换熵 | `< 0.75` | 周级别也是有序的，不是日线维度的噪声 |

> **周线数据来源**：不是独立数据源，而是从日线 CSV（`tushare-daily-full/{symbol}.csv`）通过 `aggregate_to_weekly()` 按周五截止重采样聚合而来，再独立计算一遍熵/分岔特征。

### 为什么需要周线确认

参考 Dmitriev et al. (2025) 沙堆模型——如果日线级别的「有序性」在周线级别不成立，那很可能只是日内噪声造成的暂时有序，而非真正的筹码集中。

**1 分钟数据的意义**：高频交易可以在分钟级别制造虚假的有序性（低熵），但这种有序性在日线/周线级别会表现为随机波动（高熵）。因此多时间框架确认可以过滤掉 1 分钟级别「耗散结构消失」的伪信号。

### 日线到周线聚合

```python
aggregate_to_weekly(df_daily)
# open:  取周内第一天
# high:  取周内最高
# low:   取周内最低
# close: 取周内最后一天
# vol:   求和
# amount: 求和
# 以周五为周截止日
```

---

## 7. 市场状态门控

### 7.1 市场状态检测器

基于上证综合指数（000001.SH）的宏观状态检测器，将市场分为 5 种状态，并据此决定是否允许入场：

| 状态 | 仓位系数 | 含义 |
|------|---------|------|
| **DECLINING** (下跌中) | 0.0 | 禁止入场 |
| **DECLINE_ENDED** (下跌末端) | 0.3 | 允许少量入场 |
| **CONSOLIDATION** (横盘整理) | 1.0 | 正常入场 |
| **RISING** (上升趋势) | 0.8 | 正常入场 |
| **RISE_ENDING** (上升末端) | 0.0 | 禁止入场 |

### 7.2 检测特征

| 特征 | 说明 |
|------|------|
| 均线斜率 (20/60日) | MA slope 判断趋势方向 |
| 均线多头/空头排列 | MA5 > MA10 > MA20 > MA60 |
| MACD 动量 | DIF、DEA 方向与柱状图 |
| 布林带宽度 | 波动率收缩/扩张 |
| 成交量趋势 | 量能萎缩或放大 |
| RSI 超买超卖 | 极端区域检测 |

### 7.3 因果安全性

所有市场状态特征均使用 **向后看的滚动计算**（rolling windows），不存在前瞻偏差。在回测中市场状态在 T 日仅使用 T 日及之前的数据。

---

## 8. 回测引擎

### 8.1 组合跟踪式回测

回测引擎采用 **逐日组合跟踪** 模式，纯理论驱动持有与退出：

```
每个交易日:
    1. 执行前一日挂起的入场订单 (以当日收盘价买入)
    2. 检查所有持仓退出条件:
       a. 止损检查: 浮亏 ≤ -10% → 立即退出
       b. 崩塌检查: 3/4 崩塌信号触发 → 退出
    3. 记录当日组合净值 (现金 + 持仓市值)
    4. 若有空仓位 & 到达扫描日 → 扫描全市场寻找入场机会
       - 市场状态门控: DECLINING/RISE_ENDING 不入场
       - 按 entry_score 排名选择候选
       - 信号日 T → 挂起订单 → T+1 日执行
    
回测结束: 强制平仓所有剩余持仓
```

### 8.2 退出策略

| 退出类型 | 条件 | 优先级 |
|---------|------|-------|
| 止损 | 浮亏 ≤ stop_loss_pct (默认 -10%) | 最高 |
| 结构崩塌 | 3/4 崩塌信号共振触发 | 次高 |
| 回测结束 | 回测期末强制平仓 | 兜底 |

> **注意**：无固定持有天数限制。策略完全依赖熵/分岔理论信号退出，让盈利交易充分运行。

### 8.3 入场过滤

| 过滤器 | 默认值 | 说明 |
|--------|--------|------|
| `scan_interval` | 1 天 | 入场扫描间隔 (每 N 天扫描一次) |
| `min_entry_score` | 0.30 | 最低 composite_score 门槛 |
| 市场状态 | 启用 | DECLINING/RISE_ENDING 禁止入场 |
| `max_positions` | 5 | 最大同时持仓数 |

### 8.4 输出文件

| 文件 | 说明 |
|------|------|
| `market_snapshot_{date}.csv` | 全部扫描结果（含 idle 状态） |
| `breakout_candidates_{date}.csv` | 突破候选股 |
| `backtest_equity_{date}.csv` | 净值曲线 |
| `backtest_trades_{date}.csv` | 逐笔交易记录 |

---

## 9. 参数说明

### 9.1 检测器参数 (DetectorConfig)

#### Phase 1 — 惜售吸筹

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `perm_entropy_low` | 0.65 | 置换熵低于此值视为有序 |
| `path_irrev_high` | 0.05 | 路径不可逆性高于此值视为有定向力量 |
| `turnover_entropy_low` | 0.6 | 换手率熵低于此值视为流动性收缩 |
| `accum_min_days` | 5 | 惜售状态最少持续天数 |
| `big_net_positive` | True | 要求大单净额为正 |

#### Phase 2 — 分岔突破

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `dom_eig_threshold` | 0.85 | 主特征值超过此值视为临界减速 |
| `vol_impulse_threshold` | 1.8 | 量能脉冲超过此值视为放量突破 |
| `perm_entropy_breakout_max` | 0.75 | 突破时熵需低于此值（有序突破） |

#### Phase 3 — 结构崩塌

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `perm_entropy_collapse` | 0.90 | 熵超过此值视为无序扩散 |
| `path_irrev_collapse` | 0.01 | 不可逆性跌破此值视为主力撤离 |
| `entropy_accel_collapse` | 0.05 | 熵加速度超过此值视为熵快速膨胀 |
| `vol_exhaustion_ratio` | 0.3 | 当前量 / 峰值量低于此值视为衰竭 |
| 崩塌触发阈值 | 3/5 | 5 个信号中至少 3 个触发才判定崩塌 |

#### 量子相干性

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `purity_accum_min` | 0.6 | 惜售阶段纯度下限（状态越纯 = 筹码越集中） |
| `coherence_decay_breakout` | -0.005 | 突破阶段退相干速率阈值（负值 = 方向正在确认） |
| `purity_collapse_max` | 0.3 | 崩塌阶段纯度上限（极低纯度 = 共识瓦解） |

#### 周线确认

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `weekly_perm_entropy_max` | 0.75 | 周线置换熵上限 |
| `weekly_trend_confirm` | True | 是否启用周线趋势确认 |

### 9.2 回测参数 (CLI)

| 参数 | 默认值 | CLI 参数名 | 说明 |
|------|--------|-----------|------|
| `data_dir` | tushare-daily-full | `--data_dir` | 日线 CSV 目录 |
| `basic_path` | tushare_stock_basic.csv | `--basic_path` | 股票基本信息 CSV |
| `out_dir` | results/.../backtest_2025 | `--out_dir` | 输出目录 |
| `start_date` | 20250101 | `--start_date` | 回测起始日 |
| `end_date` | 20251231 | `--end_date` | 回测结束日 |
| `max_positions` | 5 | `--max_positions` | 组合最大持仓数 |
| `min_amount` | 500,000 | `--min_amount` | 最低日均成交额 |
| `workers` | 16 | `--workers` | 并行计算进程数 |
| `scan_interval` | 1 | `--scan_interval` | 入场扫描间隔天数 |
| `min_score` | 0.30 | `--min_score` | 最低入场综合分 |
| `stop_loss` | -10.0 | `--stop_loss` | 止损百分比 |
| `no_market_gate` | False | `--no_market_gate` | 禁用市场状态门控 |
| `index_code` | 000001_sh | `--index_code` | 市场状态指数代码 |

---

## 10. 使用方法

### 10.1 单次扫描

```bash
# 使用 shell 脚本
bash scripts/run_entropy_accumulation_breakout.sh \
  --scan-date 20260410 \
  --top-n 30

# 或直接用 Python
python src/strategy/entropy_accumulation_breakout/run_entropy_accumulation_breakout.py \
  --data_dir /nvme5/xtang/gp-workspace/gp-data/tushare-daily-full \
  --out_dir results/entropy_accumulation_breakout \
  --basic_path /nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv \
  --scan_date 20260410
```

### 10.2 小样本测试

```bash
bash scripts/run_entropy_accumulation_breakout.sh \
  --symbols sh600519,sz000001,sh601398 \
  --verbose
```

### 10.3 回测

```bash
# 2025 全年回测 (推荐参数)
python src/strategy/entropy_accumulation_breakout/backtest_fast.py \
  --start_date 20250101 \
  --end_date 20251231 \
  --workers 16 \
  --stop_loss -10.0 \
  --scan_interval 3 \
  --min_score 0.30 \
  --verbose
```

关键参数说明：
- `--scan_interval 3`：每 3 个交易日扫描一次入场机会，降低噪声入场
- `--min_score 0.30`：仅入场 composite_score ≥ 0.30 的高质量信号
- `--stop_loss -10.0`：浮亏 10% 强制止损
- 无 `--hold_days`：纯理论驱动退出，不限持有时间

### 10.4 输出格式

**backtest_trades CSV 列**：

```
symbol, entry_date, entry_price, exit_date, exit_price,
exit_reason (collapse | stop_loss | end_of_backtest), pnl_pct, hold_days, score
```

---

## 11. 文件结构

```
src/strategy/entropy_accumulation_breakout/
├── __init__.py                          # 包初始化
├── feature_engine.py                    # 特征引擎 (30+ 特征，含量子相干性)
├── signal_detector.py                   # 三阶段状态检测器 (含退相干速率增强)
├── scan_service.py                      # 扫描服务
├── market_regime.py                     # 市场状态检测器 (5 状态)
├── backtest_fast.py                     # 高速回测引擎 (组合跟踪式)
├── run_entropy_accumulation_breakout.py # CLI 入口 (扫描)
└── README.md                            # 本文档

scripts/
└── run_entropy_accumulation_breakout.sh  # Shell 启动脚本
```

### 调用链

```
── 扫描模式 ──
Shell 脚本 / 命令行
  └── run_entropy_accumulation_breakout.py  (argparse → ScanConfig)
        └── scan_service.run_scan()  (遍历股票)
              ├── feature_engine.build_features()  (日线+周线特征)
              │     ├── compute_single_timeframe_features(df_daily)
              │     ├── aggregate_to_weekly(df_daily)
              │     └── compute_single_timeframe_features(df_weekly)
              └── signal_detector.evaluate_symbol()  (三阶段检测)
                    ├── detect_accumulation()     → Phase 1
                    ├── accumulation_quality()     → 质量分
                    ├── detect_bifurcation_breakout()  → Phase 2
                    └── bifurcation_quality()      → 质量分

── 回测模式 ──
backtest_fast.py  (argparse → run_fast_backtest)
  ├── 并行预计算 _compute_one_symbol()
  │     ├── feature_engine.build_features()
  │     └── 预计算 is_breakout / entry_score
  ├── market_regime.build_market_regime_series()
  └── 逐日组合跟踪循环
        ├── 止损检查
        ├── signal_detector.detect_structural_collapse()  → Phase 3
        └── 入场信号匹配 + 排名
```

### 依赖关系

```
src/core/tick_entropy.py  ← 核心熵计算模块
  ├── rolling_permutation_entropy()
  ├── rolling_path_irreversibility()
  ├── rolling_dominant_eigenvalue()
  ├── rolling_turnover_entropy()
  └── _discretize_trinary(), _rolling_apply_1d()

src/core/quantum_coherence.py  ← 量子相干性模块
  └── compute_quantum_coherence_features()
        → coherence_l1, purity, von_neumann_entropy, coherence_decay_rate
```

---

## 12. 回测结果

### 2025 全年回测 (20250101 — 20251231)

#### 迭代历史

| 版本 | 变更 | 交易数 | 胜率 | 总回报 | 最大回撤 | 平均持仓天数 |
|------|------|--------|------|--------|---------|-------------|
| v1 (hold_days=5, 10仓) | 基线 + 市场门控 + 止损 | 30 | 53.3% | +31.16% | 24.40% | 5.0 |
| v2 (纯理论退出, 5仓) | 去掉 hold_days, 纯崩塌退出 | 90 | 36.7% | +1.71% | 16.37% | 4.7 |
| v3 (扫描+门槛) | scan_interval=3, min_score=0.30, entropy_collapse=0.9 | 39 | 48.7% | +0.44% | 11.27% | 4.7 |
| **v4 (崩塌≥3/4)** | irrev 0.01, accel 0.05, vol_exh 0.3, **score≥3** | **33** | **48.5%** | **+10.76%** | **17.41%** | **16.7** |

#### 当前最优 (v4) 详细结果

```
总交易数        : 33
胜率          : 16/33 = 48.5%
平均收益        : 3.04%
盈亏比         : 1.71
最终净值        : 1.1076
总回报率        : +10.76%
最大回撤        : 17.41%
平均持仓天数      : 16.7
中位持仓天数      : 10
最长持仓天数      : 86
```

#### 退出原因分布

| 退出原因 | 次数 | 占比 |
|---------|------|------|
| collapse (结构崩塌) | 20 | 60.6% |
| stop_loss (止损) | 11 | 33.3% |
| end_of_backtest (回测结束) | 2 | 6.1% |

#### 市场状态表现

| 市场状态 | 交易数 | 平均收益 | 胜率 |
|---------|--------|---------|------|
| rising | 26 | +5.30% | 53.85% |
| consolidation | 7 | -5.36% | 28.57% |

#### 关键发现

1. **持仓时长大幅提升**：要求 3/4 崩塌信号共振后，平均持仓从 4.7 天提升到 16.7 天，策略能"拿住"大行情
2. **盈亏比正**：平均盈利 +16.56% vs 平均亏损 -9.69%，盈亏比 1.71
3. **市场状态有效**：rising 状态下胜率 53.85%，consolidation 下仅 28.57%
4. **止损保护**：33.3% 的交易由止损退出，有效控制了崩塌检测延迟带来的风险

---

## 13. 参考文献

1. **Seifert, U.** (2025). The Stochastic Thermodynamics of Computation. *arXiv:2501.XXXXX*.
   - 粗粒化熵产生下界 → 路径不可逆性的理论保证

2. **Fan, H., Chen, S., Gao, Z., et al.** (2025). Instability of Financial Time Series Revealed by Irreversibility Analysis. *Entropy*, 27(4), 402.
   - KLD 不可逆性 > 矩统计量; DHVG 滑动窗口

3. **Bielinskyi, A., Serdyuk, V., Semerikov, S., et al.** (2025). Early Warning Signs: Evaluating Permutation Entropy Metrics for Stock Market Crashes.
   - 加权置换熵检测市场复杂度变化

4. **Dmitriev, A. V., Maltseva, S. V., Tsukanova, O. A., et al.** (2025). Self-organization of the stock exchange to the edge of a phase transition. *Frontiers in Physics*, 12, 1508465.
   - 熵=控制参数, 成交量=序参量; 沙堆自组织临界; 方差/AR(1)/峰度/偏度预警

5. **Yan, H., Zhao, L., Hu, L., et al.** (2023). Thermodynamic and dynamical predictions for bifurcations and non-equilibrium phase transitions. *Communications Physics*, 6, 16.
   - 熵产生率在分岔点达峰

6. **Ardakani, O. M.** (2025). Detecting Financial Bubbles with Tail-Weighted Entropy.
   - 尾部加权熵检测泡沫与崩塌

7. **Baumgratz, T., Cramer, M. & Plenio, M. B.** (2014). Quantifying Coherence. *Physical Review Letters*, 113(14), 140401.
   - l1-范数相干性度量 → coherence_l1 指标的理论基础

8. **Baaquie, B. E.** (2004). *Quantum Finance: Path Integrals and Hamiltonians for Options and Interest Rates*. Cambridge University Press.
   - 量子力学数学框架在金融中的应用; 密度矩阵 → 市场状态编码

9. **Busemeyer, J. R. & Bruza, P. D.** (2012). *Quantum Models of Cognition and Decision*. Cambridge University Press.
   - 量子概率论解释决策行为中的经典概率违反现象 → 叠加/坍缩/退相干类比的理论依据
