# GP-Quant Wiki — 内容索引

> LLM 在每次 Ingest 操作后更新此文件。  
> 查询时先读此文件定位相关页面。

---

## 综述

| 页面 | 说明 |
|------|------|
| [overview](overview.md) | 项目全局综述：A 股量化研究工作空间 |

## 概念 (Concepts)

| 页面 | 说明 | 标签 |
|------|------|------|
| [entropy](concepts/entropy.md) | 熵、信息论与市场无序度度量 | `entropy`, `thermodynamics` |
| [bifurcation](concepts/bifurcation.md) | 分岔理论、临界转变与早期预警信号 | `bifurcation`, `critical-transition` |
| [fractal](concepts/fractal.md) | 分形几何、自相似性与多尺度结构 | `fractal`, `multiscale` |
| [dissipative-structure](concepts/dissipative-structure.md) | 耗散结构、相变与资本流入的局部熵降 | `phase-transition`, `thermodynamics` |
| [permutation-entropy](concepts/permutation-entropy.md) | 置换熵：基于序型的复杂度度量 | `entropy`, `complexity` |
| [dominant-eigenvalue](concepts/dominant-eigenvalue.md) | 主特征值与临界减速检测 | `bifurcation`, `eigenvalue` |
| [path-irreversibility](concepts/path-irreversibility.md) | 路径不可逆性：前向/反向转移概率不对称 | `entropy`, `irreversibility` |
| [strategic-abandonment](concepts/strategic-abandonment.md) | 战略放弃：高噪声环境下的最优退出 | `control-theory`, `noise` |
| [multitimeframe-resonance](concepts/multitimeframe-resonance.md) | 多时间框架共振：日/周/月级别趋势共振 | `multitimeframe`, `resonance` |

## 实体 (Entities)

| 页面 | 说明 | 源码位置 |
|------|------|----------|
| [tick-entropy-module](entities/tick-entropy-module.md) | 核心熵计算模块（5 个熵指标 + 状态分类器） | `src/core/tick_entropy.py` |
| [four-layer-system](entities/four-layer-system.md) | 四层选股系统（市场门 → 股票状态 → 执行成本 → 实验层） | `src/strategy/entropy_bifurcation_setup/` |
| [multitimeframe-scanner](entities/multitimeframe-scanner.md) | 多时间框架共振扫描器 | `src/strategy/multitimeframe/` |
| [data-pipeline](entities/data-pipeline.md) | 数据管道：Tushare/Tencent 数据同步 | `src/downloader/` |
| [agent-system](entities/agent-system.md) | Agent 调度系统：5 个数据 Agent + Supervisor | `src/agents/` |
| [continuous-decline-recovery](entities/continuous-decline-recovery.md) | 连续下跌恢复买入策略 | `src/strategy/continuous_decline_recovery/` |
| [hold-exit-system](entities/hold-exit-system.md) | 持有/退出决策系统（熵储备 + 快速膨胀 + 衰竭退出） | `src/strategy/uptrend_hold_state_flow/` |
| [entropy-accumulation-breakout](entities/entropy-accumulation-breakout.md) | 熵惜售分岔突破策略（三阶段状态机：惜售→突破→崩塌） | `src/strategy/entropy_accumulation_breakout/` |
| [web-dashboard](entities/web-dashboard.md) | Web 可视化面板 | `web/` |
| [market-trend-system](entities/market-trend-system.md) | 大盘趋势判断系统（从小见大：7维度微观聚合宏观） | `src/strategy/market_trend/` |

## 来源 (Sources)

| 页面 | 论文/来源 | 年份 |
|------|-----------|------|
| [seifert-2025-entropy-bounds](sources/seifert-2025-entropy-bounds.md) | Seifert - 粗粒化轨迹的熵产生下界 | 2025 |
| [period-doubling-eigenvalue](sources/period-doubling-eigenvalue.md) | 主特征值检测周期倍化分岔 | 2026 |
| [reservoir-computing-tipping](sources/reservoir-computing-tipping.md) | 储层计算预测超早期临界点 | 2026 |
| [pinn-vs-neural-ode](sources/pinn-vs-neural-ode.md) | PINN vs Neural ODE 在临界态的对比 | 2026 |
| [communication-induced-bifurcation](sources/communication-induced-bifurcation.md) | 信息处理成本诱导分岔 | 2026 |
| [12-papers-synthesis](sources/12-papers-synthesis.md) | 12 篇论文统一框架综合分析 | 2025-2026 |
| [fan-2025-irreversibility](sources/fan-2025-irreversibility.md) | Fan et al. — KLD 不可逆性检测金融不稳定 | 2025 |
| [dmitriev-2025-self-organization](sources/dmitriev-2025-self-organization.md) | Dmitriev et al. — 股票市场自组织到相变边缘 | 2025 |
| [yan-2023-thermodynamic-bifurcation](sources/yan-2023-thermodynamic-bifurcation.md) | Yan et al. — 热力学预测分岔与非平衡相变 | 2023 |

## 实验 (Experiments)

| 页面 | 说明 | 结果 |
|------|------|------|
| [entropy-backtest-minute](experiments/entropy-backtest-minute.md) | 分钟级熵因子回测（240min 窗口） | ❌ 失败：35% 胜率 |
| [four-layer-backtest-2025](experiments/four-layer-backtest-2025.md) | 四层系统 2025 年日线回测 | 进行中 |
| [multitimeframe-backtest](experiments/multitimeframe-backtest.md) | 多时间框架共振回测 | 进行中 |
| [market-trend-backtest-2024](experiments/market-trend-backtest-2024.md) | 大盘趋势回测 2023H2-2026Q2 (672天) | ✅ 完成: STRONG_DOWN后20日涨+3.25% |

## 决策 (Decisions)

| 页面 | 说明 |
|------|------|
| [why-daily-not-minute](decisions/why-daily-not-minute.md) | 为什么选日线而非分钟线作为主要时间框架 |
| [gray-box-over-black-box](decisions/gray-box-over-black-box.md) | 灰箱模型（结构约束）优于黑箱的理论与实证依据 |

---

*最后更新: 2026-04-13 — 新增大盘趋势判断系统 + 2023H2-2026Q2回测分析*
