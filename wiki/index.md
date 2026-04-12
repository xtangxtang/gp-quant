# GP-Quant Wiki — 内容索引

> LLM 在每次 Ingest 操作后更新此文件。  
> 查询时先读此文件定位相关页面。

---

## 综述

| 页面 | 说明 |
|------|------|
| [[overview]] | 项目全局综述：A 股量化研究工作空间 |

## 概念 (Concepts)

| 页面 | 说明 | 标签 |
|------|------|------|
| [[entropy]] | 熵、信息论与市场无序度度量 | `entropy`, `thermodynamics` |
| [[bifurcation]] | 分岔理论、临界转变与早期预警信号 | `bifurcation`, `critical-transition` |
| [[fractal]] | 分形几何、自相似性与多尺度结构 | `fractal`, `multiscale` |
| [[dissipative-structure]] | 耗散结构、相变与资本流入的局部熵降 | `phase-transition`, `thermodynamics` |
| [[permutation-entropy]] | 置换熵：基于序型的复杂度度量 | `entropy`, `complexity` |
| [[dominant-eigenvalue]] | 主特征值与临界减速检测 | `bifurcation`, `eigenvalue` |
| [[path-irreversibility]] | 路径不可逆性：前向/反向转移概率不对称 | `entropy`, `irreversibility` |
| [[strategic-abandonment]] | 战略放弃：高噪声环境下的最优退出 | `control-theory`, `noise` |
| [[multitimeframe-resonance]] | 多时间框架共振：日/周/月级别趋势共振 | `multitimeframe`, `resonance` |

## 实体 (Entities)

| 页面 | 说明 | 源码位置 |
|------|------|----------|
| [[tick-entropy-module]] | 核心熵计算模块（5 个熵指标 + 状态分类器） | `src/core/tick_entropy.py` |
| [[four-layer-system]] | 四层选股系统（市场门 → 股票状态 → 执行成本 → 实验层） | `src/strategy/entropy_bifurcation_setup/` |
| [[multitimeframe-scanner]] | 多时间框架共振扫描器 | `src/strategy/multitimeframe/` |
| [[data-pipeline]] | 数据管道：Tushare/Eastmoney/Tencent 数据同步 | `src/downloader/` |
| [[continuous-decline-recovery]] | 连续下跌恢复买入策略 | `src/strategy/continuous_decline_recovery/` |
| [[hold-exit-system]] | 持有/退出决策系统（熵储备 + 快速膨胀 + 衰竭退出） | `src/strategy/uptrend_hold_state_flow/` |
| [[web-dashboard]] | Web 可视化面板 | `web/` |

## 来源 (Sources)

| 页面 | 论文/来源 | 年份 |
|------|-----------|------|
| [[seifert-2025-entropy-bounds]] | Seifert - 粗粒化轨迹的熵产生下界 | 2025 |
| [[period-doubling-eigenvalue]] | 主特征值检测周期倍化分岔 | 2026 |
| [[reservoir-computing-tipping]] | 储层计算预测超早期临界点 | 2026 |
| [[pinn-vs-neural-ode]] | PINN vs Neural ODE 在临界态的对比 | 2026 |
| [[communication-induced-bifurcation]] | 信息处理成本诱导分岔 | 2026 |
| [[12-papers-synthesis]] | 12 篇论文统一框架综合分析 | 2025-2026 |

## 实验 (Experiments)

| 页面 | 说明 | 结果 |
|------|------|------|
| [[entropy-backtest-minute]] | 分钟级熵因子回测（240min 窗口） | ❌ 失败：35% 胜率 |
| [[four-layer-backtest-2025]] | 四层系统 2025 年日线回测 | 进行中 |
| [[multitimeframe-backtest]] | 多时间框架共振回测 | 进行中 |

## 决策 (Decisions)

| 页面 | 说明 |
|------|------|
| [[why-daily-not-minute]] | 为什么选日线而非分钟线作为主要时间框架 |
| [[gray-box-over-black-box]] | 灰箱模型（结构约束）优于黑箱的理论与实证依据 |

---

*最后更新: 2026-04-12 — 初始化 Wiki 结构*
