---
title: GP-Quant 项目综述
tags: [overview]
confidence: high
status: active
created: 2026-04-12
updated: 2026-04-12
---

# GP-Quant — A 股量化研究工作空间

## 定位

研究导向的量化分析平台，聚焦 A 股市场，核心方法论来自**复杂性理论**与**经济物理学**。

不是交易系统，而是一个用于：
- 理解市场微观结构（熵、分岔、临界转变）
- 多时间框架信号共振选股
- 持仓管理（持有/退出决策）
- 回测与因子验证

的研究工作空间。

## 核心架构

### 数据层 (`src/downloader/`)
- **Tushare**: 日线 K 线、财务数据、除权因子、交易日历
- **Eastmoney / Tencent**: 分钟级交易数据增量同步
- **定时调度**: 每日 16:00 自动更新 (`eod_data_scheduler.py`)

### 策略层 (`src/strategy/`)
1. **[[multitimeframe-scanner]]** — 主力策略：日/周/月多时间框架共振扫描
2. **[[four-layer-system]]** — 四层熵分岔选股：市场门 → 股票状态 → 执行成本 → 实验层
3. **[[continuous-decline-recovery]]** — 连续下跌恢复买入策略
4. **[[hold-exit-system]]** — 持有/退出决策：熵储备 + 状态流

### 核心计算 (`src/core/`)
- **[[tick-entropy-module]]** — 5 个熵指标 + 市场状态分类器

### 可视化 (`web/`)
- **[[web-dashboard]]** — Flask Web 面板

## 理论基础

项目的选股逻辑建立在 12 篇学术论文的综合分析之上（详见 [[12-papers-synthesis]]）：

- **[[entropy]]** — 市场无序度度量；低熵 = 资本流入 = 趋势形成
- **[[bifurcation]]** — 系统在临界点发生质变，横盘突破
- **[[path-irreversibility]]** — 前向/反向转移不对称 = 主力控盘信号
- **[[strategic-abandonment]]** — 噪声过高时战略性退出

## 关键教训

| 教训 | 详情 | 决策页 |
|------|------|--------|
| 熵因子在分钟级无效 | 35% 胜率，远低于随机 | [[why-daily-not-minute]] |
| 灰箱优于黑箱 | 结构约束的 PINN > 纯 NN | [[gray-box-over-black-box]] |
| 交易成本是杀手 | 高换手 + 低利润 = 手续费吞噬收益 | [[entropy-backtest-minute]] |

## 数据位置

```
/nvme5/xtang/gp-workspace/gp-data/
├── tushare-daily-full/       # 日线 K 线（每股一文件）
├── trade/<symbol>/           # 分钟级交易数据
├── tushare_stock_basic.csv   # 股票基本信息
├── trade_cal.csv             # 交易日历
└── *.csv                     # 财务数据等
```
