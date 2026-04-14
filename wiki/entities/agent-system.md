---
title: Agent 调度系统
tags: [agent, supervisor, scheduling, data-pipeline]
confidence: high
status: active
created: 2026-04-12
updated: 2026-04-12
---

# Agent 调度系统

**源码**: `src/agents/`

## 概述

Agent 系统将数据下载拆分为 5 个数据 Agent + 1 个策略 Agent，由 Supervisor 统一调度和监控。替代原有的 `eod_data_scheduler.py` 单进程模式。

## 架构

```
Supervisor (supervisor.py)
  ├── 依赖 DAG 调度（按拓扑序执行）
  ├── 自动重试（最多 3 次，间隔 5 分钟）
  ├── 状态看板 (status)
  ├── 告警日志 (.agent_alerts.log)
  └── 守护进程模式 (daemon, 每日 16:00)

Agent 执行顺序（按优先级）:
  P0: stock_list         → 股票列表
  P1: daily_financial     → 日线 + 财务（依赖 stock_list）
  P1: market_data         → 资金流/指数/市场数据（依赖 stock_list）
  P2: minute              → 1 分钟 K 线（依赖 stock_list）
  P3: derived             → 衍生数据/周线（依赖 daily_financial）
  P4: market_trend        → 大盘趋势判断（依赖 derived + market_data）
```

## Agent 清单

| Agent | 模块 | 说明 | 频率 |
|-------|------|------|------|
| `stock_list` | `agent_stock_list.py` | 同步 stock_basic + gplist | 每日 |
| `daily_financial` | `agent_daily_financial.py` | 日线行情 + 复权/涨跌停/停牌/分红 + 财报 | 每日 |
| `market_data` | `agent_market_data.py` | 资金流/指数/两融/大宗/北向/期货/宏观利率 | 每日 |
| `minute` | `agent_minute.py` | 1 分钟 K 线 (Tushare) | 每日 |
| `derived` | `agent_derived.py` | 衍生数据（5 日周线等） | 每日 |
| `market_trend` | `agent_market_trend.py` | 大盘趋势判断（7 维度评分 + 报告） | 每日 |

## 状态管理

每个 Agent 独立维护 JSON 状态文件：

```
gp-data/
├── .agent_stock_list_state.json
├── .agent_daily_financial_state.json
├── .agent_market_data_state.json
├── .agent_minute_state.json
├── .agent_derived_state.json
├── .agent_market_trend_state.json
├── .agent_stock_list.lock          # fcntl 排他锁
└── .agent_alerts.log               # 告警日志
```

状态字段：`status`(idle/running/success/failed), `progress`, `last_success_at`, `consecutive_failures`, `stats`

## 使用方式

```bash
# 查看所有 Agent 状态
./scripts/run_agent_supervisor.sh status

# 运行每日同步（stock_list → daily_financial/market_data → minute → derived → market_trend）
./scripts/run_agent_supervisor.sh run

# 只运行某个 Agent
./scripts/run_agent_supervisor.sh run --agent daily_financial

# 守护进程模式
./scripts/run_agent_supervisor.sh daemon --schedule-time 16:00
```

## 核心类

| 类 | 文件 | 说明 |
|----|------|------|
| `AgentState` | `base_agent.py` | JSON 状态读写，原子写入（tmp + replace） |
| `BaseAgent` | `base_agent.py` | Agent 基类：锁管理、execute 生命周期、进度上报 |
| Supervisor | `supervisor.py` | 依赖调度、重试、告警、守护进程 |

## 依赖关系

- 各 Agent 内部调用 `src/downloader/` 的已有下载函数，不重复实现下载逻辑
- `stock_list` 是所有其他 Agent 的前置依赖（提供股票列表）
- `derived` 依赖 `daily_financial`（需要日线数据生成周线）
- `market_trend` 依赖 `derived` + `market_data`（需要所有数据就绪后运行趋势判断）

## 相关实体

- [data-pipeline](data-pipeline.md) — 底层数据下载模块
- [web-dashboard](web-dashboard.md) — 展示数据
