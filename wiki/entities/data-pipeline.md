---
title: 数据管道
tags: [data, pipeline, tushare, eastmoney]
confidence: high
status: active
created: 2026-04-12
updated: 2026-04-12
---

# 数据管道 (Data Pipeline)

**源码**: `src/downloader/`

## 数据源

| 源 | 数据类型 | 频率 |
|----|----------|------|
| Tushare | 日线 K 线、财务数据、除权因子、交易日历 | 日 |
| Eastmoney | 分钟级交易数据 | 实时增量 |
| Tencent | 分钟级交易数据（备用） | 实时增量 |

## 核心模块

| 模块 | 功能 |
|------|------|
| `download_all_tushare.py` | 一次性全量下载 |
| `get_tushare_daily_full.py` | 下载全部日线 K 线 |
| `get_tushare_extended.py` | 财务数据、分红、停牌 |
| `get_tushare_stock_list.py` | 下载股票列表 |
| `fast_sync_tushare_latest.py` | 快速增量更新日线 |
| `sync_a_share_1m.py` | 分钟级增量同步 |
| `eod_data_scheduler.py` | 自动 16:00 定时更新 |
| `tushare_provider.py` | Tushare API 封装 |
| `daily_kline_provider.py` | 日线 K 线聚合 |
| `eastmoney_universe.py` | Eastmoney 数据源 |
| `csv_utils.py` | CSV 读写工具 |

## 数据存储

```
/nvme5/xtang/gp-workspace/gp-data/
├── tushare-daily-full/       # 日线 K 线（每股一文件，csv）
├── trade/<symbol>/           # 分钟级交易（每天一文件）
├── tushare_stock_basic.csv   # 股票基本信息
├── trade_cal.csv             # 交易日历
├── adj_factor.csv            # 复权因子
├── suspend_d.csv             # 停牌信息
├── income.csv                # 利润表
├── balancesheet.csv          # 资产负债表
├── cashflow.csv              # 现金流量表
├── fina_indicator.csv        # 财务指标
└── dividend.csv              # 分红数据
```

## 运行脚本

| 脚本 | 用途 |
|------|------|
| `scripts/run_get_tushare_all.sh` | 全量下载 |
| `scripts/run_sync_a_share_1m.sh` | 1 分钟增量同步 |
| `scripts/run_eod_data_scheduler.sh` | EOD 定时调度 |
| `scripts/run_get_tushare_daily_full.sh` | 日线全量 |
| `scripts/run_get_tushare_extended.sh` | 扩展数据 |

## 相关实体

- [multitimeframe-scanner](multitimeframe-scanner.md) — 消费日线数据
- [four-layer-system](four-layer-system.md) — 消费日线数据
- [web-dashboard](web-dashboard.md) — 展示数据
