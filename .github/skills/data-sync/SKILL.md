---
name: data-sync
description: "同步 A 股行情数据。Use when: 数据同步, 下载数据, sync data, download, tushare, 日线数据, 分钟数据, 1分钟, 更新数据, 增量同步, eod scheduler, 全量下载, 扩展数据, 财务数据。"
argument-hint: "数据类型，例如：日线、1分钟、全量、增量"
---

# A 股数据同步 (Data Sync)

管理 A 股行情数据的下载与更新，覆盖日线、分钟线、财务等全类型数据。

## 使用场景

- 首次全量下载（股票列表 + 日线历史 + 扩展数据）
- 每日增量同步（收盘后自动/手动触发）
- 1 分钟数据同步（免费源 Eastmoney/Tencent 或付费 Tushare）
- 历史 1 分钟数据回补
- 仅更新特定股票

## 数据目录结构

```
gp-data/
├── tushare_stock_basic.csv        # 股票基础信息
├── tushare_gplist.json            # 股票列表
├── tushare-daily-full/            # 每只股票一个 CSV（日线+资金流向）
│   ├── sh600000.csv
│   └── ...
├── tushare-extended/              # 复权因子、ST、分红、财务
└── trade/                         # 1 分钟数据
    ├── sh600000/
    │   ├── 2026-04-09.csv
    │   └── ...
    └── ...
```

## 执行步骤

### 方式一：全量初始化

首次使用时，一次性下载全部数据：

```bash
# 方法 A：分步下载（股票列表 → 日线 → 扩展数据）
./scripts/run_get_tushare_all.sh \
  -o /nvme5/xtang/gp-workspace/gp-data \
  --token <tushare_token>

# 方法 B：一次性全部下载（Tushare 2000 积分可用的全部数据）
./scripts/run_download_all_tushare.sh \
  -o /nvme5/xtang/gp-workspace/gp-data \
  --token <tushare_token>
```

方法 B 支持额外参数：
- `--threads N` — 并发线程数（默认 4）
- `--rate N` — API 调用频率（默认 180/min）
- `--category <all|daily|financial|...>` — 只下载特定类别
- `--start-date / --end-date` — 指定日期范围

### 方式二：每日自动调度

收盘后 16:00 自动触发增量同步：

```bash
# 后台常驻
nohup ./scripts/run_eod_data_scheduler.sh \
  -o /nvme5/xtang/gp-workspace/gp-data \
  > /tmp/gp_quant_eod_scheduler.log 2>&1 &

# 立即执行一次
./scripts/run_eod_data_scheduler.sh \
  -o /nvme5/xtang/gp-workspace/gp-data \
  --run-now --run-once

# 只看命令不执行
./scripts/run_eod_data_scheduler.sh \
  -o /nvme5/xtang/gp-workspace/gp-data \
  --run-now --run-once --dry-run
```

关键参数：
- `--schedule-time HH:MM` — 触发时间，默认 16:00
- `--minute-source <em|tx|ts>` — 分钟源，默认 ts
- `--skip-date-based` — 跳过日线等日期型数据
- `--skip-financials` — 跳过财务数据
- `--run-now` — 立即执行，不等待定时
- `--run-once` — 执行一次后退出

### 方式三：仅日线数据

```bash
./scripts/run_get_tushare_daily_full.sh \
  -o /nvme5/xtang/gp-workspace/gp-data \
  --token <token>

# 仅特定股票
./scripts/run_get_tushare_daily_full.sh \
  -o /nvme5/xtang/gp-workspace/gp-data \
  --symbols sh600000,sz000001
```

### 方式四：1 分钟数据

```bash
# 免费源（最近 3 个交易日）
./scripts/run_sync_a_share_1m.sh \
  -o /nvme5/xtang/gp-workspace/gp-data \
  --source em --recent-open-days 3

# Tushare 付费源（指定日期范围）
./scripts/run_sync_a_share_1m.sh \
  -o /nvme5/xtang/gp-workspace/gp-data \
  --source ts --token <token> \
  --start-date 20260401 --end-date 20260409

# 仅重试失败任务
./scripts/run_sync_a_share_1m.sh \
  -o /nvme5/xtang/gp-workspace/gp-data \
  --retry-failed-only
```

关键参数：
- `--source <em|tx|ts>` — em=Eastmoney(免费), tx=Tencent(免费), ts=Tushare(付费)
- `--recent-open-days N` — 同步最近 N 个交易日
- `--force` — 强制重拉已有文件
- `--retry-failed-rounds N` — 失败任务重试轮数

### 方式五：1 分钟历史回补

```bash
./scripts/run_backfill_tushare_1m_history.sh \
  -o /nvme5/xtang/gp-workspace/gp-data \
  --token <token>
```

## 注意事项

- Tushare API 有频率限制（默认 batch_rate=240/min, financial_rate=360/min）
- 免费分钟源（em/tx）通常只覆盖最近几个交易日
- 建议用 `--dry-run` 先预览命令再执行大规模操作
