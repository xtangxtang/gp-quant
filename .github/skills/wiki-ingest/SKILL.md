---
name: wiki-ingest
description: "生成或更新每只股票的Wiki知识库（Karpathy LLM-Wiki模式）。Use when: wiki, 知识库, ingest, 生成wiki, 更新wiki, 股票wiki, stock wiki, 知识整理, 信息归档, Karpathy, 每日总结, 雪球总结, xueqiu wiki。"
argument-hint: "股票代码和日期，例如：SH600519 / 为600519生成wiki / 更新所有股票的wiki"
---

# 股票 Wiki 知识库生成 (Wiki Ingest)

将雪球社区每日采集数据整理为结构化的 per-stock Wiki 知识库，采用 Karpathy LLM-Wiki 三层架构（raw → wiki → schema）。

## 使用场景

- 采集完雪球数据后，为指定股票生成/更新 Wiki 知识库
- 批量更新多只股票的 Wiki
- 查看已有 Wiki 内容
- 每日自动归档——生成 wiki 页面 + raw 目录下的当日总结 md

## 三层架构

```
{symbol}/
├── raw/                    ← 不可变的每日原始数据
│   ├── 20260413.json       ← 采集的原始 JSON
│   └── 20260413_summary.md ← 当日总结（自动从 JSON 提取）
├── wiki/                   ← 7 个结构化知识页面
│   ├── index.md            ← 页面目录
│   ├── log.md              ← 操作日志
│   ├── overview.md         ← 公司总览
│   ├── orders_channels.md  ← 订单与渠道
│   ├── financials.md       ← 财务与业绩
│   ├── capacity.md         ← 产能与产量
│   ├── industry_policy.md  ← 行业与政策
│   ├── management.md       ← 管理层动态
│   └── market_sentiment.md ← 市场情绪
└── SCHEMA.md               ← Wiki 结构约定
```

## 执行步骤

### 1. 确认参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--symbol` | 股票代码 (e.g. SH600519) | 必填 |
| `--date` | 指定日期 YYYYMMDD | 自动找最新 JSON |
| `--data-dir` | 数据根目录 | `/nvme5/xtang/gp-workspace/gp-data/xueqiu_orders` |
| `--use-llm` | 使用 LLM 生成页面 | 默认不使用（由 agent 直接生成） |

### 2. 方式 A: Agent 直接生成（推荐，快速）

**不调用 LLM**，由 Copilot agent 充当知识库维护员，直接读取 JSON 数据并生成 wiki 页面。

工作流：
1. 读取 `{data_dir}/{symbol}/{date}.json` 中的 `filtered_posts` 和 `summary`
2. 分析帖子内容，提取事实性信息
3. 按 7 个页面的职责分类写入 wiki/
4. 将 JSON 中的 `summary` 字段提取为 `raw/{date}_summary.md`
5. 生成 `wiki/index.md`（页面目录）和 `wiki/log.md`（操作日志）

```bash
# 先确认数据存在
ls /nvme5/xtang/gp-workspace/gp-data/xueqiu_orders/{symbol}/*.json
```

### 3. 方式 B: LLM 自动生成（慢，约 15-20 分钟）

```bash
cd /nvme5/xtang/gp-workspace/gp-quant
./scripts/run_wiki_ingest.sh --symbol SH600519 [--date 20260413]
```

### 4. 查看结果

```bash
# 查看目录结构
find /nvme5/xtang/gp-workspace/gp-data/xueqiu_orders/{symbol}/ -type f | sort

# 查看总结
cat /nvme5/xtang/gp-workspace/gp-data/xueqiu_orders/{symbol}/raw/{date}_summary.md

# 查看某个 wiki 页面
cat /nvme5/xtang/gp-workspace/gp-data/xueqiu_orders/{symbol}/wiki/overview.md
```

## 输出文件

| 文件 | 说明 |
|------|------|
| `raw/{date}.json` | 不可变的原始采集数据 |
| `raw/{date}_summary.md` | 当日情报总结（从 JSON summary 提取） |
| `wiki/overview.md` | 公司总览：基本信息、状态概述、关键指标 |
| `wiki/orders_channels.md` | 订单/合同/渠道动销/预收款 |
| `wiki/financials.md` | 营收/利润/估值/分红/现金流 |
| `wiki/capacity.md` | 产能规划/基酒产量/释放节奏 |
| `wiki/industry_policy.md` | 行业周期/竞争格局/监管/消费趋势 |
| `wiki/management.md` | 管理层变动/公司治理/资本配置 |
| `wiki/market_sentiment.md` | 多空观点/机构观点/情绪分布 |
| `wiki/index.md` | 页面目录 |
| `wiki/log.md` | 操作日志 |
| `SCHEMA.md` | Wiki 结构约定 |

## 策略代码

```
src/wiki/
├── __init__.py
└── stock_wiki_ingest.py   # CLI 入口 + LLM ingest 逻辑
```

## Wiki 页面约定

- 每页顶部有 YAML frontmatter: `title`, `updated`, `sources_count`
- 信息来源标注: `> 📌 来源: @用户名 (YYYY-MM-DD)`
- 矛盾信息: 保留两者并标注 `⚠️ 矛盾`
- 页面间引用: `[[page_name]]` 风格
- 增量更新时新信息标记 `🆕`

## 数据依赖

- 输入: 雪球采集 JSON（由 `src/downloader/xueqiu_order_monitor.py` 生成）
- 数据目录: `/nvme5/xtang/gp-workspace/gp-data/xueqiu_orders/`
