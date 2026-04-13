---
name: market-trend
description: "运行大盘趋势判断策略（从小见大：7维度微观+宏观聚合评分）。Use when: 大盘趋势, 市场趋势, market trend, 从小见大, 广度, breadth, 资金流, money flow, 波动结构, volatility, 熵有序度, entropy ordering, 动量扩散, momentum diffusion, 杠杆资金, leverage, 流动性, liquidity, SHIBOR, 两融, 涨跌比, 涨跌停, 趋势判断, trend judgement, 大盘状态, market state。"
argument-hint: "日期范围或天数，例如：20240101 20260410 或最近 30 天"
---

# 大盘趋势判断策略 (Market Trend — 从小见大)

通过全市场个股微观交易行为 + 宏观资金/流动性数据，聚合判断大盘趋势状态。

```
微观 5 维 (个股聚合)          宏观 2 维 (外部数据)
├── 广度 Breadth         25%   ├── 杠杆资金 Leverage   10%
├── 资金流 Money Flow    20%   └── 流动性 Liquidity    10%
├── 波动结构 Volatility  10%
├── 熵/有序度 Entropy    15%         ↓
└── 动量扩散 Momentum    10%   综合评分 [-1, 1] → 趋势判定
```

## 趋势状态

| 状态 | 综合得分条件 | 说明 |
|------|-------------|------|
| STRONG_UP | ≥ 0.5 且广度推力触发 | 强势上涨 |
| UP | ≥ 0.2 | 上涨趋势 |
| NEUTRAL | (-0.2, 0.2) | 震荡中性 |
| DOWN | ≤ -0.2 | 下跌趋势 |
| STRONG_DOWN | ≤ -0.5 且恐慌度 > 3% | 强势下跌 |

## 执行步骤

### 1. 默认运行（2024年至今）

```bash
cd /nvme5/xtang/gp-workspace/gp-quant
./scripts/run_market_trend.sh
```

### 2. 指定日期范围

```bash
./scripts/run_market_trend.sh 20250101 20260410
```

### 3. 直接用 Python 运行（更多参数控制）

```bash
python -m src.strategy.market_trend.run_market_trend \
  --data_dir /nvme5/xtang/gp-workspace/gp-data/tushare-daily-full \
  --index_dir /nvme5/xtang/gp-workspace/gp-data/tushare-index-daily \
  --stk_limit_dir /nvme5/xtang/gp-workspace/gp-data/tushare-stk_limit \
  --margin_path /nvme5/xtang/gp-workspace/gp-data/tushare-margin/margin.csv \
  --shibor_path /nvme5/xtang/gp-workspace/gp-data/tushare-shibor/shibor.csv \
  --index_member_path /nvme5/xtang/gp-workspace/gp-data/tushare-index_member_all/index_member_all.csv \
  --basic_path /nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv \
  --out_dir ./results/market_trend \
  --start_date 20240101 --end_date 20260410 \
  --workers 8
```

### 4. 分析已有结果

如果结果文件已存在，直接读取 `results/market_trend/market_trend_daily.csv` 或指定 `--out_dir` 下的 CSV 进行分析。

## 输出文件

结果输出到 `results/market_trend/`（或 `--out_dir` 指定目录）:

| 文件 | 内容 |
|------|------|
| `market_trend_daily.csv` | 每日趋势状态（32列：日期、趋势、综合得分、7维子分、广度原始指标等） |

### 输出字段说明

| 字段 | 含义 |
|------|------|
| `date` | 交易日期 YYYYMMDD |
| `trend` | 趋势状态（STRONG_UP/UP/NEUTRAL/DOWN/STRONG_DOWN） |
| `composite_score` | 综合得分 [-1, 1] |
| `breadth_score` ~ `liquidity_score` | 7个维度子分（各自 [-1, 1]） |
| `advance_ratio` | 上涨股占比 |
| `above_ma20_ratio` | 站上 MA20 占比 |
| `limit_up_count` / `limit_down_count` | 涨/跌停家数 |
| `breadth_thrust` | 广度推力是否触发 |

## 7 维度详解

### 微观维度（个股聚合）

| 维度 | 权重 | 核心指标 | 正分含义 |
|------|------|----------|----------|
| 广度 Breadth | 25% | 涨跌比、MA20/60站上比、涨跌停、新高比、广度推力 | 多头扩散 |
| 资金流 Money Flow | 20% | 大单净流入占比、全市场大单净额 | 主力净流入 |
| 波动结构 Volatility | 10% | 5日波动率中位数、恐慌度（跌>5%股票占比） | 低波稳定 |
| 熵/有序度 Entropy | 15% | 排列熵中位数、有序股票占比 | 市场有序运行 |
| 动量扩散 Momentum | 10% | 行业动量离散度、涨跌方向一致性 | 板块一致上涨 |

### 宏观维度（外部数据）

| 维度 | 权重 | 核心指标 | 正分含义 |
|------|------|----------|----------|
| 杠杆资金 Leverage | 10% | 两融余额变化率、MA5/MA20金叉、融资净买入 | 增量资金入场 |
| 流动性 Liquidity | 10% | 隔夜SHIBOR水平、均值偏离、日变化 | 资金面宽松 |

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ma_short` | 20 | 短期均线窗口 |
| `ma_long` | 60 | 长期均线窗口 |
| `entropy_order` | 3 | 排列熵阶数 |
| `entropy_window` | 20 | 排列熵滚动窗口 |
| `momentum_window` | 20 | 动量计算窗口 |
| `strong_up_threshold` | 0.5 | 强势上涨门槛 |
| `up_threshold` | 0.2 | 上涨门槛 |
| `down_threshold` | -0.2 | 下跌门槛 |
| `strong_down_threshold` | -0.5 | 强势下跌门槛 |
| `workers` | 8 | 多进程并行数 |
| `min_bars` | 60 | 个股最少数据天数 |

## 数据需求

| 数据源 | 路径 | 用途 |
|--------|------|------|
| 个股日线(含资金流) | `tushare-daily-full/` | 5维微观指标聚合 |
| 指数日线 | `tushare-index-daily/` | 指数对照 |
| 涨跌停数据 | `tushare-stk_limit/` | 广度（涨跌停判定） |
| 两融数据 | `tushare-margin/margin.csv` | 杠杆资金维度 |
| SHIBOR数据 | `tushare-shibor/shibor.csv` | 流动性维度 |
| 行业分类 | `tushare-index_member_all/index_member_all.csv` | 动量扩散（行业分组） |
| 股票基本信息 | `tushare_stock_basic.csv` | 名称映射 |

## 策略代码

```
src/strategy/market_trend/
├── run_market_trend.py     # CLI 入口
├── config.py               # MarketTrendConfig + TrendState 数据类
├── data_loader.py          # 批量数据加载
├── micro_indicators.py     # 5维微观指标计算（含排列熵）
├── macro_indicators.py     # 2维宏观指标计算
└── trend_engine.py         # 7维评分 → 综合 → 趋势判定
```

## Wiki 参考

- `wiki/entities/market-trend-system.md` — 策略实体页
- `wiki/experiments/market-trend-backtest-2024.md` — 672天回测分析
