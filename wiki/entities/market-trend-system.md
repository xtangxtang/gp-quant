---
title: 大盘趋势判断系统（从小见大）
tags: [market-trend, breadth, macro, entropy, leverage, shibor]
confidence: high
status: active
sources: []
created: 2026-04-13
updated: 2026-04-13
---

# 大盘趋势判断系统 — Market Trend from Micro Structure

## 定位

从 5000+ 只个股的微观交易行为 + 宏观资金/流动性数据中聚合判断大盘趋势状态。**不直接看指数K线**，而是从个股行为「从小见大」推断市场整体方向。

可作为其他选股策略的**前置过滤器**（大盘 DOWN 时不开仓），也可独立生成趋势时间线。

## 源码位置

```
src/strategy/market_trend/
├── __init__.py
├── config.py              # 配置: 7维权重、趋势门槛、数据路径
├── data_loader.py         # 数据加载: 个股日线、指数、涨跌停、两融、SHIBOR、行业分类
├── micro_indicators.py    # 微观指标引擎: 5维 (广度/资金流/波动/熵/动量)
├── macro_indicators.py    # 宏观指标引擎: 2维 (杠杆/流动性)
├── trend_engine.py        # 主引擎: 7维评分 → 综合得分 → 趋势判定
└── run_market_trend.py    # CLI 入口
```

运行脚本: `scripts/run_market_trend.sh`

## 7 维度评分体系

### 微观维度（从个股聚合）

| 维度 | 权重 | 数据来源 | 核心指标 |
|------|------|----------|----------|
| **广度 Breadth** | 25% | 个股涨跌/均线/涨跌停 | advance_ratio, above_ma20_ratio, limit_up/down_count, 广度推力 |
| **资金流 Money Flow** | 20% | 个股大单资金流 | net_inflow_ratio, big_order_net_sum |
| **波动结构 Volatility** | 10% | 个股波动率分布 | vol_median (5日实现波动率中位数), panic_ratio (跌>5%占比) |
| **熵/有序度 Entropy** | 15% | 个股排列熵分布 | entropy_median, ordering_ratio (低熵有序股占比) |
| **动量扩散 Momentum** | 10% | 行业动量一致性 | sector_momentum_std, trend_alignment |

### 宏观维度（外部数据）

| 维度 | 权重 | 数据来源 | 核心指标 |
|------|------|----------|----------|
| **杠杆资金 Leverage** | 10% | tushare-margin (两融余额) | margin_balance_chg_pct, MA5/MA20交叉, margin_net_buy |
| **流动性 Liquidity** | 10% | tushare-shibor (银行间利率) | shibor_on 水平/均值偏离/日变化 |

## 趋势判定规则

| 状态 | 条件 |
|------|------|
| STRONG_UP | 综合得分 ≥ 0.5 且广度推力触发 |
| UP | 综合得分 ≥ 0.2 |
| NEUTRAL | -0.2 < 综合得分 < 0.2 |
| DOWN | 综合得分 ≤ -0.2 |
| STRONG_DOWN | 综合得分 ≤ -0.5 且恐慌占比 > 3% |

## 数据依赖

| 数据 | 路径 | 用途 |
|------|------|------|
| 个股日线 (含资金流) | `tushare-daily-full/` (~5500 只) | 主数据源: OHLCV + 大单 + 换手率 |
| 大盘指数 | `tushare-index-daily/` | 对照基准 (上证综指) |
| 涨跌停价 | `tushare-stk_limit/` | 涨停/跌停家数 |
| 两融余额 | `tushare-margin/margin.csv` | 杠杆资金指标 |
| SHIBOR 利率 | `tushare-shibor/shibor.csv` | 流动性指标 |
| 行业分类 | `tushare-index_member_all/` | 行业动量扩散 |
| 股票名称 | `tushare_stock_basic.csv` | ST 过滤 |

## 输出

| 文件 | 内容 |
|------|------|
| `market_trend_daily.csv` | 每日一行: 趋势状态 + 综合得分 + 7维子分 + 原始指标 + 指数对照 (32列) |
| `market_trend_summary.csv` | 各趋势状态天数统计 |

## 性能

- 5182 只A股 × 672天 ≈ 400秒 (含数据加载58秒、特征计算117秒、逐日扫描229秒)
- 单日扫描约0.3秒

## 与其他策略的关系

- 可作为 [[multitimeframe-scanner]]、[[entropy-accumulation-breakout]]、[[four-layer-system]] 的大盘环境过滤器
- 当 trend=DOWN 时，建议降低新开仓；trend=STRONG_DOWN 时反而可抄底
- 杠杆维度 (margin MA5/MA20 交叉) 可作为中期仓位管理信号

## 关键发现 (2023.07 ~ 2026.04 回测)

详见 [[market-trend-backtest-2024]]

- DOWN 占 44%、NEUTRAL 占 43%、UP 仅 7%——A 股多数时间不在上涨
- STRONG_DOWN 后 20 日平均涨 +3.25%，胜率 58%——恐慌即机会
- 资金流得分在整段行情中**持续为负**——机构长期在派发，散户和杠杆在买入
- 杠杆是核心驱动: 两融余额 14000→26000 亿的扩张支撑了 3 年慢牛
- 熵得分始终为负——A股市场一直缺乏持续有序趋势（高排列熵）
