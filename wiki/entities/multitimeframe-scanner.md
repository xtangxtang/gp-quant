---
title: 多时间框架共振扫描器
tags: [strategy, multitimeframe, scanner]
confidence: high
status: active
created: 2026-04-12
updated: 2026-04-12
---

# 多时间框架共振扫描器 (Multi-Timeframe Scanner)

**源码**: `src/strategy/multitimeframe/`  
**运行**: `scripts/run_multitimeframe_resonance_scan.sh`  
**入口**: `src/strategy/multitimeframe/run_multitimeframe_resonance_scan.py`

## 功能

主力生产策略。扫描全市场，寻找**日/周/月三个时间框架同时趋势对齐**的股票。

## 核心模块

| 文件 | 功能 |
|------|------|
| `multitimeframe_feature_engine.py` | 聚合日/周/月 K 线 + 计算 5 个物理特征 |
| `multitimeframe_evaluation.py` | 单框架评分 + 多框架共振评分 |
| `multitimeframe_physics_utils.py` | 熵、Hurst、z-score、regime 工具 |
| `multitimeframe_scan_service.py` | 全市场扫描服务 |

## 输入

- 日线 K 线: `tushare-daily-full/`
- 股票基本信息: `tushare_stock_basic.csv`

## 输出

| 文件 | 说明 |
|------|------|
| `market_scan_snapshot_<date>.csv` | 全市场状态快照 |
| `resonance_candidates_<date>_all.csv` | 所有共振股票 |
| `resonance_candidates_<date>_top30.csv` | Top 30 共振 |
| `selected_portfolio_<date>_top30.csv` | 最终组合（过滤后） |
| `forward_backtest_daily_<date>.csv` | 净值与日收益 |
| `forward_backtest_trades_<date>.csv` | 逐笔交易记录 |

## 关键参数

见 [multitimeframe-resonance](../concepts/multitimeframe-resonance.md) 的生产参数表。

## Smoke Test

```bash
python src/strategy/multitimeframe/run_multitimeframe_resonance_scan.py \
    --data_dir /nvme5/xtang/gp-workspace/gp-data/tushare-daily-full \
    --basic_path /nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv \
    --symbols sh600000,sz000001,sh601398 \
    --out_dir /tmp/gp_quant_smoke
```

## 相关实体

- [tick-entropy-module](tick-entropy-module.md) — 物理特征计算基础
- [four-layer-system](four-layer-system.md) — 互补策略（熵分岔 vs 多框架共振）
- [data-pipeline](data-pipeline.md) — 数据源

## 相关概念

- [multitimeframe-resonance](../concepts/multitimeframe-resonance.md) — 理论基础
- [fractal](../concepts/fractal.md) — 多尺度自相似性
