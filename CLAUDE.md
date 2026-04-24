# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`gp-quant` is an A-share (Chinese stock market) quantitative research workspace focused on complexity theory and econophysics. The system uses entropy-based indicators and multi-timeframe resonance to identify trading opportunities.

## Core Architecture

```
gp-quant/
├── src/
│   ├── core/                      # Core entropy calculation module (tick-level)
│   ├── strategy/                  # Strategy implementations
│   │   ├── multitimeframe/        # Main strategy: daily/weekly/monthly resonance
│   │   ├── entropy_bifurcation_setup/  # 4-layer entropy system
│   │   ├── four_layer_entropy_system/  # Minute-level 4-layer trading system
│   │   ├── dual_entropy_accumulation/  # Intraday + daily entropy fusion
│   │   ├── continuous_decline_recovery/ # Mean reversion after decline
│   │   ├── uptrend_hold_state_flow/    # State-flow based hold/exit
│   │   ├── factor_model_selection/     # Factor model + Bull Hunter v3
│   │   └── adaptive_state_machine/     # 4-agent closed-loop: dynamic weights + thresholds
│   ├── downloader/                # Tushare/Tencent data sync
│   └── web/                       # Flask dashboard (auto-discovers strategies)
├── wiki/                          # LLM-maintained knowledge base (Karpathy Wiki pattern)
├── results/                       # Backtest outputs (separate from code)
├── scripts/                       # Shell entry points
└── .github/skills/                # Claude skills for recurring tasks
```

## Strategy Hierarchy

| Strategy | Timeframe | Status |
|----------|-----------|--------|
| `multitimeframe-scanner` | Daily/Weekly/Monthly | Production |
| `entropy-accumulation-breakout` | Daily (3-stage FSM + multi-source) | Production |
| `entropy-bifurcation` | Daily (20-day rolling) | Research |
| `four-layer-system` | Minute-level | Experimental |
| `dual-entropy-accumulation` | Intraday + Daily | Research |
| `continuous-decline-recovery` | Daily | Research |
| `hold-exit-system` | Daily | Research |
| `bull-hunter-v3` | Daily (LightGBM 30%/100%/200%) | Research |
| `adaptive-state-machine` | Daily (4-agent closed-loop, 37 factors) | Research |

## Environment & Commands

### Setup
```bash
pip install -r requirements.txt
# Requires: pandas, numpy, scipy, flask, tushare, chinesecalendar
```

### Data Download
```bash
# Full data sync (stock list + daily OHLCV + financials)
./scripts/run_get_tushare_all.sh -o /path/to/gp-data --token <tushare_token>

# Incremental 1-minute sync (Tencent source, free)
./scripts/run_sync_a_share_1m.sh -o /path/to/gp-data

# EOD scheduler (16:00 daily auto-update)
./scripts/run_eod_data_scheduler.sh -o /path/to/gp-data
```

### Run Strategy Scans
```bash
# Multi-timeframe resonance (main production strategy)
./scripts/run_multitimeframe_resonance_scan.sh \
  --scan-date 20260309 \
  --top_n 30 \
  --max-positions 10

# Entropy bifurcation 4-layer system
./scripts/run_entropy_bifurcation_setup.sh \
  --scan-date 20260330

# Four-layer minute-level system
python -m src.strategy.four_layer_entropy_system.run_scan \
  --data_dir /path/to/gp-data/trade \
  --max_stocks 50

# Entropy accumulation breakout (3-stage FSM, with feature cache)
python -m src.strategy.entropy_accumulation_breakout.run_entropy_accumulation_breakout \
  --data_dir /path/to/gp-data/tushare-daily-full \
  --feature_cache_dir /path/to/gp-data/feature-cache \
  --scan_date 20260416

# Bull Hunter v3 (LightGBM 4-agent pipeline: 30%/100%/200% targets)
python -m src.strategy.factor_model_selection.v3_bull_hunter.run_bull_hunter \
  --scan_date 20260419

# Bull Hunter v3 backtest with actual P&L
python -m src.strategy.factor_model_selection.v3_bull_hunter.run_bull_hunter \
  --backtest --start_date 20250301 --end_date 20251230 \
  --interval_days 20 --top_n 10

# Adaptive State Machine (4-agent closed-loop)
python -m src.strategy.adaptive_state_machine.run_adaptive_state_machine \
  --backtest --start_date 20250101 --end_date 20260331 --interval_days 20
```

### Backtesting
```bash
# Forward backtest via scan script
./scripts/run_multitimeframe_resonance_scan.sh \
  --backtest-start-date 20260101 \
  --backtest-end-date 20260331 \
  --hold-days 5

# Analyze results
python tests/analyze_backtest_results.py --results-dir <dir>
```

### Web Dashboard
```bash
python web/app.py --port 5050
# Auto-discovers strategies from src/strategy/*/README.md
```

## Key Metrics

### Entropy Indicators (src/core/tick_entropy.py)
| Metric | Range | Interpretation |
|--------|-------|----------------|
| `path_irreversibility` | 0-0.3+ | Main force control (>0.2 = strong) |
| `permutation_entropy` | 0-1 | Order/disorder (<0.5 = ordered) |
| `dominant_eigenvalue` | -1 to 1 | Critical slowing (|λ|>0.9 = warning) |
| `turnover_entropy` | 0-1 | Concentration (<0.5 = concentrated) |

### Market States
- `ordered`: Trend-following viable
- `weak_chaos`: Normal trading
- `strong_chaos`: Avoid trend strategies
- `critical`: Bifurcation warning

## Data & Code Mapping

### Data Directory (`gp-data/`)

```
gp-data/                                   ← 外部数据目录，不在版本控制中
├── tushare_stock_basic.csv                # 5,509只股票元数据 (ts_code,name,area,industry,market,list_date)
├── tushare_gplist.json                    # 股票列表 JSON 数组 (sz/sh/bj 格式)
├── .eod_data_scheduler_state.json         # EOD调度器运行状态
│
├── tushare-daily-full/*.csv               # 日线 OHLCV + 资金流 + 估值 (43列)
│   └── {sz|sh|bj}{code}.csv               # 5,509文件，最新至当日收盘
│
├── tushare-trade_cal/trade_cal.csv        # 交易日历 (2001~2026)
├── tushare-adj_factor/{symbol}.csv        # 复权因子 (ts_code,trade_date,adj_factor)
├── tushare-stk_limit/{symbol}.csv         # 涨跌停价 (trade_date,up_limit,down_limit)
├── tushare-suspend_d/{symbol}.csv         # 停复牌 (trade_date,suspend_timing,suspend_type)
├── tushare-dividend/{symbol}.csv          # 分红送转 (ann_date,ex_date,cash_div等)
│
├── tushare-income/{symbol}.csv            # 利润表 (80+列)
├── tushare-balancesheet/{symbol}.csv      # 资产负债表 (110+列)
├── tushare-cashflow/{symbol}.csv          # 现金流量表 (80+列)
├── tushare-fina_indicator/{symbol}.csv    # 财务指标 (90+列: ROE/ROA/PE/PB等)
│   └── ann_date=季度报告发布日期, end_date=报告期
│
├── trade/<symbol>/                        # 1分钟行情 (Tencent/Tushare源)
│   └── YYYY-MM-DD.csv                     # 按日存储: 时间,开盘,收盘,最高,最低,成交量,成交额,均价,换手率
│
└── feature-cache/                         # 策略特征缓存 (entropy-bifurcation用)
```

### 注意：周线/月线无独立文件

周线和月线数据在 `multitimeframe_feature_engine.py:aggregate_stock_bars()` 中从日线实时聚合：
- 周线: `df["period"] = df["dt"].dt.to_period("W-FRI")`
- 月线: `df["period"] = df["dt"].dt.to_period("M")`

### 策略 → 数据依赖关系

| 策略模块 | 数据源 | 核心文件 |
|----------|--------|----------|
| **multitimeframe** (主力) | `tushare-daily-full/` + `tushare_stock_basic.csv` | `multitimeframe_scan_service.py` 读日线 → `feature_engine.py` 聚合D/W/M → `evaluation.py` 共振评估 |
| **entropy-bifurcation** | `tushare-daily-full/` + `feature-cache/` | 20日滚动熵值分叉，特征缓存加速 |
| **four-layer-system** | `trade/<symbol>/` (1m) | 四层分钟级交易系统 |
| **bull-hunter-v3** | `tushare-daily-full/` + `tushare-fina_indicator/` + `tushare-income/` | LightGBM因子模型 (30%/100%/200%目标) |
| **entropy-accumulation-breakout** | `tushare-daily-full/` + `feature-cache/` | 3阶段FSM，特征缓存~17x加速 |
| **hold-exit-system** | `tushare-daily-full/` | 状态流持仓/退出决策 |
| **continuous-decline-recovery** | `tushare-daily-full/` | 连续下跌均值回归 |
| **dual-entropy-accumulation** | `tushare-daily-full/` + `trade/` | 日内+日线熵值融合 |
| **adaptive-state-machine** | `tushare-daily-full/` | 4 Agent 闭环: 因子→权重→状态→验证反馈 |

### 数据同步流程

```
run_get_tushare_all.sh         → 全量下载: 股票列表 + 日线历史 + 扩展数据
run_get_tushare_daily_full.sh  → 仅全量日线 (支持 --threads, --start-date, --end-date)
run_get_tushare_extended.sh    → 仅扩展数据 (复权/财务/分红/停复牌)
run_sync_a_share_1m.sh         → 增量1分钟数据 (默认最近3日, Tencent免费源)
run_eod_data_scheduler.sh      → 调度器: 16:00触发 → fast_sync_tushare_latest.py + 1分钟同步
```

调度器 (`eod_data_scheduler.py`) 行为：
1. 先运行 `fast_sync_tushare_latest.py`：增量更新股票列表、交易日历、日线、复权、停复牌、分红、财务
2. 再运行 `sync_a_share_1m.py`：同步当天1分钟数据（默认Tencent源）
3. 状态记录在 `gp-data/.eod_data_scheduler_state.json`
4. 锁文件 `gp-data/.eod_data_scheduler.lock` 防止重复启动

### Minimum CSV Columns (日线)
- `trade_date`, `open`, `close`, `amount`
- `turnover_rate`, `net_mf_amount`
- 可选: `industry`, `market`, `area`

## Design Decisions (from wiki/)

| Decision | Rationale |
|----------|-----------|
| **Daily over minute** | Minute-level entropy shows 35% win rate (vs random) |
| **Gray-box over black-box** | Structural PINN > pure neural nets |
| **Market filter required** | A-shape: 44% DOWN, 43% NEUTRAL, 7% UP |
| **STRONG_DOWN = buy signal** | Panic → +3.25% avg return, 58% win rate |

## Skills System

Available via `/skill` command:
- `backtest-analyze`: Run backtests and analyze performance metrics
- `entropy-bifurcation`: Run entropy bifurcation scans
- `entropy-accumulation-breakout`: Entropy accumulation breakout 3-stage FSM
- `market-scan`: Multi-timeframe resonance scans
- `hold-exit-decision`: Hold/exit system execution
- `wiki-ingest`: Generate stock wikis from Xueqiu data
- `complexity-reference`: Query complexity theory concepts

## Testing

```bash
# Feature engine tests
python -m pytest tests/strategy/continuous_decline_recovery/

# Smoke test with 2 stocks
./scripts/run_multitimeframe_resonance_scan.sh \
  --symbols sh600000,sz000001 \
  --out-dir /tmp/smoke_test
```

## Important Notes

- **Tushare token required** for data download (free tier available)
- **Minute data**: Free Tencent source only covers ~3 trading days
- **Entropy factors**: Only validated on daily timeframe, not minute-level
- **Results directory**: All outputs go to `results/`, not code directory
- **Feature cache**: `entropy-accumulation-breakout` uses incremental cache at `gp-data/feature-cache/` (~17x speedup)
- **Wiki pattern**: LLM maintains `wiki/` as structured knowledge base (read docs/papers/, write wiki/)
