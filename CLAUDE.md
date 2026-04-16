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
│   │   └── uptrend_hold_state_flow/    # State-flow based hold/exit
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
| `entropy-bifurcation` | Daily (20-day rolling) | Research |
| `four-layer-system` | Minute-level | Experimental |
| `dual-entropy-accumulation` | Intraday + Daily | Research |
| `continuous-decline-recovery` | Daily | Research |
| `hold-exit-system` | Daily | Research |

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

## Data Conventions

```
/path/to/gp-data/
├── tushare-daily-full/        # Daily OHLCV (~5500 stocks)
├── tushare_stock_basic.csv    # Stock metadata (industry, area, market)
├── trade/<symbol>/            # Minute data (~13.5M files)
├── tushare-moneyflow/         # Capital flow
└── tushare-adj_factor/        # Adjustment factors
```

### Minimum CSV Columns
- `trade_date`, `open`, `close`, `amount`
- `turnover_rate`, `net_mf_amount`
- Optional: `industry`, `market`, `area`

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
- **Wiki pattern**: LLM maintains `wiki/` as structured knowledge base (read docs/papers/, write wiki/)
