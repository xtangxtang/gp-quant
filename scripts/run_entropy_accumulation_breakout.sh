#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_DATA_DIR="/nvme5/xtang/gp-workspace/gp-data/tushare-daily-full"
DEFAULT_BASIC_PATH="/nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv"
DEFAULT_OUT_DIR="$WORKSPACE_DIR/results/entropy_accumulation_breakout"
DEFAULT_FEATURE_CACHE_DIR="/nvme5/xtang/gp-workspace/gp-data/feature-cache"
PYTHON_CMD=(
  /root/miniforge3/bin/conda
  run
  -p
  /root/miniforge3
  --no-capture-output
  python
)

DATA_DIR="$DEFAULT_DATA_DIR"
OUT_DIR="$DEFAULT_OUT_DIR"
SCAN_DATE=""
BASIC_PATH="$DEFAULT_BASIC_PATH"
TOP_N="30"
SYMBOLS=""
MIN_AMOUNT="500000"
MIN_TURNOVER="0.5"
EXCLUDE_ST="1"
FEATURE_CACHE_DIR="$DEFAULT_FEATURE_CACHE_DIR"
DATA_ROOT=""
BACKTEST_START_DATE=""
BACKTEST_END_DATE=""
HOLD_DAYS="5"
MAX_POSITIONS="10"
MAX_POSITIONS_PER_INDUSTRY="2"
EXTRA_ARGS=()

show_help() {
  echo "用法: $0 [选项]"
  echo ""
  echo "熵惜售分岔突破策略 - 扫描 / 回测"
  echo ""
  echo "选项:"
  echo "  --data-dir <dir>         日线 CSV 目录"
  echo "  --out-dir <dir>          输出目录"
  echo "  --scan-date <yyyymmdd>   扫描日期"
  echo "  --basic-path <file>      tushare_stock_basic.csv 路径"
  echo "  --top-n <n>              输出前 N 只候选 (默认 30)"
  echo "  --symbols <csv>          逗号分隔股票列表"
  echo "  --min-amount <num>       最低成交额过滤 (默认 500000)"
  echo "  --min-turnover <num>     最低换手率过滤 (默认 0.5)"
  echo "  --include-st             不排除 ST 股票"
  echo "  --feature-cache-dir <dir> 特征缓存目录 (默认: gp-data/feature-cache)"
  echo "  --data-root <dir>        数据根目录 (自动推断)"
  echo "  --backtest-start-date <yyyymmdd>  回测起始日期"
  echo "  --backtest-end-date <yyyymmdd>    回测结束日期"
  echo "  --hold-days <n>          持有天数 (默认 5)"
  echo "  --max-positions <n>      最大持仓数 (默认 10)"
  echo "  --verbose                详细日志"
  echo "  -h, --help               帮助"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir)
      DATA_DIR="$2"; shift 2 ;;
    --out-dir)
      OUT_DIR="$2"; shift 2 ;;
    --scan-date)
      SCAN_DATE="$2"; shift 2 ;;
    --basic-path)
      BASIC_PATH="$2"; shift 2 ;;
    --top-n)
      TOP_N="$2"; shift 2 ;;
    --symbols)
      SYMBOLS="$2"; shift 2 ;;
    --min-amount)
      MIN_AMOUNT="$2"; shift 2 ;;
    --min-turnover)
      MIN_TURNOVER="$2"; shift 2 ;;
    --include-st)
      EXCLUDE_ST="0"; shift ;;
    --feature-cache-dir)
      FEATURE_CACHE_DIR="$2"; shift 2 ;;
    --data-root)
      DATA_ROOT="$2"; shift 2 ;;
    --backtest-start-date)
      BACKTEST_START_DATE="$2"; shift 2 ;;
    --backtest-end-date)
      BACKTEST_END_DATE="$2"; shift 2 ;;
    --hold-days)
      HOLD_DAYS="$2"; shift 2 ;;
    --max-positions)
      MAX_POSITIONS="$2"; shift 2 ;;
    --max-positions-per-industry)
      MAX_POSITIONS_PER_INDUSTRY="$2"; shift 2 ;;
    -h|--help)
      show_help; exit 0 ;;
    *)
      EXTRA_ARGS+=("$1")
      shift ;;
  esac
done

mkdir -p "$OUT_DIR"

CMD=(
  src/strategy/entropy_accumulation_breakout/run_entropy_accumulation_breakout.py
  --data_dir "$DATA_DIR"
  --out_dir "$OUT_DIR"
  --basic_path "$BASIC_PATH"
  --top_n "$TOP_N"
  --min_amount "$MIN_AMOUNT"
  --min_turnover "$MIN_TURNOVER"
  --hold_days "$HOLD_DAYS"
  --max_positions "$MAX_POSITIONS"
  --max_positions_per_industry "$MAX_POSITIONS_PER_INDUSTRY"
)

if [[ -n "$FEATURE_CACHE_DIR" ]]; then
  CMD+=(--feature_cache_dir "$FEATURE_CACHE_DIR")
fi

if [[ -n "$DATA_ROOT" ]]; then
  CMD+=(--data_root "$DATA_ROOT")
fi

if [[ -n "$SCAN_DATE" ]]; then
  CMD+=(--scan_date "$SCAN_DATE")
fi

if [[ -n "$SYMBOLS" ]]; then
  CMD+=(--symbols "$SYMBOLS")
fi

if [[ "$EXCLUDE_ST" == "0" ]]; then
  CMD+=(--include_st)
fi

if [[ -n "$BACKTEST_START_DATE" ]]; then
  CMD+=(--backtest_start_date "$BACKTEST_START_DATE")
fi

if [[ -n "$BACKTEST_END_DATE" ]]; then
  CMD+=(--backtest_end_date "$BACKTEST_END_DATE")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

cd "$WORKSPACE_DIR"
"${PYTHON_CMD[@]}" "${CMD[@]}"
