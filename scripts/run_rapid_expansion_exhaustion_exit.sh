#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_DATA_DIR="/nvme5/xtang/gp-workspace/gp-data/tushare-daily-full"
DEFAULT_BASIC_PATH="/nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv"
DEFAULT_OUT_DIR="$WORKSPACE_DIR/results/rapid_expansion_exhaustion_exit"
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
SYMBOL_OR_NAME=""
START_DATE=""
SCAN_DATE=""
BASIC_PATH="$DEFAULT_BASIC_PATH"
LOOKBACK_YEARS="5"
EXIT_PERSIST_DAYS="2"
EXTRA_ARGS=()

show_help() {
  echo "用法: $0 [选项]"
  echo "选项:"
  echo "  --symbol-or-name <value> 股票代码或股票名称，例如 sh688268 / 688268.SH / 华特气体"
  echo "  --start-date <yyyymmdd>  起始持有日期，必填"
  echo "  --data-dir <dir>         日线 CSV 目录"
  echo "  --out-dir <dir>          输出目录，默认写到 results/rapid_expansion_exhaustion_exit"
  echo "  --scan-date <yyyymmdd>   评估日期；默认自动推断最新交易日"
  echo "  --basic-path <file>      tushare_stock_basic.csv 路径"
  echo "  --lookback-years <n>     特征回看年数，默认 5"
  echo "  --exit-persist-days <n>  连续多少天满足衰竭退出才确认离场，默认 2"
  echo "  -h, --help               显示帮助"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --symbol-or-name)
      SYMBOL_OR_NAME="$2"; shift 2 ;;
    --start-date)
      START_DATE="$2"; shift 2 ;;
    --data-dir)
      DATA_DIR="$2"; shift 2 ;;
    --out-dir)
      OUT_DIR="$2"; shift 2 ;;
    --scan-date)
      SCAN_DATE="$2"; shift 2 ;;
    --basic-path)
      BASIC_PATH="$2"; shift 2 ;;
    --lookback-years)
      LOOKBACK_YEARS="$2"; shift 2 ;;
    --exit-persist-days)
      EXIT_PERSIST_DAYS="$2"; shift 2 ;;
    -h|--help)
      show_help; exit 0 ;;
    *)
      EXTRA_ARGS+=("$1")
      shift ;;
  esac
done

if [[ -z "$SYMBOL_OR_NAME" ]]; then
  echo "缺少必填参数: --symbol-or-name" >&2
  exit 1
fi

if [[ -z "$START_DATE" ]]; then
  echo "缺少必填参数: --start-date" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

CMD=(
  src/strategy/uptrend_hold_state_flow/rapid_expansion_exhaustion_exit/run_rapid_expansion_exhaustion_scan.py
  --data_dir "$DATA_DIR"
  --out_dir "$OUT_DIR"
  --symbol_or_name "$SYMBOL_OR_NAME"
  --start_date "$START_DATE"
  --basic_path "$BASIC_PATH"
  --lookback_years "$LOOKBACK_YEARS"
  --exit_persist_days "$EXIT_PERSIST_DAYS"
)

if [[ -n "$SCAN_DATE" ]]; then
  CMD+=(--scan_date "$SCAN_DATE")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

cd "$WORKSPACE_DIR"
"${PYTHON_CMD[@]}" "${CMD[@]}"