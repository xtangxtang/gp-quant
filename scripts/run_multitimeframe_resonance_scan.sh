#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_DATA_DIR="/nvme5/xtang/gp-workspace/gp-data/tushare-daily-full"
DEFAULT_BASIC_PATH="/nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv"
DEFAULT_OUT_DIR="$WORKSPACE_DIR/results/multitimeframe_resonance/out_2025_multitimeframe_fullscan"
PYTHON_CMD=(
  /root/miniforge3/bin/conda
  run
  -p
  /root/miniforge3
  --no-capture-output
  python
  /root/.vscode-server/extensions/ms-python.python-2026.2.0-linux-x64/python_files/get_output_via_markers.py
)

DATA_DIR="$DEFAULT_DATA_DIR"
OUT_DIR="$DEFAULT_OUT_DIR"
TEST_YEAR="2025"
BASIC_PATH="$DEFAULT_BASIC_PATH"
INDEX_PATH=""
TOP_N="300"
SYMBOLS=""
EXTRA_ARGS=()

show_help() {
  echo "用法: $0 [选项]"
  echo "选项:"
  echo "  --data-dir <dir>         日线 CSV 目录"
  echo "  --out-dir <dir>          输出目录，默认写到 results/multitimeframe_resonance/"
  echo "  --test-year <year>       测试年份，默认 2025"
  echo "  --basic-path <file>      tushare_stock_basic.csv 路径"
  echo "  --index-path <file>      可选指数 CSV 路径"
  echo "  --top-n <n>              进入详细扫描的股票数量，默认 300"
  echo "  --symbols <csv>          可选，逗号分隔股票列表，仅做小样本时使用"
  echo "  -h, --help               显示帮助"
  echo ""
  echo "示例:"
  echo "  $0"
  echo "  $0 --test-year 2024"
  echo "  $0 --symbols sh600000,sz000001 --out-dir /tmp/gp_quant_resonance_smoke"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir)
      DATA_DIR="$2"; shift 2 ;;
    --out-dir)
      OUT_DIR="$2"; shift 2 ;;
    --test-year)
      TEST_YEAR="$2"; shift 2 ;;
    --basic-path)
      BASIC_PATH="$2"; shift 2 ;;
    --index-path)
      INDEX_PATH="$2"; shift 2 ;;
    --top-n)
      TOP_N="$2"; shift 2 ;;
    --symbols)
      SYMBOLS="$2"; shift 2 ;;
    -h|--help)
      show_help; exit 0 ;;
    *)
      EXTRA_ARGS+=("$1")
      shift ;;
  esac
done

mkdir -p "$OUT_DIR"

CMD=(
  src/analysis/run_multitimeframe_resonance_scan.py
  --data_dir "$DATA_DIR"
  --out_dir "$OUT_DIR"
  --test_year "$TEST_YEAR"
  --basic_path "$BASIC_PATH"
  --top_n "$TOP_N"
)

if [[ -n "$INDEX_PATH" ]]; then
  CMD+=(--index_path "$INDEX_PATH")
fi

if [[ -n "$SYMBOLS" ]]; then
  CMD+=(--symbols "$SYMBOLS")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

cd "$WORKSPACE_DIR"
"${PYTHON_CMD[@]}" "${CMD[@]}"