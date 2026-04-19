#!/bin/bash
# 因子模型选股 — 4-Agent Pipeline
#
# 用法:
#   # 单次运行
#   ./scripts/run_factor_model_pipeline.sh --scan-date 20260410
#
#   # 滚动回测
#   ./scripts/run_factor_model_pipeline.sh --start-date 20250101 --end-date 20250331
#
#   # 自定义 horizons
#   ./scripts/run_factor_model_pipeline.sh --scan-date 20260410 --horizons 3d,5d,1w

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 默认路径
DEFAULT_CACHE_DIR="/nvme5/xtang/gp-workspace/gp-data/feature-cache"
DEFAULT_DATA_DIR="/nvme5/xtang/gp-workspace/gp-data/tushare-daily-full"
DEFAULT_BASIC_PATH="/nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv"
DEFAULT_OUT_DIR="$PROJECT_ROOT/results/factor_model_selection"

# 解析参数
CACHE_DIR="${CACHE_DIR:-$DEFAULT_CACHE_DIR}"
DATA_DIR="${DATA_DIR:-$DEFAULT_DATA_DIR}"
BASIC_PATH="${BASIC_PATH:-$DEFAULT_BASIC_PATH}"
OUT_DIR="${OUT_DIR:-$DEFAULT_OUT_DIR}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cache-dir)   CACHE_DIR="$2";   shift 2 ;;
        --data-dir)    DATA_DIR="$2";    shift 2 ;;
        --basic-path)  BASIC_PATH="$2";  shift 2 ;;
        --out-dir)     OUT_DIR="$2";     shift 2 ;;
        *)             EXTRA_ARGS+=("$1"); shift ;;
    esac
done

cd "$PROJECT_ROOT"

python -m src.strategy.factor_model_selection.pipeline \
    --cache_dir "$CACHE_DIR" \
    --data_dir "$DATA_DIR" \
    --basic_path "$BASIC_PATH" \
    --out_dir "$OUT_DIR" \
    "${EXTRA_ARGS[@]}"
