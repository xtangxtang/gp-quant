#!/usr/bin/env bash
# 大盘趋势判断 - 从小见大
# 用法: bash scripts/run_market_trend.sh [START_DATE] [END_DATE]

set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT="/nvme5/xtang/gp-workspace/gp-data"
OUT_DIR="./results/market_trend"

START_DATE="${1:-20240101}"
END_DATE="${2:-20260410}"

mkdir -p "$OUT_DIR"

echo "=========================================="
echo " 大盘趋势判断: 从小见大"
echo " 日期范围: $START_DATE ~ $END_DATE"
echo "=========================================="

python -m src.strategy.market_trend.run_market_trend \
    --data_dir "$DATA_ROOT/tushare-daily-full" \
    --index_dir "$DATA_ROOT/tushare-index-daily" \
    --stk_limit_dir "$DATA_ROOT/tushare-stk_limit" \
    --margin_path "$DATA_ROOT/tushare-margin/margin.csv" \
    --shibor_path "$DATA_ROOT/tushare-shibor/shibor.csv" \
    --index_member_path "$DATA_ROOT/tushare-index_member_all/index_member_all.csv" \
    --basic_path "$DATA_ROOT/tushare_stock_basic.csv" \
    --out_dir "$OUT_DIR" \
    --start_date "$START_DATE" \
    --end_date "$END_DATE" \
    --workers 8

echo ""
echo "结果输出: $OUT_DIR/"
ls -lh "$OUT_DIR/"*.csv 2>/dev/null || true
