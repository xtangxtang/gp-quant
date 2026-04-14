#!/usr/bin/env bash
# 雪球讨论 — 订单与基本面信息监控
# 拉取当天所有讨论, 通过 LLM(qwen3.6-plus) 过滤+总结订单/基本面信息
#
# 用法:
#   bash scripts/run_xueqiu_order_monitor.sh                          # 全量 (从 CSV 读取)
#   bash scripts/run_xueqiu_order_monitor.sh --codes 600519,000858    # 指定股票
#   bash scripts/run_xueqiu_order_monitor.sh --stock-csv /path/to.csv # 自定义 CSV

set -euo pipefail
cd "$(dirname "$0")/.."

STOCK_CSV="/nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv"
OUTPUT_DIR="/nvme5/xtang/gp-workspace/gp-data/xueqiu_orders"

# 如果没有参数, 默认走全量 CSV
if [[ $# -eq 0 ]]; then
    python -m src.downloader.xueqiu_order_monitor \
        --stock-csv "$STOCK_CSV" \
        --output-dir "$OUTPUT_DIR"
else
    python -m src.downloader.xueqiu_order_monitor \
        --output-dir "$OUTPUT_DIR" \
        "$@"
fi
