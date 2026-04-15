#!/usr/bin/env bash
# 雪球全量 A 股帖子采集 — 历史 + 每日增量
#
# 用法:
#   # 历史全量爬取 (crawl-only, 断点续传)
#   bash scripts/run_xueqiu_batch_history.sh
#
#   # 历史 + LLM 处理
#   bash scripts/run_xueqiu_batch_history.sh --full
#
#   # 每日增量 (cron 调度, crawl-only)
#   bash scripts/run_xueqiu_batch_history.sh --daily
#
#   # 每日增量 + LLM
#   bash scripts/run_xueqiu_batch_history.sh --daily --full
#
#   # 限制数量 (测试)
#   bash scripts/run_xueqiu_batch_history.sh --limit 10
#
#   # 单只测试
#   bash scripts/run_xueqiu_batch_history.sh --codes 600519

set -euo pipefail
cd "$(dirname "$0")/.."

STOCK_CSV="/nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv"
OUTPUT_DIR="/nvme5/xtang/gp-workspace/gp-data/xueqiu_orders"
LOG_DIR="$OUTPUT_DIR/_logs"
mkdir -p "$LOG_DIR"

MODE="history"
CRAWL_ONLY="--crawl-only"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --daily)
            MODE="daily"
            shift
            ;;
        --full)
            CRAWL_ONLY=""
            shift
            ;;
        --codes)
            EXTRA_ARGS+=(--codes "$2")
            shift 2
            ;;
        --limit)
            EXTRA_ARGS+=(--limit "$2")
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# 如果没指定 --codes, 使用全量 CSV
if ! printf '%s\n' "${EXTRA_ARGS[@]}" 2>/dev/null | grep -q -- "--codes"; then
    EXTRA_ARGS+=(--stock-csv "$STOCK_CSV")
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

if [[ "$MODE" == "daily" ]]; then
    LOG_FILE="$LOG_DIR/daily_${TIMESTAMP}.log"
    echo "[$(date)] 每日增量模式" | tee "$LOG_FILE"
    python -m src.downloader.xueqiu_batch_history \
        --daily \
        --resume \
        --output-dir "$OUTPUT_DIR" \
        $CRAWL_ONLY \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee -a "$LOG_FILE"
else
    LOG_FILE="$LOG_DIR/history_${TIMESTAMP}.log"
    echo "[$(date)] 历史全量模式" | tee "$LOG_FILE"
    python -m src.downloader.xueqiu_batch_history \
        --resume \
        --output-dir "$OUTPUT_DIR" \
        $CRAWL_ONLY \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee -a "$LOG_FILE"
fi

echo "[$(date)] 完成" | tee -a "$LOG_FILE"
