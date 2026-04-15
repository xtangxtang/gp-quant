#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════
# 雪球 Wiki 流水线  (抓一只 → 立即生成 wiki → 下一只)
#
# 用法:
#   # 历史全量 (所有 A 股, 断点续传)
#   bash scripts/run_xueqiu_wiki_pipeline.sh
#
#   # 每日增量 (cron 任务)
#   bash scripts/run_xueqiu_wiki_pipeline.sh --daily
#
#   # 指定几只股票
#   bash scripts/run_xueqiu_wiki_pipeline.sh --codes 600519,000858
#
#   # 限制数量 (测试)
#   bash scripts/run_xueqiu_wiki_pipeline.sh --limit 5
# ════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Defaults
STOCK_CSV="/nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv"
OUTPUT_DIR="/nvme5/xtang/gp-workspace/gp-data/xueqiu_orders"
LOG_DIR="$OUTPUT_DIR/_logs"
DATE_TAG=$(date +%Y%m%d_%H%M%S)
EXTRA_ARGS=()

# Parse script-level args, pass rest to python
DAILY=""
CODES=""
LIMIT=""
for arg in "$@"; do
    case "$arg" in
        --daily)   DAILY="--daily" ;;
        --codes=*) CODES="${arg#--codes=}" ;;
        --limit=*) LIMIT="${arg#--limit=}" ;;
        *)         EXTRA_ARGS+=("$arg") ;;
    esac
done

mkdir -p "$LOG_DIR"

# Build python command
CMD=(python -m src.downloader.xueqiu_batch_history)

if [[ -n "$CODES" ]]; then
    CMD+=(--codes "$CODES")
else
    CMD+=(--stock-csv "$STOCK_CSV")
fi

CMD+=(--output-dir "$OUTPUT_DIR" --resume)

[[ -n "$DAILY" ]] && CMD+=($DAILY)
[[ -n "$LIMIT" ]] && CMD+=(--limit "$LIMIT")
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && CMD+=("${EXTRA_ARGS[@]}")

LOG_FILE="$LOG_DIR/pipeline_${DATE_TAG}.log"

echo "═══════════════════════════════════════════"
echo "  雪球 Wiki 流水线  —  抓一只立即生成 wiki"
echo "═══════════════════════════════════════════"
echo "  模式: ${DAILY:-历史全量}"
echo "  日志: $LOG_FILE"
echo "  命令: ${CMD[*]}"
echo "═══════════════════════════════════════════"

"${CMD[@]}" 2>&1 | tee "$LOG_FILE"
