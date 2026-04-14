#!/usr/bin/env bash
# 股票 Wiki 知识库生成（LLM 模式）
# 将雪球采集的每日 JSON 整理为 per-stock Wiki
#
# 用法:
#   bash scripts/run_wiki_ingest.sh --symbol SH600519                   # 最新数据
#   bash scripts/run_wiki_ingest.sh --symbol SH600519 --date 20260413   # 指定日期
#
# 推荐: 使用 @wiki-ingest agent 直接生成（更快、质量更高）

set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR="/nvme5/xtang/gp-workspace/gp-data/xueqiu_orders"
SYMBOL=""
DATE=""
EXTRA_ARGS=()

show_help() {
    echo "用法: $0 --symbol <SYMBOL> [--date YYYYMMDD] [--data-dir DIR]"
    echo ""
    echo "参数:"
    echo "  --symbol      股票代码 (e.g. SH600519)         [必填]"
    echo "  --date        指定日期 (e.g. 20260413)         [默认: 最新]"
    echo "  --data-dir    数据根目录                        [默认: $DATA_DIR]"
    echo ""
    echo "提示: 推荐使用 @wiki-ingest agent 直接生成（不走 LLM，秒级完成）"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) show_help ;;
        --symbol) SYMBOL="$2"; shift 2 ;;
        --date) DATE="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$SYMBOL" ]]; then
    echo "错误: 必须指定 --symbol"
    show_help
fi

CMD=(python -m src.wiki.stock_wiki_ingest --symbol "$SYMBOL" --data-dir "$DATA_DIR")

if [[ -n "$DATE" ]]; then
    CMD+=(--date "$DATE")
fi

CMD+=("${EXTRA_ARGS[@]}")

echo "=== Stock Wiki Ingest ==="
echo "Symbol: $SYMBOL"
echo "Data:   $DATA_DIR"
[[ -n "$DATE" ]] && echo "Date:   $DATE"
echo "========================="
echo ""

"${CMD[@]}"

# 生成完毕后，提取 raw summary
echo ""
echo "=== 提取 raw summary ==="

if [[ -n "$DATE" ]]; then
    JSON_FILE="$DATA_DIR/$SYMBOL/$DATE.json"
    RAW_JSON="$DATA_DIR/$SYMBOL/raw/$DATE.json"
else
    # 找最新的 JSON
    JSON_FILE=$(ls -t "$DATA_DIR/$SYMBOL"/*.json 2>/dev/null | head -1)
    RAW_JSON=$(ls -t "$DATA_DIR/$SYMBOL/raw"/*.json 2>/dev/null | head -1)
fi

# 优先取 raw 下的，其次取根目录的
TARGET_JSON="${RAW_JSON:-$JSON_FILE}"

if [[ -n "$TARGET_JSON" && -f "$TARGET_JSON" ]]; then
    BASENAME=$(basename "$TARGET_JSON" .json)
    SUMMARY_MD="$DATA_DIR/$SYMBOL/raw/${BASENAME}_summary.md"
    python3 -c "
import json, sys
with open('$TARGET_JSON', encoding='utf-8') as f:
    data = json.load(f)
summary = data.get('summary', '')
if not summary:
    print('No summary in JSON, skipping')
    sys.exit(0)
symbol = data.get('symbol', '$SYMBOL')
date = data.get('date', '$BASENAME')
with open('$SUMMARY_MD', 'w', encoding='utf-8') as f:
    f.write(summary)
print(f'✓ {\"$SUMMARY_MD\"}')
"
else
    echo "No JSON found, skipping summary extraction"
fi
