#!/bin/bash
set -euo pipefail

OUTPUT_DIR=""
THREADS=""
MINUTE_ADJ="qfq"
DAILY_ADJ="qfq"
NO_IPO_FILTER=""
TARGET_DATE=""

show_help() {
  echo "用法: $0 [选项]"
  echo "选项:"
  echo "  -o, --output-dir <dir>          输出目录根路径 (默认: ../gp-data 或 ./gp_daily)"
  echo "  -t, --threads <N>               并发线程数 (默认: 8)"
  echo "      --date <YYYY-MM-DD>         指定日期 (默认: 今天)"
  echo "      --minute-adj <qfq|none|hfq> 分钟线复权 (默认: qfq)"
  echo "      --daily-adj <none|qfq|hfq>  日线复权 (默认: qfq)"
  echo "      --no-ipo-filter             不按上市日期过滤 (全市场/自选股/日线均生效)"
  echo "  -h, --help                      显示帮助"
  echo ""
  echo "说明:"
  echo "  该脚本会依次下载："
  echo "    1) 全市场分钟数据（当天） -> <output_dir>/trade/<symbol>/<date>.csv"
  echo "    2) 自选股分钟数据（当天） -> <output_dir>/trade/<symbol>/<date>.csv"
  echo "    3) 最近交易日交易总结（全市场列表）-> <output_dir>/total-daily-view/YYYY-MM-DD.csv"
  echo "    4) 财务+估值（最新交易日缓存）-> <output_dir>/total-fundamentals/YYYY-MM-DD.csv"
  echo "    5) 全市场日线历史（断点续跑，补齐缺失段）-> <output_dir>/total-daily-trade/<symbol>.csv"
  echo ""
  echo "示例:"
  echo "  $0 -o ../gp-data"
  echo "  $0 -o ../gp-data -t 32 --minute-adj qfq --daily-adj qfq"
}

while [[ $# -gt 0 ]]; do
  case $1 in
    -o|--output-dir)
      OUTPUT_DIR="$2"; shift 2;;
    -t|--threads)
      THREADS="$2"; shift 2;;
    --date)
      TARGET_DATE="$2"; shift 2;;
    --minute-adj)
      MINUTE_ADJ="$2"; shift 2;;
    --daily-adj)
      DAILY_ADJ="$2"; shift 2;;
    --no-ipo-filter)
      NO_IPO_FILTER="1"; shift 1;;
    -h|--help)
      show_help; exit 0;;
    *)
      echo "未知选项: $1"; show_help; exit 1;;
  esac
done

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

if [[ -z "$OUTPUT_DIR" ]]; then
  if [[ -d "$SCRIPT_DIR/../gp-data" ]]; then
    OUTPUT_DIR="$SCRIPT_DIR/../gp-data"
  else
    OUTPUT_DIR="$SCRIPT_DIR/gp_daily"
  fi
fi

if [[ -z "$THREADS" ]]; then
  THREADS="8"
fi

if [[ -z "$TARGET_DATE" ]]; then
  TARGET_DATE=$(date +%F)
fi

COMMON_MINUTE_ARGS=("--start_date" "$TARGET_DATE" "--end_date" "$TARGET_DATE" "--output_dir" "$OUTPUT_DIR" "--threads" "$THREADS" "--adj" "$MINUTE_ADJ")
COMMON_DAILY_ARGS=("--output_dir" "$OUTPUT_DIR" "--threads" "$THREADS" "--adj" "$DAILY_ADJ")

if [[ -n "$NO_IPO_FILTER" ]]; then
  COMMON_MINUTE_ARGS+=("--no_ipo_filter")
  # daily trade has same flag
  COMMON_DAILY_TRADE_EXTRA=("--no_ipo_filter")
else
  COMMON_DAILY_TRADE_EXTRA=()
fi

echo "Output dir: $OUTPUT_DIR"
echo "Target date: $TARGET_DATE"
echo "Threads: $THREADS"
echo "Minute adj: $MINUTE_ADJ"
echo "Daily adj: $DAILY_ADJ"
echo "----------------------------------------"

set -x

# 1) Total minute data for the day (will auto-create/refresh total_gplist.json)
python src/downloader/get_total_daily.py "${COMMON_MINUTE_ARGS[@]}"

# 2) Selflist minute data for the day (will auto-create self_gplist.json if missing)
python src/downloader/get_selflist_daily.py "${COMMON_MINUTE_ARGS[@]}"

# 3) Latest trading-day daily summaries cached as CSV
python src/downloader/get_total_daily_view.py "${COMMON_DAILY_ARGS[@]}" --list total

# 4) Fundamentals: valuation snapshot + key finance indicators cached as CSV
python src/downloader/get_total_fundamentals.py --output_dir "$OUTPUT_DIR" --threads "$THREADS" --list total

# 5) Total daily kline history (scan continuity and fill gaps, then extend to latest trading day)
python src/downloader/get_total_daily_trade.py --output_dir "$OUTPUT_DIR" --threads "$THREADS" --adj "$DAILY_ADJ" "${COMMON_DAILY_TRADE_EXTRA[@]}"

set +x

echo "Done."
