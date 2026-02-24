#!/bin/bash

OUTPUT_DIR=""
THREADS=""
ADJ=""
NO_IPO_FILTER=""
END_DATE=""
SYMBOLS=""

show_help() {
  echo "用法: $0 [选项]"
  echo "选项:"
  echo "  -o, --output-dir <dir>          输出目录根路径 (必填)"
  echo "  -t, --threads <N>               并发线程数 (默认: 8)"
  echo "  -a, --adj <none|qfq|hfq>         日线复权模式 (默认: none 不复权)"
  echo "      --end-date <YYYY-MM-DD>     截止日期 (默认: 今天；接口自然停在最新交易日)"
  echo "      --no-ipo-filter             不按上市日期过滤 (缺失上市日期则从 1990-01-01 开始)"
  echo "      --symbols <a,b,c>           仅下载指定 symbol 列表（测试用）"
  echo "  -h, --help                      显示帮助"
  echo ""
  echo "示例:"
  echo "  $0 -o /mnt/.../gp-data"
  echo "  $0 -o /mnt/.../gp-data -t 32 -a qfq"
  echo "  $0 -o /mnt/.../gp-data --symbols sh600000,sz000001"
}

while [[ $# -gt 0 ]]; do
  case $1 in
    -o|--output-dir)
      OUTPUT_DIR="$2"; shift 2;;
    -t|--threads)
      THREADS="$2"; shift 2;;
    -a|--adj)
      ADJ="$2"; shift 2;;
    --end-date)
      END_DATE="$2"; shift 2;;
    --no-ipo-filter)
      NO_IPO_FILTER="1"; shift 1;;
    --symbols)
      SYMBOLS="$2"; shift 2;;
    -h|--help)
      show_help; exit 0;;
    *)
      echo "未知选项: $1"; show_help; exit 1;;
  esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
  echo "Error: --output-dir is required"
  show_help
  exit 1
fi

CMD="python src/downloader/get_total_daily_trade.py --output_dir $OUTPUT_DIR"

if [[ -n "$THREADS" ]]; then
  CMD="$CMD --threads $THREADS"
fi

if [[ -n "$ADJ" ]]; then
  CMD="$CMD --adj $ADJ"
fi

if [[ -n "$END_DATE" ]]; then
  CMD="$CMD --end_date $END_DATE"
fi

if [[ -n "$NO_IPO_FILTER" ]]; then
  CMD="$CMD --no_ipo_filter"
fi

if [[ -n "$SYMBOLS" ]]; then
  CMD="$CMD --symbols $SYMBOLS"
fi

echo "执行命令: $CMD"
echo "----------------------------------------"
eval $CMD
