#!/bin/bash

OUTPUT_DIR=""
THREADS="4"
TOKEN="3404e77dbe323ba4582d677ace412c0bc257f72b39f956b7bf8f975f"
START_DATE="19900101"
END_DATE=""
SYMBOLS=""
LIST_FILE="tushare_gplist.json"

show_help() {
  echo "用法: $0 [选项]"
  echo "选项:"
  echo "  -o, --output-dir <dir>          输出目录根路径 (必填)"
  echo "  -t, --threads <N>               并发线程数 (默认: 4)"
  echo "      --token <token>             Tushare API Token (默认使用内置)"
  echo "      --start-date <YYYYMMDD>     开始日期 (默认: 19900101)"
  echo "      --end-date <YYYYMMDD>       截止日期 (默认: 今天)"
  echo "      --symbols <a,b,c>           仅下载指定 symbol 列表（测试用）"
  echo "      --list-file <filename>      股票列表文件名 (默认: tushare_gplist.json)"
  echo "  -h, --help                      显示帮助"
  echo ""
  echo "示例:"
  echo "  $0 -o /mnt/.../gp-data"
  echo "  $0 -o /mnt/.../gp-data --symbols sh600000,sz000001"
}

while [[ $# -gt 0 ]]; do
  case $1 in
    -o|--output-dir)
      OUTPUT_DIR="$2"; shift 2;;
    -t|--threads)
      THREADS="$2"; shift 2;;
    --token)
      TOKEN="$2"; shift 2;;
    --start-date)
      START_DATE="$2"; shift 2;;
    --end-date)
      END_DATE="$2"; shift 2;;
    --symbols)
      SYMBOLS="$2"; shift 2;;
    --list-file)
      LIST_FILE="$2"; shift 2;;
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

CMD="python src/downloader/get_tushare_daily_full.py --output_dir $OUTPUT_DIR --token $TOKEN --threads $THREADS --start_date $START_DATE --list_file $LIST_FILE"

if [[ -n "$END_DATE" ]]; then
  CMD="$CMD --end_date $END_DATE"
fi

if [[ -n "$SYMBOLS" ]]; then
  CMD="$CMD --symbols $SYMBOLS"
fi

echo "执行命令: $CMD"
echo "----------------------------------------"
eval $CMD
