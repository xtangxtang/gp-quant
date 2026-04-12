#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR=""
THREADS="4"
RATE="180"
TOKEN="${TUSHARE_TOKEN:-3404e77dbe323ba4582d677ace412c0bc257f72b39f956b7bf8f975f}"
CATEGORY="all"
START_DATE=""
END_DATE=""

show_help() {
  echo "用法: $0 [选项]"
  echo ""
  echo "下载 Tushare 2000积分可用的全部数据"
  echo ""
  echo "选项:"
  echo "  -o, --output-dir <dir>    输出目录根路径 (必填)"
  echo "  -t, --threads <N>         并发线程数 (默认: 4)"
  echo "  -r, --rate <N>            每分钟API调用次数 (默认: 180)"
  echo "  -c, --category <cat>      数据类别 (默认: all)"
  echo "                            可选: all, stock, financial, market, index,"
  echo "                                  fund, futures, bond, macro, fx, hk, option"
  echo "      --start-date <date>   起始日期 YYYYMMDD"
  echo "      --end-date <date>     结束日期 YYYYMMDD (默认: 今天)"
  echo "      --token <token>       Tushare API Token"
  echo "  -h, --help                显示帮助"
  echo ""
  echo "示例:"
  echo "  $0 -o /nvme5/xtang/gp-workspace/gp-data"
  echo "  $0 -o /nvme5/xtang/gp-workspace/gp-data -c stock    # 只下载股票数据"
  echo "  $0 -o /nvme5/xtang/gp-workspace/gp-data -c macro    # 只下载宏观数据"
}

while [[ $# -gt 0 ]]; do
  case $1 in
    -o|--output-dir)     OUTPUT_DIR="$2"; shift 2;;
    -t|--threads)        THREADS="$2"; shift 2;;
    -r|--rate)           RATE="$2"; shift 2;;
    -c|--category)       CATEGORY="$2"; shift 2;;
    --start-date)        START_DATE="$2"; shift 2;;
    --end-date)          END_DATE="$2"; shift 2;;
    --token)             TOKEN="$2"; shift 2;;
    -h|--help)           show_help; exit 0;;
    *)                   echo "未知选项: $1"; show_help; exit 1;;
  esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
  echo "Error: --output-dir is required"
  show_help
  exit 1
fi

echo "======================================================"
echo "  Tushare 全量数据下载 (2000积分)"
echo "======================================================"
echo "输出目录:   $OUTPUT_DIR"
echo "数据类别:   $CATEGORY"
echo "并发线程:   $THREADS"
echo "API频率:    $RATE/min"
echo "开始时间:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"

CMD="python $WORKSPACE_DIR/src/downloader/download_all_tushare.py"
CMD="$CMD -o $OUTPUT_DIR"
CMD="$CMD --token $TOKEN"
CMD="$CMD --threads $THREADS"
CMD="$CMD --rate $RATE"
CMD="$CMD --category $CATEGORY"

if [[ -n "$START_DATE" ]]; then
  CMD="$CMD --start-date $START_DATE"
fi
if [[ -n "$END_DATE" ]]; then
  CMD="$CMD --end-date $END_DATE"
fi

$CMD

echo ""
echo "======================================================"
echo "  完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"
