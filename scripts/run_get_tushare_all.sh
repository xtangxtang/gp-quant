#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR=""
THREADS="4"
TOKEN="3404e77dbe323ba4582d677ace412c0bc257f72b39f956b7bf8f975f"

show_help() {
  echo "用法: $0 [选项]"
  echo "选项:"
  echo "  -o, --output-dir <dir>          输出目录根路径 (必填)"
  echo "  -t, --threads <N>               并发线程数 (默认: 4)"
  echo "      --token <token>             Tushare API Token (默认使用内置)"
  echo "  -h, --help                      显示帮助"
  echo ""
  echo "示例:"
  echo "  $0 -o /mnt/.../gp-data"
}

while [[ $# -gt 0 ]]; do
  case $1 in
    -o|--output-dir)
      OUTPUT_DIR="$2"; shift 2;;
    -t|--threads)
      THREADS="$2"; shift 2;;
    --token)
      TOKEN="$2"; shift 2;;
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

echo "======================================================"
echo "1. 获取当前所有可交易股票列表..."
echo "======================================================"
python "$WORKSPACE_DIR/src/downloader/get_tushare_stock_list.py" --output_dir "$OUTPUT_DIR" --token "$TOKEN"

if [ $? -ne 0 ]; then
    echo "获取股票列表失败，退出。"
    exit 1
fi

echo ""
echo "======================================================"
echo "2. 开始下载所有股票的历史交易记录..."
echo "======================================================"
"$SCRIPT_DIR/run_get_tushare_daily_full.sh" -o "$OUTPUT_DIR" -t "$THREADS" --token "$TOKEN" --list-file "tushare_gplist.json"

if [ $? -ne 0 ]; then
  echo "日线历史下载失败，退出。"
  exit 1
fi

echo ""
echo "======================================================"
echo "3. 开始下载/更新扩展数据(复权因子/财务等)..."
echo "======================================================"
"$SCRIPT_DIR/run_get_tushare_extended.sh" -o "$OUTPUT_DIR" -t "$THREADS" --token "$TOKEN"
