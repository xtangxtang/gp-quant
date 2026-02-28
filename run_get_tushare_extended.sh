#!/bin/bash

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

# 定义要下载的数据集列表
# 1. 全局数据
GLOBAL_DATASETS=("trade_cal")

# 2. 需要按股票代码遍历的数据集
# 包含：复权因子、涨跌停价格、停复牌信息、利润表、资产负债表、现金流量表、财务指标、分红送股
STOCK_DATASETS=("adj_factor" "stk_limit" "suspend_d" "income" "balancesheet" "cashflow" "fina_indicator" "dividend")

echo "======================================================"
echo "开始下载 Tushare 扩展数据..."
echo "输出目录: $OUTPUT_DIR"
echo "======================================================"

# 下载全局数据
for dataset in "${GLOBAL_DATASETS[@]}"; do
    echo ""
    echo ">>> 正在下载全局数据集: $dataset <<<"
    python src/downloader/get_tushare_extended.py --output_dir "$OUTPUT_DIR" --token "$TOKEN" --dataset "$dataset"
done

# 下载按股票遍历的数据
for dataset in "${STOCK_DATASETS[@]}"; do
    echo ""
    echo ">>> 正在下载股票数据集: $dataset <<<"
    python src/downloader/get_tushare_extended.py --output_dir "$OUTPUT_DIR" --token "$TOKEN" --threads "$THREADS" --dataset "$dataset"
done

echo ""
echo "======================================================"
echo "所有扩展数据下载完成！"
echo "======================================================"
