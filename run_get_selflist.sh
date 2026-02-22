#!/bin/bash

# 默认参数
START_DATE=""
END_DATE=""
OUTPUT_DIR=""

# 帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo "选项:"
    echo "  -s, --start-date <YYYY-MM-DD>  指定开始日期"
    echo "  -e, --end-date <YYYY-MM-DD>    指定结束日期"
    echo "  -o, --output-dir <dir>         指定数据保存目录 (默认: ./gp_daily)"
    echo "  -h, --help                     显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                                         # 下载今天的数据"
    echo "  $0 -s 2023-08-01                           # 从 2023-08-01 下载到今天"
    echo "  $0 -s 2023-08-01 -e 2023-08-05             # 下载指定日期范围内的数据"
    echo "  $0 -s 2023-08-01 -o /tmp/my_stock_data     # 指定输出目录"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--start-date)
            START_DATE="$2"
            shift 2
            ;;
        -e|--end-date)
            END_DATE="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 构建 Python 命令
if command -v conda >/dev/null 2>&1; then
    CMD="conda run -n xtang-gp python get_selflist_daily.py"
else
    CMD="python get_selflist_daily.py"
fi

if [ -n "$START_DATE" ]; then
    CMD="$CMD --start_date $START_DATE"
fi

if [ -n "$END_DATE" ]; then
    CMD="$CMD --end_date $END_DATE"
fi

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output_dir $OUTPUT_DIR"
fi

# 打印并执行命令
echo "执行命令: $CMD"
echo "----------------------------------------"
eval $CMD
