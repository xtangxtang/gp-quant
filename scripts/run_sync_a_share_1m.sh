#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR=""
THREADS="4"
SYMBOLS=""
LIST_FILE="tushare_gplist.json"
START_DATE=""
END_DATE=""
RECENT_OPEN_DAYS="3"
FQT="0"
SOURCE="ts"
TOKEN=""
FORCE="0"
RETRY_FAILED_ROUNDS="2"
RETRY_SLEEP_SECONDS="8"
RETRY_FAILED_ONLY="0"
NO_RESUME_FAILURES="0"

show_help() {
  echo "用法: $0 [选项]"
  echo "选项:"
  echo "  -o, --output-dir <dir>          输出目录根路径 (必填)"
  echo "  -t, --threads <N>               并发线程数 (默认: 4)"
  echo "      --symbols <a,b,c>           仅同步指定股票"
  echo "      --list-file <filename>      股票列表文件名 (默认: tushare_gplist.json)"
  echo "      --start-date <YYYYMMDD>     起始日期"
  echo "      --end-date <YYYYMMDD>       结束日期"
  echo "      --recent-open-days <N>      未指定日期时，同步最近 N 个交易日 (默认: 3)"
  echo "      --fqt <0|1|2>               复权方式，默认 0=不复权"
  echo "      --source <em|tx|ts>         主分钟源，默认 em"
  echo "      --token <token>             Tushare token，仅在 --source ts 时需要"
  echo "      --force                     即使文件已完整也强制重拉"
  echo "      --retry-failed-rounds <N>   初始跑完后，对失败任务追加重试轮数 (默认: 2)"
  echo "      --retry-sleep-seconds <S>   每轮失败续跑之间的等待秒数 (默认: 8)"
  echo "      --retry-failed-only         只处理 failed_tasks.json 里已有失败任务"
  echo "      --no-resume-failures        初始任务忽略历史失败队列"
  echo "  -h, --help                      显示帮助"
  echo ""
  echo "说明:"
  echo "  免费 Eastmoney/Tencent 分钟接口通常只能覆盖最近几个交易日。"
  echo ""
  echo "示例:"
  echo "  $0 -o /nvme5/xtang/gp-workspace/gp-data"
  echo "  $0 -o /nvme5/xtang/gp-workspace/gp-data --symbols sh600000,sz000001 --recent-open-days 1"
  echo "  $0 -o /nvme5/xtang/gp-workspace/gp-data --source ts --token <your_tushare_token> --start-date 20260327"
}

while [[ $# -gt 0 ]]; do
  case $1 in
    -o|--output-dir)
      OUTPUT_DIR="$2"; shift 2;;
    -t|--threads)
      THREADS="$2"; shift 2;;
    --symbols)
      SYMBOLS="$2"; shift 2;;
    --list-file)
      LIST_FILE="$2"; shift 2;;
    --start-date)
      START_DATE="$2"; shift 2;;
    --end-date)
      END_DATE="$2"; shift 2;;
    --recent-open-days)
      RECENT_OPEN_DAYS="$2"; shift 2;;
    --fqt)
      FQT="$2"; shift 2;;
    --source)
      SOURCE="$2"; shift 2;;
    --token)
      TOKEN="$2"; shift 2;;
    --force)
      FORCE="1"; shift 1;;
    --retry-failed-rounds)
      RETRY_FAILED_ROUNDS="$2"; shift 2;;
    --retry-sleep-seconds)
      RETRY_SLEEP_SECONDS="$2"; shift 2;;
    --retry-failed-only)
      RETRY_FAILED_ONLY="1"; shift 1;;
    --no-resume-failures)
      NO_RESUME_FAILURES="1"; shift 1;;
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

CMD=(python "$WORKSPACE_DIR/src/downloader/sync_a_share_1m.py"
  --output_dir "$OUTPUT_DIR"
  --threads "$THREADS"
  --list_file "$LIST_FILE"
  --recent_open_days "$RECENT_OPEN_DAYS"
  --fqt "$FQT"
  --source "$SOURCE"
  --retry_failed_rounds "$RETRY_FAILED_ROUNDS"
  --retry_sleep_seconds "$RETRY_SLEEP_SECONDS")

if [[ -n "$SYMBOLS" ]]; then
  CMD+=(--symbols "$SYMBOLS")
fi

if [[ -n "$START_DATE" ]]; then
  CMD+=(--start_date "$START_DATE")
fi

if [[ -n "$END_DATE" ]]; then
  CMD+=(--end_date "$END_DATE")
fi

if [[ -n "$TOKEN" ]]; then
  CMD+=(--token "$TOKEN")
fi

if [[ "$FORCE" == "1" ]]; then
  CMD+=(--force)
fi

if [[ "$RETRY_FAILED_ONLY" == "1" ]]; then
  CMD+=(--retry_failed_only)
fi

if [[ "$NO_RESUME_FAILURES" == "1" ]]; then
  CMD+=(--no_resume_failures)
fi

"${CMD[@]}"