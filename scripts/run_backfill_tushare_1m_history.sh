#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR=""
PYTHON_BIN="${PYTHON_BIN:-python}"
TOKEN="${TUSHARE_TOKEN:-3404e77dbe323ba4582d677ace412c0bc257f72b39f956b7bf8f975f}"
SYMBOLS=""
STOCK_BASIC_FILE="tushare_stock_basic.csv"
START_DATE="1990-12-19"
END_DATE=""
CHUNK_OPEN_DAYS="30"
THREADS="4"
MINUTE_API_RATE="120"
DAILY_BASIC_RATE="120"
MAX_RETRIES="4"
RETRY_FAILED_ROUNDS="1"
RETRY_SLEEP_SECONDS="15"
FQT="0"
FORCE="0"
LOG_EVERY="100"
RESET_STATE="0"
MAX_CHUNKS="0"

show_help() {
  echo "用法: $0 [选项]"
  echo "选项:"
  echo "  -o, --output-dir <dir>                gp-data 根目录 (必填)"
  echo "      --token <token>                   Tushare token"
  echo "      --symbols <a,b,c>                 仅回填指定股票"
  echo "      --stock-basic-file <file>         股票基础信息 CSV，默认 tushare_stock_basic.csv"
  echo "      --start-date <YYYYMMDD|YYYY-MM-DD> 全局起始日期，默认 1990-12-19"
  echo "      --end-date <YYYYMMDD|YYYY-MM-DD>  全局结束日期，默认最新已收盘交易日"
  echo "      --chunk-open-days <N>             每次 stk_mins 拉取的交易日块大小，默认 30"
  echo "      --threads <N>                     每个交易日块内的股票并发数，默认 4"
  echo "      --minute-api-rate <N>             历史分钟接口每 60 秒调用上限，默认 120"
  echo "      --daily-basic-rate <N>            daily_basic 每 60 秒调用上限，默认 120"
  echo "      --max-retries <N>                 单股票块最大重试次数，默认 4"
  echo "      --retry-failed-rounds <N>         每个交易日块失败任务的追加轮数，默认 1"
  echo "      --retry-sleep-seconds <S>         失败轮次之间的等待秒数，默认 15"
  echo "      --fqt <0|1|2>                     复权方式，默认 0"
  echo "      --force                           覆盖已有历史分钟 CSV"
  echo "      --log-every <N>                   每完成 N 个股票块打印一次进度，默认 100"
  echo "      --reset-state                     忽略已有 state，从头开始"
  echo "      --max-chunks <N>                  仅跑前 N 个交易日块，适合测试"
  echo "      PYTHON_BIN=/path/to/python        覆盖默认解释器"
}

while [[ $# -gt 0 ]]; do
  case $1 in
    -o|--output-dir)
      OUTPUT_DIR="$2"; shift 2;;
    --token)
      TOKEN="$2"; shift 2;;
    --symbols)
      SYMBOLS="$2"; shift 2;;
    --stock-basic-file)
      STOCK_BASIC_FILE="$2"; shift 2;;
    --start-date)
      START_DATE="$2"; shift 2;;
    --end-date)
      END_DATE="$2"; shift 2;;
    --chunk-open-days)
      CHUNK_OPEN_DAYS="$2"; shift 2;;
    --threads)
      THREADS="$2"; shift 2;;
    --minute-api-rate)
      MINUTE_API_RATE="$2"; shift 2;;
    --daily-basic-rate)
      DAILY_BASIC_RATE="$2"; shift 2;;
    --max-retries)
      MAX_RETRIES="$2"; shift 2;;
    --retry-failed-rounds)
      RETRY_FAILED_ROUNDS="$2"; shift 2;;
    --retry-sleep-seconds)
      RETRY_SLEEP_SECONDS="$2"; shift 2;;
    --fqt)
      FQT="$2"; shift 2;;
    --force)
      FORCE="1"; shift 1;;
    --log-every)
      LOG_EVERY="$2"; shift 2;;
    --reset-state)
      RESET_STATE="1"; shift 1;;
    --max-chunks)
      MAX_CHUNKS="$2"; shift 2;;
    -h|--help)
      show_help; exit 0;;
    *)
      echo "未知选项: $1"
      show_help
      exit 1;;
  esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
  echo "Error: --output-dir is required"
  show_help
  exit 1
fi

CMD=("$PYTHON_BIN" "$WORKSPACE_DIR/src/downloader/backfill_tushare_1m_history.py"
  --output_dir "$OUTPUT_DIR"
  --token "$TOKEN"
  --stock_basic_file "$STOCK_BASIC_FILE"
  --start_date "$START_DATE"
  --chunk_open_days "$CHUNK_OPEN_DAYS"
  --threads "$THREADS"
  --minute_api_rate "$MINUTE_API_RATE"
  --daily_basic_rate "$DAILY_BASIC_RATE"
  --max_retries "$MAX_RETRIES"
  --retry_failed_rounds "$RETRY_FAILED_ROUNDS"
  --retry_sleep_seconds "$RETRY_SLEEP_SECONDS"
  --fqt "$FQT"
  --log_every "$LOG_EVERY"
  --max_chunks "$MAX_CHUNKS")

if [[ -n "$SYMBOLS" ]]; then
  CMD+=(--symbols "$SYMBOLS")
fi

if [[ -n "$END_DATE" ]]; then
  CMD+=(--end_date "$END_DATE")
fi

if [[ "$FORCE" == "1" ]]; then
  CMD+=(--force)
fi

if [[ "$RESET_STATE" == "1" ]]; then
  CMD+=(--reset_state)
fi

"${CMD[@]}"