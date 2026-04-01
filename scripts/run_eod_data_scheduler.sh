#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR=""
PYTHON_BIN="${PYTHON_BIN:-python}"
TOKEN="${TUSHARE_TOKEN:-3404e77dbe323ba4582d677ace412c0bc257f72b39f956b7bf8f975f}"
SCHEDULE_TIME="16:00"
CHECK_INTERVAL_SECONDS="30"
STATUS_LOG_INTERVAL_SECONDS="300"
FAILURE_RETRY_INTERVAL_SECONDS="1800"
BACKFILL_OPEN_DAYS="1"
FINANCIAL_THREADS="16"
BATCH_RATE="240"
FINANCIAL_RATE="360"
FINANCIAL_DATASETS="income,balancesheet,cashflow,fina_indicator"
MINUTE_SOURCE="ts"
MINUTE_THREADS="6"
MINUTE_FQT="0"
MINUTE_FORCE="0"
MINUTE_RETRY_FAILED_ROUNDS="2"
MINUTE_RETRY_SLEEP_SECONDS="2"
RUN_NOW="0"
RUN_ONCE="0"
ALLOW_NON_TRADING_DAY="0"
DRY_RUN="0"
SKIP_DATE_BASED="0"
SKIP_FINANCIALS="0"

show_help() {
  echo "用法: $0 [选项]"
  echo "选项:"
  echo "  -o, --output-dir <dir>                 gp-data 根目录 (必填)"
  echo "      PYTHON_BIN=/path/to/python         可覆盖默认解释器，默认 python"
  echo "      --token <token>                    Tushare token (默认优先读取环境变量 TUSHARE_TOKEN)"
  echo "      --schedule-time <HH:MM>            本地触发时间，默认 16:00"
  echo "      --check-interval-seconds <N>       等待时轮询间隔秒数，默认 30"
  echo "      --status-log-interval-seconds <N>  空闲状态心跳日志间隔秒数，默认 300"
  echo "      --failure-retry-interval-seconds <N>  同一交易日失败后再次尝试的冷却秒数，默认 1800"
  echo "      --backfill-open-days <N>           每日增量同步回补最近 N 个开市日，默认 1"
  echo "      --financial-threads <N>            财务增量线程数，默认 16"
  echo "      --batch-rate <N>                   Tushare 批量接口每 60 秒调用上限，默认 240"
  echo "      --financial-rate <N>               Tushare 财务接口每 60 秒调用上限，默认 360"
  echo "      --financial-datasets <csv>         财务数据集列表"
  echo "      --skip-date-based                  跳过日线/复权/停复牌等日期型数据"
  echo "      --skip-financials                  跳过财务数据"
  echo "      --minute-source <tx|em|ts>         当天 1 分钟数据主源，默认 ts"
  echo "      --minute-threads <N>               当天 1 分钟同步线程数，默认 6"
  echo "      --minute-fqt <0|1|2>               当天 1 分钟复权方式，默认 0"
  echo "      --minute-force                     强制重拉当天 1 分钟文件"
  echo "      --minute-retry-failed-rounds <N>   当天 1 分钟失败续跑轮数，默认 2"
  echo "      --minute-retry-sleep-seconds <N>   当天 1 分钟失败续跑轮次间隔秒数，默认 2"
  echo "      --run-now                          立即执行一次，不等待 16:00"
  echo "      --run-once                         执行一次后退出，或只做一次调度判断后退出"
  echo "      --allow-non-trading-day            仅配合 --run-now：非交易日时使用最近开市日"
  echo "      --dry-run                          只打印命令，不真正执行"
  echo "  -h, --help                             显示帮助"
  echo ""
  echo "示例:"
  echo "  $0 -o /nvme5/xtang/gp-workspace/gp-data"
  echo "  $0 -o /nvme5/xtang/gp-workspace/gp-data --run-now --run-once --dry-run"
  echo "  nohup $0 -o /nvme5/xtang/gp-workspace/gp-data > /tmp/gp_quant_eod_scheduler.log 2>&1 &"
}

while [[ $# -gt 0 ]]; do
  case $1 in
    -o|--output-dir)
      OUTPUT_DIR="$2"; shift 2;;
    --token)
      TOKEN="$2"; shift 2;;
    --schedule-time)
      SCHEDULE_TIME="$2"; shift 2;;
    --check-interval-seconds)
      CHECK_INTERVAL_SECONDS="$2"; shift 2;;
    --status-log-interval-seconds)
      STATUS_LOG_INTERVAL_SECONDS="$2"; shift 2;;
    --failure-retry-interval-seconds)
      FAILURE_RETRY_INTERVAL_SECONDS="$2"; shift 2;;
    --backfill-open-days)
      BACKFILL_OPEN_DAYS="$2"; shift 2;;
    --financial-threads)
      FINANCIAL_THREADS="$2"; shift 2;;
    --batch-rate)
      BATCH_RATE="$2"; shift 2;;
    --financial-rate)
      FINANCIAL_RATE="$2"; shift 2;;
    --financial-datasets)
      FINANCIAL_DATASETS="$2"; shift 2;;
    --skip-date-based)
      SKIP_DATE_BASED="1"; shift 1;;
    --skip-financials)
      SKIP_FINANCIALS="1"; shift 1;;
    --minute-source)
      MINUTE_SOURCE="$2"; shift 2;;
    --minute-threads)
      MINUTE_THREADS="$2"; shift 2;;
    --minute-fqt)
      MINUTE_FQT="$2"; shift 2;;
    --minute-force)
      MINUTE_FORCE="1"; shift 1;;
    --minute-retry-failed-rounds)
      MINUTE_RETRY_FAILED_ROUNDS="$2"; shift 2;;
    --minute-retry-sleep-seconds)
      MINUTE_RETRY_SLEEP_SECONDS="$2"; shift 2;;
    --run-now)
      RUN_NOW="1"; shift 1;;
    --run-once)
      RUN_ONCE="1"; shift 1;;
    --allow-non-trading-day)
      ALLOW_NON_TRADING_DAY="1"; shift 1;;
    --dry-run)
      DRY_RUN="1"; shift 1;;
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

CMD=("$PYTHON_BIN" "$WORKSPACE_DIR/src/downloader/eod_data_scheduler.py"
  --output-dir "$OUTPUT_DIR"
  --token "$TOKEN"
  --schedule-time "$SCHEDULE_TIME"
  --check-interval-seconds "$CHECK_INTERVAL_SECONDS"
  --status-log-interval-seconds "$STATUS_LOG_INTERVAL_SECONDS"
  --failure-retry-interval-seconds "$FAILURE_RETRY_INTERVAL_SECONDS"
  --backfill-open-days "$BACKFILL_OPEN_DAYS"
  --financial-threads "$FINANCIAL_THREADS"
  --batch-rate "$BATCH_RATE"
  --financial-rate "$FINANCIAL_RATE"
  --financial-datasets "$FINANCIAL_DATASETS"
  --minute-source "$MINUTE_SOURCE"
  --minute-threads "$MINUTE_THREADS"
  --minute-fqt "$MINUTE_FQT"
  --minute-retry-failed-rounds "$MINUTE_RETRY_FAILED_ROUNDS"
  --minute-retry-sleep-seconds "$MINUTE_RETRY_SLEEP_SECONDS")

if [[ "$SKIP_DATE_BASED" == "1" ]]; then
  CMD+=(--skip-date-based)
fi

if [[ "$SKIP_FINANCIALS" == "1" ]]; then
  CMD+=(--skip-financials)
fi

if [[ "$MINUTE_FORCE" == "1" ]]; then
  CMD+=(--minute-force)
fi

if [[ "$RUN_NOW" == "1" ]]; then
  CMD+=(--run-now)
fi

if [[ "$RUN_ONCE" == "1" ]]; then
  CMD+=(--run-once)
fi

if [[ "$ALLOW_NON_TRADING_DAY" == "1" ]]; then
  CMD+=(--allow-non-trading-day)
fi

if [[ "$DRY_RUN" == "1" ]]; then
  CMD+=(--dry-run)
fi

"${CMD[@]}"