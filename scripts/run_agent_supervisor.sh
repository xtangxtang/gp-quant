#!/bin/bash
# Agent Supervisor — 管理所有数据下载 Agent
#
# Usage:
#   ./scripts/run_agent_supervisor.sh status                     # 查看所有 Agent 状态
#   ./scripts/run_agent_supervisor.sh run                        # 运行每日增量同步
#   ./scripts/run_agent_supervisor.sh run --agent minute         # 只运行 1 分钟 Agent
#   ./scripts/run_agent_supervisor.sh run --all                  # 包含非每日 Agent
#   ./scripts/run_agent_supervisor.sh daemon                     # 守护进程模式

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${GP_DATA_DIR:-/nvme5/xtang/gp-workspace/gp-data}"
TOKEN="${TUSHARE_TOKEN:-${GP_TUSHARE_TOKEN:-3404e77dbe323ba4582d677ace412c0bc257f72b39f956b7bf8f975f}}"

export PYTHONPATH="${PROJECT_DIR}/src/agents:${PROJECT_DIR}/src/downloader:${PROJECT_DIR}/src:${PYTHONPATH:-}"

CMD="${1:-status}"
shift || true

case "$CMD" in
  status)
    python "${PROJECT_DIR}/src/agents/supervisor.py" --data-dir "$DATA_DIR" status
    ;;
  run)
    python "${PROJECT_DIR}/src/agents/supervisor.py" --data-dir "$DATA_DIR" run --token "$TOKEN" "$@"
    ;;
  daemon)
    python "${PROJECT_DIR}/src/agents/supervisor.py" --data-dir "$DATA_DIR" daemon --token "$TOKEN" "$@"
    ;;
  *)
    echo "Usage: $0 {status|run|daemon} [options]"
    echo ""
    echo "Commands:"
    echo "  status                  查看所有 Agent 状态看板"
    echo "  run                     运行每日增量同步 (stock_list → daily_financial → minute → derived)"
    echo "  run --agent <name>      只运行指定 Agent"
    echo "  run --all               包含所有 Agent (含 market_data)"
    echo "  daemon                  守护进程模式，每日 17:00 自动运行"
    echo ""
    echo "Available agents: stock_list, daily_financial, market_data, minute, derived"
    exit 1
    ;;
esac
