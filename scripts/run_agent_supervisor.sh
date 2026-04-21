#!/bin/bash
# Agent Supervisor — 管理所有数据下载 Agent
#
# Usage:
#   ./scripts/run_agent_supervisor.sh status                     # 查看所有 Agent 状态
#   ./scripts/run_agent_supervisor.sh run                        # 运行每日增量同步
#   ./scripts/run_agent_supervisor.sh run --agent minute         # 只运行 1 分钟 Agent
#   ./scripts/run_agent_supervisor.sh run --all                  # 包含非每日 Agent
#   ./scripts/run_agent_supervisor.sh daemon                     # 守护进程模式（默认绑到 NUMA node1）

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${GP_DATA_DIR:-/nvme5/xtang/gp-workspace/gp-data}"
TOKEN="${TUSHARE_TOKEN:-${GP_TUSHARE_TOKEN:-3404e77dbe323ba4582d677ace412c0bc257f72b39f956b7bf8f975f}}"
SUPERVISOR_NUMA_NODE="${GP_SUPERVISOR_NUMA_NODE:-1}"

export PYTHONPATH="${PROJECT_DIR}/src/agents:${PROJECT_DIR}/src/downloader:${PROJECT_DIR}/src:${PYTHONPATH:-}"

CMD="${1:-status}"
shift || true

case "$CMD" in
  status)
    exec python "${PROJECT_DIR}/src/agents/supervisor.py" --data-dir "$DATA_DIR" status
    ;;
  run)
    exec python "${PROJECT_DIR}/src/agents/supervisor.py" --data-dir "$DATA_DIR" run --token "$TOKEN" "$@"
    ;;
  daemon)
    if ! command -v numactl >/dev/null 2>&1; then
      echo "numactl is required for daemon mode but was not found in PATH" >&2
      exit 1
    fi

    if ! numactl --hardware 2>/dev/null | grep -q "^node ${SUPERVISOR_NUMA_NODE} cpus:"; then
      echo "NUMA node ${SUPERVISOR_NUMA_NODE} is not available on this machine" >&2
      exit 1
    fi

    echo "[run_agent_supervisor] Launching daemon on NUMA node ${SUPERVISOR_NUMA_NODE}" >&2
    exec numactl \
      --cpunodebind="${SUPERVISOR_NUMA_NODE}" \
      --preferred="${SUPERVISOR_NUMA_NODE}" \
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
    echo "  daemon                  守护进程模式，每日 17:00 自动运行（默认 NUMA node1，可用 GP_SUPERVISOR_NUMA_NODE 覆盖）"
    echo ""
    echo "Available agents: stock_list, daily_financial, market_data, minute, derived"
    exit 1
    ;;
esac
