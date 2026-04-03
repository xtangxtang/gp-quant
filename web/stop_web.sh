#!/usr/bin/env bash
set -euo pipefail

# Stop the strategy console started by web/start_web.sh.
# Usage examples:
#   ./web/stop_web.sh
#   ./web/stop_web.sh --port 8010
#
# Defaults:
#   --port 5050
#   --grace-seconds 5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PORT="5050"
GRACE_SECONDS="5"

show_help() {
  cat <<EOF
Usage: ./web/stop_web.sh [options]

Options:
  --port <port>             Port used by web/app.py, default 5050
  --grace-seconds <secs>    Seconds to wait after SIGTERM before SIGKILL, default 5
  -h, --help                Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="$2"
      shift 2
      ;;
    --grace-seconds)
      GRACE_SECONDS="$2"
      shift 2
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      show_help >&2
      exit 1
      ;;
  esac
done

if ! [[ "${PORT}" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  echo "Invalid --port: ${PORT}" >&2
  exit 1
fi

if ! [[ "${GRACE_SECONDS}" =~ ^[0-9]+$ ]]; then
  echo "Invalid --grace-seconds: ${GRACE_SECONDS}" >&2
  exit 1
fi

read_cmdline() {
  local pid="$1"
  if [[ -r "/proc/${pid}/cmdline" ]]; then
    tr '\0' ' ' < "/proc/${pid}/cmdline"
  fi
}

read_cwd() {
  local pid="$1"
  if [[ -e "/proc/${pid}/cwd" ]]; then
    readlink -f "/proc/${pid}/cwd" 2>/dev/null || true
  fi
}

is_repo_web_pid() {
  local pid="$1"
  local cmdline="$(read_cmdline "${pid}")"
  local cwd="$(read_cwd "${pid}")"

  [[ -n "${cmdline}" ]] || return 1
  [[ "${cmdline}" == *"web/app.py"* ]] || return 1
  [[ "${cwd}" == "${REPO_ROOT}" ]] || return 1
}

find_port_pids() {
  if command -v lsof >/dev/null 2>&1; then
    lsof -ti TCP:"${PORT}" -sTCP:LISTEN 2>/dev/null || true
    return
  fi

  if command -v fuser >/dev/null 2>&1; then
    fuser -n tcp "${PORT}" 2>/dev/null | tr ' ' '\n' || true
    return
  fi

  if command -v ss >/dev/null 2>&1; then
    ss -ltnp "sport = :${PORT}" 2>/dev/null | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' || true
  fi
}

find_repo_web_pids_by_match() {
  if ! command -v pgrep >/dev/null 2>&1; then
    return
  fi

  while IFS= read -r pid; do
    [[ -n "${pid}" ]] || continue
    if is_repo_web_pid "${pid}"; then
      local cmdline="$(read_cmdline "${pid}")"
      if [[ "${cmdline}" == *"--port ${PORT}"* ]] || [[ "${cmdline}" == *"--port=${PORT}"* ]]; then
        echo "${pid}"
        continue
      fi
      if [[ "${PORT}" == "5050" ]] && [[ "${cmdline}" != *"--port "* ]] && [[ "${cmdline}" != *"--port="* ]]; then
        echo "${pid}"
      fi
    fi
  done < <(pgrep -f "web/app.py" 2>/dev/null || true)
}

collect_target_pids() {
  {
    find_port_pids
    find_repo_web_pids_by_match
  } | awk 'NF && !seen[$0]++'
}

mapfile -t CANDIDATE_PIDS < <(collect_target_pids)

TARGET_PIDS=()
for pid in "${CANDIDATE_PIDS[@]}"; do
  if is_repo_web_pid "${pid}"; then
    TARGET_PIDS+=("${pid}")
  fi
done

if [[ ${#TARGET_PIDS[@]} -eq 0 ]]; then
  echo "No matching web/app.py process found for port ${PORT}."
  exit 0
fi

echo "Stopping web/app.py on port ${PORT}: ${TARGET_PIDS[*]}"
kill "${TARGET_PIDS[@]}" 2>/dev/null || true

deadline=$((SECONDS + GRACE_SECONDS))
while (( SECONDS < deadline )); do
  REMAINING_PIDS=()
  for pid in "${TARGET_PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      REMAINING_PIDS+=("${pid}")
    fi
  done
  if [[ ${#REMAINING_PIDS[@]} -eq 0 ]]; then
    echo "Web process stopped."
    exit 0
  fi
  sleep 1
done

REMAINING_PIDS=()
for pid in "${TARGET_PIDS[@]}"; do
  if kill -0 "${pid}" 2>/dev/null; then
    REMAINING_PIDS+=("${pid}")
  fi
done

if [[ ${#REMAINING_PIDS[@]} -eq 0 ]]; then
  echo "Web process stopped."
  exit 0
fi

echo "Processes still alive after ${GRACE_SECONDS}s, sending SIGKILL: ${REMAINING_PIDS[*]}"
kill -9 "${REMAINING_PIDS[@]}" 2>/dev/null || true
echo "Web process stopped."