#!/usr/bin/env bash
set -euo pipefail

# Start the strategy console.
# Usage examples:
#   ./web/start_web.sh
#   ./web/start_web.sh --port 8010
#
# Defaults (see web/app.py):
#   --host 0.0.0.0
#   --port 5050

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

# If GP_WEB_PY is set, use it as the python runner (e.g. 'conda run -n xtang-gp python').
# Otherwise, prefer venv python, then fall back to 'python' on PATH.
if [[ -n "${GP_WEB_PY:-}" ]]; then
  PY_RUNNER="$GP_WEB_PY"
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PY_RUNNER="${REPO_ROOT}/.venv/bin/python"
else
  PY_RUNNER="python"
fi

# Forward all args to the app.
exec ${PY_RUNNER} web/app.py "$@"
