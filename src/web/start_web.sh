#!/usr/bin/env bash
set -euo pipefail

# Start the resonance scan dashboard.
# Usage examples:
#   ./src/web/start_web.sh
#   ./src/web/start_web.sh --port 8010
#   ./src/web/start_web.sh --scan-output-dir /nvme5/xtang/gp-workspace/gp-quant/results/multitimeframe_resonance/live_market_scan
#
# Defaults (see src/web/app.py):
#   --host 0.0.0.0
#   --port 5050

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

# If GP_WEB_PY is set, use it as the python runner (e.g. 'conda run -n xtang-gp python').
# Otherwise, use whatever 'python' is on PATH.
PY_RUNNER="${GP_WEB_PY:-python}"

# Forward all args to the app.
exec ${PY_RUNNER} src/web/app.py "$@"
