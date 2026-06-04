#!/usr/bin/env bash
set -euo pipefail

SESSION="${1:-momentnet_all}"
if [[ $# -gt 0 ]]; then
  shift
fi

RUNNER="experiments/run.py"
if [[ $# -gt 0 && "$1" == *.py ]]; then
  RUNNER="$1"
  shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session '${SESSION}' already exists"
  exit 1
fi

tmux new-session -d -s "${SESSION}" \
  "cd '${REPO_ROOT}' && python '${RUNNER}' $* 2>&1 | tee '${LOG_DIR}/${SESSION}.log'"

echo "Started tmux session: ${SESSION}"
echo "Log: ${LOG_DIR}/${SESSION}.log"
