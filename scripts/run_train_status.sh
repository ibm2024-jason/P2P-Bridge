#!/usr/bin/env bash
set -euo pipefail

SAVE_ROOT="${1:-experiments/rodr_compare_starter}"
RODR_WEIGHT="${RODR_WEIGHT:-1.0}"

for name in baseline_mse "rodr_w${RODR_WEIGHT}"; do
  pid_file="${SAVE_ROOT}/${name}.pid"
  log_file="${SAVE_ROOT}/logs/${name}.log"
  if [ -f "${pid_file}" ]; then
    pid="$(cat "${pid_file}")"
    if kill -0 "${pid}" >/dev/null 2>&1; then
      echo "${name}: running pid=${pid}"
    else
      echo "${name}: stopped pid=${pid}"
    fi
  else
    echo "${name}: no pid file"
  fi
  if [ -f "${log_file}" ]; then
    echo "--- tail ${log_file} ---"
    tail -n 8 "${log_file}"
  fi
done
