#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-punet}"
SAVE_ROOT="${2:-experiments/rodr_compare_punet_full}"

WEIGHTS=(${WEIGHTS:-0.25 0.5 0.8 1.2})
GPUS=(${GPUS:-0 1 2 3})
STEPS="${STEPS:-300000}"
BS="${BS:-8}"
WORKERS="${WORKERS:-2}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
WANDB_MODE="${WANDB_MODE:-disabled}"

if [ "${#WEIGHTS[@]}" -ne "${#GPUS[@]}" ]; then
  echo "WEIGHTS and GPUS must have the same length." >&2
  echo "WEIGHTS=${WEIGHTS[*]}" >&2
  echo "GPUS=${GPUS[*]}" >&2
  exit 2
fi

case "${DATASET}" in
  starter|StarterLocal)
    RODR_CONFIG="configs/PVDS_StarterLocal_rodr.yaml"
    ;;
  punet|PUNet)
    RODR_CONFIG="configs/PVDS_PUNet_rodr.yaml"
    ;;
  *)
    echo "Unknown DATASET=${DATASET}. Use starter or punet." >&2
    exit 2
    ;;
esac

mkdir -p "${SAVE_ROOT}/logs"

echo "Starting RODR weight sweep"
echo "  dataset: ${DATASET}"
echo "  config: ${RODR_CONFIG}"
echo "  save root: ${SAVE_ROOT}"
echo "  weights: ${WEIGHTS[*]}"
echo "  gpus: ${GPUS[*]}"
echo "  steps: ${STEPS}"
echo "  batch size: ${BS}"

for idx in "${!WEIGHTS[@]}"; do
  weight="${WEIGHTS[$idx]}"
  gpu="${GPUS[$idx]}"
  name="rodr_w${weight}"
  log_file="${SAVE_ROOT}/logs/${name}.log"
  pid_file="${SAVE_ROOT}/${name}.pid"

  if [ -f "${pid_file}" ]; then
    old_pid="$(cat "${pid_file}")"
    if kill -0 "${old_pid}" >/dev/null 2>&1; then
      echo "Refusing to start ${name}: existing process is running pid=${old_pid}" >&2
      continue
    fi
  fi

  if [ -d "${SAVE_ROOT}/${name}" ]; then
    echo "Warning: ${SAVE_ROOT}/${name} already exists; training may resume/append files in this directory." >&2
  fi

  echo "Starting ${name} on GPU ${gpu}. Log: ${log_file}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export WANDB_MODE
    python train.py \
      --config "${RODR_CONFIG}" \
      --save_dir "${SAVE_ROOT}" \
      --name "${name}" \
      --distribution_type single \
      --training.steps "${STEPS}" \
      --training.save_interval "${SAVE_INTERVAL}" \
      --training.bs "${BS}" \
      --data.workers "${WORKERS}" \
      --diffusion.rodr_tangent_weight "${weight}"
  ) >"${log_file}" 2>&1 &

  echo $! > "${pid_file}"
  echo "${name} PID: $(cat "${pid_file}")"
done

echo "Check status with:"
echo "  ./scripts/run_rodr_weight_sweep_status.sh ${SAVE_ROOT}"
