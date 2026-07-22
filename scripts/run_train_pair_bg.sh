#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-starter}"
GPU_BASE="${2:-0}"
GPU_RODR="${3:-1}"
SAVE_ROOT="${4:-experiments/rodr_compare_${DATASET}}"

STEPS="${STEPS:-2000}"
BS="${BS:-8}"
WORKERS="${WORKERS:-2}"
RODR_WEIGHT="${RODR_WEIGHT:-1.0}"
WANDB_MODE="${WANDB_MODE:-disabled}"
SAVE_INTERVAL="${SAVE_INTERVAL:-${STEPS}}"
TRAIN_BASELINE="${TRAIN_BASELINE:-1}"
RODR_NAME="${RODR_NAME:-rodr_w${RODR_WEIGHT}}"

case "${DATASET}" in
  starter|StarterLocal)
    BASE_CONFIG="configs/PVDS_StarterLocal_base.yaml"
    RODR_CONFIG="configs/PVDS_StarterLocal_rodr.yaml"
    ;;
  punet|PUNet)
    BASE_CONFIG="configs/PVDS_PUNet.yaml"
    RODR_CONFIG="configs/PVDS_PUNet_rodr.yaml"
    ;;
  *)
    echo "Unknown DATASET=${DATASET}. Use starter or punet." >&2
    exit 2
    ;;
esac

mkdir -p "${SAVE_ROOT}/logs"

BASE_LOG="${SAVE_ROOT}/logs/baseline_mse.log"
RODR_LOG="${SAVE_ROOT}/logs/${RODR_NAME}.log"
BASE_PID="${SAVE_ROOT}/baseline_mse.pid"
RODR_PID="${SAVE_ROOT}/${RODR_NAME}.pid"

if [ "${TRAIN_BASELINE}" = "1" ]; then
  echo "Starting baseline on GPU ${GPU_BASE}. Log: ${BASE_LOG}"
  (
    export CUDA_VISIBLE_DEVICES="${GPU_BASE}"
    export WANDB_MODE
    python train.py \
      --config "${BASE_CONFIG}" \
      --save_dir "${SAVE_ROOT}" \
      --name baseline_mse \
      --distribution_type single \
      --training.steps "${STEPS}" \
      --training.save_interval "${SAVE_INTERVAL}" \
      --training.bs "${BS}" \
      --data.workers "${WORKERS}"
  ) >"${BASE_LOG}" 2>&1 &
  echo $! > "${BASE_PID}"
else
  echo "Skipping baseline because TRAIN_BASELINE=${TRAIN_BASELINE}."
fi

echo "Starting RODR on GPU ${GPU_RODR}. Log: ${RODR_LOG}"
(
  export CUDA_VISIBLE_DEVICES="${GPU_RODR}"
  export WANDB_MODE
  python train.py \
    --config "${RODR_CONFIG}" \
    --save_dir "${SAVE_ROOT}" \
    --name "${RODR_NAME}" \
    --distribution_type single \
    --training.steps "${STEPS}" \
    --training.save_interval "${SAVE_INTERVAL}" \
    --training.bs "${BS}" \
    --data.workers "${WORKERS}" \
    --diffusion.rodr_tangent_weight "${RODR_WEIGHT}"
) >"${RODR_LOG}" 2>&1 &
echo $! > "${RODR_PID}"

if [ "${TRAIN_BASELINE}" = "1" ]; then
  echo "Baseline PID: $(cat "${BASE_PID}")"
fi
echo "RODR PID: $(cat "${RODR_PID}")"
echo "Check logs with:"
if [ "${TRAIN_BASELINE}" = "1" ]; then
  echo "  tail -f ${BASE_LOG}"
fi
echo "  tail -f ${RODR_LOG}"
