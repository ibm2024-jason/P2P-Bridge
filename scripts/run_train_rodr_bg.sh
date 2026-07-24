#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-punet}"
GPU="${2:-0}"
SAVE_ROOT="${3:-experiments/rodr_only_${DATASET}}"

STEPS="${STEPS:-450000}"
LR="${LR:-1e-4}"
RODR_WEIGHT="${RODR_WEIGHT:-0.5}"
OBJECTIVE="${OBJECTIVE:-pred_x0}"
WANDB_MODE="${WANDB_MODE:-disabled}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
VIZ_INTERVAL="${VIZ_INTERVAL:-10000}"
RODR_NAME="${RODR_NAME:-rodr_${OBJECTIVE}_w${RODR_WEIGHT}_lr${LR}}"

case "${DATASET}" in
  starter|StarterLocal)
    RODR_CONFIG="configs/PVDS_StarterLocal_rodr.yaml"
    DEFAULT_BS=8
    DEFAULT_WORKERS=2
    ;;
  punet|PUNet)
    RODR_CONFIG="configs/PVDS_PUNet_rodr.yaml"
    DEFAULT_BS=32
    DEFAULT_WORKERS=4
    ;;
  *)
    echo "Unknown DATASET=${DATASET}. Use starter or punet." >&2
    exit 2
    ;;
esac

BS="${BS:-${DEFAULT_BS}}"
WORKERS="${WORKERS:-${DEFAULT_WORKERS}}"

mkdir -p "${SAVE_ROOT}/logs"

RODR_LOG="${SAVE_ROOT}/logs/${RODR_NAME}.log"
RODR_PID="${SAVE_ROOT}/${RODR_NAME}.pid"

echo "Starting RODR-only training on GPU ${GPU}. Log: ${RODR_LOG}"
(
  export CUDA_VISIBLE_DEVICES="${GPU}"
  export WANDB_MODE
  python train.py \
    --config "${RODR_CONFIG}" \
    --save_dir "${SAVE_ROOT}" \
    --name "${RODR_NAME}" \
    --distribution_type single \
    --diffusion.objective "${OBJECTIVE}" \
    --diffusion.loss_type rodr \
    --diffusion.rodr_tangent_weight "${RODR_WEIGHT}" \
    --training.optimizer.lr "${LR}" \
    --training.scheduler.type StepLR \
    --training.steps "${STEPS}" \
    --training.save_interval "${SAVE_INTERVAL}" \
    --training.viz_interval "${VIZ_INTERVAL}" \
    --training.bs "${BS}" \
    --data.workers "${WORKERS}"
) >"${RODR_LOG}" 2>&1 &

echo $! > "${RODR_PID}"
echo "RODR PID: $(cat "${RODR_PID}")"
echo "Check log with:"
echo "  tail -f ${RODR_LOG}"
