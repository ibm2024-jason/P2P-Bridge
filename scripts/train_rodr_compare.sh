#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-starter}"
GPU="${2:-0}"
SAVE_ROOT="${3:-experiments/rodr_compare_${DATASET}}"

STEPS="${STEPS:-2000}"
BS="${BS:-8}"
WORKERS="${WORKERS:-2}"
WANDB_MODE="${WANDB_MODE:-disabled}"
RODR_WEIGHT="${RODR_WEIGHT:-1.0}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export WANDB_MODE

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

echo "Using GPU ${GPU}"
echo "Running baseline from ${BASE_CONFIG}"
python train.py \
  --config "${BASE_CONFIG}" \
  --save_dir "${SAVE_ROOT}" \
  --name baseline_mse \
  --distribution_type single \
  --training.steps "${STEPS}" \
  --training.bs "${BS}" \
  --data.workers "${WORKERS}"

echo "Running RODR from ${RODR_CONFIG}"
python train.py \
  --config "${RODR_CONFIG}" \
  --save_dir "${SAVE_ROOT}" \
  --name rodr_w${RODR_WEIGHT}" \
  --distribution_type single \
  --training.steps "${STEPS}" \
  --training.bs "${BS}" \
  --data.workers "${WORKERS}" \
  --diffusion.rodr_tangent_weight "${RODR_WEIGHT}"
