#!/usr/bin/env bash
set -euo pipefail

SAVE_ROOT="${1:-experiments/starterlocal_compare}"
STEPS="${STEPS:-2000}"
BS="${BS:-8}"
WORKERS="${WORKERS:-2}"
WANDB_MODE="${WANDB_MODE:-disabled}"

export WANDB_MODE

python train.py \
  --config configs/PVDS_StarterLocal_base.yaml \
  --save_dir "${SAVE_ROOT}" \
  --name baseline_mse \
  --distribution_type single \
  --training.steps "${STEPS}" \
  --training.bs "${BS}" \
  --data.workers "${WORKERS}"

python train.py \
  --config configs/PVDS_StarterLocal_rodr.yaml \
  --save_dir "${SAVE_ROOT}" \
  --name rodr_w1 \
  --distribution_type single \
  --training.steps "${STEPS}" \
  --training.bs "${BS}" \
  --data.workers "${WORKERS}"
