#!/usr/bin/env bash
set -euo pipefail

SAVE_ROOT="${1:-experiments/starterlocal_compare}"
STEP="${STEP:-2000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_WORKERS="${EVAL_WORKERS:-2}"
EVAL_ACCUM_ITER="${EVAL_ACCUM_ITER:-8}"

python evaluate_starterlocal.py \
  --model_path "${SAVE_ROOT}/baseline_mse/step_${STEP}.pth" \
  --output_dir "${SAVE_ROOT}/baseline_mse/eval_step_${STEP}" \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --workers "${EVAL_WORKERS}" \
  --accum_iter "${EVAL_ACCUM_ITER}"

python evaluate_starterlocal.py \
  --model_path "${SAVE_ROOT}/rodr_w1/step_${STEP}.pth" \
  --output_dir "${SAVE_ROOT}/rodr_w1/eval_step_${STEP}" \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --workers "${EVAL_WORKERS}" \
  --accum_iter "${EVAL_ACCUM_ITER}"
