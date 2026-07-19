#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-starter}"
GPU="${2:-0}"
SAVE_ROOT="${3:-experiments/rodr_compare_${DATASET}}"

STEP="${STEP:-2000}"
RODR_WEIGHT="${RODR_WEIGHT:-1.0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_WORKERS="${EVAL_WORKERS:-2}"
EVAL_ACCUM_ITER="${EVAL_ACCUM_ITER:-8}"

export CUDA_VISIBLE_DEVICES="${GPU}"

BASE_CKPT="${SAVE_ROOT}/baseline_mse/step_${STEP}.pth"
RODR_CKPT="${SAVE_ROOT}/rodr_w${RODR_WEIGHT}/step_${STEP}.pth"

if [ ! -f "${BASE_CKPT}" ]; then
  echo "Missing baseline checkpoint: ${BASE_CKPT}" >&2
  exit 1
fi
if [ ! -f "${RODR_CKPT}" ]; then
  echo "Missing RODR checkpoint: ${RODR_CKPT}" >&2
  exit 1
fi

case "${DATASET}" in
  starter|StarterLocal)
    python evaluate_starterlocal.py \
      --model_path "${BASE_CKPT}" \
      --output_dir "${SAVE_ROOT}/baseline_mse/eval_step_${STEP}" \
      --batch_size "${EVAL_BATCH_SIZE}" \
      --workers "${EVAL_WORKERS}" \
      --accum_iter "${EVAL_ACCUM_ITER}"

    python evaluate_starterlocal.py \
      --model_path "${RODR_CKPT}" \
      --output_dir "${SAVE_ROOT}/rodr_w${RODR_WEIGHT}/eval_step_${STEP}" \
      --batch_size "${EVAL_BATCH_SIZE}" \
      --workers "${EVAL_WORKERS}" \
      --accum_iter "${EVAL_ACCUM_ITER}"
    ;;
  punet|PUNet)
    python evaluate_objects.py \
      --model_path "${BASE_CKPT}" \
      --dataset PUNet \
      --output_root "${SAVE_ROOT}/baseline_mse/eval_objects" \
      --gpu cuda:0

    python evaluate_objects.py \
      --model_path "${RODR_CKPT}" \
      --dataset PUNet \
      --output_root "${SAVE_ROOT}/rodr_w${RODR_WEIGHT}/eval_objects" \
      --gpu cuda:0
    ;;
  *)
    echo "Unknown DATASET=${DATASET}. Use starter or punet." >&2
    exit 2
    ;;
esac
