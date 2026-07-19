#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-data}"

echo "Checking starter data under: ${DATA_ROOT}"

for path in \
  "${DATA_ROOT}/dataset_train" \
  "${DATA_ROOT}/datalist/train.txt" \
  "${DATA_ROOT}/local/clean" \
  "${DATA_ROOT}/local/noisy"; do
  if [ ! -e "${path}" ]; then
    echo "Missing: ${path}" >&2
    exit 1
  fi
done

mesh_count="$(find "${DATA_ROOT}/dataset_train" -type f -name 'model_normalized.obj' | wc -l)"
clean_count="$(find "${DATA_ROOT}/local/clean" -type f -name '*.npy' | wc -l)"
noisy_count="$(find "${DATA_ROOT}/local/noisy" -type f -name '*.npy' | wc -l)"

echo "dataset_train meshes: ${mesh_count}"
echo "local clean npy: ${clean_count}"
echo "local noisy npy: ${noisy_count}"
echo "datalist train entries: $(wc -l < "${DATA_ROOT}/datalist/train.txt")"

if [ "${mesh_count}" -eq 0 ]; then
  echo "No model_normalized.obj files found under ${DATA_ROOT}/dataset_train" >&2
  exit 1
fi

echo "Starter data layout looks OK."
